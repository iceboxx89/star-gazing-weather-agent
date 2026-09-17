"""Clear Sky agent: gate the intent, then ask, act, observe, conclude.

Outer gate (classify_intent) decides before any tool work whether the question
belongs to the advisor: OBSERVE runs the loop, HELP and OTHER short-circuit.
The loop itself is a state machine with two states:
  WAIT_FOR_MODEL -> tool_calls? -> EXECUTE_TOOLS -> WAIT_FOR_MODEL
                   no tool_calls -> DONE
Narration is filtered in code: while tools are being called, the reply's
content is logged for debugging but stripped from the transcript.
"""

import inspect
import json
import logging
from collections.abc import Callable
from typing import Any

import litellm
from litellm import ModelResponse, acompletion
from litellm.types.utils import ChatCompletionMessageToolCall, Message

from .messages import ChatMessage
from .settings import Settings
from .tools import TOOL_REGISTRY, TOOLS

settings: Settings = Settings()

logging.basicConfig(
    level=settings.log_level, format="%(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("clear-sky")

# LiteLLM narrates every call at INFO; keep the terminal for our own logs.
litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

MAX_ITERATIONS = 6

# Prefix a refusal must carry so the loop can tell "declined" from "dodged":
# only these two states end the conversation without a tool observation.
REFUSAL_PREFIX = "REFUSED:"

STEER_MESSAGE = (
    "You did not call any of your tools. Either call the relevant tool(s) to "
    "answer from observing data, or, if the question is not about observing "
    f"the night sky, reply starting with '{REFUSAL_PREFIX}' followed by a "
    "short refusal."
)

SYSTEM_PROMPT = (
    "You help decide whether tonight is good for telescope observing. "
    "When the user names an observing site, first resolve it to WGS84 "
    "decimal-degree coordinates and pass them to get_forecast(lat, lon) "
    "— the tool takes coordinates, not site names. If you do not know "
    "the site's coordinates, ask the user for them. To know today's "
    "date for a location call get_todays_date(location) — never guess "
    "the date, midnights differ by timezone. Use your tools; never "
    "guess data. If the question is not about observing the night sky, "
    'reply starting with "REFUSED: " and a short refusal.'
)

HELP_MESSAGE = (
    "I decide whether tonight is good for telescope observing. Ask me about a "
    'site and a night, e.g. "Is tonight good at Mauna Kea?" — I check the '
    "forecast, moon phase, and the local date for your site."
)

INTENT_PROMPT = (
    "You are the doorman for a star-gazing weather advisor. Classify the "
    "user's question by replying with exactly one word only:\n"
    "OBSERVE — the question asks about observing the night sky, astronomy "
    "conditions, or a site's observing weather;\n"
    "HELP — the user is asking how to use this advisor or its commands;\n"
    "OTHER — anything else.\n"
    "Reply with only that word, no punctuation or explanation."
)

REFUSAL_MESSAGE = (
    f"{REFUSAL_PREFIX} I only answer questions about observing the night sky; "
    "type --help for how to use me."
)


async def _call(model_fn: Callable[..., Any], **kwargs: Any) -> ModelResponse:
    """Await model_fn if it's a coroutine function, else call it directly.

    Tests pass sync fakes; production passes litellm's async acompletion.
    """
    if inspect.iscoroutinefunction(model_fn):
        response = await model_fn(**kwargs)
    else:
        response = model_fn(**kwargs)
    assert isinstance(response, ModelResponse), (
        f"expected a chat response, got {type(response).__name__}"
    )
    return response


def build_messages(question: str) -> list[ChatMessage]:
    """System + user messages that start an observing session."""
    return [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=question),
    ]


async def classify_intent(
    question: str,
    model_fn: Callable[..., Any] = acompletion,
) -> str:
    """Cheap pre-loop gate: is this question about observing, tool help, or none?

    One no-tool model call with a tiny token budget. Unknown replies default to
    OTHER (fail closed: refuse rather than answer off-topic). Returns one of
    "OBSERVE", "HELP", "OTHER".
    """
    response = await _call(
        model_fn,
        model=settings.qualified_model,
        messages=[
            {"role": "system", "content": INTENT_PROMPT},
            {"role": "user", "content": question},
        ],
        api_key=settings.api_key,
        max_tokens=8,
        num_retries=settings.max_retries,
    )
    content = (response.choices[0].message.content or "").strip().upper()
    for token in ("OBSERVE", "HELP", "OTHER"):
        if content.startswith(token):
            return token
    log.warning("intent gate got an unrecognised reply %r; refusing", content)
    return "OTHER"


async def ask_agent(
    question: str,
    model_fn: Callable[..., Any] = acompletion,
    max_iterations: int = MAX_ITERATIONS,
) -> str | None:
    """Gate on intent, then run the observing loop or answer help/refusal.

    The gate is the deterministic outer loop: it decides before any tool work
    whether the question belongs to the advisor at all. Returns the model's
    final answer, help text, refusal, or None on the iteration cap.
    """
    intent = await classify_intent(question, model_fn)
    if intent == "OTHER":
        return REFUSAL_MESSAGE
    if intent == "HELP":
        return HELP_MESSAGE
    messages = build_messages(question)
    return await run(messages, model_fn=model_fn, max_iterations=max_iterations)


async def run(
    messages: list[ChatMessage],
    model_fn: Callable[..., Any] = acompletion,
    max_iterations: int = MAX_ITERATIONS,
) -> str | None:
    """Drive the agent loop: call the model, execute requested tools, repeat.

    Async: tools and the model call may both be coroutine functions; sync
    callables are supported for tests. Returns the model's final answer
    (content), or None if max_iterations was hit without one. Appends to
    messages in place, so the caller can inspect the full transcript
    afterwards. Pass a fake model_fn in tests.
    """
    for _ in range(max_iterations):
        # State: WAIT_FOR_MODEL
        response = await _call(
            model_fn,
            model=settings.qualified_model,
            messages=[m.to_dict() for m in messages],
            api_key=settings.api_key,
            max_tokens=settings.max_tokens,
            num_retries=settings.max_retries,
            tools=TOOLS,
        )
        msg: Message = response.choices[0].message
        log.info("model=%s", settings.qualified_model)

        tool_calls: list[ChatCompletionMessageToolCall] = msg.tool_calls or []
        if not tool_calls:
            # State: DONE — a reply ends the loop only when grounded in at
            # least one tool observation, or when the model formally refuses.
            # Anything else counted on the model's own knowledge: steer back.
            content = msg.content or ""
            held_answer = any(m.role == "tool" for m in messages) or content.startswith(
                REFUSAL_PREFIX
            )
            messages.append(ChatMessage(role="assistant", content=content))
            if held_answer:
                return content
            messages.append(ChatMessage(role="user", content=STEER_MESSAGE))
            continue

        # Filter: narration is noise mid-loop; keep it for the debug log, drop
        # it from the transcript we resend (saves tokens, keeps state clean).
        if msg.content:
            log.info("narration=%s", msg.content)
        messages.append(
            ChatMessage(
                role="assistant",
                tool_calls=[
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            )
        )

        # State: EXECUTE_TOOLS
        for tc in tool_calls:
            log.info(
                "tool_call: id=%s name=%s args=%s",
                tc.id,
                tc.function.name,
                tc.function.arguments,
            )
            assert (
                tc.function.name is not None
            )  # a nameless tool call is a provider bug
            fn: Callable[..., Any] | None = TOOL_REGISTRY.get(tc.function.name)
            if fn is None:
                result = f"tool not found: {tc.function.name}"
                log.error(result)
            else:
                try:
                    args = json.loads(tc.function.arguments)
                    if inspect.iscoroutinefunction(fn):
                        result = await fn(**args)
                    else:
                        result = fn(**args)
                except Exception as e:
                    result = f"tool error: {e}"
                log.info("result: %s", result)
            # Observe: feed the result back, keyed to the call that asked for it.
            messages.append(
                ChatMessage(role="tool", tool_call_id=tc.id, content=json.dumps(result))
            )
    return None
