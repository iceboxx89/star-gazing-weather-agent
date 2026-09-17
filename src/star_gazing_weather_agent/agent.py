"""Clear Sky agent loop: ask, act, observe, conclude.

State machine with two states:
  WAIT_FOR_MODEL -> tool_calls? -> EXECUTE_TOOLS -> WAIT_FOR_MODEL
                   no tool_calls -> DONE
Narration is filtered in code: while tools are being called, the reply's
content is logged for debugging but stripped from the transcript.
"""

import datetime
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


async def _call(model_fn: Callable[..., Any], **kwargs: Any) -> ModelResponse:
    """Call model_fn whether it's sync or async; force a valid plus_type."""
    if inspect.iscoroutinefunction(model_fn):
        response = await model_fn(**kwargs)
    else:
        response = model_fn(**kwargs)
    assert isinstance(response, ModelResponse), (
        f"expected a chat response, got {type(response).__name__}"
    )
    return response


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
            # State: DONE — no intent, the reply is the answer.
            messages.append(ChatMessage(role="assistant", content=msg.content))
            return msg.content

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
