"""StarGazingWeatherAgent: the tool-grounding observing loop.

Configures the loop state machine:
  WAIT_FOR_MODEL -> tool_calls? -> EXECUTE_TOOLS -> WAIT_FOR_MODEL
                   no tool_calls -> DONE
Narration is filtered in code: while tools are being called, the reply's
content is logged for debugging but stripped from the transcript.
"""

import json
import logging
from collections.abc import Callable
from typing import Any

from litellm.types.utils import ChatCompletionMessageToolCall, Message

from ..constants import MAX_ITERATIONS, REFUSAL_PREFIX, STEER_MESSAGE, SYSTEM_PROMPT
from ..messages import ChatMessage
from ..tools import TOOL_REGISTRY, TOOLS
from ._base import AgentBase

log = logging.getLogger("clear-sky")


class StarGazingWeatherAgent(AgentBase):
    """Tool-grounding loop for observing questions."""

    def build_messages(self, question: str) -> list[ChatMessage]:
        """System + user messages that start an observing session."""
        return [
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=question),
        ]

    async def ask(
        self, question: str, max_iterations: int = MAX_ITERATIONS
    ) -> str | None:
        """Build the observing conversation and run the loop."""
        messages = self.build_messages(question)
        return await self.run(messages, max_iterations=max_iterations)

    async def run(
        self, messages: list[ChatMessage], max_iterations: int = MAX_ITERATIONS
    ) -> str | None:
        """Drive the agent loop: call the model, execute requested tools, repeat."""
        for _ in range(max_iterations):
            # State: WAIT_FOR_MODEL
            response = await self._call_llm(
                model=self.settings.qualified_model,
                messages=[m.to_dict() for m in messages],
                api_key=self.settings.api_key,
                max_tokens=self.settings.max_tokens,
                num_retries=self.settings.max_retries,
                tools=TOOLS,
            )
            msg: Message = response.choices[0].message
            log.info("model=%s", self.settings.qualified_model)

            tool_calls: list[ChatCompletionMessageToolCall] = msg.tool_calls or []
            if not tool_calls:
                # State: DONE — a reply ends the loop only when grounded in at
                # least one tool observation, or when the model formally refuses.
                # Anything else counted on the model's own knowledge: steer back.
                content = msg.content or ""
                held_answer = any(
                    m.role == "tool" for m in messages
                ) or content.startswith(REFUSAL_PREFIX)
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
            await self._execute_tools(messages, tool_calls)
        return None

    async def _execute_tools(
        self,
        messages: list[ChatMessage],
        tool_calls: list[ChatCompletionMessageToolCall],
    ) -> None:
        """Run each requested tool and append its observation to the transcript."""
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
                    result = await fn(**args)
                except Exception as e:
                    result = f"tool error: {e}"
                log.info("result: %s", result)
            # Observe: feed the result back, keyed to the call that asked for it.
            messages.append(
                ChatMessage(role="tool", tool_call_id=tc.id, content=json.dumps(result))
            )
