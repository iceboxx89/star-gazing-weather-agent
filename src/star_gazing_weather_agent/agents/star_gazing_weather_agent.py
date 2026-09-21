"""StarGazingWeatherAgent: the tool-grounding observing loop.

The loop is pydantic-ai's: call the model, run the tools it asks for, feed the
results back. Grounding is enforced by an output validator — a final answer is
accepted only once a tool has returned, or when the model refuses with
REFUSAL_PREFIX; anything else goes back to the model with a nudge to use tools.
"""

import logging

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from ..constants import MAX_ITERATIONS, REFUSAL_PREFIX, STEER_MESSAGE, SYSTEM_PROMPT
from ..settings import Settings
from ..tools import observing_tools

log = logging.getLogger("clear-sky")


class StarGazingWeatherAgent:
    """Tool-grounding loop for observing questions."""

    def __init__(
        self,
        model: Model | None = None,
        settings: Settings | None = None,
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        self.settings = settings or Settings()
        self.max_iterations = max_iterations
        self.agent = Agent(
            model or self.settings.build_model(),
            deps_type=type(None),
            system_prompt=SYSTEM_PROMPT,
            output_type=str,
            toolsets=[observing_tools],
            model_settings=self.settings.model_settings(),
            # Run tools even when the reply also carries prose, so a reply can
            # never end the loop before its tools have been executed.
            end_strategy="exhaustive",
            retries=max_iterations,
        )
        self.agent.output_validator(self._require_grounding)

    async def _require_grounding(self, ctx: RunContext[None], output: str) -> str:
        """Accept an answer only if a tool returned, or the model refused."""
        observed = any(
            isinstance(part, ToolReturnPart)
            for message in ctx.messages
            for part in message.parts
        )
        if observed or output.startswith(REFUSAL_PREFIX):
            return output
        raise ModelRetry(STEER_MESSAGE)

    async def ask(self, question: str) -> str | None:
        """Run the observing loop; None if it exhausts the request limit."""
        try:
            result = await self.agent.run(
                question,
                usage_limits=UsageLimits(request_limit=self.max_iterations),
            )
        except UsageLimitExceeded:
            log.warning(
                "hit request limit of %d without a grounded answer",
                self.max_iterations,
            )
            return None
        return result.output
