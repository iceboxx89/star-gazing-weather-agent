"""StarGazingIntentClassifier: decides intent, nothing more.

One cheap no-tool call returns whether the question belongs to the advisor
(OBSERVE), asks how it works (HELP), or is off-topic (OTHER). Routing those
results is the orchestrator's job, one level up.
"""

import logging
from enum import Enum

from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models import Model

from ..constants import INTENT_PROMPT
from ..settings import Settings

log = logging.getLogger("clear-sky")


class Intent(Enum):
    """Which of the three doors the question walks through."""

    OBSERVE = "OBSERVE"
    HELP = "HELP"
    OTHER = "OTHER"


class StarGazingIntentClassifier:
    """Decide a question's intent: OBSERVE, HELP, or OTHER."""

    def __init__(
        self,
        model: Model | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.agent = Agent(
            model or self.settings.build_model(),
            deps_type=type(None),
            system_prompt=INTENT_PROMPT,
            output_type=Intent,
            model_settings=self.settings.model_settings(),
            # No retries: an unusable reply fails closed to OTHER in one call.
            retries=0,
        )

    async def classify(self, question: str) -> Intent:
        """One no-tool call returning a validated Intent; OTHER on failure."""
        try:
            result = await self.agent.run(question)
        except UnexpectedModelBehavior:
            log.warning("intent gate got an unusable reply; refusing")
            return Intent.OTHER
        return result.output
