"""StarGazingIntentClassifier: decides intent, nothing more.

One cheap no-tool call returns whether the question belongs to the advisor
(OBSERVE), asks how it works (HELP), or is off-topic (OTHER). Routing those
results is the orchestrator's job, one level up.
"""

import logging
from enum import Enum

from ..constants import INTENT_PROMPT
from ._base import AgentBase

log = logging.getLogger("clear-sky")


class Intent(Enum):
    """Which of the three doors the question walks through."""

    OBSERVE = "OBSERVE"
    HELP = "HELP"
    OTHER = "OTHER"


class StarGazingIntentClassifier(AgentBase):
    """Decide a question's intent: OBSERVE, HELP, or OTHER."""

    async def classify(self, question: str) -> Intent:
        """One no-tool model call with a tiny token budget; returns an Intent."""
        response = await self._call_llm(
            model=self.settings.qualified_model,
            messages=[
                {"role": "system", "content": INTENT_PROMPT},
                {"role": "user", "content": question},
            ],
            api_key=self.settings.api_key,
            max_tokens=8,
            num_retries=self.settings.max_retries,
        )
        content = (response.choices[0].message.content or "").strip().upper()
        first_word = content.split(maxsplit=1)[0].rstrip(".,;:!?")
        for intent in Intent:
            if first_word == intent.value:
                return intent
        log.warning("intent gate got an unrecognised reply %r; refusing", content)
        return Intent.OTHER
