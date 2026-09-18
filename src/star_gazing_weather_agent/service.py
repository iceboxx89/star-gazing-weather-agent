"""ask: synchronous front end to the agent, shared by the CLI and any front end.

No Typer or Rich here — pure orchestration: settings bootstrap, the async
agent loop bridged to a synchronous ask(), and the iteration-cap outcome.
"""

import asyncio
import logging

from .agents import ask_agent
from .constants import MAX_ITERATIONS
from .settings import Settings

log = logging.getLogger("clear-sky")


def ask(question: str, settings: Settings | None = None) -> str | None:
    """Ask synchronously: returns the answer, help text, refusal, or None on
    the iteration cap."""
    settings = settings or Settings()
    answer = asyncio.run(ask_agent(question, settings=settings))
    if answer is None:
        log.warning("hit iteration cap of %d without a final answer", MAX_ITERATIONS)
    return answer
