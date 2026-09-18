"""ask_agent: the routing front door that joins classifier and loop.

Orchestration does not need encapsulation, so ask_agent is a plain function.
It classifies the question via StarGazingIntentClassifier and routes:
OBSERVE runs a StarGazingWeatherAgent built with the same model_fn and
settings; HELP and OTHER short-circuit to help text and a refusal.
"""

from collections.abc import Callable
from typing import Any

from litellm import acompletion

from ..constants import HELP_MESSAGE, MAX_ITERATIONS, REFUSAL_MESSAGE
from ..settings import Settings
from .intent_agent import Intent, StarGazingIntentClassifier
from .star_gazing_weather_agent import StarGazingWeatherAgent

__all__ = ["StarGazingIntentClassifier", "StarGazingWeatherAgent", "ask_agent"]


async def ask_agent(
    question: str,
    model_fn: Callable[..., Any] = acompletion,
    settings: Settings | None = None,
    max_iterations: int = MAX_ITERATIONS,
) -> str | None:
    """Classify the question, then refuse, help, or run the observing loop.

    OBSERVE hands off to a StarGazingWeatherAgent sharing the classifier's
    model_fn and settings. Returns the final answer, help text, refusal, or
    None on the iteration cap.
    """
    settings = settings or Settings()
    intent = await StarGazingIntentClassifier(
        model_fn=model_fn, settings=settings
    ).classify(question)

    match intent:
        case Intent.OTHER:
            return REFUSAL_MESSAGE
        case Intent.HELP:
            return HELP_MESSAGE
        case Intent.OBSERVE:
            weather = StarGazingWeatherAgent(model_fn=model_fn, settings=settings)
            return await weather.ask(question, max_iterations=max_iterations)
        case _:
            raise AssertionError(f"unhandled intent: {intent}")
