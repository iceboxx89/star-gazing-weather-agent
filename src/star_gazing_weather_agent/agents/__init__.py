"""ask_agent: the routing front door that joins classifier and loop.

Orchestration does not need encapsulation, so ask_agent is a plain function.
It classifies the question via StarGazingIntentClassifier and routes: OBSERVE
runs a StarGazingWeatherAgent built with the same model and settings; HELP and
OTHER short-circuit to help text and a refusal.
"""

from pydantic_ai.models import Model

from ..constants import HELP_MESSAGE, MAX_ITERATIONS, REFUSAL_MESSAGE
from ..settings import Settings
from .intent_agent import Intent, StarGazingIntentClassifier
from .star_gazing_weather_agent import StarGazingWeatherAgent

__all__ = ["StarGazingIntentClassifier", "StarGazingWeatherAgent", "ask_agent"]


async def ask_agent(
    question: str,
    model: Model | None = None,
    settings: Settings | None = None,
    max_iterations: int = MAX_ITERATIONS,
) -> str | None:
    """Classify the question, then refuse, help, or run the observing loop.

    OBSERVE hands off to a StarGazingWeatherAgent sharing the classifier's
    model and settings. Returns the final answer, help text, refusal, or None
    on the iteration cap.
    """
    settings = settings or Settings()
    intent = await StarGazingIntentClassifier(model=model, settings=settings).classify(
        question
    )

    match intent:
        case Intent.OTHER:
            return REFUSAL_MESSAGE
        case Intent.HELP:
            return HELP_MESSAGE
        case Intent.OBSERVE:
            weather = StarGazingWeatherAgent(
                model=model, settings=settings, max_iterations=max_iterations
            )
            return await weather.ask(question)
        case _:
            raise AssertionError(f"unhandled intent: {intent}")
