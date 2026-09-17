"""Command-line interface for the Star Gazing Weather agent.

Thin wrapper around ``run()``: owns only argument parsing and I/O, so the
agent loop in ``agent.py`` stays a pure library function. Registered as the
``star-gazing-weather-agent`` console script and reachable via
``python -m star_gazing_weather_agent``.
"""

import argparse
import asyncio
import logging

from .agent import MAX_ITERATIONS, run
from .messages import ChatMessage

log = logging.getLogger("star-gazing-weather-agent")

DEFAULT_QUESTION = (
    "Please check the forecast and moon phase for Mauna Kea for tonight. "
    "Use your tools."
)


async def _amain() -> None:
    parser = argparse.ArgumentParser(
        description="Decide if tonight is good for telescope observing."
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="the question to ask the agent (quoted or not)",
    )
    args = parser.parse_args()

    question: str = " ".join(args.question) or DEFAULT_QUESTION

    messages: list[ChatMessage] = [
        ChatMessage(
            role="system",
            content=(
                "You help decide whether tonight is good for telescope observing. "
                "When the user names an observing site, first resolve it to WGS84 "
                "decimal-degree coordinates and pass them to get_forecast(lat, lon) "
                "— the tool takes coordinates, not site names. If you do not know "
                "the site's coordinates, ask the user for them. To know today's "
                "date for a location call get_todays_date(location) — never guess "
                "the date, midnights differ by timezone. Use your tools; never "
                "guess data."
            ),
        ),
        ChatMessage(role="user", content=question),
    ]

    answer: str | None = await run(messages)
    if answer is None:
        log.warning("hit iteration cap of %d without a final answer", MAX_ITERATIONS)
    else:
        log.info("answer=%s", answer)


def main() -> None:
    """Console-script entry: run the async agent loop in a fresh event loop."""
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
