"""Star Gazing Weather agent: a Clear-Sky observing advisor.

Re-exports the public surface so callers can do either::

    from star_gazing_weather_agent import ask_agent

or reach the internals via :mod:`star_gazing_weather_agent.agent`.
"""

from .agent import MAX_ITERATIONS, ask_agent, run
from .messages import ChatMessage

__all__ = ["MAX_ITERATIONS", "ChatMessage", "ask_agent", "run"]
