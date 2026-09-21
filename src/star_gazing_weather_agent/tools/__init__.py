"""tools: the agent's tool surface — one module per tool.

Each tool module owns its function, its result TypedDict, and its data, and
registers itself on observing_tools at its definition site, so a tool's name,
docstring and signature are its schema. Importing this package is what
populates the toolset.
"""

from ._toolset import observing_tool, observing_tools
from .forecast import ForecastResult, get_forecast
from .moon_phase import MoonPhaseResult, get_moon_phase
from .todays_date import get_todays_date

__all__ = [
    "ForecastResult",
    "MoonPhaseResult",
    "get_forecast",
    "get_moon_phase",
    "get_todays_date",
    "observing_tool",
    "observing_tools",
]
