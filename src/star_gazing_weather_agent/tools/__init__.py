"""tools: the agent's tool surface — one module per tool, registry built
here from the flat list.

Each tool module owns its function, its result TypedDict, and its data. The
function is the source of truth: its name, docstring and signature drive the
schema the model sees (via litellm) and the registry the loop dispatches to.
"""

from collections.abc import Callable
from typing import Any

import litellm

from .forecast import ForecastResult, get_forecast
from .moon_phase import MoonPhaseResult, get_moon_phase
from .todays_date import get_todays_date

__all__ = [
    "TOOLS",
    "TOOL_FUNCTIONS",
    "TOOL_REGISTRY",
    "ForecastResult",
    "MoonPhaseResult",
    "get_forecast",
    "get_moon_phase",
    "get_todays_date",
]

TOOL_FUNCTIONS: tuple[Callable[..., Any], ...] = (
    get_forecast,
    get_moon_phase,
    get_todays_date,
)

TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": litellm.utils.function_to_dict(fn)}
    for fn in TOOL_FUNCTIONS
]

TOOL_REGISTRY: dict[str, Callable[..., Any]] = {
    fn.__name__: fn for fn in TOOL_FUNCTIONS
}
