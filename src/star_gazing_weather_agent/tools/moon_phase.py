"""get_moon_phase: moon illumination percent and set time for a date.

Still a stub — fixed values, no real source. A caller cannot tell the stub
from a real forecast, which is the known risk; the moon needs a real data
source before moon-dependent answers are trustworthy.
"""

from typing import TypedDict


class MoonPhaseResult(TypedDict):
    """Shape of get_moon_phase output: illumination % and moon set time."""

    date: str
    illumination: int
    sets: str


async def get_moon_phase(date: str) -> MoonPhaseResult:
    """Moon phase — illumination percent and set time for a date."""
    return {"date": date, "illumination": 60, "sets": "23:40"}
