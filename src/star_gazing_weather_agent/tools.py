"""Tool implementations for the Clear Sky agent.

get_forecast pulls real data from 7Timer's ASTRO product (free, no API key;
GFS-derived, 3-day range at 3-hourly steps). get_moon_phase and
get_todays_date are local helpers; the moon phase is still a stub.

The function is the source of truth: its name, docstring and signature drive
the schema the model sees (via litellm) and the registry the loop dispatches to.
"""

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, TypedDict
from zoneinfo import ZoneInfo

import httpx
import litellm

SEVEN_TIMER_URL = "http://www.7timer.info/bin/astro.php"
SEVEN_TIMER_TIMEOUT = 10.0

LOCATION_TO_TIMEZONE: dict[str, str] = {
    "Mauna Kea": "Pacific/Honolulu",
    "Chile": "America/Santiago",
    "Arizona": "America/Phoenix",
    "Hawaii": "Pacific/Honolulu",
    "California": "America/Los_Angeles",
    "UK": "Europe/London",
}

# 7Timer band -> midpoint, so values are usable numbers rather than bands.
SEEING_ARCSEC: dict[int, float] = {
    1: 0.42,  # <0.5"
    2: 0.62,  # 0.5-0.75"
    3: 0.88,  # 0.75-1"
    4: 1.12,  # 1-1.25"
    5: 1.38,  # 1.25-1.5"
    6: 1.75,  # 1.5-2"
    7: 2.25,  # 2-2.5"
    8: 2.75,  # >2.5"
}
CLOUD_PERCENT: dict[int, int] = {
    1: 3,
    2: 12,
    3: 25,
    4: 37,
    5: 50,
    6: 62,
    7: 75,
    8: 87,
    9: 97,
}


class ForecastResult(TypedDict):
    """Shape of get_forecast output: cloud %, seeing arcsec, wind km/h, humidity."""

    site: str
    date: Optional[str]
    cloud: int
    seeing: float
    wind: int
    humidity: int


class MoonPhaseResult(TypedDict):
    """Shape of get_moon_phase output: illumination % and moon set time."""

    date: str
    illumination: int
    sets: str


async def get_forecast(
    lat: float, lon: float, date: Optional[str] = None
) -> ForecastResult:
    """Cloud cover, seeing, wind and humidity at a decimal-degree coordinate.

    Queries 7Timer's ASTRO product for the site's lat/lon. The model resolves
    the user's site name to WGS84 decimal degrees. date is YYYY-MM-DD (default:
    today); the forecast point nearest 21:00 UTC of that date is returned.
    """
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"coordinates out of range: lat={lat}, lon={lon}")

    target_day = (
        datetime.fromisoformat(date) if date else datetime.now(timezone.utc)
    ).date()
    target = datetime(
        target_day.year, target_day.month, target_day.day, 21, tzinfo=timezone.utc
    )

    async with httpx.AsyncClient(timeout=SEVEN_TIMER_TIMEOUT) as client:
        resp = await client.get(
            SEVEN_TIMER_URL,
            params={
                "lon": lon,
                "lat": lat,
                "ac": "0",
                "unit": "metric",
                "output": "json",
                "tzshift": "0",
            },
        )
        resp.raise_for_status()
        payload = resp.json()

    init = datetime.strptime(payload["init"], "%Y%m%d%H").replace(tzinfo=timezone.utc)
    data = payload["dataseries"]
    if not data:
        raise RuntimeError("7Timer returned no forecast points")
    point = min(
        data, key=lambda p: abs(init + timedelta(hours=p["timepoint"]) - target)
    )

    return {
        "site": f"{lat:.4f}, {lon:.4f}",
        "date": target_day.isoformat(),
        "cloud": CLOUD_PERCENT[point["cloudcover"]],
        "seeing": SEEING_ARCSEC[point["seeing"]],
        "wind": round(point["wind10m"]["speed"] * 3.6),  # m/s -> km/h
        "humidity": point["rh2m"],
    }


def get_moon_phase(date: str) -> MoonPhaseResult:
    """Moon phase — illumination percent and set time for a date."""
    return {"date": date, "illumination": 60, "sets": "23:40"}


def get_todays_date(location: str) -> str:
    """Return today's date in the locale format for the given location.

    Pass a known location name (e.g. 'Mauna Kea', 'Chile', 'UK') from
    LOCATION_TO_TIMEZONE.
    """
    tz_name = LOCATION_TO_TIMEZONE.get(location, "UTC")
    now = datetime.now(ZoneInfo(tz_name))
    return now.strftime("%A %-d %B %Y")


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
