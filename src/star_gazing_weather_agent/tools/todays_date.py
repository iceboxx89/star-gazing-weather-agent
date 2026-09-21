"""get_todays_date: the locale-format date for a named observing location.

Looks the location up in LOCATION_TO_TIMEZONE; unknown names fall back to
UTC. Midnights differ by timezone, so the model must call this rather than
guess the date.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from ._toolset import observing_tool

LOCATION_TO_TIMEZONE: dict[str, str] = {
    "Mauna Kea": "Pacific/Honolulu",
    "Chile": "America/Santiago",
    "Arizona": "America/Phoenix",
    "Hawaii": "Pacific/Honolulu",
    "California": "America/Los_Angeles",
    "UK": "Europe/London",
}


@observing_tool
async def get_todays_date(location: str) -> str:
    """Return today's date in the locale format for the given location.

    Pass a known location name (e.g. 'Mauna Kea', 'Chile', 'UK') from
    LOCATION_TO_TIMEZONE.
    """
    tz_name = LOCATION_TO_TIMEZONE.get(location, "UTC")
    now = datetime.now(ZoneInfo(tz_name))
    # %-d is glibc-only; build the day unpadded so this stays cross-platform.
    return f"{now.strftime('%A')} {now.day} {now.strftime('%B %Y')}"
