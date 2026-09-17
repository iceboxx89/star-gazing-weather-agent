# Clear Sky Observing Advisor

Decides whether tonight is good for telescope observing, using an agent loop
with tool calls.

- **Package:** `star_gazing_weather_agent` (src layout)
- **CLI:** `star-gazing-weather-agent` console script or `python -m star_gazing_weather_agent`
- **Tools:** forecast and moon-phase stubs, plus `get_todays_date` which
  resolves an observing location to its timezone for the correct locale date

## Setup

```bash
uv sync
```

## Run

```bash
uv run star-gazing-weather-agent "Is tonight good at Mauna Kea?"
```

## Gates

```bash
make format     # ruff format + import order + sorted __all__
make typecheck  # mypy
make test       # pytest
```