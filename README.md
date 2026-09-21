# Clear Sky Observing Advisor

Decides whether tonight is good for telescope observing, using a tool-calling
agent loop.

- **Package:** `star_gazing_weather_agent` (src layout)
- **CLI:** `star-gazing-weather-agent` console script or `python -m star_gazing_weather_agent`
- **Public API:** `ask_agent(question)` (async) and `service.ask(question)` (sync);
  both take an injectable pydantic-ai model, so tests drive a scripted model
  end to end

```mermaid
flowchart LR
    Q[Question] --> G{Intent gate}
    G -- help --> H[Usage text]
    G -- other --> R[Refused]
    G -- observing --> M[Model]
    M -->|needs more data| T[Tools]
    T --> M
    M -->|answer ready| Stop[Stop]
```

A cheap intent gate first decides whether the question is about observing
(OBSERVE), tool usage (HELP), or neither (refused) — the observing loop only
runs for OBSERVE. The gate returns an `Intent` enum, and the orchestrator
routes on it with one `match`.

![Clear Sky demo](resources/str-1.gif)

## Architecture

Layers are strictly separated, with the UI at the top and the model seam at
the bottom:

```
cli (Typer + Rich)                      → renders, exit codes, nothing else
  → service.ask (sync)                  → settings bootstrap + asyncio.run bridge
    → agents.ask_agent (async)          → routes the Intent, plain function
      → StarGazingIntentClassifier      → OBSERVE / HELP / OTHER
      → StarGazingWeatherAgent          → tool-grounding loop
        → pydantic-ai Agent             → injected model + observing tools
```

The observing loop is pydantic-ai's `Agent`: it waits for the model, executes
any requested tools, and feeds the results back — repeating until the model
answers grounded in a tool observation, or formally refuses with a `REFUSED:`
prefix. A reply that dodges the tools is rejected by an output validator and
sent back with a nudge. The whole layer under `service.ask` is async and free
of Typer/Rich, so any front end can drive it.

## Tools

Each tool registers itself on the shared toolset with `@observing_tool` at its
definition site, so its name, docstring and signature are its schema.

- `get_forecast` — real data from 7Timer's ASTRO product (free, no API key;
  GFS-derived, 3-day range at 3-hourly steps)
- `get_moon_phase` — still a stub (fixed 60% illumination)
- `get_todays_date` — resolves an observing location to its timezone for the
  correct locale date

## Setup

```bash
make env    # copies .env.tmp to .env — then edit it
uv sync
```

The agent calls the model through pydantic-ai. Fill `API_KEY`, `PROVIDER` and
`MODEL` in `.env` with a key the provider accepts; for any OpenAI-compatible
endpoint, set `API_BASE` and put just the model id in `MODEL` — then run the
agent.

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
