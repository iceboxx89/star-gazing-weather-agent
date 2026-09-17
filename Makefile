.PHONY: env format typecheck test

env:
	@if [ ! -f .env ]; then cp .env.tmp .env && echo "Created .env from .env.tmp — fill in your API key."; \
	else echo ".env already exists — leaving it alone."; fi

format:
	uv run ruff format .
	uv run ruff check --select I,RUF022 --fix .

typecheck:
	uv run mypy .

test:
	uv run pytest -q