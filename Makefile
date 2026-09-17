.PHONY: format typecheck test

format:
	uv run ruff format .
	uv run ruff check --select I,RUF022 --fix .

typecheck:
	uv run mypy .

test:
	uv run pytest -q