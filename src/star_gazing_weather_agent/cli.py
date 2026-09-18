"""cli: the Typer + Rich surface over the service layer.

The CLI owns nothing about how the agent runs: it renders the question, the
spinner, and the outcome panels, and maps failures to exit codes. All
orchestration lives in service.ask.
"""

import logging
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from . import service
from .constants import MAX_ITERATIONS
from .logging import configure_logging

log = logging.getLogger("star-gazing-weather-agent")
console = Console()


def ask(
    ctx: typer.Context,
    question: Annotated[
        list[str] | None,
        typer.Argument(
            help='the question to ask, e.g. "Is tonight good at Mauna Kea?"'
        ),
    ] = None,
    verbose: Annotated[
        int,
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="repeat to raise log verbosity: -v INFO, -vv DEBUG",
        ),
    ] = 0,
) -> None:
    """Decide if tonight is good for telescope observing."""
    root_log_level = logging.WARNING
    if verbose >= 2:
        root_log_level = logging.DEBUG
    elif verbose == 1:
        root_log_level = logging.INFO
    logging.getLogger().setLevel(root_log_level)

    if not question:
        console.print(
            "[yellow]No question given — type the observing question you want "
            'answered, e.g. "Is tonight good at Mauna Kea?"[/]'
        )
        console.print(ctx.get_help())
        raise typer.Exit(code=2)

    text = " ".join(question)

    try:
        with console.status("[cyan]Asking the star-gazing agent…[/]", spinner="dots"):
            answer = service.ask(text)
    except Exception as e:
        console.print(Panel(str(e), title="Error", border_style="red", padding=(1, 2)))
        raise typer.Exit(code=1)

    if answer is None:
        console.print(
            f"[yellow]No answer after {MAX_ITERATIONS} iterations — raising the "
            "cap or rephrasing may help.[/]"
        )
    else:
        console.print(
            Panel(Markdown(answer), title="Answer", border_style="cyan", padding=(1, 2))
        )
        log.info("answer=%s", answer)


def main() -> None:
    """Console-script and ``python -m`` entry point."""
    configure_logging()
    typer.run(ask)
