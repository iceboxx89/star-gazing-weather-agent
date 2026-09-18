"""configure_logging: one-time app logging bootstrap for the entry points.

basicConfig must not run at import: the package is importable as a library,
and reconfiguring a host process's logging would be rude. Call
configure_logging() from cli.main — both the console script and ``python -m``
land there — before anything logs.
"""

import logging

import litellm

from .settings import Settings


def configure_logging(log_level: str | None = None) -> None:
    """Root handler, level, and LiteLLM muting; safe to call more than once.

    Defaults to Settings().log_level when no level is given. The CLI's -v
    flag still overrides the root level afterwards, inside the handler.
    """
    logging.basicConfig(
        level=log_level or Settings().log_level,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # LiteLLM narrates every call at INFO; keep the terminal for our own logs.
    litellm.suppress_debug_info = True
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
