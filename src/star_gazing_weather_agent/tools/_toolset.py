"""observing_tools: the toolset the observing tools register themselves on.

Each tool module decorates its own function with @observing_tool, so a tool's
name, description and schema live next to its implementation. This module sits
apart from the tools package so tool modules can import it without importing
the package — which imports them — and closing a cycle.
"""

import functools
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic_ai.toolsets import FunctionToolset

log = logging.getLogger("clear-sky")

observing_tools: FunctionToolset[None] = FunctionToolset()


def observing_tool(
    fn: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """Register fn as a tool, turning a raised error into an observation.

    The loop must survive a failing tool, so the model receives the failure as
    data ({"error": ...}) rather than the whole run aborting.
    """

    @functools.wraps(fn)
    async def guarded(**kwargs: Any) -> Any:
        try:
            return await fn(**kwargs)
        except Exception as e:  # tool failures are data for the model, not crashes
            log.info("tool %s failed: %s", fn.__name__, e)
            return {"error": str(e)}

    return observing_tools.tool_plain(guarded)
