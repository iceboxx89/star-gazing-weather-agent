"""ChatMessage: one transcript entry in the Clear Sky agent loop.

All four roles share one shape: role + optional fields that are only set for
the role that needs them (tool_call_id on tool results, tool_calls on filtered
assistant messages). to_dict() converts to the wire format litellm expects,
omitting None fields.
"""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ChatMessage:
    role: str
    content: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Wire format for litellm: omit fields that are None."""
        return {k: v for k, v in asdict(self).items() if v is not None}
