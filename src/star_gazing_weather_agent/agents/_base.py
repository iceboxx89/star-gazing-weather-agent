"""AgentBase: the shared seam of the agent classes — injected model callable
and settings, plus the model-call helper.

_call_llm awaits model_fn and asserts the response is a ModelResponse, the
contract every injected callable — production or fake — must satisfy.
"""

from collections.abc import Callable
from typing import Any

from litellm import ModelResponse, acompletion

from ..settings import Settings


class AgentBase:
    """Common seam for the agents: injected model callable and settings.

    _call_llm awaits model_fn and asserts the response is a ModelResponse, the
    contract every injected callable — production or fake — must satisfy.
    """

    def __init__(
        self,
        model_fn: Callable[..., Any] = acompletion,
        settings: Settings | None = None,
    ) -> None:
        self.model_fn = model_fn
        self.settings = settings or Settings()

    async def _call_llm(self, **kwargs: Any) -> ModelResponse:
        """Await model_fn and return the response it produced."""
        response = await self.model_fn(**kwargs)
        assert isinstance(response, ModelResponse), (
            f"expected a chat response, got {type(response).__name__}"
        )
        return response
