"""Settings: typed app config; values come from the environment, with the
.env file read as a fallback (env vars win)."""

from collections.abc import Callable
from typing import Any

from pydantic_ai.models import Model, infer_model
from pydantic_ai.providers import Provider
from pydantic_ai.settings import ModelSettings
from pydantic_settings import BaseSettings, SettingsConfigDict


def _openai_provider(api_key: str, base_url: str | None = None) -> Provider[Any]:
    from pydantic_ai.providers.openai import OpenAIProvider

    return OpenAIProvider(base_url=base_url, api_key=api_key or None)


def _groq_provider(api_key: str) -> Provider[Any]:
    from pydantic_ai.providers.groq import GroqProvider

    return GroqProvider(api_key=api_key or None)


def _anthropic_provider(api_key: str) -> Provider[Any]:
    from pydantic_ai.providers.anthropic import AnthropicProvider

    return AnthropicProvider(api_key=api_key or None)


def _google_provider(api_key: str) -> Provider[Any]:
    from pydantic_ai.providers.google import GoogleProvider

    return GoogleProvider(api_key=api_key or None)


# Provider SDKs are imported lazily: only the configured provider needs to be
# installed, and importing every one would slow every start.
_PROVIDER_BUILDERS: dict[str, Callable[[str], Provider[Any]]] = {
    "groq": _groq_provider,
    "openai": _openai_provider,
    "anthropic": _anthropic_provider,
    "google": _google_provider,
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_key: str = ""
    api_base: str | None = None

    provider: str = "groq"
    model: str = "qwen/qwen3.8-27b"
    max_tokens: int | None = 500
    log_level: str = "INFO"

    def model_settings(self) -> ModelSettings:
        """Model settings for this config: a token cap only when one is set."""
        if self.max_tokens is None:
            return ModelSettings()
        return ModelSettings(max_tokens=self.max_tokens)

    def build_model(self) -> Model:
        """Build the pydantic-ai model for this configuration.

        API_BASE points at any OpenAI-compatible endpoint; the openai-chat
        prefix selects the Chat Completions model, since the bare openai
        prefix means the Responses API, which compatible endpoints rarely serve.
        """
        if self.api_base:
            base_url = self.api_base
            return infer_model(
                f"openai-chat:{self.model}",
                provider_factory=lambda _: _openai_provider(self.api_key, base_url),
            )
        builder = _PROVIDER_BUILDERS.get(self.provider)
        if builder is None:
            # Unknown provider: let pydantic-ai infer it and read its own env vars.
            return infer_model(f"{self.provider}:{self.model}")
        return infer_model(
            f"{self.provider}:{self.model}",
            provider_factory=lambda _: builder(self.api_key),
        )
