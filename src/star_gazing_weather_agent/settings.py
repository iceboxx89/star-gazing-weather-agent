"""Settings: typed app config; values come from the environment, with the
.env file read as a fallback (env vars win)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_key: str = ""

    provider: str = "groq"
    model: str = "qwen/qwen3.8-27b"
    max_retries: int = 0
    max_tokens: int | None = 500
    log_level: str = "INFO"

    @property
    def qualified_model(self) -> str:
        return f"{self.provider}/{self.model}"
