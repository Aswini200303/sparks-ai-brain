"""Application configuration loaded from environment variables."""

import hashlib

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Service settings backed by environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    GEMINI_API_KEY: str
    SERVICE_API_KEY: str
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_TEMPERATURE: float = 0.1
    GEMINI_MAX_TOKENS: int = 8192
    APP_NAME: str = "Sparks AI Brain"
    APP_VERSION: str = "1.0.0"

    @field_validator("GEMINI_API_KEY", "SERVICE_API_KEY", mode="before")
    @classmethod
    def normalize_secret(cls, value: str) -> str:
        """Trim copy/paste whitespace from deployed secrets."""
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("GEMINI_API_KEY")
    @classmethod
    def validate_gemini_api_key(cls, value: str) -> str:
        """Catch common deployment mistakes before the first Gemini request."""
        lowered = value.lower()
        if lowered.startswith("bearer "):
            raise ValueError(
                "GEMINI_API_KEY must be the raw Gemini API key, not a Bearer token"
            )
        if value.startswith("{"):
            raise ValueError(
                "GEMINI_API_KEY must be a Gemini API key, not a service account JSON"
            )
        return value

    @property
    def gemini_api_key_fingerprint(self) -> str:
        """Return a non-secret fingerprint for comparing deployed env values."""
        digest = hashlib.sha256(self.GEMINI_API_KEY.encode("utf-8")).hexdigest()
        return digest[:12]


settings = Settings()
