"""Application configuration loaded from environment variables."""

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


settings = Settings()
