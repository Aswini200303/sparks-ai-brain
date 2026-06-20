"""API key authentication for protected endpoints."""

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import settings

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validate the x-api-key header against the configured service API key."""
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "missing_api_key",
                "message": "x-api-key header is required",
            },
        )

    if api_key != settings.SERVICE_API_KEY:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "invalid_api_key",
                "message": "The provided API key is not valid",
            },
        )

    return api_key
