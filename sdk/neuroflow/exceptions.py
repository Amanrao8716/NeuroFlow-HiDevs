"""Custom exceptions for the NeuroFlow SDK."""

from __future__ import annotations


class NeuroFlowError(Exception):
    """Base exception for all NeuroFlow SDK errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AuthenticationError(NeuroFlowError):
    """Raised when authentication fails (401/403)."""

    def __init__(self, message: str = "Authentication failed.") -> None:
        super().__init__(message, status_code=401)


class RateLimitError(NeuroFlowError):
    """Raised when the API returns 429 Too Many Requests."""

    def __init__(
        self, message: str = "Rate limit exceeded.", retry_after: float | None = None
    ) -> None:
        super().__init__(message, status_code=429)
        self.retry_after = retry_after


class IngestionError(NeuroFlowError):
    """Raised when document ingestion fails."""

    pass


class TimeoutError(NeuroFlowError):
    """Raised when a polling operation exceeds the configured timeout."""

    pass
