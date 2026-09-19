"""Normalized error hierarchy for runtime and provider failures."""

from typing import Any


class AIRuntimeError(Exception):
    """Base class for failures exposed by the AI runtime."""


class ConfigurationError(AIRuntimeError):
    """Raised when runtime configuration is invalid or unavailable."""


class PolicyViolationError(AIRuntimeError):
    """Raised when a request violates an applicable policy."""


class NoEligibleModelError(AIRuntimeError):
    """Raised when no configured model can satisfy a request."""


class ProviderError(AIRuntimeError):
    """Base class for normalized provider failures.

    The classification flags describe a failure for future orchestration. They
    deliberately do not perform retry or fallback themselves.
    """

    retryable = False
    fallback_eligible = False

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        self.attempts: list[Any] = []


class ProviderUnavailableError(ProviderError):
    """Raised when a provider cannot currently serve a request."""

    retryable = True
    fallback_eligible = True


class ProviderTimeoutError(ProviderError):
    """Raised when a provider does not respond before its timeout."""

    retryable = True
    fallback_eligible = True


class ProviderRateLimitError(ProviderError):
    """Raised when a provider rejects a request due to rate limiting."""

    retryable = True
    fallback_eligible = True


class ProviderAuthenticationError(ProviderError):
    """Raised when provider credentials are invalid or unavailable."""


class ProviderRequestError(ProviderError):
    """Raised when a provider rejects a malformed or invalid request."""


class AllCandidatesFailedError(AIRuntimeError):
    """Raised when no further eligible candidate can complete a request."""

    def __init__(
        self, attempts: list[Any], message: str = "all candidates failed"
    ) -> None:
        super().__init__(message)
        self.attempts = list(attempts)
