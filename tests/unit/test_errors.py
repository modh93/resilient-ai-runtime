"""Tests for normalized runtime errors."""

import pytest

from resilient_ai_runtime.core import (
    AIRuntimeError,
    ConfigurationError,
    NoEligibleModelError,
    PolicyViolationError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)


@pytest.mark.parametrize(
    "error_type",
    [
        ConfigurationError,
        PolicyViolationError,
        NoEligibleModelError,
        ProviderUnavailableError,
        ProviderTimeoutError,
        ProviderRateLimitError,
        ProviderAuthenticationError,
        ProviderRequestError,
    ],
)
def test_runtime_error_subclasses_are_distinguishable(
    error_type: type[AIRuntimeError],
) -> None:
    error = error_type("failure")

    assert isinstance(error, AIRuntimeError)
    assert type(error) is error_type


@pytest.mark.parametrize(
    "error_type",
    [ProviderUnavailableError, ProviderTimeoutError, ProviderRateLimitError],
)
def test_transient_provider_errors_allow_retry_and_fallback(
    error_type: type[AIRuntimeError],
) -> None:
    error = error_type("temporary failure")

    assert error.retryable is True
    assert error.fallback_eligible is True


@pytest.mark.parametrize(
    "error_type",
    [ProviderAuthenticationError, ProviderRequestError],
)
def test_non_transient_provider_errors_disallow_retry_and_fallback(
    error_type: type[AIRuntimeError],
) -> None:
    error = error_type("permanent failure")

    assert error.retryable is False
    assert error.fallback_eligible is False
