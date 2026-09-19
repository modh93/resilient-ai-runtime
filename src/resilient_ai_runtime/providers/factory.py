"""Construction-only registration for configured Runtime v0 providers."""

from resilient_ai_runtime.core.config import ProviderConfig
from resilient_ai_runtime.core.errors import ConfigurationError

from .anthropic import AnthropicAdapter
from .base import ProviderAdapter
from .mistral import MistralAdapter
from .ollama import OllamaAdapter
from .openai import OpenAIAdapter


def create_provider_adapter(config: ProviderConfig) -> ProviderAdapter:
    """Construct the adapter registered for one configured provider.

    This function does not select a provider or execute requests; it only maps
    an already-selected configuration entry to its adapter implementation.
    """

    if config.name == OllamaAdapter.name:
        return OllamaAdapter(base_url=config.resolve_base_url())
    if config.name == MistralAdapter.name:
        return MistralAdapter(
            base_url=config.resolve_base_url(),
            credential_env=config.credential_env or "MISTRAL_API_KEY",
        )
    if config.name == OpenAIAdapter.name:
        return OpenAIAdapter(
            base_url=config.resolve_base_url(),
            credential_env=config.credential_env or "OPENAI_API_KEY",
        )
    if config.name == AnthropicAdapter.name:
        return AnthropicAdapter(
            base_url=config.resolve_base_url(),
            credential_env=config.credential_env or "ANTHROPIC_API_KEY",
        )
    raise ConfigurationError(f"no adapter is registered for provider {config.name!r}")
