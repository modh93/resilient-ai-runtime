"""Normalized input model for text generation."""

from dataclasses import dataclass
from typing import Any


@dataclass
class GenerationRequest:
    """A provider-agnostic request for text generation."""

    prompt: str
    system_prompt: str | None = None
    policy: str | None = None
    constraints: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")
