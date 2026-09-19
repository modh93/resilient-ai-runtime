"""Explicit model registry used by deterministic routing."""

from collections.abc import Iterable

from resilient_ai_runtime.core.config import ModelConfig


class ModelRegistry:
    """Read-only access to models keyed by their logical configuration ID."""

    def __init__(self, models: dict[str, ModelConfig]) -> None:
        self._models = dict(models)

    def get(self, model_id: str) -> ModelConfig | None:
        """Return a model by logical ID, or ``None`` when it is unknown."""

        return self._models.get(model_id)

    def in_priority_order(self, model_ids: Iterable[str]) -> list[ModelConfig]:
        """Return known models once, preserving their configured priority order."""

        seen: set[str] = set()
        ordered_models: list[ModelConfig] = []
        for model_id in model_ids:
            if model_id in seen:
                continue
            seen.add(model_id)
            if model := self.get(model_id):
                ordered_models.append(model)
        return ordered_models
