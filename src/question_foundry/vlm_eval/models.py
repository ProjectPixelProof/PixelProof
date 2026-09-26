"""Versioned model aliases for the OpenRouter evaluation backend."""

from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path

MODEL_REGISTRY_PATH = Path(__file__).with_name("models.toml")
REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
IMAGE_DETAILS = frozenset({"auto", "low", "high"})


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    reference: str
    model_id: str
    reasoning_effort: str | None
    max_tokens: int
    image_detail: str

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def load_model_registry(path: Path = MODEL_REGISTRY_PATH) -> dict[str, ModelSpec]:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "vlm-api-model-registry@0.1.0":
        raise ValueError(f"unsupported model registry schema: {path}")
    output = {}
    for alias, raw in payload.get("models", {}).items():
        output[alias] = _model_spec(alias, raw)
    if not output:
        raise ValueError(f"model registry is empty: {path}")
    return output


def _model_spec(reference: str, raw: dict) -> ModelSpec:
    model_id = raw.get("model_id")
    if not isinstance(model_id, str) or "/" not in model_id:
        raise ValueError(f"model {reference!r} requires a provider/model model_id")
    effort = raw.get("reasoning_effort")
    if effort is not None and effort not in REASONING_EFFORTS:
        raise ValueError(f"model {reference!r} has unsupported reasoning effort {effort!r}")
    max_tokens = raw.get("max_tokens", 4096)
    if not isinstance(max_tokens, int) or max_tokens < 64:
        raise ValueError(f"model {reference!r} has invalid max_tokens")
    image_detail = raw.get("image_detail", "high")
    if image_detail not in IMAGE_DETAILS:
        raise ValueError(f"model {reference!r} has invalid image_detail")
    return ModelSpec(reference, model_id, effort, max_tokens, image_detail)


def resolve_model(
    reference: str,
    *,
    registry_path: Path = MODEL_REGISTRY_PATH,
    effort: str | None = None,
    max_tokens: int | None = None,
    image_detail: str | None = None,
) -> ModelSpec:
    registry = load_model_registry(registry_path)
    if reference in registry:
        spec = registry[reference]
    elif "/" in reference:
        spec = ModelSpec(reference, reference, None, 4096, "high")
    else:
        raise ValueError(
            f"unknown model alias {reference!r}; use a registry alias or provider/model slug"
        )
    changes = {}
    if effort is not None:
        if effort not in REASONING_EFFORTS:
            raise ValueError(f"unsupported reasoning effort: {effort}")
        changes["reasoning_effort"] = effort
    if max_tokens is not None:
        if max_tokens < 64:
            raise ValueError("max_tokens must be at least 64")
        changes["max_tokens"] = max_tokens
    if image_detail is not None:
        if image_detail not in IMAGE_DETAILS:
            raise ValueError(f"unsupported image detail: {image_detail}")
        changes["image_detail"] = image_detail
    return dataclasses.replace(spec, **changes)
