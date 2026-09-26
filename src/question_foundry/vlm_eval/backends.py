"""Offline fake and OpenRouter/OpenAI-compatible evaluation backends."""

from __future__ import annotations

import copy
import dataclasses
import json
import urllib.error
import urllib.request
from typing import Any

from .models import ModelSpec
from .prompting import RESPONSE_FORMAT

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclasses.dataclass(frozen=True)
class BackendResponse:
    content: str
    served_model: str
    response_id: str | None
    provider: str | None
    usage: dict
    raw_response: dict
    refusal: str | None = None
    rationale: str | None = None


class ProviderResponseError(RuntimeError):
    """A provider returned a response that could not satisfy the response contract.

    The partially decoded response is retained so the append-only evaluator ledger
    can preserve provider identity, usage, finish metadata, and raw content instead
    of reducing a paid request to an opaque parsing exception.
    """

    def __init__(self, message: str, *, response: BackendResponse) -> None:
        super().__init__(message)
        self.response = response


@dataclasses.dataclass(frozen=True)
class ProviderPolicy:
    order: tuple[str, ...] = ()
    allow_fallbacks: bool = True
    data_collection: str = "deny"
    zdr: bool = False
    sort: str | None = None

    def as_dict(self) -> dict:
        payload: dict[str, Any] = {
            "require_parameters": True,
            "data_collection": self.data_collection,
            "allow_fallbacks": self.allow_fallbacks,
        }
        if self.order:
            payload["order"] = list(self.order)
        if self.zdr:
            payload["zdr"] = True
        if self.sort:
            payload["sort"] = self.sort
        return payload


class FakeBackend:
    """A deterministic, network-free backend used by the mandatory smoke test."""

    def complete(
        self,
        *,
        model: ModelSpec,
        messages: list[dict],
        candidates: tuple[str, ...],
        provider_policy: ProviderPolicy,
    ) -> BackendResponse:
        del messages, provider_policy
        content = json.dumps({"answer": candidates[0], "confidence": 0.5})
        raw = {
            "id": "offline-smoke",
            "model": model.model_id,
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        return BackendResponse(
            content=content,
            served_model=model.model_id,
            response_id="offline-smoke",
            provider="offline-fake",
            usage=raw["usage"],
            raw_response=raw,
        )


class OpenRouterBackend:
    """Thin OpenAI-SDK wrapper with retries disabled and explicit provider policy."""

    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: float,
        base_url: str = OPENROUTER_BASE_URL,
        referer: str | None = None,
        title: str = "visual-question-discovery developmental evaluation",
        response_format: dict | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised by operator install
            raise RuntimeError("install the API evaluator with: uv sync --extra api-eval") from exc
        headers = {"X-OpenRouter-Title": title}
        if referer:
            headers["HTTP-Referer"] = referer
        self._api_key = api_key
        self._response_format = response_format or RESPONSE_FORMAT
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers=headers,
            max_retries=0,
            timeout=timeout_seconds,
        )

    def complete(
        self,
        *,
        model: ModelSpec,
        messages: list[dict],
        candidates: tuple[str, ...],
        provider_policy: ProviderPolicy,
    ) -> BackendResponse:
        del candidates
        extra_body: dict[str, Any] = {"provider": provider_policy.as_dict()}
        if model.reasoning_effort is not None:
            extra_body["reasoning"] = {
                "effort": model.reasoning_effort,
                "exclude": True,
            }
        try:
            response = self._client.chat.completions.create(
                model=model.model_id,
                messages=messages,
                max_tokens=model.max_tokens,
                response_format=self._response_format,
                extra_body=extra_body,
            )
        except Exception as exc:
            message = str(exc).replace(self._api_key, "[REDACTED]")
            raise RuntimeError(message) from exc
        raw = response.model_dump(mode="json")
        usage = response.usage.model_dump(mode="json") if response.usage else {}
        provider = raw.get("provider")
        if provider is None:
            provider = raw.get("provider_name")
        choices = response.choices or []
        if not choices or choices[0] is None or choices[0].message is None:
            partial = BackendResponse(
                content="",
                served_model=response.model,
                response_id=response.id,
                provider=provider,
                usage=usage,
                raw_response=raw,
            )
            raise ProviderResponseError(
                "provider response did not contain choices[0].message",
                response=partial,
            )
        message = choices[0].message
        content = message.content or ""
        return BackendResponse(
            content=content,
            served_model=response.model,
            response_id=response.id,
            provider=provider,
            usage=usage,
            raw_response=raw,
            refusal=getattr(message, "refusal", None),
        )


class LocalOpenAIBackend:
    """OpenAI-compatible local VLM backend without OpenRouter-only parameters."""

    def __init__(
        self,
        *,
        timeout_seconds: float,
        base_url: str,
        api_key: str = "local-not-a-secret",
        temperature: float = 0.0,
        enable_thinking: bool = False,
        structured_output: bool = True,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised by operator install
            raise RuntimeError("install the API evaluator with: uv sync --extra api-eval") from exc
        self._api_key = api_key
        self._temperature = temperature
        self._enable_thinking = enable_thinking
        self._structured_output = structured_output
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            max_retries=0,
            timeout=timeout_seconds,
        )

    def complete(
        self,
        *,
        model: ModelSpec,
        messages: list[dict],
        candidates: tuple[str, ...],
        provider_policy: ProviderPolicy,
    ) -> BackendResponse:
        del candidates, provider_policy
        local_messages = copy.deepcopy(messages)
        for message in local_messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("image_url"), dict):
                    # llama.cpp accepts the standard data URL but does not need OpenAI's
                    # hosted-model image-detail hint.
                    part["image_url"].pop("detail", None)
        request: dict[str, Any] = {
            "model": model.model_id,
            "messages": local_messages,
            "max_tokens": model.max_tokens,
            "temperature": self._temperature,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": self._enable_thinking}},
        }
        if self._structured_output:
            request["response_format"] = RESPONSE_FORMAT
        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:
            message = str(exc).replace(self._api_key, "[REDACTED]")
            raise RuntimeError(message) from exc
        raw = response.model_dump(mode="json")
        message = response.choices[0].message
        content = message.content or ""
        usage = response.usage.model_dump(mode="json") if response.usage else {}
        return BackendResponse(
            content=content,
            served_model=response.model or model.model_id,
            response_id=response.id,
            provider="local-openai-compatible",
            usage=usage,
            raw_response=raw,
            refusal=getattr(message, "refusal", None),
        )


def fetch_openai_compatible_models(
    *,
    base_url: str,
    timeout_seconds: float = 30,
    api_key: str | None = None,
) -> list[dict]:
    """Fetch an OpenAI-compatible model list without making an inference call."""
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(f"{base_url.rstrip('/')}/models", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"local model-metadata preflight failed: {exc}") from exc
    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise RuntimeError("local model-metadata response has no data list")
    return [item for item in models if isinstance(item, dict)]


def validate_local_model(model_id: str, models: list[dict], *, require_single: bool = True) -> dict:
    """Require the exact local model identity, optionally on a one-model endpoint."""
    ids = [item.get("id") for item in models if isinstance(item.get("id"), str)]
    if require_single and ids != [model_id]:
        raise ValueError(f"local endpoint model list differs from the required singleton: {ids}")
    metadata = next((item for item in models if item.get("id") == model_id), None)
    if metadata is None:
        raise ValueError(f"model is absent from the local endpoint: {model_id}")
    return metadata


def fetch_openrouter_model_metadata(
    *,
    api_key: str,
    base_url: str = OPENROUTER_BASE_URL,
    timeout_seconds: float = 30,
) -> list[dict]:
    """Fetch the live model catalogue without making an inference request."""
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"OpenRouter model-metadata preflight failed: {exc}") from exc
    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        raise RuntimeError("OpenRouter model-metadata response has no data list")
    return [item for item in models if isinstance(item, dict)]


def validate_live_model(model: ModelSpec, models: list[dict]) -> dict:
    """Fail closed when live metadata contradicts the requested VLM contract."""
    metadata = next((item for item in models if item.get("id") == model.model_id), None)
    if metadata is None:
        raise ValueError(f"model is absent from the live OpenRouter catalogue: {model.model_id}")

    architecture = metadata.get("architecture") or {}
    modalities = architecture.get("input_modalities") or metadata.get("input_modalities")
    if not isinstance(modalities, list) or "image" not in modalities:
        raise ValueError(f"live model metadata does not advertise image input: {model.model_id}")

    supported = metadata.get("supported_parameters")
    structured_names = {"response_format", "structured_outputs"}
    if not isinstance(supported, list) or not structured_names.intersection(supported):
        raise ValueError(
            f"live model metadata does not advertise structured output: {model.model_id}"
        )
    if model.reasoning_effort is not None and "reasoning" not in supported:
        raise ValueError(
            f"live model metadata does not advertise reasoning controls: {model.model_id}"
        )

    reasoning = metadata.get("reasoning")
    if isinstance(reasoning, dict) and model.reasoning_effort is not None:
        efforts = reasoning.get("supported_efforts")
        if isinstance(efforts, list) and model.reasoning_effort not in efforts:
            raise ValueError(
                f"reasoning effort {model.reasoning_effort!r} is unavailable for {model.model_id}"
            )
    return metadata
