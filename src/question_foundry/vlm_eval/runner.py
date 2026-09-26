"""Offline-first planning and append-only execution for developmental VLM evaluation."""

from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path
from typing import Protocol

from PIL import Image

from . import PROMPT_VERSION
from .backends import BackendResponse, FakeBackend, ProviderPolicy, ProviderResponseError
from .datasets import (
    DatasetSnapshot,
    EvalExample,
    discover_manifests,
    load_examples,
    verify_selected_images,
)
from .models import ModelSpec
from .prompting import messages, parse_response, sanitized_request
from .sampling import select_chunk
from .store import (
    append_event,
    create_plan,
    event_summary,
    excluded_sample_keys,
    initialize_run,
    pending_plan,
    read_events,
)


class EvaluationBackend(Protocol):
    def complete(
        self,
        *,
        model: ModelSpec,
        messages: list[dict],
        candidates: tuple[str, ...],
        provider_policy: ProviderPolicy,
    ) -> BackendResponse: ...


def _contains_forbidden_field(value: object, forbidden_fields: frozenset[str]) -> bool:
    if isinstance(value, dict):
        return bool(forbidden_fields.intersection(value)) or any(
            _contains_forbidden_field(item, forbidden_fields) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_field(item, forbidden_fields) for item in value)
    return False


def load_inputs(inputs: list[Path]) -> tuple[list[EvalExample], list[DatasetSnapshot]]:
    manifests = discover_manifests(inputs)
    return load_examples(manifests)


def run_configuration(
    *,
    model: ModelSpec,
    snapshots: list[DatasetSnapshot],
    sampling_method: str,
    provider_policy: ProviderPolicy,
    transport: dict | None = None,
) -> dict:
    return {
        "purpose": "developmental-vlm-evaluation",
        "development_only": True,
        "model": model.as_dict(),
        "prompt_version": PROMPT_VERSION,
        "sampling_method": sampling_method,
        "provider_policy": provider_policy.as_dict(),
        "transport": transport or {"backend": "offline-or-unspecified"},
        "datasets": [item.as_dict() for item in snapshots],
    }


def prepare_plan(
    *,
    run_dir: Path,
    examples: list[EvalExample],
    snapshots: list[DatasetSnapshot],
    model: ModelSpec,
    provider_policy: ProviderPolicy,
    sample_size: int,
    seed: int,
    sampling_method: str,
    retry_errors: bool,
    transport: dict | None = None,
) -> tuple[dict, list[EvalExample], dict[str, str], bool]:
    """Resume an incomplete immutable plan or create the next deterministic chunk."""
    initialize_run(
        run_dir,
        run_configuration(
            model=model,
            snapshots=snapshots,
            sampling_method=sampling_method,
            provider_policy=provider_policy,
            transport=transport,
        ),
    )
    events = read_events(run_dir)
    existing = pending_plan(run_dir, events, retry_errors=retry_errors)
    by_key = {example.sample_key: example for example in examples}
    if existing is not None:
        _path, plan = existing
        selected = []
        for item in plan["samples"]:
            example = by_key.get(item["sample_key"])
            if example is None:
                raise ValueError(
                    f"planned sample is absent from current snapshots: {item['sample_key']}"
                )
            selected.append(example)
        hashes = verify_selected_images(selected)
        for item in plan["samples"]:
            if hashes[item["sample_key"]] != item["image_sha256"]:
                raise ValueError(f"planned image changed on disk: {item['image_path']}")
        return plan, selected, hashes, True

    excluded = excluded_sample_keys(events, retry_errors=retry_errors)
    selected = select_chunk(
        examples,
        excluded_keys=excluded,
        sample_size=sample_size,
        seed=seed,
        method=sampling_method,
    )
    if not selected:
        return {}, [], {}, False
    hashes = verify_selected_images(selected)
    sample_records = []
    for example in selected:
        record = example.public_metadata()
        record["image_sha256"] = hashes[example.sample_key]
        sample_records.append(record)
    _path, plan = create_plan(
        run_dir,
        samples=sample_records,
        seed=seed,
        sampling_method=sampling_method,
    )
    return plan, selected, hashes, False


def offline_preflight(
    *,
    examples: list[EvalExample],
    hashes: dict[str, str],
    model: ModelSpec,
    provider_policy: ProviderPolicy,
) -> dict:
    """Exercise image encoding, prompt construction, and parsing without network access."""
    backend = FakeBackend()
    for example in examples:
        request = sanitized_request(
            example,
            image_detail=model.image_detail,
            image_sha256=hashes[example.sample_key],
        )
        serialized = json.dumps(request, sort_keys=True)
        forbidden_fields = frozenset({"gold", "gold_answer", "expected_answer", "label"})
        if "base64," in serialized or _contains_forbidden_field(request, forbidden_fields):
            raise AssertionError("sanitized request leaks an image payload or a labeled gold field")
        response = backend.complete(
            model=model,
            messages=messages(example, image_detail=model.image_detail),
            candidates=example.candidates,
            provider_policy=provider_policy,
        )
        parse_response(response.content, example.candidates)
    return {
        "status": "passed",
        "network_used": False,
        "examples_checked": len(examples),
        "model_id": model.model_id,
    }


def execute_plan(
    *,
    run_dir: Path,
    plan: dict,
    examples: list[EvalExample],
    hashes: dict[str, str],
    model: ModelSpec,
    provider_policy: ProviderPolicy,
    backend: EvaluationBackend,
    retry_errors: bool,
    continue_on_api_error: bool = False,
) -> dict:
    """Execute one plan, preserving an event before and after every paid attempt."""
    events = read_events(run_dir)
    excluded = excluded_sample_keys(events, retry_errors=retry_errors)
    for example in examples:
        if example.sample_key in excluded:
            continue
        attempt_id = str(uuid.uuid4())
        request_record = sanitized_request(
            example,
            image_detail=model.image_detail,
            image_sha256=hashes[example.sample_key],
        )
        append_event(
            run_dir,
            {
                "event_id": str(uuid.uuid4()),
                "attempt_id": attempt_id,
                "plan_id": plan["plan_id"],
                "sample_key": example.sample_key,
                "status": "attempt_started",
                "request": request_record,
            },
        )
        try:
            response = backend.complete(
                model=model,
                messages=messages(example, image_detail=model.image_detail),
                candidates=example.candidates,
                provider_policy=provider_policy,
            )
        except KeyboardInterrupt:
            raise
        except ProviderResponseError as exc:
            response = exc.response
            append_event(
                run_dir,
                {
                    "event_id": str(uuid.uuid4()),
                    "attempt_id": attempt_id,
                    "plan_id": plan["plan_id"],
                    "sample_key": example.sample_key,
                    "status": "api_error",
                    "failure_stage": "provider_response_validation",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:4000],
                    "served_model": response.served_model,
                    "response_id": response.response_id,
                    "provider": response.provider,
                    "usage": response.usage,
                    "raw_response": response.raw_response,
                    "response_content": response.content,
                },
            )
            if not continue_on_api_error:
                break
            continue
        except Exception as exc:  # preserve provider failures without losing prior results
            append_event(
                run_dir,
                {
                    "event_id": str(uuid.uuid4()),
                    "attempt_id": attempt_id,
                    "plan_id": plan["plan_id"],
                    "sample_key": example.sample_key,
                    "status": "api_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:4000],
                },
            )
            if not continue_on_api_error:
                break
            continue

        common = {
            "event_id": str(uuid.uuid4()),
            "attempt_id": attempt_id,
            "plan_id": plan["plan_id"],
            "sample_key": example.sample_key,
            "served_model": response.served_model,
            "response_id": response.response_id,
            "provider": response.provider,
            "usage": response.usage,
            "raw_response": response.raw_response,
        }
        if response.rationale is not None:
            common["rationale"] = response.rationale
        if response.refusal:
            append_event(
                run_dir,
                {**common, "status": "refused", "refusal": response.refusal},
            )
            continue
        try:
            parsed = parse_response(response.content, example.candidates)
        except ValueError as exc:
            append_event(
                run_dir,
                {
                    **common,
                    "status": "unparseable",
                    "parse_error": str(exc),
                    "response_content": response.content,
                    "gold_answer": example.answer,
                },
            )
            continue
        append_event(
            run_dir,
            {
                **common,
                "status": "scored",
                "prediction": parsed["answer"],
                "confidence": parsed["confidence"],
                "gold_answer": example.answer,
                "correct": parsed["answer"] == example.answer,
            },
        )
    return event_summary(read_events(run_dir))


def create_offline_fixture(root: Path) -> Path:
    """Create a tiny exported-dataset fixture for CLI smoke tests."""
    dataset = root / "offline-smoke-dataset"
    image_dir = dataset / "images"
    image_dir.mkdir(parents=True)
    rows = []
    for index, (color, answer) in enumerate((("red", "red"), ("blue", "blue"))):
        image = image_dir / f"scene-{index:03d}.png"
        Image.new("RGB", (32, 32), color=color).save(image)
        rows.append(
            {
                "schema_version": "candidate-vlm-manifest-0.1.0",
                "example_id": f"smoke-{index:03d}",
                "source_record_id": "offline-smoke-candidate",
                "campaign_id": "offline-smoke",
                "candidate_id": "candidate_001",
                "image_path": image.relative_to(dataset).as_posix(),
                "question": "What color fills the image?",
                "answer": answer,
                "candidates": ["red", "blue"],
                "prompt_family": "closed_choice",
                "margin": None,
                "quarantined": False,
                "development_only": True,
            }
        )
    manifest = dataset / "vlm_manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return dataset


def full_offline_smoke(*, input_paths: list[Path] | None = None) -> dict:
    """Run discovery through append-only scoring with no environment key or network."""
    with tempfile.TemporaryDirectory(prefix="vlm-api-smoke-") as temporary:
        root = Path(temporary)
        inputs = input_paths or [create_offline_fixture(root)]
        examples, snapshots = load_inputs(inputs)
        selected = examples[: min(3, len(examples))]
        hashes = verify_selected_images(selected)
        model = ModelSpec("offline-smoke", "offline/fake-vlm", None, 64, "low")
        policy = ProviderPolicy()
        preflight = offline_preflight(
            examples=selected,
            hashes=hashes,
            model=model,
            provider_policy=policy,
        )
        run_dir = root / "run"
        plan, planned, planned_hashes, _resumed = prepare_plan(
            run_dir=run_dir,
            examples=examples,
            snapshots=snapshots,
            model=model,
            provider_policy=policy,
            sample_size=len(selected),
            seed=42,
            sampling_method="uniform-example",
            retry_errors=False,
        )
        summary = execute_plan(
            run_dir=run_dir,
            plan=plan,
            examples=planned,
            hashes=planned_hashes,
            model=model,
            provider_policy=policy,
            backend=FakeBackend(),
            retry_errors=False,
        )
        return {"preflight": preflight, "summary": summary, "network_used": False}
