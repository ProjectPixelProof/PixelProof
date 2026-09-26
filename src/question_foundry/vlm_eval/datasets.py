"""Discovery and validation for exported developmental VLM manifests."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import mimetypes
from pathlib import Path

from PIL import Image

SUPPORTED_IMAGE_MIME = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})
MIME_IMAGE_FORMATS = {
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/webp": "WEBP",
    "image/gif": "GIF",
}
REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "example_id",
        "source_record_id",
        "campaign_id",
        "candidate_id",
        "image_path",
        "question",
        "answer",
        "candidates",
        "prompt_family",
        "quarantined",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True)
class DatasetSnapshot:
    manifest_path: Path
    manifest_sha256: str
    dataset_id: str
    row_count: int
    source: dict | None = None

    def as_dict(self) -> dict:
        output = {
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "dataset_id": self.dataset_id,
            "row_count": self.row_count,
        }
        if self.source is not None:
            output["source"] = self.source
        return output


@dataclasses.dataclass(frozen=True)
class EvalExample:
    sample_key: str
    manifest_path: Path
    manifest_sha256: str
    dataset_dir: Path
    dataset_id: str
    example_id: str
    source_record_id: str
    campaign_id: str
    candidate_id: str
    image_path: Path
    question: str
    answer: str
    candidates: tuple[str, ...]
    prompt_family: str
    margin: float | int | None

    def public_metadata(self) -> dict:
        return {
            "sample_key": self.sample_key,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "dataset_id": self.dataset_id,
            "example_id": self.example_id,
            "source_record_id": self.source_record_id,
            "campaign_id": self.campaign_id,
            "candidate_id": self.candidate_id,
            "image_path": str(self.image_path),
            "question": self.question,
            "candidates": list(self.candidates),
            "prompt_family": self.prompt_family,
            "margin": self.margin,
        }


def discover_manifests(inputs: list[Path]) -> list[Path]:
    manifests = set()
    for raw in inputs:
        path = raw.expanduser().resolve()
        if path.is_file():
            if path.name != "vlm_manifest.jsonl":
                raise ValueError(f"input file is not a vlm_manifest.jsonl: {path}")
            manifests.add(path)
        elif path.is_dir():
            own = path / "vlm_manifest.jsonl"
            if own.is_file():
                manifests.add(own.resolve())
            else:
                manifests.update(item.resolve() for item in path.rglob("vlm_manifest.jsonl"))
        else:
            raise ValueError(f"input path does not exist: {path}")
    if not manifests:
        joined = ", ".join(str(path) for path in inputs)
        raise ValueError(f"no vlm_manifest.jsonl files found under: {joined}")
    return sorted(manifests)


def _safe_image(dataset_dir: Path, relative: object, *, source: str) -> tuple[Path, str]:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{source}: image_path must be a non-empty string")
    image_path = (dataset_dir / relative).resolve()
    if dataset_dir != image_path.parent and dataset_dir not in image_path.parents:
        raise ValueError(f"{source}: image path escapes dataset: {relative}")
    if not image_path.is_file():
        raise ValueError(f"{source}: image is missing: {image_path}")
    mime, _ = mimetypes.guess_type(image_path.name)
    if mime not in SUPPORTED_IMAGE_MIME:
        raise ValueError(f"{source}: unsupported image type {mime!r}: {image_path}")
    return image_path, mime


def _required_string(row: dict, field: str, source: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: {field} must be a non-empty string")
    return value


def _closed_choice_text(value: object, *, source: str, field: str) -> str:
    """Normalize JSON string/number labels to the text sent to a VLM."""
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return json.dumps(value, allow_nan=False, separators=(",", ":"))
        except ValueError as exc:
            raise ValueError(f"{source}: {field} must be a finite JSON number") from exc
    raise ValueError(f"{source}: {field} must be a non-empty string or JSON number")


def load_examples(manifests: list[Path]) -> tuple[list[EvalExample], list[DatasetSnapshot]]:
    examples = []
    snapshots = []
    seen_keys = set()
    for manifest in manifests:
        dataset_dir = manifest.parent.resolve()
        manifest_sha = sha256_file(manifest)
        rows = []
        dataset_ids = set()
        for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            source = f"{manifest}:{number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{source}: row must be an object")
            missing = sorted(REQUIRED_FIELDS - row.keys())
            if missing:
                raise ValueError(f"{source}: missing fields {missing}")
            if row.get("schema_version") != "candidate-vlm-manifest-0.1.0":
                raise ValueError(f"{source}: unsupported schema_version")
            if row.get("quarantined") is not False:
                raise ValueError(f"{source}: downstream manifest contains quarantined row")
            candidates = row.get("candidates")
            if not isinstance(candidates, list) or len(candidates) < 2:
                raise ValueError(f"{source}: candidates must contain at least two answers")
            candidates = [
                _closed_choice_text(item, source=source, field="candidate") for item in candidates
            ]
            if len(candidates) != len(set(candidates)):
                raise ValueError(f"{source}: candidates must be unique")
            answer = _closed_choice_text(row.get("answer"), source=source, field="answer")
            if answer not in candidates:
                raise ValueError(f"{source}: gold answer is outside the closed candidates")
            image_path, _mime = _safe_image(dataset_dir, row.get("image_path"), source=source)
            example_id = _required_string(row, "example_id", source)
            dataset_id = _required_string(row, "source_record_id", source)
            sample_key = hashlib.sha256(f"{manifest_sha}\0{example_id}".encode()).hexdigest()
            if sample_key in seen_keys:
                raise ValueError(f"duplicate sample identity: {source}")
            seen_keys.add(sample_key)
            dataset_ids.add(dataset_id)
            rows.append(
                EvalExample(
                    sample_key=sample_key,
                    manifest_path=manifest.resolve(),
                    manifest_sha256=manifest_sha,
                    dataset_dir=dataset_dir,
                    dataset_id=dataset_id,
                    example_id=example_id,
                    source_record_id=dataset_id,
                    campaign_id=_required_string(row, "campaign_id", source),
                    candidate_id=_required_string(row, "candidate_id", source),
                    image_path=image_path,
                    question=_required_string(row, "question", source),
                    answer=answer,
                    candidates=tuple(candidates),
                    prompt_family=_required_string(row, "prompt_family", source),
                    margin=row.get("margin"),
                )
            )
        if not rows:
            raise ValueError(f"manifest is empty: {manifest}")
        if len(dataset_ids) != 1:
            raise ValueError(f"manifest mixes source_record_id values: {manifest}")
        examples.extend(rows)
        snapshots.append(
            DatasetSnapshot(manifest.resolve(), manifest_sha, next(iter(dataset_ids)), len(rows))
        )
    return examples, snapshots


def verify_selected_images(examples: list[EvalExample]) -> dict[str, str]:
    hashes = {}
    for example in examples:
        mime, _ = mimetypes.guess_type(example.image_path.name)
        with Image.open(example.image_path) as image:
            if image.format != MIME_IMAGE_FORMATS.get(mime):
                raise ValueError(
                    f"image bytes do not match the filename MIME type: {example.image_path}"
                )
            image.verify()
        hashes[example.sample_key] = sha256_file(example.image_path)
    return hashes
