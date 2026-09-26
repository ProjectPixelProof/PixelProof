"""Catalogue and validate immutable candidate-world review packets.

The candidate archive is deliberately separate from frozen foundry seed sets and
from the human-admitted ``worlds/`` bank. Review packets are historical evidence:
cataloguing or replaying one never changes seed membership.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from question_foundry.registry import canonical_json, sha256_candidate_tree
from question_foundry.render_runtime import DEFAULT_RENDER_RUNTIME, supported_render_runtimes

CATALOGUE_SCHEMA_VERSION = "candidate-catalogue-0.1.0"
VLM_MANIFEST_SCHEMA_VERSION = "candidate-vlm-manifest-0.1.0"

_REQUIRED_MANIFEST_FIELDS = frozenset(
    {
        "answer",
        "candidates",
        "example_id",
        "image_path",
        "margin",
        "prompt_family",
        "quarantined",
        "question",
        "scene_id",
    }
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_component(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"{field} must be one safe path component: {value!r}")
    return value


def discover_review_cohorts(studies_root: Path) -> list[Path]:
    """Return canonical public review cohorts without outer-curator copies."""

    return sorted(
        path
        for path in studies_root.glob("**/review_packet/cohort.json")
        if "outer-curator-v0" not in path.parts
    )


def _campaign_render_runtime(cohort_path: Path, campaign_id: str) -> str:
    """Recover renderer provenance from the archive collection envelope."""

    collection_path = cohort_path.parents[2] / "collection.json"
    if not collection_path.is_file():
        return DEFAULT_RENDER_RUNTIME
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    rows = [
        row
        for row in collection.get("campaigns", [])
        if isinstance(row, dict) and row.get("campaign_id") == campaign_id
    ]
    if len(rows) != 1:
        raise ValueError(
            f"{collection_path}: expected one campaign row for {campaign_id}, found {len(rows)}"
        )
    runtime = rows[0].get("render_runtime", DEFAULT_RENDER_RUNTIME)
    if runtime not in supported_render_runtimes():
        raise ValueError(f"{collection_path}: unsupported render_runtime {runtime!r}")
    return runtime


def build_candidate_catalogue(repo_root: Path) -> list[dict]:
    """Index every complete candidate exported into a canonical review packet."""

    repo_root = repo_root.resolve()
    records: list[dict] = []
    record_ids: set[str] = set()
    for cohort_path in discover_review_cohorts(repo_root / "studies"):
        cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
        campaign_id = _safe_component(cohort.get("campaign_id"), "campaign_id")
        render_runtime = _campaign_render_runtime(cohort_path, campaign_id)
        candidates = cohort.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"{cohort_path}: candidates must be an array")
        if cohort.get("candidate_count") != len(candidates):
            raise ValueError(f"{cohort_path}: candidate_count does not match candidates")

        packet = cohort_path.parent
        for source in candidates:
            if not isinstance(source, dict):
                raise ValueError(f"{cohort_path}: candidate row must be an object")
            blind_id = _safe_component(source.get("blind_id"), "blind_id")
            candidate_root = packet / "candidates" / blind_id / "candidate"
            gate_path = packet / "candidates" / blind_id / "gate_report.json"
            audit_path = packet / "candidates" / blind_id / "HUMAN_AUDIT.md"
            metadata_path = candidate_root / "candidate.json"
            required_paths = (candidate_root, gate_path, audit_path, metadata_path)
            missing = [str(path) for path in required_paths if not path.exists()]
            if missing:
                raise ValueError(f"{cohort_path}: incomplete candidate {blind_id}: {missing}")

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            candidate_id = _safe_component(metadata.get("candidate_id"), "candidate_id")
            record_id = f"{campaign_id}:{blind_id}"
            if record_id in record_ids:
                raise ValueError(f"duplicate candidate catalogue record: {record_id}")
            record_ids.add(record_id)

            record = {
                "schema_version": CATALOGUE_SCHEMA_VERSION,
                "record_id": record_id,
                "campaign_id": campaign_id,
                "blind_id": blind_id,
                "candidate_id": candidate_id,
                "title": str(metadata.get("title") or ""),
                "question": str(metadata.get("question") or ""),
                "task_signature": metadata.get("task_signature") or {},
                "candidate_path": candidate_root.relative_to(repo_root).as_posix(),
                "gate_report_path": gate_path.relative_to(repo_root).as_posix(),
                "audit_path": audit_path.relative_to(repo_root).as_posix(),
                # Source hashes identify the unblinded verifier artifact. The
                # archive hashes identify the public, identity-scrubbed packet.
                "source_candidate_sha256": source["candidate_sha256"],
                "archive_candidate_sha256": sha256_candidate_tree(candidate_root),
                "source_gate_report_sha256": source["gate_report_sha256"],
                "archive_gate_report_sha256": _sha256_file(gate_path),
                "mechanically_eligible": bool(source.get("mechanically_eligible")),
                "registry_distinct": bool(source.get("registry_distinct")),
                "semantic_review_ready": bool(source.get("semantic_review_ready")),
                "semantic_duplicate_risk": source.get("semantic_duplicate_risk", "not_evaluated"),
                "human_admission": "pending",
                "canonical_world_member": False,
                "seed_member": False,
                "render_runtime": render_runtime,
            }
            records.append(record)
    return sorted(records, key=lambda row: row["record_id"])


def catalogue_jsonl(records: Iterable[dict]) -> str:
    return "".join(canonical_json(record) + "\n" for record in records)


def read_candidate_catalogue(path: Path) -> list[dict]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON") from exc
        if record.get("schema_version") != CATALOGUE_SCHEMA_VERSION:
            raise ValueError(f"{path}:{number}: unsupported catalogue schema")
        records.append(record)
    ids = [record.get("record_id") for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate record_id")
    return records


def verify_archived_record(repo_root: Path, record: dict) -> Path:
    """Fail if a catalogued public artifact has changed since indexing."""

    repo_root = repo_root.resolve()
    candidate_root = (repo_root / record["candidate_path"]).resolve()
    gate_path = (repo_root / record["gate_report_path"]).resolve()
    for path in (candidate_root, gate_path):
        if repo_root not in path.parents:
            raise ValueError(f"catalogue path escapes repository: {path}")
    if sha256_candidate_tree(candidate_root) != record["archive_candidate_sha256"]:
        raise ValueError(f"candidate archive hash mismatch: {record['record_id']}")
    if _sha256_file(gate_path) != record["archive_gate_report_sha256"]:
        raise ValueError(f"gate-report archive hash mismatch: {record['record_id']}")
    return candidate_root


def validate_generated_dataset(dataset: Path, *, expected_scenes: int) -> dict:
    """Validate the portable surface shared by all archived candidate generators."""

    dataset = dataset.resolve()
    manifest_path = dataset / "manifest.jsonl"
    if not manifest_path.is_file():
        raise ValueError(f"generated dataset has no manifest: {manifest_path}")
    rows = []
    for number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = sorted(_REQUIRED_MANIFEST_FIELDS - row.keys())
        if missing:
            raise ValueError(f"{manifest_path}:{number}: missing fields {missing}")
        candidates = row["candidates"]
        if not isinstance(candidates, list) or row["answer"] not in candidates:
            raise ValueError(f"{manifest_path}:{number}: answer is not in closed candidates")
        image_path = (dataset / row["image_path"]).resolve()
        if dataset not in image_path.parents or not image_path.is_file():
            raise ValueError(f"{manifest_path}:{number}: unsafe or missing image")
        rows.append(row)
    if not rows:
        raise ValueError(f"{manifest_path}: manifest is empty")

    scene_ids = {row["scene_id"] for row in rows}
    if len(scene_ids) != expected_scenes:
        raise ValueError(
            f"{manifest_path}: expected {expected_scenes} scenes, found {len(scene_ids)}"
        )
    eligible = [row for row in rows if not bool(row["quarantined"])]
    if not eligible:
        raise ValueError(f"{manifest_path}: every generated row is quarantined")
    return {
        "scene_count": len(scene_ids),
        "row_count": len(rows),
        "eligible_row_count": len(eligible),
        "quarantined_row_count": len(rows) - len(eligible),
        "prompt_family_count": len({row["prompt_family"] for row in rows}),
    }


def write_development_vlm_manifest(
    *,
    source_manifest: Path,
    destination: Path,
    record: dict,
) -> int:
    """Write non-quarantined rows for later developmental VLM evaluation."""

    written = 0
    lines = []
    for line in source_manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if bool(row["quarantined"]):
            continue
        lines.append(
            canonical_json(
                {
                    "schema_version": VLM_MANIFEST_SCHEMA_VERSION,
                    "example_id": f"{record['record_id']}::{row['example_id']}",
                    "source_record_id": record["record_id"],
                    "campaign_id": record["campaign_id"],
                    "candidate_id": record["candidate_id"],
                    "image_path": row["image_path"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "candidates": row["candidates"],
                    "prompt_family": row["prompt_family"],
                    "margin": row["margin"],
                    "quarantined": False,
                    "development_only": True,
                    "human_admission": record["human_admission"],
                }
            )
        )
        written += 1
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return written
