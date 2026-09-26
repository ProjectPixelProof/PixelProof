"""Build identity-blinded, content-hashed outer-review packets from Harbor jobs."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from question_foundry.registry import canonical_json, sha256_candidate_tree, sha256_tree
from question_foundry.semantic_novelty import candidate_text, fingerprint_similarity

_TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".py", ".toml", ".txt", ".html"}
_TRANSIENT_PARTS = {".pytest_cache", "__pycache__"}


@dataclass(frozen=True)
class ReviewPacketResult:
    root: Path
    public_packet: Path
    private_map: Path
    outer_task: Path
    packet_sha256: str
    candidate_count: int


def _trial_dirs(job: Path) -> list[Path]:
    return sorted(
        path for path in job.iterdir() if path.is_dir() and (path / "result.json").is_file()
    )


def _copy_blinded(source: Path, destination: Path, replacements: dict[str, str]) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part in _TRANSIENT_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() in {".pyc", ".pyo"}:
            continue
        if relative.as_posix() == "provenance.json":
            continue
        target = destination / relative
        if path.is_symlink():
            raise ValueError(f"candidate contains symlink: {path}")
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() in _TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8")
            for old, new in replacements.items():
                if old:
                    text = text.replace(old, new)
            target.write_text(text, encoding="utf-8")
        else:
            shutil.copy2(path, target)


def _candidate_content_hash(candidate: Path) -> str:
    """Hash authored candidate content while excluding verifier runtime caches."""
    return sha256_candidate_tree(candidate)


def _candidate_gate_result(report: dict, candidate_id: str) -> dict:
    matches = [
        item
        for item in report.get("candidate_results", [])
        if item.get("candidate_id") == candidate_id
    ]
    if len(matches) != 1:
        raise ValueError(f"gate report has {len(matches)} rows for {candidate_id}")
    return matches[0]


def materialize_review_packet(
    *,
    repo_root: Path,
    job_roots: list[Path],
    output_root: Path,
    campaign_id: str,
) -> ReviewPacketResult:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"review packet target exists: {output_root}")

    collected = []
    for job in job_roots:
        job = job.resolve()
        for trial in _trial_dirs(job):
            result = json.loads((trial / "result.json").read_text(encoding="utf-8"))
            exception = result.get("exception_info")
            exception_type = (exception or {}).get("exception_type")
            if exception is not None and exception_type != "AgentTimeoutError":
                raise ValueError(f"trial has infrastructure exception: {trial}")
            submission = trial / "verifier/artifacts/submission"
            report_path = trial / "verifier/artifacts/gate_report.json"
            if not submission.is_dir() or not report_path.is_file():
                raise ValueError(f"trial lacks preserved submission/gate report: {trial}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("campaign_id") != campaign_id:
                raise ValueError(f"trial campaign differs from {campaign_id}: {trial}")
            portfolio = json.loads((submission / "portfolio.json").read_text(encoding="utf-8"))
            agent_info = result.get("agent_info") or {}
            model_info = agent_info.get("model_info") or {}
            for candidate_id in portfolio["candidate_ids"]:
                candidate = submission / "candidates" / candidate_id
                metadata = json.loads((candidate / "candidate.json").read_text(encoding="utf-8"))
                collected.append(
                    {
                        "candidate_id": candidate_id,
                        "candidate": candidate,
                        "candidate_sha256": _candidate_content_hash(candidate),
                        "gate_result": _candidate_gate_result(report, candidate_id),
                        "gate_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                        "trial": trial,
                        "agent": str(agent_info.get("name") or ""),
                        "model": str(model_info.get("name") or ""),
                        "agent_termination": (
                            "timeout" if exception_type == "AgentTimeoutError" else "completed"
                        ),
                        "metadata": metadata,
                    }
                )
    if not collected:
        raise ValueError("no candidates found")
    hashes = [item["candidate_sha256"] for item in collected]
    if len(hashes) != len(set(hashes)):
        raise ValueError("duplicate candidate content hashes in review cohort")
    collected.sort(key=lambda item: item["candidate_sha256"])

    public = output_root / "review_packet"
    private = output_root / "private"
    public_candidates = public / "candidates"
    public_candidates.mkdir(parents=True)
    private.mkdir(parents=True)
    public_rows = []
    private_rows = []
    semantic_metadata: dict[str, dict] = {}
    cells: dict[tuple[str, str, str], list[str]] = {}
    for index, item in enumerate(collected, start=1):
        blind_id = f"candidate_{index:03d}"
        destination = public_candidates / blind_id / "candidate"
        replacements = {
            item["agent"]: "[BLINDED_AGENT]",
            item["model"]: "[BLINDED_MODEL]",
            item["trial"].name: "[BLINDED_TRIAL]",
        }
        _copy_blinded(item["candidate"], destination, replacements)
        gate_result = json.loads(canonical_json(item["gate_result"]))
        gate_result["candidate_id"] = blind_id
        gate_destination = public_candidates / blind_id / "gate_report.json"
        gate_destination.write_text(
            json.dumps(gate_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        audit = public_candidates / blind_id / "HUMAN_AUDIT.md"
        semantic_notes = (gate_result.get("notes") or {}).get("semantic") or {}
        known_neighbors = semantic_notes.get("known_neighbors") or []
        neighbor_lines = "\n".join(
            f"- `{neighbor.get('world_id')}`: score {neighbor.get('score')}"
            for neighbor in known_neighbors
        )
        if not neighbor_lines:
            neighbor_lines = "- No machine-retrieved neighbors were recorded."
        audit.write_text(
            "# Human audit\n\n"
            "Reviewer: UNSIGNED\n\n"
            f"Candidate: `{blind_id}`\n\n"
            "- [ ] Answerable from pixels alone\n"
            "- [ ] Boundary is visible\n"
            "- [ ] Prompt families are semantically stable\n"
            "- [ ] Quarantine is appropriate\n"
            "- [ ] Scientifically distinct and useful\n"
            "- [ ] Nearest known mechanisms were inspected, not merely renamed\n"
            "- [ ] Negative-memory matches and sibling similarity were inspected\n"
            "- [ ] Verifier evidence is credible\n\n"
            "## Machine-retrieved known neighbors\n\n"
            f"{neighbor_lines}\n\n"
            "Machine similarity is diagnostic and cannot admit or reject a world.\n\n"
            "Decision: promote / repair / reject\n\n"
            "Rationale:\n\n"
            "Signed-off-by: ________________________\n",
            encoding="utf-8",
        )
        metadata = item["metadata"]
        semantic_metadata[blind_id] = metadata
        signature = metadata["task_signature"]
        cell = (
            signature["decision_var"],
            signature["structure"],
            signature["decision_type"],
        )
        cells.setdefault(cell, []).append(blind_id)
        public_rows.append(
            {
                "blind_id": blind_id,
                "candidate_sha256": item["candidate_sha256"],
                "gate_report_sha256": item["gate_report_sha256"],
                "task_signature": signature,
                "mechanically_eligible": bool(item["gate_result"]["mechanically_eligible"]),
                "registry_distinct": bool(item["gate_result"]["registry_distinct"]),
                "semantic_review_ready": bool(
                    item["gate_result"].get("semantic_review_ready", False)
                ),
                "semantic_duplicate_risk": item["gate_result"].get(
                    "semantic_duplicate_risk",
                    "not_evaluated",
                ),
                "known_mechanism_neighbors": known_neighbors,
                "negative_memory_neighbors": semantic_notes.get(
                    "negative_neighbors",
                    [],
                ),
                "agent_termination": item["agent_termination"],
            }
        )
        private_rows.append(
            {
                "blind_id": blind_id,
                "candidate_id": item["candidate_id"],
                "candidate_sha256": item["candidate_sha256"],
                "source_job": str(item["trial"].parent),
                "source_trial": item["trial"].name,
                "agent": item["agent"],
                "model": item["model"],
                "agent_termination": item["agent_termination"],
            }
        )

    collisions = [
        {"cell": list(cell), "blind_ids": ids}
        for cell, ids in sorted(cells.items())
        if len(ids) > 1
    ]
    semantic_edges = []
    blind_ids = sorted(semantic_metadata)
    for index, left_id in enumerate(blind_ids):
        left = semantic_metadata[left_id]
        left_fingerprint = left.get("mechanism_fingerprint")
        if not isinstance(left_fingerprint, dict):
            continue
        for right_id in blind_ids[index + 1 :]:
            right = semantic_metadata[right_id]
            right_fingerprint = right.get("mechanism_fingerprint")
            if not isinstance(right_fingerprint, dict):
                continue
            score = fingerprint_similarity(
                left_fingerprint,
                right_fingerprint,
                left_text=candidate_text(left),
                right_text=candidate_text(right),
            )
            if score >= 0.68:
                semantic_edges.append(
                    {
                        "left": left_id,
                        "right": right_id,
                        "score": score,
                    }
                )
    cohort = {
        "schema_version": "0.3.0",
        "campaign_id": campaign_id,
        "candidate_count": len(public_rows),
        "candidates": public_rows,
        "cohort_cell_collisions": collisions,
        "cohort_semantic_edges_at_0_68": semantic_edges,
        "semantic_similarity_is_diagnostic": True,
    }
    (public / "cohort.json").write_text(
        json.dumps(cohort, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (public / "README.md").write_text(
        f"# Blinded review packet: {campaign_id}\n\n"
        f"Candidates: {len(public_rows)}\n\n"
        "Inspect each candidate's code, gallery, gate report, and human-audit form. "
        "Identity mappings are intentionally outside this directory.\n",
        encoding="utf-8",
    )
    private_map = private / "blinding-map.json"
    private_map.write_text(
        json.dumps(
            {"schema_version": "0.2.0", "campaign_id": campaign_id, "mappings": private_rows},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    outer_template = repo_root / "harbor/datasets/outer-curator-v0"
    outer_dataset = output_root / "outer-curator-v0"
    shutil.copytree(outer_template, outer_dataset)
    outer_task = outer_dataset / "review-candidates"
    destination_packet = outer_task / "environment/review_packet"
    shutil.rmtree(destination_packet)
    shutil.copytree(public, destination_packet)

    packet_sha256 = sha256_tree(public)
    manifest = {
        "schema_version": "0.2.0",
        "kind": "blinded-review-packet",
        "campaign_id": campaign_id,
        "candidate_count": len(public_rows),
        "packet_sha256": packet_sha256,
        "public_packet": "review_packet",
        "private_map": "private/blinding-map.json",
        "outer_task": "outer-curator-v0/review-candidates",
    }
    (output_root / "materialization.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return ReviewPacketResult(
        root=output_root,
        public_packet=public,
        private_map=private_map,
        outer_task=outer_task,
        packet_sha256=packet_sha256,
        candidate_count=len(public_rows),
    )
