"""Archive protected-valid campaign working seeds as blinded review cohorts.

Campaign-local working seeds are disposable controller state. This module
validates every accepted row against its immutable Harbor trial, then reuses the
canonical review-packet builder so executable candidates enter the durable
candidate catalogue without becoming foundry seeds or admitted worlds.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from question_foundry.candidate_contract import task_signature_errors
from question_foundry.registry import sha256_candidate_tree
from question_foundry.render_runtime import campaign_render_runtime
from question_foundry.review_packet import ReviewPacketResult, materialize_review_packet


@dataclass(frozen=True)
class WorkingSeedRun:
    """A validated controller run and the Harbor jobs backing its working seed."""

    controller_root: Path
    campaign_id: str
    source_revision: str
    state_sha256: str
    source_candidate_count: int
    candidate_count: int
    job_roots: tuple[Path, ...]
    comparison_eligible: bool
    hypothesis_profile: str | None
    render_runtime: str
    exclusions: tuple[dict[str, str], ...]
    generated_only: bool = False
    skipped_bootstrap_count: int = 0


@dataclass(frozen=True)
class WorkingSeedArchiveResult:
    """One atomic collection of one or more blinded campaign cohorts."""

    root: Path
    candidate_count: int
    campaigns: tuple[ReviewPacketResult, ...]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_component(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"{field} must be one safe path component: {value!r}")
    return value


def _controller_root(run_root: Path) -> Path:
    run_root = run_root.resolve()
    if (run_root / "state.json").is_file() and (run_root / "jobs").is_dir():
        return run_root
    controller = run_root / "controller"
    if (controller / "state.json").is_file() and (controller / "jobs").is_dir():
        return controller
    raise ValueError(f"run has no controller state/jobs layout: {run_root}")


def _resolve_inside(root: Path, relative: object, field: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{field} must be a non-empty relative path")
    candidate = (root / relative).resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError(f"{field} escapes controller root: {relative!r}")
    return candidate


def _candidate_gate_result(report: dict, candidate_id: str) -> dict:
    rows = [
        row
        for row in report.get("candidate_results", [])
        if isinstance(row, dict) and row.get("candidate_id") == candidate_id
    ]
    if len(rows) != 1:
        raise ValueError(f"gate report has {len(rows)} rows for {candidate_id}")
    return rows[0]


def load_archive_exclusions(path: Path) -> dict[str, Any]:
    """Load a run-bound, forward-only exclusion record."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "working-seed-archive-exclusions-0.1.0":
        raise ValueError(f"unsupported archive-exclusion schema: {path}")
    campaign_id = _safe_component(payload.get("campaign_id"), "campaign_id")
    state_hash = payload.get("source_state_sha256")
    if not isinstance(state_hash, str) or len(state_hash) != 64:
        raise ValueError(f"{path}: source_state_sha256 must be a SHA-256 digest")
    rows = payload.get("exclusions")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: exclusions must be a non-empty array")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "candidate_id",
            "reason",
            "reason_code",
        }:
            raise ValueError(f"{path}: every exclusion must contain exact documented fields")
        candidate_id = _safe_component(row.get("candidate_id"), "candidate_id")
        if candidate_id in seen:
            raise ValueError(f"{path}: duplicate exclusion for {candidate_id}")
        seen.add(candidate_id)
        if row.get("reason_code") != "candidate_contract_false_positive":
            raise ValueError(f"{path}: unsupported exclusion reason for {candidate_id}")
        if len(str(row.get("reason") or "").strip()) < 40:
            raise ValueError(f"{path}: exclusion reason is too short for {candidate_id}")
    return {**payload, "campaign_id": campaign_id}


def inspect_working_seed_run(
    run_root: Path,
    *,
    exclusions: dict[str, Any] | None = None,
    generated_only: bool = False,
) -> WorkingSeedRun:
    """Fail closed unless every working-seed row matches protected trial evidence."""

    controller = _controller_root(run_root)
    state_path = controller / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    campaign_id = _safe_component(state.get("campaign_id"), "campaign_id")
    source_revision = str(state.get("source_revision") or "")
    if not source_revision:
        raise ValueError(f"{state_path}: source_revision is missing")
    working_seed = state.get("working_seed")
    if not isinstance(working_seed, list) or not working_seed:
        raise ValueError(f"{state_path}: working_seed must be a non-empty array")
    selected_seed = working_seed
    skipped_bootstrap_count = 0
    if generated_only:
        selected_seed = [
            row
            for row in working_seed
            if isinstance(row, dict)
            and int(row.get("episode_index") or 0) > 0
            and not row.get("bootstrap_source")
        ]
        skipped_bootstrap_count = len(working_seed) - len(selected_seed)
        if not selected_seed:
            raise ValueError(f"{state_path}: no generated working-seed rows remain")

    exclusion_rows = exclusions.get("exclusions", []) if exclusions else []
    if exclusions:
        if exclusions.get("campaign_id") != campaign_id:
            raise ValueError(f"archive exclusions target the wrong campaign: {campaign_id}")
        if exclusions.get("source_state_sha256") != _sha256_file(state_path):
            raise ValueError(f"archive exclusions do not bind the current state: {state_path}")
    exclusions_by_id = {str(row["candidate_id"]): row for row in exclusion_rows}
    observed_exclusions: set[str] = set()

    job_roots: list[Path] = []
    candidate_ids: set[str] = set()
    candidate_hashes: set[str] = set()
    for row in selected_seed:
        if not isinstance(row, dict):
            raise ValueError(f"{state_path}: working_seed row must be an object")
        candidate_id = _safe_component(row.get("candidate_id"), "candidate_id")
        source_job = _safe_component(row.get("source_job"), "source_job")
        source_trial = _safe_component(row.get("source_trial"), "source_trial")
        if candidate_id in candidate_ids:
            raise ValueError(f"{state_path}: duplicate working-seed candidate {candidate_id}")
        candidate_ids.add(candidate_id)

        required_true = (
            "mechanically_eligible",
            "registry_distinct",
            "reported_submission_valid",
            "submission_valid",
            "verifier_completed",
        )
        failed_invariants = [field for field in required_true if row.get(field) is not True]
        if failed_invariants:
            raise ValueError(
                f"{state_path}: {candidate_id} lacks protected-valid transition fields "
                f"{failed_invariants}"
            )
        if row.get("canonical_admitted") is not False:
            raise ValueError(f"{state_path}: {candidate_id} has invalid canonical admission state")
        if row.get("failed_gates") not in ([], None):
            raise ValueError(f"{state_path}: {candidate_id} records failed gates")

        artifact = _resolve_inside(controller, row.get("artifact_path"), "artifact_path")
        if not artifact.is_dir():
            raise ValueError(f"working-seed artifact is missing: {artifact}")
        metadata_path = artifact / "candidate.json"
        if not metadata_path.is_file():
            raise ValueError(f"working-seed candidate metadata is missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("candidate_id") != candidate_id:
            raise ValueError(f"working-seed candidate ID disagrees for {candidate_id}")
        signature_failures = task_signature_errors(metadata)
        excluded = exclusions_by_id.get(candidate_id)
        if excluded is None and signature_failures:
            raise ValueError(
                f"working-seed candidate violates frozen contract for {candidate_id}: "
                + "; ".join(signature_failures)
            )
        if excluded is not None:
            if not signature_failures:
                raise ValueError(f"refusing to exclude schema-conformant candidate {candidate_id}")
            observed_exclusions.add(candidate_id)

        expected_candidate_hash = str(row.get("candidate_sha256") or "")
        artifact_hash = sha256_candidate_tree(artifact)
        if artifact_hash != expected_candidate_hash:
            raise ValueError(f"working-seed candidate hash mismatch for {candidate_id}")
        if artifact_hash in candidate_hashes:
            raise ValueError(f"duplicate working-seed candidate content for {candidate_id}")
        candidate_hashes.add(artifact_hash)

        job = controller / "jobs" / source_job
        trial = job / source_trial
        if not (job / "result.json").is_file() or not (trial / "result.json").is_file():
            raise ValueError(f"source Harbor job/trial is incomplete for {candidate_id}")
        trial_dirs = sorted(
            path for path in job.iterdir() if path.is_dir() and (path / "result.json").is_file()
        )
        if trial_dirs != [trial]:
            raise ValueError(f"source job does not resolve uniquely to {source_trial}")

        submission = trial / "verifier/artifacts/submission"
        portfolio_path = submission / "portfolio.json"
        source_candidate = submission / "candidates" / candidate_id
        gate_path = trial / "verifier/gate_report.json"
        for required in (portfolio_path, source_candidate, gate_path):
            if not required.exists():
                raise ValueError(f"source trial artifact is missing: {required}")
        portfolio = json.loads(portfolio_path.read_text(encoding="utf-8"))
        if portfolio.get("candidate_ids") != [candidate_id]:
            raise ValueError(
                f"source episode must contain exactly working-seed candidate {candidate_id}"
            )
        if sha256_candidate_tree(source_candidate) != artifact_hash:
            raise ValueError(f"source/working-seed candidate differs for {candidate_id}")

        expected_gate_hash = str(row.get("gate_report_sha256") or "")
        if _sha256_file(gate_path) != expected_gate_hash:
            raise ValueError(f"protected gate-report hash mismatch for {candidate_id}")
        report = json.loads(gate_path.read_text(encoding="utf-8"))
        if report.get("campaign_id") != campaign_id:
            raise ValueError(f"protected gate report has wrong campaign for {candidate_id}")
        gate_row = _candidate_gate_result(report, candidate_id)
        if gate_row.get("mechanically_eligible") is not True:
            raise ValueError(f"protected gate report rejects {candidate_id}")
        if gate_row.get("registry_distinct") is not True:
            raise ValueError(f"protected registry check rejects {candidate_id}")
        if gate_row.get("failures") not in ({}, None):
            raise ValueError(f"protected gate report records failures for {candidate_id}")
        if excluded is None:
            job_roots.append(job)

    missing_exclusions = sorted(set(exclusions_by_id) - observed_exclusions)
    if missing_exclusions:
        raise ValueError(f"archive exclusions did not match working seed: {missing_exclusions}")

    budget = state.get("budget_assessment") or {}
    compiled_campaign = controller.parent / "compiled-campaign.toml"
    render_runtime = "python-pillow@0.1.0"
    if compiled_campaign.is_file():
        render_runtime = campaign_render_runtime(
            tomllib.loads(compiled_campaign.read_text(encoding="utf-8"))
        )
    return WorkingSeedRun(
        controller_root=controller,
        campaign_id=campaign_id,
        source_revision=source_revision,
        state_sha256=_sha256_file(state_path),
        source_candidate_count=len(working_seed),
        candidate_count=len(job_roots),
        job_roots=tuple(job_roots),
        comparison_eligible=bool(budget.get("comparison_eligible", False)),
        hypothesis_profile=(
            str(state["hypothesis_profile"]) if state.get("hypothesis_profile") else None
        ),
        render_runtime=render_runtime,
        exclusions=tuple(dict(row) for row in exclusion_rows),
        generated_only=generated_only,
        skipped_bootstrap_count=skipped_bootstrap_count,
    )


def inspect_working_seed_runs(
    run_roots: list[Path],
    *,
    exclusion_files: list[Path] | None = None,
    generated_only: bool = False,
) -> tuple[WorkingSeedRun, ...]:
    """Validate all inputs before materializing any durable output."""

    if not run_roots:
        raise ValueError("at least one run root is required")
    exclusion_records = [load_archive_exclusions(path) for path in (exclusion_files or [])]
    exclusions_by_campaign = {row["campaign_id"]: row for row in exclusion_records}
    if len(exclusions_by_campaign) != len(exclusion_records):
        raise ValueError("duplicate campaign_id across archive-exclusion files")
    runs_list = []
    for path in run_roots:
        controller = _controller_root(path)
        state = json.loads((controller / "state.json").read_text(encoding="utf-8"))
        campaign_id = _safe_component(state.get("campaign_id"), "campaign_id")
        runs_list.append(
            inspect_working_seed_run(
                path,
                exclusions=exclusions_by_campaign.pop(campaign_id, None),
                generated_only=generated_only,
            )
        )
    if exclusions_by_campaign:
        raise ValueError(
            f"archive exclusions supplied for absent campaigns: {sorted(exclusions_by_campaign)}"
        )
    runs = tuple(runs_list)
    campaign_ids = [run.campaign_id for run in runs]
    if len(campaign_ids) != len(set(campaign_ids)):
        raise ValueError("duplicate campaign_id across working-seed runs")
    return runs


def collect_working_seed_archives(
    *,
    repo_root: Path,
    runs: tuple[WorkingSeedRun, ...],
    output_root: Path,
) -> WorkingSeedArchiveResult:
    """Atomically materialize one blinded review cohort per validated campaign."""

    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"working-seed archive target exists: {output_root}")
    studies_root = (repo_root / "studies").resolve()
    if studies_root not in output_root.parents:
        raise ValueError("working-seed archive output must live under studies/")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent))
    try:
        results = []
        manifest_rows = []
        for run in runs:
            result = materialize_review_packet(
                repo_root=repo_root,
                job_roots=list(run.job_roots),
                output_root=temporary / run.campaign_id,
                campaign_id=run.campaign_id,
            )
            if result.candidate_count != run.candidate_count:
                raise ValueError(
                    f"{run.campaign_id}: materialized {result.candidate_count} candidates, "
                    f"expected {run.candidate_count}"
                )
            results.append(result)
            manifest_rows.append(
                {
                    "campaign_id": run.campaign_id,
                    "candidate_count": run.candidate_count,
                    "source_working_seed_count": run.source_candidate_count,
                    "generated_only": run.generated_only,
                    "skipped_bootstrap_count": run.skipped_bootstrap_count,
                    "excluded_candidates": list(run.exclusions),
                    "comparison_eligible": run.comparison_eligible,
                    "hypothesis_profile": run.hypothesis_profile,
                    "render_runtime": run.render_runtime,
                    "packet_sha256": result.packet_sha256,
                    "source_revision": run.source_revision,
                    "source_state_sha256": run.state_sha256,
                }
            )
        total = sum(run.candidate_count for run in runs)
        (temporary / "collection.json").write_text(
            json.dumps(
                {
                    "schema_version": "working-seed-archive-collection-0.1.0",
                    "kind": "blinded-working-seed-candidate-archive",
                    "candidate_count": total,
                    "campaign_count": len(runs),
                    "canonical_admissions": 0,
                    "human_admissions": 0,
                    "seed_memberships": 0,
                    "campaigns": manifest_rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return WorkingSeedArchiveResult(
        root=output_root,
        candidate_count=sum(run.candidate_count for run in runs),
        campaigns=tuple(results),
    )
