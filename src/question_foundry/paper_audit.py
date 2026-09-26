"""Prospective paper-run evidence records for audited foundry campaigns.

The controller owns these records. Agent-visible phase events establish only
when a milestone was observed; protected verifier outcomes remain authoritative
for scientific disposition.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from datetime import datetime
from pathlib import Path

from question_foundry.registry import canonical_json

RUN_CONFIG_SCHEMA = "paper-run-config-0.1.0"
LEDGER_SCHEMA = "paper-episode-ledger-0.1.0"
EVIDENCE_SCHEMA = "paper-run-evidence-manifest-0.1.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def seconds_between(start: str | None, finish: str | None) -> float | None:
    if not start or not finish:
        return None
    first = datetime.fromisoformat(start.replace("Z", "+00:00"))
    last = datetime.fromisoformat(finish.replace("Z", "+00:00"))
    return round(max(0.0, (last - first).total_seconds()), 6)


def harbor_lifecycle(result: dict) -> dict:
    """Normalize exact Harbor phase timestamps without inferring gaps."""

    phases = {}
    for source, name in (
        ("environment_setup", "environment_setup"),
        ("agent_setup", "agent_setup"),
        ("agent_execution", "agent_execution"),
        ("verifier", "protected_verification"),
    ):
        block = result.get(source) if isinstance(result.get(source), dict) else {}
        started = block.get("started_at")
        finished = block.get("finished_at")
        phases[name] = {
            "started_at": started,
            "finished_at": finished,
            "seconds": seconds_between(started, finished),
            "source": f"harbor.result.{source}",
        }
    phases["trial"] = {
        "started_at": result.get("started_at"),
        "finished_at": result.get("finished_at"),
        "seconds": seconds_between(result.get("started_at"), result.get("finished_at")),
        "source": "harbor.result",
    }
    return phases


def read_phase_events(trial: Path) -> dict:
    """Read the immutable v0.7 phase ledger from the recycled artifact copy."""

    candidates = (
        trial / "verifier/artifacts/process/phase-events.jsonl",
        trial / "agent/exported-artifacts/process/phase-events.jsonl",
    )
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        return {
            "status": "not_observed",
            "source": None,
            "events": [],
            "first_staging_at": None,
            "repair_iterations": 0,
            "public_gate_attempts": [],
            "survivor_committed_at": None,
            "survivor_candidate_sha256": None,
        }
    events = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        event = json.loads(line)
        if event.get("schema_version") != "candidate-phase-event-0.1.0":
            raise ValueError("phase event schema mismatch")
        if event.get("sequence") != line_number:
            raise ValueError("phase event sequence is not append-only")
        events.append(event)
    starts = {
        (event.get("details") or {}).get("attempt"): event
        for event in events
        if event.get("kind") == "public_gate_started"
    }
    attempts = []
    for event in events:
        if event.get("kind") != "public_gate_finished":
            continue
        attempt = (event.get("details") or {}).get("attempt")
        attempts.append(
            {
                "attempt": attempt,
                "started_at": (starts.get(attempt) or {}).get("observed_at"),
                "finished_at": event.get("observed_at"),
                "outcome": event.get("outcome"),
            }
        )
    survivor = next(
        (event for event in reversed(events) if event.get("kind") == "survivor_committed"),
        None,
    )
    return {
        "status": "observed",
        "source": path.relative_to(trial).as_posix(),
        "source_sha256": sha256_file(path),
        "events": events,
        "first_staging_at": next(
            (
                event.get("observed_at")
                for event in events
                if event.get("kind") == "candidate_staged"
            ),
            None,
        ),
        "repair_iterations": sum(event.get("kind") == "repair_started" for event in events),
        "public_gate_attempts": attempts,
        "survivor_committed_at": (survivor or {}).get("observed_at"),
        "survivor_candidate_sha256": ((survivor or {}).get("details") or {}).get(
            "candidate_sha256"
        ),
    }


def phase_evidence_complete(
    phase: dict, *, disposition: str, candidate_sha256: str | None = None
) -> bool:
    """Require the complete milestone chain only for a completed survivor."""

    if disposition != "complete":
        return True
    return bool(
        phase.get("first_staging_at")
        and phase.get("survivor_committed_at")
        and candidate_sha256
        and phase.get("survivor_candidate_sha256") == candidate_sha256
        and any(item.get("outcome") == "passed" for item in phase.get("public_gate_attempts", []))
    )


def candidate_transaction_status(
    *,
    mechanically_eligible: bool,
    paper_audit_enabled: bool,
    phase: dict,
    candidate_sha256: str,
) -> dict:
    """Separate scientific eligibility from completion of the paper transaction.

    A candidate can survive protected verification even when its networked
    session ends before the final public-envelope commit. Preserve that bundle
    as budget-incomplete, but do not admit it to the growing working seed.
    """

    commit_complete = bool(
        not paper_audit_enabled
        or not mechanically_eligible
        or phase_evidence_complete(
            phase,
            disposition="complete",
            candidate_sha256=candidate_sha256,
        )
    )
    process_incomplete = bool(paper_audit_enabled and mechanically_eligible and not commit_complete)
    return {
        "mechanically_eligible": mechanically_eligible,
        "survivor_commit_complete": commit_complete,
        "process_incomplete": process_incomplete,
        "working_seed_eligible": mechanically_eligible and not process_incomplete,
    }


def normalize_disposition(
    *,
    working_seed_eligible: bool,
    exception_type: str | None,
    verifier_completed: bool,
) -> str:
    if working_seed_eligible:
        return "complete"
    if exception_type == "AgentTimeoutError":
        return "budget-incomplete"
    if exception_type is not None or not verifier_completed:
        return "infra-fail"
    return "scientific-fail"


def candidate_metadata(path: Path) -> dict:
    try:
        payload = json.loads((path / "candidate.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "metadata_status": "unreadable",
            "task_signature": None,
            "task_signature_cell": None,
            "mechanism_fingerprint": None,
        }
    signature = payload.get("task_signature") if isinstance(payload, dict) else None
    fingerprint = payload.get("mechanism_fingerprint") if isinstance(payload, dict) else None
    cell = None
    if isinstance(signature, dict):
        cell = {key: signature.get(key) for key in ("decision_var", "structure", "decision_type")}
    return {
        "metadata_status": "parsed",
        "task_signature": signature,
        "task_signature_cell": cell,
        "mechanism_fingerprint": fingerprint,
    }


def semantic_neighbors(candidate_result: dict | None) -> dict:
    notes = (candidate_result or {}).get("notes") or {}
    semantic = notes.get("semantic") if isinstance(notes, dict) else {}
    if not isinstance(semantic, dict):
        semantic = {}
    return {
        "known": semantic.get("known_neighbors") or [],
        "negative": semantic.get("negative_neighbors") or [],
        "protected_source": semantic.get("protected_source_neighbors") or [],
        "siblings": semantic.get("sibling_neighbors") or [],
    }


def build_run_config(
    *,
    repo_root: Path,
    campaign_path: Path,
    campaign: dict,
    preflight: dict,
    source_revision: str,
    source_dirty: bool,
    materialization: dict,
    execution_context: dict | None,
    created_at: str,
    image_identity: dict,
) -> dict:
    controller_policies = {}
    if campaign.get("archive_policy"):
        controller_policies["quality_diversity"] = campaign["archive_policy"]
    if campaign.get("feedback_policy"):
        controller_policies["difficulty_feedback"] = campaign["feedback_policy"]
    profile = materialization.get("discovery_profile") or {}
    seed_name, _, seed_version = campaign["seed_set"].partition("@")
    seed_paths = sorted((repo_root / "foundry/seed_sets").glob(f"{seed_name}_v*.toml"))
    seed_path = next(
        (
            path
            for path in seed_paths
            if tomllib.loads(path.read_text(encoding="utf-8")).get("version") == seed_version
        ),
        None,
    )
    if seed_path is None:
        raise ValueError("run audit could not resolve seed definition")
    seed = tomllib.loads(seed_path.read_text(encoding="utf-8"))
    profile_definition = None
    if profile.get("reference"):
        family_version, _, profile_id = profile["reference"].partition(":")
        family, _, version = family_version.partition("@")
        profile_definition_path = (
            repo_root
            / "foundry/discovery_profiles"
            / family
            / f"v{'.'.join(version.split('.')[:2])}"
            / "profile-set.toml"
        )
        profile_definition = tomllib.loads(profile_definition_path.read_text(encoding="utf-8"))
        assert profile_id == profile.get("profile_id")
    protected_novelty = (profile_definition or {}).get("protected_novelty")
    protocol_root = repo_root / "foundry/protocols/question-world/v0.4"
    verifier_files = [
        repo_root
        / "harbor/datasets/foundry-builder-v1/build-question-worlds/tests/candidate_gatekeeper.py",
        repo_root / "harbor/datasets/sequential-builder-v1/test.sh",
        protocol_root / "candidate.schema.json",
        protocol_root / "candidate-contract.md",
    ]
    return {
        "schema_version": RUN_CONFIG_SCHEMA,
        "immutable": True,
        "created_at": created_at,
        "campaign_id": campaign["id"],
        "campaign_config": {
            "path": campaign_path.relative_to(repo_root).as_posix()
            if campaign_path.is_relative_to(repo_root)
            else str(campaign_path),
            "sha256": sha256_file(campaign_path),
            "status": campaign["status"],
        },
        "controller_policies": controller_policies,
        "budget": {
            "episode_timeout_seconds": campaign["agent"]["timeout_seconds_per_episode"],
            "agent_time_seconds": campaign["campaign"].get("agent_time_seconds"),
            "controller_wall_seconds": campaign["campaign"]["wall_time_seconds"],
            "phase_fractions": {
                "selection_deadline": 0.25,
                "durable_skeleton_deadline": 0.5,
                "first_public_gate_deadline": 0.75,
                "repair_reserve": 0.25,
            },
            "max_repair_turns": campaign["task"]["repair_turns"],
        },
        "distinctness": {
            "semantic_neighbor_limit": campaign["verification"]["semantic_neighbor_limit"],
            "semantic_duplicate_threshold": campaign["verification"][
                "semantic_duplicate_threshold"
            ],
            "protected_novelty": protected_novelty
            if protected_novelty is not None
            else {"enabled": False},
        },
        "software": {
            "source_revision": source_revision,
            "source_dirty": source_dirty,
            "protocol": campaign["protocol"],
            "strategy": campaign["strategy"],
            "harbor_version": preflight["harbor_version"],
            "agent_cli_version": preflight["agent_cli_version"],
            "verifier_contracts": {
                path.relative_to(repo_root).as_posix(): sha256_file(path) for path in verifier_files
            },
            "render_runtime": materialization.get("render_runtime"),
            "harbor_image": image_identity,
        },
        "inputs": {
            "seed_snapshot": campaign["seed_set"],
            "seed_definition_sha256": sha256_file(seed_path),
            "seed_list": seed.get("worlds"),
            "seed_list_sha256": sha256_json(seed.get("worlds")),
            "profile_reference": profile.get("reference"),
            "profile_card_sha256": profile.get("card_sha256"),
            "profile_definition_sha256": profile.get("definition_sha256"),
            "profile_sampling_seed": (profile_definition or {}).get("sampling_seed"),
        },
        "agent": {
            "adapter": campaign["agent"]["adapter"],
            "model_id": campaign["agent"]["model"],
            "reported_models": preflight["expected_reported_models"],
            "reasoning_effort": campaign["agent"]["reasoning_effort"],
            "temperature": {"value": None, "status": "not_configured_by_adapter"},
            "output_format": campaign["agent"]["output_format"],
            "provider": {
                "credential_kind": preflight["credential_kind"],
                "billing_mode": campaign["campaign"].get("cost_reporting"),
                "subscription_vs_api": (
                    "subscription_oauth"
                    if campaign["campaign"].get("cost_reporting") == "unavailable_subscription"
                    else "api"
                ),
                "routing": (
                    {
                        "upstream_provider": campaign["agent"].get("upstream_provider", "deepseek"),
                        "allow_fallbacks": campaign["agent"].get("allow_provider_fallbacks", False),
                        "source": (
                            "campaign_explicit"
                            if "upstream_provider" in campaign["agent"]
                            else "pinned_adapter_default"
                        ),
                    }
                    if campaign["agent"]["adapter"]
                    == "openrouter_opencode_foundry:OpenRouterOpenCodeFoundry"
                    else {"source": "not_applicable"}
                ),
            },
        },
        "execution": execution_context
        or {"mode": "single_campaign", "parallel": False, "max_parallel_arms": 1},
        "verification": {
            "clean_recycle": True,
            "oracle_stress_seeds": campaign["verification"]["oracle_stress_seeds"],
            "oracle_stress_scenes_per_seed": campaign["verification"][
                "oracle_stress_scenes_per_seed"
            ],
            "rng_sources": {
                "oracle": campaign["verification"]["oracle_stress_seeds"],
                "profile_sampling": (profile_definition or {}).get("sampling_seed"),
                "provider_sampling": {"value": None, "status": "not_exposed_by_adapter"},
                "controller": (
                    {
                        "value": campaign["archive_policy"]["controller_seed"],
                        "status": "quality_diversity_stable_hash",
                    }
                    if campaign.get("archive_policy")
                    else {
                        "value": {
                            "render_seed": campaign["feedback_policy"]["render_seed"],
                            "sampling_seed": campaign["feedback_policy"]["sampling_seed"],
                        },
                        "status": "difficulty_feedback_deterministic_sampling",
                    }
                    if campaign.get("feedback_policy")
                    else {"value": None, "status": "deterministic_no_rng"}
                ),
            },
        },
        "stopping": {
            "rule": campaign["campaign"].get("completion_policy"),
            "episode_limit_policy": campaign["campaign"].get("episode_limit_policy"),
            "useful_agent_time_definition": (
                "Harbor agent_execution wall time; excludes materialization, setup, "
                "protected verification, and controller retry backoff; in-call provider "
                "latency is inseparable and retained explicitly as provider-call wall time"
            ),
            "provider_retry_policy": {
                "policy": campaign["campaign"].get("transient_error_policy"),
                "backoff_seconds": campaign["campaign"].get("retry_backoff_seconds"),
                "max_consecutive_failures": campaign["campaign"].get(
                    "max_consecutive_provider_failures"
                ),
            },
            "rate_card": {
                "date": (
                    None
                    if campaign["campaign"].get("cost_reporting") == "unavailable_subscription"
                    else created_at[:10]
                ),
                "status": (
                    "not_applicable_subscription"
                    if campaign["campaign"].get("cost_reporting") == "unavailable_subscription"
                    else "provider_native_dated_at_run"
                ),
            },
        },
    }


def _safe_evidence_path(run_root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("evidence path must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe evidence path: {value!r}")
    path = (run_root / relative).resolve()
    if not path.is_relative_to(run_root):
        raise ValueError(f"evidence path escapes run root: {value!r}")
    return path


def _candidate_tree_sha256(path: Path) -> str:
    files = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError(f"candidate evidence contains symlink: {item}")
        if item.is_file():
            files[item.relative_to(path).as_posix()] = sha256_file(item)
    return sha256_json(files)


def verify_run_evidence(run_root: Path, manifest: dict | None = None) -> dict:
    """Recompute every raw-run evidence digest instead of trusting its pass flag."""

    run_root = run_root.resolve()
    if manifest is None:
        manifest = json.loads((run_root / "evidence-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != EVIDENCE_SCHEMA:
        raise ValueError("paper evidence manifest schema mismatch")
    if manifest.get("validation_passed") is not True:
        raise ValueError("paper evidence manifest was not finalized as valid")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("paper evidence manifest has no artifacts")
    artifact_paths = [row.get("path") for row in artifacts if isinstance(row, dict)]
    if len(artifact_paths) != len(artifacts) or len(set(artifact_paths)) != len(artifacts):
        raise ValueError("paper evidence artifact paths are invalid or duplicated")
    for row in artifacts:
        path = _safe_evidence_path(run_root, row["path"])
        if not path.is_file():
            raise ValueError(f"paper evidence artifact is missing: {row['path']}")
        if path.stat().st_size != row.get("bytes") or sha256_file(path) != row.get("sha256"):
            raise ValueError(f"paper evidence artifact digest mismatch: {row['path']}")

    bundles = manifest.get("candidate_bundles")
    if not isinstance(bundles, list):
        raise ValueError("paper evidence candidate bundle list is invalid")
    bundle_paths = [row.get("path") for row in bundles if isinstance(row, dict)]
    if len(bundle_paths) != len(bundles) or len(set(bundle_paths)) != len(bundles):
        raise ValueError("paper evidence candidate bundle paths are invalid or duplicated")
    for row in bundles:
        path = _safe_evidence_path(run_root, row["path"])
        if not path.is_dir():
            raise ValueError(f"paper evidence candidate bundle is missing: {row['path']}")
        if _candidate_tree_sha256(path) != row.get("tree_sha256"):
            raise ValueError(f"paper evidence candidate tree digest mismatch: {row['path']}")

    records = {
        int(record["episode_index"]): record
        for record in (
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(run_root.glob("episodes/*/episode-record.json"))
        )
    }
    packet_rows = manifest.get("packet_hashes")
    if not isinstance(packet_rows, list) or len(packet_rows) != len(records):
        raise ValueError("paper evidence packet ledger does not match episode records")
    for row in packet_rows:
        episode_index = int(row["episode_index"])
        record = records.get(episode_index)
        if record is None:
            raise ValueError(f"paper evidence has an unknown episode: {episode_index}")
        materialization_path = (
            run_root / "episodes" / f"{episode_index:03d}" / "packet" / "materialization.json"
        )
        materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
        packet_sha256 = sha256_json(materialization.get("files") or {})
        observed = {
            row.get("packet_sha256"),
            row.get("packet_content_hash"),
            record.get("packet_sha256"),
            (record.get("episode_start_state") or {}).get("packet_content_hash"),
            materialization.get("packet_sha256"),
            packet_sha256,
        }
        if len(observed) != 1:
            raise ValueError(f"paper evidence packet hash mismatch for episode {episode_index}")

    reconciliation = json.loads(
        (run_root / "usage-reconciliation.json").read_text(encoding="utf-8")
    )
    if reconciliation != manifest.get("usage_reconciliation") or not reconciliation.get("passed"):
        raise ValueError("paper evidence usage reconciliation is invalid")
    policies = manifest.get("controller_policy_evidence") or {}
    for name, passed in policies.items():
        if passed is not True:
            raise ValueError(f"paper controller policy evidence is incomplete: {name}")
    if "quality_diversity" in policies:
        if (
            not (run_root / "controller-policy-ledger.jsonl").is_file()
            or not (run_root / "quality-diversity-summary.json").is_file()
        ):
            raise ValueError("quality-diversity evidence files are missing")
    if "difficulty_feedback" in policies:
        if not (run_root / "difficulty-feedback-summary.json").is_file():
            raise ValueError("difficulty-feedback summary is missing")
        feedback = json.loads(
            (run_root / "feedback-usage-reconciliation.json").read_text(encoding="utf-8")
        )
        if not feedback.get("passed"):
            raise ValueError("difficulty-feedback reconciliation is incomplete")
    return {
        "status": "passed",
        "artifact_count": len(artifacts),
        "candidate_bundle_count": len(bundles),
        "episode_count": len(records),
    }


def finalize_run_evidence(run_root: Path) -> dict:
    """Write reconciled session-end evidence without discarding failures."""

    run_root = run_root.resolve()
    state_path = run_root / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    episode_records = sorted(run_root.glob("episodes/*/episode-record.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in episode_records]
    usage_fields = (
        "input_tokens",
        "cache_read_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
    )
    usage = {
        key: sum(
            int((record.get("provider", {}).get("usage") or {}).get(key) or 0) for record in records
        )
        for key in usage_fields
    }
    cost = round(
        sum(float(record.get("provider", {}).get("cost_usd") or 0.0) for record in records), 9
    )
    turns = sum(int(record.get("provider", {}).get("num_turns") or 0) for record in records)
    reconciliation = {
        "schema_version": "paper-run-usage-reconciliation-0.1.0",
        "episodes": len(records),
        "usage": usage,
        "provider_turns": turns,
        "cost_usd": cost,
        "state_matches": {
            "episodes": int(state.get("episodes_completed") or 0) == len(records),
            "provider_turns": int(state.get("cumulative_provider_turns") or 0) == turns,
            "cost_usd": abs(float(state.get("cumulative_cost_usd") or 0.0) - cost) < 1e-8,
        },
    }
    reconciliation["passed"] = all(reconciliation["state_matches"].values())
    reconciliation_path = run_root / "usage-reconciliation.json"
    reconciliation_path.write_text(
        json.dumps(reconciliation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    candidates = []
    for path in sorted((run_root / "candidates").glob("*/*")):
        if not path.is_dir():
            continue
        candidates.append(
            {
                "candidate_id": path.name.split("__", 1)[-1],
                "disposition_bucket": path.parent.name,
                "path": path.relative_to(run_root).as_posix(),
                "tree_sha256": sha256_json(
                    {
                        item.relative_to(path).as_posix(): sha256_file(item)
                        for item in sorted(child for child in path.rglob("*") if child.is_file())
                    }
                ),
            }
        )
    artifacts_by_path = {}
    patterns = (
        "run-config.json",
        "state.json",
        "events.jsonl",
        "episode-ledger.jsonl",
        "usage-reconciliation.json",
        "controller-policy-ledger.jsonl",
        "quality-diversity-summary.json",
        "difficulty-feedback-summary.json",
        "feedback-usage-reconciliation.json",
        "episodes/*/episode-record.json",
        "episodes/*/packet/materialization.json",
        "jobs/*/*/verifier/artifacts/gate_report.json",
        "jobs/*/*/verifier/artifacts/boundary_report.json",
        "jobs/*/*/verifier/artifacts/process/phase-events.jsonl",
        "jobs/*/*/agent/provider-turns.jsonl",
        "jobs/*/*/agent/trajectory.json",
        "jobs/*/*/agent/*.txt",
        "jobs/*/*/agent/*.json",
        "jobs/*/*/agent/*.jsonl",
        "jobs/*/*/agent/*.log",
        "jobs/*/result.json",
        "jobs/*/*/verifier/reward.json",
        "jobs/*/*/result.json",
    )
    for pattern in patterns:
        for path in sorted(run_root.glob(pattern)):
            if path.is_file():
                relative = path.relative_to(run_root).as_posix()
                artifacts_by_path[relative] = {
                    "path": relative,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
    for path in sorted(run_root.glob("episodes/*/feedback/**/*")):
        if path.is_file():
            relative = path.relative_to(run_root).as_posix()
            artifacts_by_path[relative] = {
                "path": relative,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    artifacts = [artifacts_by_path[key] for key in sorted(artifacts_by_path)]
    manifest = {
        "schema_version": EVIDENCE_SCHEMA,
        "run_status": state.get("status", "interrupted"),
        "campaign_id": state.get("campaign_id"),
        "episode_count": len(records),
        "disposition_counts": {
            name: sum(record.get("disposition") == name for record in records)
            for name in ("complete", "scientific-fail", "infra-fail", "budget-incomplete")
        },
        "candidate_bundles": candidates,
        "packet_hashes": [
            {
                "episode_index": record.get("episode_index"),
                "packet_sha256": record.get("packet_sha256"),
                "packet_content_hash": record.get("episode_start_state", {}).get(
                    "packet_content_hash"
                ),
            }
            for record in records
        ],
        "verdict_count": sum("gate_report.json" in item["path"] for item in artifacts),
        "artifacts": artifacts,
        "usage_reconciliation": reconciliation,
        "phase_audit_complete": all(
            bool(record.get("phase_evidence_complete")) for record in records
        ),
    }
    run_config = json.loads((run_root / "run-config.json").read_text(encoding="utf-8"))
    policies = run_config.get("controller_policies") or {}
    policy_requirements = {
        "quality_diversity": (
            (run_root / "controller-policy-ledger.jsonl").is_file()
            and (run_root / "quality-diversity-summary.json").is_file()
        ),
        "difficulty_feedback": (
            (run_root / "controller-policy-ledger.jsonl").is_file()
            and (run_root / "difficulty-feedback-summary.json").is_file()
            and (run_root / "feedback-usage-reconciliation.json").is_file()
        ),
    }
    manifest["controller_policy_evidence"] = {name: policy_requirements[name] for name in policies}
    manifest["validation_passed"] = bool(
        reconciliation["passed"]
        and manifest["phase_audit_complete"]
        and manifest["verdict_count"] == len(records)
        and (run_root / "run-config.json").is_file()
        and all(manifest["controller_policy_evidence"].values())
    )
    path = run_root / "evidence-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
