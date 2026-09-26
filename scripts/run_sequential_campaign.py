"""Run a controller-owned sequential campaign as isolated Harbor episodes."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from question_foundry.difficulty_feedback import (
    BRANCHING_POLICY_ID,
    EFFORT_AWARE_BRANCHING_POLICY_ID,
    RESILIENT_BRANCHING_POLICY_ID,
    RESILIENT_POLICY_IDS,
    SYNTHESIS_POLICY_IDS,
    WIRED_RESILIENT_BRANCHING_POLICY_ID,
    feedback_reconciliation,
    feedback_summary,
    fresh_feedback_candidate_count,
    maximum_paid_requests,
    policy_schema_version,
    reasoning_branch_assignment,
    update_feedback_state,
)
from question_foundry.difficulty_feedback import (
    evaluate_candidate as evaluate_candidate_difficulty,
)
from question_foundry.difficulty_feedback import (
    preflight_authorization as preflight_feedback_authorization,
)
from question_foundry.paper_audit import (
    LEDGER_SCHEMA,
    build_run_config,
    candidate_metadata,
    candidate_transaction_status,
    finalize_run_evidence,
    harbor_lifecycle,
    normalize_disposition,
    phase_evidence_complete,
    read_phase_events,
    semantic_neighbors,
    sha256_file,
)
from question_foundry.quality_diversity import (
    archive_summary as quality_diversity_summary,
)
from question_foundry.quality_diversity import (
    consider_candidate as consider_quality_diversity_candidate,
)
from question_foundry.registry import canonical_json, sha256_candidate_tree
from question_foundry.sequential import (
    automatic_working_seed_eligible,
    cost_limit_enforced,
    expected_reported_models,
    initial_state,
    load_sequential_campaign,
    materialize_sequential_episode,
    separates_budget_from_infrastructure,
    strategy_policy,
)
from scripts.check_auth_smoke import patterns_from_env, patterns_from_file, scan_tree

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HARBOR = "0.1.44"
# Claude Code attributes its own locally generated assistant messages (API
# errors, quota notices) to this sentinel rather than to a served model.
_SYNTHETIC_MODEL = "<synthetic>"
_QUOTA_MARKERS = ("usage limit reached", "rate limit", "429")
_HARD_QUOTA_MARKERS = ("usage limit reached", "quota exceeded", "insufficient_quota")
_TRANSIENT_RATE_LIMIT_MARKERS = ("rate limit", "429", "too many requests")
_TRANSIENT_CAPACITY_MARKERS = ("overloaded", "529", "server-side issue")
_CANDIDATE_ID = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
# Every supported adapter declares its pinned host CLI, credential source, and
# operator confirmations. Secrets are inspected only for the post-run leak scan.
_ADAPTER_AUTH = {
    "claude_code_foundry:ClaudeCodeFoundry": {
        "cli_binary": "claude",
        "cli_label": "Claude Code",
        "credential_env": "CLAUDE_CODE_OAUTH_TOKEN",
        "run_confirmation": "CLAUDE_SEQUENTIAL_RUN_CONFIRMED",
        "entitlement_confirmation": "CLAUDE_MODEL_ENTITLEMENT_CONFIRMED",
        "extra_confirmations": {"CLAUDE_OAUTH_REFRESH_CONFIRMED": "YES"},
        "credential_transport": "temporary-mode-0600-upload",
    },
    "codex_oauth:CodexOAuth": {
        "cli_binary": "codex",
        "cli_label": "Codex",
        "credential_file_env": "CODEX_AUTH_FILE",
        "run_confirmation": "CODEX_SEQUENTIAL_RUN_CONFIRMED",
        "entitlement_confirmation": "CODEX_MODEL_ENTITLEMENT_CONFIRMED",
        "extra_confirmations": {},
        "credential_transport": "temporary-mode-0600-upload",
    },
    "openrouter_opencode_foundry:OpenRouterOpenCodeFoundry": {
        "cli_binary": "opencode",
        "cli_label": "OpenCode",
        # The adapter installs this exact CLI version inside every Harbor
        # container; a host OpenCode installation is neither used nor required.
        "host_cli_required": False,
        "credential_env": "OPENROUTER_API_KEY",
        "run_confirmation": "OPENROUTER_SEQUENTIAL_RUN_CONFIRMED",
        "entitlement_confirmation": "OPENROUTER_MODEL_ENTITLEMENT_CONFIRMED",
        "extra_confirmations": {},
        "credential_transport": "temporary-mode-0600-upload",
    },
}


def _utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _repository_path(relative: str, *, label: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or not path.exists():
        raise ValueError(f"{label} escaped the repository or does not exist: {relative}")
    return path


def _feedback_bootstrap_policy_compatible(source_policy: str, target_policy: str) -> bool:
    """Allow only frozen v0.6 evidence into a forward resilient controller."""

    return source_policy == target_policy or (
        source_policy == BRANCHING_POLICY_ID
        and target_policy
        in {
            RESILIENT_BRANCHING_POLICY_ID,
            WIRED_RESILIENT_BRANCHING_POLICY_ID,
            EFFORT_AWARE_BRANCHING_POLICY_ID,
        }
    )


def _load_feedback_bootstrap(campaign: dict) -> tuple[dict, Path] | None:
    """Load a frozen pilot-feedback manifest before any paid inference.

    The bootstrap is developmental evidence, not a replacement for the frozen
    conceptual seed profile. Candidate and feedback hashes make the exact prior
    code/evidence auditable while keeping evaluator rasters hidden in code-only
    treatments.
    """

    config = campaign.get("feedback_bootstrap")
    if config is None:
        return None
    manifest_path = _repository_path(config["manifest"], label="feedback bootstrap manifest")
    if sha256_file(manifest_path) != config["manifest_sha256"]:
        raise ValueError("feedback bootstrap manifest SHA-256 does not match campaign")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {"schema_version", "id", "source", "records"}
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError(f"feedback bootstrap manifest keys must be exactly {sorted(required)}")
    if manifest["schema_version"] != "difficulty-feedback-bootstrap-0.1.0":
        raise ValueError("feedback bootstrap schema_version is unsupported")
    if manifest["id"] != config["id"]:
        raise ValueError("feedback bootstrap ID does not match campaign")
    records = manifest["records"]
    if not isinstance(records, list) or not 1 <= len(records) <= 10:
        raise ValueError("feedback bootstrap requires one to ten records")
    seen: set[str] = set()
    for record in records:
        expected = {
            "candidate_id",
            "candidate_path",
            "candidate_sha256",
            "feedback_path",
            "feedback_result_sha256",
            "sample_set_sha256",
            "working_seed_record",
        }
        if not isinstance(record, dict) or set(record) != expected:
            raise ValueError(f"feedback bootstrap record keys must be exactly {sorted(expected)}")
        candidate_id = record["candidate_id"]
        if _CANDIDATE_ID.fullmatch(str(candidate_id)) is None or candidate_id in seen:
            raise ValueError("feedback bootstrap candidate IDs must be unique and valid")
        seen.add(candidate_id)
        candidate_path = _repository_path(record["candidate_path"], label="bootstrap candidate")
        feedback_path = _repository_path(record["feedback_path"], label="bootstrap feedback")
        if sha256_candidate_tree(candidate_path) != record["candidate_sha256"]:
            raise ValueError(f"bootstrap candidate hash mismatch: {candidate_id}")
        result_path = feedback_path / "feedback-result.json"
        sample_path = feedback_path / "sample-set.json"
        if sha256_file(result_path) != record["feedback_result_sha256"]:
            raise ValueError(f"bootstrap feedback-result hash mismatch: {candidate_id}")
        if sha256_file(sample_path) != record["sample_set_sha256"]:
            raise ValueError(f"bootstrap sample-set hash mismatch: {candidate_id}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("candidate_id") != candidate_id
            or not _feedback_bootstrap_policy_compatible(
                str(result.get("policy_id")),
                campaign["feedback_policy"]["id"],
            )
            or result.get("status") != "complete"
            or result.get("protected_valid") is not True
        ):
            raise ValueError(
                f"bootstrap feedback is not complete protected-valid evidence: {candidate_id}"
            )
        state_record = record["working_seed_record"]
        if (
            not isinstance(state_record, dict)
            or state_record.get("candidate_id") != candidate_id
            or state_record.get("candidate_sha256") != record["candidate_sha256"]
            or state_record.get("mechanically_eligible") is not True
        ):
            raise ValueError(f"bootstrap working-seed record is inconsistent: {candidate_id}")
    return manifest, manifest_path


def _install_feedback_bootstrap(
    *, campaign: dict, manifest: dict, run_root: Path, state: dict
) -> list[Path]:
    accepted_paths: list[Path] = []
    feedback_state = state["difficulty_feedback"]
    for index, item in enumerate(manifest["records"], start=1):
        candidate_id = item["candidate_id"]
        source_candidate = _repository_path(item["candidate_path"], label="bootstrap candidate")
        source_feedback = _repository_path(item["feedback_path"], label="bootstrap feedback")
        destination = (
            run_root / "candidates/working-seed" / f"bootstrap-{index:03d}__{candidate_id}"
        )
        feedback_destination = run_root / "bootstrap-feedback" / candidate_id
        shutil.copytree(source_candidate, destination)
        shutil.copytree(source_feedback, feedback_destination)
        result = json.loads(
            (feedback_destination / "feedback-result.json").read_text(encoding="utf-8")
        )
        result["artifact_path"] = feedback_destination.relative_to(run_root).as_posix()
        result["bootstrap_source"] = manifest["id"]
        feedback_state = update_feedback_state(feedback_state, result)
        state_record = json.loads(json.dumps(item["working_seed_record"]))
        state_record["artifact_path"] = destination.relative_to(run_root).as_posix()
        state_record["episode_index"] = 0
        state_record["bootstrap_source"] = manifest["id"]
        state_record["difficulty_feedback"] = {
            key: result.get(key)
            for key in (
                "policy_id",
                "status",
                "aggregate_accuracy",
                "evaluator_accuracies",
                "examples_per_evaluator",
                "hard_seed_eligible",
                "difficulty_hypothesis",
                "sample_responses",
                "sample_set_sha256",
                "total_cost_usd",
                "artifact_path",
            )
        }
        state["working_seed"].append(state_record)
        accepted_paths.append(destination)
    state["difficulty_feedback"] = feedback_state
    source_cost = round(
        sum(
            float(row.get("total_cost_usd") or 0.0)
            for row in feedback_state["candidate_feedback"]
            if row.get("bootstrap_source")
        ),
        9,
    )
    # The source evaluations are sunk pilot evidence. Preserve and disclose
    # their cost, but do not charge it against this campaign's fresh cost cap.
    state["difficulty_feedback"]["cumulative_cost_usd"] = 0.0
    state["feedback_bootstrap"] = {
        "id": manifest["id"],
        "source": manifest["source"],
        "candidate_ids": [row["candidate_id"] for row in manifest["records"]],
        "source_cost_usd": source_cost,
    }
    return accepted_paths


def _append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(descriptor, (canonical_json(value) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _event(path: Path, *, sequence: int, kind: str, data: dict) -> None:
    _append_jsonl(
        path,
        {
            "schema_version": "sequential-controller-event-0.1.0",
            "sequence": sequence,
            "observed_at": _utc(),
            "kind": kind,
            "data": data,
        },
    )


def _paper_ledger_event(path: Path, *, sequence: int, kind: str, data: dict) -> None:
    _append_jsonl(
        path,
        {
            "schema_version": LEDGER_SCHEMA,
            "sequence": sequence,
            "observed_at": _utc(),
            "kind": kind,
            "data": data,
        },
    )


def _render_image_identity(materialization: dict, task: Path) -> dict:
    runtime = materialization.get("render_runtime") or {}
    image = runtime.get("container_image")
    if not image:
        dockerfile = task / "environment/Dockerfile"
        first = dockerfile.read_text(encoding="utf-8").splitlines()[0]
        if not first.startswith("FROM "):
            raise ValueError("materialized task Dockerfile has no base image")
        image = first.removeprefix("FROM ").strip().split(" AS ", 1)[0]
    completed = subprocess.run(
        ["docker", "inspect", "--type", "image", "--format", "{{.Id}}", str(image)],
        check=False,
        capture_output=True,
        text=True,
    )
    resolution = "direct_reference"
    if completed.returncode != 0 or not completed.stdout.strip():
        # Docker Desktop can transiently list an exact RepoTag while its
        # name-based inspect endpoint fails to resolve that same tag. Resolve
        # the exact reference to one local content ID, then inspect by ID. This
        # remains fail-closed on missing or ambiguous matches and records which
        # path established the immutable identity.
        listed = subprocess.run(
            [
                "docker",
                "image",
                "ls",
                "--filter",
                f"reference={image}",
                "--format",
                "{{.ID}}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        image_ids = sorted(set(listed.stdout.split())) if listed.returncode == 0 else []
        if len(image_ids) == 1:
            completed = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "--type",
                    "image",
                    "--format",
                    "{{.Id}}",
                    image_ids[0],
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            resolution = "exact_reference_list_to_content_id"
    if completed.returncode != 0 or not completed.stdout.strip():
        raise ValueError(f"render-runtime image is not locally inspectable: {image}")
    return {
        "runtime_id": runtime.get("id"),
        "image_reference": image,
        "image_id": completed.stdout.strip(),
        "image_resolution": resolution,
        "task_dockerfile_sha256": sha256_file(task / "environment/Dockerfile"),
        "identity_status": "docker_content_id_observed_before_provider_call",
    }


def _harbor_version(harbor: str) -> str:
    first = Path(harbor).read_text(encoding="utf-8").splitlines()[0]
    if not first.startswith("#!"):
        raise ValueError("cannot resolve Harbor Python interpreter")
    return subprocess.run(
        [
            first[2:],
            "-c",
            "import importlib.metadata as m; print(m.version('harbor'))",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _preflight(campaign: dict) -> dict:
    harbor = shutil.which("harbor")
    adapter = campaign["agent"]["adapter"]
    auth = _ADAPTER_AUTH.get(adapter)
    if auth is None:
        raise ValueError(f"no credential policy is defined for adapter {adapter}")
    cli = shutil.which(auth["cli_binary"])
    host_cli_required = bool(auth.get("host_cli_required", True))
    if not harbor or (host_cli_required and not cli):
        raise ValueError(f"harbor and {auth['cli_binary']} executables are required")
    harbor_version = _harbor_version(harbor)
    if harbor_version != EXPECTED_HARBOR:
        raise ValueError(f"Harbor {harbor_version} is installed; expected {EXPECTED_HARBOR}")
    if host_cli_required:
        cli_output = subprocess.run(
            [cli, "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        version_match = re.search(r"\b\d+\.\d+\.\d+\b", cli_output)
        if version_match is None:
            raise ValueError(f"could not parse {auth['cli_label']} version: {cli_output!r}")
        cli_version = version_match.group(0)
        if cli_version != str(campaign["agent"]["cli_version"]):
            raise ValueError(
                f"{auth['cli_label']} {cli_version} differs from pinned "
                f"{campaign['agent']['cli_version']}"
            )
        cli_version_source = "host_executable"
    else:
        cli_version = str(campaign["agent"]["cli_version"])
        cli_version_source = "container_install_pin"
    if credential_env := auth.get("credential_env"):
        if not os.environ.get(credential_env, "").strip():
            raise ValueError(f"{credential_env} is not set")
        credential_kind = "environment"
        credential_reference = credential_env
    else:
        credential_file_env = auth["credential_file_env"]
        credential_file = Path(os.environ.get(credential_file_env, "")).expanduser()
        if not credential_file.is_file() or credential_file.stat().st_size == 0:
            raise ValueError(f"{credential_file_env} does not name a non-empty credential file")
        credential_kind = "file"
        credential_reference = credential_file_env
    if os.environ.get(auth["entitlement_confirmation"]) != campaign["agent"]["model"]:
        raise ValueError("exact-model auth smoke confirmation is missing")
    for name, value in auth["extra_confirmations"].items():
        if os.environ.get(name) != value:
            raise ValueError(f"{name} confirmation is missing")
    if os.environ.get(auth["run_confirmation"]) != "YES":
        raise ValueError("provider-specific developmental run confirmation is missing")
    result = {
        "harbor": harbor,
        "harbor_version": harbor_version,
        "agent_cli": auth["cli_binary"],
        "agent_cli_version": cli_version,
        "agent_cli_version_source": cli_version_source,
        "adapter": adapter,
        "credential_kind": credential_kind,
        "credential_reference": credential_reference,
        "credential_transport": auth["credential_transport"],
        "cost_limit_enforced": cost_limit_enforced(campaign),
        "expected_reported_models": sorted(expected_reported_models(campaign)),
    }
    if auth["cli_binary"] == "claude":
        # Preserve the historical field used by the first sequential records.
        result["claude_cli_version"] = cli_version
    return result


def _trial_dir(job: Path) -> Path:
    trials = sorted(
        path for path in job.iterdir() if path.is_dir() and (path / "result.json").is_file()
    )
    if len(trials) != 1:
        raise ValueError(f"expected one Harbor trial in {job}, found {len(trials)}")
    return trials[0]


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _claude_provider_record(
    trial: Path,
    expected_model: str,
    reported_models: set[str],
) -> dict:
    stream = trial / "agent/claude-code.jsonl"
    if not stream.is_file():
        return {
            "model": expected_model,
            "reported_models": [],
            "provider_errors": [],
            "provider_error_event_count": 0,
            "recovered_provider_error_count": 0,
            "terminal_provider_failure": False,
            "transient_rate_limited": False,
            "transient_overloaded": False,
            "quota_exhausted": False,
            "model_records_checked": 0,
            "num_turns": 0,
            "stop_reason": "missing_stream",
            "cost_usd": 0.0,
            "cost_available": False,
            "usage": {},
        }
    records = []
    models = set()
    provider_errors: list[str] = []
    served_messages: dict[str, dict] = {}
    retry_events: list[dict] = []
    for line in stream.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        records.append(row)
        if row.get("type") == "system" and row.get("subtype") == "api_retry":
            retry_events.append(row)
        if row.get("type") == "assistant":
            message = row.get("message") if isinstance(row.get("message"), dict) else {}
            if message.get("model") == _SYNTHETIC_MODEL:
                # The CLI attributes its own locally generated messages — API
                # errors, quota notices — to a sentinel that is not a served
                # model. Collect them as provider errors instead of failing the
                # exact model-identity check on them.
                provider_errors.extend(
                    text
                    for item in message.get("content") or []
                    if isinstance(item, dict) and (text := (item.get("text") or "").strip())
                )
            elif message.get("model"):
                models.add(str(message["model"]))
                message_key = str(
                    message.get("id")
                    or row.get("request_id")
                    or row.get("uuid")
                    or f"assistant-row-{len(records)}"
                )
                served_messages[message_key] = message
    if models and not models <= reported_models:
        raise ValueError(
            f"provider stream model mismatch: observed {sorted(models)}, "
            f"campaign pins {sorted(reported_models)}"
        )
    result = next(
        (
            row
            for row in reversed(records)
            if row.get("type") == "result" or "total_cost_usd" in row
        ),
        {},
    )
    provider_error_text = " ".join(provider_errors).lower()
    result_is_error = (
        bool(result.get("is_error"))
        or str(result.get("terminal_reason") or "").lower() in {"api_error", "provider_error"}
        or str(result.get("stop_reason") or "").lower() == "error"
    )
    terminal_provider_failure = bool(provider_errors) and result_is_error
    quota_exhausted = terminal_provider_failure and any(
        marker in provider_error_text for marker in _HARD_QUOTA_MARKERS
    )
    transient_rate_limited = terminal_provider_failure and any(
        marker in provider_error_text for marker in _TRANSIENT_RATE_LIMIT_MARKERS
    )
    transient_overloaded = terminal_provider_failure and any(
        marker in provider_error_text for marker in _TRANSIENT_CAPACITY_MARKERS
    )
    usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
    if not usage and served_messages:
        usage = {
            key: sum(
                int((message.get("usage") or {}).get(key) or 0)
                for message in served_messages.values()
                if isinstance(message.get("usage"), dict)
            )
            for key in (
                "input_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
                "output_tokens",
            )
        }
        usage["thinking_output_tokens_estimate"] = sum(
            int(row.get("estimated_tokens_delta") or 0)
            for row in records
            if row.get("type") == "system" and row.get("subtype") == "thinking_tokens"
        )
    num_turns = int(result.get("num_turns") or len(served_messages))
    stop_reason = str(
        result.get("stop_reason")
        or result.get("subtype")
        or ("stream_ended_without_result" if served_messages else "unknown")
    )
    return {
        "model": expected_model,
        "reported_models": sorted(models),
        "provider_errors": provider_errors[:5],
        "provider_error_event_count": len(retry_events),
        "recovered_provider_error_count": (
            len(retry_events) if retry_events and not terminal_provider_failure else 0
        ),
        "terminal_provider_failure": terminal_provider_failure,
        "transient_rate_limited": transient_rate_limited,
        "transient_overloaded": transient_overloaded,
        "quota_exhausted": quota_exhausted,
        "model_records_checked": sum(row.get("type") == "assistant" for row in records),
        "num_turns": num_turns,
        "stop_reason": stop_reason,
        "terminal_reason": str(result.get("terminal_reason") or "unknown"),
        "cost_usd": float(result.get("total_cost_usd") or 0.0),
        "cost_available": result.get("total_cost_usd") is not None,
        "usage": usage,
    }


def _codex_provider_record(
    trial: Path,
    expected_model: str,
    reported_models: set[str],
) -> dict:
    trajectory_path = trial / "agent/trajectory.json"
    stream_path = trial / "agent/codex.txt"
    if not trajectory_path.is_file():
        return {
            "model": expected_model,
            "reported_models": [],
            "provider_errors": [],
            "provider_error_event_count": 0,
            "recovered_provider_error_count": 0,
            "terminal_provider_failure": False,
            "transient_rate_limited": False,
            "quota_exhausted": False,
            "model_records_checked": 0,
            "num_turns": 0,
            "stop_reason": "missing_trajectory",
            "terminal_reason": "unknown",
            "cost_usd": 0.0,
            "cost_available": False,
            "usage": {},
        }
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    agent = trajectory.get("agent") if isinstance(trajectory.get("agent"), dict) else {}
    observed_model = str(agent.get("model_name") or "")
    models = {observed_model} if observed_model else set()
    if not models or not models <= reported_models:
        raise ValueError(
            f"provider trajectory model mismatch: observed {sorted(models)}, "
            f"campaign pins {sorted(reported_models)}"
        )
    metrics = (
        trajectory.get("final_metrics") if isinstance(trajectory.get("final_metrics"), dict) else {}
    )
    extra = metrics.get("extra") if isinstance(metrics.get("extra"), dict) else {}
    stream_text = stream_path.read_text(encoding="utf-8") if stream_path.is_file() else ""
    num_turns = 0
    parsed_events: list[dict] = []
    if stream_text:
        for line in stream_text.splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            parsed_events.append(row)
            num_turns += row.get("type") == "turn.completed"
    completed_indexes = [
        index for index, row in enumerate(parsed_events) if row.get("type") == "turn.completed"
    ]
    error_events = [
        (index, row)
        for index, row in enumerate(parsed_events)
        if row.get("type") in {"error", "turn.failed"}
    ]
    last_completed = max(completed_indexes, default=-1)
    terminal_errors = [row for index, row in error_events if index > last_completed]
    terminal_error_text = " ".join(
        json.dumps(row, sort_keys=True).lower() for row in terminal_errors
    )
    quota_exhausted = bool(terminal_errors) and any(
        marker in terminal_error_text for marker in _HARD_QUOTA_MARKERS
    )
    transient_rate_limited = bool(terminal_errors) and any(
        marker in terminal_error_text for marker in _TRANSIENT_RATE_LIMIT_MARKERS
    )
    recovered_error_count = sum(index < last_completed for index, _ in error_events)
    provider_errors = []
    if terminal_errors:
        provider_errors.append(
            "terminal_hard_quota"
            if quota_exhausted
            else "terminal_transient_rate_limit"
            if transient_rate_limited
            else "terminal_provider_error"
        )
    return {
        "model": expected_model,
        "reported_models": sorted(models),
        # Inspect only provider terminal/error events. Tool output is untrusted
        # candidate data and may legitimately contain strings such as "429".
        "provider_errors": provider_errors,
        "provider_error_event_count": len(error_events),
        "recovered_provider_error_count": recovered_error_count,
        "terminal_provider_failure": bool(terminal_errors),
        "transient_rate_limited": transient_rate_limited,
        "quota_exhausted": quota_exhausted,
        "model_records_checked": 1,
        "num_turns": num_turns,
        "stop_reason": (
            "terminal_provider_error"
            if terminal_errors
            else "turn_completed"
            if num_turns
            else "trajectory_recorded"
        ),
        "terminal_reason": (
            "hard_quota"
            if quota_exhausted
            else "transient_rate_limit"
            if terminal_errors and transient_rate_limited
            else "provider_error"
            if terminal_errors
            else "unknown"
        ),
        # Subscription OAuth exposes token usage but no provider-native dollars.
        "cost_usd": 0.0,
        "cost_available": False,
        "usage": {
            "input_tokens": int(metrics.get("total_prompt_tokens") or 0),
            "cache_read_input_tokens": int(metrics.get("total_cached_tokens") or 0),
            "output_tokens": int(metrics.get("total_completion_tokens") or 0),
            "reasoning_output_tokens": int(extra.get("reasoning_output_tokens") or 0),
        },
    }


def _grok_provider_record(
    trial: Path,
    expected_model: str,
    reported_models: set[str],
) -> dict:
    stream = trial / "agent/grok.json"
    if not stream.is_file():
        return {
            "model": expected_model,
            "reported_models": [],
            "provider_errors": [],
            "quota_exhausted": False,
            "model_records_checked": 0,
            "model_identity_source": "operator_entitlement_only",
            "num_turns": 0,
            "stop_reason": "missing_stream",
            "terminal_reason": "unknown",
            "cost_usd": 0.0,
            "cost_available": False,
            "usage": {},
        }
    records = []
    for line in stream.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            records.append(row)
    terminal = next(
        (
            row
            for row in reversed(records)
            if row.get("type") == "end" or row.get("total_cost_usd") is not None
        ),
        {},
    )
    errors = [
        str(row.get("data")) for row in records if row.get("type") == "error" and row.get("data")
    ]
    quota_exhausted = any(marker in item.lower() for item in errors for marker in _QUOTA_MARKERS)
    # The Grok streaming result does not attest its served model. Exact identity
    # is therefore pinned by the pre-run `grok models` entitlement confirmation,
    # and this limitation is explicit in the durable provider record.
    return {
        "model": expected_model,
        "reported_models": sorted(reported_models),
        "provider_errors": errors[:5],
        "quota_exhausted": quota_exhausted,
        "model_records_checked": 0,
        "model_identity_source": "operator_entitlement_only",
        "num_turns": int(terminal.get("num_turns") or 0),
        "stop_reason": str(terminal.get("stopReason") or "stream_ended_without_result"),
        "terminal_reason": str(terminal.get("stopReason") or "unknown"),
        "cost_usd": float(terminal.get("total_cost_usd") or 0.0),
        "cost_available": terminal.get("total_cost_usd") is not None,
        "usage": terminal.get("usage") if isinstance(terminal.get("usage"), dict) else {},
    }


def _opencode_provider_record(
    trial: Path,
    expected_model: str,
    reported_models: set[str],
) -> dict:
    """Read exact model, usage, and billed cost from OpenCode's ATIF record."""

    trajectory_path = trial / "agent/trajectory.json"
    stream_path = trial / "agent/opencode.txt"
    if not trajectory_path.is_file():
        return {
            "model": expected_model,
            "reported_models": [],
            "provider_errors": ["missing_opencode_trajectory"],
            "provider_error_event_count": 0,
            "recovered_provider_error_count": 0,
            "terminal_provider_failure": True,
            "transient_rate_limited": False,
            "transient_overloaded": False,
            "quota_exhausted": False,
            "model_records_checked": 0,
            "num_turns": 0,
            "stop_reason": "missing_trajectory",
            "terminal_reason": "provider_error",
            "cost_usd": 0.0,
            "cost_available": False,
            "usage": {},
        }

    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    agent = trajectory.get("agent") if isinstance(trajectory.get("agent"), dict) else {}
    observed_model = str(agent.get("model_name") or "")
    models = {observed_model} if observed_model else set()
    if not models or not models <= reported_models:
        raise ValueError(
            f"provider trajectory model mismatch: observed {sorted(models)}, "
            f"campaign pins {sorted(reported_models)}"
        )
    metrics = (
        trajectory.get("final_metrics") if isinstance(trajectory.get("final_metrics"), dict) else {}
    )
    extra = metrics.get("extra") if isinstance(metrics.get("extra"), dict) else {}
    records: list[dict] = []
    if stream_path.is_file():
        for line in stream_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                records.append(row)
    completed_indexes = [
        index for index, row in enumerate(records) if row.get("type") == "step_finish"
    ]
    error_events = [
        (index, row)
        for index, row in enumerate(records)
        if row.get("type") in {"error", "step_error"}
    ]
    last_completed = max(completed_indexes, default=-1)
    terminal_errors = [row for index, row in error_events if index > last_completed]
    terminal_error_text = " ".join(
        json.dumps(row, sort_keys=True).lower() for row in terminal_errors
    )
    quota_exhausted = bool(terminal_errors) and any(
        marker in terminal_error_text for marker in _HARD_QUOTA_MARKERS
    )
    transient_rate_limited = bool(terminal_errors) and any(
        marker in terminal_error_text for marker in _TRANSIENT_RATE_LIMIT_MARKERS
    )
    transient_overloaded = bool(terminal_errors) and any(
        marker in terminal_error_text for marker in _TRANSIENT_CAPACITY_MARKERS
    )
    cost = metrics.get("total_cost_usd")
    provider_errors = []
    if terminal_errors:
        provider_errors.append(
            "terminal_hard_quota"
            if quota_exhausted
            else "terminal_transient_rate_limit"
            if transient_rate_limited
            else "terminal_provider_overloaded"
            if transient_overloaded
            else "terminal_provider_error"
        )
    return {
        "model": expected_model,
        "reported_models": sorted(models),
        "provider_errors": provider_errors,
        "provider_error_event_count": len(error_events),
        "recovered_provider_error_count": sum(index < last_completed for index, _ in error_events),
        "terminal_provider_failure": bool(terminal_errors),
        "transient_rate_limited": transient_rate_limited,
        "transient_overloaded": transient_overloaded,
        "quota_exhausted": quota_exhausted,
        "model_records_checked": 1,
        "model_identity_source": "atif_trajectory",
        "num_turns": len(completed_indexes),
        "stop_reason": "terminal_provider_error" if terminal_errors else "trajectory_recorded",
        "terminal_reason": (
            "hard_quota"
            if quota_exhausted
            else "transient_rate_limit"
            if transient_rate_limited
            else "overloaded"
            if transient_overloaded
            else "provider_error"
            if terminal_errors
            else "unknown"
        ),
        "cost_usd": float(cost or 0.0),
        "cost_available": cost is not None,
        "usage": {
            "input_tokens": int(metrics.get("total_prompt_tokens") or 0),
            "cache_read_input_tokens": int(metrics.get("total_cached_tokens") or 0),
            "output_tokens": int(metrics.get("total_completion_tokens") or 0),
            "reasoning_output_tokens": int(extra.get("reasoning_tokens") or 0),
        },
    }


def _provider_record(
    trial: Path,
    expected_model: str,
    reported_models: set[str],
    *,
    adapter: str = "claude_code_foundry:ClaudeCodeFoundry",
) -> dict:
    if adapter == "codex_oauth:CodexOAuth":
        return _codex_provider_record(trial, expected_model, reported_models)
    if adapter == "grok_build_auth:GrokBuildAuth":
        return _grok_provider_record(trial, expected_model, reported_models)
    if adapter == "openrouter_opencode_foundry:OpenRouterOpenCodeFoundry":
        return _opencode_provider_record(trial, expected_model, reported_models)
    return _claude_provider_record(trial, expected_model, reported_models)


def _load_episode_result(
    job: Path,
    expected_model: str,
    reported_models: set[str],
    *,
    adapter: str,
    min_candidates: int = 1,
    max_candidates: int | None = 1,
    candidate_limit_policy: str | None = None,
) -> dict:
    trial = _trial_dir(job)
    result = json.loads((trial / "result.json").read_text(encoding="utf-8"))
    exception = result.get("exception_info")
    exception_type = (exception or {}).get("exception_type")
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    report_path = trial / "verifier/artifacts/gate_report.json"
    boundary_path = trial / "verifier/artifacts/boundary_report.json"
    if not report_path.is_file() or not boundary_path.is_file():
        raise ValueError("Harbor trial did not preserve verifier reports")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    checks = boundary.get("checks") or {}
    if (
        boundary.get("requires_clean_recycle") is not True
        or len(checks) != 4
        or not all(checks.values())
    ):
        raise ValueError(f"clean offline verifier boundary failed: {checks}")
    agent_execution = result.get("agent_execution") or {}
    if agent_execution.get("started_at") and agent_execution.get("finished_at"):
        agent_seconds = (
            _parse_time(agent_execution["finished_at"]) - _parse_time(agent_execution["started_at"])
        ).total_seconds()
    else:
        agent_seconds = 0.0
    if result.get("finished_at") and agent_execution.get("finished_at"):
        verifier_seconds = (
            _parse_time(result["finished_at"]) - _parse_time(agent_execution["finished_at"])
        ).total_seconds()
    else:
        verifier_seconds = 0.0
    candidate_results = report.get("candidate_results") or []
    results_by_id = {
        str(item.get("candidate_id")): item
        for item in candidate_results
        if isinstance(item, dict) and item.get("candidate_id")
    }
    submission = trial / "verifier/artifacts/submission"
    candidate_entries = []
    portfolio_path = submission / "portfolio.json"
    if portfolio_path.is_file():
        portfolio = json.loads(portfolio_path.read_text(encoding="utf-8"))
        ids = portfolio.get("candidate_ids") or [] if isinstance(portfolio, dict) else []
        if not isinstance(ids, list):
            ids = []
        safe_ids = list(
            dict.fromkeys(
                candidate_id
                for candidate_id in ids
                if isinstance(candidate_id, str) and _CANDIDATE_ID.fullmatch(candidate_id)
            )
        )
    else:
        ids = []
        safe_ids = []
    if rewards.get("submission_valid") == 1.0:
        candidate_count_valid = (
            len(ids) == 1
            if candidate_limit_policy == "exactly_one_per_session"
            else len(ids) >= min_candidates
            if candidate_limit_policy == "time_bounded_append_only"
            else max_candidates is not None and min_candidates <= len(ids) <= max_candidates
        )
        if not candidate_count_valid:
            raise ValueError("session submission candidate count is outside its configured bounds")
        if len(ids) != len(set(ids)):
            raise ValueError("session submission repeats a candidate ID")
        if set(ids) != set(results_by_id):
            raise ValueError("portfolio IDs differ from protected candidate results")
    # Invalid envelopes never become eligible, but safe declared candidate
    # bundles remain valuable failure evidence. Preserve only identifier-safe,
    # protected-evaluated directories; ignore undeclared scratch entries.
    for candidate_id in safe_ids:
        if candidate_id not in results_by_id:
            if rewards.get("submission_valid") == 1.0:
                raise ValueError(f"protected candidate result is missing: {candidate_id}")
            continue
        candidate_path = submission / "candidates" / candidate_id
        if not candidate_path.is_dir():
            if rewards.get("submission_valid") == 1.0:
                raise ValueError(f"session candidate directory is missing: {candidate_id}")
            continue
        candidate_entries.append(
            {
                "candidate_id": candidate_id,
                "candidate_path": candidate_path,
                "candidate_result": results_by_id[candidate_id],
                "submission_valid": rewards.get("submission_valid") == 1.0,
            }
        )
    provider = _provider_record(
        trial,
        expected_model,
        reported_models,
        adapter=adapter,
    )
    return {
        "trial": trial,
        "trial_id": trial.name,
        "exception_type": exception_type,
        "verifier_completed": result.get("verifier_result") is not None,
        "rewards": rewards,
        "report": report,
        "report_path": report_path,
        "boundary_checks": checks,
        "candidates": candidate_entries,
        # Preserve scalar fields for historical one-candidate reports.
        "candidate_result": (
            candidate_entries[0]["candidate_result"] if len(candidate_entries) == 1 else None
        ),
        "candidate_id": (
            candidate_entries[0]["candidate_id"] if len(candidate_entries) == 1 else None
        ),
        "candidate_path": (
            candidate_entries[0]["candidate_path"] if len(candidate_entries) == 1 else None
        ),
        "agent_seconds": agent_seconds,
        "verifier_seconds": max(0.0, verifier_seconds),
        "provider": provider,
        "lifecycle": harbor_lifecycle(result),
        "phase_evidence": read_phase_events(trial),
        "harbor_result_sha256": sha256_file(trial / "result.json"),
    }


def _scan_job_for_secret(job: Path, preflight: dict) -> None:
    credential_reference = preflight["credential_reference"]
    if preflight["credential_kind"] == "environment":
        patterns = patterns_from_env([credential_reference])
    elif preflight["credential_kind"] == "file":
        patterns = patterns_from_file(Path(os.environ[credential_reference]))
    else:
        raise ValueError("unknown credential leak-scan policy")
    if not patterns:
        raise ValueError("credential leak scan has no in-memory pattern")
    leaks = scan_tree(job, patterns)
    if leaks:
        raise ValueError(
            "credential material persisted in job files: "
            + ", ".join(str(path.relative_to(job)) for path in leaks)
        )
    if any(path.name == "auth.json" for path in job.rglob("auth.json")):
        raise ValueError("auth.json persisted in the Harbor job")


def _missing_candidate_transition(*, exception_type: str | None, split_budget: bool) -> str:
    """Classify an episode with no controller-eligible candidate artifact."""

    if not split_budget:
        return "infrastructure_failure"
    if exception_type == "AgentTimeoutError":
        return "budget_incomplete"
    if exception_type is None:
        return "invalid_submission"
    return "infrastructure_failure"


def _claim_candidate_id(
    candidate_id: str,
    *,
    episode_index: int,
    first_seen_episode: dict[str, int],
) -> dict | None:
    """Claim an episode-local ID or return a durable scientific exclusion.

    A builder can repeat an ID from an earlier, budget-incomplete session that
    was preserved but not included in its working or failure memory. That is a
    candidate-level protocol failure, not controller infrastructure failure.
    Preserve the duplicate, exclude it from the working seed, and let the
    campaign consume its frozen useful-time budget.
    """

    prior_episode = first_seen_episode.get(candidate_id)
    if prior_episode is None:
        first_seen_episode[candidate_id] = episode_index
        return None
    return {
        "kind": "candidate_id_reused",
        "candidate_id": candidate_id,
        "first_seen_episode_index": prior_episode,
        "reused_in_episode_index": episode_index,
    }


def _feedback_incomplete_action(policy: dict, consecutive_count: int) -> str:
    """Return the frozen arm action for an incomplete feedback episode."""

    if policy.get("id") not in RESILIENT_POLICY_IDS:
        return "stop"
    maximum = int(policy["max_consecutive_incomplete_feedback"])
    return "continue" if consecutive_count < maximum else "stop"


def _budget_assessment(
    campaign: dict,
    *,
    controller_wall_seconds: float,
    cumulative_agent_seconds: float,
    stop_reason: str,
    source_dirty: bool,
    infrastructure_outcome_count: int = 0,
    deviation_count: int = 0,
) -> dict:
    """Decide whether a deadline-driven arm is both budget- and run-eligible."""

    wall = int(campaign["campaign"]["wall_time_seconds"])
    agent_budget = int(campaign["campaign"]["agent_time_seconds"])
    remaining_wall = max(0.0, wall - controller_wall_seconds)
    remaining_agent = max(0.0, agent_budget - cumulative_agent_seconds)
    minimum_agent_start = int(campaign["campaign"]["minimum_start_seconds"])
    budget_fulfilled = remaining_agent < minimum_agent_start
    blocking_reasons = {
        "provider_quota_exhausted",
        "provider_failure_limit_reached",
        "provider_cost_unavailable",
        "empty_submission_limit_reached",
        "operator_stop_after_current_episode",
        "session_safety_cap_reached_before_deadline",
        "continuous_session_ended_before_agent_budget",
        "feedback_candidate_budget_reached",
        "feedback_cost_limit_reached",
        "difficulty_feedback_infrastructure_failure",
    }
    comparison_eligible = (
        budget_fulfilled
        and stop_reason not in blocking_reasons
        and not source_dirty
        and infrastructure_outcome_count == 0
        and deviation_count == 0
    )
    return {
        "schema_version": "campaign-budget-assessment-0.1.0",
        "policy": campaign["campaign"]["completion_policy"],
        "configured_wall_seconds": wall,
        "configured_agent_seconds": agent_budget,
        "controller_wall_seconds": round(controller_wall_seconds, 6),
        "cumulative_agent_seconds": round(cumulative_agent_seconds, 6),
        "remaining_wall_seconds": round(remaining_wall, 6),
        "remaining_agent_seconds": round(remaining_agent, 6),
        "minimum_agent_start_seconds": minimum_agent_start,
        "agent_budget_utilization_fraction": round(
            min(cumulative_agent_seconds / agent_budget, 1.0), 6
        ),
        "controller_wall_utilization_fraction": round(min(controller_wall_seconds / wall, 1.0), 6),
        "budget_fulfilled": budget_fulfilled,
        "comparison_eligible": comparison_eligible,
        "stop_reason": stop_reason,
        "source_dirty": source_dirty,
        "infrastructure_outcome_count": infrastructure_outcome_count,
        "deviation_count": deviation_count,
    }


def _provider_retry_delay(campaign: dict, consecutive_failures: int) -> int:
    schedule = campaign["campaign"]["retry_backoff_seconds"]
    index = min(max(consecutive_failures - 1, 0), len(schedule) - 1)
    return int(schedule[index])


def _next_empty_submission_streak(
    current: int,
    *,
    transition: str,
    candidate_count: int,
    exception_type: str | None,
    terminal_provider_failure: bool,
) -> int:
    """Count only ordinary, candidate-free invalid submissions.

    Provider failures, timeouts, and protected-evaluated candidate bundles have
    their own outcome paths. This circuit breaker prevents a deterministic
    adapter or prompt defect from launching unbounded fresh empty episodes.
    """

    empty_invalid = (
        transition == "invalid_submission"
        and candidate_count == 0
        and exception_type is None
        and not terminal_provider_failure
    )
    return current + 1 if empty_invalid else 0


def _episode_numbers(campaign: dict):
    """Yield controller sessions without imposing a portfolio-size surrogate.

    A single continuous mode gets one Harbor session whose adapter owns provider
    continuation. Episodic and hybrid modes get fresh sessions until agent time
    is exhausted. Historical campaigns retain their frozen max-episode meaning.
    """

    policy = campaign["campaign"].get("episode_limit_policy")
    if policy == "single_continuous_session":
        return range(1, 2)
    if policy == "repeat_until_agent_budget":
        return itertools.count(1)
    return range(1, int(campaign["campaign"]["max_episodes"]) + 1)


def _another_episode_allowed(campaign: dict, episode_index: int) -> bool:
    policy = campaign["campaign"].get("episode_limit_policy")
    if policy == "repeat_until_agent_budget":
        return True
    if policy == "single_continuous_session":
        return False
    return episode_index < int(campaign["campaign"]["max_episodes"])


def _stop_after_current_episode_requested(run_root: Path) -> bool:
    """Return whether the operator requested a verifier-safe campaign stop.

    The sentinel is checked only after Harbor has finished the current agent,
    clean recycle, protected verifier, and controller transition. It therefore
    provides a safe alternative to interrupting a live verifier with Ctrl-C.
    """

    return (run_root / "STOP_AFTER_CURRENT_EPISODE").is_file()


def run_campaign(
    *,
    campaign_path: Path,
    run_root: Path,
    harbor_task_namespace: str | None = None,
    execution_context: dict | None = None,
) -> dict:
    campaign = load_sequential_campaign(campaign_path)
    if campaign["status"] != "frozen":
        raise ValueError("campaign status must be frozen before execution")
    feedback_bootstrap = _load_feedback_bootstrap(campaign)
    preflight = _preflight(campaign)
    feedback_preflight = None
    if feedback_policy := campaign.get("feedback_policy"):
        feedback_preflight = preflight_feedback_authorization(
            feedback_policy,
            campaign_id=campaign["id"],
            authorization_scope_id=campaign["comparison"]["experiment_id"],
        )
    run_root = run_root.resolve()
    if run_root.exists():
        raise FileExistsError(f"sequential run root exists: {run_root}")
    run_root.mkdir(parents=True)
    started_at = _utc()
    state = initial_state(campaign, run_id=run_root.name, started_at=started_at)
    state["source_revision"] = _git("rev-parse", "HEAD")
    state["source_dirty"] = bool(_git("status", "--porcelain"))
    state["preflight"] = preflight
    bootstrap_accepted_paths: list[Path] = []
    if feedback_bootstrap is not None:
        bootstrap_accepted_paths = _install_feedback_bootstrap(
            campaign=campaign,
            manifest=feedback_bootstrap[0],
            run_root=run_root,
            state=state,
        )
    if feedback_preflight is not None:
        state["feedback_preflight"] = feedback_preflight
    state_path = run_root / "state.json"
    events_path = run_root / "events.jsonl"
    paper_ledger_path = run_root / "episode-ledger.jsonl"
    policy_ledger_path = run_root / "controller-policy-ledger.jsonl"
    # A paper arm can end before any candidate reaches its controller policy
    # (for example, a protected scientific failure in its only canary episode).
    # Preserve an explicit empty append-only ledger so session-end evidence
    # distinguishes "no policy decision occurred" from "the ledger vanished."
    if campaign.get("archive_policy") or campaign.get("feedback_policy"):
        policy_ledger_path.touch(exist_ok=False)
    _atomic_json(state_path, state)
    sequence = 1
    ledger_sequence = 0
    policy_sequence = 0
    campaign_started_data = {
        "campaign_id": campaign["id"],
        "strategy": campaign["strategy"],
        "wall_time_seconds": campaign["campaign"]["wall_time_seconds"],
        "source_revision": state["source_revision"],
        "source_dirty": state["source_dirty"],
        "adapter": campaign["agent"]["adapter"],
        "expected_reported_models": preflight["expected_reported_models"],
        "cost_limit_enforced": preflight["cost_limit_enforced"],
        "harbor_task_namespace": harbor_task_namespace,
    }
    if state.get("feedback_bootstrap"):
        campaign_started_data["feedback_bootstrap"] = state["feedback_bootstrap"]
    if "agent_time_seconds" in campaign["campaign"]:
        campaign_started_data.update(
            {
                "agent_time_seconds": campaign["campaign"]["agent_time_seconds"],
                "completion_policy": campaign["campaign"]["completion_policy"],
                "provider_turn_policy": campaign["agent"]["provider_turn_policy"],
            }
        )
    _event(
        events_path,
        sequence=sequence,
        kind="campaign_started",
        data=campaign_started_data,
    )
    reported_models = expected_reported_models(campaign)
    enforce_cost_limit = cost_limit_enforced(campaign)
    split_budget_outcomes = separates_budget_from_infrastructure(campaign)
    policy = strategy_policy(campaign)
    deadline_driven = bool(policy.get("deadline_driven"))
    paper_audit_enabled = bool(policy.get("paper_audit"))
    if not enforce_cost_limit:
        stopping_bounds = (
            "agent time, controller wall time, session safety, and provider-failure policy"
            if deadline_driven
            else "wall time and episode count"
        )
        print(
            "==> provider-reported cost is an unreliable proxy for this adapter; "
            f"the campaign stops on {stopping_bounds}",
            flush=True,
        )
    start_monotonic = time.monotonic()
    deadline = start_monotonic + int(campaign["campaign"]["wall_time_seconds"])
    accepted_paths: list[Path] = bootstrap_accepted_paths
    failed_paths: list[Path] = []
    failed_verdicts: list[dict] = []
    first_seen_candidate_ids: dict[str, int] = {}
    episode_limit_policy = campaign["campaign"].get("episode_limit_policy")
    stop_reason = (
        "single_continuous_session_completed"
        if episode_limit_policy == "single_continuous_session"
        else "episode_iterator_exhausted"
        if episode_limit_policy == "repeat_until_agent_budget"
        else "max_episodes"
    )
    consecutive_provider_failures = 0
    consecutive_empty_submissions = 0
    consecutive_incomplete_feedback = 0
    sys.stdout.flush()

    for episode_index in _episode_numbers(campaign):
        if feedback_policy := campaign.get("feedback_policy"):
            feedback_state = state["difficulty_feedback"]
            # Frozen bootstrap evaluations are prior evidence, not paid calls
            # made by this run. The fresh-candidate cap must therefore exclude
            # them, just as the cost cap excludes their sunk source cost.
            if fresh_feedback_candidate_count(feedback_state) >= int(
                feedback_policy["max_evaluated_candidates"]
            ):
                stop_reason = "feedback_candidate_budget_reached"
                break
            if float(feedback_state["cumulative_cost_usd"]) >= float(
                feedback_policy["max_total_cost_usd"]
            ):
                stop_reason = "feedback_cost_limit_reached"
                break
        remaining = int(deadline - time.monotonic())
        minimum_start = int(campaign["campaign"]["minimum_start_seconds"])
        reserve = int(campaign["campaign"]["verifier_reserve_seconds"])
        available_agent = remaining - reserve
        remaining_agent_budget = (
            int(
                int(campaign["campaign"]["agent_time_seconds"])
                - float(state["cumulative_agent_seconds"])
            )
            if deadline_driven
            else available_agent
        )
        if deadline_driven and remaining_agent_budget < minimum_start:
            stop_reason = "agent_budget_consumed"
            break
        if available_agent < minimum_start:
            stop_reason = (
                "insufficient_controller_time_for_episode"
                if deadline_driven
                else "insufficient_time_for_episode"
            )
            break
        timeout = min(
            int(campaign["agent"]["timeout_seconds_per_episode"]),
            available_agent,
            remaining_agent_budget,
        )
        packet = run_root / "episodes" / f"{episode_index:03d}" / "packet"
        materialization_started_at = _utc()
        materialization_started = time.monotonic()
        materialized = materialize_sequential_episode(
            repo_root=ROOT,
            campaign_path=campaign_path,
            output_root=packet,
            state=state,
            episode_index=episode_index,
            timeout_seconds=timeout,
            accepted_candidates=accepted_paths,
            failed_candidates=failed_paths,
            failed_verdicts=failed_verdicts,
            source_revision=state["source_revision"],
            source_dirty=state["source_dirty"],
            harbor_task_namespace=harbor_task_namespace,
            feedback_artifact_root=run_root,
        )
        materialization_finished_at = _utc()
        materialization_seconds = round(time.monotonic() - materialization_started, 6)
        materialization_record = json.loads(materialized.manifest.read_text(encoding="utf-8"))
        diversity_target = None
        if campaign.get("archive_policy"):
            diversity_target = materialization_record["controller_policy"]["target"]
            policy_sequence += 1
            _append_jsonl(
                policy_ledger_path,
                {
                    "schema_version": "controller-policy-event-0.1.0",
                    "sequence": policy_sequence,
                    "observed_at": materialization_finished_at,
                    "kind": "diversity_target_selected",
                    "episode_index": episode_index,
                    "data": diversity_target,
                },
            )
        if paper_audit_enabled and not (run_root / "run-config.json").exists():
            run_config = build_run_config(
                repo_root=ROOT,
                campaign_path=campaign_path,
                campaign=campaign,
                preflight=preflight,
                source_revision=state["source_revision"],
                source_dirty=bool(state["source_dirty"]),
                materialization=materialization_record,
                execution_context=execution_context,
                created_at=materialization_finished_at,
                image_identity=_render_image_identity(materialization_record, materialized.task),
            )
            if feedback_preflight is not None:
                run_config["feedback_preflight"] = feedback_preflight
            if state.get("feedback_bootstrap"):
                run_config["feedback_bootstrap"] = state["feedback_bootstrap"]
            _atomic_json(run_root / "run-config.json", run_config)
        if paper_audit_enabled:
            ledger_sequence += 1
            _paper_ledger_event(
                paper_ledger_path,
                sequence=ledger_sequence,
                kind="episode_started",
                data={
                    "episode_index": episode_index,
                    "campaign_id": campaign["id"],
                    "profile_reference": (
                        materialization_record.get("discovery_profile") or {}
                    ).get("reference"),
                    "materialization": {
                        "started_at": materialization_started_at,
                        "finished_at": materialization_finished_at,
                        "seconds": materialization_seconds,
                    },
                    "episode_start_state": {
                        "working_library_size": len(accepted_paths),
                        "failure_memory_size": len(failed_paths),
                        "state_sha256": materialization_record["state_sha256"],
                        "packet_content_hash": materialized.packet_sha256,
                    },
                    "agent_timeout_seconds": timeout,
                },
            )
        sequence += 1
        _event(
            events_path,
            sequence=sequence,
            kind="episode_materialized",
            data={
                "episode_index": episode_index,
                "packet_sha256": materialized.packet_sha256,
                "agent_timeout_seconds": timeout,
                "remaining_campaign_seconds": remaining,
                **({"remaining_agent_seconds": remaining_agent_budget} if deadline_driven else {}),
                "working_seed_size": len(accepted_paths),
                "failed_outcome_count": len(failed_paths),
            },
        )
        jobs_root = run_root / "jobs"
        job_name = f"{campaign['id']}-episode-{episode_index:03d}"
        command = [
            preflight["harbor"],
            "run",
            "-c",
            str(ROOT / "harbor/job-configs/vector_rewards.yaml"),
            "-p",
            str(materialized.task),
            "--agent-import-path",
            campaign["agent"]["adapter"],
            "-m",
            campaign["agent"]["model"],
            "--ak",
            f"version={campaign['agent']['cli_version']}",
            "--ak",
            f"reasoning_effort={campaign['agent']['reasoning_effort']}",
            "--ak",
            f"output_format={campaign['agent']['output_format']}",
            "--ak",
            "recycle_for_verifier=true",
            "-e",
            "docker",
            "-o",
            str(jobs_root),
            "--job-name",
            job_name,
            "--delete",
            "--n-concurrent",
            "1",
        ]
        if "max_turns_per_episode" in campaign["agent"]:
            command.extend(
                [
                    "--ak",
                    f"max_turns={campaign['agent']['max_turns_per_episode']}",
                ]
            )
        if "provider_turn_policy" in campaign["agent"]:
            command.extend(
                [
                    "--ak",
                    f"provider_turn_policy={campaign['agent']['provider_turn_policy']}",
                    "--ak",
                    (
                        "continuation_delay_seconds="
                        f"{campaign['agent']['continuation_delay_seconds']}"
                    ),
                    "--ak",
                    (
                        "retry_backoff_seconds="
                        + ",".join(
                            str(item) for item in campaign["campaign"]["retry_backoff_seconds"]
                        )
                    ),
                    "--ak",
                    (
                        "max_consecutive_provider_failures="
                        f"{campaign['campaign']['max_consecutive_provider_failures']}"
                    ),
                ]
            )
        if "upstream_provider" in campaign["agent"]:
            command.extend(
                [
                    "--ak",
                    f"upstream_provider={campaign['agent']['upstream_provider']}",
                    "--ak",
                    (
                        "allow_provider_fallbacks="
                        + str(campaign["agent"]["allow_provider_fallbacks"]).lower()
                    ),
                ]
            )
        environment = os.environ.copy()
        agents = str(ROOT / "harbor/agents")
        environment["PYTHONPATH"] = agents + (
            os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
        )
        sequence += 1
        _event(
            events_path,
            sequence=sequence,
            kind="episode_started",
            data={
                "episode_index": episode_index,
                "job_name": job_name,
                "observed_remaining_seconds": int(deadline - time.monotonic()),
            },
        )
        print(
            f"==> sequential episode {episode_index}: "
            f"timeout={timeout}s working_seed={len(accepted_paths)}",
            flush=True,
        )
        completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        job = jobs_root / job_name
        if not job.is_dir():
            raise ValueError(f"Harbor did not create expected job: {job}")
        _scan_job_for_secret(job, preflight)
        episode_result = _load_episode_result(
            job,
            campaign["agent"]["model"],
            reported_models,
            adapter=campaign["agent"]["adapter"],
            min_candidates=int(campaign["task"]["min_candidates"]),
            max_candidates=(
                int(campaign["task"]["max_candidates"])
                if "max_candidates" in campaign["task"]
                else None
            ),
            candidate_limit_policy=campaign["task"].get("candidate_limit_policy"),
        )
        if not episode_result["verifier_completed"]:
            verifier_outcome = {
                "episode_index": episode_index,
                "source_job": job_name,
                "source_trial": episode_result["trial_id"],
                "kind": "protected_verifier_result_missing",
                "reason": (
                    "Harbor preserved verifier artifacts but did not return a "
                    "completed verifier result"
                ),
            }
            state["infrastructure_outcomes"].append(verifier_outcome)
            state["deviations"].append(verifier_outcome)
        if completed.returncode != 0 and episode_result["exception_type"] is None:
            raise ValueError(
                f"Harbor exited {completed.returncode} without a recorded trial exception"
            )

        state["episodes_completed"] = episode_index
        state["cumulative_agent_seconds"] = round(
            state["cumulative_agent_seconds"] + episode_result["agent_seconds"],
            6,
        )
        state["cumulative_verifier_seconds"] = round(
            state["cumulative_verifier_seconds"] + episode_result["verifier_seconds"],
            6,
        )
        state["cumulative_cost_usd"] = round(
            state["cumulative_cost_usd"] + episode_result["provider"]["cost_usd"],
            9,
        )
        if deadline_driven:
            state["cumulative_provider_turns"] += int(
                episode_result["provider"].get("num_turns") or 0
            )
        budget_exhausted = episode_result["exception_type"] == "AgentTimeoutError"
        transition = "no_valid_candidate"
        candidate_records = []
        feedback_episode_failure = None
        for entry in episode_result["candidates"]:
            candidate_id = str(entry["candidate_id"])
            candidate_result = entry["candidate_result"]
            controller_exclusion = _claim_candidate_id(
                candidate_id,
                episode_index=episode_index,
                first_seen_episode=first_seen_candidate_ids,
            )
            mechanically_eligible = automatic_working_seed_eligible(
                rewards=episode_result["rewards"],
                candidate_result=candidate_result,
                verifier_completed=episode_result["verifier_completed"],
            )
            reported_submission_valid = bool(entry.get("submission_valid"))
            submission_valid = episode_result["verifier_completed"] and reported_submission_valid
            candidate_hash = sha256_candidate_tree(entry["candidate_path"])
            transaction = candidate_transaction_status(
                mechanically_eligible=mechanically_eligible,
                paper_audit_enabled=paper_audit_enabled,
                phase=episode_result["phase_evidence"],
                candidate_sha256=candidate_hash,
            )
            eligible = bool(transaction["working_seed_eligible"] and controller_exclusion is None)
            category = (
                "working-seed"
                if eligible
                else "budget-incomplete"
                if transaction["process_incomplete"]
                else "failed"
                if submission_valid
                else "invalid-submission"
            )
            destination = (
                run_root / "candidates" / category / f"episode-{episode_index:03d}__{candidate_id}"
            )
            shutil.copytree(entry["candidate_path"], destination)
            if sha256_candidate_tree(destination) != candidate_hash:
                raise ValueError("candidate hash changed while preserving the episode artifact")
            audit_metadata = candidate_metadata(destination)
            record = {
                "episode_index": episode_index,
                "candidate_id": candidate_id,
                "candidate_sha256": candidate_hash,
                "artifact_path": destination.relative_to(run_root).as_posix(),
                "source_job": job_name,
                "source_trial": episode_result["trial_id"],
                "gate_report_sha256": hashlib.sha256(
                    episode_result["report_path"].read_bytes()
                ).hexdigest(),
                "mechanically_eligible": bool(candidate_result.get("mechanically_eligible")),
                "registry_distinct": bool(candidate_result.get("registry_distinct")),
                "semantic_review_ready": bool(candidate_result.get("semantic_review_ready")),
                "semantic_duplicate_risk": candidate_result.get(
                    "semantic_duplicate_risk", "not_evaluated"
                ),
                "failed_gates": sorted(
                    gate
                    for gate, passed in (candidate_result.get("gates") or {}).items()
                    if not passed
                ),
                "submission_valid": submission_valid,
                "reported_submission_valid": reported_submission_valid,
                "verifier_completed": episode_result["verifier_completed"],
                "canonical_admitted": False,
                **audit_metadata,
                "semantic_neighbors": semantic_neighbors(candidate_result),
            }
            if controller_exclusion is not None:
                record["controller_exclusion"] = controller_exclusion
            qd_attestation = (candidate_result.get("notes") or {}).get(
                "quality_diversity_attestation"
            )
            if qd_attestation is not None:
                record["quality_diversity_attestation"] = qd_attestation
            quality_diversity_decision = None
            if archive_policy := campaign.get("archive_policy"):
                if diversity_target is None:
                    raise ValueError("quality-diversity episode has no materialized target")
                archive, quality_diversity_decision = consider_quality_diversity_candidate(
                    archive_policy,
                    state["quality_diversity"],
                    candidate=record,
                    target=diversity_target,
                    protected_valid=eligible,
                )
                state["quality_diversity"] = archive
                record["quality_diversity"] = quality_diversity_decision
                policy_sequence += 1
                _append_jsonl(
                    policy_ledger_path,
                    {
                        "schema_version": "controller-policy-event-0.1.0",
                        "sequence": policy_sequence,
                        "observed_at": _utc(),
                        "kind": "quality_diversity_decision",
                        "episode_index": episode_index,
                        "data": quality_diversity_decision,
                    },
                )
            difficulty_feedback_result = None
            if campaign.get("feedback_policy") and eligible:
                feedback_output = (
                    run_root / "episodes" / f"{episode_index:03d}" / "feedback" / candidate_id
                )
                policy_sequence += 1
                feedback_started_at = _utc()
                feedback_started = time.monotonic()
                _append_jsonl(
                    policy_ledger_path,
                    {
                        "schema_version": "controller-policy-event-0.1.0",
                        "sequence": policy_sequence,
                        "observed_at": feedback_started_at,
                        "kind": "difficulty_feedback_started",
                        "episode_index": episode_index,
                        "data": {
                            "candidate_id": candidate_id,
                            "candidate_sha256": candidate_hash,
                            "maximum_examples_per_evaluator": campaign["feedback_policy"][
                                "max_examples_per_candidate"
                            ],
                            "evaluator_models": [
                                item["model_id"]
                                for item in campaign["feedback_policy"]["evaluators"]
                            ],
                        },
                    },
                )
                try:
                    expected_parent_candidate_id = None
                    expected_reasoning_target = None
                    if campaign["feedback_policy"]["id"] in SYNTHESIS_POLICY_IDS:
                        complete_feedback = [
                            row
                            for row in state["difficulty_feedback"]["candidate_feedback"]
                            if row.get("status") == "complete"
                        ]
                        if campaign["feedback_policy"]["id"] in {
                            BRANCHING_POLICY_ID,
                            RESILIENT_BRANCHING_POLICY_ID,
                            WIRED_RESILIENT_BRANCHING_POLICY_ID,
                            EFFORT_AWARE_BRANCHING_POLICY_ID,
                        }:
                            assignment = reasoning_branch_assignment(
                                campaign["feedback_policy"],
                                state["difficulty_feedback"],
                                episode_index,
                            )
                            expected_parent_candidate_id = assignment["parent_candidate_id"]
                            expected_reasoning_target = assignment["reasoning_target"]
                        else:
                            expected_parent_candidate_id = (
                                complete_feedback[-1]["candidate_id"]
                                if complete_feedback
                                else "initial_profile"
                            )
                    difficulty_feedback_result = evaluate_candidate_difficulty(
                        policy=campaign["feedback_policy"],
                        candidate_root=destination,
                        output_root=feedback_output,
                        campaign_id=campaign["id"],
                        candidate_id=candidate_id,
                        candidate_sha256=candidate_hash,
                        episode_index=episode_index,
                        render_runtime=campaign["environment"].get(
                            "render_runtime", "python-pillow@0.1.0"
                        ),
                        authorization_scope_id=campaign["comparison"]["experiment_id"],
                        live_model_metadata=feedback_preflight["evaluator_model_metadata"],
                        expected_parent_candidate_id=expected_parent_candidate_id,
                        expected_reasoning_target=expected_reasoning_target,
                    )
                    difficulty_feedback_result["artifact_path"] = feedback_output.relative_to(
                        run_root
                    ).as_posix()
                except Exception as exc:  # preserve evidence, then stop this arm fail-closed
                    secret = os.environ.get("OPENROUTER_API_KEY", "")
                    error = str(exc).replace(secret, "[REDACTED]") if secret else str(exc)
                    feedback_output.mkdir(parents=True, exist_ok=True)
                    difficulty_feedback_result = {
                        "schema_version": policy_schema_version(
                            campaign["feedback_policy"]["id"], "result"
                        ),
                        "policy_id": campaign["feedback_policy"]["id"],
                        "status": "incomplete",
                        "campaign_id": campaign["id"],
                        "episode_index": episode_index,
                        "candidate_id": candidate_id,
                        "candidate_sha256": candidate_hash,
                        "protected_valid": True,
                        "examples_per_evaluator": None,
                        "maximum_paid_requests": maximum_paid_requests(
                            campaign["feedback_policy"], candidate_count=1
                        ),
                        "sample_responses": [],
                        "evaluator_accuracies": {},
                        "aggregate_accuracy": None,
                        "hard_seed_eligible": None,
                        "difficulty_hypothesis": None,
                        "total_cost_usd": 0.0,
                        "artifact_path": feedback_output.relative_to(run_root).as_posix(),
                        "error_type": type(exc).__name__,
                        "error": error[:4000],
                        "evaluated_at": _utc(),
                    }
                    _atomic_json(
                        feedback_output / "feedback-failure.json",
                        difficulty_feedback_result,
                    )
                    feedback_episode_failure = {
                        "episode_index": episode_index,
                        "candidate_id": candidate_id,
                        "kind": "difficulty_feedback_incomplete",
                        "error_type": type(exc).__name__,
                        "error": error[:4000],
                    }
                if (
                    difficulty_feedback_result["status"] != "complete"
                    and feedback_episode_failure is None
                ):
                    feedback_episode_failure = {
                        "episode_index": episode_index,
                        "candidate_id": candidate_id,
                        "kind": "difficulty_feedback_incomplete",
                        "error_type": "IncompleteEvaluatorBatch",
                        "error": (
                            "one or more expected evaluator samples were not scored; "
                            "see the preserved feedback artifact"
                        ),
                    }
                if feedback_episode_failure is not None:
                    if campaign["feedback_policy"]["id"] in RESILIENT_POLICY_IDS:
                        consecutive_incomplete_feedback += 1
                        quarantine = {
                            **feedback_episode_failure,
                            "kind": "difficulty_feedback_quarantined",
                            "consecutive_incomplete_feedback": (consecutive_incomplete_feedback),
                        }
                        state.setdefault("difficulty_feedback_quarantines", []).append(quarantine)
                        maximum_incomplete = int(
                            campaign["feedback_policy"]["max_consecutive_incomplete_feedback"]
                        )
                        if (
                            _feedback_incomplete_action(
                                campaign["feedback_policy"],
                                consecutive_incomplete_feedback,
                            )
                            == "continue"
                        ):
                            feedback_episode_failure = None
                        else:
                            feedback_episode_failure = {
                                **quarantine,
                                "kind": "difficulty_feedback_circuit_breaker",
                                "error": (
                                    "maximum consecutive incomplete feedback episodes "
                                    f"reached: {maximum_incomplete}"
                                ),
                            }
                            state["infrastructure_outcomes"].append(feedback_episode_failure)
                            state["deviations"].append(feedback_episode_failure)
                    else:
                        state["infrastructure_outcomes"].append(feedback_episode_failure)
                        state["deviations"].append(feedback_episode_failure)
                else:
                    consecutive_incomplete_feedback = 0
                difficulty_feedback_result["timing"] = {
                    "started_at": feedback_started_at,
                    "finished_at": _utc(),
                    "seconds": round(time.monotonic() - feedback_started, 6),
                }
                state["difficulty_feedback"] = update_feedback_state(
                    state["difficulty_feedback"], difficulty_feedback_result
                )
                record["difficulty_feedback"] = {
                    key: difficulty_feedback_result.get(key)
                    for key in (
                        "policy_id",
                        "status",
                        "artifact_path",
                        "sample_set_sha256",
                        "examples_per_evaluator",
                        "maximum_paid_requests",
                        "sample_responses",
                        "evaluator_accuracies",
                        "aggregate_accuracy",
                        "hard_seed_eligible",
                        "difficulty_hypothesis",
                        "total_cost_usd",
                        "timing",
                    )
                }
                policy_sequence += 1
                _append_jsonl(
                    policy_ledger_path,
                    {
                        "schema_version": "controller-policy-event-0.1.0",
                        "sequence": policy_sequence,
                        "observed_at": difficulty_feedback_result["timing"]["finished_at"],
                        "kind": "difficulty_feedback_finished",
                        "episode_index": episode_index,
                        "data": record["difficulty_feedback"],
                    },
                )
            candidate_records.append(
                {
                    **record,
                    "candidate_transition": (
                        "working_seed_added"
                        if eligible
                        else "budget_incomplete_preserved"
                        if transaction["process_incomplete"]
                        else "failed_outcome_added"
                        if submission_valid
                        else "invalid_submission_preserved"
                    ),
                    "working_seed_eligible": eligible,
                    "protected_working_seed_eligible": mechanically_eligible,
                    "survivor_commit_complete": transaction["survivor_commit_complete"],
                    "process_incomplete": transaction["process_incomplete"],
                    "candidate_result": candidate_result,
                }
            )
            if eligible:
                accepted_paths.append(destination)
                state["working_seed"].append(record)
            elif not transaction["process_incomplete"]:
                failed_paths.append(destination)
                memory_verdict = candidate_result
                if controller_exclusion is not None:
                    memory_verdict = {
                        **candidate_result,
                        "gates": {
                            **(candidate_result.get("gates") or {}),
                            "controller_candidate_id_unique": False,
                        },
                        "notes": {
                            **(candidate_result.get("notes") or {}),
                            "controller_exclusion": controller_exclusion,
                        },
                    }
                failed_verdicts.append(memory_verdict)
                state["failed_outcomes"].append(record)

        eligible_count = sum(bool(record["working_seed_eligible"]) for record in candidate_records)
        process_incomplete_records = [
            record for record in candidate_records if record["process_incomplete"]
        ]
        failed_memory_added_count = sum(
            not record["working_seed_eligible"] and not record["process_incomplete"]
            for record in candidate_records
        )
        if candidate_records:
            valid_submission = (
                episode_result["verifier_completed"]
                and episode_result["rewards"].get("submission_valid") == 1.0
            )
            if not valid_submission:
                transition = "invalid_submission_preserved"
            elif len(process_incomplete_records) == len(candidate_records):
                transition = "budget_incomplete_preserved"
            elif eligible_count == len(candidate_records):
                transition = "working_seed_added"
            elif eligible_count == 0:
                transition = "failed_outcome_added"
            else:
                transition = "mixed_candidate_outcomes"
            if split_budget_outcomes and (budget_exhausted or process_incomplete_records):
                budget_outcome = {
                    "episode_index": episode_index,
                    "source_job": job_name,
                    "source_trial": episode_result["trial_id"],
                    "exception_type": episode_result["exception_type"],
                    "submission_valid": episode_result["rewards"].get("submission_valid"),
                    "candidate_transition": transition,
                    "reason": (
                        "paper_phase_evidence_incomplete"
                        if process_incomplete_records
                        else "agent_timeout"
                    ),
                }
                if campaign["strategy"] in {"sequential@0.1.0", "sequential@0.2.0"}:
                    budget_outcome["candidate_id"] = candidate_records[0]["candidate_id"]
                else:
                    budget_outcome["candidate_ids"] = [
                        record["candidate_id"] for record in candidate_records
                    ]
                state["budget_incomplete_outcomes"].append(budget_outcome)
        else:
            outcome = {
                "episode_index": episode_index,
                "source_job": job_name,
                "source_trial": episode_result["trial_id"],
                "exception_type": episode_result["exception_type"],
                "submission_valid": episode_result["rewards"].get("submission_valid"),
                "verifier_failures": (
                    episode_result["report"].get("infrastructure_failures") or []
                )[:5],
            }
            # An agent that exhausts its ordinary time budget has not hit an
            # infrastructure failure. Conflating the two misinforms later
            # episodes, which read this state as their situation report.
            transition = _missing_candidate_transition(
                exception_type=episode_result["exception_type"],
                split_budget=split_budget_outcomes,
            )
            if transition == "budget_incomplete":
                state["budget_incomplete_outcomes"].append(outcome)
            elif transition == "invalid_submission":
                state["failed_outcomes"].append(outcome)
            else:
                state["infrastructure_outcomes"].append(outcome)

        sole_candidate = candidate_records[0] if len(candidate_records) == 1 else None
        disposition = (
            "budget-incomplete"
            if sole_candidate and sole_candidate["process_incomplete"]
            else normalize_disposition(
                working_seed_eligible=bool(
                    sole_candidate and sole_candidate["working_seed_eligible"]
                ),
                exception_type=episode_result["exception_type"],
                verifier_completed=episode_result["verifier_completed"],
            )
        )
        phase_complete = (
            phase_evidence_complete(
                episode_result["phase_evidence"],
                disposition=disposition,
                candidate_sha256=(sole_candidate or {}).get("candidate_sha256"),
            )
            if paper_audit_enabled
            else True
        )
        if paper_audit_enabled and not phase_complete:
            audit_failure = {
                "episode_index": episode_index,
                "source_job": job_name,
                "source_trial": episode_result["trial_id"],
                "kind": "paper_phase_evidence_incomplete",
            }
            state["infrastructure_outcomes"].append(audit_failure)
            state["deviations"].append(audit_failure)
        episode_record = {
            "schema_version": "sequential-episode-record-0.1.0",
            "campaign_id": campaign["id"],
            "episode_index": episode_index,
            "packet_sha256": materialized.packet_sha256,
            "job_name": job_name,
            "trial_id": episode_result["trial_id"],
            "harbor_return_code": completed.returncode,
            "exception_type": episode_result["exception_type"],
            "verifier_completed": episode_result["verifier_completed"],
            "budget_exhausted": budget_exhausted,
            "adapter": campaign["agent"]["adapter"],
            "cost_reporting": campaign["campaign"].get("cost_reporting", "provider_native"),
            "rewards": episode_result["rewards"],
            "boundary_checks": episode_result["boundary_checks"],
            "candidate_id": sole_candidate["candidate_id"] if sole_candidate else None,
            "candidate_sha256": (sole_candidate["candidate_sha256"] if sole_candidate else None),
            "candidate_transition": transition,
            "working_seed_eligible": (
                sole_candidate["working_seed_eligible"] if sole_candidate else False
            ),
            "candidate_result": (sole_candidate["candidate_result"] if sole_candidate else None),
            "provider": episode_result["provider"],
            "agent_seconds": episode_result["agent_seconds"],
            "verifier_seconds": episode_result["verifier_seconds"],
            "disposition": disposition,
            "profile_reference": (materialization_record.get("discovery_profile") or {}).get(
                "reference"
            ),
            "episode_start_state": {
                "working_library_size": len(accepted_paths) - eligible_count,
                "failure_memory_size": len(failed_paths) - failed_memory_added_count,
                "state_sha256": materialization_record["state_sha256"],
                "packet_content_hash": materialized.packet_sha256,
            },
            "timing": {
                "materialization": {
                    "started_at": materialization_started_at,
                    "finished_at": materialization_finished_at,
                    "seconds": materialization_seconds,
                },
                **episode_result["lifecycle"],
                "useful_agent_time": {
                    "seconds": episode_result["agent_seconds"],
                    "definition": "Harbor agent_execution wall time",
                    "provider_call_wait_included": True,
                },
                "controller_retry_backoff_seconds": 0,
                "provider_recovered_retry_count": int(
                    episode_result["provider"].get("recovered_provider_error_count") or 0
                ),
            },
            "phase_evidence": episode_result["phase_evidence"],
            "phase_evidence_complete": phase_complete,
            "survivor_commit_complete": (
                sole_candidate["survivor_commit_complete"] if sole_candidate else None
            ),
            "harbor_result_sha256": episode_result["harbor_result_sha256"],
            "controller_observed_at": _utc(),
        }
        if diversity_target is not None:
            episode_record["controller_policy"] = {
                "kind": "quality_diversity",
                "policy_id": campaign["archive_policy"]["id"],
                "target": diversity_target,
                "decisions": [
                    record["quality_diversity"]
                    for record in candidate_records
                    if "quality_diversity" in record
                ],
                "archive_summary_after": quality_diversity_summary(state["quality_diversity"]),
            }
        if campaign.get("feedback_policy"):
            episode_record["controller_policy"] = {
                "kind": "difficulty_feedback",
                "policy_id": campaign["feedback_policy"]["id"],
                "candidate_feedback": [
                    record["difficulty_feedback"]
                    for record in candidate_records
                    if "difficulty_feedback" in record
                ],
                "feedback_state_after": {
                    key: state["difficulty_feedback"][key]
                    for key in (
                        "evaluated_candidate_count",
                        "hard_seed",
                        "valid_not_hard",
                        "incomplete",
                        "cumulative_cost_usd",
                    )
                },
            }
        if campaign["strategy"] not in {"sequential@0.1.0", "sequential@0.2.0"}:
            episode_record["schema_version"] = (
                "orchestration-session-record-0.2.0"
                if deadline_driven
                else "orchestration-session-record-0.1.0"
            )
            episode_record["orchestration_mode"] = state["orchestration_mode"]
            episode_record["candidate_records"] = candidate_records
        _atomic_json(
            run_root / "episodes" / f"{episode_index:03d}" / "episode-record.json",
            episode_record,
        )
        if paper_audit_enabled:
            ledger_sequence += 1
            _paper_ledger_event(
                paper_ledger_path,
                sequence=ledger_sequence,
                kind="episode_finalized",
                data={
                    "episode_index": episode_index,
                    "started_at": episode_result["lifecycle"]["trial"]["started_at"],
                    "finished_at": episode_result["lifecycle"]["trial"]["finished_at"],
                    "disposition": disposition,
                    "failed_gates": (
                        sole_candidate.get("failed_gates", []) if sole_candidate else []
                    ),
                    "candidate": (
                        {
                            key: sole_candidate.get(key)
                            for key in (
                                "candidate_id",
                                "candidate_sha256",
                                "task_signature_cell",
                                "task_signature",
                                "mechanism_fingerprint",
                                "semantic_neighbors",
                                "working_seed_eligible",
                                "protected_working_seed_eligible",
                                "survivor_commit_complete",
                                "process_incomplete",
                                "controller_exclusion",
                            )
                        }
                        if sole_candidate
                        else None
                    ),
                    "profile_reference": episode_record["profile_reference"],
                    "provider": episode_result["provider"],
                    "timing": episode_record["timing"],
                    "phase_evidence": episode_result["phase_evidence"],
                    "phase_evidence_complete": phase_complete,
                    "episode_start_state": episode_record["episode_start_state"],
                    "controller_policy": episode_record.get("controller_policy"),
                    "packet_sha256": materialized.packet_sha256,
                    "verifier": {
                        "completed": episode_result["verifier_completed"],
                        "rewards": episode_result["rewards"],
                        "boundary_checks": episode_result["boundary_checks"],
                    },
                },
            )
        sequence += 1
        completion_data = {
            "episode_index": episode_index,
            "job_name": job_name,
            "trial_id": episode_result["trial_id"],
            "candidate_sha256": (sole_candidate["candidate_sha256"] if sole_candidate else None),
            "transition": transition,
            "agent_seconds": episode_result["agent_seconds"],
            "verifier_seconds": episode_result["verifier_seconds"],
            "cost_usd": episode_result["provider"]["cost_usd"],
            "remaining_campaign_seconds": int(deadline - time.monotonic()),
            **(
                {
                    "remaining_agent_seconds": max(
                        0.0,
                        float(campaign["campaign"]["agent_time_seconds"])
                        - float(state["cumulative_agent_seconds"]),
                    )
                }
                if deadline_driven
                else {}
            ),
        }
        if campaign["strategy"] in {"sequential@0.1.0", "sequential@0.2.0"}:
            completion_data["candidate_id"] = (
                sole_candidate["candidate_id"] if sole_candidate else None
            )
        else:
            completion_data["candidate_ids"] = [
                record["candidate_id"] for record in candidate_records
            ]
        _event(
            events_path,
            sequence=sequence,
            kind="episode_completed",
            data=completion_data,
        )
        _atomic_json(state_path, state)
        cost_label = "cost" if enforce_cost_limit else "reported-cost(unreliable)"
        print(
            f"==> episode {episode_index} verdict: {transition}; "
            f"working_seed={len(accepted_paths)} "
            f"{cost_label}=${state['cumulative_cost_usd']:.4f}",
            flush=True,
        )
        if paper_audit_enabled and not phase_complete:
            finalize_run_evidence(run_root)
            raise ValueError("completed candidate is missing required paper phase evidence")
        if feedback_episode_failure is not None:
            stop_reason = "difficulty_feedback_infrastructure_failure"
            _atomic_json(state_path, state)
            print("==> difficulty feedback failed closed; stopping campaign", flush=True)
            break
        if episode_result["provider"].get("quota_exhausted"):
            # A provider usage cap is not an infrastructure fault and not an
            # ordinary budget: further episodes cannot succeed until it resets.
            state["deviations"].append(
                {
                    "episode_index": episode_index,
                    "kind": "provider_quota_exhausted",
                    "provider_errors": episode_result["provider"].get("provider_errors", []),
                }
            )
            stop_reason = "provider_quota_exhausted"
            _atomic_json(state_path, state)
            print("==> provider usage limit reached; stopping campaign", flush=True)
            break
        terminal_provider_failure = bool(
            episode_result["provider"].get("terminal_provider_failure")
        )
        if deadline_driven and terminal_provider_failure:
            consecutive_provider_failures += 1
            incident = {
                "episode_index": episode_index,
                "kind": (
                    "terminal_provider_overloaded"
                    if episode_result["provider"].get("transient_overloaded")
                    else "terminal_transient_rate_limit"
                    if episode_result["provider"].get("transient_rate_limited")
                    else "terminal_provider_failure"
                ),
                "consecutive_failures": consecutive_provider_failures,
                "provider_errors": episode_result["provider"].get("provider_errors", []),
            }
            state["provider_incidents"].append(incident)
            if consecutive_provider_failures >= int(
                campaign["campaign"]["max_consecutive_provider_failures"]
            ):
                stop_reason = "provider_failure_limit_reached"
                _atomic_json(state_path, state)
                print("==> repeated terminal provider failures; stopping campaign", flush=True)
                break
            if _another_episode_allowed(campaign, episode_index):
                delay = _provider_retry_delay(campaign, consecutive_provider_failures)
                remaining_before_retry = max(0, int(deadline - time.monotonic()))
                if remaining_before_retry > delay:
                    incident["retry_delay_seconds"] = delay
                    _atomic_json(state_path, state)
                    print(
                        f"==> terminal provider failure; retrying after {delay}s backoff",
                        flush=True,
                    )
                    if paper_audit_enabled:
                        ledger_sequence += 1
                        _paper_ledger_event(
                            paper_ledger_path,
                            sequence=ledger_sequence,
                            kind="provider_retry_backoff_started",
                            data={
                                "episode_index": episode_index,
                                "planned_seconds": delay,
                                "consecutive_failures": consecutive_provider_failures,
                            },
                        )
                    backoff_started = time.monotonic()
                    time.sleep(delay)
                    if paper_audit_enabled:
                        ledger_sequence += 1
                        _paper_ledger_event(
                            paper_ledger_path,
                            sequence=ledger_sequence,
                            kind="provider_retry_backoff_finished",
                            data={
                                "episode_index": episode_index,
                                "observed_seconds": round(time.monotonic() - backoff_started, 6),
                            },
                        )
        else:
            consecutive_provider_failures = 0
            if deadline_driven and int(
                episode_result["provider"].get("recovered_provider_error_count") or 0
            ):
                state["provider_incidents"].append(
                    {
                        "episode_index": episode_index,
                        "kind": "provider_error_recovered_within_session",
                        "count": int(episode_result["provider"]["recovered_provider_error_count"]),
                    }
                )
        consecutive_empty_submissions = _next_empty_submission_streak(
            consecutive_empty_submissions,
            transition=transition,
            candidate_count=len(candidate_records),
            exception_type=episode_result["exception_type"],
            terminal_provider_failure=terminal_provider_failure,
        )
        empty_submission_limit = campaign["campaign"].get("max_consecutive_empty_submissions")
        if (
            deadline_driven
            and empty_submission_limit is not None
            and consecutive_empty_submissions >= int(empty_submission_limit)
        ):
            incident = {
                "episode_index": episode_index,
                "kind": "repeated_empty_invalid_submissions",
                "consecutive_empty_submissions": consecutive_empty_submissions,
                "configured_limit": int(empty_submission_limit),
                "source_job": job_name,
                "source_trial": episode_result["trial_id"],
            }
            state["infrastructure_outcomes"].append(incident)
            stop_reason = "empty_submission_limit_reached"
            _atomic_json(state_path, state)
            print(
                "==> repeated empty invalid submissions; stopping campaign",
                flush=True,
            )
            break
        if enforce_cost_limit and not episode_result["provider"].get("cost_available"):
            state["deviations"].append(
                {
                    "episode_index": episode_index,
                    "kind": "provider_cost_unavailable",
                    "reason": "provider stream ended without a terminal cost record",
                }
            )
            stop_reason = "provider_cost_unavailable"
            _atomic_json(state_path, state)
            print(
                "==> provider-native cost is unavailable; stopping before another episode",
                flush=True,
            )
            break
        if enforce_cost_limit and state["cumulative_cost_usd"] >= float(
            campaign["campaign"]["soft_cost_limit_usd"]
        ):
            stop_reason = "soft_cost_limit_reached"
            break
        if time.monotonic() >= deadline:
            stop_reason = "campaign_wall_time_reached"
            break
        if _stop_after_current_episode_requested(run_root):
            incident = {
                "episode_index": episode_index,
                "kind": "operator_stop_after_current_episode",
                "instruction": "STOP_AFTER_CURRENT_EPISODE",
            }
            state["deviations"].append(incident)
            stop_reason = "operator_stop_after_current_episode"
            _atomic_json(state_path, state)
            print(
                "==> verifier-safe operator stop requested; campaign will not "
                "start another episode",
                flush=True,
            )
            break

    controller_wall_seconds = round(time.monotonic() - start_monotonic, 6)
    if deadline_driven and stop_reason in {
        "max_episodes",
        "single_continuous_session_completed",
        "episode_iterator_exhausted",
    }:
        remaining_agent = int(
            int(campaign["campaign"]["agent_time_seconds"])
            - float(state["cumulative_agent_seconds"])
        )
        if remaining_agent < int(campaign["campaign"]["minimum_start_seconds"]):
            stop_reason = "agent_budget_consumed"
        elif episode_limit_policy == "single_continuous_session":
            stop_reason = "continuous_session_ended_before_agent_budget"
        else:
            stop_reason = "session_safety_cap_reached_before_deadline"
    state["status"] = "completed"
    state["finished_at"] = _utc()
    state["stop_reason"] = stop_reason
    state["controller_wall_seconds"] = controller_wall_seconds
    if deadline_driven:
        state["budget_assessment"] = _budget_assessment(
            campaign,
            controller_wall_seconds=controller_wall_seconds,
            cumulative_agent_seconds=float(state["cumulative_agent_seconds"]),
            stop_reason=stop_reason,
            source_dirty=bool(state["source_dirty"]),
            infrastructure_outcome_count=len(state["infrastructure_outcomes"]),
            deviation_count=len(state["deviations"]),
        )
    _atomic_json(state_path, state)
    if campaign.get("archive_policy"):
        _atomic_json(
            run_root / "quality-diversity-summary.json",
            quality_diversity_summary(state["quality_diversity"]),
        )
    if campaign.get("feedback_policy"):
        _atomic_json(
            run_root / "feedback-usage-reconciliation.json",
            feedback_reconciliation(state["difficulty_feedback"]),
        )
        _atomic_json(
            run_root / "difficulty-feedback-summary.json",
            feedback_summary(state["difficulty_feedback"]),
        )
    sequence += 1
    _event(
        events_path,
        sequence=sequence,
        kind="campaign_completed",
        data={
            "stop_reason": stop_reason,
            "episodes_completed": state["episodes_completed"],
            "working_seed_size": len(state["working_seed"]),
            "failed_outcome_count": len(state["failed_outcomes"]),
            "infrastructure_outcome_count": len(state["infrastructure_outcomes"]),
            "budget_incomplete_count": len(state.get("budget_incomplete_outcomes", [])),
            "controller_wall_seconds": state["controller_wall_seconds"],
            "cumulative_cost_usd": state["cumulative_cost_usd"],
            **(
                {
                    "budget_fulfilled": state["budget_assessment"]["budget_fulfilled"],
                    "comparison_eligible": state["budget_assessment"]["comparison_eligible"],
                }
                if deadline_driven
                else {}
            ),
        },
    )
    if paper_audit_enabled:
        evidence = finalize_run_evidence(run_root)
        if not evidence["validation_passed"]:
            raise ValueError("paper-run evidence reconciliation failed closed")
    print(json.dumps(state, indent=2, sort_keys=True), flush=True)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        run_campaign(campaign_path=args.campaign.resolve(), run_root=args.run_root)
    except (OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise SystemExit(f"sequential campaign failed closed: {exc}") from exc


if __name__ == "__main__":
    main()
