"""Protected integrity and funnel checks for question-world search traces."""

from __future__ import annotations

import argparse
import json
import re
import stat
import tomllib
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
EVENT_KEYS = {
    "schema_version",
    "sequence",
    "elapsed_seconds",
    "attempt_id",
    "state",
    "reason_code",
    "evidence",
}
RUN_KEYS = {"schema_version", "campaign_id", "status", "budget", "deviations"}
PROPOSAL_KEYS = {
    "schema_version",
    "attempt_id",
    "title",
    "question",
    "task_signature",
    "rationale",
    "anticipated_risks",
}
DISPOSITION_KEYS = {
    "schema_version",
    "attempt_id",
    "final_state",
    "reason_code",
    "evidence",
    "summary",
}
RUNTIME_KEYS = {"schema_version", "exit_code", "termination"}
RANKING_KEYS = {
    "schema_version",
    "pass_index",
    "elapsed_seconds",
    "considered_ids",
    "survivor_ids",
    "criteria",
}
TERMINAL_STATES = {
    "build_failed",
    "self_rejected",
    "omitted_during_ranking",
    "submitted",
}
ALLOWED_TRANSITIONS = {
    "proposed": {"selected_for_build", "self_rejected"},
    "selected_for_build": {"build_started", "self_rejected"},
    "build_started": {
        "build_failed",
        "self_check_failed",
        "self_check_passed",
    },
    "self_check_failed": {"build_started", "self_rejected"},
    "self_check_passed": {
        "self_rejected",
        "omitted_during_ranking",
        "submitted",
    },
}
MAX_FILES = 10_000
MAX_BYTES = 512 * 1024 * 1024


def _read_json(path: Path, failures: list[str], label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"{label} is invalid: {exc}")
        return {}
    if not isinstance(value, dict):
        failures.append(f"{label} must contain a JSON object")
        return {}
    return value


def _safe_tree(root: Path) -> list[str]:
    failures = []
    if not root.is_dir() or root.is_symlink():
        return ["process root is missing, not a directory, or a symlink"]
    file_count = 0
    total_bytes = 0
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            failures.append(f"symlink is forbidden: {path.relative_to(root)}")
            continue
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            failures.append(f"non-regular file is forbidden: {path.relative_to(root)}")
            continue
        file_count += 1
        total_bytes += path.stat().st_size
    if file_count > MAX_FILES:
        failures.append(f"process tree has {file_count} files; maximum is {MAX_FILES}")
    if total_bytes > MAX_BYTES:
        failures.append(f"process tree has {total_bytes} bytes; maximum is {MAX_BYTES}")
    return failures


def _valid_evidence(root: Path, values: object, failures: list[str], label: str) -> bool:
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        failures.append(f"{label} evidence must be a string list")
        return False
    valid = True
    for value in values:
        relative = PurePosixPath(value)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            failures.append(f"{label} has unsafe evidence path: {value!r}")
            valid = False
            continue
        if not (root / Path(*relative.parts)).exists():
            failures.append(f"{label} evidence path does not exist: {value!r}")
            valid = False
    return valid


def _expected_budget(task_spec: dict) -> dict:
    keys = [
        "timeout_seconds",
        "min_logged_proposals",
        "min_candidates",
    ]
    if task_spec.get("candidate_limit_policy") == "time_bounded_append_only":
        keys.extend(("min_ranking_passes", "candidate_limit_policy", "proposal_refresh_policy"))
    else:
        keys.extend(("max_turns", "max_build_starts", "max_candidates"))
    return {key: task_spec.get(key) for key in keys}


def _validate_run(root: Path, task_spec: dict, failures: list[str]) -> dict:
    run = _read_json(root / "run.json", failures, "run.json")
    if set(run) != RUN_KEYS:
        failures.append("run.json keys are invalid")
    if run.get("schema_version") != "search-trace-0.1.0":
        failures.append("run.json schema_version is invalid")
    if run.get("campaign_id") != task_spec.get("campaign_id"):
        failures.append("run.json campaign_id differs from TASK_SPEC.toml")
    if run.get("status") not in {"running", "completed"}:
        failures.append("run.json status is invalid")
    if run.get("budget") != _expected_budget(task_spec):
        failures.append("run.json budget differs from TASK_SPEC.toml")
    if not isinstance(run.get("deviations"), list) or not all(
        isinstance(item, str) for item in run.get("deviations", [])
    ):
        failures.append("run.json deviations must be a string list")
    return run


def _validate_runtime(root: Path, failures: list[str]) -> dict:
    runtime = _read_json(root / "runtime.json", failures, "runtime.json")
    if set(runtime) != RUNTIME_KEYS:
        failures.append("runtime.json keys are invalid")
    if runtime.get("schema_version") != "harbor-runtime-0.1.0":
        failures.append("runtime.json schema_version is invalid")
    if not isinstance(runtime.get("exit_code"), int):
        failures.append("runtime.json exit_code must be an integer")
    if runtime.get("termination") not in {"exit", "sigint", "sigterm", "cancelled"}:
        failures.append("runtime.json termination is invalid")
    return runtime


def _load_events(root: Path, failures: list[str]) -> list[dict]:
    path = root / "attempts.jsonl"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        failures.append(f"attempts.jsonl is missing: {exc}")
        return []
    events = []
    for line_number, line in enumerate(lines, 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append(f"attempts.jsonl line {line_number} is invalid JSON: {exc}")
            continue
        if not isinstance(event, dict):
            failures.append(f"attempts.jsonl line {line_number} is not an object")
            continue
        events.append(event)
    return events


def _validate_events(
    root: Path,
    events: list[dict],
    failures: list[str],
    *,
    protocol: str,
) -> dict[str, list[dict]]:
    allowed_transitions = {
        state: set(next_states) for state, next_states in ALLOWED_TRANSITIONS.items()
    }
    if protocol == "question-world@0.4.0":
        allowed_transitions["proposed"].add("omitted_during_ranking")
    by_attempt: dict[str, list[dict]] = defaultdict(list)
    last_elapsed = -1.0
    for expected_sequence, event in enumerate(events, 1):
        label = f"event {expected_sequence}"
        if set(event) != EVENT_KEYS:
            failures.append(f"{label} keys are invalid")
        if event.get("schema_version") != "search-event-0.1.0":
            failures.append(f"{label} schema_version is invalid")
        if event.get("sequence") != expected_sequence:
            failures.append(f"{label} sequence is not contiguous")
        elapsed = event.get("elapsed_seconds")
        if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or elapsed < 0:
            failures.append(f"{label} elapsed_seconds is invalid")
        elif elapsed < last_elapsed:
            failures.append(f"{label} elapsed_seconds moved backwards")
        else:
            last_elapsed = float(elapsed)
        attempt_id = event.get("attempt_id")
        if not isinstance(attempt_id, str) or not ID_PATTERN.fullmatch(attempt_id):
            failures.append(f"{label} attempt_id is invalid")
            continue
        state = event.get("state")
        reason = event.get("reason_code")
        if state not in ALLOWED_TRANSITIONS and state not in TERMINAL_STATES:
            failures.append(f"{label} state is invalid")
        if state in TERMINAL_STATES:
            if not isinstance(reason, str) or not ID_PATTERN.fullmatch(reason):
                failures.append(f"{label} terminal reason_code is invalid")
        elif reason is not None:
            failures.append(f"{label} non-terminal reason_code must be null")
        _valid_evidence(root, event.get("evidence"), failures, label)
        by_attempt[attempt_id].append(event)

    for attempt_id, attempt_events in by_attempt.items():
        states = [event.get("state") for event in attempt_events]
        if not states or states[0] != "proposed":
            failures.append(f"{attempt_id} does not begin with proposed")
            continue
        if states.count("proposed") != 1:
            failures.append(f"{attempt_id} has more than one proposed event")
        for previous, current in zip(states, states[1:], strict=False):
            if previous in TERMINAL_STATES:
                failures.append(f"{attempt_id} has an event after terminal state {previous}")
                continue
            if current not in allowed_transitions.get(previous, set()):
                failures.append(f"{attempt_id} has invalid transition {previous} -> {current}")
    return by_attempt


def _validate_proposal(root: Path, attempt_id: str, failures: list[str]) -> None:
    label = f"{attempt_id}/proposal.json"
    proposal = _read_json(root / "candidates" / attempt_id / "proposal.json", failures, label)
    if set(proposal) != PROPOSAL_KEYS:
        failures.append(f"{label} keys are invalid")
    if proposal.get("schema_version") != "search-proposal-0.1.0":
        failures.append(f"{label} schema_version is invalid")
    if proposal.get("attempt_id") != attempt_id:
        failures.append(f"{label} attempt_id differs from its directory")
    for key, minimum in (("title", 5), ("question", 10), ("rationale", 20)):
        if not isinstance(proposal.get(key), str) or len(proposal[key]) < minimum:
            failures.append(f"{label} {key} is too short")
    signature = proposal.get("task_signature")
    if not isinstance(signature, dict) or not {
        "decision_var",
        "structure",
        "decision_type",
        "mechanism",
    } <= set(signature):
        failures.append(f"{label} task_signature is incomplete")
    risks = proposal.get("anticipated_risks")
    if not isinstance(risks, list) or not all(
        isinstance(item, str) and len(item) >= 3 for item in risks
    ):
        failures.append(f"{label} anticipated_risks is invalid")


def _validate_disposition(
    root: Path,
    attempt_id: str,
    terminal_event: dict,
    failures: list[str],
) -> None:
    label = f"{attempt_id}/disposition.json"
    disposition = _read_json(
        root / "candidates" / attempt_id / "disposition.json",
        failures,
        label,
    )
    if set(disposition) != DISPOSITION_KEYS:
        failures.append(f"{label} keys are invalid")
    if disposition.get("schema_version") != "search-disposition-0.1.0":
        failures.append(f"{label} schema_version is invalid")
    if disposition.get("attempt_id") != attempt_id:
        failures.append(f"{label} attempt_id differs from its directory")
    if disposition.get("final_state") != terminal_event.get("state"):
        failures.append(f"{label} final_state differs from the terminal event")
    if disposition.get("reason_code") != terminal_event.get("reason_code"):
        failures.append(f"{label} reason_code differs from the terminal event")
    _valid_evidence(root, disposition.get("evidence"), failures, label)
    if not isinstance(disposition.get("summary"), str) or len(disposition["summary"]) < 10:
        failures.append(f"{label} summary is too short")


def _portfolio_ids(submission: Path) -> tuple[list[str], list[str]]:
    failures = []
    try:
        portfolio = json.loads((submission / "portfolio.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], [f"portfolio.json is unavailable to the process check: {exc}"]
    values = portfolio.get("candidate_ids") if isinstance(portfolio, dict) else None
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        failures.append("portfolio candidate_ids are invalid")
        return [], failures
    return values, failures


def _validate_rankings(
    root: Path,
    task_spec: dict,
    by_attempt: dict[str, list[dict]],
    failures: list[str],
    budget_failures: list[str],
) -> int:
    if task_spec.get("protocol") != "question-world@0.4.0":
        return 0
    minimum = int(task_spec.get("min_ranking_passes", 0))
    ranking_root = root / "rankings"
    files = sorted(ranking_root.glob("pass_*.json")) if ranking_root.is_dir() else []
    if len(files) < minimum:
        budget_failures.append(f"recorded {len(files)} ranking passes; minimum is {minimum}")
    rolling = task_spec.get("proposal_refresh_policy") == "replenish_ranked_queue"
    proposal_ids = set(by_attempt)
    selected_events = [
        event
        for events in by_attempt.values()
        for event in events
        if event.get("state") == "selected_for_build"
    ]
    first_selected = min(
        (float(event["elapsed_seconds"]) for event in selected_events),
        default=float("inf"),
    )
    previous_elapsed = -1.0
    previous_survivors: set[str] | None = None
    final_survivors: set[str] = set()
    survivor_history: list[tuple[float, set[str]]] = []
    initial_ranking_passes = 0
    for expected_index, path in enumerate(files, start=1):
        label = f"rankings/{path.name}"
        ranking = _read_json(path, failures, label)
        if set(ranking) != RANKING_KEYS:
            failures.append(f"{label} keys are invalid")
        if ranking.get("schema_version") != "search-ranking-0.1.0":
            failures.append(f"{label} schema_version is invalid")
        if ranking.get("pass_index") != expected_index:
            failures.append(f"{label} pass_index is not contiguous")
        elapsed = ranking.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or elapsed < previous_elapsed
        ):
            failures.append(f"{label} elapsed_seconds is invalid or moved backwards")
        else:
            previous_elapsed = float(elapsed)
            if float(elapsed) <= first_selected:
                initial_ranking_passes += 1
            elif not rolling:
                failures.append(f"{label} was recorded after selection began")
        considered = ranking.get("considered_ids")
        survivors = ranking.get("survivor_ids")
        active_at_ranking = {
            attempt_id
            for attempt_id, attempt_events in by_attempt.items()
            if float(attempt_events[0].get("elapsed_seconds", float("inf"))) <= float(elapsed or 0)
            and (
                attempt_events[-1].get("state") not in TERMINAL_STATES
                or float(attempt_events[-1].get("elapsed_seconds", -1)) > float(elapsed or 0)
            )
        }
        if (
            not isinstance(considered, list)
            or len(considered) != len(set(considered))
            or (
                set(considered) != active_at_ranking if rolling else set(considered) != proposal_ids
            )
        ):
            failures.append(f"{label} must consider the complete active proposal pool")
        if (
            not isinstance(survivors, list)
            or not survivors
            or len(survivors) != len(set(survivors))
            or not set(survivors) <= set(considered or [])
        ):
            failures.append(f"{label} survivor_ids are invalid")
            final_survivors = set()
        else:
            final_survivors = set(survivors)
            survivor_history.append((float(elapsed or 0), final_survivors))
            if (
                not rolling
                and previous_survivors is not None
                and not final_survivors <= previous_survivors
            ):
                failures.append(f"{label} reintroduced a candidate removed by an earlier pass")
            previous_survivors = final_survivors
        criteria = ranking.get("criteria")
        if (
            not isinstance(criteria, list)
            or len(criteria) < 3
            or not all(isinstance(item, str) and len(item) >= 3 for item in criteria)
        ):
            failures.append(f"{label} criteria are invalid")
    selected_ids = {
        attempt_id
        for attempt_id, events in by_attempt.items()
        if any(event.get("state") == "selected_for_build" for event in events)
    }
    if rolling:
        if initial_ranking_passes < minimum:
            budget_failures.append(
                f"completed {initial_ranking_passes} ranking passes before first selection; "
                f"minimum is {minimum}"
            )
        for event in selected_events:
            attempt_id = str(event.get("attempt_id"))
            selected_at = float(event.get("elapsed_seconds", -1))
            if not any(
                ranked_at <= selected_at and attempt_id in survivors
                for ranked_at, survivors in survivor_history
            ):
                failures.append(f"{attempt_id} was selected without a prior surviving rank")
    elif files and not selected_ids <= final_survivors:
        failures.append("selected candidates are not a subset of final ranking survivors")
    return len(files)


def validate_process(process: Path, submission: Path, task_spec: dict) -> dict:
    failures = _safe_tree(process)
    budget_failures = []
    if failures:
        return {
            "trace_valid": False,
            "budget_compliant": False,
            "failures": failures,
            "budget_failures": ["trace is unavailable"],
            "funnel": {},
            "terminal_reason_counts": {},
        }

    run = _validate_run(process, task_spec, failures)
    runtime = _validate_runtime(process, failures)
    events = _load_events(process, failures)
    by_attempt = _validate_events(
        process,
        events,
        failures,
        protocol=str(task_spec.get("protocol", "")),
    )
    ranking_passes = _validate_rankings(
        process,
        task_spec,
        by_attempt,
        failures,
        budget_failures,
    )

    candidate_root = process / "candidates"
    actual_dirs = (
        {path.name for path in candidate_root.iterdir() if path.is_dir()}
        if candidate_root.is_dir()
        else set()
    )
    if actual_dirs != set(by_attempt):
        failures.append(
            "process candidate directories differ from attempts: "
            f"directories={sorted(actual_dirs)} attempts={sorted(by_attempt)}"
        )

    built_ids = set()
    submitted_ids = set()
    terminal_reasons = Counter()
    state_counts = Counter()
    for attempt_id, attempt_events in by_attempt.items():
        _validate_proposal(process, attempt_id, failures)
        states = [event.get("state") for event in attempt_events]
        state_counts.update(set(states))
        if "build_started" in states:
            built_ids.add(attempt_id)
            checkpoint = process / "candidates" / attempt_id / "checkpoint"
            if not checkpoint.is_dir() or not any(checkpoint.iterdir()):
                failures.append(f"{attempt_id} reached build_started without a checkpoint")
        terminal = attempt_events[-1]
        if terminal.get("state") in TERMINAL_STATES:
            _validate_disposition(process, attempt_id, terminal, failures)
            terminal_reasons[str(terminal.get("reason_code"))] += 1
        if terminal.get("state") == "submitted":
            submitted_ids.add(attempt_id)

    proposal_count = len(by_attempt)
    min_proposals = int(task_spec.get("min_logged_proposals", 0))
    if proposal_count < min_proposals:
        budget_failures.append(f"logged {proposal_count} proposals; minimum is {min_proposals}")
    if "max_build_starts" in task_spec:
        max_builds = int(task_spec["max_build_starts"])
        if len(built_ids) > max_builds:
            budget_failures.append(f"started {len(built_ids)} builds; maximum is {max_builds}")

    portfolio_ids, portfolio_failures = _portfolio_ids(submission)
    completed = run.get("status") == "completed"
    if completed:
        failures.extend(portfolio_failures)
        if set(portfolio_ids) != submitted_ids:
            failures.append(
                "submitted trace attempts differ from portfolio IDs: "
                f"trace={sorted(submitted_ids)} portfolio={sorted(portfolio_ids)}"
            )
        minimum = int(task_spec.get("min_candidates", 1))
        if task_spec.get("candidate_limit_policy") == "time_bounded_append_only":
            if len(portfolio_ids) < minimum:
                budget_failures.append(
                    f"submitted {len(portfolio_ids)} candidates; minimum is {minimum}"
                )
        else:
            maximum = int(task_spec.get("max_candidates", 1))
            if not minimum <= len(portfolio_ids) <= maximum:
                budget_failures.append(
                    f"submitted {len(portfolio_ids)} candidates; required range is "
                    f"[{minimum}, {maximum}]"
                )
    elif runtime.get("exit_code") == 0:
        failures.append("agent exited zero but run.json remained running")

    if completed and runtime.get("exit_code") != 0:
        failures.append("run.json claims completed after a nonzero agent exit")

    funnel = {
        "proposed": proposal_count,
        "selected_for_build": state_counts["selected_for_build"],
        "build_started": len(built_ids),
        "self_check_passed": state_counts["self_check_passed"],
        "self_rejected": state_counts["self_rejected"],
        "build_failed": state_counts["build_failed"],
        "omitted_during_ranking": state_counts["omitted_during_ranking"],
        "submitted": len(submitted_ids),
    }
    return {
        "trace_valid": not failures,
        "budget_compliant": not budget_failures,
        "run_status": run.get("status"),
        "runtime": runtime,
        "failures": failures,
        "budget_failures": budget_failures,
        "funnel": funnel,
        "terminal_reason_counts": dict(sorted(terminal_reasons.items())),
        "ranking_passes_completed": ranking_passes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--process", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--task-spec", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reward", type=Path, required=True)
    args = parser.parse_args()

    try:
        task_spec = tomllib.loads(args.task_spec.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        task_spec = {}
        report = {
            "schema_version": "search-trace-report-0.1.0",
            "trace_valid": False,
            "budget_compliant": False,
            "failures": [f"TASK_SPEC.toml is invalid: {exc}"],
            "budget_failures": ["task budget is unavailable"],
            "funnel": {},
            "terminal_reason_counts": {},
        }
    else:
        report = {
            "schema_version": "search-trace-report-0.1.0",
            "campaign_id": task_spec.get("campaign_id"),
            **validate_process(args.process, args.submission, task_spec),
        }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.reward.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        reward = json.loads(args.reward.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        reward = {}
    reward["process_trace_valid"] = float(report["trace_valid"])
    reward["search_budget_compliant"] = float(report["budget_compliant"])
    args.reward.write_text(json.dumps(reward, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "process trace: "
        f"valid={report['trace_valid']} budget_compliant={report['budget_compliant']} "
        f"funnel={report['funnel']}"
    )


if __name__ == "__main__":
    main()
