"""Check the public envelope and timestamp a clean survivor commit."""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from pathlib import Path

from phase_audit import (
    DEFAULT_LEDGER,
    append_event,
    candidate_tree_sha256,
    portfolio_ids,
    read_events,
)

_ID = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_PORTFOLIO_KEYS = {"schema_version", "candidate_ids", "deviations"}


def check_submission(submission: Path, task_spec: Path) -> list[str]:
    failures: list[str] = []
    try:
        spec = tomllib.loads(task_spec.read_text(encoding="utf-8"))
        portfolio = json.loads((submission / "portfolio.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        return [f"submission or task specification is unreadable: {exc}"]
    if not isinstance(portfolio, dict) or set(portfolio) != _PORTFOLIO_KEYS:
        return [f"portfolio.json keys must be exactly {sorted(_PORTFOLIO_KEYS)}"]
    if portfolio.get("schema_version") != "0.4.0":
        failures.append("portfolio.json schema_version must be 0.4.0")
    deviations = portfolio.get("deviations")
    if not isinstance(deviations, list) or not all(isinstance(item, str) for item in deviations):
        failures.append("portfolio deviations must be a string list")
    candidate_ids = portfolio.get("candidate_ids")
    if not isinstance(candidate_ids, list):
        return failures + ["candidate_ids must be a list"]
    if any(not isinstance(item, str) or not _ID.fullmatch(item) for item in candidate_ids):
        failures.append("candidate_ids contain an invalid identifier")
    if len(candidate_ids) != len(set(candidate_ids)):
        failures.append("candidate_ids must be unique")
    reserved = spec.get("reserved_candidate_ids", [])
    if not isinstance(reserved, list) or not all(
        isinstance(item, str) and _ID.fullmatch(item) for item in reserved
    ):
        failures.append("TASK_SPEC.toml reserved_candidate_ids is invalid")
    elif reused := sorted(set(candidate_ids) & set(reserved)):
        failures.append(f"candidate_ids reuse prior campaign IDs: {reused}")
    policy = spec.get("candidate_limit_policy")
    minimum = int(spec.get("min_candidates", 0))
    if policy == "exactly_one_per_session" and len(candidate_ids) != 1:
        failures.append("episodic session must contain exactly one candidate")
    elif policy == "time_bounded_append_only" and len(candidate_ids) < minimum:
        failures.append(f"candidate count is below configured minimum {minimum}")
    candidates_root = submission / "candidates"
    entries = (
        sorted(path.name for path in candidates_root.iterdir()) if candidates_root.is_dir() else []
    )
    if entries != sorted(candidate_ids):
        failures.append("portfolio IDs differ from candidate-directory entries")
    for candidate_id in candidate_ids:
        path = candidates_root / candidate_id
        if not path.is_dir() or not (path / "candidate.json").is_file():
            failures.append(f"candidate bundle is incomplete: {candidate_id}")
        for required in ("world", "tests", "evidence"):
            if not (path / required).is_dir():
                failures.append(f"{candidate_id} is missing {required} directory")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=Path("/logs/artifacts/submission"))
    parser.add_argument("--task-spec", type=Path, default=Path("/workspace/TASK_SPEC.toml"))
    parser.add_argument("--phase-ledger", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args()
    ids = portfolio_ids(args.submission)
    append_event("envelope_check_started", candidate_ids=ids, path=args.phase_ledger)
    failures = check_submission(args.submission, args.task_spec)
    append_event(
        "envelope_check_finished",
        candidate_ids=ids,
        outcome="failed" if failures else "passed",
        details={"failure_count": len(failures)},
        path=args.phase_ledger,
    )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    events = read_events(args.phase_ledger)
    if not any(
        event["kind"] == "public_gate_finished" and event["outcome"] == "passed" for event in events
    ):
        raise SystemExit("FAIL: survivor commit requires a passing public gate")
    append_event(
        "survivor_committed",
        candidate_ids=ids,
        outcome="passed",
        details={
            "source": "passing_public_envelope_check",
            "candidate_sha256": candidate_tree_sha256(args.submission / "candidates" / ids[0]),
        },
        path=args.phase_ledger,
    )
    print("OK: public submission envelope is structurally clean and committed")


if __name__ == "__main__":
    main()
