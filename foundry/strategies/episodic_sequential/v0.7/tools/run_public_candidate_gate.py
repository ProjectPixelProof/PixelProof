"""Run and timestamp the authoritative public question-world gates."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from phase_audit import DEFAULT_LEDGER, append_event, portfolio_ids, read_events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=Path("/logs/artifacts/submission"))
    parser.add_argument("--task-spec", type=Path, default=Path("/workspace/TASK_SPEC.toml"))
    parser.add_argument(
        "--mechanism-memory",
        type=Path,
        default=Path("/workspace/memory/known_mechanisms.jsonl"),
    )
    parser.add_argument(
        "--negative-memory",
        type=Path,
        default=Path("/workspace/memory/rejected_mechanisms.jsonl"),
    )
    parser.add_argument(
        "--registry", type=Path, default=Path("/workspace/memory/public_registry.jsonl")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("/workspace/scratch/public-gate-report.json")
    )
    parser.add_argument(
        "--reward", type=Path, default=Path("/workspace/scratch/public-gate-reward.json")
    )
    parser.add_argument("--phase-ledger", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args()

    ids = portfolio_ids(args.submission)
    if not any(event["kind"] == "candidate_staged" for event in read_events(args.phase_ledger)):
        raise SystemExit("FAIL: record candidate staging before the first public gate")
    attempt = 1 + sum(
        event["kind"] == "public_gate_started" for event in read_events(args.phase_ledger)
    )
    append_event(
        "public_gate_started",
        candidate_ids=ids,
        details={"attempt": attempt},
        path=args.phase_ledger,
    )

    tools = Path(__file__).resolve().parent
    args.report.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            sys.executable,
            str(tools / "candidate_gatekeeper.py"),
            "--submission",
            str(args.submission),
            "--task-spec",
            str(args.task_spec),
            "--registry",
            str(args.registry),
            "--mechanism-memory",
            str(args.mechanism_memory),
            "--negative-memory",
            str(args.negative_memory),
            "--report",
            str(args.report),
            "--reward",
            str(args.reward),
        ],
        check=False,
    )
    report = json.loads(args.report.read_text(encoding="utf-8")) if args.report.is_file() else {}
    failures: list[str] = []
    if completed.returncode != 0 or not report:
        failures.append("public candidate gate did not complete")
    elif not report.get("portfolio_valid"):
        failures.extend(report.get("infrastructure_failures") or ["portfolio is invalid"])
    for candidate in report.get("candidate_results") or []:
        candidate_id = candidate.get("candidate_id", "unknown")
        if not candidate.get("mechanically_eligible"):
            for gate, messages in sorted((candidate.get("failures") or {}).items()):
                failures.extend(f"{candidate_id}/{gate}: {message}" for message in messages)
        if not candidate.get("semantic_review_ready"):
            failures.append(f"{candidate_id}/semantic: visible-memory contrast is not review-ready")
    append_event(
        "public_gate_finished",
        candidate_ids=ids,
        outcome="failed" if failures else "passed",
        details={"attempt": attempt, "failure_count": len(failures)},
        path=args.phase_ledger,
    )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        print(f"Full public report: {args.report}")
        raise SystemExit(1)
    print(f"OK: {len(report.get('candidate_results') or [])} candidate(s) pass public gates")


if __name__ == "__main__":
    main()
