"""Run the authoritative public subset of the question-world gatekeeper.

The materializer installs this file beside byte-identical copies of the
candidate gatekeeper, exact-evidence probe, and semantic validator used by the
protected verifier. The public run uses only the visible mechanism memories and
an empty registry. Hidden registry comparison and the clean-container boundary
remain protected.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path("/logs/artifacts/submission"),
    )
    parser.add_argument(
        "--task-spec",
        type=Path,
        default=Path("/workspace/TASK_SPEC.toml"),
    )
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
        "--registry",
        type=Path,
        default=Path("/workspace/memory/public_registry.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/workspace/scratch/public-gate-report.json"),
    )
    parser.add_argument(
        "--reward",
        type=Path,
        default=Path("/workspace/scratch/public-gate-reward.json"),
    )
    args = parser.parse_args()

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
    if completed.returncode != 0 or not args.report.is_file():
        raise SystemExit("FAIL: public candidate gate did not complete")

    report = json.loads(args.report.read_text(encoding="utf-8"))
    failures = []
    if not report.get("portfolio_valid"):
        failures.extend(report.get("infrastructure_failures") or ["portfolio is invalid"])
    for candidate in report.get("candidate_results") or []:
        candidate_id = candidate.get("candidate_id", "unknown")
        if not candidate.get("mechanically_eligible"):
            for gate, messages in sorted((candidate.get("failures") or {}).items()):
                for message in messages:
                    failures.append(f"{candidate_id}/{gate}: {message}")
        if not candidate.get("semantic_review_ready"):
            failures.append(f"{candidate_id}/semantic: visible-memory contrast is not review-ready")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        print(f"Full public report: {args.report}")
        raise SystemExit(1)
    count = len(report.get("candidate_results") or [])
    print(f"OK: {count} candidate(s) pass authoritative public gates")


if __name__ == "__main__":
    main()
