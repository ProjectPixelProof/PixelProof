"""Public, agent-visible structural check for a question-world portfolio.

This deliberately checks only the transport envelope. It does not run protected
mechanical gates, inspect hidden registries, or claim scientific eligibility.
"""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from pathlib import Path

_ID = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_PORTFOLIO_KEYS = {"schema_version", "candidate_ids", "deviations"}


def check_submission(submission: Path, task_spec: Path) -> list[str]:
    failures: list[str] = []
    try:
        spec = tomllib.loads(task_spec.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        return [f"TASK_SPEC.toml is unreadable: {exc}"]

    portfolio_path = submission / "portfolio.json"
    candidates_root = submission / "candidates"
    try:
        portfolio = json.loads(portfolio_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"portfolio.json is unreadable: {exc}"]
    if not isinstance(portfolio, dict) or set(portfolio) != _PORTFOLIO_KEYS:
        failures.append(f"portfolio.json keys must be exactly {sorted(_PORTFOLIO_KEYS)}")
        return failures
    if portfolio.get("schema_version") != "0.4.0":
        failures.append("portfolio.json schema_version must be 0.4.0")
    deviations = portfolio.get("deviations")
    if not isinstance(deviations, list) or not all(isinstance(item, str) for item in deviations):
        failures.append("portfolio deviations must be a string list")

    candidate_ids = portfolio.get("candidate_ids")
    if not isinstance(candidate_ids, list):
        failures.append("candidate_ids must be a list")
        return failures
    if any(not isinstance(item, str) or not _ID.fullmatch(item) for item in candidate_ids):
        failures.append("candidate_ids contain an invalid identifier")
    if len(candidate_ids) != len(set(candidate_ids)):
        failures.append("candidate_ids must be unique")
    reserved_ids = spec.get("reserved_candidate_ids", [])
    if not isinstance(reserved_ids, list) or not all(
        isinstance(item, str) and _ID.fullmatch(item) for item in reserved_ids
    ):
        failures.append("TASK_SPEC.toml reserved_candidate_ids is invalid")
    else:
        reused = sorted(set(candidate_ids) & set(reserved_ids))
        if reused:
            failures.append(f"candidate_ids reuse prior campaign IDs: {reused}")
    minimum = int(spec.get("min_candidates", 0))
    policy = spec.get("candidate_limit_policy")
    if policy == "time_bounded_append_only":
        if len(candidate_ids) < minimum:
            failures.append(
                f"candidate count {len(candidate_ids)} is below configured minimum {minimum}"
            )
    elif policy == "exactly_one_per_session":
        if len(candidate_ids) != 1:
            failures.append("episodic session must contain exactly one candidate")
    else:
        maximum = int(spec.get("max_candidates", 0))
        if not minimum <= len(candidate_ids) <= maximum:
            failures.append(
                f"candidate count {len(candidate_ids)} is outside configured [{minimum}, {maximum}]"
            )

    if not candidates_root.is_dir():
        failures.append("submission/candidates is missing")
        return failures
    entries = sorted(path.name for path in candidates_root.iterdir())
    expected = sorted(candidate_ids)
    if entries != expected:
        failures.append(
            f"portfolio IDs {expected} differ from candidate-directory entries {entries}; "
            "move scratch files to /workspace/scratch"
        )
    for candidate_id in candidate_ids:
        path = candidates_root / candidate_id
        if not path.is_dir():
            failures.append(f"candidate directory is missing: {candidate_id}")
            continue
        if not (path / "candidate.json").is_file():
            failures.append(f"{candidate_id} is missing candidate.json file")
        for required in ("world", "tests", "evidence"):
            if not (path / required).is_dir():
                failures.append(f"{candidate_id} is missing {required} directory")
    return failures


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
    args = parser.parse_args()
    failures = check_submission(args.submission, args.task_spec)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("OK: public submission envelope is structurally clean")


if __name__ == "__main__":
    main()
