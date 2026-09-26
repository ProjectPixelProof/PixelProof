"""Test episode preparation and final verification without a model."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from scripts.release import ROOT, prepare, preview


def validate_canary(boundary, rewards):
    required_checks = {
        "clean_recycle_marker",
        "hidden_tests_present_now",
        "network_blocked",
        "oauth_absent",
    }
    checks = boundary.get("checks", {})
    if (
        boundary.get("failures")
        or not required_checks.issubset(checks)
        or not all(value is True for value in checks.values())
    ):
        raise ValueError("missing or failed protected boundary check")
    required_rewards = {
        "all_mechanically_eligible",
        "mechanically_eligible_fraction",
        "registry_distinct_fraction",
        "semantic_review_ready_fraction",
        "submission_valid",
    }
    if (
        not rewards
        or not required_rewards.issubset(rewards)
        or any(rewards[key] != 1.0 for key in required_rewards)
    ):
        raise ValueError("missing or failed fixture reward")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/docker-smoke")
    parser.add_argument(
        "--rq",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="experiment: 1 = profile-steered, 2 = spatially-steered, 3 = model-feedback-steered generation",
    )
    parser.add_argument(
        "--collection", choices=["a", "b"], default="a", help="feedback example set for --rq 3"
    )
    args = parser.parse_args()
    root = args.out.resolve()
    if root.exists():
        raise FileExistsError("choose a new --out directory")
    root.mkdir(parents=True)
    path = prepare(
        rq=args.rq,
        agent="codex",
        campaign_id="offline_canary",
        collection=args.collection,
        out=root / "campaign.toml",
    )
    packet = preview(path, root / "preview", oracle_fixture=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "harbor/agents") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [
            "harbor",
            "run",
            "-c",
            str(ROOT / "harbor/job-configs/vector_rewards.yaml"),
            "-p",
            str(packet.task),
            "--agent-import-path",
            "offline_recycle_oracle:OfflineRecycleOracle",
            "-e",
            "docker",
            "-o",
            str(root / "jobs"),
            "--job-name",
            "offline-canary",
            "--delete",
            "--n-concurrent",
            "1",
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    job = root / "jobs/offline-canary"
    trials = list(job.glob("*/result.json"))
    if len(trials) != 1:
        raise ValueError("expected one Harbor trial")
    result = json.loads(trials[0].read_text())
    if result.get("exception_info") is not None:
        raise ValueError(f"Harbor infrastructure failed: {result['exception_info']}")
    trial = trials[0].parent
    reports = list(trial.rglob("boundary_report.json"))
    if not reports:
        raise ValueError("missing protected boundary report")
    boundary = json.loads(reports[0].read_text())
    validate_canary(boundary, (result.get("verifier_result") or {}).get("rewards"))
    rewards = result["verifier_result"]["rewards"]
    summary = {
        "provider_calls": 0,
        "experiment": f"RQ{args.rq}",
        "harbor_trial_exception": None,
        "boundary": boundary,
        "rewards": rewards,
        "scope": "Fixed-example check of execution and isolation; no model generation.",
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
