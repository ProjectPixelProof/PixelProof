"""Materialize one content-hashed sequential foundry episode without model calls."""

from __future__ import annotations

import argparse
import json
import subprocess
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from question_foundry.sequential import (
    initial_state,
    load_sequential_campaign,
    materialize_sequential_episode,
)

ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def materialize(args: argparse.Namespace) -> None:
    campaign = load_sequential_campaign(args.campaign)
    dirty = bool(_git("status", "--porcelain"))
    if dirty and not args.allow_dirty:
        raise SystemExit("worktree is dirty; pass --allow-dirty for a developmental packet")
    started_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    state = initial_state(campaign, run_id="materialization-preview", started_at=started_at)
    result = materialize_sequential_episode(
        repo_root=ROOT,
        campaign_path=args.campaign,
        output_root=args.out,
        state=state,
        episode_index=1,
        timeout_seconds=(
            args.timeout_seconds or int(campaign["agent"]["timeout_seconds_per_episode"])
        ),
        accepted_candidates=[],
        failed_candidates=[],
        failed_verdicts=[],
        source_revision=_git("rev-parse", "HEAD"),
        source_dirty=dirty,
        include_oracle_fixture=args.oracle_fixture,
    )
    print(f"materialized sequential episode: {result.root}")
    print(f"packet_sha256: {result.packet_sha256}")
    print(f"task: {result.task}")
    print("model_calls: 0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--oracle-fixture", action="store_true")
    args = parser.parse_args()
    try:
        materialize(args)
    except (
        OSError,
        ValueError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
        tomllib.TOMLDecodeError,
    ) as exc:
        raise SystemExit(f"sequential materialization failed closed: {exc}") from exc


if __name__ == "__main__":
    main()
