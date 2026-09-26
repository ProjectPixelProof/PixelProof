"""Portable release CLI. Preparation and preview never contact model providers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from question_foundry.orchestration import render_compiled_campaign
from question_foundry.sequential import (
    initial_state,
    load_sequential_campaign,
    materialize_sequential_episode,
)
from scripts.run_sequential_campaign import (
    _install_feedback_bootstrap,
    _load_feedback_bootstrap,
    run_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILES = tuple(
    tomllib.loads(
        (ROOT / "foundry/discovery_profiles/paper-semantic/v0.3/profile-set.toml").read_text()
    )["profile_ids"]
)


def prepare(
    *,
    rq: int,
    agent: str,
    campaign_id: str,
    profile: str = "tracing",
    collection: str = "a",
    cli_version: str | None = None,
    agent_seconds: int = 1200,
    out: Path | None = None,
) -> Path:
    if not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", campaign_id):
        raise ValueError(
            "ID must be 3–64 lowercase letters/digits/underscores, starting with a letter"
        )
    if agent_seconds < 300 or agent_seconds > 21600:
        raise ValueError("agent-seconds must be between 300 and 21600")
    config = tomllib.loads((ROOT / f"experiments/rq{rq}/campaign.template.toml").read_text())
    preset = json.loads((ROOT / f"configs/agents/{agent}.json").read_text())
    config["id"] = campaign_id
    config["comparison"]["experiment_id"] = campaign_id
    config["comparison"]["budget_scope"] = "per_campaign_configured_agent_time"
    config["agent"] = preset["agent"]
    if cli_version:
        if not re.fullmatch(r"\d+\.\d+\.\d+", cli_version):
            raise ValueError("cli-version must be an exact X.Y.Z version")
        config["agent"]["cli_version"] = cli_version
    config["agent"]["timeout_seconds_per_episode"] = min(1200, agent_seconds)
    config["campaign"].update(
        agent_time_seconds=agent_seconds,
        minimum_start_seconds=min(300, agent_seconds),
        verifier_reserve_seconds=min(900, agent_seconds - 30),
        wall_time_seconds=max(3600, agent_seconds * (6 if rq == 3 else 2)),
        cost_reporting=preset["cost_reporting"],
        soft_cost_limit_usd=preset["soft_cost_limit_usd"],
    )
    if rq < 3:
        if profile not in PROFILES:
            raise ValueError(f"profile must be one of {PROFILES}")
        config["comparison"]["hypothesis_profile"] = f"paper-semantic@0.3.0:{profile}"
    else:
        config["comparison"]["hypothesis_profile"] = (
            f"paper-difficulty@0.3.0:frontier_hard_{collection}"
        )
        manifest = f"inputs/starting_collections/{collection}/manifest.json"
        config["feedback_bootstrap"] = {
            "id": f"starting_collection_{collection}",
            "manifest": manifest,
            "manifest_sha256": hashlib.sha256((ROOT / manifest).read_bytes()).hexdigest(),
        }
    text = render_compiled_campaign(config)
    destination = out or ROOT / "campaigns" / f"{campaign_id}.toml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        stream.write(text)
    return destination


def preview(campaign_path: Path, output: Path, *, oracle_fixture: bool = False):
    campaign = load_sequential_campaign(campaign_path)
    if output.exists():
        raise FileExistsError(f"preview output already exists: {output}")
    # Validate hashes before creating output. RQ3 previews install the real ten-world inputs.
    bootstrap = _load_feedback_bootstrap(campaign)
    output.mkdir(parents=True)
    state = initial_state(
        campaign, run_id="local-preview", started_at=datetime.now(UTC).isoformat()
    )
    accepted = []
    if bootstrap:
        accepted = _install_feedback_bootstrap(
            campaign=campaign, manifest=bootstrap[0], run_root=output, state=state
        )
    return materialize_sequential_episode(
        repo_root=ROOT,
        campaign_path=campaign_path,
        output_root=output / "episode",
        state=state,
        episode_index=1,
        timeout_seconds=campaign["agent"]["timeout_seconds_per_episode"],
        accepted_candidates=accepted,
        failed_candidates=[],
        failed_verdicts=[],
        source_revision="release-preview",
        source_dirty=True,
        include_oracle_fixture=oracle_fixture,
        feedback_artifact_root=output,
    )


def freeze(path: Path) -> None:
    config = load_sequential_campaign(path)
    _load_feedback_bootstrap(config)
    config["status"] = "frozen"
    path.write_text(render_compiled_campaign(config))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "profiles", help="list the nine discovery profiles used with --rq 1 and --rq 2"
    )
    p = commands.add_parser("prepare", help="write a new campaign without any model call")
    p.add_argument(
        "--rq",
        type=int,
        choices=(1, 2, 3),
        required=True,
        help="experiment: 1 = profile-steered, 2 = spatially-steered, 3 = model-feedback-steered generation",
    )
    p.add_argument("--agent", choices=("codex", "claude", "opencode"), required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--profile", choices=(*PROFILES, "all"), default="tracing")
    p.add_argument(
        "--collection",
        choices=("a", "b"),
        default="a",
        help="feedback example set for --rq 3",
    )
    p.add_argument("--cli-version")
    p.add_argument("--agent-seconds", type=int, default=1200)
    p.add_argument("--out", type=Path)
    p = commands.add_parser("preview", help="materialize exact first inputs without a provider")
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = commands.add_parser("freeze", help="freeze a reviewed campaign; commit it before running")
    p.add_argument("--campaign", type=Path, required=True)
    p = commands.add_parser("run", help="launch a frozen campaign; incurs provider usage")
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--confirm", required=True, help="must equal campaign ID")
    p.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "profiles":
        print("\n".join(PROFILES))
        return
    if args.command == "prepare":
        if args.profile == "all" and (args.rq == 3 or args.out):
            parser.error("--profile all is only for --rq 1 or --rq 2, without --out")
        profiles = PROFILES if args.profile == "all" else (args.profile,)
        for profile in profiles:
            cid = args.id + "_" + profile if args.profile == "all" else args.id
            print(
                prepare(
                    rq=args.rq,
                    agent=args.agent,
                    campaign_id=cid,
                    profile=profile,
                    collection=args.collection,
                    cli_version=args.cli_version,
                    agent_seconds=args.agent_seconds,
                    out=args.out,
                )
            )
    elif args.command == "preview":
        result = preview(args.campaign.resolve(), args.out.resolve())
        print(f"Provider calls: 0\nTask: {result.task}\nPacket SHA256: {result.packet_sha256}")
    elif args.command == "freeze":
        freeze(args.campaign)
        print("Frozen. Commit the reviewed configuration before running.")
    else:
        cfg = load_sequential_campaign(args.campaign)
        if args.confirm != cfg["id"]:
            parser.error("--confirm must equal the campaign ID")
        # Preserve auditable configs without requiring personal commit metadata.
        relative = args.campaign.resolve().relative_to(ROOT)
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(relative)],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
            raise ValueError("commit reviewed source/configuration changes before running")
        campaign_path = args.campaign.resolve()
        run_root = (args.out or ROOT / "artifacts" / cfg["id"]).resolve()
        os.chdir(ROOT)
        run_campaign(campaign_path=campaign_path, run_root=run_root)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        sys.exit(str(exc))
