"""Explicit, small paid authentication sentinel for one configured coding agent."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from question_foundry.sequential import load_sequential_campaign
from scripts.check_auth_smoke import patterns_from_env, patterns_from_file, scan_tree, validate_job
from scripts.release import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--confirm", required=True, help="campaign ID; permits this provider call")
    args = parser.parse_args()
    cfg = load_sequential_campaign(args.campaign)
    if args.confirm != cfg["id"]:
        parser.error("--confirm must equal the campaign ID")
    destination = args.out.resolve()
    if destination.exists():
        raise FileExistsError("auth smoke output already exists")
    agent = cfg["agent"]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "harbor/agents") + os.pathsep + env.get("PYTHONPATH", "")
    patterns = patterns_from_env(["CLAUDE_CODE_OAUTH_TOKEN", "OPENROUTER_API_KEY"])
    if agent["adapter"] == "codex_oauth:CodexOAuth":
        auth = Path(env.get("CODEX_AUTH_FILE", "~/.codex/auth.json")).expanduser().resolve()
        env["CODEX_AUTH_FILE"] = str(auth)
        patterns |= patterns_from_file(auth)
        env.pop("OPENAI_API_KEY", None)
        env.pop("OPENAI_BASE_URL", None)
    elif agent["adapter"] == "claude_code_foundry:ClaudeCodeFoundry":
        if not env.get("CLAUDE_CODE_OAUTH_TOKEN"):
            raise ValueError("CLAUDE_CODE_OAUTH_TOKEN is required")
        for key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
            env.pop(key, None)
    elif not env.get("OPENROUTER_API_KEY"):
        raise ValueError("OPENROUTER_API_KEY is required")
    command = [
        "harbor",
        "run",
        "-p",
        str(ROOT / "harbor/datasets/auth-smoke-v0/agent-auth-smoke"),
        "--agent-import-path",
        agent["adapter"],
        "-m",
        agent["model"],
        "--ak",
        f"version={agent['cli_version']}",
        "--ak",
        f"reasoning_effort={agent['reasoning_effort']}",
        "--ak",
        f"output_format={agent['output_format']}",
        "-e",
        "docker",
        "-o",
        str(destination.parent),
        "--job-name",
        destination.name,
        "--delete",
        "--n-concurrent",
        "1",
    ]
    for key in ("upstream_provider", "allow_provider_fallbacks"):
        if key in agent:
            value = str(agent[key]).lower() if isinstance(agent[key], bool) else agent[key]
            command += ["--ak", f"{key}={value}"]
    # Always scan the files even if the provider or harness failed.
    result = subprocess.run(command, cwd=ROOT, env=env, check=False)
    leaks = scan_tree(destination, patterns)
    if leaks:
        raise ValueError(f"credential scan failed in {len(leaks)} files; keep output private")
    result.check_returncode()
    validate_job(destination)
    print(
        json.dumps(
            {
                "auth_sentinel": "passed",
                "secret_scan": "passed",
                "model": agent["model"],
                "note": "Inspect the retained served-model identity before confirming entitlement.",
            }
        )
    )


if __name__ == "__main__":
    main()
