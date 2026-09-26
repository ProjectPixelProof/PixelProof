"""Fail closed before candidate execution unless the verifier is offline."""

from __future__ import annotations

import argparse
import json
import socket
import tomllib
from pathlib import Path


def network_blocked() -> bool:
    try:
        with socket.create_connection(("1.1.1.1", 53), timeout=1.0):
            return False
    except OSError:
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-spec", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    task_spec = tomllib.loads(args.task_spec.read_text(encoding="utf-8"))
    requires_recycle = task_spec.get("requires_clean_recycle") is True
    credential_paths = (
        Path("/root/.grok/auth.json"),
        Path("/home/agent/.grok/auth.json"),
        Path("/tmp/grok-foundry-oauth.json"),
        Path("/tmp/codex-oauth-upload.json"),
        Path("/logs/agent/auth.json"),
        Path("/root/.claude"),
        Path("/home/agent/.claude-foundry"),
        Path("/tmp/claude-foundry-oauth-token"),
        Path("/tmp/zai-coding-plan-key"),
    )
    checks = {
        "network_blocked": network_blocked(),
        "oauth_absent": not any(path.exists() or path.is_symlink() for path in credential_paths),
        "hidden_tests_present_now": Path("/tests/candidate_gatekeeper.py").is_file(),
        "clean_recycle_marker": (
            Path("/offline-verifier-ready").is_file() if requires_recycle else True
        ),
    }
    report = {
        "schema_version": "0.2.0",
        "campaign_id": task_spec.get("campaign_id"),
        "kind": "verifier-boundary",
        "requires_clean_recycle": requires_recycle,
        "checks": checks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    failed = sorted(key for key, value in checks.items() if not value)
    if failed:
        raise SystemExit(f"verifier boundary failed: {failed}")
    print(f"verifier boundary passed ({len(checks)} checks)")


if __name__ == "__main__":
    main()
