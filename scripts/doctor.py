"""Check local runtime dependencies without reading credentials or contacting providers."""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import subprocess
import sys

from question_foundry.render_runtime import LATEX_TIKZ_IMAGE, LATEX_TIKZ_REPLAY_IMAGE


def main():
    checks = {"python": {"ok": sys.version_info >= (3, 12), "version": sys.version.split()[0]}}
    try:
        version = importlib.metadata.version("harbor")
        checks["harbor"] = {"ok": version == "0.1.44", "version": version}
    except importlib.metadata.PackageNotFoundError:
        checks["harbor"] = {"ok": False, "hint": "uv sync --locked --extra runner --group dev"}
    commands = {
        "docker": ["docker", "info", "--format", "{{.ServerVersion}}"],
        "compose": ["docker", "compose", "version", "--short"],
    }
    for image in [LATEX_TIKZ_IMAGE, LATEX_TIKZ_REPLAY_IMAGE]:
        commands[image] = ["docker", "image", "inspect", "--format", "{{.Id}}", image]
    for name, command in commands.items():
        if not shutil.which(command[0]):
            checks[name] = {"ok": False, "hint": "install/start Docker with Compose v2"}
            continue
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, timeout=30, check=False
            )
            checks[name] = {
                "ok": result.returncode == 0,
                "version_or_id": result.stdout.strip() if result.returncode == 0 else None,
            }
        except subprocess.TimeoutExpired:
            checks[name] = {"ok": False, "hint": "Docker command timed out"}
    print(json.dumps({"provider_calls": 0, "checks": checks}, indent=2))
    raise SystemExit(0 if all(v["ok"] for v in checks.values()) else 1)


if __name__ == "__main__":
    main()
