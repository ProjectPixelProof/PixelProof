"""Scan the allowlisted public payload; never print credential-like file contents."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "uv.lock",
    ".gitignore",
    ".python-version",
}
DIRECTORIES = {
    "src",
    "scripts",
    "tests",
    "worlds",
    "harbor",
    "foundry",
    "registry",
    "configs",
    "experiments",
    "inputs",
    "docs",
    "examples",
}
# Explicitly reviewed README illustration; arbitrary image exports remain excluded.
DOCUMENTATION_IMAGES = {
    "docs/assets/figure-1.png": "495689244d9785db005da4bac7c5ce013151c7e95a823a271ec38bf70cc90464"
}
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache"}


def payload_files(root: Path = ROOT):
    for path in sorted(root.iterdir()):
        if path.is_symlink() and path.name in ROOT_FILES | DIRECTORIES:
            yield path
        elif path.is_file() and path.name in ROOT_FILES:
            yield path
        elif path.is_dir() and path.name in DIRECTORIES:
            for item in sorted(path.rglob("*")):
                if any(part in SKIP_PARTS for part in item.relative_to(root).parts):
                    continue
                if (
                    item.name.startswith("._")
                    or item.name == ".DS_Store"
                    or item.suffix in {".pyc", ".pyo"}
                ):
                    continue
                if item.is_file() or item.is_symlink():
                    yield item


def audit(root: Path = ROOT):
    issues = []
    patterns = [
        re.compile(rb"/(?:Users|Volumes|home)/[A-Za-z0-9_.-]+/"),
        re.compile(rb"(?:sk-ant-|sk-or-v1-|sk-proj-)[A-Za-z0-9_-]{20,}"),
        re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    ]
    count = size = 0
    for path in payload_files(root):
        relative = path.relative_to(root).as_posix()
        count += 1
        if path.is_symlink():
            issues.append({"path": relative, "reason": "symlink in public payload"})
            continue
        data = path.read_bytes()
        size += len(data)
        # /home/agent is the intentional unprivileged container user.
        inspected = data.replace(b"/home/agent/", b"/container-user/")
        if any(pattern.search(inspected) for pattern in patterns):
            issues.append({"path": relative, "reason": "identity/credential-like pattern"})
        approved_image = DOCUMENTATION_IMAGES.get(relative) == hashlib.sha256(data).hexdigest()
        if not approved_image and path.suffix.lower() in {
            ".pdf",
            ".tex",
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".zip",
            ".tar",
            ".gz",
            ".log",
        }:
            issues.append(
                {"path": relative, "reason": "excluded publication/runtime artifact type"}
            )
        if path.name in {"trajectory.json", "claude-code.jsonl", "auth.json", ".env"}:
            issues.append({"path": relative, "reason": "excluded trace or credential filename"})
    if (root / ".git").exists():
        tracked = subprocess.run(
            ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=False
        )
        if tracked.returncode == 0:
            allowed = {p.relative_to(root).as_posix() for p in payload_files(root)}
            for name in tracked.stdout.decode().split("\0"):
                if name and name not in allowed:
                    issues.append({"path": name, "reason": "tracked file outside public payload"})
    return {
        "status": "passed" if not issues else "failed",
        "files": count,
        "bytes": size,
        "issues": issues,
    }


def main():
    result = audit()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
