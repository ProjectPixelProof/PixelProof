"""Fail unless a Harbor auth smoke passed without persisting credentials.

The scanner receives credential sources at runtime, derives raw/base64/token
patterns in memory, and scans regular job files without printing any secret. It
then checks the trial reward and preserved sentinel rather than trusting Harbor's
job-level exit status.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
from collections.abc import Iterable
from pathlib import Path

_SENSITIVE_KEY = re.compile(r"(access|refresh|id)[_-]?token|api[_-]?key|secret|^key$", re.I)
_MAX_SCAN_BYTES = 32 * 1024 * 1024
_EXPECTED_SENTINEL = b"AUTH_SMOKE_OK\n"


def _sensitive_values(value: object, parent_key: str = "") -> Iterable[bytes]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _sensitive_values(child, str(key))
    elif isinstance(value, list):
        for child in value:
            yield from _sensitive_values(child, parent_key)
    elif _SENSITIVE_KEY.search(parent_key) and isinstance(value, str) and len(value) >= 8:
        yield value.encode()


def patterns_from_file(path: Path) -> set[bytes]:
    raw = path.read_bytes()
    patterns = {raw, base64.b64encode(raw)}
    try:
        patterns.update(_sensitive_values(json.loads(raw)))
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    return {pattern for pattern in patterns if len(pattern) >= 8}


def patterns_from_env(names: Iterable[str]) -> set[bytes]:
    patterns = set()
    for name in names:
        value = os.environ.get(name)
        if value and len(value) >= 8:
            patterns.add(value.encode())
    return patterns


def scan_tree(root: Path, patterns: set[bytes]) -> list[Path]:
    leaked: list[Path] = []
    for path in sorted(
        item for item in root.rglob("*") if item.is_file() and not item.is_symlink()
    ):
        if path.stat().st_size > _MAX_SCAN_BYTES:
            continue
        data = path.read_bytes()
        if any(pattern in data for pattern in patterns):
            leaked.append(path)
    return leaked


def validate_job(root: Path) -> None:
    result_path = root / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    stats = result.get("stats") or {}
    if (
        result.get("n_total_trials") != 1
        or stats.get("n_trials") != 1
        or stats.get("n_errors") != 0
    ):
        raise ValueError("expected exactly one error-free Harbor trial")

    trial_dirs = [
        path for path in root.iterdir() if path.is_dir() and (path / "result.json").is_file()
    ]
    if len(trial_dirs) != 1:
        raise ValueError(f"expected one trial directory, found {len(trial_dirs)}")
    trial = trial_dirs[0]
    trial_result = json.loads((trial / "result.json").read_text(encoding="utf-8"))
    rewards = (trial_result.get("verifier_result") or {}).get("rewards") or {}
    if rewards.get("submission_valid") != 1.0:
        raise ValueError("auth smoke verifier reward was not 1.0")
    if trial_result.get("exception_info") is not None:
        raise ValueError("auth smoke trial recorded an exception")

    sentinel = trial / "verifier/artifacts/auth-smoke/result.txt"
    if not sentinel.is_file() or sentinel.read_bytes() != _EXPECTED_SENTINEL:
        raise ValueError("preserved auth smoke artifact is missing or incorrect")


def reported_models(root: Path, stream_name: str) -> list[str]:
    """Return the distinct model strings a CLI stream attributed to assistant turns.

    An Anthropic-compatible third-party endpoint may echo a slug that differs
    from the requested one. A campaign must pin the observed value rather than
    relaxing the controller's model-identity check.
    """

    models: set[str] = set()
    for stream in sorted(root.rglob(stream_name)):
        for line in stream.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or record.get("type") != "assistant":
                continue
            message = record.get("message")
            if isinstance(message, dict) and message.get("model"):
                models.add(str(message["model"]))
    return sorted(models)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--secret-file", action="append", type=Path, default=[])
    parser.add_argument("--secret-env", action="append", default=[])
    parser.add_argument(
        "--report-models",
        metavar="STREAM_FILENAME",
        help="print the distinct assistant model strings found in this CLI stream",
    )
    args = parser.parse_args()

    patterns = patterns_from_env(args.secret_env)
    for path in args.secret_file:
        patterns.update(patterns_from_file(path))
    if not patterns:
        raise SystemExit("no credential patterns were supplied to the scanner")

    leaked = scan_tree(args.job_root, patterns)
    if leaked:
        relative = [str(path.relative_to(args.job_root)) for path in leaked]
        raise SystemExit(f"credential material persisted in job files: {relative}")
    print(f"credential leak scan passed ({len(patterns)} in-memory patterns)")
    try:
        validate_job(args.job_root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"auth smoke job validation failed: {exc}") from exc
    print("auth smoke reward and preserved artifact passed")
    if args.report_models:
        observed = reported_models(args.job_root, args.report_models)
        if not observed:
            raise SystemExit(f"no assistant model records found in {args.report_models}")
        print(f"reported_model observed: {', '.join(observed)}")


if __name__ == "__main__":
    main()
