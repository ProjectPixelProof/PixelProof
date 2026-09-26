"""Shared fixtures: run the real generate CLI (images + manifest, oracle on)."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def generate_dataset(dest: Path, *, n: int = 6, seed: int = 1, extra_args: tuple = ()) -> Path:
    """Run the two-circle generator into ``dest`` and return its manifest path."""
    manifest = dest / "manifest.jsonl"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "worlds.two_circles.generate",
            "--n",
            str(n),
            "--seed",
            str(seed),
            "--split",
            "pytest_tmp",
            "--out-dir",
            str(dest / "img"),
            "--manifest",
            str(manifest),
            *extra_args,
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    return manifest


@pytest.fixture(scope="session")
def generated_manifest(tmp_path_factory) -> Path:
    """One oracle-certified 6-scene dataset shared by CLI-level tests."""
    return generate_dataset(tmp_path_factory.mktemp("gen"))
