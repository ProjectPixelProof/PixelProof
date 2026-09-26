"""Tests for the exact checked-in evidence bundle."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "world"))

from verify import verify_dataset  # noqa: E402


def test_exact_submitted_evidence_is_recertified() -> None:
    report = verify_dataset(ROOT / "evidence")
    assert report["status"] == "passed", report
