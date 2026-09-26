from __future__ import annotations

import base64
import json

import pytest

from scripts.check_auth_smoke import patterns_from_file, scan_tree, validate_job


def test_scanner_detects_raw_and_base64_credentials(tmp_path):
    credential = {
        "auth_mode": "oauth",
        "tokens": {
            "access_token": "access-token-value-123",
            "refresh_token": "refresh-token-value-456",
        },
    }
    auth = tmp_path / "auth.json"
    raw = (json.dumps(credential) + "\n").encode()
    auth.write_bytes(raw)
    patterns = patterns_from_file(auth)

    clean = tmp_path / "clean-job"
    clean.mkdir()
    (clean / "result.txt").write_text("AUTH_SMOKE_OK\n")
    assert scan_tree(clean, patterns) == []

    raw_job = tmp_path / "raw-job"
    raw_job.mkdir()
    (raw_job / "config.json").write_bytes(raw)
    assert scan_tree(raw_job, patterns) == [raw_job / "config.json"]

    encoded_job = tmp_path / "encoded-job"
    encoded_job.mkdir()
    (encoded_job / "command.json").write_bytes(base64.b64encode(raw))
    assert scan_tree(encoded_job, patterns) == [encoded_job / "command.json"]


def test_scanner_ignores_symlink_to_nonpersistent_secret(tmp_path):
    secret = tmp_path / "secret"
    secret.write_text("sensitive-token-value")
    job = tmp_path / "job"
    job.mkdir()
    (job / "auth.json").symlink_to(secret)

    assert scan_tree(job, {b"sensitive-token-value"}) == []


def _write_job(root, *, reward=1.0):
    root.mkdir()
    (root / "result.json").write_text(
        json.dumps(
            {
                "n_total_trials": 1,
                "stats": {"n_trials": 1, "n_errors": 0},
            }
        )
    )
    trial = root / "trial"
    artifact = trial / "verifier/artifacts/auth-smoke"
    artifact.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps(
            {
                "verifier_result": {"rewards": {"submission_valid": reward}},
                "exception_info": None,
            }
        )
    )
    (artifact / "result.txt").write_text("AUTH_SMOKE_OK\n")


def test_validate_job_accepts_one_passing_trial(tmp_path):
    job = tmp_path / "job"
    _write_job(job)
    validate_job(job)


def test_validate_job_rejects_zero_reward(tmp_path):
    job = tmp_path / "job"
    _write_job(job, reward=0.0)
    with pytest.raises(ValueError, match="reward"):
        validate_job(job)
