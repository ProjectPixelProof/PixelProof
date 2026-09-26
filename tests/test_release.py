"""Release integration tests: complete input packets and immutable bootstrap checks."""

import json

import pytest

from question_foundry.sequential import load_sequential_campaign
from question_foundry.vlm_eval.runner import full_offline_smoke
from scripts.release import PROFILES, ROOT, prepare, preview
from scripts.run_sequential_campaign import _load_feedback_bootstrap


@pytest.mark.parametrize("agent", ["codex", "claude", "opencode"])
@pytest.mark.parametrize("rq", [1, 2, 3])
def test_three_sections_and_adapters(tmp_path, agent, rq):
    campaign = prepare(
        rq=rq, agent=agent, campaign_id=f"test_rq{rq}_{agent}", out=tmp_path / "campaign.toml"
    )
    packet = preview(campaign, tmp_path / "preview")
    starter = packet.task / "environment/starter"
    assert (packet.task / "tests/candidate_gatekeeper.py").is_file()
    assert not (starter / "tests").exists()
    assert not (starter / "oracle_solution").exists()
    for world in ["two_circles", "angle_acuteness", "counting_with_distractors"]:
        assert {p.name for p in (starter / "seeds" / world).iterdir()} == {
            "DESIGN.md",
            "world.toml",
            "renderer.py",
            "prompts.py",
            "oracle.py",
        }
    assert (starter / "DISCOVERY_PROFILE.md").is_file()
    assert (starter / "DIVERSITY_TARGET.json").exists() == (rq == 2)
    if rq == 3:
        assert len(list((starter / "campaign/working_seed").iterdir())) == 10
        feedback = json.loads((starter / "DIFFICULTY_FEEDBACK.json").read_text())
        assert len(feedback["records"]) == 10
        assert sum(len(x["sample_responses"]) for x in feedback["records"]) == 20
        assert all("oracle_answer" in s for x in feedback["records"] for s in x["sample_responses"])


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("rq", [1, 2])
def test_every_profile_card(tmp_path, profile, rq):
    campaign = prepare(
        rq=rq,
        agent="codex",
        campaign_id="test_profile",
        profile=profile,
        out=tmp_path / "campaign.toml",
    )
    packet = preview(campaign, tmp_path / "preview")
    original = ROOT / f"foundry/discovery_profiles/paper-semantic/v0.3/{profile}.md"
    assert (
        packet.task / "environment/starter/DISCOVERY_PROFILE.md"
    ).read_bytes() == original.read_bytes()
    assert original.read_text().count("- **`") == 5


@pytest.mark.parametrize("collection", ["a", "b"])
def test_starting_collections_are_self_contained(tmp_path, collection):
    path = prepare(
        rq=3,
        agent="codex",
        campaign_id="test_collection",
        collection=collection,
        out=tmp_path / "campaign.toml",
    )
    config = load_sequential_campaign(path)
    manifest, _ = _load_feedback_bootstrap(config)
    assert len(manifest["records"]) == 10
    for item in manifest["records"]:
        assert (ROOT / item["candidate_path"] / "world/generate.py").is_file()
    packet = preview(path, tmp_path / "preview")
    assert len(list((packet.task / "environment/starter/campaign/working_seed").iterdir())) == 10
    config["feedback_bootstrap"]["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        _load_feedback_bootstrap(config)


def test_no_overwrite(tmp_path):
    path = prepare(
        rq=1, agent="codex", campaign_id="test_no_overwrite", out=tmp_path / "campaign.toml"
    )
    with pytest.raises(FileExistsError):
        prepare(rq=1, agent="claude", campaign_id="test_no_overwrite", out=path)
    preview(path, tmp_path / "preview")
    with pytest.raises(FileExistsError):
        preview(path, tmp_path / "preview")


def test_feedback_pipeline_without_network():
    result = full_offline_smoke()
    assert result


def test_feedback_tests_use_scratch_copy_without_writing_source(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from question_foundry import difficulty_feedback

    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="sha256:" + "a" * 64, returncode=0)

    monkeypatch.setattr(difficulty_feedback.subprocess, "run", run)
    monkeypatch.setattr(difficulty_feedback, "validate_generated_dataset", lambda *a, **k: {})
    difficulty_feedback._render_candidate(
        candidate_root=tmp_path / "source",
        dataset=tmp_path / "out/dataset",
        render_runtime="latex-tikz@0.1.0",
        scenes=3,
        seed=101,
    )
    test_command = commands[1]
    assert str((tmp_path / "source").resolve()) + ":/candidate:ro" in test_command
    assert "--read-only" in test_command
    assert test_command[test_command.index("--network") + 1] == "none"
    assert "shutil.copytree('/candidate','/tmp/test-candidate')" in test_command[-1]
    # Test mutations cannot affect subsequent rendering: it reads the immutable input.
    assert "/candidate/world/generate.py" in commands[2]


def test_release_payload_excludes_runs_and_history(tmp_path):
    from scripts.audit_release import audit, payload_files

    (tmp_path / "README.md").write_text("public text\n")
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "trajectory.json").write_text("private trace")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("private remote")
    assert [p.relative_to(tmp_path).as_posix() for p in payload_files(tmp_path)] == ["README.md"]
    assert audit(tmp_path)["status"] == "passed"
    (tmp_path / "README.md").write_text("/" + "Users" + "/someone/private/path")
    assert audit(tmp_path)["status"] == "failed"


def test_release_audit_rejects_symlink_directory(tmp_path):
    from scripts.audit_release import audit

    external = tmp_path / "private"
    external.mkdir()
    (external / "secret.txt").write_text("not public")
    (tmp_path / "src").symlink_to(external, target_is_directory=True)
    assert audit(tmp_path)["status"] == "failed"


def test_release_audit_rejects_unallowlisted_tracked_file(tmp_path):
    import subprocess

    from scripts.audit_release import audit

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "private.txt").write_text("private")
    subprocess.run(["git", "add", "private.txt"], cwd=tmp_path, check=True)
    assert audit(tmp_path)["status"] == "failed"


def test_canary_requires_complete_success():
    from scripts.docker_smoke import validate_canary

    checks = dict.fromkeys(
        ("clean_recycle_marker", "hidden_tests_present_now", "network_blocked", "oauth_absent"),
        True,
    )
    rewards = dict.fromkeys(
        (
            "all_mechanically_eligible",
            "mechanically_eligible_fraction",
            "registry_distinct_fraction",
            "semantic_review_ready_fraction",
            "submission_valid",
        ),
        1.0,
    )
    validate_canary({"checks": checks}, rewards)
    for key in checks:
        with pytest.raises(ValueError):
            validate_canary({"checks": {k: v for k, v in checks.items() if k != key}}, rewards)
    for key in rewards:
        with pytest.raises(ValueError):
            validate_canary({"checks": checks}, {**rewards, key: 0.0})


def test_run_resolves_paths_before_changing_directory(tmp_path, monkeypatch):
    import sys

    from scripts import release

    subdir = tmp_path / "subdir"
    subdir.mkdir()
    monkeypatch.chdir(subdir)
    monkeypatch.setattr(release, "ROOT", tmp_path)
    monkeypatch.setattr(release, "load_sequential_campaign", lambda p: {"id": "test"})
    monkeypatch.setattr(release.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(release.subprocess, "check_output", lambda *a, **k: b"")
    captured = {}
    monkeypatch.setattr(release, "run_campaign", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        sys,
        "argv",
        ["release", "run", "--campaign", "campaign.toml", "--confirm", "test", "--out", "results"],
    )
    release.main()
    assert captured == {"campaign_path": subdir / "campaign.toml", "run_root": subdir / "results"}


def test_release_audit_allows_only_reviewed_documentation_image(tmp_path):
    from scripts.audit_release import audit

    destination = tmp_path / "docs/assets/figure-1.png"
    destination.parent.mkdir(parents=True)
    destination.write_bytes((ROOT / "docs/assets/figure-1.png").read_bytes())
    assert audit(tmp_path)["status"] == "passed"
    destination.write_bytes(destination.read_bytes() + b"unreviewed metadata")
    assert audit(tmp_path)["status"] == "failed"
    destination.unlink()
    (destination.parent / "unreviewed.png").write_bytes(b"unreviewed export")
    assert audit(tmp_path)["status"] == "failed"
