"""Tests for ``stitch_face_alternation``.

These run against the exact checked-in ``evidence/`` dataset plus deterministic
stress ranges, not only a convenient seed.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "world"
EVIDENCE = ROOT / "evidence"
sys.path.insert(0, str(WORLD))

import oracle  # noqa: E402
import prompts  # noqa: E402
import renderer  # noqa: E402
import verify as verify_mod  # noqa: E402

STRESS_SEEDS = (101, 2027, 9929)


def _rows() -> list[dict]:
    text = (EVIDENCE / "manifest.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# the exact submitted evidence
# --------------------------------------------------------------------------- #
def test_submitted_evidence_verifies():
    assert verify_mod.verify(EVIDENCE) == []


def test_submitted_evidence_shape():
    rows = _rows()
    assert rows
    scenes = {r["scene_id"] for r in rows}
    assert len(rows) == len(scenes) * len(prompts.PROMPT_FAMILIES)
    for r in rows:
        assert (EVIDENCE / r["image_path"]).exists()
        assert r["answer"] in r["candidates"]


def test_submitted_evidence_oracle_and_render_match():
    for r in _rows():
        scene = r["scene"]
        image = renderer.render(scene)
        assert _png(image) == (EVIDENCE / r["image_path"]).read_bytes()
        got = oracle.decision_from_image(image)
        assert got == r["decision"] or (r["quarantined"] and got == oracle.ABSTAIN)


def test_submitted_evidence_balance_and_variants():
    rows = _rows()
    variants = {r["scene"]["variant"] for r in rows}
    assert {"control", "counterfactual"} <= variants
    for family in prompts.PROMPT_FAMILIES:
        answers = {r["answer"] for r in rows if r["prompt_family"] == family
                   and not r["quarantined"]}
        assert len(answers) >= 2, family
    # the canonical prior is wrong on a real share of scenes
    conflicts = {r["scene_id"] for r in rows if r["prior_conflict"]}
    assert 0 < len(conflicts) < len({r["scene_id"] for r in rows})


def test_prior_is_never_gold():
    """Scenes whose visible truth contradicts the running-stitch prior exist and
    are labelled by the visible truth, not by the prior."""
    for r in _rows():
        if r["prompt_family"] != "pf1_alternates":
            continue
        prior = r["scene"]["canonical_prior_answer"]
        assert prior == "alternating"
        if r["decision"].startswith("broken"):
            assert r["answer"] == "no"


# --------------------------------------------------------------------------- #
# deterministic stress sweeps
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", STRESS_SEEDS)
def test_stress_oracle_agreement(seed):
    for i in range(24):
        scene = renderer.sample_scene(seed * 100003 + i * 7919 + 13)
        gold = renderer.analytic_gold(scene)
        image = renderer.render(scene)
        got = oracle.decision_from_image(image)
        assert got == gold or (renderer.is_quarantined(scene) and got == oracle.ABSTAIN)


@pytest.mark.parametrize("seed", STRESS_SEEDS)
def test_determinism(seed):
    for i in range(8):
        s = seed * 100003 + i * 7919 + 13
        a, b = renderer.sample_scene(s), renderer.sample_scene(s)
        assert a == b
        assert _png(renderer.render(a)) == _png(renderer.render(b))


def test_margin_and_quarantine_consistency():
    for i in range(60):
        scene = renderer.sample_scene(101 * 100003 + i * 7919 + 13)
        m = renderer.margin(scene)
        assert renderer.is_quarantined(scene) == (m < renderer.QUARANTINE_MARGIN)
        assert not renderer.is_quarantined(scene)  # the sampler never emits them


def test_latent_alias_pixel_identical_and_gold_preserving():
    for i in range(24):
        scene = renderer.sample_scene(2027 * 100003 + i * 7919 + 13)
        base = _png(renderer.render(scene))
        aliases = renderer.latent_symmetries(scene)
        assert aliases
        for name, alias in aliases:
            assert alias != scene, name
            assert _png(renderer.render(alias)) == base, name
            assert renderer.analytic_gold(alias) == renderer.analytic_gold(scene), name


# --------------------------------------------------------------------------- #
# oracle independence, boundary, corruption
# --------------------------------------------------------------------------- #
def test_oracle_imports_nothing_from_the_renderer_side():
    src = (WORLD / "oracle.py").read_text(encoding="utf-8")
    for bad in ("renderer", "prompts", "generate", "verify", "analytic_gold"):
        assert f"import {bad}" not in src
    import inspect
    sig = inspect.signature(oracle.decision_from_image)
    assert list(sig.parameters) == ["image"]


def test_boundary_is_observable_flipping_one_crossing_flips_the_verdict():
    """Flipping a single crossing state is a visible edit that moves gold."""
    flipped = 0
    for i in range(24):
        scene = renderer.sample_scene(9929 * 100003 + i * 7919 + 13)
        gold = renderer.analytic_gold(scene)
        for k in range(len(scene["faces"])):
            alt = dict(scene)
            faces = list(scene["faces"])
            faces[k] = "behind" if faces[k] == "front" else "front"
            alt["faces"] = faces
            new_gold = renderer.analytic_gold(alt)
            if new_gold == gold:
                continue
            flipped += 1
            got = oracle.decision_from_image(renderer.render(alt))
            assert got == new_gold
    assert flipped > 0


def test_erasing_the_tape_strip_prevents_the_answer():
    """Evidence erasure: covering the stitched strip must stop the pixel arm."""
    for i in range(6):
        scene = renderer.sample_scene(101 * 100003 + i * 7919 + 13)
        image = renderer.render(scene)
        assert oracle.decision_from_image(image) == renderer.analytic_gold(scene)
        arr = np.asarray(image).copy()
        tx0, ty0, tx1, ty1 = (int(round(float(v))) for v in scene["tape"])
        amp = int(round(float(scene["amp"]))) + 6
        arr[max(0, ty0 - amp):ty1 + amp, max(0, tx0 - 8):tx1 + 8] = 255
        assert oracle.decision_from_image(Image.fromarray(arr)) == oracle.ABSTAIN


def test_erasing_the_tag_prevents_the_answer():
    for i in range(6):
        scene = renderer.sample_scene(2027 * 100003 + i * 7919 + 13)
        arr = np.asarray(renderer.render(scene)).copy()
        tag = np.abs(arr.astype(np.int16) - np.array(renderer.TAG_RGB)).max(axis=2) <= 40
        arr[tag] = 255
        assert oracle.decision_from_image(Image.fromarray(arr)) == oracle.ABSTAIN


def test_remote_scraps_are_nondiagnostic():
    """Deleting every remote scrap leaves the decision and the oracle unchanged."""
    for i in range(8):
        scene = renderer.sample_scene(9929 * 100003 + i * 7919 + 13)
        bare = dict(scene)
        bare["distractors"] = []
        assert renderer.analytic_gold(bare) == renderer.analytic_gold(scene)
        assert oracle.decision_from_image(renderer.render(bare)) == renderer.analytic_gold(scene)


def test_corrupted_raster_and_altered_labels_fail_verification(tmp_path):
    dataset = tmp_path / "ds"
    subprocess.run(
        [sys.executable, str(WORLD / "generate.py"), "--out", str(dataset),
         "--n", "24", "--seed", "101"],
        check=True, capture_output=True,
    )
    assert verify_mod.verify(dataset) == []

    # altered label
    manifest = dataset / "manifest.jsonl"
    rows = [json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
    rows[0]["answer"] = "yes" if rows[0]["answer"] != "yes" else "no"
    manifest.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
    assert verify_mod.verify(dataset) != []

    # materially corrupted raster
    rows[0]["answer"] = prompts.correct_answer(rows[0]["prompt_family"], rows[0]["decision"])
    manifest.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
    assert verify_mod.verify(dataset) == []
    victim = dataset / rows[0]["image_path"]
    arr = np.asarray(Image.open(victim).convert("RGB")).copy()
    arr[:, :] = np.uint8(255)
    Image.fromarray(arr).save(victim, format="PNG", optimize=False)
    assert verify_mod.verify(dataset) != []


def test_prompt_api_is_total_and_constrained():
    for family in prompts.PROMPT_FAMILIES:
        cands = prompts.candidates_for(family)
        assert len(cands) >= 2 and len(set(cands)) == len(cands)
        for pattern in ("alternating", "broken"):
            for face in ("front", "behind"):
                assert prompts.correct_answer(family, f"{pattern}|{face}") in cands
    with pytest.raises(KeyError):
        prompts.candidates_for("nope")
