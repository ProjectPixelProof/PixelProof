"""Candidate tests for the end-fold pairing reveal world.

They cover the exact submitted evidence, the public stress seeds from
TASK_SPEC.toml, oracle independence, label-preserving permutations, cheap-cue
counterbalancing, the ablation-aware support geometry, erasure behaviour, and
corruption rejection.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "world"))

import generate  # noqa: E402
import oracle  # noqa: E402
import prompts  # noqa: E402
import renderer  # noqa: E402
import verify  # noqa: E402

EVIDENCE = ROOT / "evidence"
STRESS_SEEDS = (101, 2027, 9929)
STRESS_SCENES = 24


def _rows() -> list[dict]:
    lines = (EVIDENCE / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _stress_scenes():
    for base in STRESS_SEEDS:
        for index in range(STRESS_SCENES):
            seed = base + index
            yield seed, renderer.sample_scene(seed)


def _white_out(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    array = np.array(image.convert("RGB"))
    x0, y0, x1, y1 = box
    array[y0:y1, x0:x1] = 255
    return Image.fromarray(array)


# --- submitted evidence -----------------------------------------------------


def test_submitted_evidence_verifies_exactly():
    assert verify.verify(EVIDENCE) == []


def test_evidence_rows_cover_every_family_once_per_scene():
    rows = _rows()
    families = sorted(prompts.PROMPT_FAMILIES)
    seen: dict[str, list[str]] = {}
    for row in rows:
        seen.setdefault(row["scene_id"], []).append(row["prompt_family"])
    assert seen, "evidence manifest is empty"
    for scene_id, found in seen.items():
        assert sorted(found) == families, scene_id


def test_evidence_manifest_has_required_keys():
    required = {
        "example_id",
        "scene_id",
        "seed",
        "scene",
        "image_path",
        "prompt_family",
        "question",
        "decision",
        "answer",
        "candidates",
        "margin",
        "quarantined",
    }
    for row in _rows():
        assert required <= set(row), row.get("example_id")
        assert (EVIDENCE / row["image_path"]).is_file()


def test_evidence_gallery_covers_classes_and_near_boundary():
    rows = _rows()
    decisions = {row["decision"] for row in rows}
    assert decisions == set(renderer.GLYPHS)
    assert any(row["quarantined"] for row in rows), "no quarantined scene in the gallery"
    near = [
        row
        for row in rows
        if row["margin"] < renderer.QUARANTINE_CLEARANCE + 6.0
    ]
    assert near, "no near-boundary scene in the gallery"
    assert len(list((EVIDENCE / "gallery").glob("*.png"))) >= 12


def test_class_balance_per_family_off_quarantine():
    rows = [row for row in _rows() if not row["quarantined"]]
    for family in prompts.PROMPT_FAMILIES:
        answers = {row["answer"] for row in rows if row["prompt_family"] == family}
        assert len(answers) >= 2, family


# --- stress agreement, determinism, purity ----------------------------------


def test_analytic_and_pixel_agree_on_stress_seeds():
    for seed, scene in _stress_scenes():
        image = renderer.render(scene)
        assert oracle.decision_from_image(image) == renderer.analytic_gold(scene), seed


def test_stress_seeds_cover_every_glyph():
    found = {renderer.analytic_gold(scene) for _, scene in _stress_scenes()}
    assert found == set(renderer.GLYPHS)


def test_generation_is_deterministic(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    for out in (first, second):
        out.mkdir()
        generate.build(out, 3, 2027)
    assert (first / "manifest.jsonl").read_bytes() == (second / "manifest.jsonl").read_bytes()
    for path in sorted((first / "gallery").glob("*.png")):
        assert path.read_bytes() == (second / "gallery" / path.name).read_bytes()


def test_scene_sampling_and_rendering_are_pure():
    scene = renderer.sample_scene(9929)
    assert renderer.sample_scene(9929) == scene
    before = copy.deepcopy(scene)
    left = renderer.render(scene)
    assert scene == before
    right = renderer.render(scene)
    assert ImageChops.difference(left.convert("RGB"), right.convert("RGB")).getbbox() is None


# --- oracle independence ----------------------------------------------------


def test_oracle_source_is_independent():
    source = (ROOT / "world" / "oracle.py").read_text(encoding="utf-8")
    for forbidden in ("import renderer", "import prompts", "import generate", "import verify"):
        assert forbidden not in source
    for forbidden in ("analytic_gold", "sample_scene", "GLYPH_MASKS", "scene"):
        assert forbidden not in source.replace("latent scene data", "")


def test_oracle_signature_takes_only_an_image():
    import inspect

    signature = inspect.signature(oracle.decision_from_image)
    assert list(signature.parameters) == ["image"]


# --- glyph alphabet and cheap-cue counterbalancing --------------------------


def test_every_glyph_marks_nine_positions():
    for name in renderer.GLYPHS:
        assert sum(renderer.GLYPH_MASKS[name]) == 9


def test_glyphs_are_pairwise_far_apart():
    names = list(renderer.GLYPHS)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            distance = sum(
                1
                for a, b in zip(renderer.GLYPH_MASKS[left], renderer.GLYPH_MASKS[right])
                if a != b
            )
            assert distance >= 8, (left, right, distance)


def test_local_cues_are_constant_across_answers():
    """Bead count, ink area, cord length and unmarked-pair count carry no answer."""
    profiles: dict[str, set[tuple[int, int, int]]] = {}
    for _, scene in _stress_scenes():
        gold = renderer.analytic_gold(scene)
        unmarked = 25 - sum(renderer.GLYPH_MASKS[gold])
        profiles.setdefault(gold, set()).add(
            (len(scene["route"]), len(scene["tones"]), unmarked)
        )
    assert set(profiles) == set(renderer.GLYPHS)
    shared = set.union(*profiles.values())
    assert shared == {(renderer.BEADS, renderer.BEADS, 16)}


def test_dark_bead_count_does_not_separate_the_answers():
    """Every glyph is rendered with overlapping dark-bead totals."""
    totals: dict[str, set[int]] = {}
    for _, scene in _stress_scenes():
        gold = renderer.analytic_gold(scene)
        totals.setdefault(gold, set()).add(sum(scene["tones"]))
    for left in renderer.GLYPHS:
        for right in renderer.GLYPHS:
            if left != right:
                assert totals[left] & totals[right], (left, right)


def test_cheaper_pairings_never_spell_the_answer():
    """Half-on-half, neighbour and raw-half readings are rejected at sampling."""
    masks = set(renderer.GLYPH_MASKS.values())
    for seed, scene in _stress_scenes():
        tones = tuple(scene["tones"])
        for alternative in renderer._alternative_masks(tones):
            assert alternative not in masks, seed


# --- ablation-aware support geometry ---------------------------------------


def test_support_is_one_thin_connected_trace_spanning_the_canvas():
    for seed, scene in _stress_scenes():
        array = np.array(renderer.render(scene).convert("RGB"), dtype=np.int16)
        cord = np.abs(array - np.array(renderer.CORD, dtype=np.int16)).max(axis=2) <= 40
        gray = (array.max(axis=2) - array.min(axis=2) <= 12) & (
            array.mean(axis=2) >= 30
        ) & (array.mean(axis=2) <= 244)
        support = cord | gray
        if seed in (101, 2027, 9929):
            # The cord and its beads form one connected trace; the small tone key
            # is the only other ink and is never part of the decisive support.
            blobs = [c for c in oracle._components(support) if len(c) > 5000]
            assert len(blobs) == 1, (seed, [len(c) for c in blobs])
            trace = {(int(y), int(x)) for y, x in blobs[0]}
            assert all((int(y), int(x)) in trace for y, x in zip(*np.nonzero(cord)))
        ys, xs = np.nonzero(cord)
        height = ys.max() - ys.min()
        width = xs.max() - xs.min()
        canvas = int(scene["canvas"])
        assert height >= 0.7 * canvas and width >= 0.7 * canvas, (seed, height, width)
        assert ys.min() > 20 and xs.min() > 20
        assert ys.max() < canvas - 20 and xs.max() < canvas - 20
        area = float(support[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].sum())
        assert area / float((height + 1) * (width + 1)) < 0.2, seed


def test_either_end_of_the_trace_is_necessary():
    """Widely separated parts of one image are jointly necessary."""
    for seed in (101, 2027, 9929):
        scene = renderer.sample_scene(seed)
        image = renderer.render(scene)
        canvas = int(scene["canvas"])
        half = canvas // 2
        for box in (
            (0, 0, half, canvas),
            (half, 0, canvas, canvas),
            (0, 0, canvas, half),
            (0, half, canvas, canvas),
        ):
            assert oracle.decision_from_image(_white_out(image, box)) == oracle.ABSTAIN


def test_interior_erasure_forces_abstention():
    for seed in (101, 2027, 9929):
        scene = renderer.sample_scene(seed)
        image = renderer.render(scene)
        points = [
            (
                int(scene["origin"]) + node[0] * int(scene["pitch"]),
                int(scene["origin"]) + node[1] * int(scene["pitch"]),
            )
            for node in scene["route"]
        ]
        for index in (12, 24, 37):
            x, y = points[index]
            box = (x - 22, y - 22, x + 22, y + 22)
            assert oracle.decision_from_image(_white_out(image, box)) == oracle.ABSTAIN


def test_offtrace_erasure_does_not_change_the_answer():
    """Erasing the key does not touch the decisive support."""
    for seed in (101, 2027, 9929):
        scene = renderer.sample_scene(seed)
        image = renderer.render(scene)
        x0, y0, x1, y1 = renderer.key_box(int(scene["key_corner"]))
        box = (x0 - 4, y0 - 4, x1 + 4, y1 + 4)
        stripped = _white_out(image, box)
        assert oracle.decision_from_image(stripped) == renderer.analytic_gold(scene)


# --- label-preserving and label-changing perturbations ---------------------


def test_translation_preserves_the_answer():
    for seed in (101, 2027, 9929):
        scene = renderer.sample_scene(seed)
        gold = renderer.analytic_gold(scene)
        moved = copy.deepcopy(scene)
        moved["origin"] = int(scene["origin"]) + 14
        moved["decoys"] = []
        assert renderer.analytic_gold(moved) == gold
        assert oracle.decision_from_image(renderer.render(moved)) == gold


def test_distractor_permutation_preserves_the_answer():
    for seed in (101, 2027, 9929):
        scene = renderer.sample_scene(seed)
        gold = renderer.analytic_gold(scene)
        for variant in ([], list(reversed(scene["decoys"]))):
            altered = copy.deepcopy(scene)
            altered["decoys"] = variant
            assert renderer.analytic_gold(altered) == gold
            assert oracle.decision_from_image(renderer.render(altered)) == gold


def test_global_tone_swap_and_key_move_preserve_the_answer():
    """Agreement between folded partners is invariant to swapping the two grays."""
    for seed in (101, 2027, 9929):
        scene = renderer.sample_scene(seed)
        gold = renderer.analytic_gold(scene)
        swapped = copy.deepcopy(scene)
        swapped["dark"], swapped["light"] = scene["light"], scene["dark"]
        free = [c for c in renderer.free_key_corners(scene) if c != scene["key_corner"]]
        if free:
            swapped["key_corner"] = free[0]
        assert renderer.analytic_gold(swapped) == gold
        assert renderer.margin(swapped) == renderer.margin(scene)
        rendered = renderer.render(swapped)
        assert (
            ImageChops.difference(rendered.convert("RGB"), renderer.render(scene).convert("RGB"))
            .getbbox()
            is not None
        )
        assert oracle.decision_from_image(rendered) == gold


def test_reversing_the_walk_direction_is_label_preserving_here():
    """End-to-end folding is symmetric, and the brackets still mark one end."""
    for seed in (101, 2027, 9929):
        scene = renderer.sample_scene(seed)
        flipped = copy.deepcopy(scene)
        flipped["route"] = list(reversed(scene["route"]))
        flipped["tones"] = list(reversed(scene["tones"]))
        assert renderer.analytic_gold(flipped) == renderer.analytic_gold(scene)
        assert oracle.decision_from_image(renderer.render(flipped)) == renderer.analytic_gold(
            scene
        )


def test_flipping_one_bead_tone_changes_or_abstains():
    for seed in (101, 2027, 9929):
        scene = renderer.sample_scene(seed)
        for index in (0, 7, 31, 49):
            altered = copy.deepcopy(scene)
            altered["tones"][index] = 1 - int(altered["tones"][index])
            decided = oracle.decision_from_image(renderer.render(altered))
            assert decided != renderer.analytic_gold(scene)


def test_removing_the_start_bracket_forces_abstention():
    for seed in (101, 2027, 9929):
        scene = renderer.sample_scene(seed)
        image = renderer.render(scene)
        array = np.array(image.convert("RGB"))
        black = (array.max(axis=2) < 40)
        array[black] = 255
        assert oracle.decision_from_image(Image.fromarray(array)) == oracle.ABSTAIN


def test_corrupted_raster_fails():
    scene = renderer.sample_scene(101)
    array = np.array(renderer.render(scene).convert("RGB"))
    array[:, : array.shape[1] // 3] = 255
    assert oracle.decision_from_image(Image.fromarray(array)) != renderer.analytic_gold(scene)


def test_altered_stored_label_is_rejected(tmp_path):
    generate.build(tmp_path, 2, 101)
    manifest = tmp_path / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    other = [g for g in renderer.GLYPHS if g != rows[0]["decision"]][0]
    rows[0]["decision"] = other
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    assert verify.verify(tmp_path) != []


# --- declarations -----------------------------------------------------------


def test_declared_latent_alias_mode_is_not_applicable():
    candidate = json.loads((ROOT / "candidate.json").read_text(encoding="utf-8"))
    assert candidate["latent_alias"]["mode"] == "not_applicable"
    for _, scene in _stress_scenes():
        assert renderer.latent_symmetries(scene) == []
        break


def test_every_scene_field_is_rendered():
    scene = renderer.sample_scene(2027)
    baseline = renderer.render(scene).convert("RGB")
    mutations = {
        "canvas": lambda s: s.update(canvas=int(s["canvas"]) + 20),
        "pitch": lambda s: s.update(pitch=int(s["pitch"]) - 3),
        "origin": lambda s: s.update(origin=int(s["origin"]) + 7),
        "cord_w": lambda s: s.update(cord_w=int(s["cord_w"]) + 2),
        "bead_r": lambda s: s.update(bead_r=int(s["bead_r"]) - 1),
        "route": lambda s: s.update(route=list(reversed(s["route"]))),
        "tones": lambda s: s.update(tones=[1 - int(b) for b in s["tones"]]),
        "dark": lambda s: s.update(dark=int(s["dark"]) + 3),
        "light": lambda s: s.update(light=int(s["light"]) - 3),
        "key_corner": lambda s: s.update(key_corner=(int(s["key_corner"]) + 1) % 4),
        "decoys": lambda s: s.update(decoys=[]),
    }
    assert set(mutations) == set(scene), set(scene) ^ set(mutations)
    for field, mutate in mutations.items():
        altered = copy.deepcopy(scene)
        mutate(altered)
        rendered = renderer.render(altered).convert("RGB")
        if rendered.size != baseline.size:
            continue
        assert ImageChops.difference(rendered, baseline).getbbox() is not None, field


def test_quarantine_rule_is_explicit():
    for _, scene in _stress_scenes():
        assert renderer.is_quarantined(scene) == (
            renderer.margin(scene) < renderer.QUARANTINE_CLEARANCE
        )


def test_prompt_api_is_total_over_glyphs():
    for family in prompts.PROMPT_FAMILIES:
        answers = set()
        for glyph in renderer.GLYPHS:
            answer = prompts.correct_answer(family, glyph)
            assert answer in prompts.candidates_for(family)
            answers.add(answer)
        assert len(answers) >= 2
    with pytest.raises(KeyError):
        prompts.candidates_for("pf_missing")
    with pytest.raises(ValueError):
        prompts.correct_answer("pf1_symbol_choice", "not_a_glyph")
