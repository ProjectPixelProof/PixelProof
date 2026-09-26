"""Answerability conformance gates (SSOT §7.14).

The §7.6 oracle and the §7.7 admission criteria verify that a label is
*consistent* with the geometry (gold ↔ z ↔ pixels agree). They do NOT verify
that the question posed to the model is *answerable from the pixels alone* — a
distinct property. Two foundry-4 worlds passed every consistency gate yet were
scientifically void because gold and oracle shared a hidden convention the
render never exposed (a latent endpoint order; a code-only threshold constant).
These five gates close that hole structurally; see
``docs/verification-boundary.md``.

This module is **task-agnostic core**: every gate is a pure function over a
duck-typed :class:`WorldBundle` (already-imported renderer / oracle / prompts
modules, the ``[gold]`` config, and the smoke records). It imports no specific
world — the loader that constructs bundles from disk lives in
``worlds/_common/gate_runner.py`` (an instance-layer helper allowed to import
worlds), and the defect fixtures in ``tests/fixtures/`` build bundles from
stub objects with the same interface. The same code therefore judges a real
world and a minimal reproduction of a defect, which is what makes each gate
fixture-provable (a gate that never fails anything is a rubber stamp).

Gate ↔ hook contract (see SSOT §7.14):

- **latent_symmetry** needs ``renderer.latent_symmetries(scene, rng)`` yielding
  ``(name, twin_scene)`` pairs that render pixel-identically.
- **sign_blind** needs ``oracle.decision_from_image(image)`` (takes no scene).
- **stated_boundary** needs ``gold["boundary_observable"]`` non-empty.
- **class_balance** and **gallery_audit** read the smoke records / the world dir;
  no per-world hook.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GATE_NAMES = (
    "latent_symmetry",
    "sign_blind",
    "stated_boundary",
    "class_balance",
    "gallery_audit",
)


@dataclass(frozen=True)
class GateResult:
    """Outcome of one gate on one world. ``ok`` false ⇒ ``failures`` non-empty."""

    gate: str
    ok: bool
    failures: list[str] = field(default_factory=list)
    notes: dict = field(default_factory=dict)


@dataclass
class WorldBundle:
    """Everything a gate needs about one world. Duck-typed: ``renderer`` /
    ``oracle`` / ``prompts`` may be real modules or stub objects, so long as they
    expose the attributes the relevant gate uses."""

    name: str
    renderer: Any
    oracle: Any
    prompts: Any
    gold: dict
    smoke_records: list = field(default_factory=list)
    supplemental_records: list = field(default_factory=list)
    path: Path | None = None  # the world directory (for the audit gate)
    audits_dir: Path | None = None  # docs/audits/
    scene_from_dict: Any = None  # resolved scene reconstructor (loader-provided)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _scene_from_dict(renderer, scene: dict):
    if hasattr(renderer, "scene_from_dict"):
        return renderer.scene_from_dict(scene)
    return renderer.Scene(**scene)


def _reconstruct(bundle: WorldBundle, scene: dict):
    """Rebuild a Scene from a manifest dict, preferring the bundle-provided
    reconstructor (older worlds keep ``_scene_from_dict`` in ``verify.py``)."""
    if bundle.scene_from_dict is not None:
        return bundle.scene_from_dict(scene)
    return _scene_from_dict(bundle.renderer, scene)


def _sampled_bins(renderer, quarantine: str) -> list[str]:
    declared = getattr(renderer, "SAMPLED_BINS", None)
    if declared:
        return list(declared)
    return [b for b in renderer.MARGIN_BINS if b != quarantine]


def _pixels_identical(a, b) -> bool:
    """True iff two PIL images are pixel-for-pixel identical."""
    if a.size != b.size:
        return False
    return a.convert("RGB").tobytes() == b.convert("RGB").tobytes()


def _unique_scene_records(records: list) -> list[tuple[dict, list]]:
    """Group manifest records by their (unique) rendered image / scene.

    Records that share an image share a scene (paired design, SSOT §7.4); the
    ``image_path`` is the grouping key. Preserves first-seen order."""
    groups: dict[str, list] = {}
    order: list[str] = []
    for rec in records:
        if rec.image_path not in groups:
            order.append(rec.image_path)
        groups.setdefault(rec.image_path, []).append(rec)
    return [(groups[k][0].scene, groups[k]) for k in order]


# ---------------------------------------------------------------------------
# Gate 1 — latent-symmetry invariance
# ---------------------------------------------------------------------------


def gate_latent_symmetry(bundle: WorldBundle, rng: random.Random, n_per_bin: int = 2) -> GateResult:
    """Render-invariant scene transforms must leave gold unchanged (SSOT §7.14 gate 1).

    For each declared symmetry twin the gate first proves the render is
    pixel-identical (else the hook lied about it being a symmetry), then asserts
    gold is identical for every prompt family. Pixel-identical render + differing
    gold ⇒ the image under-determines the answer ⇒ fail.
    """
    r, p = bundle.renderer, bundle.prompts
    if not hasattr(r, "latent_symmetries"):
        return GateResult(
            "latent_symmetry",
            False,
            [
                f"{bundle.name}: renderer has no latent_symmetries hook — a world must "
                "enumerate its render-invariant transforms (empty list + justification if none)"
            ],
        )
    decision_attr = bundle.gold["decision_attr"]
    quar = bundle.gold["quarantine_bin"]
    failures: list[str] = []
    n_symmetries_checked = 0
    for b in _sampled_bins(r, quar):
        for _ in range(n_per_bin):
            try:
                scene = r.sample_scene(rng, margin_bin=b)
            except (ValueError, RuntimeError):
                continue
            img = r.render(scene)
            dv = getattr(scene, decision_attr)
            for name, twin in r.latent_symmetries(scene, rng):
                n_symmetries_checked += 1
                if not _pixels_identical(img, r.render(twin)):
                    failures.append(
                        f"{bundle.name}: declared symmetry {name!r} CHANGED the render — it "
                        "is not a latent symmetry (the hook is wrong)"
                    )
                    continue
                dv_twin = getattr(twin, decision_attr)
                for fam in p.PROMPT_FAMILIES:
                    a, c = p.correct_answer(fam, dv), p.correct_answer(fam, dv_twin)
                    if a != c:
                        failures.append(
                            f"{bundle.name}: symmetry {name!r} is pixel-identical but flips "
                            f"gold for {fam} ({a!r} != {c!r}) — question unanswerable from pixels"
                        )
    return GateResult(
        "latent_symmetry",
        not failures,
        failures,
        {"symmetries_checked": n_symmetries_checked},
    )


# ---------------------------------------------------------------------------
# Gate 2 — sign-blind oracle
# ---------------------------------------------------------------------------


def gate_sign_blind(bundle: WorldBundle, max_scenes: int = 12) -> GateResult:
    """The certified decision must be recomputable from the image alone (gate 2).

    Re-renders each off-quarantine scene from the tracked manifest (rendered PNGs
    are git-ignored, so we never read them from disk), asks the oracle's
    scene-free ``decision_from_image`` for the decision, and asserts
    ``correct_answer(family, decision)`` equals the frozen gold. Off-quarantine
    only, so the calibrated no-sign-flip guarantee holds. Defect #1 (a directed
    line's side is not orientable from pixels) fails here.
    """
    r, o, p = bundle.renderer, bundle.oracle, bundle.prompts
    if not hasattr(o, "decision_from_image"):
        return GateResult(
            "sign_blind",
            False,
            [
                f"{bundle.name}: oracle has no decision_from_image hook — the certified "
                "decision must be computable from measure(image) with no scene"
            ],
        )
    if not bundle.smoke_records:
        return GateResult("sign_blind", False, [f"{bundle.name}: no smoke records to check"])
    quar = bundle.gold["quarantine_bin"]
    failures: list[str] = []
    checked = 0
    for scene_dict, group in _unique_scene_records(bundle.smoke_records):
        if group[0].targets.get("margin_bin") == quar:
            continue  # off-quarantine only
        scene = _reconstruct(bundle, scene_dict)
        image = r.render(scene)
        try:
            dv_hat = o.decision_from_image(image)
        except Exception as e:  # a measurement failure is a gate failure, reported
            failures.append(f"{bundle.name}: decision_from_image raised {e!r}")
            if len(failures) >= 8:
                break
            continue
        checked += 1
        for rec in group:
            recomputed = p.correct_answer(rec.prompt_family, dv_hat)
            if recomputed != rec.targets["answer"]:
                failures.append(
                    f"{bundle.name}/{rec.prompt_family}: image-only decision gives "
                    f"{recomputed!r} but frozen gold is {rec.targets['answer']!r} "
                    f"(scene {rec.example_id}) — decision not recoverable from pixels"
                )
        if checked >= max_scenes:
            break
        if len(failures) >= 8:
            break
    if checked == 0 and not failures:
        failures.append(f"{bundle.name}: no off-quarantine scenes found to check")
    return GateResult("sign_blind", not failures, failures, {"scenes_checked": checked})


# ---------------------------------------------------------------------------
# Gate 3 — stated boundary
# ---------------------------------------------------------------------------


def gate_stated_boundary(bundle: WorldBundle) -> GateResult:
    """Every ``[gold]`` block must declare how the decision boundary is visible
    in-image (gate 3). Presence is machine-checked; the content is judged by the
    human audit. A hidden-threshold world (defect #2) has nothing honest to write."""
    val = bundle.gold.get("boundary_observable")
    if not isinstance(val, str) or not val.strip():
        return GateResult(
            "stated_boundary",
            False,
            [
                f"{bundle.name}: [gold] must declare a non-empty boundary_observable = "
                '"<how the decision boundary is visible in-image>" (answerability gate 3)'
            ],
        )
    return GateResult("stated_boundary", True, [], {"boundary_observable": val})


# ---------------------------------------------------------------------------
# Gate 4 — off-quarantine class balance
# ---------------------------------------------------------------------------


def gate_class_balance(bundle: WorldBundle) -> GateResult:
    """Every DECLARED prompt family must be either present in the smoke manifest with
    ≥2 distinct off-quarantine gold classes, OR listed in ``[gold]
    smoke_excluded_families`` (gate 4) — the machine-checked "are they touching?"
    rule, made dodge-resistant.

    A present family whose off-quarantine gold is constant cannot separate signal
    from response bias → fail. And a declared family SILENTLY absent from smoke also
    fails: to hold a family out of the eval split (a balanceable lexical battery
    emitted elsewhere, or a boundary-only coincidence family that can never be
    balanced) the world must DECLARE it excluded, so the exclusion is disclosed and
    reviewable rather than a silent way to dodge a hard family. Every excluded
    family must occur with at least two off-quarantine classes in a supplemental
    manifest; declaring an exclusion is not itself an escape hatch."""
    if not bundle.smoke_records:
        return GateResult("class_balance", False, [f"{bundle.name}: no smoke records to check"])
    quar = bundle.gold["quarantine_bin"]
    excluded = set(bundle.gold.get("smoke_excluded_families", []))
    declared = set(getattr(bundle.prompts, "PROMPT_FAMILIES", {}))
    present: set = set()
    by_family: dict[str, set] = {}
    for rec in bundle.smoke_records:
        present.add(rec.prompt_family)
        if rec.targets.get("margin_bin") == quar:
            continue
        by_family.setdefault(rec.prompt_family, set()).add(rec.targets["answer"])
    failures: list[str] = []
    unknown_exclusions = excluded - declared
    if unknown_exclusions:
        failures.append(
            f"{bundle.name}: smoke_excluded_families names undeclared families "
            f"{sorted(unknown_exclusions)}"
        )
    for fam in sorted((declared or present) - excluded):
        if fam not in present:
            failures.append(
                f"{bundle.name}/{fam}: declared but absent from smoke and not in [gold] "
                "smoke_excluded_families — a held-out/degenerate family must be disclosed, "
                "not silently dropped"
            )
            continue
        golds = by_family.get(fam, set())
        if len(golds) < 2:
            failures.append(
                f"{bundle.name}/{fam}: only gold class(es) {sorted(golds)} occur "
                "off-quarantine (need ≥2 distinct) — cannot separate signal from bias"
            )

    supplemental_by_family: dict[str, set] = {}
    for rec in bundle.supplemental_records:
        if rec.targets.get("margin_bin") == quar:
            continue
        supplemental_by_family.setdefault(rec.prompt_family, set()).add(rec.targets["answer"])
    for fam in sorted(excluded & declared):
        golds = supplemental_by_family.get(fam, set())
        if len(golds) < 2:
            failures.append(
                f"{bundle.name}/{fam}: excluded from smoke but supplemental manifests "
                f"contain only class(es) {sorted(golds)} off-quarantine (need ≥2 distinct)"
            )
    return GateResult(
        "class_balance",
        not failures,
        failures,
        {
            "classes_off_quarantine": {k: sorted(v) for k, v in by_family.items()},
            "supplemental_classes_off_quarantine": {
                k: sorted(v) for k, v in supplemental_by_family.items()
            },
            "excluded": sorted(excluded),
        },
    )


# ---------------------------------------------------------------------------
# Gate 5 — gallery + human audit
# ---------------------------------------------------------------------------

_AUDIT_REQUIRED = ("# Human audit", "Reviewer:", "| prompt family |", "Signed-off-by:")


def _audit_is_signed(text: str) -> bool:
    """A signed audit has a Reviewer name and a Signed-off-by name, both non-placeholder."""
    import re

    def _filled(label: str) -> bool:
        m = re.search(rf"{re.escape(label)}\s*(.*)", text)
        if not m:
            return False
        v = m.group(1).strip().strip("*_ ")
        return bool(v) and "UNSIGNED" not in v and "___" not in v and v.lower() != "todo"

    return _filled("Reviewer:") and _filled("Signed-off-by:")


def gate_gallery_audit(bundle: WorldBundle, gallery_ok: bool = True) -> GateResult:
    """A world is eval-eligible only if its gallery builds and a signed
    ``docs/audits/<world>.md`` exists (gate 5). The gate enforces the file's
    existence and SHAPE (required sections + a per-family table); the operator
    signs it later. Signature presence (→ eval-eligibility) is reported in
    ``notes['signed']``, separately from the shape check the gate enforces."""
    failures: list[str] = []
    if not gallery_ok:
        failures.append(f"{bundle.name}: gallery did not build")
    if bundle.audits_dir is None:
        failures.append(f"{bundle.name}: no audits_dir configured")
        return GateResult("gallery_audit", False, failures)
    audit = bundle.audits_dir / f"{bundle.name}.md"
    signed = False
    if not audit.is_file():
        failures.append(
            f"{bundle.name}: missing human-audit file {audit} — copy docs/audits/TEMPLATE.md "
            "and have the operator sign it (an agent never signs a human audit)"
        )
    else:
        text = audit.read_text(encoding="utf-8")
        missing = [s for s in _AUDIT_REQUIRED if s not in text]
        if missing:
            failures.append(f"{bundle.name}: audit {audit.name} missing sections {missing}")
        missing_families = [
            family
            for family in getattr(bundle.prompts, "PROMPT_FAMILIES", {})
            if f"`{family}`" not in text
        ]
        if missing_families:
            failures.append(
                f"{bundle.name}: audit {audit.name} omits prompt families {missing_families}"
            )
        signed = _audit_is_signed(text)
    return GateResult("gallery_audit", not failures, failures, {"signed": signed})


# ---------------------------------------------------------------------------
# run all
# ---------------------------------------------------------------------------


def run_all(bundle: WorldBundle, rng: random.Random, gallery_ok: bool = True) -> dict:
    """Run all five gates over one world; return ``{gate_name: GateResult}``."""
    return {
        "latent_symmetry": gate_latent_symmetry(bundle, rng),
        "sign_blind": gate_sign_blind(bundle),
        "stated_boundary": gate_stated_boundary(bundle),
        "class_balance": gate_class_balance(bundle),
        "gallery_audit": gate_gallery_audit(bundle, gallery_ok=gallery_ok),
    }
