"""Prompt families (SSOT §7.5): semantically equivalent framings + adversarial variants.

Prompt-framing variance across these families is itself a core metric (SSOT §8.6).
"""

from __future__ import annotations

# Bump when prompt text or the gold-answer mapping changes: manifests freeze
# both at generation time, and this version (stored in ExampleRecord.extra)
# is how downstream analysis detects drift between a manifest and this module.
# 0.2.0: pf7/pf8/pf9 complete the lexical x polarity 2x2 (+ concept probe)
# motivated by the VLMH external finding (SSOT §8.2): word-specific blindness.
PROMPTS_VERSION = "two_circles-prompts-0.2.0"

# Standard framings (SSOT §7.5, verbatim).
PROMPT_FAMILIES: dict[str, str] = {
    "pf1_overlap_yesno": "Are the two circles overlapping? Answer yes or no.",
    "pf2_touch_or_overlap_yesno": "Are the two circles touching or overlapping? Answer yes or no.",
    "pf3_gap_yesno": "Is there any visible gap between the two circles? Answer yes or no.",
    "pf4_multiple_choice": (
        "Which statement is true: A) the circles are separate, B) the circles overlap?"
    ),
    "pf5_describe_then_answer": (
        "Describe the image briefly, then answer whether the circles overlap."
    ),
    "pf6_visual_evidence": "Focus only on the visual evidence. Are the circles overlapping?",
    # Lexical x polarity completion (SSOT §7.2 axis 4): pf1/pf3 are the
    # unnegated diagonal; pf7/pf8 are their negated counterparts (negation held
    # constant between them, so pf7-vs-pf8 is a clean word contrast); pf9
    # probes the separation CONCEPT without the word "gap" and without negation.
    "pf7_no_gap_yesno": "Is there no visible gap between the two circles? Answer yes or no.",
    "pf8_not_overlapping_yesno": "Are the two circles not overlapping? Answer yes or no.",
    "pf9_apart_yesno": "Are the two circles apart? Answer yes or no.",
}

# Adversarial / misleading text variants (SSOT §7.5) — used for the
# text-prior override index (SSOT §8.7).
ADVERSARIAL_PROMPTS: dict[str, str] = {
    "adv1_false_premise": "The circles are not touching. Is that correct? Answer yes or no.",
    "adv2_ignore_priors": "Ignore any prior assumptions. Are the circles overlapping?",
}

# In-image text overlays (rendered by the renderer via scene.overlay_text).
OVERLAY_TEXTS: dict[str, str] = {
    "adv_overlay_not_touching": "not touching",
    "adv_overlay_overlapping": "overlapping",
}

_YES_NO = ("yes", "no")
_AB = ("A", "B")


def candidates_for(family: str) -> tuple[str, ...]:
    """Constrained answer set for a prompt family (SSOT §25 item 2)."""
    return _AB if family == "pf4_multiple_choice" else _YES_NO


def correct_answer(family: str, m: float) -> str:
    """Ground-truth constrained answer given signed gap m.

    Binary convention (SSOT §7.3): overlap_or_touch = (m <= 0). Note §21.6 —
    examples in the tangent bin are ambiguous at pixel scale; downstream metrics
    should report near-boundary bins separately rather than trusting this label.
    """
    overlap = m <= 0
    # Families whose gold "yes" means SEPARATE (inverted polarity):
    if family in (
        "pf3_gap_yesno",  # gap visible iff separate
        "adv1_false_premise",  # "not touching" is correct iff separate
        "pf8_not_overlapping_yesno",  # negated overlap word
        "pf9_apart_yesno",  # separation concept, no "gap", no negation
    ):
        return "no" if overlap else "yes"
    if family == "pf4_multiple_choice":
        return "B" if overlap else "A"
    # Everything else (incl. pf7 "no visible gap"): gold "yes" iff overlap_or_touch.
    return "yes" if overlap else "no"
