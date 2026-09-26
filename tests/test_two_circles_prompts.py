"""Prompt-family gold answers (SSOT §7.5, §7.3 binary convention: overlap_or_touch = m <= 0).

The full family x sign matrix is pinned here because a silent flip of any one
family's gold mapping shows up downstream as a plausible-looking accuracy
inversion. In the historical pf5 incident, the apparent inversion was model
yes-bias, not a gold bug.
"""

import pytest

from worlds.two_circles.prompts import (
    ADVERSARIAL_PROMPTS,
    OVERLAY_TEXTS,
    PROMPT_FAMILIES,
    PROMPTS_VERSION,
    candidates_for,
    correct_answer,
)

ALL_FAMILIES = {**PROMPT_FAMILIES, **ADVERSARIAL_PROMPTS}

# (family, gold when overlapping/touching m<=0, gold when separate m>0)
GOLD_MATRIX = [
    ("pf1_overlap_yesno", "yes", "no"),
    ("pf2_touch_or_overlap_yesno", "yes", "no"),
    ("pf3_gap_yesno", "no", "yes"),  # gap visible iff separate
    ("pf4_multiple_choice", "B", "A"),
    ("pf5_describe_then_answer", "yes", "no"),
    ("pf6_visual_evidence", "yes", "no"),
    ("pf7_no_gap_yesno", "yes", "no"),  # "no gap" true iff overlap_or_touch
    ("pf8_not_overlapping_yesno", "no", "yes"),  # negated overlap word
    ("pf9_apart_yesno", "no", "yes"),  # separation concept probe
    ("adv1_false_premise", "no", "yes"),  # "not touching" is correct iff separate
    ("adv2_ignore_priors", "yes", "no"),
]


def test_matrix_covers_every_family():
    assert {f for f, *_ in GOLD_MATRIX} == set(ALL_FAMILIES)


@pytest.mark.parametrize(("family", "gold_overlap", "gold_separate"), GOLD_MATRIX)
def test_correct_answer_matrix(family, gold_overlap, gold_separate):
    assert correct_answer(family, -10.0) == gold_overlap
    assert correct_answer(family, 0.0) == gold_overlap  # touching counts as overlap_or_touch
    assert correct_answer(family, 10.0) == gold_separate


@pytest.mark.parametrize("family", sorted(ALL_FAMILIES))
@pytest.mark.parametrize("m", [-5.0, 0.0, 5.0])
def test_gold_is_always_a_candidate(family, m):
    assert correct_answer(family, m) in candidates_for(family)


def test_candidate_sets():
    assert candidates_for("pf4_multiple_choice") == ("A", "B")
    for family in sorted(ALL_FAMILIES):
        if family != "pf4_multiple_choice":
            assert candidates_for(family) == ("yes", "no")


def test_pf5_gold_not_inverted_regression():
    # The observed pf5 accuracy inversion was model yes-bias, NOT a gold bug;
    # the gold mapping must stay aligned with pf1 (yes iff overlap_or_touch).
    assert correct_answer("pf5_describe_then_answer", 10.0) == "no"
    assert correct_answer("pf5_describe_then_answer", -10.0) == "yes"


def test_overlay_texts_are_short_ascii():
    for text in OVERLAY_TEXTS.values():
        assert text.isascii() and 0 < len(text) < 30


def test_prompts_version_pinned():
    # Manifests freeze prompt text + gold mapping at generation time and store
    # this version in extra; bump it whenever either changes.
    assert PROMPTS_VERSION == "two_circles-prompts-0.2.0"


def test_lexical_2x2_is_complete():
    # SSOT §7.2 axis 4: pf1/pf3 unnegated diagonal, pf7/pf8 negated
    # counterparts with opposite gold, pf9 the no-negation concept probe.
    assert correct_answer("pf7_no_gap_yesno", -10.0) == correct_answer("pf1_overlap_yesno", -10.0)
    assert correct_answer("pf8_not_overlapping_yesno", -10.0) == correct_answer(
        "pf3_gap_yesno", -10.0
    )
    assert correct_answer("pf9_apart_yesno", 10.0) == "yes"
