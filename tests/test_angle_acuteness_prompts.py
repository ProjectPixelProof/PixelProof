"""Prompt-family gold answers for angle_acuteness (design §1: acute = m > 0, strict).

The full family x sign matrix is pinned here because a silent flip of any one
family's gold mapping shows up downstream as a plausible-looking accuracy
inversion, as happened historically for two_circles/pf5.
"""

import pytest

from worlds.angle_acuteness.prompts import (
    PROMPT_FAMILIES,
    PROMPTS_VERSION,
    candidates_for,
    correct_answer,
)

# (family, gold when acute m>0, gold when obtuse m<0)
GOLD_MATRIX = [
    ("pf1_acute_yesno", "yes", "no"),
    ("pf2_less_right_yesno", "yes", "no"),
    ("pf3_sharp_yesno", "yes", "no"),
    ("pf4_obtuse_yesno", "no", "yes"),  # polarity flip
    ("pf5_acute_mc", "A", "B"),  # selection
]


def test_matrix_covers_every_family():
    assert {f for f, *_ in GOLD_MATRIX} == set(PROMPT_FAMILIES)


@pytest.mark.parametrize(("family", "gold_acute", "gold_obtuse"), GOLD_MATRIX)
def test_correct_answer_matrix(family, gold_acute, gold_obtuse):
    assert correct_answer(family, 12.0) == gold_acute
    assert correct_answer(family, -12.0) == gold_obtuse


@pytest.mark.parametrize("family", sorted(PROMPT_FAMILIES))
def test_right_angle_convention(family):
    # A right angle (m == 0) is NEITHER acute nor obtuse: pf1-pf3 gold "no" AND pf4
    # (obtuse) golds "no" as well; pf5 folds to "A" (design §3). Pinned; right angles
    # live only in the quarantined right band (SSOT §21.6). Note pf4 at m=0 is "no",
    # not the naive polarity flip "yes".
    expected = {
        "pf1_acute_yesno": "no",
        "pf2_less_right_yesno": "no",
        "pf3_sharp_yesno": "no",
        "pf4_obtuse_yesno": "no",
        "pf5_acute_mc": "A",
    }[family]
    assert correct_answer(family, 0.0) == expected


@pytest.mark.parametrize("family", sorted(PROMPT_FAMILIES))
@pytest.mark.parametrize("m", [-5.0, 0.0, 5.0])
def test_gold_is_always_a_candidate(family, m):
    assert correct_answer(family, m) in candidates_for(family)


def test_candidate_sets():
    assert candidates_for("pf5_acute_mc") == ("A", "B")
    for family in sorted(PROMPT_FAMILIES):
        if family != "pf5_acute_mc":
            assert candidates_for(family) == ("yes", "no")


def test_lexical_variants_agree_polarity_flips():
    # pf1/pf2/pf3 are lexical variants of the SAME decision -> identical gold.
    for m in (7.0, -7.0):
        assert (
            correct_answer("pf1_acute_yesno", m)
            == correct_answer("pf2_less_right_yesno", m)
            == correct_answer("pf3_sharp_yesno", m)
        )
    # pf4 (obtuse) is the exact polarity flip of pf1 (acute) OFF the boundary.
    for m in (7.0, -7.0):
        assert correct_answer("pf4_obtuse_yesno", m) != correct_answer("pf1_acute_yesno", m)


def test_unknown_family_raises():
    with pytest.raises(KeyError):
        correct_answer("pf9_nope", 5.0)


def test_prompts_version_pinned():
    assert PROMPTS_VERSION == "angle_acuteness-prompts-0.1.0"
