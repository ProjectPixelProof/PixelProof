"""Prompt-family gold answers for counting_with_distractors (design §3: count = N).

The full family x N gold matrix is pinned here because a silent flip of any one
family's gold mapping shows up downstream as a plausible-looking accuracy
inversion (as happened historically for two_circles/pf5). Unlike the distance
worlds the gold keys off the COUNT N, not a signed margin.
"""

import pytest

from worlds.counting_with_distractors.prompts import (
    PROMPT_FAMILIES,
    PROMPTS_VERSION,
    candidates_for,
    correct_answer,
)

# design §3 table: N -> (pf1/pf2, pf3 >3, pf4 <=3, pf5 ==4)
GOLD_MATRIX = {
    2: ("2", "no", "yes", "no"),
    3: ("3", "no", "yes", "no"),
    4: ("4", "yes", "no", "yes"),
    5: ("5", "yes", "no", "no"),
    6: ("6", "yes", "no", "no"),
    7: ("7", "yes", "no", "no"),
}


def test_families_are_the_five_designed():
    assert set(PROMPT_FAMILIES) == {
        "pf1_howmany_count",
        "pf2_count_count",
        "pf3_morethan3_yesno",
        "pf4_atmost3_yesno",
        "pf5_exactly4_yesno",
    }


@pytest.mark.parametrize("n", sorted(GOLD_MATRIX))
def test_correct_answer_matrix(n):
    count_gold, pf3, pf4, pf5 = GOLD_MATRIX[n]
    assert correct_answer("pf1_howmany_count", n) == count_gold
    assert correct_answer("pf2_count_count", n) == count_gold
    assert correct_answer("pf3_morethan3_yesno", n) == pf3
    assert correct_answer("pf4_atmost3_yesno", n) == pf4
    assert correct_answer("pf5_exactly4_yesno", n) == pf5


def test_count_family_answer_is_str_of_n():
    for n in range(2, 8):
        assert correct_answer("pf1_howmany_count", n) == str(n)
        assert correct_answer("pf2_count_count", n) == str(n)


@pytest.mark.parametrize("n", sorted(GOLD_MATRIX))
def test_pf3_pf4_are_exact_complements(n):
    # "more than three" is the exact logical complement of "at most three".
    assert correct_answer("pf3_morethan3_yesno", n) != correct_answer("pf4_atmost3_yesno", n)


def test_lexical_count_variants_agree():
    for n in range(2, 8):
        assert correct_answer("pf1_howmany_count", n) == correct_answer("pf2_count_count", n)


@pytest.mark.parametrize("n", sorted(GOLD_MATRIX))
def test_gold_is_always_a_candidate(n):
    for family in PROMPT_FAMILIES:
        assert correct_answer(family, n) in candidates_for(family)


def test_candidate_sets():
    assert candidates_for("pf1_howmany_count") == ("2", "3", "4", "5", "6", "7")
    assert candidates_for("pf2_count_count") == ("2", "3", "4", "5", "6", "7")
    for family in ("pf3_morethan3_yesno", "pf4_atmost3_yesno", "pf5_exactly4_yesno"):
        assert candidates_for(family) == ("yes", "no")


def test_unknown_family_raises():
    with pytest.raises(KeyError):
        correct_answer("pf9_nope", 4)
    with pytest.raises(KeyError):
        candidates_for("pf9_nope")


def test_prompts_version_pinned():
    assert PROMPTS_VERSION == "counting_with_distractors-prompts-0.1.0"
