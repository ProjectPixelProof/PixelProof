"""Prompt families for the angle-acuteness world (SSOT §7.5, §7.2 axis 4; design §3).

Five families cross the query-composition axis on identical images: pf1-pf3 are
lexical variants of the same decision (acute / smaller-than-right / sharp), pf4
flips the polarity (obtuse), and pf5 changes the decision *type* to a
multiple-choice selection. Framing variance across these families is itself a core
metric (SSOT §8.6).
"""

from __future__ import annotations

# Bump when prompt text or the gold-answer mapping changes: manifests freeze both
# at generation time, and this version (stored in ExampleRecord.extra) is how
# downstream analysis detects drift between a manifest and this module.
PROMPTS_VERSION = "angle_acuteness-prompts-0.1.0"

PROMPT_FAMILIES: dict[str, str] = {
    "pf1_acute_yesno": "Is the angle acute? Answer yes or no.",
    "pf2_less_right_yesno": (
        "Is the angle smaller than a right angle (90 degrees)? Answer yes or no."
    ),
    "pf3_sharp_yesno": "Is the angle sharp (narrower than a square corner)? Answer yes or no.",
    "pf4_obtuse_yesno": "Is the angle obtuse? Answer yes or no.",
    "pf5_acute_mc": (
        "Is the angle A) acute (less than 90 degrees) or B) obtuse (more than 90 degrees)?"
    ),
}

_YES_NO = ("yes", "no")
_AB = ("A", "B")


def candidates_for(family: str) -> tuple[str, ...]:
    """Constrained answer set for a prompt family."""
    return _AB if family == "pf5_acute_mc" else _YES_NO


def correct_answer(family: str, m: float) -> str:
    """Ground-truth constrained answer given angular margin m = 90 - θ (degrees).

    Binary convention (design §1): acute = (m > 0), obtuse = (m < 0), strict. A
    right angle (m == 0) is NEITHER acute nor obtuse, so pf1-pf3 gold "no" AND pf4
    (obtuse) golds "no" as well; pf5 (selection) folds the right-angle tie to "A" by
    the pinned convention (design §3). Right angles live only in the quarantined
    right band (SSOT §21.6), so downstream metrics never trust the boundary label.
    """
    acute = m > 0
    obtuse = m < 0
    if family in ("pf1_acute_yesno", "pf2_less_right_yesno", "pf3_sharp_yesno"):
        return "yes" if acute else "no"
    if family == "pf4_obtuse_yesno":
        return "yes" if obtuse else "no"
    if family == "pf5_acute_mc":
        return "A" if m >= 0 else "B"  # right-angle tie folds to A (design §3)
    raise KeyError(f"unknown prompt family {family!r}")
