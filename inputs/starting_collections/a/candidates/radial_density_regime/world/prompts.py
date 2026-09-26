"""Prompt families for the radial-density-regime world.

Every family is constrained to the yes/no answer set. ``correct_answer``
receives the structural decision 'core' (packed toward the centre) or 'rim'
(packed toward the outer margin) and maps it to the family's polarity.
"""

from __future__ import annotations

PROMPT_FAMILIES: dict[str, str] = {
    "pf1_core_yesno": (
        "Across the whole canvas the dark dots are distributed radially. Are the "
        "dots packed tightly toward the centre (a solid core), with the rim spread "
        "thin? Answer yes or no."
    ),
    "pf2_rim_yesno": (
        "Across the whole canvas the dark dots are distributed radially. Are the "
        "dots packed toward the outside rim (a surrounding ring), leaving the "
        "centre sparse? Answer yes or no."
    ),
}

_YES_NO = ("yes", "no")


def candidates_for(family: str) -> tuple[str, ...]:
    return _YES_NO


def correct_answer(family: str, decision: str) -> str:
    """Ground-truth constrained answer given the structural decision."""
    if decision not in {"core", "rim"}:
        raise ValueError(f"unknown decision {decision!r}")
    core = decision == "core"
    if family == "pf1_core_yesno":
        return "yes" if core else "no"
    if family == "pf2_rim_yesno":
        return "yes" if not core else "no"
    raise ValueError(f"unknown prompt family {family!r}")
