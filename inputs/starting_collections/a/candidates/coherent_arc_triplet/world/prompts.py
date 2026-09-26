"""Public prompt contract for the coherent-arc-triplet world."""

from __future__ import annotations

PROMPT_FAMILIES: dict[str, str] = {
    "pf1_common_circle": (
        "Do all three separated colored curve fragments trace parts of one imaginary "
        "circle? Answer yes or no."
    ),
    "pf2_odd_fragment": (
        "Does one of the three separated colored curve fragments bend away from the "
        "single circle suggested by the other two? Answer yes or no."
    ),
}


def candidates_for(family: str) -> tuple[str, ...]:
    if family not in PROMPT_FAMILIES:
        raise KeyError(family)
    return ("yes", "no")


def correct_answer(family: str, decision: str) -> str:
    if decision not in {"yes", "no"}:
        raise ValueError(f"unknown decision {decision!r}")
    if family == "pf1_common_circle":
        return decision
    if family == "pf2_odd_fragment":
        return "no" if decision == "yes" else "yes"
    raise KeyError(family)
