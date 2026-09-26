"""Prompt families for the global-elongation world.

Every family is a constrained four-way selection over the principal cloud axes
``horizontal``, ``rising``, ``vertical``, ``falling``. ``correct_answer`` receives
the structural decision (one of those four axes) and maps it to the family's
answer; here the decision is already the axis label.
"""

from __future__ import annotations

_AXES = ("horizontal", "rising", "vertical", "falling")

PROMPT_FAMILIES: dict[str, str] = {
    "pf1_cloud_axis": (
        "The dark dots together form one elongated cloud. Looking at the whole "
        "cloud, along which single direction is it stretched out: horizontal, "
        "rising (uphill to the right), vertical, or falling (downhill to the "
        "right)? Answer horizontal, rising, vertical, or falling."
    ),
    "pf2_global_stretch": (
        "Consider only the global shape of the entire dot cloud, not any single "
        "dot. Along which axis is the whole cloud elongated: horizontal, rising, "
        "vertical, or falling? Answer one of those four words."
    ),
}


def candidates_for(family: str) -> tuple[str, ...]:
    return _AXES


def correct_answer(family: str, decision: str) -> str:
    """Ground-truth constrained answer given the structural decision."""
    if decision not in _AXES:
        raise ValueError(f"unknown decision {decision!r}")
    return decision
