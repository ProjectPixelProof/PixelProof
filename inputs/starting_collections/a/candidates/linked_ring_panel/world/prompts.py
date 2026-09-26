"""Public prompt contract for the interlocked-ring panel world."""

from __future__ import annotations

_RULES = (
    "This image shows five framed square panels of identical size standing side "
    "by side in one row, numbered 1 to 5 from left to right. Every panel holds "
    "exactly two rings of the same thickness, drawn in two different flat "
    "colours, and in every panel the two rings cross each other at exactly two "
    "places, so ring count, ring size, colour, position and the amount of ink "
    "carry no information. The rings are opaque, so at each crossing one ring "
    "passes in front and hides a short piece of the other one, leaving a plain "
    "break in the ring behind. Two rings are interlocked when neither ring "
    "stays in front the whole way round: one ring passes in front at one "
    "crossing and behind at the other, so each of the two rings carries exactly "
    "one break and each is still a single connected curve. Two rings are merely "
    "stacked when the same ring passes in front at both crossings, so that "
    "front ring carries no break at all while the ring behind it is broken "
    "twice and falls into two separate pieces. Exactly one panel holds a pair "
    "of interlocked rings. In each of the other four panels the pair is merely "
    "stacked, and the piece of the ring behind that lies between its two breaks "
    "is at least 26 pixels long, so both of its breaks are plainly visible. "
    "Which ring is drawn in which colour, which ring is in front at a given "
    "crossing, and where the pair sits inside its panel are all free to vary."
)

_QUESTION = (
    " Which numbered panel holds the two interlocked rings, that is the panel "
    "in which each of the two rings carries exactly one break?"
)

PROMPT_FAMILIES = {
    "pf_panel": (
        _RULES
        + _QUESTION
        + " Answer with exactly one of: panel-1, panel-2, panel-3, panel-4, "
        "panel-5."
    ),
    "pf_position": (
        _RULES
        + " Where in the row does the panel holding the two interlocked rings, "
        "the panel in which each of the two rings carries exactly one break, "
        "stand? Answer with exactly one of: leftmost, second-from-left, middle, "
        "second-from-right, rightmost."
    ),
}

_PANEL_ANSWERS = ("panel-1", "panel-2", "panel-3", "panel-4", "panel-5")
_POSITION_ANSWERS = (
    "leftmost",
    "second-from-left",
    "middle",
    "second-from-right",
    "rightmost",
)


def candidates_for(family: str) -> tuple:
    if family == "pf_panel":
        return _PANEL_ANSWERS
    if family == "pf_position":
        return _POSITION_ANSWERS
    raise KeyError(f"unknown prompt family {family!r}")


def correct_answer(family: str, decision: str) -> str:
    if decision not in _PANEL_ANSWERS:
        raise ValueError(f"unknown decision {decision!r}")
    return candidates_for(family)[_PANEL_ANSWERS.index(decision)]
