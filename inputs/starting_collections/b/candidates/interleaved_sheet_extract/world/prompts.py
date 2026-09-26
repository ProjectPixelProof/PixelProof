"""Prompt families for the interleaved-sheet extraction world."""

from __future__ import annotations

PROMPTS_VERSION = "interleaved_sheet_extract-prompts-0.1.0"

_SCENE = (
    "The picture shows one framed lattice of 14 rows and 7 columns of square "
    "cells; each cell is either dark (filled) or blank. Beside every lattice "
    "row, just outside the frame on its left, exactly one marker is printed: "
    "either a filled dot or a hollow ring. Exactly 7 rows carry a filled dot "
    "and exactly 7 rows carry a hollow ring. Discard every row whose marker is "
    "a hollow ring, keep the rows whose marker is a filled dot in their "
    "original top-to-bottom order, and push those 7 kept rows together into a "
    "single block of 7 rows and 7 columns. The dark cells of that block form "
    "one of these five symbols: ring (a closed square outline with a hollow "
    "centre), wedge (a solid triangle pointing down, widest across its top "
    "row), zigzag (a bar along the top, a staircase running down to the left, "
    "and a bar along the bottom), arrow (a broad arrowhead pointing down with "
    "a short stem below it), slash (a thick diagonal band running from the top "
    "left corner to the bottom right corner). "
)

PROMPT_FAMILIES: dict[str, str] = {
    "pf1_select_symbol": _SCENE
    + "Which of the five symbols do the dark cells of that block form? Answer "
    "with one word.",
    "pf2_ring_yesno": _SCENE
    + "Do the dark cells of that block form the ring symbol? Answer yes or no.",
}

_SYMBOLS = ("arrow", "ring", "slash", "wedge", "zigzag")
_YES_NO = ("yes", "no")


def candidates_for(family: str) -> tuple[str, ...]:
    return _YES_NO if family == "pf2_ring_yesno" else _SYMBOLS


def correct_answer(family: str, decision: str) -> str:
    if family == "pf2_ring_yesno":
        return "yes" if decision == "ring" else "no"
    return decision
