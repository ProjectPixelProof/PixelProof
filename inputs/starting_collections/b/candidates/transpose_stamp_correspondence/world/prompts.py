"""Public prompt families for transpose-stamp correspondence."""

from __future__ import annotations

PROMPT_FAMILIES: dict[str, str] = {
    "pf1_rows_columns": (
        "The black-bordered mini-grid is the reference. Swap its rows and columns: "
        "the cell at row r, column c moves to row c, column r. Exactly one colored-bordered "
        "mini-grid is the result. Which border color marks it? Answer amber, cyan, or violet."
    ),
    "pf2_diagonal_reflection": (
        "Treat every mini-grid as a 3-by-3 matrix. Reflect the black-bordered reference "
        "across its top-left-to-bottom-right diagonal (equivalently, swap row and column "
        "indices). Which colored border surrounds the unique exact result? Answer amber, "
        "cyan, or violet."
    ),
}

_CANDIDATES = ("amber", "cyan", "violet")


def correct_answer(family: str, decision: str) -> str:
    if family not in PROMPT_FAMILIES:
        raise KeyError(f"unknown prompt family: {family}")
    if decision not in _CANDIDATES:
        raise ValueError(f"invalid transpose-stamp decision: {decision}")
    return decision


def candidates_for(family: str) -> tuple[str, ...]:
    if family not in PROMPT_FAMILIES:
        raise KeyError(f"unknown prompt family: {family}")
    return _CANDIDATES
