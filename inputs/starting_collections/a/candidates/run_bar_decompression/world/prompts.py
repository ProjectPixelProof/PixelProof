"""Public prompt contract for run-bar decompression."""

PROMPT_FAMILIES = {
    "pf1": (
        "In each cyan lane, read the bars from left to right. A dark bar means black "
        "cells and a pale bar means white cells; the bar height in small grid steps is "
        "that run's length. Expand the runs into one row per lane, then stack the lanes "
        "from top to bottom. Which symbol appears: zigzag, key, anchor, or bell?"
    ),
    "pf2": (
        "Decode every cyan lane left-to-right: dark bars expand to black cells, pale bars "
        "expand to white cells, and each bar's height in grid steps gives how many cells. "
        "Stack the ten decoded rows top-to-bottom. Answer zigzag, key, anchor, or bell."
    ),
}

_CANDIDATES = ("zigzag", "key", "anchor", "bell")


def correct_answer(family: str, decision: str) -> str:
    if family not in PROMPT_FAMILIES or decision not in _CANDIDATES:
        raise KeyError((family, decision))
    return decision


def candidates_for(family: str) -> tuple[str, ...]:
    if family not in PROMPT_FAMILIES:
        raise KeyError(family)
    return _CANDIDATES
