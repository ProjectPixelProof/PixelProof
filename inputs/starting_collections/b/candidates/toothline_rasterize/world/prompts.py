"""Public prompt contract for the tooth-route world."""

PROMPT_FAMILIES = {
    "pf1": (
        "Follow the connected route from the amber triangle to the teal circle. "
        "It has nine V-shaped teeth. In route order, write a dark cell for a tooth "
        "pointing up and a light cell for a tooth pointing down, filling a 3 by 3 "
        "tile left-to-right and top-to-bottom. Which symbol appears: X, plus, T, or L?"
    ),
    "pf2": (
        "Start at the amber triangle and trace the unbroken line to the teal circle. "
        "Convert its nine teeth in order: up-pointing means dark and down-pointing "
        "means light. Place the results row-wise in a 3 by 3 grid. Is the resulting "
        "symbol X, plus, T, or L?"
    ),
}

_CANDIDATES = ("X", "plus", "T", "L")


def correct_answer(family: str, decision: str) -> str:
    if family not in PROMPT_FAMILIES or decision not in _CANDIDATES:
        raise ValueError("unknown prompt family or decision")
    return decision


def candidates_for(family: str) -> tuple[str, ...]:
    if family not in PROMPT_FAMILIES:
        raise ValueError("unknown prompt family")
    return _CANDIDATES
