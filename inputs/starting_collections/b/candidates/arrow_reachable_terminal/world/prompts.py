"""Stable public wording and constrained answers."""

PROMPT_FAMILIES: dict[str, str] = {
    "pf1": (
        "Starting at the green START circle, follow black links only in the direction "
        "of their purple arrowheads. Which colored terminal can be reached? Links meet "
        "only at circles, and every link may be followed only in its arrow's direction. "
        "Answer red, blue, or orange."
    )
}


def correct_answer(family: str, decision: str) -> str:
    if family not in PROMPT_FAMILIES or decision not in {"red", "blue", "orange"}:
        raise KeyError((family, decision))
    return decision


def candidates_for(family: str) -> tuple[str, ...]:
    if family not in PROMPT_FAMILIES:
        raise KeyError(family)
    return ("red", "blue", "orange")
