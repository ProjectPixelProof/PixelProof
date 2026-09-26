PROMPT_FAMILIES = {
    "direct": "Is the red marker above the visible horizontal reference line?",
    "relative": "Relative to the gray line, is the red marker higher in the image?",
}


def correct_answer(family: str, decision: str) -> str:
    if family not in PROMPT_FAMILIES:
        raise KeyError(family)
    if decision not in {"yes", "no"}:
        raise ValueError(decision)
    return decision


def candidates_for(family: str) -> tuple[str, ...]:
    if family not in PROMPT_FAMILIES:
        raise KeyError(family)
    return ("yes", "no")
