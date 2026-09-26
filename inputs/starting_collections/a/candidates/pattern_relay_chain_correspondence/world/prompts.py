"""Stable public questions and constrained answer mappings."""

PROMPT_FAMILIES = {
    "pf1": (
        "The magenta-framed pattern is START. Each relay card maps its left pattern "
        "to its right pattern; follow the arrow. Find the card whose left pattern "
        "exactly matches START, then keep matching each output to another card's input. "
        "Stop when no input matches. What color tag is on the last card: red, green, "
        "or blue? Do not rotate or reflect patterns."
    ),
    "pf2": (
        "Begin with the 3x3 glyph in the magenta START box. On a relay, an arrow maps "
        "the left glyph to the right glyph. Repeatedly choose the unique relay whose "
        "left glyph is exactly your current glyph, then carry forward its right glyph. "
        "When no relay accepts it, answer with that final relay's tag color: red, green, "
        "or blue. Glyph orientation never changes."
    ),
}


def correct_answer(family: str, decision: str) -> str:
    if family not in PROMPT_FAMILIES:
        raise KeyError(family)
    if decision not in ("red", "green", "blue"):
        raise ValueError(decision)
    return decision


def candidates_for(family: str) -> tuple[str, ...]:
    if family not in PROMPT_FAMILIES:
        raise KeyError(family)
    return ("red", "green", "blue")
