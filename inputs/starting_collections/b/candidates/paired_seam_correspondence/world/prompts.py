"""Stable yes/no prompt families for paired seam correspondence."""

PROMPT_FAMILIES = {
    "pf1_direct": (
        "Ignoring color, position, size, and the gap, do the two separated tiles have "
        "the same sequence of bumps and recesses along their facing edges? Keep the "
        "top-to-bottom order and do not rotate or flip either tile. Answer yes or no."
    ),
    "pf2_fit": (
        "If the two tiles were moved together without rotating or flipping them, would "
        "their facing contours match from top to bottom? Ignore color, location, size, "
        "and the empty gap. Answer yes or no."
    ),
    "pf3_contour": (
        "Compare the complete left-to-right seam pair in top-to-bottom order. Is the "
        "right edge of the left tile the same bump-and-recess profile as the left edge "
        "of the right tile? Answer yes or no."
    ),
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
