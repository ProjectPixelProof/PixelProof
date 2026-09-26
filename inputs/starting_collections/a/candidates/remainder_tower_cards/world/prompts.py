"""Public question variants for remainder tower cards."""

PROMPT_FAMILIES: dict[str, str] = {
    "pf1": (
        "Order the four violet cards by their one to four amber dots. On each card, "
        "replace each tower by a dark square when its height in boxes is a multiple "
        "of three, otherwise by a pale square. Use the four results as rows of a 4 "
        "by 4 tile. Which symbol appears: hourglass, diamond, falling blocks, or rising blocks?"
    ),
    "pf2": (
        "Count each tower's boxes and decode it as dark exactly when the count is "
        "divisible by three. Sort the detached cards by amber-dot count from one to "
        "four and stack their decoded rows. Select the result: hourglass, diamond, "
        "falling blocks, or rising blocks."
    ),
}

_ANSWERS = ("hourglass", "diamond", "falling blocks", "rising blocks")


def candidates_for(family: str) -> tuple[str, ...]:
    if family not in PROMPT_FAMILIES:
        raise KeyError(family)
    return _ANSWERS


def correct_answer(family: str, decision: str) -> str:
    if family not in PROMPT_FAMILIES or decision not in _ANSWERS:
        raise KeyError((family, decision))
    return decision
