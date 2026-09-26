"""Public prompt contract for spring_coil_count."""

PROMPT_FAMILIES: dict[str, str] = {
    "pf1_visible_full_coils": (
        "How many full coils are visible in the single blue spring? Count one coil for each "
        "complete top-to-bottom-to-top cycle while following the wire from left to right. "
        "Do not count a partial cycle at either end."
    )
}


def correct_answer(family: str, decision: str) -> str:
    if family not in PROMPT_FAMILIES:
        raise KeyError(family)
    return str(decision)


def candidates_for(family: str) -> tuple[str, ...]:
    if family not in PROMPT_FAMILIES:
        raise KeyError(family)
    return ("3", "4", "5", "6", "7")

