"""Question contract for the unmarked-spur route world."""

PROMPT_FAMILIES = {
    "pf1_spur_ownership": (
        "Trace the thick dark network from the green start. Does the path to the "
        "amber terminal contain an extra unmarked dead-end spur? Ignore the "
        "colored terminal markers; answer yes or no."
    ),
    "pf2_anonymous_branch": (
        "Follow the dark route from green. Is the extra uncolored dead-end branch "
        "on the arm leading to amber rather than violet? Answer yes or no."
    ),
    "pf3_route_topology": (
        "Starting at green, inspect the connected thick route. Does its anonymous "
        "spur belong to the amber arm? Answer yes or no."
    ),
}

PROMPTS_VERSION = "unmarked-spur-route-prompts-0.1.0"
_YES_NO = ("yes", "no")


def candidates_for(family: str) -> tuple[str, ...]:
    if family not in PROMPT_FAMILIES:
        raise KeyError(f"unknown prompt family {family!r}")
    return _YES_NO


def correct_answer(family: str, decision: str) -> str:
    if family not in PROMPT_FAMILIES:
        raise KeyError(f"unknown prompt family {family!r}")
    if decision not in _YES_NO:
        raise ValueError(decision)
    return decision
