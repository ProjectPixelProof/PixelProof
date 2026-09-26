"""Stable prompt families and constrained answers for the candidate world."""

PROMPT_FAMILIES: dict[str, str] = {
    "pf1_curvature_rank": (
        "The image shows a REFERENCE panel and candidates A and B. Each panel has four "
        "uniquely colored curved strokes. Rank the strokes from 1 (least bowed) to 4 "
        "(most bowed) by normalized bow from the straight endpoint chord. Which candidate "
        "has the same color-to-curvature-rank assignment as the REFERENCE? Ignore panel "
        "position, stroke position, rotation, reflection, and line width. Answer A or B."
    ),
    "pf2_normalized_bow": (
        "Compare the REFERENCE with candidates A and B. For every colored open curve, "
        "measure how far it bows from its own endpoint-to-endpoint chord, divided by chord "
        "length, and rank the four bows from 1 through 4. Which candidate preserves the "
        "complete color-to-bow-rank map? Ignore translation, rotation, reflection, scale, "
        "and stroke width. Answer A or B."
    ),
    "pf3_curve_profile": (
        "Each of the three panels contains four separate colored arcs. Match the candidate "
        "whose colors are attached to the same ordered curvature profile as in the "
        "REFERENCE, where rank 1 is the least curved and rank 4 is the most curved after "
        "normalizing by each arc's chord. Ignore layout and size. Answer A or B."
    ),
}


def correct_answer(family: str, decision: str) -> str:
    if family not in PROMPT_FAMILIES:
        raise KeyError(f"unknown prompt family {family!r}")
    if decision not in {"A", "B"}:
        raise ValueError(f"decision is not a candidate answer: {decision!r}")
    return decision


def candidates_for(family: str) -> tuple[str, ...]:
    if family not in PROMPT_FAMILIES:
        raise KeyError(f"unknown prompt family {family!r}")
    return ("A", "B")
