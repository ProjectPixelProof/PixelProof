"""Prompt families for the texture_flow_orientation world."""

from __future__ import annotations

CLASSES = ("horizontal", "rising", "vertical", "falling")

PROMPT_FAMILIES = {
    "flow_direction": (
        "A field of many small dark strokes is scattered across the canvas. "
        "Taken together the whole stroke texture points predominantly in a "
        "single direction: most of the little strokes lean the same way, like "
        "a gentle flow field, while a few unrelated strokes point elsewhere. "
        "Which direction does the whole texture field predominantly align: "
        "horizontal, rising (uphill to the right), vertical, or falling "
        "(downhill to the right)? Answer with the single direction word."
    ),
    "field_orientation": (
        "Ignoring the few stray marks, the overall pattern of the small strokes "
        "forms one dominant orientation across the whole image. Which one "
        "orientation does the whole stroke field lean toward: horizontal, "
        "rising, vertical, or falling? Answer with the single direction word."
    ),
}


def correct_answer(family: str, decision: str) -> str:
    return decision


def candidates_for(family: str) -> tuple[str, ...]:
    return CLASSES
