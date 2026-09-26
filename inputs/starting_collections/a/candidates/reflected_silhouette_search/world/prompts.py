"""Stable prompt families for the reflected-silhouette search."""

from __future__ import annotations

PROMPTS_VERSION = "reflected-silhouette-search-prompts-0.1.0"
PROMPT_FAMILIES: dict[str, str] = {
    "pf1_reflection_pair": (
        "Each sector contains two separate filled glyphs. Imagine the line through the midpoint "
        "of the line joining their centers, perpendicular to that joining line. Which sector "
        "contains the only pair whose silhouettes are mirror images across that line? Answer NW, NE, SW, or SE."
    ),
    "pf2_shape_match": (
        "Compare the complete outlines of the two detached glyphs in each sector using the "
        "perpendicular-bisector mirror axis between their centers. Select the only sector where "
        "one outline reflects exactly onto the other: NW, NE, SW, or SE."
    ),
    "pf3_visual_search": (
        "Search all four sectors for the pair of separate asymmetric shapes that are related by "
        "reflection across their shared midpoint axis. Return NW, NE, SW, or SE."
    ),
}

SECTORS = ("NW", "NE", "SW", "SE")


def candidates_for(family: str) -> tuple[str, ...]:
    if family not in PROMPT_FAMILIES:
        raise KeyError(f"unknown prompt family {family!r}")
    return SECTORS


def correct_answer(family: str, decision: str) -> str:
    if family not in PROMPT_FAMILIES:
        raise KeyError(f"unknown prompt family {family!r}")
    if decision not in SECTORS:
        raise ValueError(f"invalid decision {decision!r}")
    return decision
