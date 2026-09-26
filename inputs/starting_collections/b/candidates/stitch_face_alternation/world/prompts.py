"""Prompt families for ``stitch_face_alternation``.

Every family states all rules needed to solve the image: how to find the
stitched tape, what front and behind look like, where the walk starts, and the
exact answer set.  Wording is identical for control and counterfactual scenes.
"""

from __future__ import annotations

PROMPTS_VERSION = "stitch_face_alternation-prompts-0.1.0"

_SCENE = (
    "This picture shows a mending board on a plain white background. Exactly one "
    "flat tape strip has a thin thread stitched through it: that thread crosses "
    "the tape exactly five times, and one end of that thread carries a small "
    "crimson square tag. At each crossing the thread either passes in FRONT of "
    "the tape, staying visible right across it, or BEHIND the tape, hidden where "
    "the tape covers it and leaving a visible gap. Loose thread scraps and bare "
    "tape pieces elsewhere in the picture are not stitched through any tape and "
    "must be ignored. Walk the thread from its tagged end and take the five "
    "crossings in the order the thread meets them."
)

PROMPT_FAMILIES: dict[str, str] = {
    "pf1_alternates": (
        _SCENE
        + " The stitch alternates if front and behind strictly take turns along "
        "that walk, never twice in a row. Does this stitch alternate? Answer with "
        "exactly one of: yes, no."
    ),
    "pf2_break": (
        _SCENE
        + " The stitch has a break in its pattern if two crossings in a row are "
        "on the same side of the tape, both front or both behind. Does this "
        "stitch have such a break? Answer with exactly one of: yes, no."
    ),
    "pf3_first_face": (
        _SCENE
        + " Consider only the first of those five crossings. At that crossing, "
        "does the thread pass in front of the tape or behind it? Answer with "
        "exactly one of: front, behind."
    ),
}

_YESNO = ("yes", "no")
_FACES = ("front", "behind")


def _parts(decision: str) -> tuple[str, str]:
    pattern, first = decision.split("|")
    if pattern not in ("alternating", "broken") or first not in _FACES:
        raise ValueError(f"unknown decision {decision!r}")
    return pattern, first


def correct_answer(family: str, decision: str) -> str:
    pattern, first = _parts(decision)
    if family == "pf1_alternates":
        return "yes" if pattern == "alternating" else "no"
    if family == "pf2_break":
        return "yes" if pattern == "broken" else "no"
    if family == "pf3_first_face":
        return first
    raise KeyError(family)


def candidates_for(family: str) -> tuple[str, ...]:
    if family in ("pf1_alternates", "pf2_break"):
        return _YESNO
    if family == "pf3_first_face":
        return _FACES
    raise KeyError(family)
