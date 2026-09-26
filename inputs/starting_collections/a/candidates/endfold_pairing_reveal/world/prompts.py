"""Prompt families for the end-fold pairing reveal world.

Each family declares every rule needed to solve the image: which cord counts,
where the walk starts, how the cord is folded end to end, which pairs count as
marked, how the 25 verdicts fill the 5x5 grid, the glyph alphabet, and the
answer set. No family reveals the route, the tones, or the glyph.
"""

from __future__ import annotations

PROMPTS_VERSION = "endfold_pairing_reveal-prompts-0.1.0"

GLYPHS = ("ell", "ex", "plus", "tee")
BOTTOM_LEFT_GLYPHS = frozenset({"ex", "ell"})

_RULES = (
    "Exactly one cord in this picture is teal; any other cord is pale lavender. The teal cord "
    "is a single unbranching thin route with exactly two ends, and it carries exactly 50 round "
    "beads, one at every corner and one at each end. Each bead is filled with one of just two "
    "grays, a darker gray and a lighter gray; the two-square key near one corner of the picture "
    "shows those two grays. Exactly one end bead of the teal cord has four small black brackets "
    "around it. Carry out this operation: walk the teal cord from that bracketed end bead to its "
    "other end and number the beads 1 to 50 in the order you meet them; then fold the cord end "
    "to end, so that bead 1 meets bead 50, bead 2 meets bead 49, bead 3 meets bead 48, and so on "
    "down to bead 25 meeting bead 26, giving 25 folded pairs; then mark every folded pair whose "
    "two beads are filled with the same gray, and leave a pair unmarked when its two beads differ. "
    "Then write the 25 verdicts into a 5 by 5 grid, filling the first row left to right with pairs "
    "1 to 5, the second row with pairs 6 to 10, and so on to the fifth row with pairs 21 to 25. "
    "Exactly 9 of the 25 positions are marked. The marked positions form one symbol on the 5 by 5 "
    "grid: 'plus' is the full centre row together with the full centre column, 'ex' is the two full "
    "diagonals, 'tee' is the full top row together with the centre column below it, and 'ell' is "
    "the full left column together with the full bottom row. The shape the cord traces across the "
    "picture is not the symbol, the lavender cords and the key are decoration, and only the folded "
    "pairing of beads from the two ends of the walk decides which positions are marked."
)

PROMPT_FAMILIES: dict[str, str] = {
    "pf1_symbol_choice": (
        _RULES + " Which symbol do the marked positions form? Answer with exactly one of: "
        "ell, ex, plus, tee."
    ),
    "pf2_bottom_left_yesno": (
        _RULES + " Is the bottom-left corner position of that 5 by 5 grid, the leftmost "
        "position of the fifth row, one of the marked positions? Answer yes or no."
    ),
}

_YES_NO = ("yes", "no")


def candidates_for(family: str) -> tuple[str, ...]:
    """Constrained answer set for one prompt family."""
    if family == "pf1_symbol_choice":
        return GLYPHS
    if family == "pf2_bottom_left_yesno":
        return _YES_NO
    raise KeyError(f"unknown prompt family {family!r}")


def correct_answer(family: str, decision: str) -> str:
    """Gold constrained answer for one family given the world decision."""
    if decision not in GLYPHS:
        raise ValueError(f"decision {decision!r} is not a world glyph")
    if family == "pf1_symbol_choice":
        return decision
    if family == "pf2_bottom_left_yesno":
        return "yes" if decision in BOTTOM_LEFT_GLYPHS else "no"
    raise KeyError(f"unknown prompt family {family!r}")
