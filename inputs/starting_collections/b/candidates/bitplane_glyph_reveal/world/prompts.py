"""Prompt families for bitplane_glyph_reveal."""

from __future__ import annotations

PROMPTS_VERSION = "bitplane_glyph_reveal-prompts-0.1.0"

PROMPT_FAMILIES = {
    "pf1_bitplane_ab": (
        "The image is a dense field of gray noise. The answer letter is hidden only "
        "in the least-significant bit (bit-plane 0) of each pixel's gray value; the "
        "other seven bits are random noise. Read that least-significant bit plane. "
        "Which letter, A or B, is revealed? Answer A or B."
    ),
    "pf2_lsb_ab": (
        "Look at the pixels. Take every gray value and keep only its lowest bit "
        "(0 or 1). Plot where that bit is 1 and you recover a single block letter. "
        "Which letter appears, A or B? Answer A or B."
    ),
    "pf3_sum_lowbit_ab": (
        "Each gray value is random background noise carrying one hidden signal bit "
        "in its last binary place. Isolate the least-significant-bit plane. What "
        "letter does that bit-plane form, A or B? Answer A or B."
    ),
}

_AB = ("A", "B")


def candidates_for(family: str) -> tuple[str, ...]:
    return _AB


def correct_answer(family: str, glyph: str) -> str:
    return glyph
