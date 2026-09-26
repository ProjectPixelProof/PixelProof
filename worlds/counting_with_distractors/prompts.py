"""Prompt families for the counting-with-distractors world (SSOT §7.5, §7.2 axis 4;
design §3).

Two decision *types* share each scene: an open count (small-integer answer set)
and threshold predicates (yes/no). pf1/pf2 are lexical variants of "how many red
dots"; pf3/pf4 are exact logical complements (> 3 vs <= 3) — a consistency check on
the model, not just the gold; pf5 is a point predicate (== 4). Framing variance
across these families on identical images is itself a core metric (SSOT §8.6).

Unlike the distance worlds, the gold answer keys off the COUNT N (a scene
parameter), not a signed margin — ``correct_answer(family, count)``.
"""

from __future__ import annotations

# Bump when prompt text or the gold-answer mapping changes: manifests freeze both
# at generation time, and this version (stored in ExampleRecord.extra) is how
# downstream analysis detects drift between a manifest and this module.
PROMPTS_VERSION = "counting_with_distractors-prompts-0.1.0"

PROMPT_FAMILIES: dict[str, str] = {
    "pf1_howmany_count": "How many red dots are there? Answer with a number.",
    "pf2_count_count": "Count the red dots. Give the number.",
    "pf3_morethan3_yesno": "Are there more than three red dots? Answer yes or no.",
    "pf4_atmost3_yesno": "Are there at most three red dots? Answer yes or no.",
    "pf5_exactly4_yesno": "Are there exactly four red dots? Answer yes or no.",
}

# Count families admit the digit strings for the sampled range N in [2, 7].
_COUNT_CANDIDATES = ("2", "3", "4", "5", "6", "7")
_YES_NO = ("yes", "no")
_COUNT_FAMILIES = ("pf1_howmany_count", "pf2_count_count")


def candidates_for(family: str) -> tuple[str, ...]:
    """Constrained answer set for a prompt family."""
    if family in _COUNT_FAMILIES:
        return _COUNT_CANDIDATES
    if family in PROMPT_FAMILIES:
        return _YES_NO
    raise KeyError(f"unknown prompt family {family!r}")


def correct_answer(family: str, count: int) -> str:
    """Ground-truth constrained answer given the target-dot count N (design §3).

    The count is definitional (a scene parameter); the threshold families apply a
    fixed, scene-independent predicate to it. pf3 (> 3) and pf4 (<= 3) are exact
    complements. The constrained answer for the count families is ``str(N)``.
    """
    if family in _COUNT_FAMILIES:
        return str(count)
    if family == "pf3_morethan3_yesno":
        return "yes" if count > 3 else "no"
    if family == "pf4_atmost3_yesno":
        return "yes" if count <= 3 else "no"
    if family == "pf5_exactly4_yesno":
        return "yes" if count == 4 else "no"
    raise KeyError(f"unknown prompt family {family!r}")
