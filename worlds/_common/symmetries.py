"""Reusable latent-symmetry twins for the answerability gate (SSOT §7.14 gate 1).

A *latent symmetry* is a transform of the scene that leaves the RENDER unchanged.
The gate proves each declared twin renders pixel-identically, then asserts gold is
identical — so a world only benefits from declaring the symmetries it genuinely
has (the gate mechanically verifies them; a bogus one that changes the render or
flips gold fails the gate). Two shapes recur across the bank:

- **slot permutation** — the scene holds objects in POSITIONAL slots (parallel
  lists, or a fixed ``c1``/``c2`` pair) with colour as an attribute of each
  object. Permuting the slots re-labels which slot holds which object without
  moving any ink, so the picture is identical; gold must not depend on slot
  order. Catches "gold keyed to object #1 vs #2 / list position" bugs.
- **no nontrivial symmetry** — a minimal parametrization (a single object; or
  objects KEYED BY a visible attribute like colour, so there is no positional
  slot to permute). Declared as an empty generator with a docstring justification;
  the gate cannot check the justification (the review does) but forces it to be
  stated.

Renderers whose objects are keyed by colour (``longest_bar``) have no slot to
permute; renderers with positional slots (``two_circles``'s ``c1``/``c2``; the
counting worlds' dot lists) do.
"""

from __future__ import annotations

import dataclasses
import random


def permute_list_group(scene, groups: list[tuple[str, ...]], rng: random.Random):
    """Return a twin of ``scene`` with each group of PARALLEL list attributes
    shuffled by a shared permutation (so parallel lists stay aligned).

    ``groups`` is a list of tuples of attribute names; the attributes within one
    tuple are parallel lists permuted together (e.g. ``("disks", "colors")``);
    separate tuples get independent permutations. A scene must be a frozen
    dataclass (uses :func:`dataclasses.replace`).
    """
    changes: dict[str, list] = {}
    for group in groups:
        lists = [list(getattr(scene, a)) for a in group]
        n = len(lists[0])
        if any(len(lst) != n for lst in lists):
            raise ValueError(f"parallel lists in {group} have unequal length")
        perm = list(range(n))
        rng.shuffle(perm)
        if perm == list(range(n)) and n > 1:  # force a real permutation when possible
            perm = perm[1:] + perm[:1]
        for a, lst in zip(group, lists, strict=True):
            changes[a] = [lst[i] for i in perm]
    return dataclasses.replace(scene, **changes)


def swap_slots(scene, pairs: list[tuple[str, str]]):
    """Return a twin with each ``(attr_a, attr_b)`` pair of scene attributes
    swapped — the two-slot special case of a slot permutation (e.g. swap
    ``("c1", "c2")``, ``("r1", "r2")``, ``("color1", "color2")`` for two_circles).
    """
    a_vals = {a: getattr(scene, a) for a, _ in pairs}
    b_vals = {b: getattr(scene, b) for _, b in pairs}
    changes = {}
    for a, b in pairs:
        changes[a] = b_vals[b]
        changes[b] = a_vals[a]
    return dataclasses.replace(scene, **changes)
