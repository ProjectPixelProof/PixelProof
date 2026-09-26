"""Deterministic chunk sampling without replacement."""

from __future__ import annotations

import hashlib
from collections import defaultdict

from .datasets import EvalExample

SAMPLING_METHODS = frozenset({"uniform-example", "balanced-dataset"})


def _rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def select_chunk(
    examples: list[EvalExample],
    *,
    excluded_keys: set[str],
    sample_size: int,
    seed: int,
    method: str,
) -> list[EvalExample]:
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if method not in SAMPLING_METHODS:
        raise ValueError(f"unsupported sampling method: {method}")
    remaining = [item for item in examples if item.sample_key not in excluded_keys]
    if method == "uniform-example":
        ordered = sorted(remaining, key=lambda item: _rank(seed, item.sample_key))
        return ordered[:sample_size]

    groups = defaultdict(list)
    for item in remaining:
        groups[item.dataset_id].append(item)
    for dataset_id in groups:
        groups[dataset_id].sort(key=lambda item: _rank(seed, item.sample_key))
    group_order = sorted(groups, key=lambda item: _rank(seed, item))
    selected = []
    offset = 0
    while len(selected) < sample_size:
        added = False
        for dataset_id in group_order:
            group = groups[dataset_id]
            if offset < len(group):
                selected.append(group[offset])
                added = True
                if len(selected) == sample_size:
                    break
        if not added:
            break
        offset += 1
    return selected
