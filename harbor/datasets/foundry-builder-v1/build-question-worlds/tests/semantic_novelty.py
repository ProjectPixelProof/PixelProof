"""Deterministic semantic-neighbor diagnostics for low-level visual worlds."""

from __future__ import annotations

import json
import re
from pathlib import Path

FINGERPRINT_KEYS = {
    "decision_family",
    "entities",
    "arity",
    "relation",
    "margin_operator",
    "oracle_primitives",
    "composition_depth",
}
ALLOWED = {
    "decision_family": {"predicate", "comparison", "selection", "count", "composition"},
    "entities": {
        "point",
        "line",
        "segment",
        "bar",
        "disk",
        "circle",
        "polygon",
        "region",
        "arc",
        "angle",
        "axis",
        "gauge",
        "set",
        "shape",
        "color",
        "attribute",
    },
    "arity": {"single", "pair", "triple", "set", "mixed"},
    "relation": {
        "proximity",
        "intersection",
        "magnitude",
        "containment",
        "angle",
        "count",
        "proportion",
        "collinearity",
        "order",
        "orientation",
        "distance",
        "spacing",
        "parity",
        "halfplane",
        "composition",
        "curvature",
        "convexity",
        "dispersion",
        "attribute_binding",
    },
    "margin_operator": {
        "signed_distance",
        "signed_difference",
        "absolute_threshold",
        "top_gap",
        "minimum",
        "count_difference",
        "crowding_clearance",
        "angular_difference",
        "discrete_boundary",
        "composition_min",
    },
    "oracle_primitives": {
        "color_segmentation",
        "connected_components",
        "bounding_box",
        "centroid",
        "line_fit",
        "circle_fit",
        "polygon_fit",
        "arc_fit",
        "pixel_count",
        "distance_geometry",
        "composition",
    },
}
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "does",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "two",
    "which",
    "with",
}
_SYNONYMS = {
    "above": "order",
    "below": "order",
    "circle": "disk",
    "circles": "disk",
    "cross": "intersection",
    "crossing": "intersection",
    "dot": "point",
    "dots": "point",
    "height": "length",
    "heights": "length",
    "inside": "containment",
    "intersect": "intersection",
    "larger": "magnitude",
    "largest": "maximum",
    "left": "order",
    "longer": "length",
    "longest": "maximum",
    "marker": "point",
    "markers": "point",
    "rectangle": "shape",
    "right": "order",
    "square": "shape",
    "tall": "length",
    "taller": "length",
    "tallest": "maximum",
    "width": "proportion",
    "wider": "proportion",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number} is not a JSON object")
        rows.append(value)
    return rows


def validate_fingerprint(value: object) -> list[str]:
    if not isinstance(value, dict):
        return ["mechanism_fingerprint must be an object"]
    failures = []
    if set(value) != FINGERPRINT_KEYS:
        failures.append(f"mechanism_fingerprint keys must be exactly {sorted(FINGERPRINT_KEYS)}")
    for key in ("decision_family", "arity", "relation", "margin_operator"):
        if value.get(key) not in ALLOWED[key]:
            failures.append(f"mechanism_fingerprint {key} is invalid")
    for key in ("entities", "oracle_primitives"):
        items = value.get(key)
        if (
            not isinstance(items, list)
            or not items
            or len(items) != len(set(items))
            or any(item not in ALLOWED[key] for item in items)
        ):
            failures.append(f"mechanism_fingerprint {key} is invalid")
    depth = value.get("composition_depth")
    if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= 4:
        failures.append("mechanism_fingerprint composition_depth must be in [1, 4]")
    return failures


def _tokens(text: str) -> set[str]:
    values = set()
    for token in _TOKEN.findall(text.lower()):
        if token in _STOP:
            continue
        values.add(_SYNONYMS.get(token, token))
    return values


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def candidate_text(metadata: dict) -> str:
    signature = metadata.get("task_signature") or {}
    return " ".join(
        str(value)
        for value in (
            metadata.get("title", ""),
            metadata.get("question", ""),
            signature.get("decision_var", ""),
            signature.get("structure", ""),
            signature.get("mechanism", ""),
        )
    )


def memory_text(row: dict) -> str:
    return " ".join(
        str(value)
        for value in (
            row.get("world_id", ""),
            row.get("proposal_id", ""),
            row.get("summary", ""),
        )
    )


def fingerprint_similarity(
    left: dict,
    right: dict,
    *,
    left_text: str = "",
    right_text: str = "",
) -> float:
    score = 0.0
    score += 0.15 * float(left.get("decision_family") == right.get("decision_family"))
    score += 0.10 * float(left.get("arity") == right.get("arity"))
    score += 0.25 * float(left.get("relation") == right.get("relation"))
    score += 0.20 * float(left.get("margin_operator") == right.get("margin_operator"))
    score += 0.10 * _jaccard(set(left.get("entities", [])), set(right.get("entities", [])))
    score += 0.05 * _jaccard(
        set(left.get("oracle_primitives", [])),
        set(right.get("oracle_primitives", [])),
    )
    score += 0.05 * float(left.get("composition_depth") == right.get("composition_depth"))
    score += 0.10 * _jaccard(_tokens(left_text), _tokens(right_text))
    return round(score, 6)


def semantic_neighbors(metadata: dict, memory: list[dict], *, limit: int = 5) -> list[dict]:
    fingerprint = metadata.get("mechanism_fingerprint") or {}
    text = candidate_text(metadata)
    rows = []
    for item in memory:
        other = item.get("fingerprint") or {}
        rows.append(
            {
                "world_id": item.get("world_id"),
                "score": fingerprint_similarity(
                    fingerprint,
                    other,
                    left_text=text,
                    right_text=memory_text(item),
                ),
                "summary": item.get("summary", ""),
            }
        )
    rows.sort(key=lambda item: (-item["score"], str(item["world_id"])))
    return rows[:limit]


def negative_neighbors(metadata: dict, memory: list[dict], *, limit: int = 5) -> list[dict]:
    tokens = _tokens(candidate_text(metadata))
    rows = []
    for item in memory:
        score = _jaccard(tokens, _tokens(memory_text(item)))
        rows.append(
            {
                "proposal_id": item.get("proposal_id"),
                "score": round(score, 6),
                "reason_code": item.get("reason_code"),
                "canonical_neighbors": item.get("canonical_neighbors", []),
                "summary": item.get("summary", ""),
            }
        )
    rows.sort(key=lambda item: (-item["score"], str(item["proposal_id"])))
    return rows[:limit]


def candidate_as_memory(metadata: dict) -> dict:
    return {
        "world_id": metadata.get("candidate_id"),
        "fingerprint": metadata.get("mechanism_fingerprint") or {},
        "summary": candidate_text(metadata),
    }
