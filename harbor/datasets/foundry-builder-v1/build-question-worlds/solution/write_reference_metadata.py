"""Adapt the deterministic Oracle fixture to the materialized protocol version."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

SUBMISSION = Path("/logs/artifacts/submission")
TASK_SPEC = Path("/workspace/TASK_SPEC.toml")


def main() -> None:
    task = tomllib.loads(TASK_SPEC.read_text(encoding="utf-8"))
    if task.get("protocol") != "question-world@0.4.0":
        return
    portfolio_path = SUBMISSION / "portfolio.json"
    portfolio = json.loads(portfolio_path.read_text(encoding="utf-8"))
    portfolio["schema_version"] = "0.4.0"
    portfolio_path.write_text(
        json.dumps(portfolio, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    candidate_path = SUBMISSION / "candidates/reference_marker/candidate.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["schema_version"] = "0.4.0"
    candidate["latent_alias"] = {
        "mode": "tested_transforms",
        "justification": (
            "The unused latent tag is varied while rendered pixels and analytic "
            "gold remain identical, directly exercising the declared alias."
        ),
        "label_inputs": ["marker_y", "reference_y"],
    }
    candidate["mechanism_fingerprint"] = {
        "decision_family": "predicate",
        "entities": ["point", "line"],
        "arity": "pair",
        "relation": "order",
        "margin_operator": "signed_difference",
        "oracle_primitives": ["color_segmentation", "line_fit", "centroid"],
        "composition_depth": 1,
    }
    candidate["semantic_contrast"] = {
        "closest_known_worlds": ["vertical_order"],
        "known_negative_matches": ["horizontal_order"],
        "contrast": (
            "This deterministic Oracle fixture is intentionally a coordinate-axis "
            "restatement of vertical_order and therefore is not scientifically novel."
        ),
        "expected_information_gain": (
            "None; it exists solely to prove that the v0.4 packaging and verifier "
            "contract can pass end to end."
        ),
    }
    candidate_path.write_text(
        json.dumps(candidate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
