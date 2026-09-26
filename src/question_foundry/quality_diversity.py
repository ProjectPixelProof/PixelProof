"""Deterministic archive policy for Section 2 diversity-directed discovery.

The archive never changes scientific validity. Protected gates decide whether a
candidate is a valid executable question; this module only decides whether that
valid candidate is an active diversity elite and which feasible niche should be
targeted next. Legacy builder-declared descriptors remain explicitly labeled;
the current policy uses descriptors recomputed by the protected verifier.
"""

from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from typing import Any

from question_foundry.candidate_contract import DECISION_TYPES
from question_foundry.registry import canonical_json
from question_foundry.semantic_novelty import ALLOWED

LEGACY_POLICY_ID = "quality-diversity@0.1.0"
POLICY_ID = "quality-diversity@0.2.0"
ADAPTIVE_POLICY_ID = "quality-diversity@0.3.0"
ROBUST_POLICY_ID = "quality-diversity@0.4.0"
REFINED_POLICY_ID = "quality-diversity@0.5.0"
HIGH_RES_POLICY_ID = "quality-diversity@0.6.0"
STEERED_POLICY_ID = "quality-diversity@0.7.0"
PREVIEW_POLICY_ID = "quality-diversity@0.8.0"
ATTESTATION_ID = "pixel-oracle-support@0.1.0"
ADAPTIVE_ATTESTATION_ID = "pixel-oracle-support@0.2.0"
ROBUST_ATTESTATION_ID = "pixel-oracle-support@0.3.0"
REFINED_ATTESTATION_ID = "pixel-oracle-support@0.4.0"
HIGH_RES_ATTESTATION_ID = "pixel-oracle-support@0.5.0"
SUPPORTED_AXES = frozenset({"relation", "decision_family", "decision_type", "composition_depth"})
CONTROLLED_VALUES = {
    "decision_family": frozenset(ALLOWED["decision_family"]),
    "decision_type": DECISION_TYPES,
    "relation": frozenset(ALLOWED["relation"]),
}
ARCHIVE_SCHEMA = "quality-diversity-archive-0.1.0"
DECISION_SCHEMA = "quality-diversity-decision-0.1.0"
ATTESTED_AXES = ("evidence_extent", "evidence_topology")
ROBUST_AXES = ("support_scale", "support_shape")
ATTESTED_VALUES = {
    "evidence_extent": frozenset({"local", "long_range", "global"}),
    "evidence_topology": frozenset({"compact", "elongated", "multipart", "diffuse"}),
}
ATTESTED_DEFINITIONS = {
    "evidence_extent": {
        "local": "A small compact part of the image determines the pixel-oracle answer.",
        "long_range": (
            "Separated or image-spanning evidence must be integrated, without most of the "
            "canvas being individually critical."
        ),
        "global": (
            "Evidence is broadly distributed; ablating most spatial tiles disrupts the answer."
        ),
    },
    "evidence_topology": {
        "compact": "The reproducible pixel-oracle support is one compact spatial component.",
        "elongated": "The support is one extended trace-, path-, or axis-like component.",
        "multipart": "Two or more separated support components must be bound or compared.",
        "diffuse": "Support is dense across the image rather than concentrated in a few regions.",
    },
}

ROBUST_VALUES = {
    "support_scale": frozenset({"focal", "regional", "distributed"}),
    "support_shape": frozenset({"compact", "pathlike", "multipart"}),
}
ROBUST_DEFINITIONS = {
    "support_scale": {
        "focal": "The decisive pixel-oracle support remains within a small local span.",
        "regional": "The decisive support spans a substantial region without crossing the image.",
        "distributed": "The decisive support spans widely separated parts of the image.",
    },
    "support_shape": {
        "compact": "The decisive support is one connected, area-like component.",
        "pathlike": "The decisive support is one connected thin or elongated component.",
        "multipart": "The decisive support contains multiple disconnected components.",
    },
}

STEERING_RECIPES = {
    "evidence_extent": {
        "local": (
            "Concentrate the decisive evidence in one small image region. Keep distant marks "
            "nondiagnostic, and make the pixel oracle depend on the local relation rather than "
            "segmenting the full canvas."
        ),
        "long_range": (
            "Make at least two distant locations or the endpoints of an image-spanning structure "
            "jointly necessary, while avoiding a rule that makes nearly every tile decisive."
        ),
        "global": (
            "Make the answer aggregate evidence distributed across several image regions. A local "
            "crop must be insufficient, but the inverse oracle must remain robust when one region "
            "is partially occluded."
        ),
    },
    "evidence_topology": {
        "compact": (
            "Arrange the causal support as one connected compact component; avoid multiple "
            "separated evidence islands and long thin paths."
        ),
        "elongated": (
            "Arrange the causal support as one long thin trace, route, axis, or ordered strip. "
            "The answer should depend on following that extended component."
        ),
        "multipart": (
            "Require binding or comparison of two or more separated support components. Removing "
            "either component should change or prevent the answer."
        ),
        "diffuse": (
            "Distribute causal evidence densely across the canvas rather than concentrating it in "
            "a few regions; avoid making oracle crashes the source of apparent support."
        ),
    },
}

ROBUST_STEERING_RECIPES = {
    "support_scale": {
        "focal": (
            "Put all decisive evidence in one small neighborhood. Randomize its absolute "
            "position across scenes, and make remote marks nondiagnostic."
        ),
        "regional": (
            "Make the answer depend on evidence spread across one medium-sized region. A tiny "
            "crop must be insufficient, but opposite image edges need not interact."
        ),
        "distributed": (
            "Make widely separated image locations jointly necessary. A crop covering only one "
            "side or one distant part must be insufficient for the inverse oracle."
        ),
    },
    "support_shape": {
        "compact": (
            "Use one connected area-like evidence component with substantial interior support; "
            "avoid a thin route or disconnected islands."
        ),
        "pathlike": (
            "Use one connected thin trace, route, chain, axis, or ordered strip. The answer must "
            "depend on following that component rather than merely seeing an endpoint."
        ),
        "multipart": (
            "Require binding or comparison of at least two disconnected evidence components. "
            "Each component must contribute to the answer."
        ),
    },
}

CELL_STEERING_RECIPES = {
    "support_scale=focal|support_shape=compact": {
        "construct": (
            "Use one thick, connected cue inside a small local inset; the answer should change "
            "when that cue is removed, while the rest of the canvas is nondiagnostic."
        ),
        "avoid": "Do not use a long stroke, distant endpoints, or two separated evidence islands.",
    },
    "support_scale=focal|support_shape=pathlike": {
        "construct": (
            "Confine one short but clearly thin connected trace to a small local inset. Make the "
            "answer depend on following its interior, not only reading one endpoint."
        ),
        "avoid": "Do not span the canvas or reduce the answer to a compact marker lookup.",
    },
    "support_scale=focal|support_shape=multipart": {
        "construct": (
            "Place two or three disconnected decisive glyphs inside the same small local inset and "
            "require comparing or binding them; background must remain visible between components."
        ),
        "avoid": (
            "Do not put the components at opposite ends of a route, connect them with a decisive "
            "stroke, or let remote distractors affect the answer."
        ),
    },
    "support_scale=regional|support_shape=compact": {
        "construct": (
            "Use one thick connected region of medium span and make an area, centroid, occupancy, "
            "or interior relation across that region decide the answer."
        ),
        "avoid": "Do not use a thin route, isolated endpoint cue, or image-spanning region.",
    },
    "support_scale=regional|support_shape=pathlike": {
        "construct": (
            "Use one connected thin route contained within a medium-sized region; multiple "
            "interior portions of the route must matter to the inverse oracle."
        ),
        "avoid": (
            "Do not use only endpoints, disconnected cards, or a route spanning opposite edges."
        ),
    },
    "support_scale=regional|support_shape=multipart": {
        "construct": (
            "Use separated decisive components distributed across one medium-sized region. Require "
            "a comparison or binding for which every component contributes."
        ),
        "avoid": (
            "Do not connect the components with a decisive stroke or spread them across the canvas."
        ),
    },
    "support_scale=distributed|support_shape=compact": {
        "construct": (
            "Use one thick, high-fill connected structure spanning widely separated locations. "
            "Make the answer aggregate properties of multiple distant parts of that same region."
        ),
        "avoid": (
            "Do not use a thin serpentine route, endpoint-only lookup, separate panels, or "
            "multiple "
            "disconnected regions; those measure as pathlike or multipart."
        ),
    },
    "support_scale=distributed|support_shape=pathlike": {
        "construct": (
            "Use one connected thin trace spanning widely separated locations, and require "
            "following or integrating the trace rather than reading a single endpoint."
        ),
        "avoid": "Do not thicken it into an area-like region or split it into disconnected panels.",
    },
    "support_scale=distributed|support_shape=multipart": {
        "construct": (
            "Place two or more disconnected decisive components far apart and require binding or "
            "comparison across them; each distant component must affect the inverse answer."
        ),
        "avoid": (
            "Do not join the components with a decisive path or let one component determine the "
            "answer."
        ),
    },
}


def _cell_key(cell: dict, axes: tuple[str, ...]) -> str:
    return "|".join(f"{axis}={cell[axis]}" for axis in axes)


def validate_policy(policy: dict) -> None:
    if policy.get("id") == PREVIEW_POLICY_ID:
        _validate_preview_policy(policy)
        return
    if policy.get("id") == STEERED_POLICY_ID:
        _validate_steered_policy(policy)
        return
    if policy.get("id") == HIGH_RES_POLICY_ID:
        _validate_high_res_policy(policy)
        return
    if policy.get("id") == REFINED_POLICY_ID:
        _validate_refined_policy(policy)
        return
    if policy.get("id") == ROBUST_POLICY_ID:
        _validate_robust_policy(policy)
        return
    if policy.get("id") == ADAPTIVE_POLICY_ID:
        _validate_adaptive_policy(policy)
        return
    if policy.get("id") == POLICY_ID:
        _validate_attested_policy(policy)
        return
    if policy.get("id") != LEGACY_POLICY_ID:
        raise ValueError(
            "archive_policy.id must be one of "
            f"{LEGACY_POLICY_ID}, {POLICY_ID}, {ADAPTIVE_POLICY_ID}, "
            f"{ROBUST_POLICY_ID}, {REFINED_POLICY_ID}, {HIGH_RES_POLICY_ID}, "
            f"{STEERED_POLICY_ID}, or {PREVIEW_POLICY_ID}"
        )
    required = {
        "id",
        "descriptor_axes",
        "feasible_cells",
        "max_elites_per_cell",
        "global_similarity_ceiling",
        "within_cell_similarity_ceiling",
        "require_complete_within_cell_neighbors",
        "target_selection",
        "replacement_policy",
        "controller_seed",
    }
    if set(policy) != required:
        raise ValueError(f"archive_policy keys must be exactly {sorted(required)}")
    axes_raw = policy["descriptor_axes"]
    if (
        not isinstance(axes_raw, list)
        or not 2 <= len(axes_raw) <= 4
        or len(axes_raw) != len(set(axes_raw))
        or any(axis not in SUPPORTED_AXES for axis in axes_raw)
    ):
        raise ValueError("archive_policy.descriptor_axes are invalid")
    axes = tuple(axes_raw)
    cells = policy["feasible_cells"]
    if not isinstance(cells, list) or len(cells) < 2:
        raise ValueError("archive_policy requires at least two feasible cells")
    keys = []
    for cell in cells:
        if not isinstance(cell, dict) or set(cell) != set(axes):
            raise ValueError("every feasible cell must define exactly the descriptor axes")
        for axis, value in cell.items():
            if axis == "composition_depth":
                if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 4:
                    raise ValueError("composition_depth must be an integer in [1, 4]")
            elif value not in CONTROLLED_VALUES[axis]:
                raise ValueError(
                    f"archive_policy cell has invalid controlled {axis} value: {value!r}"
                )
        keys.append(_cell_key(cell, axes))
    if len(keys) != len(set(keys)):
        raise ValueError("archive_policy feasible cells must be unique")
    if not 1 <= int(policy["max_elites_per_cell"]) <= 10:
        raise ValueError("max_elites_per_cell must be in [1, 10]")
    for field in ("global_similarity_ceiling", "within_cell_similarity_ceiling"):
        value = float(policy[field])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must be in [0, 1]")
    if policy["require_complete_within_cell_neighbors"] is not True:
        raise ValueError("quality-diversity local competition must fail closed on missing elites")
    if policy["target_selection"] != "least_occupied_stable_hash":
        raise ValueError("unsupported quality-diversity target_selection")
    if policy["replacement_policy"] != "lower_semantic_similarity_wins":
        raise ValueError("unsupported quality-diversity replacement_policy")
    if isinstance(policy["controller_seed"], bool) or not isinstance(
        policy["controller_seed"], int
    ):
        raise ValueError("archive_policy.controller_seed must be an integer")


def _validate_attested_policy(policy: dict) -> None:
    required = {
        "id",
        "descriptor_attestation",
        "descriptor_axes",
        "feasible_cells",
        "max_elites_per_cell",
        "global_similarity_ceiling",
        "minimum_global_behavioral_distance",
        "minimum_within_cell_behavioral_distance",
        "maximum_oracle_ablation_error_fraction",
        "target_selection",
        "replacement_policy",
        "controller_seed",
    }
    if set(policy) != required:
        raise ValueError(f"attested archive_policy keys must be exactly {sorted(required)}")
    if policy["descriptor_attestation"] != ATTESTATION_ID:
        raise ValueError(f"descriptor_attestation must be {ATTESTATION_ID}")
    if tuple(policy["descriptor_axes"]) != ATTESTED_AXES:
        raise ValueError(f"attested descriptor_axes must be {list(ATTESTED_AXES)}")
    cells = policy["feasible_cells"]
    if not isinstance(cells, list) or len(cells) < 2:
        raise ValueError("archive_policy requires at least two feasible cells")
    keys = []
    for cell in cells:
        if not isinstance(cell, dict) or set(cell) != set(ATTESTED_AXES):
            raise ValueError("every feasible cell must define exactly the attested axes")
        for axis, value in cell.items():
            if value not in ATTESTED_VALUES[axis]:
                raise ValueError(f"invalid attested {axis} value: {value!r}")
        keys.append(_cell_key(cell, ATTESTED_AXES))
    if len(keys) != len(set(keys)):
        raise ValueError("archive_policy feasible cells must be unique")
    if not 2 <= int(policy["max_elites_per_cell"]) <= 10:
        raise ValueError("attested max_elites_per_cell must be in [2, 10]")
    for field in (
        "global_similarity_ceiling",
        "minimum_global_behavioral_distance",
        "minimum_within_cell_behavioral_distance",
        "maximum_oracle_ablation_error_fraction",
    ):
        value = float(policy[field])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must be in [0, 1]")
    if policy["target_selection"] != "least_occupied_stable_hash":
        raise ValueError("unsupported quality-diversity target_selection")
    if policy["replacement_policy"] != "maximin_behavioral_dispersion":
        raise ValueError("unsupported attested quality-diversity replacement_policy")
    if isinstance(policy["controller_seed"], bool) or not isinstance(
        policy["controller_seed"], int
    ):
        raise ValueError("archive_policy.controller_seed must be an integer")


def _validate_adaptive_policy(policy: dict) -> None:
    required = {
        "id",
        "descriptor_attestation",
        "descriptor_axes",
        "feasible_cells",
        "max_elites_per_cell",
        "global_similarity_ceiling",
        "minimum_global_behavioral_distance",
        "minimum_within_cell_behavioral_distance",
        "maximum_oracle_ablation_error_fraction",
        "minimum_descriptor_stability",
        "target_selection",
        "replacement_policy",
        "target_coverage_weight",
        "target_feasibility_weight",
        "target_exploration_weight",
        "target_miss_penalty",
        "target_cooldown_episodes",
        "controller_seed",
    }
    if set(policy) != required:
        raise ValueError(f"adaptive archive_policy keys must be exactly {sorted(required)}")
    if policy["descriptor_attestation"] != ADAPTIVE_ATTESTATION_ID:
        raise ValueError(f"descriptor_attestation must be {ADAPTIVE_ATTESTATION_ID}")
    shared = {
        key: value
        for key, value in policy.items()
        if key
        not in {
            "minimum_descriptor_stability",
            "target_coverage_weight",
            "target_feasibility_weight",
            "target_exploration_weight",
            "target_miss_penalty",
            "target_cooldown_episodes",
        }
    }
    shared["id"] = POLICY_ID
    shared["descriptor_attestation"] = ATTESTATION_ID
    shared["target_selection"] = "least_occupied_stable_hash"
    _validate_attested_policy(shared)
    if policy["target_selection"] != "adaptive_feasibility_ucb":
        raise ValueError("adaptive target_selection must be adaptive_feasibility_ucb")
    for field in (
        "minimum_descriptor_stability",
        "target_coverage_weight",
        "target_feasibility_weight",
        "target_exploration_weight",
        "target_miss_penalty",
    ):
        value = policy[field]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{field} must be numeric")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{field} must be finite and non-negative")
    if not 0.0 <= float(policy["minimum_descriptor_stability"]) <= 1.0:
        raise ValueError("minimum_descriptor_stability must be in [0, 1]")
    cooldown = policy["target_cooldown_episodes"]
    if isinstance(cooldown, bool) or not isinstance(cooldown, int) or cooldown < 0:
        raise ValueError("target_cooldown_episodes must be a non-negative integer")


def _validate_robust_policy(policy: dict) -> None:
    required = {
        "id",
        "descriptor_attestation",
        "descriptor_axes",
        "feasible_cells",
        "max_elites_per_cell",
        "global_similarity_ceiling",
        "minimum_global_behavioral_distance",
        "minimum_within_cell_behavioral_distance",
        "minimum_descriptor_stability",
        "target_selection",
        "replacement_policy",
        "target_coverage_weight",
        "target_feasibility_weight",
        "target_exploration_weight",
        "target_miss_penalty",
        "target_cooldown_episodes",
        "controller_seed",
    }
    if set(policy) != required:
        raise ValueError(f"robust archive_policy keys must be exactly {sorted(required)}")
    if policy["descriptor_attestation"] != ROBUST_ATTESTATION_ID:
        raise ValueError(f"descriptor_attestation must be {ROBUST_ATTESTATION_ID}")
    if tuple(policy["descriptor_axes"]) != ROBUST_AXES:
        raise ValueError(f"robust descriptor_axes must be {list(ROBUST_AXES)}")
    cells = policy["feasible_cells"]
    if not isinstance(cells, list) or len(cells) < 2:
        raise ValueError("archive_policy requires at least two feasible cells")
    keys = []
    for cell in cells:
        if not isinstance(cell, dict) or set(cell) != set(ROBUST_AXES):
            raise ValueError("every feasible cell must define exactly the robust axes")
        for axis, value in cell.items():
            if value not in ROBUST_VALUES[axis]:
                raise ValueError(f"invalid robust {axis} value: {value!r}")
        keys.append(_cell_key(cell, ROBUST_AXES))
    if len(keys) != len(set(keys)):
        raise ValueError("archive_policy feasible cells must be unique")
    if not 2 <= int(policy["max_elites_per_cell"]) <= 10:
        raise ValueError("robust max_elites_per_cell must be in [2, 10]")
    for field in (
        "global_similarity_ceiling",
        "minimum_global_behavioral_distance",
        "minimum_within_cell_behavioral_distance",
        "minimum_descriptor_stability",
    ):
        value = policy[field]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{field} must be numeric")
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{field} must be finite and in [0, 1]")
    for field in (
        "target_coverage_weight",
        "target_feasibility_weight",
        "target_exploration_weight",
        "target_miss_penalty",
    ):
        value = policy[field]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{field} must be numeric")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{field} must be finite and non-negative")
    if policy["target_selection"] != "adaptive_feasibility_ucb":
        raise ValueError("robust target_selection must be adaptive_feasibility_ucb")
    if policy["replacement_policy"] != "maximin_behavioral_dispersion":
        raise ValueError("unsupported robust quality-diversity replacement_policy")
    cooldown = policy["target_cooldown_episodes"]
    if isinstance(cooldown, bool) or not isinstance(cooldown, int) or cooldown < 0:
        raise ValueError("target_cooldown_episodes must be a non-negative integer")
    if isinstance(policy["controller_seed"], bool) or not isinstance(
        policy["controller_seed"], int
    ):
        raise ValueError("archive_policy.controller_seed must be an integer")


def _validate_refined_policy(policy: dict) -> None:
    compatible = deepcopy(policy)
    compatible["id"] = ROBUST_POLICY_ID
    compatible["descriptor_attestation"] = ROBUST_ATTESTATION_ID
    _validate_robust_policy(compatible)
    if policy["descriptor_attestation"] != REFINED_ATTESTATION_ID:
        raise ValueError(f"descriptor_attestation must be {REFINED_ATTESTATION_ID}")


def _validate_high_res_policy(policy: dict) -> None:
    compatible = deepcopy(policy)
    compatible["id"] = ROBUST_POLICY_ID
    compatible["descriptor_attestation"] = ROBUST_ATTESTATION_ID
    _validate_robust_policy(compatible)
    if policy["descriptor_attestation"] != HIGH_RES_ATTESTATION_ID:
        raise ValueError(f"descriptor_attestation must be {HIGH_RES_ATTESTATION_ID}")


def _validate_steered_policy(policy: dict) -> None:
    compatible = deepcopy(policy)
    compatible["id"] = HIGH_RES_POLICY_ID
    _validate_high_res_policy(compatible)


def _validate_preview_policy(policy: dict) -> None:
    """Validate v0.8 without changing the protected v0.7 archive semantics."""

    compatible = deepcopy(policy)
    compatible["id"] = STEERED_POLICY_ID
    _validate_steered_policy(compatible)


def initial_archive(policy: dict) -> dict:
    validate_policy(policy)
    axes = tuple(policy["descriptor_axes"])
    archive = {
        "schema_version": (
            "quality-diversity-archive-0.8.0"
            if policy["id"] == PREVIEW_POLICY_ID
            else "quality-diversity-archive-0.7.0"
            if policy["id"] == STEERED_POLICY_ID
            else "quality-diversity-archive-0.6.0"
            if policy["id"] == HIGH_RES_POLICY_ID
            else "quality-diversity-archive-0.5.0"
            if policy["id"] == REFINED_POLICY_ID
            else "quality-diversity-archive-0.4.0"
            if policy["id"] == ROBUST_POLICY_ID
            else "quality-diversity-archive-0.3.0"
            if policy["id"] == ADAPTIVE_POLICY_ID
            else "quality-diversity-archive-0.2.0"
            if policy["id"] == POLICY_ID
            else ARCHIVE_SCHEMA
        ),
        "policy_id": policy["id"],
        "descriptor_axes": list(axes),
        "descriptor_source": (
            "protected_scene_relative_pixel_oracle_ablation_v5"
            if policy["id"] in {HIGH_RES_POLICY_ID, STEERED_POLICY_ID, PREVIEW_POLICY_ID}
            else "protected_scene_relative_pixel_oracle_ablation_v4"
            if policy["id"] == REFINED_POLICY_ID
            else "protected_scene_relative_pixel_oracle_ablation_v3"
            if policy["id"] == ROBUST_POLICY_ID
            else "protected_scene_relative_pixel_oracle_ablation"
            if policy["id"] == ADAPTIVE_POLICY_ID
            else "protected_offline_pixel_oracle_ablation"
            if policy["id"] == POLICY_ID
            else "builder_declared_not_independently_validated"
        ),
        "descriptor_attestation": policy.get("descriptor_attestation"),
        "feasible_cell_count": len(policy["feasible_cells"]),
        "cells": {_cell_key(cell, axes): [] for cell in policy["feasible_cells"]},
        "decisions": 0,
        "elite_admissions": 0,
        "elite_replacements": 0,
    }
    if policy["id"] in {
        ADAPTIVE_POLICY_ID,
        ROBUST_POLICY_ID,
        REFINED_POLICY_ID,
        HIGH_RES_POLICY_ID,
        STEERED_POLICY_ID,
        PREVIEW_POLICY_ID,
    }:
        archive["target_statistics"] = {
            key: {
                "selected_count": 0,
                "completed_count": 0,
                "target_match_count": 0,
                "elite_admission_count": 0,
                "consecutive_misses": 0,
                "last_selected_episode": None,
                "observed_cell_counts": {},
            }
            for key in archive["cells"]
        }
    return archive


def _occupancy(archive: dict, cell_key: str) -> int:
    return len(archive["cells"][cell_key])


def select_target(policy: dict, archive: dict, *, episode_index: int) -> dict:
    validate_policy(policy)
    axes = tuple(policy["descriptor_axes"])
    candidates = []
    minimum = min(_occupancy(archive, key) for key in archive["cells"])
    score_rows = []
    for cell in policy["feasible_cells"]:
        key = _cell_key(cell, axes)
        rank = hashlib.sha256(
            f"{policy['controller_seed']}\0{episode_index}\0{key}".encode()
        ).hexdigest()
        if policy["id"] not in {
            ADAPTIVE_POLICY_ID,
            ROBUST_POLICY_ID,
            REFINED_POLICY_ID,
            HIGH_RES_POLICY_ID,
            STEERED_POLICY_ID,
            PREVIEW_POLICY_ID,
        }:
            if _occupancy(archive, key) == minimum:
                candidates.append((rank, key, cell))
            continue
        stats = archive["target_statistics"][key]
        completed = int(stats["completed_count"])
        matched = int(stats["target_match_count"])
        occupancy = _occupancy(archive, key)
        feasibility = (matched + 1.0) / (completed + 2.0)
        coverage = 1.0 / (occupancy + 1.0)
        uncertainty = 1.0 / math.sqrt(completed + 1.0)
        miss_penalty = min(3, int(stats["consecutive_misses"])) / 3.0
        last = stats["last_selected_episode"]
        cooldown_active = bool(
            isinstance(last, int)
            and episode_index - last <= int(policy["target_cooldown_episodes"])
        )
        score = (
            float(policy["target_coverage_weight"]) * coverage
            + float(policy["target_feasibility_weight"]) * feasibility
            + float(policy["target_exploration_weight"]) * uncertainty
            - float(policy["target_miss_penalty"]) * miss_penalty
            - (1.0 if cooldown_active else 0.0)
        )
        components = {
            "coverage": round(coverage, 6),
            "posterior_target_feasibility": round(feasibility, 6),
            "uncertainty": round(uncertainty, 6),
            "consecutive_miss_penalty": round(miss_penalty, 6),
            "cooldown_active": cooldown_active,
        }
        score_rows.append((score, rank, key, cell, components, deepcopy(stats)))
    if policy["id"] in {
        ADAPTIVE_POLICY_ID,
        ROBUST_POLICY_ID,
        REFINED_POLICY_ID,
        HIGH_RES_POLICY_ID,
        STEERED_POLICY_ID,
        PREVIEW_POLICY_ID,
    }:
        score, _rank, key, cell, score_components, history = max(
            score_rows, key=lambda item: (item[0], item[1])
        )
    else:
        _rank, key, cell = min(candidates)
        score = None
        score_components = None
        history = None
    target = {
        "schema_version": (
            "quality-diversity-target-0.8.0"
            if policy["id"] == PREVIEW_POLICY_ID
            else "quality-diversity-target-0.7.0"
            if policy["id"] == STEERED_POLICY_ID
            else "quality-diversity-target-0.6.0"
            if policy["id"] == HIGH_RES_POLICY_ID
            else "quality-diversity-target-0.5.0"
            if policy["id"] == REFINED_POLICY_ID
            else "quality-diversity-target-0.4.0"
            if policy["id"] == ROBUST_POLICY_ID
            else "quality-diversity-target-0.3.0"
            if policy["id"] == ADAPTIVE_POLICY_ID
            else "quality-diversity-target-0.2.0"
            if policy["id"] == POLICY_ID
            else "quality-diversity-target-0.1.0"
        ),
        "policy_id": policy["id"],
        "episode_index": episode_index,
        "cell_key": key,
        "descriptor": deepcopy(cell),
        "occupancy_before": _occupancy(archive, key),
        "selection_policy": policy["target_selection"],
        "controller_seed": policy["controller_seed"],
        "archive_sha256": hashlib.sha256(canonical_json(archive).encode()).hexdigest(),
        "descriptor_source": archive["descriptor_source"],
    }
    if policy["id"] in {
        POLICY_ID,
        ADAPTIVE_POLICY_ID,
        ROBUST_POLICY_ID,
        REFINED_POLICY_ID,
        HIGH_RES_POLICY_ID,
        STEERED_POLICY_ID,
        PREVIEW_POLICY_ID,
    }:
        definitions = (
            ROBUST_DEFINITIONS
            if policy["id"]
            in {
                ROBUST_POLICY_ID,
                REFINED_POLICY_ID,
                HIGH_RES_POLICY_ID,
                STEERED_POLICY_ID,
                PREVIEW_POLICY_ID,
            }
            else ATTESTED_DEFINITIONS
        )
        target["descriptor_definitions"] = {
            axis: definitions[axis][value] for axis, value in cell.items()
        }
        target["attestation_note"] = (
            "The protected verifier recomputes this cell from deterministic grid ablations "
            "of unseen stress renders against the independent pixel-only oracle."
        )
    if policy["id"] in {
        ADAPTIVE_POLICY_ID,
        ROBUST_POLICY_ID,
        REFINED_POLICY_ID,
        HIGH_RES_POLICY_ID,
        STEERED_POLICY_ID,
        PREVIEW_POLICY_ID,
    }:
        recipes = (
            ROBUST_STEERING_RECIPES
            if policy["id"]
            in {
                ROBUST_POLICY_ID,
                REFINED_POLICY_ID,
                HIGH_RES_POLICY_ID,
                STEERED_POLICY_ID,
                PREVIEW_POLICY_ID,
            }
            else STEERING_RECIPES
        )
        target["steering_recipes"] = {axis: recipes[axis][value] for axis, value in cell.items()}
        target["selection_score"] = round(float(score), 6)
        target["selection_score_components"] = score_components
        target["prior_target_history"] = history
        target["selection_note"] = (
            "The deterministic controller balances archive coverage, a smoothed estimate of "
            "target feasibility, uncertainty, recent misses, and a short cooldown."
        )
    if policy["id"] in {STEERED_POLICY_ID, PREVIEW_POLICY_ID}:
        target["cell_recipe"] = deepcopy(CELL_STEERING_RECIPES[key])
        occupied = [
            {"cell_key": occupied_key, "occupancy": len(elites)}
            for occupied_key, elites in archive["cells"].items()
            if occupied_key != key and elites
        ]
        target["occupied_cells_to_avoid"] = sorted(
            occupied,
            key=lambda item: (-item["occupancy"], item["cell_key"]),
        )
    if policy["id"] == PREVIEW_POLICY_ID:
        target["public_preview"] = {
            "attestation_id": HIGH_RES_ATTESTATION_ID,
            "minimum_nonempty_scene_fraction": 0.666667,
            "minimum_descriptor_stability": float(policy["minimum_descriptor_stability"]),
            "authority": "development_only_protected_unseen_stress_replay_remains_authoritative",
        }
    return target


def candidate_descriptor(candidate: dict, axes: tuple[str, ...]) -> dict:
    fingerprint = candidate.get("mechanism_fingerprint") or {}
    task_signature = candidate.get("task_signature") or {}
    descriptor: dict[str, Any] = {}
    for axis in axes:
        if axis == "decision_type":
            value = task_signature.get("decision_type") or candidate.get("decision_type")
        else:
            value = fingerprint.get(axis)
        if value is None:
            raise ValueError(f"candidate lacks quality-diversity descriptor axis {axis}")
        descriptor[axis] = value
    return descriptor


def _neighbor_scores(candidate: dict) -> dict[str, float]:
    neighbors = candidate.get("semantic_neighbors") or {}
    scores = {}
    for group in ("known", "siblings", "protected_source"):
        for item in neighbors.get(group) or []:
            score = item.get("score") if isinstance(item, dict) else None
            neighbor_id = (
                item.get("world_id") or item.get("candidate_id") if isinstance(item, dict) else None
            )
            if isinstance(score, int | float) and not isinstance(score, bool):
                if isinstance(neighbor_id, str) and neighbor_id:
                    scores[neighbor_id] = max(scores.get(neighbor_id, 0.0), float(score))
    return scores


def consider_candidate(
    policy: dict,
    archive: dict,
    *,
    candidate: dict,
    target: dict,
    protected_valid: bool,
) -> tuple[dict, dict]:
    """Return a new archive and an auditable, non-scientific archive decision."""

    validate_policy(policy)
    if policy["id"] in {
        POLICY_ID,
        ADAPTIVE_POLICY_ID,
        ROBUST_POLICY_ID,
        REFINED_POLICY_ID,
        HIGH_RES_POLICY_ID,
        STEERED_POLICY_ID,
        PREVIEW_POLICY_ID,
    }:
        return _consider_attested_candidate(
            policy,
            archive,
            candidate=candidate,
            target=target,
            protected_valid=protected_valid,
        )
    updated = deepcopy(archive)
    axes = tuple(policy["descriptor_axes"])
    neighbor_scores = _neighbor_scores(candidate)
    similarity = max(neighbor_scores.values(), default=0.0)
    descriptor = None
    key = None
    if protected_valid:
        descriptor = candidate_descriptor(candidate, axes)
        key = _cell_key(descriptor, axes)
    decision = {
        "schema_version": DECISION_SCHEMA,
        "policy_id": policy["id"],
        "candidate_id": candidate.get("candidate_id"),
        "candidate_sha256": candidate.get("candidate_sha256"),
        "protected_valid": bool(protected_valid),
        "descriptor": descriptor,
        "descriptor_source": "builder_declared_not_independently_validated",
        "cell_key": key,
        "target_cell_key": target["cell_key"],
        "target_match": key == target["cell_key"],
        "maximum_semantic_similarity": similarity,
        "global_similarity_ceiling": float(policy["global_similarity_ceiling"]),
        "within_cell_similarity_ceiling": float(policy["within_cell_similarity_ceiling"]),
        "within_cell_semantic_similarity": None,
        "within_cell_neighbor_coverage": None,
        "archive_admitted": False,
        "replaced_candidate_id": None,
        "reason": None,
    }
    updated["decisions"] += 1
    if not protected_valid:
        decision["reason"] = "protected_invalid"
        return updated, decision
    assert descriptor is not None and key is not None
    if key not in updated["cells"]:
        decision["reason"] = "outside_prespecified_feasible_mask"
        return updated, decision
    if similarity >= float(policy["global_similarity_ceiling"]):
        decision["reason"] = "global_semantic_similarity_ceiling"
        return updated, decision

    elites = updated["cells"][key]
    elite_ids = {item["candidate_id"] for item in elites}
    observed_local = {
        candidate_id: score
        for candidate_id, score in neighbor_scores.items()
        if candidate_id in elite_ids
    }
    decision["within_cell_semantic_similarity"] = (
        max(observed_local.values()) if observed_local else 0.0
    )
    decision["within_cell_neighbor_coverage"] = (
        len(observed_local) / len(elite_ids) if elite_ids else 1.0
    )
    if elite_ids and set(observed_local) != elite_ids:
        decision["reason"] = "incomplete_within_cell_neighbor_evidence"
        return updated, decision
    elite = {
        "candidate_id": candidate["candidate_id"],
        "candidate_sha256": candidate["candidate_sha256"],
        "artifact_path": candidate.get("artifact_path"),
        "maximum_semantic_similarity": similarity,
        "target_match": decision["target_match"],
    }
    if len(elites) < int(policy["max_elites_per_cell"]):
        if decision["within_cell_semantic_similarity"] >= float(
            policy["within_cell_similarity_ceiling"]
        ):
            decision["reason"] = "within_cell_similarity_ceiling"
            return updated, decision
        elites.append(elite)
        elites.sort(key=lambda item: (item["maximum_semantic_similarity"], item["candidate_id"]))
        updated["elite_admissions"] += 1
        decision["archive_admitted"] = True
        decision["reason"] = "open_elite_slot"
        return updated, decision

    worst = max(
        elites, key=lambda item: (item["maximum_semantic_similarity"], item["candidate_id"])
    )
    if similarity < float(worst["maximum_semantic_similarity"]):
        elites.remove(worst)
        elites.append(elite)
        elites.sort(key=lambda item: (item["maximum_semantic_similarity"], item["candidate_id"]))
        updated["elite_admissions"] += 1
        updated["elite_replacements"] += 1
        decision["archive_admitted"] = True
        decision["replaced_candidate_id"] = worst["candidate_id"]
        decision["reason"] = "lower_similarity_replacement"
    else:
        decision["reason"] = "cell_full_no_local_improvement"
    return updated, decision


def _validated_behavioral_vector(attestation: dict) -> list[float]:
    vector = attestation.get("behavioral_vector")
    expected = (
        149
        if attestation.get("attestation_id") == HIGH_RES_ATTESTATION_ID
        else 93
        if attestation.get("attestation_id") in {ROBUST_ATTESTATION_ID, REFINED_ATTESTATION_ID}
        else 52
        if attestation.get("attestation_id") == ADAPTIVE_ATTESTATION_ID
        else 39
    )
    if not isinstance(vector, list) or len(vector) != expected:
        raise ValueError(f"attested behavioral vector must contain exactly {expected} values")
    output = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("attested behavioral vector values must be numeric")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError("attested behavioral vector values must be finite and in [0, 1]")
        output.append(number)
    return output


def _behavioral_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("behavioral vectors are incompatible")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)) / len(left))


def _minimum_pairwise_distance(elites: list[dict]) -> float:
    if len(elites) < 2:
        return 1.0
    return min(
        _behavioral_distance(left["behavioral_vector"], right["behavioral_vector"])
        for index, left in enumerate(elites)
        for right in elites[index + 1 :]
    )


def _consider_attested_candidate(
    policy: dict,
    archive: dict,
    *,
    candidate: dict,
    target: dict,
    protected_valid: bool,
) -> tuple[dict, dict]:
    updated = deepcopy(archive)
    attestation = candidate.get("quality_diversity_attestation") or {}
    descriptor = attestation.get("descriptor") if attestation.get("status") == "complete" else None
    robust_policy = policy["id"] in {
        ROBUST_POLICY_ID,
        REFINED_POLICY_ID,
        HIGH_RES_POLICY_ID,
        STEERED_POLICY_ID,
        PREVIEW_POLICY_ID,
    }
    axes = ROBUST_AXES if robust_policy else ATTESTED_AXES
    values = ROBUST_VALUES if robust_policy else ATTESTED_VALUES
    key = _cell_key(descriptor, axes) if isinstance(descriptor, dict) else None
    neighbor_scores = _neighbor_scores(candidate)
    semantic_similarity = max(neighbor_scores.values(), default=0.0)
    decision = {
        "schema_version": (
            "quality-diversity-decision-0.8.0"
            if policy["id"] == PREVIEW_POLICY_ID
            else "quality-diversity-decision-0.7.0"
            if policy["id"] == STEERED_POLICY_ID
            else "quality-diversity-decision-0.6.0"
            if policy["id"] == HIGH_RES_POLICY_ID
            else "quality-diversity-decision-0.5.0"
            if policy["id"] == REFINED_POLICY_ID
            else "quality-diversity-decision-0.4.0"
            if policy["id"] == ROBUST_POLICY_ID
            else "quality-diversity-decision-0.3.0"
            if policy["id"] == ADAPTIVE_POLICY_ID
            else "quality-diversity-decision-0.2.0"
        ),
        "policy_id": policy["id"],
        "candidate_id": candidate.get("candidate_id"),
        "candidate_sha256": candidate.get("candidate_sha256"),
        "protected_valid": bool(protected_valid),
        "attestation_id": attestation.get("attestation_id"),
        "attestation_status": attestation.get("status", "missing"),
        "descriptor": descriptor,
        "descriptor_source": (
            "protected_scene_relative_pixel_oracle_ablation_v5"
            if policy["id"] in {HIGH_RES_POLICY_ID, STEERED_POLICY_ID, PREVIEW_POLICY_ID}
            else "protected_scene_relative_pixel_oracle_ablation_v4"
            if policy["id"] == REFINED_POLICY_ID
            else "protected_scene_relative_pixel_oracle_ablation_v3"
            if policy["id"] == ROBUST_POLICY_ID
            else "protected_scene_relative_pixel_oracle_ablation"
            if policy["id"] == ADAPTIVE_POLICY_ID
            else "protected_offline_pixel_oracle_ablation"
        ),
        "cell_key": key,
        "target_cell_key": target["cell_key"],
        "target_match": key == target["cell_key"],
        "maximum_semantic_similarity": semantic_similarity,
        "global_similarity_ceiling": float(policy["global_similarity_ceiling"]),
        "minimum_global_behavioral_distance": None,
        "minimum_within_cell_behavioral_distance": None,
        "archive_admitted": False,
        "replaced_candidate_id": None,
        "dispersion_before": None,
        "dispersion_after": None,
        "reason": None,
    }
    updated["decisions"] += 1
    target_stats = None
    if policy["id"] in {
        ADAPTIVE_POLICY_ID,
        ROBUST_POLICY_ID,
        REFINED_POLICY_ID,
        HIGH_RES_POLICY_ID,
        STEERED_POLICY_ID,
        PREVIEW_POLICY_ID,
    }:
        target_stats = updated["target_statistics"][target["cell_key"]]
        target_stats["selected_count"] += 1
        target_stats["completed_count"] += 1
        target_stats["last_selected_episode"] = int(target["episode_index"])
        target_stats["consecutive_misses"] += 1
    if not protected_valid:
        decision["reason"] = "protected_invalid"
        return updated, decision
    if (
        attestation.get("attestation_id") != policy["descriptor_attestation"]
        or attestation.get("status") != "complete"
        or not isinstance(descriptor, dict)
        or set(descriptor) != set(axes)
    ):
        decision["reason"] = "descriptor_attestation_incomplete"
        return updated, decision
    for axis, value in descriptor.items():
        if value not in values[axis]:
            decision["reason"] = "descriptor_attestation_invalid"
            return updated, decision
    if target_stats is not None:
        target_stats["observed_cell_counts"][key] = (
            int(target_stats["observed_cell_counts"].get(key, 0)) + 1
        )
        if decision["target_match"]:
            target_stats["target_match_count"] += 1
            target_stats["consecutive_misses"] = 0
    assert key is not None
    if key not in updated["cells"]:
        decision["reason"] = "outside_prespecified_feasible_mask"
        return updated, decision
    if semantic_similarity >= float(policy["global_similarity_ceiling"]):
        decision["reason"] = "global_semantic_similarity_ceiling"
        return updated, decision
    try:
        vector = _validated_behavioral_vector(attestation)
    except ValueError:
        decision["reason"] = "descriptor_attestation_invalid"
        return updated, decision
    support = attestation.get("support") or {}
    oracle_error_fraction = support.get("oracle_error_fraction")
    if (
        isinstance(oracle_error_fraction, bool)
        or not isinstance(oracle_error_fraction, int | float)
        or not 0.0 <= float(oracle_error_fraction) <= 1.0
    ):
        decision["reason"] = "descriptor_attestation_invalid"
        return updated, decision
    decision["oracle_ablation_error_fraction"] = float(oracle_error_fraction)
    if not robust_policy:
        decision["maximum_oracle_ablation_error_fraction"] = float(
            policy["maximum_oracle_ablation_error_fraction"]
        )
        if float(oracle_error_fraction) > float(policy["maximum_oracle_ablation_error_fraction"]):
            decision["reason"] = "oracle_ablation_fragility_ceiling"
            return updated, decision
    else:
        decision["oracle_ablation_error_interpretation"] = (
            "diagnostic_abstention_under_destructive_intervention"
        )
    if policy["id"] in {
        ADAPTIVE_POLICY_ID,
        ROBUST_POLICY_ID,
        REFINED_POLICY_ID,
        HIGH_RES_POLICY_ID,
        STEERED_POLICY_ID,
        PREVIEW_POLICY_ID,
    }:
        stability = attestation.get("descriptor_stability_fraction")
        if (
            isinstance(stability, bool)
            or not isinstance(stability, int | float)
            or not 0.0 <= float(stability) <= 1.0
        ):
            decision["reason"] = "descriptor_attestation_invalid"
            return updated, decision
        decision["descriptor_stability_fraction"] = float(stability)
        decision["minimum_descriptor_stability"] = float(policy["minimum_descriptor_stability"])
        if float(stability) < float(policy["minimum_descriptor_stability"]):
            decision["reason"] = "descriptor_instability_floor"
            return updated, decision

    all_elites = [elite for values in updated["cells"].values() for elite in values]
    global_distance = min(
        (_behavioral_distance(vector, elite["behavioral_vector"]) for elite in all_elites),
        default=1.0,
    )
    decision["minimum_global_behavioral_distance"] = round(global_distance, 6)
    if all_elites and global_distance < float(policy["minimum_global_behavioral_distance"]):
        decision["reason"] = "global_behavioral_distance_floor"
        return updated, decision

    elites = updated["cells"][key]
    local_distance = min(
        (_behavioral_distance(vector, elite["behavioral_vector"]) for elite in elites),
        default=1.0,
    )
    decision["minimum_within_cell_behavioral_distance"] = round(local_distance, 6)
    if elites and local_distance < float(policy["minimum_within_cell_behavioral_distance"]):
        decision["reason"] = "within_cell_behavioral_distance_floor"
        return updated, decision

    elite = {
        "candidate_id": candidate["candidate_id"],
        "candidate_sha256": candidate["candidate_sha256"],
        "artifact_path": candidate.get("artifact_path"),
        "descriptor": deepcopy(descriptor),
        "behavioral_vector": vector,
        "maximum_semantic_similarity": semantic_similarity,
        "target_match": decision["target_match"],
        "attestation_summary": {
            "scenes_analyzed": attestation.get("scenes_analyzed"),
            "support": deepcopy(attestation.get("support")),
            "visual_structure": deepcopy(attestation.get("visual_structure")),
            "descriptor_distribution": deepcopy(attestation.get("descriptor_distribution")),
            "descriptor_stability_fraction": attestation.get("descriptor_stability_fraction"),
            "oracle_ablation_error_fraction": oracle_error_fraction,
            "oracle_source_sha256": attestation.get("oracle_source_sha256"),
        },
    }
    if len(elites) < int(policy["max_elites_per_cell"]):
        elites.append(elite)
        elites.sort(key=lambda item: item["candidate_id"])
        updated["elite_admissions"] += 1
        if target_stats is not None:
            target_stats["elite_admission_count"] += 1
        decision["archive_admitted"] = True
        decision["reason"] = "open_elite_slot"
        decision["dispersion_after"] = round(_minimum_pairwise_distance(elites), 6)
        return updated, decision

    before = _minimum_pairwise_distance(elites)
    decision["dispersion_before"] = round(before, 6)
    alternatives = []
    for index, existing in enumerate(elites):
        proposed = [item for position, item in enumerate(elites) if position != index] + [elite]
        alternatives.append(
            (_minimum_pairwise_distance(proposed), existing["candidate_id"], proposed)
        )
    best_dispersion, replaced_id, proposed = max(alternatives, key=lambda item: (item[0], item[1]))
    if best_dispersion > before + 1e-12:
        elites[:] = sorted(proposed, key=lambda item: item["candidate_id"])
        updated["elite_admissions"] += 1
        updated["elite_replacements"] += 1
        if target_stats is not None:
            target_stats["elite_admission_count"] += 1
        decision["archive_admitted"] = True
        decision["replaced_candidate_id"] = replaced_id
        decision["dispersion_after"] = round(best_dispersion, 6)
        decision["reason"] = "maximin_dispersion_improved"
    else:
        decision["dispersion_after"] = round(before, 6)
        decision["reason"] = "cell_full_no_dispersion_improvement"
    return updated, decision


def archive_summary(archive: dict) -> dict:
    occupancies = {key: len(value) for key, value in archive["cells"].items()}
    occupied = sum(value > 0 for value in occupancies.values())
    summary = {
        "schema_version": (
            "quality-diversity-summary-0.8.0"
            if archive["policy_id"] == PREVIEW_POLICY_ID
            else "quality-diversity-summary-0.7.0"
            if archive["policy_id"] == STEERED_POLICY_ID
            else "quality-diversity-summary-0.6.0"
            if archive["policy_id"] == HIGH_RES_POLICY_ID
            else "quality-diversity-summary-0.5.0"
            if archive["policy_id"] == REFINED_POLICY_ID
            else "quality-diversity-summary-0.4.0"
            if archive["policy_id"] == ROBUST_POLICY_ID
            else "quality-diversity-summary-0.3.0"
            if archive["policy_id"] == ADAPTIVE_POLICY_ID
            else "quality-diversity-summary-0.2.0"
            if archive["policy_id"] == POLICY_ID
            else "quality-diversity-summary-0.1.0"
        ),
        "policy_id": archive["policy_id"],
        "feasible_cell_count": archive["feasible_cell_count"],
        "occupied_cell_count": occupied,
        "coverage_fraction": occupied / archive["feasible_cell_count"],
        "elite_count": sum(occupancies.values()),
        "decisions": archive["decisions"],
        "elite_admissions": archive["elite_admissions"],
        "elite_replacements": archive["elite_replacements"],
        "occupancy": occupancies,
    }
    if archive["policy_id"] in {
        POLICY_ID,
        ADAPTIVE_POLICY_ID,
        ROBUST_POLICY_ID,
        REFINED_POLICY_ID,
        HIGH_RES_POLICY_ID,
        STEERED_POLICY_ID,
        PREVIEW_POLICY_ID,
    }:
        summary["descriptor_axes"] = archive["descriptor_axes"]
        summary["descriptor_source"] = archive["descriptor_source"]
        summary["descriptor_attestation"] = archive["descriptor_attestation"]
        # State files are serialized with sorted JSON keys, while a live archive
        # retains the feasible-cell declaration order.  Keep the summary
        # canonical across both representations so the paper-run validator
        # compares scientific content rather than incidental mapping order.
        summary["cells"] = [
            {
                "cell_key": key,
                "occupancy": len(elites),
                "minimum_pairwise_behavioral_distance": (
                    round(_minimum_pairwise_distance(elites), 6) if len(elites) >= 2 else None
                ),
                "elite_candidate_ids": [elite["candidate_id"] for elite in elites],
            }
            for key, elites in sorted(archive["cells"].items())
        ]
    if archive["policy_id"] in {
        ADAPTIVE_POLICY_ID,
        ROBUST_POLICY_ID,
        REFINED_POLICY_ID,
        HIGH_RES_POLICY_ID,
        STEERED_POLICY_ID,
        PREVIEW_POLICY_ID,
    }:
        summary["target_statistics"] = deepcopy(archive["target_statistics"])
    return summary
