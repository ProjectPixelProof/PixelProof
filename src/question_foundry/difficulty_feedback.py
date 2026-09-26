"""Cost-bounded OpenRouter feedback for Section 3 difficulty-directed search.

Protected inverse-arm verification happens first. Only a mechanically valid,
transaction-complete candidate may reach this evaluator. The evaluator renders
fresh scenes without network access, chooses at most ten deterministic examples,
then stores append-only OpenRouter attempts using the existing VLM API evaluator.
The next builder receives sanitized structured predictions and outcomes. Policy
0.3 also exposes a bounded evaluator-authored visual rationale. Policy 0.4
additionally requires the builder to commit a concise, testable difficulty
hypothesis before evaluation so the next episode can compare intended failure
mechanisms with observed outcomes. Policy 0.5 adds a falsification-aware attempt
memory and a controlled evaluator-image-access ablation. Opaque provider
reasoning and raw payloads remain controller-side.
Policy 0.9 preserves malformed provider payloads before parsing and exposes only
bounded token-count and public-rationale-length effort proxies to the next builder.
Difficulty is a separate annotation and never changes scientific validity.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime as dt
import hashlib
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

from question_foundry.candidate_archive import (
    validate_generated_dataset,
    write_development_vlm_manifest,
)
from question_foundry.registry import canonical_json, sha256_tree
from question_foundry.render_runtime import (
    DEFAULT_RENDER_RUNTIME,
    LATEX_TIKZ_RENDER_RUNTIME,
    LATEX_TIKZ_REPLAY_IMAGE,
)
from question_foundry.vlm_eval.backends import (
    OPENROUTER_BASE_URL,
    OpenRouterBackend,
    ProviderPolicy,
    ProviderResponseError,
    fetch_openrouter_model_metadata,
    validate_live_model,
)
from question_foundry.vlm_eval.datasets import load_examples, verify_selected_images
from question_foundry.vlm_eval.models import IMAGE_DETAILS, REASONING_EFFORTS, ModelSpec
from question_foundry.vlm_eval.runner import execute_plan, full_offline_smoke, offline_preflight
from question_foundry.vlm_eval.sampling import select_chunk
from question_foundry.vlm_eval.store import (
    assert_secret_absent,
    create_plan,
    event_summary,
    initialize_run,
    read_events,
    write_new_json,
)

LEGACY_POLICY_ID = "difficulty-feedback@0.2.0"
RATIONALE_POLICY_ID = "difficulty-feedback@0.3.0"
POLICY_ID = "difficulty-feedback@0.4.0"
SYNTHESIS_POLICY_ID = "difficulty-feedback@0.5.0"
BRANCHING_POLICY_ID = "difficulty-feedback@0.6.0"
RESILIENT_BRANCHING_POLICY_ID = "difficulty-feedback@0.7.0"
WIRED_RESILIENT_BRANCHING_POLICY_ID = "difficulty-feedback@0.8.0"
EFFORT_AWARE_BRANCHING_POLICY_ID = "difficulty-feedback@0.9.0"
RESILIENT_POLICY_IDS = frozenset(
    {
        RESILIENT_BRANCHING_POLICY_ID,
        WIRED_RESILIENT_BRANCHING_POLICY_ID,
        EFFORT_AWARE_BRANCHING_POLICY_ID,
    }
)
BRANCHING_POLICY_IDS = frozenset(
    {
        BRANCHING_POLICY_ID,
        RESILIENT_BRANCHING_POLICY_ID,
        WIRED_RESILIENT_BRANCHING_POLICY_ID,
        EFFORT_AWARE_BRANCHING_POLICY_ID,
    }
)
POLICY_IDS = frozenset(
    {
        LEGACY_POLICY_ID,
        RATIONALE_POLICY_ID,
        POLICY_ID,
        SYNTHESIS_POLICY_ID,
        BRANCHING_POLICY_ID,
        RESILIENT_BRANCHING_POLICY_ID,
        WIRED_RESILIENT_BRANCHING_POLICY_ID,
        EFFORT_AWARE_BRANCHING_POLICY_ID,
    }
)
FEEDBACK_SCHEMAS = {
    LEGACY_POLICY_ID: "difficulty-feedback-result-0.2.0",
    RATIONALE_POLICY_ID: "difficulty-feedback-result-0.3.0",
    POLICY_ID: "difficulty-feedback-result-0.4.0",
    SYNTHESIS_POLICY_ID: "difficulty-feedback-result-0.5.0",
    BRANCHING_POLICY_ID: "difficulty-feedback-result-0.6.0",
    RESILIENT_BRANCHING_POLICY_ID: "difficulty-feedback-result-0.7.0",
    WIRED_RESILIENT_BRANCHING_POLICY_ID: "difficulty-feedback-result-0.8.0",
    EFFORT_AWARE_BRANCHING_POLICY_ID: "difficulty-feedback-result-0.9.0",
}
PUBLIC_FEEDBACK_VISIBILITIES = {
    LEGACY_POLICY_ID: "sample_predictions_and_outcomes",
    RATIONALE_POLICY_ID: "sample_predictions_outcomes_and_rationales",
    POLICY_ID: "difficulty_hypotheses_sample_predictions_outcomes_and_rationales",
    SYNTHESIS_POLICY_ID: "attempt_synthesis_predictions_outcomes_and_rationales",
    BRANCHING_POLICY_ID: "reasoning_branch_predictions_outcomes_and_rationales",
    RESILIENT_BRANCHING_POLICY_ID: "reasoning_branch_predictions_outcomes_and_rationales",
    WIRED_RESILIENT_BRANCHING_POLICY_ID: ("reasoning_branch_predictions_outcomes_and_rationales"),
    EFFORT_AWARE_BRANCHING_POLICY_ID: (
        "reasoning_branch_predictions_outcomes_rationales_and_effort"
    ),
}
RATIONALE_POLICY_IDS = frozenset(
    {
        RATIONALE_POLICY_ID,
        POLICY_ID,
        SYNTHESIS_POLICY_ID,
        BRANCHING_POLICY_ID,
        RESILIENT_BRANCHING_POLICY_ID,
        WIRED_RESILIENT_BRANCHING_POLICY_ID,
        EFFORT_AWARE_BRANCHING_POLICY_ID,
    }
)
HYPOTHESIS_POLICY_IDS = frozenset(
    {
        POLICY_ID,
        SYNTHESIS_POLICY_ID,
        BRANCHING_POLICY_ID,
        RESILIENT_BRANCHING_POLICY_ID,
        WIRED_RESILIENT_BRANCHING_POLICY_ID,
        EFFORT_AWARE_BRANCHING_POLICY_ID,
    }
)
SYNTHESIS_POLICY_IDS = frozenset(
    {
        SYNTHESIS_POLICY_ID,
        BRANCHING_POLICY_ID,
        RESILIENT_BRANCHING_POLICY_ID,
        WIRED_RESILIENT_BRANCHING_POLICY_ID,
        EFFORT_AWARE_BRANCHING_POLICY_ID,
    }
)
DIFFICULTY_HYPOTHESIS_SCHEMA_VERSION = "difficulty-hypothesis-0.1.0"
SYNTHESIS_HYPOTHESIS_SCHEMA_VERSION = "difficulty-hypothesis-0.2.0"
BRANCHING_HYPOTHESIS_SCHEMA_VERSION = "difficulty-hypothesis-0.3.0"
DIFFICULTY_HYPOTHESIS_KEYS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "prior_evidence_ids",
        "intended_visual_computation",
        "anticipated_model_failure",
        "causal_mutation_from_prior",
        "non_artifact_check",
        "predicted_evaluator_accuracy",
    }
)
SYNTHESIS_HYPOTHESIS_KEYS = frozenset(
    {
        "schema_version",
        "candidate_id",
        "parent_candidate_id",
        "prior_evidence_ids",
        "prior_attempt_synthesis",
        "evaluator_strategy_to_block",
        "intended_visual_computation",
        "anticipated_model_failure",
        "causal_mutation_from_prior",
        "non_artifact_check",
        "predicted_evaluator_accuracy",
    }
)
BRANCHING_HYPOTHESIS_KEYS = SYNTHESIS_HYPOTHESIS_KEYS | frozenset(
    {"reasoning_target", "reasoning_departure_from_archive"}
)
REASONING_TARGETS = frozenset(
    {
        "topological_connectivity",
        "correspondence_binding",
        "state_transition",
        "counterfactual_intervention",
        "geometric_transformation",
        "quantitative_relation",
    }
)
BRANCH_PARENT_SELECTION = "best_same_reasoning_target_or_initial_profile"
BUILDER_IMAGE_ACCESS = frozenset({"none", "evaluated_samples"})
RATIONALE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "visual_question_answer_with_rationale",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "One answer copied exactly from allowed_answers.",
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "rationale": {
                    "type": "string",
                    "description": (
                        "A concise description of the visible evidence and operations "
                        "used to select the answer; do not mention hidden labels."
                    ),
                    "minLength": 1,
                    "maxLength": 1200,
                },
            },
            "required": ["answer", "confidence", "rationale"],
            "additionalProperties": False,
        },
    },
}
DEFAULT_REPLAY_IMAGE = "question-foundry-candidate-replay:0.1"
REPLAY_IMAGES = {
    DEFAULT_RENDER_RUNTIME: DEFAULT_REPLAY_IMAGE,
    LATEX_TIKZ_RENDER_RUNTIME: LATEX_TIKZ_REPLAY_IMAGE,
}
AGGREGATIONS = frozenset(
    {
        "all_evaluators_below_threshold",
        "any_evaluator_below_threshold",
        "majority_evaluators_at_or_below_threshold",
        "mean_accuracy",
    }
)


def policy_schema_version(policy_id: str, family: str) -> str:
    versions = {
        LEGACY_POLICY_ID: "0.2.0",
        RATIONALE_POLICY_ID: "0.3.0",
        POLICY_ID: "0.4.0",
        SYNTHESIS_POLICY_ID: "0.5.0",
        BRANCHING_POLICY_ID: "0.6.0",
        RESILIENT_BRANCHING_POLICY_ID: "0.7.0",
        WIRED_RESILIENT_BRANCHING_POLICY_ID: "0.8.0",
        EFFORT_AWARE_BRANCHING_POLICY_ID: "0.9.0",
    }
    try:
        return f"difficulty-feedback-{family}-{versions[policy_id]}"
    except KeyError as exc:
        raise ValueError(f"unsupported difficulty-feedback policy: {policy_id}") from exc


def maximum_paid_requests(policy: dict, *, candidate_count: int | None = None) -> int:
    """Return the frozen worst-case request count, including bounded retries."""

    candidates = (
        int(policy["max_evaluated_candidates"]) if candidate_count is None else int(candidate_count)
    )
    attempts = 1 + len(policy.get("evaluator_retry_backoff_seconds", []))
    return (
        candidates
        * int(policy["max_examples_per_candidate"])
        * len(policy["evaluators"])
        * attempts
    )


def validate_difficulty_hypothesis(
    payload: dict,
    *,
    candidate_id: str,
    policy_id: str = POLICY_ID,
    expected_parent_candidate_id: str | None = None,
    expected_reasoning_target: str | None = None,
) -> dict:
    """Validate one public prospective difficulty claim without eliciting CoT."""

    synthesis = policy_id in SYNTHESIS_POLICY_IDS
    branching = policy_id in BRANCHING_POLICY_IDS
    expected_keys = (
        BRANCHING_HYPOTHESIS_KEYS
        if branching
        else SYNTHESIS_HYPOTHESIS_KEYS
        if synthesis
        else DIFFICULTY_HYPOTHESIS_KEYS
    )
    expected_schema = (
        BRANCHING_HYPOTHESIS_SCHEMA_VERSION
        if branching
        else SYNTHESIS_HYPOTHESIS_SCHEMA_VERSION
        if synthesis
        else DIFFICULTY_HYPOTHESIS_SCHEMA_VERSION
    )
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError(f"difficulty_hypothesis.json keys must be exactly {sorted(expected_keys)}")
    if payload.get("schema_version") != expected_schema:
        raise ValueError(f"difficulty_hypothesis.json schema_version must be {expected_schema}")
    if payload.get("candidate_id") != candidate_id:
        raise ValueError("difficulty hypothesis candidate_id does not match candidate")
    if synthesis:
        parent = payload.get("parent_candidate_id")
        if not isinstance(parent, str) or not parent.strip():
            raise ValueError("difficulty hypothesis requires parent_candidate_id")
        if expected_parent_candidate_id is not None and parent != expected_parent_candidate_id:
            if branching:
                raise ValueError(
                    "difficulty hypothesis parent_candidate_id must match the "
                    "controller-assigned lineage parent: "
                    f"{expected_parent_candidate_id}"
                )
            raise ValueError(
                "difficulty hypothesis parent_candidate_id must match the latest evaluated "
                f"candidate: {expected_parent_candidate_id}"
            )
    if branching:
        target = payload.get("reasoning_target")
        if target not in REASONING_TARGETS:
            raise ValueError(
                f"difficulty hypothesis reasoning_target must be one of {sorted(REASONING_TARGETS)}"
            )
        if expected_reasoning_target is not None and target != expected_reasoning_target:
            raise ValueError(
                "difficulty hypothesis reasoning_target must match the controller-assigned "
                f"target: {expected_reasoning_target}"
            )
        departure = payload.get("reasoning_departure_from_archive")
        if not isinstance(departure, str) or not 80 <= len(departure.strip()) <= 2000:
            raise ValueError(
                "difficulty hypothesis reasoning_departure_from_archive must contain "
                "80-2000 characters"
            )
    evidence = payload.get("prior_evidence_ids")
    if (
        not isinstance(evidence, list)
        or not 1 <= len(evidence) <= 5
        or len(evidence) != len(set(evidence))
        or not all(isinstance(item, str) and item.strip() for item in evidence)
    ):
        raise ValueError("difficulty hypothesis requires one to five prior evidence IDs")
    for field in (
        "intended_visual_computation",
        "anticipated_model_failure",
        "causal_mutation_from_prior",
        "non_artifact_check",
    ):
        value = payload.get(field)
        if not isinstance(value, str) or not 40 <= len(value.strip()) <= 1200:
            raise ValueError(f"difficulty hypothesis {field} must contain 40-1200 characters")
    if synthesis:
        for field in ("prior_attempt_synthesis", "evaluator_strategy_to_block"):
            value = payload.get(field)
            if not isinstance(value, str) or not 80 <= len(value.strip()) <= 2000:
                raise ValueError(f"difficulty hypothesis {field} must contain 80-2000 characters")
    predicted = payload.get("predicted_evaluator_accuracy")
    if (
        isinstance(predicted, bool)
        or not isinstance(predicted, int | float)
        or not 0.0 <= float(predicted) <= 1.0
    ):
        raise ValueError("predicted_evaluator_accuracy must be in [0, 1]")
    return copy.deepcopy(payload)


def load_difficulty_hypothesis(
    candidate_root: Path,
    *,
    candidate_id: str,
    policy_id: str = POLICY_ID,
    expected_parent_candidate_id: str | None = None,
    expected_reasoning_target: str | None = None,
) -> dict:
    path = candidate_root / "difficulty_hypothesis.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"difficulty_hypothesis.json is missing or invalid: {exc}") from exc
    return validate_difficulty_hypothesis(
        payload,
        candidate_id=candidate_id,
        policy_id=policy_id,
        expected_parent_candidate_id=expected_parent_candidate_id,
        expected_reasoning_target=expected_reasoning_target,
    )


def validate_policy(policy: dict) -> None:
    required = {
        "id",
        "render_scenes",
        "render_seed",
        "max_examples_per_candidate",
        "sampling_seed",
        "sampling_method",
        "sample_unit",
        "hard_accuracy_threshold",
        "aggregation",
        "max_evaluated_candidates",
        "max_total_cost_usd",
        "request_timeout_seconds",
        "fail_closed_on_incomplete",
        "feedback_visibility",
        "provider",
        "evaluators",
    }
    if policy.get("id") in SYNTHESIS_POLICY_IDS:
        required |= {
            "builder_image_access",
            "builder_attempt_memory_limit",
            "builder_samples_per_attempt",
        }
    if policy.get("id") in BRANCHING_POLICY_IDS:
        required |= {"reasoning_targets", "parent_selection"}
    if policy.get("id") in RESILIENT_POLICY_IDS:
        required |= {
            "evaluator_retry_backoff_seconds",
            "incomplete_episode_policy",
            "max_consecutive_incomplete_feedback",
        }
    if set(policy) != required:
        raise ValueError(f"feedback_policy keys must be exactly {sorted(required)}")
    if policy["id"] not in POLICY_IDS:
        raise ValueError(f"feedback_policy.id must be one of {sorted(POLICY_IDS)}")
    for field in ("render_scenes", "max_examples_per_candidate", "max_evaluated_candidates"):
        value = policy[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"feedback_policy.{field} must be a positive integer")
    if int(policy["render_scenes"]) > 32:
        raise ValueError("feedback_policy.render_scenes may not exceed 32")
    if int(policy["max_examples_per_candidate"]) > 10:
        raise ValueError("feedback_policy may send at most 10 examples per candidate")
    if int(policy["render_scenes"]) < int(policy["max_examples_per_candidate"]):
        raise ValueError("feedback_policy.render_scenes must cover every requested example")
    if isinstance(policy["render_seed"], bool) or not isinstance(policy["render_seed"], int):
        raise ValueError("feedback_policy.render_seed must be an integer")
    if isinstance(policy["sampling_seed"], bool) or not isinstance(policy["sampling_seed"], int):
        raise ValueError("feedback_policy.sampling_seed must be an integer")
    if policy["sampling_method"] not in {"uniform-example", "balanced-dataset"}:
        raise ValueError("feedback_policy.sampling_method is unsupported")
    if policy["sample_unit"] != "unique_image":
        raise ValueError("difficulty feedback requires sample_unit=unique_image")
    threshold = float(policy["hard_accuracy_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("hard_accuracy_threshold must be in [0, 1]")
    if policy["aggregation"] not in AGGREGATIONS:
        raise ValueError("feedback_policy.aggregation is unsupported")
    if isinstance(policy["max_total_cost_usd"], bool) or not isinstance(
        policy["max_total_cost_usd"], int | float
    ):
        raise ValueError("feedback_policy.max_total_cost_usd must be numeric")
    if float(policy["max_total_cost_usd"]) <= 0:
        raise ValueError("feedback_policy.max_total_cost_usd must be positive")
    timeout = policy["request_timeout_seconds"]
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int | float)
        or not 30 <= timeout <= 900
    ):
        raise ValueError("feedback request timeout must be in [30, 900] seconds")
    if policy["fail_closed_on_incomplete"] is not True:
        raise ValueError("difficulty feedback must fail closed on incomplete evaluation")
    expected_visibility = PUBLIC_FEEDBACK_VISIBILITIES[policy["id"]]
    if policy["feedback_visibility"] != expected_visibility:
        raise ValueError(f"feedback_visibility must be {expected_visibility!r}")
    if policy["id"] in SYNTHESIS_POLICY_IDS:
        if policy["builder_image_access"] not in BUILDER_IMAGE_ACCESS:
            raise ValueError(f"builder_image_access must be one of {sorted(BUILDER_IMAGE_ACCESS)}")
        memory_limit = policy["builder_attempt_memory_limit"]
        if (
            isinstance(memory_limit, bool)
            or not isinstance(memory_limit, int)
            or not 1 <= memory_limit <= 10
        ):
            raise ValueError("builder_attempt_memory_limit must be in [1, 10]")
        samples_per_attempt = policy["builder_samples_per_attempt"]
        if (
            isinstance(samples_per_attempt, bool)
            or not isinstance(samples_per_attempt, int)
            or not 1 <= samples_per_attempt <= 3
        ):
            raise ValueError("builder_samples_per_attempt must be in [1, 3]")
        if samples_per_attempt > int(policy["max_examples_per_candidate"]):
            raise ValueError("builder_samples_per_attempt exceeds evaluated examples")
    if policy["id"] in BRANCHING_POLICY_IDS:
        targets = policy["reasoning_targets"]
        if (
            not isinstance(targets, list)
            or not 3 <= len(targets) <= len(REASONING_TARGETS)
            or len(targets) != len(set(targets))
            or not all(target in REASONING_TARGETS for target in targets)
        ):
            raise ValueError(
                "feedback_policy.reasoning_targets must contain three to six unique "
                "supported targets"
            )
    if policy["id"] in RESILIENT_POLICY_IDS:
        delays = policy["evaluator_retry_backoff_seconds"]
        if (
            not isinstance(delays, list)
            or not 1 <= len(delays) <= 3
            or not all(
                isinstance(delay, int) and not isinstance(delay, bool) and 1 <= delay <= 300
                for delay in delays
            )
        ):
            raise ValueError(
                "feedback_policy.evaluator_retry_backoff_seconds must contain one to "
                "three integer delays in [1, 300]"
            )
        if policy["incomplete_episode_policy"] != "quarantine_feedback_and_continue":
            raise ValueError(
                f"{policy['id']} requires "
                "incomplete_episode_policy=quarantine_feedback_and_continue"
            )
        maximum = policy["max_consecutive_incomplete_feedback"]
        if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 5:
            raise ValueError(
                "feedback_policy.max_consecutive_incomplete_feedback must be in [1, 5]"
            )
        if policy["parent_selection"] != BRANCH_PARENT_SELECTION:
            raise ValueError(f"feedback_policy.parent_selection must be {BRANCH_PARENT_SELECTION}")

    provider = policy["provider"]
    provider_keys = {
        "base_url",
        "order",
        "allow_fallbacks",
        "data_collection",
        "zdr",
        "sort",
    }
    if not isinstance(provider, dict) or set(provider) != provider_keys:
        raise ValueError(f"feedback provider keys must be exactly {sorted(provider_keys)}")
    if provider["base_url"] != OPENROUTER_BASE_URL:
        raise ValueError(f"{policy['id']} requires the official OpenRouter base URL")
    if not isinstance(provider["order"], list) or any(
        not isinstance(item, str) or not item for item in provider["order"]
    ):
        raise ValueError("feedback provider order must be a string list")
    if len(provider["order"]) != len(set(provider["order"])):
        raise ValueError("feedback provider order must not contain duplicates")
    for field in ("allow_fallbacks", "zdr"):
        if not isinstance(provider[field], bool):
            raise ValueError(f"feedback provider {field} must be boolean")
    if provider["data_collection"] != "deny":
        raise ValueError("feedback provider must deny data collection")
    if provider["sort"] not in {"price", "latency", "throughput", "none"}:
        raise ValueError("feedback provider sort is unsupported")

    evaluators = policy["evaluators"]
    if not isinstance(evaluators, list) or not 1 <= len(evaluators) <= 4:
        raise ValueError("feedback_policy requires one to four evaluators")
    names = set()
    model_ids = set()
    for evaluator in evaluators:
        required_evaluator = {
            "name",
            "model_id",
            "reasoning_effort",
            "max_tokens",
            "image_detail",
        }
        if not isinstance(evaluator, dict) or set(evaluator) != required_evaluator:
            raise ValueError("feedback evaluator fields are invalid")
        if not isinstance(evaluator["name"], str) or not evaluator["name"]:
            raise ValueError("feedback evaluator name is invalid")
        if evaluator["name"] in names:
            raise ValueError("feedback evaluator names must be unique")
        names.add(evaluator["name"])
        if not isinstance(evaluator["model_id"], str) or "/" not in evaluator["model_id"]:
            raise ValueError("feedback evaluator model_id must be provider/model")
        if evaluator["model_id"] in model_ids:
            raise ValueError("feedback evaluator model IDs must be unique")
        model_ids.add(evaluator["model_id"])
        if evaluator["reasoning_effort"] not in REASONING_EFFORTS:
            raise ValueError("feedback evaluator reasoning effort is unsupported")
        if (
            isinstance(evaluator["max_tokens"], bool)
            or not isinstance(evaluator["max_tokens"], int)
            or not 64 <= evaluator["max_tokens"] <= 65536
        ):
            raise ValueError("feedback evaluator max_tokens is invalid")
        if evaluator["image_detail"] not in IMAGE_DETAILS:
            raise ValueError("feedback evaluator image detail is unsupported")


def initial_feedback_state(policy: dict) -> dict:
    validate_policy(policy)
    return {
        "schema_version": policy_schema_version(policy["id"], "state"),
        "policy_id": policy["id"],
        "evaluated_candidate_count": 0,
        "hard_seed": [],
        "valid_not_hard": [],
        "incomplete": [],
        "cumulative_cost_usd": 0.0,
        "candidate_feedback": [],
    }


def preflight_authorization(
    policy: dict,
    *,
    campaign_id: str,
    authorization_scope_id: str | None = None,
    live_model_metadata: list[dict] | None = None,
) -> dict:
    """Validate paid-run authority and the complete offline evaluator path."""

    validate_policy(policy)
    authorization_scope_id = authorization_scope_id or campaign_id
    if os.environ.get("OPENROUTER_FEEDBACK_RUN_CONFIRMED") != authorization_scope_id:
        raise ValueError(
            "OPENROUTER_FEEDBACK_RUN_CONFIRMED must equal the exact campaign or matrix ID"
        )
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set for difficulty feedback")
    smoke = full_offline_smoke()
    metadata_fetched = live_model_metadata is None
    if live_model_metadata is None:
        live_model_metadata = fetch_openrouter_model_metadata(
            api_key=api_key,
            base_url=policy["provider"]["base_url"],
            timeout_seconds=min(float(policy["request_timeout_seconds"]), 60.0),
        )
    selected_metadata = [
        validate_live_model(_model(evaluator), live_model_metadata)
        for evaluator in policy["evaluators"]
    ]
    return {
        "schema_version": "difficulty-feedback-preflight-0.1.0",
        "policy_id": policy["id"],
        "campaign_id": campaign_id,
        "authorization_scope_id": authorization_scope_id,
        "offline_smoke": smoke,
        "network_used": metadata_fetched,
        "inference_requests": 0,
        "maximum_paid_requests": maximum_paid_requests(policy),
        "evaluator_models": [item["model_id"] for item in policy["evaluators"]],
        "evaluator_model_metadata": selected_metadata,
    }


def public_feedback_history(state: dict) -> list[dict]:
    output = []
    for row in state.get("candidate_feedback", []):
        output.append(
            {
                "candidate_id": row["candidate_id"],
                "episode_index": row.get("episode_index"),
                "protected_valid": row["protected_valid"],
                "feedback_status": row["status"],
                "hard_seed_eligible": row.get("hard_seed_eligible"),
                "aggregate_accuracy": row.get("aggregate_accuracy"),
                "evaluator_accuracies": row.get("evaluator_accuracies", {}),
                "examples_per_evaluator": row.get("examples_per_evaluator"),
                "sample_responses": row.get("sample_responses", []),
                "difficulty_hypothesis": row.get("difficulty_hypothesis"),
                "feedback_policy": row["policy_id"],
            }
        )
    return output


def builder_feedback_history(
    state: dict,
    *,
    attempt_limit: int = 5,
    samples_per_attempt: int = 1,
) -> dict:
    """Build a bounded hard-first view while preserving the full controller ledger."""

    records = public_feedback_history(state)
    if state.get("policy_id") not in HYPOTHESIS_POLICY_IDS:
        return {"records": records, "hard_records": [], "easy_summaries": []}
    complete = [row for row in records if row.get("feedback_status") == "complete"]
    if state.get("policy_id") in SYNTHESIS_POLICY_IDS:
        # Keep the lowest-accuracy evidence and the newest causal lineage in one
        # bounded bank.  Unlike 0.4, solved attempts retain representative
        # evaluator rationales: those are the counterexamples that falsify the
        # builder's prospective difficulty claim.
        by_difficulty = sorted(
            complete,
            key=lambda row: (
                float(row.get("aggregate_accuracy", 1.0)),
                -int(row.get("episode_index") or 0),
                str(row.get("candidate_id")),
            ),
        )
        by_recency = sorted(
            complete,
            key=lambda row: (-int(row.get("episode_index") or 0), str(row["candidate_id"])),
        )
        selected = []
        difficulty_slots = min((attempt_limit + 1) // 2, len(by_difficulty))
        for row in [*by_difficulty[:difficulty_slots], *by_recency]:
            if any(item["candidate_id"] == row["candidate_id"] for item in selected):
                continue
            detached = copy.deepcopy(row)
            responses = detached.get("sample_responses", [])
            ranked = sorted(
                responses,
                key=lambda sample: (
                    not any(
                        response.get("status") == "scored" and response.get("correct") is False
                        for response in sample.get("responses", [])
                    ),
                    str(sample.get("sample_id")),
                ),
            )
            detached["sample_responses"] = ranked[:samples_per_attempt]
            selected.append(detached)
            if len(selected) == attempt_limit:
                break
        selected.sort(key=lambda row: int(row.get("episode_index") or 0))
        return {
            "records": selected,
            "hard_records": [row for row in selected if row.get("hard_seed_eligible") is True],
            "easy_summaries": [],
            "controller_record_count": len(records),
            "omitted_record_count": max(0, len(complete) - len(selected)),
            "latest_complete_candidate_id": (
                by_recency[0]["candidate_id"] if by_recency else "initial_profile"
            ),
        }
    hard = [row for row in complete if row.get("hard_seed_eligible") is True]
    hard.sort(
        key=lambda row: (
            float(row.get("aggregate_accuracy", 1.0)),
            -int(row.get("episode_index") or 0),
            str(row.get("candidate_id")),
        )
    )
    selected_hard = hard[:5]
    easy = [row for row in complete if row.get("hard_seed_eligible") is False]
    easy.sort(key=lambda row: -int(row.get("episode_index") or 0))
    easy_summaries = [
        {
            "candidate_id": row["candidate_id"],
            "episode_index": row.get("episode_index"),
            "aggregate_accuracy": row.get("aggregate_accuracy"),
            "evaluator_accuracies": row.get("evaluator_accuracies", {}),
            "difficulty_hypothesis": row.get("difficulty_hypothesis"),
        }
        for row in easy[:5]
    ]
    return {
        "records": selected_hard,
        "hard_records": selected_hard,
        "easy_summaries": easy_summaries,
        "controller_record_count": len(records),
        "omitted_easy_record_count": max(0, len(easy) - len(easy_summaries)),
    }


def reasoning_branch_assignment(policy: dict, state: dict, episode_index: int) -> dict:
    """Assign a reproducible reasoning branch and its local parent for policy 0.6+."""

    if policy.get("id") not in BRANCHING_POLICY_IDS:
        raise ValueError("reasoning branch assignment requires a branching feedback policy")
    targets = policy["reasoning_targets"]
    target = targets[(episode_index - 1) % len(targets)]
    matching = []
    for row in state.get("candidate_feedback", []):
        if row.get("status") != "complete":
            continue
        hypothesis = row.get("difficulty_hypothesis") or {}
        if hypothesis.get("reasoning_target") == target:
            matching.append(row)
    matching.sort(
        key=lambda row: (
            float(row.get("aggregate_accuracy", 1.0)),
            -int(row.get("episode_index") or 0),
            str(row.get("candidate_id")),
        )
    )
    return {
        "reasoning_target": target,
        "parent_candidate_id": matching[0]["candidate_id"] if matching else "initial_profile",
        "parent_selection": policy["parent_selection"],
        "target_cycle_index": (episode_index - 1) % len(targets),
    }


def _docker_image(render_runtime: str) -> str:
    try:
        return REPLAY_IMAGES[render_runtime]
    except KeyError as exc:
        raise ValueError(f"unsupported feedback render runtime: {render_runtime}") from exc


def _render_candidate(
    *,
    candidate_root: Path,
    dataset: Path,
    render_runtime: str,
    scenes: int,
    seed: int,
) -> dict:
    volume = dataset.parent
    volume.mkdir(parents=True, exist_ok=False)
    image = _docker_image(render_runtime)
    inspect = subprocess.run(
        ["docker", "inspect", "--type", "image", "--format", "{{.Id}}", image],
        check=True,
        capture_output=True,
        text=True,
    )
    image_id = inspect.stdout.strip()
    if not image_id.startswith("sha256:"):
        raise ValueError("feedback replay image has no immutable Docker ID")
    base = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--cpus",
        "2",
        "--memory",
        "4g",
        "--pids-limit",
        "256",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,size=512m",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--env",
        "HOME=/tmp",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "PYTHONPATH=/candidate/world",
        "--volume",
        f"{candidate_root.resolve()}:/candidate:ro",
        "--volume",
        f"{volume.resolve()}:/feedback:rw",
        "--workdir",
        "/candidate",
        image,
    ]
    commands = [
        [
            *base,
            "python",
            "-c",
            "import os,shutil,subprocess,sys; "
            "shutil.copytree('/candidate','/tmp/test-candidate'); "
            "os.environ['PYTHONPATH']='/tmp/test-candidate/world'; "
            "sys.exit(subprocess.call([sys.executable,'-m','pytest','-q','-p',"
            "'no:cacheprovider','/tmp/test-candidate/tests'],cwd='/tmp/test-candidate'))",
        ],
        [
            *base,
            "python",
            "/candidate/world/generate.py",
            "--out",
            "/feedback/dataset",
            "--n",
            str(scenes),
            "--seed",
            str(seed),
        ],
        [
            *base,
            "python",
            "/candidate/world/verify.py",
            "--dataset",
            "/feedback/dataset",
        ],
    ]
    for command in commands:
        subprocess.run(command, check=True)
    stats = validate_generated_dataset(dataset, expected_scenes=scenes)
    return {
        "network_mode": "none",
        "container_image": image,
        "container_image_id": image_id,
        "render_runtime": render_runtime,
        "scenes": scenes,
        "seed": seed,
        "statistics": stats,
    }


def _model(evaluator: dict) -> ModelSpec:
    return ModelSpec(
        evaluator["name"],
        evaluator["model_id"],
        evaluator["reasoning_effort"],
        evaluator["max_tokens"],
        evaluator["image_detail"],
    )


def _provider_policy(provider: dict) -> ProviderPolicy:
    return ProviderPolicy(
        order=tuple(provider["order"]),
        allow_fallbacks=bool(provider["allow_fallbacks"]),
        data_collection=provider["data_collection"],
        zdr=bool(provider["zdr"]),
        sort=None if provider["sort"] == "none" else provider["sort"],
    )


def _aggregate(policy: dict, accuracies: dict[str, float]) -> tuple[float, bool]:
    values = list(accuracies.values())
    mean = sum(values) / len(values)
    threshold = float(policy["hard_accuracy_threshold"])
    if policy["aggregation"] == "all_evaluators_below_threshold":
        hard = all(value < threshold for value in values)
    elif policy["aggregation"] == "any_evaluator_below_threshold":
        hard = any(value < threshold for value in values)
    elif policy["aggregation"] == "majority_evaluators_at_or_below_threshold":
        # Section 3 defines task-level frontier difficulty using a strict
        # majority of the frozen evaluator panel.  With five samples and a
        # threshold of 0.6, a model must miss at least two examples.  The
        # inclusive comparison is intentional and matches the frozen seed-pool
        # eligibility rule rather than silently tightening it to three misses.
        hard = sum(value <= threshold for value in values) > len(values) / 2
    else:
        hard = mean < threshold
    return mean, hard


def _one_example_per_image(examples: list, *, seed: int) -> list:
    """Choose one deterministic prompt row for every distinct rendered image."""

    ranked = sorted(
        examples,
        key=lambda item: hashlib.sha256(f"{seed}\0{item.sample_key}".encode()).hexdigest(),
    )
    unique = {}
    for example in ranked:
        unique.setdefault(example.image_path.resolve(), example)
    return list(unique.values())


def _cost_record_count(events: list[dict]) -> int:
    latest = {}
    for event in events:
        if event.get("status") in {"scored", "unparseable", "refused"}:
            latest[event.get("sample_key")] = event
    return sum(
        isinstance((event.get("usage") or {}).get("cost"), int | float)
        and not isinstance((event.get("usage") or {}).get("cost"), bool)
        for event in latest.values()
    )


def _latest_terminal_events(events: list[dict]) -> dict[str, dict]:
    """Return one terminal response per sample without exposing provider payloads."""

    terminal = {}
    for event in events:
        if event.get("status") in {"scored", "unparseable", "refused"}:
            terminal[event["sample_key"]] = event
    return terminal


class _RationaleFeedbackBackend:
    """Request a concise public rationale while keeping provider internals private.

    OpenRouter's internal reasoning stream is provider-specific and is not a stable
    experimental artifact.  Policy 0.3 instead asks the evaluator to emit a bounded
    rationale in its final structured answer.  The generic scorer still receives its
    historical answer/confidence payload, while the rationale is carried separately
    into the append-only event and the next builder packet.
    """

    def __init__(self, backend: OpenRouterBackend) -> None:
        self._backend = backend

    def complete(self, *, model, messages, candidates, provider_policy):
        rationale_messages = copy.deepcopy(messages)
        text_part = rationale_messages[0]["content"][0]
        text_part["text"] += (
            "\nAlso return a concise rationale describing the visible evidence and the "
            "operations you performed. This rationale will be shown to a question-design "
            "agent so it can understand why the task was easy or hard. Do not mention or "
            "guess any hidden gold label."
        )
        response = self._backend.complete(
            model=model,
            messages=rationale_messages,
            candidates=candidates,
            provider_policy=provider_policy,
        )
        try:
            payload = json.loads(response.content)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError(
                "rationale response content is not JSON",
                response=response,
            ) from exc
        if not isinstance(payload, dict) or set(payload) != {
            "answer",
            "confidence",
            "rationale",
        }:
            raise ProviderResponseError(
                "rationale response must contain answer, confidence, and rationale",
                response=response,
            )
        rationale = payload["rationale"]
        if not isinstance(rationale, str) or not rationale.strip():
            raise ProviderResponseError(
                "rationale response must contain a non-empty rationale",
                response=response,
            )
        rationale = rationale.strip()[:1200]
        scorer_content = json.dumps(
            {"answer": payload["answer"], "confidence": payload["confidence"]},
            separators=(",", ":"),
        )
        return dataclasses.replace(response, content=scorer_content, rationale=rationale)


def _public_sample_responses(
    selected: list,
    evaluator_events: dict[str, list[dict]],
    evaluator_specs: dict[str, dict],
    *,
    include_rationales: bool,
    include_oracle_answers: bool = False,
) -> list[dict]:
    """Build feedback visible to the next builder.

    Policy 0.3 includes a concise evaluator-authored rationale. Policy 0.9 also
    exposes bounded effort measurements (not opaque provider reasoning) so the
    next designer can distinguish a quick correct answer from one that consumed
    substantial reasoning budget. Gold labels, costs, response IDs, raw provider
    payloads, and opaque reasoning content remain controller-side evidence.
    """

    latest_by_evaluator = {
        name: _latest_terminal_events(events) for name, events in evaluator_events.items()
    }
    output = []
    for index, example in enumerate(selected, 1):
        responses = []
        for evaluator_name, evaluator_spec in evaluator_specs.items():
            model_id = evaluator_spec["model_id"]
            event = latest_by_evaluator.get(evaluator_name, {}).get(example.sample_key)
            if event is None:
                responses.append(
                    {
                        "evaluator": evaluator_name,
                        "model_id": model_id,
                        "status": "missing",
                    }
                )
                continue
            response = {
                "evaluator": evaluator_name,
                "model_id": model_id,
                "served_model": event.get("served_model"),
                "provider": event.get("provider"),
                "status": event["status"],
            }
            if event["status"] == "scored":
                response.update(
                    {
                        "prediction": event.get("prediction"),
                        "confidence": event.get("confidence"),
                        "correct": bool(event.get("correct")),
                    }
                )
                if include_rationales:
                    rationale = event.get("rationale")
                    if not isinstance(rationale, str) or not rationale.strip():
                        raise ValueError("complete rationale feedback is missing a rationale")
                    response["rationale"] = rationale.strip()[:1200]
                if evaluator_spec.get("expose_effort_signal"):
                    usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
                    completion_value = usage.get("completion_tokens")
                    completion_tokens = (
                        int(completion_value)
                        if isinstance(completion_value, int | float)
                        and not isinstance(completion_value, bool)
                        else None
                    )
                    details = usage.get("completion_tokens_details")
                    reasoning_tokens = usage.get("reasoning_tokens")
                    if not isinstance(reasoning_tokens, int | float):
                        reasoning_tokens = (
                            details.get("reasoning_tokens") if isinstance(details, dict) else None
                        )
                    reasoning_tokens = (
                        int(reasoning_tokens)
                        if isinstance(reasoning_tokens, int | float)
                        and not isinstance(reasoning_tokens, bool)
                        else None
                    )
                    max_tokens = int(evaluator_spec["max_tokens"])
                    public_rationale = response["rationale"]
                    response["solver_effort"] = {
                        "reasoning_effort_setting": evaluator_spec["reasoning_effort"],
                        "completion_tokens": completion_tokens,
                        "reasoning_tokens": reasoning_tokens,
                        "visible_output_tokens": (
                            max(completion_tokens - reasoning_tokens, 0)
                            if completion_tokens is not None and reasoning_tokens is not None
                            else None
                        ),
                        "completion_budget_tokens": max_tokens,
                        "completion_budget_fraction": (
                            round(completion_tokens / max_tokens, 6)
                            if completion_tokens is not None and max_tokens
                            else None
                        ),
                        "reasoning_fraction_of_completion": (
                            round(reasoning_tokens / completion_tokens, 6)
                            if reasoning_tokens is not None and completion_tokens
                            else None
                        ),
                        "rationale_characters": len(public_rationale),
                        "rationale_words": len(public_rationale.split()),
                    }
            elif event["status"] == "refused":
                response["refusal"] = str(event.get("refusal") or "")[:1000]
            else:
                response["response"] = str(event.get("response_content") or "")[:1000]
                response["parse_error"] = str(event.get("parse_error") or "")[:1000]
            responses.append(response)
        sample = {
            "sample_id": f"sample-{index:02d}",
            "question": example.question,
            "candidates": list(example.candidates),
            "responses": responses,
        }
        if include_oracle_answers:
            sample["oracle_answer"] = example.answer
        output.append(sample)
    return output


def evaluate_candidate(
    *,
    policy: dict,
    candidate_root: Path,
    output_root: Path,
    campaign_id: str,
    candidate_id: str,
    candidate_sha256: str,
    episode_index: int,
    render_runtime: str,
    authorization_scope_id: str | None = None,
    backend_factory: Callable[[ModelSpec], object] | None = None,
    live_model_metadata: list[dict] | None = None,
    expected_parent_candidate_id: str | None = None,
    expected_reasoning_target: str | None = None,
) -> dict:
    """Render, evaluate, and preserve one valid candidate without hiding failures."""

    validate_policy(policy)
    difficulty_hypothesis = (
        load_difficulty_hypothesis(
            candidate_root,
            candidate_id=candidate_id,
            policy_id=policy["id"],
            expected_parent_candidate_id=expected_parent_candidate_id,
            expected_reasoning_target=expected_reasoning_target,
        )
        if policy["id"] in HYPOTHESIS_POLICY_IDS
        else None
    )
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    dataset = output_root / "rendered/dataset"
    render = _render_candidate(
        candidate_root=candidate_root,
        dataset=dataset,
        render_runtime=render_runtime,
        scenes=int(policy["render_scenes"]),
        seed=int(policy["render_seed"]) + episode_index,
    )
    record_id = f"{campaign_id}:{candidate_id}:feedback"
    write_development_vlm_manifest(
        source_manifest=dataset / "manifest.jsonl",
        destination=dataset / "vlm_manifest.jsonl",
        record={
            "record_id": record_id,
            "campaign_id": campaign_id,
            "candidate_id": candidate_id,
            "human_admission": "deferred",
        },
    )
    examples, snapshots = load_examples([dataset / "vlm_manifest.jsonl"])
    unique_examples = _one_example_per_image(
        examples,
        seed=int(policy["sampling_seed"]) + episode_index,
    )
    selected = select_chunk(
        unique_examples,
        excluded_keys=set(),
        sample_size=int(policy["max_examples_per_candidate"]),
        seed=int(policy["sampling_seed"]) + episode_index,
        method=policy["sampling_method"],
    )
    if len(selected) != int(policy["max_examples_per_candidate"]):
        raise ValueError(
            "candidate did not yield the exact requested number of distinct eligible images"
        )
    hashes = verify_selected_images(selected)
    sample_set = {
        "schema_version": "difficulty-feedback-sample-set-0.2.0",
        "candidate_id": candidate_id,
        "candidate_sha256": candidate_sha256,
        "sampling_seed": int(policy["sampling_seed"]) + episode_index,
        "sampling_method": policy["sampling_method"],
        "sample_unit": policy["sample_unit"],
        "distinct_image_count": len({example.image_path.resolve() for example in selected}),
        "samples": [
            {**example.public_metadata(), "image_sha256": hashes[example.sample_key]}
            for example in selected
        ],
    }
    (output_root / "sample-set.json").write_text(
        canonical_json(sample_set) + "\n", encoding="utf-8"
    )
    framework_smoke = full_offline_smoke()
    provider_policy = _provider_policy(policy["provider"])
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    paid = backend_factory is None
    authorization_scope_id = authorization_scope_id or campaign_id
    if paid:
        if os.environ.get("OPENROUTER_FEEDBACK_RUN_CONFIRMED") != authorization_scope_id:
            raise ValueError(
                "OPENROUTER_FEEDBACK_RUN_CONFIRMED must equal the exact campaign or matrix ID"
            )
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set for difficulty feedback")
        if live_model_metadata is None:
            live_model_metadata = fetch_openrouter_model_metadata(
                api_key=api_key,
                base_url=policy["provider"]["base_url"],
                timeout_seconds=min(float(policy["request_timeout_seconds"]), 60.0),
            )
    elif live_model_metadata is None:
        live_model_metadata = []

    evaluator_results = []
    evaluator_events = {}
    for evaluator in policy["evaluators"]:
        model = _model(evaluator)
        evaluator_root = output_root / "evaluators" / evaluator["name"]
        metadata = None
        failure = None
        try:
            initialize_run(
                evaluator_root,
                {
                    "model": model.as_dict(),
                    "prompt_version": "closed-choice-image@0.1.0",
                    "sampling_method": policy["sampling_method"],
                    "provider_policy": provider_policy.as_dict(),
                    "transport": {
                        "backend": "openrouter-openai-chat-completions",
                        "base_url": policy["provider"]["base_url"],
                        "timeout_seconds": policy["request_timeout_seconds"],
                    },
                    "datasets": [snapshot.as_dict() for snapshot in snapshots],
                },
            )
            offline_preflight(
                examples=selected,
                hashes=hashes,
                model=model,
                provider_policy=provider_policy,
            )
            if paid:
                metadata = validate_live_model(model, live_model_metadata)
                backend = OpenRouterBackend(
                    api_key=api_key,
                    timeout_seconds=float(policy["request_timeout_seconds"]),
                    base_url=policy["provider"]["base_url"],
                    title="visual-question-discovery difficulty feedback",
                    response_format=(
                        RATIONALE_RESPONSE_FORMAT if policy["id"] in RATIONALE_POLICY_IDS else None
                    ),
                )
                if policy["id"] in RATIONALE_POLICY_IDS:
                    backend = _RationaleFeedbackBackend(backend)
            else:
                backend = backend_factory(model)
                if policy["id"] in RATIONALE_POLICY_IDS:
                    backend = _RationaleFeedbackBackend(backend)
            _plan_path, plan = create_plan(
                evaluator_root,
                samples=[
                    {**example.public_metadata(), "image_sha256": hashes[example.sample_key]}
                    for example in selected
                ],
                seed=int(policy["sampling_seed"]) + episode_index,
                sampling_method=policy["sampling_method"],
            )
            retry_delays = tuple(policy.get("evaluator_retry_backoff_seconds", []))
            retry_count = 0
            while True:
                summary = execute_plan(
                    run_dir=evaluator_root,
                    plan=plan,
                    examples=selected,
                    hashes=hashes,
                    model=model,
                    provider_policy=provider_policy,
                    backend=backend,
                    retry_errors=retry_count > 0,
                    continue_on_api_error=False,
                )
                retryable = sum(
                    int(summary["latest_status_counts"].get(status, 0))
                    for status in ("attempt_started", "api_error")
                )
                if retryable == 0 or retry_count >= len(retry_delays):
                    break
                delay = retry_delays[retry_count]
                retry_count += 1
                write_new_json(
                    evaluator_root / "retries" / f"retry-{retry_count:03d}.json",
                    {
                        "schema_version": "difficulty-feedback-evaluator-retry-0.1.0",
                        "retry_index": retry_count,
                        "delay_seconds": delay,
                        "retryable_sample_count": retryable,
                        "reason": "provider_api_error",
                        "observed_at": dt.datetime.now(dt.UTC).isoformat(),
                    },
                )
                time.sleep(delay)
        except Exception as exc:
            message = str(exc).replace(api_key, "[REDACTED]") if api_key else str(exc)
            failure = {"error_type": type(exc).__name__, "error": message[:4000]}
            evaluator_root.mkdir(parents=True, exist_ok=True)
            (evaluator_root / "evaluator-failure.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            summary = event_summary(read_events(evaluator_root))
        finally:
            if api_key and evaluator_root.exists():
                assert_secret_absent(evaluator_root, api_key)
        events = read_events(evaluator_root)
        evaluator_events[evaluator["name"]] = events
        statuses = summary["latest_status_counts"]
        cost_record_count = _cost_record_count(events)
        all_samples_scored = (
            summary["attempted_samples"] == len(selected)
            and summary["scored_samples"] == len(selected)
            and set(statuses) == {"scored"}
        )
        if paid and failure is None and all_samples_scored and cost_record_count != len(selected):
            failure = {
                "error_type": "ProviderCostUnavailable",
                "error": (
                    "one or more paid OpenRouter attempts lack provider-native cost; "
                    "the feedback cost cap cannot be enforced"
                ),
            }
            (evaluator_root / "evaluator-failure.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        complete = failure is None and all_samples_scored
        if not complete and failure is None:
            failure = {
                "error_type": "IncompleteEvaluatorBatch",
                "error": (
                    "evaluator batch remained incomplete after bounded retries: "
                    f"{dict(sorted(statuses.items()))}"
                ),
            }
            (evaluator_root / "evaluator-failure.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        evaluator_results.append(
            {
                "name": evaluator["name"],
                "model": model.as_dict(),
                "metadata": metadata,
                "complete": complete,
                "summary": summary,
                "event_count": len(events),
                "provider_cost_record_count": cost_record_count,
                "provider_cost_complete": not paid or cost_record_count == len(selected),
                "retry_count": len(list((evaluator_root / "retries").glob("retry-*.json"))),
                "events_sha256": (
                    sha256_tree(evaluator_root / "events")
                    if (evaluator_root / "events").is_dir()
                    else None
                ),
                "failure": failure,
            }
        )
        if failure is not None:
            break

    complete = len(evaluator_results) == len(policy["evaluators"]) and all(
        item["complete"] for item in evaluator_results
    )
    accuracies = {
        item["name"]: float(item["summary"]["accuracy"])
        for item in evaluator_results
        if item["summary"]["accuracy"] is not None
    }
    aggregate_accuracy = None
    hard = False
    if complete and len(accuracies) == len(evaluator_results):
        aggregate_accuracy, hard = _aggregate(policy, accuracies)
    total_cost = sum(
        float(item["summary"]["usage_totals"]["cost"] or 0.0) for item in evaluator_results
    )
    sample_responses = _public_sample_responses(
        selected,
        evaluator_events,
        {
            item["name"]: {
                **item,
                "expose_effort_signal": policy["id"] == EFFORT_AWARE_BRANCHING_POLICY_ID,
            }
            for item in policy["evaluators"]
        },
        include_rationales=policy["id"] in RATIONALE_POLICY_IDS,
        include_oracle_answers=policy["id"] in SYNTHESIS_POLICY_IDS,
    )
    result = {
        "schema_version": FEEDBACK_SCHEMAS[policy["id"]],
        "policy_id": policy["id"],
        "status": "complete" if complete else "incomplete",
        "campaign_id": campaign_id,
        "authorization_scope_id": authorization_scope_id,
        "episode_index": episode_index,
        "candidate_id": candidate_id,
        "candidate_sha256": candidate_sha256,
        "protected_valid": True,
        "render": render,
        "sample_set_sha256": hashlib.sha256(canonical_json(sample_set).encode()).hexdigest(),
        "examples_per_evaluator": len(selected),
        "maximum_paid_requests": maximum_paid_requests(policy, candidate_count=1),
        "sample_responses": sample_responses,
        "evaluator_accuracies": accuracies,
        "aggregate_accuracy": aggregate_accuracy,
        "aggregation": policy["aggregation"],
        "hard_accuracy_threshold": policy["hard_accuracy_threshold"],
        "hard_seed_eligible": hard if complete else None,
        "difficulty_hypothesis": difficulty_hypothesis,
        "total_cost_usd": round(total_cost, 9),
        "evaluators": evaluator_results,
        "framework_offline_smoke": framework_smoke,
        # Authorization is not evidence that an inference request reached the
        # backend.  A missing optional SDK (or another setup failure) can occur
        # before the first append-only attempt event.  Report calls only when
        # the evaluator ledger contains an actual attempted sample.
        "provider_calls_made": paid
        and any(item["summary"]["attempted_samples"] > 0 for item in evaluator_results),
        "feedback_visibility": policy["feedback_visibility"],
        "evaluated_at": dt.datetime.now(dt.UTC).isoformat(),
    }
    (output_root / "feedback-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if api_key:
        assert_secret_absent(output_root, api_key)
    return result


def update_feedback_state(state: dict, result: dict) -> dict:
    updated = json.loads(json.dumps(state))
    updated["evaluated_candidate_count"] += 1
    updated["cumulative_cost_usd"] = round(
        float(updated["cumulative_cost_usd"]) + float(result.get("total_cost_usd") or 0.0),
        9,
    )
    compact = {
        key: result.get(key)
        for key in (
            "schema_version",
            "policy_id",
            "status",
            "campaign_id",
            "episode_index",
            "candidate_id",
            "candidate_sha256",
            "protected_valid",
            "sample_set_sha256",
            "examples_per_evaluator",
            "maximum_paid_requests",
            "sample_responses",
            "evaluator_accuracies",
            "aggregate_accuracy",
            "aggregation",
            "hard_accuracy_threshold",
            "hard_seed_eligible",
            "difficulty_hypothesis",
            "total_cost_usd",
            "artifact_path",
            "evaluated_at",
            "bootstrap_source",
        )
    }
    updated["candidate_feedback"].append(compact)
    if result["status"] != "complete":
        updated["incomplete"].append(result["candidate_id"])
    elif result["hard_seed_eligible"]:
        updated["hard_seed"].append(result["candidate_id"])
    else:
        updated["valid_not_hard"].append(result["candidate_id"])
    return updated


def fresh_feedback_candidate_count(state: dict) -> int:
    """Count candidates evaluated by this run, excluding frozen bootstrap evidence."""

    return sum(1 for row in state.get("candidate_feedback", []) if not row.get("bootstrap_source"))


def feedback_reconciliation(state: dict) -> dict:
    rows = state.get("candidate_feedback", [])
    run_rows = [row for row in rows if not row.get("bootstrap_source")]
    bootstrap_rows = [row for row in rows if row.get("bootstrap_source")]
    cost = round(sum(float(row.get("total_cost_usd") or 0.0) for row in run_rows), 9)
    bootstrap_cost = round(
        sum(float(row.get("total_cost_usd") or 0.0) for row in bootstrap_rows), 9
    )
    incomplete_is_quarantined = state.get("policy_id") in RESILIENT_POLICY_IDS
    return {
        "schema_version": policy_schema_version(state["policy_id"], "usage-reconciliation"),
        "policy_id": state["policy_id"],
        "evaluated_candidate_count": len(rows),
        "fresh_evaluated_candidate_count": len(run_rows),
        "bootstrap_candidate_count": len(bootstrap_rows),
        "hard_seed_count": len(state.get("hard_seed", [])),
        "valid_not_hard_count": len(state.get("valid_not_hard", [])),
        "incomplete_count": len(state.get("incomplete", [])),
        "incomplete_disposition": (
            "quarantined_and_excluded_from_difficulty_statistics"
            if incomplete_is_quarantined
            else "run_incomplete"
        ),
        "cost_usd": cost,
        "bootstrap_source_cost_usd": bootstrap_cost,
        "state_matches": {
            "candidate_count": state.get("evaluated_candidate_count") == len(rows),
            "cost_usd": abs(float(state.get("cumulative_cost_usd") or 0.0) - cost) < 1e-8,
        },
        "passed": (
            state.get("evaluated_candidate_count") == len(rows)
            and abs(float(state.get("cumulative_cost_usd") or 0.0) - cost) < 1e-8
            and (incomplete_is_quarantined or not state.get("incomplete"))
        ),
    }


def feedback_summary(state: dict) -> dict:
    """Summarize difficulty outcomes without replacing append-only evidence."""

    model_counts = defaultdict(Counter)
    candidate_accuracies = []
    for row in state.get("candidate_feedback", []):
        accuracy = row.get("aggregate_accuracy")
        if isinstance(accuracy, int | float) and not isinstance(accuracy, bool):
            candidate_accuracies.append(float(accuracy))
        for sample in row.get("sample_responses", []):
            for response in sample.get("responses", []):
                name = response.get("evaluator")
                status = response.get("status")
                if not isinstance(name, str) or not isinstance(status, str):
                    continue
                model_counts[name][status] += 1
                if status == "scored" and response.get("correct") is True:
                    model_counts[name]["correct"] += 1
    evaluators = {}
    for name, counts in sorted(model_counts.items()):
        scored = counts["scored"]
        evaluators[name] = {
            "scored_samples": scored,
            "correct_samples": counts["correct"],
            "accuracy": counts["correct"] / scored if scored else None,
            "status_counts": {
                status: count for status, count in sorted(counts.items()) if status != "correct"
            },
        }
    return {
        "schema_version": policy_schema_version(state["policy_id"], "summary"),
        "policy_id": state["policy_id"],
        "evaluated_candidate_count": int(state.get("evaluated_candidate_count") or 0),
        "fresh_evaluated_candidate_count": fresh_feedback_candidate_count(state),
        "bootstrap_candidate_count": sum(
            1 for row in state.get("candidate_feedback", []) if row.get("bootstrap_source")
        ),
        "hard_seed_count": len(state.get("hard_seed", [])),
        "valid_not_hard_count": len(state.get("valid_not_hard", [])),
        "incomplete_count": len(state.get("incomplete", [])),
        "mean_candidate_accuracy": (
            sum(candidate_accuracies) / len(candidate_accuracies) if candidate_accuracies else None
        ),
        "cumulative_cost_usd": float(state.get("cumulative_cost_usd") or 0.0),
        "evaluators": evaluators,
    }
