from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from question_foundry.difficulty_feedback import (
    RATIONALE_RESPONSE_FORMAT,
    _aggregate,
    _cost_record_count,
    _RationaleFeedbackBackend,
    builder_feedback_history,
    evaluate_candidate,
    feedback_reconciliation,
    feedback_summary,
    initial_feedback_state,
    preflight_authorization,
    public_feedback_history,
    reasoning_branch_assignment,
    update_feedback_state,
    validate_difficulty_hypothesis,
    validate_policy,
)
from question_foundry.vlm_eval.backends import BackendResponse, FakeBackend


def _policy() -> dict:
    return {
        "id": "difficulty-feedback@0.2.0",
        "render_scenes": 5,
        "render_seed": 42000,
        "max_examples_per_candidate": 5,
        "sampling_seed": 12027,
        "sampling_method": "balanced-dataset",
        "sample_unit": "unique_image",
        "hard_accuracy_threshold": 1.0,
        "aggregation": "all_evaluators_below_threshold",
        "max_evaluated_candidates": 20,
        "max_total_cost_usd": 100.0,
        "request_timeout_seconds": 300,
        "fail_closed_on_incomplete": True,
        "feedback_visibility": "sample_predictions_and_outcomes",
        "provider": {
            "base_url": "https://openrouter.ai/api/v1",
            "order": [],
            "allow_fallbacks": True,
            "data_collection": "deny",
            "zdr": False,
            "sort": "none",
        },
        "evaluators": [
            {
                "name": "offline_a",
                "model_id": "offline/evaluator-a",
                "reasoning_effort": "high",
                "max_tokens": 4096,
                "image_detail": "high",
            },
            {
                "name": "offline_b",
                "model_id": "offline/evaluator-b",
                "reasoning_effort": "high",
                "max_tokens": 4096,
                "image_detail": "high",
            },
        ],
    }


def _fake_render(*, dataset: Path, scenes: int, seed: int, **_kwargs) -> dict:
    images = dataset / "images"
    images.mkdir(parents=True)
    rows = []
    for index in range(scenes):
        image = images / f"scene-{index:03d}.png"
        answer = "red" if index % 2 == 0 else "blue"
        Image.new("RGB", (16, 16), color=answer).save(image)
        for prompt_family in ("closed_choice_a", "closed_choice_b"):
            rows.append(
                {
                    "schema_version": "candidate-manifest-row-0.1.0",
                    "example_id": f"example-{index:03d}--{prompt_family}",
                    "scene_id": f"scene-{index:03d}",
                    "image_path": image.relative_to(dataset).as_posix(),
                    "question": "What color fills the square?",
                    "answer": answer,
                    "candidates": ["red", "blue"],
                    "prompt_family": prompt_family,
                    "margin": index,
                    "quarantined": False,
                }
            )
    (dataset / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return {
        "network_mode": "none",
        "container_image": "offline-test",
        "container_image_id": "sha256:test",
        "render_runtime": "python-pillow@0.1.0",
        "scenes": scenes,
        "seed": seed,
        "statistics": {"scene_count": scenes, "eligible_row_count": 2 * scenes},
    }


def _hypothesis(candidate_id: str = "candidate_001") -> dict:
    return {
        "schema_version": "difficulty-hypothesis-0.1.0",
        "candidate_id": candidate_id,
        "prior_evidence_ids": ["arrowhead_flow_grid"],
        "intended_visual_computation": (
            "Track two simultaneous graph states and combine their terminal parity."
        ),
        "anticipated_model_failure": (
            "The evaluator may update branches sequentially instead of synchronously."
        ),
        "causal_mutation_from_prior": (
            "Replace one-path tracing with two interacting paths whose updates cancel."
        ),
        "non_artifact_check": (
            "All marks are large, wording is short, answers are balanced, "
            "and the oracle reads pixels."
        ),
        "predicted_evaluator_accuracy": 0.6,
    }


def _synthesis_policy(*, image_access: str = "none") -> dict:
    policy = _policy()
    policy.update(
        {
            "id": "difficulty-feedback@0.5.0",
            "feedback_visibility": "attempt_synthesis_predictions_outcomes_and_rationales",
            "builder_image_access": image_access,
            "builder_attempt_memory_limit": 5,
            "builder_samples_per_attempt": 1,
        }
    )
    return policy


def _synthesis_hypothesis(
    candidate_id: str = "candidate_001", parent_candidate_id: str = "initial_profile"
) -> dict:
    return {
        "schema_version": "difficulty-hypothesis-0.2.0",
        "candidate_id": candidate_id,
        "parent_candidate_id": parent_candidate_id,
        "prior_evidence_ids": [parent_candidate_id],
        "prior_attempt_synthesis": (
            "The parent renderer exposes a directed graph and the inverse arm segments its "
            "arrows before ordinary reachability. The evaluator followed exactly that "
            "procedure, so additional path length did not create a perceptual bottleneck."
        ),
        "evaluator_strategy_to_block": (
            "The next design must prevent a clean read-once graph extraction followed by a "
            "standard symbolic traversal, while retaining large marks and an independently "
            "recoverable pixel oracle."
        ),
        "intended_visual_computation": (
            "Bind remote local states before constructing and traversing the visible graph."
        ),
        "anticipated_model_failure": (
            "The evaluator may preserve local states but bind them to the wrong remote edges."
        ),
        "causal_mutation_from_prior": (
            "Replace independent arrows with paired remote evidence that determines edge state."
        ),
        "non_artifact_check": (
            "All marks are large, wording is short, answers are balanced, and pixels suffice."
        ),
        "predicted_evaluator_accuracy": 0.5,
    }


def _branching_policy(*, image_access: str = "none") -> dict:
    policy = _synthesis_policy(image_access=image_access)
    policy.update(
        {
            "id": "difficulty-feedback@0.6.0",
            "feedback_visibility": "reasoning_branch_predictions_outcomes_and_rationales",
            "reasoning_targets": [
                "topological_connectivity",
                "correspondence_binding",
                "state_transition",
            ],
            "parent_selection": "best_same_reasoning_target_or_initial_profile",
        }
    )
    return policy


def _resilient_branching_policy(*, image_access: str = "none") -> dict:
    policy = _branching_policy(image_access=image_access)
    policy.update(
        {
            "id": "difficulty-feedback@0.7.0",
            "evaluator_retry_backoff_seconds": [1, 2],
            "incomplete_episode_policy": "quarantine_feedback_and_continue",
            "max_consecutive_incomplete_feedback": 3,
        }
    )
    return policy


def _effort_aware_branching_policy(*, image_access: str = "none") -> dict:
    policy = _resilient_branching_policy(image_access=image_access)
    policy.update(
        {
            "id": "difficulty-feedback@0.9.0",
            "feedback_visibility": ("reasoning_branch_predictions_outcomes_rationales_and_effort"),
        }
    )
    for evaluator in policy["evaluators"]:
        evaluator["max_tokens"] = 40960
    return policy


def _branching_hypothesis(
    candidate_id: str = "candidate_001",
    parent_candidate_id: str = "initial_profile",
    reasoning_target: str = "topological_connectivity",
) -> dict:
    payload = _synthesis_hypothesis(candidate_id, parent_candidate_id)
    payload.update(
        {
            "schema_version": "difficulty-hypothesis-0.3.0",
            "reasoning_target": reasoning_target,
            "reasoning_departure_from_archive": (
                "Unlike the archived directed-route tasks, this program recovers enclosed "
                "regions and composes cut-induced connectivity without reading arrow glyphs."
            ),
        }
    )
    return payload


def test_synthesis_policy_validates_parent_bound_hypothesis() -> None:
    policy = _synthesis_policy()
    validate_policy(policy)
    assert initial_feedback_state(policy)["schema_version"] == "difficulty-feedback-state-0.5.0"
    payload = _synthesis_hypothesis()
    assert (
        validate_difficulty_hypothesis(
            payload,
            candidate_id="candidate_001",
            policy_id=policy["id"],
            expected_parent_candidate_id="initial_profile",
        )["parent_candidate_id"]
        == "initial_profile"
    )
    with pytest.raises(ValueError, match="latest evaluated candidate"):
        validate_difficulty_hypothesis(
            payload,
            candidate_id="candidate_001",
            policy_id=policy["id"],
            expected_parent_candidate_id="prior_candidate",
        )

    bad_policy = _synthesis_policy(image_access="uncontrolled")
    with pytest.raises(ValueError, match="builder_image_access"):
        validate_policy(bad_policy)


def test_branching_policy_cycles_targets_and_reuses_only_same_target_parent() -> None:
    policy = _branching_policy()
    validate_policy(policy)
    state = initial_feedback_state(policy)
    assert state["schema_version"] == "difficulty-feedback-state-0.6.0"
    assert reasoning_branch_assignment(policy, state, 1) == {
        "reasoning_target": "topological_connectivity",
        "parent_candidate_id": "initial_profile",
        "parent_selection": "best_same_reasoning_target_or_initial_profile",
        "target_cycle_index": 0,
    }
    state["candidate_feedback"] = [
        {
            "candidate_id": "topology_a",
            "episode_index": 1,
            "status": "complete",
            "aggregate_accuracy": 0.8,
            "difficulty_hypothesis": _branching_hypothesis(
                "topology_a", reasoning_target="topological_connectivity"
            ),
        },
        {
            "candidate_id": "binding_a",
            "episode_index": 2,
            "status": "complete",
            "aggregate_accuracy": 0.2,
            "difficulty_hypothesis": _branching_hypothesis(
                "binding_a", reasoning_target="correspondence_binding"
            ),
        },
    ]
    assignment = reasoning_branch_assignment(policy, state, 4)
    assert assignment["reasoning_target"] == "topological_connectivity"
    assert assignment["parent_candidate_id"] == "topology_a"


def test_resilient_branching_retries_api_errors_and_reconciles_quarantine(
    monkeypatch, tmp_path: Path
) -> None:
    policy = _resilient_branching_policy()
    validate_policy(policy)
    monkeypatch.setattr("question_foundry.difficulty_feedback._render_candidate", _fake_render)
    monkeypatch.setattr("question_foundry.difficulty_feedback.time.sleep", lambda _delay: None)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "difficulty_hypothesis.json").write_text(
        json.dumps(_branching_hypothesis()) + "\n", encoding="utf-8"
    )

    backends = []

    class TransientBackend:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, *, candidates, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient provider failure")
            content = json.dumps(
                {
                    "answer": candidates[0],
                    "confidence": 0.7,
                    "rationale": "I followed the visible relation and selected its result.",
                }
            )
            return BackendResponse(
                content=content,
                served_model="offline/transient",
                response_id=f"response-{self.calls}",
                provider="offline",
                usage={},
                raw_response={"content": content},
            )

    def factory(_model):
        backend = TransientBackend()
        backends.append(backend)
        return backend

    result = evaluate_candidate(
        policy=policy,
        candidate_root=candidate,
        output_root=tmp_path / "feedback",
        campaign_id="test_campaign",
        candidate_id="candidate_001",
        candidate_sha256="abc123",
        episode_index=1,
        render_runtime="python-pillow@0.1.0",
        backend_factory=factory,
        expected_parent_candidate_id="initial_profile",
        expected_reasoning_target="topological_connectivity",
    )
    assert result["status"] == "complete"
    assert [item["retry_count"] for item in result["evaluators"]] == [1, 1]
    assert [backend.calls for backend in backends] == [6, 6]

    result["status"] = "incomplete"
    result["hard_seed_eligible"] = None
    state = update_feedback_state(initial_feedback_state(policy), result)
    reconciliation = feedback_reconciliation(state)
    assert reconciliation["passed"] is True
    assert reconciliation["incomplete_count"] == 1
    assert reconciliation["incomplete_disposition"] == (
        "quarantined_and_excluded_from_difficulty_statistics"
    )


def test_branching_hypothesis_is_bound_to_controller_target_and_parent() -> None:
    payload = _branching_hypothesis()
    validated = validate_difficulty_hypothesis(
        payload,
        candidate_id="candidate_001",
        policy_id="difficulty-feedback@0.6.0",
        expected_parent_candidate_id="initial_profile",
        expected_reasoning_target="topological_connectivity",
    )
    assert validated["reasoning_target"] == "topological_connectivity"
    with pytest.raises(ValueError, match="controller-assigned target"):
        validate_difficulty_hypothesis(
            payload,
            candidate_id="candidate_001",
            policy_id="difficulty-feedback@0.6.0",
            expected_parent_candidate_id="initial_profile",
            expected_reasoning_target="state_transition",
        )


def test_feedback_is_bounded_append_only_and_network_free(monkeypatch, tmp_path: Path) -> None:
    policy = _policy()
    validate_policy(policy)
    monkeypatch.setattr("question_foundry.difficulty_feedback._render_candidate", _fake_render)

    result = evaluate_candidate(
        policy=policy,
        candidate_root=tmp_path / "candidate",
        output_root=tmp_path / "feedback",
        campaign_id="test_campaign",
        candidate_id="candidate_001",
        candidate_sha256="abc123",
        episode_index=1,
        render_runtime="python-pillow@0.1.0",
        backend_factory=lambda _model: FakeBackend(),
    )
    assert result["status"] == "complete"
    assert result["provider_calls_made"] is False
    assert result["examples_per_evaluator"] == 5
    assert result["maximum_paid_requests"] == 10
    assert result["aggregate_accuracy"] == 0.6
    assert result["hard_seed_eligible"] is True
    assert {item["event_count"] for item in result["evaluators"]} == {10}
    sample_set = json.loads((tmp_path / "feedback/sample-set.json").read_text())
    assert sample_set["sample_unit"] == "unique_image"
    assert sample_set["distinct_image_count"] == 5
    assert len({row["image_path"] for row in sample_set["samples"]}) == 5
    assert len(result["sample_responses"]) == 5
    assert all(len(row["responses"]) == 2 for row in result["sample_responses"])
    assert {
        response["status"] for row in result["sample_responses"] for response in row["responses"]
    } == {"scored"}
    serialized_public_feedback = json.dumps(result["sample_responses"], sort_keys=True)
    assert "gold_answer" not in serialized_public_feedback
    assert "raw_response" not in serialized_public_feedback
    assert "response_id" not in serialized_public_feedback

    result["artifact_path"] = "episodes/001/feedback/candidate_001"
    state = update_feedback_state(initial_feedback_state(policy), result)
    reconciliation = feedback_reconciliation(state)
    assert reconciliation["passed"] is True
    assert reconciliation["evaluated_candidate_count"] == 1
    assert state["hard_seed"] == ["candidate_001"]
    assert state["candidate_feedback"][0]["sample_responses"] == result["sample_responses"]
    visible = public_feedback_history(state)
    assert visible[0]["sample_responses"] == result["sample_responses"]
    assert "candidate_sha256" not in visible[0]
    summary = feedback_summary(state)
    assert summary["evaluated_candidate_count"] == 1
    assert summary["hard_seed_count"] == 1
    assert summary["mean_candidate_accuracy"] == 0.6
    assert summary["evaluators"]["offline_a"]["scored_samples"] == 5
    assert summary["evaluators"]["offline_a"]["accuracy"] == 0.6


def test_feedback_preserves_partial_evaluator_evidence_and_fails_closed(
    monkeypatch, tmp_path: Path
) -> None:
    policy = _policy()
    monkeypatch.setattr("question_foundry.difficulty_feedback._render_candidate", _fake_render)

    def backend(model):
        if model.reference == "offline_b":
            raise RuntimeError("simulated evaluator setup failure")
        return FakeBackend()

    result = evaluate_candidate(
        policy=policy,
        candidate_root=tmp_path / "candidate",
        output_root=tmp_path / "feedback",
        campaign_id="test_campaign",
        candidate_id="candidate_001",
        candidate_sha256="abc123",
        episode_index=1,
        render_runtime="python-pillow@0.1.0",
        backend_factory=backend,
    )
    assert result["status"] == "incomplete"
    assert result["hard_seed_eligible"] is None
    assert result["evaluators"][0]["complete"] is True
    assert result["evaluators"][1]["complete"] is False
    assert result["evaluators"][1]["failure"]["error_type"] == "RuntimeError"
    assert len(result["sample_responses"]) == 5
    assert {
        response["status"]
        for row in result["sample_responses"]
        for response in row["responses"]
        if response["evaluator"] == "offline_b"
    } == {"missing"}
    assert (tmp_path / "feedback/feedback-result.json").is_file()


def test_paid_feedback_setup_failure_does_not_claim_provider_calls(
    monkeypatch, tmp_path: Path
) -> None:
    policy = _policy()
    policy["evaluators"] = policy["evaluators"][:1]
    monkeypatch.setattr("question_foundry.difficulty_feedback._render_candidate", _fake_render)
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-a-real-key")
    monkeypatch.setenv("OPENROUTER_FEEDBACK_RUN_CONFIRMED", "test_campaign")

    class BrokenBackend:
        def __init__(self, **_kwargs) -> None:
            raise RuntimeError("simulated setup failure before any request")

    monkeypatch.setattr("question_foundry.difficulty_feedback.OpenRouterBackend", BrokenBackend)
    result = evaluate_candidate(
        policy=policy,
        candidate_root=tmp_path / "candidate",
        output_root=tmp_path / "feedback",
        campaign_id="test_campaign",
        candidate_id="candidate_001",
        candidate_sha256="abc123",
        episode_index=1,
        render_runtime="python-pillow@0.1.0",
        live_model_metadata=[
            {
                "id": "offline/evaluator-a",
                "architecture": {"input_modalities": ["text", "image"]},
                "supported_parameters": ["response_format", "reasoning"],
                "reasoning": {"supported_efforts": ["high"]},
            }
        ],
    )
    assert result["status"] == "incomplete"
    assert result["provider_calls_made"] is False
    assert result["evaluators"][0]["summary"]["attempted_samples"] == 0


def test_feedback_paid_authorization_can_be_scoped_to_exact_matrix(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "not-a-real-key")
    monkeypatch.setenv("OPENROUTER_FEEDBACK_RUN_CONFIRMED", "paper_feedback_matrix_001")
    observed = preflight_authorization(
        _policy(),
        campaign_id="paper_feedback_matrix_001_tracing",
        authorization_scope_id="paper_feedback_matrix_001",
        live_model_metadata=[
            {
                "id": model_id,
                "architecture": {"input_modalities": ["text", "image"]},
                "supported_parameters": ["response_format", "reasoning"],
                "reasoning": {"supported_efforts": ["high"]},
            }
            for model_id in ("offline/evaluator-a", "offline/evaluator-b")
        ],
    )
    assert observed["authorization_scope_id"] == "paper_feedback_matrix_001"
    assert observed["maximum_paid_requests"] == 200
    assert observed["network_used"] is False
    assert observed["inference_requests"] == 0


def test_feedback_policy_requires_enough_distinct_rendered_scenes() -> None:
    policy = _policy()
    policy["render_scenes"] = 4
    with pytest.raises(ValueError, match="cover every requested example"):
        validate_policy(policy)


def test_feedback_policy_rejects_duplicate_models_and_unbounded_output() -> None:
    policy = _policy()
    policy["evaluators"][1]["model_id"] = policy["evaluators"][0]["model_id"]
    with pytest.raises(ValueError, match="model IDs must be unique"):
        validate_policy(policy)

    policy = _policy()
    policy["evaluators"][0]["max_tokens"] = 65537
    with pytest.raises(ValueError, match="max_tokens is invalid"):
        validate_policy(policy)


def test_feedback_policy_accepts_a_four_model_panel() -> None:
    policy = _policy()
    policy["evaluators"].extend(
        [
            {
                "name": "offline_c",
                "model_id": "offline/evaluator-c",
                "reasoning_effort": "medium",
                "max_tokens": 8192,
                "image_detail": "high",
            },
            {
                "name": "offline_d",
                "model_id": "offline/evaluator-d",
                "reasoning_effort": "low",
                "max_tokens": 4096,
                "image_detail": "high",
            },
        ]
    )
    validate_policy(policy)


def test_majority_frontier_aggregation_matches_frozen_five_image_rule() -> None:
    policy = _policy()
    policy["aggregation"] = "majority_evaluators_at_or_below_threshold"
    policy["hard_accuracy_threshold"] = 0.6
    policy["evaluators"].append(
        {
            "name": "offline_c",
            "model_id": "offline/evaluator-c",
            "reasoning_effort": "high",
            "max_tokens": 4096,
            "image_detail": "high",
        }
    )
    validate_policy(policy)

    mean, hard = _aggregate(policy, {"a": 0.6, "b": 0.6, "c": 1.0})
    assert mean == pytest.approx(11 / 15)
    assert hard is True

    _, hard = _aggregate(policy, {"a": 0.6, "b": 0.8, "c": 1.0})
    assert hard is False

    _, hard = _aggregate(policy, {"a": 0.4, "b": 0.8, "c": 0.6})
    assert hard is True


def test_rationale_policy_is_versioned_and_requires_matching_visibility() -> None:
    policy = _policy()
    policy["id"] = "difficulty-feedback@0.3.0"
    policy["feedback_visibility"] = "sample_predictions_outcomes_and_rationales"
    validate_policy(policy)
    assert initial_feedback_state(policy)["schema_version"] == ("difficulty-feedback-state-0.3.0")

    policy["feedback_visibility"] = "sample_predictions_and_outcomes"
    with pytest.raises(ValueError, match="sample_predictions_outcomes_and_rationales"):
        validate_policy(policy)


def test_hypothesis_policy_validates_prospective_claims_and_builds_hard_first_view() -> None:
    policy = _policy()
    policy["id"] = "difficulty-feedback@0.4.0"
    policy["feedback_visibility"] = (
        "difficulty_hypotheses_sample_predictions_outcomes_and_rationales"
    )
    validate_policy(policy)
    assert initial_feedback_state(policy)["schema_version"] == "difficulty-feedback-state-0.4.0"
    assert (
        validate_difficulty_hypothesis(_hypothesis(), candidate_id="candidate_001")[
            "predicted_evaluator_accuracy"
        ]
        == 0.6
    )

    state = initial_feedback_state(policy)
    for episode, (candidate_id, accuracy, hard) in enumerate(
        (("easy_a", 1.0, False), ("hard_a", 0.8, True), ("easy_b", 1.0, False)),
        1,
    ):
        hypothesis = _hypothesis(candidate_id)
        state["candidate_feedback"].append(
            {
                "policy_id": policy["id"],
                "candidate_id": candidate_id,
                "episode_index": episode,
                "protected_valid": True,
                "status": "complete",
                "hard_seed_eligible": hard,
                "aggregate_accuracy": accuracy,
                "evaluator_accuracies": {"sol_high": accuracy},
                "examples_per_evaluator": 5,
                "sample_responses": [{"sample_id": "sample-01", "responses": []}],
                "difficulty_hypothesis": hypothesis,
            }
        )
    view = builder_feedback_history(state)
    assert [row["candidate_id"] for row in view["records"]] == ["hard_a"]
    assert [row["candidate_id"] for row in view["easy_summaries"]] == ["easy_b", "easy_a"]
    assert all("sample_responses" not in row for row in view["easy_summaries"])


def test_hypothesis_policy_requires_and_preserves_builder_claim(
    monkeypatch, tmp_path: Path
) -> None:
    policy = _policy()
    policy["id"] = "difficulty-feedback@0.4.0"
    policy["feedback_visibility"] = (
        "difficulty_hypotheses_sample_predictions_outcomes_and_rationales"
    )
    policy["evaluators"] = policy["evaluators"][:1]
    monkeypatch.setattr("question_foundry.difficulty_feedback._render_candidate", _fake_render)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "difficulty_hypothesis.json").write_text(
        json.dumps(_hypothesis()) + "\n", encoding="utf-8"
    )

    class StructuredBackend:
        def complete(self, *, candidates, **_kwargs):
            content = json.dumps(
                {
                    "answer": candidates[0],
                    "confidence": 0.7,
                    "rationale": "I followed the visible graph and applied every update.",
                }
            )
            return BackendResponse(
                content=content,
                served_model="offline/rationale",
                response_id="response-1",
                provider="offline",
                usage={},
                raw_response={"content": content},
            )

    result = evaluate_candidate(
        policy=policy,
        candidate_root=candidate,
        output_root=tmp_path / "feedback",
        campaign_id="test_campaign",
        candidate_id="candidate_001",
        candidate_sha256="abc123",
        episode_index=1,
        render_runtime="python-pillow@0.1.0",
        backend_factory=lambda _model: StructuredBackend(),
    )
    assert result["schema_version"] == "difficulty-feedback-result-0.4.0"
    assert result["difficulty_hypothesis"] == _hypothesis()

    with pytest.raises(ValueError, match="missing or invalid"):
        evaluate_candidate(
            policy=policy,
            candidate_root=tmp_path / "missing",
            output_root=tmp_path / "not-created",
            campaign_id="test_campaign",
            candidate_id="candidate_001",
            candidate_sha256="abc123",
            episode_index=2,
            render_runtime="python-pillow@0.1.0",
            backend_factory=lambda _model: StructuredBackend(),
        )


def test_synthesis_policy_preserves_oracle_answer_and_parent_binding(
    monkeypatch, tmp_path: Path
) -> None:
    policy = _synthesis_policy()
    policy["evaluators"] = policy["evaluators"][:1]
    monkeypatch.setattr("question_foundry.difficulty_feedback._render_candidate", _fake_render)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "difficulty_hypothesis.json").write_text(
        json.dumps(_synthesis_hypothesis()) + "\n", encoding="utf-8"
    )

    class StructuredBackend:
        def complete(self, *, candidates, **_kwargs):
            content = json.dumps(
                {
                    "answer": candidates[0],
                    "confidence": 0.7,
                    "rationale": "I recovered the visible fill and selected its color.",
                }
            )
            return BackendResponse(
                content=content,
                served_model="offline/rationale",
                response_id="response-1",
                provider="offline",
                usage={},
                raw_response={"content": content},
            )

    result = evaluate_candidate(
        policy=policy,
        candidate_root=candidate,
        output_root=tmp_path / "feedback",
        campaign_id="test_campaign",
        candidate_id="candidate_001",
        candidate_sha256="abc123",
        episode_index=1,
        render_runtime="python-pillow@0.1.0",
        backend_factory=lambda _model: StructuredBackend(),
        expected_parent_candidate_id="initial_profile",
    )
    assert result["schema_version"] == "difficulty-feedback-result-0.5.0"
    assert result["difficulty_hypothesis"]["parent_candidate_id"] == "initial_profile"
    assert {sample["oracle_answer"] for sample in result["sample_responses"]} == {
        "red",
        "blue",
    }


def test_rationale_backend_exposes_bounded_final_rationale_to_the_scorer() -> None:
    class StructuredBackend:
        def complete(self, *, messages, **_kwargs):
            prompt = messages[0]["content"][0]["text"]
            assert "shown to a question-design agent" in prompt
            content = json.dumps(
                {
                    "answer": "red",
                    "confidence": 0.8,
                    "rationale": "The only continuous red path reaches the marked terminal.",
                }
            )
            return BackendResponse(
                content=content,
                served_model="offline/rationale",
                response_id="response-1",
                provider="offline",
                usage={},
                raw_response={"content": content},
            )

    backend = _RationaleFeedbackBackend(StructuredBackend())
    response = backend.complete(
        model=None,
        messages=[{"role": "user", "content": [{"type": "text", "text": "Answer."}]}],
        candidates=("red", "blue"),
        provider_policy=None,
    )
    assert json.loads(response.content) == {"answer": "red", "confidence": 0.8}
    assert response.rationale == ("The only continuous red path reaches the marked terminal.")
    assert RATIONALE_RESPONSE_FORMAT["json_schema"]["strict"] is True


def test_rationale_feedback_reaches_public_candidate_result(monkeypatch, tmp_path: Path) -> None:
    policy = _policy()
    policy["id"] = "difficulty-feedback@0.3.0"
    policy["feedback_visibility"] = "sample_predictions_outcomes_and_rationales"
    policy["evaluators"] = policy["evaluators"][:1]
    monkeypatch.setattr("question_foundry.difficulty_feedback._render_candidate", _fake_render)

    class StructuredBackend:
        def complete(self, *, candidates, **_kwargs):
            content = json.dumps(
                {
                    "answer": candidates[0],
                    "confidence": 0.7,
                    "rationale": "I compared the visible fill against the allowed colors.",
                }
            )
            return BackendResponse(
                content=content,
                served_model="offline/rationale",
                response_id="response-1",
                provider="offline",
                usage={},
                raw_response={"content": content},
            )

    result = evaluate_candidate(
        policy=policy,
        candidate_root=tmp_path / "candidate",
        output_root=tmp_path / "feedback",
        campaign_id="test_campaign",
        candidate_id="candidate_001",
        candidate_sha256="abc123",
        episode_index=1,
        render_runtime="python-pillow@0.1.0",
        backend_factory=lambda _model: StructuredBackend(),
    )
    assert result["status"] == "complete"
    assert result["schema_version"] == "difficulty-feedback-result-0.3.0"
    assert {
        response["rationale"]
        for sample in result["sample_responses"]
        for response in sample["responses"]
    } == {"I compared the visible fill against the allowed colors."}
    event_text = "".join(
        path.read_text() for path in (tmp_path / "feedback/evaluators/offline_a/events").iterdir()
    )
    assert "I compared the visible fill" in event_text


def test_effort_aware_feedback_exposes_bounded_token_proxy(monkeypatch, tmp_path: Path) -> None:
    policy = _effort_aware_branching_policy()
    policy["evaluators"] = policy["evaluators"][:1]
    monkeypatch.setattr("question_foundry.difficulty_feedback._render_candidate", _fake_render)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "difficulty_hypothesis.json").write_text(
        json.dumps(_branching_hypothesis()) + "\n", encoding="utf-8"
    )

    class StructuredBackend:
        def complete(self, *, candidates, **_kwargs):
            rationale = "I traced the visible route and compared its terminal color."
            content = json.dumps(
                {"answer": candidates[0], "confidence": 0.7, "rationale": rationale}
            )
            return BackendResponse(
                content=content,
                served_model="offline/rationale",
                response_id="response-1",
                provider="offline",
                usage={
                    "completion_tokens": 1200,
                    "completion_tokens_details": {"reasoning_tokens": 1000},
                },
                raw_response={"content": content},
            )

    result = evaluate_candidate(
        policy=policy,
        candidate_root=candidate,
        output_root=tmp_path / "feedback",
        campaign_id="test_campaign",
        candidate_id="candidate_001",
        candidate_sha256="abc123",
        episode_index=1,
        render_runtime="python-pillow@0.1.0",
        backend_factory=lambda _model: StructuredBackend(),
        expected_parent_candidate_id="initial_profile",
        expected_reasoning_target="topological_connectivity",
    )
    assert result["schema_version"] == "difficulty-feedback-result-0.9.0"
    efforts = {
        tuple(sorted(response["solver_effort"].items()))
        for sample in result["sample_responses"]
        for response in sample["responses"]
    }
    assert len(efforts) == 1
    effort = dict(next(iter(efforts)))
    assert effort["reasoning_effort_setting"] == "high"
    assert effort["completion_tokens"] == 1200
    assert effort["reasoning_tokens"] == 1000
    assert effort["visible_output_tokens"] == 200
    assert effort["completion_budget_tokens"] == 40960
    assert effort["completion_budget_fraction"] == round(1200 / 40960, 6)
    assert effort["reasoning_fraction_of_completion"] == round(1000 / 1200, 6)
    assert effort["rationale_words"] == 10


def test_paid_feedback_cost_records_must_be_provider_native() -> None:
    event = {"sample_key": "sample", "status": "scored", "usage": {"total_tokens": 12}}
    assert _cost_record_count([event]) == 0
    event["usage"]["cost"] = 0.0
    assert _cost_record_count([event]) == 1
