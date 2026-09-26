"""Foundry orchestration planning and campaign compilation.

The orchestration layer changes how agent time and context are partitioned. It
does not change the question-world candidate contract, protected verifier,
working-seed transition, or human-only canonical admission rule. One experiment
file can therefore compile the selected transactional method or one of three
comparison configurations into the shared controller/materializer backend with
a single mode selection. Their empirical ranking remains open.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

from question_foundry.sequential import validate_sequential_campaign

MODES = (
    "funnel",
    "persistent_sequential",
    "episodic_sequential",
    "hybrid_checkpointed",
)

_ID = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_SCHEMA_V1 = "orchestration-experiment-0.1.0"
_SCHEMA_V2 = "orchestration-experiment-0.2.0"
_SCHEMA_V3 = "orchestration-experiment-0.3.0"
_SCHEMA_V4 = "orchestration-experiment-0.4.0"
_SCHEMA_V5 = "orchestration-experiment-0.5.0"
_SCHEMA_V6 = "orchestration-experiment-0.6.0"
_MODE_STRATEGIES_V1 = {
    "funnel": "funnel@0.2.0",
    "persistent_sequential": "persistent-sequential@0.1.0",
    "episodic_sequential": "episodic-sequential@0.1.0",
    "hybrid_checkpointed": "hybrid-checkpointed@0.1.0",
}
_MODE_STRATEGIES_V2 = {
    "funnel": "funnel@0.3.0",
    "persistent_sequential": "persistent-sequential@0.2.0",
    "episodic_sequential": "episodic-sequential@0.2.0",
    "hybrid_checkpointed": "hybrid-checkpointed@0.2.0",
}
_MODE_STRATEGIES_V3 = {
    "funnel": "funnel@0.4.0",
    "persistent_sequential": "persistent-sequential@0.4.0",
    "episodic_sequential": "episodic-sequential@0.3.0",
    "hybrid_checkpointed": "hybrid-checkpointed@0.3.0",
}
_MODE_STRATEGIES_V4 = {
    **_MODE_STRATEGIES_V3,
    "episodic_sequential": "episodic-sequential@0.4.0",
}
_MODE_STRATEGIES_V5 = {
    **_MODE_STRATEGIES_V3,
    "episodic_sequential": "episodic-sequential@0.5.0",
}
_MODE_STRATEGIES_V6 = {
    **_MODE_STRATEGIES_V3,
    "episodic_sequential": "episodic-sequential@0.6.0",
}
_MODE_CONTEXT = {
    "funnel": "single_session_proposal_pool",
    "persistent_sequential": "single_persistent_session",
    "episodic_sequential": "fresh_session_with_structured_memory",
    "hybrid_checkpointed": "fresh_sprint_with_structured_memory",
}
_MODE_VERIFICATION = {
    "funnel": "after_final_portfolio",
    "persistent_sequential": "after_persistent_session",
    "episodic_sequential": "after_every_session",
    "hybrid_checkpointed": "between_sprints",
}
_TOP_KEYS = {
    "schema_version",
    "id",
    "status",
    "question_status",
    "protocol",
    "seed_set",
    "mechanism_memory",
    "negative_memory",
    "outcome_memory",
    "agent",
    "budget",
    "verification",
    "environment",
    "transition",
    "modes",
}
_MODE_KEYS = {
    "strategy",
    "max_sessions",
    "min_candidates_per_session",
    "max_candidates_per_session",
    "max_turns_per_session",
    "timeout_seconds_per_session",
    "minimum_start_seconds",
    "verifier_reserve_seconds",
    "repair_turns",
    "context_policy",
    "protected_verification_cadence",
    "provider_conversation_resume",
    "public_self_check_required",
    "min_logged_proposals",
    "max_build_starts",
    "min_ranking_passes",
}
_MODE_V2_EXTRA_KEYS = {
    "provider_turn_policy",
    "continuation_delay_seconds",
}
_MODE_V3_KEYS = {
    "strategy",
    "session_limit_policy",
    "candidate_limit_policy",
    "min_candidates_per_session",
    "timeout_seconds_per_session",
    "minimum_start_seconds",
    "verifier_reserve_seconds",
    "repair_turns",
    "context_policy",
    "protected_verification_cadence",
    "provider_conversation_resume",
    "provider_turn_policy",
    "continuation_delay_seconds",
    "public_self_check_required",
    "min_logged_proposals",
    "min_ranking_passes",
    "proposal_refresh_policy",
}
_SESSION_LIMIT_POLICIES = {
    "funnel": "single_continuous_session",
    "persistent_sequential": "single_continuous_session",
    "episodic_sequential": "repeat_until_agent_budget",
    "hybrid_checkpointed": "repeat_until_agent_budget",
}
_CANDIDATE_LIMIT_POLICIES = {
    "funnel": "time_bounded_append_only",
    "persistent_sequential": "time_bounded_append_only",
    "episodic_sequential": "exactly_one_per_session",
    "hybrid_checkpointed": "time_bounded_append_only",
}


@dataclass(frozen=True)
class ModePlan:
    """Normalized, paper-facing plan for one equal-budget comparison arm."""

    experiment_id: str
    campaign_id: str
    mode: str
    strategy: str
    question_status: str
    wall_time_seconds: int
    agent_time_seconds: int
    max_sessions: int | None
    min_candidates_per_session: int
    max_candidates_per_session: int | None
    session_limit_policy: str
    candidate_limit_policy: str
    max_turns_per_session: int | None
    timeout_seconds_per_session: int
    minimum_start_seconds: int
    verifier_reserve_seconds: int
    context_policy: str
    protected_verification_cadence: str
    provider_conversation_resume: bool
    provider_turn_policy: str
    continuation_delay_seconds: int
    campaign_completion_policy: str
    public_self_check_required: bool
    process_trace_required: bool
    proposal_refresh_policy: str
    controller_updates_working_seed: bool
    canonical_admission: str
    agent_adapter: str
    model: str
    reasoning_effort: str

    def as_dict(self) -> dict:
        return asdict(self)


def load_orchestration_experiment(path: Path) -> dict:
    experiment = tomllib.loads(path.read_text(encoding="utf-8"))
    validate_orchestration_experiment(experiment)
    return experiment


def validate_orchestration_experiment(experiment: dict) -> None:
    if not isinstance(experiment, dict):
        raise ValueError("orchestration experiment must be a TOML table")
    if set(experiment) != _TOP_KEYS:
        raise ValueError(f"orchestration experiment keys must be exactly {sorted(_TOP_KEYS)}")
    schema_version = experiment["schema_version"]
    supported_schemas = {
        _SCHEMA_V1,
        _SCHEMA_V2,
        _SCHEMA_V3,
        _SCHEMA_V4,
        _SCHEMA_V5,
        _SCHEMA_V6,
    }
    if schema_version not in supported_schemas:
        raise ValueError(f"schema_version must be one of {sorted(supported_schemas)}")
    deadline_driven = schema_version in {
        _SCHEMA_V2,
        _SCHEMA_V3,
        _SCHEMA_V4,
        _SCHEMA_V5,
        _SCHEMA_V6,
    }
    cap_free = schema_version in {_SCHEMA_V3, _SCHEMA_V4, _SCHEMA_V5, _SCHEMA_V6}
    if not _ID.fullmatch(str(experiment["id"])):
        raise ValueError("invalid orchestration experiment id")
    if experiment["status"] not in {"draft", "awaiting_operator_approval", "frozen"}:
        raise ValueError("invalid orchestration experiment status")
    if experiment["question_status"] != "open":
        raise ValueError("the four-mode design question must remain explicitly open")
    if experiment["protocol"] != "question-world@0.4.0":
        raise ValueError("orchestration experiment requires question-world@0.4.0")

    agent = experiment["agent"]
    required_agent = {
        "adapter",
        "model",
        "reasoning_effort",
        "cli_version",
        "output_format",
    }
    optional_agent = {"reported_model", "auxiliary_reported_models"}
    if not isinstance(agent, dict) or not required_agent <= set(agent):
        raise ValueError(f"agent must contain {sorted(required_agent)}")
    if not set(agent) <= required_agent | optional_agent:
        raise ValueError("agent contains unsupported keys")

    budget = experiment["budget"]
    if not isinstance(budget, dict):
        raise ValueError("budget must be a TOML table")
    expected_budget_keys = {
        "budget_scope",
        "wall_time_seconds_per_mode",
        "cost_reporting",
        "soft_cost_limit_usd",
    }
    if deadline_driven:
        expected_budget_keys |= {
            "agent_time_seconds_per_mode",
            "completion_policy",
            "transient_error_policy",
            "retry_backoff_seconds",
            "max_consecutive_provider_failures",
        }
    if set(budget) != expected_budget_keys:
        raise ValueError("budget keys are invalid")
    if budget["budget_scope"] != "per_mode":
        raise ValueError("budget_scope must be per_mode for a controlled comparison")
    wall = int(budget["wall_time_seconds_per_mode"])
    # Cap-free continuous arms may spend four hours with the provider and then
    # use a separate hour for protected no-network verification.
    wall_ceiling = 18000 if cap_free else 7200
    if not 600 <= wall <= wall_ceiling:
        raise ValueError(f"wall_time_seconds_per_mode must be in [600, {wall_ceiling}]")
    if float(budget["soft_cost_limit_usd"]) <= 0:
        raise ValueError("soft_cost_limit_usd must be positive")
    if deadline_driven:
        agent_time = int(budget["agent_time_seconds_per_mode"])
        if not 600 <= agent_time < wall:
            raise ValueError("agent_time_seconds_per_mode must be in [600, controller wall time)")
        if budget["completion_policy"] != "consume_agent_time_budget":
            raise ValueError("deadline-driven completion_policy must consume_agent_time_budget")
        if budget["transient_error_policy"] != "retry_with_backoff":
            raise ValueError("deadline-driven transient errors must retry with backoff")
        backoff = budget["retry_backoff_seconds"]
        if (
            not isinstance(backoff, list)
            or not backoff
            or backoff != sorted(backoff)
            or any(
                isinstance(seconds, bool) or not isinstance(seconds, int) or not 5 <= seconds <= 60
                for seconds in backoff
            )
        ):
            raise ValueError("retry_backoff_seconds must be an increasing integer list in [5, 60]")
        if not 1 <= int(budget["max_consecutive_provider_failures"]) <= 10:
            raise ValueError("max_consecutive_provider_failures must be in [1, 10]")

    verification = experiment["verification"]
    if not isinstance(verification, dict) or set(verification) != {
        "semantic_neighbor_limit",
        "semantic_duplicate_threshold",
        "oracle_stress_seeds",
        "oracle_stress_scenes_per_seed",
    }:
        raise ValueError("verification keys are invalid")
    environment = experiment["environment"]
    if not isinstance(environment, dict) or set(environment) != {
        "agent_network",
        "verifier_network",
        "cpus",
        "memory_mb",
        "storage_mb",
    }:
        raise ValueError("environment keys are invalid")

    modes = experiment["modes"]
    if not isinstance(modes, dict):
        raise ValueError("modes must be a TOML table")
    if set(modes) != set(MODES):
        raise ValueError(f"modes must be exactly {list(MODES)}")
    mode_strategies = (
        _MODE_STRATEGIES_V6
        if schema_version == _SCHEMA_V6
        else _MODE_STRATEGIES_V5
        if schema_version == _SCHEMA_V5
        else _MODE_STRATEGIES_V4
        if schema_version == _SCHEMA_V4
        else _MODE_STRATEGIES_V3
        if cap_free
        else _MODE_STRATEGIES_V2
        if deadline_driven
        else _MODE_STRATEGIES_V1
    )
    for mode in MODES:
        values = modes[mode]
        if not isinstance(values, dict):
            raise ValueError(f"{mode} must be a TOML table")
        expected_mode_keys = (
            _MODE_V3_KEYS
            if cap_free
            else _MODE_KEYS | (_MODE_V2_EXTRA_KEYS if deadline_driven else set())
        )
        if set(values) != expected_mode_keys:
            raise ValueError(f"{mode} keys must be exactly {sorted(expected_mode_keys)}")
        if values["strategy"] != mode_strategies[mode]:
            raise ValueError(f"{mode} must use {mode_strategies[mode]}")
        if values["context_policy"] != _MODE_CONTEXT[mode]:
            raise ValueError(f"{mode} context_policy is invalid")
        if values["protected_verification_cadence"] != _MODE_VERIFICATION[mode]:
            raise ValueError(f"{mode} protected_verification_cadence is invalid")
        expected_resume = deadline_driven and mode != "episodic_sequential"
        if values["provider_conversation_resume"] is not expected_resume:
            if deadline_driven:
                raise ValueError(f"{mode} provider_conversation_resume must be {expected_resume}")
            raise ValueError("provider conversation resume is excluded from the v0.1 comparison")
        if deadline_driven:
            expected_turn_policy = (
                "single_turn" if mode == "episodic_sequential" else "resume_until_timeout"
            )
            if values["provider_turn_policy"] != expected_turn_policy:
                raise ValueError(f"{mode} provider_turn_policy must be {expected_turn_policy}")
            continuation_delay = int(values["continuation_delay_seconds"])
            if expected_turn_policy == "single_turn":
                if continuation_delay != 0:
                    raise ValueError("single-turn mode requires zero continuation delay")
            elif not 5 <= continuation_delay <= 120:
                raise ValueError("resumed modes require continuation delay in [5, 120]")
        if values["public_self_check_required"] is not True:
            raise ValueError("every mode must require agent-authored public self-checks")

        timeout = int(values["timeout_seconds_per_session"])
        reserve = int(values["verifier_reserve_seconds"])
        minimum_start = int(values["minimum_start_seconds"])
        sessions = None if cap_free else int(values["max_sessions"])
        if sessions is not None and not 1 <= sessions <= 20:
            raise ValueError(f"{mode} max_sessions must be in [1, 20]")
        if not 60 <= timeout < wall:
            raise ValueError(f"{mode} session timeout must be in [60, per-mode wall time)")
        if not 30 <= reserve < timeout:
            raise ValueError(f"{mode} verifier reserve must be in [30, session timeout)")
        if timeout + reserve > wall:
            raise ValueError(f"{mode} cannot fit one session and verifier inside its arm budget")
        if not 60 <= minimum_start <= timeout:
            raise ValueError(f"{mode} minimum_start_seconds is invalid")
        if not cap_free and not 1 <= int(values["max_turns_per_session"]) <= 240:
            raise ValueError(f"{mode} max_turns_per_session must be in [1, 240]")

        minimum = int(values["min_candidates_per_session"])
        if not 1 <= minimum <= 10:
            raise ValueError(f"{mode} candidate minimum is invalid")
        if cap_free:
            if values["session_limit_policy"] != _SESSION_LIMIT_POLICIES[mode]:
                raise ValueError(f"{mode} session_limit_policy is invalid")
            if values["candidate_limit_policy"] != _CANDIDATE_LIMIT_POLICIES[mode]:
                raise ValueError(f"{mode} candidate_limit_policy is invalid")
            if mode == "episodic_sequential" and minimum != 1:
                raise ValueError("episodic sequential requires one candidate per session")
            if mode == "funnel":
                if not 1 <= int(values["min_logged_proposals"]) <= 100:
                    raise ValueError("funnel proposal floor is invalid")
                if not 2 <= int(values["min_ranking_passes"]) <= 5:
                    raise ValueError("funnel ranking-pass count is invalid")
                if values["proposal_refresh_policy"] != "replenish_ranked_queue":
                    raise ValueError("funnel must replenish its ranked proposal queue")
            elif (
                int(values["min_logged_proposals"]) != 0
                or int(values["min_ranking_passes"]) != 0
                or values["proposal_refresh_policy"] != "not_applicable"
            ):
                raise ValueError(f"{mode} may not inherit funnel proposal requirements")
        else:
            maximum = int(values["max_candidates_per_session"])
            if not minimum <= maximum <= 10:
                raise ValueError(f"{mode} candidate bounds are invalid")
            if mode in {"funnel", "persistent_sequential"} and sessions != 1:
                raise ValueError(f"{mode} is a single-session arm")
            if mode == "episodic_sequential" and (minimum, maximum) != (1, 1):
                raise ValueError("episodic sequential requires exactly one candidate per session")
            if mode == "hybrid_checkpointed" and sessions is not None and sessions < 2:
                raise ValueError("hybrid checkpointed requires at least two sprints")
            if mode == "funnel":
                if not 1 <= int(values["min_logged_proposals"]) <= 100:
                    raise ValueError("funnel proposal floor is invalid")
                if not maximum <= int(values["max_build_starts"]) <= 20:
                    raise ValueError("funnel build-start cap must cover its candidate cap")
                if not 2 <= int(values["min_ranking_passes"]) <= 5:
                    raise ValueError("funnel ranking-pass count is invalid")
            elif any(
                int(values[key]) != 0
                for key in ("min_logged_proposals", "max_build_starts", "min_ranking_passes")
            ):
                raise ValueError(f"{mode} may not inherit funnel proposal quotas")

    transition = experiment["transition"]
    if not isinstance(transition, dict):
        raise ValueError("transition must be a TOML table")
    if transition != {
        "require_submission_valid": True,
        "require_all_mechanical_gates": True,
        "require_registry_distinct": True,
        "require_semantic_review_ready": True,
        "canonical_admission": "human-only",
    }:
        raise ValueError("transition policy differs from the protected contract")

    # Compilation exercises the shared backend validator for every arm. This
    # closes schema drift between the paper-facing experiment and executable
    # campaign formats.
    for mode in MODES:
        validate_sequential_campaign(compile_mode_campaign(experiment, mode, _validated=True))


def mode_plan(experiment: dict, mode: str) -> ModePlan:
    validate_orchestration_experiment(experiment)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {list(MODES)}")
    values = experiment["modes"][mode]
    cap_free = experiment["schema_version"] in {
        _SCHEMA_V3,
        _SCHEMA_V4,
        _SCHEMA_V5,
        _SCHEMA_V6,
    }
    session_limit_policy = values.get(
        "session_limit_policy",
        "single_continuous_session"
        if mode in {"funnel", "persistent_sequential"}
        else "fixed_session_safety_cap",
    )
    candidate_limit_policy = values.get(
        "candidate_limit_policy",
        "exactly_one_per_session" if mode == "episodic_sequential" else "fixed_portfolio_cap",
    )
    return ModePlan(
        experiment_id=experiment["id"],
        campaign_id=f"{experiment['id']}_{mode}",
        mode=mode,
        strategy=values["strategy"],
        question_status=experiment["question_status"],
        wall_time_seconds=int(experiment["budget"]["wall_time_seconds_per_mode"]),
        agent_time_seconds=int(
            experiment["budget"].get(
                "agent_time_seconds_per_mode",
                experiment["budget"]["wall_time_seconds_per_mode"],
            )
        ),
        max_sessions=None if cap_free else int(values["max_sessions"]),
        min_candidates_per_session=int(values["min_candidates_per_session"]),
        max_candidates_per_session=(
            None if cap_free else int(values["max_candidates_per_session"])
        ),
        session_limit_policy=str(session_limit_policy),
        candidate_limit_policy=str(candidate_limit_policy),
        max_turns_per_session=(None if cap_free else int(values["max_turns_per_session"])),
        timeout_seconds_per_session=int(values["timeout_seconds_per_session"]),
        minimum_start_seconds=int(values["minimum_start_seconds"]),
        verifier_reserve_seconds=int(values["verifier_reserve_seconds"]),
        context_policy=values["context_policy"],
        protected_verification_cadence=values["protected_verification_cadence"],
        provider_conversation_resume=bool(values["provider_conversation_resume"]),
        provider_turn_policy=str(values.get("provider_turn_policy", "single_turn")),
        continuation_delay_seconds=int(values.get("continuation_delay_seconds", 0)),
        campaign_completion_policy=str(
            experiment["budget"].get("completion_policy", "episode_or_session_cap")
        ),
        public_self_check_required=bool(values["public_self_check_required"]),
        process_trace_required=mode == "funnel",
        proposal_refresh_policy=str(values.get("proposal_refresh_policy", "fixed_proposal_pool")),
        controller_updates_working_seed=True,
        canonical_admission=experiment["transition"]["canonical_admission"],
        agent_adapter=experiment["agent"]["adapter"],
        model=experiment["agent"]["model"],
        reasoning_effort=experiment["agent"]["reasoning_effort"],
    )


def all_mode_plans(experiment: dict) -> list[ModePlan]:
    validate_orchestration_experiment(experiment)
    return [mode_plan(experiment, mode) for mode in MODES]


def compile_mode_campaign(
    experiment: dict,
    mode: str,
    *,
    _validated: bool = False,
) -> dict:
    if not _validated:
        validate_orchestration_experiment(experiment)
    if mode not in MODES:
        raise ValueError(f"mode must be one of {list(MODES)}")
    values = experiment["modes"][mode]
    cap_free = experiment["schema_version"] in {
        _SCHEMA_V3,
        _SCHEMA_V4,
        _SCHEMA_V5,
        _SCHEMA_V6,
    }
    task = {
        "min_candidates": int(values["min_candidates_per_session"]),
        "repair_turns": int(values["repair_turns"]),
    }
    if cap_free:
        task["candidate_limit_policy"] = values["candidate_limit_policy"]
    else:
        task["max_candidates"] = int(values["max_candidates_per_session"])
    if mode == "funnel":
        task["min_logged_proposals"] = int(values["min_logged_proposals"])
        task["min_ranking_passes"] = int(values["min_ranking_passes"])
        if cap_free:
            task["proposal_refresh_policy"] = values["proposal_refresh_policy"]
        else:
            task["max_build_starts"] = int(values["max_build_starts"])
    campaign_control = {
        "wall_time_seconds": int(experiment["budget"]["wall_time_seconds_per_mode"]),
        "minimum_start_seconds": int(values["minimum_start_seconds"]),
        "verifier_reserve_seconds": int(values["verifier_reserve_seconds"]),
        "cost_reporting": experiment["budget"]["cost_reporting"],
        "soft_cost_limit_usd": float(experiment["budget"]["soft_cost_limit_usd"]),
    }
    if cap_free:
        campaign_control["episode_limit_policy"] = values["session_limit_policy"]
    else:
        campaign_control["max_episodes"] = int(values["max_sessions"])
    if experiment["schema_version"] in {
        _SCHEMA_V2,
        _SCHEMA_V3,
        _SCHEMA_V4,
        _SCHEMA_V5,
        _SCHEMA_V6,
    }:
        campaign_control.update(
            {
                "completion_policy": experiment["budget"]["completion_policy"],
                "agent_time_seconds": int(experiment["budget"]["agent_time_seconds_per_mode"]),
                "transient_error_policy": experiment["budget"]["transient_error_policy"],
                "retry_backoff_seconds": list(experiment["budget"]["retry_backoff_seconds"]),
                "max_consecutive_provider_failures": int(
                    experiment["budget"]["max_consecutive_provider_failures"]
                ),
            }
        )
    return {
        "id": f"{experiment['id']}_{mode}",
        "status": ("frozen" if experiment["status"] == "frozen" else "awaiting_operator_approval"),
        "mode": "discovery",
        "protocol": experiment["protocol"],
        "strategy": values["strategy"],
        "seed_set": experiment["seed_set"],
        "mechanism_memory": experiment["mechanism_memory"],
        "negative_memory": experiment["negative_memory"],
        "outcome_memory": experiment["outcome_memory"],
        "agent": {
            **experiment["agent"],
            "timeout_seconds_per_episode": int(values["timeout_seconds_per_session"]),
            **({} if cap_free else {"max_turns_per_episode": int(values["max_turns_per_session"])}),
            **(
                {
                    "provider_turn_policy": values["provider_turn_policy"],
                    "continuation_delay_seconds": int(values["continuation_delay_seconds"]),
                }
                if experiment["schema_version"]
                in {_SCHEMA_V2, _SCHEMA_V3, _SCHEMA_V4, _SCHEMA_V5, _SCHEMA_V6}
                else {}
            ),
        },
        "campaign": campaign_control,
        "task": task,
        "verification": dict(experiment["verification"]),
        "environment": dict(experiment["environment"]),
        "transition": dict(experiment["transition"]),
        "comparison": {
            "experiment_id": experiment["id"],
            "orchestration_mode": mode,
            "question_status": experiment["question_status"],
            "budget_scope": experiment["budget"]["budget_scope"],
            "context_policy": values["context_policy"],
            "protected_verification_cadence": values["protected_verification_cadence"],
            "provider_conversation_resume": values["provider_conversation_resume"],
            **(
                {
                    "provider_turn_policy": values["provider_turn_policy"],
                    "completion_policy": experiment["budget"]["completion_policy"],
                }
                if experiment["schema_version"]
                in {_SCHEMA_V2, _SCHEMA_V3, _SCHEMA_V4, _SCHEMA_V5, _SCHEMA_V6}
                else {}
            ),
            **(
                {
                    "session_limit_policy": values["session_limit_policy"],
                    "candidate_limit_policy": values["candidate_limit_policy"],
                    "proposal_refresh_policy": values["proposal_refresh_policy"],
                }
                if cap_free
                else {}
            ),
        },
    }


def _toml_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, (int, float)):
        return str(value)
    raise TypeError(f"unsupported TOML value: {type(value).__name__}")


def _render_toml_table(lines: list[str], path: tuple[str, ...], table: dict) -> None:
    """Render one nested table, including arrays of tables, deterministically."""

    lines.extend(("", f"[{'.'.join(path)}]"))
    for key, value in table.items():
        if isinstance(value, dict) or (
            isinstance(value, list) and value and all(isinstance(item, dict) for item in value)
        ):
            continue
        lines.append(f"{key} = {_toml_value(value)}")
    for key, value in table.items():
        if isinstance(value, dict):
            _render_toml_table(lines, (*path, key), value)
        elif isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            for item in value:
                lines.extend(("", f"[[{'.'.join((*path, key))}]]"))
                for item_key, item_value in item.items():
                    if isinstance(item_value, dict) or (
                        isinstance(item_value, list)
                        and item_value
                        and all(isinstance(child, dict) for child in item_value)
                    ):
                        raise TypeError("nested tables inside arrays of tables are unsupported")
                    lines.append(f"{item_key} = {_toml_value(item_value)}")


def render_compiled_campaign(campaign: dict) -> str:
    """Render the normalized backend campaign without a third-party TOML writer."""

    validate_sequential_campaign(campaign)
    lines = []
    for key in (
        "id",
        "status",
        "mode",
        "protocol",
        "strategy",
        "seed_set",
        "mechanism_memory",
        "negative_memory",
        "outcome_memory",
    ):
        lines.append(f"{key} = {_toml_value(campaign[key])}")
    for block in (
        "agent",
        "campaign",
        "task",
        "verification",
        "environment",
        "transition",
        "comparison",
    ):
        _render_toml_table(lines, (block,), campaign[block])
    for block in ("archive_policy", "feedback_bootstrap", "feedback_policy"):
        if block in campaign:
            _render_toml_table(lines, (block,), campaign[block])
    return "\n".join(lines) + "\n"


def write_compiled_campaign(experiment: dict, mode: str, path: Path) -> dict:
    campaign = compile_mode_campaign(experiment, mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_compiled_campaign(campaign), encoding="utf-8")
    return campaign
