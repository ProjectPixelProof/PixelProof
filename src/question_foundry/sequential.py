"""Materialization and state transitions for controller-owned foundry campaigns.

Historical sequential strategies keep the question-world v0.4
candidate/verifier contract but move repetition, timestamps, and seed updates
outside the model. Prospective orchestration strategies use the same packet and
controller foundation while varying only session continuity, candidate count,
process-trace requirements, and protected-verification cadence.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path

from question_foundry.candidate_contract import require_task_signature
from question_foundry.difficulty_feedback import (
    BRANCHING_POLICY_ID,
    EFFORT_AWARE_BRANCHING_POLICY_ID,
    HYPOTHESIS_POLICY_IDS,
    RATIONALE_POLICY_IDS,
    RESILIENT_BRANCHING_POLICY_ID,
    SYNTHESIS_POLICY_ID,
    SYNTHESIS_POLICY_IDS,
    WIRED_RESILIENT_BRANCHING_POLICY_ID,
    builder_feedback_history,
    initial_feedback_state,
    public_feedback_history,
    reasoning_branch_assignment,
)
from question_foundry.difficulty_feedback import (
    POLICY_ID as DIFFICULTY_HYPOTHESIS_POLICY_ID,
)
from question_foundry.difficulty_feedback import (
    validate_policy as validate_feedback_policy,
)
from question_foundry.quality_diversity import (
    initial_archive as initial_quality_diversity_archive,
)
from question_foundry.quality_diversity import (
    select_target as select_quality_diversity_target,
)
from question_foundry.quality_diversity import (
    validate_policy as validate_quality_diversity_policy,
)
from question_foundry.registry import canonical_json, sha256_candidate_tree
from question_foundry.render_runtime import (
    campaign_render_runtime,
    configure_task_render_runtime,
)
from question_foundry.semantic_novelty import candidate_as_memory, candidate_text

_ID = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_DISCOVERY_PROFILE_REF = re.compile(
    r"^(?P<family>[a-z][a-z0-9_-]{1,31})@"
    r"(?P<major>[0-9]+)\.(?P<minor>[0-9]+)\.(?P<patch>[0-9]+):"
    r"(?P<profile_id>[a-z][a-z0-9_]{2,63})$"
)
_SUPPORTED_PROTOCOL = "question-world@0.4.0"
# Controller time is safety headroom, not scientific agent time. Section 3
# inserts deterministic rendering, three evaluator batches, and retries between
# episodes; the first two-profile run showed that a 12-hour host ceiling cannot
# reliably deliver six useful builder hours. Keep the builder budget separately
# frozen while allowing up to 36 hours of orchestration overhead.
_MAX_CAMPAIGN_WALL_SECONDS = 129600
# Strategy version -> source directory and executable orchestration policy. A
# used version is never edited in place; a behavioural change ships as a new
# version so preserved campaign records keep meaning what they meant when run.
_SUPPORTED_STRATEGIES = {
    "sequential@0.1.0": {
        "directory": "sequential/v0.1",
        "mode": "episodic_sequential",
        "candidate_bounds": (1, 1),
        "session_bounds": (1, 20),
        "max_turns": 240,
        "max_timeout_seconds": 1800,
        "process_trace_required": False,
    },
    "sequential@0.2.0": {
        "directory": "sequential/v0.2",
        "mode": "episodic_sequential",
        "candidate_bounds": (1, 1),
        "session_bounds": (1, 20),
        "max_turns": 240,
        "max_timeout_seconds": 1800,
        "process_trace_required": False,
    },
    "funnel@0.2.0": {
        "directory": "funnel/v0.2",
        "mode": "funnel",
        "candidate_bounds": (1, 10),
        "session_bounds": (1, 1),
        "max_turns": 240,
        "max_timeout_seconds": 5400,
        "process_trace_required": True,
    },
    "funnel@0.3.0": {
        "directory": "funnel/v0.3",
        "mode": "funnel",
        "candidate_bounds": (1, 10),
        "session_bounds": (1, 1),
        "max_turns": 240,
        "max_timeout_seconds": 5400,
        "process_trace_required": True,
        "task_spec_version": "0.2.0",
        "deadline_driven": True,
    },
    "funnel@0.4.0": {
        "directory": "funnel/v0.4",
        "mode": "funnel",
        "candidate_limit_policy": "time_bounded_append_only",
        "episode_limit_policy": "single_continuous_session",
        "max_timeout_seconds": 14400,
        "process_trace_required": True,
        "task_spec_version": "0.3.0",
        "deadline_driven": True,
        "cap_free": True,
    },
    "persistent-sequential@0.1.0": {
        "directory": "persistent_sequential/v0.1",
        "mode": "persistent_sequential",
        "candidate_bounds": (1, 10),
        "session_bounds": (1, 1),
        "max_turns": 240,
        "max_timeout_seconds": 5400,
        "process_trace_required": False,
    },
    "persistent-sequential@0.2.0": {
        "directory": "persistent_sequential/v0.2",
        "mode": "persistent_sequential",
        "candidate_bounds": (1, 10),
        "session_bounds": (1, 1),
        "max_turns": 240,
        "max_timeout_seconds": 5400,
        "process_trace_required": False,
        "task_spec_version": "0.2.0",
        "deadline_driven": True,
    },
    "persistent-sequential@0.3.0": {
        "directory": "persistent_sequential/v0.3",
        "mode": "persistent_sequential",
        "candidate_bounds": (1, 10),
        "session_bounds": (1, 1),
        "max_turns": 240,
        "max_timeout_seconds": 10800,
        "process_trace_required": False,
        "task_spec_version": "0.2.0",
        "deadline_driven": True,
    },
    "persistent-sequential@0.4.0": {
        "directory": "persistent_sequential/v0.4",
        "mode": "persistent_sequential",
        "candidate_limit_policy": "time_bounded_append_only",
        "episode_limit_policy": "single_continuous_session",
        "max_timeout_seconds": 14400,
        "process_trace_required": False,
        "task_spec_version": "0.3.0",
        "deadline_driven": True,
        "cap_free": True,
    },
    "episodic-sequential@0.1.0": {
        "directory": "episodic_sequential/v0.1",
        "mode": "episodic_sequential",
        "candidate_bounds": (1, 1),
        "session_bounds": (1, 20),
        "max_turns": 240,
        "max_timeout_seconds": 5400,
        "process_trace_required": False,
    },
    "episodic-sequential@0.2.0": {
        "directory": "episodic_sequential/v0.2",
        "mode": "episodic_sequential",
        "candidate_bounds": (1, 1),
        "session_bounds": (1, 100),
        "max_turns": 240,
        "max_timeout_seconds": 5400,
        "process_trace_required": False,
        "task_spec_version": "0.2.0",
        "deadline_driven": True,
    },
    "episodic-sequential@0.3.0": {
        "directory": "episodic_sequential/v0.3",
        "mode": "episodic_sequential",
        "candidate_limit_policy": "exactly_one_per_session",
        "episode_limit_policy": "repeat_until_agent_budget",
        "max_timeout_seconds": 5400,
        "process_trace_required": False,
        "task_spec_version": "0.3.0",
        "deadline_driven": True,
        "cap_free": True,
    },
    "episodic-sequential@0.4.0": {
        "directory": "episodic_sequential/v0.4",
        "mode": "episodic_sequential",
        "candidate_limit_policy": "exactly_one_per_session",
        "episode_limit_policy": "repeat_until_agent_budget",
        "max_timeout_seconds": 5400,
        "process_trace_required": False,
        "task_spec_version": "0.3.0",
        "deadline_driven": True,
        "cap_free": True,
        "authoritative_public_gate": True,
    },
    "episodic-sequential@0.5.0": {
        "directory": "episodic_sequential/v0.5",
        "mode": "episodic_sequential",
        "candidate_limit_policy": "exactly_one_per_session",
        "episode_limit_policy": "repeat_until_agent_budget",
        "max_timeout_seconds": 5400,
        "process_trace_required": False,
        "task_spec_version": "0.4.0",
        "deadline_driven": True,
        "cap_free": True,
        "authoritative_public_gate": True,
        "strategy_local_task_spec_schema": True,
    },
    "episodic-sequential@0.6.0": {
        "directory": "episodic_sequential/v0.6",
        "mode": "episodic_sequential",
        "candidate_limit_policy": "exactly_one_per_session",
        "episode_limit_policy": "repeat_until_agent_budget",
        "max_timeout_seconds": 5400,
        "process_trace_required": False,
        "task_spec_version": "0.4.0",
        "deadline_driven": True,
        "cap_free": True,
        "authoritative_public_gate": True,
        "strategy_local_task_spec_schema": True,
        "checkpoint_aware": True,
    },
    "episodic-sequential@0.7.0": {
        "directory": "episodic_sequential/v0.7",
        "mode": "episodic_sequential",
        "candidate_limit_policy": "exactly_one_per_session",
        "episode_limit_policy": "repeat_until_agent_budget",
        "max_timeout_seconds": 5400,
        "process_trace_required": False,
        "task_spec_version": "0.5.0",
        "deadline_driven": True,
        "cap_free": True,
        "authoritative_public_gate": True,
        "strategy_local_task_spec_schema": True,
        "checkpoint_aware": True,
        "paper_audit": True,
    },
    "hybrid-checkpointed@0.1.0": {
        "directory": "hybrid_checkpointed/v0.1",
        "mode": "hybrid_checkpointed",
        "candidate_bounds": (1, 5),
        "session_bounds": (2, 10),
        "max_turns": 240,
        "max_timeout_seconds": 5400,
        "process_trace_required": False,
    },
    "hybrid-checkpointed@0.2.0": {
        "directory": "hybrid_checkpointed/v0.2",
        "mode": "hybrid_checkpointed",
        "candidate_bounds": (1, 5),
        "session_bounds": (2, 100),
        "max_turns": 240,
        "max_timeout_seconds": 5400,
        "process_trace_required": False,
        "task_spec_version": "0.2.0",
        "deadline_driven": True,
    },
    "hybrid-checkpointed@0.3.0": {
        "directory": "hybrid_checkpointed/v0.3",
        "mode": "hybrid_checkpointed",
        "candidate_limit_policy": "time_bounded_append_only",
        "episode_limit_policy": "repeat_until_agent_budget",
        "max_timeout_seconds": 5400,
        "process_trace_required": False,
        "task_spec_version": "0.3.0",
        "deadline_driven": True,
        "cap_free": True,
    },
}
# v0.2 separates ordinary budget exhaustion from genuine infrastructure failure,
# so its controller state carries an extra bucket.
_STATE_SCHEMA = {
    "sequential@0.1.0": "sequential-campaign-state-0.1.0",
    "sequential@0.2.0": "sequential-campaign-state-0.2.0",
    "funnel@0.2.0": "orchestration-campaign-state-0.1.0",
    "funnel@0.3.0": "orchestration-campaign-state-0.2.0",
    "funnel@0.4.0": "orchestration-campaign-state-0.3.0",
    "persistent-sequential@0.1.0": "orchestration-campaign-state-0.1.0",
    "persistent-sequential@0.2.0": "orchestration-campaign-state-0.2.0",
    "persistent-sequential@0.3.0": "orchestration-campaign-state-0.2.0",
    "persistent-sequential@0.4.0": "orchestration-campaign-state-0.3.0",
    "episodic-sequential@0.1.0": "orchestration-campaign-state-0.1.0",
    "episodic-sequential@0.2.0": "orchestration-campaign-state-0.2.0",
    "episodic-sequential@0.3.0": "orchestration-campaign-state-0.3.0",
    "episodic-sequential@0.4.0": "orchestration-campaign-state-0.3.0",
    "episodic-sequential@0.5.0": "orchestration-campaign-state-0.3.0",
    "episodic-sequential@0.6.0": "orchestration-campaign-state-0.3.0",
    "episodic-sequential@0.7.0": "orchestration-campaign-state-0.3.0",
    "hybrid-checkpointed@0.1.0": "orchestration-campaign-state-0.1.0",
    "hybrid-checkpointed@0.2.0": "orchestration-campaign-state-0.2.0",
    "hybrid-checkpointed@0.3.0": "orchestration-campaign-state-0.3.0",
}
# A drafted third-party campaign carries this until an auth smoke reveals the
# slug the endpoint actually echoes. Materialization refuses to proceed on it.
_PENDING_REPORTED_MODEL = "PENDING_AUTH_SMOKE"
# Each adapter pins its exact model configuration and declares whether the CLI's
# self-reported cost is this provider's billing. A third-party Anthropic-compatible
# endpoint reports Anthropic-priced cost, which must never drive a stopping rule.
_SUPPORTED_AGENTS = {
    "claude_code_foundry:ClaudeCodeFoundry": {
        "configs": (
            {
                "model": "claude-opus-4-8",
                "reasoning_effort": "high",
                "output_format": "stream-json",
            },
            {
                "model": "claude-opus-5",
                "reasoning_effort": "medium",
                "output_format": "stream-json",
            },
            {
                "model": "claude-opus-5",
                "reasoning_effort": "high",
                "output_format": "stream-json",
            },
        ),
        "requires_reported_model": False,
        # Claude Code may emit an Anthropic price-table estimate when a stream
        # terminates cleanly, but subscription OAuth does not expose a billed
        # provider charge and timed-out streams may omit the estimate entirely.
        # Historical API-style campaigns retain provider_native; new OAuth
        # campaigns should use unavailable_subscription and time bounds.
        "cost_reporting": ("provider_native", "unavailable_subscription"),
    },
    "gemma_local_foundry:GemmaLocalFoundry": {
        "configs": (
            {
                "model": "gemma-4-31B-it",
                "reasoning_effort": "high",
                "output_format": "stream-json",
            },
        ),
        # The local proxy echoes the served model name verbatim, so the observed
        # slug equals the request and no separate auth-smoke value is needed.
        "requires_reported_model": False,
        # No provider billing behind the local vLLM endpoint; cost is meaningless
        # exactly as with the GLM proxy route.
        "cost_reporting": "unreliable_proxy",
    },
    "codex_oauth:CodexOAuth": {
        "configs": (
            {
                "model": "gpt-5.6-luna",
                "reasoning_effort": "high",
                "output_format": "json",
            },
            {
                "model": "gpt-5.6-luna",
                "reasoning_effort": "max",
                "output_format": "json",
            },
            {
                "model": "gpt-5.6-sol",
                "reasoning_effort": "high",
                "output_format": "json",
            },
            {
                "model": "gpt-5.6-sol",
                "reasoning_effort": "max",
                "output_format": "json",
            },
        ),
        "requires_reported_model": False,
        # A ChatGPT-subscription OAuth session exposes token usage but no
        # provider-native dollar charge. Wall time and episode count are the
        # enforceable bounds for this developmental run.
        "cost_reporting": "unavailable_subscription",
        "supports_resume_until_timeout": True,
    },
    "openrouter_opencode_foundry:OpenRouterOpenCodeFoundry": {
        "configs": (
            {
                "model": "openrouter/deepseek/deepseek-v4-flash",
                "reasoning_effort": "high",
                "output_format": "json",
            },
            {
                "model": "openrouter/deepseek/deepseek-v4-flash-0731",
                "reasoning_effort": "high",
                "output_format": "json",
            },
        ),
        "requires_reported_model": False,
        # OpenRouter returns billed request cost through OpenCode's JSON event
        # stream; the adapter preserves the sum in its ATIF final metrics.
        "cost_reporting": "provider_native",
    },
}
_PUBLIC_PROTOCOL_FILES = (
    "candidate-contract.md",
    "candidate.schema.json",
    "known-failure-modes.md",
)
_TRACE_PROTOCOL_FILES = ("process-contract.md", "search-trace.schema.json")


@dataclass(frozen=True)
class SequentialEpisodeMaterialization:
    root: Path
    task: Path
    manifest: Path
    packet_sha256: str
    episode_index: int
    agent_timeout_seconds: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def resolve_discovery_profile(
    repo_root: Path,
    profile_ref: str,
) -> tuple[Path, Path, dict, str]:
    """Resolve one public text-only discovery profile without arbitrary paths."""

    match = _DISCOVERY_PROFILE_REF.fullmatch(profile_ref)
    if match is None:
        raise ValueError("hypothesis_profile must be FAMILY@MAJOR.MINOR.PATCH:PROFILE_ID")
    family = match["family"]
    version = f"{match['major']}.{match['minor']}.{match['patch']}"
    profile_id = match["profile_id"]
    root = (
        repo_root
        / "foundry"
        / "discovery_profiles"
        / family
        / f"v{match['major']}.{match['minor']}"
    )
    definition_path = root / "profile-set.toml"
    if not definition_path.is_file():
        raise ValueError(f"unknown discovery profile set: {family}@{version}")
    definition = tomllib.loads(definition_path.read_text(encoding="utf-8"))
    if definition.get("id") != f"{family}@{version}":
        raise ValueError("discovery profile set ID does not match its reference")
    if definition.get("status") not in {"experimental", "frozen"}:
        raise ValueError("discovery profile set status is not executable")
    if definition.get("delivery") != "public_text_card_with_seed_abstracts":
        raise ValueError("discovery profile delivery policy is unsupported")
    if profile_id in definition.get("deprecated_profile_ids", []):
        raise ValueError(
            f"discovery profile {profile_ref} is deprecated-before-use and excluded from paper runs"
        )
    if profile_id not in definition.get("profile_ids", []):
        raise ValueError(f"unknown profile ID {profile_id!r} in {family}@{version}")
    visibility = definition.get("visibility")
    if visibility != {
        "executable_prototypes": False,
        "private_scenes": False,
        "gold_answers": False,
        "oracle_outputs": False,
        "review_material": False,
    }:
        raise ValueError("discovery profile visibility policy is unsafe")
    card_path = root / f"{profile_id}.md"
    if not card_path.is_file():
        raise ValueError(f"discovery profile card is missing: {profile_id}")
    return definition_path, card_path, definition, profile_id


def load_sequential_campaign(path: Path) -> dict:
    campaign = tomllib.loads(path.read_text(encoding="utf-8"))
    validate_sequential_campaign(campaign)
    return campaign


def validate_sequential_campaign(campaign: dict) -> None:
    if not _ID.fullmatch(str(campaign.get("id", ""))):
        raise ValueError("invalid campaign id")
    required_top = {
        "id",
        "status",
        "mode",
        "protocol",
        "strategy",
        "seed_set",
        "mechanism_memory",
        "negative_memory",
        "outcome_memory",
        "agent",
        "campaign",
        "task",
        "verification",
        "environment",
        "transition",
        "comparison",
    }
    optional_top = {"archive_policy", "feedback_policy", "feedback_bootstrap"}
    if not required_top <= set(campaign) or not set(campaign) <= required_top | optional_top:
        raise ValueError(
            "sequential campaign keys must contain exactly the required contract plus "
            f"supported controller policies; required={sorted(required_top)}, "
            f"optional={sorted(optional_top)}"
        )
    if campaign["status"] not in {"awaiting_operator_approval", "frozen"}:
        raise ValueError("campaign status is invalid")
    if campaign["mode"] != "discovery":
        raise ValueError("controller-owned foundry campaigns are developmental discovery only")
    if campaign["protocol"] != _SUPPORTED_PROTOCOL:
        raise ValueError(f"controller-owned foundry campaigns require {_SUPPORTED_PROTOCOL}")
    if campaign["strategy"] not in _SUPPORTED_STRATEGIES:
        raise ValueError(f"strategy must be one of {sorted(_SUPPORTED_STRATEGIES)}")
    strategy = _SUPPORTED_STRATEGIES[campaign["strategy"]]
    if campaign["mechanism_memory"] != "known-mechanisms@0.1.0":
        raise ValueError("campaign requires known-mechanisms@0.1.0")
    if campaign["negative_memory"] != "rejected-mechanisms@0.1.0":
        raise ValueError("campaign requires rejected-mechanisms@0.1.0")
    if campaign["outcome_memory"] != "funnel-outcomes@0.1.0":
        raise ValueError("campaign requires funnel-outcomes@0.1.0")
    if "archive_policy" in campaign:
        validate_quality_diversity_policy(campaign["archive_policy"])
    if "feedback_policy" in campaign:
        validate_feedback_policy(campaign["feedback_policy"])
    if "feedback_bootstrap" in campaign:
        bootstrap = campaign["feedback_bootstrap"]
        expected = {"id", "manifest", "manifest_sha256"}
        if not isinstance(bootstrap, dict) or set(bootstrap) != expected:
            raise ValueError(f"feedback_bootstrap keys must be exactly {sorted(expected)}")
        if _ID.fullmatch(str(bootstrap.get("id", ""))) is None:
            raise ValueError("feedback_bootstrap.id is invalid")
        manifest = Path(str(bootstrap.get("manifest", "")))
        if manifest.is_absolute() or ".." in manifest.parts or manifest.suffix != ".json":
            raise ValueError("feedback_bootstrap.manifest must be a repository-relative JSON path")
        digest = str(bootstrap.get("manifest_sha256", ""))
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("feedback_bootstrap.manifest_sha256 must be lowercase SHA-256")
        if "feedback_policy" not in campaign:
            raise ValueError("feedback_bootstrap requires feedback_policy")
        if campaign["feedback_policy"]["id"] not in {
            BRANCHING_POLICY_ID,
            RESILIENT_BRANCHING_POLICY_ID,
            WIRED_RESILIENT_BRANCHING_POLICY_ID,
            EFFORT_AWARE_BRANCHING_POLICY_ID,
        }:
            raise ValueError("feedback_bootstrap requires a branching feedback policy")
    if "archive_policy" in campaign and "feedback_policy" in campaign:
        raise ValueError("quality-diversity and difficulty feedback are separate paper treatments")

    agent = campaign["agent"]
    supported = _SUPPORTED_AGENTS.get(agent.get("adapter"))
    if supported is None:
        raise ValueError("sequential campaigns require a supported clean-recycle adapter")
    adapter = agent["adapter"]
    config = {key: agent.get(key) for key in ("model", "reasoning_effort", "output_format")}
    if config not in supported["configs"]:
        raise ValueError(
            f"{adapter} sequential campaigns require one of "
            f"{list(supported['configs'])}; observed {config}"
        )
    reported_model = agent.get("reported_model")
    if supported["requires_reported_model"]:
        if not isinstance(reported_model, str) or not reported_model.strip():
            raise ValueError(f"{adapter} must declare the reported_model observed in an auth smoke")
        if reported_model == _PENDING_REPORTED_MODEL:
            raise ValueError(
                f"{adapter} still carries the {_PENDING_REPORTED_MODEL} placeholder; "
                "run the exact-model auth smoke and pin the observed model string"
            )
    elif reported_model is not None and reported_model != agent["model"]:
        raise ValueError("reported_model may not contradict a native provider model")
    auxiliary = agent.get("auxiliary_reported_models", [])
    if not isinstance(auxiliary, list) or any(
        not isinstance(item, str) or not item.strip() for item in auxiliary
    ):
        raise ValueError("auxiliary_reported_models must be a list of non-empty strings")
    routing_keys = {"upstream_provider", "allow_provider_fallbacks"}
    if adapter == "openrouter_opencode_foundry:OpenRouterOpenCodeFoundry":
        if routing_keys & set(agent) and not routing_keys <= set(agent):
            raise ValueError("OpenRouter OpenCode routing must pin provider and fallback together")
        if "upstream_provider" in agent and (
            not isinstance(agent["upstream_provider"], str)
            or not agent["upstream_provider"].strip()
            or not isinstance(agent["allow_provider_fallbacks"], bool)
        ):
            raise ValueError("OpenRouter OpenCode routing fields are invalid")
    elif routing_keys & set(agent):
        raise ValueError("OpenRouter routing fields are forbidden for this adapter")
    if strategy.get("cap_free"):
        if "max_turns_per_episode" in agent:
            raise ValueError("cap-free strategies forbid max_turns_per_episode")
    else:
        turn_cap = int(strategy["max_turns"])
        if not 1 <= int(agent.get("max_turns_per_episode", 0)) <= turn_cap:
            raise ValueError(f"max_turns_per_episode must be in [1, {turn_cap}]")
    episode_timeout = int(agent.get("timeout_seconds_per_episode", 0))
    timeout_cap = int(strategy["max_timeout_seconds"])
    if not 60 <= episode_timeout <= timeout_cap:
        raise ValueError(f"timeout_seconds_per_episode must be in [60, {timeout_cap}]")
    deadline_driven = bool(strategy.get("deadline_driven"))
    provider_turn_policy = agent.get("provider_turn_policy", "single_turn")
    continuation_delay = int(agent.get("continuation_delay_seconds", 0))
    if deadline_driven:
        expected_turn_policy = (
            "single_turn" if strategy["mode"] == "episodic_sequential" else "resume_until_timeout"
        )
        if provider_turn_policy != expected_turn_policy:
            raise ValueError(
                f"{campaign['strategy']} requires provider_turn_policy={expected_turn_policy}"
            )
        if provider_turn_policy == "resume_until_timeout":
            if not supported.get("supports_resume_until_timeout", False):
                raise ValueError(f"{adapter} cannot resume one provider session until timeout")
            if not 5 <= continuation_delay <= 120:
                raise ValueError("continuation_delay_seconds must be in [5, 120]")
        elif continuation_delay != 0:
            raise ValueError("single-turn sessions require continuation_delay_seconds=0")
    elif "provider_turn_policy" in agent or "continuation_delay_seconds" in agent:
        raise ValueError("provider-turn controls require a deadline-driven strategy version")

    budget = campaign["campaign"]
    wall = int(budget.get("wall_time_seconds", 0))
    # Long episodic campaigns need wall headroom beyond their agent-time target
    # for repeated clean offline verifier cycles and provider backoff. Wall time
    # is not model time and cannot increase the separately bounded agent budget.
    if not 600 <= wall <= _MAX_CAMPAIGN_WALL_SECONDS:
        raise ValueError(f"wall_time_seconds must be in [600, {_MAX_CAMPAIGN_WALL_SECONDS}]")
    if episode_timeout >= wall:
        raise ValueError("episode timeout must be smaller than campaign wall time")
    if strategy.get("cap_free"):
        if "max_episodes" in budget:
            raise ValueError("cap-free strategies forbid max_episodes")
        if budget.get("episode_limit_policy") != strategy["episode_limit_policy"]:
            raise ValueError(
                f"{campaign['strategy']} requires episode_limit_policy="
                f"{strategy['episode_limit_policy']}"
            )
    else:
        minimum_sessions, maximum_sessions = strategy["session_bounds"]
        sessions = int(budget.get("max_episodes", 0))
        if not minimum_sessions <= sessions <= maximum_sessions:
            raise ValueError(
                f"{campaign['strategy']} max_episodes must be in "
                f"[{minimum_sessions}, {maximum_sessions}]"
            )
    minimum_start = int(budget.get("minimum_start_seconds", 0))
    verifier_reserve = int(budget.get("verifier_reserve_seconds", 0))
    if not 60 <= minimum_start <= episode_timeout:
        raise ValueError("minimum_start_seconds must be in [60, episode timeout]")
    if not 30 <= verifier_reserve < episode_timeout:
        raise ValueError("verifier_reserve_seconds must be in [30, episode timeout)")
    if float(budget.get("soft_cost_limit_usd", 0.0)) <= 0:
        raise ValueError("soft_cost_limit_usd must be positive")
    declared_cost_reporting = budget.get("cost_reporting", "provider_native")
    supported_cost_reporting = supported["cost_reporting"]
    supported_cost_modes = (
        {supported_cost_reporting}
        if isinstance(supported_cost_reporting, str)
        else set(supported_cost_reporting)
    )
    if declared_cost_reporting not in supported_cost_modes:
        requirement = (
            f"cost_reporting={supported_cost_reporting}"
            if isinstance(supported_cost_reporting, str)
            else f"cost_reporting in {sorted(supported_cost_modes)}"
        )
        raise ValueError(
            f"{adapter} requires {requirement}; CLI-reported cost is not this provider's billing"
        )
    completion_keys = {
        "completion_policy",
        "transient_error_policy",
        "retry_backoff_seconds",
        "max_consecutive_provider_failures",
    }
    if deadline_driven:
        agent_time = int(budget.get("agent_time_seconds", 0))
        if not 600 <= agent_time < wall:
            raise ValueError("agent_time_seconds must be in [600, campaign wall time)")
        empty_submission_limit = budget.get("max_consecutive_empty_submissions")
        if empty_submission_limit is not None and (
            isinstance(empty_submission_limit, bool)
            or not isinstance(empty_submission_limit, int)
            or not 1 <= empty_submission_limit <= 10
        ):
            raise ValueError("max_consecutive_empty_submissions must be an integer in [1, 10]")
        if budget.get("completion_policy") != "consume_agent_time_budget":
            raise ValueError(
                "deadline-driven campaigns require completion_policy=consume_agent_time_budget"
            )
        if budget.get("transient_error_policy") != "retry_with_backoff":
            raise ValueError(
                "deadline-driven campaigns require transient_error_policy=retry_with_backoff"
            )
        backoff = budget.get("retry_backoff_seconds")
        if (
            not isinstance(backoff, list)
            or not backoff
            or len(backoff) > 8
            or any(
                isinstance(seconds, bool) or not isinstance(seconds, int) or not 5 <= seconds <= 60
                for seconds in backoff
            )
            or backoff != sorted(backoff)
        ):
            raise ValueError(
                "retry_backoff_seconds must be a nonempty increasing integer list in [5, 60]"
            )
        if not 1 <= int(budget.get("max_consecutive_provider_failures", 0)) <= 10:
            raise ValueError("max_consecutive_provider_failures must be in [1, 10]")
    elif completion_keys & set(budget) or "max_consecutive_empty_submissions" in budget:
        raise ValueError(
            "deadline completion and empty-submission controls require "
            "a deadline-driven strategy version"
        )

    task = campaign["task"]
    minimum = int(task.get("min_candidates", 0))
    if strategy.get("cap_free"):
        if "max_candidates" in task:
            raise ValueError("cap-free strategies forbid max_candidates")
        if task.get("candidate_limit_policy") != strategy["candidate_limit_policy"]:
            raise ValueError(
                f"{campaign['strategy']} requires candidate_limit_policy="
                f"{strategy['candidate_limit_policy']}"
            )
        if not 1 <= minimum <= 10:
            raise ValueError("cap-free strategies require min_candidates in [1, 10]")
        if (
            strategy["mode"] == "episodic_sequential"
            and strategy["candidate_limit_policy"] != "exactly_one_per_session"
        ):
            raise ValueError("episodic sequential requires exactly one candidate per session")
    else:
        maximum = int(task.get("max_candidates", 0))
        lower, upper = strategy["candidate_bounds"]
        if not lower <= minimum <= maximum <= upper:
            raise ValueError(
                f"{campaign['strategy']} candidate limits must satisfy "
                f"{lower} <= min <= max <= {upper}"
            )
        if strategy["mode"] == "episodic_sequential" and (minimum, maximum) != (1, 1):
            raise ValueError("episodic sequential sessions require exactly one candidate")
    if strategy["process_trace_required"]:
        min_proposals = int(task.get("min_logged_proposals", 0))
        ranking_passes = int(task.get("min_ranking_passes", 0))
        if not 1 <= min_proposals <= 100:
            raise ValueError("funnel min_logged_proposals must be in [1, 100]")
        if not 2 <= ranking_passes <= 5:
            raise ValueError("funnel min_ranking_passes must be in [2, 5]")
        if strategy.get("cap_free"):
            if "max_build_starts" in task:
                raise ValueError("cap-free funnel forbids max_build_starts")
            if task.get("proposal_refresh_policy") != "replenish_ranked_queue":
                raise ValueError("cap-free funnel must replenish its ranked proposal queue")
        else:
            max_builds = int(task.get("max_build_starts", 0))
            if not maximum <= max_builds <= 20:
                raise ValueError("funnel max_build_starts must cover max_candidates and be <= 20")
    elif any(
        key in task
        for key in (
            "min_logged_proposals",
            "max_build_starts",
            "min_ranking_passes",
            "proposal_refresh_policy",
        )
    ):
        raise ValueError("funnel process requirements are forbidden for non-funnel strategies")
    verification = campaign["verification"]
    seeds = verification.get("oracle_stress_seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 2
        or len(seeds) != len(set(seeds))
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise ValueError("oracle_stress_seeds must contain distinct nonnegative integers")
    if not 12 <= int(verification.get("oracle_stress_scenes_per_seed", 0)) <= 100:
        raise ValueError("oracle_stress_scenes_per_seed must be in [12, 100]")
    threshold = float(verification.get("semantic_duplicate_threshold", -1))
    if not 0 <= threshold <= 1:
        raise ValueError("semantic_duplicate_threshold must be in [0, 1]")
    transition = campaign["transition"]
    required_transition = {
        "require_submission_valid": True,
        "require_all_mechanical_gates": True,
        "require_registry_distinct": True,
        "require_semantic_review_ready": True,
        "canonical_admission": "human-only",
    }
    if transition != required_transition:
        raise ValueError("controller transition policy differs from the protected contract")
    comparison = campaign["comparison"]
    if not isinstance(comparison, dict):
        raise ValueError("comparison must be a TOML table")
    profile_ref = comparison.get("hypothesis_profile")
    if profile_ref is not None and (
        not isinstance(profile_ref, str) or _DISCOVERY_PROFILE_REF.fullmatch(profile_ref) is None
    ):
        raise ValueError("hypothesis_profile must be FAMILY@MAJOR.MINOR.PATCH:PROFILE_ID")
    campaign_render_runtime(campaign)


def separates_budget_from_infrastructure(campaign: dict) -> bool:
    """Whether this strategy records exhausted budgets as a distinct outcome."""

    return campaign["strategy"] != "sequential@0.1.0"


def strategy_policy(campaign: dict) -> dict:
    """Return a copy of the executable orchestration policy for a campaign."""

    validate_sequential_campaign(campaign)
    return dict(_SUPPORTED_STRATEGIES[campaign["strategy"]])


def _one_candidate_session(campaign: dict, strategy: dict) -> bool:
    """Whether one provider session must emit exactly one candidate.

    This is an episodic unit-of-work boundary, not a campaign portfolio cap.
    Cap-free episodic campaigns repeat fresh sessions until their agent-time
    budget is consumed.
    """

    if strategy.get("cap_free"):
        return strategy["candidate_limit_policy"] == "exactly_one_per_session"
    return int(campaign["task"]["max_candidates"]) == 1


def initial_state(campaign: dict, *, run_id: str, started_at: str) -> dict:
    validate_sequential_campaign(campaign)
    strategy = _SUPPORTED_STRATEGIES[campaign["strategy"]]
    state = {
        "schema_version": _STATE_SCHEMA[campaign["strategy"]],
        "campaign_id": campaign["id"],
        "run_id": run_id,
        "strategy": campaign["strategy"],
        "protocol": campaign["protocol"],
        "status": "running",
        "started_at": started_at,
        "finished_at": None,
        "stop_reason": None,
        "episodes_completed": 0,
        "cumulative_agent_seconds": 0.0,
        "cumulative_verifier_seconds": 0.0,
        "cumulative_cost_usd": 0.0,
        "working_seed": [],
        "failed_outcomes": [],
        "infrastructure_outcomes": [],
        "deviations": [],
    }
    if campaign["strategy"] not in {"sequential@0.1.0", "sequential@0.2.0"}:
        state["orchestration_mode"] = strategy["mode"]
    if profile_ref := campaign["comparison"].get("hypothesis_profile"):
        state["hypothesis_profile"] = profile_ref
    if strategy.get("cap_free"):
        state["candidate_limit_policy"] = strategy["candidate_limit_policy"]
        state["episode_limit_policy"] = strategy["episode_limit_policy"]
    if separates_budget_from_infrastructure(campaign):
        state["budget_incomplete_outcomes"] = []
    if strategy.get("deadline_driven"):
        state["provider_incidents"] = []
        state["budget_assessment"] = None
        state["cumulative_provider_turns"] = 0
    if archive_policy := campaign.get("archive_policy"):
        state["quality_diversity"] = initial_quality_diversity_archive(archive_policy)
    if feedback_policy := campaign.get("feedback_policy"):
        state["difficulty_feedback"] = initial_feedback_state(feedback_policy)
    return state


def expected_reported_models(campaign: dict) -> set[str]:
    """Model strings the provider stream is permitted to attribute to assistant turns.

    A compatible third-party endpoint may echo its own slug, and the CLI may
    route background turns to a smaller alias. The campaign pins every value an
    auth smoke actually observed, so the controller's check stays exact rather
    than being relaxed to a substring or prefix match.
    """

    agent = campaign["agent"]
    reported = agent.get("reported_model") or agent["model"]
    auxiliary = agent.get("auxiliary_reported_models") or []
    return {str(reported), *(str(item) for item in auxiliary)}


def cost_limit_enforced(campaign: dict) -> bool:
    """Whether CLI-reported cost may drive the campaign stopping rule."""

    return campaign["campaign"].get("cost_reporting", "provider_native") == "provider_native"


def automatic_working_seed_eligible(
    *,
    rewards: dict,
    candidate_result: dict,
    verifier_completed: bool = True,
) -> bool:
    return (
        verifier_completed
        and rewards.get("submission_valid") == 1.0
        and candidate_result.get("mechanically_eligible") is True
        and candidate_result.get("registry_distinct") is True
        and candidate_result.get("semantic_review_ready") is True
    )


def _task_spec(
    campaign: dict,
    *,
    episode_index: int,
    timeout_seconds: int,
    reserved_candidate_ids: list[str] | None = None,
) -> str:
    verification = campaign["verification"]
    task = campaign["task"]
    strategy = _SUPPORTED_STRATEGIES[campaign["strategy"]]
    values = {
        "schema_version": "0.1.0",
        "campaign_id": campaign["id"],
        "mode": campaign["mode"],
        "protocol": campaign["protocol"],
        "strategy": campaign["strategy"],
        "seed_set": campaign["seed_set"],
        "episode_index": episode_index,
        "min_candidates": task["min_candidates"],
        "repair_turns": task["repair_turns"],
        "timeout_seconds": timeout_seconds,
        "requires_clean_recycle": True,
        "mechanism_memory": campaign["mechanism_memory"],
        "negative_memory": campaign["negative_memory"],
        "semantic_neighbor_limit": verification["semantic_neighbor_limit"],
        "semantic_duplicate_threshold": verification["semantic_duplicate_threshold"],
        "oracle_stress_seeds": verification["oracle_stress_seeds"],
        "oracle_stress_scenes_per_seed": verification["oracle_stress_scenes_per_seed"],
    }
    if strategy.get("cap_free"):
        values["candidate_limit_policy"] = task["candidate_limit_policy"]
        values["episode_limit_policy"] = campaign["campaign"]["episode_limit_policy"]
    else:
        values["max_candidates"] = task["max_candidates"]
        values["max_turns"] = campaign["agent"]["max_turns_per_episode"]
    if campaign["strategy"] not in {"sequential@0.1.0", "sequential@0.2.0"}:
        values["orchestration_mode"] = strategy["mode"]
        values["process_trace_required"] = strategy["process_trace_required"]
    if strategy.get("deadline_driven"):
        values["schema_version"] = strategy["task_spec_version"]
        values["provider_turn_policy"] = campaign["agent"]["provider_turn_policy"]
        values["continuation_delay_seconds"] = campaign["agent"]["continuation_delay_seconds"]
        values["campaign_completion_policy"] = campaign["campaign"]["completion_policy"]
        values["reserved_candidate_ids"] = sorted(reserved_candidate_ids or [])
    if strategy.get("paper_audit"):
        values["phase_audit_required"] = True
    if strategy["process_trace_required"]:
        values["min_logged_proposals"] = task["min_logged_proposals"]
        values["min_ranking_passes"] = task["min_ranking_passes"]
        if strategy.get("cap_free"):
            values["proposal_refresh_policy"] = task["proposal_refresh_policy"]
        else:
            values["max_build_starts"] = task["max_build_starts"]
    lines = []
    for key, value in values.items():
        if isinstance(value, str):
            lines.append(f"{key} = {json.dumps(value)}")
        elif isinstance(value, bool):
            lines.append(f"{key} = {str(value).lower()}")
        elif isinstance(value, list):
            lines.append(f"{key} = [{', '.join(json.dumps(item) for item in value)}]")
        else:
            lines.append(f"{key} = {value}")
    return "\n".join(lines) + "\n"


def _task_toml(campaign: dict, *, timeout_seconds: int) -> str:
    environment = campaign["environment"]
    strategy = _SUPPORTED_STRATEGIES[campaign["strategy"]]
    mode = strategy["mode"]
    one_candidate = _one_candidate_session(campaign, strategy)
    historical_sequential = campaign["strategy"] in {
        "sequential@0.1.0",
        "sequential@0.2.0",
    }
    task_slug = "build-one-question-world" if one_candidate else "build-question-world-session"
    task_description = (
        "Build one executable latent-z visual-question candidate in a sequential campaign"
        if historical_sequential
        else "Build one executable latent-z visual-question candidate"
        if one_candidate
        else "Build an executable portfolio of latent-z visual-question candidates"
    )
    task_version = (
        "0.1.0"
        if historical_sequential
        else "0.4.0"
        if strategy.get("cap_free")
        else "0.3.0"
        if strategy.get("deadline_driven")
        else "0.2.0"
    )
    verifier_timeout = (
        900.0 if historical_sequential else float(campaign["campaign"]["verifier_reserve_seconds"])
    )
    keywords = (
        '["vlm", "foundry", "sequential", "world-generation", "verification"]'
        if historical_sequential
        else f'["vlm", "foundry", {json.dumps(mode)}, "world-generation", "verification"]'
    )
    difficulty = (
        "Requires one complete executable world, an independent pixel oracle, "
        "and auditable evidence."
        if historical_sequential
        else "Requires complete executable worlds, independent pixel oracles, "
        "agent-authored public self-checks, and auditable evidence."
    )
    verification = (
        "A clean offline verifier executes protected gates before the controller "
        "updates campaign state."
    )
    return f"""schema_version = "1.4"
artifacts = ["/logs/artifacts/process", "/logs/artifacts/submission"]

[task]
name = "question-foundry/{task_slug}"
version = "{task_version}"
description = {json.dumps(task_description)}
keywords = {keywords}

[metadata]
category = "agentic-research"
difficulty_explanation = {json.dumps(difficulty)}
verification_explanation = {json.dumps(verification)}

[agent]
timeout_sec = {float(timeout_seconds):.1f}

[verifier]
timeout_sec = {verifier_timeout:.1f}

[environment]
allow_internet = true
build_timeout_sec = 600.0
cpus = {int(environment["cpus"])}
memory_mb = {int(environment["memory_mb"])}
storage_mb = {int(environment["storage_mb"])}
gpus = 0
"""


def _seed_definition(repo_root: Path, seed_id: str) -> tuple[Path, dict]:
    name, separator, version = seed_id.partition("@")
    if not separator or not name or not version:
        raise ValueError("seed set must use name@version")
    for path in sorted((repo_root / "foundry/seed_sets").glob(f"{name}_v*.toml")):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        if data.get("id") == name and data.get("version") == version:
            return path, data
    raise FileNotFoundError(f"seed set not found: {seed_id}")


def _candidate_metadata(path: Path) -> dict:
    return json.loads((path / "candidate.json").read_text(encoding="utf-8"))


def _failed_candidate_id(path: Path, verdict: dict) -> str:
    """Resolve a rejected candidate ID from protected verifier evidence.

    Rejected submissions are allowed to fail the public candidate contract. In
    particular, ``candidate.json`` can be malformed. The controller must still
    preserve and remember that rejection without trusting or reparsing the
    malformed agent-authored metadata on the next episode.
    """

    try:
        metadata = _candidate_metadata(path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        candidate_id = verdict.get("candidate_id")
        if not isinstance(candidate_id, str) or _ID.fullmatch(candidate_id) is None:
            raise ValueError(
                f"failed candidate verdict has invalid candidate_id: {candidate_id!r}"
            ) from None
        return candidate_id
    candidate_id = metadata.get("candidate_id")
    if not isinstance(candidate_id, str) or _ID.fullmatch(candidate_id) is None:
        raise ValueError(f"failed candidate metadata has invalid candidate_id: {candidate_id!r}")
    return candidate_id


def _failed_candidate_summary(path: Path, verdict: dict) -> str:
    """Return useful negative memory even when contract metadata is invalid."""

    candidate_id = _failed_candidate_id(path, verdict)
    try:
        return candidate_text(_candidate_metadata(path))
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
        return f"Candidate {candidate_id} did not provide valid candidate-contract metadata."


def _extended_memories(
    *,
    repo_root: Path,
    accepted_candidates: list[Path],
    failed_candidates: list[Path],
    failed_verdicts: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    known = [
        json.loads(line)
        for line in (repo_root / "registry/historical/mechanism-fingerprints-v0.1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    negative = [
        json.loads(line)
        for line in (repo_root / "registry/negative/rejected-mechanisms-v0.1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    registry = [
        json.loads(line)
        for line in (repo_root / "registry/historical/reference-worlds.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    funnel_outcomes = [
        json.loads(line)
        for line in (repo_root / "registry/outcomes/funnel-outcomes-v0.1.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    for outcome in funnel_outcomes:
        memory_id = outcome["outcome_id"]
        if outcome["mechanically_eligible"]:
            known.append(
                {
                    "schema_version": "mechanism-memory-0.1.0",
                    "world_id": memory_id,
                    "fingerprint": outcome["mechanism_fingerprint"],
                    "summary": (
                        f"Unadmitted funnel outcome {outcome['candidate_id']}: {outcome['summary']}"
                    ),
                }
            )
        else:
            failed_gates = outcome.get("failed_gates") or []
            negative.append(
                {
                    "schema_version": "negative-memory-0.1.0",
                    "proposal_id": memory_id,
                    "reason_code": (
                        "protected_gate_failure_" + "_".join(failed_gates[:3])
                        if failed_gates
                        else "unadmitted_funnel_outcome"
                    ),
                    "canonical_neighbors": [],
                    "summary": (
                        f"Funnel outcome {outcome['candidate_id']}: {outcome['summary']}; "
                        f"failed gates: {', '.join(failed_gates) if failed_gates else 'none'}."
                    ),
                }
            )
        signature = outcome["task_signature"]
        registry.append(
            {
                "name": memory_id,
                "decision_var": signature["decision_var"],
                "structure": signature["structure"],
                "decision_type": signature["decision_type"],
                "margin": "unadmitted funnel outcome; inspect source packet",
                "quarantine": "unadmitted outcome; inspect source packet",
                "cell": [
                    signature["decision_var"],
                    signature["structure"],
                    signature["decision_type"],
                ],
                "status": "unadmitted_funnel_outcome",
                "sprint": outcome["campaign_id"],
                "note": outcome["summary"],
            }
        )

    for candidate_path in accepted_candidates:
        metadata = _candidate_metadata(candidate_path)
        row = candidate_as_memory(metadata)
        row["schema_version"] = "mechanism-memory-0.1.0"
        known.append(row)
        signature = require_task_signature(
            metadata,
            context=f"accepted candidate {metadata.get('candidate_id', candidate_path.name)}",
        )
        registry.append(
            {
                "name": metadata["candidate_id"],
                "decision_var": signature["decision_var"],
                "structure": signature["structure"],
                "decision_type": signature["decision_type"],
                "margin": "campaign-local candidate; inspect immutable bundle",
                "quarantine": metadata["boundary"]["quarantine"],
                "cell": [
                    signature["decision_var"],
                    signature["structure"],
                    signature["decision_type"],
                ],
                "status": "campaign_working_seed",
                "sprint": "sequential",
                "note": candidate_text(metadata),
            }
        )

    if len(failed_candidates) != len(failed_verdicts):
        raise ValueError("failed candidate paths and verdicts differ in length")
    for candidate_path, verdict in zip(failed_candidates, failed_verdicts, strict=True):
        candidate_id = _failed_candidate_id(candidate_path, verdict)
        failed_gates = sorted(
            gate for gate, passed in (verdict.get("gates") or {}).items() if not passed
        )
        semantic = (verdict.get("notes") or {}).get("semantic") or {}
        neighbors = [
            str(item.get("world_id"))
            for item in semantic.get("known_neighbors") or []
            if item.get("world_id")
        ]
        negative.append(
            {
                "schema_version": "negative-memory-0.1.0",
                "proposal_id": candidate_id,
                "reason_code": (
                    "protected_gate_failure_" + "_".join(failed_gates[:3])
                    if failed_gates
                    else "protected_verifier_rejection"
                ),
                "canonical_neighbors": neighbors[:5],
                "summary": (
                    f"{_failed_candidate_summary(candidate_path, verdict)} Protected failures: "
                    f"{', '.join(failed_gates) if failed_gates else 'none recorded'}."
                ),
            }
        )
    return known, negative, registry


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _validate_task_spec_schema_binding(task_spec: str, schema_path: Path) -> None:
    """Fail closed when a prospective strategy ships the wrong task-spec schema.

    Frozen v0.4 packets exposed a v0.3 orchestration schema whose episodic branch
    accepted only ``episodic-sequential@0.3.0``. The schema was not executed by
    the builder, so the pilot completed, but the packet contradicted itself.
    Prospective strategy-local schemas bind every constant policy field and the
    complete top-level key set before a packet can be materialized.
    """

    values = tomllib.loads(task_spec)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(properties, dict) or not isinstance(required, list):
        raise ValueError("task-spec schema must declare properties and required keys")
    if schema.get("additionalProperties") is not False:
        raise ValueError("task-spec schema must reject additional properties")
    if set(values) != set(required) or set(required) != set(properties):
        raise ValueError("task-spec keys differ from its strategy-local schema")
    for key, rule in properties.items():
        if isinstance(rule, dict) and "const" in rule and rule["const"] != values.get(key):
            raise ValueError(f"task-spec {key} differs from its strategy-local schema")


def materialize_sequential_episode(
    *,
    repo_root: Path,
    campaign_path: Path,
    output_root: Path,
    state: dict,
    episode_index: int,
    timeout_seconds: int,
    accepted_candidates: list[Path],
    failed_candidates: list[Path],
    failed_verdicts: list[dict],
    source_revision: str,
    source_dirty: bool,
    include_oracle_fixture: bool = False,
    harbor_task_namespace: str | None = None,
    feedback_artifact_root: Path | None = None,
) -> SequentialEpisodeMaterialization:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"episode packet target exists: {output_root}")
    campaign = load_sequential_campaign(campaign_path)
    strategy = _SUPPORTED_STRATEGIES[campaign["strategy"]]
    if state.get("campaign_id") != campaign["id"] or state.get("strategy") != campaign["strategy"]:
        raise ValueError("campaign state does not match campaign")
    if episode_index != int(state.get("episodes_completed", -1)) + 1:
        raise ValueError("episode index is not the next controller state")
    configured_timeout = int(campaign["agent"]["timeout_seconds_per_episode"])
    if not 60 <= timeout_seconds <= configured_timeout:
        raise ValueError("episode timeout is outside the configured bound")

    source_dataset = repo_root / "harbor/datasets/foundry-builder-v1"
    historical_sequential = campaign["strategy"] in {
        "sequential@0.1.0",
        "sequential@0.2.0",
    }
    dataset = output_root / (
        "sequential-builder-v1" if historical_sequential else "orchestration-builder-v1"
    )
    shutil.copytree(
        source_dataset,
        dataset,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )
    original_task = dataset / "build-question-worlds"
    task_name = (
        "build-one-question-world"
        if _one_candidate_session(campaign, strategy)
        else "build-question-world-session"
    )
    if harbor_task_namespace is not None:
        if _ID.fullmatch(harbor_task_namespace) is None:
            raise ValueError("Harbor task namespace must be a valid campaign ID")
        task_name = f"{task_name}--{harbor_task_namespace}"
    task = dataset / task_name
    original_task.rename(task)
    override_root = repo_root / "harbor/datasets/sequential-builder-v1"
    strategy_root = repo_root / "foundry/strategies" / strategy["directory"]
    protocol_root = repo_root / "foundry/protocols/question-world/v0.4"
    starter = task / "environment/starter"
    tests = task / "tests"

    dataset_readme = (
        override_root / "README.md"
        if historical_sequential
        else repo_root / "harbor/datasets/orchestration-builder-v1/README.md"
    )
    _copy(dataset_readme, dataset / "README.md")
    _copy(strategy_root / "instruction.md", task / "instruction.md")
    render_runtime = configure_task_render_runtime(
        task=task,
        campaign=campaign,
        capability_source=repo_root / "foundry/render_runtimes/latex-tikz-v0.1.md",
    )
    _copy(strategy_root / "AGENTS.md", starter / "AGENTS.md")
    _copy(strategy_root / "strategy.toml", starter / "STRATEGY.toml")
    _copy(strategy_root / "episode-contract.md", starter / "strategy/episode-contract.md")
    if historical_sequential:
        _copy(strategy_root / "brief.md", starter / "protocol/brief.md")
        _copy(strategy_root / "protocol.toml", starter / "protocol/protocol.toml")
        _copy(strategy_root / "public-gates.md", starter / "protocol/public-gates.md")
        _copy(
            strategy_root / "task-spec.schema.json",
            starter / "strategy/task-spec.schema.json",
        )
    else:
        for name in ("brief.md", "protocol.toml", "public-gates.md"):
            _copy(protocol_root / name, starter / "protocol" / name)
        if strategy.get("strategy_local_task_spec_schema"):
            task_spec_schema = strategy_root / "task-spec.schema.json"
        else:
            orchestration_schema = (
                "v0.3"
                if strategy.get("cap_free")
                else "v0.2"
                if strategy.get("deadline_driven")
                else "v0.1"
            )
            task_spec_schema = (
                repo_root
                / "foundry/strategies/orchestration"
                / orchestration_schema
                / "task-spec.schema.json"
            )
        _copy(task_spec_schema, starter / "strategy/task-spec.schema.json")
        _copy(
            (
                strategy_root / "tools/check_submission_envelope.py"
                if strategy.get("paper_audit")
                else repo_root / "foundry/tools/check_submission_envelope.py"
            ),
            starter / "tools/check_submission_envelope.py",
        )
        _copy(
            (
                strategy_root / "tools/run_public_candidate_gate.py"
                if strategy.get("paper_audit")
                else repo_root / "foundry/tools/run_public_candidate_gate.py"
            ),
            starter / "tools/run_public_candidate_gate.py",
        )
        if strategy.get("paper_audit"):
            for name in ("phase_audit.py", "record_candidate_phase.py"):
                _copy(strategy_root / "tools" / name, starter / "tools" / name)
        protected_tools = (
            repo_root / "harbor/datasets/foundry-builder-v1/build-question-worlds/tests"
        )
        for name in (
            "candidate_gatekeeper.py",
            "probe_candidate.py",
            "semantic_novelty.py",
        ):
            _copy(protected_tools / name, starter / "tools" / name)
        (starter / "scratch").mkdir(parents=True, exist_ok=True)
    for name in _PUBLIC_PROTOCOL_FILES:
        _copy(protocol_root / name, starter / "protocol" / name)
    profile_ref = campaign["comparison"].get("hypothesis_profile")
    profile_record = None
    protected_profile_memory = None
    if profile_ref is not None:
        definition_path, card_path, definition, profile_id = resolve_discovery_profile(
            repo_root,
            profile_ref,
        )
        _copy(definition_path, starter / "discovery_profile/PROFILE_SET.toml")
        _copy(card_path, starter / "DISCOVERY_PROFILE.md")
        protected_novelty = definition.get("protected_novelty")
        if protected_novelty is not None:
            expected = {
                "policy": "deterministic_source_clone_rules",
                "visibility": "verifier_only",
            }
            if not isinstance(protected_novelty, dict):
                raise ValueError("protected_novelty must be a table")
            memory_name = protected_novelty.get("memory_file")
            observed = {key: protected_novelty.get(key) for key in expected}
            if (
                observed != expected
                or not isinstance(memory_name, str)
                or Path(memory_name).name != memory_name
            ):
                raise ValueError("protected discovery-profile novelty policy is invalid")
            protected_profile_memory = definition_path.parent / memory_name
            if not protected_profile_memory.is_file():
                raise ValueError("protected discovery-profile source memory is missing")
        instruction_path = task / "instruction.md"
        instruction_path.write_text(
            "# Profile-guided discovery treatment\n\n"
            "Read `/workspace/DISCOVERY_PROFILE.md` before proposing ideas. "
            "Treat it as the required discovery direction and its examples as "
            "anti-cloning inspiration, not templates. The candidate must remain "
            "new, executable, pixel-answerable, and subject to every unchanged "
            "gate below.\n\n" + instruction_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        profile_record = {
            "reference": profile_ref,
            "profile_id": profile_id,
            "profile_set_status": definition["status"],
            "delivery": definition["delivery"],
            "card_sha256": _sha256(card_path),
            "definition_sha256": _sha256(definition_path),
        }
        if protected_profile_memory is not None:
            profile_record["protected_novelty_policy"] = protected_novelty["policy"]
            profile_record["protected_memory_sha256"] = _sha256(protected_profile_memory)
    archive_target = None
    if archive_policy := campaign.get("archive_policy"):
        archive = state.get("quality_diversity")
        if not isinstance(archive, dict):
            raise ValueError("quality-diversity campaign state has no archive")
        archive_target = select_quality_diversity_target(
            archive_policy,
            archive,
            episode_index=episode_index,
        )
        (starter / "DIVERSITY_TARGET.json").write_text(
            json.dumps(archive_target, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        descriptor_lines = "\n".join(
            f"- `{axis}`: `{value}`" for axis, value in archive_target["descriptor"].items()
        )
        descriptor_definitions = "\n".join(
            f"- **{axis}**: {description}"
            for axis, description in archive_target.get("descriptor_definitions", {}).items()
        )
        if descriptor_definitions:
            descriptor_definitions = (
                "\n\nThe target meanings are operational, not self-declared:\n"
                + descriptor_definitions
                + "\n\n"
                + archive_target.get("attestation_note", "")
            )
        steering_recipes = "\n".join(
            f"- **{axis} recipe**: {description}"
            for axis, description in archive_target.get("steering_recipes", {}).items()
        )
        if steering_recipes:
            history = archive_target["prior_target_history"]
            observed = history.get("observed_cell_counts") or {}
            observed_text = (
                ", ".join(f"`{key}` × {value}" for key, value in sorted(observed.items()))
                if observed
                else "none yet"
            )
            steering_recipes = (
                "\n\n## Operational design recipe\n\n"
                + steering_recipes
                + "\n\n"
                + "Before coding, predict which ablated regions should alter the pixel oracle, "
                "then make that prediction true through the renderer and inverse arm. Do not "
                "fake the requested cell by renaming metadata.\n\n"
                + "## Prior attempts at this target\n\n"
                + f"Completed: `{history['completed_count']}`; target matches: "
                + f"`{history['target_match_count']}`; consecutive misses: "
                + f"`{history['consecutive_misses']}`; observed cells: {observed_text}.\n\n"
                + archive_target.get("selection_note", "")
            )
        cell_recipe = ""
        if recipe := archive_target.get("cell_recipe"):
            occupied = archive_target.get("occupied_cells_to_avoid") or []
            occupied_text = (
                ", ".join(f"`{item['cell_key']}` × {item['occupancy']}" for item in occupied)
                if occupied
                else "none yet"
            )
            cell_recipe = (
                "\n\n## Joint cell recipe\n\n"
                f"- **Construct**: {recipe['construct']}\n"
                f"- **Avoid**: {recipe['avoid']}\n"
                "- **Self-check before coding**: sketch the regions whose removal should change "
                "or prevent the inverse answer. Confirm that their joint scale and connectivity "
                "match both requested axes; metadata labels do not count.\n\n"
                f"Already occupied off-target cells to avoid repeating: {occupied_text}."
            )
        preview_instructions = ""
        if preview := archive_target.get("public_preview"):
            _copy(
                repo_root
                / "harbor/datasets/foundry-builder-v1/build-question-worlds/tests"
                / "qd_attestation.py",
                starter / "tools/preview_qd_descriptor.py",
            )
            preview_instructions = (
                "\n\n## Developmental descriptor preview\n\n"
                "After rendering the candidate gallery and before the final envelope check, "
                "run the public descriptor preview on your own gallery:\n\n"
                "```bash\n"
                "candidate=/logs/artifacts/submission/candidates/YOUR_ID\n"
                "python /workspace/tools/preview_qd_descriptor.py \\\n"
                '  --candidate "$candidate" \\\n'
                '  --dataset "$candidate/evidence" \\\n'
                "  --out /workspace/scratch/qd-preview.json \\\n"
                f"  --attestation-id {preview['attestation_id']}\n"
                "```\n\n"
                f"Proceed only when the preview is `complete`, its descriptor exactly matches "
                f"`{archive_target['cell_key']}`, its nonempty-scene fraction is at least "
                f"`{preview['minimum_nonempty_scene_fraction']}`, and its descriptor stability "
                f"is at least `{preview['minimum_descriptor_stability']}`. If it misses, change "
                "the renderer, sampling margins, or inverse mechanism—not metadata—and preview "
                "again. The preview is development guidance only: the controller reruns the "
                "same measurement offline on unseen stress renders after destroying the "
                "networked session.\n"
            )
        (starter / "DIVERSITY_TARGET.md").write_text(
            "# Controller-selected diversity target\n\n"
            f"Policy: `{archive_policy['id']}`  \n"
            f"Cell: `{archive_target['cell_key']}`  \n"
            f"Current occupancy: `{archive_target['occupancy_before']}`\n\n"
            f"{descriptor_lines}\n\n"
            f"{descriptor_definitions}\n\n"
            f"{steering_recipes}\n\n"
            f"{cell_recipe}\n\n"
            f"{preview_instructions}\n\n"
            "Design a genuinely new executable visual mechanism in this niche. "
            "The descriptor is a steering target, not a field you can set or permission "
            "to relabel a near-duplicate. All protected validity gates remain unchanged. The "
            "controller, not the builder, decides archive membership after offline "
            "verification.\n",
            encoding="utf-8",
        )
        instruction_path = task / "instruction.md"
        instruction_path.write_text(
            "# Diversity-directed treatment\n\n"
            "Read `/workspace/DIVERSITY_TARGET.md` before designing the candidate. "
            "Target its requested pixel-oracle support-behavior niche while remaining "
            "semantically new. "
            "Do not claim archive admission; it is a controller decision made only "
            "after protected verification.\n\n" + instruction_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    feedback_history = None
    reasoning_assignment = None
    if feedback_policy := campaign.get("feedback_policy"):
        feedback_state = state.get("difficulty_feedback")
        if not isinstance(feedback_state, dict):
            raise ValueError("difficulty-feedback campaign state is missing")
        feedback_history = public_feedback_history(feedback_state)
        builder_view = builder_feedback_history(
            feedback_state,
            attempt_limit=int(feedback_policy.get("builder_attempt_memory_limit", 5)),
            samples_per_attempt=int(feedback_policy.get("builder_samples_per_attempt", 1)),
        )
        public_records = builder_view["records"]
        easy_summaries = builder_view["easy_summaries"]
        if feedback_policy["id"] in {
            BRANCHING_POLICY_ID,
            WIRED_RESILIENT_BRANCHING_POLICY_ID,
            EFFORT_AWARE_BRANCHING_POLICY_ID,
        }:
            reasoning_assignment = reasoning_branch_assignment(
                feedback_policy, feedback_state, episode_index
            )
            assigned_parent = reasoning_assignment["parent_candidate_id"]
            if assigned_parent != "initial_profile" and not any(
                row["candidate_id"] == assigned_parent for row in public_records
            ):
                parent_record = next(
                    row for row in feedback_history if row["candidate_id"] == assigned_parent
                )
                public_records = [*public_records, parent_record]
        if feedback_policy["id"] in HYPOTHESIS_POLICY_IDS:
            version = (
                "v0.9"
                if feedback_policy["id"] == EFFORT_AWARE_BRANCHING_POLICY_ID
                else "v0.8"
                if feedback_policy["id"] == WIRED_RESILIENT_BRANCHING_POLICY_ID
                else "v0.6"
                if feedback_policy["id"] == BRANCHING_POLICY_ID
                else "v0.5"
                if feedback_policy["id"] == SYNTHESIS_POLICY_ID
                else "v0.4"
            )
            policy_root = repo_root / f"foundry/controller_policies/difficulty-feedback/{version}"
            _copy(
                policy_root / "difficulty-hypothesis.schema.json",
                starter / "DIFFICULTY_HYPOTHESIS_SCHEMA.json",
            )
            # Work on a detached public view: adding builder-local image paths must
            # never mutate the append-only controller state.
            public_records = json.loads(json.dumps(public_records))
            raw_by_id = {row["candidate_id"]: row for row in feedback_state["candidate_feedback"]}
            if public_records and feedback_artifact_root is None:
                raise ValueError("difficulty-feedback 0.4 requires its artifact root")
            artifact_root = (
                feedback_artifact_root.resolve() if feedback_artifact_root is not None else None
            )
            for row in public_records:
                raw = raw_by_id[row["candidate_id"]]
                row["candidate_code_path"] = (
                    f"/workspace/campaign/working_seed/{row['candidate_id']}"
                )
                result_root = artifact_root / raw["artifact_path"]
                sample_set = json.loads(
                    (result_root / "sample-set.json").read_text(encoding="utf-8")
                )
                if len(sample_set["samples"]) != len(raw["sample_responses"]):
                    raise ValueError("difficulty-feedback sample/image count differs")
                selected_by_id = {
                    sample["sample_id"]: selected
                    for selected, sample in zip(
                        sample_set["samples"], raw["sample_responses"], strict=True
                    )
                }
                for sample in row["sample_responses"]:
                    is_failed = any(
                        response.get("status") == "scored" and response.get("correct") is False
                        for response in sample["responses"]
                    )
                    expose_image = is_failed
                    image_field = "failed_image_path"
                    image_directory = "hard_examples"
                    if feedback_policy["id"] in SYNTHESIS_POLICY_IDS:
                        expose_image = (
                            feedback_policy["builder_image_access"] == "evaluated_samples"
                        )
                        image_field = "feedback_image_path"
                        image_directory = "evaluated_samples"
                    if not expose_image:
                        continue
                    selected = selected_by_id[sample["sample_id"]]
                    source = Path(selected["image_path"]).resolve()
                    if not source.is_relative_to(artifact_root) or not source.is_file():
                        raise ValueError("difficulty-feedback image escaped the run artifact root")
                    relative = (
                        Path("difficulty_feedback")
                        / image_directory
                        / row["candidate_id"]
                        / f"{sample['sample_id']}.png"
                    )
                    _copy(source, starter / relative)
                    sample[image_field] = f"/workspace/{relative.as_posix()}"
        feedback_payload = {
            "schema_version": (
                "difficulty-feedback-public-history-0.9.0"
                if feedback_policy["id"] == EFFORT_AWARE_BRANCHING_POLICY_ID
                else "difficulty-feedback-public-history-0.8.0"
                if feedback_policy["id"] == WIRED_RESILIENT_BRANCHING_POLICY_ID
                else "difficulty-feedback-public-history-0.6.0"
                if feedback_policy["id"] == BRANCHING_POLICY_ID
                else "difficulty-feedback-public-history-0.5.0"
                if feedback_policy["id"] == SYNTHESIS_POLICY_ID
                else "difficulty-feedback-public-history-0.4.0"
                if feedback_policy["id"] == DIFFICULTY_HYPOTHESIS_POLICY_ID
                else "difficulty-feedback-public-history-0.3.0"
                if feedback_policy["id"] in RATIONALE_POLICY_IDS
                else "difficulty-feedback-public-history-0.2.0"
            ),
            "policy_id": feedback_policy["id"],
            "visibility": feedback_policy["feedback_visibility"],
            "valid_candidate_count": len(feedback_history),
            "hard_seed_count": len(feedback_state["hard_seed"]),
            "records": public_records,
            "easy_summaries": easy_summaries,
            "omitted_easy_record_count": builder_view.get("omitted_easy_record_count", 0),
            "omitted_record_count": builder_view.get("omitted_record_count", 0),
            "latest_complete_candidate_id": builder_view.get(
                "latest_complete_candidate_id", "initial_profile"
            ),
            "builder_image_access": feedback_policy.get("builder_image_access", "legacy"),
        }
        if reasoning_assignment is not None:
            feedback_payload["reasoning_assignment"] = reasoning_assignment
        (starter / "DIFFICULTY_FEEDBACK.json").write_text(
            json.dumps(feedback_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        history_lines = []
        for row in public_records:
            history_lines.append(
                f"- `{row['candidate_id']}`: protected-valid; feedback "
                f"`{row['feedback_status']}`; aggregate accuracy "
                f"`{row['aggregate_accuracy']}`; hard-seed "
                f"`{row['hard_seed_eligible']}`"
            )
            if code_path := row.get("candidate_code_path"):
                history_lines.append(
                    "  - Prior implementation: "
                    f"`{code_path}` (inspect renderer, prompts, oracle, verifier, and tests)."
                )
            if hypothesis := row.get("difficulty_hypothesis"):
                history_lines.extend(
                    [
                        f"  - Builder predicted evaluator accuracy: "
                        f"`{hypothesis['predicted_evaluator_accuracy']}`.",
                        f"  - Intended visual computation: "
                        f"{hypothesis['intended_visual_computation']}",
                        f"  - Anticipated failure: {hypothesis['anticipated_model_failure']}",
                        f"  - Causal mutation: {hypothesis['causal_mutation_from_prior']}",
                        f"  - Non-artifact check: {hypothesis['non_artifact_check']}",
                    ]
                )
                if reasoning_target := hypothesis.get("reasoning_target"):
                    history_lines.extend(
                        [
                            f"  - Reasoning target: `{reasoning_target}`.",
                            "  - Reasoning departure: "
                            f"{hypothesis['reasoning_departure_from_archive']}",
                        ]
                    )
            for sample in row.get("sample_responses", []):
                options = ", ".join(f"`{item}`" for item in sample["candidates"])
                history_lines.append(
                    f"  - **{sample['sample_id']}** — {sample['question']} Options: {options}."
                )
                if failed_image := sample.get("failed_image_path"):
                    history_lines.append(f"    - Failed image: `{failed_image}`")
                if feedback_image := sample.get("feedback_image_path"):
                    history_lines.append(f"    - Evaluated image: `{feedback_image}`")
                if oracle_answer := sample.get("oracle_answer"):
                    history_lines.append(f"    - Pixel-oracle answer: `{oracle_answer}`")
                for response in sample["responses"]:
                    if response["status"] == "scored":
                        history_lines.append(
                            "    - "
                            f"`{response['evaluator']}` predicted "
                            f"`{response['prediction']}` with confidence "
                            f"`{response['confidence']}` — "
                            f"**{'correct' if response['correct'] else 'incorrect'}**."
                        )
                        if rationale := response.get("rationale"):
                            history_lines.append(f"      - Evaluator rationale: {rationale}")
                        if effort := response.get("solver_effort"):
                            history_lines.append(
                                "      - Solver effort proxy: "
                                f"{effort['reasoning_tokens']} reasoning / "
                                f"{effort['completion_tokens']} completion tokens; "
                                f"{effort['completion_budget_fraction']} of the configured "
                                f"{effort['completion_budget_tokens']}-token ceiling; "
                                f"{effort['rationale_words']} rationale words."
                            )
                    else:
                        history_lines.append(
                            f"    - `{response['evaluator']}` response status: "
                            f"`{response['status']}`."
                        )
        if easy_summaries:
            history_lines.append("\n## Recently solved hypotheses (compressed)\n")
            for row in easy_summaries:
                line = (
                    f"- `{row['candidate_id']}`: aggregate accuracy `{row['aggregate_accuracy']}`"
                )
                if hypothesis := row.get("difficulty_hypothesis"):
                    line += (
                        "; predicted "
                        f"`{hypothesis['predicted_evaluator_accuracy']}`; anticipated failure: "
                        f"{hypothesis['anticipated_model_failure']}"
                    )
                history_lines.append(line)
        history_text = "\n".join(history_lines) if history_lines else "- No prior candidates yet."
        (starter / "DIFFICULTY_FEEDBACK.md").write_text(
            "# Protected-valid question difficulty feedback\n\n"
            "Every listed candidate passed the unchanged inverse-arm and mechanical "
            "gates. Accuracy is a separate, cost-bounded OpenRouter measurement on "
            "fresh deterministic scenes. A solved candidate remains scientifically "
            "valid but is not a hard seed. Do not optimize ambiguity, illegible cues, "
            "or renderer defects. Structured evaluator predictions, outcomes, and any "
            "policy-authorized concise rationales are shown below; gold labels, opaque "
            "provider reasoning, and raw provider payloads remain controller-side. Policy "
            "0.9 additionally shows provider-reported reasoning/completion token counts and "
            "public-rationale length as auxiliary solver-effort proxies. These values are "
            "not correctness labels and are not assumed comparable across providers. Hard "
            "records are ranked first. Policies 0.5 and 0.6 retain a bounded mixture of "
            "the lowest-accuracy and most recent attempts so falsified hypotheses remain "
            "actionable.\n\n"
            f"{history_text}\n",
            encoding="utf-8",
        )
        instruction_path = task / "instruction.md"
        if feedback_policy["id"] in {
            BRANCHING_POLICY_ID,
            WIRED_RESILIENT_BRANCHING_POLICY_ID,
            EFFORT_AWARE_BRANCHING_POLICY_ID,
        }:
            image_instruction = (
                "Inspect every referenced evaluated image with the image-viewing tool and "
                "connect its visible evidence to the evaluator rationale."
                if feedback_policy["builder_image_access"] == "evaluated_samples"
                else "No evaluator image is supplied in this arm. Work from prior source, "
                "inverse-oracle code, and textual evaluator records; do not reconstruct the "
                "exact evaluator rasters."
            )
            target = reasoning_assignment["reasoning_target"]
            parent = reasoning_assignment["parent_candidate_id"]
            effort_instruction = (
                "For newly evaluated attempts, `solver_effort` reports completion tokens, "
                "provider-reported reasoning tokens, completion-budget utilization, and "
                "public-rationale length. Use these only as auxiliary within-evaluator "
                "evidence: a correct answer that required substantially more reasoning may "
                "still reveal a promising mechanism. Do not optimize verbosity or compare "
                "raw token counts across providers.\n\n"
                if feedback_policy["id"] == EFFORT_AWARE_BRANCHING_POLICY_ID
                else ""
            )
            treatment = (
                "# Reasoning-diversity branching treatment\n\n"
                "Read `/workspace/DIFFICULTY_FEEDBACK.md` and "
                "`/workspace/DIFFICULTY_FEEDBACK.json`. The controller assigns this episode "
                f"to reasoning target `{target}` with lineage parent `{parent}`. "
                f"{image_instruction}\n\n"
                f"{effort_instruction}"
                "Do not continue the globally latest renderer merely because it is easy to "
                "reuse. Design a task whose essential solution program belongs to the assigned "
                "reasoning target and is operationally different from every retained attempt. "
                "A new palette, noun, graph rule, answer label, or extra chain length is not a "
                "new reasoning program. Keep wording short and marks legible; difficulty must "
                "come from binding and composing visible evidence.\n\n"
                "If the assigned parent is `initial_profile`, branch from the five conceptual "
                "seeds rather than copying prior candidate code. Otherwise, inspect that local "
                "parent and make one causal change within the assigned reasoning family. You "
                "may use other attempts only as positive or negative evidence.\n\n"
                "Before final verification, write `difficulty_hypothesis.json` beside "
                "`candidate.json` inside your sole candidate directory at "
                "`/logs/artifacts/submission/candidates/<candidate_id>/`; never write it at "
                "the submission root. Use `/workspace/DIFFICULTY_HYPOTHESIS_SCHEMA.json`. "
                "Set `reasoning_target` exactly "
                f"to `{target}` and `parent_candidate_id` exactly to `{parent}`. Explain how the "
                "new executable reasoning program departs from the archive, why the evaluator's "
                "known strategy should fail, and how clarity is preserved. The controller "
                "validates these bindings before any paid evaluation.\n\n"
            )
        elif feedback_policy["id"] == SYNTHESIS_POLICY_ID:
            image_instruction = (
                "Inspect every referenced evaluated image with the image-viewing tool and "
                "connect its visible evidence to the evaluator rationale."
                if feedback_policy["builder_image_access"] == "evaluated_samples"
                else "No evaluator image is supplied in this arm. Work only from the prior "
                "renderer, prompt, inverse-oracle, verifier, tests, and textual evaluator "
                "record; do not seek or reconstruct the evaluator's exact raster samples."
            )
            treatment = (
                "# Falsification-aware difficulty treatment\n\n"
                "Read `/workspace/DIFFICULTY_FEEDBACK.md`. For every retained attempt, "
                "inspect the prior candidate source at its listed path, especially the "
                "forward renderer and independent pixel inverse arm. Reconstruct the "
                "intended solution procedure, compare it with the evaluator's successful "
                "or failed strategy, and identify why the prospective difficulty claim was "
                f"supported or falsified. {image_instruction}\n\n"
                "Treat the latest evaluated candidate as the lineage parent. Make one causal "
                "structural mutation that blocks the demonstrated evaluator strategy while "
                "preserving short wording, large readable marks, pixel answerability, and "
                "the unchanged protected gates. More steps, more symbols, clutter, tiny cues, "
                "or a cosmetic reskin are not acceptable difficulty mechanisms.\n\n"
                "Before final verification, write `difficulty_hypothesis.json` beside "
                "`candidate.json` inside your sole candidate directory at "
                "`/logs/artifacts/submission/candidates/<candidate_id>/`; never write it at "
                "the submission root. Use `/workspace/DIFFICULTY_HYPOTHESIS_SCHEMA.json`. "
                "Its retrospective synthesis "
                "must summarize the parent renderer/inverse computation, the evaluator's "
                "actual strategy, and the lesson carried forward. Its `parent_candidate_id` "
                "must equal `latest_complete_candidate_id` in `DIFFICULTY_FEEDBACK.json` "
                "(`initial_profile` in episode one). This is a concise, auditable design "
                "analysis—not private chain-of-thought. The controller alone evaluates the "
                "next protected-valid task.\n\n"
            )
        elif feedback_policy["id"] == DIFFICULTY_HYPOTHESIS_POLICY_ID:
            treatment = (
                "# Hypothesis-directed difficulty treatment\n\n"
                "Read `/workspace/DIFFICULTY_FEEDBACK.md` and inspect every referenced "
                "failed image before designing the next candidate. Treat each evaluator "
                "outcome as evidence: explain why earlier mechanisms were hard or easy, "
                "then make one causal structural mutation expected to increase visual "
                "computation without increasing ambiguity. Do not manufacture difficulty "
                "through tiny marks, clutter, illegible rendering, verbose wording, or a "
                "cosmetic reskin.\n\n"
                "Before final verification, write `difficulty_hypothesis.json` beside "
                "`candidate.json` inside your sole candidate directory at "
                "`/logs/artifacts/submission/candidates/<candidate_id>/`; never write it at "
                "the submission root. Use `/workspace/DIFFICULTY_HYPOTHESIS_SCHEMA.json`. "
                "It must record prior evidence IDs, the intended visual computation, the "
                "anticipated evaluator failure, the causal mutation, the clarity guard, "
                "and a prospective accuracy prediction. This is a concise testable design "
                "hypothesis, not private chain-of-thought. Missing or invalid hypotheses "
                "fail the feedback treatment closed. The controller alone calls evaluator "
                "models after offline verification; do not call a VLM yourself.\n\n"
            )
        else:
            treatment = (
                "# Difficulty-directed treatment\n\n"
                "Read `/workspace/DIFFICULTY_FEEDBACK.md` before designing the next "
                "candidate. Build on protected-valid evidence and seek a clearer but more "
                "perceptually demanding mechanism. The controller alone queries evaluator "
                "models after offline inverse-arm verification; do not call a VLM yourself.\n\n"
            )
        instruction_path.write_text(
            treatment + instruction_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    if strategy["process_trace_required"]:
        for name in _TRACE_PROTOCOL_FILES:
            source = strategy_root / name if strategy.get("cap_free") else protocol_root / name
            _copy(source, starter / "protocol" / name)
    if len(failed_candidates) != len(failed_verdicts):
        raise ValueError("failed candidate paths and verdicts differ in length")
    reserved_candidate_ids = [
        _candidate_metadata(path)["candidate_id"] for path in accepted_candidates
    ] + [
        _failed_candidate_id(path, verdict)
        for path, verdict in zip(failed_candidates, failed_verdicts, strict=True)
    ]
    task_spec = _task_spec(
        campaign,
        episode_index=episode_index,
        timeout_seconds=timeout_seconds,
        reserved_candidate_ids=reserved_candidate_ids,
    )
    if strategy.get("strategy_local_task_spec_schema"):
        _validate_task_spec_schema_binding(task_spec, strategy_root / "task-spec.schema.json")
    (starter / "TASK_SPEC.toml").write_text(task_spec, encoding="utf-8")
    (tests / "TASK_SPEC.toml").write_text(task_spec, encoding="utf-8")
    if (archive_policy := campaign.get("archive_policy")) and archive_policy.get("id") in {
        "quality-diversity@0.2.0",
        "quality-diversity@0.3.0",
        "quality-diversity@0.4.0",
        "quality-diversity@0.5.0",
        "quality-diversity@0.6.0",
        "quality-diversity@0.7.0",
        "quality-diversity@0.8.0",
    }:
        (tests / "QUALITY_DIVERSITY_POLICY.json").write_text(
            json.dumps(archive_policy, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    verifier_script = (
        repo_root / "harbor/datasets/foundry-builder-v1/build-question-worlds/tests/test.sh"
        if strategy["process_trace_required"]
        else override_root / "test.sh"
    )
    _copy(verifier_script, tests / "test.sh")
    (tests / "test.sh").chmod(0o755)
    (task / "task.toml").write_text(
        _task_toml(campaign, timeout_seconds=timeout_seconds),
        encoding="utf-8",
    )

    seed_path, seed = _seed_definition(repo_root, campaign["seed_set"])
    _copy(seed_path, starter / "SEED_SET.toml")
    worlds = seed.get("worlds")
    include = seed.get("include")
    if not isinstance(worlds, list) or not worlds or not isinstance(include, list) or not include:
        raise ValueError("seed set must declare worlds and include files")
    for world in worlds:
        if not isinstance(world, str) or not _ID.fullmatch(world):
            raise ValueError("seed set contains an invalid world ID")
        for name in include:
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("seed include entries must be plain filenames")
            _copy(repo_root / "worlds" / world / name, starter / "seeds" / world / name)

    known, negative, registry = _extended_memories(
        repo_root=repo_root,
        accepted_candidates=accepted_candidates,
        failed_candidates=failed_candidates,
        failed_verdicts=failed_verdicts,
    )
    _write_jsonl(starter / "memory/known_mechanisms.jsonl", known)
    _write_jsonl(starter / "memory/rejected_mechanisms.jsonl", negative)
    _write_jsonl(starter / "memory/public_registry.jsonl", [])
    _write_jsonl(tests / "mechanism_memory.jsonl", known)
    _write_jsonl(tests / "negative_memory.jsonl", negative)
    _write_jsonl(tests / "historical_registry.jsonl", registry)
    if protected_profile_memory is not None:
        _copy(protected_profile_memory, tests / "profile_source_mechanisms.jsonl")
    _copy(
        repo_root / "foundry/ontologies/mechanism-fingerprint-v0.1.schema.json",
        starter / "protocol/mechanism-fingerprint.schema.json",
    )

    campaign_dir = starter / "campaign"
    campaign_dir.mkdir(parents=True, exist_ok=True)
    (campaign_dir / "state.json").write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    episode = {
        "schema_version": "sequential-episode-input-0.1.0",
        "campaign_id": campaign["id"],
        "episode_index": episode_index,
        "harbor_task_namespace": harbor_task_namespace,
        "agent_timeout_seconds": timeout_seconds,
        "working_seed_candidate_ids": [
            _candidate_metadata(path)["candidate_id"] for path in accepted_candidates
        ],
        "failed_candidate_ids": [
            _failed_candidate_id(path, verdict)
            for path, verdict in zip(failed_candidates, failed_verdicts, strict=True)
        ],
        "controller_owns_transition": True,
    }
    if profile_ref is not None:
        episode["hypothesis_profile"] = profile_ref
    if archive_target is not None:
        episode["archive_policy"] = campaign["archive_policy"]["id"]
        episode["diversity_target"] = archive_target
    if feedback_history is not None:
        episode["feedback_policy"] = campaign["feedback_policy"]["id"]
        episode["feedback_history_size"] = len(feedback_history)
        if reasoning_assignment is not None:
            episode["reasoning_assignment"] = reasoning_assignment
    (campaign_dir / "episode.json").write_text(
        json.dumps(episode, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for candidate_path in accepted_candidates:
        candidate_id = _candidate_metadata(candidate_path)["candidate_id"]
        shutil.copytree(candidate_path, campaign_dir / "working_seed" / candidate_id)
    if (feedback_policy := campaign.get("feedback_policy")) and feedback_policy[
        "id"
    ] in SYNTHESIS_POLICY_IDS:
        # Both 0.5 arms receive identical prior code and textual evidence. Remove
        # historical raster galleries from the copied working library so the sole
        # between-arm visual difference is the controller-selected evaluated image.
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.gif"):
            for image_path in (campaign_dir / "working_seed").rglob(pattern):
                image_path.unlink()
    if failed_candidates:
        previous_path = failed_candidates[-1]
        candidate_id = _failed_candidate_id(previous_path, failed_verdicts[-1])
        shutil.copytree(previous_path, campaign_dir / "previous_failed_candidate" / candidate_id)
        (campaign_dir / "previous_verdict.json").write_text(
            json.dumps(failed_verdicts[-1], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    # The shared Oracle solution is retained for no-model canaries. Only funnel
    # mode keeps reference process-trace generation and protected trace gates.
    solve = task / "solution/solve.sh"
    solve_text = solve.read_text(encoding="utf-8")
    marker = (
        "\nif grep -Eq 'protocol = \"question-world@0\\.(3|4)\\.0\"' "
        "/workspace/TASK_SPEC.toml; then\n"
        "  python /solution/write_reference_process.py\n"
        "fi\n"
    )
    if marker not in solve_text:
        raise ValueError("shared Oracle solution process block changed unexpectedly")
    if not strategy["process_trace_required"]:
        solve.write_text(solve_text.replace(marker, "\n"), encoding="utf-8")
    if include_oracle_fixture:
        oracle_solution = starter / "oracle_solution"
        shutil.copytree(task / "solution", oracle_solution)
        oracle_solve = oracle_solution / "solve.sh"
        oracle_solve.write_text(
            oracle_solve.read_text(encoding="utf-8").replace(
                "/solution",
                "/workspace/oracle_solution",
            ),
            encoding="utf-8",
        )
        oracle_solve.chmod(0o755)

    output_root.mkdir(parents=True, exist_ok=True)
    packet_campaign = output_root / "campaign.toml"
    _copy(campaign_path, packet_campaign)
    files = {
        path.relative_to(output_root).as_posix(): _sha256(path)
        for path in sorted(item for item in dataset.rglob("*") if item.is_file())
    }
    packet_sha256 = hashlib.sha256(canonical_json(files).encode()).hexdigest()
    strategy_status = tomllib.loads((strategy_root / "strategy.toml").read_text(encoding="utf-8"))[
        "status"
    ]
    manifest = {
        "schema_version": "sequential-episode-packet-0.1.0",
        "kind": "sequential-foundry-episode",
        "campaign_id": campaign["id"],
        "campaign_status": campaign["status"],
        "protocol": campaign["protocol"],
        "strategy": campaign["strategy"],
        "strategy_status": strategy_status,
        "episode_index": episode_index,
        "source_revision": source_revision,
        "source_dirty": source_dirty,
        "freezeable": False,
        "packet_sha256": packet_sha256,
        "campaign_sha256": _sha256(packet_campaign),
        "state_sha256": hashlib.sha256(canonical_json(state).encode()).hexdigest(),
        "working_seed_hashes": {
            _candidate_metadata(path)["candidate_id"]: sha256_candidate_tree(path)
            for path in accepted_candidates
        },
        "failed_outcome_hashes": {
            _failed_candidate_id(path, verdict): sha256_candidate_tree(path)
            for path, verdict in zip(failed_candidates, failed_verdicts, strict=True)
        },
        "file_count": len(files),
        "files": files,
        "agent": {
            "adapter": campaign["agent"]["adapter"],
            "model": campaign["agent"]["model"],
            "expected_reported_models": sorted(expected_reported_models(campaign)),
            "cost_reporting": campaign["campaign"].get("cost_reporting", "provider_native"),
            "reasoning_effort": campaign["agent"]["reasoning_effort"],
            "cli_version": campaign["agent"]["cli_version"],
            "output_format": campaign["agent"]["output_format"],
            "timeout_seconds": timeout_seconds,
            "recycle_for_verifier": True,
        },
        "render_runtime": render_runtime,
        "visibility": {
            "canonical_checkout_mounted": False,
            "hidden_registry_in_verifier_only": True,
            "working_seed_visible": True,
            "failed_outcome_memory_visible": True,
            "clean_offline_verifier_recycle_required": True,
            "controller_state_read_only_by_contract": True,
            "oracle_fixture_visible": include_oracle_fixture,
            "discovery_profile_text_visible": profile_record is not None,
            "discovery_profile_executable_prototypes_visible": False,
            "discovery_profile_private_material_visible": False,
        },
    }
    if profile_record is not None:
        manifest["discovery_profile"] = profile_record
    if archive_target is not None:
        manifest["controller_policy"] = {
            "kind": "quality_diversity",
            "policy_id": campaign["archive_policy"]["id"],
            "target": archive_target,
            "policy_sha256": hashlib.sha256(
                canonical_json(campaign["archive_policy"]).encode()
            ).hexdigest(),
        }
        manifest["visibility"]["diversity_target_visible"] = True
    if feedback_history is not None:
        manifest["controller_policy"] = {
            "kind": "difficulty_feedback",
            "policy_id": campaign["feedback_policy"]["id"],
            "public_history_size": len(feedback_history),
            "policy_sha256": hashlib.sha256(
                canonical_json(campaign["feedback_policy"]).encode()
            ).hexdigest(),
        }
        manifest["visibility"]["aggregate_difficulty_feedback_visible"] = True
        manifest["visibility"]["feedback_sample_predictions_visible"] = True
        manifest["visibility"]["feedback_evaluator_rationales_visible"] = (
            campaign["feedback_policy"]["id"] in RATIONALE_POLICY_IDS
        )
        manifest["visibility"]["feedback_builder_difficulty_hypotheses_visible"] = (
            campaign["feedback_policy"]["id"] in HYPOTHESIS_POLICY_IDS
        )
        manifest["visibility"]["feedback_failed_images_visible"] = (
            campaign["feedback_policy"]["id"] == DIFFICULTY_HYPOTHESIS_POLICY_ID
        )
        manifest["visibility"]["feedback_evaluated_images_visible"] = (
            campaign["feedback_policy"]["id"] in SYNTHESIS_POLICY_IDS
            and campaign["feedback_policy"]["builder_image_access"] == "evaluated_samples"
        )
        manifest["visibility"]["feedback_gold_labels_visible"] = (
            campaign["feedback_policy"]["id"] in SYNTHESIS_POLICY_IDS
        )
        manifest["visibility"]["feedback_reasoning_target_visible"] = campaign["feedback_policy"][
            "id"
        ] in {
            BRANCHING_POLICY_ID,
            WIRED_RESILIENT_BRANCHING_POLICY_ID,
            EFFORT_AWARE_BRANCHING_POLICY_ID,
        }
        manifest["visibility"]["feedback_solver_effort_visible"] = (
            campaign["feedback_policy"]["id"] == EFFORT_AWARE_BRANCHING_POLICY_ID
        )
        manifest["visibility"]["feedback_private_reasoning_visible"] = False
        manifest["visibility"]["feedback_raw_provider_payloads_visible"] = False
    if campaign["strategy"] not in {"sequential@0.1.0", "sequential@0.2.0"}:
        manifest["schema_version"] = (
            "orchestration-session-packet-0.3.0"
            if strategy.get("cap_free")
            else "orchestration-session-packet-0.2.0"
            if strategy.get("deadline_driven")
            else "orchestration-session-packet-0.1.0"
        )
        manifest["kind"] = "orchestration-foundry-session"
        manifest["orchestration_mode"] = strategy["mode"]
        manifest["process_trace_required"] = strategy["process_trace_required"]
        if strategy.get("cap_free"):
            manifest["candidate_limit_policy"] = campaign["task"]["candidate_limit_policy"]
            manifest["episode_limit_policy"] = campaign["campaign"]["episode_limit_policy"]
    if strategy.get("deadline_driven"):
        manifest["agent"].update(
            {
                "provider_turn_policy": campaign["agent"]["provider_turn_policy"],
                "continuation_delay_seconds": campaign["agent"]["continuation_delay_seconds"],
            }
        )
        manifest["campaign_completion_policy"] = campaign["campaign"]["completion_policy"]
        manifest["reserved_candidate_ids"] = reserved_candidate_ids
    if not strategy.get("cap_free"):
        manifest["agent"]["max_turns"] = campaign["agent"]["max_turns_per_episode"]
    manifest_path = output_root / "materialization.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return SequentialEpisodeMaterialization(
        root=output_root,
        task=task,
        manifest=manifest_path,
        packet_sha256=packet_sha256,
        episode_index=episode_index,
        agent_timeout_seconds=timeout_seconds,
    )
