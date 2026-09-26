"""Materialize minimal, content-hashed Harbor packets for foundry campaigns."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path

from question_foundry.render_runtime import (
    campaign_render_runtime,
    configure_task_render_runtime,
)

_WORLD_ID = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_CAMPAIGN_ID = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_PUBLIC_PROTOCOL_FILES = (
    "brief.md",
    "candidate-contract.md",
    "candidate.schema.json",
    "known-failure-modes.md",
    "protocol.toml",
    "public-gates.md",
    "task-spec.schema.json",
)
_TRACE_PROTOCOL_FILES = ("process-contract.md", "search-trace.schema.json")
_SUPPORTED_PROTOCOLS = {
    "question-world@0.2.0",
    "question-world@0.3.0",
    "question-world@0.4.0",
}
_MAX_AGENT_TURNS = 240
_SUPPORTED_AGENT_CONFIGS = {
    "grok_build_auth:GrokBuildAuth": {
        "model": "grok-4.5",
        "reasoning_effort": "high",
        "output_format": "streaming-json",
    },
    "claude_code_foundry:ClaudeCodeFoundry": {
        "model": "claude-opus-4-8",
        "reasoning_effort": "high",
        "output_format": "stream-json",
    },
    "gemma_local_foundry:GemmaLocalFoundry": {
        "model": "gemma-4-31B-it",
        "reasoning_effort": "high",
        "output_format": "stream-json",
    },
}


@dataclass(frozen=True)
class FoundryMaterialization:
    root: Path
    dataset: Path
    task: Path
    manifest: Path
    packet_sha256: str
    file_count: int
    freezeable: bool


def freezeable_inputs(
    *,
    campaign_status: str,
    protocol_status: str,
    seed_status: str,
    source_dirty: bool,
) -> bool:
    return (
        not source_dirty
        and campaign_status == "frozen"
        and protocol_status == "frozen"
        and seed_status == "frozen"
    )


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


def _task_spec_toml(campaign: dict) -> str:
    agent = campaign["agent"]
    task = campaign["task"]
    protocol = campaign["protocol"]
    values = {
        "schema_version": protocol.rpartition("@")[2],
        "campaign_id": campaign["id"],
        "mode": campaign["mode"],
        "protocol": protocol,
        "seed_set": campaign["seed_set"],
        "min_candidates": task["min_candidates"],
        "max_candidates": task["max_candidates"],
        "repair_turns": task["repair_turns"],
        "max_turns": agent["max_turns"],
        "timeout_seconds": agent["timeout_seconds"],
        "requires_clean_recycle": True,
    }
    if protocol in {"question-world@0.3.0", "question-world@0.4.0"}:
        search = campaign["search"]
        values.update(
            {
                "min_logged_proposals": search["min_logged_proposals"],
                "max_build_starts": search["max_build_starts"],
                "search_phase_seconds": search["search_phase_seconds"],
                "triage_phase_seconds": search["triage_phase_seconds"],
                "build_phase_seconds": search["build_phase_seconds"],
                "repair_phase_seconds": search["repair_phase_seconds"],
                "finalization_phase_seconds": search["finalization_phase_seconds"],
            }
        )
    if protocol == "question-world@0.4.0":
        values.update(
            {
                "mechanism_memory": campaign["mechanism_memory"],
                "negative_memory": campaign["negative_memory"],
                "min_ranking_passes": search["min_ranking_passes"],
                "semantic_neighbor_limit": search["semantic_neighbor_limit"],
                "semantic_duplicate_threshold": search["semantic_duplicate_threshold"],
                "oracle_stress_seeds": search["oracle_stress_seeds"],
                "oracle_stress_scenes_per_seed": search["oracle_stress_scenes_per_seed"],
            }
        )
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


def _task_toml(campaign: dict) -> str:
    environment = campaign["environment"]
    timeout = float(campaign["agent"]["timeout_seconds"])
    protocol_version = campaign["protocol"].rpartition("@")[2]
    traced = campaign["protocol"] in {
        "question-world@0.3.0",
        "question-world@0.4.0",
    }
    artifacts = (
        '["/logs/artifacts/process", "/logs/artifacts/submission"]'
        if traced
        else '["/logs/artifacts/submission"]'
    )
    difficulty = (
        "Requires complete executable worlds, independent pixel oracles, and auditable evidence."
    )
    verification = (
        "A separate offline verifier executes generation, protected gates, "
        "corruptions, and registry comparisons."
    )
    return f"""schema_version = "1.4"
artifacts = {artifacts}

[task]
name = "question-foundry/build-question-worlds"
version = "{protocol_version}"
description = "Build a frozen portfolio of executable latent-z visual-question candidates"
keywords = ["vlm", "foundry", "world-generation", "verification"]

[metadata]
category = "agentic-research"
difficulty_explanation = {json.dumps(difficulty)}
verification_explanation = {json.dumps(verification)}

[agent]
timeout_sec = {timeout:.1f}

[verifier]
timeout_sec = 900.0

[environment]
allow_internet = true
build_timeout_sec = 600.0
cpus = {int(environment["cpus"])}
memory_mb = {int(environment["memory_mb"])}
storage_mb = {int(environment["storage_mb"])}
gpus = 0
"""


def _validate_campaign(campaign: dict) -> None:
    if not _CAMPAIGN_ID.fullmatch(str(campaign.get("id", ""))):
        raise ValueError("invalid campaign id")
    if campaign.get("mode") not in {"discovery", "formal"}:
        raise ValueError("campaign mode must be discovery or formal")
    protocol = campaign.get("protocol")
    if protocol not in _SUPPORTED_PROTOCOLS:
        raise ValueError(f"foundry-builder-v1 requires one of {sorted(_SUPPORTED_PROTOCOLS)}")
    agent = campaign.get("agent")
    if not isinstance(agent, dict):
        raise ValueError("campaign has no agent block")
    adapter = agent.get("adapter")
    required = _SUPPORTED_AGENT_CONFIGS.get(adapter)
    if required is None:
        raise ValueError("campaign requires a supported clean-recycle custom adapter")
    for field, expected in required.items():
        if agent.get(field) != expected:
            raise ValueError(f"{adapter} requires {field}={expected}")
    if not 1 <= int(agent.get("max_turns", 0)) <= _MAX_AGENT_TURNS:
        raise ValueError(f"max_turns must be in [1, {_MAX_AGENT_TURNS}]")
    task = campaign.get("task")
    if not isinstance(task, dict):
        raise ValueError("campaign has no task block")
    minimum, maximum = int(task.get("min_candidates", 0)), int(task.get("max_candidates", 0))
    if not 1 <= minimum <= maximum <= 10:
        raise ValueError("candidate limits must satisfy 1 <= min <= max <= 10")
    if campaign["mode"] == "formal" and (minimum, maximum) != (1, 1):
        raise ValueError("formal campaigns require exactly one candidate")
    if protocol in {"question-world@0.3.0", "question-world@0.4.0"}:
        search = campaign.get("search")
        if not isinstance(search, dict):
            raise ValueError("question-world@0.3.0 requires a search block")
        min_proposals = int(search.get("min_logged_proposals", 0))
        max_builds = int(search.get("max_build_starts", 0))
        if not 1 <= min_proposals <= 100:
            raise ValueError("min_logged_proposals must be in [1, 100]")
        if not 1 <= max_builds <= 20:
            raise ValueError("max_build_starts must be in [1, 20]")
        if max_builds < maximum:
            raise ValueError("max_build_starts must cover max_candidates")
        phase_keys = (
            "search_phase_seconds",
            "triage_phase_seconds",
            "build_phase_seconds",
            "repair_phase_seconds",
            "finalization_phase_seconds",
        )
        phase_total = sum(int(search.get(key, 0)) for key in phase_keys)
        if any(int(search.get(key, 0)) < 1 for key in phase_keys):
            raise ValueError("every search phase must receive at least one second")
        if phase_total != int(agent["timeout_seconds"]):
            raise ValueError("search phase seconds must sum to timeout_seconds")
    if protocol == "question-world@0.4.0":
        if campaign.get("mechanism_memory") != "known-mechanisms@0.1.0":
            raise ValueError("v0.4 requires known-mechanisms@0.1.0")
        if campaign.get("negative_memory") != "rejected-mechanisms@0.1.0":
            raise ValueError("v0.4 requires rejected-mechanisms@0.1.0")
        if not 2 <= int(search.get("min_ranking_passes", 0)) <= 5:
            raise ValueError("min_ranking_passes must be in [2, 5]")
        if not 1 <= int(search.get("semantic_neighbor_limit", 0)) <= 10:
            raise ValueError("semantic_neighbor_limit must be in [1, 10]")
        threshold = float(search.get("semantic_duplicate_threshold", -1))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("semantic_duplicate_threshold must be in [0, 1]")
        stress_seeds = search.get("oracle_stress_seeds")
        if (
            not isinstance(stress_seeds, list)
            or len(stress_seeds) < 2
            or len(stress_seeds) != len(set(stress_seeds))
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in stress_seeds
            )
        ):
            raise ValueError("oracle_stress_seeds must contain at least two unique integers")
        if not 12 <= int(search.get("oracle_stress_scenes_per_seed", 0)) <= 100:
            raise ValueError("oracle_stress_scenes_per_seed must be in [12, 100]")
    environment = campaign.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("campaign has no environment block")
    campaign_render_runtime(campaign)
    if environment.get("agent_network") != "public":
        raise ValueError("installed remote agents require explicit public agent network")
    if environment.get("verifier_network") != "no-network":
        raise ValueError("the hidden verifier must be network-disabled")


def materialize_foundry_campaign(
    *,
    repo_root: Path,
    campaign_path: Path,
    output_root: Path,
    source_revision: str,
    source_dirty: bool,
) -> FoundryMaterialization:
    repo_root = repo_root.resolve()
    campaign_path = campaign_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"materialization target already exists: {output_root}")

    campaign = tomllib.loads(campaign_path.read_text(encoding="utf-8"))
    _validate_campaign(campaign)
    seed_name, separator, seed_version = campaign["seed_set"].partition("@")
    if not separator:
        raise ValueError("seed_set must use name@version")
    seed_set_path = None
    for path in sorted((repo_root / "foundry/seed_sets").glob(f"{seed_name}_v*.toml")):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        if data.get("id") == seed_name and data.get("version") == seed_version:
            seed_set_path = path
            break
    if seed_set_path is None:
        raise FileNotFoundError(f"seed set file not found for {campaign['seed_set']}")
    seed_set = tomllib.loads(seed_set_path.read_text(encoding="utf-8"))
    seed_status = str(seed_set.get("status", ""))
    worlds = seed_set.get("worlds")
    include = seed_set.get("include")
    if not isinstance(worlds, list) or not worlds:
        raise ValueError("seed set must declare worlds")
    if not isinstance(include, list) or not include:
        raise ValueError("seed set must declare include")
    if any(not isinstance(world, str) or not _WORLD_ID.fullmatch(world) for world in worlds):
        raise ValueError("seed set contains invalid world IDs")
    if any(not isinstance(name, str) or Path(name).name != name for name in include):
        raise ValueError("seed include entries must be plain filenames")

    template = repo_root / "harbor/datasets/foundry-builder-v1"
    dataset = output_root / "foundry-builder-v1"
    shutil.copytree(
        template,
        dataset,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )
    task = dataset / "build-question-worlds"
    starter = task / "environment/starter"
    tests = task / "tests"
    render_runtime = configure_task_render_runtime(
        task=task,
        campaign=campaign,
        capability_source=repo_root / "foundry/render_runtimes/latex-tikz-v0.1.md",
    )
    protocol_version = campaign["protocol"].rpartition("@")[2]
    protocol_source = (
        repo_root / "foundry/protocols/question-world" / ("v" + protocol_version.rsplit(".", 1)[0])
    )
    protocol_data = tomllib.loads((protocol_source / "protocol.toml").read_text(encoding="utf-8"))
    protocol_status = str(protocol_data.get("status", ""))
    _copy(protocol_source / "AGENTS.md", starter / "AGENTS.md")
    traced = campaign["protocol"] in {
        "question-world@0.3.0",
        "question-world@0.4.0",
    }
    public_protocol_files = _PUBLIC_PROTOCOL_FILES + (_TRACE_PROTOCOL_FILES if traced else ())
    for name in public_protocol_files:
        _copy(protocol_source / name, starter / "protocol" / name)
    if campaign["protocol"] == "question-world@0.4.0":
        mechanism_memory = repo_root / "registry/historical/mechanism-fingerprints-v0.1.jsonl"
        negative_memory = repo_root / "registry/negative/rejected-mechanisms-v0.1.jsonl"
        ontology = repo_root / "foundry/ontologies/mechanism-fingerprint-v0.1.schema.json"
        _copy(mechanism_memory, starter / "memory/known_mechanisms.jsonl")
        _copy(negative_memory, starter / "memory/rejected_mechanisms.jsonl")
        _copy(ontology, starter / "protocol/mechanism-fingerprint.schema.json")
        _copy(mechanism_memory, tests / "mechanism_memory.jsonl")
        _copy(negative_memory, tests / "negative_memory.jsonl")
    _copy(seed_set_path, starter / "SEED_SET.toml")
    for world in worlds:
        for name in include:
            _copy(repo_root / "worlds" / world / name, starter / "seeds" / world / name)

    task_spec = _task_spec_toml(campaign)
    (starter / "TASK_SPEC.toml").write_text(task_spec, encoding="utf-8")
    (tests / "TASK_SPEC.toml").write_text(task_spec, encoding="utf-8")
    _copy(
        repo_root / "registry/historical/reference-worlds.jsonl",
        tests / "historical_registry.jsonl",
    )
    (task / "task.toml").write_text(_task_toml(campaign), encoding="utf-8")
    packet_campaign = output_root / "campaign.toml"
    _copy(campaign_path, packet_campaign)

    files: dict[str, str] = {}
    for path in sorted(item for item in dataset.rglob("*") if item.is_file()):
        files[path.relative_to(output_root).as_posix()] = _sha256(path)
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    packet_sha256 = hashlib.sha256(canonical).hexdigest()
    freezeable = freezeable_inputs(
        campaign_status=str(campaign.get("status", "")),
        protocol_status=protocol_status,
        seed_status=seed_status,
        source_dirty=source_dirty,
    )
    manifest_data = {
        "schema_version": "0.2.0",
        "kind": "foundry-campaign-packet",
        "campaign_id": campaign["id"],
        "campaign_status": campaign["status"],
        "protocol_status": protocol_status,
        "seed_status": seed_status,
        "mode": campaign["mode"],
        "protocol": campaign["protocol"],
        "seed_set": campaign["seed_set"],
        "source_revision": source_revision,
        "source_dirty": source_dirty,
        "freezeable": freezeable,
        "packet_sha256": packet_sha256,
        "campaign_sha256": _sha256(packet_campaign),
        "file_count": len(files),
        "files": files,
        "agent": {
            "adapter": campaign["agent"]["adapter"],
            "model": campaign["agent"]["model"],
            "reasoning_effort": campaign["agent"]["reasoning_effort"],
            "cli_version": campaign["agent"]["cli_version"],
            "output_format": campaign["agent"]["output_format"],
            "max_turns": campaign["agent"]["max_turns"],
            "timeout_seconds": campaign["agent"]["timeout_seconds"],
            "recycle_for_verifier": True,
        },
        "render_runtime": render_runtime,
        "search": campaign.get("search"),
        "visibility": {
            "visible_worlds": worlds,
            "hidden_registry_in_verifier_only": True,
            "canonical_checkout_mounted": False,
            "sibling_outputs_visible": False,
            "clean_offline_verifier_recycle_required": True,
            "versioned_mechanism_memory_visible": (campaign["protocol"] == "question-world@0.4.0"),
            "versioned_negative_memory_visible": (campaign["protocol"] == "question-world@0.4.0"),
        },
    }
    manifest = output_root / "materialization.json"
    manifest.write_text(
        json.dumps(manifest_data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return FoundryMaterialization(
        root=output_root,
        dataset=dataset,
        task=task,
        manifest=manifest,
        packet_sha256=packet_sha256,
        file_count=len(files),
        freezeable=freezeable,
    )
