"""Security and trajectory checks for the OpenCode/OpenRouter Harbor adapter."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

AGENTS_DIR = Path(__file__).resolve().parents[1] / "harbor/agents"
_STUBBED = (
    "harbor",
    "harbor.agents",
    "harbor.agents.installed",
    "harbor.agents.installed.base",
    "harbor.environments",
    "harbor.environments.base",
    "harbor.models",
    "harbor.models.agent",
    "harbor.models.agent.context",
    "harbor.models.trial",
    "harbor.models.trial.paths",
    "offline_verifier",
)


class _ExecInput:
    def __init__(self, command: str) -> None:
        self.command = command
        self.env = None


class _BaseInstalledAgent:
    def __init__(
        self,
        *args,
        model_name: str = "",
        version: str | None = None,
        logs_dir: Path | None = None,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        self._version = version
        self.logs_dir = logs_dir or Path("/tmp/opencode-test")

    def version(self) -> str | None:
        return self._version


def _install_stubs() -> dict[str, types.ModuleType | None]:
    saved = {name: sys.modules.get(name) for name in _STUBBED}
    for name in _STUBBED:
        sys.modules[name] = types.ModuleType(name)
    base = sys.modules["harbor.agents.installed.base"]
    base.BaseInstalledAgent = _BaseInstalledAgent
    base.ExecInput = _ExecInput
    sys.modules["harbor.environments.base"].BaseEnvironment = object
    sys.modules["harbor.models.agent.context"].AgentContext = object
    sys.modules["harbor.models.trial.paths"].EnvironmentPaths = types.SimpleNamespace(
        agent_dir=Path("/logs/agent")
    )

    async def _unused(*args, **kwargs):
        raise AssertionError("offline recycle helper unexpectedly called")

    offline = sys.modules["offline_verifier"]
    offline.checkpoint_cancelled_agent = _unused
    offline.recycle_for_offline_verifier = _unused
    return saved


@pytest.fixture
def opencode_adapter():
    saved = _install_stubs()
    sys.path.insert(0, str(AGENTS_DIR))
    sys.modules.pop("openrouter_opencode_foundry", None)
    try:
        from openrouter_opencode_foundry import OpenRouterOpenCodeFoundry

        yield OpenRouterOpenCodeFoundry
    finally:
        sys.path.remove(str(AGENTS_DIR))
        sys.modules.pop("openrouter_opencode_foundry", None)
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _agent(adapter, monkeypatch, *, logs_dir: Path | None = None):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-not-a-real-secret-value")
    return adapter(
        logs_dir=logs_dir,
        model_name="openrouter/deepseek/deepseek-v4-flash",
        version="1.18.11",
        variant="high",
        upstream_provider="deepseek",
        allow_provider_fallbacks=False,
    )


def test_command_never_embeds_or_forwards_the_host_key(opencode_adapter, monkeypatch) -> None:
    agent = _agent(opencode_adapter, monkeypatch)
    command_input = agent.create_run_agent_commands("write the sentinel")[0]
    assert command_input.env is None
    assert "sk-or-not-a-real-secret-value" not in command_input.command
    assert "OPENROUTER_API_KEY" not in command_input.command
    assert "/tmp/openrouter-api-key-upload" in command_input.command
    assert '"$data_home/opencode/auth.json"' in command_input.command


def test_command_pins_model_provider_reasoning_and_cleanup(opencode_adapter, monkeypatch) -> None:
    command = _agent(opencode_adapter, monkeypatch).create_run_agent_commands("build")[0].command
    assert "--model openrouter/deepseek/deepseek-v4-flash" in command
    assert "--variant high" in command
    assert "--format json" in command
    assert "--thinking" in command
    assert "--auto" in command
    assert "--dir /workspace" in command
    assert '"order":["deepseek"]' in command
    assert '"allow_fallbacks":false' in command
    assert 'rm -rf "$data_home" "$state_home" "$config_home"' in command


def test_command_accepts_the_exact_deepseek_v4_flash_0731_slug(
    opencode_adapter, monkeypatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret-never-rendered")
    agent = opencode_adapter(
        logs_dir=Path("/tmp/test-openrouter-opencode-0731"),
        model_name="openrouter/deepseek/deepseek-v4-flash-0731",
        version="1.18.11",
        variant="high",
        upstream_provider="deepinfra",
        allow_provider_fallbacks=False,
    )

    command = agent.create_run_agent_commands("build")[0].command
    assert "--model openrouter/deepseek/deepseek-v4-flash-0731" in command
    assert '"order":["deepinfra"]' in command
    assert '"allow_fallbacks":false' in command


def test_controller_campaign_kwargs_map_to_one_high_reasoning_turn(
    opencode_adapter, monkeypatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret-never-rendered")
    agent = opencode_adapter(
        logs_dir=Path("/tmp/test-openrouter-opencode"),
        model_name="openrouter/deepseek/deepseek-v4-flash",
        reasoning_effort="high",
        output_format="json",
        provider_turn_policy="single_turn",
        continuation_delay_seconds=0,
        retry_backoff_seconds="15,30,60",
        max_consecutive_provider_failures=3,
        recycle_for_verifier=True,
    )

    command = agent.create_run_agent_commands("build")[0].command
    assert "--variant high" in command
    assert "--model openrouter/deepseek/deepseek-v4-flash" in command


def test_controller_rejects_resumed_opencode_sessions(opencode_adapter, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret-never-rendered")
    with pytest.raises(ValueError, match="provider_turn_policy=single_turn"):
        opencode_adapter(
            logs_dir=Path("/tmp/test-openrouter-opencode"),
            model_name="openrouter/deepseek/deepseek-v4-flash",
            reasoning_effort="high",
            provider_turn_policy="resume_until_timeout",
        )


def test_missing_key_and_floating_alias_fail_closed(opencode_adapter, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        opencode_adapter(
            model_name="openrouter/deepseek/deepseek-v4-flash"
        ).create_run_agent_commands("build")

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-not-a-real-secret-value")
    with pytest.raises(ValueError, match="floating"):
        opencode_adapter(
            model_name="openrouter/~deepseek/deepseek-v4-flash-latest"
        ).create_run_agent_commands("build")


def test_json_events_become_a_normalized_trajectory(
    opencode_adapter, monkeypatch, tmp_path
) -> None:
    events = [
        {
            "type": "user",
            "timestamp": 1700000000000,
            "sessionID": "session-1",
            "parts": [{"type": "text", "text": "write the sentinel"}],
        },
        {
            "type": "step_start",
            "timestamp": 1700000001000,
            "sessionID": "session-1",
            "part": {"type": "step-start"},
        },
        {
            "type": "reasoning",
            "timestamp": 1700000001500,
            "sessionID": "session-1",
            "part": {"type": "reasoning", "text": "Use the write tool."},
        },
        {
            "type": "tool_use",
            "timestamp": 1700000002000,
            "sessionID": "session-1",
            "part": {
                "type": "tool",
                "callID": "call-1",
                "tool": "write",
                "state": {
                    "input": {"filePath": "/logs/artifacts/auth-smoke/result.txt"},
                    "output": "written",
                },
            },
        },
        {
            "type": "text",
            "timestamp": 1700000002500,
            "sessionID": "session-1",
            "part": {"type": "text", "text": "Done."},
        },
        {
            "type": "step_finish",
            "timestamp": 1700000003000,
            "sessionID": "session-1",
            "part": {
                "type": "step-finish",
                "cost": 0.012,
                "tokens": {
                    "input": 100,
                    "output": 20,
                    "reasoning": 8,
                    "cache": {"read": 30, "write": 0},
                },
            },
        },
    ]
    (tmp_path / "opencode.txt").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )
    agent = _agent(opencode_adapter, monkeypatch, logs_dir=tmp_path)
    context = types.SimpleNamespace(
        cost_usd=None,
        n_input_tokens=None,
        n_cache_tokens=None,
        n_output_tokens=None,
    )
    agent.populate_context_post_run(context)

    trajectory = json.loads((tmp_path / "trajectory.json").read_text(encoding="utf-8"))
    assert trajectory["schema_version"] == "ATIF-v1.6"
    assert trajectory["agent"]["model_name"] == "openrouter/deepseek/deepseek-v4-flash"
    assert trajectory["agent"]["extra"]["openrouter_upstream_provider"] == "deepseek"
    assert trajectory["steps"][0]["source"] == "user"
    assert trajectory["steps"][1]["reasoning_content"] == "Use the write tool."
    assert trajectory["steps"][1]["tool_calls"][0]["function_name"] == "write"
    assert trajectory["final_metrics"]["total_prompt_tokens"] == 130
    assert trajectory["final_metrics"]["total_completion_tokens"] == 20
    assert trajectory["final_metrics"]["extra"]["reasoning_tokens"] == 8
    assert context.cost_usd == 0.012
    assert context.n_input_tokens == 130
    assert context.n_cache_tokens == 30
    assert context.n_output_tokens == 20
