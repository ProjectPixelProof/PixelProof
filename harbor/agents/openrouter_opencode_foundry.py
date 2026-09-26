"""Harbor 0.1.44 OpenCode adapter for exact OpenRouter model slugs.

The built-in Harbor 0.1.44 OpenCode adapter rejects the ``openrouter`` provider,
forwards credentials through ``ExecInput.env``, and does not retain a normalized
trajectory. This repository-owned adapter keeps the validated Harbor version and
uses the foundry's existing security boundary instead:

* the host key is copied to a mode-0600 temporary file and uploaded;
* the key never appears in ``ExecInput.env``, a command argument, or task text;
* OpenCode receives a disposable auth file inside the agent container;
* the auth/config/state trees are destroyed before clean offline verification;
* JSON events are retained and converted to a compact ATIF-v1.6 trajectory.

Only exact OpenRouter model slugs are accepted. Floating ``~...latest`` aliases
are intentionally rejected because they are unsuitable for paper experiments.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent, ExecInput
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths
from offline_verifier import checkpoint_cancelled_agent, recycle_for_offline_verifier


class OpenRouterOpenCodeFoundry(BaseInstalledAgent):
    """OpenCode authenticated to OpenRouter by a late-materialized API key."""

    _HOST_TOKEN_ENV = "OPENROUTER_API_KEY"
    _UPLOADED_TOKEN_FILE = "/tmp/openrouter-api-key-upload"
    _PID_FILE = "/tmp/opencode-agent-session.pid"
    _WRAPPER_PID_FILE = "/tmp/opencode-agent-wrapper.pid"
    _OUTPUT_FILENAME = "opencode.txt"
    _LOG_FILENAME = "opencode.log"

    def __init__(
        self,
        *args,
        variant: str | None = None,
        reasoning_effort: str | None = None,
        output_format: str = "json",
        upstream_provider: str = "deepseek",
        allow_provider_fallbacks: bool = False,
        recycle_for_verifier: bool = False,
        provider_turn_policy: str = "single_turn",
        continuation_delay_seconds: int = 0,
        retry_backoff_seconds: str | None = None,
        max_consecutive_provider_failures: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if variant and reasoning_effort and variant.strip() != reasoning_effort.strip():
            raise ValueError("variant and reasoning_effort must agree when both are supplied")
        selected_variant = (variant or reasoning_effort or "high").strip()
        if not selected_variant:
            raise ValueError("OpenCode requires a non-empty reasoning variant")
        if output_format != "json":
            raise ValueError("OpenCode foundry runs require JSON event output")
        if not upstream_provider.strip():
            raise ValueError("OpenRouter runs require an explicit upstream provider")
        if provider_turn_policy != "single_turn":
            raise ValueError("OpenCode episodic campaigns require provider_turn_policy=single_turn")
        if int(continuation_delay_seconds) != 0:
            raise ValueError("single-turn OpenCode campaigns require zero continuation delay")
        self._variant = selected_variant
        self._output_format = output_format
        # OpenRouter's routing fields use lowercase provider slugs such as
        # ``deepseek`` and ``deepinfra``. Freeze that slug in every trajectory.
        self._upstream_provider = upstream_provider.strip().lower()
        self._allow_provider_fallbacks = (
            allow_provider_fallbacks.strip().lower() in {"1", "true", "yes"}
            if isinstance(allow_provider_fallbacks, str)
            else bool(allow_provider_fallbacks)
        )
        self._recycle_for_verifier = (
            recycle_for_verifier.strip().lower() in {"1", "true", "yes"}
            if isinstance(recycle_for_verifier, str)
            else bool(recycle_for_verifier)
        )
        # Fresh-session repetition and provider retry policy are controller-owned.
        # Harbor passes these frozen campaign fields to every installed adapter;
        # accepting them here must not turn one episode into a resumed conversation.
        self._provider_turn_policy = provider_turn_policy
        self._retry_backoff_seconds = retry_backoff_seconds
        self._max_consecutive_provider_failures = max_consecutive_provider_failures
        self._instruction: str | None = None

    @staticmethod
    def name() -> str:
        return "openrouter-opencode-foundry"

    @property
    def _install_agent_template_path(self) -> Path:
        return Path(__file__).parent / "install-opencode-foundry.sh.j2"

    def _require_host_token(self) -> str:
        token = os.environ.get(self._HOST_TOKEN_ENV, "").strip()
        if not token:
            raise ValueError(
                f"{self._HOST_TOKEN_ENV} is not set; export the OpenRouter key securely "
                "in the launch environment"
            )
        return token

    def _model_id(self) -> str:
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("model must use provider/model format")
        provider, model_id = self.model_name.split("/", 1)
        if provider != "openrouter":
            raise ValueError("this adapter requires an openrouter/... model")
        if not model_id or model_id.startswith("~") or model_id.endswith("-latest"):
            raise ValueError("floating OpenRouter model aliases are forbidden; pin an exact slug")
        return model_id

    def _opencode_config(self, model_id: str) -> str:
        config = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "openrouter": {
                    "models": {
                        model_id: {
                            "options": {
                                "provider": {
                                    "order": [self._upstream_provider],
                                    "allow_fallbacks": self._allow_provider_fallbacks,
                                }
                            }
                        }
                    }
                }
            },
        }
        return json.dumps(config, sort_keys=True, separators=(",", ":"))

    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        self._require_host_token()
        model_id = self._model_id()
        config_json = self._opencode_config(model_id)
        output = (EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME).as_posix()
        log = (EnvironmentPaths.agent_dir / self._LOG_FILENAME).as_posix()
        command = f"""
set -euo pipefail
umask 077
agent_home=/home/agent
uploaded_token={shlex.quote(self._UPLOADED_TOKEN_FILE)}
data_home=/tmp/opencode-data
state_home=/tmp/opencode-state
config_home=/tmp/opencode-config
cache_home=/tmp/opencode-cache
auth_file="$data_home/opencode/auth.json"
config_file="$config_home/opencode/opencode.json"
termination="exit"
cleanup() {{
  status=$?
  trap - EXIT INT TERM
  set +e
  mkdir -p /logs/artifacts/process
  runtime_tmp=/logs/artifacts/process/.runtime.json.tmp
  printf '{{"schema_version":"harbor-runtime-0.1.0","exit_code":%d,"termination":"%s"}}\n' \
    "$status" "$termination" > "$runtime_tmp"
  mv "$runtime_tmp" /logs/artifacts/process/runtime.json
  rm -rf /logs/agent/exported-artifacts
  if [ -d /logs/artifacts ]; then
    cp -a /logs/artifacts /logs/agent/exported-artifacts
  fi
  rm -rf "$data_home" "$state_home" "$config_home" "$cache_home" "$uploaded_token"
  touch /logs/artifacts/process/.agent-cleanup-complete
  exit "$status"
}}
on_int() {{ termination=sigint; exit 130; }}
on_term() {{ termination=sigterm; exit 143; }}
trap cleanup EXIT
trap on_int INT
trap on_term TERM
printf '%s\n' "$$" > {shlex.quote(self._WRAPPER_PID_FILE)}
test -s "$uploaded_token"
mkdir -p "$(dirname "$auth_file")" "$(dirname "$config_file")" \
  "$state_home" "$cache_home" /logs/artifacts/process
python - "$uploaded_token" "$auth_file" <<'PY'
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
key = source.read_text(encoding="utf-8").strip()
if not key:
    raise SystemExit("uploaded OpenRouter key is empty")
destination.write_text(
    json.dumps({{"openrouter": {{"type": "api", "key": key}}}}) + "\\n",
    encoding="utf-8",
)
destination.chmod(0o600)
PY
# shellcheck disable=SC2016  # $schema is a literal JSON key.
printf '%s\n' {shlex.quote(config_json)} > "$config_file"
chmod 600 "$config_file"
touch {shlex.quote(output)} {shlex.quote(log)}
chown -R agent:agent "$data_home" "$state_home" "$config_home" "$cache_home" \
  /workspace /logs/artifacts {shlex.quote(output)} {shlex.quote(log)}
cd /workspace
setsid runuser -u agent -- env -i \
  HOME="$agent_home" \
  PATH="/usr/local/bin:/usr/bin:/bin" \
  XDG_DATA_HOME="$data_home" \
  XDG_STATE_HOME="$state_home" \
  XDG_CONFIG_HOME="$config_home" \
  XDG_CACHE_HOME="$cache_home" \
  OPENCODE_FAKE_VCS=git \
  opencode run \
    --pure \
    --model {shlex.quote(self.model_name)} \
    --format {shlex.quote(self._output_format)} \
    --thinking \
    --auto \
    --dir /workspace \
    --variant {shlex.quote(self._variant)} \
    -- {shlex.quote(instruction)} \
    > {shlex.quote(output)} 2> {shlex.quote(log)} &
agent_pid=$!
printf '%s\n' "$agent_pid" > {shlex.quote(self._PID_FILE)}
wait "$agent_pid"
""".strip()
        return [ExecInput(command=command)]

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        token = self._require_host_token()
        self._instruction = instruction
        descriptor, temporary_name = tempfile.mkstemp(prefix="openrouter-key-")
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(token + "\n")
            descriptor = -1
            await environment.upload_file(temporary_name, self._UPLOADED_TOKEN_FILE)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            Path(temporary_name).unlink(missing_ok=True)
        try:
            await super().run(instruction, environment, context)
        except asyncio.CancelledError:
            if self._recycle_for_verifier:
                await checkpoint_cancelled_agent(
                    environment,
                    pid_file=self._PID_FILE,
                    wrapper_pid_file=self._WRAPPER_PID_FILE,
                    cleanup_paths=(
                        self._UPLOADED_TOKEN_FILE,
                        "/tmp/opencode-data",
                        "/tmp/opencode-state",
                        "/tmp/opencode-config",
                        "/tmp/opencode-cache",
                    ),
                )
                await recycle_for_offline_verifier(
                    environment,
                    required_artifact="process/runtime.json",
                )
            raise
        except Exception:
            await environment.exec(
                command=(
                    "rm -rf /tmp/openrouter-api-key-upload /tmp/opencode-data "
                    "/tmp/opencode-state /tmp/opencode-config /tmp/opencode-cache"
                )
            )
            raise
        else:
            if self._recycle_for_verifier:
                await recycle_for_offline_verifier(
                    environment,
                    required_artifact="process/runtime.json",
                )

    @staticmethod
    def _iso(timestamp_ms: int | float | None) -> str | None:
        if timestamp_ms is None:
            return None
        try:
            return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).isoformat()
        except (OSError, OverflowError, TypeError, ValueError):
            return None

    def _parse_events(self) -> list[dict[str, Any]]:
        output = self.logs_dir / self._OUTPUT_FILENAME
        if not output.is_file():
            return []
        events: list[dict[str, Any]] = []
        for line in output.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    def _trajectory(self, events: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not events:
            return None
        session_id = next(
            (str(event["sessionID"]) for event in events if event.get("sessionID")),
            "unknown",
        )
        turns: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        user_message: str | None = None
        user_timestamp: int | float | None = None
        for event in events:
            event_type = event.get("type")
            if event_type == "user" and user_message is None:
                parts = event.get("parts")
                if isinstance(parts, list):
                    texts = [
                        str(part.get("text", ""))
                        for part in parts
                        if isinstance(part, dict) and part.get("type") == "text"
                    ]
                    user_message = "\n".join(text for text in texts if text) or None
                    user_timestamp = event.get("timestamp")
                continue
            if event_type == "step_start":
                current = {
                    "parts": [],
                    "finish": {},
                    "timestamp": event.get("timestamp"),
                }
                continue
            if event_type == "step_finish":
                if current is not None:
                    current["finish"] = event.get("part") or {}
                    turns.append(current)
                    current = None
                continue
            if current is not None and event_type in {"text", "reasoning", "tool_use"}:
                part = event.get("part")
                if isinstance(part, dict):
                    current["parts"].append(part)

        steps: list[dict[str, Any]] = []
        prompt_tokens = completion_tokens = cached_tokens = reasoning_tokens = 0
        total_cost = 0.0
        user_text = user_message or self._instruction
        if user_text:
            user_step: dict[str, Any] = {
                "step_id": 1,
                "source": "user",
                "message": user_text,
            }
            if timestamp := self._iso(user_timestamp):
                user_step["timestamp"] = timestamp
            steps.append(user_step)

        for turn in turns:
            texts: list[str] = []
            reasoning: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            observations: list[dict[str, Any]] = []
            for part in turn["parts"]:
                part_type = part.get("type")
                if part_type == "text" and part.get("text"):
                    texts.append(str(part["text"]))
                elif part_type == "reasoning" and part.get("text"):
                    reasoning.append(str(part["text"]))
                elif part_type == "tool":
                    state = part.get("state") or {}
                    arguments = state.get("input")
                    if not isinstance(arguments, dict):
                        arguments = {"value": arguments} if arguments is not None else {}
                    call_id = str(part.get("callID") or part.get("id") or "")
                    tool_calls.append(
                        {
                            "tool_call_id": call_id,
                            "function_name": str(part.get("tool") or ""),
                            "arguments": arguments,
                        }
                    )
                    if state.get("output") is not None:
                        observations.append(
                            {
                                "source_call_id": call_id or None,
                                "content": str(state["output"]),
                            }
                        )
            finish = turn.get("finish") or {}
            tokens = finish.get("tokens") or {}
            cache = tokens.get("cache") or {}
            input_count = int(tokens.get("input") or 0)
            output_count = int(tokens.get("output") or 0)
            cache_read = int(cache.get("read") or 0)
            reasoning_count = int(tokens.get("reasoning") or 0)
            cost = float(finish.get("cost") or 0.0)
            prompt_tokens += input_count + cache_read
            completion_tokens += output_count
            cached_tokens += cache_read
            reasoning_tokens += reasoning_count
            total_cost += cost
            step: dict[str, Any] = {
                "step_id": len(steps) + 1,
                "source": "agent",
                "model_name": self.model_name,
                "message": "\n".join(texts),
                "metrics": {
                    "prompt_tokens": input_count + cache_read,
                    "completion_tokens": output_count,
                    "cached_tokens": cache_read or None,
                    "cost_usd": cost or None,
                    "extra": {"reasoning_tokens": reasoning_count} if reasoning_count else None,
                },
            }
            if timestamp := self._iso(turn.get("timestamp")):
                step["timestamp"] = timestamp
            if reasoning:
                step["reasoning_content"] = "\n\n".join(reasoning)
            if tool_calls:
                step["tool_calls"] = tool_calls
            if observations:
                step["observation"] = {"results": observations}
            steps.append(step)
        if not any(step["source"] == "agent" for step in steps):
            return None
        return {
            "schema_version": "ATIF-v1.6",
            "session_id": session_id,
            "agent": {
                "name": self.name(),
                "version": self.version() or "unknown",
                "model_name": self.model_name,
                "extra": {
                    "openrouter_upstream_provider": self._upstream_provider,
                    "openrouter_allow_fallbacks": self._allow_provider_fallbacks,
                    "reasoning_variant": self._variant,
                },
            },
            "steps": steps,
            "final_metrics": {
                "total_prompt_tokens": prompt_tokens or None,
                "total_completion_tokens": completion_tokens or None,
                "total_cached_tokens": cached_tokens or None,
                "total_cost_usd": total_cost or None,
                "total_steps": len(steps),
                "extra": {"reasoning_tokens": reasoning_tokens} if reasoning_tokens else None,
            },
        }

    def populate_context_post_run(self, context: AgentContext) -> None:
        trajectory = self._trajectory(self._parse_events())
        if trajectory is None:
            return
        (self.logs_dir / "trajectory.json").write_text(
            json.dumps(trajectory, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metrics = trajectory["final_metrics"]
        context.cost_usd = metrics["total_cost_usd"]
        context.n_input_tokens = metrics["total_prompt_tokens"] or 0
        context.n_cache_tokens = metrics["total_cached_tokens"] or 0
        context.n_output_tokens = metrics["total_completion_tokens"] or 0
