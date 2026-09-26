"""Harbor 0.1.44 Claude Code adapter for clean-recycle foundry runs.

The launcher forwards a local Claude subscription token only through
``CLAUDE_CODE_OAUTH_TOKEN``. This adapter runs the pinned Claude Code CLI as the
unprivileged ``agent`` user with customizations disabled, removes its temporary
configuration on exit, and replaces the provider-connected container with a
clean offline container before Harbor uploads hidden verifier tests.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import tempfile
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent, ExecInput
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths
from offline_verifier import checkpoint_cancelled_agent, recycle_for_offline_verifier


class ClaudeCodeFoundry(BaseInstalledAgent):
    """Claude Code authenticated by an ephemeral subscription OAuth token."""

    _OUTPUT_FILENAME = "claude-code.jsonl"
    _LOG_FILENAME = "claude-code.log"
    _PID_FILE = "/tmp/claude-agent-session.pid"
    _WRAPPER_PID_FILE = "/tmp/claude-agent-wrapper.pid"
    _TOKEN_FILE = "/tmp/claude-foundry-oauth-token"
    # The host variable that carries the credential and the container variable
    # the CLI reads. A provider-routed subclass may differ in both.
    _HOST_TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"
    _CONTAINER_TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"

    def __init__(
        self,
        *args,
        reasoning_effort: str | None = None,
        max_turns: int | None = None,
        output_format: str = "stream-json",
        recycle_for_verifier: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._reasoning_effort = reasoning_effort
        # Claude Code 2.1.218 has no CLI turn-limit flag. The value remains in
        # TASK_SPEC.toml as an agent-visible ceiling; Harbor's wall timeout is
        # the externally enforced stopping rule for this adapter.
        self._max_turns = int(max_turns) if max_turns is not None else None
        self._output_format = output_format
        self._recycle_for_verifier = (
            recycle_for_verifier.strip().lower() in {"1", "true", "yes"}
            if isinstance(recycle_for_verifier, str)
            else bool(recycle_for_verifier)
        )

    @staticmethod
    def name() -> str:
        return "claude-code-foundry"

    @property
    def _install_agent_template_path(self) -> Path:
        return Path(__file__).parent / "install-claude-code-foundry.sh.j2"

    def _provider_environment(self) -> dict[str, str]:
        """Extra non-secret variables injected into the scrubbed agent environment.

        The CLI runs under ``env -i``, so a subclass that routes the same binary
        to a different provider must declare its endpoint and model aliases here.
        Credentials never travel through this mapping.
        """

        return {}

    def _require_host_token(self) -> str:
        token = os.environ.get(self._HOST_TOKEN_ENV, "")
        if not token.strip():
            raise ValueError(
                f"{self._HOST_TOKEN_ENV} is not set; the launcher must forward "
                "a local provider credential"
            )
        return token

    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        self._require_host_token()
        if not self.model_name:
            raise ValueError("a pinned Claude model is required")
        if self._output_format != "stream-json":
            raise ValueError("foundry runs require Claude stream-json output")

        model = self.model_name.split("/")[-1]
        effort_flag = (
            f"--effort {shlex.quote(self._reasoning_effort)} " if self._reasoning_effort else ""
        )
        provider_env = "".join(
            f" {name}={shlex.quote(value)}"
            for name, value in sorted(self._provider_environment().items())
        )
        output = (EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME).as_posix()
        log = (EnvironmentPaths.agent_dir / self._LOG_FILENAME).as_posix()
        agent_dir = EnvironmentPaths.agent_dir.as_posix()
        command = f"""
set -euo pipefail
umask 077
agent_home="/home/agent"
config_dir="$agent_home/.claude-foundry"
token_file={shlex.quote(self._TOKEN_FILE)}
termination="exit"
cleanup() {{
  status=$?
  trap - EXIT INT TERM
  set +e
  mkdir -p /logs/artifacts/process
  runtime_tmp="/logs/artifacts/process/.runtime.json.tmp"
  printf '{{"schema_version":"harbor-runtime-0.1.0","exit_code":%d,"termination":"%s"}}\\n' \
    "$status" "$termination" > "$runtime_tmp"
  mv "$runtime_tmp" /logs/artifacts/process/runtime.json
  rm -rf /logs/agent/exported-artifacts
  if [ -d /logs/artifacts ]; then
    cp -a /logs/artifacts /logs/agent/exported-artifacts
  fi
  rm -rf "$config_dir"
  rm -f "$token_file"
  touch /logs/artifacts/process/.agent-cleanup-complete
  exit "$status"
}}
on_int() {{
  termination="sigint"
  exit 130
}}
on_term() {{
  termination="sigterm"
  exit 143
}}
trap cleanup EXIT
trap on_int INT
trap on_term TERM
mkdir -p "$config_dir" {shlex.quote(agent_dir)} /logs/artifacts/process
printf '%s\\n' "$$" > {shlex.quote(self._WRAPPER_PID_FILE)}
test -s "$token_file"
chmod 600 "$token_file"
chown -R agent:agent "$config_dir" /workspace /logs/artifacts
touch {shlex.quote(output)} {shlex.quote(log)}
chown agent:agent {shlex.quote(output)} {shlex.quote(log)}
cd /workspace
setsid runuser -u agent -- env -i \
  HOME="$agent_home" \
  PATH="/usr/local/bin:/usr/bin:/bin" \
  {self._CONTAINER_TOKEN_ENV}="$(tr -d '\\r\\n' < "$token_file")" \
  CLAUDE_CONFIG_DIR="$config_dir" \
  CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
  IS_SANDBOX=1{provider_env} \
  claude --print --verbose \
  --output-format {shlex.quote(self._output_format)} \
  --permission-mode bypassPermissions \
  --model {shlex.quote(model)} \
  {effort_flag}\
  --safe-mode --disable-slash-commands --no-chrome \
  --no-session-persistence --strict-mcp-config \
  --mcp-config '{{"mcpServers":{{}}}}' \
  -p {shlex.quote(instruction)} \
  > {shlex.quote(output)} 2> {shlex.quote(log)} &
agent_pid=$!
printf '%s\\n' "$agent_pid" > {shlex.quote(self._PID_FILE)}
wait "$agent_pid"
""".strip()
        return [ExecInput(command=command)]

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        credential = self._require_host_token()
        descriptor, temporary_name = tempfile.mkstemp(prefix="claude-foundry-oauth-")
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(credential.encode("utf-8"))
            descriptor = -1
            await environment.upload_file(temporary_name, self._TOKEN_FILE)
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
                        "/home/agent/.claude-foundry",
                        self._TOKEN_FILE,
                    ),
                )
                await recycle_for_offline_verifier(
                    environment,
                    required_artifact="process/runtime.json",
                )
            raise
        except Exception:
            await environment.exec(command=f"rm -f {shlex.quote(self._TOKEN_FILE)}")
            raise
        else:
            if self._recycle_for_verifier:
                await recycle_for_offline_verifier(
                    environment,
                    required_artifact="process/runtime.json",
                )

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Lift final usage and cost metadata from Claude's JSONL stream."""

        output = self.logs_dir / self._OUTPUT_FILENAME
        if not output.exists():
            self.logger.warning("Claude output not found at %s", output)
            return
        records: list[dict] = []
        for line in output.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
        result = next(
            (record for record in reversed(records) if record.get("type") == "result"),
            {},
        )
        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        context.n_input_tokens = int(usage.get("input_tokens") or 0)
        context.n_cache_tokens = int(usage.get("cache_read_input_tokens") or 0)
        context.n_output_tokens = int(usage.get("output_tokens") or 0)
        if result.get("total_cost_usd") is not None:
            context.cost_usd = result["total_cost_usd"]
