"""Harbor 0.1.44 Codex adapter using a local subscription OAuth file.

The built-in adapter only materializes ``OPENAI_API_KEY``. This subclass copies
the host's ``~/.codex/auth.json`` into a mode-0600 temporary file, uploads it to
the disposable container, and unlinks the host copy immediately. The OAuth value
never enters ``ExecInput.env`` or a host process argument.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import tempfile
from pathlib import Path

from harbor.agents.installed.base import ExecInput
from harbor.agents.installed.codex import Codex
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths
from offline_verifier import checkpoint_cancelled_agent, recycle_for_offline_verifier


class CodexOAuth(Codex):
    """Codex CLI authenticated by a copied, writable OAuth session."""

    _UPLOADED_AUTH_FILE = "/tmp/codex-oauth-upload.json"
    _HOST_AUTH_FILE_ENV = "CODEX_AUTH_FILE"
    _PID_FILE = "/tmp/codex-agent-session.pid"
    _WRAPPER_PID_FILE = "/tmp/codex-agent-wrapper.pid"
    _LOG_FILENAME = "codex.log"

    def __init__(
        self,
        *args,
        reasoning_effort: str | None = None,
        max_turns: int | None = None,
        output_format: str = "json",
        recycle_for_verifier: bool = False,
        provider_turn_policy: str = "single_turn",
        continuation_delay_seconds: int = 0,
        retry_backoff_seconds: str = "15,30,60",
        max_consecutive_provider_failures: int = 3,
        **kwargs,
    ):
        super().__init__(*args, reasoning_effort=reasoning_effort, **kwargs)
        # Codex 0.145.0 has no hard turn-count flag. The controller places this
        # value in TASK_SPEC.toml as an agent-visible ceiling; Harbor's episode
        # timeout remains the external limit.
        self._max_turns = int(max_turns) if max_turns is not None else None
        self._output_format = output_format
        self._recycle_for_verifier = (
            recycle_for_verifier.strip().lower() in {"1", "true", "yes"}
            if isinstance(recycle_for_verifier, str)
            else bool(recycle_for_verifier)
        )
        if provider_turn_policy not in {"single_turn", "resume_until_timeout"}:
            raise ValueError("provider_turn_policy must be single_turn or resume_until_timeout")
        self._provider_turn_policy = provider_turn_policy
        self._continuation_delay_seconds = int(continuation_delay_seconds)
        if self._provider_turn_policy == "single_turn":
            if self._continuation_delay_seconds != 0:
                raise ValueError("single_turn requires continuation_delay_seconds=0")
        elif not 5 <= self._continuation_delay_seconds <= 120:
            raise ValueError("resume_until_timeout requires continuation_delay_seconds in [5, 120]")
        try:
            self._retry_backoff_seconds = [
                int(item) for item in retry_backoff_seconds.split(",") if item
            ]
        except (AttributeError, ValueError) as exc:
            raise ValueError("retry_backoff_seconds must be comma-separated integers") from exc
        if (
            not self._retry_backoff_seconds
            or self._retry_backoff_seconds != sorted(self._retry_backoff_seconds)
            or any(not 5 <= seconds <= 60 for seconds in self._retry_backoff_seconds)
        ):
            raise ValueError("retry_backoff_seconds must be increasing values in [5, 60]")
        self._max_consecutive_provider_failures = int(max_consecutive_provider_failures)
        if not 1 <= self._max_consecutive_provider_failures <= 10:
            raise ValueError("max_consecutive_provider_failures must be in [1, 10]")

    @staticmethod
    def name() -> str:
        return "codex-oauth"

    @property
    def _install_agent_template_path(self) -> Path:
        return Path(__file__).parent / "install-codex-foundry.sh.j2"

    def _require_host_auth_file(self) -> Path:
        configured = os.environ.get(self._HOST_AUTH_FILE_ENV, "")
        path = Path(configured).expanduser() if configured else Path.home() / ".codex/auth.json"
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(
                f"missing Codex OAuth file at {path}; run `codex login` or set "
                f"{self._HOST_AUTH_FILE_ENV}"
            )
        return path

    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        self._require_host_auth_file()
        if not self.model_name:
            raise ValueError("Codex OAuth runs require an explicit model name")
        if self._output_format != "json":
            raise ValueError("Codex foundry runs require JSONL output")

        model = self.model_name.split("/")[-1]
        effort_flag = (
            f"-c model_reasoning_effort={shlex.quote(self._reasoning_effort)} "
            if self._reasoning_effort
            else ""
        )
        codex_home = EnvironmentPaths.agent_dir.as_posix()
        output = (EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME).as_posix()
        log = (EnvironmentPaths.agent_dir / self._LOG_FILENAME).as_posix()
        turns = (EnvironmentPaths.agent_dir / "provider-turns.jsonl").as_posix()
        continuation_prompt = (
            "The Harbor session is still active. Continue the assigned foundry task now; "
            "do not merely summarize or stop because a checkpoint exists. Re-read "
            "TASK_SPEC.toml and obey its candidate_limit_policy or legacy max_candidates. "
            "Preserve every valid durable checkpoint. Under time_bounded_append_only, "
            "immediately execute the next complete design-build-test-repair-checkpoint "
            "loop and never replace or delete prior checkpoints. Under a legacy bounded "
            "task, remain within its declared bounds. For a funnel, preserve append-only "
            "events and replenish/rerank the queue when proposal_refresh_policy requires "
            "it. Use "
            "/workspace/scratch for temporary files and run the public submission-envelope "
            "check before yielding."
        )
        turn_record_format = (
            '{"schema_version":"provider-turn-0.1.0","turn_index":%d,'
            '"kind":"%s","exit_code":%d,"observed_at":"%s"}\\n'
        )
        command = f"""
set -euo pipefail
umask 077
agent_home=/home/agent
export CODEX_HOME={shlex.quote(codex_home)}
secret_dir=/tmp/codex-oauth
auth_link="$CODEX_HOME/auth.json"
uploaded_auth={shlex.quote(self._UPLOADED_AUTH_FILE)}
provider_turn_policy={shlex.quote(self._provider_turn_policy)}
continuation_delay={self._continuation_delay_seconds}
retry_delays=({" ".join(str(item) for item in self._retry_backoff_seconds)})
max_provider_failures={self._max_consecutive_provider_failures}
continuation_prompt={shlex.quote(continuation_prompt)}
turns_file={shlex.quote(turns)}
termination=exit
active_turn_index=0
active_turn_kind=setup
turn_recorded=1
cleanup() {{
  status=$?
  trap - EXIT INT TERM
  set +e
  if [ "$turn_recorded" -eq 0 ]; then
    record_turn "$active_turn_index" "${{active_turn_kind}}_interrupted" "$status"
    turn_recorded=1
  fi
  mkdir -p /logs/artifacts/process
  runtime_tmp=/logs/artifacts/process/.runtime.json.tmp
  printf '{{"schema_version":"harbor-runtime-0.1.0","exit_code":%d,"termination":"%s"}}\\n' \
    "$status" "$termination" > "$runtime_tmp"
  mv "$runtime_tmp" /logs/artifacts/process/runtime.json
  rm -rf /logs/agent/exported-artifacts
  if [ -d /logs/artifacts ]; then
    cp -a /logs/artifacts /logs/agent/exported-artifacts
  fi
  rm -rf "$secret_dir" "$uploaded_auth"
  rm -f "$auth_link"
  touch /logs/artifacts/process/.agent-cleanup-complete
  exit "$status"
}}
on_int() {{
  termination=sigint
  exit 130
}}
on_term() {{
  termination=sigterm
  exit 143
}}
trap cleanup EXIT
trap on_int INT
trap on_term TERM
mkdir -p "$secret_dir" "$CODEX_HOME" /logs/artifacts/process
printf '%s\\n' "$$" > {shlex.quote(self._WRAPPER_PID_FILE)}
test -s "$uploaded_auth"
mv "$uploaded_auth" "$secret_dir/auth.json"
chmod 600 "$secret_dir/auth.json"
ln -sf "$secret_dir/auth.json" "$auth_link"
touch {shlex.quote(output)} {shlex.quote(log)} "$turns_file"
chown -R agent:agent "$secret_dir" "$CODEX_HOME" /workspace /logs/artifacts
chown agent:agent {shlex.quote(output)} {shlex.quote(log)} "$turns_file"
cd /workspace
record_turn() {{
  turn_index="$1"
  turn_kind="$2"
  turn_status="$3"
  observed_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf {shlex.quote(turn_record_format)} \
    "$turn_index" "$turn_kind" "$turn_status" "$observed_at" >> "$turns_file"
}}
run_initial() {{
  setsid runuser -u agent -- env -i \
    HOME="$agent_home" \
    PATH="/usr/local/bin:/usr/bin:/bin" \
    CODEX_HOME={shlex.quote(codex_home)} \
    codex exec \
    --dangerously-bypass-approvals-and-sandbox \
    --skip-git-repo-check \
    --ignore-user-config \
    -C /workspace \
    --model {shlex.quote(model)} \
    --json \
    --enable unified_exec \
    {effort_flag}\
    -- {shlex.quote(instruction)} \
    >> {shlex.quote(output)} 2>> {shlex.quote(log)} &
  agent_pid=$!
  printf '%s\\n' "$agent_pid" > {shlex.quote(self._PID_FILE)}
  wait "$agent_pid"
}}
run_resume() {{
  setsid runuser -u agent -- env -i \
    HOME="$agent_home" \
    PATH="/usr/local/bin:/usr/bin:/bin" \
    CODEX_HOME={shlex.quote(codex_home)} \
    codex exec resume \
    --dangerously-bypass-approvals-and-sandbox \
    --skip-git-repo-check \
    --ignore-user-config \
    --model {shlex.quote(model)} \
    --json \
    --enable unified_exec \
    {effort_flag}\
    --last \
    -- "$continuation_prompt" \
    >> {shlex.quote(output)} 2>> {shlex.quote(log)} &
  agent_pid=$!
  printf '%s\\n' "$agent_pid" > {shlex.quote(self._PID_FILE)}
  wait "$agent_pid"
}}

set +e
turn_index=1
failure_streak=0
session_started=0
active_turn_index="$turn_index"
active_turn_kind=initial
turn_recorded=0
run_initial
turn_status=$?
[ "$turn_status" -ne 0 ] || session_started=1
record_turn "$turn_index" initial "$turn_status"
turn_recorded=1
if [ "$provider_turn_policy" = single_turn ]; then
  exit "$turn_status"
fi

while true; do
  if [ "$turn_status" -eq 0 ]; then
    failure_streak=0
    delay="$continuation_delay"
  else
    failure_streak=$((failure_streak + 1))
    if [ "$failure_streak" -ge "$max_provider_failures" ]; then
      exit "$turn_status"
    fi
    retry_index=$((failure_streak - 1))
    [ "$retry_index" -lt "${{#retry_delays[@]}}" ] || \
      retry_index=$((${{#retry_delays[@]}} - 1))
    delay="${{retry_delays[$retry_index]}}"
  fi
  printf '%s\\n' "$$" > {shlex.quote(self._PID_FILE)}
  sleep "$delay"
  turn_index=$((turn_index + 1))
  if [ "$session_started" -eq 0 ]; then
    active_turn_kind=initial_retry
    active_turn_index="$turn_index"
    turn_recorded=0
    run_initial
    turn_kind=initial_retry
  else
    active_turn_kind=resume
    active_turn_index="$turn_index"
    turn_recorded=0
    run_resume
    turn_kind=resume
  fi
  turn_status=$?
  [ "$turn_status" -ne 0 ] || session_started=1
  record_turn "$turn_index" "$turn_kind" "$turn_status"
  turn_recorded=1
done
""".strip()
        return [ExecInput(command=command)]

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        source = self._require_host_auth_file()
        descriptor, temporary_name = tempfile.mkstemp(prefix="codex-oauth-")
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(source.read_bytes())
            descriptor = -1
            await environment.upload_file(temporary_name, self._UPLOADED_AUTH_FILE)
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
                        "/logs/agent/auth.json",
                        "/tmp/codex-oauth",
                        self._UPLOADED_AUTH_FILE,
                    ),
                )
                await recycle_for_offline_verifier(
                    environment,
                    required_artifact="process/runtime.json",
                )
            raise
        except Exception:
            await environment.exec(
                command=f"rm -rf /tmp/codex-oauth {shlex.quote(self._UPLOADED_AUTH_FILE)}"
            )
            raise
        else:
            if self._recycle_for_verifier:
                await recycle_for_offline_verifier(
                    environment,
                    required_artifact="process/runtime.json",
                )
