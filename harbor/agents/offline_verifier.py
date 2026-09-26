"""Harbor 0.1.44 clean-container handoff for protected offline verification."""

from __future__ import annotations

import asyncio
import shlex
from pathlib import PurePosixPath

from harbor.environments.base import BaseEnvironment

EXPORTED_ARTIFACTS = "/logs/agent/exported-artifacts"


def _safe_cleanup_paths(paths: tuple[str, ...]) -> str:
    """Render explicit container paths for timeout cleanup.

    Credential cleanup must never accept a relative path or the container root.
    The returned string is suitable only as the operand list of ``rm -rf --``.
    """

    rendered = []
    for raw in paths:
        path = PurePosixPath(raw)
        if not path.is_absolute() or path == PurePosixPath("/"):
            raise ValueError(f"unsafe timeout cleanup path: {raw!r}")
        rendered.append(shlex.quote(path.as_posix()))
    return " ".join(rendered)


async def checkpoint_cancelled_agent(
    environment: BaseEnvironment,
    *,
    pid_file: str,
    wrapper_pid_file: str | None = None,
    cleanup_paths: tuple[str, ...] = (),
) -> None:
    """Terminate the provider and wrapper, scrub secrets, and record cancellation.

    Harbor 0.1.44 cancels its host-side ``docker compose exec`` process without
    reliably forwarding a signal to the command in the container. Installed
    agents therefore track both the provider process-group leader and the shell
    wrapper. The wrapper must be terminated too: a deadline-driven wrapper may
    otherwise interpret the killed provider as retryable and bypass its EXIT
    cleanup trap.
    """

    cleanup_operands = _safe_cleanup_paths(cleanup_paths)
    cleanup_command = f"rm -rf -- {cleanup_operands}; " if cleanup_operands else ""
    wrapper_file = wrapper_pid_file or ""
    result = await environment.exec(
        command=(
            "set -u; "
            f'pid_file="{pid_file}"; '
            f'wrapper_pid_file="{wrapper_file}"; '
            'cleanup_marker="/logs/artifacts/process/.agent-cleanup-complete"; '
            'pid=""; wrapper_pid=""; '
            'if [ -s "$pid_file" ]; then '
            '  pid="$(cat "$pid_file")"; '
            '  case "$pid" in *[!0-9]*|"") pid="" ;; esac; '
            '  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then '
            '    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true; '
            "  fi; "
            "fi; "
            'if [ -n "$wrapper_pid_file" ] && [ -s "$wrapper_pid_file" ]; then '
            '  wrapper_pid="$(cat "$wrapper_pid_file")"; '
            '  case "$wrapper_pid" in *[!0-9]*|"") wrapper_pid="" ;; esac; '
            '  if [ -n "$wrapper_pid" ] && [ "$wrapper_pid" != "$pid" ] '
            '     && kill -0 "$wrapper_pid" 2>/dev/null; then '
            '    kill -TERM "$wrapper_pid" 2>/dev/null || true; '
            "  fi; "
            "fi; "
            "count=0; "
            'while [ ! -e "$cleanup_marker" ] && [ "$count" -lt 50 ]; do '
            "  sleep 0.1; count=$((count + 1)); "
            "done; "
            'if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then '
            '  kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true; '
            "fi; "
            'if [ -n "$wrapper_pid" ] && kill -0 "$wrapper_pid" 2>/dev/null; then '
            '  kill -KILL "$wrapper_pid" 2>/dev/null || true; '
            "fi; " + cleanup_command + "mkdir -p /logs/artifacts/process; "
            'runtime_tmp="/logs/artifacts/process/.runtime.json.cancelled"; '
            "printf '%s\\n' "
            '\'{"schema_version":"harbor-runtime-0.1.0","exit_code":124,'
            '"termination":"cancelled"}\' '
            '> "$runtime_tmp"; '
            'mv "$runtime_tmp" /logs/artifacts/process/runtime.json; '
            'rm -f "$cleanup_marker" "$pid_file"; '
            'if [ -n "$wrapper_pid_file" ]; then rm -f "$wrapper_pid_file"; fi'
        )
    )
    if result.return_code != 0:
        detail = (result.stdout or "") + (result.stderr or "")
        raise RuntimeError(f"failed to checkpoint cancelled agent: {detail[-1000:]}")


async def _recycle_for_offline_verifier(
    environment: BaseEnvironment,
    *,
    required_artifact: str,
) -> None:
    """Preserve artifacts, replace the agent container, and fail closed offline."""

    preserve = await environment.exec(
        command=(
            "set -eu; "
            f"rm -rf {EXPORTED_ARTIFACTS}; "
            f"mkdir -p {EXPORTED_ARTIFACTS}; "
            f"cp -a /logs/artifacts/. {EXPORTED_ARTIFACTS}/; "
            f"test -f {EXPORTED_ARTIFACTS}/{required_artifact}"
        )
    )
    if preserve.return_code != 0:
        detail = (preserve.stdout or "") + (preserve.stderr or "")
        raise RuntimeError(f"agent produced no required artifact: {detail[-1000:]}")

    await environment.stop(delete=False)
    environment.task_env_config.allow_internet = False
    await environment.start(force_build=False)
    restore = await environment.exec(
        command=(
            "set -eu; "
            "test ! -e /tests; "
            "test ! -e /agent-phase-sentinel; "
            "test ! -e /root/.grok/auth.json; "
            "test ! -e /home/agent/.grok/auth.json; "
            "test ! -e /tmp/grok-foundry-oauth.json; "
            "test ! -e /tmp/codex-oauth-upload.json; "
            "test ! -e /logs/agent/auth.json; "
            "test ! -L /logs/agent/auth.json; "
            "test ! -e /root/.claude; "
            "test ! -e /home/agent/.claude-foundry; "
            "test ! -e /tmp/claude-foundry-oauth-token; "
            "test ! -e /tmp/zai-coding-plan-key; "
            "test ! -e /tmp/openrouter-api-key-upload; "
            "test ! -e /tmp/opencode-data; "
            "test ! -e /tmp/opencode-state; "
            "test ! -e /tmp/opencode-config; "
            "test ! -e /tmp/opencode-cache; "
            "mkdir -p /logs/artifacts; "
            "rm -rf /logs/artifacts/*; "
            f"cp -a {EXPORTED_ARTIFACTS}/. /logs/artifacts/; "
            f"test -f /logs/artifacts/{required_artifact}; "
            "touch /offline-verifier-ready; "
            "python - <<'PY'\n"
            "import socket\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 53), timeout=1.0)\n"
            "except OSError:\n"
            "    raise SystemExit(0)\n"
            "raise SystemExit('offline verifier container still has network')\n"
            "PY"
        )
    )
    if restore.return_code != 0:
        detail = (restore.stdout or "") + (restore.stderr or "")
        raise RuntimeError(f"failed to prepare clean offline verifier: {detail[-1000:]}")


async def recycle_for_offline_verifier(
    environment: BaseEnvironment,
    *,
    required_artifact: str,
) -> None:
    """Complete the protected handoff even if Harbor's agent timer expires.

    Harbor 0.1.44 applies the task's agent timeout to the adapter's entire
    ``run`` coroutine, including the security-critical container recycle that
    follows provider execution.  Shielding only the inner task is insufficient:
    the verifier could otherwise start while recycling still runs in the
    background.  This wrapper waits for the shielded handoff to finish and then
    re-raises cancellation, preserving both the timeout outcome and the clean,
    networkless verifier boundary.
    """

    recycle = asyncio.create_task(
        _recycle_for_offline_verifier(
            environment,
            required_artifact=required_artifact,
        )
    )
    try:
        await asyncio.shield(recycle)
    except asyncio.CancelledError:
        await recycle
        raise
