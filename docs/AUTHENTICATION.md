# Authentication: Claude Code, Codex, OpenCode

Keep credentials outside the repository. Do not put them in campaign TOML files, Docker
build arguments, or command-line arguments. Use a secret manager to populate environment
variables or a login file readable only by its owner (mode 0600). The examples below use
Bash and avoid echoing secrets. Do not use `set -x`, which prints shell commands and can
expose credentials.

The adapters install the CLI version specified in the campaign configuration inside the
container. Codex and Claude also require that exact CLI version on the host; the
controller checks it before a run. OpenCode needs no host installation. The supplied
configurations use: Codex `0.147.0`, Claude Code `2.1.224`, OpenCode `1.18.11`. Check
that your account can access the configured model before starting a campaign.

## Credential names

The credential inputs are `CODEX_AUTH_FILE`, `CLAUDE_CODE_OAUTH_TOKEN`, and
`OPENROUTER_API_KEY`. They identify the provider or agent tool; no project-specific
API-key name or Keychain service is required. Variables ending in `_CONFIRMED` record
your approval or tested model access and must not contain a secret.

## Codex

Install Node.js 22 (including npm) on the host, then install the pinned CLI and log in:

```bash
npm install -g @openai/codex@0.147.0
codex login
export CODEX_AUTH_FILE="$HOME/.codex/auth.json"
unset OPENAI_API_KEY OPENAI_BASE_URL
```

The preset selects `gpt-5.6-sol` with `max` reasoning. The auth file is uploaded as a
temporary private copy; the whole home directory is never mounted into the coding-agent
container.

## Claude Code

Install the exact host version selected in your campaign using the vendor installer:

```bash
curl -fsSL https://claude.ai/install.sh | bash -s -- 2.1.224
claude --version
claude setup-token
```

Securely supply the resulting setup token:

```bash
read -r -s -p 'Claude setup token: ' CLAUDE_CODE_OAUTH_TOKEN
printf '\n'
export CLAUDE_CODE_OAUTH_TOKEN
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL
```

The preset selects `claude-opus-5` with `high` reasoning. Use a setup token suitable for
the intended run duration. The adapter does not implement an OAuth refresh service.

## OpenCode through OpenRouter

```bash
read -r -s -p 'OpenRouter API key: ' OPENROUTER_API_KEY
printf '\n'
export OPENROUTER_API_KEY
```

The preset selects `openrouter/deepseek/deepseek-v4-flash-0731`, `high` reasoning,
DeepInfra routing and no provider fallback. Credentials are written to temporary
OpenCode auth configuration, not exposed in invocation arguments. No other coding-agent
backend is packaged or supported by the release CLI.

## Test authentication (paid)

After preparing a campaign, test authentication with its configured agent and model:

```bash
uv run --extra runner python -m scripts.auth_smoke \
  --campaign campaigns/YOUR_ID.toml --out artifacts/auth-YOUR_ID --confirm YOUR_ID
```

Each check starts one agent, with a five-minute agent timeout and no retry. Run the
command once per agent you intend to use.

This makes a small **paid model call** and asks the agent to write a file with specified
text. The script checks the file and scans saved outputs for the supplied credential.
Inspect the model identity reported in the output as well: writing the file successfully
does not establish which model served the request.

Only after that inspection, set the matching run variables. Substitute the exact model
from your campaign if you added another supported configuration:

```bash
# Codex
export CODEX_SEQUENTIAL_RUN_CONFIRMED=YES
export CODEX_MODEL_ENTITLEMENT_CONFIRMED=gpt-5.6-sol

# Claude Code
export CLAUDE_SEQUENTIAL_RUN_CONFIRMED=YES
export CLAUDE_MODEL_ENTITLEMENT_CONFIRMED=claude-opus-5
export CLAUDE_OAUTH_REFRESH_CONFIRMED=YES

# OpenCode
export OPENROUTER_SEQUENTIAL_RUN_CONFIRMED=YES
export OPENROUTER_MODEL_ENTITLEMENT_CONFIRMED=openrouter/deepseek/deepseek-v4-flash-0731
```

`CLAUDE_OAUTH_REFRESH_CONFIRMED` confirms that you have supplied a usable token. Despite
the variable name, the release does not refresh that token.

Model-feedback-steered generation additionally uses an **OpenRouter key for evaluator
calls**, even with a Codex or Claude coding agent. Its confirmation must equal the exact
campaign ID:

```bash
export OPENROUTER_FEEDBACK_RUN_CONFIRMED=YOUR_ID
```

The evaluator models, request limits, and cost budget are specified in
`[feedback_policy]`. These are separate calls from coding-agent usage.

For an unattended run, load secrets from your own secret manager in the launching
process. Never publish `artifacts/`, provider traces, copied auth files, or shell
history. The `prepare.sh` scripts and the three agent-specific shortcuts in `examples/`
only create configurations. `examples/run_campaign.sh` launches a paid campaign;
`examples/shared_examples/render.sh` renders reference worlds locally.
