# Coding-agent configurations

This directory provides one example configuration for each supported agent tool: Codex,
Claude Code, and OpenCode through OpenRouter. Each JSON file specifies the model,
reasoning level, output format, CLI version, and method for recording usage costs.
Credentials are supplied separately; see [authentication](../../docs/AUTHENTICATION.md).

The CLI versions are fixed. To use another tested version, pass `--cli-version` when
preparing a campaign. Changing the model also requires updating the campaign validator
and relevant tests. The paper evaluates five model/reasoning configurations; these three
files are examples rather than the full experiment matrix.
