# Agent orchestration

The release supports three coding-agent adapters in `harbor/agents/`: Codex,
Claude Code and OpenCode/OpenRouter. Provider-independent controller behavior
lives in `question_foundry.sequential` and `scripts.run_sequential_campaign`.

See [setup](../../../docs/SETUP.md),
[authentication](../../../docs/AUTHENTICATION.md), and the
[agent presets](../../../configs/agents/README.md).

To add a model, register its model name, CLI version, reasoning level, and output
format in the campaign validator and configuration files. A new agent interface
also needs credential removal, a clean offline verification container, validation
of the provider's response records, and an authentication test. Supporting an agent
tool does not automatically allow every model available through that tool.
