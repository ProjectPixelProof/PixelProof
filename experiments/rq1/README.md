# Profile-steered generation

This directory contains the campaign template for profile-steered generation (`--rq 1`).
See the [walkthrough](../../examples/rq1/README.md) for the paper context and run
commands.

The agent receives one fixed `paper-semantic@0.3.0` discovery profile, all five
text-only example questions together, and the implementation demonstrations. It invents
a new world satisfying the profile rather than implementing a supplied profile-specific
program.

```bash
uv run --extra runner python -m scripts.release prepare --rq 1 --agent codex \
  --id rq1_tracing_demo --profile tracing
uv run --extra runner python -m scripts.release preview --campaign campaigns/rq1_tracing_demo.toml \
  --out artifacts/rq1-tracing-preview
```

Use `--agent claude` or `--agent opencode` with the same command. To prepare the nine
separate profile campaigns:

```bash
uv run --extra runner python -m scripts.release prepare --rq 1 --agent codex \
  --id rq1_all --profile all --agent-seconds 21600
```

Profiles: tracing, topology, correspondence, search, state_tracking, measurement,
prior_conflict, reveal_global_structure (Global structure), and
reveal_declared_transform (Declared transform).

Read the [exact profiles](../../foundry/discovery_profiles/paper-semantic/v0.3/) and
[input guide](../../docs/INPUTS.md). Each episode repeats the same profile; accepted
worlds and the latest rejected submission and report change independently per campaign.

After inspection, follow the root README to authenticate, freeze, commit and run.
