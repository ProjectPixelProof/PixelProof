# Spatially-steered generation

This directory contains the campaign template for spatially-steered generation
(`--rq 2`). See the [walkthrough](../../examples/rq2/README.md) for the paper context
and run commands.

This experiment uses the same nine discovery profiles as profile-steered generation and
adds spatial guidance. Before each episode, the controller requests a spatial pattern
and provides construction guidance. After submission, it measures the spatial pattern of
the generated world and updates its category record. The internal policy is named
`quality-diversity@0.8.0`. This experiment does not use the frontier evaluators of
model-feedback-steered generation.

```bash
uv run --extra runner python -m scripts.release prepare --rq 2 --agent claude \
  --id rq2_tracing_demo --profile tracing --agent-seconds 1800
uv run --extra runner python -m scripts.release preview --campaign campaigns/rq2_tracing_demo.toml \
  --out artifacts/rq2-tracing-preview
```

Inspect `DIVERSITY_TARGET.md`, `DIVERSITY_TARGET.json`, and
`tools/preview_qd_descriptor.py` in the preview's starter directory. Targets adapt
across episodes; the profile's example questions stay fixed. The nine spatial patterns
combine three scales (focal, regional, distributed) with three shapes (compact,
pathlike, multipart).

Use `--profile all --agent-seconds 21600` to prepare nine separate six-hour campaigns.
Select `--agent codex` or `--agent opencode` to change the coding-agent configuration.

After inspection, follow the root README to authenticate, freeze, commit and run.
