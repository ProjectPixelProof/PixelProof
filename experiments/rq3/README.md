# Model-feedback-steered generation

This directory contains the campaign template for model-feedback-steered generation
(`--rq 3`). See the [walkthrough](../../examples/rq3/README.md) for the paper context
and run commands.

This experiment starts with ten world implementations from feedback example set A or B,
their fixed feedback summaries, the three implementation demonstrations, and a card
describing the starting worlds and difficulty objective. It does not use the nine
discovery profiles. The feedback policy is `difficulty-feedback@0.9.0`; this release
uses the `episodic-sequential@0.7.0` controller variant. Some historical campaigns used
a later controller version; see the release notes.

```bash
uv run --extra runner python -m scripts.release prepare --rq 3 --agent opencode \
  --id rq3_collection_a_demo --collection a
uv run --extra runner python -m scripts.release preview --campaign campaigns/rq3_collection_a_demo.toml \
  --out artifacts/rq3-a-preview
```

Repeat with `--collection b` and a new ID for feedback example set B. Use
`--agent codex` or `--agent claude` for either of the other two coding-agent adapters.
Use `--agent-seconds 21600` for six hours of agent-session time; feedback and
verification have a separate elapsed-time allowance. This experiment requires an
OpenRouter evaluator key regardless of the coding-agent adapter; see
[authentication](../../docs/AUTHENTICATION.md).

The three frontier evaluators are Sol, Opus 5, and Gemini 3.7 Flash, each with high
reasoning effort. Each evaluates five instances. A world is hard for frontier models
when at least two of the three evaluators answer at most three of five instances
correctly, with all 15 responses available. Initial feedback contains two detailed
samples per starting world; aggregate scores represent five. Submitted worlds that pass
final checks are rendered and verified offline before evaluation. Later agents
receive summaries of predictions, correctness, public rationales, available measures of
reasoning effort, and guidance for further generation.

**Visibility:** initial JSON includes `oracle_answer`, predictions and correctness.
These are starting-world labels, not reference answers for a new submission. No initial
image files are supplied. New code may render images, and later rejected submission
directories may contain galleries. See [release
differences](../../docs/RELEASE_NOTES.md) for omitted historical render evidence and
stale wording retained in versioned instructions.

After inspection, follow the root README to authenticate, freeze, commit and run.
