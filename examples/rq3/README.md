# Model-feedback-steered generation

**Paper:** §4.5 (experiment and results).

This experiment starts from ten existing worlds that were hard for frontier models.
Later coding-agent episodes use the frontier evaluators' errors to propose new worlds.
Commands select it with `--rq 3`.

## Inputs and controller behavior

Choose feedback example set A or B (`--collection a` or `--collection b`). Each contains
ten starting world implementations and tests with fixed feedback summaries. The campaign
also receives the three implementation demonstrations and a card describing the starting
worlds and difficulty objective; it does not use the nine discovery profiles.

After a new world passes the final checks, the controller renders ten instances
using fixed random seeds and sends five selected instances to each of three evaluator
models. A world is hard for frontier models when at least two of the three evaluators
answer at most three of five instances correctly, with all 15 responses present.
Hardness does not replace final verification. Later episodes receive summaries of
completed evaluations.

The initial feedback JSON includes recorded answers, predictions, and correctness. There
are two detailed samples per starting world, while aggregate scores use five instances
per world. No initial image files are supplied. Starting-world answer visibility is
separate from the final checks of a new submission.

## Prepare a small run without calling a model

Complete the root [setup](../../README.md#run-your-first-example), then run:

```bash
bash examples/rq3/prepare.sh --agent opencode --collection a --id rq3_collection_a_demo
uv run --extra runner python -m scripts.release preview \
  --campaign campaigns/rq3_collection_a_demo.toml --out artifacts/rq3_collection_a_demo_preview
```

The first command writes `campaigns/rq3_collection_a_demo.toml` with a 20-minute
campaign budget. The second writes the first-episode inputs and prints their directory.
Inspect `instruction.md` and `environment/starter/` beneath that directory before
running. Neither command calls a model.

Substitute `--agent codex`, `--agent claude`, or `--agent opencode` to choose another
agent tool. Use `--help` on the preparation script to see its options.

## Check execution without a model

After building the Docker images, run the fixed-example check for this experiment:

```bash
uv run --extra runner python -m scripts.docker_smoke \
  --rq 3 --out artifacts/rq3_smoke
```

This checks the episode inputs, container replacement, and final verification. It
does not generate a new world or call a model. A successful run prints a summary with
all isolation checks `true` and all five verifier scores equal to `1.0`. Use a new
output directory to repeat the check.

## Authenticate and start the loop

Follow [authentication](../../docs/AUTHENTICATION.md) for the selected agent, including
its host prerequisites and provider confirmation variables. This experiment additionally
requires `OPENROUTER_API_KEY` for evaluator calls, even when using a Codex or Claude
coding agent. Set the feedback confirmation to this campaign ID:

```bash
export OPENROUTER_FEEDBACK_RUN_CONFIRMED=rq3_collection_a_demo
```

The authentication check below makes a small **paid** model call. Inspect its result
before launching the full campaign:

```bash
uv run --extra runner python -m scripts.auth_smoke \
  --campaign campaigns/rq3_collection_a_demo.toml \
  --out artifacts/rq3_collection_a_demo_auth --confirm rq3_collection_a_demo
uv run --extra runner python -m scripts.release freeze \
  --campaign campaigns/rq3_collection_a_demo.toml
git add campaigns/rq3_collection_a_demo.toml
git commit -m "Freeze model-feedback-steered example campaign"
bash examples/run_campaign.sh rq3_collection_a_demo
```

The final command starts the paid campaign. Source and configuration must be committed
with a clean working tree.

## Use the paper's campaign budget

For a six-hour campaign on feedback example set B:

```bash
bash examples/rq3/prepare.sh --agent opencode --collection b \
  --id rq3_collection_b_sixhour --agent-seconds 21600
```

Preview and launch that new ID separately, including updating the feedback confirmation
variable to match. The paper evaluates five model/reasoning configurations on each of
two feedback example sets; this release supplies three example presets and the
episodic-sequential 0.7 controller variant. It does not reproduce every historical
controller variant. See [release differences](../../docs/RELEASE_NOTES.md).

## Inspect and stop

Outputs are saved under `artifacts/rq3_collection_a_demo/`: `state.json`, per-episode
inputs and records under `episodes/`, submitted code, verification reports, and
evaluator feedback. To stop after the current episode's verification:

```bash
touch artifacts/rq3_collection_a_demo/STOP_AFTER_CURRENT_EPISODE
```

An interrupted campaign cannot be resumed in place. Use a new campaign ID. Local outputs
are ignored by Git; the source ZIP does not back them up. The paper's separate 200-scene
replay and historical result tables are not reproduced by this walkthrough. See
[operational details](../../docs/SETUP.md#operations-and-failure-handling).
