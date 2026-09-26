# Spatially-steered generation

**Paper:** §4.2 (experiment and results).

This experiment tests whether the agent can control which regions of an image its
inverse program depends on. It extends profile-steered generation by requesting one of
nine spatial patterns for where the answer's evidence lies. Commands select it with
`--rq 2`.

## Inputs and controller behavior

Each episode receives a fixed discovery profile and the three implementation
demonstrations, plus a requested spatial pattern: a scale (`focal`, `regional`,
`distributed`) and a shape (`compact`, `pathlike`, `multipart`). The controller updates
its category record and selects later requests from prior measurements.

The spatial pattern is measured by masking image regions and checking whether the
inverse answer changes or the inverse program fails. This is inverse-program
sensitivity, not human attention. A difference between the requested and measured
categories does not by itself reject a world. Final verification and the retention
criteria still apply. The internal implementation uses `quality_diversity` names for
this category record.

## Prepare a small run without calling a model

Complete the root [setup](../../README.md#run-your-first-example), then run:

```bash
bash examples/rq2/prepare.sh --agent claude --profile tracing --id rq2_tracing_demo \
  --agent-seconds 1800
uv run --extra runner python -m scripts.release preview \
  --campaign campaigns/rq2_tracing_demo.toml --out artifacts/rq2_tracing_demo_preview
```

The first command writes `campaigns/rq2_tracing_demo.toml` with a 30-minute campaign
budget. The second writes the first-episode inputs and prints their directory. Inspect
`instruction.md` and `environment/starter/` beneath that directory before running.
Neither command calls a model.

Substitute `--agent codex`, `--agent claude`, or `--agent opencode` to choose another
agent tool. Use `--help` on the preparation script to see its options.

## Check execution without a model

After building the Docker images, run the fixed-example check for this experiment:

```bash
uv run --extra runner python -m scripts.docker_smoke \
  --rq 2 --out artifacts/rq2_smoke
```

This checks the episode inputs, container replacement, and final verification. It
does not generate a new world or call a model. A successful run prints a summary with
all isolation checks `true` and all five verifier scores equal to `1.0`. Use a new
output directory to repeat the check.

## Authenticate and start the loop

Follow [authentication](../../docs/AUTHENTICATION.md) for the selected agent, including
its host prerequisites and provider confirmation variables. The authentication check
below makes a small **paid** model call. Inspect its result before launching the full
campaign:

```bash
uv run --extra runner python -m scripts.auth_smoke \
  --campaign campaigns/rq2_tracing_demo.toml \
  --out artifacts/rq2_tracing_demo_auth --confirm rq2_tracing_demo
uv run --extra runner python -m scripts.release freeze \
  --campaign campaigns/rq2_tracing_demo.toml
git add campaigns/rq2_tracing_demo.toml
git commit -m "Freeze spatially-steered example campaign"
bash examples/run_campaign.sh rq2_tracing_demo
```

The final command starts the paid campaign. Source and configuration must be committed
with a clean working tree.

## Use the paper's campaign budget

To prepare all nine profile campaigns with six hours each for one agent tool:

```bash
bash examples/rq2/prepare.sh --agent claude --profile all \
  --id rq2_sixhour --agent-seconds 21600
```

This creates nine configurations named `rq2_sixhour_<profile>.toml`. Review the
configuration and preview for each profile. Then freeze and commit all nine files;
leaving any untracked configuration in the repository prevents a campaign from starting
because the runner requires a clean working tree:

```bash
for campaign in campaigns/rq2_sixhour_*.toml; do
  uv run --extra runner python -m scripts.release freeze --campaign "$campaign"
done
git add campaigns/rq2_sixhour_*.toml
git commit -m "Freeze nine spatially-steered campaigns"
```

Authenticate and launch each selected ID separately, following the steps above.
`--profile all` does not launch campaigns. The paper evaluates each of five
model/reasoning configurations on nine profiles; these scripts supply one preset per
agent tool. Reproducing all 45 combinations requires the corresponding model and
reasoning configurations.

## Inspect and stop

Outputs are saved under `artifacts/rq2_tracing_demo/`: `state.json`, per-episode inputs
and records under `episodes/`, submitted code, and verification reports. To stop after
the current episode's verification:

```bash
touch artifacts/rq2_tracing_demo/STOP_AFTER_CURRENT_EPISODE
```

An interrupted campaign cannot be resumed in place. Use a new campaign ID. Local outputs
are ignored by Git; the source ZIP does not back them up. The paper's separate 200-scene
replay and historical result tables are not reproduced by this walkthrough. See
[operational details](../../docs/SETUP.md#operations-and-failure-handling).
