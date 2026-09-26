# Profile-steered generation

**Paper:** §4.1 (experiment and results).

This experiment tests whether a coding agent can repeatedly invent visual question
worlds that pass final verification. One campaign uses one fixed discovery profile.
Commands select it with `--rq 1`.

## Inputs and controller behavior

Each episode receives the same three implementation demonstrations and the same
discovery profile, including all five text-only example questions. Accepted world code
and the most recent rejected submission and its report change as the campaign
progresses. The example questions do not come with profile-specific images or
implementations.

The nine choices are `tracing`, `topology`, `correspondence`, `search`,
`state_tracking`, `measurement`, `prior_conflict`, `reveal_global_structure`, and
`reveal_declared_transform`. See the [exact discovery
profiles](../../foundry/discovery_profiles/paper-semantic/v0.3/).

## Prepare a small run without calling a model

Complete the root [setup](../../README.md#run-your-first-example), then run:

```bash
bash examples/rq1/prepare.sh --agent codex --profile tracing --id rq1_tracing_demo
uv run --extra runner python -m scripts.release preview \
  --campaign campaigns/rq1_tracing_demo.toml --out artifacts/rq1_tracing_demo_preview
```

The first command writes `campaigns/rq1_tracing_demo.toml` with a 20-minute campaign
budget. The second writes the first-episode inputs and prints their directory. Inspect
`instruction.md` and `environment/starter/` beneath that directory before running.
Neither command calls a model.

Substitute `--agent codex`, `--agent claude`, or `--agent opencode` to choose another
agent tool. Use `--help` on the preparation script to see its options.

## Check execution without a model

After building the Docker images, run the fixed-example check for this experiment:

```bash
uv run --extra runner python -m scripts.docker_smoke \
  --rq 1 --out artifacts/rq1_smoke
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
  --campaign campaigns/rq1_tracing_demo.toml \
  --out artifacts/rq1_tracing_demo_auth --confirm rq1_tracing_demo
uv run --extra runner python -m scripts.release freeze \
  --campaign campaigns/rq1_tracing_demo.toml
git add campaigns/rq1_tracing_demo.toml
git commit -m "Freeze profile-steered example campaign"
bash examples/run_campaign.sh rq1_tracing_demo
```

The final command starts the paid campaign. Source and configuration must be committed
with a clean working tree.

## Use the paper's campaign budget

To prepare all nine profile campaigns with six hours each for one agent tool:

```bash
bash examples/rq1/prepare.sh --agent codex --profile all \
  --id rq1_sixhour --agent-seconds 21600
```

This creates nine configurations named `rq1_sixhour_<profile>.toml`. Review the
configuration and preview for each profile. Then freeze and commit all nine files;
leaving any untracked configuration in the repository prevents a campaign from starting
because the runner requires a clean working tree:

```bash
for campaign in campaigns/rq1_sixhour_*.toml; do
  uv run --extra runner python -m scripts.release freeze --campaign "$campaign"
done
git add campaigns/rq1_sixhour_*.toml
git commit -m "Freeze nine profile-steered campaigns"
```

Authenticate and launch each selected ID separately, following the steps above.
`--profile all` does not launch campaigns. The paper evaluates each of five
model/reasoning configurations on nine profiles; these scripts supply one preset per
agent tool. Reproducing all 45 combinations requires the corresponding model and
reasoning configurations.

## Inspect and stop

Outputs are saved under `artifacts/rq1_tracing_demo/`: `state.json`, per-episode inputs
and records under `episodes/`, submitted code, and verification reports. To stop after
the current episode's verification:

```bash
touch artifacts/rq1_tracing_demo/STOP_AFTER_CURRENT_EPISODE
```

An interrupted campaign cannot be resumed in place. Use a new campaign ID. Local outputs
are ignored by Git; the source ZIP does not back them up. The paper's separate 200-scene
replay and historical result tables are not reproduced by this walkthrough. See
[operational details](../../docs/SETUP.md#operations-and-failure-handling).
