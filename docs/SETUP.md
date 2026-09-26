# Setup and runtime

Run commands from the repository root. Python >=3.12 is required; `uv sync` creates an
isolated environment using the versions in `uv.lock`. Install the `runner` extra to run
campaigns. Use `uv run --extra runner ...` for Harbor commands so a later uv operation
does not drop that optional dependency.

Docker must be running and `docker compose version` must succeed. A practical local
allocation is at least 8 GB Docker memory and 20 GB free disk for one campaign at a
time; each configured episode requests 2 CPUs, 4 GB RAM and 20 GB storage. Build images:

```bash
bash harbor/build_latex_tikz_runtime.sh
uv run --extra runner python -m scripts.doctor
```

This builds `question-foundry-latex-tikz-runtime:0.1` and
`question-foundry-candidate-replay-latex-tikz:0.1`. Python and Pillow remain available;
TikZ is optional for the agent, but the released experiment presets enable that runtime.
Model-feedback-steered generation uses the replay image to generate and verify instances
of submitted worlds offline before sending selected images to the evaluator API. No GPU
is needed locally.

Docker builds and coding-agent setup download dependencies. Agent containers have public
network access to reach their provider. Before final verification, the controller
replaces that container with one that has no network access. The final-check tests are
uploaded after the agent phase. Credentials are copied temporarily with mode 0600,
removed after use, and checked absent in the replacement container. Submitted code is
executed inside containers; Docker access itself should be limited to trusted operators.

## Local checks

```bash
uv run --extra runner pytest -q
uv run python -m scripts.release profiles
uv run --extra runner python -m scripts.docker_smoke --out artifacts/docker-check-001
uv run python -m scripts.audit_release
```

`docker_smoke` checks the execution system using a fixed solution. That solution is
included only in the temporary smoke-test files; normal generation episodes do not
receive it.

Select an experiment with `--rq 1` (profile-steered), `--rq 2` (spatially-steered), or
`--rq 3` (model-feedback-steered). The last uses feedback example set A by default; add
`--collection b` to check feedback example set B. Use a different `--out` directory for
each check. These commands test input preparation and final verification; they do not
call a coding model or a frontier evaluator.

## Check pinned coding-agent installation without an account

```bash
uv run --extra runner python -m scripts.check_agent_installations
```

This builds three disposable CLI images using the same pinned install templates as live
runs. It downloads vendor packages but supplies no credentials and makes no inference
request. Use the paid check in the authentication guide to test access to the configured
model.

## Check the feedback example sets

```bash
uv run --extra runner python -m scripts.check_starting_collections
```

This regenerates 32 evidence scenes in a temporary copy of each starting world, then
runs its original tests and an independent 32-scene rendering and inverse-verification
check in the feedback container. Some tests expect `evidence/` to exist and some write a
self-check file there; run this wrapper instead of invoking their tests directly against
the supplied input directories. No provider calls are made. These checks do not replace
final verification during a campaign.

## Generate reference questions without a model

```bash
uv run python -m worlds.two_circles.generate --n 12 --seed 101 \
  --out-dir artifacts/two-circles/images --manifest artifacts/two-circles/manifest.jsonl
uv run python -m worlds.angle_acuteness.generate --n 12 --seed 101 \
  --out-dir artifacts/angles/images --manifest artifacts/angles/manifest.jsonl
uv run python -m worlds.counting_with_distractors.generate --n 12 --seed 101 \
  --out-dir artifacts/counting/images --manifest artifacts/counting/manifest.jsonl
```

Generation invokes the independent inverse verification. The release includes the module
entry points for these demonstrations; campaign agents still receive the historical
five-file source snapshots specified in the seed-set manifest.

## Operations and failure handling

Use a new ID and output path for each run. The runner refuses existing output
directories. It saves campaign state, episode inputs, submitted code, verification
results, and feedback under `artifacts/<campaign-id>/`. A world enters the accepted set
after passing final checks and the controller's retention criteria. This does not
imply human review or approval.

To request a safe stop after the current verification completes:

```bash
touch artifacts/YOUR_CAMPAIGN_ID/STOP_AFTER_CURRENT_EPISODE
```

The runner cannot resume an interrupted campaign. Keep its saved outputs and use a new
campaign ID. Agent-session time and total elapsed time have separate limits; Docker
builds, verification, and evaluator calls also consume elapsed time. The runner cannot
reliably measure the dollar cost of subscription-based agent usage. OpenRouter spending
limits for the coding agent and evaluators are separate. Requests already in progress
can take spending above a configured threshold. Set provider-side spending limits.

The example configurations cap each agent episode at 20 minutes and start another only
if at least five minutes of agent time remain. A campaign can therefore finish with some
unused budget. A submission that passes verification but lacks the required completion
records is preserved as `budget-incomplete` and is not accepted. A CLI timeout can still
leave an accepted world if those records were completed before the timeout and all
retention checks pass. Inspect `state.json` and `episode-ledger.jsonl` to see what was
retained.

If `scripts.doctor` reports missing Docker images, run the build command above. If you
need a different CLI version, prepare with `--cli-version X.Y.Z`, test authentication at
that version, and freeze the new configuration. The validator rejects unsupported model
configurations. Adding one requires updating the preset, validator, and relevant tests.
