# Validation

## Run the checks

```bash
uv sync --locked --extra runner --group dev
uv run --extra runner pytest -q
uv run ruff check scripts src tests worlds harbor
uv run ruff format --check scripts src tests worlds harbor
bash harbor/build_latex_tikz_runtime.sh
uv run --extra runner python -m scripts.doctor
uv run --extra runner python -m scripts.docker_smoke --rq 1 --out artifacts/rq1-smoke
uv run --extra runner python -m scripts.docker_smoke --rq 2 --out artifacts/rq2-smoke
uv run --extra runner python -m scripts.docker_smoke --rq 3 --out artifacts/rq3-smoke
uv run --extra runner python -m scripts.check_agent_installations
uv run --extra runner python -m scripts.check_starting_collections
```

The unit and integration tests need no network access or API keys. The Docker smoke
tests run each experiment (`--rq 1`, `--rq 2`, and `--rq 3`) with a fixed solution and
check that the final-check tests stay out of the agent's container. They test execution
and isolation, not the quality of newly generated worlds. Use new output paths when
repeating checks; diagnostics are saved under the ignored `artifacts/` folder.

Lint and format checks cover `scripts/`, `src/`, `tests/`, `worlds/`, and `harbor/`.
The frozen snapshots under `inputs/` keep their original style.

## Not yet tested

Live OpenCode runs, evaluator API calls for model-feedback-steered generation, full
six-hour campaigns, and Linux x86_64 or WSL hosts have not been tested with this
release. The feedback-processing tests use simulated evaluator responses.

## Known small-sample failure

With ten scenes and random seed 101, `stitch_face_alternation` produces three answer
classes, while its verifier requires at least four, so that check fails. The other 19
starting worlds pass, and all 20 pass with 32 scenes, the standalone checker's default.
Feedback example sets load fixed feedback, so campaign initialization does not repeat
this check. To reproduce the ten-scene failure:

```bash
uv run --extra runner python -m scripts.check_starting_collections \
  --only stitch_face_alternation --scenes 10 --out artifacts/stitch-ten-scene-check
```
