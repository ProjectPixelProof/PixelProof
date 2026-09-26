# Render the three implementation demonstrations

**Paper §3.2:** the implementation demonstrations teach agents how a question world is
implemented. They cover two-circle contact, angle acuteness, and counting with
distractors. They are distinct from the five text-only example questions in each
discovery profile.

After installing the Python environment from the root README:

```bash
bash examples/shared_examples/render.sh
```

This renders 12 scenes per world with random seed 101, computes the recorded answers and
runs the built-in inverse verification. It needs no coding agent, provider credentials,
Docker, or GPU. The number of question records can exceed the number of images because
worlds support multiple fixed question wordings.

Open the images and `manifest.jsonl` under `artifacts/shared-examples/<world>/`. Each
manifest records scene specifications, exact question text, and recorded answers. To
repeat, use a new output directory (relative paths are relative to the repository):

```bash
bash examples/shared_examples/render.sh artifacts/shared-examples-002
```

The release includes generators and supporting modules so these worlds can run locally.
During generation, agents receive only the historical five-file snapshots: `DESIGN.md`,
`world.toml`, `renderer.py`, `prompts.py`, and `oracle.py`. See [input
composition](../../docs/INPUTS.md) for the distinction.

This example renders the three reference worlds. To generate new world implementations,
use [profile-steered generation](../rq1/README.md).
