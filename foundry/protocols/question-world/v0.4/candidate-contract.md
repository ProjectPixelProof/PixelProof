# Executable candidate contract under protocol v0.4

Submit this tree under `/logs/artifacts/submission/`:

```text
submission/
├── portfolio.json
└── candidates/
    └── <candidate_id>/
        ├── candidate.json
        ├── provenance.json
        ├── deviations.md
        ├── world/
        │   ├── world.toml
        │   ├── renderer.py
        │   ├── oracle.py
        │   ├── prompts.py
        │   ├── generate.py
        │   └── verify.py
        ├── tests/test_candidate.py
        └── evidence/
            ├── manifest.jsonl
            ├── self_check.json
            └── gallery/
                ├── index.html
                └── *.png
```

`portfolio.json` uses schema version `0.4.0` and the exact keys
`schema_version`, `candidate_ids`, and `deviations`.

`candidate.json` uses schema version `0.4.0`. In addition to the executable-world
metadata from v0.2, it contains:

- `latent_alias.mode`: `tested_transforms` or `not_applicable`;
- `latent_alias.justification`: a concrete claim about aliases or injectivity;
- `latent_alias.label_inputs`: scene fields permitted to influence analytic gold;
- `mechanism_fingerprint`: values from the controlled ontology;
- `semantic_contrast.closest_known_worlds`: one to five visible memory IDs;
- `semantic_contrast.known_negative_matches`: any relevant negative-memory IDs;
- `semantic_contrast.contrast`: why the mechanism is not a renamed neighbor;
- `semantic_contrast.expected_information_gain`: what distinct VLM capability it isolates.

## Required Python interface

`renderer.py`:

- `sample_scene(seed: int) -> dict`
- `render(scene: dict) -> PIL.Image.Image`
- `analytic_gold(scene: dict) -> str`
- `margin(scene: dict) -> float`
- `is_quarantined(scene: dict) -> bool`
- `latent_symmetries(scene: dict) -> list[tuple[str, dict]]`

`oracle.py`:

- `decision_from_image(image: PIL.Image.Image) -> str`
- must not import renderer, prompts, generate, verify, or analytic-gold code
- must not accept latent scene data

`prompts.py`:

- `PROMPT_FAMILIES: dict[str, str]`
- `correct_answer(family: str, decision: str) -> str`
- `candidates_for(family: str) -> tuple[str, ...]`

`generate.py` must support:

```text
python world/generate.py --out <directory> --n <scene-count> --seed <seed>
```

It writes real PNGs plus `manifest.jsonl`. Every manifest row contains at least
the following exact keys; additional diagnostic keys are allowed:

```json
{
  "example_id": "scene_0000__pf1",
  "scene_id": "scene_0000",
  "seed": 101,
  "scene": {},
  "image_path": "gallery/scene_0000.png",
  "prompt_family": "pf1",
  "question": "Question text",
  "decision": "yes",
  "answer": "yes",
  "candidates": ["yes", "no"],
  "margin": 12.5,
  "quarantined": false
}
```

Emit every scene once per prompt family. Rows for the same scene share one image
and use the declared prompt APIs. Do not substitute aliases such as `id`, `gold`,
`image`, or `prompt_answers` for the required keys.

`verify.py` must support:

```text
python world/verify.py --dataset <directory>
```

It exits zero only when stored images, analytic decisions, prompt answers,
answer candidate sets, margins, quarantine flags, and the scene-free pixel
oracle agree.

For `tested_transforms`, `latent_symmetries` must exercise at least one
nontrivial transform. For `not_applicable`, it must return an empty list and the
hidden evaluator reports the symmetry gate as `not_applicable`, not `pass`.

The checked-in evidence manifest is a protected verifier input. Candidate tests
must verify that exact dataset, not only a convenient seed range. Before final
submission, run generation and verification at every public stress seed from
`TASK_SPEC.toml`, then run `verify.py` on the exact submitted `evidence/`.
