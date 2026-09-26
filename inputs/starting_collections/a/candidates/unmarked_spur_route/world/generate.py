"""Generate a deterministic PNG gallery and manifest for the spur world."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import renderer
import prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.n < 1:
        raise SystemExit("--n must be positive")
    out = args.out
    gallery = out / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    rows = []
    scenes = []
    for index in range(args.n):
        scene_seed = int(args.seed) + index * 104729
        scene = renderer.sample_scene(scene_seed)
        scene_id = f"scene_{index:04d}"
        image_path = gallery / f"{scene_id}.png"
        renderer.render(scene).save(image_path, format="PNG", optimize=False)
        scenes.append(scene)
        decision = renderer.analytic_gold(scene)
        for family, question in prompts.PROMPT_FAMILIES.items():
            rows.append(
                {
                    "example_id": f"{scene_id}__{family}",
                    "scene_id": scene_id,
                    "seed": int(args.seed),
                    "scene": scene,
                    "image_path": f"gallery/{scene_id}.png",
                    "prompt_family": family,
                    "question": question,
                    "decision": decision,
                    "answer": prompts.correct_answer(family, decision),
                    "candidates": list(prompts.candidates_for(family)),
                    "margin": renderer.margin(scene),
                    "quarantined": renderer.is_quarantined(scene),
                }
            )
    (out / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    (out / "self_check.json").write_text(
        json.dumps(
            {
                "schema_version": "0.4.0",
                "scene_count": len(scenes),
                "record_count": len(rows),
                "seed": int(args.seed),
                "prompt_families": list(prompts.PROMPT_FAMILIES),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
