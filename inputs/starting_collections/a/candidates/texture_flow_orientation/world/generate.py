"""Generation entrypoint for the texture_flow_orientation world.

Writes a manifest.jsonl plus gallery/*.png dataset. Every scene is rendered
once and emitted once per prompt family, sharing the same image path. Dominant
orientations are stratified across the four classes so each prompt family keeps
class balance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from renderer import analytic_gold, is_quarantined, margin, render, sample_scene
from prompts import PROMPT_FAMILIES, correct_answer, candidates_for


def _write_dataset(out: Path, n: int, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    gallery = out / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    families = sorted(PROMPT_FAMILIES)
    rows = []
    for index in range(n):
        scene_seed = seed * 100000 + index
        scene = sample_scene(scene_seed)
        decision = analytic_gold(scene)
        scene_id = f"scene_{index:04d}"
        image_path = f"gallery/{scene_id}.png"
        render(scene).save(gallery / f"{scene_id}.png")
        for family in families:
            example_id = f"{scene_id}__{family}"
            rows.append(
                {
                    "example_id": example_id,
                    "scene_id": scene_id,
                    "seed": seed,
                    "scene": scene,
                    "image_path": image_path,
                    "prompt_family": family,
                    "question": PROMPT_FAMILIES[family],
                    "decision": decision,
                    "answer": correct_answer(family, decision),
                    "candidates": list(candidates_for(family)),
                    "margin": margin(scene),
                    "quarantined": bool(is_quarantined(scene)),
                }
            )
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    _write_dataset(args.out, args.n, args.seed)
    print(f"wrote {args.n} scenes to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
