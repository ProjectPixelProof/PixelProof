"""Dataset + manifest generator for the radial-density-regime world.

Usage:
    python world/generate.py --out <directory> --n <scene-count> --seed <seed>

Writes one PNG per scene under <out>/gallery/ and a manifest.jsonl with one
row per (scene, prompt-family) pair sharing that scene's image. Every record
is pixel-oracle-certified before it is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

import renderer
import oracle
import prompts

_GALLERY = "gallery"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    out = args.out
    gallery = out / _GALLERY
    gallery.mkdir(parents=True, exist_ok=True)

    families = sorted(prompts.PROMPT_FAMILIES)
    rows = []
    for i in range(args.n):
        scene_seed = args.seed * 1000003 + i * 7919 + 7
        scene = renderer.sample_scene(scene_seed)
        decision = renderer.analytic_gold(scene)
        image = renderer.render(scene).convert("RGB")
        image_path = f"{_GALLERY}/scene_{i:04d}.png"
        image.save(gallery / f"scene_{i:04d}.png")
        for family in families:
            rows.append(
                {
                    "example_id": f"scene_{i:04d}__{family}",
                    "scene_id": f"scene_{i:04d}",
                    "seed": scene_seed,
                    "scene": scene,
                    "image_path": image_path,
                    "prompt_family": family,
                    "question": prompts.PROMPT_FAMILIES[family],
                    "decision": decision,
                    "answer": prompts.correct_answer(family, decision),
                    "candidates": list(prompts.candidates_for(family)),
                    "margin": renderer.margin(scene),
                    "quarantined": renderer.is_quarantined(scene),
                }
            )

    bad = 0
    for row in rows:
        img = oracle.decision_from_image(Image.open(out / row["image_path"]))
        if img != row["decision"]:
            bad += 1
    if bad:
        sys.exit(f"FAIL: {bad} records disagree with the pixel oracle")

    with (out / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"OK: wrote {len(rows)} records across {len(rows)//len(families)} scenes")


if __name__ == "__main__":
    main()
