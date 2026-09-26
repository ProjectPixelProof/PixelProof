"""Dataset + manifest generator for bitplane_glyph_reveal.

Usage:
    python world/generate.py --out <dir> --n <count> --seed <seed>

Writes one PNG per scene and a manifest row per (scene, prompt-family).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import renderer
import prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()

    gallery = args.out / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(args.n):
        glyph = "A" if i % 2 == 0 else "B"
        noise = args.seed * 10000 + i
        scene = renderer.build_scene(noise=noise, glyph=glyph)
        image = renderer.render(scene)
        rel = f"gallery/scene_{i:04d}.png"
        image.save(args.out / rel)
        decision = renderer.analytic_gold(scene)
        margin = renderer.margin(scene)
        quarantined = renderer.is_quarantined(scene)
        for family, question in prompts.PROMPT_FAMILIES.items():
            rows.append(
                {
                    "example_id": f"scene_{i:04d}__{family}",
                    "scene_id": f"scene_{i:04d}",
                    "seed": args.seed,
                    "scene": scene,
                    "image_path": rel,
                    "prompt_family": family,
                    "question": question,
                    "decision": decision,
                    "answer": prompts.correct_answer(family, decision),
                    "candidates": list(prompts.candidates_for(family)),
                    "margin": margin,
                    "quarantined": quarantined,
                }
            )
    with (args.out / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} rows, {args.n} scenes to {args.out}")


if __name__ == "__main__":
    main()
