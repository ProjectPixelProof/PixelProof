from __future__ import annotations

import argparse
import json
from pathlib import Path

import prompts
import renderer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    images = args.out / "images"
    gallery = args.out / "gallery"
    images.mkdir(parents=True, exist_ok=True)
    gallery.mkdir(parents=True, exist_ok=True)
    rows = []
    gallery_rows = []
    for index in range(args.n):
        seed = args.seed + index
        scene = renderer.sample_scene(seed)
        decision = renderer.analytic_gold(scene)
        image_name = f"scene_{index:04d}.png"
        renderer.render(scene).save(images / image_name)
        renderer.render(scene).save(gallery / image_name)
        for family, question in prompts.PROMPT_FAMILIES.items():
            answer = prompts.correct_answer(family, decision)
            rows.append(
                {
                    "example_id": f"{index:04d}-{family}",
                    "scene_id": f"scene-{index:04d}",
                    "seed": seed,
                    "scene": scene,
                    "image_path": f"images/{image_name}",
                    "prompt_family": family,
                    "question": question,
                    "decision": decision,
                    "answer": answer,
                    "candidates": list(prompts.candidates_for(family)),
                    "margin": renderer.margin(scene),
                    "quarantined": renderer.is_quarantined(scene),
                }
            )
        gallery_rows.append(f"<li>{image_name}: decision={decision}</li>")

    manifest = args.out / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (args.out / "self_check.json").write_text(
        json.dumps({"schema_version": "0.2.0", "generated_scenes": args.n}, indent=2) + "\n",
        encoding="utf-8",
    )
    families = ", ".join(sorted(prompts.PROMPT_FAMILIES))
    (gallery / "index.html").write_text(
        "<!doctype html><html><body>"
        f"<h1>Reference marker gallery</h1><p>Prompt families: {families}</p><ul>"
        + "".join(gallery_rows)
        + "</ul></body></html>\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
