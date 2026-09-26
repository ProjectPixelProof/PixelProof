"""Generate deterministic PNG datasets for the candidate contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oracle import decision_from_image
from prompts import PROMPT_FAMILIES, candidates_for, correct_answer
from renderer import analytic_gold, is_quarantined, margin, render, sample_scene


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.n < 1:
        raise SystemExit("scene count must be positive")
    gallery = args.out / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    records = []
    captions = []
    for index in range(args.n):
        scene_seed = args.seed + index
        scene = sample_scene(scene_seed)
        image = render(scene)
        decision = analytic_gold(scene)
        if decision_from_image(image) != decision:
            raise RuntimeError(f"pixel inverse disagrees for seed {scene_seed}")
        scene_id = f"scene_{index:04d}"
        image_name = f"{scene_id}.png"
        image.save(gallery / image_name)
        captions.append((image_name, decision, margin(scene), is_quarantined(scene)))
        for family, question in PROMPT_FAMILIES.items():
            records.append(
                {
                    "example_id": f"{scene_id}__{family}",
                    "scene_id": scene_id,
                    "seed": scene_seed,
                    "scene": scene,
                    "image_path": f"gallery/{image_name}",
                    "prompt_family": family,
                    "question": question,
                    "decision": decision,
                    "answer": correct_answer(family, decision),
                    "candidates": list(candidates_for(family)),
                    "margin": margin(scene),
                    "quarantined": is_quarantined(scene),
                }
            )
    (args.out / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8"
    )
    (args.out / "self_check.json").write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "scene_count": args.n,
                "record_count": len(records),
                "seed": args.seed,
                "pixel_oracle_agreement": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    cards = "\n".join(
        f'<figure><img src="{name}" width="480"><figcaption>{label}; margin={gap}; quarantined={str(q).lower()}</figcaption></figure>'
        for name, label, gap, q in captions
    )
    question = next(iter(PROMPT_FAMILIES.values()))
    (gallery / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Run-bar gallery</title>"
        f"<h1>Run-bar decompression gallery</h1><p>{question}</p>{cards}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
