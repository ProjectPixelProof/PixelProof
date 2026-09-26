"""Generate deterministic PNG evidence and its manifest."""

from __future__ import annotations

import argparse
import html
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
    gallery = args.out / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    rows = []
    cards = []
    for index in range(args.n):
        scene_seed = args.seed + index
        scene_id = f"scene_{index:04d}"
        scene = sample_scene(scene_seed)
        image = render(scene)
        decision = analytic_gold(scene)
        recovered = decision_from_image(image)
        if recovered != decision:
            raise RuntimeError(f"pixel oracle disagreement for {scene_id}: {recovered} != {decision}")
        relative_image = f"gallery/{scene_id}.png"
        image.save(args.out / relative_image, format="PNG", optimize=False)
        for family, question in PROMPT_FAMILIES.items():
            rows.append(
                {
                    "example_id": f"{scene_id}__{family}",
                    "scene_id": scene_id,
                    "seed": scene_seed,
                    "scene": scene,
                    "image_path": relative_image,
                    "prompt_family": family,
                    "question": question,
                    "decision": decision,
                    "answer": correct_answer(family, decision),
                    "candidates": list(candidates_for(family)),
                    "margin": margin(scene),
                    "quarantined": is_quarantined(scene),
                }
            )
        cards.append(
            f'<figure><img src="{scene_id}.png" alt="spring scene {index}">'
            f'<figcaption>{html.escape(decision)} full coils</figcaption></figure>'
        )
    (args.out / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    question = next(iter(PROMPT_FAMILIES.values()))
    (gallery / "index.html").write_text(
        "<!doctype html><meta charset=\"utf-8\"><title>Spring coil gallery</title>"
        "<style>body{font-family:sans-serif}figure{display:inline-block;width:340px;margin:12px}"
        "img{width:330px;border:1px solid #bbb}figcaption{text-align:center}</style>"
        f"<h1>{html.escape(question)}</h1>" + "".join(cards),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "self-check-0.1.0",
        "scene_count": args.n,
        "record_count": len(rows),
        "seed": args.seed,
        "pixel_oracle_agreement": True,
        "classes": sorted({row["decision"] for row in rows}),
    }
    (args.out / "self_check.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

