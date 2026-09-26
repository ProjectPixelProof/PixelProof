"""Dataset generator for the tooth-route world."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oracle import measure_from_image
from prompts import PROMPT_FAMILIES, candidates_for, correct_answer
from renderer import analytic_gold, is_quarantined, margin, render, sample_scene


def generate(out: Path, n: int, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    gallery = out / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    rows = []
    cards = []
    for index in range(n):
        scene_seed = int(seed) + index
        scene = sample_scene(scene_seed)
        image = render(scene)
        decision = analytic_gold(scene)
        measured = measure_from_image(image)
        if measured["decision"] != decision:
            raise RuntimeError(f"pixel inverse disagrees for seed {scene_seed}")
        scene_id = f"scene_{index:04d}"
        image_rel = f"gallery/{scene_id}.png"
        image.save(out / image_rel, format="PNG", optimize=False)
        cards.append(f'<figure><img src="{scene_id}.png"><figcaption>{scene_id}: {decision}</figcaption></figure>')
        for family, question in PROMPT_FAMILIES.items():
            rows.append({
                "example_id": f"{scene_id}__{family}",
                "scene_id": scene_id,
                "seed": scene_seed,
                "scene": scene,
                "image_path": image_rel,
                "prompt_family": family,
                "question": question,
                "decision": decision,
                "answer": correct_answer(family, decision),
                "candidates": list(candidates_for(family)),
                "margin": margin(scene),
                "quarantined": is_quarantined(scene),
                "pixel_runner_up_margin": measured["runner_up_margin"],
                "pixel_minimum_tooth_clearance": round(measured["minimum_tooth_clearance"], 3),
            })
    (out / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    question = PROMPT_FAMILIES["pf1"]
    html = """<!doctype html><meta charset="utf-8"><title>Toothline gallery</title>
<style>body{font-family:sans-serif;background:#eee;padding:20px}main{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}figure{margin:0;background:white;padding:8px}img{width:100%;height:auto}figcaption{font-size:13px}</style>
<h1>Nine-Tooth Route Rasterization</h1><p>""" + question + "</p><main>" + "".join(cards) + "</main>\n"
    (gallery / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.n < 1:
        raise SystemExit("n must be positive")
    generate(args.out, args.n, args.seed)


if __name__ == "__main__":
    main()
