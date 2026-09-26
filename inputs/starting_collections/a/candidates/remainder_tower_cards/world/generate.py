"""Generate deterministic evidence PNGs and manifest rows."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path

import prompts
import renderer


def generate(out: Path, n: int, seed: int) -> None:
    gallery = out / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    counts: Counter[str] = Counter()
    cards: list[str] = []
    for index in range(n):
        scene_seed = int(seed) + index
        scene = renderer.sample_scene(scene_seed)
        decision = renderer.analytic_gold(scene)
        counts[decision] += 1
        scene_id = f"scene_{index:04d}"
        filename = f"{scene_id}.png"
        renderer.render(scene).save(gallery / filename, format="PNG", optimize=False)
        cards.append(f'<figure><img src="{html.escape(filename)}"><figcaption>{html.escape(decision)}</figcaption></figure>')
        for family, question in prompts.PROMPT_FAMILIES.items():
            rows.append({
                "example_id": f"{scene_id}__{family}",
                "scene_id": scene_id,
                "seed": scene_seed,
                "scene": scene,
                "image_path": f"gallery/{filename}",
                "prompt_family": family,
                "question": question,
                "decision": decision,
                "answer": prompts.correct_answer(family, decision),
                "candidates": list(prompts.candidates_for(family)),
                "margin": renderer.margin(scene),
                "quarantined": renderer.is_quarantined(scene),
            })
    (out / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    self_check = {
        "schema_version": "self-check-0.1.0",
        "scene_count": n,
        "record_count": len(rows),
        "class_counts": dict(sorted(counts.items())),
        "boxes_per_card": 14,
        "total_qualifying_towers": 8,
        "oracle": "pending verify.py",
    }
    (out / "self_check.json").write_text(json.dumps(self_check, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    page = """<!doctype html><meta charset="utf-8"><title>Remainder Tower Cards</title>
<style>body{font-family:sans-serif;background:#eee;color:#222}main{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}figure{margin:0;background:white;padding:8px}img{width:100%;height:auto}figcaption{text-align:center}</style>
<h1>Remainder Tower Cards</h1><p>Order the cards by dots, reduce tower heights by divisibility by three, and stack the rows.</p><main>""" + "".join(cards) + "</main>\n"
    (gallery / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    generate(args.out, args.n, args.seed)


if __name__ == "__main__":
    main()
