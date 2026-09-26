"""Generate deterministic transpose-stamp datasets."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path

from oracle import decision_from_image
from prompts import PROMPT_FAMILIES, candidates_for, correct_answer
from renderer import analytic_gold, is_quarantined, margin, render, sample_scene


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate(out: Path, n: int, seed: int) -> None:
    if n < 1:
        raise ValueError("n must be positive")
    out.mkdir(parents=True, exist_ok=True)
    gallery = out / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    for stale in gallery.glob("scene_*.png"):
        stale.unlink()

    rows: list[dict] = []
    class_counts: Counter[str] = Counter()
    cards: list[str] = []
    for index in range(n):
        scene_seed = int(seed) + index
        scene_id = f"scene_{index:04d}"
        scene = sample_scene(scene_seed)
        image = render(scene)
        decision = analytic_gold(scene)
        pixel_decision = decision_from_image(image)
        if pixel_decision != decision:
            raise RuntimeError(f"pixel oracle disagrees for {scene_id}")
        image_name = f"{scene_id}.png"
        image.save(gallery / image_name, format="PNG", optimize=False)
        cards.append(
            f'<figure><img src="{html.escape(image_name)}" width="256" height="256" '
            f'alt="{html.escape(scene_id)}"><figcaption>{html.escape(scene_id)} — '
            f'{html.escape(decision)}, margin {margin(scene):.0f}</figcaption></figure>'
        )
        class_counts[decision] += 1
        for family, question in PROMPT_FAMILIES.items():
            rows.append(
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

    (out / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    _write_json(
        out / "self_check.json",
        {
            "schema_version": "0.4.0",
            "seed": int(seed),
            "scene_count": n,
            "record_count": len(rows),
            "oracle_agreement_count": n,
            "quarantined_count": sum(bool(row["quarantined"]) for row in rows[:: len(PROMPT_FAMILIES)]),
            "decision_counts": dict(sorted(class_counts.items())),
            "minimum_margin": min(float(row["margin"]) for row in rows),
        },
    )
    title = "Transpose Stamp Correspondence"
    question = next(iter(PROMPT_FAMILIES.values()))
    (gallery / "index.html").write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>"
        + title
        + "</title><style>body{font-family:sans-serif;background:#f4f5f6;color:#20242a}"
        + "main{max-width:1100px;margin:auto}section{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}"
        + "figure{margin:0;background:white;padding:8px}img{width:100%;height:auto;image-rendering:pixelated}"
        + "figcaption{font-size:12px}</style></head><body><main><h1>"
        + title
        + "</h1><p>"
        + html.escape(question)
        + "</p><section>"
        + "".join(cards)
        + "</section></main></body></html>\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    generate(args.out, args.n, args.seed)


if __name__ == "__main__":
    main()
