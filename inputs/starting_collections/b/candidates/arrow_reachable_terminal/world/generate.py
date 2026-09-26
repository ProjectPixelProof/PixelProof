"""Generate deterministic PNG datasets and their public manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prompts import PROMPT_FAMILIES, candidates_for, correct_answer
from renderer import analytic_gold, is_quarantined, margin, render, sample_scene


def generate_dataset(out: Path, n: int, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    gallery = out / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    for old in gallery.glob("scene_*.png"):
        old.unlink()

    rows: list[dict] = []
    cards: list[str] = []
    for index in range(n):
        scene_seed = int(seed) + index
        scene_id = f"scene_{index:04d}"
        scene = sample_scene(scene_seed)
        image_name = f"{scene_id}.png"
        render(scene).save(gallery / image_name, format="PNG", optimize=False)
        decision = analytic_gold(scene)
        cards.append(f'<figure><img src="{image_name}" width="320"><figcaption>{scene_id}: {decision}</figcaption></figure>')
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
    question = next(iter(PROMPT_FAMILIES.values()))
    (gallery / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Arrow reachable terminal</title>"
        f"<h1>Arrow reachable terminal</h1><p>{question}</p>"
        "<style>body{font:16px sans-serif;background:#faf8f4;color:#2a2f37}figure{display:inline-block;margin:10px}img{border:1px solid #bbb}</style>"
        + "".join(cards),
        encoding="utf-8",
    )
    counts = {color: sum(row["decision"] == color for row in rows) for color in ("red", "blue", "orange")}
    (out / "self_check.json").write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "scene_count": n,
                "row_count": len(rows),
                "decision_counts": counts,
                "all_off_quarantine": all(not row["quarantined"] for row in rows),
                "minimum_margin": min((row["margin"] for row in rows), default=None),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.n < 1:
        raise SystemExit("--n must be positive")
    generate_dataset(args.out, args.n, args.seed)


if __name__ == "__main__":
    main()
