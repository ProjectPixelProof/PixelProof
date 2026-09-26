"""Generate deterministic PNG evidence and manifest rows."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import oracle
import prompts
import renderer


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_index(path: Path, rows: list[dict]) -> None:
    question = html.escape(rows[0]["question"] if rows else "Paired seam correspondence")
    links = []
    seen: set[str] = set()
    for row in rows:
        image_path = row["image_path"]
        if image_path in seen:
            continue
        seen.add(image_path)
        links.append(
            f'<li><a href="{html.escape(image_path)}">{html.escape(image_path)}</a>'
            f' — {html.escape(row["decision"])}</li>'
        )
    body = "\n".join(links)
    path.write_text(
        "<!doctype html><meta charset=\"utf-8\"><title>Paired seam correspondence</title>"
        f"<h1>Paired seam correspondence</h1><p>{question}</p><ul>{body}</ul>\n",
        encoding="utf-8",
    )


def generate(out: Path, n: int, seed: int) -> None:
    if n < 1:
        raise ValueError("n must be positive")
    gallery = out / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for index in range(n):
        scene_seed = int(seed) + index
        scene = renderer.sample_scene(scene_seed)
        image = renderer.render(scene).convert("RGB")
        decision = renderer.analytic_gold(scene)
        if oracle.decision_from_image(image) != decision:
            raise RuntimeError(f"pixel oracle disagreement at scene {index}")
        image_path = gallery / f"scene_{index:04d}.png"
        image.save(image_path, format="PNG", optimize=False)
        for family, question in prompts.PROMPT_FAMILIES.items():
            rows.append(
                {
                    "example_id": f"scene_{index:04d}__{family}",
                    "scene_id": f"scene_{index:04d}",
                    "seed": scene_seed,
                    "scene": scene,
                    "image_path": f"gallery/{image_path.name}",
                    "prompt_family": family,
                    "question": question,
                    "decision": decision,
                    "answer": prompts.correct_answer(family, decision),
                    "candidates": list(prompts.candidates_for(family)),
                    "margin": renderer.margin(scene),
                    "quarantined": renderer.is_quarantined(scene),
                }
            )
    _write_jsonl(out / "manifest.jsonl", rows)
    _write_index(gallery / "index.html", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    generate(args.out, args.n, args.seed)


if __name__ == "__main__":
    main()
