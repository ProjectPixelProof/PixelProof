"""Deterministic dataset generator for the interlocked-ring panel world."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import prompts
import renderer


def build(out: Path, n: int, seed: int) -> list:
    gallery = out / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(n):
        scene_seed = seed * 10_000 + index
        scene = renderer.sample_scene(scene_seed)
        scene_id = f"scene_{index:04d}"
        image = renderer.render(scene)
        image_path = f"gallery/{scene_id}.png"
        image.save(out / image_path)
        decision = renderer.analytic_gold(scene)
        scene_margin = float(renderer.margin(scene))
        quarantined = bool(renderer.is_quarantined(scene))
        for family in sorted(prompts.PROMPT_FAMILIES):
            rows.append(
                {
                    "example_id": f"{scene_id}__{family}",
                    "scene_id": scene_id,
                    "seed": scene_seed,
                    "scene": scene,
                    "image_path": image_path,
                    "prompt_family": family,
                    "question": prompts.PROMPT_FAMILIES[family],
                    "decision": decision,
                    "answer": prompts.correct_answer(family, decision),
                    "candidates": list(prompts.candidates_for(family)),
                    "margin": scene_margin,
                    "quarantined": quarantined,
                }
            )
    (out / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    write_index(gallery, rows)
    return rows


def write_index(gallery: Path, rows: list) -> None:
    question = prompts.PROMPT_FAMILIES["pf_panel"]
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>linked_ring_panel gallery</title></head><body>",
        f"<h1>{html.escape(question)}</h1><table border='1'>",
        "<tr><th>scene</th><th>image</th><th>decision</th><th>margin</th>"
        "<th>quarantined</th></tr>",
    ]
    seen = set()
    for row in rows:
        if row["scene_id"] in seen:
            continue
        seen.add(row["scene_id"])
        name = Path(row["image_path"]).name
        parts.append(
            f"<tr><td>{row['scene_id']}</td>"
            f"<td><img src='{name}' width='480'></td>"
            f"<td>{row['decision']}</td><td>{row['margin']:.3f}</td>"
            f"<td>{row['quarantined']}</td></tr>"
        )
    parts.append("</table></body></html>")
    (gallery / "index.html").write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = build(args.out, args.n, args.seed)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
