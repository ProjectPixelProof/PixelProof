"""Dataset generator for the interleaved-sheet extraction world."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import prompts
import renderer


def build(out: Path, n: int, seed: int) -> None:
    gallery = out / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(n):
        scene = renderer.sample_scene(seed * 1000 + index)
        image = renderer.render(scene)
        scene_id = f"scene_{index:04d}"
        rel = f"gallery/{scene_id}.png"
        image.save(out / rel)
        decision = renderer.analytic_gold(scene)
        margin = float(renderer.margin(scene))
        quarantined = bool(renderer.is_quarantined(scene))
        for family in sorted(prompts.PROMPT_FAMILIES):
            rows.append(
                {
                    "example_id": f"{scene_id}__{family}",
                    "scene_id": scene_id,
                    "seed": seed,
                    "scene": scene,
                    "image_path": rel,
                    "prompt_family": family,
                    "question": prompts.PROMPT_FAMILIES[family],
                    "decision": decision,
                    "answer": prompts.correct_answer(family, decision),
                    "candidates": list(prompts.candidates_for(family)),
                    "margin": margin,
                    "quarantined": quarantined,
                }
            )
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    scene_rows = [r for r in rows if r["prompt_family"] == "pf1_select_symbol"]
    summary = {
        "scenes": len(scene_rows),
        "examples": len(rows),
        "seed": seed,
        "prompt_families": sorted(prompts.PROMPT_FAMILIES),
        "decisions": sorted({r["decision"] for r in scene_rows}),
        "quarantined": sum(1 for r in scene_rows if r["quarantined"]),
        "min_margin": min(r["margin"] for r in scene_rows),
        "max_margin": max(r["margin"] for r in scene_rows),
    }
    (out / "self_check.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>interleaved_sheet_extract gallery</title>",
        "<h1>interleaved_sheet_extract</h1>",
    ]
    for row in scene_rows:
        parts.append(
            "<figure style='display:inline-block;margin:6px'>"
            f"<img src='{row['image_path']}' width='240'>"
            f"<figcaption>{row['scene_id']}: {row['decision']} "
            f"(margin {row['margin']:.0f}"
            f"{', quarantined' if row['quarantined'] else ''})</figcaption>"
            "</figure>"
        )
    (out / "gallery" / "index.html").write_text(
        "\n".join(parts) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    build(args.out, args.n, args.seed)
    print(f"OK: wrote {args.n} scenes to {args.out}")


if __name__ == "__main__":
    main()
