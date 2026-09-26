"""Generate deterministic PNG evidence and the required paired manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import prompts
import renderer


def _write_gallery_index(out: Path, rows: list[dict]) -> None:
    question = prompts.PROMPT_FAMILIES["pf1_curvature_rank"]
    cards: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if row["scene_id"] in seen:
            continue
        seen.add(row["scene_id"])
        cards.append(
            "<figure><img src=\"%s\" alt=\"%s\"><figcaption>%s: answer %s, margin %s, quarantined %s</figcaption></figure>"
            % (
                row["image_path"],
                row["scene_id"],
                row["scene_id"],
                row["answer"],
                row["margin"],
                row["quarantined"],
            )
        )
    html = (
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>Arc curvature correspondence gallery</title>"
        "<style>body{font-family:sans-serif;background:#fafaf7}figure{display:inline-block;vertical-align:top;margin:10px;width:440px}img{width:440px;display:block;border:1px solid #bbb}figcaption{font-size:14px;margin-top:5px}</style></head><body>"
        f"<h1>{question}</h1><p>Deterministic pixel evidence gallery.</p>"
        + "".join(cards)
        + "</body></html>\n"
    )
    (out / "gallery" / "index.html").write_text(html, encoding="utf-8")


def generate(out: Path, count: int, seed: int) -> None:
    if count <= 0:
        raise ValueError("count must be positive")
    out.mkdir(parents=True, exist_ok=True)
    gallery = out / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for index in range(count):
        scene_seed = int(seed) + index
        scene_id = f"scene_{index:04d}"
        scene = renderer.sample_scene(scene_seed)
        image = renderer.render(scene).convert("RGB")
        image_path = gallery / f"{scene_id}.png"
        image.save(image_path, format="PNG", optimize=False)
        decision = renderer.analytic_gold(scene)
        margin = renderer.margin(scene)
        quarantined = renderer.is_quarantined(scene)
        for family, question in prompts.PROMPT_FAMILIES.items():
            rows.append(
                {
                    "example_id": f"{scene_id}__{family}",
                    "scene_id": scene_id,
                    "seed": scene_seed,
                    "scene": scene,
                    "image_path": f"gallery/{scene_id}.png",
                    "prompt_family": family,
                    "question": question,
                    "decision": decision,
                    "answer": prompts.correct_answer(family, decision),
                    "candidates": list(prompts.candidates_for(family)),
                    "margin": margin,
                    "quarantined": quarantined,
                }
            )
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    _write_gallery_index(out, rows)
    self_check = {
        "generator_version": "colored-arc-curvature-generate-0.1.0",
        "scene_count": count,
        "seed": int(seed),
        "prompt_families": sorted(prompts.PROMPT_FAMILIES),
        "records": len(rows),
        "verification": "run world/verify.py --dataset on this directory",
    }
    (out / "self_check.json").write_text(
        json.dumps(self_check, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
