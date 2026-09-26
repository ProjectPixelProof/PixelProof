"""Generate deterministic relay-chain PNGs and their exact manifest."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from oracle import decision_from_image
from prompts import PROMPT_FAMILIES, candidates_for, correct_answer
from renderer import analytic_gold, is_quarantined, margin, render, sample_scene


def generate(out: Path, count: int, seed: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    gallery = out / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    cards: list[str] = []
    quarantine_count = 0
    for index in range(count):
        scene_seed = seed + index
        scene_id = f"scene_{index:04d}"
        scene = sample_scene(scene_seed)
        image = render(scene).convert("RGB")
        decision = analytic_gold(scene)
        recovered = decision_from_image(image)
        if recovered != decision:
            raise RuntimeError(f"{scene_id}: pixel decision {recovered!r} != {decision!r}")
        relative = f"gallery/{scene_id}.png"
        image.save(out / relative, format="PNG", optimize=False)
        scene_margin = margin(scene)
        quarantined = is_quarantined(scene)
        quarantine_count += int(quarantined)
        for family, question in PROMPT_FAMILIES.items():
            rows.append(
                {
                    "example_id": f"{scene_id}__{family}",
                    "scene_id": scene_id,
                    "seed": scene_seed,
                    "scene": scene,
                    "image_path": relative,
                    "prompt_family": family,
                    "question": question,
                    "decision": decision,
                    "answer": correct_answer(family, decision),
                    "candidates": list(candidates_for(family)),
                    "margin": scene_margin,
                    "quarantined": quarantined,
                }
            )
        cards.append(
            f'<figure><img src="{html.escape(scene_id)}.png" alt="{html.escape(scene_id)}">'
            f'<figcaption>{html.escape(scene_id)} — terminal tag {html.escape(decision)}, '
            f'runner-up Hamming gap {scene_margin:g}, quarantined '
            f'{str(quarantined).lower()}</figcaption></figure>'
        )

    (out / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    (out / "self_check.json").write_text(
        json.dumps(
            {
                "schema_version": "0.4.0",
                "scene_count": count,
                "record_count": len(rows),
                "seed": seed,
                "pixel_oracle_agreement": True,
                "quarantined_scene_count": quarantine_count,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    question = html.escape(PROMPT_FAMILIES["pf1"])
    (gallery / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Pattern relay evidence</title>'
        '<style>body{font:15px sans-serif;background:#f4f1ec;color:#20242c}'
        'main{max-width:1040px;margin:auto}figure{background:white;padding:12px;border-radius:8px}'
        'img{width:100%;height:auto}figcaption{padding-top:6px}</style><main>'
        '<h1>Pattern relay chain correspondence</h1><p>'
        + question
        + "</p>"
        + "".join(cards)
        + "</main>\n",
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
    generate(args.out, args.n, args.seed)


if __name__ == "__main__":
    main()
