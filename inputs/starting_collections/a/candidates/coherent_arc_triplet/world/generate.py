"""Generate a deterministic coherent-arc-triplet dataset."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from collections import Counter
from pathlib import Path

from oracle import decision_from_image
from prompts import PROMPT_FAMILIES, candidates_for, correct_answer
from renderer import analytic_gold, is_quarantined, margin, render, sample_scene


def _json_line(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"


def generate_dataset(out: Path, n: int, seed: int) -> None:
    if n < 4 or n > 512:
        raise ValueError("n must be in [4, 512]")
    out.mkdir(parents=True, exist_ok=True)
    gallery = out / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    cards: list[str] = []
    classes: Counter[str] = Counter()
    margins: list[float] = []
    for index in range(n):
        scene_seed = int(seed) * 1009 + index
        scene = sample_scene(scene_seed)
        image = render(scene).convert("RGB")
        decision = analytic_gold(scene)
        observed = decision_from_image(image)
        if observed != decision:
            raise RuntimeError(f"pixel oracle disagreed for scene {index}: {observed} != {decision}")
        scene_id = f"scene_{index:04d}"
        relative = f"gallery/{scene_id}.png"
        image.save(out / relative, format="PNG", optimize=False)
        scene_margin = float(margin(scene))
        quarantined = bool(is_quarantined(scene))
        classes[decision] += int(not quarantined)
        margins.append(scene_margin)
        for family, question in PROMPT_FAMILIES.items():
            answer = correct_answer(family, decision)
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
                    "answer": answer,
                    "candidates": list(candidates_for(family)),
                    "margin": scene_margin,
                    "quarantined": quarantined,
                    "oracle_decision": observed,
                }
            )
        cards.append(
            f'<figure><img src="{scene_id}.png" width="256" height="256" '
            f'alt="three separated curve fragments"><figcaption>{scene_id}: '
            f'{html.escape(decision)}, margin {scene_margin:g}</figcaption></figure>'
        )
    manifest_text = "".join(_json_line(row) for row in rows)
    (out / "manifest.jsonl").write_text(manifest_text, encoding="utf-8")
    self_check = {
        "schema_version": "candidate-self-check-0.1.0",
        "candidate_id": "coherent_arc_triplet",
        "scene_count": n,
        "record_count": len(rows),
        "seed": int(seed),
        "off_quarantine_classes": dict(sorted(classes.items())),
        "minimum_margin": min(margins),
        "oracle_agreement": True,
        "manifest_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
    }
    (out / "self_check.json").write_text(
        json.dumps(self_check, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    index_html = (
        '<!doctype html><html><head><meta charset="utf-8"><title>Coherent arc triplet</title>'
        '<style>body{font-family:sans-serif;background:#f6f5f1;color:#222}main{display:flex;flex-wrap:wrap}'
        'figure{margin:10px}img{border:1px solid #aaa;background:white}</style></head><body>'
        '<h1>Coherent arc triplet</h1><p>Do all three separated colored curve fragments '
        'trace parts of one imaginary circle?</p><main>' + "".join(cards) + "</main></body></html>\n"
    )
    (gallery / "index.html").write_text(index_html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    generate_dataset(args.out, args.n, args.seed)


if __name__ == "__main__":
    main()
