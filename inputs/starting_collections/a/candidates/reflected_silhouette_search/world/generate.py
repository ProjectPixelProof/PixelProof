"""Generate deterministic PNG evidence and paired prompt records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import oracle
import prompts
import renderer


def generate(out: Path, n: int, seed: int) -> None:
    out = out.resolve()
    gallery = out / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for index in range(int(n)):
        scene_seed = int(seed) + index * 1009
        scene = renderer.sample_scene(scene_seed)
        image = renderer.render(scene).convert("RGB")
        decision = renderer.analytic_gold(scene)
        observed = oracle.decision_from_image(image)
        if observed != decision:
            raise RuntimeError(
                f"pixel oracle disagreed for scene {index}: {observed!r} != {decision!r}"
            )
        scene_id = f"scene_{index:04d}"
        image_path = f"gallery/{scene_id}.png"
        image.save(out / image_path)
        quarantined = renderer.is_quarantined(scene)
        for family, question in prompts.PROMPT_FAMILIES.items():
            rows.append(
                {
                    "example_id": f"{scene_id}__{family}",
                    "scene_id": scene_id,
                    "seed": scene_seed,
                    "scene": scene,
                    "image_path": image_path,
                    "prompt_family": family,
                    "question": question,
                    "decision": decision,
                    "answer": prompts.correct_answer(family, decision),
                    "candidates": list(prompts.candidates_for(family)),
                    "margin": float(renderer.margin(scene)),
                    "quarantined": bool(quarantined),
                }
            )
    (out / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    (out / "self_check.json").write_text(
        json.dumps(
            {
                "status": "generated",
                "scene_count": int(n),
                "record_count": len(rows),
                "seed": int(seed),
                "prompt_families": sorted(prompts.PROMPT_FAMILIES),
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
    generate(args.out, args.n, args.seed)


if __name__ == "__main__":
    main()
