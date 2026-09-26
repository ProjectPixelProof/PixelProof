"""Deterministic dataset generator for the end-fold pairing reveal world."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import oracle
import prompts
import renderer


def build(out: Path, count: int, seed: int) -> list[dict]:
    gallery = out / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    families = sorted(prompts.PROMPT_FAMILIES)
    rows: list[dict] = []
    for index in range(count):
        scene_seed = seed + index
        scene = renderer.sample_scene(scene_seed)
        scene_id = f"scene_{index:04d}"
        image_path = f"gallery/{scene_id}.png"
        renderer.render(scene).save(out / image_path, format="PNG", optimize=True)
        decision = renderer.analytic_gold(scene)
        scene_margin = float(renderer.margin(scene))
        quarantined = bool(renderer.is_quarantined(scene))
        for family in families:
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
                    "glyph_hamming_margin": int(renderer.hamming_margin(scene)),
                    "bead_count": len(scene["route"]),
                }
            )
    (out / "manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    return rows


def write_reports(out: Path, rows: list[dict], count: int, seed: int) -> None:
    scenes = {row["scene_id"]: row for row in rows}
    decisions: dict[str, int] = {}
    for row in scenes.values():
        decisions[row["decision"]] = decisions.get(row["decision"], 0) + 1
    quarantined = sorted(key for key, row in scenes.items() if row["quarantined"])
    margins = sorted(row["margin"] for row in scenes.values())
    self_check = {
        "schema_version": "0.4.0",
        "world": "endfold_pairing_reveal",
        "renderer_version": renderer.RENDERER_VERSION,
        "oracle_version": oracle.ORACLE_VERSION,
        "prompts_version": prompts.PROMPTS_VERSION,
        "base_seed": seed,
        "scene_count": count,
        "record_count": len(rows),
        "prompt_families": sorted(prompts.PROMPT_FAMILIES),
        "decision_counts": dict(sorted(decisions.items())),
        "quarantined_scene_ids": quarantined,
        "quarantine_clearance": renderer.QUARANTINE_CLEARANCE,
        "min_margin": margins[0] if margins else None,
        "max_margin": margins[-1] if margins else None,
    }
    (out / "self_check.json").write_text(
        json.dumps(self_check, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    question = prompts.PROMPT_FAMILIES["pf1_symbol_choice"]
    cards = []
    for scene_id in sorted(scenes):
        row = scenes[scene_id]
        cards.append(
            "<figure><img src='gallery/{sid}.png' width='320'>"
            "<figcaption>{sid} &middot; decision={dec} &middot; margin={mar:.2f}"
            " &middot; quarantined={qua}</figcaption></figure>".format(
                sid=scene_id,
                dec=row["decision"],
                mar=row["margin"],
                qua=str(row["quarantined"]).lower(),
            )
        )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>endfold_pairing_reveal gallery</title></head><body>"
        "<h1>endfold_pairing_reveal</h1><p>{question}</p>"
        "<p>Marked grid positions are the folded pairs whose two beads share a gray, "
        "after walking the teal cord from its bracketed end bead and pairing bead k with "
        "bead 51-k; quarantined scenes have a per-bead tone clearance below {thr} luma "
        "units.</p>"
        "{cards}</body></html>\n"
    ).format(question=question, thr=renderer.QUARANTINE_CLEARANCE, cards="".join(cards))
    (out / "gallery/index.html").write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = build(args.out, args.n, args.seed)
    write_reports(args.out, rows, args.n, args.seed)
    print(f"wrote {len(rows)} records for {args.n} scenes to {args.out}")


if __name__ == "__main__":
    main()
