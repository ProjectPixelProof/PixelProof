"""Verify stored pixels, public gold, and the separate image-only inverse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops

from oracle import decision_from_image
from prompts import PROMPT_FAMILIES, candidates_for, correct_answer
from renderer import analytic_gold, is_quarantined, margin, render


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    failures = []
    try:
        rows = [json.loads(line) for line in (args.dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception as exc:
        raise SystemExit(f"invalid manifest: {exc}")
    if not rows:
        raise SystemExit("empty manifest")
    seen = set()
    for row in rows:
        try:
            family = row["prompt_family"]
            if family not in PROMPT_FAMILIES or row["question"] != PROMPT_FAMILIES[family]:
                failures.append(f"{row.get('example_id')}: prompt mismatch")
            image_path = (args.dataset / row["image_path"]).resolve()
            image_path.relative_to(args.dataset.resolve())
            image = Image.open(image_path).convert("RGB")
            expected = render(row["scene"]).convert("RGB")
            if ImageChops.difference(image, expected).getbbox() is not None:
                failures.append(f"{row['example_id']}: raster mismatch")
            decision = analytic_gold(row["scene"])
            if row["decision"] != decision or decision_from_image(image) != decision:
                failures.append(f"{row['example_id']}: decision mismatch")
            if float(row["margin"]) != margin(row["scene"]):
                failures.append(f"{row['example_id']}: margin mismatch")
            if bool(row["quarantined"]) != is_quarantined(row["scene"]):
                failures.append(f"{row['example_id']}: quarantine mismatch")
            if row["answer"] != correct_answer(family, decision):
                failures.append(f"{row['example_id']}: answer mismatch")
            if tuple(row["candidates"]) != candidates_for(family):
                failures.append(f"{row['example_id']}: candidates mismatch")
            seen.add((row["scene_id"], family))
        except Exception as exc:
            failures.append(f"{row.get('example_id', '?')}: {exc!r}")
    if failures:
        raise SystemExit("\n".join(failures[:30]))
    print(f"OK: verified {len(rows)} records across {len(seen)} scene-family pairs")


if __name__ == "__main__":
    main()
