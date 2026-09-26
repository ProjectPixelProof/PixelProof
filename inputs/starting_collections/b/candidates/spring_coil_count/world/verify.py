"""Verify a stored dataset without trusting its labels or rasters."""

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
    manifest = args.dataset / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise SystemExit("empty manifest")
    for row in rows:
        try:
            family = row["prompt_family"]
            scene = row["scene"]
            image_path = (args.dataset / row["image_path"]).resolve()
            image_path.relative_to(args.dataset.resolve())
            image = Image.open(image_path).convert("RGB")
            expected = render(scene).convert("RGB")
            if ImageChops.difference(image, expected).getbbox() is not None:
                failures.append(f"{row['example_id']}: raster mismatch")
            decision = analytic_gold(scene)
            if row["question"] != PROMPT_FAMILIES[family]:
                failures.append(f"{row['example_id']}: question mismatch")
            if row["decision"] != decision or row["answer"] != correct_answer(family, decision):
                failures.append(f"{row['example_id']}: label mismatch")
            if tuple(row["candidates"]) != candidates_for(family):
                failures.append(f"{row['example_id']}: candidates mismatch")
            if abs(float(row["margin"]) - margin(scene)) > 1e-9:
                failures.append(f"{row['example_id']}: margin mismatch")
            if bool(row["quarantined"]) != is_quarantined(scene):
                failures.append(f"{row['example_id']}: quarantine mismatch")
            if decision_from_image(image) != decision:
                failures.append(f"{row['example_id']}: pixel decision mismatch")
        except Exception as exc:
            failures.append(f"{row.get('example_id', 'unknown')}: {exc!r}")
    if failures:
        raise SystemExit("\n".join(failures[:20]))
    print(f"OK: verified {len(rows)} rows")


if __name__ == "__main__":
    main()

