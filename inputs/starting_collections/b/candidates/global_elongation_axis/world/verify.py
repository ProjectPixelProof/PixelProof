"""Re-certify an existing global-elongation dataset against its manifest.

Usage:
    python world/verify.py --dataset <directory>

Exits zero only when stored images, analytic decisions, prompt answers,
answer candidate sets, margins, quarantine flags, and the scene-free pixel
oracle agree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageChops

import renderer
import oracle
import prompts


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()

    dataset = args.dataset
    manifest = dataset / "manifest.jsonl"
    if not manifest.is_file():
        _fail("manifest.jsonl is missing")

    failures = 0
    checks = 0
    rows = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        _fail("manifest has no records")

    for row in rows:
        image_path = dataset / row["image_path"]
        if not image_path.is_file():
            print(f"FAIL: missing image {row['image_path']}")
            failures += 1
            continue
        image = Image.open(image_path).convert("RGB")
        checks += 1

        expected = renderer.render(row["scene"]).convert("RGB")
        if ImageChops.difference(image, expected).getbbox() is not None:
            print(f"FAIL: {row['example_id']} rerender mismatch")
            failures += 1

        decision = renderer.analytic_gold(row["scene"])
        if decision != row["decision"]:
            print(f"FAIL: {row['example_id']} decision mismatch")
            failures += 1

        if abs(renderer.margin(row["scene"]) - float(row["margin"])) > 1e-9:
            print(f"FAIL: {row['example_id']} margin mismatch")
            failures += 1

        if bool(renderer.is_quarantined(row["scene"])) != bool(row["quarantined"]):
            print(f"FAIL: {row['example_id']} quarantine mismatch")
            failures += 1

        if prompts.correct_answer(row["prompt_family"], decision) != row["answer"]:
            print(f"FAIL: {row['example_id']} answer mismatch")
            failures += 1

        if tuple(row["candidates"]) != tuple(prompts.candidates_for(row["prompt_family"])):
            print(f"FAIL: {row['example_id']} candidates mismatch")
            failures += 1

        if oracle.decision_from_image(image) != decision:
            print(f"FAIL: {row['example_id']} pixel oracle mismatch")
            failures += 1

    if failures:
        _fail(f"{failures} check(s) failed across {checks} record(s)")
    print(f"OK: verified {len(rows)} records, {checks} images")


if __name__ == "__main__":
    main()
