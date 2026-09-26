"""Verifier for the texture_flow_orientation world.

Re-certifies an existing generated dataset (manifest.jsonl + gallery/*.png)
against the renderer, analytic gold, prompt APIs, quarantine flags, and the
scene-free pixel oracle. Exits zero only when every record is consistent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageChops

from oracle import decision_from_image
from renderer import analytic_gold, is_quarantined, margin, render
from prompts import PROMPT_FAMILIES, correct_answer, candidates_for


def verify(dataset: Path) -> list[str]:
    manifest = dataset / "manifest.jsonl"
    failures = []
    rows = []
    with manifest.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append((number, json.loads(line)))
            except json.JSONDecodeError as exc:
                failures.append(f"manifest line {number}: invalid JSON: {exc}")
    if not rows:
        failures.append("manifest has no records")

    images = {}
    for number, row in rows:
        scene = row["scene"]
        decision = analytic_gold(scene)
        if decision != row["decision"]:
            failures.append(f"{row['example_id']}: analytic gold != stored decision")
        m = float(margin(scene))
        if abs(m - float(row["margin"])) > 1e-9:
            failures.append(f"{row['example_id']}: margin mismatch")
        q = bool(is_quarantined(scene))
        if q != row["quarantined"]:
            failures.append(f"{row['example_id']}: quarantine mismatch")
        family = row["prompt_family"]
        if family not in PROMPT_FAMILIES:
            failures.append(f"{row['example_id']}: undeclared family")
            continue
        if tuple(row["candidates"]) != tuple(candidates_for(family)):
            failures.append(f"{row['example_id']}: candidate set mismatch")
        if correct_answer(family, decision) != row["answer"]:
            failures.append(f"{row['example_id']}: stored answer differs from prompt gold")

        image_path = (dataset / row["image_path"]).resolve()
        try:
            image_path.relative_to(dataset.resolve())
        except ValueError:
            failures.append(f"{row['example_id']}: unsafe image path")
            continue
        if not image_path.is_file():
            failures.append(f"{row['example_id']}: missing image {row['image_path']}")
            continue
        image = images.get(row["scene_id"])
        if image is None:
            image = Image.open(image_path).convert("RGB")
            images[row["scene_id"]] = image
        expected = render(scene).convert("RGB")
        if ImageChops.difference(image, expected).getbbox() is not None:
            failures.append(f"{row['example_id']}: rerender mismatch")
        pixel_decision = decision_from_image(image)
        if pixel_decision != decision:
            failures.append(
                f"{row['example_id']}: pixel oracle {pixel_decision!r} != analytic {decision!r}"
            )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    failures = verify(args.dataset)
    if failures:
        for failure in failures[:100]:
            print(f"FAIL: {failure}")
        print(f"{len(failures)} verification failure(s)")
        sys.exit(1)
    print(f"OK: dataset {args.dataset} verified ({len(open(args.dataset / 'manifest.jsonl').readlines())} lines)")


if __name__ == "__main__":
    sys.exit(main())
