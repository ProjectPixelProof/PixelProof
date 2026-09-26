"""Standalone verifier: stored dataset versus recomputed analytic and pixel arms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageChops

import oracle
import prompts
import renderer


def verify(dataset: Path) -> list:
    failures = []
    manifest = dataset / "manifest.jsonl"
    if not manifest.is_file():
        return [f"missing manifest {manifest}"]
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return ["manifest is empty"]
    cache = {}
    for row in rows:
        example = row["example_id"]
        scene = row["scene"]
        image_path = dataset / row["image_path"]
        if not image_path.is_file():
            failures.append(f"{example}: missing image {row['image_path']}")
            continue
        stored = Image.open(image_path).convert("RGB")
        expected = renderer.render(scene).convert("RGB")
        if ImageChops.difference(stored, expected).getbbox() is not None:
            failures.append(f"{example}: stored raster differs from re-render")
        decision = renderer.analytic_gold(scene)
        if decision != row["decision"]:
            failures.append(f"{example}: decision differs from analytic gold")
        if abs(float(renderer.margin(scene)) - float(row["margin"])) > 1e-9:
            failures.append(f"{example}: margin differs from recomputation")
        quarantined = bool(renderer.is_quarantined(scene))
        if quarantined != bool(row["quarantined"]):
            failures.append(f"{example}: quarantine flag differs from recomputation")
        if prompts.correct_answer(row["prompt_family"], decision) != row["answer"]:
            failures.append(f"{example}: answer differs from prompt gold")
        if tuple(row["candidates"]) != tuple(prompts.candidates_for(row["prompt_family"])):
            failures.append(f"{example}: candidate set differs from prompt API")
        if row["question"] != prompts.PROMPT_FAMILIES.get(row["prompt_family"]):
            failures.append(f"{example}: question text differs from prompt API")
        key = str(image_path)
        if key not in cache:
            cache[key] = oracle.decision_from_image(stored)
        pixel = cache[key]
        if quarantined:
            if pixel not in (decision, oracle.AMBIGUOUS):
                failures.append(
                    f"{example}: quarantined pixel decision {pixel!r} is neither "
                    f"gold {decision!r} nor an abstention"
                )
        elif pixel != decision:
            failures.append(
                f"{example}: pixel-only decision {pixel!r} != gold {decision!r}"
            )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    failures = verify(args.dataset)
    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        sys.exit(1)
    print(f"OK: dataset {args.dataset} is consistent")


if __name__ == "__main__":
    main()
