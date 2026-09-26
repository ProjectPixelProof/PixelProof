"""Standalone verifier: stored dataset vs. analytic gold and the pixel oracle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageChops

import oracle
import prompts
import renderer


def verify(dataset: Path) -> list[str]:
    failures: list[str] = []
    manifest = dataset / "manifest.jsonl"
    if not manifest.is_file():
        return [f"missing manifest {manifest}"]
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return ["manifest has no records"]

    pixel_cache: dict[str, str] = {}
    for row in rows:
        label = row.get("example_id", "?")
        scene = row.get("scene")
        if not isinstance(scene, dict):
            failures.append(f"{label}: manifest row has no scene object")
            continue
        image_path = dataset / row["image_path"]
        if not image_path.is_file():
            failures.append(f"{label}: missing image {row['image_path']}")
            continue
        stored = Image.open(image_path).convert("RGB")
        expected = renderer.render(scene).convert("RGB")
        if stored.size != expected.size:
            failures.append(f"{label}: stored raster size {stored.size} != {expected.size}")
        elif ImageChops.difference(stored, expected).getbbox() is not None:
            failures.append(f"{label}: stored raster differs from a re-render of its scene")

        decision = renderer.analytic_gold(scene)
        if decision != row.get("decision"):
            failures.append(f"{label}: stored decision {row.get('decision')!r} != {decision!r}")
        expected_margin = float(renderer.margin(scene))
        if abs(expected_margin - float(row.get("margin", float("nan")))) > 1e-9:
            failures.append(f"{label}: stored margin != recomputed {expected_margin}")
        if bool(renderer.is_quarantined(scene)) != bool(row.get("quarantined")):
            failures.append(f"{label}: stored quarantine flag != recomputation")
        family = row.get("prompt_family")
        if family not in prompts.PROMPT_FAMILIES:
            failures.append(f"{label}: undeclared prompt family {family!r}")
            continue
        if row.get("question") != prompts.PROMPT_FAMILIES[family]:
            failures.append(f"{label}: stored question text differs from the prompt API")
        if row.get("answer") != prompts.correct_answer(family, decision):
            failures.append(f"{label}: stored answer differs from prompt gold")
        if tuple(row.get("candidates") or ()) != tuple(prompts.candidates_for(family)):
            failures.append(f"{label}: stored answer candidates differ from the prompt API")

        scene_id = row["scene_id"]
        if scene_id not in pixel_cache:
            pixel_cache[scene_id] = oracle.decision_from_image(stored)
        if pixel_cache[scene_id] != decision:
            failures.append(
                f"{label}: pixel-only decision {pixel_cache[scene_id]!r} != gold {decision!r}"
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
    print(f"OK: dataset {args.dataset} agrees with analytic gold and the pixel oracle")


if __name__ == "__main__":
    main()
