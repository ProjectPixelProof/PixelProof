"""Recompute every field of an existing paired-seam dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops

import oracle
import prompts
import renderer


REQUIRED = {
    "example_id",
    "scene_id",
    "seed",
    "scene",
    "image_path",
    "prompt_family",
    "question",
    "decision",
    "answer",
    "candidates",
    "margin",
    "quarantined",
}


def verify(dataset: Path) -> list[str]:
    failures: list[str] = []
    manifest = dataset / "manifest.jsonl"
    if not manifest.is_file():
        return ["manifest.jsonl is missing"]
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return ["manifest is empty"]
    for row in rows:
        missing = REQUIRED - set(row)
        if missing:
            failures.append(f"{row.get('example_id', '?')}: missing {sorted(missing)}")
            continue
        try:
            image_path = (dataset / row["image_path"]).resolve()
            image_path.relative_to(dataset.resolve())
            image = Image.open(image_path).convert("RGB")
            expected_image = renderer.render(row["scene"]).convert("RGB")
            if ImageChops.difference(image, expected_image).getbbox() is not None:
                failures.append(f"{row['example_id']}: render mismatch")
            decision = renderer.analytic_gold(row["scene"])
            if row["decision"] != decision:
                failures.append(f"{row['example_id']}: analytic decision mismatch")
            if abs(float(row["margin"]) - renderer.margin(row["scene"])) > 1e-9:
                failures.append(f"{row['example_id']}: margin mismatch")
            if bool(row["quarantined"]) != renderer.is_quarantined(row["scene"]):
                failures.append(f"{row['example_id']}: quarantine mismatch")
            family = row["prompt_family"]
            if row["question"] != prompts.PROMPT_FAMILIES[family]:
                failures.append(f"{row['example_id']}: question mismatch")
            if row["answer"] != prompts.correct_answer(family, decision):
                failures.append(f"{row['example_id']}: answer mismatch")
            if tuple(row["candidates"]) != prompts.candidates_for(family):
                failures.append(f"{row['example_id']}: candidates mismatch")
            if oracle.decision_from_image(image) != decision:
                failures.append(f"{row['example_id']}: pixel oracle mismatch")
        except Exception as exc:
            failures.append(f"{row.get('example_id', '?')}: {exc}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    failures = verify(args.dataset)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("OK: dataset verified")


if __name__ == "__main__":
    main()
