"""Verify stored coherent-arc-triplet records from their actual PNGs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops

from oracle import decision_from_image
from prompts import PROMPT_FAMILIES, candidates_for, correct_answer
from renderer import analytic_gold, is_quarantined, margin, render

REQUIRED = {
    "example_id", "scene_id", "seed", "scene", "image_path", "prompt_family",
    "question", "decision", "answer", "candidates", "margin", "quarantined",
}


def verify_dataset(dataset: Path) -> list[str]:
    failures: list[str] = []
    manifest = dataset / "manifest.jsonl"
    if not manifest.is_file():
        return ["missing manifest.jsonl"]
    rows: list[dict] = []
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append(f"line {number}: invalid JSON: {exc}")
            continue
        missing = REQUIRED - set(row)
        if missing:
            failures.append(f"line {number}: missing {sorted(missing)}")
            continue
        rows.append(row)
    if not rows:
        return failures + ["manifest contains no records"]
    seen: dict[str, tuple[dict, str]] = {}
    classes: dict[str, set[str]] = {family: set() for family in PROMPT_FAMILIES}
    for row in rows:
        tag = str(row["example_id"])
        try:
            image_path = (dataset / str(row["image_path"])).resolve()
            image_path.relative_to(dataset.resolve())
            image = Image.open(image_path).convert("RGB")
            expected = render(row["scene"]).convert("RGB")
            if ImageChops.difference(image, expected).getbbox() is not None:
                failures.append(f"{tag}: rerender mismatch")
            decision = analytic_gold(row["scene"])
            if decision != row["decision"]:
                failures.append(f"{tag}: analytic decision mismatch")
            if abs(float(margin(row["scene"])) - float(row["margin"])) > 1e-9:
                failures.append(f"{tag}: margin mismatch")
            if bool(is_quarantined(row["scene"])) != bool(row["quarantined"]):
                failures.append(f"{tag}: quarantine mismatch")
            family = str(row["prompt_family"])
            if row["question"] != PROMPT_FAMILIES[family]:
                failures.append(f"{tag}: question mismatch")
            if row["answer"] != correct_answer(family, decision):
                failures.append(f"{tag}: answer mismatch")
            if tuple(row["candidates"]) != candidates_for(family):
                failures.append(f"{tag}: candidates mismatch")
            if decision_from_image(image) != decision:
                failures.append(f"{tag}: pixel decision mismatch")
            if not row["quarantined"]:
                classes[family].add(str(row["answer"]))
            previous = seen.setdefault(str(row["scene_id"]), (row["scene"], str(row["image_path"])))
            if previous != (row["scene"], str(row["image_path"])):
                failures.append(f"{tag}: inconsistent repeated scene")
        except Exception as exc:
            failures.append(f"{tag}: {type(exc).__name__}: {exc}")
    for family, values in classes.items():
        if values != {"yes", "no"}:
            failures.append(f"{family}: off-quarantine classes are {sorted(values)}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    failures = verify_dataset(args.dataset)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("OK: dataset verified from pixels")


if __name__ == "__main__":
    main()
