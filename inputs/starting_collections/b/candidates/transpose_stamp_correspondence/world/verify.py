"""Verify a stored dataset against code and the independent pixel oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops

from oracle import decision_from_image
from prompts import PROMPT_FAMILIES, candidates_for, correct_answer
from renderer import analytic_gold, is_quarantined, margin, render

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
    try:
        lines = (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [f"manifest unreadable: {exc}"]
    rows: list[dict] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append(f"line {number}: invalid JSON: {exc}")
            continue
        if not REQUIRED <= set(row):
            failures.append(f"line {number}: missing keys {sorted(REQUIRED - set(row))}")
            continue
        rows.append(row)
    if not rows:
        return failures + ["manifest contains no rows"]

    checked_images: dict[str, str] = {}
    for row in rows:
        label = str(row["example_id"])
        family = str(row["prompt_family"])
        if family not in PROMPT_FAMILIES:
            failures.append(f"{label}: unknown prompt family")
            continue
        try:
            image_path = (dataset / str(row["image_path"])).resolve()
            image_path.relative_to(dataset.resolve())
            image = Image.open(image_path).convert("RGB")
            expected = render(row["scene"]).convert("RGB")
            if ImageChops.difference(image, expected).getbbox() is not None:
                failures.append(f"{label}: stored raster differs from rerender")
            decision = analytic_gold(row["scene"])
            if row["decision"] != decision:
                failures.append(f"{label}: analytic decision mismatch")
            if abs(float(row["margin"]) - margin(row["scene"])) > 1e-9:
                failures.append(f"{label}: margin mismatch")
            if bool(row["quarantined"]) != is_quarantined(row["scene"]):
                failures.append(f"{label}: quarantine mismatch")
            if row["question"] != PROMPT_FAMILIES[family]:
                failures.append(f"{label}: question mismatch")
            if row["answer"] != correct_answer(family, decision):
                failures.append(f"{label}: prompt answer mismatch")
            if tuple(row["candidates"]) != candidates_for(family):
                failures.append(f"{label}: candidate set mismatch")
            if decision_from_image(image) != decision:
                failures.append(f"{label}: pixel-only decision mismatch")
            old = checked_images.setdefault(str(row["scene_id"]), str(row["image_path"]))
            if old != str(row["image_path"]):
                failures.append(f"{label}: scene points to multiple images")
        except Exception as exc:
            failures.append(f"{label}: verification raised {exc!r}")
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
    print("OK: dataset matches analytic and pixel-only decisions")


if __name__ == "__main__":
    main()
