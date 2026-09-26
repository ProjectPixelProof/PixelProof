"""Strict verifier for generated and checked-in datasets."""

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


def verify_dataset(dataset: Path) -> list[str]:
    failures: list[str] = []
    try:
        lines = (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return [f"cannot read manifest: {exc}"]
    rows: list[dict] = []
    for number, line in enumerate(lines, 1):
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
        else:
            rows.append(row)
    if not rows:
        return failures + ["manifest has no rows"]

    seen_images: dict[str, str] = {}
    classes: dict[str, set[str]] = {family: set() for family in PROMPT_FAMILIES}
    for row in rows:
        label = row["example_id"]
        try:
            family = row["prompt_family"]
            if family not in PROMPT_FAMILIES or row["question"] != PROMPT_FAMILIES[family]:
                raise ValueError("prompt contract mismatch")
            image_path = (dataset / row["image_path"]).resolve()
            image_path.relative_to(dataset.resolve())
            image = Image.open(image_path).convert("RGB")
            rerendered = render(row["scene"]).convert("RGB")
            if ImageChops.difference(image, rerendered).getbbox() is not None:
                raise ValueError("rerender mismatch")
            decision = analytic_gold(row["scene"])
            if row["decision"] != decision:
                raise ValueError("stored analytic decision mismatch")
            if decision_from_image(image) != decision:
                raise ValueError("pixel-only decision mismatch")
            if abs(float(row["margin"]) - margin(row["scene"])) > 1e-9:
                raise ValueError("margin mismatch")
            if bool(row["quarantined"]) != is_quarantined(row["scene"]):
                raise ValueError("quarantine mismatch")
            if row["answer"] != correct_answer(family, decision):
                raise ValueError("answer mismatch")
            if tuple(row["candidates"]) != candidates_for(family):
                raise ValueError("candidate set mismatch")
            previous = seen_images.setdefault(row["scene_id"], row["image_path"])
            if previous != row["image_path"]:
                raise ValueError("one scene references multiple images")
            if not row["quarantined"]:
                classes[family].add(row["answer"])
        except Exception as exc:
            failures.append(f"{label}: {exc}")
    for family, answers in classes.items():
        if len(answers) < 2:
            failures.append(f"{family}: fewer than two off-quarantine classes")
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
    print("OK: dataset is analytically and pixel-wise consistent")


if __name__ == "__main__":
    main()
