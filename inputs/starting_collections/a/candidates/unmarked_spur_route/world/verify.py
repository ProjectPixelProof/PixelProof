"""Standalone exact-dataset verifier for the spur world."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops

import oracle
import prompts
import renderer


REQUIRED = {
    "example_id", "scene_id", "seed", "scene", "image_path", "prompt_family",
    "question", "decision", "answer", "candidates", "margin", "quarantined",
}


def verify(dataset: Path) -> list[str]:
    failures: list[str] = []
    manifest = dataset / "manifest.jsonl"
    if not manifest.is_file():
        return ["manifest missing"]
    rows = []
    for line_no, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append(f"line {line_no}: invalid JSON: {exc}")
            continue
        if REQUIRED - set(row):
            failures.append(f"line {line_no}: missing {sorted(REQUIRED - set(row))}")
        rows.append(row)
    if not rows:
        failures.append("manifest empty")
        return failures
    seen_images = {}
    for row in rows:
        label = str(row.get("example_id"))
        family = row.get("prompt_family")
        if family not in prompts.PROMPT_FAMILIES:
            failures.append(f"{label}: unknown prompt family")
            continue
        image_path = (dataset / str(row.get("image_path", ""))).resolve()
        try:
            image_path.relative_to(dataset.resolve())
        except ValueError:
            failures.append(f"{label}: image escapes dataset")
            continue
        if not image_path.is_file():
            failures.append(f"{label}: image missing")
            continue
        image = Image.open(image_path).convert("RGB")
        seen_images.setdefault(str(image_path), image)
        scene = row["scene"]
        expected = renderer.render(scene).convert("RGB")
        if ImageChops.difference(image, expected).getbbox() is not None:
            failures.append(f"{label}: render mismatch")
        decision = renderer.analytic_gold(scene)
        if row["decision"] != decision:
            failures.append(f"{label}: analytic decision mismatch")
        if abs(float(row["margin"]) - renderer.margin(scene)) > 1e-9:
            failures.append(f"{label}: margin mismatch")
        if bool(row["quarantined"]) != renderer.is_quarantined(scene):
            failures.append(f"{label}: quarantine mismatch")
        if row["question"] != prompts.PROMPT_FAMILIES[family]:
            failures.append(f"{label}: question mismatch")
        if row["answer"] != prompts.correct_answer(family, decision):
            failures.append(f"{label}: answer mismatch")
        if tuple(row["candidates"]) != prompts.candidates_for(family):
            failures.append(f"{label}: candidate set mismatch")
        try:
            observed = oracle.decision_from_image(image)
        except Exception as exc:
            failures.append(f"{label}: oracle raised {exc!r}")
        else:
            if observed != decision:
                failures.append(f"{label}: pixel oracle mismatch")
    if len(seen_images) < 4:
        failures.append("fewer than four distinct images")
    for family in prompts.PROMPT_FAMILIES:
        answers = {row["answer"] for row in rows if row.get("prompt_family") == family and not row.get("quarantined")}
        if len(answers) < 2:
            failures.append(f"{family}: class balance is below two")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    failures = verify(args.dataset)
    if failures:
        for failure in failures:
            print(failure)
        raise SystemExit(1)
    print("verified", args.dataset)


if __name__ == "__main__":
    main()
