"""Recompute pixels, labels, margins, questions, and pixel-only decisions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageChops

from oracle import decision_from_image
from prompts import PROMPT_FAMILIES, candidates_for, correct_answer
from renderer import analytic_gold, is_quarantined, margin, render

REQUIRED = {
    "example_id", "scene_id", "seed", "scene", "image_path", "prompt_family",
    "question", "decision", "answer", "candidates", "margin", "quarantined",
}


def verify(dataset: Path) -> list[str]:
    failures: list[str] = []
    try:
        rows = [
            json.loads(line)
            for line in (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except Exception as exc:
        return [f"manifest unreadable: {exc}"]
    if not rows:
        return ["manifest has no rows"]

    seen: set[str] = set()
    grouped: dict[str, list[dict]] = defaultdict(list)
    for number, row in enumerate(rows, start=1):
        missing = REQUIRED - set(row)
        if missing:
            failures.append(f"row {number} missing {sorted(missing)}")
            continue
        if row["example_id"] in seen:
            failures.append(f"duplicate example_id {row['example_id']}")
        seen.add(row["example_id"])
        grouped[row["scene_id"]].append(row)

    for scene_id, group in grouped.items():
        first = group[0]
        try:
            relative = Path(first["image_path"])
            image_path = (dataset / relative).resolve()
            image_path.relative_to(dataset.resolve())
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            failures.append(f"{scene_id}: image unreadable or unsafe: {exc}")
            continue
        latent = first["scene"]
        try:
            expected = render(latent).convert("RGB")
            if ImageChops.difference(image, expected).getbbox() is not None:
                failures.append(f"{scene_id}: stored PNG differs from rerender")
            decision = analytic_gold(latent)
            recovered = decision_from_image(image)
            if recovered != decision:
                failures.append(f"{scene_id}: pixel decision {recovered!r} != {decision!r}")
            expected_margin = margin(latent)
            expected_quarantine = is_quarantined(latent)
        except Exception as exc:
            failures.append(f"{scene_id}: recomputation failed: {exc}")
            continue
        family_counts = Counter(row["prompt_family"] for row in group)
        if family_counts != Counter({family: 1 for family in PROMPT_FAMILIES}):
            failures.append(f"{scene_id}: prompt coverage is {dict(family_counts)}")
        for row in group:
            family = row["prompt_family"]
            if row["scene"] != latent or row["image_path"] != first["image_path"]:
                failures.append(f"{row['example_id']}: shared scene fields disagree")
            if family not in PROMPT_FAMILIES:
                failures.append(f"{row['example_id']}: unknown prompt family")
                continue
            checks = (
                (row["question"] == PROMPT_FAMILIES[family], "question drift"),
                (row["decision"] == decision, "decision drift"),
                (row["answer"] == correct_answer(family, decision), "answer drift"),
                (tuple(row["candidates"]) == candidates_for(family), "candidate-set drift"),
                (abs(float(row["margin"]) - expected_margin) <= 1e-9, "margin drift"),
                (bool(row["quarantined"]) == expected_quarantine, "quarantine drift"),
            )
            for ok, message in checks:
                if not ok:
                    failures.append(f"{row['example_id']}: {message}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    failures = verify(args.dataset)
    if failures:
        for failure in failures[:60]:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print(f"OK: verified {args.dataset}")


if __name__ == "__main__":
    main()
