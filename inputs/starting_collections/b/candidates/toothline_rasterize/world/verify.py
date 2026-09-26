"""Verify a generated dataset, including the exact submitted evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops

from oracle import measure_from_image
from prompts import PROMPT_FAMILIES, candidates_for, correct_answer
from renderer import analytic_gold, is_quarantined, margin, render


def verify(dataset: Path) -> None:
    rows = [json.loads(line) for line in (dataset / "manifest.jsonl").read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty manifest")
    seen = {}
    family_answers = {family: set() for family in PROMPT_FAMILIES}
    for row in rows:
        required = {"example_id","scene_id","seed","scene","image_path","prompt_family","question","decision","answer","candidates","margin","quarantined"}
        if not required <= set(row):
            raise ValueError("manifest row lacks required keys")
        image_path = (dataset / row["image_path"]).resolve()
        image_path.relative_to(dataset.resolve())
        image = Image.open(image_path).convert("RGB")
        expected = render(row["scene"]).convert("RGB")
        if ImageChops.difference(image, expected).getbbox() is not None:
            raise ValueError(f"rerender mismatch: {row['example_id']}")
        decision = analytic_gold(row["scene"])
        if decision != row["decision"]:
            raise ValueError("analytic decision mismatch")
        if float(row["margin"]) != margin(row["scene"]):
            raise ValueError("analytic margin mismatch")
        if bool(row["quarantined"]) != is_quarantined(row["scene"]):
            raise ValueError("quarantine mismatch")
        family = row["prompt_family"]
        if row["question"] != PROMPT_FAMILIES[family]:
            raise ValueError("question mismatch")
        if row["answer"] != correct_answer(family, decision):
            raise ValueError("answer mismatch")
        if tuple(row["candidates"]) != candidates_for(family):
            raise ValueError("candidate set mismatch")
        measured = measure_from_image(image)
        if measured["decision"] != decision:
            raise ValueError("pixel-only inverse mismatch")
        if "pixel_runner_up_margin" in row and float(row["pixel_runner_up_margin"]) != measured["runner_up_margin"]:
            raise ValueError("stored pixel margin mismatch")
        prior = seen.setdefault(row["scene_id"], (row["image_path"], decision))
        if prior != (row["image_path"], decision):
            raise ValueError("scene rows disagree")
        family_answers[family].add(row["answer"])
    if any(len(values) < 2 for values in family_answers.values()):
        raise ValueError("dataset lacks off-boundary class coverage")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    verify(args.dataset)
    print(f"OK: verified {args.dataset}")


if __name__ == "__main__":
    main()
