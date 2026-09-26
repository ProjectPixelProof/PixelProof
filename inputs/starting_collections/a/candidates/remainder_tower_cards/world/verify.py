"""Verify stored evidence against render, analytic, prompt, and image-only arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops

import oracle
import prompts
import renderer


def verify_dataset(dataset: Path) -> list[str]:
    failures: list[str] = []
    try:
        manifest = dataset / "manifest.jsonl"
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception as exc:
        return [f"manifest unreadable: {exc}"]
    if not rows:
        return ["empty manifest"]
    scenes: dict[str, dict] = {}
    families = {family: set() for family in prompts.PROMPT_FAMILIES}
    for row in rows:
        try:
            scene = row["scene"]
            family = row["prompt_family"]
            image_path = (dataset / row["image_path"]).resolve()
            image_path.relative_to(dataset.resolve())
            image = Image.open(image_path).convert("RGB")
            expected = renderer.render(scene).convert("RGB")
            if ImageChops.difference(image, expected).getbbox() is not None:
                failures.append(f"{row['example_id']}: raster mismatch")
            decision = renderer.analytic_gold(scene)
            checks = (
                (row["decision"] == decision, "decision"),
                (row["answer"] == prompts.correct_answer(family, decision), "answer"),
                (tuple(row["candidates"]) == prompts.candidates_for(family), "candidates"),
                (float(row["margin"]) == renderer.margin(scene), "margin"),
                (bool(row["quarantined"]) == renderer.is_quarantined(scene), "quarantine"),
                (row["question"] == prompts.PROMPT_FAMILIES[family], "question"),
                (oracle.decision_from_image(image) == decision, "pixel oracle"),
            )
            failures.extend(f"{row['example_id']}: bad {name}" for ok, name in checks if not ok)
            families[family].add(decision)
            scenes.setdefault(row["scene_id"], scene)
        except Exception as exc:
            failures.append(f"row verification error: {exc}")
    for family, decisions in families.items():
        if len(decisions) < 2:
            failures.append(f"{family}: insufficient class coverage")
    if len(rows) != len(scenes) * len(prompts.PROMPT_FAMILIES):
        failures.append("record count does not equal scenes times prompt families")
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
    print("OK: dataset verified")


if __name__ == "__main__":
    main()
