"""Standalone verifier: analytic scene arm versus scene-free pixel arm."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from PIL import Image

import oracle
import prompts
import renderer


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def check(dataset: Path) -> list[str]:
    problems: list[str] = []
    manifest = dataset / "manifest.jsonl"
    if not manifest.exists():
        return [f"missing manifest: {manifest}"]

    seen_scene: dict[str, str] = {}
    families: dict[str, set] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        scene = row["scene"]
        tag = row["example_id"]

        decision = renderer.analytic_gold(scene)
        margin = float(renderer.margin(scene))
        quarantined = bool(renderer.is_quarantined(scene))
        if decision != row["decision"]:
            problems.append(f"{tag}: decision {row['decision']} != {decision}")
        if abs(margin - float(row["margin"])) > 1e-9:
            problems.append(f"{tag}: margin {row['margin']} != {margin}")
        if quarantined != bool(row["quarantined"]):
            problems.append(f"{tag}: quarantine flag mismatch")

        family = row["prompt_family"]
        if row["question"] != prompts.PROMPT_FAMILIES[family]:
            problems.append(f"{tag}: question text drift")
        if row["answer"] != prompts.correct_answer(family, decision):
            problems.append(f"{tag}: answer mismatch")
        if tuple(row["candidates"]) != prompts.candidates_for(family):
            problems.append(f"{tag}: candidate set mismatch")
        if row["answer"] not in row["candidates"]:
            problems.append(f"{tag}: answer outside candidate set")

        image_path = dataset / row["image_path"]
        if not image_path.exists():
            problems.append(f"{tag}: missing image {image_path}")
            continue
        stored = image_path.read_bytes()
        if stored != _png_bytes(renderer.render(scene)):
            problems.append(f"{tag}: raster does not reproduce from scene")

        with Image.open(image_path) as handle:
            pixel = oracle.decision_from_image(handle.convert("RGB"))
        if quarantined:
            if pixel not in (decision, oracle.ABSTAIN):
                problems.append(f"{tag}: quarantined pixel arm said {pixel}")
        elif pixel != decision:
            problems.append(f"{tag}: pixel arm {pixel} != analytic {decision}")

        if seen_scene.setdefault(row["scene_id"], row["image_path"]) != row["image_path"]:
            problems.append(f"{tag}: scene reuses more than one image")
        if not quarantined:
            families.setdefault(family, set()).add(row["answer"])

    for family in prompts.PROMPT_FAMILIES:
        if len(families.get(family, set())) < 2:
            problems.append(f"{family}: fewer than two off-quarantine classes")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    problems = check(args.dataset)
    for problem in problems:
        print(f"FAIL {problem}")
    if problems:
        print(f"FAILED: {len(problems)} problems")
        return 1
    print(f"OK: {args.dataset} verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
