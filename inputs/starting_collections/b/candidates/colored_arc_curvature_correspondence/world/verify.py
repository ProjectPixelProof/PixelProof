"""Re-certify an existing dataset against renderer, prompts, and pixel oracle."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

import oracle
import prompts
import renderer


REQUIRED_KEYS = {
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
    try:
        rows = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        return [f"manifest is invalid: {exc}"]
    if not rows:
        failures.append("manifest has no rows")
    scenes: dict[str, dict] = {}
    images: dict[str, bytes] = {}
    by_family: dict[str, set[str]] = {family: set() for family in prompts.PROMPT_FAMILIES}
    for row_number, row in enumerate(rows, start=1):
        missing = REQUIRED_KEYS - set(row)
        if missing:
            failures.append(f"row {row_number} missing {sorted(missing)}")
            continue
        family = row["prompt_family"]
        if family not in prompts.PROMPT_FAMILIES:
            failures.append(f"row {row_number} has undeclared prompt family {family}")
            continue
        image_path = (dataset / row["image_path"]).resolve()
        try:
            image_path.relative_to(dataset.resolve())
        except ValueError:
            failures.append(f"row {row_number} image path escapes dataset")
            continue
        if not image_path.is_file() or image_path.suffix.lower() != ".png":
            failures.append(f"row {row_number} image is missing or not PNG")
            continue
        try:
            image = Image.open(image_path).convert("RGB")
            image_bytes = image.tobytes()
            if image.size != (renderer.WIDTH, renderer.HEIGHT):
                failures.append(f"row {row_number} image has size {image.size}")
            expected = renderer.render(row["scene"]).convert("RGB")
            if image_bytes != expected.tobytes():
                failures.append(f"row {row_number} render fidelity mismatch")
            decision = renderer.analytic_gold(row["scene"])
            if decision != row["decision"]:
                failures.append(f"row {row_number} analytic decision mismatch")
            if abs(float(renderer.margin(row["scene"])) - float(row["margin"])) > 1e-9:
                failures.append(f"row {row_number} margin mismatch")
            if bool(renderer.is_quarantined(row["scene"])) != bool(row["quarantined"]):
                failures.append(f"row {row_number} quarantine mismatch")
            if prompts.correct_answer(family, decision) != row["answer"]:
                failures.append(f"row {row_number} prompt answer mismatch")
            if tuple(row["candidates"]) != prompts.candidates_for(family):
                failures.append(f"row {row_number} candidate set mismatch")
            if row["question"] != prompts.PROMPT_FAMILIES[family]:
                failures.append(f"row {row_number} prompt text mismatch")
            if oracle.decision_from_image(image) != decision:
                failures.append(f"row {row_number} pixel oracle mismatch")
            scenes.setdefault(row["scene_id"], row["scene"])
            images.setdefault(row["scene_id"], image_bytes)
            if not row["quarantined"]:
                by_family[family].add(str(row["answer"]))
        except Exception as exc:
            failures.append(f"row {row_number} verification raised {exc!r}")
    for family, classes in by_family.items():
        if len(classes) < 2:
            failures.append(f"{family} lacks two off-quarantine answer classes: {sorted(classes)}")
    if len(images) != len(scenes):
        failures.append("scene/image index is inconsistent")
    return failures


def main() -> None:
    parser = __import__("argparse").ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    failures = verify(args.dataset)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print(f"OK: verified {len([line for line in args.dataset.joinpath('manifest.jsonl').read_text().splitlines() if line.strip()])} records")


if __name__ == "__main__":
    main()
