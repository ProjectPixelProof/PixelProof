"""Re-certify an existing dataset against its manifest and pixel oracle.

Usage:
    python world/verify.py --dataset <dir>
exit zero only when stored images, analytic decisions, prompt answers, answer
candidate sets, margins, quarantine flags, and the scene-free pixel oracle agree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import renderer
import prompts
from oracle import decision_from_image

from PIL import Image, ImageChops


def verify(dataset: Path) -> list[str]:
    failures = []
    manifest = dataset / "manifest.jsonl"
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return ["empty manifest"]
    for index, row in enumerate(rows):
        ex = row["example_id"]
        scene = row["scene"]
        try:
            expected = renderer.render(scene).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{ex}: render raised {exc!r}")
            continue
        image_path = dataset / row["image_path"]
        if not image_path.is_file():
            failures.append(f"{ex}: missing image {row['image_path']}")
            continue
        stored = Image.open(image_path).convert("RGB")
        if ImageChops.difference(stored, expected).getbbox() is not None:
            failures.append(f"{ex}: stored image differs from rerender")
        decision = renderer.analytic_gold(scene)
        if decision != row["decision"]:
            failures.append(f"{ex}: analytic decision mismatch")
        if abs(float(renderer.margin(scene)) - float(row["margin"])) > 1e-9:
            failures.append(f"{ex}: margin mismatch")
        if renderer.is_quarantined(scene) != bool(row["quarantined"]):
            failures.append(f"{ex}: quarantine mismatch")
        family = row["prompt_family"]
        if prompts.correct_answer(family, decision) != row["answer"]:
            failures.append(f"{ex}: prompt answer mismatch")
        if tuple(row["candidates"]) != tuple(prompts.candidates_for(family)):
            failures.append(f"{ex}: candidates mismatch")
        if decision_from_image(stored) != decision:
            failures.append(f"{ex}: pixel oracle disagrees with analytic gold")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    args = parser.parse_args()
    failures = verify(args.dataset)
    if failures:
        for failure in failures[:50]:
            print("FAIL:", failure)
        print(f"{len(failures)} failure(s)")
        raise SystemExit(1)
    print("OK: dataset verified against rerender, gold, and pixel oracle")


if __name__ == "__main__":
    main()
