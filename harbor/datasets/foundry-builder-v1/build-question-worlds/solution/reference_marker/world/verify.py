from __future__ import annotations

import argparse
import json
from pathlib import Path

import oracle
import prompts
import renderer
from PIL import Image, ImageChops


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()

    failures = []
    for line in (args.dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image = Image.open(args.dataset / row["image_path"]).convert("RGB")
        expected = renderer.render(row["scene"]).convert("RGB")
        if ImageChops.difference(image, expected).getbbox() is not None:
            failures.append(f"{row['example_id']}: raster mismatch")
        decision = renderer.analytic_gold(row["scene"])
        if decision != row["decision"]:
            failures.append(f"{row['example_id']}: analytic decision mismatch")
        if oracle.decision_from_image(image) != decision:
            failures.append(f"{row['example_id']}: pixel decision mismatch")
        answer = prompts.correct_answer(row["prompt_family"], decision)
        if answer != row["answer"]:
            failures.append(f"{row['example_id']}: prompt answer mismatch")
    if failures:
        raise SystemExit("\n".join(failures[:20]))
    print("candidate verification passed")


if __name__ == "__main__":
    main()
