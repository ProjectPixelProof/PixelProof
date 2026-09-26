"""Re-render and re-invert an exact submitted dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageChops

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


def verify_dataset(dataset: Path, *, write_report: bool = True) -> dict:
    dataset = dataset.resolve()
    failures: list[str] = []
    try:
        raw_rows = [
            line for line in (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError as exc:
        return {"status": "failed", "failures": [str(exc)]}

    rows: list[dict] = []
    for number, line in enumerate(raw_rows, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append(f"manifest line {number}: {exc}")
            continue
        missing = REQUIRED_KEYS - set(row)
        if missing:
            failures.append(f"manifest line {number} missing {sorted(missing)}")
        else:
            rows.append(row)

    scenes: dict[str, dict] = {}
    images: dict[str, Image.Image] = {}
    by_family: dict[str, set[str]] = {family: set() for family in prompts.PROMPT_FAMILIES}
    family_scene_ids: dict[str, set[str]] = {family: set() for family in prompts.PROMPT_FAMILIES}
    for row in rows:
        family = row.get("prompt_family")
        if family not in prompts.PROMPT_FAMILIES:
            failures.append(f"{row.get('example_id')}: unknown prompt family {family!r}")
            continue
        image_path = (dataset / str(row["image_path"])).resolve()
        try:
            image_path.relative_to(dataset)
            image = Image.open(image_path).convert("RGB")
        except (OSError, ValueError) as exc:
            failures.append(f"{row.get('example_id')}: image unreadable: {exc}")
            continue
        scene_id = str(row["scene_id"])
        if scene_id in images and ImageChops.difference(images[scene_id], image).getbbox() is not None:
            failures.append(f"{scene_id}: prompt rows do not share one image")
        images[scene_id] = image
        scenes.setdefault(scene_id, row["scene"])
        try:
            rerendered = renderer.render(row["scene"]).convert("RGB")
            if ImageChops.difference(image, rerendered).getbbox() is not None:
                failures.append(f"{row['example_id']}: rerender mismatch")
            decision = renderer.analytic_gold(row["scene"])
            if decision != row["decision"]:
                failures.append(f"{row['example_id']}: analytic decision mismatch")
            if abs(float(renderer.margin(row["scene"])) - float(row["margin"])) > 1e-9:
                failures.append(f"{row['example_id']}: margin mismatch")
            quarantined = bool(renderer.is_quarantined(row["scene"]))
            if quarantined != bool(row["quarantined"]):
                failures.append(f"{row['example_id']}: quarantine mismatch")
            if prompts.correct_answer(family, decision) != row["answer"]:
                failures.append(f"{row['example_id']}: prompt answer mismatch")
            if tuple(row["candidates"]) != prompts.candidates_for(family):
                failures.append(f"{row['example_id']}: answer candidates mismatch")
            if oracle.decision_from_image(image) != decision:
                failures.append(f"{row['example_id']}: pixel oracle mismatch")
            if not quarantined:
                by_family[family].add(str(row["answer"]))
            family_scene_ids[family].add(scene_id)
        except Exception as exc:
            failures.append(f"{row['example_id']}: verification raised {exc!r}")

    for family in prompts.PROMPT_FAMILIES:
        if len(by_family[family]) < 2:
            failures.append(f"{family}: fewer than two off-quarantine answer classes")
        if family_scene_ids[family] != set(scenes):
            failures.append(f"{family}: scene coverage differs from the gallery")

    symmetries_checked = 0
    for scene_id, scene in list(scenes.items())[:12]:
        original = renderer.render(scene).convert("RGB")
        gold = renderer.analytic_gold(scene)
        for name, twin in renderer.latent_symmetries(scene):
            symmetries_checked += 1
            if ImageChops.difference(original, renderer.render(twin).convert("RGB")).getbbox():
                failures.append(f"{scene_id}/{name}: symmetry changed pixels")
            if renderer.analytic_gold(twin) != gold:
                failures.append(f"{scene_id}/{name}: symmetry changed gold")
    if symmetries_checked == 0:
        failures.append("no latent symmetry was exercised")

    report = {
        "status": "passed" if not failures else "failed",
        "record_count": len(rows),
        "scene_count": len(scenes),
        "prompt_families": sorted(prompts.PROMPT_FAMILIES),
        "symmetries_checked": symmetries_checked,
        "failures": failures,
    }
    if write_report:
        (dataset / "self_check.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    report = verify_dataset(args.dataset)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
