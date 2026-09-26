"""Re-certify a rendered dataset against its manifest with the pixel oracle (SSOT §7.6).

Checks every unique image referenced by the manifest against the analytic scene
stored in its records — use after transferring datasets between machines, after
renderer changes, or on any manifest whose provenance is unclear:

    uv run python -m worlds.two_circles.verify data/manifests/smoke.jsonl
    uv run python -m worlds.two_circles.verify data/manifests/smoke.jsonl --repair

``--repair`` re-renders failing images from their manifest geometry and
re-certifies them. The manifest is the ground-truth ledger, so a failing image
is a corrupted artifact to restore — never a reason to touch a tolerance
(SSOT §7.6 rule 2). Default is a read-only dry run. Exits non-zero if any
image fails (or, with --repair, still fails after restoration).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from question_foundry.manifest import read_manifest

from .oracle import UnsupportedSceneError, verify_example
from .renderer import TwoCircleScene, render


def _scene_from_dict(scene_dict: dict) -> TwoCircleScene:
    return TwoCircleScene(
        **{**scene_dict, "c1": tuple(scene_dict["c1"]), "c2": tuple(scene_dict["c2"])}
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest")
    ap.add_argument(
        "--image-root",
        default=".",
        help="root that relative image_paths resolve against (default: cwd = repo root)",
    )
    ap.add_argument("--limit", type=int, default=None, help="verify at most N unique images")
    ap.add_argument(
        "--repair",
        action="store_true",
        help="re-render failing images from manifest geometry, then re-certify",
    )
    args = ap.parse_args()

    # One scene per unique image (records share images across prompt families).
    scenes: dict[str, dict] = {}
    for rec in read_manifest(args.manifest):
        scenes.setdefault(rec.image_path, rec.scene)
    items = list(scenes.items())[: args.limit]

    from PIL import Image

    n_ok = n_low = 0
    failures: list[tuple[str, list[str]]] = []
    repaired: list[str] = []
    unsupported: list[str] = []
    for image_path, scene_dict in tqdm(items, desc="oracle verify"):
        scene = _scene_from_dict(scene_dict)
        full_path = Path(args.image_root) / image_path
        image = Image.open(full_path)
        try:
            report = verify_example(image, scene)
            if not report.ok and args.repair:
                render(scene).save(full_path)
                report = verify_example(Image.open(full_path), scene)
                if report.ok:
                    repaired.append(image_path)
        except UnsupportedSceneError as e:
            unsupported.append(f"{image_path}: {e}")
            continue
        if report.ok:
            n_ok += 1
            n_low += report.confidence == "low"
        else:
            failures.append((image_path, report.failures))

    print(
        f"{len(items)} images: {n_ok} ok ({n_low} low-confidence), "
        f"{len(failures)} FAILED, {len(repaired)} repaired from manifest geometry, "
        f"{len(unsupported)} unsupported style"
    )
    for path in repaired[:10]:
        print(f"  REPAIRED {path}")
    for path, fails in failures[:10]:
        print(f"  FAIL {path}: {fails}")
    for line in unsupported[:5]:
        print(f"  SKIP {line}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
