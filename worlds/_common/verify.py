"""Shared re-certification driver (SSOT §7.6).

``make_verify_main(world, renderer, oracle)`` returns a ``main()`` that re-runs
the pixel oracle on every unique image referenced by a manifest against the
analytic scene stored in its records — the transfer / CI gate independent of
generation. ``--repair`` re-renders a failing image from its manifest geometry
(the manifest is the ground-truth ledger, so a failing image is a corrupted
artifact to restore, never a reason to touch a tolerance). Exits non-zero if any
image still fails.

Required world interface: ``renderer.render(scene)`` and a
``renderer.scene_from_dict(dict) -> Scene`` (or ``renderer.Scene(**dict)``), plus
``oracle.verify_example(image, scene)`` and ``oracle.UnsupportedSceneError``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from question_foundry.manifest import read_manifest


def _scene_ctor(renderer):
    if hasattr(renderer, "scene_from_dict"):
        return renderer.scene_from_dict
    return lambda d: renderer.Scene(**d)


def make_verify_main(world: str, renderer, oracle):
    from_dict = _scene_ctor(renderer)

    def main() -> None:
        ap = argparse.ArgumentParser(description=f"re-certify a rendered {world} dataset")
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

        scenes: dict[str, dict] = {}
        for rec in read_manifest(args.manifest):
            scenes.setdefault(rec.image_path, rec.scene)
        items = list(scenes.items())[: args.limit]

        from PIL import Image

        n_ok = 0
        failures: list[tuple[str, list[str]]] = []
        repaired: list[str] = []
        unsupported: list[str] = []
        for image_path, scene_dict in tqdm(items, desc="oracle verify"):
            scene = from_dict(scene_dict)
            full_path = Path(args.image_root) / image_path
            image = Image.open(full_path)
            try:
                report = oracle.verify_example(image, scene)
                if not report.ok and args.repair:
                    renderer.render(scene).save(full_path)
                    report = oracle.verify_example(Image.open(full_path), scene)
                    if report.ok:
                        repaired.append(image_path)
            except oracle.UnsupportedSceneError as e:
                unsupported.append(f"{image_path}: {e}")
                continue
            if report.ok:
                n_ok += 1
            else:
                failures.append((image_path, report.failures))

        print(
            f"{len(items)} images: {n_ok} ok, {len(failures)} FAILED, "
            f"{len(repaired)} repaired from manifest geometry, "
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

    return main
