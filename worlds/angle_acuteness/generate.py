"""Generate the angle-acuteness dataset (design §1; mirrors point_in_polygon/relative_length).

Design (SSOT §7.4–§7.6, §8.6):

- Each scene is rendered ONCE and emitted once per prompt family, sharing the
  image. Framing variance is therefore measured on identical images (paired
  design), not confounded with scene sampling.
- Every rendered image is certified by the pixel oracle (``oracle.py``) before its
  records enter the manifest: the inner builders need not look at
  images, so the manifest must be self-certifying. Generation fails loudly on the
  first oracle failure.
- ``--sweep`` generates fixed-margin sequences (renderer.MARGIN_SWEEP, degrees) for
  continuity analysis instead of bin-balanced sampling.

Run as a module from the repo root:

    uv run python -m worlds.angle_acuteness.generate --n 200 --split smoke
    uv run python -m worlds.angle_acuteness.generate --n 20 --sweep --split sweep

Rendering is CPU-only and deterministic by seed. Rendered images stay under this
instance's data/rendered/ and are git-ignored; the manifest is the tracked
ground-truth ledger.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from tqdm import tqdm

from question_foundry.manifest import ExampleRecord, write_manifest

from .oracle import verify_example
from .prompts import PROMPT_FAMILIES, PROMPTS_VERSION, candidates_for, correct_answer
from .renderer import (
    MARGIN_BINS,
    MARGIN_SWEEP,
    RENDERER_VERSION,
    bin_for_m,
    render,
    sample_scene,
    scene_targets,
)

WORLD = "angle_acuteness"
INSTANCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = INSTANCE_DIR.parents[1]
ALL_PROMPTS = dict(PROMPT_FAMILIES)


def _manifest_image_path(image_path: Path) -> str:
    try:
        return str(image_path.relative_to(REPO_ROOT))
    except ValueError:
        return str(image_path)


def _parse_families(spec: str) -> list[str]:
    families = list(PROMPT_FAMILIES) if spec == "all" else [f.strip() for f in spec.split(",")]
    unknown = [f for f in families if f not in ALL_PROMPTS]
    if unknown:
        raise SystemExit(f"unknown prompt families {unknown}; known: {sorted(ALL_PROMPTS)}")
    return families


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=10_000, help="scenes (x sweep points if --sweep)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="pilot")
    ap.add_argument(
        "--families",
        default="all",
        help="comma-separated prompt families; 'all' = the standard five. "
        "Records = scenes x families, all sharing one image per scene.",
    )
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="generate --n scenes at each fixed margin in renderer.MARGIN_SWEEP "
        "(continuity analysis, SSOT §7.4) instead of bin-balanced sampling",
    )
    ap.add_argument("--out-dir", default=None, help="images dir (default: data/rendered/<split>)")
    ap.add_argument("--manifest", default=None, help="manifest (default: data/manifests/)")
    ap.add_argument(
        "--no-verify",
        action="store_true",
        help="skip pixel-oracle certification (SSOT §7.6). Unverified manifests must "
        "not feed evals; use only for debugging the renderer itself.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else INSTANCE_DIR / "data/rendered" / args.split
    manifest = (
        Path(args.manifest)
        if args.manifest
        else INSTANCE_DIR / "data/manifests" / f"{args.split}.jsonl"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    families = _parse_families(args.families)
    rng = random.Random(args.seed)
    bins = list(MARGIN_BINS)
    if args.sweep:
        plan = [(float(m), bin_for_m(m)) for m in MARGIN_SWEEP for _ in range(args.n)]
    else:
        plan = [(None, bins[i % len(bins)]) for i in range(args.n)]

    if args.no_verify:
        print("WARNING: --no-verify — this manifest is NOT oracle-certified (SSOT §7.6)")

    records: list[ExampleRecord] = []
    confidences: dict[str, int] = {}
    for i, (m, margin_bin) in enumerate(tqdm(plan, desc=f"rendering {args.split}")):
        scene = sample_scene(rng, margin_bin=margin_bin) if m is None else sample_scene(rng, m=m)
        image_path = out_dir / f"{WORLD}_{args.split}_{i:06d}.png"
        image = render(scene)
        image.save(image_path)

        oracle_extra: dict = {}
        if not args.no_verify:
            report = verify_example(image, scene)
            if not report.ok:
                raise RuntimeError(
                    f"pixel oracle rejected scene {i} ({image_path}): {report.failures}. "
                    "The rendered image does not match the analytic ground truth — do "
                    "not relax oracle tolerances without an SSOT §7.6 change."
                )
            confidences[report.confidence] = confidences.get(report.confidence, 0) + 1
            oracle_extra = {"oracle": report.to_extra()}

        for family in families:
            records.append(
                ExampleRecord(
                    example_id=f"{WORLD}_{args.split}_{i:06d}_{family}",
                    image_path=_manifest_image_path(image_path),
                    split=args.split,
                    seed=args.seed,
                    world=WORLD,
                    scene=scene.to_dict(),
                    targets={
                        **scene_targets(scene, margin_bin),
                        "answer": correct_answer(family, scene.signed_margin),
                        "candidates": list(candidates_for(family)),
                    },
                    prompt_family=family,
                    prompt_text=ALL_PROMPTS[family],
                    renderer_version=RENDERER_VERSION,
                    extra={"prompts_version": PROMPTS_VERSION, "scene_index": i, **oracle_extra},
                )
            )

    write_manifest(records, manifest)
    verified = "UNVERIFIED" if args.no_verify else f"oracle-certified ({confidences})"
    print(
        f"wrote {len(records)} records ({len(plan)} scenes x {len(families)} families, "
        f"{verified}) -> {manifest}; images -> {out_dir}/"
    )


if __name__ == "__main__":
    main()
