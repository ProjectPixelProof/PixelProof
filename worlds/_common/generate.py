"""Shared dataset-generation driver (SSOT §7.4–§7.6, §8.6).

``make_generate_main(world, renderer, prompts, oracle)`` returns a ``main()``
identical in behaviour to the hand-written ``generate.py`` of the pre-library
worlds: each scene is rendered ONCE and emitted once per prompt family (paired
design — framing variance is measured on identical images), every image is
oracle-certified before its records enter the manifest, and ``--sweep`` renders
fixed-margin sequences for continuity analysis.

Required world interface:

- ``renderer``: ``MARGIN_BINS``, ``MARGIN_SWEEP``, ``RENDERER_VERSION``,
  ``bin_for_m(m)``, ``render(scene)``, ``sample_scene(rng, margin_bin=, m=)``,
  ``scene_targets(scene, bin)``, and a ``Scene`` whose ``.to_dict()`` round-trips.
  Optional ``SAMPLED_BINS`` narrows bin-balanced generation (e.g. one-sided worlds
  that quarantine a boundary band unreachable by the sampler).
- Each scene exposes ``.decision_value`` — the quantity ``prompts.correct_answer``
  keys off (a signed margin, a count, ...).
- ``prompts``: ``PROMPT_FAMILIES``, ``PROMPTS_VERSION``, ``correct_answer``,
  ``candidates_for``.
- ``oracle``: ``verify_example(image, scene) -> report`` with ``.ok``,
  ``.confidence``, ``.failures``, ``.to_extra()``.
"""

from __future__ import annotations

import argparse
import random
import tomllib
from pathlib import Path

from tqdm import tqdm

from question_foundry.manifest import ExampleRecord, write_manifest


def declared_excluded_families(instance_dir: Path) -> list[str]:
    """The world's ``[gold] smoke_excluded_families`` declaration (gate 4, SSOT §7.14)."""
    toml_path = Path(instance_dir) / "world.toml"
    if not toml_path.exists():
        return []
    cfg = tomllib.loads(toml_path.read_text())
    return list(cfg.get("gold", {}).get("smoke_excluded_families", []))


def resolve_families(spec: str, prompt_families, excluded: list[str]) -> list[str]:
    """Resolve a ``--families`` spec against the declared exclusions.

    ``"all"`` means every family MINUS the world's declared
    ``smoke_excluded_families`` — so a naive default regeneration reproduces the
    tracked (gated) manifest instead of silently resurrecting families gate 4
    excluded. An explicit comma-list may still name an excluded family (that is
    a deliberate operator choice, e.g. emitting a dedicated lexical split).
    """
    all_prompts = dict(prompt_families)
    if spec == "all":
        return [f for f in all_prompts if f not in set(excluded)]
    families = [f.strip() for f in spec.split(",")]
    unknown = [f for f in families if f not in all_prompts]
    if unknown:
        raise SystemExit(f"unknown prompt families {unknown}; known: {sorted(all_prompts)}")
    return families


def make_generate_main(world: str, renderer, prompts, oracle):
    instance_dir = Path(renderer.__file__).resolve().parent
    repo_root = instance_dir.parents[1]
    all_prompts = dict(prompts.PROMPT_FAMILIES)
    excluded_families = declared_excluded_families(instance_dir)

    def _manifest_image_path(image_path: Path) -> str:
        try:
            return str(image_path.relative_to(repo_root))
        except ValueError:
            return str(image_path)

    def _parse_families(spec: str) -> list[str]:
        return resolve_families(spec, prompts.PROMPT_FAMILIES, excluded_families)

    def main() -> None:
        ap = argparse.ArgumentParser(description=f"generate the {world} dataset")
        ap.add_argument("--n", type=int, default=10_000, help="scenes (x sweep points if --sweep)")
        ap.add_argument("--seed", type=int, default=0)
        ap.add_argument("--split", default="pilot")
        ap.add_argument(
            "--families",
            default="all",
            help="comma-separated prompt families; 'all' = every family. "
            "Records = scenes x families, all sharing one image per scene.",
        )
        ap.add_argument(
            "--sweep",
            action="store_true",
            help="generate --n scenes at each fixed margin in renderer.MARGIN_SWEEP "
            "(continuity analysis, SSOT §7.4) instead of bin-balanced sampling",
        )
        ap.add_argument(
            "--out-dir", default=None, help="images dir (default: data/rendered/<split>)"
        )
        ap.add_argument("--manifest", default=None, help="manifest (default: data/manifests/)")
        ap.add_argument(
            "--no-verify",
            action="store_true",
            help="skip pixel-oracle certification (SSOT §7.6). Unverified manifests must "
            "not feed evals; use only for debugging the renderer itself.",
        )
        args = ap.parse_args()

        out_dir = (
            Path(args.out_dir) if args.out_dir else instance_dir / "data/rendered" / args.split
        )
        manifest = (
            Path(args.manifest)
            if args.manifest
            else instance_dir / "data/manifests" / f"{args.split}.jsonl"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        families = _parse_families(args.families)
        rng = random.Random(args.seed)
        bins = list(getattr(renderer, "SAMPLED_BINS", None) or renderer.MARGIN_BINS)
        if args.sweep:
            plan = [
                (float(m), renderer.bin_for_m(m))
                for m in renderer.MARGIN_SWEEP
                for _ in range(args.n)
            ]
        else:
            plan = [(None, bins[i % len(bins)]) for i in range(args.n)]

        if args.no_verify:
            print("WARNING: --no-verify — this manifest is NOT oracle-certified (SSOT §7.6)")

        records: list[ExampleRecord] = []
        confidences: dict[str, int] = {}
        for i, (m, margin_bin) in enumerate(tqdm(plan, desc=f"rendering {args.split}")):
            scene = (
                renderer.sample_scene(rng, margin_bin=margin_bin)
                if m is None
                else renderer.sample_scene(rng, m=m)
            )
            image_path = out_dir / f"{world}_{args.split}_{i:06d}.png"
            image = renderer.render(scene)
            image.save(image_path)

            oracle_extra: dict = {}
            if not args.no_verify:
                report = oracle.verify_example(image, scene)
                if not report.ok:
                    raise RuntimeError(
                        f"pixel oracle rejected scene {i} ({image_path}): {report.failures}. "
                        "The rendered image does not match the analytic ground truth — do "
                        "not relax oracle tolerances without an SSOT §7.6 change."
                    )
                confidences[report.confidence] = confidences.get(report.confidence, 0) + 1
                oracle_extra = {"oracle": report.to_extra()}

            dv = scene.decision_value
            for family in families:
                records.append(
                    ExampleRecord(
                        example_id=f"{world}_{args.split}_{i:06d}_{family}",
                        image_path=_manifest_image_path(image_path),
                        split=args.split,
                        seed=args.seed,
                        world=world,
                        scene=scene.to_dict(),
                        targets={
                            **renderer.scene_targets(scene, margin_bin),
                            "answer": prompts.correct_answer(family, dv),
                            "candidates": list(prompts.candidates_for(family)),
                        },
                        prompt_family=family,
                        prompt_text=all_prompts[family],
                        renderer_version=renderer.RENDERER_VERSION,
                        extra={
                            "prompts_version": prompts.PROMPTS_VERSION,
                            "scene_index": i,
                            **oracle_extra,
                        },
                    )
                )

        write_manifest(records, manifest)
        verified = "UNVERIFIED" if args.no_verify else f"oracle-certified ({confidences})"
        print(
            f"wrote {len(records)} records ({len(plan)} scenes x {len(families)} families, "
            f"{verified}) -> {manifest}; images -> {out_dir}/"
        )

    return main
