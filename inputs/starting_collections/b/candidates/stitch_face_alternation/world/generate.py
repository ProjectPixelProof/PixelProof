"""Dataset generator for ``stitch_face_alternation``.

    python world/generate.py --out <directory> --n <scene-count> --seed <seed>

Writes ``<out>/gallery/scene_XXXX.png``, ``<out>/gallery/index.html``,
``<out>/manifest.jsonl`` (one row per scene x prompt family) and
``<out>/self_check.json``.  Every scene is certified against the independent
pixel-only oracle before it is written.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image  # noqa: E402

import oracle  # noqa: E402
import prompts  # noqa: E402
import renderer  # noqa: E402


def scene_seed(base: int, index: int) -> int:
    return int(base) * 100003 + index * 7919 + 13


def build(out: Path, n: int, seed: int) -> dict:
    gallery = out / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    stats: dict = {
        "decisions": {},
        "answers_by_variant": {},
        "variants": {},
        "margin_bins": {},
        "first_faces": {},
        "tag_side": {},
        "prior_conflicts": 0,
        "quarantined": 0,
        "oracle_agree": 0,
        "oracle_abstain_quarantined": 0,
        "prompt_answer_counts": {},
    }

    for i in range(n):
        sid = f"scene_{i:04d}"
        sc_seed = scene_seed(seed, i)
        scene = renderer.sample_scene(sc_seed)
        image = renderer.render(scene)
        decision = renderer.analytic_gold(scene)
        m = renderer.margin(scene)
        q = renderer.is_quarantined(scene)

        rel = f"gallery/{sid}.png"
        image.save(out / rel, format="PNG", optimize=False)
        with Image.open(out / rel) as stored:
            got = oracle.decision_from_image(stored.convert("RGB"))
        if got == decision:
            stats["oracle_agree"] += 1
        elif q and got == oracle.ABSTAIN:
            stats["oracle_abstain_quarantined"] += 1
        else:
            raise SystemExit(
                f"FAIL {sid}: pixel oracle said {got!r}, analytic gold {decision!r} "
                f"(margin {m:.2f}, quarantined={q})"
            )

        mb = renderer.margin_bin(scene)
        pattern, first_face = decision.split("|")
        variant = scene["variant"]
        prior = scene["canonical_prior_answer"]
        stats["decisions"][decision] = stats["decisions"].get(decision, 0) + 1
        stats["variants"][variant] = stats["variants"].get(variant, 0) + 1
        stats["margin_bins"][mb] = stats["margin_bins"].get(mb, 0) + 1
        stats["first_faces"][first_face] = stats["first_faces"].get(first_face, 0) + 1
        tag_side = renderer._canon(scene)[2]
        stats["tag_side"][tag_side] = stats["tag_side"].get(tag_side, 0) + 1
        stats["answers_by_variant"].setdefault(variant, {})
        stats["answers_by_variant"][variant][pattern] = (
            stats["answers_by_variant"][variant].get(pattern, 0) + 1
        )
        stats["prior_conflicts"] += int(prior != pattern)
        stats["quarantined"] += int(q)

        for family in prompts.PROMPT_FAMILIES:
            answer = prompts.correct_answer(family, decision)
            cands = list(prompts.candidates_for(family))
            rows.append(
                {
                    "example_id": f"{sid}__{family}",
                    "scene_id": sid,
                    "seed": sc_seed,
                    "scene": scene,
                    "image_path": rel,
                    "prompt_family": family,
                    "question": prompts.PROMPT_FAMILIES[family],
                    "decision": decision,
                    "answer": answer,
                    "candidates": cands,
                    "margin": float(m),
                    "quarantined": bool(q),
                    "margin_bin": mb,
                    "variant": variant,
                    "canonical_prior_answer": prior,
                    "prior_conflict": bool(prior != pattern),
                }
            )
            ck = f"{family}:{answer}"
            stats["prompt_answer_counts"][ck] = stats["prompt_answer_counts"].get(ck, 0) + 1

    with (out / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    stats["n_scenes"] = n
    stats["n_rows"] = len(rows)
    stats["seed"] = seed
    stats["renderer_version"] = renderer.RENDERER_VERSION
    stats["oracle_version"] = oracle.ORACLE_VERSION
    stats["prompts_version"] = prompts.PROMPTS_VERSION
    (out / "self_check.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")

    _write_index(gallery, rows)
    return stats


def _write_index(gallery: Path, rows: list[dict]) -> None:
    seen: dict[str, dict] = {}
    for r in rows:
        seen.setdefault(r["scene_id"], r)
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>stitch_face_alternation gallery</title>",
        "<style>body{font:13px sans-serif}figure{display:inline-block;margin:6px}"
        "img{width:224px;border:1px solid #999}figcaption{width:224px}</style>",
        "<h1>stitch_face_alternation</h1>",
        "<p>decision = whether the front/behind sequence alternates along the "
        "walk from the tagged thread end, and the first crossing's face.</p>",
    ]
    for sid in sorted(seen):
        r = seen[sid]
        cap = (
            f"{sid} &middot; {html.escape(r['decision'])} &middot; {html.escape(r['variant'])}"
            f"<br>prior={html.escape(str(r['canonical_prior_answer']))}"
            f" margin={r['margin']:.2f} &middot; {html.escape(r['margin_bin'])}"
            + (" &middot; QUARANTINED" if r["quarantined"] else "")
        )
        parts.append(
            f"<figure><img src='{sid}.png' alt='{sid}'><figcaption>{cap}</figcaption></figure>"
        )
    (gallery / "index.html").write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    stats = build(args.out, args.n, args.seed)
    print(
        json.dumps(
            {
                k: stats[k]
                for k in (
                    "n_scenes",
                    "n_rows",
                    "variants",
                    "margin_bins",
                    "first_faces",
                    "prior_conflicts",
                    "quarantined",
                    "oracle_agree",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
