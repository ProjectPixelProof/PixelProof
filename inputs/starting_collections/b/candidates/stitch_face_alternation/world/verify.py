"""Re-certify a generated ``stitch_face_alternation`` dataset.

    python world/verify.py --dataset <directory>

Exits zero only when, for every manifest row: the stored PNG exists and is
byte-identical to a fresh render of the stored scene; the analytic decision,
margin, quarantine flag, prompt question, prompt answer and answer-candidate set
are exactly the recomputed consequences of that scene; the scene-free pixel
oracle recovers the same decision (abstention tolerated only on quarantined
rows); and the declared latent alias is pixel-identical and gold-preserving.
Also checks per-family class balance and control/counterfactual coverage.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image  # noqa: E402

import oracle  # noqa: E402
import prompts  # noqa: E402
import renderer  # noqa: E402

REQUIRED_KEYS = (
    "example_id", "scene_id", "seed", "scene", "image_path", "prompt_family",
    "question", "decision", "answer", "candidates", "margin", "quarantined",
)


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def verify(dataset: Path) -> list[str]:
    failures: list[str] = []
    manifest = dataset / "manifest.jsonl"
    if not manifest.exists():
        return [f"missing manifest {manifest}"]

    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return ["manifest is empty"]

    rendered: dict[str, bytes] = {}
    per_family: dict[str, set[str]] = {}
    scene_decisions: dict[str, str] = {}
    variants: set[str] = set()

    for row in rows:
        rid = row.get("example_id", "?")
        missing = [k for k in REQUIRED_KEYS if k not in row]
        if missing:
            failures.append(f"{rid}: missing manifest keys {missing}")
            continue

        scene = row["scene"]
        family = row["prompt_family"]
        decision = renderer.analytic_gold(scene)
        m = renderer.margin(scene)
        q = renderer.is_quarantined(scene)
        variants.add(scene.get("variant", "?"))

        if row["decision"] != decision:
            failures.append(f"{rid}: stored decision {row['decision']!r} != analytic {decision!r}")
        if abs(float(row["margin"]) - m) > 1e-3:
            failures.append(f"{rid}: stored margin {row['margin']} != analytic {m:.4f}")
        if bool(row["quarantined"]) != q:
            failures.append(f"{rid}: stored quarantine {row['quarantined']} != analytic {q}")
        if family not in prompts.PROMPT_FAMILIES:
            failures.append(f"{rid}: unknown prompt family {family!r}")
            continue
        if row["question"] != prompts.PROMPT_FAMILIES[family]:
            failures.append(f"{rid}: question text does not match prompts.PROMPT_FAMILIES")
        exp_answer = prompts.correct_answer(family, decision)
        if row["answer"] != exp_answer:
            failures.append(f"{rid}: stored answer {row['answer']!r} != {exp_answer!r}")
        exp_cands = list(prompts.candidates_for(family))
        if list(row["candidates"]) != exp_cands:
            failures.append(f"{rid}: candidates {row['candidates']} != {exp_cands}")
        if row["answer"] not in exp_cands:
            failures.append(f"{rid}: answer not inside its candidate set")
        if "canonical_prior_answer" in row and row["canonical_prior_answer"] != scene.get(
            "canonical_prior_answer"
        ):
            failures.append(f"{rid}: stored canonical prior does not match the scene")

        img_path = dataset / row["image_path"]
        if not img_path.exists():
            failures.append(f"{rid}: missing image {row['image_path']}")
            continue

        sid = row["scene_id"]
        if sid not in rendered:
            rendered[sid] = _png_bytes(renderer.render(scene))
        if img_path.read_bytes() != rendered[sid]:
            failures.append(f"{rid}: stored raster differs from a fresh render of the stored scene")
            continue

        with Image.open(img_path) as im:
            got = oracle.decision_from_image(im.convert("RGB"))
        if got != decision and not (q and got == oracle.ABSTAIN):
            failures.append(
                f"{rid}: pixel oracle {got!r} != analytic {decision!r} (margin {m:.2f}, q={q})"
            )

        if sid in scene_decisions and scene_decisions[sid] != decision:
            failures.append(f"{sid}: inconsistent decision across prompt families")
        scene_decisions[sid] = decision
        if not q:
            per_family.setdefault(family, set()).add(row["answer"])

        for name, alias in renderer.latent_symmetries(scene):
            if renderer.analytic_gold(alias) != decision:
                failures.append(f"{rid}: latent alias {name} changed the analytic decision")
            if _png_bytes(renderer.render(alias)) != rendered[sid]:
                failures.append(f"{rid}: latent alias {name} is not pixel-identical")

    for family in prompts.PROMPT_FAMILIES:
        classes = per_family.get(family, set())
        if len(classes) < 2:
            failures.append(
                f"class balance: family {family} has {len(classes)} off-quarantine "
                "classes (need >= 2)"
            )
    need = min(4, max(2, len(scene_decisions)))
    if len(set(scene_decisions.values())) < need:
        failures.append(
            f"only {len(set(scene_decisions.values()))} distinct decisions in dataset "
            f"(need >= {need})"
        )
    if not {"control", "counterfactual"} <= variants:
        failures.append(f"dataset lacks matched control/counterfactual coverage: {sorted(variants)}")

    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    args = ap.parse_args()
    failures = verify(args.dataset)
    if failures:
        for f in failures[:60]:
            print(f"FAIL {f}")
        print(f"FAIL: {len(failures)} verification failure(s)")
        raise SystemExit(1)
    print(f"OK: {args.dataset} verified")


if __name__ == "__main__":
    main()
