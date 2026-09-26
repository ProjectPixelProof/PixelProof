"""Run semantic candidate checks in a bounded verifier subprocess."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageChops

REQUIRED_MANIFEST_KEYS = {
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


def _fail(failures: dict[str, list[str]], gate: str, message: str) -> None:
    failures[gate].append(message)


def _safe_relative(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    path.relative_to(root.resolve())
    return path


def _reachable_scene_keys(source: Path, entrypoint: str) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    pending = [entrypoint]
    visited = set()
    keys = set()
    while pending:
        name = pending.pop()
        if name in visited or name not in functions:
            continue
        visited.add(name)
        function = functions[name]
        positional = [*function.args.posonlyargs, *function.args.args]
        scene_names = {positional[0].arg} if positional else set()
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id in scene_names
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                keys.add(node.slice.value)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in scene_names
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                keys.add(node.args[0].value)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in functions
            ):
                pending.append(node.func.id)
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    candidate = args.candidate.resolve()
    dataset = args.dataset.resolve()
    world = candidate / "world"
    sys.path.insert(0, str(world))
    renderer = importlib.import_module("renderer")
    oracle = importlib.import_module("oracle")
    prompts = importlib.import_module("prompts")
    metadata = json.loads((candidate / "candidate.json").read_text(encoding="utf-8"))
    candidate_schema = metadata.get("schema_version")

    failures: dict[str, list[str]] = defaultdict(list)
    rows = []
    for number, line in enumerate(
        (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail(failures, "contract", f"manifest line {number}: {exc}")
            continue
        missing = REQUIRED_MANIFEST_KEYS - set(row)
        if missing:
            _fail(failures, "contract", f"manifest line {number} missing {sorted(missing)}")
            continue
        rows.append(row)

    if not rows:
        _fail(failures, "contract", "generated manifest has no records")

    families = set(getattr(prompts, "PROMPT_FAMILIES", {}))
    if not families:
        _fail(failures, "contract", "PROMPT_FAMILIES is empty")
    by_family: dict[str, set[str]] = defaultdict(set)
    scenes: dict[str, dict] = {}
    images: dict[str, Image.Image] = {}
    pixel_decisions: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        family = row["prompt_family"]
        if family not in families:
            _fail(failures, "contract", f"undeclared prompt family {family!r}")
            continue
        try:
            image_path = _safe_relative(dataset, row["image_path"])
        except (ValueError, OSError):
            _fail(failures, "safe_artifact", f"unsafe image path {row['image_path']!r}")
            continue
        if not image_path.is_file() or image_path.suffix.lower() != ".png":
            _fail(failures, "render_fidelity", f"missing PNG {row['image_path']!r}")
            continue
        image = Image.open(image_path).convert("RGB")
        if image.width < 32 or image.height < 32:
            _fail(failures, "render_fidelity", f"image too small: {image.size}")
        images[row["scene_id"]] = image
        scene = row["scene"]
        scenes.setdefault(row["scene_id"], scene)

        try:
            expected_image = renderer.render(scene).convert("RGB")
            if ImageChops.difference(image, expected_image).getbbox() is not None:
                _fail(failures, "render_fidelity", f"{row['example_id']}: rerender mismatch")
        except Exception as exc:
            _fail(failures, "render_fidelity", f"{row['example_id']}: render raised {exc!r}")
            continue

        try:
            decision = renderer.analytic_gold(scene)
            if decision != row["decision"]:
                _fail(
                    failures,
                    "analytic_consistency",
                    f"{row['example_id']}: stored decision differs from analytic gold",
                )
            margin = float(renderer.margin(scene))
            if abs(margin - float(row["margin"])) > 1e-9:
                _fail(
                    failures,
                    "analytic_consistency",
                    f"{row['example_id']}: stored margin differs from recomputation",
                )
            quarantined = bool(renderer.is_quarantined(scene))
            if quarantined != row["quarantined"]:
                _fail(
                    failures,
                    "quarantine",
                    f"{row['example_id']}: quarantine flag differs from recomputation",
                )
            answer = prompts.correct_answer(family, decision)
            if answer != row["answer"]:
                _fail(
                    failures,
                    "analytic_consistency",
                    f"{row['example_id']}: stored answer differs from prompt gold",
                )
            if tuple(row["candidates"]) != tuple(prompts.candidates_for(family)):
                _fail(
                    failures,
                    "contract",
                    f"{row['example_id']}: candidate answer set differs from prompt API",
                )
            pixel_decision = oracle.decision_from_image(image)
            if pixel_decision != decision:
                _fail(
                    failures,
                    "sign_blind_oracle",
                    f"{row['example_id']}: image-only decision differs from analytic gold",
                )
            if not quarantined:
                by_family[family].add(answer)
            pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
            pixel_decisions[pixel_hash].add(str(decision))
        except Exception as exc:
            _fail(failures, "sign_blind_oracle", f"{row['example_id']}: probe raised {exc!r}")

    for family in sorted(families):
        classes = by_family.get(family, set())
        if len(classes) < 2:
            _fail(
                failures,
                "class_balance",
                f"{family}: only off-quarantine classes {sorted(classes)}",
            )

    symmetry_count = 0
    for scene_id, scene in list(scenes.items())[:12]:
        try:
            original_image = renderer.render(scene).convert("RGB")
            original_gold = renderer.analytic_gold(scene)
            for name, twin in renderer.latent_symmetries(scene):
                symmetry_count += 1
                twin_image = renderer.render(twin).convert("RGB")
                if ImageChops.difference(original_image, twin_image).getbbox() is not None:
                    _fail(
                        failures,
                        "latent_symmetry",
                        f"{scene_id}/{name}: declared symmetry changes pixels",
                    )
                if renderer.analytic_gold(twin) != original_gold:
                    _fail(
                        failures,
                        "latent_symmetry",
                        f"{scene_id}/{name}: declared symmetry changes gold",
                    )
        except Exception as exc:
            _fail(failures, "latent_symmetry", f"{scene_id}: symmetry probe raised {exc!r}")

    latent_alias_status = "pass"
    if candidate_schema == "0.4.0":
        alias = metadata.get("latent_alias") or {}
        mode = alias.get("mode")
        label_inputs = set(alias.get("label_inputs") or [])
        latent_fields = set((metadata.get("latent_z") or {}).get("fields") or [])
        scene_fields = (
            set.intersection(*(set(scene) for scene in scenes.values())) if scenes else set()
        )
        if not label_inputs or not label_inputs <= latent_fields:
            _fail(
                failures,
                "latent_symmetry",
                "label_inputs must be a nonempty subset of latent_z.fields",
            )
        if not latent_fields <= scene_fields:
            _fail(
                failures,
                "latent_symmetry",
                f"declared latent fields are missing from generated scenes: "
                f"{sorted(latent_fields - scene_fields)}",
            )
        renderer_source = candidate / "world/renderer.py"
        try:
            gold_keys = _reachable_scene_keys(renderer_source, "analytic_gold")
            render_keys = _reachable_scene_keys(renderer_source, "render")
            if gold_keys != label_inputs:
                _fail(
                    failures,
                    "latent_symmetry",
                    "latent_alias.label_inputs differs from analytic-gold scene access: "
                    f"declared={sorted(label_inputs)} observed={sorted(gold_keys)}",
                )
            hidden = gold_keys - render_keys
            if hidden:
                _fail(
                    failures,
                    "latent_symmetry",
                    f"analytic gold reads scene fields not used by rendering: {sorted(hidden)}",
                )
        except (OSError, SyntaxError) as exc:
            _fail(failures, "latent_symmetry", f"rendered-field audit failed: {exc}")
        ambiguous = [key for key, values in pixel_decisions.items() if len(values) > 1]
        if ambiguous:
            _fail(
                failures,
                "latent_symmetry",
                f"{len(ambiguous)} sampled pixel hashes map to multiple decisions",
            )
        if mode == "tested_transforms" and symmetry_count == 0:
            _fail(
                failures,
                "latent_symmetry",
                "tested_transforms requires at least one exercised transformation",
            )
        elif mode == "not_applicable":
            latent_alias_status = "not_applicable"
            if symmetry_count:
                _fail(
                    failures,
                    "latent_symmetry",
                    "not_applicable requires latent_symmetries(scene) to return an empty list",
                )
        elif mode not in {"tested_transforms", "not_applicable"}:
            _fail(failures, "latent_symmetry", "latent_alias.mode is invalid")
    elif symmetry_count == 0:
        _fail(failures, "latent_symmetry", "no render-invariant symmetry was exercised")

    if failures.get("latent_symmetry"):
        latent_alias_status = "fail"

    result = {
        "schema_version": "0.4.0" if candidate_schema == "0.4.0" else "0.2.0",
        "record_count": len(rows),
        "scene_count": len(scenes),
        "prompt_families": sorted(families),
        "classes_off_quarantine": {key: sorted(value) for key, value in by_family.items()},
        "symmetries_checked": symmetry_count,
        "latent_alias_status": latent_alias_status,
        "failures": dict(failures),
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
