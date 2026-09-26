from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scripts.audit_qd_attestations import _source_stress_config, audit

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "harbor/datasets/foundry-builder-v1/build-question-worlds/tests/qd_attestation.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("protected_qd_attestation", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qd_attestation_recomputes_local_compact_support(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    world = candidate / "world"
    dataset = tmp_path / "dataset"
    gallery = dataset / "gallery"
    world.mkdir(parents=True)
    gallery.mkdir(parents=True)
    (world / "oracle.py").write_text(
        """from PIL import Image
import numpy as np

def decision_from_image(image: Image.Image) -> str:
    a = np.asarray(image.convert("RGB"))
    dark = (a[:, :, 0] < 80) & (a[:, :, 1] < 80) & (a[:, :, 2] < 80)
    left = int(dark[:, :32].sum())
    right = int(dark[:, 32:].sum())
    return "left" if left > right else "right"
""",
        encoding="utf-8",
    )
    rows = []
    for index in range(6):
        image = Image.new("RGB", (64, 64), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((5, 5, 12, 12), fill="black")
        name = f"scene_{index:04d}.png"
        image.save(gallery / name)
        rows.append({"image_path": f"gallery/{name}", "decision": "left"})
    (dataset / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    result = _load_module().characterize_candidate(candidate, dataset)

    assert result["status"] == "complete"
    assert result["descriptor"] == {
        "evidence_extent": "local",
        "evidence_topology": "compact",
    }
    assert result["descriptor_source"] == "protected_offline_pixel_oracle_ablation"
    assert len(result["behavioral_vector"]) == 39
    assert result["scenes_analyzed"] == 6


def test_scene_relative_attestation_does_not_smear_a_moving_local_cue(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    world = candidate / "world"
    dataset = tmp_path / "dataset"
    gallery = dataset / "gallery"
    world.mkdir(parents=True)
    gallery.mkdir(parents=True)
    (world / "oracle.py").write_text(
        """from PIL import Image
import numpy as np

def decision_from_image(image: Image.Image) -> str:
    a = np.asarray(image.convert("RGB"))
    dark = (a[:, :, 0] < 80) & (a[:, :, 1] < 80) & (a[:, :, 2] < 80)
    return "present" if int(dark.sum()) >= 16 else "absent"
""",
        encoding="utf-8",
    )
    rows = []
    positions = ((8, 8), (24, 8), (40, 8), (56, 8), (8, 24), (40, 40))
    for index, (x, y) in enumerate(positions):
        image = Image.new("RGB", (64, 64), "white")
        ImageDraw.Draw(image).rectangle((x - 3, y - 3, x + 3, y + 3), fill="black")
        name = f"scene_{index:04d}.png"
        image.save(gallery / name)
        rows.append({"image_path": f"gallery/{name}", "decision": "present"})
    (dataset / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    module = _load_module()
    legacy = module.characterize_candidate(candidate, dataset)
    adaptive = module.characterize_candidate(
        candidate,
        dataset,
        attestation_id="pixel-oracle-support@0.2.0",
    )

    assert legacy["status"] == "incomplete"
    assert adaptive["status"] == "complete"
    assert adaptive["descriptor"] == {
        "evidence_extent": "local",
        "evidence_topology": "compact",
    }
    assert adaptive["descriptor_stability_fraction"] == 1.0
    assert adaptive["descriptor_method"] == "per_scene_geometry_then_robust_aggregate"
    assert len(adaptive["behavioral_vector"]) == 52

    robust = module.characterize_candidate(
        candidate,
        dataset,
        attestation_id="pixel-oracle-support@0.3.0",
    )
    assert robust["status"] == "complete"
    assert robust["descriptor"] == {
        "support_scale": "focal",
        "support_shape": "compact",
    }
    assert robust["descriptor_stability_fraction"] == 1.0
    assert robust["grid"] == [6, 6]
    assert len(robust["behavioral_vector"]) == 93

    refined = module.characterize_candidate(
        candidate,
        dataset,
        attestation_id="pixel-oracle-support@0.4.0",
    )
    assert refined["status"] == "complete"
    assert refined["descriptor"] == {
        "support_scale": "focal",
        "support_shape": "pathlike",
    }
    assert refined["descriptor_stability_fraction"] == 0.5
    assert refined["grid"] == [6, 6]
    assert "overlap_1_over_4" in refined["descriptor_method"]
    assert len(refined["behavioral_vector"]) == 93

    high_res = module.characterize_candidate(
        candidate,
        dataset,
        attestation_id="pixel-oracle-support@0.5.0",
    )
    assert high_res["status"] == "complete"
    assert high_res["descriptor"] == {
        "support_scale": "focal",
        "support_shape": "compact",
    }
    assert high_res["descriptor_stability_fraction"] == 1.0
    assert high_res["grid"] == [8, 8]
    assert len(high_res["behavioral_vector"]) == 149


def test_robust_support_scale_and_shape_cells_are_constructively_reachable() -> None:
    module = _load_module()
    masks: dict[tuple[str, str], np.ndarray] = {}

    def mask(*points: tuple[int, int]) -> np.ndarray:
        value = np.zeros((6, 6), dtype=bool)
        for y, x in points:
            value[y, x] = True
        return value

    masks[("focal", "compact")] = mask((2, 2))
    masks[("focal", "pathlike")] = mask((2, 1), (2, 2), (2, 3))
    masks[("focal", "multipart")] = mask((2, 1), (2, 3))
    regional_block = np.zeros((6, 6), dtype=bool)
    regional_block[1:4, 1:4] = True
    masks[("regional", "compact")] = regional_block
    masks[("regional", "pathlike")] = mask((2, 1), (2, 2), (2, 3), (2, 4))
    masks[("regional", "multipart")] = mask((1, 1), (3, 3))
    masks[("distributed", "compact")] = np.ones((6, 6), dtype=bool)
    masks[("distributed", "pathlike")] = mask(*((2, x) for x in range(6)))
    masks[("distributed", "multipart")] = mask((0, 0), (5, 5))

    observed = set()
    for support in masks.values():
        geometry = module._normalized_geometry(support, grid_size=6)
        observed.add(
            tuple(
                module._robust_descriptor(
                    span=geometry[1],
                    components=round(geometry[2] * 8.0),
                    elongation=2.0 ** (geometry[3] * math.log2(20.0)),
                    fill=module._support_fill(support),
                ).values()
            )
        )

    assert observed == set(masks)


def test_backfill_regenerates_a_fixed_stress_gallery(tmp_path: Path) -> None:
    controller = tmp_path / "controller"
    candidate = controller / "candidates/working-seed/episode-001__synthetic"
    world = candidate / "world"
    world.mkdir(parents=True)
    (candidate / "candidate.json").write_text(
        json.dumps({"candidate_id": "synthetic", "question": "Left or right?"}) + "\n",
        encoding="utf-8",
    )
    (world / "oracle.py").write_text(
        """from PIL import Image
import numpy as np

def decision_from_image(image: Image.Image) -> str:
    a = np.asarray(image.convert("RGB"))
    dark = (a[:, :, 0] < 80) & (a[:, :, 1] < 80) & (a[:, :, 2] < 80)
    return "left" if int(dark[:, :32].sum()) > int(dark[:, 32:].sum()) else "right"
""",
        encoding="utf-8",
    )
    (world / "generate.py").write_text(
        """import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    gallery = args.out / "gallery"
    gallery.mkdir(parents=True)
    rows = []
    for index in range(args.n):
        image = Image.new("RGB", (64, 64), "white")
        ImageDraw.Draw(image).rectangle((5, 5, 12, 12), fill="black")
        name = f"scene_{index:04d}.png"
        image.save(gallery / name)
        rows.append({"image_path": f"gallery/{name}", "decision": "left"})
    (args.out / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\\n" for row in rows), encoding="utf-8"
    )

if __name__ == "__main__":
    main()
""",
        encoding="utf-8",
    )

    result = audit([controller], stress_seed=9929, stress_scenes=6)

    assert result["stress_seed"] == 9929
    assert result["stress_scenes"] == 6
    assert result["candidate_count"] == 1
    assert result["complete_count"] == 1
    assert result["records"][0]["attestation"]["descriptor"] == {
        "evidence_extent": "local",
        "evidence_topology": "compact",
    }


def test_backfill_infers_the_live_verifier_stress_configuration(tmp_path: Path) -> None:
    controllers = []
    for index in range(2):
        controller = tmp_path / f"controller-{index}"
        controller.mkdir()
        (controller / "run-config.json").write_text(
            json.dumps(
                {
                    "verification": {
                        "oracle_stress_seeds": [101, 2027, 9929],
                        "oracle_stress_scenes_per_seed": 24,
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )
        controllers.append(controller)

    assert _source_stress_config(controllers) == (101, 24)
