from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

import oracle
import prompts
import renderer
import verify


ROOT = Path(__file__).parents[1]
DATASET = ROOT / "evidence"


def test_exact_evidence_is_complete() -> None:
    assert DATASET.joinpath("manifest.jsonl").is_file()
    assert not verify.verify(DATASET)
    rows = [json.loads(line) for line in DATASET.joinpath("manifest.jsonl").read_text().splitlines() if line]
    assert len(rows) >= 12
    assert {row["prompt_family"] for row in rows} == set(prompts.PROMPT_FAMILIES)
    assert len({row["scene_id"] for row in rows}) >= 4


def test_pixel_oracle_reads_the_exact_gallery() -> None:
    rows = [json.loads(line) for line in DATASET.joinpath("manifest.jsonl").read_text().splitlines() if line]
    for row in rows:
        image = Image.open(DATASET / row["image_path"]).convert("RGB")
        assert oracle.decision_from_image(image) == row["decision"]


def test_generation_is_deterministic(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    for destination in (a, b):
        subprocess.run(
            [sys.executable, str(ROOT / "world/generate.py"), "--out", str(destination), "--n", "8", "--seed", "2027"],
            check=True,
            cwd=ROOT / "world",
        )
    assert sorted(path.name for path in (a / "gallery").glob("*.png")) == sorted(path.name for path in (b / "gallery").glob("*.png"))
    for pa in sorted((a / "gallery").glob("*.png")):
        assert pa.read_bytes() == (b / "gallery" / pa.name).read_bytes()
