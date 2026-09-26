"""Manifest schema round-trip (SSOT §15 — manifests are the tracked ground-truth ledger)."""

from question_foundry.manifest import (
    MANIFEST_SCHEMA_VERSION,
    ExampleRecord,
    read_manifest,
    write_manifest,
)


def _record(i: int = 0) -> ExampleRecord:
    # NB: JSON round-trips tuples as lists, so nested dicts use lists on purpose.
    return ExampleRecord(
        example_id=f"two_circles-pytest-{i:04d}",
        image_path=f"worlds/two_circles/data/rendered/pytest/{i:04d}.png",
        split="pytest",
        seed=i,
        world="two_circles",
        scene={"c1": [100.0, 256.0], "c2": [220.0, 256.0], "r1": 50.0, "r2": 70.0},
        targets={"m": 0.0, "margin_bin": "tangent", "label": "touching", "overlap_or_touch": True},
        prompt_family="pf1_overlap_yesno",
        prompt_text="Are the two circles overlapping? Answer yes or no.",
        renderer_version="two_circles-0.1.0",
    )


def test_round_trip_identity(tmp_path):
    records = [_record(i) for i in range(3)]
    path = tmp_path / "manifest.jsonl"
    write_manifest(records, path)
    assert list(read_manifest(path)) == records


def test_schema_version_default():
    assert _record().schema_version == MANIFEST_SCHEMA_VERSION
    assert MANIFEST_SCHEMA_VERSION.startswith("manifest-")


def test_read_skips_blank_lines(tmp_path):
    path = tmp_path / "manifest.jsonl"
    write_manifest([_record()], path)
    path.write_text(path.read_text() + "\n\n", encoding="utf-8")
    assert len(list(read_manifest(path))) == 1


def test_write_creates_parent_dirs(tmp_path):
    path = tmp_path / "deep" / "nested" / "manifest.jsonl"
    write_manifest([_record()], path)
    assert path.is_file()
