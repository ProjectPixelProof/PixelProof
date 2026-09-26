"""Protected, deterministic quality-diversity descriptor attestation.

The builder never supplies the descriptors admitted to the archive. They are
recomputed after ordinary candidate verification from rendered pixels and the
independent pixel-only oracle. A developmental public preview may reuse this
algorithm on the builder's own gallery, but protected admission always reruns
it offline on unseen stress renders after agent teardown. A fixed grid-ablation
probe estimates which separated image regions the oracle depends on. The result
is useful as a behavioral archive descriptor; it is not an additional
scientific-validity gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Public previews must never alter the candidate tree whose digest is audited.
# Protected verifier containers already set PYTHONDONTWRITEBYTECODE, but keep
# the shared implementation safe when the builder invokes it directly.
sys.dont_write_bytecode = True

ATTESTATION_ID = "pixel-oracle-support@0.1.0"
SCHEMA_VERSION = "quality-diversity-attestation-0.1.0"
ADAPTIVE_ATTESTATION_ID = "pixel-oracle-support@0.2.0"
ADAPTIVE_SCHEMA_VERSION = "quality-diversity-attestation-0.2.0"
ROBUST_ATTESTATION_ID = "pixel-oracle-support@0.3.0"
ROBUST_SCHEMA_VERSION = "quality-diversity-attestation-0.3.0"
REFINED_ATTESTATION_ID = "pixel-oracle-support@0.4.0"
REFINED_SCHEMA_VERSION = "quality-diversity-attestation-0.4.0"
HIGH_RES_ATTESTATION_ID = "pixel-oracle-support@0.5.0"
HIGH_RES_SCHEMA_VERSION = "quality-diversity-attestation-0.5.0"
GRID_SIZE = 4
ROBUST_GRID_SIZE = 6
MAX_SCENES = 6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_oracle(candidate_root: Path):
    world = candidate_root / "world"
    oracle_path = world / "oracle.py"
    if not oracle_path.is_file():
        raise ValueError("candidate has no world/oracle.py")
    module_name = f"_qd_oracle_{hashlib.sha256(str(candidate_root).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, oracle_path)
    if spec is None or spec.loader is None:
        raise ValueError("could not load candidate pixel oracle")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(world))
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    function = getattr(module, "decision_from_image", None)
    if not callable(function):
        raise ValueError("candidate oracle has no callable decision_from_image")
    return function, oracle_path


def _unique_scene_rows(dataset_root: Path) -> list[dict[str, Any]]:
    manifest = dataset_root / "manifest.jsonl"
    if not manifest.is_file():
        raise ValueError("dataset has no manifest.jsonl")
    by_image: dict[str, dict[str, Any]] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image_path = row.get("image_path")
        if not isinstance(image_path, str) or not image_path:
            continue
        by_image.setdefault(image_path, row)
    rows = [by_image[key] for key in sorted(by_image)]
    if len(rows) < 4:
        raise ValueError("QD attestation requires at least four distinct rendered scenes")
    if len(rows) <= MAX_SCENES:
        return rows
    # Evenly cover the deterministic gallery instead of taking only its prefix.
    indices = np.linspace(0, len(rows) - 1, MAX_SCENES, dtype=int)
    return [rows[int(index)] for index in indices]


def _background(array: np.ndarray) -> np.ndarray:
    border = np.concatenate(
        (
            array[:3].reshape(-1, 3),
            array[-3:].reshape(-1, 3),
            array[:, :3].reshape(-1, 3),
            array[:, -3:].reshape(-1, 3),
        ),
        axis=0,
    )
    return np.median(border, axis=0).astype(np.uint8)


def _visual_metrics(array: np.ndarray, background: np.ndarray) -> tuple[float, float, float]:
    delta = np.linalg.norm(array.astype(np.float32) - background.astype(np.float32), axis=2)
    foreground_fraction = float(np.mean(delta > 24.0))
    gray = np.mean(array.astype(np.float32), axis=2)
    gradients = np.concatenate(
        (
            np.abs(np.diff(gray, axis=0)).ravel(),
            np.abs(np.diff(gray, axis=1)).ravel(),
        )
    )
    edge_density = float(np.mean(gradients > 18.0))
    quantized = (array // 32).reshape(-1, 3)
    _, counts = np.unique(quantized, axis=0, return_counts=True)
    probabilities = counts.astype(np.float64) / counts.sum()
    entropy = float(-(probabilities * np.log2(probabilities)).sum() / 9.0)
    return foreground_fraction, edge_density, min(1.0, entropy)


def _connected_components(mask: np.ndarray) -> int:
    seen: set[tuple[int, int]] = set()
    count = 0
    height, width = mask.shape
    for y, x in np.argwhere(mask):
        point = (int(y), int(x))
        if point in seen:
            continue
        count += 1
        stack = [point]
        seen.add(point)
        while stack:
            cy, cx = stack.pop()
            for neighbor in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                ny, nx = neighbor
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
    return count


def _support_geometry(mask: np.ndarray, *, grid_size: int = GRID_SIZE) -> tuple[float, int, float]:
    points = np.argwhere(mask).astype(np.float64)
    if not len(points):
        return 0.0, 0, 1.0
    if len(points) == 1:
        return 0.0, 1, 1.0
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    diagonal = math.sqrt(2.0) * (grid_size - 1)
    span = float(distances.max() / diagonal)
    centered = points - points.mean(axis=0, keepdims=True)
    values = np.linalg.eigvalsh(centered.T @ centered / len(points))
    elongation = float((values[-1] + 0.05) / (values[0] + 0.05))
    return min(1.0, span), _connected_components(mask), min(20.0, elongation)


def _descriptor(
    *, support_fraction: float, span: float, components: int, elongation: float
) -> dict[str, str]:
    if support_fraction >= 0.55:
        extent = "global"
    elif support_fraction >= 0.25 or span >= 0.65:
        extent = "long_range"
    else:
        extent = "local"

    if support_fraction >= 0.60:
        topology = "diffuse"
    elif components >= 2:
        topology = "multipart"
    elif elongation >= 2.75 or (span >= 0.60 and support_fraction <= 0.50):
        topology = "elongated"
    else:
        topology = "compact"
    return {"evidence_extent": extent, "evidence_topology": topology}


def _robust_descriptor(*, span: float, components: int, elongation: float, fill: float) -> dict:
    """Map independently measured scale and shape into nine reachable niches."""

    if span < 0.30:
        scale = "focal"
    elif span < 0.65:
        scale = "regional"
    else:
        scale = "distributed"

    if components >= 2:
        shape = "multipart"
    elif elongation >= 2.75 or (fill <= 0.55 and span > 0.0):
        shape = "pathlike"
    else:
        shape = "compact"
    return {"support_scale": scale, "support_shape": shape}


def _support_fill(mask: np.ndarray) -> float:
    points = np.argwhere(mask)
    if not len(points):
        return 0.0
    height = int(points[:, 0].max() - points[:, 0].min() + 1)
    width = int(points[:, 1].max() - points[:, 1].min() + 1)
    return float(len(points) / (height * width))


def _overlapping_ablation_bounds(
    length: int, index: int, grid_size: int, *, overlap_divisor: int = 2
) -> tuple[int, int]:
    """Return a grid cell expanded by a fixed cell fraction on each side.

    Non-overlapping fine grids can split a small cue across a boundary so that
    no intervention removes it. The overlap makes cue discovery insensitive to
    that arbitrary tessellation while retaining a fixed deterministic probe.
    """

    start = length * index // grid_size
    end = length * (index + 1) // grid_size
    halo = math.ceil((end - start) / overlap_divisor)
    return max(0, start - halo), min(length, end + halo)


def _centered(array: np.ndarray, support: np.ndarray, *, grid_size: int = GRID_SIZE) -> np.ndarray:
    """Translate a per-scene grid to its support centroid without wraparound."""

    points = np.argwhere(support)
    if not len(points):
        return array.copy()
    centroid = points.mean(axis=0)
    target = np.asarray([(grid_size - 1) / 2.0, (grid_size - 1) / 2.0])
    dy, dx = np.rint(target - centroid).astype(int)
    output = np.zeros_like(array)
    for y in range(grid_size):
        for x in range(grid_size):
            ny, nx = y + int(dy), x + int(dx)
            if 0 <= ny < grid_size and 0 <= nx < grid_size:
                output[ny, nx] = array[y, x]
    return output


def _normalized_geometry(
    mask: np.ndarray, *, grid_size: int = GRID_SIZE
) -> tuple[float, float, float, float]:
    fraction = float(np.mean(mask))
    span, components, elongation = _support_geometry(mask, grid_size=grid_size)
    return (
        fraction,
        span,
        min(1.0, components / 8.0),
        min(1.0, math.log2(max(1.0, elongation)) / math.log2(20.0)),
    )


def _characterize_v2(candidate_root: Path, dataset_root: Path) -> dict[str, Any]:
    """Characterize support scene-relatively before robust aggregation.

    Version 0.1 averaged absolute grid coordinates across scenes. That turns a
    compact cue whose position is randomized into an apparently global map.
    Version 0.2 computes geometry and descriptors per scene, centroid-aligns
    the maps only for behavioral comparison, and then aggregates robustly.
    """

    oracle, oracle_path = _load_oracle(candidate_root)
    rows = _unique_scene_rows(dataset_root)
    aligned_impacts = []
    aligned_errors = []
    geometries = []
    visual_rows = []
    scene_records = []
    scene_descriptors = []
    total_errors = 0

    for row in rows:
        image_path = dataset_root / str(row["image_path"])
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        array = np.asarray(image, dtype=np.uint8)
        background = _background(array)
        visual_rows.append(_visual_metrics(array, background))
        baseline = oracle(image)
        expected = row.get("decision", row.get("answer"))
        if expected is not None and str(baseline) != str(expected):
            raise ValueError(
                "pixel oracle disagrees with the submitted manifest during QD attestation"
            )
        height, width = array.shape[:2]
        clean = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
        errors = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
        for gy in range(GRID_SIZE):
            y0, y1 = height * gy // GRID_SIZE, height * (gy + 1) // GRID_SIZE
            for gx in range(GRID_SIZE):
                x0, x1 = width * gx // GRID_SIZE, width * (gx + 1) // GRID_SIZE
                ablated = array.copy()
                ablated[y0:y1, x0:x1] = background
                try:
                    observed = oracle(Image.fromarray(ablated, mode="RGB"))
                except Exception:
                    errors[gy, gx] = 1.0
                    total_errors += 1
                else:
                    clean[gy, gx] = float(str(observed) != str(baseline))
        combined = (clean + errors) > 0.0
        geometry = _normalized_geometry(combined)
        descriptor = _descriptor(
            support_fraction=geometry[0],
            span=geometry[1],
            components=round(geometry[2] * 8.0),
            elongation=2.0 ** (geometry[3] * math.log2(20.0)),
        )
        geometries.append(geometry)
        scene_descriptors.append(descriptor if combined.any() else None)
        aligned_impacts.append(_centered((clean + errors).clip(0.0, 1.0), combined))
        aligned_errors.append(_centered(errors, combined))
        scene_records.append(
            {
                "image_path": str(row["image_path"]),
                "clean_decision_changes": int(clean.sum()),
                "oracle_errors": int(errors.sum()),
                "support": {
                    "active_fraction": round(geometry[0], 6),
                    "normalized_span": round(geometry[1], 6),
                    "connected_components": round(geometry[2] * 8.0),
                    "elongation": round(2.0 ** (geometry[3] * math.log2(20.0)), 6),
                },
                "descriptor": descriptor if combined.any() else None,
            }
        )

    geometry_rows = np.asarray(geometries, dtype=np.float64)
    median_geometry = np.median(geometry_rows, axis=0)
    q75, q25 = np.percentile(geometry_rows, [75, 25], axis=0)
    iqr_geometry = q75 - q25
    components = round(float(median_geometry[2]) * 8.0)
    elongation = 2.0 ** (float(median_geometry[3]) * math.log2(20.0))
    descriptor = _descriptor(
        support_fraction=float(median_geometry[0]),
        span=float(median_geometry[1]),
        components=components,
        elongation=elongation,
    )
    descriptor_stability = sum(item == descriptor for item in scene_descriptors) / len(rows)
    nonempty_scene_fraction = sum(item is not None for item in scene_descriptors) / len(rows)
    extent_order = ("local", "long_range", "global")
    topology_order = ("compact", "elongated", "multipart", "diffuse")
    descriptor_distribution = [
        sum(bool(item and item["evidence_extent"] == value) for item in scene_descriptors)
        / len(rows)
        for value in extent_order
    ] + [
        sum(bool(item and item["evidence_topology"] == value) for item in scene_descriptors)
        / len(rows)
        for value in topology_order
    ]
    aligned_impact_probability = np.mean(np.asarray(aligned_impacts), axis=0)
    aligned_error_probability = np.mean(np.asarray(aligned_errors), axis=0)
    visual = np.median(np.asarray(visual_rows, dtype=np.float64), axis=0)
    oracle_error_fraction = total_errors / (len(rows) * GRID_SIZE * GRID_SIZE)
    vector = [
        *aligned_impact_probability.ravel().tolist(),
        *aligned_error_probability.ravel().tolist(),
        *median_geometry.tolist(),
        *iqr_geometry.tolist(),
        *descriptor_distribution,
        *visual.tolist(),
        descriptor_stability,
        oracle_error_fraction,
    ]
    if len(vector) != 52:
        raise AssertionError("adaptive QD behavioral vector length changed")
    failures = []
    warnings = []
    status = "complete"
    if nonempty_scene_fraction < 2.0 / 3.0:
        status = "incomplete"
        failures.append("fewer than two thirds of stress scenes exposed an ablation dependency")
    if oracle_error_fraction > 0.50:
        warnings.append(
            "more than half of ablations made the pixel oracle unable to return a decision"
        )
    return {
        "schema_version": ADAPTIVE_SCHEMA_VERSION,
        "attestation_id": ADAPTIVE_ATTESTATION_ID,
        "status": status,
        "descriptor_source": "protected_scene_relative_pixel_oracle_ablation",
        "descriptor_method": "per_scene_geometry_then_robust_aggregate",
        "descriptor": descriptor,
        "descriptor_stability_fraction": round(descriptor_stability, 6),
        "nonempty_scene_fraction": round(nonempty_scene_fraction, 6),
        "scenes_analyzed": len(rows),
        "grid": [GRID_SIZE, GRID_SIZE],
        "support": {
            "median_active_fraction": round(float(median_geometry[0]), 6),
            "median_normalized_span": round(float(median_geometry[1]), 6),
            "median_connected_components": components,
            "median_elongation": round(elongation, 6),
            "geometry_iqr_normalized": np.round(iqr_geometry, 6).tolist(),
            "oracle_error_fraction": round(oracle_error_fraction, 6),
            "aligned_impact_probability": np.round(aligned_impact_probability, 6).tolist(),
            "aligned_error_probability": np.round(aligned_error_probability, 6).tolist(),
        },
        "descriptor_distribution": {
            "evidence_extent": dict(zip(extent_order, descriptor_distribution[:3], strict=True)),
            "evidence_topology": dict(
                zip(topology_order, descriptor_distribution[3:], strict=True)
            ),
        },
        "visual_structure": {
            "foreground_fraction": round(float(visual[0]), 6),
            "edge_density": round(float(visual[1]), 6),
            "quantized_color_entropy": round(float(visual[2]), 6),
        },
        "behavioral_vector": [round(float(value), 6) for value in vector],
        "oracle_source_sha256": _sha256(oracle_path),
        "scene_records": scene_records,
        "failures": failures,
        "warnings": warnings,
    }


def _characterize_v3(
    candidate_root: Path,
    dataset_root: Path,
    *,
    overlap_divisor: int = 2,
    attestation_id: str = ROBUST_ATTESTATION_ID,
    schema_version: str = ROBUST_SCHEMA_VERSION,
    descriptor_source: str = "protected_scene_relative_pixel_oracle_ablation_v3",
    grid_size: int = ROBUST_GRID_SIZE,
) -> dict[str, Any]:
    """Measure scene-relative support scale and shape on a 6-by-6 grid.

    Scale is determined only by normalized support span. Shape is determined
    independently from connectivity, elongation, and bounding-box fill. This
    yields a balanced, interpretable 3-by-3 archive without the narrow and
    coupled cells created by the version-0.2 extent/topology thresholds.
    Oracle exceptions under destructive ablation remain a separate diagnostic:
    they expose decisive support but are not evidence that a clean world is
    scientifically invalid.
    """

    oracle, oracle_path = _load_oracle(candidate_root)
    rows = _unique_scene_rows(dataset_root)
    aligned_impacts = []
    aligned_errors = []
    geometries = []
    visual_rows = []
    scene_records = []
    scene_descriptors = []
    total_errors = 0

    for row in rows:
        image_path = dataset_root / str(row["image_path"])
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        array = np.asarray(image, dtype=np.uint8)
        background = _background(array)
        visual_rows.append(_visual_metrics(array, background))
        baseline = oracle(image)
        expected = row.get("decision", row.get("answer"))
        if expected is not None and str(baseline) != str(expected):
            raise ValueError(
                "pixel oracle disagrees with the submitted manifest during QD attestation"
            )
        height, width = array.shape[:2]
        clean = np.zeros((grid_size, grid_size), dtype=np.float64)
        errors = np.zeros((grid_size, grid_size), dtype=np.float64)
        for gy in range(grid_size):
            y0, y1 = _overlapping_ablation_bounds(
                height, gy, grid_size, overlap_divisor=overlap_divisor
            )
            for gx in range(grid_size):
                x0, x1 = _overlapping_ablation_bounds(
                    width, gx, grid_size, overlap_divisor=overlap_divisor
                )
                ablated = array.copy()
                ablated[y0:y1, x0:x1] = background
                try:
                    observed = oracle(Image.fromarray(ablated, mode="RGB"))
                except Exception:
                    errors[gy, gx] = 1.0
                    total_errors += 1
                else:
                    clean[gy, gx] = float(str(observed) != str(baseline))
        combined = (clean + errors) > 0.0
        geometry4 = _normalized_geometry(combined, grid_size=grid_size)
        fill = _support_fill(combined)
        geometry = (*geometry4, fill)
        components = round(geometry4[2] * 8.0)
        elongation = 2.0 ** (geometry4[3] * math.log2(20.0))
        descriptor = _robust_descriptor(
            span=geometry4[1],
            components=components,
            elongation=elongation,
            fill=fill,
        )
        geometries.append(geometry)
        scene_descriptors.append(descriptor if combined.any() else None)
        aligned_impacts.append(
            _centered((clean + errors).clip(0.0, 1.0), combined, grid_size=grid_size)
        )
        aligned_errors.append(_centered(errors, combined, grid_size=grid_size))
        scene_records.append(
            {
                "image_path": str(row["image_path"]),
                "clean_decision_changes": int(clean.sum()),
                "oracle_abstentions": int(errors.sum()),
                "support": {
                    "active_fraction": round(geometry4[0], 6),
                    "normalized_span": round(geometry4[1], 6),
                    "connected_components": components,
                    "elongation": round(elongation, 6),
                    "bounding_box_fill": round(fill, 6),
                },
                "descriptor": descriptor if combined.any() else None,
            }
        )

    geometry_rows = np.asarray(geometries, dtype=np.float64)
    median_geometry = np.median(geometry_rows, axis=0)
    q75, q25 = np.percentile(geometry_rows, [75, 25], axis=0)
    iqr_geometry = q75 - q25
    components = round(float(median_geometry[2]) * 8.0)
    elongation = 2.0 ** (float(median_geometry[3]) * math.log2(20.0))
    descriptor = _robust_descriptor(
        span=float(median_geometry[1]),
        components=components,
        elongation=elongation,
        fill=float(median_geometry[4]),
    )
    descriptor_stability = sum(item == descriptor for item in scene_descriptors) / len(rows)
    nonempty_scene_fraction = sum(item is not None for item in scene_descriptors) / len(rows)
    scale_order = ("focal", "regional", "distributed")
    shape_order = ("compact", "pathlike", "multipart")
    descriptor_distribution = [
        sum(bool(item and item["support_scale"] == value) for item in scene_descriptors) / len(rows)
        for value in scale_order
    ] + [
        sum(bool(item and item["support_shape"] == value) for item in scene_descriptors) / len(rows)
        for value in shape_order
    ]
    aligned_impact_probability = np.mean(np.asarray(aligned_impacts), axis=0)
    aligned_error_probability = np.mean(np.asarray(aligned_errors), axis=0)
    visual = np.median(np.asarray(visual_rows, dtype=np.float64), axis=0)
    oracle_error_fraction = total_errors / (len(rows) * grid_size * grid_size)
    vector = [
        *aligned_impact_probability.ravel().tolist(),
        *aligned_error_probability.ravel().tolist(),
        *median_geometry.tolist(),
        *iqr_geometry.tolist(),
        *descriptor_distribution,
        *visual.tolist(),
        descriptor_stability,
        oracle_error_fraction,
    ]
    expected_vector_size = 2 * grid_size * grid_size + 21
    if len(vector) != expected_vector_size:
        raise AssertionError("robust QD behavioral vector length changed")
    failures = []
    warnings = []
    status = "complete"
    if nonempty_scene_fraction < 2.0 / 3.0:
        status = "incomplete"
        failures.append("fewer than two thirds of stress scenes exposed an ablation dependency")
    if oracle_error_fraction > 0.50:
        warnings.append(
            "more than half of destructive ablations made the pixel oracle abstain; "
            "recorded as support evidence and a diagnostic, not an admission failure"
        )
    return {
        "schema_version": schema_version,
        "attestation_id": attestation_id,
        "status": status,
        "descriptor_source": descriptor_source,
        "descriptor_method": (
            f"overlap_1_over_{overlap_divisor}_scene_relative_support_span_and_"
            "independent_shape_then_robust_aggregate"
        ),
        "descriptor": descriptor,
        "descriptor_stability_fraction": round(descriptor_stability, 6),
        "nonempty_scene_fraction": round(nonempty_scene_fraction, 6),
        "scenes_analyzed": len(rows),
        "grid": [grid_size, grid_size],
        "support": {
            "median_active_fraction": round(float(median_geometry[0]), 6),
            "median_normalized_span": round(float(median_geometry[1]), 6),
            "median_connected_components": components,
            "median_elongation": round(elongation, 6),
            "median_bounding_box_fill": round(float(median_geometry[4]), 6),
            "geometry_iqr_normalized": np.round(iqr_geometry, 6).tolist(),
            "oracle_error_fraction": round(oracle_error_fraction, 6),
            "oracle_error_interpretation": ("diagnostic_abstention_under_destructive_intervention"),
            "aligned_impact_probability": np.round(aligned_impact_probability, 6).tolist(),
            "aligned_error_probability": np.round(aligned_error_probability, 6).tolist(),
        },
        "descriptor_distribution": {
            "support_scale": dict(zip(scale_order, descriptor_distribution[:3], strict=True)),
            "support_shape": dict(zip(shape_order, descriptor_distribution[3:], strict=True)),
        },
        "visual_structure": {
            "foreground_fraction": round(float(visual[0]), 6),
            "edge_density": round(float(visual[1]), 6),
            "quantized_color_entropy": round(float(visual[2]), 6),
        },
        "behavioral_vector": [round(float(value), 6) for value in vector],
        "oracle_source_sha256": _sha256(oracle_path),
        "scene_records": scene_records,
        "failures": failures,
        "warnings": warnings,
    }


def characterize_candidate(
    candidate_root: Path,
    dataset_root: Path,
    *,
    attestation_id: str = ATTESTATION_ID,
) -> dict[str, Any]:
    candidate_root = candidate_root.resolve()
    dataset_root = dataset_root.resolve()
    if attestation_id == HIGH_RES_ATTESTATION_ID:
        return _characterize_v3(
            candidate_root,
            dataset_root,
            overlap_divisor=4,
            attestation_id=HIGH_RES_ATTESTATION_ID,
            schema_version=HIGH_RES_SCHEMA_VERSION,
            descriptor_source="protected_scene_relative_pixel_oracle_ablation_v5",
            grid_size=8,
        )
    if attestation_id == REFINED_ATTESTATION_ID:
        return _characterize_v3(
            candidate_root,
            dataset_root,
            overlap_divisor=4,
            attestation_id=REFINED_ATTESTATION_ID,
            schema_version=REFINED_SCHEMA_VERSION,
            descriptor_source="protected_scene_relative_pixel_oracle_ablation_v4",
        )
    if attestation_id == ROBUST_ATTESTATION_ID:
        return _characterize_v3(candidate_root, dataset_root)
    if attestation_id == ADAPTIVE_ATTESTATION_ID:
        return _characterize_v2(candidate_root, dataset_root)
    if attestation_id != ATTESTATION_ID:
        raise ValueError(f"unsupported QD attestation: {attestation_id}")
    oracle, oracle_path = _load_oracle(candidate_root)
    rows = _unique_scene_rows(dataset_root)
    impact_counts = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    error_counts = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)
    visual_rows = []
    scene_records = []

    for row in rows:
        image_path = dataset_root / str(row["image_path"])
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        array = np.asarray(image, dtype=np.uint8)
        background = _background(array)
        foreground_fraction, edge_density, entropy = _visual_metrics(array, background)
        visual_rows.append((foreground_fraction, edge_density, entropy))
        baseline = oracle(image)
        expected = row.get("decision", row.get("answer"))
        if expected is not None and str(baseline) != str(expected):
            raise ValueError(
                "pixel oracle disagrees with the submitted manifest during QD attestation"
            )
        height, width = array.shape[:2]
        clean_changes = 0
        oracle_errors = 0
        for gy in range(GRID_SIZE):
            y0, y1 = height * gy // GRID_SIZE, height * (gy + 1) // GRID_SIZE
            for gx in range(GRID_SIZE):
                x0, x1 = width * gx // GRID_SIZE, width * (gx + 1) // GRID_SIZE
                ablated = array.copy()
                ablated[y0:y1, x0:x1] = background
                try:
                    observed = oracle(Image.fromarray(ablated, mode="RGB"))
                except Exception:
                    error_counts[gy, gx] += 1.0
                    impact_counts[gy, gx] += 1.0
                    oracle_errors += 1
                else:
                    if str(observed) != str(baseline):
                        impact_counts[gy, gx] += 1.0
                        clean_changes += 1
        scene_records.append(
            {
                "image_path": str(row["image_path"]),
                "clean_decision_changes": clean_changes,
                "oracle_errors": oracle_errors,
            }
        )

    denominator = float(len(rows))
    impact_probability = impact_counts / denominator
    error_probability = error_counts / denominator
    active = impact_probability >= (2.0 / denominator if len(rows) < 6 else 1.0 / 3.0)
    support_fraction = float(np.mean(active))
    span, components, elongation = _support_geometry(active)
    oracle_error_fraction = float(error_counts.sum() / (denominator * GRID_SIZE * GRID_SIZE))
    visual = np.median(np.asarray(visual_rows, dtype=np.float64), axis=0)
    descriptor = _descriptor(
        support_fraction=support_fraction,
        span=span,
        components=components,
        elongation=elongation,
    )
    vector = [
        *impact_probability.ravel().tolist(),
        *error_probability.ravel().tolist(),
        support_fraction,
        span,
        min(1.0, components / 8.0),
        min(1.0, math.log2(max(1.0, elongation)) / math.log2(20.0)),
        *visual.tolist(),
    ]
    status = "complete"
    failures = []
    warnings = []
    if not active.any():
        status = "incomplete"
        failures.append("no reproducible grid-ablation dependency was observed")
    if oracle_error_fraction > 0.50:
        warnings.append(
            "more than half of ablations made the pixel oracle unable to return a decision"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "attestation_id": ATTESTATION_ID,
        "status": status,
        "descriptor_source": "protected_offline_pixel_oracle_ablation",
        "descriptor": descriptor,
        "scenes_analyzed": len(rows),
        "grid": [GRID_SIZE, GRID_SIZE],
        "active_tile_threshold": round(2.0 / denominator if len(rows) < 6 else 1.0 / 3.0, 6),
        "support": {
            "active_fraction": round(support_fraction, 6),
            "normalized_span": round(span, 6),
            "connected_components": components,
            "elongation": round(elongation, 6),
            "oracle_error_fraction": round(oracle_error_fraction, 6),
            "impact_probability": np.round(impact_probability, 6).tolist(),
            "error_probability": np.round(error_probability, 6).tolist(),
        },
        "visual_structure": {
            "foreground_fraction": round(float(visual[0]), 6),
            "edge_density": round(float(visual[1]), 6),
            "quantized_color_entropy": round(float(visual[2]), 6),
        },
        "behavioral_vector": [round(float(value), 6) for value in vector],
        "oracle_source_sha256": _sha256(oracle_path),
        "scene_records": scene_records,
        "failures": failures,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--attestation-id",
        choices=(
            ATTESTATION_ID,
            ADAPTIVE_ATTESTATION_ID,
            ROBUST_ATTESTATION_ID,
            REFINED_ATTESTATION_ID,
            HIGH_RES_ATTESTATION_ID,
        ),
        default=ATTESTATION_ID,
    )
    args = parser.parse_args()
    result = characterize_candidate(
        args.candidate,
        args.dataset,
        attestation_id=args.attestation_id,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"status": result["status"], "descriptor": result["descriptor"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
