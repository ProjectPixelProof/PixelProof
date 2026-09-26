"""Independent pixel-only inverse for the directed network.

The implementation segments marker and arrow colors, recovers node centers by
connected components, detects straight links by sampled ink support, infers each
triangular arrowhead's signed PCA direction, and performs directed reachability.
It receives no latent scene and imports no world-side module.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image

PALETTE = {
    "background": (250, 248, 244),
    "ink": (42, 47, 55),
    "arrow": (126, 65, 190),
    "joint": (221, 229, 237),
    "green": (36, 157, 92),
    "red": (215, 63, 68),
    "blue": (55, 105, 210),
    "orange": (232, 139, 38),
}


class OracleError(ValueError):
    pass


def _mask(arr: np.ndarray, color: tuple[int, int, int], tolerance: int = 12) -> np.ndarray:
    target = np.asarray(color, dtype=np.int16)
    return np.max(np.abs(arr.astype(np.int16) - target), axis=2) <= tolerance


def _components(mask: np.ndarray, minimum: int) -> list[np.ndarray]:
    height, width = mask.shape
    remaining = set(map(tuple, np.argwhere(mask)))
    found: list[np.ndarray] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        points = [start]
        while stack:
            y, x = stack.pop()
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                ny, nx = neighbor
                if 0 <= ny < height and 0 <= nx < width and neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    points.append(neighbor)
        if len(points) >= minimum:
            found.append(np.asarray(points, dtype=np.float64))
    return found


def _one_center(arr: np.ndarray, color: str) -> tuple[float, float]:
    parts = _components(_mask(arr, PALETTE[color]), 120)
    if len(parts) != 1:
        raise OracleError(f"expected one {color} marker, found {len(parts)}")
    yx = parts[0].mean(axis=0)
    return float(yx[1]), float(yx[0])


def _nodes(arr: np.ndarray) -> list[dict]:
    nodes = [{"id": "start", "point": _one_center(arr, "green"), "color": "green"}]
    for color in ("red", "blue", "orange"):
        nodes.append({"id": color, "point": _one_center(arr, color), "color": color})
    joints = _components(_mask(arr, PALETTE["joint"]), 100)
    if len(joints) != 9:
        raise OracleError(f"expected nine joint markers, found {len(joints)}")
    for index, part in enumerate(joints):
        yx = part.mean(axis=0)
        nodes.append({"id": f"joint_{index}", "point": (float(yx[1]), float(yx[0])), "color": None})
    return nodes


def _line_coverage(ink: np.ndarray, a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    length = float(np.hypot(bx - ax, by - ay))
    if length < 35.0:
        return 0.0
    endpoint_fraction = min(0.31, 21.0 / length)
    ts = np.linspace(endpoint_fraction, 1.0 - endpoint_fraction, 52)
    xs = ax + ts * (bx - ax)
    ys = ay + ts * (by - ay)
    hits = 0
    height, width = ink.shape
    for x, y in zip(xs, ys):
        ix, iy = int(round(x)), int(round(y))
        x0, x1 = max(0, ix - 3), min(width, ix + 4)
        y0, y1 = max(0, iy - 3), min(height, iy + 4)
        hits += bool(ink[y0:y1, x0:x1].any())
    return hits / len(ts)


def _undirected_edges(arr: np.ndarray, nodes: list[dict]) -> list[tuple[int, int]]:
    ink = _mask(arr, PALETTE["ink"], 22) | _mask(arr, PALETTE["arrow"], 22)
    edges: list[tuple[int, int]] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i]["point"], nodes[j]["point"]
            distance = float(np.hypot(b[0] - a[0], b[1] - a[1]))
            if distance <= 92.0 and _line_coverage(ink, a, b) >= 0.88:
                edges.append((i, j))
    if len(edges) != 12:
        raise OracleError(f"expected twelve links, recovered {len(edges)}")
    degree = [0] * len(nodes)
    for i, j in edges:
        degree[i] += 1
        degree[j] += 1
    if degree[0] != 3 or any(degree[i] != 1 for i in (1, 2, 3)):
        raise OracleError(f"invalid endpoint degrees {degree[:4]}")
    return edges


def _arrow_data(arr: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    parts = _components(_mask(arr, PALETTE["arrow"], 18), 45)
    if len(parts) != 12:
        raise OracleError(f"expected twelve arrowheads, found {len(parts)}")
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for part in parts:
        xy = part[:, ::-1]
        center = xy.mean(axis=0)
        centered = xy - center
        covariance = centered.T @ centered / len(centered)
        values, vectors = np.linalg.eigh(covariance)
        axis = vectors[:, int(np.argmax(values))]
        projections = centered @ axis
        if float(projections.max()) < float(-projections.min()):
            axis = -axis
            projections = -projections
        if float(projections.max() + projections.min()) < 2.0:
            raise OracleError("arrowhead has insufficient signed tip asymmetry")
        result.append((center, axis))
    return result


def _direct_edges(
    nodes: list[dict], edges: list[tuple[int, int]], arrows: list[tuple[np.ndarray, np.ndarray]]
) -> list[tuple[int, int]]:
    directed: list[tuple[int, int]] = []
    used: set[int] = set()
    for i, j in edges:
        a = np.asarray(nodes[i]["point"], dtype=np.float64)
        b = np.asarray(nodes[j]["point"], dtype=np.float64)
        vector = b - a
        length = float(np.linalg.norm(vector))
        unit = vector / length
        matches: list[tuple[float, int]] = []
        for index, (center, _) in enumerate(arrows):
            along = float((center - a) @ unit)
            offset = center - a
            perpendicular = abs(float(unit[0] * offset[1] - unit[1] * offset[0]))
            if 0.22 * length < along < 0.78 * length and perpendicular < 13.0:
                matches.append((perpendicular, index))
        if len(matches) != 1:
            raise OracleError(f"link has {len(matches)} matching arrowheads")
        index = matches[0][1]
        if index in used:
            raise OracleError("one arrowhead was assigned to two links")
        used.add(index)
        direction = arrows[index][1]
        directed.append((i, j) if float(direction @ unit) > 0 else (j, i))
    if len(used) != 12:
        raise OracleError("not every arrowhead was assigned")
    return directed


def decision_from_image(image: Image.Image) -> str:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if arr.shape[:2] != (640, 640):
        raise OracleError(f"unexpected image size {image.size}")
    nodes = _nodes(arr)
    edges = _undirected_edges(arr, nodes)
    directed = _direct_edges(nodes, edges, _arrow_data(arr))
    adjacency: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    for source, target in directed:
        adjacency[source].append(target)
    reached = {0}
    queue: deque[int] = deque([0])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor not in reached:
                reached.add(neighbor)
                queue.append(neighbor)
    answers = [nodes[i]["color"] for i in (1, 2, 3) if i in reached]
    if len(answers) != 1:
        raise OracleError(f"expected one reachable terminal, found {len(answers)}")
    return str(answers[0])
