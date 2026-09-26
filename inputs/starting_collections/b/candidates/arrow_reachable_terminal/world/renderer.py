"""Deterministic latent scene and Pillow renderer for a directed path network."""

from __future__ import annotations

import copy
import heapq
import math
import random

from PIL import Image, ImageDraw

WIDTH = 640
HEIGHT = 640
BACKGROUND = (250, 248, 244)
INK = (42, 47, 55)
ARROW = (126, 65, 190)
JOINT = (221, 229, 237)
COLORS = {
    "green": (36, 157, 92),
    "red": (215, 63, 68),
    "blue": (55, 105, 210),
    "orange": (232, 139, 38),
}


def _rotate(x: float, y: float, angle: float) -> tuple[float, float]:
    c, s = math.cos(angle), math.sin(angle)
    return c * x - s * y, s * x + c * y


def sample_scene(seed: int) -> dict:
    """Return a balanced symmetric graph for ``seed``.

    Consecutive seeds cycle answer color every scene and answer arm every three
    scenes, covering every color/position pairing in each nine-scene block.
    """

    rng = random.Random(int(seed))
    answer_color_index = int(seed) % 3
    answer_arm = (int(seed) // 3) % 3
    terminal_colors = ["red", "blue", "orange"]
    answer_color = terminal_colors[answer_color_index]
    remaining_colors = [c for c in terminal_colors if c != answer_color]
    rng.shuffle(remaining_colors)

    arm_colors: list[str | None] = [None, None, None]
    arm_colors[answer_arm] = answer_color
    for arm, color in zip((i for i in range(3) if i != answer_arm), remaining_colors):
        arm_colors[arm] = color

    # All three arms are rigid rotations of this same four-link polyline.
    template = [(0.0, 0.0), (66.0, -12.0), (124.0, 16.0), (184.0, -12.0), (244.0, 0.0)]
    global_angle = rng.uniform(0.0, 2.0 * math.pi)
    center = (WIDTH / 2.0, HEIGHT / 2.0)

    nodes: list[dict] = [
        {"id": "start", "x": center[0], "y": center[1], "kind": "start", "color": "green"}
    ]
    edges: list[dict] = []

    distractors = [i for i in range(3) if i != answer_arm]
    rng.shuffle(distractors)
    # One late and one earlier reversal: terminal-local direction alone never
    # uniquely identifies the answer, while every distractor remains blocked.
    blocked_at = {distractors[0]: 3, distractors[1]: 1 + rng.randrange(2)}

    for arm in range(3):
        theta = global_angle + arm * 2.0 * math.pi / 3.0
        ids = ["start"]
        for step, (lx, ly) in enumerate(template[1:], start=1):
            dx, dy = _rotate(lx, ly, theta)
            terminal = step == 4
            node_id = f"t{arm}" if terminal else f"a{arm}_{step}"
            nodes.append(
                {
                    "id": node_id,
                    "x": round(center[0] + dx, 4),
                    "y": round(center[1] + dy, 4),
                    "kind": "terminal" if terminal else "joint",
                    "color": arm_colors[arm] if terminal else "joint",
                }
            )
            ids.append(node_id)

        for index in range(4):
            outward = arm == answer_arm or index != blocked_at[arm]
            source, target = (ids[index], ids[index + 1])
            if not outward:
                source, target = target, source
            edges.append({"source": source, "target": target})

    # Record order is a genuine latent alias and carries no draw-order meaning.
    rng.shuffle(nodes)
    rng.shuffle(edges)
    return {"nodes": nodes, "edges": edges}


def _node_map(scene: dict) -> dict[str, dict]:
    return {node["id"]: node for node in scene["nodes"]}


def _directed_costs(scene: dict) -> dict[str, int]:
    """Minimum number of wrong-way traversals from START to every node."""

    nodes = scene["nodes"]
    edges = scene["edges"]
    adjacency: dict[str, list[tuple[str, int]]] = {node["id"]: [] for node in nodes}
    for edge in edges:
        source, target = edge["source"], edge["target"]
        adjacency[source].append((target, 0))
        adjacency[target].append((source, 1))
    distances = {node_id: 10**9 for node_id in adjacency}
    distances["start"] = 0
    queue: list[tuple[int, str]] = [(0, "start")]
    while queue:
        cost, node_id = heapq.heappop(queue)
        if cost != distances[node_id]:
            continue
        for neighbor, penalty in adjacency[node_id]:
            new_cost = cost + penalty
            if new_cost < distances[neighbor]:
                distances[neighbor] = new_cost
                heapq.heappush(queue, (new_cost, neighbor))
    return distances


def analytic_gold(scene: dict) -> str:
    nodes = scene["nodes"]
    edges = scene["edges"]
    # Keep both accesses explicit for the protected rendered-field audit.
    _ = len(edges)
    costs = _directed_costs(scene)
    reachable = [node for node in nodes if node["kind"] == "terminal" and costs[node["id"]] == 0]
    if len(reachable) != 1:
        raise ValueError(f"expected one reachable terminal, found {len(reachable)}")
    return str(reachable[0]["color"])


def margin(scene: dict) -> float:
    """Runner-up gap in wrong-way arrows required by the best two terminals."""

    costs = _directed_costs(scene)
    terminal_costs = sorted(
        costs[node["id"]] for node in scene["nodes"] if node["kind"] == "terminal"
    )
    return float(terminal_costs[1] - terminal_costs[0])


def is_quarantined(scene: dict) -> bool:
    try:
        if analytic_gold(scene) not in {"red", "blue", "orange"} or margin(scene) < 1.0:
            return True
        nodes = _node_map(scene)
        if len(nodes) != 13 or len(scene["edges"]) != 12:
            return True
        for edge in scene["edges"]:
            a, b = nodes[edge["source"]], nodes[edge["target"]]
            if math.hypot(a["x"] - b["x"], a["y"] - b["y"]) < 48.0:
                return True
        return False
    except (KeyError, TypeError, ValueError):
        return True


def _arrow_polygon(a: dict, b: dict, scale: int) -> list[tuple[float, float]]:
    ax, ay, bx, by = float(a["x"]), float(a["y"]), float(b["x"]), float(b["y"])
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
    tip = (mx + 13.0 * ux, my + 13.0 * uy)
    base = (mx - 9.0 * ux, my - 9.0 * uy)
    return [
        (tip[0] * scale, tip[1] * scale),
        ((base[0] + 8.0 * nx) * scale, (base[1] + 8.0 * ny) * scale),
        ((base[0] - 8.0 * nx) * scale, (base[1] - 8.0 * ny) * scale),
    ]


def render(scene: dict) -> Image.Image:
    nodes = scene["nodes"]
    edges = scene["edges"]
    node_by_id = {node["id"]: node for node in nodes}
    scale = 4
    image = Image.new("RGB", (WIDTH * scale, HEIGHT * scale), BACKGROUND)
    draw = ImageDraw.Draw(image)

    for edge in edges:
        a, b = node_by_id[edge["source"]], node_by_id[edge["target"]]
        draw.line(
            [(a["x"] * scale, a["y"] * scale), (b["x"] * scale, b["y"] * scale)],
            fill=INK,
            width=7 * scale,
        )
    for edge in edges:
        a, b = node_by_id[edge["source"]], node_by_id[edge["target"]]
        draw.polygon(_arrow_polygon(a, b, scale), fill=ARROW)

    for node in nodes:
        radius = 18 if node["kind"] in {"start", "terminal"} else 15
        fill = JOINT if node["kind"] == "joint" else COLORS[node["color"]]
        box = [
            (node["x"] - radius) * scale,
            (node["y"] - radius) * scale,
            (node["x"] + radius) * scale,
            (node["y"] + radius) * scale,
        ]
        draw.ellipse(box, fill=fill, outline=INK, width=3 * scale)

    return image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


def latent_symmetries(scene: dict) -> list[tuple[str, dict]]:
    twin = copy.deepcopy(scene)
    twin["nodes"] = list(reversed(twin["nodes"]))
    twin["edges"] = list(reversed(twin["edges"]))
    return [("reverse_record_order", twin)]
