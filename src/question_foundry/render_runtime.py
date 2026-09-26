"""Opt-in renderer runtimes for materialized Harbor question-foundry tasks."""

from __future__ import annotations

import shutil
from pathlib import Path

DEFAULT_RENDER_RUNTIME = "python-pillow@0.1.0"
LATEX_TIKZ_RENDER_RUNTIME = "latex-tikz@0.1.0"
LATEX_TIKZ_IMAGE = "question-foundry-latex-tikz-runtime:0.1"
LATEX_TIKZ_REPLAY_IMAGE = "question-foundry-candidate-replay-latex-tikz:0.1"

_RUNTIME_IMAGES = {
    DEFAULT_RENDER_RUNTIME: None,
    LATEX_TIKZ_RENDER_RUNTIME: LATEX_TIKZ_IMAGE,
}


def supported_render_runtimes() -> tuple[str, ...]:
    return tuple(sorted(_RUNTIME_IMAGES))


def campaign_render_runtime(campaign: dict) -> str:
    environment = campaign.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("campaign has no environment block")
    runtime = environment.get("render_runtime", DEFAULT_RENDER_RUNTIME)
    if runtime not in _RUNTIME_IMAGES:
        raise ValueError(f"render_runtime must be one of {list(supported_render_runtimes())}")
    return runtime


def _replace_base_image(dockerfile: Path, image: str) -> None:
    lines = dockerfile.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("FROM "):
        raise ValueError(f"task Dockerfile has no leading FROM instruction: {dockerfile}")
    lines[0] = f"FROM {image}"
    dockerfile.write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_task_render_runtime(*, task: Path, campaign: dict, capability_source: Path) -> dict:
    """Apply an explicitly selected runtime without duplicating the task template."""

    runtime = campaign_render_runtime(campaign)
    image = _RUNTIME_IMAGES[runtime]
    record = {
        "id": runtime,
        "container_image": image,
        "default": runtime == DEFAULT_RENDER_RUNTIME,
    }
    if image is None:
        return record

    for relative in ("environment/Dockerfile", "tests/Dockerfile"):
        _replace_base_image(task / relative, image)

    if not capability_source.is_file():
        raise FileNotFoundError(capability_source)
    destination = task / "environment/starter/RENDERER_CAPABILITY.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(capability_source, destination)

    instruction = task / "instruction.md"
    instruction.write_text(
        "# Optional LaTeX/TikZ renderer available\n\n"
        "Read `/workspace/RENDERER_CAPABILITY.md` before using the renderer. "
        "LaTeX is optional: use it only when it materially improves the visual "
        "world. The independent inverse arm must still recover the answer from "
        "the rasterized PNG alone.\n\n" + instruction.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return record
