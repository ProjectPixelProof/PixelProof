#!/usr/bin/env python3
"""Compile one constrained LaTeX page and rasterize it to a PNG."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _run(command: list[str], *, cwd: Path, timeout: int, environment: dict[str, str]) -> None:
    subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
    )


def render(source: Path, output: Path, *, dpi: int, timeout: int) -> None:
    source = source.resolve(strict=True)
    if source.suffix.lower() != ".tex" or not source.is_file():
        raise ValueError("source must be an existing .tex file")
    output = output.resolve()
    if output.suffix.lower() != ".png":
        raise ValueError("output must end in .png")
    if not 72 <= dpi <= 600:
        raise ValueError("dpi must be in [72, 600]")
    if not 1 <= timeout <= 120:
        raise ValueError("timeout must be in [1, 120]")

    environment = dict(os.environ)
    environment.update(
        {
            "SOURCE_DATE_EPOCH": "0",
            "TZ": "UTC",
            "openin_any": "p",
            "openout_any": "p",
        }
    )
    with tempfile.TemporaryDirectory(prefix="latex-render-") as temporary:
        build = Path(temporary)
        _run(
            [
                "pdflatex",
                "-no-shell-escape",
                "-interaction=batchmode",
                "-halt-on-error",
                "-file-line-error",
                f"-output-directory={build}",
                source.name,
            ],
            cwd=source.parent,
            timeout=timeout,
            environment=environment,
        )
        pdf = build / f"{source.stem}.pdf"
        if not pdf.is_file():
            raise RuntimeError("pdflatex did not produce the expected PDF")
        info = subprocess.run(
            ["pdfinfo", str(pdf)],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
        ).stdout
        pages = [line for line in info.splitlines() if line.startswith("Pages:")]
        if len(pages) != 1 or pages[0].partition(":")[2].strip() != "1":
            raise RuntimeError("LaTeX renderer requires exactly one PDF page")

        prefix = build / "raster"
        _run(
            [
                "pdftoppm",
                "-singlefile",
                "-png",
                "-r",
                str(dpi),
                str(pdf),
                str(prefix),
            ],
            cwd=build,
            timeout=timeout,
            environment=environment,
        )
        raster = prefix.with_suffix(".png")
        if not raster.is_file() or raster.stat().st_size == 0:
            raise RuntimeError("PDF rasterization did not produce a PNG")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(raster, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    render(args.source, args.output, dpi=args.dpi, timeout=args.timeout)


if __name__ == "__main__":
    main()
