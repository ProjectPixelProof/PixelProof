"""Build pinned coding-agent CLIs without credentials or inference (downloads packages)."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from jinja2 import Template

from scripts.release import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=("codex", "claude", "opencode", "all"), default="all")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/agent-installations")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("choose a new output directory")
    templates = {
        "codex": "install-codex-foundry.sh.j2",
        "claude": "install-claude-code-foundry.sh.j2",
        "opencode": "install-opencode-foundry.sh.j2",
    }
    names = tuple(templates) if args.agent == "all" else (args.agent,)
    for name in names:
        context = args.out.resolve() / name
        context.mkdir(parents=True)
        pin = json.loads((ROOT / f"configs/agents/{name}.json").read_text())["agent"]["cli_version"]
        installer = Template((ROOT / "harbor/agents" / templates[name]).read_text()).render(
            version=pin
        )
        (context / "install.sh").write_text(installer)
        (context / "Dockerfile").write_text(
            "FROM question-foundry-latex-tikz-runtime:0.1\n"
            "RUN useradd --create-home --shell /bin/bash agent\n"
            "COPY install.sh /tmp/install.sh\nRUN bash /tmp/install.sh\n"
        )
        subprocess.run(
            [
                "docker",
                "build",
                "--provenance=false",
                "-t",
                f"pixelproof-cli-check-{name}:local",
                str(context),
            ],
            check=True,
        )
    print("CLI installation checks passed; no credentials supplied and no inference requested.")


if __name__ == "__main__":
    main()
