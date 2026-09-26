"""Render, test and inverse-verify all worlds in both feedback example sets offline in Docker."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from question_foundry.difficulty_feedback import _render_candidate
from question_foundry.render_runtime import LATEX_TIKZ_RENDER_RUNTIME, LATEX_TIKZ_REPLAY_IMAGE
from scripts.release import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/starting-world-check")
    parser.add_argument("--scenes", type=int, default=32)
    parser.add_argument(
        "--only", help="check one world ID instead of both complete feedback example sets"
    )
    args = parser.parse_args()
    if args.scenes < 4:
        parser.error("--scenes must be at least 4 for these world contracts")
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError("choose a new output path")
    out.mkdir(parents=True)
    results = []
    for collection in ("a", "b"):
        manifest = json.loads(
            (ROOT / f"inputs/starting_collections/{collection}/manifest.json").read_text()
        )
        for record in manifest["records"]:
            candidate_id = record["candidate_id"]
            if args.only and args.only != candidate_id:
                continue
            print(f"{collection}/{candidate_id}", flush=True)
            try:
                staged = out / collection / candidate_id / "candidate"
                shutil.copytree(ROOT / record["candidate_path"], staged)
                subprocess.run(
                    [
                        "docker",
                        "run",
                        "--rm",
                        "--network",
                        "none",
                        "--cpus",
                        "2",
                        "--memory",
                        "4g",
                        "--pids-limit",
                        "256",
                        "--read-only",
                        "--tmpfs",
                        "/tmp:rw,nosuid,size=512m",
                        "--user",
                        f"{os.getuid()}:{os.getgid()}",
                        "--env",
                        "HOME=/tmp",
                        "--env",
                        "PYTHONDONTWRITEBYTECODE=1",
                        "--volume",
                        f"{staged}:/candidate:rw",
                        "--workdir",
                        "/candidate",
                        LATEX_TIKZ_REPLAY_IMAGE,
                        "python",
                        "/candidate/world/generate.py",
                        "--out",
                        "/candidate/evidence",
                        "--n",
                        "32",
                        "--seed",
                        "101",
                    ],
                    check=True,
                )
                result = _render_candidate(
                    candidate_root=staged,
                    dataset=out / collection / candidate_id / "replay" / "dataset",
                    render_runtime=LATEX_TIKZ_RENDER_RUNTIME,
                    scenes=args.scenes,
                    seed=101,
                )
                results.append(
                    {"collection": collection, "candidate_id": candidate_id, "result": result}
                )
            except (subprocess.CalledProcessError, ValueError, OSError) as exc:
                results.append(
                    {
                        "collection": collection,
                        "candidate_id": candidate_id,
                        "error": type(exc).__name__,
                    }
                )
                print(f"FAILED: {collection}/{candidate_id}: {type(exc).__name__}", flush=True)
    (out / "summary.json").write_text(
        json.dumps({"provider_calls": 0, "worlds": results}, indent=2) + "\n"
    )
    if not results:
        raise ValueError("no candidate matched --only")
    failed = sum("error" in item for item in results)
    print(f"Checked {len(results)} starting worlds; failures={failed}; no provider calls")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
