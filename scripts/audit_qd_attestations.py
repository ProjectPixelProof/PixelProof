"""Backfill protected QD descriptors over preserved executable candidates.

This tool performs no provider calls and changes no candidate.  Each candidate
is characterized in a fresh subprocess so its oracle module cannot contaminate
the next candidate's import state.  The output is an audit record used to freeze
the feasible QD cell mask before a paid campaign.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ATTESTER = ROOT / "harbor/datasets/foundry-builder-v1/build-question-worlds/tests/qd_attestation.py"
DEFAULT_DOCKER_IMAGE = "question-foundry-candidate-replay-latex-tikz:0.1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _discover(controllers: list[Path]) -> list[Path]:
    candidates = []
    for controller in controllers:
        candidates.extend((controller / "candidates/working-seed").glob("episode-*"))
    return sorted({path.resolve() for path in candidates})


def _source_stress_config(controllers: list[Path]) -> tuple[int, int]:
    configurations = set()
    for controller in controllers:
        config_path = controller / "run-config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        verification = config.get("verification") or {}
        seeds = verification.get("oracle_stress_seeds")
        scenes = verification.get("oracle_stress_scenes_per_seed")
        if (
            not isinstance(seeds, list)
            or not seeds
            or isinstance(seeds[0], bool)
            or not isinstance(seeds[0], int)
            or isinstance(scenes, bool)
            or not isinstance(scenes, int)
        ):
            raise ValueError(f"{config_path} has no usable verifier stress configuration")
        configurations.add((seeds[0], scenes))
    if len(configurations) != 1:
        raise ValueError(
            "source controllers disagree on their first verifier stress seed or scene count"
        )
    return next(iter(configurations))


def audit(
    controllers: list[Path],
    *,
    limit: int | None = None,
    stress_seed: int,
    stress_scenes: int,
    runtime: str = "local",
    docker_image: str = DEFAULT_DOCKER_IMAGE,
    attestation_id: str = "pixel-oracle-support@0.1.0",
) -> dict:
    if runtime not in {"local", "docker"}:
        raise ValueError("runtime must be local or docker")
    image_id = None
    if runtime == "docker":
        inspected = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", docker_image],
            text=True,
            capture_output=True,
            check=False,
        )
        if inspected.returncode != 0 or not inspected.stdout.strip():
            raise ValueError(f"Docker calibration image is unavailable: {docker_image}")
        image_id = inspected.stdout.strip()
    candidates = _discover(controllers)
    if limit is not None:
        candidates = candidates[:limit]
    rows = []
    with tempfile.TemporaryDirectory(prefix="qd-attestation-audit-") as temp:
        temp_root = Path(temp)
        for index, candidate in enumerate(candidates, 1):
            world = candidate / "world"
            metadata_path = candidate / "candidate.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            row = {
                "candidate_id": metadata.get("candidate_id"),
                "question": metadata.get("question"),
                "candidate_path": str(candidate),
                "candidate_json_sha256": _sha256(metadata_path),
            }
            if runtime == "docker":
                trial_root = temp_root / f"{index:05d}"
                trial_root.mkdir()
                output = trial_root / "attestation.json"
                uid = os.getuid()
                gid = os.getgid()
                command = [
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
                    f"{uid}:{gid}",
                    "--env",
                    "HOME=/tmp",
                    "--env",
                    "PYTHONDONTWRITEBYTECODE=1",
                    "--env",
                    "PYTHONPATH=/candidate/world",
                    "--volume",
                    f"{candidate}:/candidate:ro",
                    "--volume",
                    f"{trial_root}:/output:rw",
                    "--volume",
                    f"{ATTESTER}:/protected/qd_attestation.py:ro",
                    "--workdir",
                    "/candidate",
                    docker_image,
                    "bash",
                    "-lc",
                    (
                        "python /candidate/world/generate.py "
                        f"--out /output/dataset --n {stress_scenes} --seed {stress_seed} && "
                        "python /protected/qd_attestation.py --candidate /candidate "
                        "--dataset /output/dataset --out /output/attestation.json "
                        f"--attestation-id {attestation_id}"
                    ),
                ]
                completed = subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=180,
                )
                if completed.returncode == 0 and output.is_file():
                    row["attestation"] = json.loads(output.read_text(encoding="utf-8"))
                else:
                    row["attestation"] = {
                        "status": "incomplete",
                        "descriptor": None,
                        "failures": [
                            f"Docker calibration exit {completed.returncode}: "
                            f"{(completed.stderr or completed.stdout)[-1000:]}"
                        ],
                    }
            else:
                dataset = temp_root / f"{index:05d}-stress"
                output = temp_root / f"{index:05d}.json"
                environment = dict(os.environ)
                environment["PYTHONPATH"] = str(world)
                generation = subprocess.run(
                    [
                        sys.executable,
                        str(world / "generate.py"),
                        "--out",
                        str(dataset),
                        "--n",
                        str(stress_scenes),
                        "--seed",
                        str(stress_seed),
                    ],
                    cwd=world,
                    env=environment,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=120,
                )
                command = [
                    sys.executable,
                    str(ATTESTER),
                    "--candidate",
                    str(candidate),
                    "--dataset",
                    str(dataset),
                    "--out",
                    str(output),
                    "--attestation-id",
                    attestation_id,
                ]
                if generation.returncode != 0:
                    row["attestation"] = {
                        "status": "incomplete",
                        "descriptor": None,
                        "failures": [
                            f"stress generation exit {generation.returncode}: "
                            f"{(generation.stderr or generation.stdout)[-1000:]}"
                        ],
                    }
                    rows.append(row)
                    print(
                        f"[{index}/{len(candidates)}] {row['candidate_id']}: "
                        f"{row['attestation'].get('status')} "
                        f"{row['attestation'].get('descriptor')}"
                    )
                    continue
                completed = subprocess.run(
                    command,
                    cwd=world,
                    env=environment,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=120,
                )
                if completed.returncode == 0 and output.is_file():
                    row["attestation"] = json.loads(output.read_text(encoding="utf-8"))
                else:
                    row["attestation"] = {
                        "status": "incomplete",
                        "descriptor": None,
                        "failures": [
                            f"attestation exit {completed.returncode}: "
                            f"{(completed.stderr or completed.stdout)[-1000:]}"
                        ],
                    }
            rows.append(row)
            print(
                f"[{index}/{len(candidates)}] {row['candidate_id']}: "
                f"{row['attestation'].get('status')} "
                f"{row['attestation'].get('descriptor')}"
            )
    complete = [row for row in rows if row["attestation"].get("status") == "complete"]
    occupancy = Counter(
        "|".join(f"{key}={value}" for key, value in row["attestation"]["descriptor"].items())
        for row in complete
    )
    return {
        "schema_version": (
            "quality-diversity-attestation-audit-0.5.0"
            if attestation_id == "pixel-oracle-support@0.5.0"
            else "quality-diversity-attestation-audit-0.4.0"
            if attestation_id == "pixel-oracle-support@0.4.0"
            else "quality-diversity-attestation-audit-0.3.0"
            if attestation_id == "pixel-oracle-support@0.3.0"
            else "quality-diversity-attestation-audit-0.2.0"
            if attestation_id == "pixel-oracle-support@0.2.0"
            else "quality-diversity-attestation-audit-0.1.0"
        ),
        "attestation_id": attestation_id,
        "network_used": False,
        "runtime": runtime,
        "docker_image": docker_image if runtime == "docker" else None,
        "docker_image_id": image_id,
        "stress_seed": stress_seed,
        "stress_scenes": stress_scenes,
        "controllers": [str(path.resolve()) for path in controllers],
        "candidate_count": len(rows),
        "complete_count": len(complete),
        "incomplete_count": len(rows) - len(complete),
        "completion_fraction": len(complete) / len(rows) if rows else 0.0,
        "occupancy": dict(sorted(occupancy.items())),
        "records": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", action="append", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stress-seed", type=int)
    parser.add_argument("--stress-scenes", type=int)
    parser.add_argument("--runtime", choices=("docker", "local"), default="docker")
    parser.add_argument("--docker-image", default=DEFAULT_DOCKER_IMAGE)
    parser.add_argument(
        "--attestation-id",
        choices=(
            "pixel-oracle-support@0.1.0",
            "pixel-oracle-support@0.2.0",
            "pixel-oracle-support@0.3.0",
            "pixel-oracle-support@0.4.0",
            "pixel-oracle-support@0.5.0",
        ),
        default="pixel-oracle-support@0.1.0",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    source_seed, source_scenes = _source_stress_config(args.controller)
    stress_seed = args.stress_seed if args.stress_seed is not None else source_seed
    stress_scenes = args.stress_scenes if args.stress_scenes is not None else source_scenes
    if stress_scenes < 6:
        raise SystemExit("--stress-scenes must be at least 6")
    result = audit(
        args.controller,
        limit=args.limit,
        stress_seed=stress_seed,
        stress_scenes=stress_scenes,
        runtime=args.runtime,
        docker_image=args.docker_image,
        attestation_id=args.attestation_id,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: result[key] for key in ("candidate_count", "complete_count", "occupancy")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
