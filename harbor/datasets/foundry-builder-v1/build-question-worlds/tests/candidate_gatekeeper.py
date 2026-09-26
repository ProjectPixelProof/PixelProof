"""Protected offline verifier for executable question-world portfolios."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image
from semantic_novelty import (
    candidate_text,
    fingerprint_similarity,
    load_jsonl,
    negative_neighbors,
    semantic_neighbors,
    validate_fingerprint,
)

BASE_MECHANICAL_GATES = (
    "safe_artifact",
    "contract",
    "executable",
    "deterministic",
    "analytic_consistency",
    "render_fidelity",
    "sign_blind_oracle",
    "oracle_independence",
    "latent_symmetry",
    "observable_boundary",
    "quarantine",
    "class_balance",
    "corruption_coverage",
    "gallery_coverage",
)
V04_MECHANICAL_GATES = BASE_MECHANICAL_GATES + ("exact_evidence",)
REQUIRED_FILES = (
    "candidate.json",
    "provenance.json",
    "deviations.md",
    "world/world.toml",
    "world/renderer.py",
    "world/oracle.py",
    "world/prompts.py",
    "world/generate.py",
    "world/verify.py",
    "tests/test_candidate.py",
    "evidence/manifest.jsonl",
    "evidence/self_check.json",
    "evidence/gallery/index.html",
)
FORBIDDEN_ORACLE_IMPORTS = {
    "renderer",
    "prompts",
    "generate",
    "verify",
    "analytic_gold",
    "scene",
}
MAX_FILES = 2000
MAX_BYTES = 256 * 1024 * 1024
QD_ATTESTER = Path(__file__).with_name("qd_attestation.py")


def _contains_concept(text: str, concept: str) -> bool:
    """Match one protected concept without exposing source artifacts to builders."""

    normalized_text = " ".join(re.findall(r"[a-z0-9]+", text.lower()))
    normalized_concept = " ".join(re.findall(r"[a-z0-9]+", concept.lower()))
    return bool(normalized_concept) and f" {normalized_concept} " in f" {normalized_text} "


def _protected_source_matches(metadata: dict, memory: list[dict]) -> list[dict]:
    """Return deterministic direct-descendant warnings from verifier-only memory.

    The rule is deliberately conservative and auditable: a row matches only when
    every curated operation-concept group is present and the ordinary mechanism
    fingerprint similarity clears that row's floor. It is a triage guard, not a
    claim that semantic novelty can be proved mechanically.
    """

    text = candidate_text(metadata)
    fingerprint = metadata.get("mechanism_fingerprint") or {}
    matches = []
    for row in memory:
        groups = row.get("clone_concepts") or []
        hits = [
            sorted(concept for concept in group if _contains_concept(text, concept))
            for group in groups
        ]
        if not groups or not all(hits):
            continue
        score = fingerprint_similarity(
            fingerprint,
            row.get("fingerprint") or {},
            left_text=text,
            right_text=str(row.get("summary", "")),
        )
        floor = float(row.get("fingerprint_floor", 0.0))
        if score >= floor:
            matches.append(
                {
                    "source_id": row.get("world_id"),
                    "score": score,
                    "fingerprint_floor": floor,
                    "matched_concepts": hits,
                }
            )
    return sorted(matches, key=lambda item: (-item["score"], str(item["source_id"])))


def _validate_protected_source_memory(rows: list[dict]) -> list[str]:
    failures = []
    seen = set()
    for index, row in enumerate(rows, start=1):
        label = f"protected source row {index}"
        source_id = row.get("world_id")
        if not isinstance(source_id, str) or not source_id:
            failures.append(f"{label} has no world_id")
        elif source_id in seen:
            failures.append(f"{label} repeats world_id {source_id!r}")
        else:
            seen.add(source_id)
        failures.extend(
            f"{label}: {failure}" for failure in validate_fingerprint(row.get("fingerprint"))
        )
        groups = row.get("clone_concepts")
        if (
            not isinstance(groups, list)
            or len(groups) < 3
            or any(
                not isinstance(group, list)
                or not group
                or any(not isinstance(value, str) or not value.strip() for value in group)
                for group in groups
            )
        ):
            failures.append(f"{label} clone_concepts must contain at least three string groups")
        floor = row.get("fingerprint_floor")
        if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not 0 <= floor <= 1:
            failures.append(f"{label} fingerprint_floor must be in [0, 1]")
    return failures


@dataclass
class CandidateResult:
    candidate_id: str
    gate_names: tuple[str, ...] = BASE_MECHANICAL_GATES
    gates: dict[str, bool] = field(init=False)
    gate_statuses: dict[str, str] = field(init=False)
    failures: dict[str, list[str]] = field(default_factory=dict)
    notes: dict = field(default_factory=dict)
    registry_distinct: bool = True
    semantic_review_ready: bool = False
    semantic_duplicate_risk: str = "not_evaluated"
    metadata: dict | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.gates = {name: True for name in self.gate_names}
        self.gate_statuses = {name: "pass" for name in self.gate_names}

    def fail(self, gate: str, message: str) -> None:
        self.gates[gate] = False
        self.gate_statuses[gate] = "fail"
        self.failures.setdefault(gate, []).append(message)

    def mark_not_applicable(self, gate: str) -> None:
        if self.gates.get(gate, False):
            self.gate_statuses[gate] = "not_applicable"

    @property
    def mechanically_eligible(self) -> bool:
        return all(self.gates.values())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hash(root: Path) -> str:
    files = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files[path.relative_to(root).as_posix()] = _sha256(path)
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _scan_safe(root: Path) -> list[str]:
    failures = []
    files = 0
    total_bytes = 0
    if root.is_symlink():
        return ["submission root is a symlink"]
    for path in root.rglob("*"):
        if path.is_symlink():
            failures.append(f"symlink is forbidden: {path.relative_to(root)}")
            continue
        if path.is_file():
            files += 1
            total_bytes += path.stat().st_size
        elif not path.is_dir():
            failures.append(f"special file is forbidden: {path.relative_to(root)}")
    if files > MAX_FILES:
        failures.append(f"artifact contains {files} files; maximum is {MAX_FILES}")
    if total_bytes > MAX_BYTES:
        failures.append(f"artifact is {total_bytes} bytes; maximum is {MAX_BYTES}")
    return failures


def _run(
    command: list[str],
    *,
    cwd: Path,
    timeout: int = 90,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    child_env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "HOME": "/tmp/verifier-home",
    }
    if env:
        child_env.update(env)
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=child_env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as expired:
        # A candidate that exceeds its execution budget must fail the gate that
        # invoked it, not abort the whole verifier. Crashing here would leave no
        # gate report at all and destroy the campaign that produced it.
        captured = expired.stdout or ""
        if isinstance(captured, bytes):
            captured = captured.decode("utf-8", "replace")
        return subprocess.CompletedProcess(
            args=command,
            returncode=124,
            stdout=captured,
            stderr=f"command exceeded the {timeout}s verifier budget and was killed",
        )


def _output_summary(result: subprocess.CompletedProcess) -> str:
    text = (result.stdout + "\n" + result.stderr).strip()
    return text[-2000:]


def _load_registry(path: Path) -> dict[tuple[str, str, str], list[str]]:
    cells: dict[tuple[str, str, str], list[str]] = {}
    if not path.is_file():
        return cells
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cell = tuple(row.get("cell") or ())
        if len(cell) == 3:
            cells.setdefault(cell, []).append(row.get("name", "unknown"))
    return cells


def _validate_metadata(
    path: Path,
    result: CandidateResult,
    *,
    protocol: str,
    mechanism_ids: set[str],
    negative_ids: set[str],
) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result.fail("contract", f"candidate.json is invalid: {exc}")
        return None
    required = {
        "schema_version",
        "candidate_id",
        "title",
        "question",
        "decision_type",
        "task_signature",
        "latent_z",
        "boundary",
        "oracle_independence",
        "deviations",
    }
    if protocol == "question-world@0.4.0":
        required.update(
            {
                "latent_alias",
                "mechanism_fingerprint",
                "semantic_contrast",
            }
        )
    if set(data) != required:
        result.fail("contract", f"candidate.json keys must be exactly {sorted(required)}")
    expected_schema = "0.4.0" if protocol == "question-world@0.4.0" else "0.2.0"
    if data.get("schema_version") != expected_schema:
        result.fail("contract", f"candidate schema_version must be {expected_schema}")
    if data.get("candidate_id") != result.candidate_id:
        result.fail("contract", "candidate_id does not match directory")
    signature = data.get("task_signature")
    if not isinstance(signature, dict):
        result.fail("contract", "task_signature must be an object")
    else:
        signature_keys = {"decision_var", "structure", "decision_type", "mechanism"}
        decision_types = {"yesno", "comparison", "selection", "threshold", "count_open"}
        identifier = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
        if set(signature) != signature_keys:
            result.fail(
                "contract",
                f"task_signature keys must be exactly {sorted(signature_keys)}",
            )
        for field in ("decision_var", "structure"):
            value = signature.get(field)
            if not isinstance(value, str) or identifier.fullmatch(value) is None:
                result.fail("contract", f"task_signature.{field} is invalid")
        if data.get("decision_type") not in decision_types:
            result.fail("contract", "candidate decision_type is outside the protocol enum")
        if signature.get("decision_type") not in decision_types:
            result.fail("contract", "task_signature decision_type is outside the protocol enum")
        elif signature.get("decision_type") != data.get("decision_type"):
            result.fail("contract", "task_signature decision_type disagrees with candidate")
        mechanism = signature.get("mechanism")
        if not isinstance(mechanism, str) or len(mechanism.strip()) < 20:
            result.fail("contract", "task_signature mechanism is missing or too short")
    boundary = data.get("boundary")
    if not isinstance(boundary, dict) or len(str(boundary.get("observable", "")).strip()) < 20:
        result.fail("observable_boundary", "boundary.observable is missing or too vague")
    if not isinstance(boundary, dict) or len(str(boundary.get("quarantine", "")).strip()) < 10:
        result.fail("quarantine", "boundary.quarantine is missing or too vague")
    independence = data.get("oracle_independence")
    if independence != {"image_only": True, "forbidden_imports_absent": True}:
        result.fail("oracle_independence", "oracle independence declaration is incomplete")
    if protocol == "question-world@0.4.0":
        alias = data.get("latent_alias")
        if not isinstance(alias, dict) or set(alias) != {
            "mode",
            "justification",
            "label_inputs",
        }:
            result.fail("contract", "latent_alias keys are invalid")
        else:
            if alias.get("mode") not in {"tested_transforms", "not_applicable"}:
                result.fail("contract", "latent_alias mode is invalid")
            if len(str(alias.get("justification", "")).strip()) < 40:
                result.fail("contract", "latent_alias justification is too short")
            label_inputs = alias.get("label_inputs")
            if (
                not isinstance(label_inputs, list)
                or not label_inputs
                or len(label_inputs) != len(set(label_inputs))
                or not all(isinstance(item, str) and item for item in label_inputs)
            ):
                result.fail("contract", "latent_alias label_inputs are invalid")
        for failure in validate_fingerprint(data.get("mechanism_fingerprint")):
            result.fail("contract", failure)
        contrast = data.get("semantic_contrast")
        if not isinstance(contrast, dict) or set(contrast) != {
            "closest_known_worlds",
            "known_negative_matches",
            "contrast",
            "expected_information_gain",
        }:
            result.fail("contract", "semantic_contrast keys are invalid")
        else:
            closest = contrast.get("closest_known_worlds")
            negatives = contrast.get("known_negative_matches")
            if (
                not isinstance(closest, list)
                or not 1 <= len(closest) <= 5
                or len(closest) != len(set(closest))
                or any(item not in mechanism_ids for item in closest)
            ):
                result.fail(
                    "contract",
                    "semantic_contrast closest_known_worlds are invalid or unknown",
                )
            if (
                not isinstance(negatives, list)
                or len(negatives) > 10
                or len(negatives) != len(set(negatives))
                or any(item not in negative_ids for item in negatives)
            ):
                result.fail(
                    "contract",
                    "semantic_contrast known_negative_matches are invalid or unknown",
                )
            if len(str(contrast.get("contrast", "")).strip()) < 80:
                result.fail("contract", "semantic contrast is too short")
            if len(str(contrast.get("expected_information_gain", "")).strip()) < 40:
                result.fail("contract", "expected_information_gain is too short")
    return data


def _check_oracle_ast(candidate: Path, result: CandidateResult) -> None:
    try:
        tree = ast.parse((candidate / "world/oracle.py").read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        result.fail("oracle_independence", f"cannot parse oracle.py: {exc}")
        return
    imports = set()
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "decision_from_image":
                target = node
    forbidden = sorted(imports & FORBIDDEN_ORACLE_IMPORTS)
    if forbidden:
        result.fail("oracle_independence", f"oracle imports forbidden modules {forbidden}")
    if target is None:
        result.fail("oracle_independence", "oracle has no decision_from_image function")
        return
    positional = [*target.args.posonlyargs, *target.args.args]
    if len(positional) != 1 or target.args.vararg or target.args.kwarg:
        result.fail(
            "oracle_independence",
            "decision_from_image must accept exactly one positional argument",
        )


def _mutate_answer(source: Path, destination: Path) -> str | None:
    shutil.copytree(source, destination)
    manifest = destination / "manifest.jsonl"
    try:
        rows = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        candidates = list(rows[0]["candidates"])
        answer = rows[0]["answer"]
        replacement = next(value for value in candidates if value != answer)
    except (OSError, json.JSONDecodeError, IndexError, KeyError, StopIteration, TypeError) as exc:
        return f"cannot construct wrong-answer mutation from malformed manifest: {exc}"
    rows[0]["answer"] = replacement
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return None


def _mutate_image(source: Path, destination: Path) -> str | None:
    shutil.copytree(source, destination)
    try:
        rows = [
            json.loads(line)
            for line in (destination / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        image_path = destination / rows[0]["image_path"]
        image = Image.open(image_path).convert("RGB")
    except (
        OSError,
        json.JSONDecodeError,
        IndexError,
        KeyError,
        TypeError,
    ) as exc:
        return f"cannot construct blank-image mutation from malformed manifest: {exc}"
    Image.new("RGB", image.size, "white").save(image_path)
    return None


def _validate_final_evidence(candidate: Path, result: CandidateResult) -> None:
    gallery = candidate / "evidence/gallery"
    pngs = sorted(gallery.glob("*.png"))
    if len(pngs) < 4:
        result.fail("gallery_coverage", "gallery must contain at least four PNG images")
    try:
        index = (gallery / "index.html").read_text(encoding="utf-8").lower()
        metadata = json.loads((candidate / "candidate.json").read_text(encoding="utf-8"))
        if metadata["question"].split()[0].lower() not in index:
            result.notes["gallery_index_note"] = "index does not repeat the primary question"
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        result.fail("gallery_coverage", f"gallery index cannot be inspected: {exc}")
    if not (candidate / "evidence/manifest.jsonl").is_file():
        result.fail("gallery_coverage", "final evidence manifest is missing")


def _apply_probe_result(
    *,
    probe_data: dict,
    result: CandidateResult,
    label: str,
    exact_evidence: bool,
) -> None:
    result.notes.setdefault("probes", {})[label] = {
        key: value for key, value in probe_data.items() if key != "failures"
    }
    alias_status = probe_data.get("latent_alias_status")
    if alias_status == "not_applicable":
        result.mark_not_applicable("latent_symmetry")
    for gate, failures in probe_data.get("failures", {}).items():
        for failure in failures:
            if gate in result.gates:
                result.fail(gate, f"{label}: {failure}")
            if exact_evidence and "exact_evidence" in result.gates:
                result.fail("exact_evidence", f"{gate}: {failure}")


def _probe_dataset(
    *,
    candidate: Path,
    dataset: Path,
    probe_script: Path,
    output: Path,
    result: CandidateResult,
    pythonpath: str,
    label: str,
    exact_evidence: bool = False,
) -> None:
    probe = _run(
        [
            sys.executable,
            str(probe_script),
            "--candidate",
            str(candidate),
            "--dataset",
            str(dataset),
            "--out",
            str(output),
        ],
        cwd=candidate,
        env={"PYTHONPATH": pythonpath},
    )
    if probe.returncode != 0 or not output.is_file():
        result.fail("executable", f"{label} protected probe failed: {_output_summary(probe)}")
        if exact_evidence and "exact_evidence" in result.gates:
            result.fail("exact_evidence", f"{label} protected probe did not complete")
        return
    try:
        probe_data = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result.fail("executable", f"{label} protected probe output is invalid: {exc}")
        if exact_evidence and "exact_evidence" in result.gates:
            result.fail("exact_evidence", f"{label} protected probe output is invalid")
        return
    _apply_probe_result(
        probe_data=probe_data,
        result=result,
        label=label,
        exact_evidence=exact_evidence,
    )


def verify_candidate(
    candidate: Path,
    candidate_id: str,
    registry: dict[tuple[str, str, str], list[str]],
    probe_script: Path,
    *,
    task_spec: dict,
    mechanism_memory: list[dict],
    negative_memory: list[dict],
    protected_source_memory: list[dict],
    quality_diversity_policy_id: str | None = None,
) -> CandidateResult:
    protocol = str(task_spec.get("protocol", ""))
    gate_names = (
        V04_MECHANICAL_GATES if protocol == "question-world@0.4.0" else BASE_MECHANICAL_GATES
    )
    result = CandidateResult(candidate_id, gate_names=gate_names)
    for failure in _scan_safe(candidate):
        result.fail("safe_artifact", failure)
    for relative in REQUIRED_FILES:
        path = candidate / relative
        if not path.is_file() or path.stat().st_size == 0:
            result.fail("contract", f"missing or empty {relative}")

    mechanism_ids = {str(item.get("world_id")) for item in mechanism_memory if item.get("world_id")}
    negative_ids = {
        str(item.get("proposal_id")) for item in negative_memory if item.get("proposal_id")
    }
    metadata = _validate_metadata(
        candidate / "candidate.json",
        result,
        protocol=protocol,
        mechanism_ids=mechanism_ids,
        negative_ids=negative_ids,
    )
    result.metadata = metadata
    _check_oracle_ast(candidate, result)
    _validate_final_evidence(candidate, result)
    if metadata and isinstance(metadata.get("task_signature"), dict):
        signature = metadata["task_signature"]
        cell = (
            signature.get("decision_var"),
            signature.get("structure"),
            signature.get("decision_type"),
        )
        collisions = registry.get(cell, [])
        if collisions:
            result.registry_distinct = False
            result.notes["registry_collisions"] = collisions
    if protocol == "question-world@0.4.0" and metadata:
        limit = int(task_spec.get("semantic_neighbor_limit", 5))
        neighbors = semantic_neighbors(metadata, mechanism_memory, limit=limit)
        rejected = negative_neighbors(metadata, negative_memory, limit=limit)
        protected_neighbors = semantic_neighbors(metadata, protected_source_memory, limit=limit)
        protected_matches = _protected_source_matches(metadata, protected_source_memory)
        top_score = float(neighbors[0]["score"]) if neighbors else 0.0
        threshold = float(task_spec.get("semantic_duplicate_threshold", 0.68))
        if top_score >= threshold or protected_matches:
            result.semantic_duplicate_risk = "high"
        elif top_score >= max(0.0, threshold - 0.15):
            result.semantic_duplicate_risk = "medium"
        else:
            result.semantic_duplicate_risk = "low"
        contrast = metadata.get("semantic_contrast") or {}
        declared = set(contrast.get("closest_known_worlds") or [])
        top_id = neighbors[0].get("world_id") if neighbors else None
        addressed = bool(top_id and top_id in declared)
        result.semantic_review_ready = addressed and not protected_matches
        result.notes["semantic"] = {
            "known_neighbors": neighbors,
            "negative_neighbors": rejected,
            "protected_source_neighbors": protected_neighbors,
            "protected_source_clone_matches": protected_matches,
            "top_known_neighbor_addressed": addressed,
            "declared_known_neighbors": sorted(declared),
            "duplicate_threshold": threshold,
        }

    world = candidate / "world"
    pythonpath = str(world)
    tests = _run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
        cwd=candidate,
        env={"PYTHONPATH": pythonpath},
    )
    if tests.returncode != 0:
        result.fail("executable", f"candidate tests failed: {_output_summary(tests)}")

    if protocol == "question-world@0.4.0":
        exact_verify = _run(
            [
                sys.executable,
                str(candidate / "world/verify.py"),
                "--dataset",
                str(candidate / "evidence"),
            ],
            cwd=world,
            env={"PYTHONPATH": pythonpath},
        )
        if exact_verify.returncode != 0:
            result.fail(
                "exact_evidence",
                f"candidate verifier rejected submitted evidence: {_output_summary(exact_verify)}",
            )

    with tempfile.TemporaryDirectory(prefix=f"verify-{candidate_id}-") as temp:
        temp_root = Path(temp)
        run_a = temp_root / "run-a"
        run_b = temp_root / "run-b"
        generate = candidate / "world/generate.py"
        stress_seeds = (
            [int(value) for value in task_spec.get("oracle_stress_seeds", [])]
            if protocol == "question-world@0.4.0"
            else [2027]
        )
        if not stress_seeds:
            stress_seeds = [2027]
        scene_count = (
            int(task_spec.get("oracle_stress_scenes_per_seed", 24))
            if protocol == "question-world@0.4.0"
            else 24
        )
        command = [
            sys.executable,
            str(generate),
            "--out",
            str(run_a),
            "--n",
            str(scene_count),
            "--seed",
            str(stress_seeds[0]),
        ]
        generated_a = _run(command, cwd=world, env={"PYTHONPATH": pythonpath})
        command[command.index(str(run_a))] = str(run_b)
        generated_b = _run(command, cwd=world, env={"PYTHONPATH": pythonpath})
        if generated_a.returncode != 0 or generated_b.returncode != 0:
            result.fail(
                "executable",
                "generator failed: "
                + _output_summary(generated_a)
                + "\n"
                + _output_summary(generated_b),
            )
            return result
        if _tree_hash(run_a) != _tree_hash(run_b):
            result.fail("deterministic", "identical generation inputs changed output hashes")

        verify_command = [
            sys.executable,
            str(candidate / "world/verify.py"),
            "--dataset",
            str(run_a),
        ]
        clean = _run(verify_command, cwd=world, env={"PYTHONPATH": pythonpath})
        if clean.returncode != 0:
            result.fail(
                "executable", f"clean candidate verification failed: {_output_summary(clean)}"
            )

        _probe_dataset(
            candidate=candidate,
            dataset=run_a,
            probe_script=probe_script,
            output=temp_root / f"probe-stress-{stress_seeds[0]}.json",
            result=result,
            pythonpath=pythonpath,
            label=f"stress_seed_{stress_seeds[0]}",
        )
        if protocol == "question-world@0.4.0":
            _probe_dataset(
                candidate=candidate,
                dataset=candidate / "evidence",
                probe_script=probe_script,
                output=temp_root / "probe-exact-evidence.json",
                result=result,
                pythonpath=pythonpath,
                label="exact_evidence",
                exact_evidence=True,
            )
            for seed in stress_seeds[1:]:
                stress = temp_root / f"stress-{seed}"
                stress_generate = _run(
                    [
                        sys.executable,
                        str(generate),
                        "--out",
                        str(stress),
                        "--n",
                        str(scene_count),
                        "--seed",
                        str(seed),
                    ],
                    cwd=world,
                    env={"PYTHONPATH": pythonpath},
                )
                if stress_generate.returncode != 0:
                    result.fail(
                        "executable",
                        f"stress generator seed {seed} failed: {_output_summary(stress_generate)}",
                    )
                    continue
                _probe_dataset(
                    candidate=candidate,
                    dataset=stress,
                    probe_script=probe_script,
                    output=temp_root / f"probe-stress-{seed}.json",
                    result=result,
                    pythonpath=pythonpath,
                    label=f"stress_seed_{seed}",
                )

        wrong_answer = temp_root / "wrong-answer"
        blank_image = temp_root / "blank-image"
        mutation_errors = {
            "wrong answer": _mutate_answer(run_a, wrong_answer),
            "blank image": _mutate_image(run_a, blank_image),
        }
        for label, mutated in (("wrong answer", wrong_answer), ("blank image", blank_image)):
            if mutation_errors[label]:
                result.fail("corruption_coverage", mutation_errors[label])
                continue
            verify_command[-1] = str(mutated)
            mutation = _run(verify_command, cwd=world, env={"PYTHONPATH": pythonpath})
            if mutation.returncode == 0:
                result.fail(
                    "corruption_coverage",
                    f"candidate verifier accepted the {label} mutation",
                )

        # QD descriptors are recomputed only for mechanically eligible worlds,
        # after every ordinary protected gate has completed. They do not alter
        # scientific validity: an incomplete descriptor is an archive-policy
        # outcome, never a mechanical failure. The first deterministic stress
        # dataset prevents the builder from choosing archive-placement scenes.
        if (
            quality_diversity_policy_id is not None
            and result.mechanically_eligible
            and QD_ATTESTER.is_file()
        ):
            attestation_id = (
                "pixel-oracle-support@0.5.0"
                if quality_diversity_policy_id
                in {
                    "quality-diversity@0.6.0",
                    "quality-diversity@0.7.0",
                    "quality-diversity@0.8.0",
                }
                else "pixel-oracle-support@0.4.0"
                if quality_diversity_policy_id == "quality-diversity@0.5.0"
                else "pixel-oracle-support@0.3.0"
                if quality_diversity_policy_id == "quality-diversity@0.4.0"
                else "pixel-oracle-support@0.2.0"
                if quality_diversity_policy_id == "quality-diversity@0.3.0"
                else "pixel-oracle-support@0.1.0"
            )
            attestation_path = temp_root / "quality-diversity-attestation.json"
            attestation = _run(
                [
                    sys.executable,
                    str(QD_ATTESTER),
                    "--candidate",
                    str(candidate),
                    "--dataset",
                    str(run_a),
                    "--out",
                    str(attestation_path),
                    "--attestation-id",
                    attestation_id,
                ],
                cwd=world,
                timeout=(
                    180
                    if quality_diversity_policy_id
                    in {
                        "quality-diversity@0.4.0",
                        "quality-diversity@0.5.0",
                        "quality-diversity@0.6.0",
                        "quality-diversity@0.7.0",
                        "quality-diversity@0.8.0",
                    }
                    else 90
                ),
                env={"PYTHONPATH": pythonpath},
            )
            try:
                if attestation.returncode != 0 or not attestation_path.is_file():
                    raise ValueError(
                        f"attester exit {attestation.returncode}: {_output_summary(attestation)}"
                    )
                result.notes["quality_diversity_attestation"] = json.loads(
                    attestation_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                result.notes["quality_diversity_attestation"] = {
                    "schema_version": (
                        "quality-diversity-attestation-0.5.0"
                        if quality_diversity_policy_id
                        in {
                            "quality-diversity@0.6.0",
                            "quality-diversity@0.7.0",
                            "quality-diversity@0.8.0",
                        }
                        else "quality-diversity-attestation-0.4.0"
                        if quality_diversity_policy_id == "quality-diversity@0.5.0"
                        else "quality-diversity-attestation-0.3.0"
                        if quality_diversity_policy_id == "quality-diversity@0.4.0"
                        else "quality-diversity-attestation-0.2.0"
                        if quality_diversity_policy_id == "quality-diversity@0.3.0"
                        else "quality-diversity-attestation-0.1.0"
                    ),
                    "attestation_id": attestation_id,
                    "status": "incomplete",
                    "descriptor_source": (
                        "protected_scene_relative_pixel_oracle_ablation_v5"
                        if quality_diversity_policy_id
                        in {
                            "quality-diversity@0.6.0",
                            "quality-diversity@0.7.0",
                            "quality-diversity@0.8.0",
                        }
                        else "protected_scene_relative_pixel_oracle_ablation_v4"
                        if quality_diversity_policy_id == "quality-diversity@0.5.0"
                        else "protected_scene_relative_pixel_oracle_ablation_v3"
                        if quality_diversity_policy_id == "quality-diversity@0.4.0"
                        else "protected_scene_relative_pixel_oracle_ablation"
                        if quality_diversity_policy_id == "quality-diversity@0.3.0"
                        else "protected_offline_pixel_oracle_ablation"
                    ),
                    "descriptor": None,
                    "behavioral_vector": None,
                    "failures": [f"{type(exc).__name__}: {str(exc)[:500]}"],
                }

    return result


def _write_outputs(
    *,
    report_path: Path,
    reward_path: Path,
    task_spec: dict,
    portfolio_valid: bool,
    results: list[CandidateResult],
    infrastructure_failures: list[str],
) -> None:
    count = len(results)
    eligible = sum(item.mechanically_eligible for item in results)
    distinct = sum(item.registry_distinct for item in results)
    review_ready = sum(item.semantic_review_ready for item in results)
    high_duplicate_risk = sum(item.semantic_duplicate_risk == "high" for item in results)
    gate_names = sorted({gate for item in results for gate in item.gates})
    gate_rates = {
        gate: (sum(item.gates.get(gate, False) for item in results) / count if count else 0.0)
        for gate in gate_names
    }
    protocol = task_spec.get("protocol")
    report = {
        "schema_version": "0.4.0" if protocol == "question-world@0.4.0" else "0.2.0",
        "campaign_id": task_spec.get("campaign_id"),
        "mode": task_spec.get("mode"),
        "portfolio_valid": portfolio_valid,
        "infrastructure_failures": infrastructure_failures,
        "candidate_results": [
            {
                "candidate_id": item.candidate_id,
                "gates": item.gates,
                "gate_statuses": item.gate_statuses,
                "mechanically_eligible": item.mechanically_eligible,
                "registry_distinct": item.registry_distinct,
                "semantic_review_ready": item.semantic_review_ready,
                "semantic_duplicate_risk": item.semantic_duplicate_risk,
                "failures": item.failures,
                "notes": item.notes,
            }
            for item in results
        ],
        "summary": {
            "candidate_count": count,
            "mechanically_eligible_count": eligible,
            "registry_distinct_count": distinct,
            "semantic_review_ready_count": review_ready,
            "high_semantic_duplicate_risk_count": high_duplicate_risk,
            "gate_pass_rates": gate_rates,
            "latent_alias_status_counts": {
                status: sum(item.gate_statuses.get("latent_symmetry") == status for item in results)
                for status in ("pass", "not_applicable", "fail")
            },
        },
    }
    reward = {
        "submission_valid": float(portfolio_valid and not infrastructure_failures),
        "mechanically_eligible_fraction": eligible / count if count else 0.0,
        "all_mechanically_eligible": float(bool(results) and eligible == count),
        "registry_distinct_fraction": distinct / count if count else 0.0,
    }
    if protocol == "question-world@0.4.0":
        reward["semantic_review_ready_fraction"] = review_ready / count if count else 0.0
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    reward_path.write_text(json.dumps(reward, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--task-spec", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--mechanism-memory", type=Path)
    parser.add_argument("--negative-memory", type=Path)
    parser.add_argument("--profile-source-memory", type=Path)
    parser.add_argument("--quality-diversity-policy", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reward", type=Path, required=True)
    args = parser.parse_args()

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.reward.parent.mkdir(parents=True, exist_ok=True)
    infrastructure_failures = []
    results: list[CandidateResult] = []
    portfolio_valid = True
    try:
        task_spec = tomllib.loads(args.task_spec.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        task_spec = {}
        infrastructure_failures.append(f"task spec is invalid: {exc}")
    protocol = task_spec.get("protocol")
    mechanism_memory = []
    negative_memory = []
    protected_source_memory = []
    quality_diversity_policy_id = None
    if protocol == "question-world@0.4.0":
        if args.mechanism_memory is None or args.negative_memory is None:
            infrastructure_failures.append("v0.4 semantic memory arguments are missing")
        else:
            try:
                mechanism_memory = load_jsonl(args.mechanism_memory)
                negative_memory = load_jsonl(args.negative_memory)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                infrastructure_failures.append(f"semantic memory is invalid: {exc}")
        if args.profile_source_memory is not None:
            try:
                protected_source_memory = load_jsonl(args.profile_source_memory)
                infrastructure_failures.extend(
                    _validate_protected_source_memory(protected_source_memory)
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                infrastructure_failures.append(f"protected profile source memory is invalid: {exc}")
        if args.quality_diversity_policy is not None:
            try:
                policy = json.loads(args.quality_diversity_policy.read_text(encoding="utf-8"))
                if policy.get("id") not in {
                    "quality-diversity@0.2.0",
                    "quality-diversity@0.3.0",
                    "quality-diversity@0.4.0",
                    "quality-diversity@0.5.0",
                    "quality-diversity@0.6.0",
                    "quality-diversity@0.7.0",
                    "quality-diversity@0.8.0",
                }:
                    raise ValueError("unsupported verifier QD policy")
                quality_diversity_policy_id = policy["id"]
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                infrastructure_failures.append(f"quality-diversity policy marker is invalid: {exc}")

    submission = args.submission
    safety_failures = _scan_safe(submission) if submission.is_dir() else ["submission missing"]
    if safety_failures:
        portfolio_valid = False
        infrastructure_failures.extend(safety_failures)
    try:
        portfolio = json.loads((submission / "portfolio.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        portfolio = {}
        portfolio_valid = False
        infrastructure_failures.append(f"portfolio.json is invalid: {exc}")

    expected_keys = {"schema_version", "candidate_ids", "deviations"}
    expected_portfolio_schema = "0.4.0" if protocol == "question-world@0.4.0" else "0.2.0"
    if (
        set(portfolio) != expected_keys
        or portfolio.get("schema_version") != expected_portfolio_schema
    ):
        portfolio_valid = False
        infrastructure_failures.append("portfolio schema or keys are invalid")
    candidate_ids = portfolio.get("candidate_ids", [])
    if not isinstance(candidate_ids, list) or not all(
        isinstance(item, str) for item in candidate_ids
    ):
        candidate_ids = []
        portfolio_valid = False
        infrastructure_failures.append("portfolio candidate_ids must be a string list")
    if len(candidate_ids) != len(set(candidate_ids)):
        portfolio_valid = False
        infrastructure_failures.append("portfolio candidate_ids contain duplicates")
    minimum = int(task_spec.get("min_candidates", 1))
    candidate_policy = task_spec.get("candidate_limit_policy")
    if candidate_policy == "time_bounded_append_only":
        if len(candidate_ids) < minimum:
            portfolio_valid = False
            infrastructure_failures.append(
                f"candidate count {len(candidate_ids)} is below minimum {minimum}"
            )
    elif candidate_policy == "exactly_one_per_session":
        if len(candidate_ids) != 1:
            portfolio_valid = False
            infrastructure_failures.append("episodic session must contain exactly one candidate")
    else:
        maximum = int(task_spec.get("max_candidates", 1))
        if not minimum <= len(candidate_ids) <= maximum:
            portfolio_valid = False
            infrastructure_failures.append(
                f"candidate count {len(candidate_ids)} is outside [{minimum}, {maximum}]"
            )

    candidates_root = submission / "candidates"
    actual_ids = (
        sorted(path.name for path in candidates_root.iterdir() if path.is_dir())
        if candidates_root.is_dir()
        else []
    )
    if sorted(candidate_ids) != actual_ids:
        portfolio_valid = False
        infrastructure_failures.append(
            f"portfolio IDs {sorted(candidate_ids)} differ from directories {actual_ids}"
        )

    registry = _load_registry(args.registry)
    probe_script = Path(__file__).with_name("probe_candidate.py")
    for candidate_id in candidate_ids:
        candidate = candidates_root / candidate_id
        results.append(
            verify_candidate(
                candidate,
                candidate_id,
                registry,
                probe_script,
                task_spec=task_spec,
                mechanism_memory=mechanism_memory,
                negative_memory=negative_memory,
                protected_source_memory=protected_source_memory,
                quality_diversity_policy_id=quality_diversity_policy_id,
            )
        )

    if protocol == "question-world@0.4.0":
        threshold = float(task_spec.get("semantic_duplicate_threshold", 0.68))
        for index, left in enumerate(results):
            if not left.metadata:
                continue
            siblings = []
            for right in results[index + 1 :]:
                if not right.metadata:
                    continue
                score = fingerprint_similarity(
                    left.metadata.get("mechanism_fingerprint") or {},
                    right.metadata.get("mechanism_fingerprint") or {},
                    left_text=candidate_text(left.metadata),
                    right_text=candidate_text(right.metadata),
                )
                siblings.append({"candidate_id": right.candidate_id, "score": score})
                right.notes.setdefault("semantic", {}).setdefault("sibling_neighbors", []).append(
                    {"candidate_id": left.candidate_id, "score": score}
                )
                if score >= threshold:
                    left.semantic_duplicate_risk = "high"
                    right.semantic_duplicate_risk = "high"
            left.notes.setdefault("semantic", {}).setdefault("sibling_neighbors", []).extend(
                siblings
            )

    _write_outputs(
        report_path=args.report,
        reward_path=args.reward,
        task_spec=task_spec,
        portfolio_valid=portfolio_valid,
        results=results,
        infrastructure_failures=infrastructure_failures,
    )
    print(
        f"verified {len(results)} candidate(s); "
        f"mechanically eligible={sum(item.mechanically_eligible for item in results)}"
    )


if __name__ == "__main__":
    main()
