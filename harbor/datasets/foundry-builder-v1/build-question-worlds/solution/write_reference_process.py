"""Write a deterministic traced process for the Harbor Oracle fixture."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

PROCESS = Path("/logs/artifacts/process")
TASK_SPEC = Path("/workspace/TASK_SPEC.toml")


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def proposal(attempt_id: str, index: int) -> dict:
    return {
        "schema_version": "search-proposal-0.1.0",
        "attempt_id": attempt_id,
        "title": f"Oracle fixture proposal {index:02d}",
        "question": "Does the visible reference marker satisfy its declared relation?",
        "task_signature": {
            "decision_var": f"fixture_relation_{index:02d}",
            "structure": "oracle_fixture",
            "decision_type": "yesno",
            "mechanism": "Deterministic packaging-only mechanism for verifier smoke coverage.",
        },
        "rationale": "This proposal exists only to exercise the traced Harbor Oracle fixture.",
        "anticipated_risks": ["not a scientific proposal"],
    }


def disposition(attempt_id: str, state: str, reason: str, evidence: list[str]) -> dict:
    return {
        "schema_version": "search-disposition-0.1.0",
        "attempt_id": attempt_id,
        "final_state": state,
        "reason_code": reason,
        "evidence": evidence,
        "summary": f"Oracle fixture {attempt_id} terminates as {state}.",
    }


def event(
    sequence: int,
    attempt_id: str,
    state: str,
    evidence: list[str],
    reason_code: str | None = None,
) -> dict:
    return {
        "schema_version": "search-event-0.1.0",
        "sequence": sequence,
        "elapsed_seconds": float(sequence),
        "attempt_id": attempt_id,
        "state": state,
        "reason_code": reason_code,
        "evidence": evidence,
    }


def main() -> None:
    task = tomllib.loads(TASK_SPEC.read_text(encoding="utf-8"))
    minimum = int(task["min_logged_proposals"])
    attempt_ids = ["reference_marker"] + [
        f"fixture_rejected_{index:02d}" for index in range(1, minimum)
    ]
    budget_keys = [
        "timeout_seconds",
        "min_logged_proposals",
        "min_candidates",
    ]
    if task.get("candidate_limit_policy") == "time_bounded_append_only":
        budget_keys.extend(
            (
                "min_ranking_passes",
                "candidate_limit_policy",
                "proposal_refresh_policy",
            )
        )
    else:
        budget_keys.extend(("max_turns", "max_build_starts", "max_candidates"))
    budget = {key: task[key] for key in budget_keys}
    write_json(
        PROCESS / "run.json",
        {
            "schema_version": "search-trace-0.1.0",
            "campaign_id": task["campaign_id"],
            "status": "running",
            "budget": budget,
            "deviations": [
                "Oracle smoke fixture, not an agent search or scientific proposal.",
            ],
        },
    )

    events = []
    sequence = 0
    for index, attempt_id in enumerate(attempt_ids):
        proposal_path = f"candidates/{attempt_id}/proposal.json"
        write_json(PROCESS / proposal_path, proposal(attempt_id, index))
        sequence += 1
        events.append(event(sequence, attempt_id, "proposed", [proposal_path]))

    if task.get("protocol") == "question-world@0.4.0":
        survivors = list(attempt_ids)
        ranking_root = PROCESS / "rankings"
        for pass_index in range(1, int(task["min_ranking_passes"]) + 1):
            if pass_index == int(task["min_ranking_passes"]):
                survivors = ["reference_marker"]
            write_json(
                ranking_root / f"pass_{pass_index:03d}.json",
                {
                    "schema_version": "search-ranking-0.1.0",
                    "pass_index": pass_index,
                    "elapsed_seconds": float(sequence) + pass_index / 10,
                    "considered_ids": attempt_ids,
                    "survivor_ids": survivors,
                    "criteria": [
                        "mechanical feasibility",
                        "known-mechanism distance",
                        "negative-memory avoidance",
                    ],
                },
            )

    for attempt_id in attempt_ids[1:]:
        proposal_path = f"candidates/{attempt_id}/proposal.json"
        disposition_path = f"candidates/{attempt_id}/disposition.json"
        write_json(
            PROCESS / disposition_path,
            disposition(
                attempt_id,
                "self_rejected",
                "oracle_fixture_not_selected",
                [proposal_path],
            ),
        )
        sequence += 1
        events.append(
            event(
                sequence,
                attempt_id,
                "self_rejected",
                [disposition_path],
                "oracle_fixture_not_selected",
            )
        )

    checkpoint = PROCESS / "candidates/reference_marker/checkpoint"
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "notes.txt").write_text(
        "Reference marker selected by the deterministic Oracle fixture.\n",
        encoding="utf-8",
    )
    reference_disposition = "candidates/reference_marker/disposition.json"
    write_json(
        PROCESS / reference_disposition,
        disposition(
            "reference_marker",
            "submitted",
            "submitted_survivor",
            ["candidates/reference_marker/checkpoint/notes.txt"],
        ),
    )
    for state, evidence, reason in (
        ("selected_for_build", [], None),
        (
            "build_started",
            ["candidates/reference_marker/checkpoint/notes.txt"],
            None,
        ),
        (
            "self_check_passed",
            ["candidates/reference_marker/checkpoint/notes.txt"],
            None,
        ),
        ("submitted", [reference_disposition], "submitted_survivor"),
    ):
        sequence += 1
        events.append(event(sequence, "reference_marker", state, evidence, reason))

    (PROCESS / "attempts.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in events),
        encoding="utf-8",
    )
    write_json(
        PROCESS / "runtime.json",
        {
            "schema_version": "harbor-runtime-0.1.0",
            "exit_code": 0,
            "termination": "exit",
        },
    )
    write_json(
        PROCESS / "run.json",
        {
            "schema_version": "search-trace-0.1.0",
            "campaign_id": task["campaign_id"],
            "status": "completed",
            "budget": budget,
            "deviations": [
                "Oracle smoke fixture, not an agent search or scientific proposal.",
            ],
        },
    )


if __name__ == "__main__":
    main()
