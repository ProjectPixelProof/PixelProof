"""Record a validated staging or repair milestone for one candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

from phase_audit import DEFAULT_LEDGER, append_event, portfolio_ids, read_events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("stage", "repair"))
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path("/logs/artifacts/submission"),
    )
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args()

    ids = portfolio_ids(args.submission)
    if ids != [args.candidate_id]:
        raise SystemExit("phase candidate must be the sole provisional portfolio entry")
    if not (args.submission / "candidates" / args.candidate_id).is_dir():
        raise SystemExit("candidate directory must exist before phase recording")
    prior = read_events(args.ledger)
    if args.phase == "stage":
        if any(event["kind"] == "candidate_staged" for event in prior):
            raise SystemExit("candidate staging may be recorded only once")
        kind = "candidate_staged"
    else:
        if not prior or not (
            prior[-1]["kind"] == "public_gate_finished" and prior[-1]["outcome"] == "failed"
        ):
            raise SystemExit("repair requires a new failed public-gate attempt")
        kind = "repair_started"
    event = append_event(
        kind,
        candidate_ids=ids,
        details={"source": "validated_agent_visible_phase_tool"},
        path=args.ledger,
    )
    print(f"OK: recorded {event['kind']} sequence {event['sequence']}")


if __name__ == "__main__":
    main()
