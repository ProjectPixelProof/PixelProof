# Audited checkpoint-aware transactional episodic session contract v0.7

- Session unit: exactly one candidate and one provider invocation.
- Campaign: unbounded episode count within the fixed agent-time budget.
- Context: fresh provider session plus controller-authored structured memory.
- Deadline: the exact session budget is visible in `TASK_SPEC.toml`.
- Durable staging: survivor code, evidence, and provisional portfolio are built
  under `/logs/artifacts/submission/`; scratch is disposable only.
- Audit staging: immediately after creating the provisional candidate tree, run
  `record_candidate_phase.py stage --candidate-id ID`.
- Public verification: every invocation of the authoritative public gate is
  timestamped automatically; call `record_candidate_phase.py repair` before
  repairing a failed public-gate attempt.
- Survivor commit: a passing public envelope check records the final commit
  timestamp automatically.
- Protected verification: after every session, including timed-out sessions
  that left a staged bundle.
- State transition: before the next session is materialized.
- Candidate policy: exactly one per session, append-only across the campaign.
- Canonical admission: human-only.

The campaign ends only when less than the minimum useful agent-time tail
remains, an attested provider policy stops it, or an operator cancels it.
