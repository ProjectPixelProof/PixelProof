# Build one audited checkpoint-aware candidate; repeat fresh episodes until time expires

Read `TASK_SPEC.toml`, `AGENTS.md`, campaign state, the candidate contract, one
seed as an interface example, and targeted mechanism/outcome memory. Design
exactly one new pixel-answerable question. Propose and criticize briefly, then
commit early enough to implement and verify it.

Create the provisional portfolio and candidate directory directly under
`/logs/artifacts/submission/`, then run the required `stage` audit command from
`AGENTS.md`. Build there, not in scratch. Run the authoritative public gate
before the final quarter. Every gate attempt is recorded automatically; record
`repair` before each repair pass after a failure.

Before yielding, run the public gate and then the envelope check. The latter
records a survivor commit only when the envelope is clean. Do not modify the
bundle after that point unless you record another repair and rerun both checks.

The host destroys the networked session, verifies the transaction independently,
reconciles its evidence ledger, updates structured memory, and starts a fresh
session while agent-time budget remains.
