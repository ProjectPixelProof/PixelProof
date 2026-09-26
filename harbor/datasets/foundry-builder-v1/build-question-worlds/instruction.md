# Build executable low-level visual-question worlds

Begin by reading `/workspace/AGENTS.md`, `/workspace/TASK_SPEC.toml`, and every
file under `/workspace/protocol/`. Then inspect the declared visible examples
under `/workspace/seeds/`.

Work autonomously within the frozen budget. Propose, implement, render, test, and
self-verify the required number of candidates. If
`/workspace/protocol/process-contract.md` exists, initialize and maintain its
append-only search record before proposing any candidate; preserve rejected and
failed build checkpoints as required there. Publish final survivors at
`/logs/artifacts/submission/`.

Do not inspect authentication data, environment variables, sibling trials,
external repositories, hidden verifier files, historical registries, or the
canonical checkout. Do not claim certification or human approval.
