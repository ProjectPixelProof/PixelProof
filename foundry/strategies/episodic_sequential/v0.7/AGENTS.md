# Audited checkpoint-aware transactional episodic inner-builder contract

This fresh session owns exactly one candidate transaction. Work only in
`/workspace` and `/logs/artifacts`. Read `timeout_seconds` from
`TASK_SPEC.toml` and budget the work against that explicit session deadline.

After brief proposal and criticism, select one survivor. Immediately create a
provisional `portfolio.json` and its candidate tree under
`/logs/artifacts/submission/`, then record the real staging observation:

```text
python /workspace/tools/record_candidate_phase.py stage --candidate-id YOUR_ID
```

Implement, render, test, and repair in that durable tree. `/workspace/scratch`
is disposable only. Use no more than the first quarter for selection, publish
an executable skeleton before halfway, run the first public gate before the
final quarter, and reserve the final quarter for repair and completion.

The public gate records every attempt automatically. If it fails, record the
start of the corresponding repair before editing:

```text
python /workspace/tools/record_candidate_phase.py repair --candidate-id YOUR_ID
python /workspace/tools/run_public_candidate_gate.py
```

Before yielding, both commands must pass in this order:

```text
python /workspace/tools/run_public_candidate_gate.py
python /workspace/tools/check_submission_envelope.py
```

The passing envelope check records the survivor-commit timestamp. Do not edit
the bundle afterward; if hardening is necessary, treat it as another repair and
rerun both checks. Do not create a second candidate. Do not claim protected
eligibility, novelty, human approval, or canonical admission.
