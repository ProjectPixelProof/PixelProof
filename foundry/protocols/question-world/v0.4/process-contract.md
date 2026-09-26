# Search-process contract v0.2

Protocol v0.4 retains the v0.3 append-only proposal/build/disposition trace and
adds explicit ranking artifacts.

Initialize:

```text
/logs/artifacts/process/
├── run.json
├── attempts.jsonl
├── runtime.json
├── rankings/
│   ├── pass_001.json
│   └── pass_002.json
└── candidates/<attempt_id>/
    ├── proposal.json
    ├── disposition.json
    └── checkpoint/
```

All identifiers use `^[a-z][a-z0-9_]{2,63}$`. JSON objects use UTF-8 and the
exact keys in `search-trace.schema.json`. Write ordinary JSON files atomically.
Append one complete object per line to `attempts.jsonl`; never edit earlier
lines.

Every attempt begins with one `proposed` event. v0.4 allows:

```text
proposed
  -> self_rejected | omitted_during_ranking | selected_for_build
selected_for_build
  -> self_rejected | build_started
build_started
  -> build_failed | self_check_failed | self_check_passed
self_check_failed
  -> build_started | self_rejected
self_check_passed
  -> self_rejected | omitted_during_ranking | submitted
```

`submitted`, `self_rejected`, `build_failed`, and `omitted_during_ranking` are
terminal. Every terminal attempt has a `disposition.json`. Every built attempt
has a non-empty `checkpoint/`; final bundles are copied from those checkpoints.

Every event contains exactly:

```json
{
  "schema_version": "search-event-0.1.0",
  "sequence": 1,
  "elapsed_seconds": 0.1,
  "attempt_id": "proposal_id",
  "state": "proposed",
  "reason_code": null,
  "evidence": ["candidates/proposal_id/proposal.json"]
}
```

`reason_code` is always `null` for non-terminal progress. Terminal events use a
stable identifier. `evidence` is never prose: every entry is a safe relative
path beneath `/logs/artifacts/process/` and must already exist when recorded.
The terminal disposition uses the same terminal state/reason and path-only
evidence. For example:

```json
{
  "schema_version": "search-disposition-0.1.0",
  "attempt_id": "proposal_id",
  "final_state": "omitted_during_ranking",
  "reason_code": "dominated_during_ranking",
  "evidence": [
    "candidates/proposal_id/proposal.json",
    "rankings/pass_002.json"
  ],
  "summary": "Removed after the second ranking pass."
}
```

`run.json` starts as `running` with the exact task budget and becomes
`completed` only after all terminal dispositions and `portfolio.json` are
durable. The adapter owns `runtime.json`; never overwrite it.

Each ranking file contains exactly:

```json
{
  "schema_version": "search-ranking-0.1.0",
  "pass_index": 1,
  "elapsed_seconds": 120.0,
  "considered_ids": ["proposal_id"],
  "survivor_ids": ["proposal_id"],
  "criteria": [
    "mechanical feasibility",
    "known-mechanism distance",
    "negative-memory avoidance"
  ]
}
```

Ranking passes are contiguous, time-ordered, and completed before the first
`selected_for_build` event. Every pass considers the full logged proposal pool;
later survivors must be a subset of earlier survivors. Final selected IDs must
be a subset of the final ranking survivors.

Before finalization, mechanically check:

- every event evidence string resolves to an existing process-relative path;
- every non-terminal `reason_code` is `null`;
- every proposal reaches exactly one terminal disposition;
- every `build_started` attempt has a non-empty checkpoint;
- submitted trace IDs exactly equal portfolio IDs.

Budget semantics:

- timeout and turns are hard caps;
- proposals and ranking passes are outcome requirements;
- build starts count distinct attempt IDs;
- final candidates remain bounded by `min_candidates` and `max_candidates`;
- no process metric controls scientific admission.
