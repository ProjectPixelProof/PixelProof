# Difficulty-feedback controller policy 0.9

`difficulty-feedback@0.9.0` is the effort-aware, lossless-response correction to
policy 0.8. It preserves the same reasoning-target branching, protected visual
verification, deterministic five-image panel, evaluator accuracy aggregation,
bounded retries, quarantine policy, and circuit breaker.

Two transport and visibility changes are prospective and versioned:

1. Evaluator responses that violate the structured rationale contract preserve
   their raw provider payload, response ID, served model, provider, token usage,
   content, and failure stage in the controller-only append-only ledger before a
   retry is considered. Opaque provider reasoning remains private.
2. For successfully scored samples, the next question designer sees an auxiliary
   `solver_effort` record: configured reasoning effort, completion tokens,
   provider-reported reasoning tokens, visible-output tokens, completion-budget
   utilization, and public-rationale length. These measurements are evidence
   about solver effort, not a hardness label, and must not be compared numerically
   across providers without qualification.

The corresponding prospective campaign uses a 40,960-token completion ceiling.
OpenRouter counts reasoning tokens inside completion tokens; high effort may
allocate most of the requested ceiling to reasoning, so the historical 8,192
ceiling could leave insufficient space for the required JSON answer. The larger
ceiling is a maximum, not a target, and the unchanged per-arm dollar cap remains
binding.

Policy 0.8 executions remain immutable. Completed 0.8 feedback panels retain
their original meaning; incomplete panels are not upgraded or relabeled.
