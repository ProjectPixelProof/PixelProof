# Inner-builder brief: memory-aware executable question worlds v0.4

Search for low-level visual decisions that are answerable from pixels, admit an
independent image-only verifier, and add a genuinely different scientific
mechanism to the visible memory.

Before building, log the required proposal pool and perform at least
`min_ranking_passes` explicit ranking passes. Compare every survivor against
`memory/known_mechanisms.jsonl` and `memory/rejected_mechanisms.jsonl`. Preserve
the ranking artifacts and reject semantic re-skins before consuming build
budget.

This protocol is standalone: follow its exact manifest and process-event
examples. Do not invent shorter field names, put prose in an `evidence` array,
or attach a reason code to non-terminal events. Run the five process
finalization checks in `process-contract.md` before setting `run.json` to
`completed`.

Every submitted candidate must provide:

1. deterministic latent-scene sampling and RGB rendering;
2. analytic gold and a continuous or one-sided verifiability margin;
3. stable prompt families with constrained answer sets;
4. an independent `decision_from_image(image)` oracle;
5. an explicit latent-alias mode:
   - `tested_transforms`, with at least one nontrivial pixel-identical transform;
   - `not_applicable`, with an injectivity justification and declared label inputs;
6. a controlled mechanism fingerprint and contrast against the nearest known worlds;
7. an observable boundary, explicit quarantine, corruption tests, exact evidence
   verification, deterministic stress sweeps, and real gallery images.

Passing folders are not sufficient. The hidden evaluator independently retrieves
semantic neighbors, clusters siblings, probes the exact submitted evidence, and
stress-tests different generation seeds. Machine retrieval is diagnostic; only
blinded outer review and human admission can certify novelty.
