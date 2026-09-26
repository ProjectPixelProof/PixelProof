# Feedback example sets for model-feedback-steered generation

Feedback example sets A (`a/`) and B (`b/`) each contain ten world implementations and
fixed evaluator feedback. Each manifest records SHA-256 hashes for the world code,
sampled instances, and feedback files. The controller checks these before preparing a
run.

Historical rendered evidence and source bookkeeping are excluded, so release hashes
differ from the original campaign bundles. World answer logic and inverse algorithms are
unchanged. Feedback JSON retains recorded answers, which are visible to the agent. See
[release differences](../../docs/RELEASE_NOTES.md) and the [model-feedback-steered
generation walkthrough](../../examples/rq3/README.md).

The `deviations.md` files under `candidates/` are preserved authoring records. Some
mention selection notes or galleries omitted from the release. These references do not
indicate additional files needed to run the supplied implementations.
