# Training on generated questions: release scope

The paper describes supervised fine-tuning and its results in **§4.7**. It tests
whether data from generated worlds improves open-weight models on held-out worlds and
external benchmarks.

This repository releases the three generation experiments (`examples/rq1` to
`examples/rq3`). It does **not** include the fine-tuning experiment's training data,
dataset splits, LoRA training pipeline, checkpoints, or code for evaluating external
benchmarks. There is therefore no run script in this directory. The generation
examples elsewhere in this folder do not reproduce that experiment.

The three executable reference worlds can demonstrate image and question generation
locally, but they are not the paper's fine-tuning dataset. See the
[implementation-demonstration walkthrough](../shared_examples/README.md).
