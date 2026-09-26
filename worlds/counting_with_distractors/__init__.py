"""Counting-with-distractors question world (historical SSOT §7.2 and §7.7; design
docs/designs/counting_with_distractors.md).

A standalone world under the foundry. It owns its renderer,
prompt families, target computation, and dataset generation; it depends on the
core package (question_foundry.*) only for task-agnostic machinery (manifest
schema, and model wrappers / constrained-logit scoring when eval runs).

This is the first aggregation / counting world: a variable object count, an
integer-valued answer, and a crowding margin defined over the whole set of disks.

To create a new failure-mode study, copy this folder, rename it, and rewrite
renderer.py / prompts.py / world.toml / README.md. Internal imports are
relative, so the copy keeps working under its new name.
"""

WORLD = "counting_with_distractors"
