"""Angle-acuteness question world (historical SSOT §7.2 and §7.7).

A standalone world under the foundry. It owns its renderer,
prompt families, target computation, and dataset generation; it depends on the
core package (question_foundry.*) only for task-agnostic machinery (manifest
schema, and model wrappers / constrained-logit scoring when eval runs).

To create a new failure-mode study, copy this folder, rename it, and rewrite
renderer.py / prompts.py / world.toml / README.md. Internal imports are
relative, so the copy keeps working under its new name.
"""

WORLD = "angle_acuteness"
