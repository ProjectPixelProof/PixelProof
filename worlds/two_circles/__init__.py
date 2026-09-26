"""Two-circle collision question world (historical SSOT §7.3–7.5, §15).

A standalone world under the foundry. It owns its renderer,
prompt families, target computation, and dataset generation; it depends on the
core package (question_foundry.*) only for task-agnostic machinery (model
wrappers, constrained-logit scoring, probes, manifest schema).

To create a new failure-mode study, copy this folder, rename it, and rewrite
renderer.py / prompts.py / world.toml / README.md. Internal imports are
relative, so the copy keeps working under its new name.
"""

WORLD = "two_circles"
