# Release scope and differences

This release runs the three generation experiments. It includes the controller, three
coding-agent interfaces, self-checks and final checks, three implementation
demonstrations, nine discovery profiles, and two ten-world feedback example sets. The
fine-tuning experiment, historical campaign outputs, and the paper's separate 200-scene
replay analysis are not included.

## Versions

The release uses `question-world` 0.4, `episodic-sequential` 0.7, `baseline_three` 0.2,
`paper-semantic` 0.3, `quality-diversity` 0.8, and `difficulty-feedback` 0.9. Some
historical model-feedback-steered campaigns used `episodic-sequential` 0.8. That variant
is not included.

There is one example configuration for each of Codex, Claude Code, and OpenCode. The
paper uses five model/reasoning configurations. Reproducing the complete experiment
matrix requires those configurations and model access. Available models and their
generated outputs may differ from the original runs.

## Changes for this release

- The prepare, preview, freeze, and run commands provide a common interface to the
  three coding agents. Credentials come from environment variables or files
  supplied by the user.
- Package and container names use neutral names. Original Git history,
  author identities, personal paths, and personal remotes were not copied.
- Tests of submitted worlds run in a temporary copy because some write
  `self_check.json`. The original source remains mounted read-only. Rendering uses that
  source in a separate offline container. No test of a submitted world is skipped.
- Manuscript sources, historical agent traces, rendered galleries, and author
  review files are excluded. The full Figure 1 diagram is included for the README.

The world algorithms, answer classes, inverse tolerances, final checks, and
spatial-pattern thresholds are unchanged.

Five source labels in the semantic discovery profiles use a neutral benchmark name in
this release. The example questions and construction instructions are unchanged. This
label substitution is an anonymization change to the supplied text; these profiles are
not byte-identical copies of the historical inputs.

## Starting inputs for model-feedback-steered generation

Feedback example sets retain world source, tests, submission metadata, and deviation
records. Their feedback includes aggregate accuracies, question text, answer choices,
recorded answers, and predictions. These fields are visible to the coding agent.

Previously rendered images and their accompanying records are excluded. This includes
feedback example set A's 32-record `run_bar_decompression` manifest and its HTML
gallery, so the release provides fewer starting scene specifications and answers than
that historical input bundle. World logic and retained feedback values are unchanged.
Content hashes were recomputed for the release files.

Historical bookkeeping was removed from feedback summaries and manifests. Image paths in
the sample records are references to the original evaluation; the image files are
absent. Evaluations of new submissions render images locally. Feedback example sets
load their fixed feedback without rerunning evaluation at initialization.

## Known inconsistencies in frozen instructions

The instruction files preserve the experimental content, with the source-label
substitution noted above. Some wording is inconsistent with the files actually supplied:

| Frozen wording | Actual input |
| --- | --- |
| Semantic profiles say “Author choice pending” | The release uses the fixed `paper-semantic` 0.3 discovery profiles |
| Some model-feedback-steered instructions say “five conceptual seeds” | Each feedback example set contains ten worlds |
| Feedback prose says gold labels are hidden | Starting-feedback JSON includes recorded answers |

Use [the input guide](INPUTS.md) for the supplied files and what agents can read.
Changing the frozen prompts would require a new experimental version. The input guide
also explains older terms such as “working seed,” “inverse arm,” and “survivor commit,”
and identifies references to development documents that are not packaged.

## Local outputs and packaging

Campaign outputs and virtual environments are excluded from the source release.
`scripts.package_release` produces a ZIP from the approved source and input files, with
no Git history or run logs. It does not back up local campaigns.

See [validation](VALIDATION.md) for how to run the checks and for the known ten-scene answer-class coverage failure in `stitch_face_alternation`.
