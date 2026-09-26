# Shared question-world task template

This Harbor template supports both multi-submission tasks and single-submission trials.
Do not launch an agent directly against the template. The preparation code combines it
with a versioned protocol, implementation demonstrations, task limits, and container
settings, and records hashes of the resulting files. The release uses single-submission
episodes; follow the [walkthroughs](../../../examples/README.md).

The bundled reference-marker solution is an executable world used to test packaging and
final verification. It is supplied only during fixed-example smoke tests, not during
normal generation episodes.
