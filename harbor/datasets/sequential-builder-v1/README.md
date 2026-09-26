# Sequential episode template

These files adapt the shared question-world v0.4 task to one-submission episodes. This
directory cannot be run directly as a Harbor dataset.

When preparing an episode, the controller copies the verifier from
`harbor/datasets/foundry-builder-v1/`, substitutes the sequential instructions and
verification driver, and adds the current campaign state and accepted worlds. It records
a hash for every resulting file. The shared verifier has one source implementation.

See the [experiment walkthroughs](../../../examples/README.md) for run commands.
