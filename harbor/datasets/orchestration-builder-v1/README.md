# Campaign task template

The controller assembles each task from the shared question-world v0.4 template and the
selected strategy. `TASK_SPEC.toml` records the task's limits and required outputs. The
released sequential strategy requests one submission per episode.

After the coding agent stops, the controller replaces its networked container with a
clean offline container, then supplies the final-check tests and protected registries.
The controller decides whether to retain the submitted world for later episodes.
Retention does not establish novelty or constitute human approval.

Use the [experiment walkthroughs](../../../examples/README.md) to prepare and run a
campaign.
