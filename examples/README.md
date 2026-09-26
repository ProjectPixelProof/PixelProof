# Run the paper's generation experiments

Start with the root [README](../README.md) for the method and Docker setup. All commands
below assume that setup is complete. Run the walkthrough commands from the repository
root. The helper scripts also locate that root when called from another directory.

| Paper section | Walkthrough | Preparation script |
| --- | --- | --- |
| §4.1 | [Profile-steered generation](rq1/README.md) | `bash examples/rq1/prepare.sh --agent codex --profile tracing --id rq1_demo` |
| §4.2 | [Spatially-steered generation](rq2/README.md) | `bash examples/rq2/prepare.sh --agent claude --profile tracing --id rq2_demo --agent-seconds 1800` |
| §4.5 | [Model-feedback-steered generation](rq3/README.md) | `bash examples/rq3/prepare.sh --agent opencode --collection a --id rq3_demo` |
| §4.7 | [Training on generated questions: not packaged](rq4/README.md) | No training launcher is included |
| §3.2 | [Render the implementation demonstrations without a model](shared_examples/README.md) | `bash examples/shared_examples/render.sh` |

The three preparation scripts create a TOML file; they do not launch a coding agent.
Each passes its options to `scripts.release prepare`. Add `--help` for available
options. Supported agents are `codex`, `claude`, and `opencode`. Each walkthrough also
shows how to preview the input files, authenticate, freeze and commit the configuration,
start the loop, and inspect saved outputs.

`bash examples/run_campaign.sh CAMPAIGN_ID` starts a paid campaign. It requires
authentication and the confirmation variables described in [the authentication
guide](../docs/AUTHENTICATION.md). It does not automatically freeze or commit a
configuration. Run `bash examples/run_campaign.sh --help` for help.

Use a fresh ID for each attempt. Default preparation assigns 1,200 seconds of
agent-session time; `--agent-seconds 21600` requests six hours for that campaign.
`--profile all` prepares nine separate campaigns for profile-steered or
spatially-steered generation, one per profile. Launch each campaign separately.

`codex.sh`, `claude.sh`, and `opencode.sh` are alternative preparation shortcuts. Use
the experiment walkthroughs above for the full sequence.
