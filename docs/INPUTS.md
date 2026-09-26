# What the coding agent receives

| Input | Profile-steered (`--rq 1`) | Spatially-steered (`--rq 2`) | Model-feedback-steered (`--rq 3`) |
| --- | --- | --- | --- |
| Implementation demonstrations | Three, identical | Same three | Same three |
| Nine discovery profiles | One selected profile | One selected profile | None |
| Five text-only example questions | All five in that profile | All five in that profile | Uses its ten-world feedback-example-set card instead |
| Ten starting implementations | No | No | Feedback example set A or B |
| Spatial-pattern targets | No | Adaptive request in text and JSON | No |
| Evaluator feedback | No | No | Starting summaries, then summaries of new evaluations |
| Accepted world code | Grows as worlds are retained | Grows as worlds are retained | Starts with ten worlds; later retained worlds are added |
| Most recent rejected code and report | Available after rejection | Available after rejection | Available after rejection |
| Self-checks and submission requirements | Yes | Yes | Yes |

`instruction.md` directs the agent to read files in `/workspace/`. Profile question
sentences are supplied in `DISCOVERY_PROFILE.md`; the initial instruction directs the
agent to that file. All five remain fixed for that profile across episodes. Each profile
also specifies the target computation, guidance for construction and inverse solving,
validity requirements, and rules against superficial copies of existing worlds. A source
ID such as `blind_network_farthest` records where an example question came from. It does
not load task code or an image. No corresponding profile-specific implementation or
reader illustration is supplied.

The implementation demonstrations are `two_circles`, `angle_acuteness`, and
`counting_with_distractors`. The configuration (`shared_examples`) supplies only
`DESIGN.md`, `world.toml`, `renderer.py`, `prompts.py`, and `oracle.py` for each. These
source snapshots show how to sample and render scenes, map questions to answers, and
recover answers independently from pixels. Answer computation may be defined in these
modules rather than in a separate forward-program file. The release also includes
runnable generators for users; those entry points are not added to the coding agent's
five-file snapshots.

Discovery profiles do not define complete instances. For example, “Which terminal is
farthest from green along the lines?” is one example question; the agent supplies a
complete public question specification, sampler, renderer, forward program, and inverse
program for its new world.

Generation, repair, submission and verification behavior comes from separate versioned
strategy instructions and public tools, not from the three implementation
demonstrations. The final-check test files are excluded from the agent's starting
directory and uploaded to a clean container after submission. Their source is included
in this repository, so anyone reading the repository can inspect it. The restriction
applies to the files delivered during an agent episode.

Model-feedback-steered feedback JSON includes answers and correctness for starting
worlds and earlier submissions. Visibility descriptions in the profile metadata apply
only to the discovery profile; they do not describe the separate feedback example set
and feedback files. Final verification of a new submission remains a separate stage.
Availability of any file does not guarantee a future agent will read it; inspect the
local run's saved agent transcript to see which files it read. No historical
trajectories are included.

## Terms used in the preserved instructions

Some files retain terminology from the original implementation. The terms below refer to
different parts of the workflow.

| Term in source files | Meaning |
| --- | --- |
| Builder, inner builder | The coding agent that implements one new world |
| Candidate | A submitted world |
| Public gate, public checks | The self-checks the agent can run during its session |
| Protected gate, protected checks | The final checks run by the verifier after submission |
| Profile card | A discovery profile |
| Starting collection (`inputs/starting_collections/`) | A feedback example set |
| Shared examples (`examples/shared_examples/`) | The implementation demonstrations |
| RQ1, RQ2, RQ3 (`--rq 1/2/3`) | Profile-steered, spatially-steered, and model-feedback-steered generation |
| Inverse arm, pixel arm, oracle | The inverse program, which recovers the answer from the rendered image alone |
| Analytic gold | The recorded answer, computed by the forward program from the scene specification |
| Working seed | The collection of accepted worlds supplied to later episodes |
| Seed snapshot, seed set | The three implementation demonstrations; model-feedback-steered generation also has a separate feedback example set |
| Materialized packet | The task files assembled for one agent episode |
| Gate | A check that a submission must pass |
| Survivor commit, envelope check | The recorded completion of a submission; this is not a Git commit |
| Canonical admission | Human approval for inclusion in the research world bank; the release controller does not perform it |
| Descriptor cell, elite archive | A spatial-pattern category and the retained representatives of those categories in spatially-steered generation |
| Random seed | A number or string used to make sampling reproducible |

The original `DESIGN.md` files supplied with the implementation demonstrations include
references to an older SSOT (single source of truth), `docs/verification-boundary.md`,
and calibration scripts that are not packaged here. Those references describe the
examples' development history. Use the [implementation-demonstration
walkthrough](../examples/shared_examples/README.md) for supported release commands and
the versioned contracts under `foundry/` for new-world requirements.
