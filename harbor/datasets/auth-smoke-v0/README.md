# Authentication test template

This task makes one short authenticated model call. The agent writes a file containing
`AUTH_SMOKE_OK` and stops. It receives no discovery profiles, implementation
demonstrations, protected registries, or outputs from other trials.

The container has network access to install the agent CLI and contact its provider.
Harbor 0.1.44 reuses that container to check the output file. This test checks
authentication and output collection; it does not test the offline verification boundary
or generate a scientific submission. Use `scripts.docker_smoke` for the
container-isolation check.

See [authentication](../../../docs/AUTHENTICATION.md) for instructions.
