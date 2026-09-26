# Minimal authentication smoke

This is only an authentication and artifact-transport check.

Create `/logs/artifacts/auth-smoke/result.txt` containing exactly:

```text
AUTH_SMOKE_OK
```

Then stop. Do not inspect or print environment variables, authentication files,
configuration files, system credentials, or unrelated filesystem content. Do not
perform network activity except what your coding-agent CLI requires to answer this
instruction. Do not create any other artifact.
