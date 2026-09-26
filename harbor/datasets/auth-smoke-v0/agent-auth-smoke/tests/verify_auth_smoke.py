from pathlib import Path

ROOT = Path("/logs/artifacts/auth-smoke")
EXPECTED = b"AUTH_SMOKE_OK\n"


def main() -> None:
    files = sorted(path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if path.is_file())
    if files != ["result.txt"]:
        raise SystemExit(f"expected only result.txt, found {files}")
    if (ROOT / "result.txt").read_bytes() != EXPECTED:
        raise SystemExit("result.txt does not contain the exact smoke sentinel")
    print("authenticated agent artifact smoke passed")


if __name__ == "__main__":
    main()
