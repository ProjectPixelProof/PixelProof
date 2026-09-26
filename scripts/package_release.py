"""Package only allowlisted public files, excluding Git history and local run outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from scripts.audit_release import ROOT, audit, payload_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "dist/anonymous-code.zip")
    args = parser.parse_args()
    result = audit()
    if result["status"] != "passed":
        raise ValueError("release scan failed; run scripts.audit_release for file-level findings")
    if args.out.exists():
        raise FileExistsError("archive already exists; choose another output path")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    files = {p.relative_to(ROOT).as_posix(): p for p in payload_files()}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    with zipfile.ZipFile(args.out, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            info = zipfile.ZipInfo("code/" + name, date_time=(2026, 1, 1, 0, 0, 0))
            info.external_attr = (0o100755 if path.suffix == ".sh" else 0o100644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
        info = zipfile.ZipInfo("code/RELEASE_SHA256.json", date_time=(2026, 1, 1, 0, 0, 0))
        info.external_attr = 0o100644 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    print(f"Created {args.out}: {len(files)} source/input files; no Git history or execution logs")


if __name__ == "__main__":
    main()
