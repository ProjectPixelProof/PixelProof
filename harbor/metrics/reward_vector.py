"""Aggregate Harbor verifier reward dictionaries key by key."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def aggregate(rows: list[dict[str, float | int] | None]) -> dict[str, float]:
    keys = sorted({key for row in rows if row for key in row})
    if not keys:
        return {"mean_reward": 0.0}
    metrics = {}
    for key in keys:
        values = []
        for row in rows:
            value = 0.0 if row is None else row.get(key, 0.0)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"reward {key!r} must be numeric")
            values.append(float(value))
        metrics[f"mean_{key}"] = sum(values) / len(values)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    args.output.write_text(
        json.dumps(aggregate(rows), sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
