"""Fail-closed checks for fields consumed from the frozen candidate contract.

The authoritative schema remains
``foundry/protocols/question-world/v0.4/candidate.schema.json``. These checks
cover the task-signature fields consumed by orchestration and archival code so
an agent-authored verifier false positive cannot become controller memory.
"""

from __future__ import annotations

import re
from typing import Any

DECISION_TYPES = frozenset({"yesno", "comparison", "selection", "threshold", "count_open"})
TASK_SIGNATURE_KEYS = frozenset({"decision_var", "structure", "decision_type", "mechanism"})
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{2,63}$")


def task_signature_errors(metadata: dict[str, Any]) -> tuple[str, ...]:
    """Return frozen-v0.4 task-signature violations without mutation."""

    errors: list[str] = []
    decision_type = metadata.get("decision_type")
    if decision_type not in DECISION_TYPES:
        errors.append(f"decision_type must be one of {sorted(DECISION_TYPES)}")
    signature = metadata.get("task_signature")
    if not isinstance(signature, dict):
        return (*errors, "task_signature must be an object")
    if set(signature) != TASK_SIGNATURE_KEYS:
        errors.append(f"task_signature keys must be exactly {sorted(TASK_SIGNATURE_KEYS)}")
    for field in ("decision_var", "structure"):
        value = signature.get(field)
        if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
            errors.append(f"task_signature.{field} must be a protocol identifier")
    signature_type = signature.get("decision_type")
    if signature_type not in DECISION_TYPES:
        errors.append(f"task_signature.decision_type must be one of {sorted(DECISION_TYPES)}")
    elif signature_type != decision_type:
        errors.append("task_signature.decision_type disagrees with candidate")
    mechanism = signature.get("mechanism")
    if not isinstance(mechanism, str) or len(mechanism.strip()) < 20:
        errors.append("task_signature.mechanism must contain at least 20 characters")
    return tuple(errors)


def require_task_signature(metadata: dict[str, Any], *, context: str) -> dict[str, str]:
    """Return a valid signature or raise a contextual deterministic error."""

    errors = task_signature_errors(metadata)
    if errors:
        raise ValueError(f"{context}: invalid candidate task signature: {'; '.join(errors)}")
    return metadata["task_signature"]
