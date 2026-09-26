"""Stable domain contracts for PixelProof.

These types describe scientific state independently of Harbor. Harbor trials are
one execution mechanism; worlds, candidates, gates, campaigns, and admission are
owned by this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class CandidateDisposition(StrEnum):
    """Durable disposition assigned after verification and review."""

    PROPOSED = "proposed"
    REPAIR = "repair"
    REJECTED = "rejected"
    DEFERRED = "deferred"
    PROMOTED = "promoted"


@dataclass(frozen=True)
class WorldRef:
    """Content-addressable reference to one world implementation."""

    world_id: str
    version: str
    path: Path
    source_commit: str | None = None


@dataclass(frozen=True)
class CandidateBundle:
    """One inner trial's immutable output after artifact collection."""

    candidate_id: str
    campaign_id: str
    trial_id: str
    artifact_path: Path
    artifact_sha256: str
    world: WorldRef
    deviations: tuple[str, ...] = ()


@dataclass(frozen=True)
class GateEvidence:
    """Serializable evidence for one protected mechanical gate."""

    gate: str
    ok: bool
    failures: tuple[str, ...] = ()
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GateReport:
    """Conjunctive mechanical eligibility result for one candidate."""

    candidate_id: str
    protocol_version: str
    evidence: tuple[GateEvidence, ...]

    @property
    def eligible(self) -> bool:
        return bool(self.evidence) and all(item.ok for item in self.evidence)


@dataclass(frozen=True)
class CampaignSpec:
    """Frozen inputs shared by all comparable trial arms."""

    campaign_id: str
    protocol_version: str
    seed_set_version: str
    canonical_commit: str
    arms: tuple[str, ...]
    candidate_limit_per_trial: int = 1
    repair_turns: int = 0


@dataclass(frozen=True)
class OuterDecision:
    """Sighted advisory judgment; never an admission certificate."""

    candidate_id: str
    recommendation: CandidateDisposition
    evidence: tuple[str, ...]
    uncertainties: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdmissionRecord:
    """Human-controlled promotion record for a mechanically eligible candidate."""

    candidate_id: str
    gate_report_sha256: str
    reviewer: str
    signed_at: str
    disposition: CandidateDisposition
    rationale: str
