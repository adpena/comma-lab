# SPDX-License-Identifier: MIT
"""Canonical local-audit result contract.

Audit tools in this repository are custody and readiness guards. They should
not make score claims or mutate provider state. This module centralizes the
small JSON/text contract so recovered audits do not each invent a different
schema for blockers, summaries, and dispatch-safety markers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

RESERVED_TOP_LEVEL_KEYS = frozenset(
    {
        "audit",
        "blockers",
        "dispatch_attempted",
        "score_claim",
        "summary",
    }
)


@dataclass(frozen=True)
class AuditReport:
    """Stable JSON/text result for one local audit."""

    audit: str
    readiness_key: str
    ready: bool
    blockers: tuple[str, ...] = ()
    summary: dict[str, Any] = field(default_factory=dict)
    score_claim: bool = False
    dispatch_attempted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reserved = RESERVED_TOP_LEVEL_KEYS | {self.readiness_key}
        collisions = sorted(reserved.intersection(self.metadata))
        if collisions:
            joined = ", ".join(collisions)
            raise ValueError(f"AuditReport metadata cannot override reserved top-level key(s): {joined}")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "audit": self.audit,
            self.readiness_key: self.ready,
            "blockers": list(self.blockers),
            "dispatch_attempted": self.dispatch_attempted,
            "score_claim": self.score_claim,
            "summary": self.summary,
        }
        payload.update(self.metadata)
        return payload

    def render_text(self, *, pass_detail: str = "") -> str:
        title = self.audit.replace("_", " ")
        if not self.ready:
            lines = [f"{title}: FAIL"]
            lines.extend(f"  - {blocker}" for blocker in self.blockers)
            return "\n".join(lines)
        suffix = f" {pass_detail}" if pass_detail else ""
        return f"{title}: PASS{suffix}"


def audit_exit_code(report: AuditReport) -> int:
    """Return the conventional local-audit CLI exit code."""

    return 0 if report.ready else 2


__all__ = ["RESERVED_TOP_LEVEL_KEYS", "AuditReport", "audit_exit_code"]
