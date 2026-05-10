"""Fail-closed GCP dispatch scaffold.

This module intentionally does not create Compute Engine VMs. It records the
deterministic plan shape that a future GCP actuator must satisfy before real
spend is allowed.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GCPDispatchPlan:
    """Dry-run-only plan for a future GCP GPU dispatch."""

    lane_id: str
    project: str
    zone: str = "us-central1-a"
    gpu_type: str = "nvidia-tesla-t4"
    machine_type: str = "n1-standard-4"
    dry_run: bool = True
    score_claim: bool = False
    promotion_eligible: bool = False
    requires_lane_claim_before_dispatch: bool = True
    terminal_claim_required: bool = True
    custody_manifest_required: bool = True
    setup_blockers: tuple[str, ...] = field(
        default=(
            "gcloud auth/application-default login",
            "GPU quota in selected zone",
            "deterministic tarball or mounted-code manifest",
            "GCS harvest bucket with artifact SHA manifest",
        )
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "provider": "gcp",
            "lane_id": self.lane_id,
            "project": self.project,
            "zone": self.zone,
            "gpu_type": self.gpu_type,
            "machine_type": self.machine_type,
            "dry_run": self.dry_run,
            "score_claim": self.score_claim,
            "promotion_eligible": self.promotion_eligible,
            "requires_lane_claim_before_dispatch": self.requires_lane_claim_before_dispatch,
            "terminal_claim_required": self.terminal_claim_required,
            "custody_manifest_required": self.custody_manifest_required,
            "setup_blockers": list(self.setup_blockers),
        }


def plan_gcp_dispatch(*, lane_id: str, project: str, zone: str = "us-central1-a") -> GCPDispatchPlan:
    """Return a deterministic GCP dispatch plan without launching anything."""
    if not lane_id.strip():
        raise ValueError("lane_id is required before planning a GCP dispatch")
    if not project.strip():
        raise ValueError("project is required before planning a GCP dispatch")
    return GCPDispatchPlan(lane_id=lane_id.strip(), project=project.strip(), zone=zone.strip())
