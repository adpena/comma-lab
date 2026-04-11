from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromotedResult:
    run_id: str
    track: str
    score: float
    pose_distortion: float
    seg_distortion: float
    rate: float
    archive_bytes: int
    authoritative_report_path: str
    authoritative_report_copy_path: str
    summary_path: str
    artifact_path: str
    variant: str
    platform: str
    epoch: int | None
    promoted_at: str

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PromotedResult":
        return cls(
            run_id=str(payload["run_id"]),
            track=str(payload["track"]),
            score=float(payload["score"]),
            pose_distortion=float(payload["pose_distortion"]),
            seg_distortion=float(payload["seg_distortion"]),
            rate=float(payload["rate"]),
            archive_bytes=int(payload["archive_bytes"]),
            authoritative_report_path=str(payload["authoritative_report_path"]),
            authoritative_report_copy_path=str(payload["authoritative_report_copy_path"]),
            summary_path=str(payload["summary_path"]),
            artifact_path=str(payload["artifact_path"]),
            variant=str(payload["variant"]),
            platform=str(payload["platform"]),
            epoch=None if payload.get("epoch") is None else int(payload["epoch"]),
            promoted_at=str(payload["promoted_at"]),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def summary_payload(self) -> dict[str, object]:
        return {
            "track": self.track,
            "device": "cpu",
            "report_path": self.authoritative_report_path,
            "copied_report_path": self.authoritative_report_copy_path,
            "current_workflow_archive_bytes": self.archive_bytes,
            "pose_distortion": self.pose_distortion,
            "seg_distortion": self.seg_distortion,
            "original_uncompressed_bytes": 37545489,
            "current_workflow_rate": self.rate,
            "current_workflow_score": self.score,
        }

    def results_row(self) -> dict[str, object]:
        return {
            "archive_bytes": self.archive_bytes,
            "artifacts": {
                "raw_report": self.authoritative_report_path,
                "summary_json": self.summary_path,
            },
            "config": {
                "variant": self.variant,
                "platform": self.platform,
                "epoch": self.epoch,
            },
            "current_workflow_score": self.score,
            "delta_vs_published_baseline": round(self.score - 4.39, 2),
            "device": "cpu",
            "notes": [
                "authoritative scorer confirmed",
                f"{self.variant} promoted result",
                "generated from canonical promoted_result.json",
            ],
            "packaging_view": "current_workflow",
            "posenet_distortion": self.pose_distortion,
            "published_baseline_score": 4.39,
            "rate": self.rate,
            "rule_faithful_bundle_bytes": None,
            "rule_faithful_rate": None,
            "rule_faithful_score": None,
            "rule_faithful_status": "pending_honest_bytes_audit",
            "run_id": self.run_id,
            "segnet_distortion": self.seg_distortion,
            "track": self.track,
            "ts_utc": self.promoted_at,
            "upstream_commit": "ec82c291ffeae5212e9a38253791d58995518a80",
        }

    def timeline_event(self) -> dict[str, object]:
        return {
            "ts": self.promoted_at,
            "event": "promotion",
            "score": self.score,
            "variant": self.variant,
            "source": self.platform,
            "epoch": self.epoch,
            "pose": self.pose_distortion,
            "seg": self.seg_distortion,
            "rate": self.rate,
            "notes": f"authoritative scorer confirmed for {self.variant}",
        }


@dataclass(frozen=True)
class DriftFinding:
    code: str
    severity: str
    path: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DoctorReport:
    findings: tuple[DriftFinding, ...]

    def to_dict(self) -> dict[str, object]:
        return {"findings": [finding.to_dict() for finding in self.findings]}


@dataclass(frozen=True)
class SyncResult:
    changed_paths: tuple[str, ...]
    findings: tuple[DriftFinding, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "changed_paths": list(self.changed_paths),
            "changed_count": len(self.changed_paths),
            "findings": [finding.to_dict() for finding in self.findings],
        }


def canonical_record_path(repo_root: Path) -> Path:
    return repo_root / ".omx" / "state" / "promoted_result.json"
