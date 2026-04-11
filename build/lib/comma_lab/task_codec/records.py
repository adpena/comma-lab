"""Serializable evaluation and proxy record loaders for current-workflow artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


_JSON_DECODER = json.JSONDecoder()

_PROXY_TEXT_PATTERNS = {
    "weights": re.compile(r"^\[proxy-faithful\]\s+weights:\s+(.+)$", re.MULTILINE),
    "archive": re.compile(r"^\[proxy-faithful\]\s+archive:\s+(.+)$", re.MULTILINE),
    "device": re.compile(r"^\[proxy-faithful\]\s+device:\s+(.+)$", re.MULTILINE),
    "pose_distortion": re.compile(r"PoseNet distortion:\s*([0-9.]+)"),
    "seg_distortion": re.compile(r"SegNet distortion:\s*([0-9.]+)"),
    "current_workflow_rate": re.compile(r"Compression rate:\s*([0-9.]+)"),
    "current_workflow_score": re.compile(r"Final score:\s*([0-9.]+)"),
}


def _extract_last_json_object(text: str) -> dict[str, object] | None:
    best: dict[str, object] | None = None
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, end = _JSON_DECODER.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if text[idx + end :].strip():
            continue
        best = payload
    return best


def _read_required_float(payload: Mapping[str, object], key: str) -> float:
    if key not in payload:
        raise KeyError(f"Missing required key: {key}")
    return float(payload[key])


def _read_optional_int(payload: Mapping[str, object], key: str) -> int | None:
    value = payload.get(key)
    return None if value is None else int(value)


@dataclass(frozen=True)
class EvaluationRecord:
    """Normalized evaluator summary record."""

    source_path: Path
    track: str
    device: str
    report_path: str
    copied_report_path: str | None
    current_workflow_archive_bytes: int
    pose_distortion: float
    seg_distortion: float
    original_uncompressed_bytes: int
    current_workflow_rate: float
    current_workflow_score: float
    rule_faithful_bundle_bytes: int | None = None
    rule_faithful_bundle_paths: list[str] | None = None
    rule_faithful_rate: float | None = None
    rule_faithful_score: float | None = None
    rule_faithful_status: str | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> "EvaluationRecord":
        source_path = Path(path)
        payload = json.loads(source_path.read_text())
        required = {
            "track",
            "device",
            "report_path",
            "current_workflow_archive_bytes",
            "pose_distortion",
            "seg_distortion",
            "original_uncompressed_bytes",
            "current_workflow_rate",
            "current_workflow_score",
        }
        missing = sorted(required.difference(payload))
        if missing:
            raise KeyError(f"Missing evaluation summary keys: {', '.join(missing)}")
        return cls(
            source_path=source_path,
            track=str(payload["track"]),
            device=str(payload["device"]),
            report_path=str(payload["report_path"]),
            copied_report_path=payload.get("copied_report_path"),
            current_workflow_archive_bytes=int(payload["current_workflow_archive_bytes"]),
            pose_distortion=float(payload["pose_distortion"]),
            seg_distortion=float(payload["seg_distortion"]),
            original_uncompressed_bytes=int(payload["original_uncompressed_bytes"]),
            current_workflow_rate=float(payload["current_workflow_rate"]),
            current_workflow_score=float(payload["current_workflow_score"]),
            rule_faithful_bundle_bytes=_read_optional_int(payload, "rule_faithful_bundle_bytes"),
            rule_faithful_bundle_paths=payload.get("rule_faithful_bundle_paths"),
            rule_faithful_rate=None if payload.get("rule_faithful_rate") is None else float(payload["rule_faithful_rate"]),
            rule_faithful_score=None if payload.get("rule_faithful_score") is None else float(payload["rule_faithful_score"]),
            rule_faithful_status=None if payload.get("rule_faithful_status") is None else str(payload["rule_faithful_status"]),
        )


@dataclass(frozen=True)
class ProxyEvaluationRecord:
    """Normalized proxy evaluation record loaded from JSON or text logs."""

    source_path: Path
    weights_path: Path | None
    archive_path: Path | None
    device: str | None
    pose_distortion: float
    seg_distortion: float
    current_workflow_rate: float
    current_workflow_score: float
    current_workflow_archive_bytes: int | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> "ProxyEvaluationRecord":
        source_path = Path(path)
        text = source_path.read_text(errors="ignore")
        payload = _extract_last_json_object(text)
        if payload is not None and "current_workflow_score" in payload:
            weights = payload.get("weights")
            archive = payload.get("archive")
            device = payload.get("device")
            return cls(
                source_path=source_path,
                weights_path=None if weights is None else Path(str(weights)),
                archive_path=None if archive is None else Path(str(archive)),
                device=None if device is None else str(device),
                pose_distortion=_read_required_float(payload, "pose_distortion"),
                seg_distortion=_read_required_float(payload, "seg_distortion"),
                current_workflow_rate=_read_required_float(payload, "current_workflow_rate"),
                current_workflow_score=_read_required_float(payload, "current_workflow_score"),
                current_workflow_archive_bytes=_read_optional_int(payload, "current_workflow_archive_bytes"),
            )

        values: dict[str, object] = {}
        for key, pattern in _PROXY_TEXT_PATTERNS.items():
            match = pattern.search(text)
            if match is None:
                continue
            values[key] = match.group(1).strip()

        required_text = {
            "pose_distortion",
            "seg_distortion",
            "current_workflow_rate",
            "current_workflow_score",
        }
        missing = sorted(required_text.difference(values))
        if missing:
            raise ValueError(f"Could not parse proxy record fields from {source_path}: {', '.join(missing)}")

        weights = values.get("weights")
        archive = values.get("archive")
        device = values.get("device")
        return cls(
            source_path=source_path,
            weights_path=None if weights is None else Path(str(weights)),
            archive_path=None if archive is None else Path(str(archive)),
            device=None if device is None else str(device),
            pose_distortion=float(values["pose_distortion"]),
            seg_distortion=float(values["seg_distortion"]),
            current_workflow_rate=float(values["current_workflow_rate"]),
            current_workflow_score=float(values["current_workflow_score"]),
            current_workflow_archive_bytes=None,
        )
