#!/usr/bin/env python3
"""Plan C091-native tile-action atoms from exact component traces.

This is a local planning tool. It does not dispatch GPU jobs and it does not
create candidate archives unless future evidence clears the explicit dispatch
gate. The planner treats C091 PR75-minp public replay as the score anchor and
uses exact component traces from PR75 action subsets, PR77, and the failed
PR77-action/C089-pose fixed-slice run as atom-level feedback.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
UNPACKER_PATH = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"
TOOL = "experiments/plan_c091_native_action_atoms.py"
SCHEMA = "c091_native_action_atoms_policy_v1"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments/results/c091_native_action_atoms_20260503_codex"
)

RATE_SCORE_PER_BYTE = 25.0 / 37_545_489
SUB314_TARGET = 0.314
C091_SCORE = 0.31516575028285976
C091_BYTES = 276_481
C091_SHA256 = "03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746"
EPS = 1e-12

DEFAULT_ANCHOR_TRACE = REPO_ROOT / (
    "experiments/results/lightning_batch/"
    "exact_eval_pr75_minp_public_replay_t4_20260503T1049Z/component_trace.json"
)
DEFAULT_ANCHOR_ARCHIVE = REPO_ROOT / (
    "experiments/results/lightning_batch/"
    "exact_eval_pr75_minp_public_replay_t4_20260503T1049Z/archive.zip"
)
DEFAULT_PR77_ARCHIVE = (
    REPO_ROOT / "experiments/results/top_submission_reverse_engineering_20260503_pr77/archive.zip"
)
DEFAULT_PR77_FIXEDSLICE_ARCHIVE = REPO_ROOT / (
    "experiments/results/pr77_action_pose_mixed_container_20260503_codex/"
    "pr77_actions_pr75mask_renderer_c089pose_fixedslice/archive.zip"
)


@dataclass(frozen=True)
class ActionRecord:
    pair_index: int
    tile_id: int
    action_id: int
    source_action_id: int | None = None
    source_index: int | None = None
    transform: str = "identity"
    evidence_source: str = ""

    @property
    def atom_id(self) -> str:
        source_action = self.action_id if self.source_action_id is None else self.source_action_id
        return (
            f"pair{self.pair_index:03d}_tile{self.tile_id:03d}_"
            f"src{source_action:03d}_act{self.action_id:03d}_{self.transform}"
        )


@dataclass(frozen=True)
class LoadedTrace:
    label: str
    path: Path
    archive_bytes: int | None
    archive_sha256: str | None
    score: float | None
    seg_score: float
    pose_score: float
    samples_by_pair: dict[int, dict[str, float]]

    @property
    def component_score(self) -> float:
        return self.seg_score + self.pose_score


@dataclass(frozen=True)
class Observation:
    label: str
    trace: LoadedTrace
    action_records: tuple[ActionRecord, ...]
    family: str
    evidence_role: str
    attribution: str
    confidence: float
    manifest_path: Path | None = None
    archive_path: Path | None = None


@dataclass(frozen=True)
class ObservationSpec:
    label: str
    trace_path: Path
    family: str
    evidence_role: str
    attribution: str
    confidence: float
    manifest_path: Path | None = None
    archive_path: Path | None = None


def _rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _load_trace(path: Path, *, label: str) -> LoadedTrace:
    payload = json.loads(path.read_text())
    samples: dict[int, dict[str, float]] = {}
    for sample in payload.get("samples", []):
        pair = int(sample["pair_index"])
        samples[pair] = {
            "seg": float(sample["score_seg_contribution_exact"]),
            "pose": float(sample["score_pose_contribution_first_order"]),
            "combined": float(sample["score_combined_contribution_first_order"]),
        }
    if not samples:
        raise ValueError(f"{path} has no component-trace samples")
    trace_inputs = payload.get("trace_inputs") if isinstance(payload, dict) else None
    archive_sha = None
    if isinstance(trace_inputs, dict):
        archive_sha = trace_inputs.get("archive_sha256")
    return LoadedTrace(
        label=label,
        path=path,
        archive_bytes=(
            int(payload["archive_size_bytes"])
            if payload.get("archive_size_bytes") is not None
            else None
        ),
        archive_sha256=str(archive_sha) if archive_sha else None,
        score=(
            float(payload["score_recomputed_from_components"])
            if payload.get("score_recomputed_from_components") is not None
            else None
        ),
        seg_score=float(payload["score_seg_contribution"]),
        pose_score=float(payload["score_pose_contribution"]),
        samples_by_pair=samples,
    )


def _pair_deltas(anchor: LoadedTrace, candidate: LoadedTrace) -> dict[int, dict[str, float]]:
    missing = sorted(set(anchor.samples_by_pair) - set(candidate.samples_by_pair))
    if missing:
        raise ValueError(
            f"{candidate.path} is missing {len(missing)} anchor pairs; first={missing[:5]}"
        )
    out: dict[int, dict[str, float]] = {}
    for pair, base in anchor.samples_by_pair.items():
        cand = candidate.samples_by_pair[pair]
        seg_delta = base["seg"] - cand["seg"]
        pose_delta = base["pose"] - cand["pose"]
        out[pair] = {
            "seg_delta_vs_c091": seg_delta,
            "pose_delta_vs_c091": pose_delta,
            "combined_delta_vs_c091": seg_delta + pose_delta,
            "candidate_seg_contribution": cand["seg"],
            "candidate_pose_contribution": cand["pose"],
            "candidate_combined_contribution": cand["combined"],
        }
    return out


def _load_unpacker() -> Any:
    spec = importlib.util.spec_from_file_location("c091_action_atom_unpacker", UNPACKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load unpacker from {UNPACKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _safe_zip_member_name(name: str) -> str:
    parts = Path(name).parts
    if (
        not name
        or name.startswith("/")
        or ".." in parts
        or len(parts) != 1
        or name.startswith(".")
        or name.startswith("__MACOSX/")
        or name.startswith("._")
    ):
        raise ValueError(f"unsafe archive member path: {name!r}")
    return name


def _read_single_p_member(path: Path) -> bytes:
    with zipfile.ZipFile(path, "r") as zf:
        seen: set[str] = set()
        names: list[str] = []
        for info in zf.infolist():
            name = _safe_zip_member_name(info.filename)
            if name in seen:
                raise ValueError(f"duplicate archive member: {name}")
            seen.add(name)
            if not info.is_dir():
                names.append(name)
        if names != ["p"]:
            raise ValueError(f"{path} must contain exactly member 'p'; got {names!r}")
        return zf.read("p")


def _parse_action_records(raw: bytes, *, evidence_source: str) -> tuple[ActionRecord, ...]:
    if len(raw) % 4:
        raise ValueError(f"tile-action stream length must be divisible by 4, got {len(raw)}")
    records: list[ActionRecord] = []
    for offset in range(0, len(raw), 4):
        pair_index = int.from_bytes(raw[offset : offset + 2], "little")
        tile_id = raw[offset + 2]
        action_id = raw[offset + 3]
        if pair_index < 0 or pair_index >= 10_000:
            raise ValueError(f"pair index out of range: {pair_index}")
        records.append(
            ActionRecord(
                pair_index=pair_index,
                tile_id=tile_id,
                action_id=action_id,
                source_action_id=action_id,
                source_index=offset // 4,
                evidence_source=evidence_source,
            )
        )
    return tuple(records)


def _read_archive_action_records(path: Path, *, label: str) -> tuple[ActionRecord, ...]:
    payload = _read_single_p_member(path)
    unpacker = _load_unpacker()
    _header, decoded = unpacker._parse_payload(payload)  # noqa: SLF001
    raw = decoded.get("seg_tile_actions.bin")
    if raw is None:
        raise ValueError(f"{path} has no decoded seg_tile_actions.bin")
    return _parse_action_records(raw, evidence_source=label)


def _load_manifest_action_records(path: Path, *, label: str) -> tuple[ActionRecord, ...]:
    payload = json.loads(path.read_text())
    selected = payload.get("selected_records")
    if not isinstance(selected, list) or not selected:
        raise ValueError(f"{path} has no selected_records")
    records: list[ActionRecord] = []
    for item in selected:
        records.append(
            ActionRecord(
                pair_index=int(item["pair_index"]),
                tile_id=int(item["tile_id"]),
                action_id=int(item["action_id"]),
                source_action_id=(
                    int(item["source_action_id"])
                    if item.get("source_action_id") is not None
                    else int(item["action_id"])
                ),
                source_index=(
                    int(item["source_index"]) if item.get("source_index") is not None else None
                ),
                transform=str(item.get("transform") or "identity"),
                evidence_source=label,
            )
        )
    return tuple(records)


def _archive_profile(path: Path) -> dict[str, Any]:
    payload = _read_single_p_member(path)
    return {
        "archive_bytes": path.stat().st_size,
        "archive_sha256": _sha256_file(path),
        "member_name": "p",
        "payload_bytes": len(payload),
        "status": "passed",
    }


def _default_observation_specs() -> list[ObservationSpec]:
    return [
        ObservationSpec(
            label="pr75_top25_p3_t4",
            trace_path=REPO_ROOT / (
                "experiments/results/lightning_batch/"
                "exact_eval_c067_pr75_actions_top25_p3_t4_20260503T0440Z/"
                "component_trace.json"
            ),
            manifest_path=REPO_ROOT / (
                "experiments/results/c067_pr75_tile_action_subset_candidates_20260503/"
                "c067_pr75_actions_top25_p3/manifest.json"
            ),
            family="pr75_subset",
            evidence_role="subset_exact_trace",
            attribution="pair_delta_equal_share_non_c091_stream_context",
            confidence=0.55,
        ),
        ObservationSpec(
            label="pr75_top40_p3_t4",
            trace_path=REPO_ROOT / (
                "experiments/results/lightning_batch/"
                "exact_eval_c067_pr75_actions_top40_p3_t4_20260503T0440Z/"
                "component_trace.json"
            ),
            manifest_path=REPO_ROOT / (
                "experiments/results/c067_pr75_tile_action_subset_candidates_20260503/"
                "c067_pr75_actions_top40_p3/manifest.json"
            ),
            family="pr75_subset",
            evidence_role="subset_exact_trace",
            attribution="pair_delta_equal_share_non_c091_stream_context",
            confidence=0.55,
        ),
        ObservationSpec(
            label="pr75_top25_ampminus1_p3_t4",
            trace_path=REPO_ROOT / (
                "experiments/results/lightning_batch/"
                "exact_eval_c067_pr75_actions_top25_ampminus1_p3_t4_20260503T0520Z/"
                "component_trace.json"
            ),
            manifest_path=REPO_ROOT / (
                "experiments/results/c067_pr75_tile_action_amp_p3_candidates_20260503_codex/"
                "c067_pr75_actions_top25_ampminus1_p3/manifest.json"
            ),
            family="pr75_subset_amp",
            evidence_role="subset_exact_trace",
            attribution="pair_delta_equal_share_non_c091_stream_context",
            confidence=0.5,
        ),
        ObservationSpec(
            label="pr75_top40_p6_t4",
            trace_path=REPO_ROOT / (
                "experiments/results/lightning_batch/"
                "exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/"
                "component_trace.json"
            ),
            manifest_path=REPO_ROOT / (
                "experiments/results/c067_pr75_tile_action_compiler_p6_trace_ev_20260503_codex/"
                "c067_pr75_actions_top40_p6/manifest.json"
            ),
            family="pr75_p6_policy",
            evidence_role="subset_exact_trace",
            attribution="pair_delta_equal_share_non_c091_stream_context",
            confidence=0.55,
        ),
        ObservationSpec(
            label="pr75_pose_safe_positive_ampminus1_p6_t4",
            trace_path=REPO_ROOT / (
                "experiments/results/lightning_batch/"
                "exact_eval_c067_pr75_qp1_pose_safe_positive_ampminus1_p6_t4_20260503T0632Z/"
                "component_trace.json"
            ),
            manifest_path=REPO_ROOT / (
                "experiments/results/pr75_qp1_nextwave_sub314_20260503_worker/"
                "action_compiler_candidates/c067_pr75_actions_pose_safe_positive_ampminus1_p6/"
                "manifest.json"
            ),
            family="pr75_p6_policy",
            evidence_role="subset_exact_trace",
            attribution="pair_delta_equal_share_non_c091_stream_context",
            confidence=0.5,
        ),
        ObservationSpec(
            label="pr75_lag_eval_top67_p6_t4",
            trace_path=REPO_ROOT / (
                "experiments/results/lightning_batch/"
                "exact_eval_c067_pr75_qp1_lag_eval_top67_p6_t4_20260503T0608Z/"
                "component_trace.json"
            ),
            manifest_path=REPO_ROOT / (
                "experiments/results/c067_pr75_tile_action_compiler_p6_trace_ev_20260503_codex/"
                "c067_pr75_actions_lag_eval_top67_p6/manifest.json"
            ),
            family="pr75_p6_policy",
            evidence_role="subset_exact_trace",
            attribution="pair_delta_equal_share_non_c091_stream_context",
            confidence=0.55,
        ),
        ObservationSpec(
            label="pr75_lag_eval_pose2_top67_p6_t4",
            trace_path=REPO_ROOT / (
                "experiments/results/lightning_batch/"
                "exact_eval_c067_pr75_qp1_lag_eval_pose2_top67_p6_t4_20260503T0608Z/"
                "component_trace.json"
            ),
            manifest_path=REPO_ROOT / (
                "experiments/results/c067_pr75_tile_action_compiler_p6_trace_ev_20260503_codex/"
                "c067_pr75_actions_lag_eval_pose2_top67_p6/manifest.json"
            ),
            family="pr75_p6_policy",
            evidence_role="subset_exact_trace",
            attribution="pair_delta_equal_share_non_c091_stream_context",
            confidence=0.55,
        ),
        ObservationSpec(
            label="pr75_lag_eval_pose4_top67_p6_t4",
            trace_path=REPO_ROOT / (
                "experiments/results/lightning_batch/"
                "exact_eval_c067_pr75_qp1_lag_eval_pose4_top67_p6_t4_20260503T0626Z/"
                "component_trace.json"
            ),
            manifest_path=REPO_ROOT / (
                "experiments/results/c067_pr75_tile_action_compiler_p6_trace_ev_20260503_codex/"
                "c067_pr75_actions_lag_eval_pose4_top67_p6/manifest.json"
            ),
            family="pr75_p6_policy",
            evidence_role="subset_exact_trace",
            attribution="pair_delta_equal_share_non_c091_stream_context",
            confidence=0.55,
        ),
        ObservationSpec(
            label="pr77_public_replay_t4",
            trace_path=REPO_ROOT / (
                "experiments/results/lightning_batch/"
                "exact_eval_pr77_tile_delta_public_replay_t4_20260503T1116Z/"
                "component_trace.json"
            ),
            archive_path=DEFAULT_PR77_ARCHIVE,
            family="pr77_action_only",
            evidence_role="direct_pr77_action_feedback",
            attribution="pair_delta_equal_share_c091_native_action_stream_context",
            confidence=1.0,
        ),
        ObservationSpec(
            label="pr77_action_c089_pose_fixedslice_t4",
            trace_path=REPO_ROOT / (
                "experiments/results/lightning_batch/"
                "exact_eval_pr77_action_pose_fixedslice_t4_20260503T114254Z/"
                "component_trace.json"
            ),
            archive_path=DEFAULT_PR77_FIXEDSLICE_ARCHIVE,
            family="pr77_action_c089_pose_negative",
            evidence_role="pose_toxicity_feedback_not_candidate",
            attribution="pair_delta_equal_share_pose_transplant_negative_context",
            confidence=1.0,
        ),
    ]


def _load_observations(specs: Sequence[ObservationSpec]) -> list[Observation]:
    observations: list[Observation] = []
    for spec in specs:
        if not spec.trace_path.exists():
            continue
        if spec.manifest_path is not None:
            if not spec.manifest_path.exists():
                continue
            records = _load_manifest_action_records(spec.manifest_path, label=spec.label)
        elif spec.archive_path is not None:
            if not spec.archive_path.exists():
                continue
            records = _read_archive_action_records(spec.archive_path, label=spec.label)
        else:
            continue
        observations.append(
            Observation(
                label=spec.label,
                trace=_load_trace(spec.trace_path, label=spec.label),
                action_records=records,
                family=spec.family,
                evidence_role=spec.evidence_role,
                attribution=spec.attribution,
                confidence=spec.confidence,
                manifest_path=spec.manifest_path,
                archive_path=spec.archive_path,
            )
        )
    return observations


def _records_by_pair(records: Iterable[ActionRecord]) -> dict[int, list[ActionRecord]]:
    out: dict[int, list[ActionRecord]] = {}
    for record in records:
        out.setdefault(record.pair_index, []).append(record)
    return out


def _classify_atom(row: dict[str, Any], *, pose_heavy_pairs: set[int]) -> str:
    if row["pair_index"] in pose_heavy_pairs:
        return "pose_toxic_pair"
    if row["pose_toxic_votes"] > 0 and row["component_positive_votes"] > 0:
        return "component_positive_pose_risky"
    if row["pose_toxic_votes"] > 0:
        return "pose_toxic"
    if row["component_positive_votes"] > 0 and row["pose_positive_votes"] > 0:
        return "component_positive_pose_safe"
    if row["component_positive_votes"] > 0:
        return "component_positive_pose_neutral"
    return "neutral_or_negative"


def _aggregate_atoms(
    *,
    anchor: LoadedTrace,
    observations: Sequence[Observation],
    pose_heavy_pairs: set[int],
) -> list[dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = {}
    for observation in observations:
        pair_deltas = _pair_deltas(anchor, observation.trace)
        records_by_pair = _records_by_pair(observation.action_records)
        for pair, records in records_by_pair.items():
            if pair not in pair_deltas:
                continue
            delta = pair_deltas[pair]
            share_count = max(1, len(records))
            for record in records:
                row = aggregates.setdefault(
                    record.atom_id,
                    {
                        "action_id": record.action_id,
                        "atom_id": record.atom_id,
                        "classification": "",
                        "component_positive_votes": 0,
                        "evidence_sources": [],
                        "family_votes": {},
                        "pair_index": record.pair_index,
                        "pose_positive_votes": 0,
                        "pose_toxic_votes": 0,
                        "seg_positive_votes": 0,
                        "source_action_id": record.source_action_id,
                        "source_indices": [],
                        "tile_id": record.tile_id,
                        "total_weight": 0.0,
                        "transform": record.transform,
                        "weighted_combined_equal_share": 0.0,
                        "weighted_pose_equal_share": 0.0,
                        "weighted_seg_equal_share": 0.0,
                        "worst_pose_delta_vs_c091": 0.0,
                    },
                )
                weight = float(observation.confidence)
                seg_share = delta["seg_delta_vs_c091"] / share_count
                pose_share = delta["pose_delta_vs_c091"] / share_count
                combined_share = delta["combined_delta_vs_c091"] / share_count
                row["total_weight"] += weight
                row["weighted_seg_equal_share"] += weight * seg_share
                row["weighted_pose_equal_share"] += weight * pose_share
                row["weighted_combined_equal_share"] += weight * combined_share
                row["worst_pose_delta_vs_c091"] = min(
                    float(row["worst_pose_delta_vs_c091"]),
                    delta["pose_delta_vs_c091"],
                )
                if delta["combined_delta_vs_c091"] > EPS:
                    row["component_positive_votes"] += 1
                if delta["pose_delta_vs_c091"] > EPS:
                    row["pose_positive_votes"] += 1
                if delta["pose_delta_vs_c091"] < -EPS:
                    row["pose_toxic_votes"] += 1
                if delta["seg_delta_vs_c091"] > EPS:
                    row["seg_positive_votes"] += 1
                if record.source_index is not None:
                    row["source_indices"].append(record.source_index)
                source_summary = {
                    "attribution": observation.attribution,
                    "combined_delta_vs_c091": delta["combined_delta_vs_c091"],
                    "evidence_role": observation.evidence_role,
                    "family": observation.family,
                    "label": observation.label,
                    "pair_share_count": share_count,
                    "pose_delta_vs_c091": delta["pose_delta_vs_c091"],
                    "seg_delta_vs_c091": delta["seg_delta_vs_c091"],
                    "weighted_combined_equal_share": weight * combined_share,
                }
                row["evidence_sources"].append(source_summary)
                family_votes = row["family_votes"]
                family_votes[observation.family] = int(family_votes.get(observation.family, 0)) + 1

    rows = []
    for row in aggregates.values():
        total_weight = max(float(row["total_weight"]), EPS)
        row["mean_weighted_combined_equal_share"] = (
            float(row["weighted_combined_equal_share"]) / total_weight
        )
        row["mean_weighted_pose_equal_share"] = (
            float(row["weighted_pose_equal_share"]) / total_weight
        )
        row["mean_weighted_seg_equal_share"] = (
            float(row["weighted_seg_equal_share"]) / total_weight
        )
        row["source_indices"] = sorted(set(row["source_indices"]))
        row["classification"] = _classify_atom(row, pose_heavy_pairs=pose_heavy_pairs)
        row["dispatchable_atom"] = row["classification"] in {
            "component_positive_pose_safe",
            "component_positive_pose_neutral",
        }
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            not bool(row["dispatchable_atom"]),
            -float(row["mean_weighted_combined_equal_share"]),
            row["pair_index"],
            row["tile_id"],
            row["action_id"],
        ),
    )


def _top_pair_indices(
    trace: LoadedTrace,
    *,
    field: str,
    count: int,
    reverse: bool = True,
) -> list[int]:
    return [
        pair
        for pair, _sample in sorted(
            trace.samples_by_pair.items(),
            key=lambda item: item[1][field],
            reverse=reverse,
        )[:count]
    ]


def _trace_summary(anchor: LoadedTrace, trace: LoadedTrace) -> dict[str, Any]:
    deltas = _pair_deltas(anchor, trace)
    component_delta = trace.component_score - anchor.component_score
    summary = {
        "archive_bytes": trace.archive_bytes,
        "archive_sha256": trace.archive_sha256,
        "component_score": trace.component_score,
        "component_score_delta_vs_c091_trace": component_delta,
        "contest_auth_eval": _contest_auth_eval_summary(trace.path),
        "path": _rel(trace.path),
        "pose_score": trace.pose_score,
        "score_recomputed_from_trace": trace.score,
        "score_delta_vs_c091_trace": (
            trace.score - anchor.score if trace.score is not None and anchor.score is not None else None
        ),
        "seg_score": trace.seg_score,
        "top_component_positive_pairs_vs_c091": [
            pair
            for pair, _delta in sorted(
                deltas.items(),
                key=lambda item: item[1]["combined_delta_vs_c091"],
                reverse=True,
            )[:12]
        ],
        "top_pose_heavy_pairs": _top_pair_indices(trace, field="pose", count=20),
        "top_seg_heavy_pairs": _top_pair_indices(trace, field="seg", count=12),
    }
    official_score = summary["contest_auth_eval"].get("score_recomputed_from_components")
    if official_score is not None:
        summary["official_score_delta_vs_c091"] = float(official_score) - C091_SCORE
    return summary


def _contest_auth_eval_summary(trace_path: Path) -> dict[str, Any]:
    path = trace_path.parent / "contest_auth_eval.json"
    if not path.exists():
        path = trace_path.parent / "contest_auth_eval.adjudicated.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    provenance = payload.get("provenance") if isinstance(payload, dict) else None
    archive_sha = provenance.get("archive_sha256") if isinstance(provenance, dict) else None
    return {
        "archive_bytes": payload.get("archive_size_bytes"),
        "archive_sha256": archive_sha,
        "avg_posenet_dist": payload.get("avg_posenet_dist"),
        "avg_segnet_dist": payload.get("avg_segnet_dist"),
        "path": _rel(path),
        "score_pose_contribution": payload.get("score_pose_contribution"),
        "score_rate_contribution": payload.get("score_rate_contribution"),
        "score_recomputed_from_components": payload.get("score_recomputed_from_components"),
        "score_seg_contribution": payload.get("score_seg_contribution"),
    }


def _break_even(archive_bytes: int) -> dict[str, Any]:
    delta_bytes = archive_bytes - C091_BYTES
    rate_delta = delta_bytes * RATE_SCORE_PER_BYTE
    score_if_components_unchanged = C091_SCORE + rate_delta
    required = max(0.0, score_if_components_unchanged - SUB314_TARGET)
    return {
        "archive_delta_bytes_vs_c091": delta_bytes,
        "rate_score_delta_vs_c091": rate_delta,
        "score_if_components_unchanged": score_if_components_unchanged,
        "sub314_component_score_improvement_needed": required,
        "sub314_equivalent_bytes_needed_after_candidate": (
            math.ceil(required / RATE_SCORE_PER_BYTE) if required > 0.0 else 0
        ),
    }


def _policy_upper_bound(atoms: Sequence[dict[str, Any]]) -> dict[str, Any]:
    dispatchable = [row for row in atoms if row["dispatchable_atom"]]
    positive = [
        row
        for row in dispatchable
        if float(row["mean_weighted_combined_equal_share"]) > 0.0
    ]
    top = sorted(
        positive,
        key=lambda row: float(row["mean_weighted_combined_equal_share"]),
        reverse=True,
    )[:40]
    modeled = sum(float(row["mean_weighted_combined_equal_share"]) for row in top)
    return {
        "eligible_atom_count": len(dispatchable),
        "positive_eligible_atom_count": len(positive),
        "top_atom_count_used": len(top),
        "modelled_component_improvement_upper_bound": modeled,
        "top_atom_ids": [row["atom_id"] for row in top[:20]],
    }


def _write_atom_csv(path: Path, atoms: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "atom_id",
        "classification",
        "dispatchable_atom",
        "pair_index",
        "tile_id",
        "source_action_id",
        "action_id",
        "transform",
        "mean_weighted_combined_equal_share",
        "mean_weighted_pose_equal_share",
        "mean_weighted_seg_equal_share",
        "component_positive_votes",
        "pose_toxic_votes",
        "pose_positive_votes",
        "seg_positive_votes",
        "worst_pose_delta_vs_c091",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, atom in enumerate(atoms, start=1):
            row = {field: atom.get(field) for field in fields}
            row["rank"] = rank
            writer.writerow(row)


def _safe_archive_profiles(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for path in paths:
        if path.exists():
            profiles[_rel(path) or str(path)] = _archive_profile(path)
    return profiles


def build_policy(
    *,
    output_dir: Path,
    anchor_trace_path: Path = DEFAULT_ANCHOR_TRACE,
    anchor_archive: Path = DEFAULT_ANCHOR_ARCHIVE,
    specs: Sequence[ObservationSpec] | None = None,
) -> dict[str, Any]:
    anchor = _load_trace(anchor_trace_path, label="c091_pr75_minp_public_replay")
    observations = _load_observations(specs or _default_observation_specs())
    if not observations:
        raise ValueError("no action observations loaded")
    fixedslice = next(
        (
            obs.trace
            for obs in observations
            if obs.label == "pr77_action_c089_pose_fixedslice_t4"
        ),
        None,
    )
    pose_heavy_pairs = set(_top_pair_indices(fixedslice, field="pose", count=20)) if fixedslice else set()
    atoms = _aggregate_atoms(anchor=anchor, observations=observations, pose_heavy_pairs=pose_heavy_pairs)
    upper_bound = _policy_upper_bound(atoms)
    best_byte_screen_bytes = min(
        [C091_BYTES]
        + [obs.trace.archive_bytes for obs in observations if obs.trace.archive_bytes is not None]
    )
    break_even = _break_even(int(best_byte_screen_bytes))
    exact_eval_justified = (
        bool(upper_bound["positive_eligible_atom_count"])
        and float(upper_bound["modelled_component_improvement_upper_bound"])
        >= float(break_even["sub314_component_score_improvement_needed"])
    )
    trace_summaries = {
        observation.label: _trace_summary(anchor, observation.trace)
        for observation in observations
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = {
        "anchor": {
            "archive_bytes": C091_BYTES,
            "archive_path": _rel(anchor_archive),
            "archive_sha256": C091_SHA256,
            "component_trace_path": _rel(anchor_trace_path),
            "score": C091_SCORE,
            "trace_component_score": anchor.component_score,
            "trace_score_recomputed": anchor.score,
        },
        "atom_count": len(atoms),
        "break_even_for_best_byte_screen": break_even,
        "dispatch_decision": {
            "exact_eval_justified": exact_eval_justified,
            "next_command_if_yes": None,
            "reason": (
                "no exact-eval dispatch: all observed exact global action candidates are "
                "worse than C091, PR77/C089-pose fixed-slice is strongly pose-toxic, "
                "and the conservative atom upper bound does not clear the sub-0.314 "
                "component break-even"
                if not exact_eval_justified
                else "exact eval would require a fresh lane claim before dispatch"
            ),
        },
        "evidence_grade": "empirical_exact_trace_planning_no_score_claim",
        "no_candidate_archives_emitted": True,
        "observations": [
            {
                "archive_path": _rel(obs.archive_path),
                "attribution": obs.attribution,
                "confidence": obs.confidence,
                "evidence_role": obs.evidence_role,
                "family": obs.family,
                "label": obs.label,
                "manifest_path": _rel(obs.manifest_path),
                "record_count": len(obs.action_records),
                "trace_path": _rel(obs.trace.path),
            }
            for obs in observations
        ],
        "parser_safety": _safe_archive_profiles(
            [anchor_archive]
            + [obs.archive_path for obs in observations if obs.archive_path is not None]
        ),
        "policy_upper_bound": upper_bound,
        "ranked_atoms": atoms,
        "schema": SCHEMA,
        "score_claim": False,
        "sub314_target": SUB314_TARGET,
        "tool": TOOL,
        "trace_summaries": trace_summaries,
    }
    _write_json(output_dir / "ranked_atom_policy.json", policy)
    _write_atom_csv(output_dir / "ranked_atom_policy.csv", atoms)
    return policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--anchor-trace", type=Path, default=DEFAULT_ANCHOR_TRACE)
    parser.add_argument("--anchor-archive", type=Path, default=DEFAULT_ANCHOR_ARCHIVE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    policy = build_policy(
        output_dir=args.output_dir,
        anchor_trace_path=args.anchor_trace,
        anchor_archive=args.anchor_archive,
    )
    print(
        json.dumps(
            {
                "atom_count": policy["atom_count"],
                "exact_eval_justified": policy["dispatch_decision"]["exact_eval_justified"],
                "output_dir": str(args.output_dir.resolve()),
                "policy_upper_bound": policy["policy_upper_bound"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
