#!/usr/bin/env python3
"""Plan C102-native tile-action atoms from public action priors.

This is a local planning-only tool. It mines PR75/PR77 action records and exact
component traces, ranks atoms against the C102/top192 exact frontier, and emits
JSON/CSV planning artifacts. It does not build archives, dispatch GPU jobs, or
make score claims.
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

try:
    import brotli
except ImportError:  # pragma: no cover - local Pact env has brotli.
    brotli = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
UNPACKER_PATH = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"
TOOL = "experiments/plan_c102_native_action_atoms.py"
SCHEMA = "c102_native_action_atoms_policy_v1"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments/results/c102_native_action_atoms_20260503_worker"
)

RATE_SCORE_PER_BYTE = 25.0 / 37_545_489
TARGET_SCORE = 0.31
C102_SCORE = 0.31514430182167497
C102_BYTES = 276_485
C102_SHA256 = "79091f2c3f0c30ef3ca512808f3adc0306010e7f57fed3a09b3664c16fea4ea8"
C101_RENDERER_X_TOP192_NEGATIVE_SHA256 = (
    "d79d1556b55ba7e36c5aaf91d5b04320587975f1303698d8f1089bd5f399d0f3"
)
PR77_ACTION_TRANSPLANT_NEGATIVE_SHA256 = (
    "27866172e76d27113e86a30f722588fd668f81a949be3acbe1e92cddc9a6a1d8"
)
EPS = 1e-12

DEFAULT_ANCHOR_TRACE = REPO_ROOT / (
    "experiments/results/lightning_batch/"
    "exact_eval_c091_next_cem_pose_waterfill_top192_s0125_m06_t4_20260503T1238Z/"
    "component_trace.json"
)
DEFAULT_ANCHOR_ARCHIVE = DEFAULT_ANCHOR_TRACE.parent / "archive.zip"
DEFAULT_PR77_ARCHIVE = (
    REPO_ROOT / "experiments/results/top_submission_reverse_engineering_20260503_pr77/archive.zip"
)
DEFAULT_PR77_FIXEDSLICE_ARCHIVE = REPO_ROOT / (
    "experiments/results/pr77_action_pose_mixed_container_20260503_codex/"
    "pr77_actions_pr75mask_renderer_c089pose_fixedslice/archive.zip"
)
DEFAULT_EXISTING_ACTION_BUILD_DIR = (
    REPO_ROOT / "experiments/results/c101_native_action_atom_worker_20260503"
)
DEFAULT_C101_NEGATIVE_EVAL = REPO_ROOT / (
    "experiments/results/lightning_batch/"
    "exact_eval_c101_renderer_x_top192_stack_t4_20260503T1540Z/"
    "contest_auth_eval.adjudicated.json"
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

    def encode4(self) -> bytes:
        return (
            int(self.pair_index).to_bytes(2, "little")
            + bytes([int(self.tile_id), int(self.action_id)])
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
    archive_sha = trace_inputs.get("archive_sha256") if isinstance(trace_inputs, dict) else None
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
            "seg_delta_vs_c102": seg_delta,
            "pose_delta_vs_c102": pose_delta,
            "combined_delta_vs_c102": seg_delta + pose_delta,
            "candidate_seg_contribution": cand["seg"],
            "candidate_pose_contribution": cand["pose"],
            "candidate_combined_contribution": cand["combined"],
        }
    return out


def _load_unpacker() -> Any:
    spec = importlib.util.spec_from_file_location("c102_action_atom_unpacker", UNPACKER_PATH)
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
    return _parse_action_records(bytes(raw), evidence_source=label)


def _load_manifest_action_records(path: Path, *, label: str) -> tuple[ActionRecord, ...]:
    payload = json.loads(path.read_text())
    selected = payload.get("selected_records")
    if not isinstance(selected, list) or not selected:
        raise ValueError(f"{path} has no selected_records")
    records: list[ActionRecord] = []
    for index, item in enumerate(selected):
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
                    int(item["source_index"]) if item.get("source_index") is not None else index
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
            attribution="pair_delta_equal_share_non_c102_stream_context",
            confidence=0.45,
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
            attribution="pair_delta_equal_share_non_c102_stream_context",
            confidence=0.45,
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
            attribution="pair_delta_equal_share_non_c102_stream_context",
            confidence=0.45,
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
            attribution="pair_delta_equal_share_non_c102_stream_context",
            confidence=0.5,
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
            attribution="pair_delta_equal_share_c102_native_action_stream_context",
            confidence=0.8,
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


def _anchor_record_keys(records: Sequence[ActionRecord]) -> set[tuple[int, int, int]]:
    return {(record.pair_index, record.tile_id, record.action_id) for record in records}


def _classify_atom(row: dict[str, Any], *, pose_heavy_pairs: set[int]) -> str:
    if bool(row["no_op_relative_to_c102"]):
        return "exact_anchor_duplicate_noop"
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
    anchor_records: Sequence[ActionRecord],
    pose_heavy_pairs: set[int],
) -> list[dict[str, Any]]:
    anchor_keys = _anchor_record_keys(anchor_records)
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
                anchor_duplicate = (
                    record.pair_index,
                    record.tile_id,
                    record.action_id,
                ) in anchor_keys and record.transform == "identity"
                row = aggregates.setdefault(
                    record.atom_id,
                    {
                        "action_id": record.action_id,
                        "atom_id": record.atom_id,
                        "classification": "",
                        "component_positive_votes": 0,
                        "dispatchable_atom": False,
                        "evidence_sources": [],
                        "family_votes": {},
                        "no_op_relative_to_c102": anchor_duplicate,
                        "no_op_status": (
                            "exact_anchor_duplicate"
                            if anchor_duplicate
                            else "changes_c102_action_stream_if_built"
                        ),
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
                        "worst_pose_delta_vs_c102": 0.0,
                    },
                )
                row["no_op_relative_to_c102"] = bool(
                    row["no_op_relative_to_c102"] or anchor_duplicate
                )
                if bool(row["no_op_relative_to_c102"]):
                    row["no_op_status"] = "exact_anchor_duplicate"
                weight = float(observation.confidence)
                seg_share = delta["seg_delta_vs_c102"] / share_count
                pose_share = delta["pose_delta_vs_c102"] / share_count
                combined_share = delta["combined_delta_vs_c102"] / share_count
                row["total_weight"] += weight
                row["weighted_seg_equal_share"] += weight * seg_share
                row["weighted_pose_equal_share"] += weight * pose_share
                row["weighted_combined_equal_share"] += weight * combined_share
                row["worst_pose_delta_vs_c102"] = min(
                    float(row["worst_pose_delta_vs_c102"]),
                    delta["pose_delta_vs_c102"],
                )
                if delta["combined_delta_vs_c102"] > EPS:
                    row["component_positive_votes"] += 1
                if delta["pose_delta_vs_c102"] > EPS:
                    row["pose_positive_votes"] += 1
                if delta["pose_delta_vs_c102"] < -EPS:
                    row["pose_toxic_votes"] += 1
                if delta["seg_delta_vs_c102"] > EPS:
                    row["seg_positive_votes"] += 1
                if record.source_index is not None:
                    row["source_indices"].append(record.source_index)
                row["evidence_sources"].append(
                    {
                        "attribution": observation.attribution,
                        "combined_delta_vs_c102": delta["combined_delta_vs_c102"],
                        "evidence_role": observation.evidence_role,
                        "family": observation.family,
                        "label": observation.label,
                        "pair_share_count": share_count,
                        "pose_delta_vs_c102": delta["pose_delta_vs_c102"],
                        "seg_delta_vs_c102": delta["seg_delta_vs_c102"],
                        "weighted_combined_equal_share": weight * combined_share,
                    }
                )
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
        row["dispatchable_atom"] = False
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            bool(row["no_op_relative_to_c102"]),
            not row["classification"].startswith("component_positive"),
            -float(row["mean_weighted_combined_equal_share"]),
            row["pair_index"],
            row["tile_id"],
            row["action_id"],
        ),
    )


def _top_pair_indices(trace: LoadedTrace, *, field: str, count: int) -> list[int]:
    return [
        pair
        for pair, _sample in sorted(
            trace.samples_by_pair.items(),
            key=lambda item: item[1][field],
            reverse=True,
        )[:count]
    ]


def _contest_auth_eval_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": _rel(path), "status": "missing"}
    payload = json.loads(path.read_text())
    provenance = payload.get("provenance") if isinstance(payload, dict) else None
    archive_sha = provenance.get("archive_sha256") if isinstance(provenance, dict) else None
    return {
        "archive_bytes": payload.get("archive_size_bytes"),
        "archive_sha256": archive_sha,
        "avg_posenet_dist": payload.get("avg_posenet_dist"),
        "avg_segnet_dist": payload.get("avg_segnet_dist"),
        "path": _rel(path),
        "score_recomputed_from_components": payload.get("score_recomputed_from_components"),
    }


def _trace_summary(anchor: LoadedTrace, trace: LoadedTrace) -> dict[str, Any]:
    deltas = _pair_deltas(anchor, trace)
    return {
        "archive_bytes": trace.archive_bytes,
        "archive_sha256": trace.archive_sha256,
        "component_score": trace.component_score,
        "component_score_delta_vs_c102_trace": trace.component_score - anchor.component_score,
        "path": _rel(trace.path),
        "pose_score": trace.pose_score,
        "score_recomputed_from_trace": trace.score,
        "score_delta_vs_c102_trace": (
            trace.score - anchor.score if trace.score is not None and anchor.score is not None else None
        ),
        "seg_score": trace.seg_score,
        "top_component_positive_pairs_vs_c102": [
            pair
            for pair, _delta in sorted(
                deltas.items(),
                key=lambda item: item[1]["combined_delta_vs_c102"],
                reverse=True,
            )[:12]
        ],
        "top_pose_heavy_pairs": _top_pair_indices(trace, field="pose", count=20),
        "top_seg_heavy_pairs": _top_pair_indices(trace, field="seg", count=12),
    }


def _break_even(archive_bytes: int) -> dict[str, Any]:
    delta_bytes = archive_bytes - C102_BYTES
    rate_delta = delta_bytes * RATE_SCORE_PER_BYTE
    score_if_components_unchanged = C102_SCORE + rate_delta
    required = max(0.0, score_if_components_unchanged - TARGET_SCORE)
    return {
        "archive_delta_bytes_vs_c102": delta_bytes,
        "rate_score_delta_vs_c102": rate_delta,
        "score_if_components_unchanged": score_if_components_unchanged,
        "target_component_score_improvement_needed": required,
        "target_equivalent_bytes_needed_after_candidate": (
            math.ceil(required / RATE_SCORE_PER_BYTE) if required > 0.0 else 0
        ),
    }


def _estimate_action_stream_bytes(records: Sequence[ActionRecord]) -> dict[str, Any]:
    raw = b"".join(record.encode4() for record in records)
    if brotli is not None:
        compressed = brotli.compress(raw, quality=9, lgwin=10)
        method = "brotli_q9_lgwin10_raw4_record_proxy"
    else:
        import zlib

        compressed = zlib.compress(raw, level=9)
        method = "zlib9_fallback_proxy"
    return {
        "method": method,
        "selected_record_count": len(records),
        "selected_raw_bytes": len(raw),
        "selected_encoded_proxy_bytes": len(compressed),
        "proxy_archive_bytes_replace_c102_action_stream": C102_BYTES - 255 + len(compressed),
        "proxy_note": "rate proxy only; not an archive-builder byte claim",
    }


def _record_from_atom(atom: dict[str, Any]) -> ActionRecord:
    return ActionRecord(
        pair_index=int(atom["pair_index"]),
        tile_id=int(atom["tile_id"]),
        action_id=int(atom["action_id"]),
        source_action_id=(
            int(atom["source_action_id"]) if atom.get("source_action_id") is not None else None
        ),
        transform=str(atom.get("transform") or "identity"),
    )


def _policy_from_atoms(
    *,
    policy_id: str,
    atoms: Sequence[dict[str, Any]],
    rationale: str,
    actual_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    records = [_record_from_atom(atom) for atom in atoms]
    byte_proxy = _estimate_action_stream_bytes(records)
    actual_builder = None
    archive_bytes = int(byte_proxy["proxy_archive_bytes_replace_c102_action_stream"])
    if actual_manifest is not None:
        output_archive = actual_manifest.get("output_archive")
        if isinstance(output_archive, dict) and output_archive.get("bytes") is not None:
            archive_bytes = int(output_archive["bytes"])
            actual_builder = {
                "archive_bytes": archive_bytes,
                "archive_path": output_archive.get("path"),
                "archive_sha256": output_archive.get("sha256"),
                "manifest_schema": actual_manifest.get("schema"),
                "source_manifest_policy": actual_manifest.get("policy"),
            }
    expected = sum(float(atom["mean_weighted_combined_equal_share"]) for atom in atoms)
    no_op_count = sum(1 for atom in atoms if bool(atom["no_op_relative_to_c102"]))
    no_op_status = (
        "exact_noop_all_selected_records_duplicate_c102"
        if atoms and no_op_count == len(atoms)
        else "mixed_noop_risk"
        if no_op_count
        else "non_noop_if_builder_changes_action_stream"
    )
    break_even = _break_even(archive_bytes)
    support_labels = sorted(
        {
            str(source["label"])
            for atom in atoms
            for source in atom.get("evidence_sources", [])
            if isinstance(source, dict) and source.get("label")
        }
    )
    records_selected = [
        {
            "action_id": int(atom["action_id"]),
            "atom_id": atom["atom_id"],
            "classification": atom["classification"],
            "expected_component_benefit_proxy": atom["mean_weighted_combined_equal_share"],
            "no_op_status": atom["no_op_status"],
            "pair_index": int(atom["pair_index"]),
            "source_action_id": atom.get("source_action_id"),
            "tile_id": int(atom["tile_id"]),
            "transform": atom.get("transform"),
        }
        for atom in atoms
    ]
    return {
        "candidate_policy_id": policy_id,
        "charged_byte_proxy": byte_proxy,
        "actual_builder_bytes": actual_builder,
        "expected_component_benefit_proxy": expected,
        "break_even_vs_0_31": break_even,
        "dispatchable": False,
        "dispatchable_reason": (
            "planning-only: no exact CUDA evidence for this C102-native policy clears "
            "the target break-even and lane-claim gates"
        ),
        "no_op_status": no_op_status,
        "records_selected": records_selected,
        "records_selected_count": len(records_selected),
        "rationale": rationale,
        "score_claim": False,
        "source_priors": support_labels,
    }


def _load_existing_action_build_manifests(path: Path) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return manifests
    for manifest_path in sorted(path.glob("*/manifest.json")):
        payload = json.loads(manifest_path.read_text())
        candidate_id = str(payload.get("candidate_id") or manifest_path.parent.name)
        payload["_manifest_path"] = _rel(manifest_path)
        manifests[candidate_id] = payload
    return manifests


def _build_policy_rows(
    *,
    atoms: Sequence[dict[str, Any]],
    anchor_records: Sequence[ActionRecord],
    existing_manifests: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    eligible = [
        atom
        for atom in atoms
        if atom["classification"] in {
            "component_positive_pose_safe",
            "component_positive_pose_neutral",
        }
        and not bool(atom["no_op_relative_to_c102"])
        and float(atom["mean_weighted_combined_equal_share"]) > 0.0
    ]
    pose_safe = [
        atom
        for atom in eligible
        if atom["classification"] == "component_positive_pose_safe"
    ]
    policies = [
        _policy_from_atoms(
            policy_id="c102_anchor_identity_noop_control",
            atoms=[
                {
                    "action_id": record.action_id,
                    "atom_id": record.atom_id,
                    "classification": "exact_anchor_duplicate_noop",
                    "evidence_sources": [],
                    "mean_weighted_combined_equal_share": 0.0,
                    "no_op_relative_to_c102": True,
                    "no_op_status": "exact_anchor_duplicate",
                    "pair_index": record.pair_index,
                    "source_action_id": record.source_action_id,
                    "tile_id": record.tile_id,
                    "transform": record.transform,
                }
                for record in anchor_records[: min(32, len(anchor_records))]
            ],
            rationale="No-op control proving the planner does not mark anchor duplicates dispatchable.",
        ),
        _policy_from_atoms(
            policy_id="c102_consensus_positive_top32_proxy",
            atoms=eligible[:32],
            rationale="Highest-ranked non-noop component-positive atoms after excluding pose-toxic pairs.",
        ),
        _policy_from_atoms(
            policy_id="c102_pose_safe_positive_top48_proxy",
            atoms=pose_safe[:48],
            rationale="Pose-positive subset only; lower expected benefit but lower PoseNet toxicity risk.",
        ),
        _policy_from_atoms(
            policy_id="c102_consensus_positive_top64_proxy",
            atoms=eligible[:64],
            rationale="Broader top64 atom packet for nonlinear interaction screening; still planning-only.",
            actual_manifest=existing_manifests.get(
                "c101_top192_actions_consensus_positive_top64_ampfit_p6"
            ),
        ),
        _policy_from_atoms(
            policy_id="c102_native_pose_guard_ampfit_existing_builder",
            atoms=pose_safe[:64],
            rationale="Existing C101/top192 native-action builder artifact reused only for byte context.",
            actual_manifest=existing_manifests.get("c101_top192_actions_native_pose_guard_ampfit_p6"),
        ),
        _policy_from_atoms(
            policy_id="c102_all_ampminus1_existing_builder",
            atoms=eligible[: min(108, len(eligible))],
            rationale="Existing all-ampminus1 builder artifact is a high-no-op/high-pose-risk prior, not a dispatch candidate.",
            actual_manifest=existing_manifests.get("c101_top192_actions_all_ampminus1_p6"),
        ),
    ]
    return policies


def _write_atom_csv(path: Path, atoms: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "atom_id",
        "classification",
        "dispatchable_atom",
        "no_op_status",
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
        "worst_pose_delta_vs_c102",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, atom in enumerate(atoms, start=1):
            row = {field: atom.get(field) for field in fields}
            row["rank"] = rank
            writer.writerow(row)


def _write_policy_csv(path: Path, policies: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "candidate_policy_id",
        "records_selected_count",
        "expected_component_benefit_proxy",
        "archive_delta_bytes_vs_c102",
        "target_component_score_improvement_needed",
        "no_op_status",
        "dispatchable",
        "source_priors",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        ranked = sorted(
            policies,
            key=lambda row: (
                bool(row["dispatchable"]),
                float(row["expected_component_benefit_proxy"])
                - float(row["break_even_vs_0_31"]["target_component_score_improvement_needed"]),
            ),
            reverse=True,
        )
        for rank, policy in enumerate(ranked, start=1):
            break_even = policy["break_even_vs_0_31"]
            writer.writerow(
                {
                    "rank": rank,
                    "candidate_policy_id": policy["candidate_policy_id"],
                    "records_selected_count": policy["records_selected_count"],
                    "expected_component_benefit_proxy": policy["expected_component_benefit_proxy"],
                    "archive_delta_bytes_vs_c102": break_even["archive_delta_bytes_vs_c102"],
                    "target_component_score_improvement_needed": break_even[
                        "target_component_score_improvement_needed"
                    ],
                    "no_op_status": policy["no_op_status"],
                    "dispatchable": policy["dispatchable"],
                    "source_priors": ";".join(policy["source_priors"]),
                }
            )


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
    existing_action_build_dir: Path = DEFAULT_EXISTING_ACTION_BUILD_DIR,
    c101_negative_eval: Path = DEFAULT_C101_NEGATIVE_EVAL,
    specs: Sequence[ObservationSpec] | None = None,
) -> dict[str, Any]:
    anchor = _load_trace(anchor_trace_path, label="c102_top192_frontier")
    anchor_records = _read_archive_action_records(anchor_archive, label="c102_top192_frontier")
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
    atoms = _aggregate_atoms(
        anchor=anchor,
        observations=observations,
        anchor_records=anchor_records,
        pose_heavy_pairs=pose_heavy_pairs,
    )
    existing_manifests = _load_existing_action_build_manifests(existing_action_build_dir)
    policies = _build_policy_rows(
        atoms=atoms,
        anchor_records=anchor_records,
        existing_manifests=existing_manifests,
    )
    policies = sorted(
        policies,
        key=lambda row: (
            float(row["expected_component_benefit_proxy"])
            - float(row["break_even_vs_0_31"]["target_component_score_improvement_needed"]),
            -row["records_selected_count"],
        ),
        reverse=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = {
        "anchor": {
            "archive_bytes": C102_BYTES,
            "archive_path": _rel(anchor_archive),
            "archive_sha256": C102_SHA256,
            "component_trace_path": _rel(anchor_trace_path),
            "score": C102_SCORE,
            "target_score": TARGET_SCORE,
            "trace_component_score": anchor.component_score,
            "trace_score_recomputed": anchor.score,
        },
        "atom_count": len(atoms),
        "dispatch_decision": {
            "exact_eval_justified": False,
            "next_command_if_yes": None,
            "reason": (
                "planning-only C102 action atoms: direct PR75/PR77 transplants are exact "
                "negatives or non-native priors, the best policy proxies miss the <=0.31 "
                "break-even by orders of magnitude, and no fresh non-noop C102 archive "
                "has exact CUDA support"
            ),
        },
        "evidence_grade": "empirical_exact_trace_planning_no_score_claim",
        "existing_negative_controls": {
            "c101_renderer_x_top192": _contest_auth_eval_summary(c101_negative_eval),
            "known_negative_archive_shas": [
                C101_RENDERER_X_TOP192_NEGATIVE_SHA256,
                PR77_ACTION_TRANSPLANT_NEGATIVE_SHA256,
            ],
        },
        "no_remote_dispatch_performed": True,
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
            [anchor_archive] + [obs.archive_path for obs in observations if obs.archive_path is not None]
        ),
        "ranked_atoms": atoms,
        "ranked_policies": policies,
        "required_policy_fields": [
            "candidate_policy_id",
            "source_priors",
            "records_selected",
            "charged_byte_proxy",
            "actual_builder_bytes",
            "expected_component_benefit_proxy",
            "break_even_vs_0_31",
            "no_op_status",
            "dispatchable",
        ],
        "schema": SCHEMA,
        "score_claim": False,
        "tool": TOOL,
        "trace_summaries": {obs.label: _trace_summary(anchor, obs.trace) for obs in observations},
    }
    _write_json(output_dir / "ranked_atom_policy.json", policy)
    _write_atom_csv(output_dir / "ranked_atoms.csv", atoms)
    _write_policy_csv(output_dir / "ranked_policies.csv", policies)
    return policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--anchor-trace", type=Path, default=DEFAULT_ANCHOR_TRACE)
    parser.add_argument("--anchor-archive", type=Path, default=DEFAULT_ANCHOR_ARCHIVE)
    parser.add_argument(
        "--existing-action-build-dir",
        type=Path,
        default=DEFAULT_EXISTING_ACTION_BUILD_DIR,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    policy = build_policy(
        output_dir=args.output_dir,
        anchor_trace_path=args.anchor_trace,
        anchor_archive=args.anchor_archive,
        existing_action_build_dir=args.existing_action_build_dir,
    )
    print(
        json.dumps(
            {
                "atom_count": policy["atom_count"],
                "exact_eval_justified": policy["dispatch_decision"]["exact_eval_justified"],
                "output_dir": str(args.output_dir.resolve()),
                "top_policies": [
                    {
                        "candidate_policy_id": row["candidate_policy_id"],
                        "dispatchable": row["dispatchable"],
                        "expected_component_benefit_proxy": row[
                            "expected_component_benefit_proxy"
                        ],
                        "target_component_score_improvement_needed": row[
                            "break_even_vs_0_31"
                        ]["target_component_score_improvement_needed"],
                    }
                    for row in policy["ranked_policies"][:3]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
