#!/usr/bin/env python3
"""Build C101/top192-native seg_tile_actions atom candidates.

This is a local archive builder only. It preserves the exact C101/top192 mask,
renderer, and pose streams, synthesizes a new charged ``seg_tile_actions.bin``
stream from mined exact traces/public action streams, validates the result
through the robust-current runtime parser, and emits deterministic single-member
archives plus manifests. It does not dispatch GPU work and does not make score
claims.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
import zipfile
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import brotli


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.submission_archive import validate_seg_tile_actions_payload


UNPACKER_PATH = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"
DEFAULT_BASE_ARCHIVE = REPO_ROOT / (
    "experiments/results/lightning_batch/"
    "exact_eval_c091_next_cem_pose_waterfill_top192_s0125_m06_t4_20260503T1238Z/"
    "archive.zip"
)
DEFAULT_BASE_TRACE = DEFAULT_BASE_ARCHIVE.parent / "component_trace.json"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments/results/c101_native_action_atom_worker_20260503"
)

TOOL = "experiments/build_c101_native_action_atom_candidate.py"
SCHEMA = "c101_native_action_atom_candidate_v1"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
PAYLOAD_MEMBER = "p"
RATE_DENOM = 37_545_489
TARGET_SCORE = 0.31  # [heuristic: aspirational floor below current PR-frontier]
TOP192_SHA256 = "79091f2c3f0c30ef3ca512808f3adc0306010e7f57fed3a09b3664c16fea4ea8"
PR77_ACTION_TRANSPLANT_NEGATIVE_SHA256 = (
    "27866172e76d27113e86a30f722588fd668f81a949be3acbe1e92cddc9a6a1d8"
)
PR75_POSE_SAFE_P6_NEGATIVE_SHA256 = (
    "6e6ec4609c6da581b4113c5c06e67970542a9d8e22c4866959fe618aaff2c796"
)
DEFAULT_FORBIDDEN_ARCHIVE_SHAS = {
    TOP192_SHA256,
    PR77_ACTION_TRANSPLANT_NEGATIVE_SHA256,
    PR75_POSE_SAFE_P6_NEGATIVE_SHA256,
}
CUDA_AUTH_EVAL_REQUIRED = (
    "Claim the lane first, then run archive.zip -> inflate.sh -> "
    "upstream/evaluate.py via experiments/contest_auth_eval.py --device cuda"
)


@dataclass(frozen=True)
class ActionRecord:
    index: int
    pair_index: int
    tile_id: int
    action_id: int
    source_action_id: int | None = None
    transform: str = "identity"
    support: dict[str, Any] | None = None

    def encode4(self) -> bytes:
        return (
            int(self.pair_index).to_bytes(2, "little")
            + bytes([int(self.tile_id), int(self.action_id)])
        )


@dataclass(frozen=True)
class SourceArchive:
    path: Path
    archive_bytes: int
    archive_sha256: str
    payload: bytes
    payload_sha256: str
    mask_br: bytes
    renderer_br: bytes
    actions_br: bytes
    pose_br: bytes
    decoded: dict[str, bytes]
    runtime_members: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class Trace:
    label: str
    path: Path
    score: float | None
    archive_bytes: int | None
    samples_by_pair: dict[int, dict[str, float]]


@dataclass(frozen=True)
class ObservationSpec:
    label: str
    trace_path: Path
    manifest_path: Path | None
    archive_path: Path | None
    confidence: float
    role: str


class CandidateBuildError(ValueError):
    """Raised when a candidate is not safe to write or dispatch."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _safe_zip_member_name(name: str) -> str:
    path = Path(name)
    if (
        not name
        or name.startswith("/")
        or ".." in path.parts
        or len(path.parts) != 1
        or name.startswith(".")
        or name.startswith("__MACOSX/")
        or name.startswith("._")
    ):
        raise CandidateBuildError(f"unsafe ZIP member path: {name!r}")
    return name


def _load_unpacker() -> Any:
    spec = importlib.util.spec_from_file_location("c101_native_action_unpacker", UNPACKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load unpacker from {UNPACKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_single_payload(path: Path) -> bytes:
    with zipfile.ZipFile(path, "r") as zf:
        seen: set[str] = set()
        names: list[str] = []
        for info in zf.infolist():
            name = _safe_zip_member_name(info.filename)
            if name in seen:
                raise CandidateBuildError(f"duplicate ZIP member: {name}")
            seen.add(name)
            if not info.is_dir():
                names.append(name)
        if names != [PAYLOAD_MEMBER]:
            raise CandidateBuildError(
                f"{path} must contain exactly member {PAYLOAD_MEMBER!r}; got {names!r}"
            )
        return zf.read(PAYLOAD_MEMBER)


def _member_summary(header: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    members = header.get("members")
    if not isinstance(members, list):
        raise CandidateBuildError("runtime parser returned no member table")
    out: dict[str, dict[str, Any]] = {}
    for item in members:
        if not isinstance(item, Mapping):
            raise CandidateBuildError("runtime parser returned malformed member metadata")
        name = str(item.get("name"))
        out[name] = {
            "bytes": int(item["bytes"]),
            "codec": str(item["codec"]),
            "decoded_bytes": int(item["decoded_bytes"]),
            "decoded_sha256": str(item["decoded_sha256"]),
            "sha256": str(item["sha256"]),
        }
    return out


def _parse_p3_slices(payload: bytes) -> tuple[bytes, bytes, bytes, bytes]:
    if not payload.startswith(b"P3"):
        raise CandidateBuildError(
            f"C101/top192 source must be a self-describing P3 payload, got {payload[:4]!r}"
        )
    header_size = 2 + struct.calcsize("<IHH")
    if len(payload) <= header_size:
        raise CandidateBuildError("P3 payload is too short")
    mask_len, renderer_len, actions_len = struct.unpack_from("<IHH", payload, 2)
    if min(mask_len, renderer_len, actions_len) <= 0:
        raise CandidateBuildError("P3 payload has an empty required stream")
    cursor = header_size
    mask_end = cursor + mask_len
    renderer_end = mask_end + renderer_len
    actions_end = renderer_end + actions_len
    if actions_end >= len(payload):
        raise CandidateBuildError("P3 stream lengths leave no pose stream")
    return (
        payload[cursor:mask_end],
        payload[mask_end:renderer_end],
        payload[renderer_end:actions_end],
        payload[actions_end:],
    )


def load_source_archive(path: Path, *, unpacker: Any | None = None) -> SourceArchive:
    if unpacker is None:
        unpacker = _load_unpacker()
    path = path.resolve()
    payload = _read_single_payload(path)
    mask_br, renderer_br, actions_br, pose_br = _parse_p3_slices(payload)
    header, decoded_raw = unpacker._parse_payload(payload)  # noqa: SLF001
    payload_format = str(header.get("payload_format"))
    if payload_format != "public_pr75_qzs3_qp1_segactions_p3":
        raise CandidateBuildError(f"unsupported source payload_format={payload_format!r}")
    decoded = {str(name): bytes(data) for name, data in decoded_raw.items()}
    required = {"masks.mkv", "renderer.bin", "seg_tile_actions.bin", "optimized_poses.qp1"}
    missing = sorted(required - set(decoded))
    if missing:
        raise CandidateBuildError(f"runtime parser missed source members: {missing}")
    if not decoded["renderer.bin"].startswith(b"QZS3"):
        raise CandidateBuildError("decoded renderer.bin is not QZS3")
    if not decoded["optimized_poses.qp1"].startswith(b"QP1"):
        raise CandidateBuildError("decoded optimized_poses.qp1 is not QP1")
    validate_seg_tile_actions_payload(
        decoded["seg_tile_actions.bin"],
        source_name="source seg_tile_actions.bin",
    )
    return SourceArchive(
        path=path,
        archive_bytes=path.stat().st_size,
        archive_sha256=_sha256_file(path),
        payload=payload,
        payload_sha256=_sha256_bytes(payload),
        mask_br=mask_br,
        renderer_br=renderer_br,
        actions_br=actions_br,
        pose_br=pose_br,
        decoded=decoded,
        runtime_members=_member_summary(header),
    )


def _parse_action_records(raw: bytes, *, support_label: str = "source") -> list[ActionRecord]:
    validate_seg_tile_actions_payload(raw, source_name=f"{support_label} seg_tile_actions.bin")
    if len(raw) % 4:
        raise CandidateBuildError(f"{support_label}: expected 4-byte action records")
    records: list[ActionRecord] = []
    for offset in range(0, len(raw), 4):
        pair_index = int.from_bytes(raw[offset : offset + 2], "little")
        tile_id = raw[offset + 2]
        action_id = raw[offset + 3]
        if action_id >= 108:
            raise CandidateBuildError(
                f"{support_label}: action id outside PR75 fixed dictionary: {action_id}"
            )
        records.append(
            ActionRecord(
                index=offset // 4,
                pair_index=pair_index,
                tile_id=tile_id,
                action_id=action_id,
                source_action_id=action_id,
                support={"source": support_label},
            )
        )
    if not records:
        raise CandidateBuildError(f"{support_label}: decoded no action records")
    return records


def _load_trace(path: Path, *, label: str) -> Trace:
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
        raise CandidateBuildError(f"{path} has no component-trace samples")
    return Trace(
        label=label,
        path=path,
        score=(
            float(payload["score_recomputed_from_components"])
            if payload.get("score_recomputed_from_components") is not None
            else None
        ),
        archive_bytes=(
            int(payload["archive_size_bytes"])
            if payload.get("archive_size_bytes") is not None
            else None
        ),
        samples_by_pair=samples,
    )


def _load_manifest_records(path: Path, *, label: str) -> list[ActionRecord]:
    payload = json.loads(path.read_text())
    selected = payload.get("selected_records")
    if not isinstance(selected, list) or not selected:
        pair_filter = payload.get("pair_filter")
        selected_pairs = None
        if isinstance(pair_filter, Mapping):
            selected_pairs = pair_filter.get("pair_indices")
        if selected_pairs is None:
            no_op_proof = payload.get("no_op_proof")
            if isinstance(no_op_proof, Mapping):
                selected_pairs = no_op_proof.get("selected_pairs")
        if isinstance(selected_pairs, list) and selected_pairs:
            return [
                ActionRecord(
                    index=index,
                    pair_index=int(pair),
                    tile_id=-1,
                    action_id=-1,
                    source_action_id=-1,
                    transform="pair_level_support_only",
                    support={"source": label, "pair_level_support_only": True},
                )
                for index, pair in enumerate(selected_pairs)
            ]
        raise CandidateBuildError(f"{path} has no selected_records or selected pair list")
    records: list[ActionRecord] = []
    for index, item in enumerate(selected):
        records.append(
            ActionRecord(
                index=int(item.get("source_index", index)),
                pair_index=int(item["pair_index"]),
                tile_id=int(item["tile_id"]),
                action_id=int(item["action_id"]),
                source_action_id=(
                    int(item["source_action_id"])
                    if item.get("source_action_id") is not None
                    else int(item["action_id"])
                ),
                transform=str(item.get("transform") or "identity"),
                support={"source": label},
            )
        )
    return records


def _load_archive_records(path: Path, *, label: str, unpacker: Any) -> list[ActionRecord]:
    payload = _read_single_payload(path)
    _header, decoded = unpacker._parse_payload(payload)  # noqa: SLF001
    actions = decoded.get("seg_tile_actions.bin")
    if actions is None:
        raise CandidateBuildError(f"{path} has no decoded seg_tile_actions.bin")
    return _parse_action_records(bytes(actions), support_label=label)


def _default_observation_specs() -> list[ObservationSpec]:
    return [
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
            archive_path=None,
            confidence=0.45,
            role="ampminus_exact_trace_non_native",
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
            archive_path=None,
            confidence=0.45,
            role="lagrangian_exact_trace_non_native",
        ),
        ObservationSpec(
            label="pr75_qpost_bias040_stack_t4",
            trace_path=REPO_ROOT / (
                "experiments/results/lightning_batch/"
                "exact_eval_pr65_qpost_ix_lagtop67_p6_bias_top040_t4_20260503T1035Z/"
                "component_trace.json"
            ),
            manifest_path=REPO_ROOT / (
                "experiments/results/pr65_qpost_atom_v2_worker_20260503/candidates/"
                "v2_pr65_qpost_bias_poseadv_top040/manifest.json"
            ),
            archive_path=None,
            confidence=0.35,
            role="pr65_qpost_action_interaction_trace",
        ),
        ObservationSpec(
            label="pr77_public_replay_t4",
            trace_path=REPO_ROOT / (
                "experiments/results/lightning_batch/"
                "exact_eval_pr77_tile_delta_public_replay_t4_20260503T1116Z/"
                "component_trace.json"
            ),
            manifest_path=None,
            archive_path=REPO_ROOT
            / "experiments/results/top_submission_reverse_engineering_20260503_pr77/archive.zip",
            confidence=0.9,
            role="pr77_direct_negative_feedback",
        ),
        ObservationSpec(
            label="pr77_action_pose_fixedslice_t4",
            trace_path=REPO_ROOT / (
                "experiments/results/lightning_batch/"
                "exact_eval_pr77_action_pose_fixedslice_t4_20260503T114254Z/"
                "component_trace.json"
            ),
            manifest_path=None,
            archive_path=REPO_ROOT
            / (
                "experiments/results/pr77_action_pose_mixed_container_20260503_codex/"
                "pr77_actions_pr75mask_renderer_c089pose_fixedslice/archive.zip"
            ),
            confidence=0.7,
            role="pr77_action_transplant_pose_toxicity",
        ),
    ]


def _load_observations(
    specs: Sequence[ObservationSpec],
    *,
    anchor: Trace,
    unpacker: Any,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for spec in specs:
        if not spec.trace_path.exists():
            continue
        if spec.manifest_path is not None:
            if not spec.manifest_path.exists():
                continue
            records = _load_manifest_records(spec.manifest_path, label=spec.label)
        elif spec.archive_path is not None:
            if not spec.archive_path.exists():
                continue
            records = _load_archive_records(spec.archive_path, label=spec.label, unpacker=unpacker)
        else:
            continue
        trace = _load_trace(spec.trace_path, label=spec.label)
        observations.append(
            {
                "label": spec.label,
                "role": spec.role,
                "confidence": spec.confidence,
                "trace": trace,
                "records": records,
                "deltas": _pair_deltas(anchor, trace),
                "manifest_path": spec.manifest_path,
                "archive_path": spec.archive_path,
            }
        )
    return observations


def _pair_deltas(anchor: Trace, candidate: Trace) -> dict[int, dict[str, float]]:
    missing = sorted(set(anchor.samples_by_pair) - set(candidate.samples_by_pair))
    if missing:
        raise CandidateBuildError(
            f"{candidate.path} is missing {len(missing)} anchor pairs; first={missing[:5]}"
        )
    out: dict[int, dict[str, float]] = {}
    for pair, base in anchor.samples_by_pair.items():
        cand = candidate.samples_by_pair[pair]
        seg_delta = base["seg"] - cand["seg"]
        pose_delta = base["pose"] - cand["pose"]
        out[pair] = {
            "seg_delta": seg_delta,
            "pose_delta": pose_delta,
            "combined_delta": seg_delta + pose_delta,
        }
    return out


def _record_key(record: ActionRecord) -> tuple[int, int, int]:
    return (
        int(record.pair_index),
        int(record.tile_id),
        int(record.source_action_id if record.source_action_id is not None else record.action_id),
    )


def _rank_source_records(
    source_records: Sequence[ActionRecord],
    observations: Sequence[dict[str, Any]],
) -> list[ActionRecord]:
    exact_support: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    pair_support: dict[int, list[dict[str, Any]]] = {}
    for obs in observations:
        records = obs["records"]
        by_pair_count: dict[int, int] = {}
        for rec in records:
            by_pair_count[rec.pair_index] = by_pair_count.get(rec.pair_index, 0) + 1
        for rec in records:
            delta = obs["deltas"].get(rec.pair_index)
            if delta is None:
                continue
            share = max(1, by_pair_count.get(rec.pair_index, 1))
            support = {
                "label": obs["label"],
                "role": obs["role"],
                "confidence": obs["confidence"],
                "share_count": share,
                "seg_delta": delta["seg_delta"] / share,
                "pose_delta": delta["pose_delta"] / share,
                "combined_delta": delta["combined_delta"] / share,
            }
            exact_support.setdefault(_record_key(rec), []).append(support)
            pair_support.setdefault(rec.pair_index, []).append(support)

    ranked: list[ActionRecord] = []
    for rec in source_records:
        supports = list(exact_support.get(_record_key(rec), ()))
        pair_only = list(pair_support.get(rec.pair_index, ()))
        if not supports:
            supports = [
                {**item, "confidence": float(item["confidence"]) * 0.25, "pair_only": True}
                for item in pair_only
            ]
        total_weight = sum(float(item["confidence"]) for item in supports)
        if total_weight <= 0:
            summary = {
                "support_count": 0,
                "mean_combined_delta": 0.0,
                "mean_seg_delta": 0.0,
                "mean_pose_delta": 0.0,
                "pose_toxic_votes": 0,
                "positive_votes": 0,
                "support_labels": [],
            }
        else:
            mean_seg = sum(float(item["confidence"]) * float(item["seg_delta"]) for item in supports) / total_weight
            mean_pose = sum(float(item["confidence"]) * float(item["pose_delta"]) for item in supports) / total_weight
            mean_combined = mean_seg + mean_pose
            summary = {
                "support_count": len(supports),
                "mean_combined_delta": mean_combined,
                "mean_seg_delta": mean_seg,
                "mean_pose_delta": mean_pose,
                "pose_toxic_votes": sum(1 for item in supports if float(item["pose_delta"]) < 0.0),
                "positive_votes": sum(1 for item in supports if float(item["combined_delta"]) > 0.0),
                "support_labels": sorted({str(item["label"]) for item in supports}),
            }
        ranked.append(replace(rec, support=summary))
    return sorted(
        ranked,
        key=lambda item: (
            float(item.support["mean_combined_delta"] if item.support else 0.0),
            float(item.support["mean_pose_delta"] if item.support else 0.0),
            -item.index,
        ),
        reverse=True,
    )


def _action_amp_shift(record: ActionRecord, shift: int, *, transform: str) -> ActionRecord:
    direction = int(record.action_id) // 12
    within = int(record.action_id) % 12
    amp_index = within // 2
    sign_index = within % 2
    shifted_amp = min(5, max(0, amp_index + shift))
    return replace(
        record,
        action_id=direction * 12 + shifted_amp * 2 + sign_index,
        source_action_id=(
            int(record.source_action_id)
            if record.source_action_id is not None
            else int(record.action_id)
        ),
        transform=transform,
    )


def _synthesize_policy(ranked: Sequence[ActionRecord], policy: str) -> list[ActionRecord]:
    source_order = sorted(ranked, key=lambda item: item.index)
    if policy == "all_ampminus1":
        return [_action_amp_shift(rec, -1, transform="all_ampminus1") for rec in source_order]
    if policy == "native_pose_guard_ampfit":
        selected: list[ActionRecord] = []
        for rec in source_order:
            support = rec.support or {}
            pose_delta = float(support.get("mean_pose_delta", 0.0))
            seg_delta = float(support.get("mean_seg_delta", 0.0))
            toxic_votes = int(support.get("pose_toxic_votes", 0))
            if toxic_votes > 0 or pose_delta < 0.0:
                selected.append(_action_amp_shift(rec, -1, transform="native_pose_guard_ampminus1"))
            elif pose_delta > 0.0 and seg_delta >= 0.0:
                selected.append(_action_amp_shift(rec, 1, transform="native_pose_guard_ampplus1"))
            else:
                selected.append(rec)
        return selected
    if policy == "consensus_positive_top64_ampfit":
        pool = [
            rec
            for rec in ranked
            if rec.support
            and int(rec.support.get("positive_votes", 0)) > 0
            and int(rec.support.get("pose_toxic_votes", 0)) == 0
        ][:64]
        if not pool:
            pool = list(ranked[:64])
        selected = []
        for rec in sorted(pool, key=lambda item: item.index):
            support = rec.support or {}
            pose_delta = float(support.get("mean_pose_delta", 0.0))
            seg_delta = float(support.get("mean_seg_delta", 0.0))
            if pose_delta < 0.0:
                selected.append(_action_amp_shift(rec, -1, transform="consensus_ampfit_minus"))
            elif pose_delta > 0.0 and seg_delta >= 0.0:
                selected.append(_action_amp_shift(rec, 1, transform="consensus_ampfit_plus"))
            else:
                selected.append(rec)
        return selected
    raise CandidateBuildError(f"unsupported policy: {policy}")


def _uleb128(value: int) -> bytes:
    if value < 0:
        raise CandidateBuildError(f"cannot varint-encode negative value {value}")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _encode_p6_actions(records: Sequence[ActionRecord]) -> tuple[bytes, bytes]:
    ordered = sorted(records, key=lambda item: (item.pair_index, item.index, item.tile_id))
    runtime_raw = b"".join(rec.encode4() for rec in ordered)
    packed = bytearray()
    previous_pair = 0
    for index, rec in enumerate(ordered):
        pair_index = int(rec.pair_index)
        tile_id = int(rec.tile_id)
        action_id = int(rec.action_id)
        if pair_index < 0 or pair_index >= 10_000:
            raise CandidateBuildError(f"P6 pair index out of range: {pair_index}")
        if tile_id < 0 or tile_id >= 256:
            raise CandidateBuildError(f"P6 tile id out of range: {tile_id}")
        if action_id < 0 or action_id >= 108:
            raise CandidateBuildError(f"P6 fixed action id out of range: {action_id}")
        delta = pair_index if index == 0 else pair_index - previous_pair
        if delta < 0:
            raise CandidateBuildError("P6 delta-varint action encoding requires nondecreasing pairs")
        packed.extend(_uleb128(delta))
        packed.append(tile_id)
        packed.append(action_id)
        previous_pair = pair_index
    return runtime_raw, bytes(packed)


def _build_p6_payload(source: SourceArchive, records: Sequence[ActionRecord]) -> tuple[bytes, bytes, bytes]:
    runtime_raw, packed = _encode_p6_actions(records)
    actions_br = brotli.compress(packed, quality=11)
    payload = (
        b"P6"
        + struct.pack(
            "<IHHH",
            len(source.mask_br),
            len(source.renderer_br),
            len(actions_br),
            len(records),
        )
        + source.mask_br
        + source.renderer_br
        + actions_br
        + source.pose_br
    )
    return payload, runtime_raw, actions_br


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(_safe_zip_member_name(name), FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _write_archive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(_zip_info(PAYLOAD_MEMBER), payload)


def _action_guard(records: Sequence[ActionRecord], *, base_raw: bytes, decoded_raw: bytes) -> dict[str, Any]:
    exact_seen: dict[tuple[int, int, int], int] = {}
    for rec in records:
        key = (int(rec.pair_index), int(rec.tile_id), int(rec.action_id))
        exact_seen[key] = exact_seen.get(key, 0) + 1
    duplicate_exact = sum(count - 1 for count in exact_seen.values() if count > 1)
    changed_records = [
        rec
        for rec in records
        if int(rec.action_id)
        != int(rec.source_action_id if rec.source_action_id is not None else rec.action_id)
    ]
    if duplicate_exact:
        raise CandidateBuildError(f"candidate has {duplicate_exact} duplicate exact action records")
    if decoded_raw == base_raw:
        raise CandidateBuildError("unchanged decoded action semantics")
    if not changed_records and len(decoded_raw) == len(base_raw):
        raise CandidateBuildError("no-op action stream: no action ids changed")
    return {
        "record_count": len(records),
        "changed_action_id_record_count": len(changed_records),
        "duplicate_exact_action_record_count": duplicate_exact,
        "decoded_action_sha256": _sha256_bytes(decoded_raw),
        "source_decoded_action_sha256": _sha256_bytes(base_raw),
    }


def _validate_candidate_payload(
    *,
    source: SourceArchive,
    payload: bytes,
    records: Sequence[ActionRecord],
    unpacker: Any,
) -> dict[str, Any]:
    header, decoded_raw = unpacker._parse_payload(payload)  # noqa: SLF001
    payload_format = str(header.get("payload_format"))
    if payload_format != "public_pr75_qzs3_qp1_segactions_p6_delta_varint":
        raise CandidateBuildError(f"runtime parser returned invalid P6 payload_format={payload_format!r}")
    decoded = {str(name): bytes(data) for name, data in decoded_raw.items()}
    for name in ("masks.mkv", "renderer.bin", "optimized_poses.qp1"):
        if decoded.get(name) != source.decoded[name]:
            raise CandidateBuildError(f"non-action stream changed unexpectedly: {name}")
    actions = decoded.get("seg_tile_actions.bin")
    if actions is None:
        raise CandidateBuildError("runtime parser did not decode seg_tile_actions.bin")
    action_validation = validate_seg_tile_actions_payload(
        actions,
        source_name="candidate seg_tile_actions.bin",
    )
    guard = _action_guard(
        records,
        base_raw=source.decoded["seg_tile_actions.bin"],
        decoded_raw=actions,
    )
    runtime_members = _member_summary(header)
    return {
        "payload_format": payload_format,
        "runtime_parser": _repo_rel(UNPACKER_PATH),
        "runtime_members": runtime_members,
        "seg_tile_actions_validation": action_validation,
        "action_semantic_guard": guard,
        "non_action_streams_preserved": True,
    }


def _record_manifest(record: ActionRecord) -> dict[str, Any]:
    return {
        "index": int(record.index),
        "pair_index": int(record.pair_index),
        "tile_id": int(record.tile_id),
        "action_id": int(record.action_id),
        "source_action_id": (
            int(record.source_action_id)
            if record.source_action_id is not None
            else int(record.action_id)
        ),
        "transform": record.transform,
        "support": record.support,
    }


def build_one_candidate(
    *,
    source: SourceArchive,
    anchor_trace: Trace,
    ranked_records: Sequence[ActionRecord],
    observations: Sequence[dict[str, Any]],
    output_dir: Path,
    policy: str,
    unpacker: Any,
    forbidden_archive_shas: set[str] | None = None,
) -> dict[str, Any]:
    if forbidden_archive_shas is None:
        forbidden_archive_shas = set(DEFAULT_FORBIDDEN_ARCHIVE_SHAS)
    selected = _synthesize_policy(ranked_records, policy)
    if not selected:
        raise CandidateBuildError(f"policy {policy} selected no actions")
    payload, runtime_raw, actions_br = _build_p6_payload(source, selected)
    validation = _validate_candidate_payload(
        source=source,
        payload=payload,
        records=selected,
        unpacker=unpacker,
    )
    candidate_id = f"c101_top192_actions_{policy}_p6"
    candidate_dir = output_dir / candidate_id
    archive_path = candidate_dir / "archive.zip"
    _write_archive(archive_path, payload)
    archive_sha = _sha256_file(archive_path)
    if archive_sha in forbidden_archive_shas:
        raise CandidateBuildError(f"{candidate_id}: duplicate exact SHA {archive_sha}")
    if _read_single_payload(archive_path) != payload:
        raise CandidateBuildError(f"{candidate_id}: archive readback payload mismatch")
    archive_bytes = archive_path.stat().st_size
    rate_delta = 25.0 * (archive_bytes - source.archive_bytes) / RATE_DENOM
    base_score = anchor_trace.score
    unchanged_component_score = None if base_score is None else base_score + rate_delta
    required_component_improvement = (
        None
        if unchanged_component_score is None
        else max(0.0, unchanged_component_score - TARGET_SCORE)
    )
    support_delta_upper_bound = sum(
        max(0.0, float((record.support or {}).get("mean_combined_delta", 0.0)))
        for record in selected
    )
    manifest = {
        "schema": SCHEMA,
        "tool": TOOL,
        "candidate_id": candidate_id,
        "score_claim": False,
        "remote_dispatch_performed": False,
        "cuda_auth_eval_required": CUDA_AUTH_EVAL_REQUIRED,
        "policy": policy,
        "wire_format": "p6",
        "source": {
            "archive_path": str(source.path),
            "archive_repo_relative_path": _repo_rel(source.path),
            "archive_bytes": source.archive_bytes,
            "archive_sha256": source.archive_sha256,
            "payload_sha256": source.payload_sha256,
            "decoded_action_sha256": _sha256_bytes(source.decoded["seg_tile_actions.bin"]),
        },
        "observations_used": [
            {
                "label": obs["label"],
                "role": obs["role"],
                "confidence": obs["confidence"],
                "trace_path": _repo_rel(obs["trace"].path),
                "manifest_path": _repo_rel(obs["manifest_path"]) if obs["manifest_path"] else None,
                "archive_path": _repo_rel(obs["archive_path"]) if obs["archive_path"] else None,
            }
            for obs in observations
        ],
        "output_archive": {
            "path": str(archive_path),
            "repo_relative_path": _repo_rel(archive_path),
            "bytes": archive_bytes,
            "sha256": archive_sha,
            "member": PAYLOAD_MEMBER,
            "payload_bytes": len(payload),
            "payload_sha256": _sha256_bytes(payload),
            "deterministic_zip_timestamp": FIXED_ZIP_TIMESTAMP,
        },
        "action_stream": {
            "runtime_action_raw_bytes": len(runtime_raw),
            "runtime_action_raw_sha256": _sha256_bytes(runtime_raw),
            "encoded_delta_varint_bytes": len(runtime_raw) // 4 * 3,
            "encoded_action_brotli_bytes": len(actions_br),
            "encoded_action_brotli_sha256": _sha256_bytes(actions_br),
        },
        "validation": validation,
        "rate_screen": {
            "base_score_from_component_trace": base_score,
            "target_score": TARGET_SCORE,
            "delta_bytes_vs_top192": archive_bytes - source.archive_bytes,
            "formula_rate_delta_vs_top192": rate_delta,
            "unchanged_component_score_after_rate": unchanged_component_score,
            "required_component_improvement_to_reach_target": required_component_improvement,
            "positive_support_delta_upper_bound": support_delta_upper_bound,
            "evidence_grade": "empirical_byte_screen_only_until_exact_cuda",
        },
        "rationale": _policy_rationale(policy),
        "selected_records": [_record_manifest(record) for record in selected],
    }
    (candidate_dir / "manifest.json").write_bytes(_json_bytes(manifest))
    return manifest


def _policy_rationale(policy: str) -> str:
    if policy == "all_ampminus1":
        return (
            "Uniform fixed-dictionary amplitude shrink over the native top192 action body; "
            "tests whether the top25 ampminus1 exact signal generalizes without changing "
            "mask, renderer, or pose bytes."
        )
    if policy == "native_pose_guard_ampfit":
        return (
            "Per-record amplitude fit from mined pair feedback: shrink pose-toxic support, "
            "boost only pose-positive/seg-nonnegative support, preserve the rest."
        )
    if policy == "consensus_positive_top64_ampfit":
        return (
            "Low-rate consensus subset from positive, non-pose-toxic support with the same "
            "pose-aware amplitude fit; intended as an atom-level escape, not a PR77 or "
            "PR75 pose-safe replay duplicate."
        )
    return "No rationale registered."


def build_candidates(
    *,
    base_archive: Path,
    base_trace: Path,
    output_dir: Path,
    policies: Sequence[str],
    forbidden_archive_shas: set[str] | None = None,
) -> dict[str, Any]:
    unpacker = _load_unpacker()
    source = load_source_archive(base_archive, unpacker=unpacker)
    if source.archive_sha256 != TOP192_SHA256 and base_archive == DEFAULT_BASE_ARCHIVE:
        raise CandidateBuildError(
            f"default top192 archive SHA mismatch: expected {TOP192_SHA256}, got {source.archive_sha256}"
        )
    anchor = _load_trace(base_trace, label="c101_top192_anchor")
    source_records = _parse_action_records(
        source.decoded["seg_tile_actions.bin"],
        support_label="c101_top192_source",
    )
    observations = _load_observations(
        _default_observation_specs(),
        anchor=anchor,
        unpacker=unpacker,
    )
    if not observations:
        raise CandidateBuildError("no trace/action observations were loadable")
    ranked = _rank_source_records(source_records, observations)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    for policy in policies:
        manifests.append(
            build_one_candidate(
                source=source,
                anchor_trace=anchor,
                ranked_records=ranked,
                observations=observations,
                output_dir=output_dir,
                policy=policy,
                unpacker=unpacker,
                forbidden_archive_shas=forbidden_archive_shas,
            )
        )
    summary = {
        "schema": f"{SCHEMA}_summary",
        "tool": TOOL,
        "score_claim": False,
        "remote_dispatch_performed": False,
        "source_archive": {
            "path": str(source.path),
            "bytes": source.archive_bytes,
            "sha256": source.archive_sha256,
        },
        "candidate_count": len(manifests),
        "candidates": [
            {
                "candidate_id": manifest["candidate_id"],
                "archive": manifest["output_archive"],
                "rate_screen": manifest["rate_screen"],
                "changed_action_id_record_count": manifest["validation"][
                    "action_semantic_guard"
                ]["changed_action_id_record_count"],
                "rationale": manifest["rationale"],
            }
            for manifest in manifests
        ],
    }
    (output_dir / "candidate_matrix.json").write_bytes(_json_bytes(summary))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-archive", type=Path, default=DEFAULT_BASE_ARCHIVE)
    parser.add_argument("--base-component-trace", type=Path, default=DEFAULT_BASE_TRACE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--policy",
        action="append",
        default=[],
        choices=(
            "all_ampminus1",
            "native_pose_guard_ampfit",
            "consensus_positive_top64_ampfit",
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    policies = args.policy or [
        "native_pose_guard_ampfit",
        "all_ampminus1",
        "consensus_positive_top64_ampfit",
    ]
    summary = build_candidates(
        base_archive=args.base_archive,
        base_trace=args.base_component_trace,
        output_dir=args.output_dir,
        policies=policies,
    )
    for candidate in summary["candidates"]:
        archive = candidate["archive"]
        print(candidate["candidate_id"], archive["bytes"], archive["sha256"], archive["path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
