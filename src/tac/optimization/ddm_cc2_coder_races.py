# SPDX-License-Identifier: MIT
"""Typed, custody-bound helpers for the DDM CC2 quantizer and coder races.

Race 2 applies three terminal quantization operators to the same landed J8F
theta and compiles each result through the real W_joint receiver compiler.
Race 3 recursively inventories the exact counted composition archive and runs
the established MS7 same-object discipline on every physical leaf stream.

No result in this module is a contest score.  Frozen-scorer evaluation belongs
to the resumable runner, which labels every row ``[macOS-CPU advisory]``.
"""

from __future__ import annotations

import hashlib
import io
import math
import stat
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Final

import numpy as np

from tac.canonical_equations.ddm_ms2r_tolerance_capped_solve_20260724 import (
    RATE_DENOMINATOR_BYTES,
)
from tac.optimization.ddm_ms7_receiver_edges import race_counted_stream_contexts
from tac.optimization.ddm_pc1_pose_stream import (
    PC1PosePacketV1,
    build_counted_composition_archive,
    parse_counted_composition_archive,
    serialize_pc1_packet,
)
from tac.optimization.direct_description_joint_descent import (
    compile_parameterized_archive,
    lift_v15_archive,
)

RACE2_SCHEMA: Final = "ddm_cc2_quantization_race.v1"
RACE3_SCHEMA: Final = "ddm_cc2_per_counted_stream_coder_race.v1"
PRICE_TABLE_SCHEMA: Final = "ddm_cc2_c1_costate_stream_price_table.v1"
POINTER: Final = "0.1910828242 [contest-CPU]"
EVIDENCE_AXIS: Final = "[macOS-CPU advisory]"
SEED: Final = 0


class CC2CoderRacesError(ValueError):
    """Raised when exact payload custody, parse-back, or race identity differs."""


@dataclass(frozen=True, slots=True)
class QuantizationArm:
    arm_id: str
    realized_theta: np.ndarray
    parent_archive: bytes
    composition_archive: bytes
    schedule: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CountedLeaf:
    stream_id: str
    payload: bytes
    category: str


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _soft_round(value: np.ndarray, temperature: float) -> np.ndarray:
    source = np.asarray(value, dtype=np.float64)
    midpoint = np.floor(source) + 0.5
    return midpoint + np.tanh((source - midpoint) / temperature) / (2.0 * math.tanh(0.5 / temperature))


def _soft_round_inverse(value: np.ndarray, temperature: float) -> np.ndarray:
    source = np.asarray(value, dtype=np.float64)
    midpoint = np.floor(source) + 0.5
    argument = (source - midpoint) * 2.0 * math.tanh(0.5 / temperature)
    argument = np.clip(argument, np.nextafter(-1.0, 0.0), np.nextafter(1.0, 0.0))
    return midpoint + np.arctanh(argument) * temperature


def _c3_conditional_mean(value: np.ndarray, temperature: float) -> np.ndarray:
    return _soft_round_inverse(np.asarray(value, dtype=np.float64) - 0.5, temperature) + 0.5


def terminal_quantization_thetas(
    theta: np.ndarray,
    *,
    seed: int = SEED,
) -> dict[str, tuple[np.ndarray, dict[str, Any]]]:
    """Return the three pre-registered terminal schedule operators.

    C3 and Cool-Chic-v5 are explicitly bounded terminal single-pass proxies,
    not retraining claims.  Their constants and equations are transcribed from
    the SHA-bound upstream harvest named in each schedule record.
    """

    source = np.asarray(theta, dtype=np.float64)
    if source.ndim != 1 or not np.all(np.isfinite(source)):
        raise CC2CoderRacesError("J8F theta must be a finite one-dimensional array")

    c3_rng = np.random.default_rng(seed)
    c3_a = 1.0
    c3_b = (2.0**c3_a * (c3_a - 1.0) + 1.0) / c3_a
    uniform = c3_rng.random(source.shape)
    c3_noise = np.power(1.0 - np.power(1.0 - uniform, 1.0 / c3_b), 1.0 / c3_a) - 0.5
    c3_temperature = 0.1
    c3 = _c3_conditional_mean(
        _soft_round(source, c3_temperature) + c3_noise,
        c3_temperature,
    )

    v5_rng = np.random.default_rng(seed)
    v5_temperature = 0.08
    v5_sigma = 0.15
    v5 = _soft_round(
        _soft_round(source, v5_temperature) + v5_rng.normal(0.0, v5_sigma, size=source.shape),
        v5_temperature,
    )

    return {
        "CAMERA_Q8_EXACT": (
            source.copy(),
            {
                "operator_scope": "EXACT_LANDED_J8F_CAMERA_Q8_COMPILE",
                "quantization": "np.rint in direct_description_joint_descent.realize_parameter_theta",
                "seed": None,
            },
        ),
        "C3_ORIGINAL_TERMINAL_PROXY": (
            c3,
            {
                "operator_scope": "TERMINAL_SINGLE_PASS_SOURCE_SCHEDULE_PROXY_NOT_RETRAINING",
                "temperature": c3_temperature,
                "kumaraswamy_a": c3_a,
                "kumaraswamy_b": c3_b,
                "seed": seed,
                "source_commit": "e63e7519641db3d431cff623127fb2dcb825069f",
                "source_files": {
                    "configs/uvg.py": "76ec0aee20bd25416cb4e3b7e19ee06413715ece93b4b29fd924bbb1f1075cd8",
                    "model/latents.py": "642430488a75cfc514fafedff8ef1cdb0929effdf6a387eb0a14e5c27dabda3e",
                },
            },
        ),
        "COOL_CHIC_V5_TERMINAL_PROXY": (
            v5,
            {
                "operator_scope": "TERMINAL_SINGLE_PASS_SOURCE_SCHEDULE_PROXY_NOT_RETRAINING",
                "temperature": v5_temperature,
                "gaussian_sigma": v5_sigma,
                "seed": seed,
                "source_commit": "a6fe38a414dd098b39c41636bd6e423626402f7e",
                "source_files": {
                    "coolchic/training/presets.py": (
                        "bd8eba5c2ade1194c4a74a4717574ece63c7c914335f4fa41f7b15e3361dec34"
                    ),
                    "coolchic/component/core/quantizer.py": (
                        "0ae82568ab1eea853f754e6e6cad4b889ddcadeb562acffd5299c0987a4726c8"
                    ),
                },
            },
        ),
    }


def build_quantization_arms(
    *,
    source_archive: bytes,
    theta: np.ndarray,
    pose_packet: PC1PosePacketV1,
    seed: int = SEED,
) -> dict[str, QuantizationArm]:
    """Compile each quantizer through W_joint and the counted PC1 wrapper."""

    packet_bytes = serialize_pc1_packet(pose_packet)
    if len(packet_bytes) != 40:
        raise CC2CoderRacesError("PC1 pose home must remain the exact 40-byte packet")
    if np.any(pose_packet.q_xi) or np.any(pose_packet.q_luma_phase) or not pose_packet.active:
        raise CC2CoderRacesError("Race 2 requires the active zero-effect PC1 packet")
    lift = lift_v15_archive(source_archive)
    schedules = terminal_quantization_thetas(theta, seed=seed)
    schedules_repeat = terminal_quantization_thetas(theta, seed=seed)
    for arm_id in schedules:
        if not np.array_equal(schedules[arm_id][0], schedules_repeat[arm_id][0]):
            raise CC2CoderRacesError(f"{arm_id} terminal schedule is nondeterministic")

    arms: dict[str, QuantizationArm] = {}
    for arm_id, (candidate_theta, schedule) in schedules.items():
        parent, realized = compile_parameterized_archive(
            lift,
            candidate_theta.astype(np.float32),
            include_lane_programs=False,
        )
        composition = build_counted_composition_archive(
            parent_archive=parent,
            parent_sha256=sha256_bytes(parent),
            packet=pose_packet,
        )
        parsed_parent, parsed_packet, _ = parse_counted_composition_archive(composition)
        if parsed_parent != parent or serialize_pc1_packet(parsed_packet) != packet_bytes:
            raise CC2CoderRacesError(f"{arm_id} counted composition parse-back differs")
        arms[arm_id] = QuantizationArm(
            arm_id=arm_id,
            realized_theta=realized,
            parent_archive=parent,
            composition_archive=composition,
            schedule=schedule,
        )
    return arms


def _safe_member_name(name: str) -> str:
    path = PurePosixPath(name)
    if not name or name.startswith("/") or "\\" in name or any(part in ("", ".", "..") for part in path.parts):
        raise CC2CoderRacesError(f"recursive ZIP member name is unsafe: {name!r}")
    return path.as_posix()


def _stream_category(stream_id: str) -> str:
    lowered = stream_id.lower()
    if lowered.endswith("pc1.ddp"):
        return "POSE_40B_HOME"
    if lowered.endswith(".g1s"):
        return "V15_G1_PAYLOAD"
    if "/manifest" in lowered or lowered.endswith(".json"):
        return "COUNTED_MANIFEST_OR_CUSTODY"
    if lowered.endswith((".lz", ".br", ".ddrp", ".ddst", ".ddcm", ".ddq8", ".ddlp", ".ddsr")):
        return "W_JOINT_STATE_STREAM"
    return "OTHER_COUNTED_STREAM"


def extract_recursive_zip_leaves(payload: bytes) -> tuple[list[CountedLeaf], int]:
    """Expand stored nested ZIPs into physical leaves and exact fixed overhead."""

    leaves: list[CountedLeaf] = []

    def visit(blob: bytes, owner: str) -> None:
        stream = io.BytesIO(blob)
        if not zipfile.is_zipfile(stream):
            leaves.append(
                CountedLeaf(
                    stream_id=owner,
                    payload=blob,
                    category=_stream_category(owner),
                )
            )
            return
        stream.seek(0)
        try:
            with zipfile.ZipFile(stream, "r") as archive:
                infos = archive.infolist()
                names = [_safe_member_name(info.filename) for info in infos]
                if len(names) != len(set(names)):
                    raise CC2CoderRacesError(f"duplicate recursive ZIP member under {owner}")
                for info, name in zip(infos, names, strict=True):
                    if info.is_dir():
                        continue
                    mode = info.external_attr >> 16
                    if stat.S_ISLNK(mode):
                        raise CC2CoderRacesError(f"symlink ZIP member under {owner}: {name}")
                    if info.compress_type != zipfile.ZIP_STORED:
                        raise CC2CoderRacesError(
                            f"Race 3 requires current physical stored streams; compressed member: {owner}!/{name}"
                        )
                    visit(archive.read(info), f"{owner}!/{name}")
        except (zipfile.BadZipFile, RuntimeError) as exc:
            raise CC2CoderRacesError(f"invalid recursive ZIP under {owner}") from exc

    visit(bytes(payload), "composition.zip")
    leaf_bytes = sum(len(row.payload) for row in leaves)
    fixed_overhead = len(payload) - leaf_bytes
    if fixed_overhead < 0 or fixed_overhead + leaf_bytes != len(payload):
        raise CC2CoderRacesError("recursive ZIP byte partition is inconsistent")
    return leaves, fixed_overhead


def build_per_stream_price_table(composition_archive: bytes) -> dict[str, Any]:
    """Race all five coders per counted leaf and emit c1/costate SENSE prices."""

    leaves, fixed_overhead = extract_recursive_zip_leaves(composition_archive)
    rows: list[dict[str, Any]] = []
    current_leaf_bytes = 0
    selected_leaf_bytes = 0
    for leaf in leaves:
        race, _ = race_counted_stream_contexts(leaf.payload)
        winner = race["winner"]
        current_bytes = len(leaf.payload)
        winner_bytes = int(winner["framed_bytes"])
        delta_bytes = winner_bytes - current_bytes
        delta_rate = 25.0 * delta_bytes / RATE_DENOMINATOR_BYTES
        current_leaf_bytes += current_bytes
        selected_leaf_bytes += winner_bytes
        rows.append(
            {
                "stream_id": leaf.stream_id,
                "category": leaf.category,
                "current_bytes": current_bytes,
                "current_sha256": sha256_bytes(leaf.payload),
                "arms": race["rows"],
                "selected_codec": winner["codec"],
                "selected_framed_bytes": winner_bytes,
                "delta_bytes": delta_bytes,
                "delta_dseg": 0.0,
                "delta_dpose": 0.0,
                "delta_rate_score": delta_rate,
                "delta_advisory_action": delta_rate,
                "costate_sense": {
                    "d_distortion_d_byte": 0.0,
                    "d_action_d_byte": 25.0 / RATE_DENOMINATOR_BYTES,
                    "admission": (
                        "ADMIT_LOSSLESS_RECODING_TO_RECEIVER_INTEGRATION_QUEUE"
                        if delta_bytes < 0
                        else "KEEP_CURRENT_STREAM"
                    ),
                },
                "parseback_exact_all_arms": all(arm["available"] and arm["parseback_exact"] for arm in race["rows"]),
                "negative_verdict_scope": race["negative_verdict_scope"],
            }
        )
    if fixed_overhead + current_leaf_bytes != len(composition_archive):
        raise CC2CoderRacesError("Race 3 current-byte reconciliation differs")
    selected_total = fixed_overhead + selected_leaf_bytes
    return {
        "schema": RACE3_SCHEMA,
        "price_table_schema": PRICE_TABLE_SCHEMA,
        "composition_archive_bytes": len(composition_archive),
        "composition_archive_sha256": sha256_bytes(composition_archive),
        "recursive_fixed_zip_overhead_bytes": fixed_overhead,
        "counted_leaf_stream_count": len(rows),
        "current_leaf_bytes": current_leaf_bytes,
        "selected_leaf_bytes": selected_leaf_bytes,
        "selected_total_archive_estimate_bytes": selected_total,
        "selected_total_delta_bytes": selected_total - len(composition_archive),
        "selected_total_delta_dseg": 0.0,
        "selected_total_delta_dpose": 0.0,
        "selected_total_delta_advisory_action": (
            25.0 * (selected_total - len(composition_archive)) / RATE_DENOMINATOR_BYTES
        ),
        "rows": rows,
        "c1_waterfill_order": [
            row["stream_id"]
            for row in sorted(
                rows,
                key=lambda row: (
                    float(row["delta_advisory_action"]),
                    str(row["stream_id"]),
                ),
            )
            if int(row["delta_bytes"]) < 0
        ],
        "receiver_status": (
            "PRICE_TABLE_ONLY_NEW_CONTEXT_FRAMES_PARSE_BACK_EXACT_BUT_COMPOSITION_INFLATE_INTERPRETER_NOT_LANDED"
        ),
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
        "pointer": POINTER,
        "pointer_moved": False,
        "research_only": True,
        "main_review_required": True,
    }


def reprice_zero_pose_composition(
    parent_verdict: dict[str, Any],
    *,
    parent_archive: bytes,
    composition_archive: bytes,
) -> dict[str, Any]:
    """Attach the exact counted PC1 wrapper while preserving zero-effect score."""

    if int(parent_verdict.get("archive_bytes", -1)) != len(parent_archive) or parent_verdict.get(
        "archive_sha256"
    ) != sha256_bytes(parent_archive):
        raise CC2CoderRacesError("parent verdict archive custody differs")
    parsed_parent, packet, _ = parse_counted_composition_archive(composition_archive)
    if parsed_parent != parent_archive or np.any(packet.q_xi) or np.any(packet.q_luma_phase) or not packet.active:
        raise CC2CoderRacesError("PC1 wrapper is not the active zero-effect composition")
    result = dict(parent_verdict)
    result["parent_archive_bytes"] = len(parent_archive)
    result["parent_archive_sha256"] = sha256_bytes(parent_archive)
    result["archive_bytes"] = len(composition_archive)
    result["archive_sha256"] = sha256_bytes(composition_archive)
    result["pc1_packet_bytes"] = len(serialize_pc1_packet(packet))
    result["pc1_output_effect"] = "IDENTITY_ACTIVE_ALL_QUANTIZED_COORDINATES_ZERO"
    result["advisory_action"] = (
        100.0 * float(result["d_seg"])
        + math.sqrt(10.0 * float(result["d_pose"]))
        + 25.0 * len(composition_archive) / RATE_DENOMINATOR_BYTES
    )
    result["evidence_axis"] = EVIDENCE_AXIS
    result["score_claim"] = False
    result["promotion_eligible"] = False
    result["pointer"] = POINTER
    result["pointer_moved"] = False
    return result


def race2_delta(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float | int]:
    """Return exact component and total deltas relative to camera-Q8."""

    delta_bytes = int(candidate["archive_bytes"]) - int(reference["archive_bytes"])
    delta_dseg = float(candidate["d_seg"]) - float(reference["d_seg"])
    delta_dpose = float(candidate["d_pose"]) - float(reference["d_pose"])
    delta_seg_term = 100.0 * delta_dseg
    delta_pose_term = math.sqrt(10.0 * float(candidate["d_pose"])) - math.sqrt(10.0 * float(reference["d_pose"]))
    delta_rate_term = 25.0 * delta_bytes / RATE_DENOMINATOR_BYTES
    return {
        "delta_bytes": delta_bytes,
        "delta_dseg": delta_dseg,
        "delta_dpose": delta_dpose,
        "delta_seg_term": delta_seg_term,
        "delta_pose_term": delta_pose_term,
        "delta_rate_term": delta_rate_term,
        "delta_advisory_action": delta_seg_term + delta_pose_term + delta_rate_term,
    }


__all__ = [
    "EVIDENCE_AXIS",
    "POINTER",
    "PRICE_TABLE_SCHEMA",
    "RACE2_SCHEMA",
    "RACE3_SCHEMA",
    "CC2CoderRacesError",
    "CountedLeaf",
    "QuantizationArm",
    "build_per_stream_price_table",
    "build_quantization_arms",
    "extract_recursive_zip_leaves",
    "race2_delta",
    "reprice_zero_pose_composition",
    "sha256_bytes",
    "terminal_quantization_thetas",
]
