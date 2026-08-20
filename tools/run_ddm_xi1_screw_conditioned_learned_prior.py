#!/usr/bin/env python3
"""DDM XI1: learned screw-conditioned lossless entropy priors.

Leg A jointly trains PR130's attested HPAC at two matched rate lambdas on a
seeded, stratified-random n120 sample.  Its only treatment is the previous-plane
input: an all-zero spatial control versus a partition transported by the carried
pose through ``tac.lie``.  The exact selected tokens are Range-coded and decoded.

Leg B losslessly races the current CP135 carrier as direct CPR1, its incumbent
CAP1 AR(1)+bias representation, and a counted fixed-point linear model whose
features include a ``tac.lie`` relative screw.  Because CP135 does not carry a
geometric screw, the learned packet includes and charges the decoded 600x6 fp16
pose plane.  Every candidate decodes to the same canonical CPR1 bytes, so the
CP135 realized d_pose is unchanged by construction.

This is a scope-reduced screen, not an exact contest score.  Every materialized
payload is retained under ``/Volumes/APDataStore/pact/ddm_xi1_20260812``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import io
import json
import lzma
import math
import os
import platform
import random
import shutil
import struct
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any

import constriction
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.admission_guard import assert_governed_admission  # noqa: E402
from tac.lie import _se3_numpy as se3  # noqa: E402
from tac.pr130_lift.train_semantic_quantized_resumable import resolve_ema_policy  # noqa: E402
from tac.training import EMA  # noqa: E402

OUTPUT = Path("/Volumes/APDataStore/pact/ddm_xi1_20260812")
FIX_OUTPUT = OUTPUT / "fix"
STATE = FIX_OUTPUT / "state.json"
LEG_A_RETAINED = FIX_OUTPUT / "retained/leg_a"
LEG_A_RESULT = FIX_OUTPUT / "LEG_A_RESULT.json"
LEG_A_PREPARATION = FIX_OUTPUT / "LEG_A_PREPARATION.json"
LEG_A_FIRE_ORDER = FIX_OUTPUT / "queue/leg_a_mps.json"
FINAL_RESULT = FIX_OUTPUT / "FINAL_RESULT.json"
HP3_PATH = ROOT / "experiments/ddm_hp3_hpac_section_and_zip_frame.py"
POSE_WARP_PATH = ROOT / "tools/measure_pose_warp_dseg.py"
INTAKE = Path("/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo")
INTAKE_CODE = INTAKE / "code"
HPAC_CACHE = Path("/Volumes/VertigoDataTier/pact/ddm_op1r_20260809/authority_cache/gt_cache_600_official_ada.pt")  # GT_LINEAGE_OK: bytes are registry-classified DALI_NVDEC sha256 382d7dfe38b37c0c
HPAC_INIT = Path("/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/checkpoints/gt/hpac_p64_exact_from_archive.pt")
FX1_RUNTIME = ROOT / "src/tac/pr130_runtime/fx1_runtime_tree"
CP135_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_cp135_20260810/adapted_runtime")
CP135_ARCHIVE = CP135_ROOT / "archive.zip"
BOOK_SRC = Path("/Volumes/VertigoDataTier/pact/pr135_intake_20260810/experiment_book/src")
POSE_CONTAINER = Path("/Volumes/VertigoDataTier/pact/ddm_ep2_20260731/gr1_eval/smoke/arch/state/pose_warp.stp")
POSE_RAW = Path("/Volumes/VertigoDataTier/pact/ddm_tf1_20260812/final_v2/retained/inputs/pose_targets_n600_f16.bin")
CALIBRATION_RAW = Path("/Volumes/VertigoDataTier/pact/ddm_tf1_20260812/final_v2/retained/inputs/warp_calibration_f64.bin")

SCHEMA = "ddm_xi1_screw_conditioned_learned_prior.v1"
CHECKPOINT_SCHEMA = "ddm_xi1_hpac_checkpoint.v2"
LEGACY_CHECKPOINT_SCHEMA = "ddm_xi1_hpac_checkpoint.v1"
AXIS_A = "[macOS-MPS research-signal training; macOS-CPU advisory real Range bytes; scorer-free]"
AXIS_B = "[macOS-CPU advisory; n600 exact carrier decode; inherited contest-CUDA CP135 d_pose]"
SEED = 20260716
SAMPLE_SEED = 20260812
FRAME_COUNT = 600
STRATA = 10
SAMPLE_PER_STRATUM = 12
SAMPLE_COUNT = STRATA * SAMPLE_PER_STRATUM
H, W, CLASSES = 384, 512, 5
TOKENS_PER_FRAME = H * W
RATE_LAMBDAS = (1.0, 0.5)
CONTEXT_MODES = ("spatial", "xi")
DEFAULT_EPOCHS = 6
EXPECTED_BIT_DEPTH_NAMES = (
    "conv_a.bit_depth",
    "conv_b1.bit_depth",
    "conv_b2.bit_depth",
    "conv_past.bit_depth",
    "frame_scale.bit_depth",
    "frame_shift.bit_depth",
    "head.bit_depth",
    "spm_dw.bit_depth",
    "spm_pw.bit_depth",
)
POSE_D = 0.00000688
CP135_SCORE = 0.16195513827824176
CP135_BYTES = 186_252
EXPECTED = {
    HPAC_CACHE: "382d7dfe38b37c0cc5017e5645032faa045af6924db66e0b67549cc96c840195",
    HPAC_INIT: "0e6c30cef6b36c4e530779c92c56e9128c1d86c62e85e9fc5358a7e9f40ec985",
    CP135_ARCHIVE: "6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6",
    POSE_CONTAINER: "3121e6e5045d7e3167f385b0b1639327ff6dc52fc6015b2e64b7008dbb637af7",
    POSE_RAW: "4c14c3195f676888a8f9511e1ab8ac5a6d621d58c16791c3ae2e9648cfa5c29e",
    CALIBRATION_RAW: "3c6db7263465151cd744b4be40eb0a949059613b34881d7e7afeacfe32d92b42",
}


class XI1Error(RuntimeError):
    """Raised when an XI1 custody, parity, or decode invariant fails."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _atomic_bytes(path: Path, payload: bytes, *, replace: bool = False) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        if path.read_bytes() != payload:
            raise XI1Error(f"refusing to overwrite retained payload: {path}")
        return file_record(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return file_record(path)


def retain_payload(path: Path, payload: bytes) -> dict[str, Any]:
    """Canonical P0-named wrapper used by the static payload-retention gate."""

    return _atomic_bytes(path, payload)


def atomic_json(path: Path, value: Any, *, replace: bool = False) -> dict[str, Any]:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    return _atomic_bytes(path, payload, replace=replace)


def atomic_npy(path: Path, array: np.ndarray) -> dict[str, Any]:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return _atomic_bytes(path, buffer.getvalue())


def atomic_torch(path: Path, value: Any, *, replace: bool = False) -> dict[str, Any]:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return _atomic_bytes(path, buffer.getvalue(), replace=replace)


def import_path(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise XI1Error(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pin_inputs() -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for path, expected in EXPECTED.items():
        if not path.is_file():
            raise XI1Error(f"required input is absent: {path}")
        observed = file_record(path)
        if observed["sha256"] != expected:
            raise XI1Error(f"input hash changed for {path}: {observed['sha256']}")
        rows[path.name + ":" + sha256_bytes(str(path).encode())[:8]] = observed
    return rows


def storage_preflight(required_free_bytes: int) -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(OUTPUT)
    if usage.free < required_free_bytes:
        raise XI1Error(f"APData storage preflight requires {required_free_bytes}, found {usage.free}")
    return {
        "path": str(OUTPUT),
        "required_free_bytes": required_free_bytes,
        "observed_free_bytes": usage.free,
        "status": "PASS",
    }


def selected_frame_ids(seed: int = SAMPLE_SEED) -> np.ndarray:
    width = FRAME_COUNT // STRATA
    rng = np.random.default_rng(seed)
    values: list[int] = []
    for stratum in range(STRATA):
        draw = rng.choice(np.arange(stratum * width, (stratum + 1) * width), SAMPLE_PER_STRATUM, replace=False)
        values.extend(sorted(int(item) for item in draw))
    result = np.asarray(values, dtype=np.int16)
    if result.shape != (SAMPLE_COUNT,) or len(np.unique(result)) != SAMPLE_COUNT:
        raise XI1Error("stratified selection is not unique n120")
    return result


def _round_divide(values: np.ndarray, denominator: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.int64)
    return np.where(
        source >= 0,
        (source + denominator // 2) // denominator,
        -((-source + denominator // 2) // denominator),
    ).astype(np.int32)


def _signed_mod(values: np.ndarray) -> np.ndarray:
    return ((np.asarray(values, dtype=np.int64) + 2048) & 0xFFF).astype(np.int32) - 2048


def calibrated_transform(pose6: np.ndarray, calibration: np.ndarray, *, translation: bool) -> np.ndarray:
    """Build the reference-calibrated transform through tac.lie's log/exp oracle."""

    s_t, s_r, _ = (float(value) for value in calibration)
    pose = np.asarray(pose6, dtype=np.float64)
    t = (s_t if translation else 0.0) * np.asarray((pose[2], pose[1], pose[0]), dtype=np.float64)
    rotation = se3.exp_so3(s_r * pose[3:6])
    xi = se3.log_se3(se3.make_T(rotation, t))
    return se3.exp_se3(xi)


def homography_from_pose(pose6: np.ndarray, calibration: np.ndarray, *, translation: bool, mw: ModuleType) -> np.ndarray:
    transform = calibrated_transform(pose6, calibration, translation=translation)
    rotation = se3.rotation_of(transform)
    shift = se3.translation_of(transform)
    pitch = float(calibration[2])
    normal = np.asarray((0.0, -math.cos(pitch), -math.sin(pitch)), dtype=np.float64)
    intrinsics = mw.intrinsics_at(W, H)
    plane = rotation - np.outer(shift, normal) / float(mw.CAMERA_HEIGHT_M)
    return intrinsics @ plane @ np.linalg.inv(intrinsics)


def warp_previous_partition(src: np.ndarray, pose6: np.ndarray, calibration: np.ndarray, mw: ModuleType) -> np.ndarray:
    """TF1's decode-realizable class composite, with geometry supplied by tac.lie."""

    target_grid = mw._target_grid(H, W)
    ground, ground_valid = mw.warp_labels(
        src,
        homography_from_pose(pose6, calibration, translation=True, mw=mw),
        target_grid,
    )
    rotated, rotated_valid = mw.warp_labels(
        src,
        homography_from_pose(pose6, calibration, translation=False, mw=mw),
        target_grid,
    )
    result = src.copy()
    ground_in = ground_valid & np.isin(ground, (0, 1, 3))
    result = np.where(ground_in, ground, result)
    sky_in = rotated_valid & (rotated == 2) & ~ground_in
    result = np.where(sky_in, rotated, result)
    return np.where(src == 4, src, result).astype(np.uint8)


def build_contexts(cache: torch.Tensor, ids: np.ndarray, poses: np.ndarray, calibration: np.ndarray) -> dict[str, Any]:
    root = OUTPUT / "retained/contexts"
    spatial = np.zeros((len(ids), H, W), dtype=np.uint8)
    xi = np.zeros_like(spatial)
    mw = import_path(POSE_WARP_PATH, "ddm_xi1_pose_warp")
    for local, frame in enumerate(ids.astype(int)):
        if frame:
            xi[local] = warp_previous_partition(cache[frame - 1].numpy(), poses[frame], calibration, mw)
    spatial_record = atomic_npy(root / "spatial_zero_n120.npy", spatial)
    xi_record = atomic_npy(root / "xi_warped_previous_n120.npy", xi)
    repeat = np.zeros_like(xi)
    for local, frame in enumerate(ids.astype(int)):
        if frame:
            repeat[local] = warp_previous_partition(cache[frame - 1].numpy(), poses[frame], calibration, mw)
    repeat_record = atomic_npy(root / "xi_warped_previous_n120.repeat.npy", repeat)
    if xi_record["sha256"] != repeat_record["sha256"] or not np.array_equal(xi, repeat):
        raise XI1Error("tac.lie xi context repeat is not deterministic")
    return {
        "spatial": spatial,
        "xi": xi,
        "records": {"spatial": spatial_record, "xi": xi_record, "xi_repeat": repeat_record},
        "xi_repeat_exact": True,
        "construction": "previous decoded partition + carried pose -> tac.lie SE3 -> TF1 stratified class composite",
    }


def relative_screw_features(poses: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    """Return log(inv(T[t-1]) T[t]) from the decoded pose plane via tac.lie."""

    transforms = [calibrated_transform(row, calibration, translation=True) for row in poses]
    result = np.zeros((len(poses), 6), dtype=np.float64)
    for frame in range(1, len(poses)):
        result[frame] = se3.log_se3(se3.inverse(transforms[frame - 1]) @ transforms[frame])
    return result


def _coefficient_features(coefficients: np.ndarray, xi_codes: np.ndarray) -> np.ndarray:
    rows = np.zeros((len(coefficients), 30), dtype=np.int32)
    for frame in range(1, len(coefficients)):
        rows[frame, :12] = coefficients[frame - 1]
        if frame > 1:
            rows[frame, 12:24] = _signed_mod(coefficients[frame - 1] - coefficients[frame - 2])
        rows[frame, 24:] = xi_codes[frame]
    return rows


def fit_linear_prior(coefficients: np.ndarray, poses: np.ndarray, calibration: np.ndarray) -> dict[str, np.ndarray | int]:
    screws = relative_screw_features(poses, calibration)
    maxima = np.maximum(np.max(np.abs(screws), axis=0), 1e-12)
    scales = maxima / 2047.0
    xi_codes = np.rint(screws / scales).clip(-2047, 2047).astype(np.int16)
    features = _coefficient_features(coefficients, xi_codes)
    design = np.column_stack((features.astype(np.float64), np.ones(len(features))))
    fitted, *_ = np.linalg.lstsq(design, coefficients.astype(np.float64), rcond=1e-8)
    q = 4096
    weights = np.rint(fitted[:-1] * q).clip(-32768, 32767).astype(np.int16)
    baseline = _round_divide(features.astype(np.int64) @ weights.astype(np.int64), q)
    biases = np.rint(np.median(coefficients.astype(np.int64) - baseline, axis=0) * q).astype(np.int32)
    return {"q": q, "scales": scales.astype("<f4"), "weights": weights, "biases": biases}


def learned_predictions(features: np.ndarray, model: dict[str, np.ndarray | int]) -> np.ndarray:
    q = int(model["q"])
    weights = np.asarray(model["weights"], dtype=np.int16)
    biases = np.asarray(model["biases"], dtype=np.int32)
    accumulator = features.astype(np.int64) @ weights.astype(np.int64) + biases.astype(np.int64)
    return _signed_mod(_round_divide(accumulator, q))


def encode_learned_pose_packet(
    raw_cpr1: bytes,
    selector: bytes,
    poses: np.ndarray,
    calibration: np.ndarray,
    repack: ModuleType,
) -> tuple[bytes, dict[str, Any]]:
    """Encode a self-contained counted ξ-conditioned model plus exact Rice residual."""

    basis_bits, _, fields = repack._parse_cpr1(raw_cpr1, dimensions=12)
    _, coefficients = repack._coefficients_from_cpr1(raw_cpr1, frames=FRAME_COUNT, dimensions=12)
    model = fit_linear_prior(coefficients, poses, calibration)
    screws = relative_screw_features(poses, calibration)
    scales = np.asarray(model["scales"], dtype=np.float32)
    xi_codes = np.rint(screws / scales).clip(-2047, 2047).astype(np.int16)
    features = _coefficient_features(coefficients, xi_codes)
    predictions = learned_predictions(features, model)
    residuals = _signed_mod(coefficients - predictions)
    ks, rice, residual_bits = repack._rice_encode(repack._zigzag(residuals), 1)
    pose_payload = np.asarray(poses, dtype="<f2").tobytes()
    model_payload = (
        np.asarray(model["scales"], dtype="<f4").tobytes()
        + np.asarray(model["weights"], dtype="<i2").tobytes()
        + np.asarray(model["biases"], dtype="<i4").tobytes()
    )
    candidate_root = OUTPUT / "retained/leg_b/learned_xi_linear_plus_residual"
    pose_record = retain_payload(candidate_root / "counted_pose_context.f16", pose_payload)
    model_record = retain_payload(candidate_root / "learned_model.bin", model_payload)
    header = struct.pack(
        "<4sBBBBIIIIH",
        b"XIP1",
        1,
        12,
        1,
        0,
        int(basis_bits),
        int(residual_bits),
        len(pose_payload),
        len(model_payload),
        len(selector),
    )
    packet = (
        header
        + pose_payload
        + model_payload
        + fields["scales"]
        + fields["lengths"]
        + ks.reshape(-1).tobytes()
        + fields["basis"]
        + rice
        + selector
    )
    packet_record = retain_payload(candidate_root / "packet.bin", packet)
    restored = decode_learned_pose_packet(packet, calibration, repack)
    if restored != raw_cpr1:
        raise XI1Error("learned pose packet failed canonical CPR1 equality")
    return packet, {
        "header_bytes": len(header),
        "counted_pose_context_bytes": len(pose_payload),
        "counted_pose_context": pose_record,
        "learned_model_bytes": len(model_payload),
        "learned_model": model_record,
        "common_scales_bytes": len(fields["scales"]),
        "common_lengths_bytes": len(fields["lengths"]),
        "rice_parameter_bytes": ks.size,
        "basis_bytes": len(fields["basis"]),
        "rice_payload_bytes": len(rice),
        "selector_bytes": len(selector),
        "packet_bytes": len(packet),
        "packet": packet_record,
        "residual_bits": int(residual_bits),
        "model_kind": "counted fixed-point multivariate linear dynamics prior with relative tac.lie screw features",
    }


def decode_learned_pose_packet(packet: bytes, calibration: np.ndarray, repack: ModuleType) -> bytes:
    header_size = struct.calcsize("<4sBBBBIIIIH")
    if len(packet) < header_size:
        raise XI1Error("truncated XIP1 packet")
    magic, version, dimensions, segments, reserved, basis_bits, residual_bits, pose_bytes, model_bytes, selector_bytes = struct.unpack_from(
        "<4sBBBBIIIIH", packet
    )
    if (magic, version, dimensions, segments, reserved) != (b"XIP1", 1, 12, 1, 0):
        raise XI1Error("invalid XIP1 header")
    offset = header_size
    poses = np.frombuffer(packet[offset : offset + pose_bytes], dtype="<f2").astype(np.float64).reshape(FRAME_COUNT, 6)
    offset += pose_bytes
    model_end = offset + model_bytes
    if model_bytes != 6 * 4 + 30 * 12 * 2 + 12 * 4:
        raise XI1Error("invalid XIP1 model length")
    scales = np.frombuffer(packet[offset : offset + 24], dtype="<f4").copy()
    offset += 24
    weights = np.frombuffer(packet[offset : offset + 720], dtype="<i2").reshape(30, 12).copy()
    offset += 720
    biases = np.frombuffer(packet[offset : offset + 48], dtype="<i4").copy()
    offset += 48
    if offset != model_end:
        raise XI1Error("XIP1 model field accounting mismatch")
    common_scales = packet[offset : offset + 96]
    offset += 96
    lengths = packet[offset : offset + 32]
    offset += 32
    ks = np.frombuffer(packet[offset : offset + 12], dtype=np.uint8).reshape(12, 1).copy()
    offset += 12
    basis_bytes = (basis_bits + 7) // 8
    basis = packet[offset : offset + basis_bytes]
    offset += basis_bytes
    rice_bytes = (residual_bits + 7) // 8
    rice = packet[offset : offset + rice_bytes]
    offset += rice_bytes
    selector = packet[offset : offset + selector_bytes]
    offset += selector_bytes
    if offset != len(packet):
        raise XI1Error("XIP1 has trailing bytes")
    if selector_bytes:
        from runtime.frame0_selector import decode_selector

        decode_selector(selector)
    screws = relative_screw_features(poses, calibration)
    xi_codes = np.rint(screws / scales).clip(-2047, 2047).astype(np.int16)
    residuals = repack._unzigzag(repack._rice_decode(ks, rice, residual_bits, FRAME_COUNT, 12))
    coefficients = np.empty_like(residuals)
    model = {"q": 4096, "scales": scales, "weights": weights, "biases": biases}
    for frame in range(FRAME_COUNT):
        features = _coefficient_features(coefficients[: frame + 1], xi_codes[: frame + 1])[-1:]
        prediction = learned_predictions(features, model)[0]
        coefficients[frame] = _signed_mod(prediction + residuals[frame])
    original = repack._predict_residuals(coefficients, np.zeros(12, dtype=np.uint8), 0)
    original_ks, original_rice, original_bits = repack._rice_encode(repack._zigzag(original), 1)
    return (
        b"CPR1"
        + struct.pack("<II", basis_bits, original_bits)
        + common_scales
        + lengths
        + original_ks.reshape(-1).tobytes()
        + basis
        + original_rice
    )


def run_leg_b(poses: np.ndarray, calibration: np.ndarray) -> dict[str, Any]:
    root = OUTPUT / "retained/leg_b"
    for path in (str(CP135_ROOT), str(BOOK_SRC)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from cpr1_sub4 import carrier_repack as repack
    from cpr1_sub4.entropy.coefficient_ar1_codec import decode_cap1, inspect_cap1
    from runtime.carrier_repack import pack_frame0_selector_carrier, split_frame0_selector_carrier
    from runtime.residual_archive import read_residual_archive

    parts = read_residual_archive(CP135_ARCHIVE)
    cap1, selector = split_frame0_selector_carrier(parts.carrier_blob)
    if selector is None:
        raise XI1Error("CP135 current carrier lacks its counted selector")
    raw_cpr1 = decode_cap1(cap1, frames=FRAME_COUNT, dimensions=12)
    direct = pack_frame0_selector_carrier(raw_cpr1, selector)
    incumbent = pack_frame0_selector_carrier(cap1, selector)
    learned, learned_fields = encode_learned_pose_packet(raw_cpr1, selector, poses, calibration, repack)
    rows: list[dict[str, Any]] = []
    candidates = (
        ("direct_cpr1_current_object", direct, raw_cpr1, {"mechanism": "direct canonical CPR1 plus unchanged selector"}),
        ("ar1_bias_cap1_current_object", incumbent, decode_cap1(cap1, frames=FRAME_COUNT, dimensions=12), {"mechanism": "fixed-point AR1+bias plus Rice residual", "fields": inspect_cap1(cap1, frames=FRAME_COUNT, dimensions=12)}),
        ("learned_xi_linear_plus_residual", learned, decode_learned_pose_packet(learned, calibration, repack), {"mechanism": learned_fields["model_kind"], "fields": learned_fields}),
    )
    for name, packet, restored, detail in candidates:
        packet_record = _atomic_bytes(root / name / "packet.bin", packet)
        repeat_record = _atomic_bytes(root / name / "packet.repeat.bin", packet)
        decoded_record = _atomic_bytes(root / name / "decoded.cpr1", restored)
        if packet_record["sha256"] != repeat_record["sha256"] or restored != raw_cpr1:
            raise XI1Error(f"Leg B exact decode or repeat failed for {name}")
        row = {
            "candidate": name,
            "packet": packet_record,
            "packet_repeat": repeat_record,
            "decoded_cpr1": decoded_record,
            "decoded_cpr1_exact": True,
            "realized_d_pose": POSE_D,
            "d_pose_provenance": "inherited CP135 contest-CUDA n600 printed-8dp component; exact rendered bytes unchanged after canonical CPR1 restoration",
            "score_claim": False,
            "axis": AXIS_B,
            **detail,
        }
        atomic_json(root / name / "RESULT.json", row, replace=True)
        rows.append(row)
    direct_bytes = rows[0]["packet"]["bytes"]
    non_direct = rows[1:]
    fb_fires = all(row["packet"]["bytes"] >= direct_bytes for row in non_direct)
    result = {
        "schema": "ddm_xi1_leg_b.v1",
        "rows": rows,
        "matched_realized_d_pose": POSE_D,
        "falsifier_FB": {
            "fires": fb_fires,
            "rule": "both exact screw-dynamics formulations are at least direct bytes at matched d_pose",
            "verdict_scope": "FAMILY" if fb_fires else "FORMULATION",
            "note": (
                "The learned row charges its 7200-byte decoded xi plane because CP135 does not already carry geometric xi."
            ),
        },
        "cp135_custody": {"archive": file_record(CP135_ARCHIVE), "bytes": CP135_BYTES, "score": CP135_SCORE},
        "axis": AXIS_B,
        "score_claim": False,
    }
    atomic_json(OUTPUT / "LEG_B_RESULT.json", result, replace=True)
    return result


def configure_hpac() -> tuple[ModuleType, ModuleType, ModuleType, ModuleType]:
    # These three PR130 modules use absolute sibling imports.  They must share
    # the canonical ``hpac_integer`` module object: loading hpac_integer.py
    # under a private alias creates distinct IntegerConv2d class identities,
    # causing every self-compression isinstance check to fail silently.
    for path in (FX1_RUNTIME, INTAKE_CODE):
        text = str(path)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)
    integer = importlib.import_module("hpac_integer")
    compression = importlib.import_module("hpac_self_compress")
    packer = importlib.import_module("pack_hpac_self_compress")
    expected_modules = {
        "hpac_integer": INTAKE_CODE / "hpac_integer.py",
        "hpac_self_compress": INTAKE_CODE / "hpac_self_compress.py",
        "pack_hpac_self_compress": INTAKE_CODE / "pack_hpac_self_compress.py",
    }
    for name, expected_path in expected_modules.items():
        module = sys.modules[name]
        observed_path = Path(module.__file__).resolve()
        if observed_path != expected_path.resolve():
            raise XI1Error(f"{name} resolved to {observed_path}, expected {expected_path}")
    if (
        compression.IntegerConv2d is not integer.IntegerConv2d
        or compression.IntegerLinear is not integer.IntegerLinear
        or packer.IntegerHPAC is not integer.IntegerHPAC
    ):
        raise XI1Error("HPAC module class identities diverged; self-compression registration is unsafe")
    inflate = import_path(FX1_RUNTIME / "inflate.py", "ddm_xi1_hpac_inflate")
    return integer, compression, packer, inflate


def model_args() -> Any:
    class Args:
        channels = 64
        patch = 64
        delta = 2
        frame_dim = 8
        weight_bound = 127
        activation_bound = 127
        weight_exponent_min = -6

    return Args()


def build_train_model(integer: ModuleType, compression: ModuleType, device: torch.device) -> torch.nn.Module:
    model = integer.IntegerHPAC(
        channels=64,
        patch=64,
        delta=2,
        frame_dim=8,
        norm_mode="none",
        activation="relu",
        use_frame_scale=True,
        weight_bound=127,
        activation_bound=127,
        use_weight_scales=True,
        weight_exponent_min=-6,
        use_spm=True,
        use_norm_gates=False,
    ).to(device)
    compression.enable_self_compression(model, 8.0)
    _require_bit_depth_schema(model, label="fresh trainer")
    initial = torch.load(HPAC_INIT, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(initial["state_dict"], strict=False)
    unexpected_missing = {name for name in incompatible.missing_keys if not name.endswith(".bit_depth")}
    if incompatible.unexpected_keys or unexpected_missing:
        raise XI1Error(f"HPAC initializer is incompatible: {incompatible}")
    _require_bit_depth_schema(model, label="initialized trainer")
    return model


def _bit_depth_names(value: torch.nn.Module | dict[str, Any]) -> tuple[str, ...]:
    names = value.keys() if isinstance(value, dict) else dict(value.named_parameters()).keys()
    return tuple(sorted(name for name in names if name.endswith(".bit_depth")))


def _require_bit_depth_schema(value: torch.nn.Module | dict[str, Any], *, label: str) -> tuple[str, ...]:
    observed = _bit_depth_names(value)
    if observed != EXPECTED_BIT_DEPTH_NAMES:
        missing = sorted(set(EXPECTED_BIT_DEPTH_NAMES) - set(observed))
        unexpected = sorted(set(observed) - set(EXPECTED_BIT_DEPTH_NAMES))
        raise XI1Error(
            f"{label} has an invalid learned bit-depth schema; missing={missing}, unexpected={unexpected}"
        )
    return observed


def optimizer_for(model: torch.nn.Module) -> torch.optim.Optimizer:
    parameters = dict(model.named_parameters())
    bit_names = set(_require_bit_depth_schema(model, label="optimizer model"))
    exponent_names = {name for name in parameters if name.endswith(".exponent")}
    other_names = set(parameters) - bit_names - exponent_names
    groups: list[dict[str, Any]] = [
        {"params": [parameters[name] for name in sorted(other_names)], "lr": 0.003, "eps": 1e-8},
        {
            "params": [parameters[name] for name in sorted(bit_names)],
            "lr": 0.01,
            "eps": 1e-6,
            "weight_decay": 0.0,
        },
    ]
    if exponent_names:
        groups.append({"params": [parameters[name] for name in sorted(exponent_names)], "lr": 0.0002, "eps": 1e-8})
    return torch.optim.AdamW(groups, weight_decay=1e-5)


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def checkpoint_payload(
    *,
    epoch: int,
    epochs: int,
    context_mode: str,
    rate_lambda: float,
    ids: np.ndarray,
    model: torch.nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    live_state = _cpu_tree(model.state_dict())
    ema_shadow = _cpu_tree(ema.state_dict())
    bit_depth_names = _require_bit_depth_schema(live_state, label="checkpoint live state")
    _require_bit_depth_schema(ema_shadow, label="checkpoint EMA shadow")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "epoch": epoch,
        "epochs": epochs,
        "phase": "discrete_qat" if epoch > epochs // 2 else ("initial" if epoch == 0 else "continuous"),
        "context_mode": context_mode,
        "rate_lambda": rate_lambda,
        "sample_frame_ids": ids.astype(int).tolist(),
        "live_state_dict": live_state,
        "ema_shadow": ema_shadow,
        "self_compression": {
            "bit_depth_parameter_names": list(bit_depth_names),
            "trainer_registration": "canonical hpac_integer class identity",
        },
        "ema_decay": ema.decay,
        "ema_updates": ema._num_updates,
        "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
        "scheduler_state_dict": _cpu_tree(scheduler.state_dict()),
        "shuffle_generator_state": generator.get_state(),
        "torch_cpu_rng_state": torch.random.get_rng_state(),
        "mps_rng_state": torch.mps.get_rng_state().cpu(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "history": history,
        "deployment_weights": "ema_shadow",
        "config": {
            "batch_size": 8,
            "eval_batch_size": 4,
            "lr": 0.003,
            "lr_exponent": 0.0002,
            "lr_bits": 0.01,
            "bit_eps": 1e-6,
            "qat_fraction": 0.5,
            "seed": SEED,
            "selection": "seeded_stratified_random_n120_10x12",
        },
    }


def _restore_checkpoint(
    checkpoint: dict[str, Any],
    *,
    context_mode: str,
    rate_lambda: float,
    epochs: int,
    ids: np.ndarray,
    model: torch.nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    generator: torch.Generator,
) -> tuple[int, list[dict[str, Any]]]:
    schema = checkpoint.get("schema")
    if schema == LEGACY_CHECKPOINT_SCHEMA:
        raise XI1Error(
            "legacy Leg-A checkpoint has no trained bit_depth state or optimizer history; "
            "fresh 20-epoch rerun required under the fix output root"
        )
    if (
        schema != CHECKPOINT_SCHEMA
        or checkpoint.get("context_mode") != context_mode
        or checkpoint.get("rate_lambda") != rate_lambda
        or checkpoint.get("epochs") != epochs
        or checkpoint.get("sample_frame_ids") != ids.astype(int).tolist()
    ):
        raise XI1Error("Leg A resume identity changed")
    _require_bit_depth_schema(checkpoint["live_state_dict"], label="resume live state")
    _require_bit_depth_schema(checkpoint["ema_shadow"], label="resume EMA shadow")
    model.load_state_dict(checkpoint["live_state_dict"])
    ema.shadow = {name: value.to(next(model.parameters()).device) for name, value in checkpoint["ema_shadow"].items()}
    ema._num_updates = int(checkpoint["ema_updates"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    generator.set_state(checkpoint["shuffle_generator_state"])
    torch.random.set_rng_state(checkpoint["torch_cpu_rng_state"])
    torch.mps.set_rng_state(checkpoint["mps_rng_state"])
    random.setstate(checkpoint["python_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    return int(checkpoint["epoch"]), list(checkpoint["history"])


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    ema: EMA,
    compression: ModuleType,
    target: torch.Tensor,
    ids: torch.Tensor,
    context: torch.Tensor,
) -> dict[str, Any]:
    live = _cpu_tree(model.state_dict())
    model.load_state_dict(ema.shadow)
    compression.set_deployed_bit_depths(model, True)
    model.eval()
    nats = 0.0
    misses = 0
    for start in range(0, len(target), 4):
        logits = model(target[start : start + 4], ids[start : start + 4], context[start : start + 4])
        nats += float(F.cross_entropy(logits, target[start : start + 4], reduction="sum"))
        misses += int((logits.argmax(dim=1) != target[start : start + 4]).sum().item())
    pixels = target.numel()
    result = {
        "bpp": nats / math.log(2) / pixels,
        "top1_error": misses / pixels,
        "estimated_token_bytes": math.ceil(nats / math.log(2) / 8),
        "estimated_model_bytes": math.ceil(compression.estimated_model_bits(model) / 8),
        "bit_depth_histogram": compression.bit_depth_histogram(model),
        "byte_authority": "ADVISORY_ESTIMATE_NOT_SERIALIZED",
        "evaluated_weights": "ema_shadow",
    }
    model.load_state_dict(live)
    return result


def train_cell(
    *,
    context_mode: str,
    rate_lambda: float,
    epochs: int,
    ids: np.ndarray,
    target_cpu: torch.Tensor,
    context_cpu: np.ndarray,
    integer: ModuleType,
    compression: ModuleType,
) -> dict[str, Any]:
    cell_name = f"lambda_{str(rate_lambda).replace('.', 'p')}_{context_mode}"
    root = LEG_A_RETAINED / cell_name
    result_path = root / "RESULT.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    device = torch.device("mps")
    torch.manual_seed(SEED)
    torch.mps.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    model = build_train_model(integer, compression, device)
    optimizer = optimizer_for(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.003 * 0.02)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    updates = epochs * math.ceil(SAMPLE_COUNT / 8)
    ema_policy = resolve_ema_policy(updates, target_seed_fraction=0.01)
    ema = EMA(model, decay=float(ema_policy["decay"]), warmup=True)
    target = target_cpu.to(device)
    context = torch.from_numpy(context_cpu.astype(np.int64)).to(device)
    frame_ids = torch.from_numpy(ids.astype(np.int64)).to(device)
    start_epoch = 0
    history: list[dict[str, Any]] = []
    latest = root / "checkpoints/latest.pt"
    if latest.is_file():
        checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
        start_epoch, history = _restore_checkpoint(
            checkpoint,
            context_mode=context_mode,
            rate_lambda=rate_lambda,
            epochs=epochs,
            ids=ids,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            generator=generator,
        )
    else:
        payload = checkpoint_payload(
            epoch=0,
            epochs=epochs,
            context_mode=context_mode,
            rate_lambda=rate_lambda,
            ids=ids,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            generator=generator,
            history=history,
        )
        atomic_torch(root / "checkpoints/initial_stage_start.pt", payload)
        atomic_torch(latest, payload, replace=True)
    started = time.time()
    pixels = target.numel()
    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        discrete = epoch > epochs // 2
        compression.set_deployed_bit_depths(model, discrete)
        permutation = torch.randperm(SAMPLE_COUNT, generator=generator).to(device)
        for start in range(0, SAMPLE_COUNT, 8):
            index = permutation[start : start + 8]
            logits = model(target[index], frame_ids[index], context[index])
            task_loss = F.cross_entropy(logits, target[index])
            rate_loss = rate_lambda * math.log(2) * compression.variable_weight_bits(model, deployed=False) / pixels
            loss = task_loss + rate_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            ema.update(model)
        scheduler.step()
        metrics = evaluate_model(model, ema, compression, target, frame_ids, context)
        record = {"epoch": epoch, "phase": "discrete_qat" if discrete else "continuous", **metrics}
        history.append(record)
        print(json.dumps({"cell": cell_name, **record, "elapsed_s": time.time() - started}), flush=True)
        payload = checkpoint_payload(
            epoch=epoch,
            epochs=epochs,
            context_mode=context_mode,
            rate_lambda=rate_lambda,
            ids=ids,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            generator=generator,
            history=history,
        )
        periodic = root / f"checkpoints/periodic/epoch_{epoch:04d}.pt"
        atomic_torch(periodic, payload)
        if epoch == epochs // 2:
            atomic_torch(root / f"checkpoints/continuous_stage_end_epoch_{epoch:04d}.pt", payload)
        if epoch == epochs:
            atomic_torch(root / f"checkpoints/qat_stage_end_epoch_{epoch:04d}.pt", payload)
        atomic_torch(latest, payload, replace=True)
    terminal = torch.load(latest, map_location="cpu", weights_only=False)
    return pack_and_encode_cell(
        cell_name=cell_name,
        context_mode=context_mode,
        rate_lambda=rate_lambda,
        ids=ids,
        target_cpu=target_cpu,
        context_cpu=context_cpu,
        terminal=terminal,
        history=history,
    )


def _array_blocks(length: int, block: int = 65_536) -> Iterable[tuple[int, int]]:
    for start in range(0, length, block):
        yield start, min(start + block, length)


@torch.no_grad()
def materialize_code_chunks(
    *,
    root: Path,
    model: torch.nn.Module,
    inflate: ModuleType,
    ids: np.ndarray,
    target: torch.Tensor,
    context: np.ndarray,
) -> dict[str, Any]:
    manifest_path = root / "code_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") is True:
            for row in manifest["chunks"]:
                for field in ("symbols", "codes"):
                    record = row[field]
                    if file_record(Path(record["path"])) != record:
                        raise XI1Error("retained Leg A code chunk failed custody")
            return manifest
    masks = inflate.group_masks(torch.device("cpu"))
    sparse = inflate.SparseIntegerHPAC(model, H, W)
    rows: list[dict[str, Any]] = []
    for chunk_start in range(0, SAMPLE_COUNT, 12):
        chunk_end = min(chunk_start + 12, SAMPLE_COUNT)
        symbols_parts: list[np.ndarray] = []
        code_parts: list[np.ndarray] = []
        started = time.time()
        for local in range(chunk_start, chunk_end):
            frame = int(ids[local])
            previous = torch.from_numpy(context[local : local + 1].astype(np.int64))
            current = torch.zeros_like(previous)
            prepared = model.prepare_frame_context(torch.tensor([frame]), previous)
            for group, mask in enumerate(masks):
                logits = sparse.selected_logits(current, prepared, group)
                codes = logits.mul(8).round().clamp(-32768, 32767).to(torch.int16).numpy()
                symbols = target[local][mask].numpy().astype(np.uint8)
                code_parts.append(codes)
                symbols_parts.append(symbols)
                current[0, mask] = torch.from_numpy(symbols.astype(np.int64))
        symbols_array = np.concatenate(symbols_parts)
        codes_array = np.concatenate(code_parts)
        expected_tokens = (chunk_end - chunk_start) * TOKENS_PER_FRAME
        if symbols_array.shape != (expected_tokens,) or codes_array.shape != (expected_tokens, CLASSES):
            raise XI1Error("Leg A materialized code geometry changed")
        row = {
            "sample_start": chunk_start,
            "sample_end": chunk_end,
            "frame_ids": ids[chunk_start:chunk_end].astype(int).tolist(),
            "symbols": atomic_npy(root / f"symbols_{chunk_start:03d}_{chunk_end:03d}.npy", symbols_array),
            "codes": atomic_npy(root / f"codes_{chunk_start:03d}_{chunk_end:03d}.npy", codes_array),
            "materialize_wall_s": time.time() - started,
        }
        rows.append(row)
        atomic_json(
            manifest_path,
            {
                "schema": "ddm_xi1_sample_code_chunks.v1",
                "complete": chunk_end == SAMPLE_COUNT,
                "selection": "seeded_stratified_random_n120_10x12",
                "tokens": chunk_end * TOKENS_PER_FRAME,
                "chunks": rows,
            },
            replace=True,
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _range_encode(manifest: dict[str, Any], hp3: ModuleType) -> bytes:
    family = constriction.stream.model.Categorical(perfect=False)
    encoder = constriction.stream.queue.RangeEncoder()
    for row in manifest["chunks"]:
        symbols = np.load(row["symbols"]["path"], mmap_mode="r", allow_pickle=False)
        codes = np.load(row["codes"]["path"], mmap_mode="r", allow_pickle=False)
        for start, end in _array_blocks(len(symbols)):
            encoder.encode(
                np.asarray(symbols[start:end], dtype=np.int32),
                family,
                hp3.probability_tables(codes[start:end]),
            )
    return encoder.get_compressed().tobytes()


def encode_and_decode_range(root: Path, manifest: dict[str, Any], hp3: ModuleType) -> dict[str, Any]:
    payload = _range_encode(manifest, hp3)
    repeat = _range_encode(manifest, hp3)
    payload_record = _atomic_bytes(root / "tokens.range", payload)
    repeat_record = _atomic_bytes(root / "tokens.repeat.range", repeat)
    if payload_record["sha256"] != repeat_record["sha256"]:
        raise XI1Error("Leg A Range payload repeat differs")
    decoder = constriction.stream.queue.RangeDecoder(np.frombuffer(payload, dtype=np.uint32))
    family = constriction.stream.model.Categorical(perfect=False)
    decoded_parts: list[np.ndarray] = []
    source_digest = hashlib.sha256()
    for row in manifest["chunks"]:
        symbols = np.load(row["symbols"]["path"], mmap_mode="r", allow_pickle=False)
        codes = np.load(row["codes"]["path"], mmap_mode="r", allow_pickle=False)
        source_digest.update(np.asarray(symbols).tobytes(order="C"))
        for start, end in _array_blocks(len(symbols)):
            decoded = decoder.decode(family, hp3.probability_tables(codes[start:end])).astype(np.uint8)
            if not np.array_equal(decoded, symbols[start:end]):
                raise XI1Error("Leg A Range decode changed a selected token")
            decoded_parts.append(decoded)
    decoded_record = _atomic_bytes(root / "tokens.decoded.u8", np.concatenate(decoded_parts).tobytes())
    if decoded_record["sha256"] != source_digest.hexdigest():
        raise XI1Error("Leg A decoded token digest differs")
    return {
        "range": payload_record,
        "range_repeat": repeat_record,
        "decoded": decoded_record,
        "decode_exact": True,
        "repeat_exact": True,
    }


def pack_and_encode_cell(
    *,
    cell_name: str,
    context_mode: str,
    rate_lambda: float,
    ids: np.ndarray,
    target_cpu: torch.Tensor,
    context_cpu: np.ndarray,
    terminal: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    root = LEG_A_RETAINED / cell_name
    hp3 = import_path(HP3_PATH, f"ddm_xi1_hp3_{cell_name}")
    _, _, packer, inflate = configure_hpac()
    args = model_args()
    source = load_terminal_ema_for_pack(terminal=terminal, packer=packer, args=args)
    raw = packer.serialize_self_compressed(source)
    raw_record = _atomic_bytes(root / "hpac.raw", raw)
    model_xz = lzma.compress(raw, format=lzma.FORMAT_XZ, filters=hp3.LZMA_FILTERS)
    xz_record = _atomic_bytes(root / "hpac.xz", model_xz)
    restored = packer.model_from_args(args, False).eval()
    packer.deserialize_self_compressed(restored, raw)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    parity_current = torch.randint(0, CLASSES, (2, H, W), generator=generator)
    parity_previous = torch.randint(0, CLASSES, (2, H, W), generator=generator)
    parity_ids = torch.tensor([0, 599])
    max_logit_diff = float((source(parity_current, parity_ids, parity_previous) - restored(parity_current, parity_ids, parity_previous)).abs().max())
    if max_logit_diff != 0.0:
        raise XI1Error(f"packed model changed logits for {cell_name}: {max_logit_diff}")
    manifest = materialize_code_chunks(
        root=root / "codes",
        model=restored,
        inflate=inflate,
        ids=ids,
        target=target_cpu,
        context=context_cpu,
    )
    coded = encode_and_decode_range(root, manifest, hp3)
    token_bytes = coded["range"]["bytes"]
    result = {
        "schema": "ddm_xi1_leg_a_cell.v1",
        "candidate": cell_name,
        "context_mode": context_mode,
        "rate_lambda": rate_lambda,
        "selection": "seeded_stratified_random_n120; 10 strata x 12 frames",
        "sample_frames": SAMPLE_COUNT,
        "sample_tokens": SAMPLE_COUNT * TOKENS_PER_FRAME,
        "hpac_raw": raw_record,
        "hpac_xz": xz_record,
        "range": coded["range"],
        "range_repeat": coded["range_repeat"],
        "decoded_tokens": coded["decoded"],
        "decode_exact": coded["decode_exact"],
        "packed_logit_max_abs_diff": max_logit_diff,
        "sample_token_bytes": token_bytes,
        "sample_joint_bytes": token_bytes + xz_record["bytes"],
        "projected_n600_token_bytes": token_bytes * 5,
        "projected_n600_joint_bytes": token_bytes * 5 + xz_record["bytes"],
        "projection_warning": "5x selected-token projection; model counted once; not a standalone n600 Range stream",
        "epochs": terminal["epochs"],
        "terminal_epoch": terminal["epoch"],
        "training_history": history,
        "matched_config": terminal["config"],
        "deployment_weights": "terminal ema_shadow",
        "axis": AXIS_A,
        "score_claim": False,
    }
    atomic_json(root / "RESULT.json", result)
    return result


def load_terminal_ema_for_pack(*, terminal: dict[str, Any], packer: ModuleType, args: Any) -> torch.nn.Module:
    """Build the production pack source and strictly load trained bit depths."""

    if terminal.get("schema") == LEGACY_CHECKPOINT_SCHEMA:
        raise XI1Error(
            "legacy Leg-A terminal cannot be packed: bit_depth was never registered or trained; "
            "fresh 20-epoch rerun required"
        )
    if terminal.get("schema") != CHECKPOINT_SCHEMA:
        raise XI1Error(f"unsupported Leg-A terminal schema: {terminal.get('schema')!r}")
    _require_bit_depth_schema(terminal["ema_shadow"], label="pack terminal EMA shadow")
    source = packer.model_from_args(args, True).eval()
    _require_bit_depth_schema(source, label="pack source")
    source.load_state_dict(terminal["ema_shadow"], strict=True)
    packer.set_deployed_bit_depths(source, True)
    return source


def run_leg_a(
    *,
    cache: torch.Tensor,
    ids: np.ndarray,
    contexts: dict[str, Any],
    epochs: int,
) -> dict[str, Any]:
    integer, compression, _, _ = configure_hpac()
    target = cache[torch.from_numpy(ids.astype(np.int64))].long()
    rows: list[dict[str, Any]] = []
    for rate_lambda in RATE_LAMBDAS:
        for context_mode in CONTEXT_MODES:
            rows.append(
                train_cell(
                    context_mode=context_mode,
                    rate_lambda=rate_lambda,
                    epochs=epochs,
                    ids=ids,
                    target_cpu=target,
                    context_cpu=contexts[context_mode],
                    integer=integer,
                    compression=compression,
                )
            )
    by_key = {(row["rate_lambda"], row["context_mode"]): row for row in rows}
    rung_verdicts = []
    for rate_lambda in RATE_LAMBDAS:
        spatial = by_key[(rate_lambda, "spatial")]
        xi = by_key[(rate_lambda, "xi")]
        ratio = xi["sample_token_bytes"] / spatial["sample_token_bytes"]
        rung_verdicts.append(
            {
                "rate_lambda": rate_lambda,
                "spatial_token_bytes": spatial["sample_token_bytes"],
                "xi_token_bytes": xi["sample_token_bytes"],
                "xi_over_spatial": ratio,
                "xi_below_0p98": ratio < 0.98,
            }
        )
    fa_fires = all(not row["xi_below_0p98"] for row in rung_verdicts)
    result = {
        "schema": "ddm_xi1_leg_a.v1",
        "rows": rows,
        "rung_verdicts": rung_verdicts,
        "falsifier_FA": {
            "fires": fa_fires,
            "rule": "xi conditional token bytes are at least 0.98 times spatial-only at both capacity rungs",
            "verdict_scope": "FAMILY" if fa_fires else "FORMULATION",
            "scope_note": f"jointly trained n120 stratified screen with {epochs} epochs; full n600 60-epoch promotion remains separate",
        },
        "context_records": contexts["records"],
        "axis": AXIS_A,
        "score_claim": False,
    }
    atomic_json(LEG_A_RESULT, result, replace=True)
    return result


def queue_dispositions(leg_a: dict[str, Any] | None, leg_b: dict[str, Any] | None) -> list[str]:
    rows: list[str] = []
    queue = FIX_OUTPUT / "queue"
    if leg_a is not None:
        if leg_a["falsifier_FA"]["fires"]:
            rows.append(
                "ddm_xi1_leg_a_n600: FOLDED. Owner: ddm_xi1. Consumer store: "
                f"{LEG_A_RESULT}. Fire trigger: none; both n120 matched rungs met the FA close threshold."
            )
        else:
            order = {
                "schema": "ddm_xi1_leg_a_n600_fire_order.v1",
                "disposition": "QUEUED-WITH-A-FIRE-ORDER",
                "owner": "MAIN",
                "consumer_store": str(queue / "leg_a_n600.json"),
                "fire_trigger": "one n120 xi row is below 0.98 times its matched spatial control; run the winning lambda at full n600/60 epochs through the CL1 governed chain",
            }
            atomic_json(queue / "leg_a_n600.json", order)
            rows.append(
                "ddm_xi1_leg_a_n600: QUEUED-WITH-A-FIRE-ORDER. Owner: MAIN. Consumer store: "
                f"{queue / 'leg_a_n600.json'}. Fire trigger: one n120 xi row is below 0.98 times its matched spatial control; run the winning lambda at full n600/60 epochs through the CL1 governed chain."
            )
    if leg_b is not None:
        learned = next(row for row in leg_b["rows"] if row["candidate"].startswith("learned_"))
        direct = leg_b["rows"][0]
        if learned["packet"]["bytes"] < direct["packet"]["bytes"]:
            order = {
                "schema": "ddm_xi1_leg_b_runtime_fire_order.v1",
                "disposition": "QUEUED-WITH-A-FIRE-ORDER",
                "owner": "MAIN",
                "consumer_store": str(queue / "leg_b_runtime.json"),
                "fire_trigger": "learned XIP1 packet is smaller than direct current-object CPR1 and restores it byte-exactly; integrate the decoder and repackage CP135",
            }
            atomic_json(queue / "leg_b_runtime.json", order)
            rows.append(
                "ddm_xi1_leg_b_runtime: QUEUED-WITH-A-FIRE-ORDER. Owner: MAIN. Consumer store: "
                f"{queue / 'leg_b_runtime.json'}. Fire trigger: learned XIP1 packet is smaller than direct current-object CPR1 and restores it byte-exactly; integrate the decoder and repackage CP135."
            )
        else:
            rows.append(
                "ddm_xi1_leg_b_runtime: FOLDED. Owner: ddm_xi1. Consumer store: "
                f"{OUTPUT / 'LEG_B_RESULT.json'}. Fire trigger: none; CAP1 is already the CP135 incumbent and counted geometric-xi conditioning did not beat direct storage."
            )
    atomic_json(FIX_OUTPUT / "QUEUE_DISPOSITIONS.json", {"rows": rows}, replace=True)
    _atomic_bytes(
        FIX_OUTPUT / "QUEUE_DISPOSITIONS.md",
        ("\n".join(f"- {row}" for row in rows) + "\n").encode(),
        replace=True,
    )
    return rows


def self_test() -> None:
    ids = selected_frame_ids()
    assert len(ids) == 120 and len(np.unique(ids)) == 120
    assert all(np.any((ids >= start) & (ids < start + 60)) for start in range(0, 600, 60))
    values = np.asarray((-4097, -2049, -2048, 0, 2047, 2048, 4097))
    assert np.array_equal(_signed_mod(values), np.asarray((-1, 2047, -2048, 0, 2047, -2048, 1)))
    calibration = np.asarray((0.01, 0.02, 0.03), dtype=np.float64)
    identity = calibrated_transform(np.zeros(6), calibration, translation=True)
    assert np.allclose(identity, np.eye(4), atol=1e-14)
    rng = np.random.default_rng(7)
    coefficients = rng.integers(-20, 21, size=(16, 12), dtype=np.int32)
    poses = rng.normal(size=(16, 6)).astype(np.float16)
    model = fit_linear_prior(coefficients, poses, calibration)
    screws = relative_screw_features(poses, calibration)
    xi_codes = np.rint(screws / np.asarray(model["scales"])).clip(-2047, 2047).astype(np.int16)
    predictions = learned_predictions(_coefficient_features(coefficients, xi_codes), model)
    restored = _signed_mod(predictions + _signed_mod(coefficients - predictions))
    assert np.array_equal(restored, coefficients)
    print("ddm_xi1 self-test: PASS")


def update_state(*, stages: dict[str, str], pins: dict[str, Any], preflight: dict[str, Any], complete: bool) -> None:
    payload = {
        "schema": "ddm_xi1_state.v1",
        "arm": "ddm_xi1",
        "complete": complete,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mission": "matched learned xi context rate screen plus exact CP135 pose-carrier prior race",
        "output_root": str(FIX_OUTPUT),
        "source_payload_root": str(OUTPUT),
        "payload_policy": "retain every materialized payload with bytes and sha256",
        "stages": stages,
        "input_pins": pins,
        "storage_preflight": preflight,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "hardware": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": sys.version,
            "torch": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
        },
    }
    atomic_json(STATE, payload, replace=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leg", choices=("both", "a", "b", "prepare-a"), default="both")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--required-free-bytes", type=int, default=4 << 30)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    assert_governed_admission("run_ddm_xi1_screw_conditioned_learned_prior")
    if args.epochs < 2 or args.epochs % 2:
        raise XI1Error("--epochs must be an even integer at least 2 for the continuous/QAT boundary")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise XI1Error("PYTHONHASHSEED=0 is required")
    if os.environ.get("TAC_ADMISSION_ENFORCE") != "1":
        raise XI1Error("TAC_ADMISSION_ENFORCE=1 is required")
    if args.leg in {"both", "a"} and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise XI1Error("PYTORCH_ENABLE_MPS_FALLBACK=0 is required for Leg A")
    preflight = storage_preflight(args.required_free_bytes)
    pins = pin_inputs()
    stages = {"leg_a": "pending", "leg_b": "pending", "receipt": "pending"}
    update_state(stages=stages, pins=pins, preflight=preflight, complete=False)
    retained_inputs = OUTPUT / "retained/inputs"
    _atomic_bytes(retained_inputs / "pose_targets_n600_f16.bin", POSE_RAW.read_bytes())
    _atomic_bytes(retained_inputs / "warp_calibration_f64.bin", CALIBRATION_RAW.read_bytes())
    _atomic_bytes(retained_inputs / "pose_warp.stp", POSE_CONTAINER.read_bytes())
    poses = np.frombuffer(POSE_RAW.read_bytes(), dtype="<f2").astype(np.float64).reshape(FRAME_COUNT, 6)
    calibration = np.frombuffer(CALIBRATION_RAW.read_bytes(), dtype="<f8").copy()
    if calibration.shape != (3,):
        raise XI1Error("calibration plane is not three float64 scalars")
    leg_b: dict[str, Any] | None = None
    leg_a: dict[str, Any] | None = None
    if args.leg in {"both", "b"}:
        leg_b = run_leg_b(poses, calibration)
        stages["leg_b"] = "complete"
        update_state(stages=stages, pins=pins, preflight=preflight, complete=False)
    contexts: dict[str, Any] | None = None
    ids: np.ndarray | None = None
    cache: torch.Tensor | None = None
    if args.leg in {"both", "a", "prepare-a"}:
        cache = torch.load(HPAC_CACHE, map_location="cpu", weights_only=False)["seg"].to(torch.uint8)
        if tuple(cache.shape) != (FRAME_COUNT, H, W):
            raise XI1Error("HPAC cache shape changed")
        ids = selected_frame_ids()
        atomic_npy(retained_inputs / "stratified_frame_ids.npy", ids)
        contexts = build_contexts(cache, ids, poses, calibration)
    if args.leg == "prepare-a":
        assert ids is not None and contexts is not None
        preparation = {
            "schema": "ddm_xi1_leg_a_preparation.v1",
            "status": "READY_BLOCKED_NO_METAL_DEVICE",
            "sample_frame_ids": ids.astype(int).tolist(),
            "context_records": contexts["records"],
            "context_repeat_exact": contexts["xi_repeat_exact"],
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "cpu_substitution_refused": True,
            "resume_command": (
                "PYTHONHASHSEED=0 TAC_ADMISSION_ENFORCE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 "
                ".venv/bin/python tools/safe_run.py --rss-mb 12288 --projected-gib 12 --timeout 2390 "
                "--label ddm_xi1f_leg_a_n120 --status-receipt "
                "/Volumes/APDataStore/pact/ddm_xi1_20260812/fix/leg_a.safe_run.json -- "
                ".venv/bin/python tools/run_ddm_xi1_screw_conditioned_learned_prior.py --leg a --epochs 20"
            ),
            "axis": AXIS_A,
            "score_claim": False,
        }
        atomic_json(LEG_A_PREPARATION, preparation, replace=True)
        fire_order = {
            "schema": "ddm_xi1_leg_a_mps_fire_order.v1",
            "disposition": "QUEUED-WITH-A-FIRE-ORDER",
            "owner": "MAIN Metal executor",
            "consumer_store": str(LEG_A_RESULT),
            "fire_trigger": "a governed process reports torch.backends.mps.is_available() == True; execute the pinned resume_command without CPU substitution",
            "preparation": file_record(LEG_A_PREPARATION),
        }
        atomic_json(LEG_A_FIRE_ORDER, fire_order, replace=True)
        row = (
            "ddm_xi1_leg_a_mps: QUEUED-WITH-A-FIRE-ORDER. Owner: MAIN Metal executor. Consumer store: "
            f"{LEG_A_RESULT}. Fire trigger: a governed process reports "
            "torch.backends.mps.is_available() == True; execute the pinned resume_command without CPU substitution."
        )
        existing_leg_b = None
        if (OUTPUT / "LEG_B_RESULT.json").is_file():
            existing_leg_b = json.loads((OUTPUT / "LEG_B_RESULT.json").read_text(encoding="utf-8"))
        dispositions = queue_dispositions(None, existing_leg_b)
        dispositions.append(row)
        atomic_json(FIX_OUTPUT / "QUEUE_DISPOSITIONS.json", {"rows": dispositions}, replace=True)
        _atomic_bytes(
            FIX_OUTPUT / "QUEUE_DISPOSITIONS.md",
            ("\n".join(f"- {item}" for item in dispositions) + "\n").encode(),
            replace=True,
        )
        partial_final = {
            "schema": SCHEMA,
            "status": "PARTIAL_BLOCKED_NO_METAL_DEVICE",
            "leg_a_preparation": file_record(LEG_A_PREPARATION),
            "leg_a_result": None,
            "leg_b": (
                file_record(OUTPUT / "LEG_B_RESULT.json")
                if (OUTPUT / "LEG_B_RESULT.json").is_file()
                else None
            ),
            "queue_dispositions": dispositions,
            "axis": [AXIS_A, AXIS_B],
            "score_claim": False,
            "frontier_moved": False,
        }
        atomic_json(FINAL_RESULT, partial_final, replace=True)
        stages["leg_a"] = "ready_blocked_no_metal_device"
        stages["leg_b"] = "complete" if (OUTPUT / "LEG_B_RESULT.json").is_file() else "pending"
        stages["receipt"] = "complete"
        update_state(stages=stages, pins=pins, preflight=preflight, complete=False)
        print(json.dumps(preparation, indent=2, sort_keys=True))
        return
    if args.leg in {"both", "a"}:
        if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            raise XI1Error("Leg A requires local MPS; CPU substitution is forbidden")
        assert cache is not None and ids is not None and contexts is not None
        leg_a = run_leg_a(cache=cache, ids=ids, contexts=contexts, epochs=args.epochs)
        stages["leg_a"] = "complete"
        update_state(stages=stages, pins=pins, preflight=preflight, complete=False)
    dispositions = queue_dispositions(leg_a, leg_b)
    final = {
        "schema": SCHEMA,
        "leg_a": None if leg_a is None else file_record(LEG_A_RESULT),
        "leg_b": None if leg_b is None else file_record(OUTPUT / "LEG_B_RESULT.json"),
        "queue_dispositions": dispositions,
        "axis": [AXIS_A, AXIS_B],
        "score_claim": False,
        "frontier_moved": False,
    }
    atomic_json(FINAL_RESULT, final, replace=True)
    stages["receipt"] = "complete"
    update_state(stages=stages, pins=pins, preflight=preflight, complete=True)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
