#!/usr/bin/env python3
"""Characterize and stage real DX2 receiver precompensation candidates.

The scorer-free ``characterize`` stage joins the retained MST1 realization
ladder to WJ1's complete gross-manufactured support.  It persists one row for
every one of the 28,602 gross native breaks and measures separation between
the 11,685 downstream-repaired and 16,917 terminal-persistent positions.

The scorer-free ``materialize`` stage creates four executable receiver
variants.  Every fitted value is appended to the single counted ZIP member;
the copied receiver contains only the generic parser and transform
algorithms.  The shipped DX2 runtime remains read-only.  Candidate archives,
repeat archives, extracted parameter payloads, and complete runtime trees are
retained under the charter-mandated local receipt root.

This execution unit does not run a scorer or launch an advisory.  The ``score``
consumer is staged for use only after the canonical local-advisory queue's
recorded fire trigger transfers the fleet's sole scorer slot to RX3.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
import shutil
import struct
import sys
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from scipy import ndimage, stats

sys.dont_write_bytecode = True

REPO: Final = Path(__file__).resolve().parents[1]
RECEIPT_ROOT: Final = (
    REPO / ".omx/tmp/arm_receipts_local/ddm_rx3_receiver_precompensation"
)
MST1_ROOT: Final = (
    REPO
    / ".omx/tmp/arm_receipts_local/ddm_mst1_manufactured_stage_split/capture_r2_local"
)
WJ1_ROOT: Final = (
    REPO
    / ".omx/tmp/arm_receipts_local/ddm_wj1_cost_error_position_join/measurement_v1"
)
SOURCE_RUNTIME: Final = Path(
    "/Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2"
)
SOURCE_ARCHIVE: Final = SOURCE_RUNTIME / "archive.zip"
GT_POSE: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/"
    "direct_v6/full_n600_eval/retained/pose_vectors/gt_first6_dali_n600.npy"
)

N: Final = 600
H: Final = 384
W: Final = 512
PLANE: Final = H * W
PACKED_FRAME_BYTES: Final = PLANE // 8
SEG_PIXELS: Final = N * PLANE
CLASS_NAMES: Final = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
CLASS_DENOMINATORS: Final = (27_407_372, 690_754, 58_413_067, 1_460_386, 29_993_221)
ARCHIVE_BYTES: Final = 180_368
ARCHIVE_SHA256: Final = (
    "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674"
)
TOKEN_BYTES: Final = 113_777
TOKEN_SHA256: Final = (
    "e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5"
)
TARGET_SHA256: Final = (
    "bb1c42698e38deb94d9bee8edbdf44261a40a95554defef38d6088730be5da7d"
)
GT_SEG_SHA256: Final = (
    "91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248"
)
GT_POSE_SHA256: Final = (
    "8d5cfa83df55b89493ba43b1e5386d792c836c32791666192499a089068e7eff"
)
UPSTREAM: Final = Path("/Volumes/APDataStore/pact/upstream_eval_mirror_20260815")
VIDEO_NAMES: Final = UPSTREAM / "public_test_video_names.txt"
AP1_CONTROL: Final = (
    REPO
    / ".omx/tmp/arm_receipts_local/ddm_ap1_residue_purchase_scorer/scorer/control/RESULT.json"
)
GROSS_MASK_SHA256: Final = (
    "b756ca948f5db3dd085a61803e24c5a90db946d89ea1894ba552f024a74b1d5d"
)
FINAL_MASK_SHA256: Final = (
    "89d09fbf1dc6a0bf8d1117287e2fbc5473e1a6c218e9975604c8fac94a9a3127"
)
RATE_S_PER_BYTE: Final = 25.0 / 37_545_489.0
MIN_FREE_BYTES: Final = 48 * (1 << 30)
FOOTER: Final = struct.Struct("<4sBB")
FOOTER_MAGIC: Final = b"RX3T"


class RX3Error(RuntimeError):
    """A custody, measurement, or executable-candidate invariant failed."""


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    algorithm_id: int
    algorithm: str
    fitted_parameter_count: int
    parameter_bytes: bytes
    rationale: str


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise RX3Error(f"required file is absent: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_fact(record: Mapping[str, Any]) -> Path:
    path = Path(str(record["path"]))
    actual = file_fact(path)
    if actual["bytes"] != record["bytes"] or actual["sha256"] != record["sha256"]:
        raise RX3Error(f"retained artifact drifted: {path}")
    return path


def atomic_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return file_fact(path)


def atomic_json(path: Path, value: object) -> dict[str, Any]:
    return atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(),
    )


def atomic_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    array = np.ascontiguousarray(value)
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return file_fact(path) | {"shape": list(array.shape), "dtype": array.dtype.str}


def tree_fact(root: Path) -> dict[str, Any]:
    rows = []
    total = 0
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if path.name.startswith("._") or "__pycache__" in relative.parts:
            continue
        fact = file_fact(path)
        row = {
            "relative_path": relative.as_posix(),
            "bytes": fact["bytes"],
            "sha256": fact["sha256"],
        }
        rows.append(row)
        total += int(fact["bytes"])
        digest.update((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())
    return {
        "path": str(root.resolve()),
        "file_count": len(rows),
        "bytes": total,
        "tree_sha256": digest.hexdigest(),
    }


def storage_preflight() -> dict[str, Any]:
    local = shutil.disk_usage(REPO)
    if local.free < MIN_FREE_BYTES:
        raise RX3Error(
            f"local receipt tier has {local.free} free bytes; RX3 requires {MIN_FREE_BYTES}"
        )
    tiers = []
    for path in (REPO, Path("/Volumes/VertigoDataTier/pact"), Path("/Volumes/APDataStore/pact")):
        usage = shutil.disk_usage(path)
        tiers.append(
            {
                "path": str(path),
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
            }
        )
    receipt = {
        "schema": "ddm_rx3_storage_preflight.v1",
        "created_utc": utc_now(),
        "selected_tier": str(RECEIPT_ROOT),
        "selection": "local_disk_explicit_opt_in_because_both_ssd_tiers_are_full",
        "minimum_free_bytes": MIN_FREE_BYTES,
        "tiers": tiers,
        "volume_writes": False,
        "cleanup": "none; all candidate and measurement bytes are retained evidence",
        "certify_or_block": True,
    }
    atomic_json(RECEIPT_ROOT / "STORAGE_PREFLIGHT.json", receipt)
    return receipt


def verify_pins() -> dict[str, Any]:
    archive = file_fact(SOURCE_ARCHIVE)
    if archive["bytes"] != ARCHIVE_BYTES or archive["sha256"] != ARCHIVE_SHA256:
        raise RX3Error("DX2 archive pin drifted")
    target = file_fact(
        WJ1_ROOT / "retained/targets/top_10pct_render_manufactured_positions.npy"
    )
    if target["sha256"] != TARGET_SHA256:
        raise RX3Error("WJ1 target-list pin drifted")
    gt_seg = file_fact(WJ1_ROOT / "retained/inputs/gt_argmax_n600.npy")
    if gt_seg["sha256"] != GT_SEG_SHA256:
        raise RX3Error("DALI Seg GT pin drifted")
    gt_pose = file_fact(GT_POSE)
    if gt_pose["sha256"] != GT_POSE_SHA256:
        raise RX3Error("DALI Pose GT pin drifted")
    gross = file_fact(
        WJ1_ROOT / "retained/inputs/gross_manufactured_native_render_head.n600.packbits"
    )
    final = file_fact(WJ1_ROOT / "retained/inputs/final_error_support.n600.packbits")
    if gross["sha256"] != GROSS_MASK_SHA256 or final["sha256"] != FINAL_MASK_SHA256:
        raise RX3Error("WJ1 membership-mask pin drifted")
    parts = _source_parts()
    token_sha = sha256_bytes(parts.token_stream)
    if len(parts.token_stream) != TOKEN_BYTES or token_sha != TOKEN_SHA256:
        raise RX3Error("RC64 token-stream pin drifted")
    return {
        "archive": archive,
        "token_stream": {"bytes": len(parts.token_stream), "sha256": token_sha},
        "wj1_target_list": target,
        "dali_seg_gt": gt_seg,
        "dali_pose_gt": gt_pose,
        "gross_mask": gross,
        "final_mask": final,
    }


def _source_parts() -> Any:
    runtime = SOURCE_RUNTIME.resolve()
    inserted = [str(runtime), str(runtime / "cpr1")]
    for value in inserted:
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    for name in list(sys.modules):
        if name == "runtime" or name.startswith("runtime."):
            del sys.modules[name]
    module = importlib.import_module("runtime.residual_archive")
    if Path(module.__file__).resolve().parent.parent != runtime:
        raise RX3Error("source residual parser resolved outside the DX2 runtime")
    return module.read_residual_archive(SOURCE_ARCHIVE)


def packed_frame(path: Path, frame: int) -> np.ndarray:
    packed = np.memmap(
        path,
        mode="r",
        dtype=np.uint8,
        offset=frame * PACKED_FRAME_BYTES,
        shape=(PACKED_FRAME_BYTES,),
    )
    return np.unpackbits(packed, bitorder="little", count=PLANE).reshape(H, W).astype(bool)


def _chunk_dir(frame: int) -> tuple[Path, int]:
    start = (frame // 16) * 16
    stop = min(start + 16, N)
    return MST1_ROOT / f"retained/chunks/{start:04d}_{stop - 1:04d}", frame - start


def _margin(logits: np.ndarray, gt: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    rows = logits[:, ys, xs].T.astype(np.float64, copy=False)
    true = rows[np.arange(rows.shape[0]), gt]
    rows = rows.copy()
    rows[np.arange(rows.shape[0]), gt] = -np.inf
    return true - rows.max(axis=1)


def _gt_boundary_distance(labels: np.ndarray) -> np.ndarray:
    edge = np.zeros((H, W), dtype=bool)
    edge[1:] |= labels[1:] != labels[:-1]
    edge[:-1] |= labels[:-1] != labels[1:]
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    edge[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return ndimage.distance_transform_edt(~edge).astype(np.float32)


def _local_features(
    labels: np.ndarray,
    native_rgb: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
) -> dict[str, np.ndarray]:
    count = ys.size
    same3 = np.empty(count, dtype=np.float32)
    same5 = np.empty(count, dtype=np.float32)
    unique5 = np.empty(count, dtype=np.uint8)
    luma = (
        0.299 * native_rgb[0].astype(np.float64)
        + 0.587 * native_rgb[1].astype(np.float64)
        + 0.114 * native_rgb[2].astype(np.float64)
    )
    grad_y = np.zeros_like(luma)
    grad_x = np.zeros_like(luma)
    grad_y[1:] = np.abs(luma[1:] - luma[:-1])
    grad_x[:, 1:] = np.abs(luma[:, 1:] - luma[:, :-1])
    gradient = np.maximum(grad_y, grad_x)
    blur = ndimage.uniform_filter(native_rgb.astype(np.float64), size=(1, 3, 3), mode="nearest")
    highpass = native_rgb.astype(np.float64) - blur
    for index, (y, x) in enumerate(zip(ys, xs, strict=True)):
        label = labels[y, x]
        y0, y1 = max(0, y - 1), min(H, y + 2)
        x0, x1 = max(0, x - 1), min(W, x + 2)
        y2, y3 = max(0, y - 2), min(H, y + 3)
        x2, x3 = max(0, x - 2), min(W, x + 3)
        same3[index] = np.mean(labels[y0:y1, x0:x1] == label)
        patch5 = labels[y2:y3, x2:x3]
        same5[index] = np.mean(patch5 == label)
        unique5[index] = np.unique(patch5).size
    return {
        "same_class_fraction_3x3": same3,
        "same_class_fraction_5x5": same5,
        "unique_classes_5x5": unique5,
        "native_luma_gradient": gradient[ys, xs].astype(np.float32),
        "native_highpass_r": highpass[0, ys, xs].astype(np.float32),
        "native_highpass_g": highpass[1, ys, xs].astype(np.float32),
        "native_highpass_b": highpass[2, ys, xs].astype(np.float32),
    }


FEATURE_DTYPE: Final = np.dtype(
    [
        ("flat_index", "<u8"),
        ("frame", "<u2"),
        ("y", "<u2"),
        ("x", "<u2"),
        ("repaired", "u1"),
        ("gt_class", "u1"),
        ("native_class", "u1"),
        ("boundary_distance", "<f4"),
        ("margin_native", "<f4"),
        ("margin_preuint8", "<f4"),
        ("margin_uint8", "<f4"),
        ("same_class_fraction_3x3", "<f4"),
        ("same_class_fraction_5x5", "<f4"),
        ("unique_classes_5x5", "u1"),
        ("native_luma_gradient", "<f4"),
        ("native_highpass_r", "<f4"),
        ("native_highpass_g", "<f4"),
        ("native_highpass_b", "<f4"),
        ("pre_residual_r", "<f4"),
        ("pre_residual_g", "<f4"),
        ("pre_residual_b", "<f4"),
        ("uint8_residual_r", "<f4"),
        ("uint8_residual_g", "<f4"),
        ("uint8_residual_b", "<f4"),
        ("total_residual_r", "<f4"),
        ("total_residual_g", "<f4"),
        ("total_residual_b", "<f4"),
        ("pre_residual_l2", "<f4"),
        ("uint8_residual_l2", "<f4"),
        ("total_residual_l2", "<f4"),
    ]
)


def _auc(value: np.ndarray, label: np.ndarray) -> float:
    ranks = stats.rankdata(value, method="average")
    positives = label.astype(bool)
    n1 = int(positives.sum())
    n0 = int((~positives).sum())
    return float((ranks[positives].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _continuous_summary(value: np.ndarray, label: np.ndarray) -> dict[str, Any]:
    repaired = value[label]
    persistent = value[~label]
    auc = _auc(value, label)
    pooled = math.sqrt(
        (
            (repaired.size - 1) * repaired.var(ddof=1)
            + (persistent.size - 1) * persistent.var(ddof=1)
        )
        / (repaired.size + persistent.size - 2)
    )
    ks = stats.ks_2samp(repaired, persistent, method="asymp")
    return {
        "denominator": int(value.size),
        "repaired_n": int(repaired.size),
        "persistent_n": int(persistent.size),
        "repaired_mean": float(repaired.mean()),
        "persistent_mean": float(persistent.mean()),
        "repaired_median": float(np.median(repaired)),
        "persistent_median": float(np.median(persistent)),
        "auc_repaired_high": auc,
        "auc_separation": max(auc, 1.0 - auc),
        "auc_direction": "repaired_high" if auc >= 0.5 else "repaired_low",
        "cohen_d_repaired_minus_persistent": (
            (float(repaired.mean()) - float(persistent.mean())) / pooled if pooled > 0 else 0.0
        ),
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "repaired_q10_q90": np.quantile(repaired, (0.1, 0.9)).tolist(),
        "persistent_q10_q90": np.quantile(persistent, (0.1, 0.9)).tolist(),
    }


def _categorical_summary(codes: Sequence[str], label: np.ndarray) -> dict[str, Any]:
    repaired = Counter(code for code, flag in zip(codes, label, strict=True) if flag)
    persistent = Counter(code for code, flag in zip(codes, label, strict=True) if not flag)
    keys = sorted(set(repaired) | set(persistent))
    n1 = int(label.sum())
    n0 = int((~label).sum())
    total_variation = 0.5 * sum(
        abs(repaired[key] / n1 - persistent[key] / n0) for key in keys
    )
    rows = []
    for key in keys:
        denominator = repaired[key] + persistent[key]
        rows.append(
            {
                "category": key,
                "denominator": denominator,
                "repaired": repaired[key],
                "persistent": persistent[key],
                "repair_rate": repaired[key] / denominator,
            }
        )
    rows.sort(key=lambda row: (-row["denominator"], row["category"]))
    return {
        "denominator": len(codes),
        "category_count": len(keys),
        "total_variation": total_variation,
        "rows": rows,
    }


def _choose_gradient_threshold(value: np.ndarray, label: np.ndarray) -> dict[str, Any]:
    candidates = np.unique(np.quantile(value, np.linspace(0.05, 0.95, 91)))
    best: tuple[float, float, float, float] | None = None
    for threshold in candidates:
        selected = value >= threshold
        tpr = float(selected[label].mean())
        fpr = float(selected[~label].mean())
        row = (tpr - fpr, float(threshold), tpr, fpr)
        if best is None or row > best:
            best = row
    assert best is not None
    return {
        "threshold": best[1],
        "youden_j": best[0],
        "repaired_selected_fraction": best[2],
        "persistent_selected_fraction": best[3],
        "denominator": int(value.size),
    }


def characterize() -> dict[str, Any]:
    RECEIPT_ROOT.mkdir(parents=True, exist_ok=True)
    storage = storage_preflight()
    pins = verify_pins()
    result_path = RECEIPT_ROOT / "discriminator/RESULT.json"
    rows_path = RECEIPT_ROOT / "discriminator/retained/gross_positions_features.npy"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        validate_fact(result["features"])
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    gross_path = WJ1_ROOT / "retained/inputs/gross_manufactured_native_render_head.n600.packbits"
    final_path = WJ1_ROOT / "retained/inputs/final_error_support.n600.packbits"
    gt_all = np.load(WJ1_ROOT / "retained/inputs/gt_argmax_n600.npy", mmap_mode="r")
    if gt_all.shape != (N, H, W) or gt_all.dtype != np.uint8:
        raise RX3Error("DALI Seg GT geometry drifted")
    output = np.empty(28_602, dtype=FEATURE_DTYPE)
    cursor = 0
    for frame in range(N):
        gross = packed_frame(gross_path, frame)
        final = packed_frame(final_path, frame)
        ys, xs = np.nonzero(gross)
        count = ys.size
        if not count:
            continue
        repaired = ~final[ys, xs]
        gt = np.asarray(gt_all[frame])
        chunk, local = _chunk_dir(frame)
        native = np.load(chunk / "native_rgb.float32.npy", mmap_mode="r")[local]
        pre = np.load(chunk / "evaluator_resized_preuint8_rgb.float32.npy", mmap_mode="r")[local]
        uint8 = np.load(chunk / "evaluator_resized_uint8_rgb.float32.npy", mmap_mode="r")[local]
        logits_native = np.load(chunk / "logits_native.float32.npy", mmap_mode="r")[local]
        logits_pre = np.load(chunk / "logits_preuint8.float32.npy", mmap_mode="r")[local]
        logits_uint8 = np.load(chunk / "logits_uint8.float32.npy", mmap_mode="r")[local]
        native_class = logits_native.argmax(axis=0)[ys, xs].astype(np.uint8)
        if np.any(native_class == gt[ys, xs]):
            raise RX3Error("gross native break contains a native-correct position")
        pre_residual = (pre[:, ys, xs] - native[:, ys, xs]).T
        uint8_residual = (uint8[:, ys, xs] - pre[:, ys, xs]).T
        total_residual = (uint8[:, ys, xs] - native[:, ys, xs]).T
        local_features = _local_features(gt, np.asarray(native), ys, xs)
        destination = output[cursor : cursor + count]
        destination["flat_index"] = frame * PLANE + ys * W + xs
        destination["frame"] = frame
        destination["y"] = ys
        destination["x"] = xs
        destination["repaired"] = repaired
        destination["gt_class"] = gt[ys, xs]
        destination["native_class"] = native_class
        destination["boundary_distance"] = _gt_boundary_distance(gt)[ys, xs]
        destination["margin_native"] = _margin(logits_native, gt[ys, xs], ys, xs)
        destination["margin_preuint8"] = _margin(logits_pre, gt[ys, xs], ys, xs)
        destination["margin_uint8"] = _margin(logits_uint8, gt[ys, xs], ys, xs)
        for name, value in local_features.items():
            destination[name] = value
        for prefix, value in (
            ("pre_residual", pre_residual),
            ("uint8_residual", uint8_residual),
            ("total_residual", total_residual),
        ):
            for channel, suffix in enumerate(("r", "g", "b")):
                destination[f"{prefix}_{suffix}"] = value[:, channel]
            destination[f"{prefix}_l2"] = np.linalg.norm(value, axis=1)
        cursor += count
        if (frame + 1) % 50 == 0 or frame + 1 == N:
            print(f"characterize {frame + 1}/{N}: {cursor} positions", flush=True)
    if cursor != output.size:
        raise RX3Error(f"gross support count {cursor} != 28602")
    label = output["repaired"].astype(bool)
    if int(label.sum()) != 11_685 or int((~label).sum()) != 16_917:
        raise RX3Error("repaired/persistent split differs from WJ1")
    if np.unique(output["flat_index"]).size != output.size:
        raise RX3Error("gross-position feature table contains duplicate coordinates")
    feature_record = atomic_npy(rows_path, output)
    continuous_names = [
        name
        for name in output.dtype.names or ()
        if output.dtype[name].kind == "f" and name not in {"flat_index"}
    ]
    summaries = {
        name: _continuous_summary(output[name].astype(np.float64), label)
        for name in continuous_names
    }
    class_pairs = [
        f"{int(gt)}->{int(native)}"
        for gt, native in zip(output["gt_class"], output["native_class"], strict=True)
    ]
    residual_signs = [
        "".join("+" if value > 0 else "-" if value < 0 else "0" for value in row)
        for row in np.column_stack(
            [output["total_residual_r"], output["total_residual_g"], output["total_residual_b"]]
        )
    ]
    strongest = sorted(
        (
            {"feature": name, **summary}
            for name, summary in summaries.items()
        ),
        key=lambda row: (-row["auc_separation"], row["feature"]),
    )
    result = {
        "schema": "ddm_rx3_repair_discriminator.v1",
        "created_utc": utc_now(),
        "status": "COMPLETE",
        "axis": "[scorer-free read of retained MST1 macOS-CPU fields joined to contest-CUDA DALI-GT membership]",
        "score_claim": False,
        "pairs": N,
        "gross_positions": int(output.size),
        "repaired_positions": int(label.sum()),
        "terminal_persistent_positions": int((~label).sum()),
        "membership_definition": (
            "gross native-render wrong transition AND absent/present respectively in the "
            "retained final-error support"
        ),
        "pins": pins,
        "features": feature_record,
        "continuous": summaries,
        "strongest_continuous": strongest[:12],
        "categorical": {
            "class_pair_gt_to_native": _categorical_summary(class_pairs, label),
            "total_residual_channel_sign": _categorical_summary(residual_signs, label),
        },
        "gradient_threshold_proxy": _choose_gradient_threshold(
            output["native_luma_gradient"].astype(np.float64), label
        ),
        "per_gt_class": [
            {
                "class_id": class_id,
                "class_name": class_name,
                "denominator": int(np.sum(output["gt_class"] == class_id)),
                "repaired": int(np.sum((output["gt_class"] == class_id) & label)),
                "persistent": int(np.sum((output["gt_class"] == class_id) & ~label)),
            }
            for class_id, class_name in enumerate(CLASS_NAMES)
        ],
        "storage": storage,
        "retention": "complete 28,602-row feature payload retained locally with hash and bytes",
    }
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


RX3_HELPER_SOURCE: Final = r'''"""Generic counted-parameter RX3 receiver precompensation."""
from __future__ import annotations
import struct
import zipfile
from pathlib import Path
import numpy as np
from torch.nn import functional

FOOTER = struct.Struct("<4sBB")
MAGIC = b"RX3T"

def split_outer(outer: bytes):
    if len(outer) < FOOTER.size:
        raise ValueError("RX3 payload footer is absent")
    magic, algorithm_id, parameter_bytes = FOOTER.unpack_from(outer, len(outer) - FOOTER.size)
    if magic != MAGIC or parameter_bytes == 0:
        raise ValueError("RX3 payload footer is malformed")
    start = len(outer) - FOOTER.size - parameter_bytes
    if start <= 0:
        raise ValueError("RX3 parameter section overlaps the base payload")
    params = outer[start : start + parameter_bytes]
    return outer[:start], {"algorithm_id": int(algorithm_id), "parameters": params}

def load_candidate_params(archive_path: Path):
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != ["p"]:
            raise ValueError("RX3 archive must contain exactly member p")
        _, spec = split_outer(archive.read("p"))
    return spec

def _offsets(parameters: bytes, tensor):
    values = np.frombuffer(parameters[:6], dtype=np.int8).astype(np.float32) * 0.25
    return tensor.new_tensor(values.reshape(2, 3))

def apply_precomp(tensor, frame_role: int, spec):
    algorithm_id = spec["algorithm_id"]
    parameters = spec["parameters"]
    if frame_role not in (0, 1):
        raise ValueError("frame role must be 0 or 1")
    if algorithm_id == 1:
        if len(parameters) != 6:
            raise ValueError("RX3 global offset parameter size differs")
        return tensor + _offsets(parameters, tensor)[frame_role][None, :, None, None]
    if algorithm_id == 2:
        if len(parameters) != 6:
            raise ValueError("RX3 local highpass parameter size differs")
        gains = tensor.new_tensor(
            np.frombuffer(parameters, dtype=np.int8).astype(np.float32).reshape(2, 3) / 64.0
        )
        local_mean = functional.avg_pool2d(tensor, 3, stride=1, padding=1)
        return tensor + gains[frame_role][None, :, None, None] * (tensor - local_mean)
    if algorithm_id == 3:
        if len(parameters) != 7:
            raise ValueError("RX3 gradient-band parameter size differs")
        offsets = _offsets(parameters, tensor)[frame_role][None, :, None, None]
        threshold = float(parameters[6])
        luma = 0.299 * tensor[:, 0:1] + 0.587 * tensor[:, 1:2] + 0.114 * tensor[:, 2:3]
        gy = functional.pad((luma[:, :, 1:] - luma[:, :, :-1]).abs(), (0, 0, 1, 0))
        gx = functional.pad((luma[:, :, :, 1:] - luma[:, :, :, :-1]).abs(), (1, 0, 0, 0))
        band = (gy.maximum(gx) >= threshold).to(dtype=tensor.dtype)
        return tensor + offsets * band
    raise ValueError(f"unsupported RX3 algorithm id: {algorithm_id}")
'''


def _single_member(path: Path) -> tuple[bytes, zipfile.ZipInfo]:
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["p"]:
            raise RX3Error("source archive is not a canonical single-member ZIP")
        return archive.read("p"), archive.getinfo("p")


def _emit_zip(payload: bytes, inherited: zipfile.ZipInfo) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        info = zipfile.ZipInfo("p", date_time=inherited.date_time)
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = inherited.external_attr
        info.create_system = inherited.create_system
        archive.writestr(info, payload)
    return buffer.getvalue()


def _pack_candidate(base: bytes, spec: CandidateSpec) -> bytes:
    if len(spec.parameter_bytes) > 255:
        raise RX3Error("RX3 parameter section exceeds one-byte length field")
    return (
        base
        + spec.parameter_bytes
        + FOOTER.pack(FOOTER_MAGIC, spec.algorithm_id, len(spec.parameter_bytes))
    )


def _copy_runtime(destination: Path) -> None:
    if destination.exists():
        raise RX3Error(f"refusing to overwrite an existing retained runtime: {destination}")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    if temporary.exists():
        raise RX3Error(f"interrupted runtime copy is retained and needs adjudication: {temporary}")
    shutil.copytree(
        SOURCE_RUNTIME,
        temporary,
        ignore=shutil.ignore_patterns("__pycache__", "._*", ".DS_Store"),
    )
    os.replace(temporary, destination)


def _patch_once(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    if source.count(old) != 1:
        raise RX3Error(f"patch anchor count differs for {path}: {source.count(old)}")
    atomic_bytes(path, source.replace(old, new, 1).encode())


def _patch_runtime(runtime: Path, archive_record: Mapping[str, Any], base_outer_sha: str) -> None:
    atomic_bytes(runtime / "runtime/rx3_precomp.py", RX3_HELPER_SOURCE.encode())
    residual = runtime / "runtime/residual_archive.py"
    _patch_once(
        residual,
        '        outer = archive.read("p")\n    rx1 = _decode_rx1_models(outer)\n',
        '        outer = archive.read("p")\n    from .rx3_precomp import split_outer\n    outer, _rx3_spec = split_outer(outer)\n    rx1 = _decode_rx1_models(outer)\n',
    )
    renderer = runtime / "cpr1/inflate.py"
    _patch_once(
        renderer,
        "from torch.nn import functional\n",
        "from torch.nn import functional\nfrom runtime.rx3_precomp import apply_precomp, load_candidate_params\n",
    )
    _patch_once(
        renderer,
        "    started = time.time()\n    semantic_batch = 8 if device.type == \"cuda\" else 1\n",
        "    started = time.time()\n    rx3_spec = load_candidate_params(Path(__file__).resolve().parents[1] / \"archive.zip\")\n    semantic_batch = 8 if device.type == \"cuda\" else 1\n",
    )
    _patch_once(
        renderer,
        '''        master = (
            functional.interpolate(
                semantic(tokens[start:end].long().to(device), indices),
                size=(CAMERA_H, CAMERA_W),
                mode="bilinear",
                align_corners=False,
            )
            .clamp(0.0, 255.0)
            .round()
        )
''',
        '''        master = functional.interpolate(
            semantic(tokens[start:end].long().to(device), indices),
            size=(CAMERA_H, CAMERA_W),
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 255.0)
''',
    )
    _patch_once(
        renderer,
        "        master_np = master.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()\n",
        "        master = apply_precomp(master, 1, rx3_spec).clamp(0.0, 255.0).round()\n        master_np = master.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()\n",
    )
    _patch_once(
        renderer,
        '''        slave = (
            functional.interpolate(
                (127.5 + CARRIER_AMPLITUDE * carrier).clamp(0.0, 255.0).round(),
                size=(CAMERA_H, CAMERA_W),
                mode="bicubic",
                align_corners=False,
            )
            .clamp(0.0, 255.0)
            .round()
        )
''',
        '''        slave = functional.interpolate(
            (127.5 + CARRIER_AMPLITUDE * carrier).clamp(0.0, 255.0).round(),
            size=(CAMERA_H, CAMERA_W),
            mode="bicubic",
            align_corners=False,
        ).clamp(0.0, 255.0)
''',
    )
    _patch_once(
        renderer,
        "        slave_np = slave.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()\n",
        "        slave = apply_precomp(slave, 0, rx3_spec).clamp(0.0, 255.0).round()\n        slave_np = slave.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()\n",
    )
    top = runtime / "inflate.py"
    source = top.read_text(encoding="utf-8")
    source = source.replace(
        "from runtime.f26_inflate import inflate_archive\n",
        "from runtime.f26_inflate import inflate_archive\nfrom runtime.rx3_precomp import split_outer\n",
        1,
    )
    old_verify = '''    if _sha256(archive_path) != ARCHIVE_SHA256:
        raise ValueError("archive.zip does not match the promoted F26 artifact")
    if archive_path.stat().st_size != ARCHIVE_BYTES:
        raise ValueError("archive.zip has an unexpected size")
    payload_path = data_dir / "p"
'''
    new_verify = f'''    payload_path = data_dir / "p"
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != ["p"]:
            raise ValueError("RX3 archive must contain exactly member p")
        archived_payload = archive.read("p")
    base_outer, _rx3_spec = split_outer(archived_payload)
    if hashlib.sha256(base_outer).hexdigest() != "{base_outer_sha}":
        raise ValueError("RX3 archive does not retain the exact DX2 base payload")
    if archive_path.stat().st_size != {int(archive_record['bytes'])}:
        raise ValueError("RX3 archive has an unexpected size")
'''
    if source.count(old_verify) != 1:
        raise RX3Error("top-level archive verifier patch anchor differs")
    source = source.replace(old_verify, new_verify, 1)
    source = source.replace(
        '    with zipfile.ZipFile(archive_path) as archive:\n        if archive.namelist() != ["p"]:\n            raise ValueError("archive.zip must contain exactly the payload file p")\n        if payload_path.read_bytes() != archive.read("p"):\n            raise ValueError("extracted payload does not match archive.zip")\n',
        '    if payload_path.read_bytes() != archived_payload:\n        raise ValueError("extracted payload does not match RX3 archive.zip")\n',
        1,
    )
    atomic_bytes(top, source.encode())


def _fit_specs(features: np.ndarray, discriminator: Mapping[str, Any]) -> tuple[CandidateSpec, ...]:
    repaired = features["repaired"].astype(bool)
    residual = np.column_stack(
        [
            features["total_residual_r"],
            features["total_residual_g"],
            features["total_residual_b"],
        ]
    ).astype(np.float64)
    desired = np.clip(residual[repaired].mean(axis=0), -2.0, 2.0)
    quantized_offsets = np.rint(desired / 0.25).astype(np.int8)
    offsets = np.concatenate([np.zeros(3, dtype=np.int8), quantized_offsets]).tobytes()
    highpass = np.column_stack(
        [
            features["native_highpass_r"],
            features["native_highpass_g"],
            features["native_highpass_b"],
        ]
    ).astype(np.float64)
    gains = np.zeros(3, dtype=np.float64)
    for channel in range(3):
        x = highpass[repaired, channel]
        y = residual[repaired, channel]
        denominator = float(np.dot(x, x))
        gains[channel] = float(np.dot(x, y) / denominator) if denominator > 0 else 0.0
    quantized_gains = np.rint(np.clip(gains, -0.5, 0.5) * 64.0).astype(np.int8)
    local_params = np.concatenate([np.zeros(3, dtype=np.int8), quantized_gains]).tobytes()
    threshold = int(
        np.clip(round(float(discriminator["gradient_threshold_proxy"]["threshold"])), 0, 255)
    )
    band_params = offsets + bytes([threshold])
    return (
        CandidateSpec(
            "l28_exact_counted",
            1,
            "global_per_frame_per_channel_offset",
            6,
            np.asarray([-4, 0, -4, 0, -4, 0], dtype=np.int8).tobytes(),
            "direct PR98/L28 mechanism analogue; ancestor constants are counted, not hidden",
        ),
        CandidateSpec(
            "global_repair_mean",
            1,
            "global_per_frame_per_channel_offset",
            6,
            offsets,
            "frame-1 offsets fitted to the mean observed downstream repair residual",
        ),
        CandidateSpec(
            "local_highpass_regression",
            2,
            "local_3x3_highpass",
            6,
            local_params,
            "frame-1 channel gains fit by repaired-position residual regression",
        ),
        CandidateSpec(
            "gradient_band_repair_mean",
            3,
            "gradient_band_scoped_offset",
            7,
            band_params,
            "repair-mean offsets restricted by the measured best observable gradient threshold",
        ),
    )


def materialize() -> dict[str, Any]:
    RECEIPT_ROOT.mkdir(parents=True, exist_ok=True)
    storage = storage_preflight()
    pins = verify_pins()
    discriminator_path = RECEIPT_ROOT / "discriminator/RESULT.json"
    if not discriminator_path.is_file():
        raise RX3Error("run characterize before materialize")
    discriminator = json.loads(discriminator_path.read_text(encoding="utf-8"))
    features = np.load(validate_fact(discriminator["features"]), allow_pickle=False)
    specs = _fit_specs(features, discriminator)
    base_outer, inherited = _single_member(SOURCE_ARCHIVE)
    base_outer_sha = sha256_bytes(base_outer)
    results = []
    for spec in specs:
        root = RECEIPT_ROOT / "retained/candidates" / spec.candidate_id
        result_path = root / "MATERIALIZE_RESULT.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            for key in ("archive", "archive_repeat", "parameter_payload"):
                validate_fact(result[key])
            results.append(result)
            continue
        member = _pack_candidate(base_outer, spec)
        archive_bytes = _emit_zip(member, inherited)
        repeat_bytes = _emit_zip(member, inherited)
        if archive_bytes != repeat_bytes:
            raise RX3Error(f"archive repeat differs for {spec.candidate_id}")
        archive = atomic_bytes(root / "archive.zip", archive_bytes)
        repeat = atomic_bytes(root / "archive.repeat.zip", repeat_bytes)
        parameter_payload = atomic_bytes(root / "retained/parameters.bin", spec.parameter_bytes)
        runtime = root / "runtime"
        _copy_runtime(runtime)
        runtime_archive = atomic_bytes(runtime / "archive.zip", archive_bytes)
        _patch_runtime(runtime, archive, base_outer_sha)
        with zipfile.ZipFile(runtime / "archive.zip") as parsed:
            parsed_base, parsed_spec = _split_candidate_outer(parsed.read("p"))
        if parsed_base != base_outer or parsed_spec["algorithm_id"] != spec.algorithm_id:
            raise RX3Error(f"candidate parse-back differs for {spec.candidate_id}")
        if parsed_spec["parameters"] != spec.parameter_bytes:
            raise RX3Error(f"candidate parameter parse-back differs for {spec.candidate_id}")
        runtime_archive = file_fact(runtime / "archive.zip")
        if (
            runtime_archive["bytes"] != archive["bytes"]
            or runtime_archive["sha256"] != archive["sha256"]
        ):
            raise RX3Error(f"runtime archive copy differs for {spec.candidate_id}")
        result = {
            "schema": "ddm_rx3_materialized_candidate.v1",
            "created_utc": utc_now(),
            "status": "COMPLETE",
            "candidate_id": spec.candidate_id,
            "algorithm_id": spec.algorithm_id,
            "algorithm": spec.algorithm,
            "rationale": spec.rationale,
            "fitted_parameter_count": spec.fitted_parameter_count,
            "parameter_section_bytes": len(spec.parameter_bytes),
            "archive_control_bytes": FOOTER.size,
            "real_archive_delta_bytes": int(archive["bytes"]) - ARCHIVE_BYTES,
            "delta_s_rate": (int(archive["bytes"]) - ARCHIVE_BYTES) * RATE_S_PER_BYTE,
            "source_archive": pins["archive"],
            "archive": archive,
            "runtime_archive": runtime_archive,
            "archive_repeat": repeat,
            "parameter_payload": parameter_payload,
            "parameter_payload_hex": spec.parameter_bytes.hex(),
            "base_outer_sha256": base_outer_sha,
            "base_outer_byte_identical": True,
            "token_stream": pins["token_stream"],
            "render_tokens_model_byte_identical": True,
            "receiver_parse_back": True,
            "candidate_runtime_tree": tree_fact(runtime),
            "source_runtime_tree": tree_fact(SOURCE_RUNTIME),
            "insertion_point": (
                "camera-resolution float RGB after the fixed native-to-camera lift and before "
                "clamp/round/uint8; evaluator resize and frozen scorers remain downstream"
            ),
            "storage": "local_disk_explicit_opt_in; no /Volumes writes",
            "scorer_status": "NOT_RUN; fleet n600 lane held by ddm_ap1_residue_purchase_scorer",
            "shipping_candidate": False,
            "retention": "archive + deterministic repeat + parameter bytes + complete runtime tree",
        }
        atomic_json(result_path, result)
        results.append(result)
        print(
            f"materialized {spec.candidate_id}: +{result['real_archive_delta_bytes']} B",
            flush=True,
        )
    manifest = {
        "schema": "ddm_rx3_materialize_all.v1",
        "created_utc": utc_now(),
        "status": "COMPLETE",
        "candidate_count": len(results),
        "candidates": [
            {
                "candidate_id": row["candidate_id"],
                "runtime_dir": str(
                    RECEIPT_ROOT / "retained/candidates" / row["candidate_id"] / "runtime"
                ),
                "archive": row["archive"],
                "fitted_parameter_count": row["fitted_parameter_count"],
                "real_archive_delta_bytes": row["real_archive_delta_bytes"],
                "delta_s_rate": row["delta_s_rate"],
            }
            for row in results
        ],
        "storage": storage,
        "next_stage": {
            "disposition": "QUEUED-WITH-A-FIRE-ORDER",
            "owner": "ddm_rx3_receiver_precompensation",
            "consumer_store": str(RECEIPT_ROOT / "advisory_and_dali_scorer"),
            "fire_trigger": (
                "ddm_ap1 releases the exclusive full-n600 scorer lane and RX3 claims it; then "
                "run each candidate sequentially through tools/fire_local_advisory.py with "
                "F26_TOKEN_DECODER left at python, retain raw, and post-score against the pinned "
                "DALI Seg/Pose tables in chunks <=120"
            ),
        },
    }
    atomic_json(RECEIPT_ROOT / "MATERIALIZE_ALL.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def rebuild_runtimes() -> dict[str, Any]:
    """Preserve the pre-fix runtimes and rebuild at the exact float insertion point."""
    materialized_path = RECEIPT_ROOT / "MATERIALIZE_ALL.json"
    if not materialized_path.is_file():
        raise RX3Error("materialize stage is incomplete")
    materialized = json.loads(materialized_path.read_text(encoding="utf-8"))
    base_outer, _ = _single_member(SOURCE_ARCHIVE)
    base_outer_sha = sha256_bytes(base_outer)
    rows = []
    for item in materialized["candidates"]:
        candidate_id = item["candidate_id"]
        root = RECEIPT_ROOT / "retained/candidates" / candidate_id
        result_path = root / "MATERIALIZE_RESULT.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        runtime = root / "runtime"
        superseded = root / "retained/superseded_runtime_postround_v1"
        if superseded.exists():
            if not runtime.exists():
                raise RX3Error(f"preserved predecessor exists but live runtime is absent: {candidate_id}")
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "runtime": tree_fact(runtime),
                    "superseded": tree_fact(superseded),
                    "resumed": True,
                }
            )
            continue
        archive_bytes = validate_fact(result["archive"]).read_bytes()
        primary_archive = atomic_bytes(root / "archive.zip", archive_bytes)
        superseded.parent.mkdir(parents=True, exist_ok=True)
        os.replace(runtime, superseded)
        _copy_runtime(runtime)
        runtime_archive = atomic_bytes(runtime / "archive.zip", archive_bytes)
        _patch_runtime(runtime, primary_archive, base_outer_sha)
        runtime_fact = tree_fact(runtime)
        superseded_fact = tree_fact(superseded)
        result["archive"] = primary_archive
        result["runtime_archive"] = runtime_archive
        result["candidate_runtime_tree"] = runtime_fact
        result["superseded_runtime_postround_v1"] = superseded_fact
        result["insertion_point"] = (
            "camera-resolution float RGB after the fixed native-to-camera lift and before "
            "the sole final clamp/round/uint8"
        )
        atomic_json(result_path, result)
        rows.append(
            {
                "candidate_id": candidate_id,
                "runtime": runtime_fact,
                "superseded": superseded_fact,
                "resumed": False,
            }
        )
    for item in materialized["candidates"]:
        result = json.loads(
            (
                RECEIPT_ROOT
                / "retained/candidates"
                / item["candidate_id"]
                / "MATERIALIZE_RESULT.json"
            ).read_text(encoding="utf-8")
        )
        item["archive"] = result["archive"]
    atomic_json(materialized_path, materialized)
    receipt = {
        "schema": "ddm_rx3_runtime_rebuild.v1",
        "created_utc": utc_now(),
        "status": "COMPLETE",
        "reason": "generation 1 applied RX3 after an inherited pre-uint8 round; preserved and superseded by true camera-float-before-round insertion",
        "rows": rows,
        "payload_deletion": False,
    }
    atomic_json(RECEIPT_ROOT / "RUNTIME_REBUILD.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def verify_runtimes() -> dict[str, Any]:
    materialized = json.loads((RECEIPT_ROOT / "MATERIALIZE_ALL.json").read_text(encoding="utf-8"))
    rows = []
    for item in materialized["candidates"]:
        candidate_id = item["candidate_id"]
        root = RECEIPT_ROOT / "retained/candidates" / candidate_id
        runtime = root / "runtime"
        result = json.loads((root / "MATERIALIZE_RESULT.json").read_text(encoding="utf-8"))
        for name in list(sys.modules):
            if (
                name == "runtime"
                or name.startswith("runtime.")
                or name
                in {
                    "inflate",
                    "carrier_codec",
                    "hpac_integer",
                    "hpac_integer_sparse",
                    "integer_model_io",
                    "ddm_mp2_semantic_receiver",
                }
            ):
                del sys.modules[name]
        inserted = (str(runtime), str(runtime / "cpr1"))
        sys.path[:0] = list(inserted)
        try:
            residual_archive = importlib.import_module("runtime.residual_archive")
            parts = residual_archive.read_residual_archive(runtime / "archive.zip")
            helper = importlib.import_module("runtime.rx3_precomp")
            parsed_spec = helper.load_candidate_params(runtime / "archive.zip")
            extracted = RECEIPT_ROOT / "verification" / candidate_id / "extracted"
            extracted.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(runtime / "archive.zip") as archive:
                payload = archive.read("p")
            extracted_payload = extracted / "p"
            if extracted_payload.exists() and extracted_payload.read_bytes() != payload:
                raise RX3Error(f"persisted verification extraction drifted: {candidate_id}")
            if not extracted_payload.exists():
                atomic_bytes(extracted_payload, payload)
            module_spec = importlib.util.spec_from_file_location(
                f"ddm_rx3_top_{candidate_id}", runtime / "inflate.py"
            )
            if module_spec is None or module_spec.loader is None:
                raise RX3Error(f"cannot load candidate top-level verifier: {candidate_id}")
            top = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(top)
            top._verify_input(extracted, runtime / "archive.zip")
        finally:
            sys.path[:] = [value for value in sys.path if value not in inserted]
        token_sha = sha256_bytes(parts.token_stream)
        if len(parts.token_stream) != TOKEN_BYTES or token_sha != TOKEN_SHA256:
            raise RX3Error(f"candidate token stream drifted: {candidate_id}")
        parameter_bytes = parsed_spec["parameters"]
        if parameter_bytes.hex() != result["parameter_payload_hex"]:
            raise RX3Error(f"candidate parameter parse-back drifted: {candidate_id}")
        source_text = (runtime / "cpr1/inflate.py").read_text(encoding="utf-8")
        master_block = source_text[source_text.index("        master = functional.interpolate") : source_text.index("        for offset in range(end - start):")]
        expected_master = (
            "        ).clamp(0.0, 255.0)\n"
            "        master = apply_precomp(master, 1, rx3_spec).clamp(0.0, 255.0).round()\n"
        )
        if master_block.count(".round()") != 1 or expected_master not in master_block:
            raise RX3Error(f"candidate master insertion order drifted: {candidate_id}")
        slave_block = source_text[
            source_text.index("        slave = functional.interpolate") : source_text.index(
                "        slave_np = slave.to(torch.uint8)"
            )
        ]
        expected_slave = (
            "        ).clamp(0.0, 255.0)\n"
            "        slave = apply_precomp(slave, 0, rx3_spec).clamp(0.0, 255.0).round()\n"
        )
        if slave_block.count(".round()") != 2 or expected_slave not in slave_block:
            raise RX3Error(f"candidate slave insertion order drifted: {candidate_id}")
        helper_text = (runtime / "runtime/rx3_precomp.py").read_text(encoding="utf-8")
        if result["parameter_payload_hex"] in helper_text:
            raise RX3Error(f"candidate parameters leaked into receiver source: {candidate_id}")
        rows.append(
            {
                "candidate_id": candidate_id,
                "archive": file_fact(runtime / "archive.zip"),
                "runtime_tree": tree_fact(runtime),
                "token_bytes": len(parts.token_stream),
                "token_sha256": token_sha,
                "algorithm_id": parsed_spec["algorithm_id"],
                "parameter_bytes": len(parameter_bytes),
                "top_level_extraction_verify": True,
                "camera_float_before_round_verify": True,
                "parameters_absent_from_receiver_source": True,
            }
        )
    receipt = {
        "schema": "ddm_rx3_runtime_verification.v3",
        "created_utc": utc_now(),
        "status": "PASS",
        "rows": rows,
    }
    atomic_json(RECEIPT_ROOT / "RUNTIME_VERIFICATION.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def fire_order() -> dict[str, Any]:
    materialized_path = RECEIPT_ROOT / "MATERIALIZE_ALL.json"
    discriminator_path = RECEIPT_ROOT / "discriminator/RESULT.json"
    if not materialized_path.is_file() or not discriminator_path.is_file():
        raise RX3Error("characterize and materialize must complete before fire-order")
    materialized = json.loads(materialized_path.read_text(encoding="utf-8"))
    discriminator = json.loads(discriminator_path.read_text(encoding="utf-8"))
    features = np.load(validate_fact(discriminator["features"]), allow_pickle=False)
    repaired = features["repaired"].astype(bool)
    target = np.column_stack(
        [
            features["total_residual_r"],
            features["total_residual_g"],
            features["total_residual_b"],
        ]
    ).astype(np.float64)
    highpass = np.column_stack(
        [
            features["native_highpass_r"],
            features["native_highpass_g"],
            features["native_highpass_b"],
        ]
    ).astype(np.float64)
    gradient = features["native_luma_gradient"].astype(np.float64)
    rows = []
    for item in materialized["candidates"]:
        candidate_id = item["candidate_id"]
        result = json.loads(
            (
                RECEIPT_ROOT
                / "retained/candidates"
                / candidate_id
                / "MATERIALIZE_RESULT.json"
            ).read_text(encoding="utf-8")
        )
        parameters = bytes.fromhex(result["parameter_payload_hex"])
        if result["algorithm_id"] == 1:
            prediction = np.broadcast_to(
                np.frombuffer(parameters, dtype=np.int8).astype(np.float64).reshape(2, 3)[1]
                * 0.25,
                target.shape,
            )
        elif result["algorithm_id"] == 2:
            gains = (
                np.frombuffer(parameters, dtype=np.int8).astype(np.float64).reshape(2, 3)[1]
                / 64.0
            )
            prediction = highpass * gains[None]
        elif result["algorithm_id"] == 3:
            offsets = (
                np.frombuffer(parameters[:6], dtype=np.int8).astype(np.float64).reshape(2, 3)[1]
                * 0.25
            )
            prediction = (gradient >= parameters[6])[:, None] * offsets[None]
        else:
            raise RX3Error(f"unsupported algorithm id in fire order: {result['algorithm_id']}")
        # Positive means the approximation moves toward the already-observed
        # downstream repair residual rather than away from it.  This is only a
        # scorer-free ordering proxy; it is never converted into distortion.
        gain = 2.0 * np.sum(target * prediction, axis=1) - np.sum(prediction**2, axis=1)
        repaired_gain = float(gain[repaired].mean())
        persistent_gain = float(gain[~repaired].mean())
        rows.append(
            {
                "candidate_id": candidate_id,
                "proxy_definition": "mean reduction in squared distance to observed total R+uint8 residual",
                "proxy_denominator": int(gain.size),
                "repaired_denominator": int(repaired.sum()),
                "persistent_denominator": int((~repaired).sum()),
                "repaired_proxy_gain": repaired_gain,
                "persistent_proxy_gain": persistent_gain,
                "repair_selectivity": repaired_gain - persistent_gain,
                "not_a_distortion_estimate": True,
            }
        )
    ranking = sorted(
        rows,
        key=lambda row: (
            -row["repaired_proxy_gain"],
            -row["repair_selectivity"],
            row["candidate_id"],
        ),
    )
    for ordinal, row in enumerate(ranking, 1):
        candidate_id = row["candidate_id"]
        runtime = RECEIPT_ROOT / f"retained/candidates/{candidate_id}/runtime"
        attempt = RECEIPT_ROOT / f"advisory_and_dali_scorer/{candidate_id}"
        row["ordinal"] = ordinal
        row["advisory_command"] = [
            str(REPO / ".venv/bin/python"),
            "tools/fire_local_advisory.py",
            "--runtime-dir",
            str(runtime),
            "--attempt-dir",
            str(attempt),
            "--label",
            f"ddm_rx3_{candidate_id}",
            "--projected-gib",
            "20",
            "--inflate-timeout",
            "7200",
            "--evaluate-timeout",
            "14400",
        ]
        row["dali_postscore_command"] = [
            str(REPO / ".venv/bin/python"),
            "experiments/ddm_rx3_receiver_precompensation.py",
            "score",
            "--candidate-id",
            candidate_id,
            "--raw",
            str(attempt / "work/inflated/0.raw"),
            "--batch-pairs",
            "16",
        ]
    result = {
        "schema": "ddm_rx3_fire_order.v1",
        "created_utc": utc_now(),
        "status": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "ddm_rx3_receiver_precompensation",
        "consumer_store": str(RECEIPT_ROOT / "advisory_and_dali_scorer"),
        "fire_trigger": (
            "main_hot_state no longer grants the sole full-n600 scorer lane to AP1, AP1's "
            "latest QUEUE_STATE is terminal, and RX3 has appended a non-conflicting active "
            "local-scorer claim; fire one row at a time in ordinal order"
        ),
        "lane_rules": {
            "full_n600_concurrency": 1,
            "chunk_pairs": 16,
            "F26_TOKEN_DECODER": "python default; native-hpac forbidden",
            "prefix": "forbidden/unavailable",
            "advisory_launcher": "tools/fire_local_advisory.py only",
        },
        "ranking_basis": "scorer-free repair-residual alignment; no distortion interpolation",
        "rows": ranking,
        "terminal_aggregate_command": [
            str(REPO / ".venv/bin/python"),
            "experiments/ddm_rx3_receiver_precompensation.py",
            "aggregate",
        ],
    }
    atomic_json(RECEIPT_ROOT / "FIRE_ORDER.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _candidate_ids() -> tuple[str, ...]:
    manifest_path = RECEIPT_ROOT / "MATERIALIZE_ALL.json"
    if not manifest_path.is_file():
        raise RX3Error("materialize stage is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return tuple(str(row["candidate_id"]) for row in manifest["candidates"])


def _array_fact(path: Path, value: np.ndarray) -> dict[str, Any]:
    return atomic_npy(path, value)


def _load_array_fact(record: Mapping[str, Any]) -> np.ndarray:
    path = validate_fact(record)
    value = np.load(path, allow_pickle=False)
    if list(value.shape) != list(record["shape"]) or value.dtype.str != record["dtype"]:
        raise RX3Error(f"retained array geometry drifted: {path}")
    return value


def _chunk_score_statistics(
    gt_seg: np.ndarray,
    candidate_seg: np.ndarray,
    gt_pose: np.ndarray,
    candidate_pose: np.ndarray,
) -> dict[str, Any]:
    if gt_seg.shape != candidate_seg.shape or gt_seg.ndim != 3:
        raise RX3Error("candidate Seg argmax chunk geometry differs")
    if gt_pose.shape != candidate_pose.shape or gt_pose.shape != (gt_seg.shape[0], 6):
        raise RX3Error("candidate Pose6 chunk geometry differs")
    if np.any(gt_seg >= len(CLASS_NAMES)) or np.any(candidate_seg >= len(CLASS_NAMES)):
        raise RX3Error("candidate Seg argmax chunk contains an invalid class")
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(confusion, (gt_seg.reshape(-1), candidate_seg.reshape(-1)), 1)
    residual = candidate_pose.astype(np.float64) - gt_pose.astype(np.float64)
    return {
        "pairs": int(gt_seg.shape[0]),
        "seg_pixels": int(gt_seg.size),
        "flips": int(gt_seg.size - np.trace(confusion)),
        "confusion_gt_rows_candidate_columns": confusion.tolist(),
        "pose_squared_error_sum": float(np.square(residual).sum(dtype=np.float64)),
    }


def _aggregate_score_receipts(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    pairs = pixels = flips = 0
    pose_sse = 0.0
    expected_start = 0
    for receipt in receipts:
        start = int(receipt["pair_start"])
        stop = int(receipt["pair_stop_exclusive"])
        if start != expected_start or stop <= start:
            raise RX3Error("retained candidate scorer chunks are not contiguous")
        statistics = receipt["statistics"]
        if int(statistics["pairs"]) != stop - start:
            raise RX3Error("retained candidate scorer chunk span differs from statistics")
        pairs += int(statistics["pairs"])
        pixels += int(statistics["seg_pixels"])
        flips += int(statistics["flips"])
        pose_sse += float(statistics["pose_squared_error_sum"])
        confusion += np.asarray(statistics["confusion_gt_rows_candidate_columns"], dtype=np.int64)
        expected_start = stop
    if pairs != N or pixels != SEG_PIXELS or int(confusion.sum()) != SEG_PIXELS:
        raise RX3Error("retained candidate scorer does not cover full n600")
    if flips != SEG_PIXELS - int(np.trace(confusion)):
        raise RX3Error("candidate Seg reduction does not close")
    class_denominators = tuple(int(value) for value in confusion.sum(axis=1))
    if class_denominators != CLASS_DENOMINATORS:
        raise RX3Error(
            f"DALI GT class census differs: {class_denominators} != {CLASS_DENOMINATORS}"
        )
    per_class = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        denominator = int(confusion[class_id].sum())
        class_flips = denominator - int(confusion[class_id, class_id])
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "gt_pixels": denominator,
                "flips_from_gt_class": class_flips,
                "conditional_d_seg": class_flips / denominator,
                "contribution_to_total_d_seg": class_flips / SEG_PIXELS,
            }
        )
    return {
        "pairs": pairs,
        "seg_denominator": SEG_PIXELS,
        "pose_denominator": N * 6,
        "flips": flips,
        "d_seg": flips / SEG_PIXELS,
        "pose_squared_error_sum": pose_sse,
        "d_pose": pose_sse / (N * 6),
        "confusion_gt_rows_candidate_columns": confusion.tolist(),
        "per_class": per_class,
        "lane_class_1": per_class[1],
    }


def score_candidate(candidate_id: str, raw_path: Path, batch_pairs: int) -> dict[str, Any]:
    if candidate_id not in _candidate_ids():
        raise RX3Error(f"unknown materialized candidate: {candidate_id}")
    if not 1 <= batch_pairs <= 120:
        raise RX3Error("--batch-pairs must be in [1, 120]")
    pins = verify_pins()
    candidate_root = RECEIPT_ROOT / "retained/candidates" / candidate_id
    materialized = json.loads(
        (candidate_root / "MATERIALIZE_RESULT.json").read_text(encoding="utf-8")
    )
    validate_fact(materialized["archive"])
    raw_path = raw_path.resolve()
    expected_raw = RECEIPT_ROOT / f"advisory_and_dali_scorer/{candidate_id}/work/inflated/0.raw"
    if raw_path != expected_raw.resolve():
        raise RX3Error(f"candidate raw must be retained at {expected_raw}")
    if raw_path.stat().st_size != 3_662_409_600:
        raise RX3Error("candidate raw byte count differs from complete n600 camera frames")
    out_dir = RECEIPT_ROOT / "advisory_and_dali_scorer" / candidate_id / "dali_score"
    result_path = out_dir / "RESULT.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        validate_fact(result["candidate_raw"])
        _load_array_fact(result["candidate_argmax_n600"])
        _load_array_fact(result["candidate_pose6_n600"])
        gt_seg = np.load(Path(pins["dali_seg_gt"]["path"]), mmap_mode="r")
        gt_pose = np.load(GT_POSE, mmap_mode="r")
        retained_receipts = []
        for record in result["chunk_receipts"]:
            receipt_path = validate_fact(record)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            candidate_seg = _load_array_fact(receipt["candidate_argmax"])
            candidate_pose = _load_array_fact(receipt["candidate_pose6"])
            start = int(receipt["pair_start"])
            stop = int(receipt["pair_stop_exclusive"])
            statistics = _chunk_score_statistics(
                np.asarray(gt_seg[start:stop]),
                candidate_seg,
                np.asarray(gt_pose[start:stop]),
                candidate_pose,
            )
            if statistics != receipt.get("statistics"):
                raise RX3Error(f"retained score chunk drifted: {receipt_path}")
            retained_receipts.append(receipt)
        if _aggregate_score_receipts(retained_receipts) != result["summary"]:
            raise RX3Error(f"retained scorer reduction drifted for {candidate_id}")
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    import torch
    from frame_utils import TensorVideoDataset
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    torch.manual_seed(12_341)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    scorer = DistortionNet().eval().to("cpu")
    scorer.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    names = [line for line in VIDEO_NAMES.read_text(encoding="utf-8").splitlines() if line]
    if len(names) != 1:
        raise RX3Error("RX3 expects exactly one public video")
    dataset = TensorVideoDataset(
        names,
        data_dir=raw_path.parent,
        batch_size=batch_pairs,
        device=torch.device("cpu"),
        num_threads=2,
        seed=1234,
    )
    dataset.prepare_data()
    loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=0)
    gt_seg = np.load(Path(pins["dali_seg_gt"]["path"]), mmap_mode="r")
    gt_pose = np.load(GT_POSE, mmap_mode="r")
    if gt_seg.shape != (N, H, W) or gt_seg.dtype != np.uint8:
        raise RX3Error("pinned DALI segmentation GT geometry differs")
    if gt_pose.shape != (N, 6) or gt_pose.dtype != np.float32:
        raise RX3Error("pinned DALI pose GT geometry differs")
    receipts = []
    start = 0
    for chunk_index, (_, _, batch) in enumerate(loader):
        stop = start + int(batch.shape[0])
        receipt_path = out_dir / "chunks" / f"{start:04d}_{stop - 1:04d}.json"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            candidate_seg = _load_array_fact(receipt["candidate_argmax"])
            candidate_pose = _load_array_fact(receipt["candidate_pose6"])
            statistics = _chunk_score_statistics(
                np.asarray(gt_seg[start:stop]),
                candidate_seg,
                np.asarray(gt_pose[start:stop]),
                candidate_pose,
            )
            if statistics != receipt.get("statistics"):
                raise RX3Error(f"retained score chunk drifted: {receipt_path}")
        else:
            with torch.inference_mode():
                pose_output, seg_logits = scorer(batch.to("cpu"))
                candidate_seg = seg_logits.argmax(dim=1).to(torch.uint8).cpu().numpy()
                candidate_pose = (
                    pose_output["pose"][..., :6].to(torch.float32).cpu().numpy().reshape(stop - start, 6)
                )
            seg_record = _array_fact(
                out_dir / "chunks" / f"{start:04d}_{stop - 1:04d}.candidate_argmax.npy",
                candidate_seg.astype(np.uint8, copy=False),
            )
            pose_record = _array_fact(
                out_dir / "chunks" / f"{start:04d}_{stop - 1:04d}.candidate_pose6.npy",
                candidate_pose.astype(np.float32, copy=False),
            )
            receipt = {
                "schema": "ddm_rx3_dali_score_chunk.v1",
                "chunk_index": chunk_index,
                "pair_start": start,
                "pair_stop_exclusive": stop,
                "candidate_argmax": seg_record,
                "candidate_pose6": pose_record,
                "statistics": _chunk_score_statistics(
                    np.asarray(gt_seg[start:stop]),
                    candidate_seg,
                    np.asarray(gt_pose[start:stop]),
                    candidate_pose,
                ),
                "checkpoint_complete": True,
            }
            atomic_json(receipt_path, receipt)
        receipts.append(receipt)
        start = stop
        print(f"score {candidate_id}: {stop}/{N}", flush=True)
    if start != N:
        raise RX3Error("candidate DALI scorer stopped before n600")
    summary = _aggregate_score_receipts(receipts)
    full_seg = np.concatenate(
        [_load_array_fact(receipt["candidate_argmax"]) for receipt in receipts], axis=0
    )
    full_pose = np.concatenate(
        [_load_array_fact(receipt["candidate_pose6"]) for receipt in receipts], axis=0
    )
    result = {
        "schema": "ddm_rx3_dali_score.v1",
        "created_utc": utc_now(),
        "status": "COMPLETE",
        "candidate_id": candidate_id,
        "axis": "[macOS-CPU advisory; pinned contest-CUDA DALI-GT tables; n600]",
        "promotable": False,
        "score_claim": False,
        "gt_lineage": {
            "seg_argmax": pins["dali_seg_gt"],
            "pose_first6": pins["dali_pose_gt"],
            "description": "DALI_NVDEC GT tables; frozen CPU-torch candidate forward",
        },
        "candidate_raw": file_fact(raw_path),
        "candidate_archive": materialized["archive"],
        "candidate_argmax_n600": _array_fact(
            out_dir / "candidate_argmax_n600.uint8.npy", full_seg
        ),
        "candidate_pose6_n600": _array_fact(
            out_dir / "candidate_pose6_n600.float32.npy", full_pose
        ),
        "segnet_weights": file_fact(Path(segnet_sd_path)),
        "posenet_weights": file_fact(Path(posenet_sd_path)),
        "batch_pairs": batch_pairs,
        "chunk_receipts": [file_fact(out_dir / "chunks" / f"{r['pair_start']:04d}_{r['pair_stop_exclusive'] - 1:04d}.json") for r in receipts],
        "summary": summary,
        "retention": "full transformed raw + every chunk argmax/Pose6 + concatenated n600 fields",
    }
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _class_delta(control: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for baseline, item in zip(control["per_class"], candidate["per_class"], strict=True):
        if baseline["class_id"] != item["class_id"] or baseline["gt_pixels"] != item["gt_pixels"]:
            raise RX3Error("candidate/control class census differs")
        rows.append(
            {
                "class_id": item["class_id"],
                "class_name": item["class_name"],
                "denominator": item["gt_pixels"],
                "control_flips": baseline["flips_from_gt_class"],
                "candidate_flips": item["flips_from_gt_class"],
                "delta_flips": item["flips_from_gt_class"] - baseline["flips_from_gt_class"],
                "delta_conditional_d_seg": item["conditional_d_seg"] - baseline["conditional_d_seg"],
                "delta_contribution_to_total_d_seg": (
                    item["contribution_to_total_d_seg"] - baseline["contribution_to_total_d_seg"]
                ),
            }
        )
    return rows


def aggregate_scores() -> dict[str, Any]:
    if not AP1_CONTROL.is_file():
        raise RX3Error("AP1 matched DX2 DALI control is not complete; aggregation remains queued")
    control_receipt = json.loads(AP1_CONTROL.read_text(encoding="utf-8"))
    control = control_receipt["summary"]
    rows = []
    for candidate_id in _candidate_ids():
        candidate_root = RECEIPT_ROOT / "retained/candidates" / candidate_id
        materialized = json.loads((candidate_root / "MATERIALIZE_RESULT.json").read_text())
        scored_path = RECEIPT_ROOT / f"advisory_and_dali_scorer/{candidate_id}/dali_score/RESULT.json"
        if not scored_path.is_file():
            raise RX3Error(f"candidate score is incomplete: {candidate_id}")
        scored = json.loads(scored_path.read_text(encoding="utf-8"))["summary"]
        archive_bytes = int(materialized["archive"]["bytes"])
        delta_seg = scored["d_seg"] - control["d_seg"]
        delta_pose = scored["d_pose"] - control["d_pose"]
        delta_distortion = 100.0 * delta_seg + (
            math.sqrt(10.0 * scored["d_pose"]) - math.sqrt(10.0 * control["d_pose"])
        )
        delta_rate = (archive_bytes - ARCHIVE_BYTES) * RATE_S_PER_BYTE
        net = delta_distortion + delta_rate
        class_delta = _class_delta(control, scored)
        rows.append(
            {
                "candidate_id": candidate_id,
                "fitted_parameter_count": materialized["fitted_parameter_count"],
                "parameter_section_bytes": materialized["parameter_section_bytes"],
                "real_archive_delta_bytes": archive_bytes - ARCHIVE_BYTES,
                "archive_bytes": archive_bytes,
                "d_seg": scored["d_seg"],
                "d_pose": scored["d_pose"],
                "delta_d_seg": delta_seg,
                "delta_d_pose": delta_pose,
                "delta_s_distortion": delta_distortion,
                "delta_s_rate": delta_rate,
                "net_delta_s": net,
                "per_class_delta": class_delta,
                "lane_class_1": class_delta[1],
                "denominators": {"seg_pixels": SEG_PIXELS, "pose_values": N * 6},
            }
        )
    result = {
        "schema": "ddm_rx3_candidate_table.v1",
        "created_utc": utc_now(),
        "status": "COMPLETE",
        "axis": "[macOS-CPU advisory; pinned contest-CUDA DALI-GT tables; n600]",
        "promotable": False,
        "score_claim": False,
        "control": {
            "source": file_fact(AP1_CONTROL),
            "archive_bytes": ARCHIVE_BYTES,
            "d_seg": control["d_seg"],
            "d_pose": control["d_pose"],
        },
        "rows": rows,
        "best_candidate": min(rows, key=lambda row: (row["net_delta_s"], row["candidate_id"])),
        "all_net_positive": all(row["net_delta_s"] > 0 for row in rows),
        "shipping_candidate_built": False,
    }
    atomic_json(RECEIPT_ROOT / "CANDIDATE_TABLE.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _split_candidate_outer(outer: bytes) -> tuple[bytes, dict[str, Any]]:
    if len(outer) < FOOTER.size:
        raise RX3Error("candidate footer is absent")
    magic, algorithm_id, parameter_count = FOOTER.unpack_from(outer, len(outer) - FOOTER.size)
    if magic != FOOTER_MAGIC or parameter_count == 0:
        raise RX3Error("candidate footer is malformed")
    start = len(outer) - FOOTER.size - parameter_count
    return outer[:start], {
        "algorithm_id": int(algorithm_id),
        "parameters": outer[start : start + parameter_count],
    }


def self_test() -> dict[str, Any]:
    base = b"base-payload"
    info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
    spec = CandidateSpec("test", 3, "gradient", 7, bytes((1, 2, 3, 4, 5, 6, 7)), "test")
    member = _pack_candidate(base, spec)
    parsed_base, parsed = _split_candidate_outer(member)
    archive = _emit_zip(member, info)
    repeat = _emit_zip(member, info)
    if parsed_base != base or parsed["parameters"] != spec.parameter_bytes:
        raise RX3Error("candidate trailer self-test failed")
    if archive != repeat or len(archive) - len(_emit_zip(base, info)) != len(spec.parameter_bytes) + FOOTER.size:
        raise RX3Error("real ZIP delta self-test failed")
    labels = np.array([False, False, True, True])
    values = np.array([0.0, 1.0, 2.0, 3.0])
    summary = _continuous_summary(values, labels)
    if summary["auc_repaired_high"] != 1.0:
        raise RX3Error("AUC self-test failed")
    result = {
        "status": "PASS",
        "archive_repeat_identity": True,
        "archive_delta_bytes": len(spec.parameter_bytes) + FOOTER.size,
        "auc_toy": summary["auc_repaired_high"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("characterize", "materialize", "rebuild-runtimes", "verify-runtimes", "fire-order", "score", "aggregate", "self-test"),
    )
    parser.add_argument("--candidate-id", choices=_candidate_ids() if (RECEIPT_ROOT / "MATERIALIZE_ALL.json").is_file() else (), default=None)
    parser.add_argument("--raw", type=Path, default=None)
    parser.add_argument("--batch-pairs", type=int, default=16)
    args = parser.parse_args()
    if args.command == "characterize":
        characterize()
    elif args.command == "materialize":
        materialize()
    elif args.command == "rebuild-runtimes":
        rebuild_runtimes()
    elif args.command == "verify-runtimes":
        verify_runtimes()
    elif args.command == "fire-order":
        fire_order()
    elif args.command == "score":
        if args.candidate_id is None or args.raw is None:
            parser.error("score requires --candidate-id and --raw")
        score_candidate(args.candidate_id, args.raw, args.batch_pairs)
    elif args.command == "aggregate":
        aggregate_scores()
    else:
        self_test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
