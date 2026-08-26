# SPDX-License-Identifier: MIT
"""ddm_td1 -- instrument-guided token-drop byte surface for #869.

This runner is deliberately scorer-slot respectful.  It reads the live qo1 IX2
archive, builds a 768-cell x 4-rung sensitivity ledger from already-custodied
instruments, stages candidate token-drop maps through the real IX2 receiver
format, and optionally runs inflate only.  It does not run SegNet/PoseNet:
td1 does not own the scorer slot, so realized distortion rows are queued by the
receipt instead of fabricated from proxies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
EXP = REPO / "experiments"
for path in (str(SRC), str(EXP)):
    if path not in sys.path:
        sys.path.insert(0, path)

from tac.optimization import ddm_ix2_archive_container as IX2  # noqa: E402
from tac.submission_chain import build_byte_ledger, run_inflate, sha256_file, stage_submission  # noqa: E402
from tac.witness_dsl.ax1_pool_a_levers_20260730 import margin_coupled_level_map  # noqa: E402

import ddm_r7_token_coder as R7  # noqa: E402


DEN = 37_545_489
SEG_PIXELS_N600 = 600 * 384 * 512
BASELINE = {
    "source": "ddm_sb1 C RESULT / main_hot_state live own-vehicle row",
    "archive_bytes": 357_836,
    "archive_sha256": "d5e814d5b9f65c3094b0e65fecdd7771734d03c420c63d1d2033a671b766986a",
    "d_seg": 0.00431179,
    "d_pose": 0.00071459,
    "score": 0.7539807296911207,
    "axis": "[macOS-CPU advisory]",
}
RT1_NEGATIVE = {
    "source": "/Volumes/VertigoDataTier/pact/ddm_sb1_20260804/B_rt1_margin_16_12_8_4_n600_eval_receipt.json",
    "archive_bytes": 244_436,
    "delta_archive_bytes": -113_400,
    "d_seg": 0.00515854,
    "d_pose": 0.16815221,
    "score": 1.9753490686354727,
    "delta_d_seg": 0.00515854 - BASELINE["d_seg"],
    "delta_d_pose": 0.16815221 - BASELINE["d_pose"],
}
R8_POSE_TERM_EROSION_LIMIT = 0.005
RUNTIME_FILES = (
    "inflate.sh",
    "inflate_runner.py",
    "ddm_ix2_archive_container.py",
    "ddm_tr1_runtime.py",
    "pfs1_warp_receiver.py",
    "ddm_r7_token_coder.py",
    "repair_entropy_coder_runtime_adapters.py",
)
RUNG_LEVELS = (16, 14, 12, 8, 4)


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "__dict__"):
        return asdict(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cannot serialize {type(obj)!r}")


def _sha256_array(arr: np.ndarray) -> str:
    value = np.ascontiguousarray(arr)
    return hashlib.sha256(value.tobytes()).hexdigest()


def _read_payload_archive(archive: Path) -> tuple[bytes, list[bytes]]:
    with zipfile.ZipFile(archive) as zf:
        payload = zf.read("0.bin")
    bulk, sections = IX2.parse_payload(payload)
    return bulk, list(sections)


def _codes_to_pm1(codes16: np.ndarray) -> np.ndarray:
    return codes16.astype(np.float64) / 15.0 * 2.0 - 1.0


def _pm1_to_codes16(values: np.ndarray) -> np.ndarray:
    x01 = (np.clip(values, -1.0, 1.0) + 1.0) * 0.5
    return np.rint(x01 * 15.0).astype(np.uint8)


def _requant_global(codes16: np.ndarray, levels: int) -> np.ndarray:
    x01 = codes16.astype(np.float64) / 15.0
    return np.rint(x01 * (levels - 1)).astype(np.uint8)


def _apply_cell_level_map(tokens: np.ndarray, level_map: np.ndarray) -> np.ndarray:
    t = np.clip(_codes_to_pm1(tokens), -1.0, 1.0)
    levels = (level_map.astype(np.float64) - 1.0)[None, :, :, None]
    x01 = (t + 1.0) * 0.5
    snapped = np.round(x01 * levels) / levels * 2.0 - 1.0
    return _pm1_to_codes16(snapped)


def _per_cell_activity(tokens: np.ndarray) -> np.ndarray:
    base, _ = IX2._factor_mode_delta(tokens, 16)
    return (tokens != base[None]).sum(axis=(0, 3)).astype(np.float64)


def _per_cell_pair_residual(tokens: np.ndarray) -> np.ndarray:
    base, _ = IX2._factor_mode_delta(tokens, 16)
    return np.abs(tokens.astype(np.int16) - base[None].astype(np.int16)).sum(axis=3).astype(np.float64)


def _cell_sums_384x512(array: np.ndarray) -> np.ndarray:
    if array.shape[-2:] != (384, 512):
        raise ValueError(f"expected trailing 384x512 scorer lattice, got {array.shape}")
    reshaped = array.reshape(array.shape[:-2] + (24, 16, 32, 16))
    return reshaped.sum(axis=(-3, -1))


def _normalize(field: np.ndarray) -> np.ndarray:
    value = np.asarray(field, dtype=np.float64)
    lo = float(np.min(value))
    hi = float(np.max(value))
    if hi <= lo:
        return np.zeros_like(value, dtype=np.float64)
    return (value - lo) / (hi - lo)


def _load_g3_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != 600:
        raise ValueError(f"expected 600 g3 rows, got {len(rows)} from {path}")
    rows.sort(key=lambda r: int(r["pair_index"]))
    return rows


def _instrument_fields(
    *,
    tokens: np.ndarray,
    gt_argmax: Path,
    cx1_argmax: Path,
    g3_jsonl: Path,
    g4_recurrence: Path,
    sg1_cell_flip_mass: Path,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    g3_rows = _load_g3_rows(g3_jsonl)
    pair_score = np.array(
        [float(r["score_mass"]["distortion_score_mass"]) for r in g3_rows], dtype=np.float64
    )
    pair_pose = np.array(
        [float(r["score_mass"]["pose_score_mass"]) for r in g3_rows], dtype=np.float64
    )
    pair_pose_sq = np.array(
        [float(r["costate_signal"]["pose_squared_error_sum"]) for r in g3_rows], dtype=np.float64
    )
    pair_flip_count = np.array(
        [float(r["segmentation"]["flip_count"]) for r in g3_rows], dtype=np.float64
    )

    gt = np.load(gt_argmax, mmap_mode="r")
    cx1 = np.load(cx1_argmax, mmap_mode="r")
    if gt.shape != (600, 384, 512) or cx1.shape != gt.shape:
        raise ValueError(f"argmax cache shape drift: gt={gt.shape} cx1={cx1.shape}")

    cell_error = np.zeros((24, 32), dtype=np.float64)
    cell_score_weighted_error = np.zeros((24, 32), dtype=np.float64)
    for start in range(0, 600, 25):
        stop = min(600, start + 25)
        err = np.asarray(cx1[start:stop] != gt[start:stop], dtype=np.uint8)
        cells = _cell_sums_384x512(err).astype(np.float64)
        cell_error += cells.sum(axis=0)
        cell_score_weighted_error += (cells * pair_score[start:stop, None, None]).sum(axis=0)

    g4 = np.load(g4_recurrence)
    g4_flip_frequency = np.asarray(g4["flip_frequency"], dtype=np.float64)
    g4_cell_recurrence = _cell_sums_384x512(g4_flip_frequency).astype(np.float64)

    sg1_flip = np.asarray(np.load(sg1_cell_flip_mass), dtype=np.float64)
    if sg1_flip.shape != (24, 32):
        raise ValueError(f"sg1 cell flip shape drift: {sg1_flip.shape}")

    activity = _per_cell_activity(tokens)
    pair_residual = _per_cell_pair_residual(tokens)
    pose_weight = pair_pose + pair_pose_sq
    pose_weight = pose_weight / pose_weight.sum() if pose_weight.sum() > 0 else np.ones(600) / 600.0
    pose_tube_proxy = (pair_residual * pose_weight[:, None, None]).sum(axis=0)

    seg_guard = (
        0.45 * _normalize(cell_score_weighted_error)
        + 0.25 * _normalize(cell_error)
        + 0.20 * _normalize(sg1_flip)
        + 0.10 * _normalize(g4_cell_recurrence)
    )
    pose_guard = 0.70 * _normalize(pose_tube_proxy) + 0.30 * _normalize(activity)
    joint_guard = 0.72 * seg_guard + 0.28 * pose_guard

    fields = {
        "cell_error": cell_error,
        "cell_score_weighted_error": cell_score_weighted_error,
        "g4_cell_recurrence": g4_cell_recurrence,
        "sg1_cell_flip_mass": sg1_flip,
        "activity": activity,
        "pose_tube_proxy": pose_tube_proxy,
        "seg_guard": seg_guard,
        "pose_guard": pose_guard,
        "joint_guard": joint_guard,
        "pair_score_mass": pair_score,
        "pair_pose_score_mass": pair_pose,
        "pair_pose_squared_error_sum": pair_pose_sq,
        "pair_flip_count": pair_flip_count,
    }
    source = {
        "gt_argmax": {"path": str(gt_argmax), "sha256": sha256_file(gt_argmax), "shape": list(gt.shape)},
        "cx1_argmax": {"path": str(cx1_argmax), "sha256": sha256_file(cx1_argmax), "shape": list(cx1.shape)},
        "g3_jsonl": {"path": str(g3_jsonl), "sha256": sha256_file(g3_jsonl), "rows": len(g3_rows)},
        "g4_recurrence": {"path": str(g4_recurrence), "sha256": sha256_file(g4_recurrence)},
        "sg1_cell_flip_mass": {
            "path": str(sg1_cell_flip_mass),
            "sha256": sha256_file(sg1_cell_flip_mass),
            "shape": list(sg1_flip.shape),
        },
        "formula": {
            "seg_guard": "0.45*norm(g3_score_weighted_argmax_errors)+0.25*norm(cx1_argmax_errors)+0.20*norm(sg1_cell_flip_mass)+0.10*norm(g4_flip_recurrence)",
            "pose_guard": "0.70*norm(g3_pose_weighted_token_mode_residual)+0.30*norm(token_mode_activity)",
            "joint_guard": "0.72*seg_guard+0.28*pose_guard",
            "authority": "DERIVED instrument ledger only; scorer forward pass not run by td1",
        },
    }
    return source, np.asarray(joint_guard, dtype=np.float64), [
        {"name": key, "sha256_of_float64": _sha256_array(np.asarray(value, dtype=np.float64))}
        for key, value in fields.items()
        if np.asarray(value).shape == (24, 32)
    ], fields


def _level_map_by_counts(joint_guard: np.ndarray, counts_by_level: dict[int, int]) -> np.ndarray:
    if sum(counts_by_level.values()) != joint_guard.size:
        raise ValueError(f"counts do not cover grid: {counts_by_level}")
    order = np.argsort(joint_guard.reshape(-1), kind="mergesort")
    levels = np.empty(joint_guard.size, dtype=np.int64)
    cursor = 0
    for level in sorted(counts_by_level):
        count = counts_by_level[level]
        idx = order[cursor : cursor + count]
        levels[idx] = int(level)
        cursor += count
    return levels.reshape(joint_guard.shape)


def _candidate_maps(joint_guard: np.ndarray, sg1_flip: np.ndarray) -> dict[str, np.ndarray]:
    margin_rt1 = margin_coupled_level_map(sg1_flip, base_levels=16, min_levels=4, n_tiers=4)
    return {
        "global_L14": np.full(joint_guard.shape, 14, dtype=np.int64),
        "td1_joint_guard_16_14_12_8": _level_map_by_counts(
            joint_guard, {8: 384, 12: 192, 14: 96, 16: 96}
        ),
        "td1_joint_guard_16_12_8_4": _level_map_by_counts(
            joint_guard, {4: 552, 8: 87, 12: 86, 16: 43}
        ),
        "td1_joint_guard_16_14_12_10": _level_map_by_counts(
            joint_guard, {10: 384, 12: 160, 14: 128, 16: 96}
        ),
        "rt1_margin_reconstructed_16_12_8_4": margin_rt1.astype(np.int64),
    }


def _drop_weight(level_map: np.ndarray) -> np.ndarray:
    return (16.0 - level_map.astype(np.float64)) / 12.0


def _exposure(level_map: np.ndarray, risk: np.ndarray) -> float:
    weight = _drop_weight(level_map)
    return float(np.sum(np.asarray(risk, dtype=np.float64) * weight))


def _score_from_components(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    return 100.0 * d_seg + float(np.sqrt(10.0 * d_pose)) + 25.0 * archive_bytes / DEN


def _token_coder_surface(codes: np.ndarray, levels: int) -> dict[str, Any]:
    ix2_frame = IX2.encode_token_frame(codes, levels=16)
    smevr_frame = R7.encode_token_codes(codes, levels=16, codec="smevr")
    brotli11_frame = R7.encode_token_codes(codes, levels=16, codec="brotli11")
    r7_auto = R7.encode_token_codes(codes, levels=16, codec="auto")
    return {
        "ix2_token_frame_bytes": len(ix2_frame),
        "r7_smevr_token_frame_bytes": len(smevr_frame),
        "r7_brotli11_token_frame_bytes": len(brotli11_frame),
        "r7_auto_token_frame_bytes": len(r7_auto),
        "r7_auto_winner": "smevr" if len(smevr_frame) <= len(brotli11_frame) else "brotli11",
        "levels_argument": levels,
        "note": "candidate is kept in the live IX2 global-levels=16 format; per-cell sublattices are data values, not a new counted map",
    }


def _stage_candidate(
    *,
    name: str,
    level_map: np.ndarray,
    tokens: np.ndarray,
    sections: list[bytes],
    out_dir: Path,
    base_sub: Path,
    receiver_close: bool,
    names_file: Path,
) -> dict[str, Any]:
    candidate_dir = out_dir / name
    codes = _apply_cell_level_map(tokens, level_map)
    token_frame = IX2.encode_token_frame(codes, levels=16)
    payload = IX2.build_payload(token_frame, sections)
    archive = stage_submission(
        payload,
        dest=candidate_dir,
        runtime_src=base_sub,
        runtime_files=RUNTIME_FILES,
    )
    ledger = build_byte_ledger(archive)
    archive_dir = candidate_dir / "archive"
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    archive_dir.mkdir(parents=True)
    from tac.submission_archive import safe_extract_zip

    safe_extract_zip(archive, archive_dir)

    row: dict[str, Any] = {
        "name": name,
        "candidate_dir": str(candidate_dir),
        "archive": str(archive),
        "archive_sha256": sha256_file(archive),
        "archive_bytes": archive.stat().st_size,
        "level_counts": {str(k): int(v) for k, v in zip(*np.unique(level_map, return_counts=True))},
        "map_sha256_of_int64": _sha256_array(level_map.astype(np.int64)),
        "codes_sha256": _sha256_array(codes),
        "token_frame_sha256": hashlib.sha256(token_frame).hexdigest(),
        "token_coder_surface": _token_coder_surface(codes, 16),
        "byte_ledger": {
            "closes": ledger.closes(),
            "residual_bytes": ledger.residual_bytes,
            "payload_reencodes_identically": ledger.payload_reencodes_identically,
            "bulk_bytes": ledger.bulk_bytes,
            "joint_coded_bytes": ledger.joint_coded_bytes,
            "archive_bytes": ledger.archive_bytes,
            "archive_sha256": ledger.archive_sha256,
        },
        "receiver_close": {"ran": False, "reason": "not selected"},
    }
    if receiver_close:
        infl = run_inflate(
            candidate_dir,
            archive_dir=archive_dir,
            out_dir=candidate_dir / "inflated",
            video_names_file=names_file,
            timeout=3600,
        )
        row["receiver_close"] = {
            "ran": True,
            "returncode": infl.returncode,
            "out_dir": infl.out_dir,
            "raw_files": infl.raw_files,
            "raw_bytes": infl.raw_bytes,
            "seconds": infl.seconds,
            "stdout_tail": infl.stdout_tail,
            "stderr_tail": infl.stderr_tail,
        }
    return row


def _write_cell_ledger(
    *,
    path: Path,
    fields: dict[str, Any],
    joint_guard: np.ndarray,
) -> dict[str, Any]:
    flat_rank = np.empty(joint_guard.size, dtype=np.int64)
    flat_rank[np.argsort(joint_guard.reshape(-1), kind="mergesort")] = np.arange(joint_guard.size)
    total = 0
    with path.open("w") as handle:
        for r in range(24):
            for c in range(32):
                for level in (16, 14, 12, 8):
                    row = {
                        "schema": "ddm_td1_cell_rung_sensitivity.v1",
                        "evidence_axis": "[cached scorer instruments; macOS-CPU frozen-scorer advisory sources]",
                        "cell_row": r,
                        "cell_col": c,
                        "rung_levels": level,
                        "drop_weight_vs_L16_to_L4": float((16 - level) / 12.0),
                        "joint_guard_rank_low_is_safer": int(flat_rank[r * 32 + c]),
                        "joint_guard": float(joint_guard[r, c]),
                        "seg_guard": float(fields["seg_guard"][r, c]),
                        "pose_guard": float(fields["pose_guard"][r, c]),
                        "cx1_argmax_error_pixels": int(fields["cell_error"][r, c]),
                        "g3_score_weighted_argmax_error_mass": float(fields["cell_score_weighted_error"][r, c]),
                        "sg1_cell_flip_mass": float(fields["sg1_cell_flip_mass"][r, c]),
                        "g4_flip_recurrence_mass": float(fields["g4_cell_recurrence"][r, c]),
                        "token_mode_activity": float(fields["activity"][r, c]),
                        "g3_pose_weighted_token_mode_residual": float(fields["pose_tube_proxy"][r, c]),
                        "distortion_measured_by_td1": False,
                        "coder_singleton_measured_by_td1": False,
                        "verdict_scope": "DERIVED cell-rung ledger; scorer subset is queued because td1 does not own scorer slot",
                    }
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    total += 1
    return {"path": str(path), "rows": total, "sha256": sha256_file(path)}


def _rank_and_project(candidates: list[dict[str, Any]], maps: dict[str, np.ndarray], fields: dict[str, Any]) -> None:
    rt1_map = maps["rt1_margin_reconstructed_16_12_8_4"]
    rt1_seg_exp = _exposure(rt1_map, fields["seg_guard"])
    rt1_pose_exp = _exposure(rt1_map, fields["pose_guard"])
    for row in candidates:
        level_map = maps[row["name"]]
        delta_bytes = int(row["archive_bytes"] - BASELINE["archive_bytes"])
        row["delta_archive_bytes_vs_qo1"] = delta_bytes
        row["rate_delta_S_measured_bytes"] = 25.0 * delta_bytes / DEN
        seg_exp = _exposure(level_map, fields["seg_guard"])
        pose_exp = _exposure(level_map, fields["pose_guard"])
        row["exposure"] = {
            "seg_guard_drop_exposure": seg_exp,
            "pose_guard_drop_exposure": pose_exp,
            "rt1_seg_guard_drop_exposure": rt1_seg_exp,
            "rt1_pose_guard_drop_exposure": rt1_pose_exp,
            "calibration_source": RT1_NEGATIVE["source"],
        }
        inferred_delta_d_seg = RT1_NEGATIVE["delta_d_seg"] * (seg_exp / rt1_seg_exp) if rt1_seg_exp else None
        inferred_delta_d_pose = RT1_NEGATIVE["delta_d_pose"] * (pose_exp / rt1_pose_exp) if rt1_pose_exp else None
        if inferred_delta_d_seg is not None and inferred_delta_d_pose is not None:
            projected_d_seg = BASELINE["d_seg"] + inferred_delta_d_seg
            projected_d_pose = BASELINE["d_pose"] + inferred_delta_d_pose
            projected_score = _score_from_components(projected_d_seg, projected_d_pose, row["archive_bytes"])
            row["rt1_calibrated_projection"] = {
                "status": "INFERRED_NOT_A_SCORE",
                "delta_d_seg": inferred_delta_d_seg,
                "delta_d_pose": inferred_delta_d_pose,
                "projected_d_seg": projected_d_seg,
                "projected_d_pose": projected_d_pose,
                "projected_score": projected_score,
                "projected_delta_S_vs_qo1": projected_score - BASELINE["score"],
                "R8_pose_term_erosion": float(np.sqrt(10.0 * projected_d_pose) - np.sqrt(10.0 * BASELINE["d_pose"])),
                "R8_pass_projected": bool(
                    np.sqrt(10.0 * projected_d_pose) - np.sqrt(10.0 * BASELINE["d_pose"])
                    <= R8_POSE_TERM_EROSION_LIMIT
                ),
                "warning": "calibrated from one measured negative; scorer-realized subset was not run",
            }
        else:
            row["rt1_calibrated_projection"] = {"status": "UNAVAILABLE"}
    candidates.sort(
        key=lambda r: (
            r.get("rt1_calibrated_projection", {}).get("projected_score", float("inf")),
            r["archive_bytes"],
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-sub", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/sub_auto_pairbit"))
    ap.add_argument("--token-source", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_br1_20260803/cx1_tokens.npy"))
    ap.add_argument("--gt-argmax", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/gt_argmax_n600.npy"))
    ap.add_argument("--cx1-argmax", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/cx1_argmax_n600.npy"))
    ap.add_argument("--g3-jsonl", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_g3_score_atlas_n600_20260722T204000Z/ddm_g3_score_atlas_n600.jsonl"))
    ap.add_argument("--g4-recurrence", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_g4_spatial_stationarity_n600_20260722T212138Z/stage_checkpoints/01_recurrence_arrays.npz"))
    ap.add_argument("--sg1-cell-flip-mass", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_sg1_20260731/cell_flip_mass.npy"))
    ap.add_argument("--bulk-dir", type=Path, default=Path("/Volumes/VertigoDataTier/pact/ddm_td1_20260804"))
    ap.add_argument("--receipt-dir", type=Path, default=REPO / ".omx/research/ddm_td1_20260804")
    ap.add_argument("--names-file", type=Path, default=REPO / "upstream/public_test_video_names.txt")
    ap.add_argument("--receiver-close-top", type=int, default=0)
    args = ap.parse_args()

    started = time.time()
    args.bulk_dir.mkdir(parents=True, exist_ok=True)
    args.receipt_dir.mkdir(parents=True, exist_ok=True)

    base_archive = args.base_sub / "archive.zip"
    bulk, sections = _read_payload_archive(base_archive)
    tokens = IX2.decode_token_frame(bulk)
    if tokens.shape != (600, 24, 32, 4):
        raise ValueError(f"live token shape drift: {tokens.shape}")
    token_source = np.load(args.token_source)
    token_source_equal = bool(np.array_equal(tokens, token_source))
    if not token_source_equal:
        raise ValueError("live qo1 token bulk differs from cx1 token source; refusing cross-object ledger")

    instrument_source, joint_guard, field_hashes, fields = _instrument_fields(
        tokens=tokens,
        gt_argmax=args.gt_argmax,
        cx1_argmax=args.cx1_argmax,
        g3_jsonl=args.g3_jsonl,
        g4_recurrence=args.g4_recurrence,
        sg1_cell_flip_mass=args.sg1_cell_flip_mass,
    )
    maps = _candidate_maps(joint_guard, fields["sg1_cell_flip_mass"])

    npz_path = args.bulk_dir / "td1_candidate_maps_and_fields.npz"
    np.savez_compressed(
        npz_path,
        joint_guard=joint_guard,
        seg_guard=fields["seg_guard"],
        pose_guard=fields["pose_guard"],
        **{f"map__{name}": value for name, value in maps.items()},
    )

    ledger_path = args.receipt_dir / "td1_cell_rung_ledger.jsonl"
    cell_ledger = _write_cell_ledger(path=ledger_path, fields=fields, joint_guard=joint_guard)

    stage_names = [name for name in maps if not name.startswith("rt1_")]
    candidates: list[dict[str, Any]] = []
    for name in stage_names:
        candidates.append(
            _stage_candidate(
                name=name,
                level_map=maps[name],
                tokens=tokens,
                sections=sections,
                out_dir=args.bulk_dir,
                base_sub=args.base_sub,
                receiver_close=False,
                names_file=args.names_file,
            )
        )
    _rank_and_project(candidates, maps, fields)

    receiver_to_close = {row["name"] for row in candidates[: max(0, args.receiver_close_top)]}
    if receiver_to_close:
        closed: list[dict[str, Any]] = []
        by_name = {row["name"]: row for row in candidates}
        for name in receiver_to_close:
            closed_row = _stage_candidate(
                name=name,
                level_map=maps[name],
                tokens=tokens,
                sections=sections,
                out_dir=args.bulk_dir,
                base_sub=args.base_sub,
                receiver_close=True,
                names_file=args.names_file,
            )
            by_name[name].update(closed_row)
            closed.append(closed_row)

    receipt = {
        "schema": "ddm_td1_token_drop_guided_surface.v1",
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_axis": "[macOS-CPU advisory inputs; no scorer forward pass run]",
        "task": "td1 TOKEN-DROP AT OPTIMAL FORM (#869)",
        "seconds": time.time() - started,
        "baseline": BASELINE,
        "rt1_negative_calibration": RT1_NEGATIVE,
        "scorer_slot": {
            "td1_owns_full_n600_slot": False,
            "observed_live_owner": "sq2 [SCORER] in .omx/state/main_hot_state.md",
            "action": "byte surface measured; scorer subset/full-n600 must be queued",
        },
        "base_archive": {
            "path": str(base_archive),
            "sha256": sha256_file(base_archive),
            "bytes": base_archive.stat().st_size,
            "token_bulk_bytes": len(bulk),
            "token_bulk_reencodes": bool(IX2.encode_token_frame(tokens, levels=16) == bulk),
            "token_source_path": str(args.token_source),
            "token_source_sha256": sha256_file(args.token_source),
            "token_source_equal_to_archive": token_source_equal,
            "tokens_sha256": _sha256_array(tokens),
            "section_bytes_preserved": [len(s) for s in sections],
            "section_sha256_preserved": [hashlib.sha256(s).hexdigest() for s in sections],
        },
        "instrument_sources": instrument_source,
        "field_hashes": field_hashes,
        "candidate_maps_npz": {
            "path": str(npz_path),
            "sha256": sha256_file(npz_path),
            "bytes": npz_path.stat().st_size,
        },
        "cell_rung_ledger": cell_ledger,
        "candidates_ranked_by_inferred_projection": candidates,
        "measured": {
            "coder_surface": "MEASURED real IX2 archive bytes plus R7 smevr/brotli token frame bytes for each map",
            "receiver_close": f"MEASURED inflate only for top {args.receiver_close_top} ranked candidate(s)",
        },
        "not_measured": {
            "subset_d_seg_d_pose": "QUEUED: td1 did not own scorer slot; no SegNet/PoseNet forward pass run",
            "full_n600": "NOT RUN: sq2 owns scorer slot",
        },
        "next_if_resumed": [
            "Run <=32-pair matched-base subset scorer on the first receiver-closed td1 candidate and qo1 base.",
            "Reject immediately if R8 pose-term erosion exceeds 0.005 or if subset projection cannot beat S=0.7539807296911207.",
            "Only append/fire full-n600 scorer spec for a receiver-closed candidate after the subset passes.",
        ],
    }
    receipt_path = args.receipt_dir / "td1_surface_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=1, default=_json_default) + "\n")

    summary = {
        "receipt": str(receipt_path),
        "cell_ledger": str(ledger_path),
        "maps_npz": str(npz_path),
        "candidate_order": [
            {
                "name": row["name"],
                "archive_bytes": row["archive_bytes"],
                "delta_archive_bytes_vs_qo1": row["delta_archive_bytes_vs_qo1"],
                "rate_delta_S_measured_bytes": row["rate_delta_S_measured_bytes"],
                "projection": row["rt1_calibrated_projection"],
                "receiver_close": row["receiver_close"],
            }
            for row in candidates
        ],
    }
    print(json.dumps(summary, indent=1, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
