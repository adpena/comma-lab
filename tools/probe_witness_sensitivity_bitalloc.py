#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""#336 n600 witness precision-response curves and measured KKT allocation.

This is the witness-format sibling of ``jrd_pr110_section_response_curves.v1``.
Every load-bearing response row is measured over all 600 real-GT pairs with the
NumPy-fp32 receiver and CPU frozen scorers.  Work is pair-checkpointed and can be
driven in bounded foreground chunks.  The final selected allocation is replayed
through the actual byte-closed inflate runtime in resumable pair chunks.

No trainer knob is added.  No cloud, paid evaluator, MPS, or live-run write occurs.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _path in (_REPO, _REPO / "src", _REPO / "experiments", _REPO / "tools", _REPO / "upstream"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.witness_sensitivity_bitalloc import (  # noqa: E402
    SCHEMA,
    classify_response_rows,
    repeat_noise_floor,
    score_delta,
    solve_measured_reverse_waterfill,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _archive_bytes(blob: bytes) -> bytes:
    """Exact ``levelset_byte_close_and_eval.assemble_packet`` ZIP bytes in memory."""
    out = io.BytesIO()
    info = zipfile.ZipInfo(filename="0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr(info, blob)
    return out.getvalue()


def _manifest(params: dict[str, np.ndarray], cfg: dict[str, Any], so: dict[str, Any]):
    import levelset_byte_close_and_eval as bc

    blob, breakdown = bc.build_levelset_blob(params, cfg, so, None)
    manifest, *_ = bc._read_blob_bytes(blob)
    if manifest.get("lane_render_band") or manifest.get("pose_carrier"):
        raise ValueError("#336 currently requires the plain #406 checkpoint: lane/pose carriers absent")
    return manifest, blob, breakdown


def _render_pair(bc, ctx: dict[str, Any], pi: int) -> tuple[np.ndarray, np.ndarray]:
    """Canonical NumPy-fp32 witness receiver for one pair, before CPU scorers."""
    from tac.boundary_math.lever_b_levelset_generator import levelset_rgb_forward_numpy

    manifest = ctx["manifest"]
    curv = ctx["curv"]
    coords = ctx["coords"]
    params = ctx["params"]
    code = ctx["code"]
    rh, rw, ch, cw = ctx["rh"], ctx["rw"], ctx["ch"], ctx["cw"]
    fwd_kw = ctx["fwd_kw"]
    if bool(manifest["self_orient"]):
        ndf = int(manifest["n_dir_freqs"])
        dirf = np.zeros((curv.shape[0], 4 * ndf), np.float32)
        previous = None
        for _ in range(int(manifest["so_iters"])):
            feats = np.concatenate([curv, dirf], axis=-1)
            _rgb, phi = levelset_rgb_forward_numpy(params, feats, code[2 * pi + 1], **fwd_kw)
            argmax = phi.argmax(-1).reshape(rh, rw).astype(np.int64)
            if previous is not None and np.array_equal(argmax, previous):
                break
            dirf = bc._canon_dir_feats(
                coords,
                argmax,
                ndf,
                float(manifest["so_freq_along"]),
                float(manifest["so_freq_across"]),
                float(manifest["so_tau"]),
            )
            previous = argmax
        feats = np.concatenate([curv, dirf], axis=-1)
    else:
        feats = curv
    frames = []
    for frame_kind in (0, 1):
        rgb, _phi = levelset_rgb_forward_numpy(
            params, feats, code[2 * pi + frame_kind], **fwd_kw
        )
        frames.append(bc._torch_R_reference(np.asarray(rgb, np.float32), rh, rw, ch, cw))
    return frames[0], frames[1]


def _realize(
    params_fp: dict[str, np.ndarray],
    code_fp: np.ndarray,
    tensor: str | None,
    operation: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    from apply_sensitivity_bitalloc_witness import _realize_alloc

    params = {k: np.asarray(v, np.float32) for k, v in params_fp.items()}
    code = np.asarray(code_fp, np.float32)
    if operation.startswith("int"):
        bits = int(operation[3:])
        allocation = {} if tensor is None or bits == 8 else {tensor: bits}
        p, c, *_ = _realize_alloc(params, code, allocation)
        return p, c
    if tensor is None:
        raise ValueError(f"operation {operation!r} needs a tensor")
    target = code if tensor == "code" else params[tensor]
    if operation == "zero":
        replacement = np.zeros_like(target, dtype=np.float32)
    elif operation == "mean":
        replacement = np.full_like(target, float(np.mean(target)), dtype=np.float32)
    else:
        raise ValueError(f"unknown operation {operation!r}")
    if tensor == "code":
        code = replacement
    else:
        params[tensor] = replacement
    p, c, *_ = _realize_alloc(params, code, {})
    return p, c


def _realize_combined(
    params_fp: dict[str, np.ndarray], code_fp: np.ndarray, nbits: dict[str, int]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    from apply_sensitivity_bitalloc_witness import _realize_alloc

    p, c, *_ = _realize_alloc(params_fp, code_fp, nbits)
    return p, c


def _candidate(
    bc,
    cfg: dict[str, Any],
    so: dict[str, Any],
    params_dq: dict[str, np.ndarray],
    code_dq: np.ndarray,
) -> dict[str, Any]:
    full = {**params_dq, "code": code_dq}
    manifest, blob, breakdown = _manifest(full, cfg, so)
    archive = _archive_bytes(blob)
    return {
        "params": params_dq,
        "code": code_dq,
        "manifest": manifest,
        "blob": blob,
        "blob_sha256": _sha256_bytes(blob),
        "archive": archive,
        "archive_bytes": len(archive),
        "archive_sha256": _sha256_bytes(archive),
        "breakdown": breakdown,
    }


def _score_many_rows(
    seg_cpu, pose_cpu, gt, frames: list[tuple[np.ndarray, np.ndarray, int]]
) -> list[dict[str, float]]:
    from train_witness_realized_through_R_mlx import (
        cpu_verdict_d_pose_batch,
        cpu_verdict_d_seg_batch,
    )

    frame0 = [item[0] for item in frames]
    frame1 = [item[1] for item in frames]
    pair_ids = [item[2] for item in frames]
    # The scorers are distinct frozen read-only modules.  Run their batch forwards
    # concurrently; each Torch kernel remains single-threaded under --torch-threads=1,
    # so this changes wall-clock only, not either arithmetic path.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        seg_future = pool.submit(
            cpu_verdict_d_seg_batch,
            seg_cpu,
            frame1,
            [gt.lstars[pi] for pi in pair_ids],
        )
        pose_future = pool.submit(
            cpu_verdict_d_pose_batch,
            pose_cpu,
            frame0,
            frame1,
            [gt.gt_poses[pi] for pi in pair_ids],
        )
        dseg = seg_future.result()
        dpose = pose_future.result()
    return [
        {"pair_id": int(pi), "d_seg": float(ds), "d_pose": float(dp)}
        for pi, ds, dp in zip(pair_ids, dseg, dpose, strict=True)
    ]


def _score_many(seg_cpu, pose_cpu, gt, frames: list[tuple[np.ndarray, np.ndarray, int]]):
    return {
        str(row["pair_id"]): {"d_seg": row["d_seg"], "d_pose": row["d_pose"]}
        for row in _score_many_rows(seg_cpu, pose_cpu, gt, frames)
    }


def _aggregate(records: dict[str, Any], n_pairs: int) -> tuple[float, float]:
    ordered = [records[str(i)] for i in range(n_pairs)]
    return (
        float(np.mean([float(r["d_seg"]) for r in ordered], dtype=np.float64)),
        float(np.mean([float(r["d_pose"]) for r in ordered], dtype=np.float64)),
    )


def _measure_batch_axis_control(
    bc,
    ctx: dict[str, Any],
    gt,
    seg_cpu,
    pose_cpu,
    n_pairs: int,
    verdict_batch: int,
) -> dict[str, Any]:
    """Diagnostic-only singleton-vs-selected-batch scorer drift on up to four pairs."""
    pair_ids = list(range(min(4, n_pairs)))
    rendered = [(*_render_pair(bc, ctx, pi), pi) for pi in pair_ids]
    singleton: dict[str, Any] = {}
    for item in rendered:
        singleton.update(_score_many(seg_cpu, pose_cpu, gt, [item]))
    padded = [*rendered]
    while len(padded) < verdict_batch:
        padded.append(rendered[0])
    selected_rows = _score_many_rows(seg_cpu, pose_cpu, gt, padded)[: len(rendered)]
    selected = {
        str(row["pair_id"]): {"d_seg": row["d_seg"], "d_pose": row["d_pose"]}
        for row in selected_rows
    }
    per_pair = {
        str(pi): {
            "singleton": singleton[str(pi)],
            "selected_batch": selected[str(pi)],
            "abs_delta": {
                key: abs(float(selected[str(pi)][key]) - float(singleton[str(pi)][key]))
                for key in ("d_seg", "d_pose")
            },
        }
        for pi in pair_ids
    }
    return {
        "classification": "MEASURED DIAGNOSTIC; n<=4; NOT load-bearing evidence",
        "selected_batch": verdict_batch,
        "pairs": per_pair,
        "max_abs_delta": {
            key: max((row["abs_delta"][key] for row in per_pair.values()), default=0.0)
            for key in ("d_seg", "d_pose")
        },
        "interpretation": (
            "All n600 response rows use one fixed selected batch axis. This diagnostic records "
            "rather than conceals any arithmetic change versus singleton CPU inference."
        ),
        "score_claim": False,
    }


def _measurement_plan(
    tensors: list[str], bits: list[int], verdict_batch: int
) -> list[dict[str, Any]]:
    plan = [
        {"label": "baseline_repeat_a", "tensor": None, "operation": "int8", "batch": verdict_batch},
        {"label": "baseline_repeat_b", "tensor": None, "operation": "int8", "batch": verdict_batch},
    ]
    for tensor in tensors:
        for bit in bits:
            if bit == 8:
                continue
            plan.append({"label": f"{tensor}:int{bit}", "tensor": tensor,
                         "operation": f"int{bit}", "batch": verdict_batch})
        plan.append({"label": f"{tensor}:zero", "tensor": tensor,
                     "operation": "zero", "batch": verdict_batch})
        plan.append({"label": f"{tensor}:mean", "tensor": tensor,
                     "operation": "mean", "batch": verdict_batch})
    return plan


def _fingerprint_payload(args, ckpt_sha: str, tensors: list[str], bits: list[int]) -> dict[str, Any]:
    bound_sources = [
        Path(__file__).resolve(),
        _REPO / "tools" / "levelset_byte_close_and_eval.py",
        _REPO / "src" / "tac" / "witness_sensitivity_bitalloc.py",
        _REPO / "src" / "tac" / "losses" / "variable_level_waterfill_allocator.py",
    ]
    return {
        "schema": SCHEMA,
        "ckpt_sha256": ckpt_sha,
        "gt_cache_sha256": _sha256_file(Path(args.gt_cache)),
        "n_pairs": int(args.n_pairs),
        "bits": bits,
        "tensors": tensors,
        "so": [args.so_freq_across, args.so_freq_along, args.so_tau, args.so_iters],
        "torch_threads": int(args.torch_threads),
        "verdict_batch": int(args.verdict_batch),
        "inflate_workers": int(args.inflate_workers),
        "bound_source_sha256": {
            str(path.relative_to(_REPO)): _sha256_file(path) for path in bound_sources
        },
    }


def _load_or_init_state(
    path: Path,
    fingerprint_payload: dict[str, Any],
    *,
    allow_complete_postprocess_drift: bool = False,
    required_complete_labels: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    fingerprint = _sha256_bytes(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    )
    if path.exists():
        state = json.loads(path.read_text())
        if state.get("fingerprint") != fingerprint:
            if not allow_complete_postprocess_drift:
                raise ValueError(
                    "resume fingerprint mismatch; refusing to mix checkpoint/GT/config evidence"
                )
            stored_payload = state.get("fingerprint_payload")
            if not isinstance(stored_payload, dict):
                raise ValueError("postprocess migration requires the stored fingerprint payload")
            stored_fingerprint = _sha256_bytes(
                json.dumps(stored_payload, sort_keys=True, separators=(",", ":")).encode()
            )
            if state.get("fingerprint") != stored_fingerprint:
                raise ValueError("stored measurement fingerprint is internally inconsistent")
            stored_config = {k: v for k, v in stored_payload.items() if k != "bound_source_sha256"}
            current_config = {
                k: v for k, v in fingerprint_payload.items() if k != "bound_source_sha256"
            }
            if stored_config != current_config:
                raise ValueError("postprocess migration refuses measurement config drift")
            stored_sources = stored_payload.get("bound_source_sha256", {})
            current_sources = fingerprint_payload.get("bound_source_sha256", {})
            if set(stored_sources) != set(current_sources):
                raise ValueError("postprocess migration refuses bound-source set drift")
            changed_sources = sorted(
                key for key in stored_sources if stored_sources[key] != current_sources[key]
            )
            allowed_sources = sorted(
                [
                    "src/tac/witness_sensitivity_bitalloc.py",
                    "tools/probe_witness_sensitivity_bitalloc.py",
                ]
            )
            if changed_sources != allowed_sources:
                raise ValueError(
                    "postprocess migration permits only the classifier and probe-tool fixes; "
                    f"changed={changed_sources}"
                )
            expected_pairs = {str(i) for i in range(int(fingerprint_payload["n_pairs"]))}
            missing_or_partial = []
            for label in required_complete_labels or []:
                pairs = state.get("units", {}).get(label, {}).get("pairs", {})
                if set(pairs) != expected_pairs:
                    missing_or_partial.append(label)
            if missing_or_partial:
                raise ValueError(
                    "postprocess migration requires a complete measured surface; "
                    f"incomplete={missing_or_partial[:8]}"
                )
            lineage = {
                "schema": "witness_sensitivity_bitalloc_postprocess_lineage.v1",
                "measurement_fingerprint": stored_fingerprint,
                "postprocess_fingerprint": fingerprint,
                "changed_sources": changed_sources,
                "measurement_bound_source_sha256": stored_sources,
                "postprocess_bound_source_sha256": current_sources,
                "acceptance_checks": {
                    "measurement_config_identical": True,
                    "all_measured_units_complete": True,
                    "candidate_custody_rederived_before_allocation": True,
                },
            }
            return state, lineage
        return state, None
    state = {
        "schema": "witness_sensitivity_bitalloc_resume.v1",
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "units": {},
        "candidate_custody": {},
        "final_inflate_chunks": {},
    }
    _atomic_json(path, state)
    return state, None


def _response_payload(
    state: dict[str, Any], plan: list[dict[str, Any]], tensors: list[str], bits: list[int], n_pairs: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    a = state["units"]["baseline_repeat_a"]
    b = state["units"]["baseline_repeat_b"]
    dseg0, dpose0 = _aggregate(a["pairs"], n_pairs)
    dseg1, dpose1 = _aggregate(b["pairs"], n_pairs)
    baseline = {
        "label": "baseline_repeat_a",
        "archive_bytes": int(a["archive_bytes"]),
        "archive_sha256": a["archive_sha256"],
        "blob_sha256": a["blob_sha256"],
        "d_seg": dseg0,
        "d_pose": dpose0,
        "eval_pairs": n_pairs,
        "axis": "[macOS-CPU advisory; NumPy-fp32 receiver; CPU frozen scorers]",
        "score_claim": False,
        "verdict_scope": "INSTANCE: named frozen #406 witness checkpoint, all 600 real-GT pairs",
    }
    repeat = {
        "d_seg": dseg1,
        "d_pose": dpose1,
        "archive_bytes": int(b["archive_bytes"]),
        "archive_sha256": b["archive_sha256"],
    }
    noise = repeat_noise_floor(baseline, repeat)
    raw_rows: list[dict[str, Any]] = []
    for tensor in tensors:
        raw_rows.append(
            {
                "tensor": tensor,
                "bits": 8,
                "operation": "int8",
                "label": f"{tensor}:int8",
                "archive_bytes": baseline["archive_bytes"],
                "archive_sha256": baseline["archive_sha256"],
                "blob_sha256": baseline["blob_sha256"],
                "d_seg": dseg0,
                "d_pose": dpose0,
                "eval_pairs": n_pairs,
            }
        )
    for spec in plan:
        if spec["tensor"] is None:
            continue
        unit = state["units"][spec["label"]]
        dseg, dpose = _aggregate(unit["pairs"], n_pairs)
        operation = str(spec["operation"])
        raw_rows.append(
            {
                "tensor": spec["tensor"],
                "bits": int(operation[3:]) if operation.startswith("int") else None,
                "operation": operation,
                "label": spec["label"],
                "archive_bytes": int(unit["archive_bytes"]),
                "archive_sha256": unit["archive_sha256"],
                "blob_sha256": unit["blob_sha256"],
                "d_seg": dseg,
                "d_pose": dpose,
                "eval_pairs": n_pairs,
            }
        )
    # Populate distortion deltas before the Pareto comparison consumes them.
    prepared = []
    for row in raw_rows:
        row["distortion_delta_S"] = 100.0 * (float(row["d_seg"]) - dseg0) + (
            np.sqrt(10.0 * float(row["d_pose"])) - np.sqrt(10.0 * dpose0)
        )
        prepared.append(row)
    classified = classify_response_rows(prepared, baseline=baseline, noise_floor=noise)
    allocation = solve_measured_reverse_waterfill(classified, baseline)
    response = {
        "schema": SCHEMA,
        "schema_lineage": "generalizes jrd_pr110_section_response_curves.v1 to LVLS1 witness npz",
        "fingerprint": state["fingerprint"],
        "baseline": baseline,
        "deterministic_repeat": repeat,
        "measured_repeat_noise_floor": noise,
        "rows": classified,
        "precision_bits": bits,
        "eval_pairs": n_pairs,
        "authority": "[macOS-CPU advisory] NON-PROMOTABLE",
        "score_claim": False,
        "pointer_moved": False,
        "verdict_scope": "INSTANCE: frozen V9 ep150 EMA-best; post-hoc intN/zero/mean witness tensors",
    }
    return response, allocation


def _write_final_candidate(
    out_dir: Path,
    candidate: dict[str, Any],
    params: dict[str, np.ndarray],
    code: np.ndarray,
    cfg: dict[str, Any],
) -> None:
    archive_path = out_dir / "allocated_archive.zip"
    archive_path.write_bytes(candidate["archive"])
    npz_path = out_dir / "allocated_witness.npz"
    arrays = {k: np.asarray(v) for k, v in params.items()}
    arrays["code"] = np.asarray(code)
    for key, value in cfg.items():
        if key == "npz_name" or key in arrays:
            continue
        if isinstance(value, (str, int, float, bool, np.number)) or value is None:
            arrays[f"__cfg_{key}"] = np.array(value if value is not None else "__NONE__")
    tmp = npz_path.with_suffix(".npz.tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, npz_path)
    custody = {
        "archive": {"path": str(archive_path), "bytes": archive_path.stat().st_size,
                    "sha256": _sha256_file(archive_path)},
        "allocated_npz": {"path": str(npz_path), "bytes": npz_path.stat().st_size,
                          "sha256": _sha256_file(npz_path)},
        "score_claim": False,
    }
    _atomic_json(out_dir / "allocated_artifact_custody.json", custody)


def _measure_runtime_candidate(
    *,
    bc,
    label: str,
    candidate: dict[str, Any],
    unit: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    out_dir: Path,
    scratch_root: Path,
    gt,
    seg_cpu,
    pose_cpu,
    n_pairs: int,
    verdict_batch: int,
    inflate_workers: int,
    deadline: float | None,
    max_new_pairs: int,
    new_pairs: int,
    batch_control_path: Path,
) -> tuple[bool, int]:
    """Inflate one byte-closed candidate with the shipped optimized runtime, then score it.

    The raw is durable on SSD until every pair score is checkpointed.  On success a
    reproducibility manifest is written before the rebuildable raw is deleted.
    """
    if len(unit.get("pairs", {})) == n_pairs:
        return True, new_pairs
    label_hash = hashlib.sha256(label.encode()).hexdigest()[:12]
    work = scratch_root / "response_candidates" / f"{label_hash}_{label.replace(':', '_')[:48]}"
    work.mkdir(parents=True, exist_ok=True)
    blob_path = work / "candidate.bin"
    raw_path = work / "candidate.raw"
    runtime_path = work / "inflate.py"
    manifest_pairs = int(candidate["manifest"]["n_pairs"])
    measure_blob = (
        candidate["blob"]
        if n_pairs == manifest_pairs
        else _chunk_blob(bc, candidate["blob"], 0, n_pairs)
    )
    measure_blob_sha = _sha256_bytes(measure_blob)
    if not blob_path.exists():
        blob_path.write_bytes(measure_blob)
    if _sha256_file(blob_path) != measure_blob_sha:
        raise RuntimeError(f"{label}: scratch blob SHA drift")
    if not runtime_path.exists():
        runtime_path.write_text(bc._INFLATE_PY)
    frame_bytes = bc.CAMERA_H * bc.CAMERA_W * 3
    expected = 2 * n_pairs * frame_bytes
    if not raw_path.exists():
        free = shutil.disk_usage(work).free
        required = int(expected * 1.10)
        if free < required:
            raise RuntimeError(
                f"{label}: storage preflight failed: {free} B free < {required} B required"
            )
        env = dict(os.environ)
        env["INFLATE_WORKERS"] = str(inflate_workers)
        proc = subprocess.run(
            [sys.executable, str(runtime_path), str(blob_path), str(raw_path)],
            capture_output=True,
            text=True,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"{label}: inflate rc={proc.returncode}: {proc.stderr[-4000:]}")
    if raw_path.stat().st_size != expected:
        raise RuntimeError(f"{label}: raw size {raw_path.stat().st_size} != {expected}")

    def read_pair(handle, pi: int) -> tuple[np.ndarray, np.ndarray, int]:
        handle.seek(2 * pi * frame_bytes)
        f0b = handle.read(frame_bytes)
        f1b = handle.read(frame_bytes)
        if len(f0b) != frame_bytes or len(f1b) != frame_bytes:
            raise RuntimeError(f"{label}: short raw read at pair {pi}")
        return (
            np.frombuffer(f0b, dtype=np.uint8).reshape(bc.CAMERA_H, bc.CAMERA_W, 3),
            np.frombuffer(f1b, dtype=np.uint8).reshape(bc.CAMERA_H, bc.CAMERA_W, 3),
            pi,
        )

    with raw_path.open("rb") as handle:
        if label == "baseline_repeat_a" and not batch_control_path.exists():
            control_ids = list(range(min(4, n_pairs)))
            rendered = [read_pair(handle, pi) for pi in control_ids]
            singleton: dict[str, Any] = {}
            for item in rendered:
                singleton.update(_score_many(seg_cpu, pose_cpu, gt, [item]))
            padded = [*rendered]
            while len(padded) < verdict_batch:
                padded.append(rendered[0])
            selected_rows = _score_many_rows(seg_cpu, pose_cpu, gt, padded)[: len(rendered)]
            selected = {
                str(row["pair_id"]): {"d_seg": row["d_seg"], "d_pose": row["d_pose"]}
                for row in selected_rows
            }
            per_pair = {
                str(pi): {
                    "singleton": singleton[str(pi)],
                    "selected_batch": selected[str(pi)],
                    "abs_delta": {
                        key: abs(float(selected[str(pi)][key]) - float(singleton[str(pi)][key]))
                        for key in ("d_seg", "d_pose")
                    },
                }
                for pi in control_ids
            }
            _atomic_json(
                batch_control_path,
                {
                    "classification": "MEASURED DIAGNOSTIC; n<=4; NOT load-bearing evidence",
                    "selected_batch": verdict_batch,
                    "pairs": per_pair,
                    "max_abs_delta": {
                        key: max(row["abs_delta"][key] for row in per_pair.values())
                        for key in ("d_seg", "d_pose")
                    },
                    "score_claim": False,
                },
            )
        missing = [pi for pi in range(n_pairs) if str(pi) not in unit["pairs"]]
        for offset in range(0, len(missing), verdict_batch):
            pair_ids = missing[offset : offset + verdict_batch]
            rendered = [read_pair(handle, pi) for pi in pair_ids]
            padded = [*rendered]
            while len(padded) < verdict_batch:
                padded.append(rendered[0])
            rows = _score_many_rows(seg_cpu, pose_cpu, gt, padded)[: len(rendered)]
            for row in rows:
                unit["pairs"][str(row["pair_id"])] = {
                    "d_seg": row["d_seg"], "d_pose": row["d_pose"]
                }
            new_pairs += len(pair_ids)
            _atomic_json(state_path, state)
            boundary_due = (max_new_pairs and new_pairs >= max_new_pairs) or (
                deadline is not None and time.monotonic() >= deadline
            )
            if boundary_due and len(unit["pairs"]) < n_pairs:
                return False, new_pairs

    raw_sha = _sha256_file(raw_path)
    cleanup = {
        "original_path": str(raw_path),
        "bytes": expected,
        "sha256": raw_sha,
        "source_blob_sha256": candidate["blob_sha256"],
        "measurement_blob_sha256": measure_blob_sha,
        "archive_sha256": candidate["archive_sha256"],
        "command": [sys.executable, "inflate.py", "candidate.bin", "candidate.raw"],
        "env": {"INFLATE_WORKERS": inflate_workers},
        "reason": "deterministically rebuildable response raw; measured pair rows are durable",
        "cold_store_destination": None,
        "false_authority_flags": {"score_claim": False, "promotion_eligible": False},
        "success_only_delete": True,
    }
    _atomic_json(out_dir / f"cleanup_manifest_response_{label_hash}.json", cleanup)
    shutil.rmtree(work)
    return True, new_pairs


def _chunk_blob(bc, blob: bytes, start: int, stop: int) -> bytes:
    import brotli

    manifest, base_b, code_b, pose_b, lane_b, pcar_b, _chart_b = bc._read_blob_bytes(blob)  # (#497) 7th chart block: mechanical unpack update
    if pose_b or lane_b or pcar_b:
        raise ValueError("chunked final seal supports the plain #406 witness packet only")
    q = np.frombuffer(brotli.decompress(code_b), dtype=np.int8).reshape(manifest["code_shape"])
    q_chunk = q[2 * start : 2 * stop]
    manifest["n_pairs"] = stop - start
    manifest["code_shape"] = list(q_chunk.shape)
    mj = json.dumps(manifest, separators=(",", ":")).encode()
    return bc._io_pack(mj, base_b, brotli.compress(q_chunk.tobytes(), quality=11), None)


def _inflate_final_chunk(
    bc,
    state: dict[str, Any],
    state_path: Path,
    out_dir: Path,
    blob: bytes,
    start: int,
    stop: int,
    gt,
    seg_cpu,
    pose_cpu,
    scratch_root: Path,
) -> None:
    key = f"{start}:{stop}"
    if key in state["final_inflate_chunks"]:
        return
    chunk_dir = scratch_root / f"chunk_{start:04d}_{stop:04d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_blob = _chunk_blob(bc, blob, start, stop)
    src = chunk_dir / "chunk.bin"
    dst = chunk_dir / "chunk.raw"
    runtime = chunk_dir / "inflate.py"
    src.write_bytes(chunk_blob)
    runtime.write_text(bc._INFLATE_PY)
    frame_bytes = bc.CAMERA_H * bc.CAMERA_W * 3
    expected = 2 * (stop - start) * frame_bytes
    free = shutil.disk_usage(chunk_dir).free
    required = int(expected * 1.05)
    if free < required:
        raise RuntimeError(
            f"storage preflight failed: {free} B free < {required} B required for final chunk"
        )
    proc = subprocess.run([sys.executable, str(runtime), str(src), str(dst)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"final chunk inflate failed rc={proc.returncode}: {proc.stderr[-2000:]}")
    if dst.stat().st_size != expected:
        raise RuntimeError(f"chunk raw size {dst.stat().st_size} != {expected}")
    h = hashlib.sha256()
    pairs = {}
    rendered: list[tuple[np.ndarray, np.ndarray, int]] = []
    with dst.open("rb") as handle:
        for _local, pi in enumerate(range(start, stop)):
            f0b, f1b = handle.read(frame_bytes), handle.read(frame_bytes)
            h.update(f0b)
            h.update(f1b)
            f0 = np.frombuffer(f0b, dtype=np.uint8).reshape(bc.CAMERA_H, bc.CAMERA_W, 3)
            f1 = np.frombuffer(f1b, dtype=np.uint8).reshape(bc.CAMERA_H, bc.CAMERA_W, 3)
            rendered.append((f0, f1, pi))
    pairs.update(_score_many(seg_cpu, pose_cpu, gt, rendered))
    cleanup = {
        "original_path": str(dst),
        "bytes": expected,
        "sha256": h.hexdigest(),
        "chunk_blob_sha256": _sha256_bytes(chunk_blob),
        "full_archive_sha256": _sha256_file(out_dir / "allocated_archive.zip"),
        "command": [sys.executable, "inflate.py", "chunk.bin", "chunk.raw"],
        "reason": "deterministically rebuildable chunk scratch; exact archive and runtime are preserved",
        "cold_store_destination": None,
        "false_authority_flags": {"score_claim": False, "promotion_eligible": False},
        "success_only_delete": True,
    }
    state["final_inflate_chunks"][key] = {"pairs": pairs, "cleanup": cleanup}
    _atomic_json(state_path, state)
    _atomic_json(out_dir / f"cleanup_manifest_{start:04d}_{stop:04d}.json", cleanup)
    shutil.rmtree(chunk_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--npz-name", default="levelset_witness_ema_BEST.npz")
    ap.add_argument("--gt-cache", type=Path, default=Path("experiments/results/mlx_fleet_gt_cache/gt_n600.npz"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-pairs", type=int, default=600)
    ap.add_argument("--bits", default="8,7,6,5,4,3,2")
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--verdict-batch", type=int, default=4,
                    help="frozen CPU-scorer batch; recorded as an advisory-axis parameter")
    ap.add_argument("--inflate-workers", type=int, default=8,
                    help="local pair workers for the exact shipped NumPy inflate runtime")
    ap.add_argument("--so-freq-across", type=float, default=32.0)
    ap.add_argument("--so-freq-along", type=float, default=8.0)
    ap.add_argument("--so-tau", type=float, default=4.0)
    ap.add_argument("--so-iters", type=int, default=4)
    ap.add_argument("--max-new-pairs", type=int, default=0,
                    help="bounded foreground work; 0 means run until complete")
    ap.add_argument("--chunk-seconds", type=float, default=0.0)
    ap.add_argument("--final-inflate-chunk-pairs", type=int, default=8)
    ap.add_argument("--scratch-root", type=Path,
                    default=Path("experiments/.scratch/witness_bitalloc_336"),
                    help="transient certified raw root; local fallback is explicitly authorized by #336")
    ap.add_argument("--skip-zero-mean", action="store_true", help="diagnostic smoke only")
    ap.add_argument(
        "--finalize-complete-state-after-postprocess-fix",
        action="store_true",
        help=(
            "allow the recorded classifier/probe postprocess-only source drift only when the "
            "entire measured surface is complete; re-derives every candidate custody before solve"
        ),
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not (1 <= int(args.n_pairs) <= 600):
        raise ValueError("--n-pairs must be in [1,600]")
    if int(args.verdict_batch) < 1:
        raise ValueError("--verdict-batch must be >=1")
    if int(args.inflate_workers) < 1:
        raise ValueError("--inflate-workers must be >=1")
    bits = sorted({int(x) for x in str(args.bits).split(",") if x.strip()}, reverse=True)
    if 8 not in bits or any(bit not in range(2, 9) for bit in bits):
        raise ValueError("--bits must be a subset of 8,7,6,5,4,3,2 and include 8")
    out_dir = args.out_dir.resolve()
    if str(out_dir).startswith(("/tmp/", "/private/tmp/", "/var/tmp/")):
        raise ValueError("durable --out-dir may not be temporary")
    out_dir.mkdir(parents=True, exist_ok=True)
    source = (args.ckpt_dir / args.npz_name).resolve()
    frozen = out_dir / "frozen_source_checkpoint.npz"
    if not frozen.exists():
        shutil.copy2(source, frozen)
    source_sha = _sha256_file(source)
    if _sha256_file(frozen) != source_sha:
        raise ValueError("frozen checkpoint copy SHA mismatch")

    import levelset_byte_close_and_eval as bc
    import torch
    from train_witness_realized_through_R_mlx import load_gt_from_cache

    torch.set_num_threads(int(args.torch_threads))
    loaded, cfg = bc._load_levelset_ckpt(out_dir, frozen.name)
    code_fp = np.asarray(loaded.pop("code"), np.float32)
    params_fp = {k: np.asarray(v, np.float32) for k, v in loaded.items()}
    so = bc.detect_self_orient(cfg, {
        "freq_across": args.so_freq_across,
        "freq_along": args.so_freq_along,
        "tau": args.so_tau,
        "iters": args.so_iters,
    })
    p0, c0 = _realize(params_fp, code_fp, None, "int8")
    baseline_candidate = _candidate(bc, cfg, so, p0, c0)
    tensors = [*baseline_candidate["manifest"]["base_param_order"], "code"]
    plan = _measurement_plan(tensors, bits, int(args.verdict_batch))
    if args.skip_zero_mean:
        plan = [p for p in plan if p["operation"] not in {"zero", "mean"}]
    fingerprint_payload = _fingerprint_payload(args, source_sha, tensors, bits)
    fingerprint_payload["skip_zero_mean"] = bool(args.skip_zero_mean)
    state_path = out_dir / "resume_state.json"
    state, postprocess_lineage = _load_or_init_state(
        state_path,
        fingerprint_payload,
        allow_complete_postprocess_drift=bool(
            args.finalize_complete_state_after_postprocess_fix
        ),
        required_complete_labels=[str(spec["label"]) for spec in plan],
    )
    gt, seg_cpu, pose_cpu = load_gt_from_cache(args.gt_cache, int(args.n_pairs))
    batch_control_path = out_dir / "scorer_batch_axis_control.json"

    deadline = time.monotonic() + args.chunk_seconds if args.chunk_seconds > 0 else None
    new_pairs = 0
    for spec in plan:
        label = spec["label"]
        unit = state["units"].setdefault(label, {"pairs": {}})
        if spec["tensor"] is None:
            candidate = baseline_candidate
        else:
            pdq, cdq = _realize(params_fp, code_fp, spec["tensor"], spec["operation"])
            candidate = _candidate(bc, cfg, so, pdq, cdq)
        custody = {
            "archive_bytes": candidate["archive_bytes"],
            "archive_sha256": candidate["archive_sha256"],
            "blob_sha256": candidate["blob_sha256"],
        }
        for key, value in custody.items():
            if key in unit and unit[key] != value:
                raise ValueError(f"{label}: candidate custody drift in {key}")
            unit[key] = value
        complete, new_pairs = _measure_runtime_candidate(
            bc=bc,
            label=label,
            candidate=candidate,
            unit=unit,
            state=state,
            state_path=state_path,
            out_dir=out_dir,
            scratch_root=args.scratch_root,
            gt=gt,
            seg_cpu=seg_cpu,
            pose_cpu=pose_cpu,
            n_pairs=int(args.n_pairs),
            verdict_batch=int(args.verdict_batch),
            inflate_workers=int(args.inflate_workers),
            deadline=deadline,
            max_new_pairs=int(args.max_new_pairs),
            new_pairs=new_pairs,
            batch_control_path=batch_control_path,
        )
        if not complete:
            print(
                f"[#336] resumable boundary after {new_pairs} new pair measurements; "
                f"state={state_path} raw_preserved_on_ssd=true"
            )
            return 7
        dseg, dpose = _aggregate(unit["pairs"], int(args.n_pairs))
        print(f"[#336] complete {label}: n={args.n_pairs} d_seg={dseg:.9g} d_pose={dpose:.9g} "
              f"archive={unit['archive_bytes']}B", flush=True)
        if (args.max_new_pairs and new_pairs >= args.max_new_pairs) or (
            deadline is not None and time.monotonic() >= deadline
        ):
            print(f"[#336] resumable boundary after completed candidate {label}; state={state_path}")
            return 7

    repeat_a = state["units"]["baseline_repeat_a"]["pairs"]
    repeat_b = state["units"]["baseline_repeat_b"]["pairs"]
    repeat_mismatches = [
        pi for pi in range(int(args.n_pairs)) if repeat_a[str(pi)] != repeat_b[str(pi)]
    ]
    if repeat_mismatches:
        raise RuntimeError(
            f"selected scorer axis is not deterministic on {len(repeat_mismatches)}/"
            f"{args.n_pairs} repeated pairs; refusing response curves"
        )

    if args.skip_zero_mean:
        print("[#336] diagnostic precision-only run complete; zero/mean #153 rows intentionally skipped")
        return 0

    response, allocation = _response_payload(state, plan, tensors, bits, int(args.n_pairs))
    if postprocess_lineage is not None:
        response["postprocess_lineage"] = postprocess_lineage
        allocation["postprocess_lineage"] = postprocess_lineage
        _atomic_json(out_dir / "postprocess_lineage.json", postprocess_lineage)
    response["scorer_batch_axis"] = {
        "selected_batch": int(args.verdict_batch),
        "control": json.loads(batch_control_path.read_text()),
        "classification": "ASSUMED advisory scorer axis; upstream exact-eval not run",
    }
    _atomic_json(out_dir / "section_precision_response_curves.json", response)
    _atomic_json(out_dir / "allocated_bit_budget.json", allocation)

    combined_p, combined_c = _realize_combined(params_fp, code_fp, allocation["nbits"])
    combined = _candidate(bc, cfg, so, combined_p, combined_c)
    combined_unit = state["units"].setdefault("combined_kkt", {"pairs": {}})
    if postprocess_lineage is not None:
        combined_unit["postprocess_lineage"] = postprocess_lineage
    for key in ("archive_bytes", "archive_sha256", "blob_sha256"):
        value = combined[key]
        if key in combined_unit and combined_unit[key] != value:
            raise ValueError(f"combined KKT custody drift in {key}")
        combined_unit[key] = value
    complete, new_pairs = _measure_runtime_candidate(
        bc=bc,
        label="combined_kkt",
        candidate=combined,
        unit=combined_unit,
        state=state,
        state_path=state_path,
        out_dir=out_dir,
        scratch_root=args.scratch_root,
        gt=gt,
        seg_cpu=seg_cpu,
        pose_cpu=pose_cpu,
        n_pairs=int(args.n_pairs),
        verdict_batch=int(args.verdict_batch),
        inflate_workers=int(args.inflate_workers),
        deadline=deadline,
        max_new_pairs=int(args.max_new_pairs),
        new_pairs=new_pairs,
        batch_control_path=batch_control_path,
    )
    if not complete:
        print(f"[#336] resumable boundary in combined replay; state={state_path}")
        return 7
    combined_dseg, combined_dpose = _aggregate(combined_unit["pairs"], int(args.n_pairs))
    baseline = response["baseline"]
    combined_row = {
        "archive_bytes": combined["archive_bytes"],
        "archive_sha256": combined["archive_sha256"],
        "blob_sha256": combined["blob_sha256"],
        "d_seg_numpy_oracle": combined_dseg,
        "d_pose_numpy_oracle": combined_dpose,
        "net_delta_S_advisory_numpy_oracle": score_delta(
            combined_dseg,
            combined_dpose,
            combined["archive_bytes"],
            baseline["d_seg"],
            baseline["d_pose"],
            baseline["archive_bytes"],
        ),
        "eval_pairs": int(args.n_pairs),
        "score_claim": False,
        "verdict_scope": "INSTANCE: measured combined KKT allocation on frozen V9 ep150 checkpoint",
    }
    _atomic_json(out_dir / "combined_numpy_oracle_row.json", combined_row)
    _write_final_candidate(out_dir, combined, combined_p, combined_c, cfg)

    # ``combined_unit`` came from the exact shipped inflate runtime above; no second
    # renderer is necessary to establish the receiver-closed seal.
    final_dseg, final_dpose = combined_dseg, combined_dpose
    final_row = {
        **combined_row,
        "d_seg_byte_closed": final_dseg,
        "d_pose_byte_closed": final_dpose,
        "net_delta_S_advisory_byte_closed": score_delta(
            final_dseg,
            final_dpose,
            combined["archive_bytes"],
            baseline["d_seg"],
            baseline["d_pose"],
            baseline["archive_bytes"],
        ),
        "numpy_vs_inflate_abs_delta": {"d_seg": 0.0, "d_pose": 0.0},
        "receiver_closed": True,
        "axis": "[macOS-CPU advisory; actual LVLS1 inflate; CPU frozen scorers; n600]",
        "upstream_evaluate_py_run": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    if postprocess_lineage is not None:
        final_row["postprocess_lineage"] = postprocess_lineage
    _atomic_json(out_dir / "byte_closed_advisory_row.json", final_row)
    print(f"[#336] COMPLETE receiver-closed n{args.n_pairs}: delta_S={final_row['net_delta_S_advisory_byte_closed']:+.9g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
