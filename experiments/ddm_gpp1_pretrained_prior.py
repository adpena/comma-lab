#!/usr/bin/env python3
# ruff: noqa: I001
"""Measure a public generic motion prior as causal side information for LS1.

This is a scorer-free n600 research instrument.  It runs the public
TorchVision RAFT-small C_T_V2 model only on the previous, already-decoded RGB
pair from the unchanged move-44 render.  Every quantized flow and derived
belief map is retained under the owned SSD root before it is measured.

The result is an in-sample Miller--Madow conditional-surprise upper bound, not
a physical code length, candidate archive, score, or claim that model weights
are free under contest rule 118.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_key] = "4"

import numpy as np

from experiments.ddm_ls1_shipped_surprise import FIELD, FIELD_SHA, H, K, N, TOTAL, W

ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_gpp1")
RAW = Path("/Volumes/VertigoDataTier/pact/ddm_rlc5_cure_on_move43/public_rlc4/output/0.raw")
RAW_RECEIPT = RAW.parents[1] / "RESULT.json"
RAW_SHA = "2b762eba4a20a315c104f8447d6ea0e604f73c3d8b8b69b3fc63b0fc792d59fc"
WEIGHT_SOURCE = Path("/Users/adpena/.cache/torch/hub/checkpoints/raft_small_C_T_V2-01064c6d.pth")
WEIGHT_SHA = "01064c6dba73b0fc9fc8edf772248560a00a3acfd62ac6677e9eeebad9680e27"
LICENSE_SOURCE = REPO / ".venv/lib/python3.13/site-packages/torchvision-0.27.1.dist-info/LICENSE"
LICENSE_SHA = "6502f676851cfe25f8af75531dfb32375b7325b73c37e7b43741fa422893e71d"
LS1_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_ls1")
LS1_COUNTS = LS1_ROOT / "atlas/counts_0600.npz"
LS1_RESULT = LS1_ROOT / "atlas/RESULT.json"
ARCHIVE = Path("/Volumes/VertigoDataTier/pact/ddm_rlc5_cure_on_move43/candidate_runtime/archive.zip")
ARCHIVE_SHA = "04758c0dfb8d94ebe801602aac96d93a4f260aad24f58b6b9cdbbe2270ad460e"
NATIVE_H, NATIVE_W = 874, 1164
FRAMES = N * 2
PLANE = H * W
SEED = 20260911
RESERVE_BYTES = 16 << 30
AXIS = "[macOS-CPU advisory / scorer-free n600 receiver probabilities]"
PRIMARY_NAME = "raft_motion_boundary16"
BELIEF_LEVELS = 17  # sixteen integer flow cells plus pair-0 missing sentinel
THRESHOLD_BYTES = 8000
GEOMETRIES = [
    "isolated_boundary_adjacent",
    "isolated_interior",
    "short_boundary_adjacent",
    "short_interior",
    "long_boundary_adjacent",
    "long_interior",
    "correct_prediction",
]


def fact(path: Path) -> dict[str, object]:
    """Return a complete durable file fact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def read_frame(frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one exact LS1 observation row without importing scorer dependencies."""
    path = LS1_ROOT / "rows" / f"frame_{frame:04d}.npz"
    receipt = json.loads(path.with_suffix(".json").read_text())
    if receipt != fact(path):
        raise ValueError(f"LS1 row fact changed: {path}")
    with np.load(path, allow_pickle=False) as saved:
        frequencies = saved["frequencies"]
        truth = saved["symbols"]
        bits = saved["bits"]
    if frequencies.shape != (PLANE, K):
        raise ValueError("LS1 frequency shape changed")
    if not np.all(frequencies.sum(axis=1, dtype=np.uint64) == TOTAL):
        raise ValueError("LS1 frequency mass changed")
    selected = frequencies[np.arange(PLANE), truth]
    np.testing.assert_array_equal(bits, -np.log2(selected.astype(np.float64) / TOTAL))
    return frequencies, truth, bits


def oracle_table(
    keys: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Use LS1's exact plug-in MLE and Miller--Madow allocation."""
    cells, inverse = np.unique(keys // K, return_inverse=True)
    counts = np.zeros((len(cells), K), dtype=np.int64)
    counts[inverse, (keys % K).astype(np.int64)] = values
    totals = counts.sum(axis=1)
    nonzero = counts > 0
    support = nonzero.sum(axis=1)
    loss = np.zeros_like(counts, dtype=np.float64)
    np.log2(
        np.divide(totals[:, None], counts, out=np.ones_like(loss), where=nonzero),
        out=loss,
    )
    correction = np.divide(
        (1 - 1 / support)[:, None] / (2 * math.log(2)),
        counts,
        out=np.zeros_like(loss),
        where=nonzero,
    )
    plugin_bits = float((counts * loss).sum())
    correction_bits = float((counts * correction).sum())
    np.testing.assert_allclose(
        correction_bits,
        float((support - 1).sum()) / (2 * math.log(2)),
        rtol=1e-12,
    )
    summary = {
        "symbols": int(totals.sum()),
        "occupied_cells": len(cells),
        "singleton_cells": int(np.count_nonzero(totals == 1)),
        "plugin_bits": plugin_bits,
        "correction_bits": correction_bits,
        "mm_bits": plugin_bits + correction_bits,
        "correction": "Miller-Madow; exact LS1 allocation convention",
        "scope": "fixed full-n600 in-sample cells; free oracle parameters",
    }
    return cells, counts, loss, correction, summary


def admit(path: Path, need: int = 0) -> None:
    """Fail closed unless a write stays in the arm and preserves SSD reserve."""
    root = ROOT.resolve()
    target = path.resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"write escaped {root}: {target}")
    ROOT.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(ROOT).free < RESERVE_BYTES + need:
        raise RuntimeError("STORAGE_BLOCK: keep all evidence; no deletion or alternate tier is authorized")
    path.parent.mkdir(parents=True, exist_ok=True)


def record(path: Path, value: object) -> None:
    """Atomically retain JSON metadata."""
    admit(path)
    temporary = path.with_suffix(path.suffix + ".new")
    with temporary.open("w") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def retain_arrays(path: Path, values: dict[str, np.ndarray]) -> dict[str, object]:
    """Atomically retain one payload; an existing receipt makes restart read-only."""
    receipt_path = path.with_suffix(".json")
    if path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["payload"] != fact(path):
            raise ValueError(f"retained payload fact changed: {path}")
        return receipt
    admit(path, sum(array.nbytes for array in values.values()))
    temporary = path.with_suffix(".npz.new")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    receipt = {"payload": fact(path)}
    record(receipt_path, receipt)
    return receipt


def copy_pinned(source: Path, destination: Path, sha256: str) -> dict[str, object]:
    """Copy an immutable dependency into the arm and verify exact identity."""
    if fact(source)["sha256"] != sha256:
        raise ValueError(f"source pin failed: {source}")
    admit(destination, source.stat().st_size)
    if not destination.exists():
        shutil.copyfile(source, destination)
    copied = fact(destination)
    if copied["sha256"] != sha256:
        raise ValueError(f"copied pin failed: {destination}")
    return copied


def prepare() -> dict[str, object]:
    """Verify all immutable inputs and persist dependency/custody pins."""
    weight = copy_pinned(WEIGHT_SOURCE, ROOT / "model" / WEIGHT_SOURCE.name, WEIGHT_SHA)
    license_fact = copy_pinned(LICENSE_SOURCE, ROOT / "model" / "torchvision-BSD-3-Clause.txt", LICENSE_SHA)
    if fact(ARCHIVE)["sha256"] != ARCHIVE_SHA:
        raise ValueError("move-44 archive pin failed")
    if fact(FIELD)["sha256"] != FIELD_SHA:
        raise ValueError("shipped field pin failed")
    raw_receipt = json.loads(RAW_RECEIPT.read_text())
    raw = fact(RAW)
    if raw["sha256"] != RAW_SHA or raw["bytes"] != FRAMES * NATIVE_H * NATIVE_W * 3:
        raise ValueError("move-44 raw render pin failed")
    if raw_receipt["candidate_raw"] != raw:
        raise ValueError("move-44 raw receipt binding failed")
    if not LS1_COUNTS.exists() or not LS1_RESULT.exists():
        raise ValueError("complete LS1 n600 instrument is required")
    import torch
    import torchvision

    pins = {
        "axis": AXIS,
        "archive": fact(ARCHIVE),
        "field": fact(FIELD),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "license": license_fact,
        "ls1_counts": fact(LS1_COUNTS),
        "ls1_result": fact(LS1_RESULT),
        "model": {
            "architecture": "torchvision raft_small",
            "weights_enum": "Raft_Small_Weights.C_T_V2",
            "weights": weight,
            "training_data": ["FlyingChairs", "FlyingThings3D"],
            "video_specific_training": False,
            "scorer_derived": False,
            "semantic_claim": "generic optical-flow and motion-boundary prior; not a lane detector",
        },
        "platform": platform.platform(),
        "python": sys.version,
        "raw_render": raw,
        "raw_receipt": fact(RAW_RECEIPT),
        "seed": SEED,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "producer": fact(Path(__file__)),
        "retention": (
            "KEEP: model, license, every q4 flow and belief map, stage checkpoint, "
            "and gain map are required evidence; no cleanup is authorized"
        ),
        "rule118_status": (
            "measurement only; whether generic public neural weights are uncounted is "
            "an unresolved publication decision"
        ),
    }
    path = ROOT / "INPUTS.json"
    if path.exists():
        original = json.loads(path.read_text())
        stable_names = set(pins) - {"git_head", "producer"}
        changed = [name for name in stable_names if original[name] != pins[name]]
        if changed:
            raise ValueError("restart numerical input changed: " + ", ".join(changed))
        preserved_v1 = ROOT / "source/ddm_gpp1_pretrained_prior_v1.py"
        preserved_fact = fact(preserved_v1)
        if any(
            original["producer"][name] != preserved_fact[name]
            for name in ("bytes", "sha256")
        ):
            raise ValueError("preserved first-run producer does not match INPUTS")
        migration = ROOT / "source/PROVENANCE_MIGRATION.json"
        migration_value = {
            "first_run_git_head": original["git_head"],
            "first_run_producer": original["producer"],
            "restart_git_head": pins["git_head"],
            "restart_producer": pins["producer"],
            "reason": (
                "concurrent HEAD movement was removed from numerical restart equality; "
                "model, raw RGB, field, LS1, package, weights, license, seed, platform, "
                "and inference algorithm remain exact"
            ),
        }
        if migration.exists():
            prior_migration = json.loads(migration.read_text())
            for name in ("first_run_git_head", "first_run_producer", "restart_producer", "reason"):
                if prior_migration[name] != migration_value[name]:
                    raise ValueError("provenance migration changed: " + name)
        else:
            record(migration, migration_value)
        return original
    record(path, pins)
    return pins


def flow_beliefs(flow_q4: np.ndarray) -> dict[str, np.ndarray]:
    """Derive preregistered integer context cells from retained quarter-pixel flow."""
    if flow_q4.shape != (2, H, W) or flow_q4.dtype != np.int16:
        raise ValueError("flow_q4 shape/dtype mismatch")
    qx = flow_q4[0].astype(np.int32)
    qy = flow_q4[1].astype(np.int32)
    magnitude = np.abs(qx) + np.abs(qy)
    magnitude4 = np.searchsorted(np.array([4, 12, 32]), magnitude).astype(np.uint8)
    edge = np.zeros((H, W), dtype=np.int32)
    horizontal = np.abs(qx[:, 1:] - qx[:, :-1]) + np.abs(qy[:, 1:] - qy[:, :-1])
    vertical = np.abs(qx[1:] - qx[:-1]) + np.abs(qy[1:] - qy[:-1])
    edge[:, 1:] = np.maximum(edge[:, 1:], horizontal)
    edge[:, :-1] = np.maximum(edge[:, :-1], horizontal)
    edge[1:] = np.maximum(edge[1:], vertical)
    edge[:-1] = np.maximum(edge[:-1], vertical)
    boundary4 = np.searchsorted(np.array([2, 8, 24]), edge).astype(np.uint8)
    primary = (magnitude4 * 4 + boundary4).astype(np.uint8)
    return {
        "belief_motion4": magnitude4,
        "belief_boundary4": boundary4,
        "belief_motion_boundary16": primary,
    }


def load_pair(path: Path) -> dict[str, np.ndarray]:
    """Load and validate one retained belief payload."""
    receipt = json.loads(path.with_suffix(".json").read_text())
    if receipt["payload"] != fact(path):
        raise ValueError(f"belief payload changed: {path}")
    with np.load(path, allow_pickle=False) as saved:
        values = {name: saved[name] for name in saved.files}
    if set(values) != {
        "flow_q4",
        "belief_motion4",
        "belief_boundary4",
        "belief_motion_boundary16",
        "source_frames",
    }:
        raise ValueError(f"belief payload schema changed: {path}")
    return values


def infer_pair(model, raw: np.memmap, pair: int) -> dict[str, np.ndarray]:
    """Run RAFT on pair-1 only; pair 0 receives a distinguished missing cell."""
    import torch
    import torch.nn.functional as functional

    if pair == 0:
        q4 = np.zeros((2, H, W), dtype=np.int16)
        beliefs = flow_beliefs(q4)
        beliefs["belief_motion_boundary16"].fill(16)
        return {"flow_q4": q4, **beliefs, "source_frames": np.array([-1, -1])}
    source_frames = np.array([2 * pair - 2, 2 * pair - 1], dtype=np.int64)
    tensors = []
    for frame in source_frames.tolist():
        image = torch.from_numpy(np.array(raw[frame], copy=True)).permute(2, 0, 1)
        image = functional.interpolate(
            image[None].to(torch.float32),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
            antialias=False,
        )
        tensors.append(image.div_(127.5).sub_(1.0))
    with torch.inference_mode():
        flow = model(tensors[0], tensors[1])[-1][0]
    q4_torch = torch.round(flow * 4.0).clamp_(-32768, 32767).to(torch.int16)
    q4 = q4_torch.cpu().numpy()
    return {"flow_q4": q4, **flow_beliefs(q4), "source_frames": source_frames}


def inference(resume_from: Path, stop_after: int) -> None:
    """Run resumable n600 inference with stage and local-repeat receipts."""
    if resume_from.resolve() != (ROOT / "beliefs").resolve():
        raise ValueError("--resume-from must be the canonical belief store")
    if not 1 <= stop_after <= N:
        raise ValueError("--stop-after must be in [1,600]")
    pins = prepare()
    import torch
    from torchvision.models.optical_flow import raft_small

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model = raft_small(weights=None, progress=False)
    state = torch.load(pins["model"]["weights"]["path"], map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    raw = np.memmap(RAW, dtype=np.uint8, mode="r", shape=(FRAMES, NATIVE_H, NATIVE_W, 3))
    receipts: list[dict[str, object]] = []
    for pair in range(stop_after):
        path = ROOT / "beliefs" / f"pair_{pair:04d}.npz"
        if path.exists():
            load_pair(path)
            receipt = json.loads(path.with_suffix(".json").read_text())
        else:
            started = time.perf_counter()
            values = infer_pair(model, raw, pair)
            elapsed = time.perf_counter() - started
            receipt = retain_arrays(path, values)
            receipt.update(
                {
                    "pair": pair,
                    "wall_seconds": elapsed,
                    "causal_source_frames": values["source_frames"].tolist(),
                }
            )
            record(path.with_suffix(".json"), receipt)
        receipts.append(receipt)
        if pair in {1, 100, 300, 599} and pair < stop_after:
            repeated = infer_pair(model, raw, pair)
            original = load_pair(path)
            for name in original:
                np.testing.assert_array_equal(repeated[name], original[name])
            repeat_path = ROOT / "beliefs" / f"pair_{pair:04d}.repeat.npz"
            retain_arrays(repeat_path, repeated)
        if (pair + 1) % 25 == 0 or pair + 1 == stop_after:
            record(
                ROOT / "checkpoints" / f"inference_{pair + 1:04d}.json",
                {
                    "schema": "ddm_gpp1.raft_belief_checkpoint.v1",
                    "completed_pairs": pair + 1,
                    "payloads": [item["payload"] for item in receipts],
                    "resume_from": str(ROOT / "beliefs"),
                },
            )
            print(json.dumps({"completed_pairs": pair + 1}), flush=True)
    wall = sum(float(item.get("wall_seconds", 0.0)) for item in receipts)
    inference_result = {
        "axis": AXIS,
        "complete_n600": stop_after == N,
        "inference_pairs": max(0, stop_after - 1),
        "missing_sentinel_pairs": 1,
        "model_wall_seconds_local_cpu": wall,
        "model_wall_seconds_local_cpu_per_inferred_pair": wall / max(1, stop_after - 1),
        "payload_count": len(receipts),
        "repeat_pairs": [pair for pair in (1, 100, 300, 599) if pair < stop_after],
        "repeat_identity": True,
        "score_claim": False,
        "t4_identity_measured": False,
        "t4_wall_measured": False,
        "timing_scope": "model preprocessing+forward+integer belief derivation+payload write",
        "inputs": fact(ROOT / "INPUTS.json"),
        "source_migration": fact(ROOT / "source/PROVENANCE_MIGRATION.json"),
    }
    record(ROOT / ("INFERENCE_RESULT.json" if stop_after == N else f"SCOPE_{stop_after:04d}.json"), inference_result)


def sparse_add(counts: dict[int, int], code: np.ndarray) -> None:
    keys, values = np.unique(code, return_counts=True)
    for key, value in zip(keys.tolist(), values.tolist(), strict=True):
        counts[key] = counts.get(key, 0) + value


def count_joint(resume_from: Path) -> Path:
    """Count the exact nested LS1+belief cells with 25-frame complete checkpoints."""
    store = ROOT / "oracle"
    if resume_from.resolve() != store.resolve():
        raise ValueError("--resume-from must be the canonical oracle store")
    inference_result = json.loads((ROOT / "INFERENCE_RESULT.json").read_text())
    if not inference_result["complete_n600"]:
        raise ValueError("full n600 belief set is required")
    binding = {
        "producer": fact(Path(__file__)),
        "inputs": fact(ROOT / "INPUTS.json"),
        "inference": fact(ROOT / "INFERENCE_RESULT.json"),
        "ls1_counts": fact(LS1_COUNTS),
        "belief_name": PRIMARY_NAME,
        "belief_levels": BELIEF_LEVELS,
    }
    binding_path = store / "BINDING.json"
    if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
        raise ValueError("oracle binding changed")
    record(binding_path, binding)
    latest = store / "LATEST.json"
    counts: dict[int, int] = {}
    start = 0
    if latest.exists():
        receipt = json.loads(latest.read_text())
        checkpoint = Path(receipt["path"])
        if fact(checkpoint) != receipt:
            raise ValueError("oracle checkpoint fact changed")
        with np.load(checkpoint, allow_pickle=False) as saved:
            start = int(saved["frame"][0])
            counts = dict(zip(saved["keys"].tolist(), saved["counts"].tolist(), strict=True))
    for frame in range(start, N):
        cells_path = LS1_ROOT / "atlas/cells" / f"frame_{frame:04d}.npz"
        with np.load(cells_path, allow_pickle=False) as cells:
            base = cells["receiver_lane"]
        _, truth, _ = read_frame(frame)
        belief = load_pair(ROOT / "beliefs" / f"pair_{frame:04d}.npz")["belief_motion_boundary16"].reshape(-1)
        if belief.max() >= BELIEF_LEVELS:
            raise ValueError("belief cell escaped preregistered alphabet")
        joint = base * BELIEF_LEVELS + belief.astype(np.uint64)
        sparse_add(counts, joint * K + truth)
        if (frame + 1) % 25 == 0:
            keys = np.array(sorted(counts), dtype=np.uint64)
            values = {
                "frame": np.array([frame + 1], dtype=np.int64),
                "keys": keys,
                "counts": np.array([counts[int(key)] for key in keys], dtype=np.int64),
            }
            path = store / f"counts_{frame + 1:04d}.npz"
            retain_arrays(path, values)
            record(latest, fact(path))
            print(
                json.dumps({"stage": "joint_counts", "frames": frame + 1, "occupied": len(counts)}),
                flush=True,
            )
    return store


def measure(resume_from: Path) -> None:
    """Attribute the joint MM saving against LS1 receiver-visible cells."""
    store = count_joint(resume_from)
    trace = json.loads((LS1_ROOT / "TRACE.json").read_text())
    if not trace["full_n600"] or not trace["field_identity"]:
        raise ValueError("LS1 trace is not a complete shipped-field identity")
    with np.load(LS1_COUNTS, allow_pickle=False) as saved:
        base_table = oracle_table(saved["receiver_lane_keys"], saved["receiver_lane_counts"])
    with np.load(store / "counts_0600.npz", allow_pickle=False) as saved:
        joint_table = oracle_table(saved["keys"], saved["counts"])
    base_cells, base_counts, base_loss, base_correction, base_summary = base_table
    joint_cells, joint_counts, joint_loss, joint_correction, joint_summary = joint_table
    if base_summary["symbols"] != N * PLANE or joint_summary["symbols"] != N * PLANE:
        raise ValueError("oracle denominator changed")
    class_geometry = np.zeros((K, len(GEOMETRIES)), dtype=np.float64)
    pair_class = np.zeros((N, K), dtype=np.float64)
    gain_path = store / "incremental_gain_bits.npy"
    gain_new = gain_path.with_suffix(".npy.new")
    gains = np.lib.format.open_memmap(gain_new, mode="w+", dtype=np.float64, shape=(N, PLANE))
    for frame in range(N):
        _, truth, _ = read_frame(frame)
        cells_path = LS1_ROOT / "atlas/cells" / f"frame_{frame:04d}.npz"
        with np.load(cells_path, allow_pickle=False) as cells:
            base = cells["receiver_lane"]
            geometry = cells["geometry"]
        belief = load_pair(ROOT / "beliefs" / f"pair_{frame:04d}.npz")["belief_motion_boundary16"].reshape(-1)
        joint = base * BELIEF_LEVELS + belief.astype(np.uint64)
        base_where = np.searchsorted(base_cells, base)
        joint_where = np.searchsorted(joint_cells, joint)
        np.testing.assert_array_equal(base_cells[base_where], base)
        np.testing.assert_array_equal(joint_cells[joint_where], joint)
        gain = (
            base_loss[base_where, truth]
            + base_correction[base_where, truth]
            - joint_loss[joint_where, truth]
            - joint_correction[joint_where, truth]
        )
        gains[frame] = gain
        codes = truth.astype(np.int64) * len(GEOMETRIES) + geometry
        class_geometry += np.bincount(codes, weights=gain, minlength=K * len(GEOMETRIES)).reshape(K, len(GEOMETRIES))
        pair_class[frame] = np.bincount(truth, weights=gain, minlength=K)
    gains.flush()
    if gain_path.exists():
        if fact(gain_path)["sha256"] != fact(gain_new)["sha256"]:
            raise ValueError("incremental gain replay changed")
        repeat_path = store / "incremental_gain_bits.repeat.npy"
        gain_new.replace(repeat_path)
    else:
        gain_new.replace(gain_path)
    incremental_bytes = float(class_geometry.sum()) / 8
    expected_incremental = (base_summary["mm_bits"] - joint_summary["mm_bits"]) / 8
    np.testing.assert_allclose(incremental_bytes, expected_incremental, atol=1e-7)
    actual_ideal_bytes = trace["bits"] / 8
    base_total_gain = actual_ideal_bytes - base_summary["mm_bits"] / 8
    joint_total_gain = actual_ideal_bytes - joint_summary["mm_bits"] / 8
    result = {
        "axis": AXIS,
        "score_claim": False,
        "full_n600": True,
        "symbols": N * PLANE,
        "pairs": N,
        "belief": {
            "name": PRIMARY_NAME,
            "levels": BELIEF_LEVELS,
            "definition": (
                "quarter-pixel RAFT flow L1 magnitude bins [0..3] crossed with "
                "integer flow-gradient boundary bins [0..3]; pair 0 uses cell 16"
            ),
            "causality": "pair t uses only rendered RGB frames 2t-2 and 2t-1; pair 0 sentinel",
        },
        "baseline": base_summary,
        "joint": joint_summary,
        "actual_receiver_ideal_bytes": actual_ideal_bytes,
        "ls1_receiver_visible_mm_gain_bytes": base_total_gain,
        "joint_mm_gain_bytes": joint_total_gain,
        "incremental_mm_gain_bytes": incremental_bytes,
        "demand_bytes": 25899,
        "remaining_shortfall_bytes": 25899 - joint_total_gain,
        "prototype_threshold_incremental_bytes": THRESHOLD_BYTES,
        "prototype_required": incremental_bytes >= THRESHOLD_BYTES,
        "class_geometry_incremental_gain_bytes": (class_geometry / 8).tolist(),
        "class_pair_incremental_gain_bytes": (pair_class / 8).tolist(),
        "geometry_names": GEOMETRIES,
        "gain_payload": fact(gain_path),
        "scope": (
            "fixed receiver_lane x preregistered public-model belief cells, full n600 "
            "in-sample MLE with Miller-Madow correction; free oracle probability "
            "parameters; upper bound only, not a realizable code length"
        ),
        "union_not_sum": (
            "joint conditional entropy measured directly; incremental value is joint "
            "minus receiver_lane, never a sum of independent bounds"
        ),
        "producer": fact(Path(__file__)),
        "inputs": fact(ROOT / "INPUTS.json"),
        "inference": fact(ROOT / "INFERENCE_RESULT.json"),
    }
    record(store / "RESULT.json", result)
    record(
        ROOT / "COMPLETE.json",
        {
            "schema": "ddm_gpp1.complete.v1",
            "disposition": ("PROTOTYPE_REQUIRED" if result["prototype_required"] else "FORMULATION_CLOSED"),
            "result": fact(store / "RESULT.json"),
            "checkpoint": fact(store / "counts_0600.npz"),
            "resume_from": str(store),
        },
    )
    print(json.dumps(result, sort_keys=True, indent=2), flush=True)


def self_test() -> None:
    """Exercise deterministic integer beliefs and MM nesting on a small fixture."""
    q4 = np.zeros((2, H, W), dtype=np.int16)
    q4[0, 10:, 20:] = 16
    first = flow_beliefs(q4)
    second = flow_beliefs(q4.copy())
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    assert first["belief_motion_boundary16"].dtype == np.uint8
    assert int(first["belief_motion_boundary16"].max()) < 16
    keys = np.array([0, 1, 5, 6], dtype=np.uint64)
    values = np.array([3, 2, 4, 1], dtype=np.int64)
    _, _, _, _, summary = oracle_table(keys, values)
    assert summary["symbols"] == 10
    assert math.isfinite(summary["mm_bits"])
    print("ddm_gpp1 self-test passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    infer_parser = subparsers.add_parser("infer")
    infer_parser.add_argument("--resume-from", required=True, type=Path)
    infer_parser.add_argument("--stop-after", type=int, default=N)
    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("--resume-from", required=True, type=Path)
    subparsers.add_parser("self-test")
    arguments = parser.parse_args()
    if arguments.command == "infer":
        inference(arguments.resume_from, arguments.stop_after)
    elif arguments.command == "measure":
        measure(arguments.resume_from)
    else:
        self_test()
