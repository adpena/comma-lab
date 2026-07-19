#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded real-n600 joint seg/pose interval solve and rate telemetry.

Each invocation is capped at twelve real cache pairs.  Per-pair NPZ stages and
an atomic JSON state make the solve resumable; multiple receipts compose into
the n600-capable evidence surface.  This is advisory measurement only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for path in (REPO, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tac.optimization.joint_seg_pose_rate import (  # noqa: E402
    JointSolveError,
    MarginBandConfig,
    derive_hyperplane_channel_band,
    derive_margin_rgb_band,
    generated_fill_predictor,
    range_payload_bytes_and_tiles,
    solve_interval_frame,
    solve_measured_waterfill,
)
from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator  # noqa: E402
from tools.measure_uint8_lattice_feasibility import (  # noqa: E402
    _sha256_file,
    _stat_tree_snapshot,
    stored_npy_memmap,
)

SCHEMA = "joint_seg_pose_inverse_rate_receipt.v1"
STATE_SCHEMA = "joint_seg_pose_inverse_rate_state.v1"
MAX_SUBSET = 12
CAMERA_HW = (874, 1164)
SCORER_HW = (384, 512)
DEFAULT_CACHE = Path("/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
DEFAULT_UPSTREAM = Path("/Users/adpena/Projects/pact/upstream")
SACRED = Path("/Users/adpena/Projects/pact/experiments/results/levelset_n600_witness_20260717T113932Z")
POINTER = "0.1910828242 [contest-CPU] UNMOVED"
SEED = 20260719


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(_canonical(value) + b"\n")
    os.replace(tmp, path)


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _load_cache(path: Path) -> dict[str, np.memmap]:
    fields = {key: stored_npy_memmap(path, key) for key in ("n_pairs", "gt_f0", "gt_f1", "lstars", "margins", "gt_poses")}
    if int(np.asarray(fields["n_pairs"]).reshape(())) != 600:
        raise JointSolveError("only the real n600 cache is admissible")
    if fields["gt_f0"].shape != (600, *CAMERA_HW, 3) or fields["gt_f1"].shape != (600, *CAMERA_HW, 3):
        raise JointSolveError("real cache camera geometry mismatch")
    if fields["lstars"].shape != (600, *SCORER_HW) or fields["margins"].shape != (600, *SCORER_HW):
        raise JointSolveError("real cache scorer geometry mismatch")
    if fields["gt_poses"].shape != (600, 6):
        raise JointSolveError("real cache pose geometry mismatch")
    return fields


def _load_seg_pullback_sidecar(path: Path, pairs: Sequence[int]) -> dict[int, dict[str, np.ndarray]]:
    """Load a bounded, native-arithmetic winner/rival VJP sidecar fail-closed."""

    with np.load(path, allow_pickle=False) as data:
        required = {"pair_ids", "winner", "rival", "unit_head_normal_pullback_rgb", "pair_norms", "receiver_arithmetic"}
        missing = required.difference(data.files)
        if missing:
            raise JointSolveError(f"Seg pullback sidecar lacks keys: {sorted(missing)}")
        arithmetic = str(np.asarray(data["receiver_arithmetic"]).reshape(()))
        if arithmetic != "native_float32_cpu_torch":
            raise JointSolveError("Seg pullback sidecar receiver arithmetic is not native_float32_cpu_torch")
        ids = [int(x) for x in np.asarray(data["pair_ids"]).tolist()]
        if ids != list(pairs):
            raise JointSolveError("Seg pullback sidecar pair ids/order differ from invocation")
        winner = np.asarray(data["winner"])
        rival = np.asarray(data["rival"])
        pullback = np.asarray(data["unit_head_normal_pullback_rgb"])
        norms = np.asarray(data["pair_norms"])
    expected = (len(ids), *SCORER_HW)
    if winner.shape != expected or rival.shape != expected or norms.shape != expected:
        raise JointSolveError("Seg pullback sidecar arrangement geometry mismatch")
    if pullback.shape != (*expected, 3):
        raise JointSolveError("Seg pullback sidecar RGB-VJP geometry mismatch")
    return {
        pair_id: {"winner": winner[i], "rival": rival[i], "pullback": pullback[i], "pair_norms": norms[i]}
        for i, pair_id in enumerate(ids)
    }


def _load_scorers(upstream: Path, threads: int) -> tuple[Any, Any, Any]:
    if not (upstream / "modules.py").is_file():
        raise JointSolveError(f"frozen upstream missing: {upstream}")
    sys.path.insert(0, str(upstream))
    import torch
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    torch.set_num_threads(threads)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    dn = DistortionNet().eval().to("cpu")
    dn.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    for parameter in dn.parameters():
        parameter.requires_grad_(False)
    return dn.segnet, dn.posenet, torch


def _hard_verdict(segnet: Any, posenet: Any, torch: Any, f0: np.ndarray, f1: np.ndarray, labels: np.ndarray, target_pose: np.ndarray) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    import einops

    pair = torch.from_numpy(np.stack((f0, f1), axis=0)[None]).float()
    x = einops.rearrange(pair, "b t h w c -> b t c h w")
    with torch.inference_mode():
        logits = segnet(segnet.preprocess_input(x))[0]
        argmax = logits.argmax(dim=0).cpu().numpy().astype(np.int64)
        logits_np = logits.cpu().numpy()
        masked = logits_np.copy()
        np.put_along_axis(masked, argmax[None], -np.inf, axis=0)
        rival = masked.argmax(axis=0).astype(np.int64)
        pose_out = posenet(posenet.preprocess_input(x))
        pose = pose_out["pose"] if isinstance(pose_out, dict) else pose_out
        pose6 = pose[0, :6].cpu().numpy().astype(np.float64)
    mismatch = argmax != labels
    return {
        "d_seg": float(np.mean(mismatch)),
        "seg_mismatched_pixels": int(np.count_nonzero(mismatch)),
        "d_pose": float(np.mean((pose6 - target_pose) ** 2)),
        "pose6": pose6.tolist(),
    }, argmax, rival


def _attribution(rate: dict[str, Any], labels: np.ndarray | None, rival: np.ndarray | None, margins: np.ndarray | None) -> dict[str, Any]:
    sums: dict[str, dict[str, int]] = defaultdict(lambda: {"brotli_q11_bytes": 0, "zstd_19_bytes": 0, "tiles": 0})
    for tile in rate["tiles"]:
        if labels is None:
            key = "pose_global/frame0"
        else:
            cy = min(SCORER_HW[0] - 1, int(tile["y"] + tile["h"] / 2))
            cx = min(SCORER_HW[1] - 1, int(tile["x"] + tile["w"] / 2))
            margin = float(margins[cy, cx])
            codim = "boundary_codim1" if margin < 0.039180326461791926 else "cell_interior"
            winner_id, rival_id = int(labels[cy, cx]), int(rival[cy, cx])
            key = f"cell_winner_{winner_id}/hyperplane_{winner_id}-{rival_id}/{codim}/frame1"
        row = sums[key]
        row["brotli_q11_bytes"] += int(tile["brotli_q11_bytes"])
        row["zstd_19_bytes"] += int(tile["zstd_19_bytes"])
        row["tiles"] += 1
    return dict(sorted(sums.items()))


def _pair_ids(explicit: Sequence[int] | None, count: int) -> list[int]:
    result = [int(x) for x in explicit] if explicit else [int(x) for x in np.linspace(0, 599, count, dtype=np.int64)]
    if not result or len(result) > MAX_SUBSET or len(set(result)) != len(result) or any(x < 0 or x >= 600 for x in result):
        raise JointSolveError(f"pair selection must contain 1..{MAX_SUBSET} unique ids in [0,600)")
    return result


def run_measurement(args: argparse.Namespace) -> dict[str, Any]:
    cache_path, upstream = args.cache.resolve(), args.upstream.resolve()
    output, state, stage_dir = args.output.resolve(), args.state.resolve(), args.stage_dir.resolve()
    if output.exists():
        raise JointSolveError(f"receipt already exists: {output}")
    if any(str(path).startswith(("/tmp/", "/private/tmp/", "/var/tmp/")) for path in (output, state, stage_dir)):
        raise JointSolveError("durable evidence paths may not use tmp")
    pairs = _pair_ids(args.pair_indices, args.sample_pairs)
    if args.seg_band_scale != 0.0 and args.seg_pullback_sidecar is None:
        raise JointSolveError("positive Seg band requires --seg-pullback-sidecar with native winner/rival VJP custody")
    if args.pose_rgb_band != 0.0:
        raise JointSolveError("positive pose bands require a custodied real PoseNet-6 Jacobian sidecar; zero-band control only")
    sacred_before = _stat_tree_snapshot(SACRED)
    fields = _load_cache(cache_path)
    config = {
        "schema": STATE_SCHEMA, "pairs": pairs, "seg_band_scale": args.seg_band_scale,
        "local_lipschitz": args.local_lipschitz, "max_seg_rgb_radius": args.max_seg_rgb_radius,
        "pose_rgb_band": args.pose_rgb_band, "pose_tolerance": args.pose_tolerance,
        "repair_steps": args.repair_steps, "max_nodes_per_block": args.max_nodes_per_block,
        "seg_pullback_sidecar": None if args.seg_pullback_sidecar is None else str(args.seg_pullback_sidecar.resolve()),
        "seg_pullback_sidecar_sha256": None if args.seg_pullback_sidecar is None else _sha256_file(args.seg_pullback_sidecar.resolve()),
        "cache_sha256": _sha256_file(cache_path), "solver_sha256": _sha256_file(SRC / "tac/optimization/joint_seg_pose_rate.py"),
        "tool_sha256": _sha256_file(Path(__file__).resolve()), "predictor": "generated piecewise-constant fill of counted scorer-plane description",
    }
    config_sha = hashlib.sha256(_canonical(config)).hexdigest()
    rows: list[dict[str, Any]] = []
    if args.resume:
        loaded = json.loads(state.read_text())
        if loaded.get("config_sha256") != config_sha:
            raise JointSolveError("resume config/custody mismatch")
        rows = list(loaded.get("rows", []))
        for row in rows:
            stage = Path(row["stage"]["path"])
            if not stage.is_file() or _sha256_file(stage) != row["stage"]["sha256"]:
                raise JointSolveError("resume stage custody mismatch")
    elif state.exists() or (stage_dir.exists() and any(stage_dir.iterdir())):
        raise JointSolveError("preserved state/stages exist; use --resume or new paths")
    else:
        _atomic_json(state, {"schema": STATE_SCHEMA, "config_sha256": config_sha, "config": config, "rows": []})

    operator = DisjointResizeOperator.build(camera_h=CAMERA_HW[0], camera_w=CAMERA_HW[1], scorer_h=SCORER_HW[0], scorer_w=SCORER_HW[1])
    segnet, posenet, torch = _load_scorers(upstream, args.cpu_threads)
    pullback_rows = None if args.seg_pullback_sidecar is None else _load_seg_pullback_sidecar(args.seg_pullback_sidecar.resolve(), pairs)
    completed = {int(row["pair_id"]) for row in rows}
    for pair_id in pairs:
        if pair_id in completed:
            continue
        started = time.monotonic()
        source0 = np.asarray(fields["gt_f0"][pair_id], dtype=np.uint8).copy()
        source1 = np.asarray(fields["gt_f1"][pair_id], dtype=np.uint8).copy()
        labels = np.asarray(fields["lstars"][pair_id], dtype=np.int64).copy()
        margins = np.asarray(fields["margins"][pair_id], dtype=np.float64).copy()
        target_pose = np.asarray(fields["gt_poses"][pair_id], dtype=np.float64).copy()
        source_control, native_winner, native_rival = _hard_verdict(
            segnet, posenet, torch, source0, source1, labels, target_pose
        )
        cache_disagreement = int(np.count_nonzero(native_winner != labels))
        n0, den0 = operator.apply_numerators(source0)
        n1, den1 = operator.apply_numerators(source1)
        y0, y1 = n0.astype(np.float64) / den0, n1.astype(np.float64) / den1
        predictor0, predictor1 = generated_fill_predictor(operator, y0), generated_fill_predictor(operator, y1)
        del source0, source1

        attempt_rows = []
        accepted = None
        solve_seconds = 0.0
        verify_seconds = 0.0
        attempt_count = 1 if args.seg_band_scale == 0.0 and args.pose_rgb_band == 0.0 else args.repair_steps + 1
        for repair in range(attempt_count):
            shrink = 0.5 ** repair
            bound_started = time.monotonic()
            band_config = MarginBandConfig(
                scale=args.seg_band_scale * shrink, local_lipschitz=args.local_lipschitz,
                max_rgb_radius=args.max_seg_rgb_radius,
            )
            if args.seg_band_scale == 0.0:
                band1 = derive_margin_rgb_band(margins, band_config)
            else:
                assert pullback_rows is not None
                pullback = pullback_rows[pair_id]
                if not np.array_equal(pullback["winner"], native_winner) or not np.array_equal(pullback["rival"], native_rival):
                    raise JointSolveError(f"pair {pair_id} pullback arrangement differs from native source control")
                band1 = derive_hyperplane_channel_band(
                    margins, native_winner, native_rival, pullback["pullback"], pullback["pair_norms"], band_config,
                ).channel_radii
            band0 = np.full(SCORER_HW, args.pose_rgb_band * shrink, dtype=np.float64)
            bound_seconds = time.monotonic() - bound_started
            solve_started = time.monotonic()
            solved0 = solve_interval_frame(operator, n0, den0, band0, predictor=predictor0, max_nodes_per_block=args.max_nodes_per_block)
            solved1 = solve_interval_frame(operator, n1, den1, band1, predictor=predictor1, max_nodes_per_block=args.max_nodes_per_block)
            solve_seconds += time.monotonic() - solve_started
            verify_started = time.monotonic()
            verdict, _candidate_winner, _candidate_rival = _hard_verdict(
                segnet, posenet, torch, solved0.frame, solved1.frame, native_winner, target_pose
            )
            verify_seconds += time.monotonic() - verify_started
            passed = verdict["d_seg"] == 0.0 and verdict["d_pose"] <= args.pose_tolerance
            attempt_rows.append({"repair": repair, "shrink": shrink, "bound_seconds": bound_seconds, "hard_oracle": verdict, "PASS": passed})
            if passed:
                accepted = (solved0, solved1, verdict, repair)
                break
        if accepted is None:
            failure_path = stage_dir / f"pair_{pair_id:04d}.hard_oracle_refusal.json"
            _atomic_json(failure_path, {
                "schema": "joint_seg_pose_hard_oracle_refusal.v1", "pair_id": pair_id,
                "config_sha256": config_sha, "source_positive_control": source_control,
                "cached_vs_native_winner_disagreement_pixels": cache_disagreement,
                "attempts": attempt_rows,
                "verdict_scope": "this pair and operating point only; not a formulation-family verdict",
            })
            raise JointSolveError(
                f"pair {pair_id} exhausted hard-oracle repair; durable refusal={failure_path}; "
                f"last={attempt_rows[-1]['hard_oracle']}"
            )
        solved0, solved1, verdict, repair_count = accepted
        rate_started = time.monotonic()
        predictor_num0, predictor_den0 = operator.apply_numerators(predictor0)
        predictor_num1, predictor_den1 = operator.apply_numerators(predictor1)
        if predictor_den0 != den0 or predictor_den1 != den1:
            raise JointSolveError("predictor range-coordinate denominator mismatch")
        rate0 = range_payload_bytes_and_tiles(solved0.chosen_numerators, predictor_num0)
        rate1 = range_payload_bytes_and_tiles(solved1.chosen_numerators, predictor_num1)
        rate_seconds = time.monotonic() - rate_started
        stage_path = stage_dir / f"pair_{pair_id:04d}.json"
        stage_payload = {
            "schema": "joint_seg_pose_pair_stage.v1", "pair_id": pair_id,
            "config_sha256": config_sha, "hard_oracle": verdict,
            "frame0_sha256": _sha256_array(solved0.frame), "frame1_sha256": _sha256_array(solved1.frame),
            "binding0_sha256": _sha256_array(solved0.binding_map), "binding1_sha256": _sha256_array(solved1.binding_map),
            "chosen_numerators0_sha256": _sha256_array(solved0.chosen_numerators),
            "chosen_numerators1_sha256": _sha256_array(solved1.chosen_numerators),
            "winner_sha256": _sha256_array(native_winner), "rival_sha256": _sha256_array(native_rival),
            "reconstruction": "deterministic from frozen cache scorer numerators + config + generated-fill predictor; camera frames are rebuildable and not persisted locally",
        }
        _atomic_json(stage_path, stage_payload)
        stage_sha = _sha256_file(stage_path)
        row = {
            "pair_id": pair_id, "operating_point": {"seg_band_scale": args.seg_band_scale, "pose_rgb_band": args.pose_rgb_band, "pose_tolerance": args.pose_tolerance},
            "source_positive_control": source_control,
            "cached_vs_native_winner_disagreement_pixels": cache_disagreement,
            "hard_oracle": verdict, "hard_oracle_repair_count": repair_count, "attempts": attempt_rows,
            "frame0": {"telemetry": solved0.telemetry.__dict__, "binding_map_sha256": _sha256_array(solved0.binding_map), "rate": {k: v for k, v in rate0.items() if k != "tiles"}, "byte_attribution": _attribution(rate0, None, None, None)},
            "frame1": {"telemetry": solved1.telemetry.__dict__, "binding_map_sha256": _sha256_array(solved1.binding_map), "rate": {k: v for k, v in rate1.items() if k != "tiles"}, "byte_attribution": _attribution(rate1, native_winner, native_rival, margins)},
            "profile_seconds": {"integer_search_and_repair": solve_seconds, "hard_oracle_verify": verify_seconds, "rate_and_tile_compression": rate_seconds, "total": time.monotonic() - started},
            "stage": {"path": str(stage_path), "sha256": stage_sha},
        }
        rows.append(row)
        _atomic_json(state, {"schema": STATE_SCHEMA, "config_sha256": config_sha, "config": config, "rows": rows})

    if _stat_tree_snapshot(SACRED) != sacred_before:
        raise JointSolveError("sacred result tree changed during measurement")
    receipt = {
        "schema": SCHEMA, "written_at_utc": datetime.now(UTC).isoformat(),
        "axis": f"[{platform.system()}-{platform.machine()} CPU advisory subset] NON-PROMOTABLE",
        "authority": {"score_claim": False, "promotion_eligible": False, "pointer": POINTER, "pointer_moved": False,
                      "verdict_scope": "selected real n600-cache pairs; frozen CPU-torch SegNet/PoseNet; no contest CPU/CUDA or receiver archive claim"},
        "labels": {"MEASURED": ["actual Brotli-Q11 residual bytes", "actual zstd-19 residual bytes", "frozen CPU scorer d_seg/d_pose", "stage timings"],
                   "DERIVED": ["rank-4 winner/rival hyperplane pullback channel bands", "pose derivative 5/sqrt(10d)", "crossover d_pose=2.5e-4"],
                   "INFERRED": []},
        "config": config, "config_sha256": config_sha, "pairs": rows,
        "receiver_arithmetic_declaration": {"dtype": "native float32", "semantics": "CPU-Torch conv/eval", "tie_policy": "authority scorer native argmax; generic-f64 is not substituted", "declared": True},
        "range_kernel_rate_law": {"counted_coordinates": "range(A) scorer numerator residual only", "ker_A_payload_bytes": 0, "ker_A_fill": "generated deterministically from declared decoder predictor and lattice solve", "camera_residual_is_not_serialized": True},
        "aggregate": {"pair_count": len(rows), "unique_pair_ids": sorted(int(r["pair_id"]) for r in rows),
                      "mean_d_seg": float(np.mean([r["hard_oracle"]["d_seg"] for r in rows])),
                      "mean_d_pose": float(np.mean([r["hard_oracle"]["d_pose"] for r in rows])),
                      "total_brotli_q11_bytes": int(sum(r[f]["rate"]["brotli_q11_bytes"] for r in rows for f in ("frame0", "frame1"))),
                      "total_zstd_19_bytes": int(sum(r[f]["rate"]["zstd_19_bytes"] for r in rows for f in ("frame0", "frame1")))},
        "resumability": {"state": str(state), "stage_dir": str(stage_dir), "all_stage_checkpoints_preserved": True,
                         "checkpoint_form": "small write-once custody manifests; deterministic candidates rebuild from frozen cache and config"},
        "sacred_tree_unchanged": True,
    }
    _atomic_json(output, receipt)
    return receipt


def compose(receipts: Sequence[Path], output: Path) -> dict[str, Any]:
    if output.exists():
        raise JointSolveError(f"composed receipt already exists: {output}")
    docs = [json.loads(path.read_text()) for path in receipts]
    if not docs or any(doc.get("schema") != SCHEMA for doc in docs):
        raise JointSolveError("all inputs must be joint receipt v1")
    rows = [row for doc in docs for row in doc["pairs"]]
    pair_obs = {(int(row["pair_id"]), json.dumps(row["operating_point"], sort_keys=True)) for row in rows}
    if len(pair_obs) != len(rows):
        raise JointSolveError("duplicate pair/operating-point observation in composition")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[json.dumps(row["operating_point"], sort_keys=True)].append(row)
    curves = []
    for key, group in sorted(grouped.items()):
        curves.append({"operating_point": json.loads(key), "pair_count": len(group),
                       "bytes": float(np.mean([sum(row[f]["rate"]["brotli_q11_bytes"] for f in ("frame0", "frame1")) for row in group])),
                       "d_seg": float(np.mean([row["hard_oracle"]["d_seg"] for row in group])),
                       "d_pose": float(np.mean([row["hard_oracle"]["d_pose"] for row in group]))})
    seg_curve = [{"bytes": x["bytes"], "distortion": x["d_seg"]} for x in curves]
    pose_curve = [{"bytes": x["bytes"], "distortion": x["d_pose"]} for x in curves]
    result = {"schema": "joint_seg_pose_inverse_rate_composed.v1", "written_at_utc": datetime.now(UTC).isoformat(),
              "authority": {"score_claim": False, "pointer": POINTER, "verdict_scope": "composed real-cache advisory chunks only"},
              "source_receipts": [{"path": str(p.resolve()), "sha256": _sha256_file(p)} for p in receipts],
              "observation_count": len(rows), "unique_pair_count": len({int(r["pair_id"]) for r in rows}),
              "measured_curves": curves, "waterfill": solve_measured_waterfill(seg_curve, pose_curve)}
    _atomic_json(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--stage-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sample-pairs", type=int, default=12)
    parser.add_argument("--pair-indices", nargs="+", type=int)
    parser.add_argument("--seg-pullback-sidecar", type=Path)
    parser.add_argument("--seg-band-scale", type=float, default=0.0)
    parser.add_argument("--local-lipschitz", type=float, default=1.0)
    parser.add_argument("--max-seg-rgb-radius", type=float, default=8.0)
    parser.add_argument("--pose-rgb-band", type=float, default=0.0)
    parser.add_argument("--pose-tolerance", type=float, default=1e-8)
    parser.add_argument("--repair-steps", type=int, default=4)
    parser.add_argument("--max-nodes-per-block", type=int, default=4096)
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.compose:
        result = compose(args.compose, args.output.resolve())
    else:
        if args.state is None or args.stage_dir is None:
            raise SystemExit("measurement requires --state and --stage-dir")
        result = run_measurement(args)
    print(json.dumps({"output": str(args.output.resolve()), "schema": result["schema"]}, sort_keys=True))


if __name__ == "__main__":
    main()
