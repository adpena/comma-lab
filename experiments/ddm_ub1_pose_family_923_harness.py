# SPDX-License-Identifier: MIT
"""UB1 #923 pose-family recovery harness.

This module is deliberately split into scorer-free planning/cache validation and
scorer-consuming measurement commands. UB1 uses only the former. The latter is a
recovered runnable surface for the next governed scorer slot; it refuses missing
depth/texture sidecars instead of silently substituting another formulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

DEFAULT_SELECTION = REPO / ".omx/research/ddm_na3_20260805/stratified_pose_selection_923.json"
DEFAULT_GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DEFAULT_CHECKPOINT = (
    REPO / "experiments/results/levelset_n600_crucible_v6_run1_20260708T095730Z"
    / "levelset_witness_ema_mlx.npz"
)
DEFAULT_OUTDIR = REPO / ".omx/research/ddm_ub1_20260805"
DEFAULT_RENDER_CACHE = (
    DEFAULT_OUTDIR / "pose923_run1_stratified_n120_oracle_render_cache.npz"
)

SOURCE_RECEIPTS: dict[str, Path] = {
    "pose_l2_truedepth": REPO / ".omx/research/pose_l2_truedepth_probe_measured_20260708.md",
    "pose_carrier_arms": REPO / ".omx/research/pose_carrier_arms_measured_20260708.md",
    "pose_mladder_depthwarp": REPO / ".omx/research/pose_mladder_depthwarp_measured_20260708.md",
    "pose_stratified_texture": REPO / ".omx/research/pose_stratified_texture_probe_measured_20260708.md",
}

LOST_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "pose_l2_truedepth": (
        "pose_l2_truedepth_probe.py",
        "l2_n24.json",
        "l2_n8.json",
        "l2_depths_n24.npz",
    ),
    "pose_carrier_arms": (
        "pose_mladder.py",
        "renders_n24.npz",
    ),
    "pose_mladder_depthwarp": (
        "pose_mladder.py",
        "a2_n24.jsonl",
        "a2plus_n8.jsonl",
        "renders_n24.npz",
    ),
    "pose_stratified_texture": (
        "pose_stratified_texture_probe.py",
        "pose_aperture_probe.py",
        "a1t_grid_n24.json",
        "scale_sweep_n24.json",
        "renders_n24.npz",
    ),
}

FORBIDDEN_TMP_PREFIXES = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")


class HarnessRecoveryError(RuntimeError):
    """Raised when a requested recovered surface is not honestly runnable."""


@dataclass(frozen=True)
class SelectionSpec:
    path: Path
    indices: tuple[int, ...]
    provenance: dict[str, Any]

    @property
    def n(self) -> int:
        return len(self.indices)

    @property
    def max_pair(self) -> int:
        return max(self.indices)


def _refuse_tmp(path: Path, field: str) -> None:
    s = str(path)
    if any(s.startswith(prefix) for prefix in FORBIDDEN_TMP_PREFIXES):
        raise ValueError(f"{field}={path} is a /tmp-class path; use repo or SSD tier")


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path_record(path: Path, *, hash_limit_bytes: int = 64_000_000) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "bytes": None, "sha256": None}
    size = path.stat().st_size
    if size > hash_limit_bytes:
        return {
            "path": str(path),
            "exists": True,
            "bytes": size,
            "sha256": None,
            "sha256_status": f"skipped_large_input_over_{hash_limit_bytes}_bytes",
        }
    return {"path": str(path), "exists": True, "bytes": size, "sha256": _sha256_path(path)}


def load_selection(path: Path = DEFAULT_SELECTION) -> SelectionSpec:
    rec = json.loads(Path(path).read_text())
    sel = rec.get("selection", {})
    indices = tuple(int(i) for i in sel.get("indices", ()))
    if not indices:
        raise HarnessRecoveryError(f"{path} does not carry selection.indices")
    if len(indices) != int(sel.get("n", len(indices))):
        raise HarnessRecoveryError(
            f"{path} selection length mismatch: n={sel.get('n')} indices={len(indices)}"
        )
    if int(sel.get("population", 0)) < max(indices) + 1:
        raise HarnessRecoveryError(f"{path} selection population does not contain max index")
    return SelectionSpec(path=Path(path), indices=indices, provenance=sel)


def _source_custody() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item, path in SOURCE_RECEIPTS.items():
        rows.append(
            {
                "item": item,
                "path": str(path.relative_to(REPO)),
                "exists": path.is_file(),
                "sha256": _sha256_path(path) if path.is_file() else None,
            }
        )
    return rows


def _checkpoint_self_orient_overrides(checkpoint: Path) -> dict[str, Any]:
    with np.load(checkpoint, allow_pickle=False) as z:
        return {
            "freq_across": float(z["__cfg_freq_across"]) if "__cfg_freq_across" in z.files else 32.0,
            "freq_along": float(z["__cfg_freq_along"]) if "__cfg_freq_along" in z.files else 4.0,
            "tau": 4.0,
            "iters": 4,
        }


def load_xi_effective(checkpoint: Path = DEFAULT_CHECKPOINT) -> np.ndarray:
    with np.load(checkpoint, allow_pickle=False) as z:
        if "pose_carrier.xi_stored" not in z.files or "pose_carrier.dxi" not in z.files:
            raise HarnessRecoveryError(
                f"{checkpoint} lacks pose_carrier.xi_stored/dxi; cannot recover geometric arms"
            )
        return (
            np.asarray(z["pose_carrier.xi_stored"], dtype=np.float64)
            + np.asarray(z["pose_carrier.dxi"], dtype=np.float64)
        )


def build_fire_orders(
    *,
    selection_path: Path = DEFAULT_SELECTION,
    gt_cache: Path = DEFAULT_GT_CACHE,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    render_cache: Path = DEFAULT_RENDER_CACHE,
) -> dict[str, Any]:
    sel = load_selection(selection_path)
    cache_cmd = (
        ".venv/bin/python experiments/ddm_ub1_pose_family_923_harness.py build-render-cache "
        f"--selection {selection_path} --checkpoint {checkpoint} --out-cache {render_cache}"
    )
    base = (
        ".venv/bin/python experiments/ddm_ub1_pose_family_923_harness.py score "
        f"--selection {selection_path} --gt-cache {gt_cache} --render-cache {render_cache}"
    )
    orders = [
        {
            "id": "gr1_stratified_n48_selector_then_rerace",
            "status": "READY_SELECTOR_BUILT_NOT_RUN",
            "exact_argv": (
                ".venv/bin/python experiments/ddm_gr1_granularity_rerace.py "
                "--mode sweep --pairs 48 --selection-mode stratified --selection-seed 20260805 "
                "--outdir /Volumes/VertigoDataTier/pact/ddm_gr1_20260730"
            ),
            "notes": "Scorer-consuming command; UB1 only builds selector metadata.",
        },
        {
            "id": "pose923_oracle_render_cache_n120",
            "status": "READY_CACHE_BUILD_NOT_RUN",
            "exact_argv": cache_cmd,
            "notes": "Scorer-free but expensive; builds selected uint8 camera frames through the byte-close oracle.",
        },
        {
            "id": "pose_carrier_arms_stratified_n120_retest",
            "status": "READY_REBUILT_FROM_RECEIPT_NOT_RUN",
            "exact_argv": f"{base} --item pose_carrier_arms",
            "source_path": str(SOURCE_RECEIPTS["pose_carrier_arms"].relative_to(REPO)),
        },
        {
            "id": "pose_mladder_depthwarp_a0_stratified_n120_retest",
            "status": "PARTIAL_READY_A0_REBUILT_FROM_RECEIPT_NOT_RUN",
            "exact_argv": f"{base} --item pose_mladder_depthwarp --rungs A0",
            "source_path": str(SOURCE_RECEIPTS["pose_mladder_depthwarp"].relative_to(REPO)),
            "bounded_gap": "A2/A2+ solve logs and original solver harness were not recovered in searched scope.",
        },
        {
            "id": "pose_l2_truedepth_stratified_n120_retest",
            "status": "BLOCKED_DEPTH_CACHE_ABSENT",
            "exact_argv": f"{base} --item pose_l2_truedepth --depth-cache <RECOVERED_L2_DEPTHS_N600_NPZ>",
            "source_path": str(SOURCE_RECEIPTS["pose_l2_truedepth"].relative_to(REPO)),
            "bounded_gap": "l2_depths_n24.npz and true-depth flow harness were not recovered in searched scope.",
        },
        {
            "id": "pose_stratified_texture_stratified_n120_retest",
            "status": "BLOCKED_TEXTURE_GRID_ABSENT",
            "exact_argv": f"{base} --item pose_stratified_texture --texture-grid <RECOVERED_A1T_GRID_JSON>",
            "source_path": str(SOURCE_RECEIPTS["pose_stratified_texture"].relative_to(REPO)),
            "bounded_gap": "a1t_grid_n24.json, scale_sweep_n24.json, and texture harness were not recovered in searched scope.",
        },
    ]
    return {
        "schema": "ddm_ub1.pose_family_923_fire_orders.v1",
        "score_claim": False,
        "scorer_runs_by_ub1": 0,
        "selection": sel.provenance,
        "selection_path": str(selection_path),
        "selection_sha256": _sha256_path(selection_path),
        "gt_cache": _path_record(gt_cache),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256_path(checkpoint) if checkpoint.is_file() else None,
            "xi_effective_available": checkpoint.is_file(),
            "self_orient_overrides": _checkpoint_self_orient_overrides(checkpoint)
            if checkpoint.is_file()
            else None,
        },
        "source_receipts": _source_custody(),
        "lost_artifacts_searched": LOST_ARTIFACTS,
        "fire_orders": orders,
    }


def write_fire_orders(out: Path, **kwargs: Any) -> Path:
    _refuse_tmp(out, "out")
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = build_fire_orders(**kwargs)
    out.write_text(json.dumps(rec, indent=1) + "\n")
    return out


def build_render_cache(
    *,
    selection_path: Path = DEFAULT_SELECTION,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    out_cache: Path = DEFAULT_RENDER_CACHE,
) -> Path:
    """Build selected witness frame0/frame1 cache via the byte-close oracle path.

    This is scorer-free but compute-heavy. It renders through the canonical
    level-set oracle, then writes only the selected pairs plus xi_eff.
    """

    _refuse_tmp(out_cache, "out_cache")
    sel = load_selection(selection_path)
    from tools.levelset_byte_close_and_eval import (
        _dequant_blob,
        _load_levelset_ckpt,
        build_levelset_blob,
        detect_self_orient,
        numpy_oracle_reference_frames,
    )

    params, cfg = _load_levelset_ckpt(checkpoint.parent, checkpoint.name)
    so = detect_self_orient(cfg, _checkpoint_self_orient_overrides(checkpoint))
    blob, _manifest = build_levelset_blob(params, cfg, so, pose_sidecar=None)
    manifest, dparams, code, lane_pairs, pose_carrier, chart_payload = _dequant_blob(blob)
    frames, argmaxes = numpy_oracle_reference_frames(
        dparams,
        code,
        manifest,
        sel.max_pair + 1,
        lane_pairs=lane_pairs,
        pose_carrier=pose_carrier,
        chart_payload=chart_payload,
    )
    f0 = np.stack([frames[2 * i] for i in sel.indices]).astype(np.uint8)
    f1 = np.stack([frames[2 * i + 1] for i in sel.indices]).astype(np.uint8)
    argmax = np.stack([argmaxes[i] for i in sel.indices]).astype(np.int64)
    xi = load_xi_effective(checkpoint)[list(sel.indices)].astype(np.float32)
    out_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_cache,
        schema=np.asarray("ddm_ub1.pose923_oracle_render_cache.v1"),
        pair_indices=np.asarray(sel.indices, dtype=np.int64),
        witness_f0=f0,
        witness_f1=f1,
        witness_argmax=argmax,
        xi_effective=xi,
        selection_sha256=np.asarray(_sha256_path(selection_path)),
        checkpoint_sha256=np.asarray(_sha256_path(checkpoint)),
    )
    return out_cache


def _load_cache(cache: Path, sel: SelectionSpec) -> dict[str, np.ndarray]:
    with np.load(cache, allow_pickle=False) as z:
        pairs = tuple(int(i) for i in np.asarray(z["pair_indices"], dtype=np.int64).tolist())
        if pairs != sel.indices:
            raise HarnessRecoveryError(
                f"{cache} pair_indices do not match {sel.path}; cache={pairs[:8]}... "
                f"selection={sel.indices[:8]}..."
            )
        return {k: np.asarray(z[k]) for k in z.files}


def score_rebuilt_item(
    *,
    item: str,
    selection_path: Path = DEFAULT_SELECTION,
    gt_cache: Path = DEFAULT_GT_CACHE,
    render_cache: Path = DEFAULT_RENDER_CACHE,
    rungs: tuple[str, ...] = ("A0",),
) -> dict[str, Any]:
    if item in {"pose_l2_truedepth", "pose_stratified_texture"}:
        raise HarnessRecoveryError(
            f"{item} is not honestly runnable from landed bytes: required lost artifacts "
            f"{LOST_ARTIFACTS[item]} remain absent; see plan/fire-order status."
        )
    if item not in {"pose_carrier_arms", "pose_mladder_depthwarp"}:
        raise HarnessRecoveryError(f"unknown item {item!r}")
    sel = load_selection(selection_path)
    cache = _load_cache(render_cache, sel)

    from experiments.train_witness_realized_through_R_mlx import cpu_verdict_d_pose_batch, load_gt_from_cache
    from tac.boundary_math.warp_real_luma_frame0 import GroundHomographyGeom, warp_frame0_uint8_numpy

    gt, _seg, posenet_cpu = load_gt_from_cache(gt_cache, sel.max_pair + 1)
    gt_f0 = [np.asarray(gt.gt_f0[i], dtype=np.uint8) for i in sel.indices]
    poses = [np.asarray(gt.gt_poses[i], dtype=np.float64) for i in sel.indices]
    witness_f0 = [np.asarray(x, dtype=np.uint8) for x in cache["witness_f0"]]
    witness_f1 = [np.asarray(x, dtype=np.uint8) for x in cache["witness_f1"]]
    xi = np.asarray(cache["xi_effective"], dtype=np.float64)
    geom = GroundHomographyGeom.eon()

    rows: list[dict[str, Any]] = []
    if item == "pose_carrier_arms":
        arms = {
            "store_nothing": (witness_f0, witness_f1),
            "real_f0_plus_witness_f1": (gt_f0, witness_f1),
            "warp_real_luma": (
                [warp_frame0_uint8_numpy(src, x, geom) for src, x in zip(gt_f0, xi, strict=True)],
                witness_f1,
            ),
        }
        for name, (f0s, f1s) in arms.items():
            vals = cpu_verdict_d_pose_batch(posenet_cpu, f0s, f1s, poses)
            rows.append({"arm": name, "mean_d_pose": float(np.mean(vals)), "median_d_pose": float(np.median(vals))})
    else:
        if tuple(rungs) != ("A0",):
            raise HarnessRecoveryError("Only the A0 depthwarp rung was rebuilt; A2/A2+ solver logs are absent.")
        f0s = [warp_frame0_uint8_numpy(src, x, geom) for src, x in zip(witness_f0, xi, strict=True)]
        vals = cpu_verdict_d_pose_batch(posenet_cpu, f0s, witness_f1, poses)
        rows.append({"arm": "A0", "mean_d_pose": float(np.mean(vals)), "median_d_pose": float(np.median(vals))})

    return {
        "schema": "ddm_ub1.pose_family_923_rebuilt_measurement.v1",
        "score_claim": False,
        "item": item,
        "selection": sel.provenance,
        "render_cache": str(render_cache),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    p_plan.add_argument("--gt-cache", type=Path, default=DEFAULT_GT_CACHE)
    p_plan.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p_plan.add_argument("--render-cache", type=Path, default=DEFAULT_RENDER_CACHE)
    p_plan.add_argument("--out", type=Path, default=None)

    p_cache = sub.add_parser("build-render-cache")
    p_cache.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    p_cache.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p_cache.add_argument("--out-cache", type=Path, default=DEFAULT_RENDER_CACHE)

    p_score = sub.add_parser("score")
    p_score.add_argument("--item", choices=sorted(SOURCE_RECEIPTS), required=True)
    p_score.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    p_score.add_argument("--gt-cache", type=Path, default=DEFAULT_GT_CACHE)
    p_score.add_argument("--render-cache", type=Path, default=DEFAULT_RENDER_CACHE)
    p_score.add_argument("--depth-cache", type=Path, default=None)
    p_score.add_argument("--texture-grid", type=Path, default=None)
    p_score.add_argument("--rungs", default="A0")
    p_score.add_argument("--out", type=Path, default=None)

    args = ap.parse_args(argv)
    if args.cmd == "plan":
        rec = build_fire_orders(
            selection_path=args.selection,
            gt_cache=args.gt_cache,
            checkpoint=args.checkpoint,
            render_cache=args.render_cache,
        )
        if args.out:
            _refuse_tmp(args.out, "out")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(rec, indent=1) + "\n")
        print(json.dumps(rec, indent=1), flush=True)
        return 0
    if args.cmd == "build-render-cache":
        out = build_render_cache(
            selection_path=args.selection,
            checkpoint=args.checkpoint,
            out_cache=args.out_cache,
        )
        print(json.dumps({"render_cache": str(out), "sha256": _sha256_path(out)}, indent=1), flush=True)
        return 0
    rungs = tuple(x.strip() for x in args.rungs.split(",") if x.strip())
    rec = score_rebuilt_item(
        item=args.item,
        selection_path=args.selection,
        gt_cache=args.gt_cache,
        render_cache=args.render_cache,
        rungs=rungs,
    )
    if args.out:
        _refuse_tmp(args.out, "out")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
