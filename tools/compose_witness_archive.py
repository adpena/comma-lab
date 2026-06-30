#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""compose_witness_archive — the single automated entry point for the v2 hybrid-witness pipeline.

Composes already-MEASURED/BUILT components into the v2 hybrid witness, in two phases around the one
unavoidable GPU step (the residual-INR run). Spec:
``.omx/research/v2_coherent_automated_composition_pipeline_spec_20260630T201213Z.md``.

PHASE-A (CPU / $0 / no GPU / no launch):
  1. load baseline GT cache (lstars, gt_f1, gt_poses)
  2. Step 2: encode_known_split(warp-recoverability, reach k*) -> KNOWN per-class STORE/GENERATE/LEARN
  3. Step 3: generate the deterministic bulk argmax THROUGH R (proven warp + render + frozen SegNet)
  4. Step 4: compute + save the RESIDUAL TARGET (residual = GT - bulk_through_R; bulk SUBTRACTED)
  5. Step 5: build the dual-use pose sidecar (from the cache's gt_poses; $0, no PoseNet re-run)
  6. EMIT the flag-validated residual-INR launch command (HOLD for operator GO; NOT fired)

[operator GO -> the residual INR trains on the one GPU, resumable/per-stage -- NOT this tool's job]

PHASE-B (CPU / $0):
  7. Step 6: assemble the 4-section archive.zip (store + residual + pose + manifest) + MLX-free
     inflate.py; run inflate (DETERMINISTIC FLOOR: empty residual) -> realized d_seg through R
     (advisory) -> compute_contest_score -> emit the STAGED dual CPU/CUDA upstream/evaluate.py command

means != ends (NO-FAKE): this is composition PLUMBING. The pointer (0.19110) moves ONLY when the
residual-INR run lands + the 4-section archive byte-closes + exact-evals below 0.19110. Every number
emitted is ``[advisory] NON-PROMOTABLE``. NO GPU, NO training launch, NO touching the live n600 run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "src", _REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.contest_score import compute_contest_score, rate_term  # noqa: E402
from tac.v2_compose import bulk_generator as bg  # noqa: E402
from tac.v2_compose import archive_grammar as ag  # noqa: E402
from tac.v2_compose import launch_command as lc  # noqa: E402
from tac.v2_compose import pose_sidecar as ps  # noqa: E402
from tac.v2_compose import residual_target as rt  # noqa: E402
from tac.v2_compose.store_learn_split import (  # noqa: E402
    WARP_GROUND_HOMOGRAPHY,
    WARP_IDENTITY,
    WARP_ROTATION_ONLY,
    DECISION_LEARN,
    encode_known_split,
    load_reach_kstar,
    load_warp_recoverability_from_grok,
)

_D_POSE_SIDECAR = 3.4e-5  # the d_pose the stored sidecar enables (FEED-lj; advisory budget)
_DEF_CACHE = "experiments/results/mlx_fleet_gt_cache/gt_n96.npz"
_DEF_GROK = "experiments/results/grok_pose_warp_dseg_20260629T181000Z/results.json"
_DEF_REACH = "experiments/results/screw_reach/reach_n96.json"

# map the encode_known_split warp-type -> the store_blob warp-code (per-class).
_WARP_NAME_TO_CODE = {
    WARP_GROUND_HOMOGRAPHY: ag.WARP_TYPE_CODE["ground_homography"],
    WARP_IDENTITY: ag.WARP_TYPE_CODE["identity"],
    WARP_ROTATION_ONLY: ag.WARP_TYPE_CODE["rotation_only"],
    None: ag.WARP_TYPE_CODE["learn"],
}


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (_REPO / pp)


def phase_a(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    """PHASE-A: plan + deterministic bulk + residual target + pose sidecar + launch command."""
    cache = _resolve(args.gt_cache)
    print(f"[phase-A] loading GT cache {cache} ...", file=sys.stderr)
    z = np.load(cache, allow_pickle=False)
    lstars = np.asarray(z["lstars"], dtype=np.int64)
    gt_f1 = np.asarray(z["gt_f1"])
    gt_poses = np.asarray(z["gt_poses"], dtype=np.float64)
    n_pairs = int(lstars.shape[0])

    # --- Step 2: encode the KNOWN split ---
    rec = load_warp_recoverability_from_grok(_resolve(args.grok_results))
    kstar = load_reach_kstar(_resolve(args.reach_json))
    plan = encode_known_split(rec, kstar, n_pairs=args.n_pairs_clip)
    print(f"[phase-A] split: GENERATE={plan.generate_classes} LEARN={plan.learn_classes} "
          f"keyframes={plan.keyframe_count} reach_k*={kstar}", file=sys.stderr)

    # --- Step 3: deterministic bulk through R (proven warp path) ---
    cfg = bg.load_calibration_from_reach(_resolve(args.reach_json), reach_kstar=kstar, n_pairs=n_pairs)
    print(f"[phase-A] generating deterministic bulk through R for {n_pairs} pairs "
          f"(calib s_t={cfg.s_t:.5f} s_r={cfg.s_r} pitch={cfg.pitch:.4f}) ...", file=sys.stderr)
    bulk = bg.generate_bulk_argmax_stack(lstars, gt_f1, gt_poses, cfg, verbose=True)
    print(f"[phase-A] bulk d_seg floor: full={bulk.bulk_dseg_full:.4f} bulk={bulk.bulk_dseg_bulk:.4f} "
          f"| SegNet selfcheck ok={bulk.k0_faithful} (max_px={bulk.segnet_selfcheck_max_px}) "
          f"| k0 R1-floor={bulk.k0_bulk_mean:.4f} (within n96 ref={bulk.k0_within_r1_ref})", file=sys.stderr)
    if not bulk.k0_faithful:
        print("[phase-A] WARNING: SegNet selfcheck FAILED (gt_f1->argmax != cached lstars) — the "
              "render->R->SegNet path is NOT the contest path; treat ALL bulk numbers as suspect.",
              file=sys.stderr)

    # --- Step 4: residual target (bulk SUBTRACTED) ---
    residual = rt.compute_residual_target(bulk.bulk_argmax_through_R, lstars)
    residual_path = out_dir / "residual_target.npz"
    residual_bytes = rt.save_residual_target(residual, residual_path)
    print(f"[phase-A] residual target: bulk_dseg_floor={residual.bulk_dseg:.4f} "
          f"residual_classes_ranked={residual.residual_classes_ranked[:3]} "
          f"-> {residual_path} ({residual_bytes} B training artifact)", file=sys.stderr)

    # --- Step 5: pose sidecar (dual-use; $0 from cache poses) ---
    pose_path = out_dir / "posenet_targets.bin"
    pose_bytes = ps.build_pose_sidecar_from_cache_poses(gt_poses, pose_path)
    print(f"[phase-A] pose sidecar: {pose_bytes} B -> {pose_path}", file=sys.stderr)

    # --- the REAL store-blob byte measurement (keyframes contour-coded + palette + calib + mask) ---
    warp_codes = [_WARP_NAME_TO_CODE[plan.per_class[c].warp_type] if c in plan.per_class else 3
                  for c in ("Road", "Lane", "Undriv", "Movable", "MyCar")]
    store_blob = ag.build_store_blob(
        keyframe_indices=bulk.keyframes,
        keyframe_lstars=bulk.keyframe_lstars,
        palette=bulk.palette,
        calib=cfg.params,
        warp_type_codes=warp_codes,
        reach_kstar=kstar,
        n_pairs=n_pairs,
    )
    store_bytes = len(store_blob)
    # NOTE: keyframes were selected on the CACHE subset (n_pairs); for the full 600-pair clip the
    # store scales by (600/n_pairs) keyframes. Report both the measured-subset + the projected-600.
    proj_kf_full = plan.keyframe_count
    measured_kf = len(bulk.keyframes)
    store_bytes_proj600 = int(round(store_bytes * (proj_kf_full / max(measured_kf, 1))))

    # --- Step 6 (emit): the flag-validated residual-INR launch command (HOLD) ---
    residual_run_dir = str(_resolve(args.residual_run_dir)) if args.residual_run_dir else str(out_dir / "residual_inr_run")
    launch = lc.build_residual_inr_command(
        out_dir=residual_run_dir,
        gt_cache=str(cache),
        num_pairs=args.n_pairs_clip,
        epochs=args.residual_epochs,
        seed=args.seed,
        hidden_dim=args.residual_hidden_dim,
        mod_dim=args.residual_mod_dim,
        strict=True,
    )
    print(f"[phase-A] residual-INR launch command FLAG-VALIDATED "
          f"(all_valid={launch.all_flags_valid}); HOLD for operator GO.", file=sys.stderr)

    # --- byte budget arithmetic ---
    known_store = store_bytes_proj600 + pose_bytes
    budget = {
        "store_blob_bytes_measured_subset": store_bytes,
        "store_keyframes_measured": measured_kf,
        "store_blob_bytes_projected_600": store_bytes_proj600,
        "projected_keyframes_600": proj_kf_full,
        "pose_sidecar_bytes": pose_bytes,
        "known_store_total_bytes_proj600": known_store,
        "known_store_rate_proj600": rate_term(known_store),
        "residual_inr_bytes": None,  # OPEN — the GPU run
        "free_generated_bytes": 0,
        "deterministic_bulk_dseg_floor": residual.bulk_dseg,
        "break_even_d_seg_at_known_store": float(
            (0.19110 - (10.0 * _D_POSE_SIDECAR) ** 0.5 - rate_term(known_store)) / 100.0
        ),
        "sub015_d_seg_at_known_store": float(
            (0.15 - (10.0 * _D_POSE_SIDECAR) ** 0.5 - rate_term(known_store)) / 100.0
        ),
    }

    report = {
        "phase": "A",
        "tool": "compose_witness_archive",
        "utc": _utc(),
        "authority": "[macOS-CPU advisory] NON-PROMOTABLE",
        "score_claim": False,
        "promotable": False,
        "frontier_pointer": "UNMOVED 0.19110",
        "gt_cache": str(cache),
        "n_pairs_cache": n_pairs,
        "n_pairs_clip": args.n_pairs_clip,
        "split": plan.to_json(),
        "bulk": {
            "d_seg_full": bulk.bulk_dseg_full,
            "d_seg_bulk_classes": bulk.bulk_dseg_bulk,
            "segnet_selfcheck_ok": bulk.k0_faithful,
            "segnet_selfcheck_max_px": bulk.segnet_selfcheck_max_px,
            "k0_r1_floor_this_cache": bulk.k0_bulk_mean,
            "k0_within_n96_ref": bulk.k0_within_r1_ref,
            "keyframes": bulk.keyframes,
        },
        "residual_target": {**residual.to_summary(), "artifact": str(residual_path), "artifact_bytes": residual_bytes},
        "pose_sidecar": ps.pose_sidecar_byte_cost(pose_path),
        "byte_budget": budget,
        "residual_inr_launch_command": launch.to_json(),
        "the_open_quantity": (
            "Can a SMALL residual INR close d_seg from the "
            f"{residual.bulk_dseg:.4f} deterministic floor to ~{budget['break_even_d_seg_at_known_store']:.5f} "
            "(beat 0.19110) at a small byte cost? Settled ONLY by the GPU residual run + byte-closed exact eval."
        ),
        "means_not_ends": (
            "PHASE-A built the composition + emitted the residual-INR command (a MEANS). The pointer "
            "moves only when that run lands + the 4-section archive byte-closes + evaluate.py < 0.19110."
        ),
    }
    return report


def phase_b(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    """PHASE-B: assemble the 4-section archive (DETERMINISTIC FLOOR: empty residual), inflate,
    realized d_seg through R (advisory), emit the staged dual CPU/CUDA eval command."""
    cache = _resolve(args.gt_cache)
    z = np.load(cache, allow_pickle=False)
    lstars = np.asarray(z["lstars"], dtype=np.int64)
    gt_f1 = np.asarray(z["gt_f1"])
    gt_poses = np.asarray(z["gt_poses"], dtype=np.float64)
    n_pairs = int(lstars.shape[0])
    kstar = load_reach_kstar(_resolve(args.reach_json))
    cfg = bg.load_calibration_from_reach(_resolve(args.reach_json), reach_kstar=kstar, n_pairs=n_pairs)

    palette = bg.compute_class_mean_palette(gt_f1, lstars)
    keyframes = bg.select_keyframes(n_pairs, kstar)
    warp_codes = [0, 3, 2, 3, 1]  # Road=ground, Lane=learn, Undriv=rot, Movable=learn, MyCar=identity

    store_blob = ag.build_store_blob(
        keyframe_indices=keyframes,
        keyframe_lstars=lstars[keyframes].copy(),
        palette=palette,
        calib=cfg.params,
        warp_type_codes=warp_codes,
        reach_kstar=kstar,
        n_pairs=n_pairs,
    )
    pose_path = out_dir / "posenet_targets.bin"
    if not pose_path.exists():
        ps.build_pose_sidecar_from_cache_poses(gt_poses, pose_path)
    pose_blob = pose_path.read_bytes()

    # DETERMINISTIC FLOOR archive: empty residual (the residual-INR weights are the GPU run).
    residual_blob = b""
    if args.residual_inr_weights:
        raise SystemExit(
            "PHASE-B with --residual-inr-weights is NEEDS-WIRING: the residual-INR section format + "
            "inflate compose hook are the GPU-side trainer change (not yet wired). Byte-close the "
            "DETERMINISTIC FLOOR (empty residual) until the residual run + compose mode land. (NO-FAKE.)"
        )

    manifest = {
        "format_version": "v2.0",
        "n_pairs": n_pairs,
        "n_classes": 5,
        "reach_kstar": kstar,
        "keyframe_count": len(keyframes),
        "residual_inr_present": False,
        "mlx_free_inflate": True,
        "loads_no_scorers_at_inflate": True,
        "note": "deterministic FLOOR archive (empty residual); the residual INR closes d_seg + d_pose.",
    }
    manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode("utf-8")

    blob = ag.pack_v2_archive(store_blob, residual_blob, pose_blob, manifest_bytes)
    packet_dir = out_dir / "packet"
    zip_path, zip_bytes = ag.assemble_v2_packet(blob, packet_dir)
    acct = ag.byte_accounting(zip_path, store_blob, residual_blob, pose_blob, manifest_bytes)
    print(f"[phase-B] archive.zip = {zip_bytes} B (rate {acct['rate']:.5f}); sections={acct['section_bytes']}",
          file=sys.stderr)

    realized: dict[str, Any] = {"skipped": bool(args.skip_inflate)}
    if not args.skip_inflate:
        realized = _inflate_and_realize(packet_dir, lstars, max_pairs=args.max_pairs)

    d_seg = realized.get("d_seg_realized", None)
    s_budget = None
    if d_seg is not None:
        # S budget = realized FLOOR d_seg + the sidecar d_pose BUDGET + the real rate. The flat-bulk
        # realized d_pose is high (the residual run closes it); we report the achievable-target budget.
        s_budget = compute_contest_score(d_seg, _D_POSE_SIDECAR, zip_bytes)

    contest_cmd = (
        f".venv/bin/python experiments/contest_auth_eval.py "
        f"--archive {zip_path} --inflate-sh {packet_dir / 'inflate.sh'} "
        f"--device cpu  # [contest-CPU] authoritative ONLY on Linux x86_64; then --device cuda on T4"
    )

    return {
        "phase": "B",
        "tool": "compose_witness_archive",
        "utc": _utc(),
        "authority": "[advisory] NON-PROMOTABLE",
        "score_claim": False,
        "promotable": False,
        "frontier_pointer": "UNMOVED 0.19110",
        "archive_zip": str(zip_path),
        "byte_accounting": acct,
        "realized_floor": realized,
        "s_budget_advisory": s_budget,
        "s_budget_note": (
            "S_budget = realized FLOOR d_seg + sidecar d_pose BUDGET (3.4e-5) + real rate. Flat-bulk "
            "realized d_pose is high until the residual INR renders texture; this is the deterministic "
            "FLOOR row (confirms FACT 2), not the v2 row. The v2 row needs the GPU residual + compose."
        ),
        "staged_exact_eval_cmd": contest_cmd,
        "means_not_ends": (
            "PHASE-B byte-closed the DETERMINISTIC FLOOR. The pointer moves only when the residual "
            "INR lands + the FULL 4-section archive byte-closes + evaluate.py (CPU+CUDA) < 0.19110."
        ),
    }


def _inflate_and_realize(packet_dir: Path, lstars: np.ndarray, *, max_pairs: int | None) -> dict[str, Any]:
    """Run inflate.py on a (capped) archive -> realized d_seg through the frozen CPU-torch SegNet."""
    import zipfile

    from tac.boundary_math.seg_core import load_real_segnet
    from tac.optimization.frame1_seg_repair_atoms import measure_segnet_argmax

    archive_dir = packet_dir / "archive"
    inflated_dir = packet_dir / "inflated"
    archive_dir.mkdir(exist_ok=True)
    inflated_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(packet_dir / "archive.zip") as zf:
        zf.extractall(archive_dir)
    src_bin = archive_dir / "0.bin"
    dst_raw = inflated_dir / "0.raw"
    cmd = [sys.executable, str(packet_dir / "inflate.py"), str(src_bin), str(dst_raw)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(packet_dir))
    if proc.returncode != 0:
        return {"inflate_failed": True, "stderr": proc.stderr[-2000:], "stdout": proc.stdout[-500:]}

    # read back camera frames, run SegNet argmax on frame1 -> realized d_seg vs GT lstars
    n_pairs = int(lstars.shape[0]) if max_pairs is None else min(int(max_pairs), int(lstars.shape[0]))
    fb = ag._INFLATE_PY_V2  # noqa: F841 (kept for provenance)
    CAM_H, CAM_W = 874, 1164
    frame_bytes = CAM_H * CAM_W * 3
    seg = load_real_segnet("cpu")
    d_segs: list[float] = []
    with open(dst_raw, "rb") as f:
        for p in range(n_pairs):
            _f0 = np.frombuffer(f.read(frame_bytes), dtype=np.uint8).reshape(CAM_H, CAM_W, 3)
            f1 = np.frombuffer(f.read(frame_bytes), dtype=np.uint8).reshape(CAM_H, CAM_W, 3)
            am, _m = measure_segnet_argmax(seg, f1.astype(np.float64))
            d_segs.append(float((am != lstars[p]).mean()))
    return {
        "pairs_scored": n_pairs,
        "d_seg_realized": float(np.mean(d_segs)) if d_segs else None,
        "inflate_stdout": proc.stdout.strip()[-300:],
        "note": "realized d_seg on the INFLATE'D bulk frames (the archive->inflate->SegNet chain).",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["A", "B", "both"], default="A")
    ap.add_argument("--gt-cache", default=_DEF_CACHE, help="GT npz (lstars, gt_f1, gt_poses)")
    ap.add_argument("--grok-results", default=_DEF_GROK, help="measure_pose_warp_dseg results.json")
    ap.add_argument("--reach-json", default=_DEF_REACH, help="measure_screw_reach_through_R reach.json")
    ap.add_argument("--n-pairs-clip", type=int, default=600, help="full clip pair count (for byte projection)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--residual-epochs", type=int, default=1500)
    ap.add_argument("--residual-hidden-dim", type=int, default=48, help="residual INR width (< full 96)")
    ap.add_argument("--residual-mod-dim", type=int, default=16, help="residual INR code dim (< full 32)")
    ap.add_argument("--residual-run-dir", default=None, help="where the GPU residual run would write")
    ap.add_argument("--residual-inr-weights", default=None, help="(PHASE-B) trained residual INR (NEEDS-WIRING)")
    ap.add_argument("--max-pairs", type=int, default=None, help="(PHASE-B) cap inflate+parity pairs for speed")
    ap.add_argument("--skip-inflate", action="store_true", help="(PHASE-B) byte-close only, no inflate/parity")
    ap.add_argument("--out-dir", default=None, help="output dir (default experiments/results/v2_compose_<ts>)")
    args = ap.parse_args(argv)

    out_dir = _resolve(args.out_dir) if args.out_dir else (_REPO / "experiments" / "results" / f"v2_compose_{_utc()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    reports: dict[str, Any] = {}
    if args.phase in ("A", "both"):
        reports["phase_a"] = phase_a(args, out_dir)
        (out_dir / "phase_a_report.json").write_text(json.dumps(reports["phase_a"], indent=2))
        print(f"[phase-A] report -> {out_dir / 'phase_a_report.json'}", flush=True)
        print("\n=== RESIDUAL-INR LAUNCH COMMAND (HOLD for operator GO) ===")
        print(reports["phase_a"]["residual_inr_launch_command"]["command"])
        print("===========================================================\n")
    if args.phase in ("B", "both"):
        reports["phase_b"] = phase_b(args, out_dir)
        (out_dir / "phase_b_report.json").write_text(json.dumps(reports["phase_b"], indent=2))
        print(f"[phase-B] report -> {out_dir / 'phase_b_report.json'}", flush=True)
        if reports["phase_b"].get("staged_exact_eval_cmd"):
            print("\n=== STAGED EXACT-EVAL COMMAND (not run) ===")
            print(reports["phase_b"]["staged_exact_eval_cmd"])
            print("===========================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
