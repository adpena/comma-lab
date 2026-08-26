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
     (advisory) -> audit exact coupled score surface -> emit the STAGED dual CPU/CUDA eval command

means != ends (NO-FAKE): this is composition PLUMBING. The competitive target is read from the
canonical frontier pointer; it moves only when the residual-INR run lands + the 4-section archive
byte-closes + exact CPU/CUDA eval beats that target. Every number emitted is
``[advisory] NON-PROMOTABLE``. NO GPU, NO training launch, NO touching the live n600 run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "src", _REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.contest_compliance import compute_upstream_snapshot_sha256  # noqa: E402
from tac.contest_score import rate_term  # noqa: E402
from tac.v2_compose import archive_grammar as ag  # noqa: E402
from tac.v2_compose import bulk_generator as bg  # noqa: E402
from tac.v2_compose import launch_command as lc  # noqa: E402
from tac.v2_compose import pose_sidecar as ps  # noqa: E402
from tac.v2_compose import residual_target as rt  # noqa: E402
from tac.v2_compose.residual_compose import (  # noqa: E402
    DEFAULT_ANNULUS_DILATE,
    LEARN_CLASSES,
    MASK_MODE_BOUNDARY_ANNULUS,
    build_residual_training_bundle,
    load_residual_training_bundle,
    measure_composition_coverage,
    save_residual_training_bundle,
)
from tac.v2_compose.store_learn_split import (  # noqa: E402
    encode_known_split,
    load_reach_kstar,
    load_warp_recoverability_from_grok,
    validated_frontier_target_payload,
)
from tac.witness_dsl.dynamic_frontier_target import (  # noqa: E402
    DynamicFrontierTargetSnapshot,
    load_dynamic_frontier_target,
    score_sublevel_against_dynamic_frontier,
)

# SCREW_REGIME is the single source of truth for the per-class PHYSICAL warp regime (A3.1: one
# derivation shared by phase_a + phase_b; A3.2: the inflate CONSUMES the stored codes).
from tools.measure_screw_reach_through_R import SCREW_REGIME  # noqa: E402

_D_POSE_SIDECAR = 3.4e-5  # Legacy stored-sidecar planning assumption only; never an admission gate.
_DEF_CACHE = "experiments/results/mlx_fleet_gt_cache/gt_n96.npz"
_DEF_GROK = "experiments/results/grok_pose_warp_dseg_20260629T181000Z/results.json"
_DEF_REACH = "experiments/results/screw_reach/reach_n96.json"
_RESIDUAL_BUNDLE_NAME = "residual_bundle.npz"  # the --residual-target-npz the trainer consumes
_DEFAULT_FRONTIER_POINTER = ".omx/state/canonical_frontier_pointer.json"


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (_REPO / pp)


def _frontier_snapshot_for_args(args: argparse.Namespace) -> DynamicFrontierTargetSnapshot:
    """Load only the canonical target path through the fail-closed dynamic API."""

    requested = _resolve(getattr(args, "frontier_pointer", _DEFAULT_FRONTIER_POINTER))
    canonical = _resolve(_DEFAULT_FRONTIER_POINTER)
    if os.path.abspath(os.fspath(requested)) != os.path.abspath(os.fspath(canonical)):
        raise ValueError(
            "--frontier-pointer cannot substitute a non-canonical competitive target; "
            f"expected {canonical}, got {requested}"
        )
    return load_dynamic_frontier_target(repo_root=_REPO)


def coupled_score_surface_payload(
    *,
    frontier_target: DynamicFrontierTargetSnapshot,
    d_seg: float,
    d_pose: float,
    archive_zip_bytes: int,
    external_counted_bytes: int = 0,
) -> dict[str, Any]:
    """Return the exact coupled audit with every counted byte charged."""
    honest_total_bytes = archive_zip_bytes + external_counted_bytes
    audit = score_sublevel_against_dynamic_frontier(
        frontier_target,
        d_seg=d_seg,
        d_pose=d_pose,
        archive_bytes=honest_total_bytes,
    )
    return {
        "target_score": audit.target_score,
        "score": audit.score,
        "signed_target_slack": audit.signed_target_slack,
        "inside_strict_sublevel": audit.inside_strict_sublevel,
        "seg_term": audit.seg_term,
        "pose_term": audit.pose_term,
        "rate_term": audit.rate_term,
        "archive_bytes_charged": honest_total_bytes,
        "archive_zip_bytes": archive_zip_bytes,
        "external_counted_bytes": external_counted_bytes,
        "conditional_d_seg_boundary": audit.boundary_d_seg_given_pose_and_bytes,
        "conditional_d_pose_boundary": audit.boundary_d_pose_given_seg_and_bytes,
        "conditional_archive_byte_boundary": (audit.boundary_archive_bytes_given_distortions),
        "admission_rule": "exact coupled score only; component coordinates are conditional",
    }


def _git_head() -> str | None:
    """``git rev-parse HEAD`` (provenance; A1.1). None if git is unavailable."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(_REPO), capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        return None


def _provenance() -> dict[str, Any]:
    """Deterministic-reproducibility provenance (A1.1): git hash + upstream-snapshot-sha + utc.

    Per the CLAUDE.md deterministic-reproducibility non-negotiable item (6): every result records
    git hash + upstream-snapshot-sha so any host can reproduce the exact bytes. Best-effort (never
    raises): the upstream sha uses the canonical ``tac.contest_compliance`` helper."""
    try:
        upstream_sha = compute_upstream_snapshot_sha256(_REPO)
    except Exception:
        upstream_sha = None
    return {
        "git_head": _git_head(),
        "upstream_snapshot_sha256": upstream_sha,
        "utc": _utc(),
    }


def _warp_codes_for_clip(n_classes: int = 5) -> list[int]:
    """The per-class PHYSICAL warp-regime codes stored in the store_blob warp-mask (A3.1 single
    derivation; A3.2 consumed by the inflate). Derived from SCREW_REGIME -> WARP_TYPE_CODE."""
    return ag.screw_regime_warp_codes(SCREW_REGIME, n_classes=n_classes)


def keyframe_payload_accounting(args: argparse.Namespace) -> dict[str, Any]:
    """EXPLICIT COUNTED byte accounting for the real-luma pose-carrier keyframes (rule-118 / NO-FAKE).

    The warp-real-luma pose carrier (``--pose-carrier``; ``tac.boundary_math.warp_real_luma_frame0``,
    the #205 sealed-config pose path) produces the render's FRAME0 by warping a STORED REAL keyframe by
    the per-pair ego twist. Rule-118 boundary:

      * FREE  (inflate.py, uncounted): the plane homography + ``exp_se3`` + inverse-warp bilinear + R
        (generic algorithm) AND the per-pair twist ``xi`` (DUAL-USE — already counted in the pose
        sidecar for d_pose, so ~0 MARGINAL bytes).
      * COUNTED (archive.zip): the STORED REAL keyframe luma (video-derived) — the payload
        ``warp_real_luma_frame0.py`` explicitly flags as "the vehicle's ... concern ... NOT smuggled
        into this module's byte claim". THIS is that line item, so the composed byte-closed rate is
        HONEST rather than silently omitting the keyframes (the borrowed "13 keyframes ~ 0.0060"
        partition figure the ``warp_keyframe_payload_rate_minimization`` memo corrects).

    Sources (priority): ``--keyframe-payload-path`` (counts a REAL blob's ``st_size``) OR
    ``--keyframe-payload-bytes`` (an explicit MEASURED codec byte count from the memo §2 sweep).
    Default 0 -> NO real-luma carrier (pose via the stored sidecar, which the scorer never reads as
    frames -> does NOT lower realized d_pose); a real-luma ``--pose-carrier`` ROW REQUIRES payload > 0.
    """
    path = getattr(args, "keyframe_payload_path", None)
    explicit = int(getattr(args, "keyframe_payload_bytes", 0) or 0)
    if explicit < 0:
        raise ValueError("--keyframe-payload-bytes must be >= 0")
    source = "none"
    real_path: str | None = None
    if path:
        pp = _resolve(path)
        if not pp.exists():
            raise FileNotFoundError(f"--keyframe-payload-path not found: {pp}")
        kbytes = int(pp.stat().st_size)
        source = "real_payload_file"
        real_path = str(pp)
    elif explicit > 0:
        kbytes = explicit
        source = "explicit_measured_bytes"
    else:
        kbytes = 0
    return {
        "keyframe_blob_bytes": kbytes,
        "keyframe_blob_rate": rate_term(kbytes),
        "source": source,
        "payload_path": real_path,
        "keyframe_count_hint": getattr(args, "keyframe_count", None),
        "rule_118": {
            "COUNTED (archive.zip)": (
                "stored real keyframe luma (video-derived); warped at decode by "
                "the per-pair ego twist to synthesize frame0"
            ),
            "FREE (inflate.py)": (
                "plane homography + exp_se3 + inverse-warp bilinear + R (generic "
                "algorithm) + the DUAL-USE twist (already counted in the pose sidecar)"
            ),
        },
        "note": (
            "NO real-luma pose-carrier keyframes counted (0 B). Pose path = the stored screw/twist "
            "sidecar, which the scorer never reads as frames -> does NOT lower realized d_pose. A "
            "real-luma --pose-carrier ROW (the #205 sealed config) REQUIRES a keyframe payload > 0 "
            "here; pass --keyframe-payload-bytes (measured codec, memo warp_keyframe_payload_rate_"
            "minimization_20260702 §2) or --keyframe-payload-path <blob>."
            if kbytes == 0
            else f"real-luma pose-carrier keyframes: {kbytes} B ({source}) COUNTED into the rate "
            f"(+{rate_term(kbytes):.5f}). Codec/resolution/count MEASURED in the "
            "warp_keyframe_payload_rate_minimization memo; the cheap-vs-expensive branch is decided by "
            "the residual-compensation measurement (memo §4.2, needs the trained #205 residual)."
        ),
    }


def phase_a(
    args: argparse.Namespace,
    out_dir: Path,
    *,
    frontier_target: DynamicFrontierTargetSnapshot,
) -> dict[str, Any]:
    """PHASE-A: plan + deterministic bulk + residual target + pose sidecar + launch command."""
    # Verify at entry; target-dependent helpers and final report serialization
    # repeat the reopen so a mid-run pointer refresh fails closed.
    validated_frontier_target_payload(frontier_target)
    target_score = frontier_target.target_score
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
    plan = encode_known_split(
        rec,
        kstar,
        frontier_target=frontier_target,
        n_pairs=args.n_pairs_clip,
    )
    print(
        f"[phase-A] split: GENERATE={plan.generate_classes} LEARN={plan.learn_classes} "
        f"keyframes={plan.keyframe_count} reach_k*={kstar}",
        file=sys.stderr,
    )

    # --- Step 3: deterministic bulk through R (proven warp path) ---
    cfg = bg.load_calibration_from_reach(_resolve(args.reach_json), reach_kstar=kstar, n_pairs=n_pairs)
    print(
        f"[phase-A] generating deterministic bulk through R for {n_pairs} pairs "
        f"(calib s_t={cfg.s_t:.5f} s_r={cfg.s_r} pitch={cfg.pitch:.4f}) ...",
        file=sys.stderr,
    )
    bulk = bg.generate_bulk_argmax_stack(lstars, gt_f1, gt_poses, cfg, verbose=True)
    print(
        f"[phase-A] bulk d_seg floor: full={bulk.bulk_dseg_full:.4f} bulk={bulk.bulk_dseg_bulk:.4f} "
        f"| SegNet selfcheck ok={bulk.k0_faithful} (max_px={bulk.segnet_selfcheck_max_px}) "
        f"| k0 R1-floor={bulk.k0_bulk_mean:.4f} (within n96 ref={bulk.k0_within_r1_ref})",
        file=sys.stderr,
    )
    if not bulk.k0_faithful:
        print(
            "[phase-A] WARNING: SegNet selfcheck FAILED (gt_f1->argmax != cached lstars) — the "
            "render->R->SegNet path is NOT the contest path; treat ALL bulk numbers as suspect.",
            file=sys.stderr,
        )

    # --- Step 4: residual target (bulk SUBTRACTED) -- the DIAGNOSTIC floor (which cells bulk gets
    # wrong, by GT class). The residual_mask here is the coverage-gate reference (B2). ---
    residual = rt.compute_residual_target(bulk.bulk_argmax_through_R, lstars)
    residual_path = out_dir / "residual_target.npz"
    residual_bytes = rt.save_residual_target(residual, residual_path)
    print(
        f"[phase-A] residual target (diagnostic): bulk_dseg_floor={residual.bulk_dseg:.4f} "
        f"residual_classes_ranked={residual.residual_classes_ranked[:3]} "
        f"-> {residual_path} ({residual_bytes} B training artifact)",
        file=sys.stderr,
    )

    # --- Step 5: pose sidecar (dual-use STORED screw/twist; $0 from cache poses) ---
    pose_path = out_dir / "posenet_targets.bin"
    pose_bytes = ps.build_pose_sidecar_from_cache_poses(gt_poses, pose_path)
    print(f"[phase-A] pose sidecar (stored screw/twist): {pose_bytes} B -> {pose_path}", file=sys.stderr)

    # --- the REAL store-blob byte measurement (keyframes contour-coded + palette + calib + mask).
    # warp_codes are the PHYSICAL per-class regime codes (A3.1 single derivation; A3.2 consumed by
    # the inflate's _composite_warped -- no more dead bytes / hardcoded routing). ---
    warp_codes = _warp_codes_for_clip(n_classes=5)
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
    store_bytes_proj600 = round(store_bytes * (proj_kf_full / max(measured_kf, 1)))

    # --- byte budget arithmetic ---
    # EXPLICIT real-luma pose-carrier keyframe line item (rule-118 COUNTED; the accounting gap the
    # warp_keyframe_payload_rate_minimization memo flags — no longer silently omitted).
    kf = keyframe_payload_accounting(args)
    kf_bytes = int(kf["keyframe_blob_bytes"])
    known_store_excl_kf = store_bytes_proj600 + pose_bytes
    known_store = known_store_excl_kf + kf_bytes  # HONEST total: partition store + pose sidecar + keyframes
    conditional_surface = score_sublevel_against_dynamic_frontier(
        frontier_target,
        d_seg=0.0,
        d_pose=_D_POSE_SIDECAR,
        archive_bytes=known_store,
    )
    break_even_dseg = conditional_surface.boundary_d_seg_given_pose_and_bytes
    print(
        f"[phase-A] keyframe payload (real-luma pose carrier): {kf_bytes} B "
        f"({kf['source']}; rate +{kf['keyframe_blob_rate']:.5f}) -> honest known-store "
        f"{known_store} B (rate {rate_term(known_store):.5f})",
        file=sys.stderr,
    )
    budget = {
        "store_blob_bytes_measured_subset": store_bytes,
        "store_keyframes_measured": measured_kf,
        "store_blob_bytes_projected_600": store_bytes_proj600,
        "projected_keyframes_600": proj_kf_full,
        "pose_sidecar_bytes": pose_bytes,
        # EXPLICIT keyframe line item (real-luma pose carrier); COUNTED in the honest total below.
        "keyframe_blob_bytes": kf_bytes,
        "keyframe_blob_rate": kf["keyframe_blob_rate"],
        "keyframe_payload": kf,
        "known_store_excl_keyframes_bytes_proj600": known_store_excl_kf,
        "known_store_total_bytes_proj600": known_store,  # NOW includes the real-luma keyframe payload
        "known_store_rate_proj600": rate_term(known_store),
        "residual_inr_bytes": None,  # OPEN — the GPU run
        "free_generated_bytes": 0,
        "deterministic_bulk_dseg_floor": residual.bulk_dseg,
        "conditional_d_seg_boundary_at_known_store_and_pose_assumption": break_even_dseg,
        "coupled_target_score": target_score,
        # Filled from a final pointer reopen immediately before report return.
        "coupled_target_custody": None,
        "d_pose_planning_assumption": _D_POSE_SIDECAR,
        "d_pose_assumption_note": (
            "BUDGET assumption (the stored-screw/twist TARGET), NOT yet validated for THIS composed "
            "pipeline. d_pose is an OPEN axis until measured on the composed witness (B3)."
        ),
    }

    # --- Step 4b (NEW, the B1 fix): build + save the residual TRAINING BUNDLE the trainer's
    # --residual-mode actually consumes (NOT residual_target.npz). The deterministic bulk RGB +
    # the boundary-annulus composition mask (B2) at render res; the COUNTED bytes are the INR
    # weights the trainer PRODUCES, never this bundle. ---
    print("[phase-A] generating deterministic bulk render + warped labels for the residual bundle ...", file=sys.stderr)
    bulk_rgb, warped_labels = bg.generate_bulk_render_and_labels(lstars, gt_poses, cfg, bulk.palette)
    bundle = build_residual_training_bundle(
        bulk_rgb,
        warped_labels,
        learn_classes=LEARN_CLASSES,
        dilate=DEFAULT_ANNULUS_DILATE,
        mode=MASK_MODE_BOUNDARY_ANNULUS,
    )
    bundle_path = out_dir / _RESIDUAL_BUNDLE_NAME
    bundle_bytes = save_residual_training_bundle(bundle, bundle_path)
    print(
        f"[phase-A] residual BUNDLE (trainer input): mask_mode={bundle.mask_mode} "
        f"dilate={bundle.dilate} override_frac={float(bundle.composition_mask.mean()):.4f} "
        f"-> {bundle_path} ({bundle_bytes} B training artifact)",
        file=sys.stderr,
    )

    # --- Step 4c (NEW, the B2 GEOMETRY gate): does the composition mask COVER the residual the
    # INR must close? Gate only on the CONDITIONAL d_seg coordinate of the live score surface. ---
    if break_even_dseg is None:
        raise ValueError(
            "known store plus pose assumption already exceeds the coupled target before Seg; "
            "no non-negative d_seg boundary exists"
        )
    coverage = measure_composition_coverage(
        residual.residual_mask, bundle.composition_mask, dseg_budget=break_even_dseg
    )
    cov_summary = coverage.to_summary()
    print(
        f"[phase-A] COVERAGE GATE: coverage={coverage.coverage:.4f} "
        f"unreachable_dseg={coverage.unreachable_dseg:.5f} "
        f"(conditional target boundary={break_even_dseg:.5f}) "
        f"-> GATE {'PASS' if coverage.passes_gate else 'FLAG (NO-GO: geometry ceiling)'}",
        file=sys.stderr,
    )
    if not coverage.passes_gate:
        print(
            "[phase-A] WARNING: the composition mask does NOT cover enough of the residual — the "
            "INR can NEVER close the uncovered flips (a GEOMETRY ceiling, NOT an INR-capacity wall). "
            "Widen the annulus (--residual-dilate) or switch mask_mode=union BEFORE the GPU GO.",
            file=sys.stderr,
        )

    # --- Step 6 (emit): the CORRECTED residual-ONLY launch command (--residual-mode +
    # --residual-target-npz pointed at the BUNDLE). NO --structured-init (that bakes the bulk into
    # the weights = NO rate shrink). HOLD for operator GO. ---
    residual_run_dir = (
        str(_resolve(args.residual_run_dir)) if args.residual_run_dir else str(out_dir / "residual_inr_run")
    )
    launch = lc.build_residual_only_command(
        out_dir=residual_run_dir,
        gt_cache=str(cache),
        residual_target_npz=str(bundle_path),
        num_pairs=args.n_pairs_clip,
        epochs=args.residual_epochs,
        seed=args.seed,
        hidden_dim=args.residual_hidden_dim,
        mod_dim=args.residual_mod_dim,
        strict=True,
    )
    print(
        f"[phase-A] residual-ONLY launch command FLAG-VALIDATED "
        f"(all_valid={launch.all_flags_valid}); HOLD for operator GO.",
        file=sys.stderr,
    )

    frontier_target_receipt = validated_frontier_target_payload(frontier_target)
    budget["coupled_target_custody"] = frontier_target_receipt
    report = {
        "phase": "A",
        "tool": "compose_witness_archive",
        "provenance": _provenance(),
        "utc": _utc(),
        "authority": "[macOS-CPU advisory] NON-PROMOTABLE",
        "score_claim": False,
        "promotable": False,
        "frontier_target": frontier_target_receipt,
        "pointer_moved": False,
        "gt_cache": str(cache),
        "n_pairs_cache": n_pairs,
        "n_pairs_clip": args.n_pairs_clip,
        "split": plan.to_json(),
        "warp_type_codes": warp_codes,
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
        "residual_bundle": {**bundle.to_summary(), "artifact": str(bundle_path), "artifact_bytes": bundle_bytes},
        "composition_coverage": cov_summary,
        "coverage_gate_passes": coverage.passes_gate,
        "pose_sidecar": ps.pose_sidecar_byte_cost(pose_path),
        "byte_budget": budget,
        "residual_inr_launch_command": launch.to_json(),
        "open_axes": {
            "d_seg": (
                f"OPEN: can the SMALL residual INR close d_seg from the {residual.bulk_dseg:.4f} "
                f"deterministic floor to ~{break_even_dseg:.5f} (conditional coordinate for target "
                f"{target_score:g})? Bounded BELOW by the "
                f"coverage gate's unreachable_dseg={coverage.unreachable_dseg:.5f}. Settled by the GPU "
                "residual run + byte-closed exact eval."
            ),
            "d_pose": (
                "OPEN: d_pose of the COMPOSED witness is NOT yet validated for this pipeline. The pose "
                "sidecar stores the screw/twist (dual-use: warp->d_seg AND pose->d_pose); whether "
                "PoseNet(composed pair) ~ PoseNet(original pair) holds is measured in PHASE-B (advisory) "
                "and on the byte-closed exact eval. The 3.4e-5 figure is the stored-target BUDGET, not a "
                "measured d_pose for this composition (NO-FAKE)."
            ),
            "rate": (
                f"PARTIAL: known store {known_store} B (rate {rate_term(known_store):.5f}) is MEASURED "
                f"= partition store {store_bytes_proj600} + pose sidecar {pose_bytes} + real-luma "
                f"keyframe payload {kf_bytes} B (EXPLICIT rule-118 line item; {kf['source']}). The "
                "residual INR bytes are OPEN until the GPU run byte-closes; the keyframe payload's "
                "final size is decided by the residual-compensation measurement (memo §4.2)."
            ),
        },
        "means_not_ends": (
            "PHASE-A built the composition + the residual BUNDLE + emitted the residual-ONLY command "
            "(a MEANS). The pointer moves only when that run lands + the 4-section archive byte-closes "
            f"+ evaluate.py (CPU+CUDA) < current canonical target {target_score:g}."
        ),
    }
    return report


def residual_blob_from_weights_npz(
    weights_npz: str | Path,
    *,
    learn_classes: tuple[int, ...] = LEARN_CLASSES,
    dilate: int = DEFAULT_ANNULUS_DILATE,
    mask_mode: str = MASK_MODE_BOUNDARY_ANNULUS,
) -> tuple[bytes, dict[str, Any]]:
    """Build the COUNTED residual_blob from a trained residual-INR EMA-shadow npz (B1 / G1.3).

    The trainer saves the EMA shadow as a flat npz: the INR params (in_proj/film/hidden.*/out_sdf/
    out_tex/palette/code, + optional film_pl/concat_pl) plus ``__cfg_*``/``__bank_*`` provenance
    scalars (the ``_build_ema_checkpoint_arrays`` schema). This reconstructs the forward cfg from
    those scalars (mod_dim from the code shape; n_classes from out_sdf) and threads the bundle's
    ``learn_classes``/``dilate``/``mask_mode`` into the manifest so the inflate re-derives the SAME
    composition mask (train == inflate). Returns (residual_blob_bytes, inr_cfg). NO-FAKE: the COUNTED
    bytes ARE the trained weights; nothing is fabricated."""
    weights_npz = Path(weights_npz)
    z = np.load(weights_npz, allow_pickle=False)
    params = {k: np.asarray(z[k], np.float32) for k in z.files if not k.startswith("__")}
    if "code" not in params:
        raise ValueError(f"{weights_npz}: trained INR weights must include the per-frame 'code' latent")
    if "out_sdf.weight" not in params:
        raise ValueError(f"{weights_npz}: trained INR weights must include 'out_sdf.weight'")
    for req in (
        "__cfg_n_hidden",
        "__cfg_hidden_dim",
        "__bank_n_scales",
        "__bank_n_orient0",
        "__bank_f0",
        "__bank_base",
        "__bank_n_iso",
    ):
        if req not in z.files:
            raise ValueError(
                f"{weights_npz}: missing required cfg key {req!r} (not an EMA-checkpoint npz from "
                "_build_ema_checkpoint_arrays). Cannot reconstruct the forward contract."
            )

    def _g(key: str, cast, default=None):
        return cast(z[key]) if key in z.files else default

    render_hw = z["__render_hw"] if "__render_hw" in z.files else None
    render_h = int(render_hw[0]) if render_hw is not None else 384
    render_w = int(render_hw[1]) if render_hw is not None else 512
    max_bank = _g("__cfg_max_bank_freq", float, -1.0)
    inr_cfg: dict[str, Any] = {
        "n_hidden": _g("__cfg_n_hidden", int),
        "hidden_dim": _g("__cfg_hidden_dim", int),
        "mod_dim": int(params["code"].shape[-1]),
        "n_classes": int(params["out_sdf.weight"].shape[0]),
        "activation": _g("__cfg_activation", str, "hosc"),
        "hosc_beta": _g("__cfg_hosc_beta", float, 4.0),
        "hosc_omega": _g("__cfg_hosc_omega", float, 1.0),
        "softmax_temp": _g("__cfg_softmax_temp", float, 0.1),
        "wire_w0": _g("__cfg_wire_w0", float, 20.0),
        "wire_s0": _g("__cfg_wire_s0", float, 10.0),
        "chroma": bool(_g("__cfg_chroma", int, 1)),
        "render_h": render_h,
        "render_w": render_w,
        "bank_n_scales": _g("__bank_n_scales", int),
        "bank_n_orient0": _g("__bank_n_orient0", int),
        "bank_f0": _g("__bank_f0", float),
        "bank_base": _g("__bank_base", float),
        "bank_n_iso": _g("__bank_n_iso", int),
        "max_bank_freq": (None if (max_bank is None or max_bank < 0) else max_bank),
        "learn_classes": [int(c) for c in learn_classes],
        "dilate": int(dilate),
        "mask_mode": str(mask_mode),
    }
    return ag.build_residual_blob(params, inr_cfg), inr_cfg


def phase_b(
    args: argparse.Namespace,
    out_dir: Path,
    *,
    frontier_target: DynamicFrontierTargetSnapshot,
) -> dict[str, Any]:
    """PHASE-B: assemble the 4-section archive, inflate, realize d_seg (+ advisory d_pose) through R,
    emit the staged dual CPU/CUDA eval command.

    With ``--residual-inr-weights`` it byte-closes the REAL v2 row (store + residual INR + pose);
    without it, the DETERMINISTIC FLOOR (empty residual). (B1: the residual-INR section + inflate
    compose hook are now wired -- no SystemExit.)"""
    validated_frontier_target_payload(frontier_target)
    target_score = frontier_target.target_score
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
    # PHYSICAL per-class warp-regime codes (A3.1: SAME derivation as phase_a; A3.2: consumed by the
    # inflate's _composite_warped -- no hardcoded [0,3,2,3,1] two-sources-of-truth seam).
    warp_codes = _warp_codes_for_clip(n_classes=5)

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

    # --- residual INR section (B1: wired; assembles the REAL v2 row from trained weights) ---
    residual_blob = b""
    residual_info: dict[str, Any] = {"present": False}
    if args.residual_inr_weights:
        weights = _resolve(args.residual_inr_weights)
        if not weights.exists():
            raise SystemExit(f"--residual-inr-weights not found: {weights}")
        # the residual mask params (learn_classes/dilate/mask_mode) MUST match the bundle the INR
        # trained against, so the inflate re-derives the SAME composition mask (train == inflate).
        bundle_path = _resolve(args.residual_bundle) if args.residual_bundle else (out_dir / _RESIDUAL_BUNDLE_NAME)
        learn_classes: tuple[int, ...] = LEARN_CLASSES
        dilate = DEFAULT_ANNULUS_DILATE
        mask_mode = MASK_MODE_BOUNDARY_ANNULUS
        if bundle_path.exists():
            bundle = load_residual_training_bundle(bundle_path)
            learn_classes = bundle.learn_classes
            dilate = bundle.dilate
            mask_mode = bundle.mask_mode
            print(
                f"[phase-B] residual mask params from bundle {bundle_path.name}: "
                f"learn_classes={list(learn_classes)} dilate={dilate} mask_mode={mask_mode}",
                file=sys.stderr,
            )
        else:
            print(
                f"[phase-B] WARNING: no residual bundle at {bundle_path} -- using DEFAULTS "
                f"(learn_classes={list(learn_classes)} dilate={dilate} mask_mode={mask_mode}). "
                "Pass --residual-bundle to guarantee the inflate mask matches the trained mask.",
                file=sys.stderr,
            )
        residual_blob, inr_cfg = residual_blob_from_weights_npz(
            weights, learn_classes=learn_classes, dilate=dilate, mask_mode=mask_mode
        )
        residual_info = {
            "present": True,
            "weights_npz": str(weights),
            "bundle": str(bundle_path) if bundle_path.exists() else None,
            "residual_blob_bytes": len(residual_blob),
            "inr_cfg": {
                k: v for k, v in inr_cfg.items() if k not in ("base_shapes", "base_scales", "base_param_order")
            },
        }
        print(f"[phase-B] residual INR section: {len(residual_blob)} B (the COUNTED LEARN-tier rate)", file=sys.stderr)

    manifest = {
        "format_version": "v2.0",
        "n_pairs": n_pairs,
        "n_classes": 5,
        "reach_kstar": kstar,
        "keyframe_count": len(keyframes),
        "residual_inr_present": bool(residual_blob),
        "mlx_free_inflate": True,
        "loads_no_scorers_at_inflate": True,
        "note": (
            "v2 row (store + residual INR + pose)"
            if residual_blob
            else "deterministic FLOOR archive (empty residual); the residual INR closes d_seg + d_pose."
        ),
    }
    manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode("utf-8")

    blob = ag.pack_v2_archive(store_blob, residual_blob, pose_blob, manifest_bytes)
    packet_dir = out_dir / "packet"
    zip_path, zip_bytes = ag.assemble_v2_packet(blob, packet_dir)
    acct = ag.byte_accounting(zip_path, store_blob, residual_blob, pose_blob, manifest_bytes)
    # EXPLICIT real-luma pose-carrier keyframe line item (rule-118). This archive's pose path is the
    # stored sidecar, so its zip does NOT pack keyframes; when a real-luma --pose-carrier is adopted
    # the keyframe payload is COUNTED into the honest rate (no silent omission). accounting-only here.
    kf = keyframe_payload_accounting(args)
    kf_bytes = int(kf["keyframe_blob_bytes"])
    honest_total_bytes = int(zip_bytes) + kf_bytes
    acct = {
        **acct,
        "keyframe_payload": kf,
        "keyframe_blob_bytes": kf_bytes,
        "keyframe_in_this_zip": False,
        "honest_total_bytes_incl_keyframes": honest_total_bytes,
        "honest_rate_incl_keyframes": rate_term(honest_total_bytes),
        "honest_rate_note": (
            "archive.zip st_size is the rate of the store+residual+pose sections (stored-sidecar pose "
            "path). The real-luma pose-carrier keyframe payload is NOT packed in THIS zip; its EXPLICIT "
            "COUNTED bytes are added here for the honest real-luma-carrier rate. Fully byte-closing a "
            "real-luma --pose-carrier row additionally requires the pose-carrier warp decode wired into "
            "inflate (a separate wiring dependency, flagged; not fabricated here)."
        ),
    }
    print(
        f"[phase-B] archive.zip = {zip_bytes} B (rate {acct['rate']:.5f}); sections={acct['section_bytes']}"
        + (
            f"; + keyframe payload {kf_bytes} B -> honest rate {acct['honest_rate_incl_keyframes']:.5f}"
            if kf_bytes
            else "; keyframe payload 0 B (stored-sidecar pose path)"
        ),
        file=sys.stderr,
    )

    realized: dict[str, Any] = {"skipped": bool(args.skip_inflate)}
    if not args.skip_inflate:
        realized = _inflate_and_realize(
            packet_dir,
            lstars,
            gt_poses=(gt_poses if args.measure_dpose else None),
            max_pairs=args.max_pairs,
            dpose_max_pairs=args.dpose_max_pairs,
        )

    d_seg = realized.get("d_seg_realized", None)
    d_pose_measured = realized.get("d_pose_realized", None)
    is_floor = not residual_blob
    s_budget = None
    s_budget_note = None
    coupled_score_surface = None
    if d_seg is not None:
        # Use the MEASURED composed d_pose when available (B3 -- the honest pipeline value); else the
        # stored-target BUDGET, clearly labeled as an assumption (NO-FAKE #8).
        d_pose_for_s = d_pose_measured if d_pose_measured is not None else _D_POSE_SIDECAR
        coupled_score_surface = coupled_score_surface_payload(
            frontier_target=frontier_target,
            d_seg=d_seg,
            d_pose=d_pose_for_s,
            archive_zip_bytes=zip_bytes,
            external_counted_bytes=kf_bytes,
        )
        s_budget = float(coupled_score_surface["score"])
        s_budget_note = (
            "S = realized d_seg + "
            + (
                "MEASURED composed d_pose"
                if d_pose_measured is not None
                else "stored-target d_pose BUDGET 3.4e-5 (NOT measured for this composition)"
            )
            + " + honest counted rate (archive.zip plus any external keyframe payload). "
            + (
                "DETERMINISTIC FLOOR row (empty residual; f0==f1 so PoseNet sees ~no motion -> "
                "d_pose high). Confirms FACT 2; NOT the v2 row."
                if is_floor
                else "v2 row (store + residual INR + pose)."
            )
            + " [advisory] until upstream/evaluate.py runs these bytes (CPU+CUDA)."
        )

    contest_cmd = (
        f".venv/bin/python experiments/contest_auth_eval.py "
        f"--archive {zip_path} --inflate-sh {packet_dir / 'inflate.sh'} "
        f"--device cpu  # [contest-CPU] authoritative ONLY on Linux x86_64; then --device cuda on T4"
    )

    frontier_target_receipt = validated_frontier_target_payload(frontier_target)
    return {
        "phase": "B",
        "tool": "compose_witness_archive",
        "provenance": _provenance(),
        "utc": _utc(),
        "authority": "[advisory] NON-PROMOTABLE",
        "score_claim": False,
        "promotable": False,
        "frontier_target": frontier_target_receipt,
        "pointer_moved": False,
        "archive_zip": str(zip_path),
        "warp_type_codes": warp_codes,
        "residual_inr": residual_info,
        "byte_accounting": acct,
        "realized": realized,
        "s_budget_advisory": s_budget,
        "s_budget_note": s_budget_note,
        "coupled_score_surface": coupled_score_surface,
        "d_pose_axis": (
            "OPEN: d_pose of the composed witness is validated through PoseNet on the inflated pairs "
            "(advisory) "
            + (
                f"= {d_pose_measured:.6g} on {realized.get('dpose_pairs_scored')} pairs"
                if d_pose_measured is not None
                else "(not measured this run; pass --measure-dpose)"
            )
            + ". The pose sidecar stores the screw/twist (dual-use warp->d_seg AND pose->d_pose); "
            "the 3.4e-5 figure is the stored-target BUDGET, NOT a measured composed d_pose (NO-FAKE)."
        ),
        "staged_exact_eval_cmd": contest_cmd,
        "means_not_ends": (
            "PHASE-B byte-closed the "
            + ("v2 row (store + residual INR + pose)." if residual_blob else "DETERMINISTIC FLOOR.")
            + " The pointer moves only when the FULL 4-section archive byte-closes + evaluate.py "
            f"(CPU+CUDA) < current canonical target {target_score:g}."
        ),
    }


def _load_posenet_cpu():
    """Load the frozen contest PoseNet on CPU (NEVER MPS) for the advisory d_pose measurement."""
    from tac.boundary_math.seg_core import _ensure_upstream_on_path

    _ensure_upstream_on_path()
    from modules import PoseNet  # type: ignore  # upstream
    from safetensors.torch import load_file

    ckpt = _REPO / "upstream" / "models" / "posenet.safetensors"
    if not ckpt.exists():
        raise FileNotFoundError(f"PoseNet checkpoint not found: {ckpt}")
    pose = PoseNet().eval().to("cpu")
    pose.load_state_dict(load_file(str(ckpt), device="cpu"))
    return pose


def _inflate_and_realize(
    packet_dir: Path,
    lstars: np.ndarray,
    *,
    gt_poses: np.ndarray | None = None,
    max_pairs: int | None,
    dpose_max_pairs: int | None = 16,
) -> dict[str, Any]:
    """Run inflate.py on the archive -> realized d_seg through the frozen CPU-torch SegNet (+ an
    OPTIONAL advisory d_pose through the frozen CPU PoseNet when ``gt_poses`` is given). NEVER MPS."""
    import zipfile

    from tac.boundary_math.seg_core import load_real_segnet
    from tac.optimization.frame1_seg_repair_atoms import measure_segnet_argmax

    archive_dir = packet_dir / "archive"
    inflated_dir = packet_dir / "inflated"
    archive_dir.mkdir(exist_ok=True)
    inflated_dir.mkdir(exist_ok=True)
    import shutil

    from tac.submission_archive import safe_extract_zip

    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    safe_extract_zip(packet_dir / "archive.zip", archive_dir)
    src_bin = archive_dir / "0.bin"
    dst_raw = inflated_dir / "0.raw"
    cmd = [sys.executable, str(packet_dir / "inflate.py"), str(src_bin), str(dst_raw)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(packet_dir))
    if proc.returncode != 0:
        return {"inflate_failed": True, "stderr": proc.stderr[-2000:], "stdout": proc.stdout[-500:]}

    # read back camera frames, run SegNet argmax on frame1 -> realized d_seg vs GT lstars.
    n_pairs = int(lstars.shape[0]) if max_pairs is None else min(int(max_pairs), int(lstars.shape[0]))
    CAM_H, CAM_W = 874, 1164
    frame_bytes = CAM_H * CAM_W * 3
    seg = load_real_segnet("cpu")
    d_segs: list[float] = []
    dpose_n = 0 if gt_poses is None else min(int(dpose_max_pairs or 0), int(lstars.shape[0]))
    dpose_frames: list[Any] = []  # (H,W,3) uint8 torch tensors for the advisory d_pose pairs
    import torch

    with open(dst_raw, "rb") as f:
        for p in range(n_pairs):
            f0 = np.frombuffer(f.read(frame_bytes), dtype=np.uint8).reshape(CAM_H, CAM_W, 3)
            f1 = np.frombuffer(f.read(frame_bytes), dtype=np.uint8).reshape(CAM_H, CAM_W, 3)
            am, _m = measure_segnet_argmax(seg, f1.astype(np.float64))
            d_segs.append(float((am != lstars[p]).mean()))
            if p < dpose_n:
                dpose_frames.append(torch.from_numpy(f0.copy()))
                dpose_frames.append(torch.from_numpy(f1.copy()))

    out: dict[str, Any] = {
        "pairs_scored": n_pairs,
        "d_seg_realized": float(np.mean(d_segs)) if d_segs else None,
        "inflate_stdout": proc.stdout.strip()[-300:],
        "note": "realized d_seg on the INFLATE'D frames (the archive->inflate->SegNet chain).",
    }

    # advisory d_pose (B3): PoseNet(composed pair) vs the stored GT poses (which ARE PoseNet(original)).
    if gt_poses is not None and dpose_frames:
        try:
            from tac.scorer_targets import extract_posenet_targets

            pose = _load_posenet_cpu()
            res = extract_posenet_targets(dpose_frames, pose, device="cpu", verbose=False)
            witness_poses = np.asarray(res["targets"], dtype=np.float64)  # (dpose_n, 6)
            gt = np.asarray(gt_poses[:dpose_n], dtype=np.float64)
            d_pose = float(np.mean((witness_poses - gt) ** 2))
            out["d_pose_realized"] = d_pose
            out["dpose_pairs_scored"] = int(dpose_n)
            out["d_pose_note"] = (
                "MSE(PoseNet(composed pair)[:6], stored GT pose[:6]) on the first "
                f"{dpose_n} pairs. For the FLOOR archive f0==f1 so PoseNet sees ~no motion -> high "
                "d_pose; the residual INR (which differs f0/f1 in the override region) closes it. "
                "Advisory CPU; authority is upstream/evaluate.py."
            )
        except Exception as exc:
            out["d_pose_error"] = f"{type(exc).__name__}: {exc}"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["A", "B", "both"], default="A")
    ap.add_argument("--gt-cache", default=_DEF_CACHE, help="GT npz (lstars, gt_f1, gt_poses)")
    ap.add_argument("--grok-results", default=_DEF_GROK, help="measure_pose_warp_dseg results.json")
    ap.add_argument("--reach-json", default=_DEF_REACH, help="measure_screw_reach_through_R reach.json")
    ap.add_argument(
        "--frontier-pointer",
        default=_DEFAULT_FRONTIER_POINTER,
        help="canonical pointer supplying the effective competitive target",
    )
    ap.add_argument("--n-pairs-clip", type=int, default=600, help="full clip pair count (for byte projection)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--residual-epochs", type=int, default=1500)
    ap.add_argument("--residual-hidden-dim", type=int, default=48, help="residual INR width (< full 96)")
    ap.add_argument("--residual-mod-dim", type=int, default=16, help="residual INR code dim (< full 32)")
    ap.add_argument("--residual-run-dir", default=None, help="where the GPU residual run would write")
    ap.add_argument(
        "--residual-inr-weights",
        default=None,
        help="(PHASE-B) trained residual-INR EMA-shadow npz -> assembles the REAL v2 row",
    )
    ap.add_argument(
        "--residual-bundle",
        default=None,
        help="(PHASE-B) the residual bundle the INR trained against (learn_classes/dilate/"
        "mask_mode for the inflate mask). Default <out_dir>/residual_bundle.npz",
    )
    # real-luma pose-carrier keyframe payload (EXPLICIT rule-118 COUNTED line item; the accounting
    # gap warp_keyframe_payload_rate_minimization_20260702 flags). Default 0 -> stored-sidecar pose.
    ap.add_argument(
        "--keyframe-payload-bytes",
        type=int,
        default=0,
        help="EXPLICIT measured byte count of the stored real-luma pose-carrier keyframes "
        "(video-derived -> COUNTED in the rate). From the codec sweep in memo "
        "warp_keyframe_payload_rate_minimization_20260702 §2 (e.g. 384x512 HEVC crf34 "
        "13-keyframe ~32875 B). Default 0 = no real-luma carrier (pose via stored sidecar).",
    )
    ap.add_argument(
        "--keyframe-payload-path",
        type=str,
        default=None,
        help="path to a REAL stored keyframe blob; its st_size is COUNTED (overrides "
        "--keyframe-payload-bytes). The warp decode stays FREE (rule-118).",
    )
    ap.add_argument(
        "--keyframe-count",
        type=int,
        default=None,
        help="observability hint: number of stored keyframes (reach-gate schedule; ~13).",
    )
    ap.add_argument("--max-pairs", type=int, default=None, help="(PHASE-B) cap inflate+d_seg pairs for speed")
    ap.add_argument(
        "--measure-dpose",
        action="store_true",
        help="(PHASE-B) measure advisory composed d_pose through the frozen CPU PoseNet (B3)",
    )
    ap.add_argument(
        "--dpose-max-pairs",
        type=int,
        default=16,
        help="(PHASE-B) cap the advisory d_pose pairs (PoseNet CPU forward is slow)",
    )
    ap.add_argument("--skip-inflate", action="store_true", help="(PHASE-B) byte-close only, no inflate/parity")
    ap.add_argument("--out-dir", default=None, help="output dir (default experiments/results/v2_compose_<ts>)")
    args = ap.parse_args(argv)

    # One source-bound target snapshot is explicit input to every phase.  Each
    # phase reopens it before target-dependent arithmetic and report emission.
    frontier_target = _frontier_snapshot_for_args(args)

    out_dir = _resolve(args.out_dir) if args.out_dir else (_REPO / "experiments" / "results" / f"v2_compose_{_utc()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    reports: dict[str, Any] = {}
    if args.phase in ("A", "both"):
        reports["phase_a"] = phase_a(
            args,
            out_dir,
            frontier_target=frontier_target,
        )
        (out_dir / "phase_a_report.json").write_text(json.dumps(reports["phase_a"], indent=2))
        print(f"[phase-A] report -> {out_dir / 'phase_a_report.json'}", flush=True)
        print("\n=== RESIDUAL-INR LAUNCH COMMAND (HOLD for operator GO) ===")
        print(reports["phase_a"]["residual_inr_launch_command"]["command"])
        print("===========================================================\n")
    if args.phase in ("B", "both"):
        reports["phase_b"] = phase_b(
            args,
            out_dir,
            frontier_target=frontier_target,
        )
        (out_dir / "phase_b_report.json").write_text(json.dumps(reports["phase_b"], indent=2))
        print(f"[phase-B] report -> {out_dir / 'phase_b_report.json'}", flush=True)
        if reports["phase_b"].get("staged_exact_eval_cmd"):
            print("\n=== STAGED EXACT-EVAL COMMAND (not run) ===")
            print(reports["phase_b"]["staged_exact_eval_cmd"])
            print("===========================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
