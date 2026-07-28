#!/usr/bin/env python
"""ddm_fd2 — disambiguate fd1r's zero-accept window (realization vs locality).

This is a COMPOSE-not-build probe. It reuses the fd1 GN engine + the launcher's
exact realize/compile/verdict machinery (module
``tools.launch_ddm_joint_descent``) to answer, cheaply and rigorously:

PART A — CANARY (L3 verdict-clearance / Confound self-protection, BLOCKING).
  An acceptance instrument that rejects everything is uninterpretable unless it
  is PROVEN to be able to accept. Positive control: replay a KNOWN-ACCEPTED
  in-family descent (ws3 W_joint SOURCE advisory 26.280224 -> STEP4 advisory
  26.186453) through the EXACT ``_chunked_n600_verdict`` + v19 gate. Assert my
  pipeline reproduces both recorded advisories and the gate returns accept.
  If the canary FAILS -> the instrument is broken; STOP; do NOT adjudicate.

PART B — Q2 REALIZATION vs LOCALITY (the fork adjudicator).
  fd1's n600 realized d_seg was bit-identical for 5/6 GN candidates, but the
  active DOF are GLOBAL (shared_template_dof + island_worldsheet) so an n600
  bit-identical d_seg does NOT imply the block (447-450) argmax is unchanged
  (global DOF can flip block pixels down while flipping off-block pixels up,
  cancelling in the n600 mean). This probe REGENERATES fd1's gn_step-1 delta
  deterministically and measures the realized ARGMAX on the block pairs through
  the exact compile -> receiver -> R -> frozen SegNet path:
    * block-vs-baseline argmax FLIP COUNT (== 0 -> REALIZATION gap at uint8;
      the frame perturbation is below the argmax-flip threshold even on its own
      training block).
    * block d_seg vs GT, candidate vs baseline (direction: improved -> LOCALITY;
      not improved with flips -> block moves wrong).
    * off-block sample argmax flips (to witness the n600 cancellation).

Advisory only: ``[macOS-CPU frozen-scorer advisory]`` — score_claim=False,
promotion_eligible=False, pointer UNMOVED.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# ws3 recorded canonical n600 verdicts (source of the canary reference values).
WS3_SOURCE_ADVISORY = 26.28022354503111  # stage00 (sha 5aa45850)
WS3_STEP4_ADVISORY = 26.186452519033523  # step-4 (sha 9601e777) = fd1 baseline
WS3_STEP4_SHA = "9601e777010b1dc45ed0841e118fcf34c58452324f8730fe9958a3440502e3a4"
WS3_SOURCE_SHA = "5aa45850ab05d47f411583fd7582e27644c5bf289cd6d5bc32c05a52706c433e"
BASE_TICKET = REPO / ".omx/research/configs/ddm_ws2_j7_366_w_joint_20260724.json"
RESUME_FROM = Path(
    "/Volumes/VertigoDataTier/pact/experiments/results/"
    "ddm_ws3_w_joint_exact_history_20260724T132200Z/checkpoints/"
    "01_residual_bucket_realized_acceptance_intra_global000004.npz"
)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _block_argmax(receiver_archive: bytes, ids: tuple[int, ...], segnet: Any) -> np.ndarray:
    """Realized SegNet argmax on frame_1 for the given pair ids (through R)."""
    from experiments.train_witness_realized_through_R_mlx import cpu_verdict_d_seg_argmax_batch
    from tools.launch_ddm_joint_descent import receive_joint_descent_archive

    receiver = receive_joint_descent_archive(receiver_archive)
    camera = receiver.render_camera_pairs(ids)
    # dummy targets: we only need the argmax, so pass zeros of correct shape then
    # recompute; but cpu_verdict_d_seg_argmax_batch returns (rows, argmax).
    dummy = [np.zeros((384, 512), dtype=np.int64) for _ in ids]
    _rows, argmax = cpu_verdict_d_seg_argmax_batch(
        segnet, [camera[i, 1] for i in range(len(ids))], dummy
    )
    return np.stack([np.asarray(a) for a in argmax])  # (len(ids), 384, 512)


def _block_dseg_vs_gt(argmax: np.ndarray, gt: np.ndarray) -> float:
    return float(np.count_nonzero(argmax != gt)) / float(argmax.size)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--gn-steps", type=int, default=1, help="regenerate this many GN steps (Q2 uses gn_step 1)")
    ap.add_argument("--skip-canary", action="store_true", help="DEBUG ONLY — never for a load-bearing verdict")
    ap.add_argument("--offblock-sample", type=int, default=60, help="off-block pairs to sample for cancellation witness")
    args = ap.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx

    from tac.local_acceleration.metal_fused_r_operator import assert_metal_matches_cpu_oracle
    from tac.optimization.ddm_family_d_gn_description import FamilyDGaussNewtonEngineV1
    from tac.optimization.direct_description_joint_descent import (
        DirectDescriptionJointDescentTypedConfigV1,
        ProposalGeometryInfeasibleError,
    )
    from tools.launch_ddm_joint_descent import (
        AdamStateV1,
        DirectDescriptionJointDescentMLXModule,
        _chunked_n600_verdict,
        _load_cpu_frozen_scorers,
        _resolve_input,
        _verify_regular,
        compile_parameterized_archive,
        lift_v15_archive,
        load_mlx_distortion_scorer_adapter_from_upstream,
        load_stage_checkpoint,
        open_stored_npy_memmap,
        parameter_group_indices,
        project_adam_state_geometry,
        realize_parameter_theta,
        realized_training_state,
        temporary_mlx_device,
    )

    config = DirectDescriptionJointDescentTypedConfigV1.from_ticket(BASE_TICKET)
    assert config.typed_config_hash() == "346975b25fce972766ef89ebce437ba87cc2e11e37f709683682246116dbcf93", (
        f"config hash mismatch: {config.typed_config_hash()}"
    )

    os.environ["TAC_MLX_CUSTOM_GROUPED_BACKWARD"] = "1"
    np.random.seed(config.seed)
    started = time.monotonic()
    receipt: dict[str, Any] = {
        "schema": "ddm_fd2_posenull_gn_disambiguation.v1",
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU]",
        "pointer_moved": False,
        "typed_config_hash": config.typed_config_hash(),
        "resume_from": str(RESUME_FROM),
        "acceptance_rule": "v19 realized joint advisory action strictly decreases (UNCHANGED)",
    }

    with temporary_mlx_device("gpu"):
        mx.random.seed(config.seed)
        assert_metal_matches_cpu_oracle(seed=config.seed)
        source_path = _resolve_input(config.source_archive_path)
        cache_path = _resolve_input(config.target_cache_path, allow_authority_cache=True)
        _verify_regular(source_path, expected_bytes=config.source_archive_bytes, expected_sha256=config.source_archive_sha256)
        _verify_regular(cache_path, expected_bytes=config.target_cache_bytes, expected_sha256=config.target_cache_sha256)
        archive = source_path.read_bytes()
        lift = lift_v15_archive(archive)
        groups = parameter_group_indices(lift)
        labels = open_stored_npy_memmap(cache_path, "lstars")
        poses = open_stored_npy_memmap(cache_path, "gt_poses")
        adapter = load_mlx_distortion_scorer_adapter_from_upstream(config.upstream_root, device="cpu")
        model = DirectDescriptionJointDescentMLXModule(
            lift=lift, scorer_adapter=adapter, seg_targets=labels, pose_targets=poses
        )
        state, metadata = load_stage_checkpoint(RESUME_FROM, config=config)
        segnet, posenet = _load_cpu_frozen_scorers(config.upstream_root)

        # Baseline (step-4 W_joint) compiled archive — the fd1 window baseline.
        base_archive, _ = compile_parameterized_archive(lift, state.theta, include_lane_programs=False)
        base_sha = _sha(base_archive)
        receipt["baseline_step4_archive_sha256"] = base_sha
        receipt["baseline_step4_archive_sha_matches_recorded"] = base_sha == WS3_STEP4_SHA

        # ─────────── PART A — CANARY (L3, BLOCKING) ───────────
        if not args.skip_canary:
            t0 = time.monotonic()
            # candidate = step4 (better); baseline reference = ws3 source (worse).
            step4_verdict = _chunked_n600_verdict(
                archive=base_archive, labels=labels, poses=poses,
                segnet=segnet, posenet=posenet, batch_size=config.verdict_batch,
            )
            source_verdict = _chunked_n600_verdict(
                archive=archive, labels=labels, poses=poses,
                segnet=segnet, posenet=posenet, batch_size=config.verdict_batch,
            )
            adv_step4 = float(step4_verdict["advisory_action"])
            adv_source = float(source_verdict["advisory_action"])
            repro_step4 = abs(adv_step4 - WS3_STEP4_ADVISORY) < 1e-5
            repro_source = abs(adv_source - WS3_SOURCE_ADVISORY) < 1e-5
            gate_accepts = adv_step4 < adv_source  # v19 gate: candidate < baseline
            canary_pass = bool(repro_step4 and repro_source and gate_accepts and receipt["baseline_step4_archive_sha_matches_recorded"])
            receipt["canary"] = {
                "kind": "ws3_in_family_source_to_step4_replay",
                "candidate_step4_advisory_measured": adv_step4,
                "candidate_step4_advisory_recorded": WS3_STEP4_ADVISORY,
                "candidate_step4_reproduces": repro_step4,
                "baseline_source_advisory_measured": adv_source,
                "baseline_source_advisory_recorded": WS3_SOURCE_ADVISORY,
                "baseline_source_reproduces": repro_source,
                "v19_gate_accepts_step4_over_source": gate_accepts,
                "canary_pass": canary_pass,
                "wall_seconds": time.monotonic() - t0,
                "sister_corroboration": {
                    "mc1_cb1_hood_static_reassert_admitted_delta_S_joint": -0.0516,
                    "custody": "ddm_mc1_hood_static_reassert_20260724T003346Z/02_measurements + cs1 line 115",
                    "note": "different base (CB1/RG4-PC1 receiver) — corroborates the frozen-scorer instrument CAN emit a strict-negative admitted joint row; NOT replayed through this _chunked_n600_verdict path",
                },
            }
            receipt["canary_pass"] = canary_pass
            print(json.dumps({"canary_pass": canary_pass, "adv_step4": adv_step4, "adv_source": adv_source}))
            if not canary_pass:
                receipt["verdict"] = "PLUMBING_FAULT_CANARY_FAILED__FORK_NOT_ADJUDICATED"
                _write(out_dir, receipt, started)
                print(json.dumps({"verdict": receipt["verdict"]}))
                return 3

        # ─────────── PART B — Q2 REALIZATION vs LOCALITY ───────────
        schedule = config.full_run_schedule
        stage0 = schedule.stages[0]
        active_groups = tuple(stage0.active_groups)
        train_batch = int(schedule.train_batch)
        pair_start = int(schedule.warm_start_pair)
        pair_ids = tuple((pair_start + off) % 600 for off in range(train_batch))
        active = sorted(set().union(*(groups[name] for name in active_groups)))
        receipt["block_pair_ids"] = list(pair_ids)
        receipt["active_groups"] = list(active_groups)

        # baseline block + off-block argmax (frozen reference)
        gt_block = np.stack([np.asarray(labels[i], dtype=np.int64) for i in pair_ids])
        base_block_argmax = _block_argmax(base_archive, pair_ids, segnet)
        base_block_dseg = _block_dseg_vs_gt(base_block_argmax, gt_block)
        rng = np.random.default_rng(0)
        offblock_ids = tuple(
            int(x) for x in rng.choice(
                [p for p in range(600) if p not in set(pair_ids)],
                size=min(args.offblock_sample, 600 - train_batch), replace=False,
            )
        )
        gt_off = np.stack([np.asarray(labels[i], dtype=np.int64) for i in offblock_ids])
        base_off_argmax = _block_argmax(base_archive, offblock_ids, segnet)

        damping = 1.0e-3
        q2_rows: list[dict[str, Any]] = []
        for gn_step in range(1, int(args.gn_steps) + 1):
            base_camera, template_masks, basis, basis_indices, local_theta, _ = realized_training_state(
                lift, state.theta, pair_ids=pair_ids, active_groups=active_groups, include_lane_programs=False
            )
            loss, gradient = model.loss_and_grad(
                local_theta, pair_ids=pair_ids, base_camera=base_camera, template_masks=template_masks,
                realized_secant_basis=basis, realized_secant_indices=basis_indices, pose_objective_weight=0.0,
            )
            step_active = sorted(set(active) | set(basis_indices))
            gradient = gradient.copy()
            gradient[[i for i in range(len(gradient)) if i not in set(step_active)]] = 0.0
            engine = FamilyDGaussNewtonEngineV1(
                model, repository_root=str(REPO), hutchinson_probes=4, seed=config.seed
            )
            delta, diagnostics = engine.propose(
                local_theta, gradient, pair_ids=pair_ids, base_camera=base_camera, template_masks=template_masks,
                realized_secant_basis=basis, realized_secant_indices=basis_indices,
                active_indices=step_active, damping=damping, cg_iterations=6,
            )
            for multiplier in (1.0, 0.5, 0.25):
                candidate_theta = state.theta + np.float32(multiplier) * delta
                try:
                    projected_state, _ev = project_adam_state_geometry(
                        lift, AdamStateV1(step=state.step, theta=np.asarray(candidate_theta, dtype=np.float32),
                                          ema=state.ema, first_moment=state.first_moment, second_moment=state.second_moment),
                    )
                except ProposalGeometryInfeasibleError as exc:
                    q2_rows.append({"gn_step": gn_step, "multiplier": multiplier, "geometry_infeasible": str(exc)})
                    continue
                candidate_theta = projected_state.theta
                # description-level realization (matches launcher realized_changed)
                cand_realized = realize_parameter_theta(lift, candidate_theta)
                base_realized = realize_parameter_theta(lift, state.theta)
                description_changed = not np.array_equal(cand_realized, base_realized)
                try:
                    cand_archive, _ = compile_parameterized_archive(lift, candidate_theta, include_lane_programs=False)
                except ProposalGeometryInfeasibleError as exc:
                    q2_rows.append({"gn_step": gn_step, "multiplier": multiplier, "compile_infeasible": str(exc)})
                    continue
                cand_sha = _sha(cand_archive)
                cand_bytes = len(cand_archive)
                # block realized argmax
                cand_block_argmax = _block_argmax(cand_archive, pair_ids, segnet)
                block_flips = int(np.count_nonzero(cand_block_argmax != base_block_argmax))
                cand_block_dseg = _block_dseg_vs_gt(cand_block_argmax, gt_block)
                # off-block sample realized argmax (cancellation witness)
                cand_off_argmax = _block_argmax(cand_archive, offblock_ids, segnet)
                off_flips = int(np.count_nonzero(cand_off_argmax != base_off_argmax))
                base_off_dseg = _block_dseg_vs_gt(base_off_argmax, gt_off)
                cand_off_dseg = _block_dseg_vs_gt(cand_off_argmax, gt_off)
                row = {
                    "gn_step": gn_step,
                    "multiplier": multiplier,
                    "candidate_archive_sha256": cand_sha,
                    "candidate_archive_bytes": cand_bytes,
                    "description_changed": bool(description_changed),
                    "block_pair_ids": list(pair_ids),
                    "block_argmax_flip_count": block_flips,
                    "block_sites": int(base_block_argmax.size),
                    "block_dseg_vs_gt_baseline": base_block_dseg,
                    "block_dseg_vs_gt_candidate": cand_block_dseg,
                    "block_dseg_delta": cand_block_dseg - base_block_dseg,
                    "offblock_sample_ids_count": len(offblock_ids),
                    "offblock_argmax_flip_count": off_flips,
                    "offblock_sites": int(base_off_argmax.size),
                    "offblock_dseg_vs_gt_baseline": base_off_dseg,
                    "offblock_dseg_vs_gt_candidate": cand_off_dseg,
                    "offblock_dseg_delta": cand_off_dseg - base_off_dseg,
                    "block_proxy_loss": float(loss),
                    "step_norm": float(np.linalg.norm(np.asarray(delta))),
                }
                # per-candidate realization/locality classification
                if block_flips == 0:
                    row["q2_classification"] = "REALIZATION_GAP_NO_BLOCK_ARGMAX_FLIP"
                elif (cand_block_dseg - base_block_dseg) < 0:
                    row["q2_classification"] = "BLOCK_IMPROVED_LOCALITY"
                else:
                    row["q2_classification"] = "BLOCK_MOVED_NOT_IMPROVED"
                q2_rows.append(row)
                print(json.dumps({k: row[k] for k in ("gn_step", "multiplier", "description_changed",
                      "block_argmax_flip_count", "block_dseg_delta", "offblock_argmax_flip_count", "q2_classification")}))
            damping *= 4.0  # match launcher rejection adaptation (no accept here)

        receipt["q2_rows"] = q2_rows
        receipt["diagnostics_gn_step1"] = diagnostics.to_payload()

    _write(out_dir, receipt, started)
    return 0


def _write(out_dir: Path, receipt: dict[str, Any], started: float) -> None:
    receipt["elapsed_seconds"] = time.monotonic() - started
    path = out_dir / "fd2_disambiguation_receipt.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))
    print(json.dumps({"receipt": str(path), "score_claim": False}))


if __name__ == "__main__":
    raise SystemExit(main())
