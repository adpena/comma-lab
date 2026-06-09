#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Launch the SNeRV-B / G1a MECHANISM CHECK (uncross-the-wires retrain).

Operator-approved 2026-06-09 MECHANISM EXPERIMENT (verbatim: "run F1/F5/SNeRV-B
as mechanism experiments, then exact-eval only if the mechanism passes").

This driver is a THIN actuator. It does NOT reimplement training. It invokes the
canonical SNeRV path-B trainer (the official MFU/HFR/TUB conv renderer WITH the
residual skip-add) through ``tools/run_compact_renderer_mlx_spine_runner.py
--execute-family snerv`` with the score-aware loss objective TURNED ON, which is
the binding fix the full-stack audit
(``.omx/research/snerv_fullstack_extreme_scrutiny_vs_evaluate_py_20260609.md``)
pinpointed: the right architecture was previously trained by the wrong loss
(recon-MSE base with scorer distill weights defaulting to 0.0).

WHAT THE LAUNCH CONFIG CHANGES vs the prior ep22399-class runs (the wire fix):

1. ``--snerv-model-size-adapter snerv_official_mfu_hfr_tub_numeric_primitives_v1``
   selects the REAL official conv MFU/HFR/TUB renderer (Path B, the skip-bearing
   one), NOT the Haar score renderer the ep22399 run used.
2. ``--segnet-direct-live-distillation-weight`` + ``--pose-direct-live-distillation-weight``
   turn ON the renderer-gradient REAL-scorer VJP terms (backprop through the
   frozen SegNet/PoseNet into the decoded pixels), NOT just the lightweight
   student-head surrogates. (The pose-direct-live lever was severed for the SNeRV
   path until commit f5c66f43c; this driver depends on that fix.)
3. ``--segnet-direct-live-class-balanced-ce-weight`` adds the class-balanced
   subcontrol (d_seg is the per-pixel argmax-flip rate; minority/score-relevant
   classes must not be swamped by the dominant road/undrivable mass).
4. ``--recon-loss-stage-weight`` anneals the reconstruction MSE term DOWN (not
   off) so the conditional-mean blur minimizer no longer dominates and collapse
   SegNet's argmax.
5. ``--snerv-official-skip-high-mode full`` keeps the finest skip (do NOT use the
   ``*_mean`` byte-saver modes that collapse it to a frame-invariant blob).

AUTHORITY: ``[macOS-MLX research-signal]`` mechanism check ONLY. ``score_claim=false``,
``promotion_eligible=false``, ``ready_for_exact_eval_dispatch=false``,
``rank_or_kill_eligible=false``. NOT a score. The byte-closed SCORABLE archive is
gated by G1b (bind trained official-conv MLX weights into the official decoder
payload; export blocked today per ``carrier.py`` ``official_mfu_hfr_tub_export_blockers``).

The driver writes the ``snerv_mistake_b_retrain.v1`` manifest, claims the lane,
then spawns the spine runner as a detached ``nohup`` daemon (the foreground bash
dies at SIGURG ~3 min; the daemon survives). It prints the manifest path, output
dir, lane claim, and the launched PID.
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

REPO_ROOT = Path(__file__).resolve().parents[1]

# SSD tier per CLAUDE.md "Local Disk, SSD Spill" non-negotiable (NEVER /tmp in evidence).
DEFAULT_SSD_TIER = Path("/Volumes/VertigoDataTier/pact")
DEFAULT_SSD_TIER_FALLBACK = Path("/Volumes/APDataStore/pact")

MANIFEST_SCHEMA = "snerv_mistake_b_retrain.v1"
LANE_ID = "lane_snerv_mistake_b_g1a_crosswire_20260609"

# The explicit official-primitives modelsize candidate id. The auto-selector
# refuses the official adapter under any hard byte ceiling because its
# pre-training (random-init) predicted archive is ~1.4 GB and requires measured
# post-training feedback; passing the candidate id directly bypasses the
# auto-under-ceiling raise (run_compact_renderer_mlx_spine_runner.py resolves an
# explicit id without re-running the auto selector). Bytes are NOT the G1a goal
# (that is G1b); the mechanism check measures d_seg / d_pose movement.
OFFICIAL_CANDIDATE_ID = (
    "snerv_np600_haar_lv1_lfb1_stepb1_fc36e0_p1_mfu1-2-4_hfr0_t0_"
    "adofficial_oms0p285_int8_symmetric_ceil1500000000"
)
OFFICIAL_ADAPTER = "snerv_official_mfu_hfr_tub_numeric_primitives_v1"
HARD_BYTE_CEILING = 1_500_000_000


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _select_output_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve(strict=False)
    if DEFAULT_SSD_TIER.parent.exists():
        return DEFAULT_SSD_TIER
    if DEFAULT_SSD_TIER_FALLBACK.parent.exists():
        return DEFAULT_SSD_TIER_FALLBACK
    raise SystemExit(
        "no SSD tier available (/Volumes/VertigoDataTier or /Volumes/APDataStore); "
        "pass --output-root explicitly (must NOT be /tmp)"
    )


def _build_spine_argv(args: argparse.Namespace, output_dir: Path) -> list[str]:
    return [
        ".venv/bin/python",
        "tools/run_compact_renderer_mlx_spine_runner.py",
        "--execute-family", "snerv",
        # Mechanism-check manual launch (self-tags manual_launch_false_authority).
        "--allow-manual-compact-family-launch",
        # Path B: the REAL official conv MFU/HFR/TUB renderer (the skip-bearing one).
        "--snerv-model-size-adapter", OFFICIAL_ADAPTER,
        "--modelsize-candidate-id", OFFICIAL_CANDIDATE_ID,
        "--snerv-official-modelsize-mparams", "0.285",
        "--hard-byte-ceiling", str(HARD_BYTE_CEILING),
        # Keep the finest skip (NOT a *_mean collapse mode).
        "--snerv-official-skip-high-mode", "full",
        "--num-pairs", str(int(args.num_pairs)),
        "--epochs", str(int(args.epochs)),
        "--snerv-score-aware-long-training-epochs", str(int(args.epochs)),
        "--snerv-score-aware-long-training-lr", str(float(args.lr)),
        "--snerv-score-aware-long-training-batch-pairs", str(int(args.batch_pairs)),
        "--snerv-score-aware-long-training-optimizer", "pact_muon_adamw",
        "--snerv-score-aware-long-training-eval-roundtrip-ste",
        "--snerv-score-aware-long-training-pr95-faithful-curriculum",
        # Score-aware objective ON (the binding fix).
        "--distillation-device", str(args.distillation_device),
        "--segnet-distillation-weight", str(float(args.segnet_distillation_weight)),
        "--pose-distillation-weight", str(float(args.pose_distillation_weight)),
        # The REAL-scorer renderer-gradient VJP terms (the antidote to blur/collapse).
        "--segnet-direct-live-distillation-weight",
        str(float(args.segnet_direct_live_distillation_weight)),
        "--pose-direct-live-distillation-weight",
        str(float(args.pose_direct_live_distillation_weight)),
        # Class-balanced subcontrol (d_seg = per-pixel argmax-flip rate).
        "--segnet-direct-live-class-balanced-ce-weight",
        str(float(args.segnet_direct_live_class_balanced_ce_weight)),
        # Anneal recon DOWN (not off).
        "--recon-loss-stage-weight", str(float(args.recon_loss_stage_weight)),
        # Scorer-space step guard ON (records TRUE argmax d_seg + pose score term
        # per accepted step => the early exact-pair probe surface). Default on; we
        # are explicit for the manifest.
        # (no flag needed to enable; --no-scorer-space-step-guard would disable it.)
        # Periodic checkpoints so the ep50/100/250 trajectory is auditable.
        "--checkpoint-retention-keep-every-n-epochs",
        str(int(args.checkpoint_keep_every_n_epochs)),
        # Mechanism check: do NOT attempt the byte-closed archive (that is G1b).
        "--skip-snerv-native-mlx-archive-export",
        "--snerv-native-mlx-receiver-proof-timeout", "1800",
        "--output-dir", output_dir.as_posix(),
    ]


def _build_manifest(
    args: argparse.Namespace,
    output_dir: Path,
    spine_argv: list[str],
) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "utc": _utc_stamp(),
        "lane_id": LANE_ID,
        "authority_tier": "telemetry_proxy",
        "axis_tag": "[macOS-MLX research-signal]",
        "metric_family": "mlx_score_aware_long_training_telemetry",
        # False-authority contract (CLAUDE.md MLX non-negotiable).
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotable": False,
        # Operator framing.
        "experiment_kind": "mechanism_check_g1a_uncross_the_wires",
        "operator_approval_verbatim": (
            "run F1/F5/SNeRV-B as mechanism experiments, then exact-eval only if "
            "the mechanism passes"
        ),
        "audit_source": (
            ".omx/research/snerv_fullstack_extreme_scrutiny_vs_evaluate_py_20260609.md"
        ),
        # Renderer / architecture facts.
        "renderer": "path_B_official_mfu_hfr_tub_score_renderer_with_residual_skip_add",
        "renderer_schema": "snerv_mlx_official_mfu_hfr_tub_score_renderer.v1",
        "snerv_model_size_adapter": OFFICIAL_ADAPTER,
        "modelsize_candidate_id": OFFICIAL_CANDIDATE_ID,
        "official_skip_high_mode": "full",
        # Loss config (the binding fix).
        "recon_loss_stage_weight": float(args.recon_loss_stage_weight),
        "recon_anneal": (
            "stage-weight down-weight (not removed); recon base term retained as a "
            "small anchor via loss_weights['recon']"
        ),
        "segnet_distillation_weight": float(args.segnet_distillation_weight),
        "posenet_distillation_weight": float(args.pose_distillation_weight),
        "segnet_direct_live_distillation_weight": float(
            args.segnet_direct_live_distillation_weight
        ),
        "pose_direct_live_distillation_weight": float(
            args.pose_direct_live_distillation_weight
        ),
        "segnet_direct_live_class_balanced_ce_weight": float(
            args.segnet_direct_live_class_balanced_ce_weight
        ),
        "scorer_space_step_guard_enabled": True,
        "depends_on_commit": "f5c66f43c",
        "depends_on_commit_reason": (
            "pose_direct_live_distillation_weight was severed for the SNeRV path "
            "(execute_snerv def param + 2 inner attachment calls + CLI dispatch); "
            "f5c66f43c threads it so the pose-direct-live VJP actually activates"
        ),
        # QAT / source-forward / TUB status (honest, per audit).
        "qat_status": "coder_aware_qat_not_enabled_for_g1a_mechanism_check",
        "source_forward_proof_status": (
            "unproven (official MFU/HFR/TUB source-forward replay missing; "
            "fail-closed blocker, does NOT block mechanism check)"
        ),
        "tub_status": (
            "official TUB inputs not consumed at frame synthesis (DROP_OR_REIFY "
            "open); does NOT block mechanism check"
        ),
        # Early exact-pair probe surface (where TRUE d_seg / d_pose appear).
        "early_exact_pair_probe": {
            "mechanism": (
                "scorer_space_step_guard computes the TRUE argmax d_seg and the "
                "pose score term sqrt(10*d_pose) per accepted training step when "
                "the direct-live SegNet/PoseNet terms are on; these are persisted "
                "to the long-training telemetry rows"
            ),
            "true_dseg_metric_keys": [
                "score_aware_long_training.best_checkpoint_selection."
                "segnet_direct_live_argmax_disagreement",
                "scorer_space_step_guard_post_segnet_argmax_disagreement_<suffix>",
            ],
            "true_dpose_metric_keys": [
                "score_aware_long_training.best_checkpoint_selection."
                "score_aware_composite_parts.raw_pose_direct_live_score_term",
                "score_aware_long_training.best_checkpoint_selection."
                "score_aware_composite_parts.raw_pose_score_term",
                "scorer_space_step_guard_post_pose_score_term_<suffix>",
            ],
            "checkpoint_keep_every_n_epochs": int(args.checkpoint_keep_every_n_epochs),
            "probe_epochs_of_interest": [50, 100, 250],
        },
        # Falsifiable prediction (operator G1 prediction).
        "falsifiable_prediction": (
            "with the REAL-scorer renderer-gradient objective on (segnet + pose "
            "direct-live VJP) + recon annealed down + skip='full' on the "
            "skip-bearing official conv path, the early TRUE argmax d_seg should "
            "drop from the ep22399 ~0.71 baseline toward <0.2 within the probe "
            "window. If it does NOT, that is a real mechanism finding (the "
            "cross-wiring was not the whole story) and MUST be recorded honestly "
            "(Forbidden premature KILL; no name-laundering)."
        ),
        "baseline_reference": {
            "ep22399_avg_segnet_dist": 0.7114705238739649,
            "ep22399_avg_posenet_dist": 163.19407329559326,
            "ep22399_renderer": "Haar score renderer (NOT path B)",
            "ep22399_note": (
                "the ep22399 baseline used the Haar adapter + student-head distill "
                "with recon at base weight; THIS run uses the official conv path B "
                "+ direct-live VJP + recon annealed down"
            ),
        },
        # G1b promotion gate (NOT $0; not attempted here).
        "g1b_promotion_gate": {
            "blocker": (
                "bind trained official-conv MLX weights into the official MFU/HFR/"
                "TUB decoder payload (carrier.py official_mfu_hfr_tub_export_blockers) "
                "for a byte-closed SCORABLE archive"
            ),
            "export_blockers": [
                "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload",
                "snerv_official_mfu_hfr_tub_weight_mapping_missing",
                "snerv_official_mfu_hfr_tub_source_forward_replay_missing",
            ],
            "status": "DEFERRED_required_for_promotion_not_for_mechanism_check",
        },
        # Run config + custody.
        "output_dir": output_dir.as_posix(),
        "num_pairs": int(args.num_pairs),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "batch_pairs": int(args.batch_pairs),
        "distillation_device": str(args.distillation_device),
        "spine_runner_argv": spine_argv,
        "telemetry_fields_requested": [
            "epoch",
            "recon_loss (pr95_stage_recon)",
            "segnet distill loss (pr95_stage_segnet_direct_live_distill)",
            "posenet distill loss (raw_pose_direct_live_score_term)",
            "true argmax d_seg (segnet_direct_live_argmax_disagreement)",
            "true PoseNet d_pose / pose score term (raw_pose_score_term)",
            "source_forward_survival (official source-forward blockers)",
            "renderer noncollapse (occupied_class_fraction / argmax disagreement)",
            "frame0/frame1 diversity (scorer_input contrast/std telemetry)",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-pairs", type=int, default=600)
    parser.add_argument(
        "--epochs",
        type=int,
        default=600,
        help=(
            "Bounded epoch budget. Enough to see the ep50/100/250 probes move "
            "(mechanism check, not a full campaign)."
        ),
    )
    parser.add_argument("--lr", type=float, default=2.7e-05)
    parser.add_argument("--batch-pairs", type=int, default=4)
    parser.add_argument("--distillation-device", default="cpu")
    parser.add_argument("--segnet-distillation-weight", type=float, default=1.0)
    parser.add_argument("--pose-distillation-weight", type=float, default=1.0)
    parser.add_argument(
        "--segnet-direct-live-distillation-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--pose-direct-live-distillation-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--segnet-direct-live-class-balanced-ce-weight", type=float, default=0.5
    )
    parser.add_argument(
        "--recon-loss-stage-weight",
        type=float,
        default=0.2,
        help="Recon stage weight (down-weighted from 1.0, NOT removed).",
    )
    parser.add_argument(
        "--checkpoint-keep-every-n-epochs",
        type=int,
        default=50,
        help="Keep a checkpoint every N epochs (aligns to the probe epochs).",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--agent",
        default="snerv_crosswire_fix_and_launch_20260609",
        help="Lane-claim agent label.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the manifest + print the argv WITHOUT claiming the lane or launching.",
    )
    args = parser.parse_args(argv)

    stamp = _utc_stamp()
    output_root = _select_output_root(args.output_root)
    output_dir = output_root / f"snerv_mistake_b_g1a_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    spine_argv = _build_spine_argv(args, output_dir)
    manifest = _build_manifest(args, output_dir, spine_argv)
    manifest_path = output_dir / "snerv_mistake_b_retrain_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.dry_run:
        print(json.dumps({
            "dry_run": True,
            "manifest_path": manifest_path.as_posix(),
            "output_dir": output_dir.as_posix(),
            "spine_argv": spine_argv,
        }, indent=2, sort_keys=True))
        return 0

    # Claim the lane (cross-agent coordination non-negotiable).
    instance_job_id = f"snerv_mistake_b_g1a_{stamp}"
    claim = subprocess.run(
        [
            ".venv/bin/python", "tools/claim_lane_dispatch.py", "claim",
            "--lane-id", LANE_ID,
            "--platform", "local_mlx_macos",
            "--instance-job-id", instance_job_id,
            "--agent", str(args.agent),
            "--status", "active_local_mlx",
            "--notes", (
                "SNeRV-B / G1a MECHANISM CHECK [macOS-MLX research-signal] "
                "NON-PROMOTABLE: path-B official MFU/HFR/TUB conv renderer + "
                "REAL-scorer direct-live SegNet+PoseNet VJP ON + recon annealed "
                "down + skip=full (depends on commit f5c66f43c pose_direct_live "
                "wire fix). G1b export gate deferred. "
                f"manifest={manifest_path.as_posix()}"
            ),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    sys.stderr.write(claim.stdout + claim.stderr)
    if claim.returncode != 0:
        print(json.dumps({
            "launched": False,
            "lane_claim_refused": True,
            "lane_claim_rc": claim.returncode,
            "manifest_path": manifest_path.as_posix(),
        }, indent=2, sort_keys=True))
        return claim.returncode

    # Detached nohup daemon (SIGURG-safe; foreground bash dies ~3 min).
    log_path = output_dir / "spine_runner_daemon.log"
    # Shell-quote argv safely.
    import shlex
    quoted = " ".join(shlex.quote(a) for a in spine_argv)
    daemon_cmd = (
        f"nohup bash -c {shlex.quote(quoted)} "
        f"< /dev/null > {shlex.quote(log_path.as_posix())} 2>&1 & echo $!"
    )
    proc = subprocess.run(
        ["bash", "-lc", daemon_cmd],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ},
    )
    pid_str = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    try:
        pid = int(pid_str)
    except ValueError:
        pid = None

    manifest["launched_pid"] = pid
    manifest["daemon_log_path"] = log_path.as_posix()
    manifest["lane_claim_instance_job_id"] = instance_job_id
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "launched": pid is not None,
        "pid": pid,
        "lane_id": LANE_ID,
        "instance_job_id": instance_job_id,
        "manifest_path": manifest_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "daemon_log_path": log_path.as_posix(),
        "axis_tag": "[macOS-MLX research-signal]",
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }, indent=2, sort_keys=True))
    return 0 if pid is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
