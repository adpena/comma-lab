#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Emit the B1 HiNeRV 229K full-curriculum launch manifest (PRE-LAUNCH GATE).

Thin CLI around :func:`tac.substrates.hi_nerv.launch_manifest.build_launch_manifest`.
Builds the REAL MLX model (param count + Muon/AdamW partition), reads the REAL
PR95 8-stage factory, verifies stage-8 Muon wiring, and writes the manifest to
``.omx/research/`` (durable) + an SSD copy. The 64-hour MLX run does NOT launch
unless ``manifest_complete_and_self_consistent`` is ``True``.

Usage::

    .venv/bin/python tools/build_b1_launch_manifest.py \
        --run-id 20260609T043613Z \
        --telemetry-path /Volumes/VertigoDataTier/pact/<run_id>/telemetry.jsonl \
        --best-checkpoint-manifest-path .omx/research/b1_best_checkpoint_<run_id>.json

All emitted numbers are ``[macOS-MLX research-signal]`` — the manifest attests
TRAINING-CONFIG readiness, NOT a contest score.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_DIR = REPO_ROOT / ".omx/research"
DEFAULT_SSD_ROOT = Path("/Volumes/VertigoDataTier/pact")


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - git-less env
        return "unknown"


def _default_hallucination_audit() -> dict[str, Any]:
    """The P0 anti-hallucination audit findings, recorded in the manifest."""
    return {
        "directive": (
            "operator 2026-06-09: 'must make sure there are not other "
            "hallucinations or errors like that too' + 'pr95 is not straight "
            "up muon' + 'each vehicle will look different especially vehicle 3'"
        ),
        "param_count_built_from_real_model": True,
        "param_count_taper_36_30_23_17_14_11_8_equals_228903": True,
        "default_340802_param_config_avoided": True,
        "muon_partition_rule_is_selective_not_pure_muon": True,
        "muon_partition_rule_source": (
            "profile_pr95_hnerv_muon_intake.py:183 == tac.optimization.muon."
            "partition_params_for_muon == tac.local_acceleration.pr95_hnerv_mlx."
            "partition_pr95_mlx_parameter_names"
        ),
        "stage8_muon_only_stages_1_to_7_all_adamw": True,
        "factory_schedule_matches_pr95_static_source_ground_truth": True,
        "telemetry_module_reads_real_canonical_surfaces_not_assumptions": True,
        "ema_decay_tension_found": (
            "harness PR95_SOURCE_EMA_DECAY=0.999 when faithful curriculum; "
            "operator + CLAUDE.md EMA non-negotiable require 0.997; resolved "
            "by new --ema-decay trainer flag (default 0.997, overrides 0.999)"
        ),
        "deforestation_atoms_have_zero_loss_consumers_today": True,
        "deforestation_argmax_margin_idea_has_wired_form": (
            "--segnet-distillation-objective boundary_tckd / boundary_kl_t2 / "
            "boundary_argmax_hinge is a wired boundary-concentration seg loss"
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--telemetry-path", required=True)
    parser.add_argument("--best-checkpoint-manifest-path", required=True)
    parser.add_argument("--resume-command", default="")
    parser.add_argument("--exact-eval-command-stub", default="")
    parser.add_argument("--ema-decay", type=float, default=0.997)
    parser.add_argument("--checkpoint-cadence-epochs", type=int, default=250)
    parser.add_argument(
        "--sidecar-export-enabled",
        action="store_true",
        help="ONLY set when the sidecar pays rent (off by default).",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--ssd-root",
        type=Path,
        default=DEFAULT_SSD_ROOT,
        help="SSD tier root for the durable manifest copy.",
    )
    parser.add_argument("--no-write", action="store_true")
    # --- B1 clean-relaunch (BLOCKER 2) ------------------------------------
    parser.add_argument(
        "--clean-baseline",
        action="store_true",
        help=(
            "Emit the b1_clean_pr95_baseline_launch_manifest.v1 (SCALED stage "
            "boundaries + clean-baseline gate fields). Supersedes the diverging "
            "off-spec pilot. Exits rc=2 if the clean gate fails (do NOT launch)."
        ),
    )
    parser.add_argument(
        "--research-total-epochs",
        type=int,
        default=3000,
        help="Clean-baseline reduced pilot budget (scaled PR95 curriculum).",
    )
    parser.add_argument(
        "--grad-clip-max-norm",
        type=float,
        default=1.0,
        help="Wave N+11 stabilizer max_norm (must be > 0 for clean baseline).",
    )
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument(
        "--no-cosine-decay",
        action="store_true",
        help="Disable cosine LR decay in the clean-baseline stabilizer string.",
    )
    parser.add_argument(
        "--superseded-run-id",
        default="",
        help="The KILLED off-spec run this clean relaunch supersedes.",
    )
    return parser.parse_args(argv)


def _write_manifest_text(
    *, args: argparse.Namespace, run_id: str, text: str, default_filename: str
) -> None:
    """Write the manifest text to the durable .omx/research path + SSD copy."""
    out_path = args.output or (DEFAULT_MANIFEST_DIR / default_filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path}")
    try:
        ssd_dir = Path(args.ssd_root) / run_id
        ssd_dir.mkdir(parents=True, exist_ok=True)
        ssd_path = ssd_dir / default_filename
        ssd_path.write_text(text, encoding="utf-8")
        print(f"wrote {ssd_path}")
    except OSError as exc:
        print(f"WARNING: SSD copy skipped ({exc})")


def _main_clean_baseline(args: argparse.Namespace, commit_sha: str) -> int:
    """Emit the B1 clean-relaunch manifest (BLOCKER 2). rc=2 if gate fails."""
    from tac.substrates.hi_nerv.launch_manifest import (
        B1_CANONICAL_DECODER_CHANNELS,
        build_clean_pr95_baseline_launch_manifest,
    )

    run_id = str(args.run_id)
    default_resume = (
        ".venv/bin/python experiments/train_substrate_hi_nerv_mlx_local.py "
        "--full --allow-direct-research-full-launch "
        f"--research-curriculum-total-epochs {int(args.research_total_epochs)} "
        "--pr95-faithful-curriculum --pr95-muon-policy faithful_stage8_only "
        f"--decoder-channels {','.join(str(c) for c in B1_CANONICAL_DECODER_CHANNELS)} "
        "--ema-decay 0.997 --ema-archive-selection "
        f"--grad-clip-max-norm {float(args.grad_clip_max_norm):g} "
        "--resume-from-checkpoint <SSD_RUN_DIR>/checkpoints"
    )
    resume_command = str(args.resume_command) or default_resume
    default_exact_eval = (
        "# B2 exact-eval bridge (contest hardware, NOT MLX): materialize "
        "archive.zip from best EMA-shadow checkpoint -> inflate.sh -> "
        "upstream/evaluate.py --device cpu (Linux x86_64) AND --device cuda (T4)"
    )
    exact_eval_stub = str(args.exact_eval_command_stub) or default_exact_eval

    manifest = build_clean_pr95_baseline_launch_manifest(
        commit_sha=commit_sha,
        run_id=run_id,
        telemetry_path=str(args.telemetry_path),
        best_checkpoint_manifest_path=str(args.best_checkpoint_manifest_path),
        resume_command=resume_command,
        exact_eval_command_stub=exact_eval_stub,
        research_total_epochs=int(args.research_total_epochs),
        ema_decay=float(args.ema_decay),
        checkpoint_cadence_epochs=int(args.checkpoint_cadence_epochs),
        grad_clip_max_norm=float(args.grad_clip_max_norm),
        cosine_decay_enabled=not bool(args.no_cosine_decay),
        warmup_epochs=int(args.warmup_epochs),
        superseded_run_id=str(args.superseded_run_id),
    )
    payload = manifest.as_dict()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    gate = payload["manifest_complete_and_self_consistent"]

    if args.no_write:
        print(text, end="")
        print(f"GATE manifest_complete_and_self_consistent={gate}")
        return 0 if gate else 2

    _write_manifest_text(
        args=args,
        run_id=run_id,
        text=text,
        default_filename=f"b1_clean_pr95_baseline_launch_manifest_{run_id}.json",
    )
    print(f"GATE manifest_complete_and_self_consistent={gate}")
    print(f"  param_count={payload['param_count']} (confirmed={payload['param_count_confirmed']})")
    print(f"  research_total_epochs={payload['research_total_epochs']}")
    print(f"  total_curriculum_epochs={payload['total_curriculum_epochs']} (scaled_not_29650={payload['stage_boundaries_are_scaled_not_canonical_29650']})")
    print(f"  source_weight_amplification={payload['source_weight_amplification']}")
    print(f"  extra_guard_tether_floor_losses={payload['extra_guard_tether_floor_losses']}")
    print(f"  grad_clip_active={payload['grad_clip_active']} stabilizer={payload['stabilizer']!r}")
    print(f"  sidecar_exported={payload['sidecar_exported']} pay_rent_gate_active={payload['pay_rent_gate_active']}")
    print(f"  stage8_muon_status={payload['stage8_muon_status']} stages_all_validated={payload['stages_all_validated']}")
    return 0 if gate else 2


def main(argv: list[str] | None = None) -> int:
    from tac.substrates.hi_nerv.launch_manifest import (
        B1_CANONICAL_DECODER_CHANNELS,
        build_launch_manifest,
    )

    args = parse_args(argv)
    run_id = str(args.run_id)
    commit_sha = _git_sha()

    if bool(args.clean_baseline):
        return _main_clean_baseline(args, commit_sha)

    default_resume = (
        ".venv/bin/python experiments/train_substrate_hi_nerv_mlx_local.py "
        "--full --pr95-faithful-curriculum --pr95-muon-policy faithful_stage8_only "
        f"--decoder-channels {','.join(str(c) for c in B1_CANONICAL_DECODER_CHANNELS)} "
        "--ema-decay 0.997 --ema-archive-selection "
        f"--resume-from-checkpoint <SSD_RUN_DIR>/checkpoints"
    )
    resume_command = str(args.resume_command) or default_resume

    default_exact_eval = (
        "# B2 exact-eval bridge (contest hardware, NOT MLX): "
        "materialize archive.zip from best EMA-shadow checkpoint -> "
        "inflate.sh archive_dir output_dir public_test_video_names.txt -> "
        "upstream/evaluate.py --device cpu (Linux x86_64) AND --device cuda (T4); "
        "emit hi_nerv_backend_only_exact_eval.v1 with archive sha256 + bytes + "
        "runtime-tree sha + recomputed 100*d_seg+sqrt(10*d_pose)+25*bytes/37545489"
    )
    exact_eval_stub = str(args.exact_eval_command_stub) or default_exact_eval

    manifest = build_launch_manifest(
        commit_sha=commit_sha,
        run_id=run_id,
        telemetry_path=str(args.telemetry_path),
        best_checkpoint_manifest_path=str(args.best_checkpoint_manifest_path),
        resume_command=resume_command,
        exact_eval_command_stub=exact_eval_stub,
        ema_decay=float(args.ema_decay),
        checkpoint_cadence_epochs=int(args.checkpoint_cadence_epochs),
        sidecar_export_enabled=bool(args.sidecar_export_enabled),
        hallucination_audit=_default_hallucination_audit(),
        notes=(
            "B1-CAMPAIGN-V2 pre-launch gate. 229K-parity HiNeRV PR95-faithful "
            "8-stage MLX curriculum. operator frontier-override 2026-06-09: "
            "'Full send now: 229K full curriculum' + 'MLX' + 'get full "
            "telemetry all stages all behavior best checkpoint to learn and "
            "analyze' + 'iterate and optimize' + 'pr95 is not straight up "
            "muon' + 'pr95 sets the baseline for rigor ... but it is a "
            "baseline to beat ... each vehicle will look different especially "
            "vehicle 3'."
        ),
    )
    payload = manifest.as_dict()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    if args.no_write:
        print(text, end="")
        gate = payload["manifest_complete_and_self_consistent"]
        print(f"GATE manifest_complete_and_self_consistent={gate}")
        return 0 if gate else 2

    out_path = args.output or (DEFAULT_MANIFEST_DIR / f"b1_launch_manifest_{run_id}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_path}")

    # SSD durable copy (best-effort; manifest is small so local is the SoT).
    try:
        ssd_dir = Path(args.ssd_root) / run_id
        ssd_dir.mkdir(parents=True, exist_ok=True)
        ssd_path = ssd_dir / f"b1_launch_manifest_{run_id}.json"
        ssd_path.write_text(text, encoding="utf-8")
        print(f"wrote {ssd_path}")
    except OSError as exc:
        print(f"WARNING: SSD copy skipped ({exc})")

    gate = payload["manifest_complete_and_self_consistent"]
    print(f"GATE manifest_complete_and_self_consistent={gate}")
    print(f"  param_count={payload['param_count']} (confirmed={payload['param_count_confirmed']})")
    print(f"  muon={payload['muon_param_count']} adamw={payload['adamw_param_count']}")
    print(f"  stage8_muon_status={payload['stage8_muon_status']}")
    print(f"  stages_all_validated={payload['stages_all_validated']}")
    print(f"  ema_decay={payload['ema_decay']}")
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
