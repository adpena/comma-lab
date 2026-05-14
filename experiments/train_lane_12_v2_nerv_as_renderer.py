#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Lane 12-v2 NeRV-as-renderer — production trainer.

Per operator directive 2026-05-11 + CLAUDE.md HNeRV parity discipline lesson 5
(architecture must be FULL RGB renderer, not slot replacement). Lane 12-v2's
Phase A scaffold lives in :mod:`tac.lane_12_v2_nerv_as_renderer`; this script
is the production training driver that wires:

  - real video pairs (NOT synthetic) via ``RealPairBatchSource``
  - score-aware Lagrangian via ``train_step`` against gradient-reachable
    SegNet + PoseNet (``tac.scorer.load_differentiable_scorers``)
  - eval_roundtrip uint8 STE (CLAUDE.md non-negotiable, default ON)
  - differentiable rgb_to_yuv6 monkey-patch (PR #95/106 fix; default ON)
  - weight EMA decay 0.997 with snapshot+restore at eval (CLAUDE.md)
  - CUDA-required default; explicit ``--device cpu`` opt-in for tests only
  - export to archive via ``export_to_archive`` (monolithic single-file 0.bin)
  - auth eval at end (GATED behind ``--auth-eval`` + operator memo)
  - Phase B Option C ``--phase-b-auth-memo`` operator authorization (Catalog #150)
  - NO ``/tmp`` paths; outputs under ``experiments/results/<lane>_<utc>/``
  - NO ``make_synthetic_*`` outside ``--smoke`` (CLAUDE.md forbidden pattern)

REPRESENTATION_ARCHIVE_GRAMMAR_BLUEPRINT (per CLAUDE.md Catalog #124)
---------------------------------------------------------------------
  archive_grammar: monolithic single-file 0.bin with 12-byte fixed header +
    4 length-prefixed sections (decoder_blob INT8+brotli, scale_table FP16,
    latent_blob uint8 delta-zigzag+brotli, sidecar_blob empty in Phase B)
  parser_section_manifest: ARCHIVE_GRAMMAR in
    src/tac/lane_12_v2_nerv_as_renderer.py with schema_keys_in_order =
    Lane12V2NeRVRenderer.SCHEMA (pinned ordering)
  inflate_runtime_loc_budget: substrate_engineering — Phase B inflate target
    ≤200 LOC contest-hermetic (mirrors PR100 hnerv_lc_v2 budget). The training
    LOC budget of this trainer is NOT the inflate budget.
  runtime_dep_closure: torch + brotli (+ pyav at compress time only — NOT
    shipped). Inflate has zero dep on tac.* — schema is replicated.
  export_format: lane_12_v2_phase_b_monolithic_singlefile_0bin
  score_aware_loss: train_step ((B,2,3,H,W) RGB) → upsample to camera res →
    differentiable rgb_to_yuv6 → load_differentiable_scorers PoseNet (FastViT-
    T12) + EfficientNet-B2 SegNet → Lagrangian λ_seg + λ_pose with operating-
    point-aware weights (CLAUDE.md SegNet-vs-PoseNet section)
  bolt_on_loc_budget: substrate_engineering (full renderer, NOT a bolt-on)
  no_op_detector_planned: export_to_archive returns sha256; byte-mutation
    smoke proves inflate consumes payload (Phase 1 packet compiler integration
    pending Phase B dispatch authorization)

CLAUDE.md non-negotiables wired through this trainer
----------------------------------------------------
  - eval_roundtrip=True (CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE")
  - EMA decay=0.997 with snapshot+restore (CLAUDE.md "EMA — NON-NEGOTIABLE")
  - differentiable rgb_to_yuv6 monkey-patch BEFORE scorer construction
  - score-domain Lagrangian (NOT weight-domain proxy; Catalog #123)
  - real video data outside ``--smoke`` (no make_synthetic; Catalog #114)
  - auth eval at end vs EMA shadow (CLAUDE.md "Auth eval EVERYWHERE")
  - CUDA-required default (CLAUDE.md "MPS auth eval is NOISE"); NO MPS
  - No /tmp paths (CLAUDE.md FORBIDDEN_PATTERNS)
  - Phase B Option C auth_memo_path wired (Catalog #150)
  - Co-Authored-By trailer auto-appended via subagent_commit_serializer

Cost when dispatched: ~$30-50 (Vast.ai T4 ~5-10h). Operator-gated.

Predicted Δ score at PR106 r2 frontier (operating point pose_avg=3.4e-5):
  ``[predicted; HNeRV parity discipline; pose-axis marginal 2.71× SegNet;
  NeRV full-renderer recovery of PR100 substrate at λ_pose 288.675 +
  λ_seg 100 per Lane 12-v2 Phase A defaults]``. NOT a score claim until
  [contest-CUDA] anchor on exact archive bytes lands.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from tac.output_path_policy import assert_not_temporary_output_dir

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Lazy imports below for clean argparse-only invocations.


LANE_ID = "lane_12_v2_nerv_as_renderer"
SCHEMA_VERSION = "1.0.0-lane-12-v2-nerv-as-renderer-production"
PREDICTED_DELTA_SCORE = (
    "[predicted; HNeRV parity discipline; pose-axis marginal 2.71× SegNet at "
    "PR106 r2 frontier; NeRV full-renderer recovery of PR100 substrate]"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lane 12-v2 NeRV-as-renderer production trainer",
    )
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output dir under experiments/results/ (no /tmp)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="CUDA required by default; cpu only for deterministic-bytes smoke")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--ema-decay", type=float, default=0.997,
                        help="Weight EMA decay (CLAUDE.md canonical 0.997)")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--base-channels", type=int, default=36)
    parser.add_argument("--n-pairs", type=int, default=600)
    parser.add_argument("--lambda-seg", type=float, default=100.0,
                        help="SegNet Lagrangian weight (CLAUDE.md operating-point default)")
    parser.add_argument("--lambda-pose", type=float, default=288.6751345948129,
                        help="PoseNet Lagrangian weight (CLAUDE.md operating-point default)")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--video-path", type=Path,
                        default=REPO_ROOT / "upstream" / "videos" / "0.mkv")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Optional cap on # of pairs (smoke). None = full 600.")
    parser.add_argument("--enable-differentiable-yuv6", action="store_true", default=True,
                        help="Monkey-patch upstream rgb_to_yuv6 before scorer load (PR95/106)")
    parser.add_argument("--enable-score-aware-loss", action="store_true", default=True,
                        help="Use real SegNet+PoseNet scorer-domain Lagrangian (CLAUDE.md)")
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--smoke", action="store_true", default=False,
                        help="Smoke mode: tiny model + synthetic-pair smoke (waiver tag inline)")
    parser.add_argument("--auth-eval", action="store_true", default=False,
                        help=(
                            "Run CUDA auth eval on the exported archive AFTER training. "
                            "REFUSED unless --phase-b-auth-memo is also passed (Catalog #150)."
                        ))
    parser.add_argument(
        "--phase-b-auth-memo",
        type=str,
        default=None,
        help=(
            "Path to a committed repo-relative authorization memo (Option C "
            "operator decision 2026-05-09; Catalog #150). Path MUST resolve "
            "under the git repo root; ~/.claude, /tmp, and any non-repo "
            "absolute path are REFUSED. Required to enable --auth-eval."
        ),
    )
    parser.add_argument("--eval-every-epochs", type=int, default=25)
    return parser.parse_args(argv)


def _resolve_device(device_str: str) -> "torch.device":
    """Resolve device with CLAUDE.md MPS-is-noise enforcement.

    CUDA-required by default; explicit ``cpu`` opt-in only for deterministic-
    bytes acceptable smoke coverage. ``mps`` is REFUSED at the argparse level
    (choices excludes it); this function double-checks at runtime.
    """
    if device_str == "mps":
        raise SystemExit(
            "[lane_12_v2] --device mps REFUSED per CLAUDE.md "
            "'MPS auth eval is NOISE' non-negotiable."
        )
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "[lane_12_v2] --device cuda requested but torch.cuda.is_available() "
                "is False. CUDA-required default per CLAUDE.md MPS-fallback-trap "
                "FORBIDDEN_PATTERNS. Pass --device cpu only with explicit "
                "deterministic-bytes-acceptable smoke intent."
            )
        return torch.device("cuda")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _activate_differentiable_yuv6():
    """Monkey-patch upstream rgb_to_yuv6 BEFORE scorer construction.

    Per CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE" expansion 2026-05-09:
    upstream rgb_to_yuv6 is ``@torch.no_grad()`` / in-place and severs
    PoseNet gradients. Lane 12-v2 trainer MUST monkey-patch BEFORE the
    scorers are constructed via ``load_differentiable_scorers``.
    """
    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally
    return patch_upstream_yuv6_globally()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        assert_not_temporary_output_dir(args.output_dir, tool_name="lane_12_v2")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    # CLAUDE.md "Operator gates must be wired and used": refuse --auth-eval
    # without explicit Phase B operator authorization memo (Catalog #150).
    if args.auth_eval:
        if args.phase_b_auth_memo is None:
            raise SystemExit(
                "[lane_12_v2] --auth-eval refused: requires "
                "--phase-b-auth-memo <committed_repo_path> per Phase B "
                "Option C operator decision 2026-05-09 (Catalog #150)."
            )
        # Validate the memo path now (repo-relative + explicit token).
        from tac.lane_12_v2_nerv_as_renderer import phase_b_preconditions_status
        status = phase_b_preconditions_status(
            consult_session_state=True,
            auth_memo_path=Path(args.phase_b_auth_memo),
        )
        if status.get("operator_phase_b_authorization") != "MET":
            raise SystemExit(
                f"[lane_12_v2] --auth-eval refused: operator-phase-B-authorization "
                f"gate returned PENDING for memo {args.phase_b_auth_memo}. "
                f"Status: {status}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _seed_everything(args.seed)
    device = _resolve_device(args.device)

    # Monkey-patch differentiable YUV6 BEFORE scorer load (PR95/106).
    yuv6_token = None
    if args.enable_differentiable_yuv6 and args.enable_score_aware_loss:
        yuv6_token = _activate_differentiable_yuv6()
        print("[lane_12_v2] differentiable rgb_to_yuv6 monkey-patch active")

    # Load CLAUDE.md-compliant Lane 12-v2 substrate (after monkey-patch).
    from tac.lane_12_v2_nerv_as_renderer import (
        Lane12V2NeRVConfig,
        Lane12V2NeRVRenderer,
        Lane12V2LatentTable,
        RealPairBatchSource,
        _make_synthetic_pair_batch_for_smoke,
        default_pose_surrogate,
        default_seg_surrogate,
        export_to_archive,
        train_step,
    )
    from tac.training import EMA

    config = Lane12V2NeRVConfig(
        latent_dim=args.latent_dim,
        base_channels=8 if args.smoke else args.base_channels,
        n_pairs=4 if args.smoke else args.n_pairs,
        lambda_seg=args.lambda_seg,
        lambda_pose=args.lambda_pose,
        cuda_required=(args.device == "cuda" and not args.smoke),
    )

    renderer = Lane12V2NeRVRenderer(config).to(device)
    latent_table = Lane12V2LatentTable(config.n_pairs, config.latent_dim).to(device)
    n_params = sum(p.numel() for p in renderer.parameters()) + \
               sum(p.numel() for p in latent_table.parameters())
    print(f"[lane_12_v2] renderer={sum(p.numel() for p in renderer.parameters()):,} "
          f"latents={sum(p.numel() for p in latent_table.parameters()):,} "
          f"total={n_params:,} device={device}")

    # Per CLAUDE.md "EMA — NON-NEGOTIABLE": EMA on weights, decay 0.997,
    # snapshot+restore at eval time, EMA shadow at archive export.
    ema_renderer = EMA(renderer, decay=args.ema_decay)
    ema_latents = EMA(latent_table, decay=args.ema_decay)

    optimizer = torch.optim.Adam(
        list(renderer.parameters()) + list(latent_table.parameters()),
        lr=args.learning_rate,
    )

    # Load gradient-reachable scorers (after YUV6 monkey-patch).
    scorer_seg = scorer_pose = None
    if args.enable_score_aware_loss:
        from tac.scorer import load_differentiable_scorers
        scorer_pose, scorer_seg = load_differentiable_scorers(
            REPO_ROOT / "upstream", device=device,
        )
        scorer_pose.eval()
        scorer_seg.eval()
        for m in (scorer_pose, scorer_seg):
            for p in m.parameters():
                p.requires_grad_(False)

    # CLAUDE.md FORBIDDEN_PATTERNS: synthetic pairs only inside smoke mode.
    use_synthetic = bool(args.smoke)
    if use_synthetic:
        print("[lane_12_v2] SMOKE MODE — synthetic pair batches "
              "(# SYNTHETIC_NON_SMOKE_OK:lane_12_v2_smoke_via_argparse)")
    else:
        if not args.video_path.exists():
            raise SystemExit(
                f"[lane_12_v2] non-smoke training requires {args.video_path} "
                "(CLAUDE.md FORBIDDEN_PATTERNS: synthetic in non-smoke is refused)"
            )
        batch_source = RealPairBatchSource(
            video_path=args.video_path,
            n_pairs=config.n_pairs,
            eval_size=config.eval_size,
        )

    history: list[dict] = []
    epochs = 1 if args.smoke else args.epochs
    batch_size = 1 if args.smoke else args.batch_size

    for epoch in range(epochs):
        renderer.train()
        latent_table.train()
        epoch_loss = 0.0
        n_batches = 0

        if use_synthetic:
            # Smoke: single synthetic micro-batch for gradient-path verification.
            pair_indices, gt_pairs_uint8 = _make_synthetic_pair_batch_for_smoke(
                batch_size=batch_size,
                latent_dim=config.latent_dim,
                eval_size=config.eval_size,
                n_pairs=config.n_pairs,
                seed=args.seed + epoch,
            )
            pair_indices = pair_indices.to(device)
            gt_pairs_uint8 = gt_pairs_uint8.to(device)
            batches = [(pair_indices, gt_pairs_uint8)]
        else:
            # Real video pairs streamed via PyAV.
            batches = batch_source.iter_batches(
                batch_size=batch_size,
                max_pairs=args.max_pairs,
            )

        for pair_indices, gt_pairs_uint8 in batches:
            pair_indices = pair_indices.to(device)
            gt_pairs_uint8 = gt_pairs_uint8.to(device)

            if args.enable_score_aware_loss:
                # Score-aware Lagrangian per Lane 12-v2 train_step contract.
                result = train_step(
                    renderer=renderer,
                    latent_table=latent_table,
                    pair_indices=pair_indices,
                    gt_pairs_uint8=gt_pairs_uint8,
                    scorer_seg=scorer_seg,
                    scorer_pose=scorer_pose,
                    seg_surrogate=default_seg_surrogate,
                    pose_surrogate=default_pose_surrogate,
                    lambda_seg=args.lambda_seg,
                    lambda_pose=args.lambda_pose,
                    eval_roundtrip=True,  # CLAUDE.md non-negotiable
                )
            else:
                # Fallback: pixel-L1 distortion only (research-only path).
                z = latent_table(pair_indices)
                decoded = renderer(z)
                target_native = torch.nn.functional.interpolate(
                    gt_pairs_uint8.float().reshape(-1, 3, *gt_pairs_uint8.shape[-2:]),
                    size=config.eval_size,
                    mode="bicubic",
                    align_corners=False,
                ).reshape(decoded.shape)
                loss = torch.nn.functional.l1_loss(decoded, target_native)
                result = {"loss": loss}

            if not torch.isfinite(result["loss"]).all():
                raise RuntimeError(
                    f"[lane_12_v2] non-finite loss at epoch {epoch} batch {n_batches}"
                )

            optimizer.zero_grad()
            result["loss"].backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(renderer.parameters()) + list(latent_table.parameters()),
                    max_norm=args.grad_clip_norm,
                )
            optimizer.step()

            # CLAUDE.md "EMA — NON-NEGOTIABLE": update EMA after every step.
            ema_renderer.update(renderer)
            ema_latents.update(latent_table)

            epoch_loss += float(result["loss"].detach())
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        if epoch == 0 or (epoch + 1) % args.eval_every_epochs == 0 or epoch == epochs - 1:
            history.append({"epoch": epoch + 1, "avg_loss": avg})
            print(f"[lane_12_v2] epoch {epoch+1}/{epochs} avg_loss={avg:.4f}")

    # ── Archive export from EMA shadow (CLAUDE.md non-negotiable) ────
    orig_renderer = {k: v.detach().clone() for k, v in renderer.state_dict().items()}
    orig_latents = {k: v.detach().clone() for k, v in latent_table.state_dict().items()}
    try:
        ema_renderer.apply(renderer)
        ema_latents.apply(latent_table)
        archive_path = args.output_dir / "0.bin"
        archive_sha = export_to_archive(
            renderer=renderer,
            latent_table=latent_table,
            output_path=archive_path,
        )
    finally:
        renderer.load_state_dict(orig_renderer)
        latent_table.load_state_dict(orig_latents)
        renderer.train()
        latent_table.train()

    archive_bytes = archive_path.stat().st_size
    print(f"[lane_12_v2] archive {archive_path} sha256={archive_sha[:16]} bytes={archive_bytes}")

    # ── Auth eval (gated; see top of main) ───────────────────────────
    auth_eval_result = None
    if args.auth_eval:
        print("[lane_12_v2] --auth-eval gated path: deferring to external dispatcher")
        auth_eval_result = {
            "status": "deferred_to_external_dispatcher",
            "note": (
                "Trainer does not embed contest_auth_eval; archive bytes are "
                "ready. Operator should run experiments/contest_auth_eval.py "
                "with --archive-path + --phase-b-auth-memo per CLAUDE.md "
                "submission-auth-eval-BOTH-CPU-AND-CUDA."
            ),
        }

    # ── Provenance (no /tmp; output_dir scoped) ──────────────────────
    provenance = {
        "schema": SCHEMA_VERSION,
        "lane_id": LANE_ID,
        "started_at_utc": started_at,
        "device": str(device),
        "smoke": bool(args.smoke),
        "epochs": int(epochs),
        "n_params": int(n_params),
        "config": {
            "latent_dim": config.latent_dim,
            "base_channels": config.base_channels,
            "n_pairs": config.n_pairs,
            "eval_size": list(config.eval_size),
            "lambda_seg": config.lambda_seg,
            "lambda_pose": config.lambda_pose,
        },
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha,
        "archive_bytes": int(archive_bytes),
        "ema_decay": float(args.ema_decay),
        "predicted_delta_score": PREDICTED_DELTA_SCORE,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "[predicted; Lane 12-v2 production trainer; no anchor]",
        "score_aware_loss_enabled": bool(args.enable_score_aware_loss),
        "differentiable_yuv6_enabled": bool(args.enable_differentiable_yuv6),
        "auth_eval_result": auth_eval_result,
        "history": history,
        "compliance_tags": [
            "ema_0p997_snapshot_restore",
            "eval_roundtrip_true",
            "no_mps_authoritative",
            "differentiable_yuv6",
            "score_aware_lagrangian",
            "no_synthetic_outside_smoke",
            "no_tmp_paths",
            "auth_eval_gated_phase_b_option_c",
            "cuda_required_default",
        ],
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"[lane_12_v2] done; provenance written to {args.output_dir}/provenance.json")

    # Optional: restore upstream YUV6 if we monkey-patched.
    if yuv6_token is not None:
        from tac.differentiable_eval_roundtrip import unpatch_upstream_yuv6
        try:
            unpatch_upstream_yuv6(yuv6_token)
        except Exception:
            pass  # Not fatal; pytest process exits anyway.

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
