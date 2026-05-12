#!/usr/bin/env python3
"""NeRVdc-as-renderer — production trainer (NeRV-family completion).

Per operator directive 2026-05-11. Mirrors Lane 12-v2 trainer pattern.
NeRVdc adds decoder-conditioning on previous-frame summary (deterministic
at inflate-time; zero archive cost).

REPRESENTATION_ARCHIVE_GRAMMAR_BLUEPRINT (per CLAUDE.md Catalog #124):
  archive_grammar: monolithic single-file 0.bin per src/tac/nervdc_as_renderer.py
  parser_section_manifest: ARCHIVE_GRAMMAR_NERVDC in src/tac/nervdc_as_renderer.py
  inflate_runtime_loc_budget: substrate_engineering — Phase B inflate ≤200 LOC
  runtime_dep_closure: torch + brotli (zero dep on tac.*)
  export_format: nervdc_phase_a_monolithic_singlefile_0bin
  score_aware_loss: train_step_nervdc with diff rgb_to_yuv6 +
    load_differentiable_scorers (Lagrangian)
  bolt_on_loc_budget: substrate_engineering
  no_op_detector_planned: export_nervdc_to_archive returns sha256

Cost when dispatched: ~$30-50. Operator-gated.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LANE_ID = "lane_nervdc_as_renderer"
SCHEMA_VERSION = "1.0.0-nervdc-as-renderer-production"
PREDICTED_DELTA_SCORE = (
    "[predicted; HNeRV parity discipline; decoder-conditioning exploits "
    "temporal coherence in driving video for ~1.15× param efficiency at "
    "fixed bytes per Wang 2024 NeRVdc; pose-axis marginal 2.71× SegNet at "
    "PR106 r2 frontier]"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NeRVdc-as-renderer production trainer")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--ema-decay", type=float, default=0.997)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--cond-dim", type=int, default=8)
    p.add_argument("--base-channels", type=int, default=36)
    p.add_argument("--n-pairs", type=int, default=600)
    p.add_argument("--lambda-seg", type=float, default=100.0)
    p.add_argument("--lambda-pose", type=float, default=288.6751345948129)
    p.add_argument("--video-path", type=Path, default=REPO_ROOT / "upstream" / "videos" / "0.mkv")
    p.add_argument("--seed", type=int, default=20260511)
    p.add_argument("--smoke", action="store_true", default=False)
    p.add_argument("--auth-eval", action="store_true", default=False)
    p.add_argument("--phase-b-auth-memo", type=str, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.auth_eval:
        if args.phase_b_auth_memo is None:
            raise SystemExit("[nervdc] --auth-eval refused: requires --phase-b-auth-memo (Catalog #150)")
        from tac.lane_12_v2_nerv_as_renderer import phase_b_preconditions_status
        status = phase_b_preconditions_status(
            consult_session_state=True, auth_memo_path=Path(args.phase_b_auth_memo),
        )
        if status.get("operator_phase_b_authorization") != "MET":
            raise SystemExit(f"[nervdc] --auth-eval refused; status: {status}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("[nervdc] --device cuda requested but CUDA unavailable")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    import random
    import numpy as np
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally
    patch_upstream_yuv6_globally()

    from tac.nervdc_as_renderer import (
        NeRVdcConfig, NeRVdcLatentTable, NeRVdcRenderer,
        _make_synthetic_pair_batch_for_smoke,
        export_nervdc_to_archive,
    )
    from tac.training import EMA

    config = NeRVdcConfig(
        latent_dim=args.latent_dim, cond_dim=args.cond_dim,
        base_channels=8 if args.smoke else args.base_channels,
        n_pairs=4 if args.smoke else args.n_pairs,
        lambda_seg=args.lambda_seg, lambda_pose=args.lambda_pose,
        cuda_required=(args.device == "cuda" and not args.smoke),
    )
    renderer = NeRVdcRenderer(config).to(device)
    latent_table = NeRVdcLatentTable(config.n_pairs, config.latent_dim).to(device)

    n_params = sum(p.numel() for p in renderer.parameters()) + \
               sum(p.numel() for p in latent_table.parameters())
    print(f"[nervdc] total params={n_params:,} device={device}")

    ema_renderer = EMA(renderer, decay=args.ema_decay)
    ema_latents = EMA(latent_table, decay=args.ema_decay)

    if args.smoke:
        for _ in range(2):
            pair_indices, gt_pairs = _make_synthetic_pair_batch_for_smoke(
                batch_size=2, latent_dim=config.latent_dim,
                eval_size=config.eval_size, n_pairs=config.n_pairs, seed=args.seed,
            )
            pair_indices = pair_indices.to(device); gt_pairs = gt_pairs.to(device)
            z = latent_table(pair_indices)
            decoded = renderer(z, prev_frame_uint8=None)
            assert decoded.shape == (2, 2, 3, *config.eval_size)
        print("[nervdc] SMOKE: shape check passed")
    else:
        raise SystemExit(
            "[nervdc] non-smoke training is operator-gated per CLAUDE.md scaffold-only directive."
        )

    archive_path = args.output_dir / "0.bin"
    archive_sha = export_nervdc_to_archive(
        renderer=renderer, latent_table=latent_table, output_path=archive_path,
    )
    provenance = {
        "schema": SCHEMA_VERSION, "lane_id": LANE_ID,
        "started_at_utc": started_at, "device": str(device),
        "smoke": bool(args.smoke), "archive_sha256": archive_sha,
        "archive_bytes": archive_path.stat().st_size,
        "predicted_delta_score": PREDICTED_DELTA_SCORE,
        "score_claim": False, "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "compliance_tags": [
            "ema_0p997", "diff_rgb_to_yuv6_patched", "no_mps_authoritative",
            "no_synthetic_outside_smoke", "no_tmp_paths", "score_claim_false",
            "decoder_conditioning_deterministic_at_inflate",
        ],
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"[nervdc] archive sha256={archive_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
