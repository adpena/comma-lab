#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""TOP-2 LOCAL-CPU advisory smoke: F4 PoseNet summary(512) per-pattern inverter.

Authority: routing directive
``.omx/research/codex_routing_directive_top2_f4_summary_512_per_pattern_inverter_prototype_20260518.md``
(commit ``248120281``) + operator PROCEED-ALL-3 LOCAL-CPU authorization
2026-05-18 verbatim ``"Can those be local cpu"`` + ``"approved all"``.

Lane: ``lane_top_3_reclamation_local_cpu_smoke_20260518``
(research_only=true; substrate_engineering scope; framework validation only).

Hook: PoseNet ``summarizer`` Sequential output (per upstream/modules.py:67-78):
  vision(2048) -> Linear(2048->512) -> ReLU -> ResBlock(512) = summary

The hook captures `summary` AFTER the summarizer Sequential, which IS what
the directive memo refers to as "summary(512)".

[verified-against:upstream/modules.py:lines-22-26-67-78-PoseNet-summarizer
+ submissions/a1/src/codec.py (per-pattern A1 latents drive frame selection)
+ tac.contest_exploits.a1_specialized_inverter sister Codex landing
+ 0701c323b A1 binary distillation framework spec]

Canonical-vs-unique decision per layer:
  - VQ + FP4 + 50% sparse + Brotli engine: ADOPT canonical
    ``tac.contest_exploits.a1_specialized_inverter.build_a1_specialized_inverter``.
  - PoseNet feature extraction at summary(512): FORK locally (canonical helper
    takes precomputed feature matrices; we register a forward hook on PoseNet).
  - Video decode via canonical ``frame_utils.AVVideoDataset``: ADOPT canonical.
  - Local-CPU advisory wrapping + cascade verdict: FORK locally (same as TOP-1).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "upstream"))
sys.path.insert(0, str(REPO_ROOT / "submissions" / "a1" / "src"))

from tac.contest_exploits.a1_specialized_inverter import (  # noqa: E402
    A1SpecializedInverterConfig,
    build_a1_specialized_inverter,
)


def extract_posenet_summary_features_cpu(
    *,
    video_path: Path,
    max_pairs: int,
    posenet_weights_path: Path,
    verbose: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run PoseNet forward on real video pairs; hook summary(512) activations.

    Per upstream/modules.py:67-78 the summary is:
      summary = self.summarizer(vision_out)
    where summarizer = Sequential(Linear(2048->512), ReLU, ResBlock(512)).

    Returns (features[N <= max_pairs, 512] as float32, metadata).

    Per CLAUDE.md "Forbidden device-selection defaults": explicit
    torch.device('cpu') only; MPS forbidden.
    """
    import einops
    from frame_utils import AVVideoDataset
    from modules import PoseNet
    from safetensors.torch import load_file

    device = torch.device("cpu")
    posenet = PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(str(posenet_weights_path), device="cpu"))

    captured: list[torch.Tensor] = []

    def _hook(module, args, output):
        captured.append(output.detach().cpu().clone())

    handle = posenet.summarizer.register_forward_hook(_hook)

    files = ["0.mkv"]
    ds = AVVideoDataset(
        file_names=files,
        data_dir=video_path.parent,
        batch_size=4,
        device=device,
    )
    ds.prepare_data()

    feats_list: list[np.ndarray] = []
    n_pairs = 0
    try:
        with torch.inference_mode():
            for _path, _idx, batch in ds:
                batch = batch.float() if batch.dtype == torch.uint8 else batch.to(torch.float32)
                # batch: (B, seq_len=2, H, W, 3) - convert to (B, S, C, H, W)
                batch = einops.rearrange(batch, "b s h w c -> b s c h w")
                posenet_in = posenet.preprocess_input(batch)
                _ = posenet(posenet_in)
                # Hook captured summary(B, 512) per call
                if captured:
                    summary = captured[-1]
                    feats_list.append(summary.numpy().astype(np.float32))
                    captured.clear()
                n_pairs += batch.shape[0]
                if n_pairs >= max_pairs:
                    break
                if verbose and n_pairs % 8 == 0:
                    print(f"  [top2-f4] processed {n_pairs} pairs")
    finally:
        handle.remove()

    features = np.concatenate(feats_list, axis=0)[:max_pairs]
    if features.shape[0] < 2:
        raise RuntimeError(
            f"Need at least 2 pairs for VQ codebook; got {features.shape[0]}. "
            f"Increase --max-pairs or check video decode."
        )
    metadata = {
        "feature_extraction_path": "PoseNet.summarizer forward hook output (summary(512))",
        "video_path": str(video_path),
        "n_pairs_decoded": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "feature_dtype": "float32",
    }
    return features, metadata


def wrap_score_in_advisory_provenance(
    *, blob_path: Path, archive_sha256: str, metric_name: str, metric_value: float,
) -> dict[str, Any]:
    """Catalog #192 + #317 + #323 canonical [macOS-CPU advisory] wrap (no score claim)."""
    from tac.provenance import (
        audit_score_claim_dict,
        build_provenance_for_macos_cpu_advisory,
        provenance_to_dict,
    )
    prov = build_provenance_for_macos_cpu_advisory(
        archive_sha256=archive_sha256, source_path=str(blob_path),
    )
    row = {
        "metric_name": metric_name,
        "metric_value": metric_value,
        "metric_units": "bytes",
        "score_axis": "[macOS-CPU advisory]",
        "evidence_grade": "macOS-CPU-advisory",
        "hardware_substrate": "darwin_arm64_apple_m5_max_cpu",
        "archive_sha256": archive_sha256,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "score_claim_valid": False,
        "is_contest_score_claim": False,
        "framework_validation_only": True,
        "provenance": provenance_to_dict(prov),
    }
    is_ok, blockers = audit_score_claim_dict(row, expected_axis="[macOS-CPU advisory]")
    row["catalog_323_audit_ok"] = is_ok
    row["catalog_323_audit_blockers"] = list(blockers)
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="TOP-2 LOCAL-CPU advisory smoke: F4 PoseNet summary(512) per-pattern inverter"
    )
    parser.add_argument(
        "--archive", type=Path,
        default=REPO_ROOT / "submissions" / "a1" / "archive.zip",
        help="A1 archive path (canonical: submissions/a1/archive.zip) - for archive_sha256 only",
    )
    parser.add_argument(
        "--video", type=Path,
        default=REPO_ROOT / "upstream" / "videos" / "0.mkv",
        help="Contest video for PoseNet forward (canonical: upstream/videos/0.mkv)",
    )
    parser.add_argument(
        "--posenet-weights", type=Path,
        default=REPO_ROOT / "upstream" / "models" / "posenet.safetensors",
        help="PoseNet weights (canonical: upstream/models/posenet.safetensors)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--k", type=int, default=64,
                        help="VQ codebook size (default 64 per directive: K=64 per 8-subvector PQ-8x8 spirit)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu"],
                        help="Device (CPU only; MPS forbidden per CLAUDE.md non-negotiable)")
    parser.add_argument("--max-pairs", type=int, default=25,
                        help="Smoke gate: max pairs to forward through PoseNet (default 25 = ~1-2 min CPU)")
    parser.add_argument("--km-iters", type=int, default=8, help="k-means iterations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.device != "cpu":
        raise RuntimeError("Only --device cpu supported per CLAUDE.md non-negotiable")

    args.out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # Archive sha256 (for Provenance custody anchor)
    archive_bytes = args.archive.read_bytes()
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()

    if args.verbose:
        print(f"[top2-f4] extracting PoseNet summary(512) features (max_pairs={args.max_pairs})")
    t_extract = time.time()
    features, extract_meta = extract_posenet_summary_features_cpu(
        video_path=args.video,
        max_pairs=args.max_pairs,
        posenet_weights_path=args.posenet_weights,
        verbose=args.verbose,
    )
    extract_elapsed = time.time() - t_extract
    if args.verbose:
        print(f"[top2-f4] features shape={features.shape} extracted in {extract_elapsed:.1f}s")

    n_patterns, feature_dim = features.shape

    # Effective K = min(args.k, n_patterns)
    effective_k = min(args.k, n_patterns)
    if args.verbose:
        print(f"[top2-f4] building VQ-FP4-sparse-Brotli packet K={effective_k} (requested {args.k})")
    t_build = time.time()
    config = A1SpecializedInverterConfig(
        codebook_size=effective_k,
        sparsity=0.50,
        kmeans_iterations=args.km_iters,
        brotli_quality=11,
        seed=args.seed,
    )
    artifact = build_a1_specialized_inverter(
        features=features,
        config=config,
        input_label="posenet_summary_512_activations_on_contest_video_pairs",
        source_sha256=archive_sha256,
    )
    build_elapsed = time.time() - t_build

    inverter_path = args.out / "inverter.bin"
    inverter_path.write_bytes(artifact.compressed_blob)
    inverter_sha256 = hashlib.sha256(artifact.compressed_blob).hexdigest()
    compressed_bytes = len(artifact.compressed_blob)
    raw_bytes = len(artifact.uncompressed_blob)

    residual = features - artifact.reconstructed_features
    pattern_std = float(np.std(features))
    pattern_var = float(np.var(features))
    proxy = {
        "mse_mean": float(np.mean(np.square(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "max_abs_error": float(np.max(np.abs(residual))) if residual.size else 0.0,
        "mse_p95": float(np.percentile(np.mean(residual ** 2, axis=1), 95))
        if residual.size else 0.0,
        "pattern_count": int(features.shape[0]),
    }
    proxy_relative_rmse = proxy["rmse"] / pattern_std if pattern_std > 0 else float("inf")
    proxy_threshold_relative_rmse = 0.5
    cascade_threshold_bytes = 40 * 1024  # 40 KB per directive

    cascade_bytes_pass = compressed_bytes <= cascade_threshold_bytes
    cascade_proxy_pass = proxy_relative_rmse <= proxy_threshold_relative_rmse

    if cascade_bytes_pass and cascade_proxy_pass:
        cascade_verdict = "framework_validated_advisory"
    elif not cascade_bytes_pass and not cascade_proxy_pass:
        cascade_verdict = "framework_falsified_advisory"
    else:
        cascade_verdict = "partial_framework_validated_advisory"

    score_row = wrap_score_in_advisory_provenance(
        blob_path=inverter_path,
        archive_sha256=archive_sha256,
        metric_name="compressed_bytes_per_pattern_inverter",
        metric_value=float(compressed_bytes),
    )

    elapsed = time.time() - t_start

    manifest: dict[str, Any] = {
        "schema_version": "top_2_f4_summary_local_cpu_advisory_smoke_v1_20260518",
        "tool": "tools/build_top_2_f4_summary_local_cpu_advisory_smoke.py",
        "lane_id": "lane_top_3_reclamation_local_cpu_smoke_20260518",
        "routing_directive": ".omx/research/codex_routing_directive_top2_f4_summary_512_per_pattern_inverter_prototype_20260518.md",
        "authority_commit": "248120281",
        "operator_authorization": "PROCEED-ALL-3 LOCAL-CPU 2026-05-18",
        "evidence_grade": "macOS-CPU-advisory",
        "score_axis": "[macOS-CPU advisory]",
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "score_claim_valid": False,
        "axis_label_required_for_promotion": (
            "[contest-CPU] via Linux x86_64 paired anchor per Catalog #192"
        ),
        "device": "cpu",
        "hardware_substrate": "darwin_arm64_apple_m5_max_cpu",
        "args": {
            "archive": str(args.archive),
            "video": str(args.video),
            "posenet_weights": str(args.posenet_weights),
            "out": str(args.out),
            "k": args.k,
            "max_pairs": args.max_pairs,
            "km_iters": args.km_iters,
            "seed": args.seed,
        },
        "archive_sha256_for_provenance_custody": archive_sha256,
        "feature_extraction_meta": extract_meta,
        "feature_extraction_seconds": extract_elapsed,
        "patterns_used": {"n": n_patterns, "d": feature_dim},
        "canonical_helper_delegated": (
            "tac.contest_exploits.a1_specialized_inverter.build_a1_specialized_inverter "
            "(sister Codex landing; ADOPT-CANONICAL)"
        ),
        "vq_fp4_sparse_brotli_config": {
            "codebook_size_requested": args.k,
            "codebook_size_effective": int(config.codebook_size),
            "sparsity": float(config.sparsity),
            "kmeans_iterations": int(config.kmeans_iterations),
            "brotli_quality": int(config.brotli_quality),
            "actual_codebook_zero_fraction": float(artifact.report["actual_codebook_zero_fraction"]),
        },
        "compressed_bytes": compressed_bytes,
        "raw_bytes": raw_bytes,
        "compression_ratio": (raw_bytes / compressed_bytes) if compressed_bytes else 0,
        "inverter_sha256": inverter_sha256,
        "proxy_distortion": proxy,
        "pattern_variance": pattern_var,
        "pattern_std": pattern_std,
        "proxy_relative_rmse": proxy_relative_rmse,
        "cascade_thresholds": {
            "compressed_bytes_threshold": cascade_threshold_bytes,
            "proxy_relative_rmse_threshold": proxy_threshold_relative_rmse,
        },
        "cascade_verdict": cascade_verdict,
        "cascade_compressed_bytes_pass": cascade_bytes_pass,
        "cascade_proxy_pass": cascade_proxy_pass,
        "wall_clock_seconds": elapsed,
        "build_seconds": build_elapsed,
        "score_row": score_row,
        "evidence_tag": "[macOS-CPU advisory]",
        "evidence_summary": (
            f"F4-summary(512) VQ-FP4-sparse-Brotli K={effective_k} compressed_bytes={compressed_bytes} "
            f"[macOS-CPU advisory]; proxy rel_rmse={proxy_relative_rmse:.3f} [macOS-CPU advisory]; "
            f"verdict={cascade_verdict}"
        ),
        "canonical_provenance_per_catalog_323": True,
        "audit_passes_catalog_323": score_row["catalog_323_audit_ok"],
        "sister_helper_report": artifact.report,
    }

    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print(f"[top2-f4] inverter: {inverter_path} ({compressed_bytes} bytes)")
    print(f"[top2-f4] manifest: {manifest_path}")
    print(f"[top2-f4] verdict: {cascade_verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
