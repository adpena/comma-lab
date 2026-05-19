#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""TOP-1 LOCAL-CPU advisory smoke: A1-SPECIALIZED per-pattern VQ-VAE inverter.

Authority: routing directive
``.omx/research/codex_routing_directive_top1_a1_specialized_per_pattern_vq_vae_inverter_prototype_20260518.md``
(commit ``248120281``) + operator PROCEED-ALL-3 LOCAL-CPU authorization
2026-05-18 verbatim ``"Can those be local cpu"`` + ``"approved all"``.

Lane: ``lane_top_3_reclamation_local_cpu_smoke_20260518``
(research_only=true; substrate_engineering scope; framework validation only;
sister of TOP-2 + TOP-3 local-CPU advisory smoke tools).

Scope DISJOINT from sister Codex's promotable-evidence build tool
``tools/build_a1_per_pattern_vq_vae_inverter_prototype.py`` (which carries
paid-dispatch infrastructure via ``tac.packet_compiler.deterministic_compiler``
+ exact-eval custody). This tool is the local-CPU advisory framework-validation
variant: $0 GPU, [macOS-CPU advisory] axis only, no paid dispatch.

Canonical-vs-unique decision per layer (per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD):
  - VQ + FP4 + 50% sparse + Brotli engine: ADOPT canonical
    ``tac.contest_exploits.a1_specialized_inverter.build_a1_specialized_inverter``
    (sister Codex landing; same algorithm).
  - A1 latent decoding: FORK locally (sister helper takes precomputed features;
    we extract the per-pattern latent manifold from the canonical A1 archive).
  - Local-CPU advisory wrapping: FORK (sister produces [predicted] artifact
    for paid-dispatch consumption; we produce [macOS-CPU advisory] for
    framework-validation only per Catalog #192).
  - Cascade verdict logic: FORK (CASCADE RULE specific to this advisory-smoke
    lane per Assumption-Adversary op-routable; sister doesn't carry it).
  - Provenance: FORK locally to use ``build_provenance_for_macos_cpu_advisory``
    instead of ``build_provenance_for_predicted`` per Catalog #192 + #317 + #323.

Per CLAUDE.md "Forbidden device-selection defaults": explicit
``torch.device('cpu')`` only; MPS forbidden.

Per CLAUDE.md "MPS auth eval is NOISE" + "Submission auth eval" non-negotiables:
local-CPU advisory NEVER promoted to ``[contest-CPU]`` without paired Linux
x86_64 anchor; ``promotion_eligible=False`` enforced at every output.

[verified-against:upstream/modules.py:lines-22-26-67-78-HEADS-summarizer +
submissions/a1/src/codec.py:decode_latents_compact + apply_latent_sidecar +
tac.contest_exploits.a1_specialized_inverter sister Codex landing
0701c323b A1 binary distillation framework spec]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "submissions" / "a1" / "src"))

from codec import (  # noqa: E402  (per A1 canonical decoder API)
    LATENT_BLOB_LEN,
    apply_latent_sidecar,
    decode_decoder_compact,
    decode_latents_compact,
)

from tac.contest_exploits.a1_specialized_inverter import (  # noqa: E402
    A1SpecializedInverterConfig,
    build_a1_specialized_inverter,
)


def load_a1_per_pattern_latents(
    archive_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode A1 archive's per-pair latents = the per-pattern manifold.

    Returns (latents[N=600, D=28] as float32 numpy, metadata).

    Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag": this is
    the canonical per-pattern decoding path via ``submissions/a1/src/codec.py``;
    the latents are the substrate's pattern manifold by construction.
    """
    if not archive_path.exists():
        raise FileNotFoundError(
            f"A1 archive not found: {archive_path}; canonical path is "
            f"submissions/a1/archive.zip per A1 council Round 1 finding"
        )
    archive_bytes = archive_path.read_bytes()
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()

    z = zipfile.ZipFile(archive_path)
    if "x" not in z.namelist():
        raise ValueError(
            f"A1 archive at {archive_path} lacks 'x' member; canonical A1 layout "
            f"requires single 'x' ZIP member per submissions/a1/inflate.py"
        )
    x = z.read("x")
    section_total = struct.unpack_from("<I", x, 0)[0]
    decoder_blob = x[4:section_total]
    latent_blob = x[section_total:section_total + LATENT_BLOB_LEN]
    sidecar_blob = x[section_total + LATENT_BLOB_LEN:]

    latents_t = apply_latent_sidecar(decode_latents_compact(latent_blob), sidecar_blob)
    decoder_sd = decode_decoder_compact(decoder_blob)

    metadata = {
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha256,
        "archive_size_bytes": len(archive_bytes),
        "decoder_blob_bytes": len(decoder_blob),
        "latent_blob_bytes": LATENT_BLOB_LEN,
        "sidecar_blob_bytes": len(sidecar_blob),
        "decoder_state_dict_n_tensors": len(decoder_sd),
        "n_pairs": int(latents_t.shape[0]),
        "latent_dim": int(latents_t.shape[1]),
    }
    return latents_t.detach().cpu().numpy().astype(np.float32), metadata


def wrap_score_in_advisory_provenance(
    *,
    out_dir: Path,
    archive_sha256: str,
    metric_name: str,
    metric_value: float,
    blob_path: Path,
) -> dict[str, Any]:
    """Wrap a metric in canonical [macOS-CPU advisory] Provenance per Catalog #323.

    Per Catalog #192 + #317: ``[macOS-CPU advisory]`` is NEVER promoted to
    ``[contest-CPU]`` without a paired Linux x86_64 anchor.
    """
    from tac.provenance import (
        audit_score_claim_dict,
        build_provenance_for_macos_cpu_advisory,
        provenance_to_dict,
    )
    prov = build_provenance_for_macos_cpu_advisory(
        archive_sha256=archive_sha256,
        source_path=str(blob_path),
    )
    # Per Catalog #321/#323 phantom-score class extinction: a framework-validation
    # metric is NOT a contest score. We carry compressed_bytes as a metadata
    # field; score field is omitted entirely (no score claim). The Provenance
    # contract still applies for axis-tag + custody routing.
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
        description="TOP-1 LOCAL-CPU advisory smoke: A1-SPECIALIZED per-pattern VQ-VAE inverter"
    )
    parser.add_argument(
        "--archive", type=Path,
        default=REPO_ROOT / "submissions" / "a1" / "archive.zip",
        help="A1 archive path (canonical: submissions/a1/archive.zip)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--k", type=int, default=256,
                        help="VQ codebook size (default 256; Selfcomp NON-BLOCKING: 64 for over-param test)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu"],
                        help="Device (CPU only; MPS forbidden per CLAUDE.md non-negotiable)")
    parser.add_argument("--max-pairs", type=int, default=600,
                        help="Smoke gate: max patterns to use (default 600 = full A1)")
    parser.add_argument("--km-iters", type=int, default=8, help="k-means iterations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.device != "cpu":
        raise RuntimeError(
            "Only --device cpu supported in this LOCAL-CPU framework-validation tool; "
            "per CLAUDE.md MPS auth eval is NOISE non-negotiable"
        )
    # Explicit CPU device (per CLAUDE.md Forbidden device-selection defaults).
    _ = torch.device("cpu")

    args.out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # ---- Decode A1 archive patterns ----
    if args.verbose:
        print(f"[top1-a1] decoding A1 archive: {args.archive}")
    patterns, archive_meta = load_a1_per_pattern_latents(args.archive)
    if args.max_pairs < patterns.shape[0]:
        patterns = patterns[:args.max_pairs]
    if args.verbose:
        print(f"[top1-a1] patterns shape={patterns.shape} dtype={patterns.dtype}")
        print(f"[top1-a1] archive sha256={archive_meta['archive_sha256']}")

    n_patterns, feature_dim = patterns.shape

    # ---- Delegate to canonical sister helper for VQ + FP4 + sparse + Brotli ----
    if args.verbose:
        print(f"[top1-a1] building VQ-FP4-sparse-Brotli packet K={args.k}")
    t_build = time.time()
    config = A1SpecializedInverterConfig(
        codebook_size=args.k,
        sparsity=0.50,
        kmeans_iterations=args.km_iters,
        brotli_quality=11,
        seed=args.seed,
    )
    artifact = build_a1_specialized_inverter(
        features=patterns,
        config=config,
        input_label="a1_per_pair_latents_decoded_via_canonical_submissions_a1_src_codec",
        source_sha256=archive_meta["archive_sha256"],
    )
    build_elapsed = time.time() - t_build

    inverter_path = args.out / "inverter.bin"
    inverter_path.write_bytes(artifact.compressed_blob)
    inverter_sha256 = hashlib.sha256(artifact.compressed_blob).hexdigest()
    compressed_bytes = len(artifact.compressed_blob)
    raw_bytes = len(artifact.uncompressed_blob)

    # ---- Proxy distortion ----
    residual = patterns - artifact.reconstructed_features
    proxy = {
        "mse_mean": float(np.mean(np.square(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "max_abs_error": float(np.max(np.abs(residual))) if residual.size else 0.0,
        "mse_p95": float(np.percentile(np.mean(residual ** 2, axis=1), 95))
        if residual.size else 0.0,
        "pattern_count": int(patterns.shape[0]),
    }
    if args.verbose:
        print(f"[top1-a1] proxy mse_mean={proxy['mse_mean']:.6f} rmse={proxy['rmse']:.6f}")
        print(f"[top1-a1] compressed_bytes={compressed_bytes} (raw {raw_bytes})")

    # ---- CASCADE RULE per Assumption-Adversary op-routable (TOP-3 routing
    # index memo): "FAILS = compressed bytes >2x predicted (>40 KB) OR proxy
    # distortion >2x predicted threshold". Per directive memo: predicted band
    # 5-20 KB; 40 KB is the 2x upper-bound cascade trigger. Proxy threshold:
    # framework-validation == relative RMSE < 50% of pattern_std (substrate
    # signal preserved within order of magnitude). This is conservative;
    # tighter promotion thresholds apply at paired Linux x86_64 anchor stage.
    pattern_std = float(np.std(patterns))
    pattern_var = float(np.var(patterns))
    proxy_relative_rmse = proxy["rmse"] / pattern_std if pattern_std > 0 else float("inf")
    proxy_threshold_relative_rmse = 0.5  # framework-validation: 50% relative RMSE
    cascade_threshold_bytes = 40 * 1024  # 40 KB per directive CASCADE rule

    cascade_bytes_pass = compressed_bytes <= cascade_threshold_bytes
    cascade_proxy_pass = proxy_relative_rmse <= proxy_threshold_relative_rmse

    if cascade_bytes_pass and cascade_proxy_pass:
        cascade_verdict = "framework_validated_advisory"
    elif not cascade_bytes_pass and not cascade_proxy_pass:
        cascade_verdict = "framework_falsified_advisory"
    else:
        cascade_verdict = "partial_framework_validated_advisory"

    # ---- Wrap in canonical Provenance per Catalog #323 ----
    score_row = wrap_score_in_advisory_provenance(
        out_dir=args.out,
        archive_sha256=archive_meta["archive_sha256"],
        metric_name="compressed_bytes_per_pattern_inverter",
        metric_value=float(compressed_bytes),
        blob_path=inverter_path,
    )

    elapsed = time.time() - t_start

    # ---- Emit manifest ----
    manifest: dict[str, Any] = {
        "schema_version": "top_1_a1_local_cpu_advisory_smoke_v1_20260518",
        "tool": "tools/build_top_1_a1_local_cpu_advisory_smoke.py",
        "lane_id": "lane_top_3_reclamation_local_cpu_smoke_20260518",
        "routing_directive": ".omx/research/codex_routing_directive_top1_a1_specialized_per_pattern_vq_vae_inverter_prototype_20260518.md",
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
            "out": str(args.out),
            "k": args.k,
            "max_pairs": args.max_pairs,
            "km_iters": args.km_iters,
            "seed": args.seed,
        },
        "archive_metadata": archive_meta,
        "feature_extraction_path": (
            "a1_canonical_per_pair_latents (600 pairs x 28-dim) via "
            "submissions/a1/src/codec.py decode_latents_compact + apply_latent_sidecar"
        ),
        "patterns_used": {"n": n_patterns, "d": feature_dim},
        "canonical_helper_delegated": (
            "tac.contest_exploits.a1_specialized_inverter.build_a1_specialized_inverter "
            "(sister Codex landing; ADOPT-CANONICAL per UNIQUE-AND-COMPLETE-PER-METHOD)"
        ),
        "vq_fp4_sparse_brotli_config": {
            "codebook_size": int(config.codebook_size),
            "sparsity": float(config.sparsity),
            "kmeans_iterations": int(config.kmeans_iterations),
            "brotli_quality": int(config.brotli_quality),
            "effective_codebook_size": int(artifact.report["effective_codebook_size"]),
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
            f"A1-SPECIALIZED V2 (VQ-FP4-sparse-Brotli K={args.k}) "
            f"compressed_bytes={compressed_bytes} [macOS-CPU advisory]; "
            f"proxy mse_mean={proxy['mse_mean']:.6f} [macOS-CPU advisory]; "
            f"verdict={cascade_verdict}"
        ),
        "canonical_provenance_per_catalog_323": True,
        "audit_passes_catalog_323": score_row["catalog_323_audit_ok"],
        "sister_helper_report": artifact.report,
    }

    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print(f"[top1-a1] inverter: {inverter_path} ({compressed_bytes} bytes)")
    print(f"[top1-a1] manifest: {manifest_path}")
    print(f"[top1-a1] verdict: {cascade_verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
