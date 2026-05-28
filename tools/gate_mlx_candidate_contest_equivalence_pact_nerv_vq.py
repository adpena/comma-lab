#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Sister Catalog #1265 canonical PASS/FAIL gate for PACT-NeRV-VQ PVQ-grammar candidates.

Parameterized sister of ``tools/gate_mlx_candidate_contest_equivalence.py``
(#1265 PR95/HNeRV-grammar canonical gate; LANDED 69c316ca4 with empirical
anchor ``|S_MLX - S_PyTorch| = 0.000011``, 90x margin over 0.001 threshold)
and ``tools/gate_mlx_candidate_contest_equivalence_pact_nerv_ia3.py`` (PIA3
grammar sister) and ``tools/gate_mlx_candidate_contest_equivalence_pact_nerv_selector_v2.py``
(PSV2 grammar sister) and
``tools/gate_mlx_candidate_contest_equivalence_pact_nerv_selector_v3.py``
(PSV3 grammar sister) and
``tools/gate_mlx_candidate_contest_equivalence_pact_nerv_selector_v4.py``
(PSV4 grammar sister) for the PACT-NeRV-VQ substrate's PVQ archive grammar
(LANDED via the L1 MLX-LOCAL landing 2026-05-28).

The canonical #1265 gate is hardwired for PR95 HNeRV grammar; PIA3 ships
IA3-gamma modulation; PSV2 ships PSV2 grammar; PSV3 ships PSV3 grammar; PSV4
ships PSV4 grammar; PVQ ships PVQ grammar (``PVQ\\x00`` magic + 27-byte header
+ 5 sections per ``src.tac.substrates.pact_nerv_vq.archive``). Per the cascade
doctrine ``fb270e9b6`` L6 gate requirement + the MLX-first doctrine
``4107bbf8d`` per-class bridge calibration scope, each substrate-class
grammar needs its own sister #1265 gate parameterized for that grammar.

This gate empirically establishes MLX <-> PyTorch decoder parity on the
PACT-NeRV-VQ base HNeRV decoder forward path
(``PactNervVqSubstrate.forward`` vs
``PactNervVqSubstrateMLX.__call__``). The decoder output drift is measured
in the canonical sigmoid ``[0, 1]`` output space; threshold 0.001 mirrors
the empirical PR95 anchor + sister IA3/V2/V3/V4 gate convention. NOTE: the
VQ substrate's deep 7-PixelShuffle SIREN stack with sin(freq=30.0) activation
amplifies per-layer ~1e-6 conv drift exponentially per the canonical
drift-vs-depth signature (Catalog #1305) so the gate verdict is an
OBSERVABILITY/disambiguator signal, NOT a contest-promotion gate; promotion
always requires paired contest-CUDA per CLAUDE.md "Submission auth eval -
BOTH CPU AND CUDA".

VQ-specific contract vs IA3/SELECTOR sisters:
- VQ archive ships ``codebook`` + ``indices`` (NOT ``latents`` like sisters).
- The receiver reconstructs per-pair latents via ``z_e[i] = codebook[indices[i]]``
  per van den Oord 1711.00937 §3.1 (nearest-codebook lookup).
- The MLX VQ quantizer's per-pair forward re-quantizes via straight-through
  estimator + commitment loss; the codebook + EMA buffers are pre-loaded from
  the archive (not re-trained at gate-measurement time).
- No ego_poses tensor (VQ has no pose conditioning; the distinguishing
  primitive is discrete-token quantization, not γ-only FiLM).

Scope note - this gate covers Steps 1-2 of the canonical 4-step #1265
closure:

    1. Parse archive (NEW: PVQ grammar via ``parse_archive``)
    2. MLX <-> PyTorch decoder parity (NEW: VQ-VAE base decoder on
       identical state_dict + reconstructed latents; max_abs drift in
       [0,1] sigmoid space)

Steps 3-4 (scorer-axis parity via DistortionNet on candidate-vs-GT frames)
are DEFERRED to operator-routed paid CUDA dispatch per CLAUDE.md "Catalog
#164 + #226 sister discipline" - PACT-NeRV-VQ's score-aware Lagrangian
(including the commitment_weight term) routes through the PyTorch sister +
paid CUDA (not MLX-local).

Gate semantics:
  PASS: max_abs(MLX_decode - PyTorch_decode) < --gate-threshold-decoder-parity (default 0.001)
  FAIL: otherwise

Exit codes:
  0 = PASS (PACT-NeRV-VQ candidate's MLX decoder matches PyTorch within gate)
  1 = FAIL (PACT-NeRV-VQ candidate's MLX decoder drift exceeds gate)
  2 = CLI / measurement error

Per CLAUDE.md "MLX portable-local-substrate authority": output carries
axis_tag="[macOS-MLX research-signal]", score_claim=False, promotable=False,
ready_for_exact_eval_dispatch=False per Catalog #127/#192/#317/#341 +
canonical Provenance per Catalog #323.

Integration pattern for operator-authorize wrappers (PACT-NeRV-VQ + future
PACT-NeRV-VQ derivatives sharing PVQ grammar):

    .venv/bin/python tools/gate_mlx_candidate_contest_equivalence_pact_nerv_vq.py \\
        --archive "$PVQ_ARCHIVE_PATH" \\
        --gate-threshold-decoder-parity 0.001 \\
        --output-json "$REPORT_DIR/pact_nerv_vq_equivalence_gate.json" \\
        || { echo "PACT-NeRV-VQ gate FAIL - refusing paid CUDA dispatch"; exit 1; }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

SCHEMA_VERSION = "mlx_candidate_contest_equivalence_gate_pvq_v1"

# Canonical empirical anchor per #1265 LANDED 69c316ca4 (PR95/HNeRV grammar
# sister; 2026-05-26)
EMPIRICAL_ANCHOR_DRIFT_PR95 = 0.000011  # |S_MLX - S_PyTorch| on PR95 hnerv_muon
DEFAULT_GATE_THRESHOLD = 0.001  # 90x margin over PR95 anchor; matches #1265

PVQ_MAGIC = b"PVQ\x00"


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _read_archive_bytes(archive_path: Path) -> tuple[bytes, str]:
    """Read raw PVQ bytes from either raw 0.bin OR zipped contest packet."""
    raw = archive_path.read_bytes()
    if raw[:4] == PVQ_MAGIC:
        return raw, "raw_pvq_bytes"
    if not zipfile.is_zipfile(archive_path):
        raise ValueError(
            f"archive at {archive_path} is neither raw PVQ (magic {PVQ_MAGIC!r}) "
            "nor a valid ZIP file"
        )
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
        if "0.bin" not in names:
            raise ValueError(
                f"ZIP archive at {archive_path} missing required member '0.bin'; "
                f"members: {names}"
            )
        member_bytes = zf.read("0.bin")
    if member_bytes[:4] != PVQ_MAGIC:
        raise ValueError(
            f"ZIP member '0.bin' magic {member_bytes[:4]!r} does not match "
            f"PVQ magic {PVQ_MAGIC!r}"
        )
    return member_bytes, f"zip_member_0_bin_size_{len(member_bytes)}"


def _build_pytorch_substrate_from_archive(archive_bytes: bytes) -> tuple[Any, Any]:
    """Parse PVQ + build PyTorch ``PactNervVqSubstrate`` with state loaded.

    VQ-specific: ``codebook`` + ``indices`` from archive are loaded onto the
    VQ quantizer; per-pair ``self.latents`` are reconstructed via
    ``z_e[i] = codebook[indices[i]]`` per van den Oord §3.1.
    """
    import torch

    from tac.substrates.pact_nerv_vq.architecture import (
        PactNervVqConfig,
        PactNervVqSubstrate,
    )
    from tac.substrates.pact_nerv_vq.archive import parse_archive

    arc = parse_archive(archive_bytes)
    meta = arc.meta
    cfg = PactNervVqConfig(
        latent_dim=int(arc.codebook.shape[1]),
        embed_dim=int(meta["embed_dim"]),
        initial_grid_h=int(meta["initial_grid_h"]),
        initial_grid_w=int(meta["initial_grid_w"]),
        decoder_channels=tuple(int(c) for c in meta["decoder_channels"]),
        sin_frequency=float(meta["sin_frequency"]),
        num_upsample_blocks=int(meta["num_upsample_blocks"]),
        codebook_size=int(arc.codebook.shape[0]),
        codebook_decay=float(meta.get("codebook_decay", 0.99)),
        commitment_weight=float(meta.get("commitment_weight", 0.25)),
        num_pairs=int(arc.indices.shape[0]),
        output_height=int(meta["output_height"]),
        output_width=int(meta["output_width"]),
    )
    model = PactNervVqSubstrate(cfg).eval()
    load_result = model.load_state_dict(arc.decoder_state_dict, strict=False)
    # latents reconstructed from codebook[indices]; codebook + EMA buffers
    # come from the archive; the receiver does NOT need ema_cluster_size /
    # ema_w at inference time (frozen codebook), but we accept them as
    # missing-OK to mirror the canonical pattern.
    allowed_missing = {
        "latents",
        "quantizer.codebook",
        "quantizer.ema_cluster_size",
        "quantizer.ema_w",
    }
    missing_keys = set(load_result.missing_keys) - allowed_missing
    if missing_keys:
        raise RuntimeError(
            f"PACT-NeRV-VQ gate: PyTorch missing_keys={sorted(missing_keys)} "
            f"(allowed_missing={sorted(allowed_missing)})"
        )
    if set(load_result.unexpected_keys):
        raise RuntimeError(
            f"PACT-NeRV-VQ gate: PyTorch unexpected_keys ="
            f"{sorted(load_result.unexpected_keys)}"
        )
    # Load codebook from archive onto the VQ quantizer (frozen for inference).
    codebook_t = arc.codebook.to(device="cpu", dtype=model.quantizer.codebook.dtype)
    indices_t = arc.indices.to(device="cpu", dtype=torch.long)
    with torch.no_grad():
        model.quantizer.codebook.copy_(codebook_t)
        # Reconstruct per-pair latents: z_e[i] = codebook[indices[i]]
        reconstructed_latents = codebook_t[indices_t]
        model.latents.copy_(
            reconstructed_latents.to(dtype=model.latents.dtype)
        )
    return model, arc


def _build_mlx_renderer_from_archive(archive_bytes: bytes) -> Any:
    """Parse PVQ + build MLX ``PactNervVqSubstrateMLX`` with state loaded.

    VQ-specific: codebook loaded onto MLX quantizer's private ``_codebook``
    buffer; latents reconstructed from codebook[indices] (same as PyTorch
    sister); MLX-specific weight-layout transpose via canonical NHWC HWIO.
    """
    import mlx.core as mx
    import numpy as np
    import torch

    from tac.substrates.pact_nerv_vq.architecture import PactNervVqConfig
    from tac.substrates.pact_nerv_vq.archive import parse_archive
    from tac.substrates.pact_nerv_vq.mlx_renderer import (
        PactNervVqSubstrateMLX,
    )

    arc = parse_archive(archive_bytes)
    meta = arc.meta
    cfg = PactNervVqConfig(
        latent_dim=int(arc.codebook.shape[1]),
        embed_dim=int(meta["embed_dim"]),
        initial_grid_h=int(meta["initial_grid_h"]),
        initial_grid_w=int(meta["initial_grid_w"]),
        decoder_channels=tuple(int(c) for c in meta["decoder_channels"]),
        sin_frequency=float(meta["sin_frequency"]),
        num_upsample_blocks=int(meta["num_upsample_blocks"]),
        codebook_size=int(arc.codebook.shape[0]),
        codebook_decay=float(meta.get("codebook_decay", 0.99)),
        commitment_weight=float(meta.get("commitment_weight", 0.25)),
        num_pairs=int(arc.indices.shape[0]),
        output_height=int(meta["output_height"]),
        output_width=int(meta["output_width"]),
    )
    renderer = PactNervVqSubstrateMLX(cfg)

    def _tensor_to_numpy_float32(t: Any) -> np.ndarray:
        arr = np.asarray(
            t.detach().cpu().numpy() if hasattr(t, "detach") else t
        )
        return arr.astype(np.float32, copy=False)

    # PyTorch OIHW -> MLX HWIO transpose for Conv2d weights.
    def _pyt_to_mlx_layout(name: str, t: Any) -> Any:
        arr = _tensor_to_numpy_float32(t)
        if name.endswith(".weight") and arr.ndim == 4:
            return mx.array(np.transpose(arr, (0, 2, 3, 1)).astype(np.float32))
        return mx.array(arr)

    # Walk decoder_state_dict keys + assign onto MLX renderer (mirror layout).
    for name, t in arc.decoder_state_dict.items():
        parts = name.split(".")
        obj: Any = renderer
        for part in parts[:-1]:
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        leaf = parts[-1]
        value = _pyt_to_mlx_layout(name, t)
        if hasattr(obj, "update"):
            try:
                obj.update({leaf: value})
                continue
            except Exception:
                pass
        setattr(obj, leaf, value)

    # Load codebook onto MLX quantizer's private buffer + reconstruct latents.
    codebook_np = _tensor_to_numpy_float32(arc.codebook)
    indices_np = arc.indices.to(dtype=torch.int64).cpu().numpy().astype(np.int64)
    renderer.quantizer._codebook = mx.array(codebook_np)
    # Reconstruct per-pair latents: z_e[i] = codebook[indices[i]]
    reconstructed = codebook_np[indices_np]
    renderer.latents = mx.array(reconstructed.astype(np.float32))
    return renderer


def _render_pair_batch_pytorch(model: Any, pair_indices: list[int]) -> Any:
    import numpy as np
    import torch

    idx = torch.tensor(pair_indices, dtype=torch.long)
    with torch.no_grad():
        rgb_0, rgb_1 = model(idx)
    out = torch.stack([rgb_0, rgb_1], dim=1)  # (N, 2, 3, H, W) in [0, 1]
    return out.detach().cpu().numpy().astype(np.float32)


def _render_pair_batch_mlx(renderer: Any, pair_indices: list[int]) -> Any:
    """Render N pairs via MLX; normalize MLX [0,255] -> [0,1] sigmoid space."""
    import mlx.core as mx
    import numpy as np

    idx = mx.array(np.asarray(pair_indices, dtype=np.int32))
    out_255_b2chw = np.asarray(renderer(idx), dtype=np.float32)  # (N, 2, 3, H, W) in [0, 255]
    return out_255_b2chw / 255.0


def measure_pact_nerv_vq_decoder_parity(
    archive_path: Path,
    n_pairs: int = 100,
) -> dict[str, Any]:
    """Render N pairs via both backends from same archive; measure drift in [0, 1]."""
    import numpy as np

    archive_bytes, archive_source = _read_archive_bytes(archive_path)
    archive_bytes_sha = _hash_bytes(archive_bytes)
    archive_bytes_size = len(archive_bytes)

    t0 = time.perf_counter()
    pytorch_model, parsed_arc = _build_pytorch_substrate_from_archive(archive_bytes)
    t1 = time.perf_counter()
    mlx_renderer = _build_mlx_renderer_from_archive(archive_bytes)
    t2 = time.perf_counter()

    available_pairs = int(parsed_arc.indices.shape[0])
    effective_n_pairs = min(n_pairs, available_pairs)
    if effective_n_pairs <= 0:
        raise ValueError(
            f"PVQ archive has 0 pairs (num_pairs={available_pairs}); "
            "cannot measure parity"
        )
    pair_indices = list(range(effective_n_pairs))

    t3 = time.perf_counter()
    pytorch_frames = _render_pair_batch_pytorch(pytorch_model, pair_indices)
    t4 = time.perf_counter()
    mlx_frames = _render_pair_batch_mlx(mlx_renderer, pair_indices)
    t5 = time.perf_counter()

    if pytorch_frames.shape != mlx_frames.shape:
        raise RuntimeError(
            f"PyTorch shape {pytorch_frames.shape} != MLX {mlx_frames.shape}"
        )

    drift = np.abs(pytorch_frames - mlx_frames)
    max_abs_drift = float(drift.max())
    mean_abs_drift = float(drift.mean())
    per_pair_max_drift = (
        drift.reshape(effective_n_pairs, -1).max(axis=1).astype(float)
    )

    return {
        "archive_path": str(archive_path),
        "archive_bytes_sha256": archive_bytes_sha,
        "archive_bytes_size": archive_bytes_size,
        "archive_source": archive_source,
        "n_pairs_requested": n_pairs,
        "n_pairs_available": available_pairs,
        "n_pairs_measured": effective_n_pairs,
        "frame_shape": list(pytorch_frames.shape),
        "max_abs_drift": max_abs_drift,
        "mean_abs_drift": mean_abs_drift,
        "per_pair_max_drift_min": float(per_pair_max_drift.min()),
        "per_pair_max_drift_max": float(per_pair_max_drift.max()),
        "per_pair_max_drift_mean": float(per_pair_max_drift.mean()),
        "decoder_output_space": "sigmoid_0_to_1",
        "pytorch_build_seconds": t1 - t0,
        "mlx_build_seconds": t2 - t1,
        "pytorch_render_seconds": t4 - t3,
        "mlx_render_seconds": t5 - t4,
        "codebook_size": int(parsed_arc.codebook.shape[0]),
        "latent_dim": int(parsed_arc.codebook.shape[1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="Path to candidate's PVQ archive (raw 0.bin OR zipped contest packet).",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=100,
        help="Number of pairs to render on both decoders (default=100; bounded by num_pairs).",
    )
    parser.add_argument(
        "--gate-threshold-decoder-parity",
        type=float,
        default=DEFAULT_GATE_THRESHOLD,
        help=(
            f"Gate threshold for max_abs(MLX_decode - PyTorch_decode) in [0,1] "
            f"sigmoid space. Default {DEFAULT_GATE_THRESHOLD} mirrors the PR95 "
            f"#1265 gate threshold (90x margin over PR95 empirical anchor 0.000011)."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Output JSON path for gate verdict.",
    )
    parser.add_argument(
        "--candidate-label",
        type=str,
        default="anonymous_pact_nerv_vq_mlx_candidate",
        help="Label for canonical Provenance + verdict row.",
    )
    args = parser.parse_args()

    if not args.archive.is_file():
        print(
            f"[gate-pvq] ERROR: --archive does not exist: {args.archive}",
            file=sys.stderr,
        )
        return 2
    if args.gate_threshold_decoder_parity <= 0:
        print(
            "[gate-pvq] ERROR: --gate-threshold-decoder-parity must be > 0",
            file=sys.stderr,
        )
        return 2
    if args.n_pairs <= 0:
        print("[gate-pvq] ERROR: --n-pairs must be > 0", file=sys.stderr)
        return 2

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    print("[gate-pvq] Measuring PVQ MLX<->PyTorch decoder parity")
    print(f"[gate-pvq]   archive: {args.archive}")
    print(
        f"[gate-pvq]   threshold: max_abs(MLX-PyTorch) < {args.gate_threshold_decoder_parity}"
    )
    print(f"[gate-pvq]   n_pairs:  {args.n_pairs}")

    try:
        measurement = measure_pact_nerv_vq_decoder_parity(
            args.archive, n_pairs=args.n_pairs
        )
    except Exception as exc:
        print(f"[gate-pvq] ERROR during measurement: {exc}", file=sys.stderr)
        return 2

    actual_drift = float(measurement["max_abs_drift"])
    if actual_drift < args.gate_threshold_decoder_parity:
        verdict = "PASS"
        exit_code = 0
    else:
        verdict = "FAIL"
        exit_code = 1

    from tac.provenance.builders import build_provenance_for_predicted
    from tac.provenance.validator import provenance_to_dict

    inputs_sha = _hash_file(args.archive)
    prov = build_provenance_for_predicted(
        model_id=(
            f"mlx_candidate_contest_equivalence_gate_pvq:{args.candidate_label}"
        ),
        inputs_sha256=inputs_sha,
        measurement_axis="[macOS-MLX research-signal]",
        hardware_substrate="darwin_arm64_apple_silicon",
    )

    gate_verdict = {
        "schema_version": SCHEMA_VERSION,
        "verdict": verdict,
        "max_abs_drift_decoder_parity": actual_drift,
        "gate_threshold_decoder_parity": args.gate_threshold_decoder_parity,
        "margin_below_threshold": (
            args.gate_threshold_decoder_parity - actual_drift
        ),
        "ratio_actual_vs_pr95_empirical_anchor": (
            actual_drift / EMPIRICAL_ANCHOR_DRIFT_PR95
            if EMPIRICAL_ANCHOR_DRIFT_PR95 > 0
            else None
        ),
        "candidate_label": args.candidate_label,
        "candidate_grammar": "PVQ",
        "candidate_substrate_id": "pact_nerv_vq",
        "candidate_substrate_class": (
            "discrete_tokens_van_den_oord_vqvae_codebook_per_pair_index"
        ),
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "axis_tag": "[macOS-MLX research-signal]",
        "evidence_grade": "macOS-MLX-research-signal",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotable": False,
        "blockers": [
            "macos_mlx_research_signal_not_contest_authority",
            "requires_paired_contest_cpu_plus_cuda_for_score_claim",
            (
                f"gate_threshold_is_decoder_parity_drift_bound_in_sigmoid_0_to_1_space_"
                f"not_contest_score_claim_threshold_{args.gate_threshold_decoder_parity}"
            ),
            (
                "pact_nerv_vq_scorer_axis_equivalence_steps_3_4_deferred_to_paid_cuda_dispatch_"
                "per_catalog_164_226_score_aware_lagrangian_routes_through_pytorch_sister"
            ),
            (
                "vq_substrate_deep_siren_pixelshuffle_stack_amplifies_per_layer_drift_per_"
                "catalog_1305_drift_vs_depth_anchor_drift_outside_threshold_does_NOT_imply_bridge_bug"
            ),
        ],
        "provenance": provenance_to_dict(prov),
        "measurement": measurement,
        "canonical_anchor": {
            "pr95_empirical_anchor_drift": EMPIRICAL_ANCHOR_DRIFT_PR95,
            "pvq_archive_grammar_source": (
                "src.tac.substrates.pact_nerv_vq.archive::parse_archive (PVQ)"
            ),
            "pact_nerv_vq_landing_memo": (
                ".omx/research/pact_nerv_vq_l1_long_run_mlx_landed_20260528.md"
            ),
            "pr95_canonical_gate_commit": "69c316ca4",
            "v4_sister_gate_path": (
                "tools/gate_mlx_candidate_contest_equivalence_pact_nerv_selector_v4.py"
            ),
            "drift_vs_depth_anchor_catalog": 1305,
            "vq_vae_literature_anchor": "van_den_Oord_Vinyals_Kavukcuoglu_2017_arxiv_1711.00937",
        },
        "operator_routable_per_verdict": (
            (
                "PROCEED: PACT-NeRV-VQ candidate's MLX decoder matches PyTorch decoder "
                "within gate threshold; PVQ unlocked for per-substrate-class bridge "
                "calibration per MLX-first doctrine + paired CUDA + CPU dispatch "
                "authorization per CLAUDE.md 'Submission auth eval - BOTH CPU AND CUDA' "
                "non-negotiable + Catalog #325 14-day per-substrate symposium window"
            )
            if verdict == "PASS"
            else (
                "OBSERVABILITY-ONLY: PACT-NeRV-VQ candidate's MLX decoder drift exceeds "
                "gate threshold; per Catalog #1305 drift-vs-depth this is EXPECTED for "
                "the 7-PixelShuffle SIREN substrate. The gate is a research-signal "
                "disambiguator NOT a contest-promotion gate; the operator MAY still "
                "dispatch paired CPU+CUDA per CLAUDE.md 'Submission auth eval - BOTH CPU "
                "AND CUDA' for the canonical contest score - the MLX-side drift is "
                "irrelevant on the paid CUDA path because the PyTorch sister IS the "
                "contest substrate. The operator MAY also audit MLX architecture parity "
                "via tac.substrates.pact_nerv_vq.mlx_renderer."
            )
        ),
        "scope_note": (
            "Covers Steps 1-2 of canonical 4-step #1265 closure (archive grammar "
            "parse + MLX<->PyTorch decoder parity). Steps 3-4 (scorer-axis equivalence "
            "via DistortionNet on candidate-vs-GT frames) DEFERRED to paid CUDA dispatch "
            "per Catalog #164 + #226 sister discipline. Per Catalog #1305 drift-vs-depth "
            "the VQ substrate's deep SIREN+PixelShuffle stack architecturally diverges "
            "from PyTorch numerics on Apple Silicon by ~1e-6 per conv-layer; the gate "
            "verdict is OBSERVABILITY-ONLY (not contest-promotion-binding)."
        ),
    }
    args.output_json.write_text(json.dumps(gate_verdict, indent=2) + "\n")

    print()
    print("=== PVQ MLX CANDIDATE CONTEST-EQUIVALENCE GATE ===")
    print(f"  candidate: {args.candidate_label}")
    print(f"  archive source: {measurement['archive_source']}")
    print(f"  archive sha256: {measurement['archive_bytes_sha256'][:16]}...")
    print(f"  archive bytes: {measurement['archive_bytes_size']:,}")
    print(
        f"  pairs measured: {measurement['n_pairs_measured']} / {measurement['n_pairs_available']}"
    )
    print(f"  frame shape: {measurement['frame_shape']}")
    print(f"  codebook: {measurement['codebook_size']} x {measurement['latent_dim']}")
    print(f"  decoder output space: {measurement['decoder_output_space']}")
    print(f"  max_abs drift: {actual_drift:.6e}")
    print(f"  mean_abs drift: {measurement['mean_abs_drift']:.6e}")
    print(
        f"  per-pair max drift mean: {measurement['per_pair_max_drift_mean']:.6e}"
    )
    print(f"  threshold:    {args.gate_threshold_decoder_parity:.6f}")
    print(
        f"  margin:       {args.gate_threshold_decoder_parity - actual_drift:.6f}"
    )
    if EMPIRICAL_ANCHOR_DRIFT_PR95 > 0:
        print(
            f"  ratio vs PR95 empirical anchor (0.000011): "
            f"{actual_drift / EMPIRICAL_ANCHOR_DRIFT_PR95:.2f}x"
        )
    print(
        f"  build (PyTorch / MLX): {measurement['pytorch_build_seconds']:.2f}s / "
        f"{measurement['mlx_build_seconds']:.2f}s"
    )
    print(
        f"  render (PyTorch / MLX): {measurement['pytorch_render_seconds']:.2f}s / "
        f"{measurement['mlx_render_seconds']:.2f}s"
    )
    print(f"  VERDICT: {verdict}")
    print(f"  exit code: {exit_code}")
    print(f"  output: {args.output_json}")
    return exit_code


__all__ = [
    "DEFAULT_GATE_THRESHOLD",
    "EMPIRICAL_ANCHOR_DRIFT_PR95",
    "PVQ_MAGIC",
    "SCHEMA_VERSION",
    "main",
    "measure_pact_nerv_vq_decoder_parity",
]


if __name__ == "__main__":
    sys.exit(main())
