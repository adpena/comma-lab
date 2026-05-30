#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Sister Catalog #1265 canonical PASS/FAIL gate for Z6PCWM1-grammar candidates.

Parameterized sister of ``tools/gate_mlx_candidate_contest_equivalence.py``
(#1265 PR95/HNeRV-grammar canonical gate; LANDED 69c316ca4 with empirical
anchor |S_MLX − S_PyTorch| = 0.000011, 90× margin over 0.001 threshold) for
the D=Z6 substrate's Z6PCWM1 archive grammar (LANDED 8833b9db5).

The canonical #1265 gate is hardwired for PR95 HNeRV archive grammar
(``parse_pr95_public_archive_zip``); D=Z6 ships the NEW Z6PCWM1 grammar
(``Z6WM`` magic + 39-byte header + 7 sections per
``src.tac.substrates.time_traveler_l5_z6.archive``). Per the cascade doctrine
``fb270e9b6`` L6 gate requirement + the MLX-first doctrine ``4107bbf8d``
per-class bridge calibration scope, each substrate-class grammar needs its
own sister #1265 gate parameterized for that grammar.

This gate empirically establishes MLX↔PyTorch decoder parity on the Z6
autoregressive ``reconstruct_pair`` recurrence (encoder + FiLM predictor +
decoder via the canonical PyTorch ``Z6PredictiveCodingSubstrate`` vs the
canonical MLX ``Z6PredictiveCodingMLXRenderer``). The decoder output drift is
measured in the canonical sigmoid [0, 1] output space; threshold 0.001 mirrors
the empirical PR95 anchor.

Scope note — this gate covers Steps 1-2 of the canonical 4-step #1265 closure:

    1. Parse archive (NEW: Z6PCWM1 grammar via ``parse_archive``)
    2. MLX↔PyTorch decoder parity (NEW: ``reconstruct_pair`` autoregressive
       recurrence on identical state_dict; max_abs drift in [0,1] sigmoid space)

Steps 3-4 (scorer-axis parity via DistortionNet on candidate-vs-GT frames) are
DEFERRED to operator-routed paid CUDA dispatch per Yousfi's L1 promotion
symposium dissent: Z6's score-aware Lagrangian routes through PyTorch sister +
paid CUDA (not MLX-local) per Catalog #164 + #226 sister discipline. The
canonical PR95 #1265 gate includes Steps 3-4 because PR95 HNeRV has scorer
parity already validated on Apple Silicon; Z6's class-shift architecture
(predictive-coding + FiLM ego-motion conditioning) gets that validation
empirically on paid CUDA, not MLX-local.

Gate semantics:
  PASS: max_abs(MLX_decode − PyTorch_decode) < --gate-threshold-decoder-parity (default 0.001)
  FAIL: otherwise

The default 0.001 threshold is the same 90× margin the canonical PR95 gate uses
over the empirical anchor 0.000011 (per MLX-first doctrine 4107bbf8d) and is
operationally meaningful in the [0, 1] sigmoid output space the Z6 decoder
produces. Per the cascade doctrine fb270e9b6 L6 gate, PASS unlocks paid-CUDA
dispatch eligibility for D=Z6.

Exit codes:
  0 = PASS (Z6 candidate's MLX decoder matches PyTorch decoder within gate)
  1 = FAIL (Z6 candidate's MLX decoder drift exceeds gate; do NOT dispatch)
  2 = CLI / measurement error

Per CLAUDE.md "MLX portable-local-substrate authority": output carries
axis_tag="[macOS-MLX research-signal]", score_claim=False, promotable=False,
ready_for_exact_eval_dispatch=False per Catalog #127/#192/#317/#341 +
canonical Provenance per Catalog #323.

Integration pattern for operator-authorize wrappers (D=Z6 + future Z6 derivatives):

    # In scripts/operator_authorize_substrate_time_traveler_l5_z6_modal_t4_dispatch.sh:
    .venv/bin/python tools/gate_mlx_candidate_contest_equivalence_z6.py \\
        --archive "$Z6_ARCHIVE_PATH" \\
        --gate-threshold-decoder-parity 0.001 \\
        --output-json "$REPORT_DIR/z6_equivalence_gate.json" \\
        || { echo "Z6 gate FAIL — refusing paid CUDA dispatch"; exit 1; }
    # if gate PASSES, proceed to canonical operator_authorize.py

Future Z6 derivatives (e.g. O=Z6-v2, multi-layer FiLM Wave 2 BUILD candidates,
Z6 + Wyner-Ziv side-info compositions) that ship the same Z6PCWM1 grammar
inherit this gate without re-parameterization. Substrates that fork the
grammar (e.g. Z6PCWM2 if it ever lands) need a new sister gate; the canonical
#1265 + #1265-Z6 pair is the reusable template.
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

from tac.framework_agnostic import require_mlx_core  # noqa: E402

SCHEMA_VERSION = "mlx_candidate_contest_equivalence_gate_z6pcwm1_v1"

# Canonical empirical anchor per #1258 corrected closure footer + #1265 LANDED
# 69c316ca4 (PR95/HNeRV grammar sister; 2026-05-26)
EMPIRICAL_ANCHOR_DRIFT_PR95 = 0.000011  # |S_MLX − S_PyTorch| on PR95 hnerv_muon
DEFAULT_GATE_THRESHOLD = 0.001  # 90× margin over PR95 anchor; matches #1265

Z6PCWM1_MAGIC = b"Z6WM"


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _read_archive_bytes(archive_path: Path) -> tuple[bytes, str]:
    """Read raw Z6PCWM1 bytes from either raw 0.bin OR zipped contest packet.

    Z6 L1 SCAFFOLD/PROMOTION emits raw Z6PCWM1 bytes; contest packaging wraps
    them in a single-member ZIP per the canonical inflate.sh contract
    (Catalog #146). This helper handles both transparently.

    Returns (raw_z6pcwm1_bytes, source_description).
    """
    raw = archive_path.read_bytes()
    if raw[:4] == Z6PCWM1_MAGIC:
        return raw, "raw_z6pcwm1_bytes"
    # Treat as ZIP and look for member "0.bin"
    if not zipfile.is_zipfile(archive_path):
        raise ValueError(
            f"archive at {archive_path} is neither raw Z6PCWM1 (magic {Z6PCWM1_MAGIC!r}) "
            f"nor a valid ZIP file"
        )
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
        if "0.bin" not in names:
            raise ValueError(
                f"ZIP archive at {archive_path} missing required member '0.bin'; "
                f"members: {names}"
            )
        member_bytes = zf.read("0.bin")
    if member_bytes[:4] != Z6PCWM1_MAGIC:
        raise ValueError(
            f"ZIP member '0.bin' magic {member_bytes[:4]!r} does not match "
            f"Z6PCWM1 magic {Z6PCWM1_MAGIC!r}"
        )
    return member_bytes, f"zip_member_0_bin_size_{len(member_bytes)}"


def _build_pytorch_substrate_from_archive(archive_bytes: bytes) -> tuple[Any, Any]:
    """Parse Z6PCWM1 + build PyTorch ``Z6PredictiveCodingSubstrate`` with state loaded.

    Returns (model_eval_mode, parsed_archive). The model is on CPU per the
    canonical inflate device contract (Catalog #205); no MPS per Catalog #1.
    """
    import torch

    from tac.substrates.time_traveler_l5_z6.architecture import (
        EVAL_HW,
        Z6PredictiveCodingConfig,
        Z6PredictiveCodingSubstrate,
    )
    from tac.substrates.time_traveler_l5_z6.archive import parse_archive

    arc = parse_archive(archive_bytes)
    meta = arc.meta
    pcwm_meta = meta.get("predictive_coding_world_model_meta", {})
    cfg = Z6PredictiveCodingConfig(
        latent_dim=int(arc.latent_init.shape[0]),
        encoder_input_channels=int(meta.get("encoder_input_channels", 3)),
        encoder_hidden_dim=int(meta.get("encoder_hidden_dim", 64)),
        decoder_embed_dim=int(meta["decoder_embed_dim"]),
        decoder_initial_grid_h=int(meta["decoder_initial_grid_h"]),
        decoder_initial_grid_w=int(meta["decoder_initial_grid_w"]),
        decoder_channels=tuple(int(c) for c in meta["decoder_channels"]),
        decoder_num_upsample_blocks=int(meta["decoder_num_upsample_blocks"]),
        num_pairs=int(arc.residuals.shape[0]),
        output_height=int(meta.get("output_height", EVAL_HW[0])),
        output_width=int(meta.get("output_width", EVAL_HW[1])),
        predictor_hidden_dim=int(meta.get("predictor_hidden_dim", 64)),
        predictor_film_mlp_hidden_dim=int(
            meta.get("predictor_film_mlp_hidden_dim", 32)
        ),
        predictor_kernel_size=int(pcwm_meta.get("predictor_kernel_size", 3)),
        predictor_ego_motion_dim=int(arc.ego_motion.shape[1]),
        identity_predictor=bool(pcwm_meta.get("identity_predictor", False)),
        latent_init_std=float(meta.get("latent_init_std", 0.02)),
    )
    model = Z6PredictiveCodingSubstrate(cfg).eval()
    for sub_name, sub_mod, sd in (
        ("encoder", model.encoder, arc.encoder_state_dict),
        ("decoder", model.decoder, arc.decoder_state_dict),
        ("predictor", model.predictor, arc.predictor_state_dict),
    ):
        if sub_name == "predictor" and cfg.identity_predictor:
            continue
        load_res = sub_mod.load_state_dict(sd, strict=False)
        if set(load_res.missing_keys) or set(load_res.unexpected_keys):
            raise RuntimeError(
                f"Z6 sister gate: PyTorch {sub_name} state_dict mismatch: "
                f"missing={sorted(load_res.missing_keys)} "
                f"unexpected={sorted(load_res.unexpected_keys)}"
            )
    with torch.no_grad():
        model.latent_init.copy_(arc.latent_init.to(dtype=model.latent_init.dtype))
        model.residuals.copy_(arc.residuals.to(dtype=model.residuals.dtype))
        model.ego_motion_buffer.copy_(
            arc.ego_motion.to(dtype=model.ego_motion_buffer.dtype)
        )
    return model, arc


def _build_mlx_renderer_from_archive(archive_bytes: bytes) -> Any:
    """Parse Z6PCWM1 + build MLX ``Z6PredictiveCodingMLXRenderer`` with state loaded.

    Mirrors :func:`_build_pytorch_substrate_from_archive` but on the MLX side.
    The MLX renderer's :meth:`reconstruct_pair` produces decoder output in the
    same canonical [0, 1] sigmoid space as the PyTorch sister.
    """
    mx = require_mlx_core()
    import numpy as np

    from tac.substrates.time_traveler_l5_z6.architecture import (
        EVAL_HW,
        Z6PredictiveCodingConfig,
    )
    from tac.substrates.time_traveler_l5_z6.archive import parse_archive
    from tac.substrates.time_traveler_l5_z6.mlx_renderer import (
        Z6PredictiveCodingMLXRenderer,
    )

    arc = parse_archive(archive_bytes)
    meta = arc.meta
    pcwm_meta = meta.get("predictive_coding_world_model_meta", {})
    cfg = Z6PredictiveCodingConfig(
        latent_dim=int(arc.latent_init.shape[0]),
        encoder_input_channels=int(meta.get("encoder_input_channels", 3)),
        encoder_hidden_dim=int(meta.get("encoder_hidden_dim", 64)),
        decoder_embed_dim=int(meta["decoder_embed_dim"]),
        decoder_initial_grid_h=int(meta["decoder_initial_grid_h"]),
        decoder_initial_grid_w=int(meta["decoder_initial_grid_w"]),
        decoder_channels=tuple(int(c) for c in meta["decoder_channels"]),
        decoder_num_upsample_blocks=int(meta["decoder_num_upsample_blocks"]),
        num_pairs=int(arc.residuals.shape[0]),
        output_height=int(meta.get("output_height", EVAL_HW[0])),
        output_width=int(meta.get("output_width", EVAL_HW[1])),
        predictor_hidden_dim=int(meta.get("predictor_hidden_dim", 64)),
        predictor_film_mlp_hidden_dim=int(
            meta.get("predictor_film_mlp_hidden_dim", 32)
        ),
        predictor_kernel_size=int(pcwm_meta.get("predictor_kernel_size", 3)),
        predictor_ego_motion_dim=int(arc.ego_motion.shape[1]),
        identity_predictor=bool(pcwm_meta.get("identity_predictor", False)),
        latent_init_std=float(meta.get("latent_init_std", 0.02)),
    )
    renderer = Z6PredictiveCodingMLXRenderer(cfg)

    # Load encoder state_dict (PyTorch OIHW → MLX HWIO transpose for Conv2d)
    def _torch_oihw_to_mlx_hwio(arr: np.ndarray) -> np.ndarray:
        if arr.ndim != 4:
            return arr
        return np.transpose(arr, (0, 2, 3, 1))

    enc_sd = {k: np.asarray(v) for k, v in arc.encoder_state_dict.items()}
    renderer.encoder.stem.weight = mx.array(_torch_oihw_to_mlx_hwio(enc_sd["stem.weight"]))
    renderer.encoder.stem.bias = mx.array(enc_sd["stem.bias"])
    renderer.encoder.head_mu.weight = mx.array(enc_sd["head_mu.weight"])
    renderer.encoder.head_mu.bias = mx.array(enc_sd["head_mu.bias"])
    renderer.encoder.head_logvar.weight = mx.array(enc_sd["head_logvar.weight"])
    renderer.encoder.head_logvar.bias = mx.array(enc_sd["head_logvar.bias"])

    # Load predictor (FiLM MLP Linear + input/output Conv2d)
    pred_sd = {k: np.asarray(v) for k, v in arc.predictor_state_dict.items()}
    if not cfg.identity_predictor:
        renderer.predictor.film_mlp_0.weight = mx.array(pred_sd["film_mlp.0.weight"])
        renderer.predictor.film_mlp_0.bias = mx.array(pred_sd["film_mlp.0.bias"])
        renderer.predictor.film_mlp_2.weight = mx.array(pred_sd["film_mlp.2.weight"])
        renderer.predictor.film_mlp_2.bias = mx.array(pred_sd["film_mlp.2.bias"])
        renderer.predictor.input_conv.weight = mx.array(
            _torch_oihw_to_mlx_hwio(pred_sd["input_conv.weight"])
        )
        renderer.predictor.input_conv.bias = mx.array(pred_sd["input_conv.bias"])
        renderer.predictor.output_conv.weight = mx.array(
            _torch_oihw_to_mlx_hwio(pred_sd["output_conv.weight"])
        )
        renderer.predictor.output_conv.bias = mx.array(pred_sd["output_conv.bias"])

    # Load decoder (initial_proj Linear + blocks Conv2d sequence)
    dec_sd = {k: np.asarray(v) for k, v in arc.decoder_state_dict.items()}
    renderer.decoder.initial_proj.weight = mx.array(dec_sd["initial_proj.weight"])
    renderer.decoder.initial_proj.bias = mx.array(dec_sd["initial_proj.bias"])
    for i in range(renderer.decoder.num_upsample_blocks):
        pytorch_idx = 3 * i
        mlx_conv = getattr(renderer.decoder, f"_block_conv_{pytorch_idx}")
        mlx_conv.weight = mx.array(
            _torch_oihw_to_mlx_hwio(dec_sd[f"blocks.{pytorch_idx}.weight"])
        )
        mlx_conv.bias = mx.array(dec_sd[f"blocks.{pytorch_idx}.bias"])
    final_idx = renderer.decoder._final_conv_index
    final_conv = getattr(renderer.decoder, f"_block_conv_{final_idx}")
    final_conv.weight = mx.array(
        _torch_oihw_to_mlx_hwio(dec_sd[f"blocks.{final_idx}.weight"])
    )
    final_conv.bias = mx.array(dec_sd[f"blocks.{final_idx}.bias"])

    # Load latent_init, residuals, ego_motion (numpy → MLX)
    renderer.latent_init = mx.array(np.asarray(arc.latent_init))
    renderer.residuals = mx.array(np.asarray(arc.residuals))
    renderer.ego_motion_buffer = mx.array(np.asarray(arc.ego_motion))

    return renderer


def _render_pair_batch_pytorch(model: Any, pair_indices: list[int]) -> Any:
    """Render N pairs via PyTorch substrate; returns (N, 2, 3, H, W) float32 [0,1]."""
    import numpy as np
    import torch

    idx_tensor = torch.tensor(pair_indices, dtype=torch.long)
    with torch.no_grad():
        rgb_0, rgb_1, _z = model.reconstruct_pair(idx_tensor)
    # rgb_0, rgb_1 are (N, 3, H, W) each in [0, 1]
    out = torch.stack([rgb_0, rgb_1], dim=1)  # (N, 2, 3, H, W)
    return out.detach().cpu().numpy().astype(np.float32)


def _render_pair_batch_mlx(renderer: Any, pair_indices: list[int]) -> Any:
    """Render N pairs via MLX renderer; returns (N, 2, 3, H, W) float32 [0,1].

    The MLX decoder produces NHWC outputs (B, H, W, 3); this helper transposes
    to NCHW (B, 3, H, W) so direct comparison to the PyTorch sister is
    layout-aligned.
    """
    mx = require_mlx_core()
    import numpy as np

    idx_mx = mx.array(np.asarray(pair_indices, dtype=np.int32))
    rgb_0, rgb_1, _z = renderer.reconstruct_pair(idx_mx)
    # rgb_0, rgb_1 are MLX (N, H, W, 3); convert NHWC → NCHW
    rgb_0_np = np.asarray(rgb_0)
    rgb_1_np = np.asarray(rgb_1)
    if rgb_0_np.ndim != 4 or rgb_1_np.ndim != 4:
        raise RuntimeError(
            f"unexpected MLX decoder output shape: rgb_0={rgb_0_np.shape}, rgb_1={rgb_1_np.shape}"
        )
    # NHWC → NCHW
    rgb_0_chw = np.transpose(rgb_0_np, (0, 3, 1, 2))
    rgb_1_chw = np.transpose(rgb_1_np, (0, 3, 1, 2))
    out = np.stack([rgb_0_chw, rgb_1_chw], axis=1)  # (N, 2, 3, H, W)
    return out.astype(np.float32)


def measure_z6_decoder_parity(
    archive_path: Path,
    n_pairs: int = 100,
) -> dict[str, Any]:
    """Render N pairs via both PyTorch + MLX from the same archive; measure drift.

    Returns a dict with max_abs_drift, mean_abs_drift, per-pair stats, and
    timing — the operational measurements the gate verdict consumes.
    """
    import numpy as np

    archive_bytes, archive_source = _read_archive_bytes(archive_path)
    archive_bytes_sha = _hash_bytes(archive_bytes)
    archive_bytes_size = len(archive_bytes)

    t0 = time.perf_counter()
    pytorch_model, parsed_arc = _build_pytorch_substrate_from_archive(archive_bytes)
    t1 = time.perf_counter()
    mlx_renderer = _build_mlx_renderer_from_archive(archive_bytes)
    t2 = time.perf_counter()

    available_pairs = int(parsed_arc.residuals.shape[0])
    effective_n_pairs = min(n_pairs, available_pairs)
    if effective_n_pairs <= 0:
        raise ValueError(
            f"Z6 archive has 0 pairs (num_pairs={available_pairs}); cannot measure parity"
        )
    pair_indices = list(range(effective_n_pairs))

    t3 = time.perf_counter()
    pytorch_frames = _render_pair_batch_pytorch(pytorch_model, pair_indices)
    t4 = time.perf_counter()
    mlx_frames = _render_pair_batch_mlx(mlx_renderer, pair_indices)
    t5 = time.perf_counter()

    if pytorch_frames.shape != mlx_frames.shape:
        raise RuntimeError(
            f"PyTorch frame shape {pytorch_frames.shape} != MLX {mlx_frames.shape}"
        )

    drift = np.abs(pytorch_frames - mlx_frames)
    max_abs_drift = float(drift.max())
    mean_abs_drift = float(drift.mean())
    per_pair_max_drift = drift.reshape(effective_n_pairs, -1).max(axis=1).astype(float)

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
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        required=True,
        help="Path to candidate's Z6PCWM1 archive (raw 0.bin OR zipped contest packet).",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=100,
        help="Number of pairs to render on both decoders (default=100; bounded by archive's num_pairs).",
    )
    parser.add_argument(
        "--gate-threshold-decoder-parity",
        type=float,
        default=DEFAULT_GATE_THRESHOLD,
        help=(
            f"Gate threshold for max_abs(MLX_decode − PyTorch_decode) in [0,1] "
            f"sigmoid space. Default {DEFAULT_GATE_THRESHOLD} mirrors the PR95 "
            f"#1265 gate threshold (90× margin over PR95 empirical anchor 0.000011)."
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
        default="anonymous_z6_mlx_candidate",
        help="Label for canonical Provenance + verdict row.",
    )
    args = parser.parse_args()

    if not args.archive.is_file():
        print(f"[gate-z6] ERROR: --archive does not exist: {args.archive}", file=sys.stderr)
        return 2
    if args.gate_threshold_decoder_parity <= 0:
        print(
            "[gate-z6] ERROR: --gate-threshold-decoder-parity must be > 0",
            file=sys.stderr,
        )
        return 2
    if args.n_pairs <= 0:
        print("[gate-z6] ERROR: --n-pairs must be > 0", file=sys.stderr)
        return 2

    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    print("[gate-z6] Measuring Z6PCWM1 MLX↔PyTorch decoder parity")
    print(f"[gate-z6]   archive: {args.archive}")
    print(f"[gate-z6]   threshold: max_abs(MLX−PyTorch) < {args.gate_threshold_decoder_parity}")
    print(f"[gate-z6]   n_pairs:  {args.n_pairs}")

    try:
        measurement = measure_z6_decoder_parity(args.archive, n_pairs=args.n_pairs)
    except Exception as exc:
        print(f"[gate-z6] ERROR during measurement: {exc}", file=sys.stderr)
        return 2

    actual_drift = float(measurement["max_abs_drift"])

    # Gate decision
    if actual_drift < args.gate_threshold_decoder_parity:
        verdict = "PASS"
        exit_code = 0
    else:
        verdict = "FAIL"
        exit_code = 1

    # Build canonical Provenance per Catalog #323
    from tac.provenance.builders import build_provenance_for_predicted
    from tac.provenance.validator import provenance_to_dict

    inputs_sha = _hash_file(args.archive)
    prov = build_provenance_for_predicted(
        model_id=(
            f"mlx_candidate_contest_equivalence_gate_z6pcwm1:{args.candidate_label}"
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
        "margin_below_threshold": args.gate_threshold_decoder_parity - actual_drift,
        "ratio_actual_vs_pr95_empirical_anchor": (
            actual_drift / EMPIRICAL_ANCHOR_DRIFT_PR95
            if EMPIRICAL_ANCHOR_DRIFT_PR95 > 0
            else None
        ),
        "candidate_label": args.candidate_label,
        "candidate_grammar": "Z6PCWM1",
        "candidate_substrate_id": "time_traveler_l5_z6",
        "candidate_substrate_class": "predictive_coding_world_model",
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
                "z6_scorer_axis_equivalence_steps_3_4_deferred_to_paid_cuda_dispatch_"
                "per_yousfi_l1_promotion_symposium_dissent_score_aware_lagrangian_routes_"
                "through_pytorch_sister_per_catalog_164_226"
            ),
        ],
        "provenance": provenance_to_dict(prov),
        "measurement": measurement,
        "canonical_anchor": {
            "pr95_empirical_anchor_drift": EMPIRICAL_ANCHOR_DRIFT_PR95,
            "pr95_empirical_anchor_source_landing_memo": (
                ".omx/research/pr95_mlx_full_decoder_downstream_scorer_drift_landed_20260526.md"
            ),
            "z6_archive_grammar_source": (
                "src.tac.substrates.time_traveler_l5_z6.archive::parse_archive (Z6PCWM1)"
            ),
            "z6_l1_promotion_landing_memo": (
                ".omx/research/path_3_d_z6_l1_promotion_landed_20260526.md"
            ),
            "z6_l1_promotion_commit": "8833b9db5",
            "pr95_canonical_gate_commit": "69c316ca4",
            "cascade_doctrine_commit": "fb270e9b6",
            "mlx_first_doctrine_commit": "4107bbf8d",
            "sister_lane_id": (
                "lane_path_3_sister_1265_gate_z6pcwm1_grammar_20260526"
            ),
        },
        "operator_routable_per_verdict": (
            (
                "PROCEED: Z6 candidate's MLX decoder matches PyTorch decoder within "
                "gate threshold; D=Z6 unlocked for per-substrate-class bridge "
                "calibration per MLX-first doctrine + paid CUDA dispatch authorization "
                "per CLAUDE.md 'Submission auth eval — BOTH CPU AND CUDA' non-negotiable"
            )
            if verdict == "PASS"
            else (
                "REFUSE: Z6 candidate's MLX decoder drift exceeds gate threshold; "
                "do NOT dispatch to paid CUDA; audit Z6 MLX↔PyTorch architecture "
                "parity via tac.substrates.time_traveler_l5_z6.mlx_renderer + "
                "tac.substrates.time_traveler_l5_z6.architecture per the canonical "
                "Z6PredictiveCodingMLXRenderer + Z6PredictiveCodingSubstrate "
                "mirror-architecture contract"
            )
        ),
        "scope_note": (
            "Covers Steps 1-2 of canonical 4-step #1265 closure (archive grammar "
            "parse + MLX↔PyTorch decoder parity). Steps 3-4 (scorer-axis "
            "equivalence via DistortionNet on candidate-vs-GT frames) DEFERRED to "
            "paid CUDA dispatch per Yousfi L1 promotion symposium dissent + "
            "Catalog #164 + #226 sister discipline."
        ),
    }
    args.output_json.write_text(json.dumps(gate_verdict, indent=2))

    print()
    print("=== Z6PCWM1 MLX CANDIDATE CONTEST-EQUIVALENCE GATE ===")
    print(f"  candidate: {args.candidate_label}")
    print(f"  archive source: {measurement['archive_source']}")
    print(f"  archive sha256: {measurement['archive_bytes_sha256'][:16]}...")
    print(f"  archive bytes: {measurement['archive_bytes_size']:,}")
    print(f"  pairs measured: {measurement['n_pairs_measured']} / {measurement['n_pairs_available']}")
    print(f"  frame shape (PyTorch + MLX): {measurement['frame_shape']}")
    print(f"  decoder output space: {measurement['decoder_output_space']}")
    print(f"  max_abs drift: {actual_drift:.6f}")
    print(f"  mean_abs drift: {measurement['mean_abs_drift']:.6f}")
    print(f"  per-pair max drift mean: {measurement['per_pair_max_drift_mean']:.6f}")
    print(f"  threshold:    {args.gate_threshold_decoder_parity:.6f}")
    print(f"  margin:       {args.gate_threshold_decoder_parity - actual_drift:.6f}")
    if EMPIRICAL_ANCHOR_DRIFT_PR95 > 0:
        print(
            f"  ratio vs PR95 empirical anchor (0.000011): "
            f"{actual_drift / EMPIRICAL_ANCHOR_DRIFT_PR95:.2f}×"
        )
    print(f"  build (PyTorch / MLX): {measurement['pytorch_build_seconds']:.2f}s / {measurement['mlx_build_seconds']:.2f}s")
    print(f"  render (PyTorch / MLX): {measurement['pytorch_render_seconds']:.2f}s / {measurement['mlx_render_seconds']:.2f}s")
    print(f"  VERDICT: {verdict}")
    print(f"  exit code: {exit_code}")
    print(f"  output: {args.output_json}")
    return exit_code


__all__ = [
    "DEFAULT_GATE_THRESHOLD",
    "EMPIRICAL_ANCHOR_DRIFT_PR95",
    "SCHEMA_VERSION",
    "Z6PCWM1_MAGIC",
    "main",
    "measure_z6_decoder_parity",
]


if __name__ == "__main__":
    sys.exit(main())
