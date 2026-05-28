#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Sister Catalog #1265 canonical PASS/FAIL gate for Z6-v2 Z6V2CU1-grammar candidates.

Parameterized sister of:
- ``tools/gate_mlx_candidate_contest_equivalence.py`` (PR95/HNeRV-grammar canonical; commit ``69c316ca4``)
- ``tools/gate_mlx_candidate_contest_equivalence_pact_nerv_ia3.py`` (IA3 grammar)
- ``tools/gate_mlx_candidate_contest_equivalence_pact_nerv_selector_v2.py`` (PSV2 grammar)
- ``tools/gate_mlx_candidate_contest_equivalence_pact_nerv_selector_v3.py`` (PSV3 grammar; commit ``2f69d0ea6``)

Z6-v2 ships Z6V2CU1 grammar (``Z6V2`` magic + 28-byte header + 4 sections per
``src.tac.substrates.z6_v2_cargo_cult_unwind.archive``). Per the cascade
doctrine ``fb270e9b6`` L6 gate requirement + the MLX-first doctrine
``4107bbf8d`` per-class bridge calibration scope, each substrate-class
grammar needs its own sister #1265 gate parameterized for that grammar.

This gate empirically establishes MLX <-> PyTorch decoder parity on the
Z6-v2 base decoder forward path (``Z6V2Substrate.forward`` vs
``Z6V2SubstrateMLX.__call__``). The decoder output drift is measured in the
canonical sigmoid ``[0, 1]`` output space; threshold 0.001 mirrors the
empirical PR95 anchor + sister gate convention. NOTE: Z6-v2's 2-level
Rao-Ballard FiLM hierarchy with per-pair (latent ⊕ ego_vec) modulation
adds additional architectural complexity beyond pure NeRV-class decoders;
the gate verdict is an OBSERVABILITY/disambiguator signal, NOT a
contest-promotion gate per CLAUDE.md "Submission auth eval - BOTH CPU AND
CUDA" non-negotiable.

Scope note - this gate covers Steps 1-2 of the canonical 4-step #1265 closure:

    1. Parse archive (NEW: Z6V2CU1 grammar via ``parse_archive``)
    2. MLX <-> PyTorch decoder parity (NEW: Z6-v2 base decoder on identical
       state_dict; max_abs drift in [0,1] sigmoid space)

Steps 3-4 (scorer-axis parity via DistortionNet on candidate-vs-GT frames)
are DEFERRED to operator-routed paid CUDA dispatch per CLAUDE.md "Catalog
#164 + #226 sister discipline" - Z6-v2's score-aware Lagrangian with
Atick-Redlich cooperative-receiver gradient binding (Catalog #311) routes
through the PyTorch sister + paid CUDA (not MLX-local).

Gate semantics:
  PASS: max_abs(MLX_decode - PyTorch_decode) < --gate-threshold-decoder-parity (default 0.001)
  FAIL: otherwise

Exit codes:
  0 = PASS (Z6-v2 candidate's MLX decoder matches PyTorch within gate)
  1 = FAIL (Z6-v2 candidate's MLX decoder drift exceeds gate)
  2 = CLI / measurement error

Per CLAUDE.md "MLX portable-local-substrate authority": output carries
axis_tag="[macOS-MLX research-signal]", score_claim=False, promotable=False,
ready_for_exact_eval_dispatch=False per Catalog #127/#192/#317/#341 +
canonical Provenance per Catalog #323.

Z6-v2 DISTINGUISHING vs sister gates (per Catalog #290):

- Z6V2CU1 archive grammar has TWO per-pair tensors (``latents`` + ``ego_vecs``)
  whereas sister PACT-NeRV grammars have ONE (latents).
- Z6-v2's distinguishing primitive (2-level Rao-Ballard FiLM hierarchy +
  Atick-Redlich cooperative-receiver) is captured at the MLX renderer
  state_dict layer; the gate's MLX<->PyTorch parity check exercises THE FULL
  FORWARD PATH including FiLM modulation per per-pair (latent ⊕ ego_vec).
- Per Catalog #1305 drift-vs-depth signature: 7 PixelShuffle + SIREN sin(30)
  blocks amplify per-layer ~1e-6 conv drift exponentially; expect higher
  drift than sister NeRV-class substrates without FiLM modulation.

Integration pattern for operator-authorize wrappers:

    .venv/bin/python tools/gate_mlx_candidate_contest_equivalence_z6_v2.py \\
        --archive "$Z6V2CU1_ARCHIVE_PATH" \\
        --gate-threshold-decoder-parity 0.001 \\
        --output-json "$REPORT_DIR/z6_v2_equivalence_gate.json" \\
        || { echo "Z6-v2 gate FAIL - refusing paid CUDA dispatch"; exit 1; }
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

SCHEMA_VERSION = "mlx_candidate_contest_equivalence_gate_z6v2cu1_v1"

# Canonical empirical anchor per #1265 LANDED 69c316ca4 (PR95/HNeRV grammar
# sister; 2026-05-26).
EMPIRICAL_ANCHOR_DRIFT_PR95 = 0.000011  # |S_MLX - S_PyTorch| on PR95 hnerv_muon
DEFAULT_GATE_THRESHOLD = 0.001  # 90x margin over PR95 anchor; matches #1265

Z6V2_MAGIC = b"Z6V2"


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _read_archive_bytes(archive_path: Path) -> tuple[bytes, str]:
    """Read raw Z6V2CU1 bytes from either raw 0.bin OR zipped contest packet."""
    raw = archive_path.read_bytes()
    if raw[:4] == Z6V2_MAGIC:
        return raw, "raw_z6v2_bytes"
    if not zipfile.is_zipfile(archive_path):
        raise ValueError(
            f"archive at {archive_path} is neither raw Z6V2CU1 (magic {Z6V2_MAGIC!r}) "
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
    if member_bytes[:4] != Z6V2_MAGIC:
        raise ValueError(
            f"ZIP member '0.bin' magic {member_bytes[:4]!r} does not match "
            f"Z6V2CU1 magic {Z6V2_MAGIC!r}"
        )
    return member_bytes, f"zip_member_0_bin_size_{len(member_bytes)}"


def _build_pytorch_substrate_from_archive(archive_bytes: bytes) -> tuple[Any, Any]:
    """Parse Z6V2CU1 + build PyTorch ``Z6V2Substrate`` with state loaded."""
    import torch

    from tac.substrates.z6_v2_cargo_cult_unwind.architecture import (
        Z6V2Config,
        Z6V2Substrate,
    )
    from tac.substrates.z6_v2_cargo_cult_unwind.archive import parse_archive

    arc = parse_archive(archive_bytes)
    meta = arc.meta
    cfg = Z6V2Config(
        latent_dim=int(arc.latents.shape[1]),
        ego_dim=int(arc.ego_vecs.shape[1]),
        embed_dim=int(meta["embed_dim"]),
        initial_grid_h=int(meta["initial_grid_h"]),
        initial_grid_w=int(meta["initial_grid_w"]),
        decoder_channels=tuple(int(c) for c in meta["decoder_channels"]),
        sin_frequency=float(meta["sin_frequency"]),
        num_upsample_blocks=int(meta["num_upsample_blocks"]),
        rao_ballard_level_boundary=int(meta.get("rao_ballard_level_boundary", 3)),
        film_generator_depth=int(meta.get("film_generator_depth", 3)),
        film_hidden_width=int(meta.get("film_hidden_width", 80)),
        cooperative_receiver_beta=float(meta.get("cooperative_receiver_beta", 0.5)),
        num_pairs=int(arc.latents.shape[0]),
        output_height=int(meta["output_height"]),
        output_width=int(meta["output_width"]),
    )
    model = Z6V2Substrate(cfg).eval()
    load_result = model.load_state_dict(arc.decoder_state_dict, strict=False)
    # Z6-v2 has TWO per-pair tensors loaded SEPARATELY from decoder state dict.
    allowed_missing = {"latents", "ego_vecs"}
    missing_keys = set(load_result.missing_keys) - allowed_missing
    if missing_keys:
        raise RuntimeError(
            f"Z6-v2 gate: PyTorch missing_keys={sorted(missing_keys)} "
            f"(allowed_missing={sorted(allowed_missing)})"
        )
    if set(load_result.unexpected_keys):
        raise RuntimeError(
            f"Z6-v2 gate: PyTorch unexpected_keys = "
            f"{sorted(load_result.unexpected_keys)}"
        )
    with torch.no_grad():
        model.latents.copy_(
            arc.latents.to(device="cpu", dtype=model.latents.dtype)
        )
        model.ego_vecs.copy_(
            arc.ego_vecs.to(device="cpu", dtype=model.ego_vecs.dtype)
        )
    return model, arc


def _build_mlx_renderer_from_archive(archive_bytes: bytes) -> Any:
    """Parse Z6V2CU1 + build MLX ``Z6V2SubstrateMLX`` with state loaded."""
    import mlx.core as mx
    import numpy as np

    from tac.substrates.z6_v2_cargo_cult_unwind.architecture import Z6V2Config
    from tac.substrates.z6_v2_cargo_cult_unwind.archive import parse_archive
    from tac.substrates.z6_v2_cargo_cult_unwind.mlx_renderer import (
        Z6V2SubstrateMLX,
    )

    arc = parse_archive(archive_bytes)
    meta = arc.meta
    cfg = Z6V2Config(
        latent_dim=int(arc.latents.shape[1]),
        ego_dim=int(arc.ego_vecs.shape[1]),
        embed_dim=int(meta["embed_dim"]),
        initial_grid_h=int(meta["initial_grid_h"]),
        initial_grid_w=int(meta["initial_grid_w"]),
        decoder_channels=tuple(int(c) for c in meta["decoder_channels"]),
        sin_frequency=float(meta["sin_frequency"]),
        num_upsample_blocks=int(meta["num_upsample_blocks"]),
        rao_ballard_level_boundary=int(meta.get("rao_ballard_level_boundary", 3)),
        film_generator_depth=int(meta.get("film_generator_depth", 3)),
        film_hidden_width=int(meta.get("film_hidden_width", 80)),
        cooperative_receiver_beta=float(meta.get("cooperative_receiver_beta", 0.5)),
        num_pairs=int(arc.latents.shape[0]),
        output_height=int(meta["output_height"]),
        output_width=int(meta["output_width"]),
    )
    renderer = Z6V2SubstrateMLX(cfg)

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

    # Walk state_dict keys + assign onto MLX renderer (mirror layout).
    # PyTorch sister wraps FiLM-generator MLP in nn.Sequential so keys are
    # ``blocks.<i>.film_gen.mlp.<idx>.{weight,bias}`` where idx is 0/2/4/...
    # (even = Linear; odd = _SinAct activation). The MLX sister stores
    # Linears in a plain ``layers`` list so the canonical mapping is:
    #   PyTorch ``film_gen.mlp.<2*j>`` -> MLX ``film_gen.layers.<j>``
    # We rewrite the dotted name accordingly before walking the MLX tree.
    for name, t in arc.decoder_state_dict.items():
        mlx_name = name
        if ".film_gen.mlp." in name:
            head, tail = name.split(".film_gen.mlp.", 1)
            idx_str, rest = tail.split(".", 1)
            pyt_idx = int(idx_str)
            if pyt_idx % 2 != 0:
                # Odd index is _SinAct activation (no trainable params); skip.
                continue
            mlx_idx = pyt_idx // 2
            mlx_name = f"{head}.film_gen.layers.{mlx_idx}.{rest}"
        parts = mlx_name.split(".")
        obj: Any = renderer
        for part in parts[:-1]:
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        leaf = parts[-1]
        setattr(obj, leaf, _pyt_to_mlx_layout(name, t))

    # Per-pair latents + ego_vecs (top-level mx.array attribute; same layout).
    renderer.latents = mx.array(_tensor_to_numpy_float32(arc.latents))
    renderer.ego_vecs = mx.array(_tensor_to_numpy_float32(arc.ego_vecs))
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
    out_255_b2chw = np.asarray(renderer(idx), dtype=np.float32)
    return out_255_b2chw / 255.0


def measure_z6_v2_decoder_parity(
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

    available_pairs = int(parsed_arc.latents.shape[0])
    effective_n_pairs = min(n_pairs, available_pairs)
    if effective_n_pairs <= 0:
        raise ValueError(
            f"Z6V2CU1 archive has 0 pairs (num_pairs={available_pairs}); "
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
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, required=True,
        help="Path to candidate's Z6V2CU1 archive (raw 0.bin OR zipped contest packet).",
    )
    parser.add_argument(
        "--n-pairs", type=int, default=100,
        help="Number of pairs to render on both decoders (default=100; bounded by num_pairs).",
    )
    parser.add_argument(
        "--gate-threshold-decoder-parity",
        type=float, default=DEFAULT_GATE_THRESHOLD,
        help=(
            f"Gate threshold for max_abs(MLX_decode - PyTorch_decode) in [0,1] "
            f"sigmoid space. Default {DEFAULT_GATE_THRESHOLD} mirrors the PR95 "
            f"#1265 gate threshold (90x margin over PR95 empirical anchor 0.000011)."
        ),
    )
    parser.add_argument(
        "--output-json", type=Path, required=True,
        help="Output JSON path for gate verdict.",
    )
    parser.add_argument(
        "--candidate-label", type=str,
        default="anonymous_z6_v2_mlx_candidate",
        help="Label for canonical Provenance + verdict row.",
    )
    args = parser.parse_args()

    if not args.archive.is_file():
        print(
            f"[gate-z6v2] ERROR: --archive does not exist: {args.archive}",
            file=sys.stderr,
        )
        return 2
    if args.gate_threshold_decoder_parity <= 0:
        print(
            "[gate-z6v2] ERROR: --gate-threshold-decoder-parity must be > 0",
            file=sys.stderr,
        )
        return 2
    if args.n_pairs <= 0:
        print("[gate-z6v2] ERROR: --n-pairs must be > 0", file=sys.stderr)
        return 2

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    print("[gate-z6v2] Measuring Z6V2CU1 MLX<->PyTorch decoder parity")
    print(f"[gate-z6v2]   archive: {args.archive}")
    print(
        f"[gate-z6v2]   threshold: max_abs(MLX-PyTorch) < {args.gate_threshold_decoder_parity}"
    )
    print(f"[gate-z6v2]   n_pairs:  {args.n_pairs}")

    try:
        measurement = measure_z6_v2_decoder_parity(
            args.archive, n_pairs=args.n_pairs
        )
    except Exception as exc:
        print(f"[gate-z6v2] ERROR during measurement: {exc}", file=sys.stderr)
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
            f"mlx_candidate_contest_equivalence_gate_z6v2cu1:{args.candidate_label}"
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
        "candidate_grammar": "Z6V2CU1",
        "candidate_substrate_id": "z6_v2_cargo_cult_unwind",
        "candidate_substrate_class": (
            "ego_motion_conditioned_predictive_coding_2_level_rao_ballard_film_hierarchy_"
            "with_atick_redlich_cooperative_receiver_per_catalog_311"
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
                "z6_v2_scorer_axis_equivalence_steps_3_4_deferred_to_paid_cuda_dispatch_"
                "per_catalog_164_226_score_aware_lagrangian_routes_through_pytorch_sister_"
                "with_atick_redlich_cooperative_receiver_gradient_binding_per_catalog_311"
            ),
            (
                "z6_v2_2_level_rao_ballard_film_hierarchy_with_7_pixelshuffle_siren_blocks_"
                "amplifies_per_layer_drift_per_catalog_1305_drift_vs_depth_anchor_drift_"
                "outside_threshold_does_NOT_imply_bridge_bug"
            ),
        ],
        "provenance": provenance_to_dict(prov),
        "measurement": measurement,
        "canonical_anchor": {
            "pr95_empirical_anchor_drift": EMPIRICAL_ANCHOR_DRIFT_PR95,
            "z6v2cu1_archive_grammar_source": (
                "src.tac.substrates.z6_v2_cargo_cult_unwind.archive::parse_archive (Z6V2CU1)"
            ),
            "z6_v2_landing_memo": (
                ".omx/research/z6_v2_cargo_cult_unwind_l1_long_run_mlx_landed_20260528.md"
            ),
            "pr95_canonical_gate_commit": "69c316ca4",
            "pact_nerv_selector_v3_sister_gate_path": (
                "tools/gate_mlx_candidate_contest_equivalence_pact_nerv_selector_v3.py"
            ),
            "drift_vs_depth_anchor_catalog": 1305,
            "ego_motion_predictive_coding_paradigm_catalog": 311,
        },
        "operator_routable_per_verdict": (
            (
                "PROCEED: Z6-v2 candidate's MLX decoder matches PyTorch decoder within "
                "gate threshold; Z6V2CU1 unlocked for per-substrate-class bridge "
                "calibration per MLX-first doctrine + paired CUDA + CPU dispatch "
                "authorization per CLAUDE.md 'Submission auth eval - BOTH CPU AND CUDA' "
                "non-negotiable + Catalog #325 14-day per-substrate symposium window"
            )
            if verdict == "PASS"
            else (
                "OBSERVABILITY-ONLY: Z6-v2 candidate's MLX decoder drift exceeds gate "
                "threshold; per Catalog #1305 drift-vs-depth this is EXPECTED for the "
                "Z6-v2 7-PixelShuffle SIREN substrate with 2-level Rao-Ballard FiLM "
                "hierarchy. The gate is a research-signal disambiguator NOT a "
                "contest-promotion gate; the operator MAY still dispatch paired CPU+CUDA "
                "per CLAUDE.md 'Submission auth eval - BOTH CPU AND CUDA' for the "
                "canonical contest score - the MLX-side drift is irrelevant on the paid "
                "CUDA path because the PyTorch sister IS the contest substrate."
            )
        ),
        "scope_note": (
            "Covers Steps 1-2 of canonical 4-step #1265 closure (archive grammar parse + "
            "MLX<->PyTorch decoder parity). Steps 3-4 (scorer-axis equivalence via "
            "DistortionNet on candidate-vs-GT frames) DEFERRED to paid CUDA dispatch per "
            "Catalog #164 + #226 sister discipline. Per Catalog #1305 drift-vs-depth the "
            "Z6-v2 substrate's 2-level Rao-Ballard FiLM hierarchy on top of 7 SIREN+"
            "PixelShuffle blocks architecturally diverges from PyTorch numerics on Apple "
            "Silicon; the gate verdict is OBSERVABILITY-ONLY (not contest-promotion-binding)."
        ),
    }
    args.output_json.write_text(json.dumps(gate_verdict, indent=2) + "\n")

    print()
    print("=== Z6V2CU1 MLX CANDIDATE CONTEST-EQUIVALENCE GATE ===")
    print(f"  candidate: {args.candidate_label}")
    print(f"  archive source: {measurement['archive_source']}")
    print(f"  archive sha256: {measurement['archive_bytes_sha256'][:16]}...")
    print(f"  archive bytes: {measurement['archive_bytes_size']:,}")
    print(
        f"  pairs measured: {measurement['n_pairs_measured']} / {measurement['n_pairs_available']}"
    )
    print(f"  frame shape: {measurement['frame_shape']}")
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
    "SCHEMA_VERSION",
    "Z6V2_MAGIC",
    "main",
    "measure_z6_v2_decoder_parity",
]


if __name__ == "__main__":
    sys.exit(main())
