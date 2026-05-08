#!/usr/bin/env python3
"""PR101 pre-stage anchor: arch_shrink × Lagrangian × UNIWARD composed stack.

Background
----------
Three independent score-lowering primitives have empirical byte-anchors on the
PR101 substrate:

* **arch_shrink** (post-hoc channel L2-truncate at ratio r) — best anchor at
  r=0.4 lands ``83,571 B`` (canonical evidence
  ``reports/raw/pr101_arch_shrink_20260508T003827Z/manifest.json``). This is
  a CPU byte anchor only; the SCORE impact requires retraining (Lightning T4
  ``arch-shrink-x0-4-lightning-20260508t010514z`` is in flight).
* **Lagrangian per-tensor allocation** (uniform-weight λ-bisection over K-
  curves) — canonical at :mod:`tac.optimization.lagrangian_per_tensor_allocation`.
  Empirically saves bytes vs uniform-K baselines on the PR101 substrate.
* **UNIWARD-weighted Lagrangian** (inverse-variance weights on rel_err²) —
  canonical at :class:`tac.optimization.lagrangian_per_tensor_allocation.UniwardWeightedAllocator`.
  Saves +4,074 B vs uniform Lagrangian on PR101 substrate at rms=0.05
  (commit ``be715fac``).

Composition hypothesis
----------------------
The stack ``arch_shrink × Lagrangian × UNIWARD`` predicts COMPOUNDING savings:
arch_shrink reduces parameter count → the Lagrangian + UNIWARD allocator then
operates on the reduced symbol stream. Per the variance-heterogeneity finding
on HNeRV tensors (variance ratio 15.5×), UNIWARD's inverse-variance weights
should still find meaningful per-tensor allocation differences after channel
pruning.

This tool is the **pre-stage byte anchor** for that composed stack. It applies
arch_shrink (channel L2-truncation at a configurable ratio) to the raw fp32
state_dict, quantizes the truncated state_dict via the canonical PR101
quantizer, then runs the canonical Lagrangian and UNIWARD-weighted allocators
over per-tensor K-curves and reports joint-encoded byte counts at each rms
target.

Composition is conceptual (pre-quantization arch_shrink + post-quantization
Lagrangian × UNIWARD). The underlying byte-count primitives all live in
:mod:`tac.codec.cost_curves`, :mod:`tac.codec.per_tensor_codecs`, and
:mod:`tac.optimization.lagrangian_per_tensor_allocation`. The CodecPipeline
canonical orchestrator (:class:`tac.codec_pipeline.CodecPipeline`, CPL2) is
the natural home for this stack as a fully-baked CodecOp; the present tool
is the empirical pre-stage anchor BEFORE that wiring lands (CPLX1 fallback:
the present tool is a numpy/brotli pure-pipeline, not a CodecPipeline of
encoder ops). When ORCH-SYNC delivers CPL2 substrate-transform CodecOps for
arch_shrink, this tool's output becomes the regression anchor.

CLAUDE.md compliance
--------------------
* Pure CPU + numpy + brotli; no scorer load; no contest score claims.
* No Q-FAITHFUL retraining is proposed.
* Output rows tagged ``[CPU-prep faithful arch_shrink × Lagrangian × UNIWARD test]``,
  ``score_claim=False``, ``promotion_eligible=False``,
  ``ready_for_exact_eval_dispatch=False``, ``cuda_eval_worth_testing`` per
  result, ``family_falsified=False``,
  ``falsification_scope="composed_stack_post_hoc_only"``.
* Score impact of post-hoc arch truncation is unknown without retraining —
  preserved in dispatch_blockers per the tool's predecessor
  ``pr101_arch_shrink_post_hoc_sweep.py``.
* Canonical primitives are NOT duplicated; this tool delegates to:
    - :func:`tac.codec.per_tensor_codecs.encode_brotli_only` / `encode_lossy_K_coarsen`
    - :func:`tac.codec.cost_curves.precompute_per_tensor_K_curves`
    - :class:`tac.optimization.lagrangian_per_tensor_allocation.LagrangianPerTensorAllocator`
    - :class:`tac.optimization.lagrangian_per_tensor_allocation.UniwardWeightedAllocator`

Per ``forbidden_premature_class_level_falsification``: any negative result is
tagged ``MEASURED_CONFIG_NOT_DISPATCHABLE`` only — never as a class-level
falsification of arch_shrink, Lagrangian, or UNIWARD individually.

Per ``forbidden_CPU_MPS_derived_dispatch_readiness_flag``:
``ready_for_exact_eval_dispatch=False`` is hardcoded into every emitted row.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

# Canonical primitives — DO NOT duplicate these helpers per CANONICALIZE-OSS.
from tac.codec.cost_curves import (  # noqa: E402
    DEFAULT_K_RANGE,
    precompute_per_tensor_K_curves,
)
from tac.optimization.lagrangian_per_tensor_allocation import (  # noqa: E402
    LagrangianPerTensorAllocator,
    UniwardWeightedAllocator,
    compute_local_variance_proxy,
    compute_uniward_weights,
)
from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
)

# Reuse the canonical joint-encoder + arch-shrink helpers via tool modules.
from pr101_lossy_coarsening_analytical import (  # noqa: E402
    TensorBlob,
    encode_with_per_tensor_K,
)
from pr101_arch_shrink_post_hoc_sweep import (  # noqa: E402
    truncate_to_top_channels,
)

TOOL_NAME = "tools/pr101_arch_shrink_x_lagrangian_x_uniward_empirical.py"
SCHEMA_VERSION = "pr101_arch_shrink_x_lagrangian_x_uniward_empirical.v1"
EVIDENCE_GRADE = "[CPU-prep faithful arch_shrink × Lagrangian × UNIWARD test]"
EVIDENCE_SEMANTICS = (
    "cpu_post_hoc_arch_shrink_then_lagrangian_uniward_byte_anchor_no_score"
)
DEFAULT_RMS_TARGETS = [0.01, 0.02, 0.0386, 0.05, 0.10]
K_RANGE = list(DEFAULT_K_RANGE)

DISPATCH_BLOCKERS = [
    "post_hoc_arch_truncate_not_retrained",
    "score_impact_unknown_without_contest_cuda",
    "byte_rel_err_proxy_only_no_score_test",
    "no_runtime_dequantize_path_built",
    "byte_closed_arch_shrink_runtime_packet_missing",
    "missing_exact_cuda_auth_eval",
    "requires_exact_cuda_auth_eval_before_any_score_use",
    "channel_truncation_breaks_inference_until_retrained",
    "variance_proxy_substitutes_for_uniward_wavelet_residual",
    "no_iterative_primal_dual_ADMM_consensus",
    "composition_assumes_independence_pre_q_arch_post_q_alloc",
]

REACTIVATION_CRITERIA_REMAINING = [
    "arch_shrink_retrained_state_dict_with_quantizr_class_score_anchor",
    "wavelet_domain_uniward_residual_variance_proxy",
    "score_aware_per_tensor_distortion_weights_detector_in_loop",
    "iterative_primal_dual_ADMM_with_consensus",
    "uniward_weighted_lagrangian_with_CUDA_score_validation",
    "codec_pipeline_CPL2_substrate_transform_arch_shrink_op",
]


# ---------------------------------------------------------------------------
# Custody-flag block (single source of truth)
# ---------------------------------------------------------------------------


def proxy_evidence_contract(
    *, cuda_eval_worth_testing: bool = False
) -> dict[str, object]:
    """Return the immutable custody flag block for every emitted row.

    Per CLAUDE.md ``forbidden_CPU_MPS_derived_dispatch_readiness_flag``
    every CPU-derived row hardcodes ``ready_for_exact_eval_dispatch=False``.
    """
    return {
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_marker": EVIDENCE_GRADE,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "proxy_row": True,
        "cuda_eval_worth_testing": bool(cuda_eval_worth_testing),
        "family_falsified": False,
        "falsification_scope": "composed_stack_post_hoc_only",
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
    }


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ShrinkSubstrate:
    """Quantized symbols + tensor blobs for a given arch_shrink ratio."""

    shrink_ratio: float
    n_elements_orig: int
    n_elements_kept: int
    fraction_kept: float
    tensors: list[TensorBlob]


# ---------------------------------------------------------------------------
# Pre-stage 1: arch_shrink (channel L2-truncate)
# ---------------------------------------------------------------------------


def apply_arch_shrink(
    state_dict: dict, shrink_ratio: float
) -> ShrinkSubstrate:
    """Truncate each FIXED_STATE_SCHEMA tensor to top-channel L2 magnitude.

    Delegates to :func:`tools.pr101_arch_shrink_post_hoc_sweep.truncate_to_top_channels`
    for the per-tensor truncation (same primitive used by the upstream
    ``pr101_arch_shrink_post_hoc_sweep`` tool that produced the
    83,571 B r=0.4 anchor).

    Each truncated tensor is then quantized via the canonical PR101
    :func:`tac.pr101_split_brotli_codec._quantize_tensor` (signed-int7,
    ``N_QUANT=127``); the resulting int symbols become the substrate fed
    into the Lagrangian × UNIWARD allocators.

    Args:
        state_dict: input fp32 state_dict (must contain every name in
            FIXED_STATE_SCHEMA).
        shrink_ratio: channel-keep ratio in (0.0, 1.0]. ``1.0`` is the
            no-shrink control; values < 1.0 channel-prune by L2 magnitude.

    Returns:
        :class:`ShrinkSubstrate` with quantized symbols ready for K-curve
        precompute.
    """
    if not (0.0 < shrink_ratio <= 1.0):
        raise ValueError(
            f"shrink_ratio must be in (0.0, 1.0]; got {shrink_ratio}"
        )

    import torch

    tensors: list[TensorBlob] = []
    n_elements_orig = 0
    n_elements_kept = 0

    for name, _shape in FIXED_STATE_SCHEMA:
        if name not in state_dict:
            raise KeyError(
                f"state_dict missing required FIXED_STATE_SCHEMA tensor "
                f"{name!r}"
            )
        t_full = state_dict[name].detach().cpu().to(torch.float32).numpy()
        n_elements_orig += int(t_full.size)

        if shrink_ratio >= 1.0:
            t_trunc_np = t_full
        else:
            n_channels = t_full.shape[0] if t_full.ndim >= 1 else 1
            n_keep = max(1, round(n_channels * shrink_ratio))
            t_trunc_np = truncate_to_top_channels(t_full, n_keep)
        n_elements_kept += int(t_trunc_np.size)

        # Quantize the truncated tensor through the canonical PR101 path.
        # We must wrap as a torch tensor since _quantize_tensor expects torch.
        t_trunc_torch = torch.from_numpy(np.ascontiguousarray(t_trunc_np))
        qt = _quantize_tensor(name, t_trunc_torch, n_quant=N_QUANT)
        symbols_i32 = qt.q_i8.astype(np.int32).flatten()
        tensors.append(TensorBlob(name=name, raw=symbols_i32))

    return ShrinkSubstrate(
        shrink_ratio=float(shrink_ratio),
        n_elements_orig=n_elements_orig,
        n_elements_kept=n_elements_kept,
        fraction_kept=n_elements_kept / max(n_elements_orig, 1),
        tensors=tensors,
    )


# ---------------------------------------------------------------------------
# Pre-stage 2: Lagrangian × UNIWARD allocation joint encoder
# ---------------------------------------------------------------------------


def _make_joint_encoder(tensors: list[TensorBlob]):
    """Joint encoder hook for the canonical λ-bisection allocator.

    Returns a callable that maps per-tensor curve selections back to ``Ks``
    and runs :func:`encode_with_per_tensor_K`. The hook is the canonical
    interface between :class:`LagrangianPerTensorAllocator` and the joint
    brotli encoder.
    """

    def hook(selections: list[dict]) -> dict:
        Ks = [int(s["K"]) for s in selections]
        result = encode_with_per_tensor_K(tensors, Ks)
        return {
            "total_bytes": int(result["archive_bytes"]),
            "rel_err": float(result["rel_err"]),
            **{
                k: v
                for k, v in result.items()
                if k not in {"archive_bytes", "rel_err"}
            },
        }

    return hook


def run_uniform_lagrangian(
    tensors: list[TensorBlob],
    curves: list[list[dict]],
    rms_target: float,
    *,
    max_iter: int = 80,
) -> dict:
    """Uniform-weight Lagrangian λ-bisection (delegates to canonical)."""
    res = LagrangianPerTensorAllocator(
        joint_encoder=_make_joint_encoder(tensors)
    ).bisect_for_rms_target(
        curves, rms_target, max_iter=max_iter, lam_hi=1e15
    )
    return {
        "lambda": res.lam,
        "rel_err": res.rel_err,
        "rms_rel_err": res.rel_err,
        "archive_bytes": int(res.total_bytes),
        **res.joint_extras,
    }


def run_uniward_lagrangian(
    tensors: list[TensorBlob],
    curves: list[list[dict]],
    rms_target: float,
    *,
    max_iter: int = 80,
) -> dict:
    """UNIWARD inverse-variance Lagrangian λ-bisection (delegates to canonical).

    Variances are computed from the SHRUNK quantized symbols (not the
    full-substrate variances), so the UNIWARD weights are properly conditioned
    on the post-arch-shrink statistics. This is the score-relevant move per
    the canonical UNIWARD formulation: weights MUST be computed on the same
    symbols the allocator sees.
    """
    res = UniwardWeightedAllocator(
        [t.raw for t in tensors],
        joint_encoder=_make_joint_encoder(tensors),
    ).bisect_for_rms_target(
        curves, rms_target, max_iter=max_iter, lam_hi=1e15
    )
    return {
        "lambda": res.lam,
        "rel_err": res.rel_err,
        "rms_rel_err": res.rel_err,
        "archive_bytes": int(res.total_bytes),
        **res.joint_extras,
    }


# ---------------------------------------------------------------------------
# Top-level run
# ---------------------------------------------------------------------------


def run_experiment(
    state_dict_path: Path,
    shrink_ratios: list[float],
    rms_targets: list[float],
) -> dict:
    """Run arch_shrink × Lagrangian × UNIWARD sweep over (ratio, rms_target)."""
    import torch

    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        raise SystemExit(f"state_dict at {state_dict_path} is not a dict")

    # Per-shrink substrate cache (ratio → ShrinkSubstrate + curves + variances).
    rows: list[dict] = []

    for r in sorted(set(shrink_ratios)):
        substrate = apply_arch_shrink(sd, r)
        n_tensors = len(substrate.tensors)

        # Baseline: lossless (K=1 everywhere) on the SHRUNK substrate.
        baseline = encode_with_per_tensor_K(
            substrate.tensors, [1] * n_tensors
        )
        baseline_bytes = int(baseline["archive_bytes"])

        # Per-tensor K curves (canonical primitive).
        curves = precompute_per_tensor_K_curves(
            substrate.tensors, K_range=K_RANGE
        )

        # UNIWARD variance proxy on shrunk symbols (canonical primitive).
        variances = compute_local_variance_proxy(
            [t.raw for t in substrate.tensors]
        )
        weights = compute_uniward_weights(variances)
        weight_ratio = (
            max(weights) / min(weights) if min(weights) > 0 else float("inf")
        )

        for rms_t in rms_targets:
            uniform_res = run_uniform_lagrangian(
                substrate.tensors, curves, rms_t
            )
            uniward_res = run_uniward_lagrangian(
                substrate.tensors, curves, rms_t
            )

            uniward_savings_vs_uniform = (
                uniform_res["archive_bytes"]
                - uniward_res["archive_bytes"]
            )
            stack_savings_vs_lossless = (
                baseline_bytes - uniward_res["archive_bytes"]
            )

            rows.append(
                {
                    "shrink_ratio": float(r),
                    "rms_target": float(rms_t),
                    "n_tensors": int(n_tensors),
                    "n_elements_orig": int(substrate.n_elements_orig),
                    "n_elements_kept": int(substrate.n_elements_kept),
                    "fraction_kept": float(substrate.fraction_kept),
                    "shrunk_lossless_bytes": baseline_bytes,
                    "variance_min": float(min(variances)),
                    "variance_max": float(max(variances)),
                    "weight_ratio_max_min": float(weight_ratio),
                    "lagrangian_uniform": {
                        "archive_bytes": int(uniform_res["archive_bytes"]),
                        "rel_err": float(uniform_res["rel_err"]),
                        "lambda": float(uniform_res["lambda"]),
                        "Ks": list(uniform_res.get("Ks", [])),
                    },
                    "lagrangian_uniward": {
                        "archive_bytes": int(uniward_res["archive_bytes"]),
                        "rel_err": float(uniward_res["rel_err"]),
                        "lambda": float(uniward_res["lambda"]),
                        "Ks": list(uniward_res.get("Ks", [])),
                    },
                    "uniward_savings_vs_uniform_bytes": int(
                        uniward_savings_vs_uniform
                    ),
                    "stack_savings_vs_lossless_shrunk_bytes": int(
                        stack_savings_vs_lossless
                    ),
                    **proxy_evidence_contract(
                        cuda_eval_worth_testing=False
                    ),
                }
            )

    # Annotate "vs full substrate lossless" if r=1.0 swept.
    full_lossless_bytes = None
    for row in rows:
        if abs(row["shrink_ratio"] - 1.0) < 1e-9:
            # Pick the smallest rms_target to use the most-conservative
            # reference (closest to lossless even for the K curve path).
            if (
                full_lossless_bytes is None
                or row["shrunk_lossless_bytes"] < full_lossless_bytes
            ):
                full_lossless_bytes = row["shrunk_lossless_bytes"]
    if full_lossless_bytes is not None:
        for row in rows:
            row["stack_savings_vs_full_substrate_lossless_bytes"] = (
                int(full_lossless_bytes - row["lagrangian_uniward"]["archive_bytes"])
            )

    # Best stacked row (smallest archive bytes from UNIWARD allocator).
    best_row = min(
        rows,
        key=lambda r: r["lagrangian_uniward"]["archive_bytes"],
    )

    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        **proxy_evidence_contract(),
        "input_state_dict": str(state_dict_path),
        "shrink_ratios_swept": sorted(set(shrink_ratios)),
        "rms_targets_swept": rms_targets,
        "K_range": [K_RANGE[0], K_RANGE[-1]],
        "n_rows": len(rows),
        "rows": rows,
        "best_archive_bytes": int(best_row["lagrangian_uniward"]["archive_bytes"]),
        "best_shrink_ratio": float(best_row["shrink_ratio"]),
        "best_rms_target": float(best_row["rms_target"]),
        "headline": (
            "arch_shrink × Lagrangian × UNIWARD pre-stage. Best UNIWARD-allocator "
            f"archive: {best_row['lagrangian_uniward']['archive_bytes']:,} B at "
            f"shrink_ratio={best_row['shrink_ratio']}, "
            f"rms_target={best_row['rms_target']} "
            "[CPU-prep proxy; SCORE unknown without retrained arch_shrink + CUDA eval]"
        ),
        "reactivation_criteria_remaining": list(REACTIVATION_CRITERIA_REMAINING),
        "composition_provenance": {
            "arch_shrink_primitive": (
                "tools/pr101_arch_shrink_post_hoc_sweep.py:truncate_to_top_channels"
            ),
            "quantizer_primitive": (
                "tac.pr101_split_brotli_codec._quantize_tensor"
            ),
            "K_curve_primitive": (
                "tac.codec.cost_curves.precompute_per_tensor_K_curves"
            ),
            "lagrangian_primitive": (
                "tac.optimization.lagrangian_per_tensor_allocation.LagrangianPerTensorAllocator"
            ),
            "uniward_primitive": (
                "tac.optimization.lagrangian_per_tensor_allocation.UniwardWeightedAllocator"
            ),
            "joint_encoder_primitive": (
                "tools/pr101_lossy_coarsening_analytical.py:encode_with_per_tensor_K"
            ),
            "codec_pipeline_canonical": (
                "tac.codec_pipeline.CodecPipeline (CPL2; substrate-transform "
                "arch_shrink CodecOp pending ORCH-SYNC)"
            ),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--state-dict",
        type=Path,
        default=REPO_ROOT
        / "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt",
        help=(
            "Input PR101 fp32 state_dict (default: PR101 88K-element "
            "Quantizr-class output, the substrate matching arch_shrink "
            "Lightning T4 dispatch)."
        ),
    )
    p.add_argument(
        "--shrink-ratios",
        type=float,
        nargs="+",
        default=[0.4, 0.6, 0.8, 1.0],
        help="Channel-keep ratios to sweep.",
    )
    p.add_argument(
        "--rms-targets",
        type=float,
        nargs="+",
        default=DEFAULT_RMS_TARGETS,
        help="rel_err RMS targets for Lagrangian λ-bisection.",
    )
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-evidence", type=Path, default=None)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: only sweep shrink=1.0 and rms_target=0.05 for speed.",
    )
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict}")

    if args.smoke:
        args.shrink_ratios = [1.0]
        args.rms_targets = [0.05]

    print(
        "[arch_shrink × Lagrangian × UNIWARD pre-stage] "
        f"state_dict={args.state_dict}"
    )
    print(
        f"  shrink_ratios={args.shrink_ratios} rms_targets={args.rms_targets}"
    )
    manifest = run_experiment(
        args.state_dict, args.shrink_ratios, args.rms_targets
    )

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_json is None:
        out_dir = (
            REPO_ROOT
            / f"reports/raw/pr101_arch_shrink_x_lagrangian_x_uniward_{ts}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_json = out_dir / "manifest.json"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nmanifest: {args.output_json}\n")
    print(
        f"  {'ratio':>5} | {'rms':>7} | {'lossless':>10} | "
        f"{'uniform':>10} {'uni_λ':>10} | "
        f"{'uniward':>10} {'uni_λ':>10} | {'Δ_vs_unif':>10}"
    )
    for row in sorted(
        manifest["rows"],
        key=lambda r: (r["shrink_ratio"], r["rms_target"]),
    ):
        u = row["lagrangian_uniform"]
        w = row["lagrangian_uniward"]
        print(
            f"  {row['shrink_ratio']:>5.2f} | {row['rms_target']:>7.4f} | "
            f"{row['shrunk_lossless_bytes']:>10,} | "
            f"{u['archive_bytes']:>10,} {u['lambda']:>10.2e} | "
            f"{w['archive_bytes']:>10,} {w['lambda']:>10.2e} | "
            f"{row['uniward_savings_vs_uniform_bytes']:>+10,}"
        )
    print(f"\n  {manifest['headline']}")

    if args.output_evidence:
        # Pick the canonical anchor row: shrink=0.4 if present, else best.
        anchor_row = next(
            (r for r in manifest["rows"] if abs(r["shrink_ratio"] - 0.4) < 1e-9),
            None,
        )
        if anchor_row is None:
            anchor_row = min(
                manifest["rows"],
                key=lambda r: r["lagrangian_uniward"]["archive_bytes"],
            )

        evidence_row = {
            "technique": "arch_shrink_x_lagrangian_x_uniward_post_hoc",
            "empirical_archive_bytes": int(
                anchor_row["lagrangian_uniward"]["archive_bytes"]
            ),
            **proxy_evidence_contract(),
            "shrink_ratio": float(anchor_row["shrink_ratio"]),
            "rms_target": float(anchor_row["rms_target"]),
            "fraction_kept": float(anchor_row["fraction_kept"]),
            "uniward_savings_vs_uniform_bytes": int(
                anchor_row["uniward_savings_vs_uniform_bytes"]
            ),
            "shrunk_lossless_bytes": int(anchor_row["shrunk_lossless_bytes"]),
            "source": (
                f"{EVIDENCE_GRADE} {args.output_json} "
                f"(arch_shrink ratio={anchor_row['shrink_ratio']} "
                f"+ Lagrangian × UNIWARD rms={anchor_row['rms_target']}; "
                f"uniward_archive={anchor_row['lagrangian_uniward']['archive_bytes']:,} B; "
                f"vs_uniform={anchor_row['uniward_savings_vs_uniform_bytes']:+,} B)"
            ),
            "contest_dispatch_verdict": "DEFERRED-pending-research",
            "supersedes_prior_FALSIFIED_tag": False,
            "reactivation_criteria_tested": [
                "post_hoc_arch_shrink_then_lagrangian_uniward_byte_compounding"
            ],
            "reactivation_criteria_remaining": list(
                REACTIVATION_CRITERIA_REMAINING
            ),
            "timestamp": _dt.datetime.now(_dt.UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        }
        args.output_evidence.parent.mkdir(parents=True, exist_ok=True)
        with args.output_evidence.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evidence_row) + "\n")
        print(f"\nevidence row appended: {args.output_evidence}")

    print(
        "\nNOTE: composed-stack pre-stage byte anchor only. SCORE impact "
        "of post-hoc arch truncation is unknown without retraining; "
        "the arch_shrink Lightning T4 dispatch in flight will produce the "
        "first contest-CUDA architecture anchor. Catalog row "
        "arch_shrink_x_lagrangian_x_uniward remains DEFERRED-pending-research."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
