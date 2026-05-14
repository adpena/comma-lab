# SPDX-License-Identifier: MIT
"""Z3 × C6 composition probe-disambiguator (T1-F).

Per CLAUDE.md "Anti-arbitrariness primitive: the probe-disambiguator pattern" +
the Grand Council Tier-1 dispatch authorization
(`.omx/research/grand_council_tier_dispatch_authorizations_20260514.md` T1-F):

When a design choice has 2+ defensible interpretations, ship BOTH modes via
callable interface + build a probe that returns the regime-conditional
verdict. THIS probe arbitrates the **Z3 × C6 stacking question**:

    Z3 alone     reduces archive bytes via Ballé hyperprior amortization
                  on A1's latent_blob → ΔS_Z3 = -25 · Δb_Z3 / N
    C6 alone     substrate-class shift to MDL-IBPS latent grammar
                  → ΔS_C6 = -25 · Δb_C6 / N + Δd_C6
    Z3 + C6      hyperprior amortizing IBPS1 latent
                  → ΔS_stacked = -25 · (additivity_factor) · (Δb_Z3 + Δb_C6) / N

The relationship has THREE defensible interpretations per Ballé 2018 §IV.A
sub-additivity factor on saturated substrates + Atick-Redlich 1990
cooperative-receiver framing:

  - **Optimistic (additive)**: -25·(Δb_Z3 + Δb_C6)/N   (α = 1.0)
  - **Realistic (sub-additive)**: -25·α·(Δb_Z3 + Δb_C6)/N where α ∈ [0.3, 0.9]
  - **Pessimistic (saturating)**: -25·max(Δb_Z3, Δb_C6)/N   (α ≤ 0.3)

The empirical α is what THIS probe estimates from a synthetic CPU-only
forward pass on the two frozen substrate archives.

Math model
==========

Let:
  B_A1     = baseline A1 archive bytes (frozen public anchor)
  B_C6     = C6 IBPS1 archive bytes
  B_Z3     = A1 + Z3HP1 sidecar archive bytes
  B_stack  = C6 IBPS1 + Z3HP1 sidecar archive bytes

Per-substrate savings:
  Δb_Z3   = B_A1 - B_Z3   (positive if Z3 saves bytes vs A1)
  Δb_C6   = B_A1 - B_C6   (positive if C6 saves bytes vs A1; can be negative
                            because C6 is a class-shift, not byte-shrink)
  Δb_stack = B_A1 - B_stack

Composition factor α:
  α = (Δb_Z3 + Δb_C6 - Δb_stack) / max(|Δb_Z3|, |Δb_C6|)
      [absolute values to handle sign-flipping baselines]

Verdict bands (council-pinned):
  α >  0.7  → ADDITIVE      — stacking justified; dispatch A1×Z3×C6 cell
  0.3 < α ≤ 0.7 → SUB-ADDITIVE — marginal; pick single best substrate
  α ≤ 0.3  → SATURATING    — dominated; DEFER stacking per
                              CLAUDE.md "KILL is LAST RESORT"

WAIT: that's the SAVINGS-COMPOSITION direction. Our probe asks instead:
**how much does stacking ADD on top of the per-substrate savings?**

Concretely:
  realized_stack_savings = Δb_stack
  predicted_additive    = Δb_Z3 + Δb_C6
  α = realized_stack_savings / predicted_additive  (when predicted_additive > 0)

So:
  α >  0.7  → ADDITIVE         — realized ≥ 70% of predicted-additive
  0.3 < α ≤ 0.7 → SUB-ADDITIVE — partially-additive; one path saturates the other
  α ≤ 0.3  → SATURATING        — stacking dominated by the single best move

For the **degenerate** case where Δb_Z3 + Δb_C6 ≤ 0 (i.e. C6 ADDS bytes
because class-shift trades bytes for distortion), the probe reports
`predicted_additive_le_zero` as the verdict and lets the operator decide
based on the per-substrate distortion-deltas instead.

Provenance
==========

This probe is CPU-only synthetic. It uses:
  - submissions/a1/archive.zip — frozen public A1 anchor
  - The Z3 substrate's `total_balle_rate_bits` estimator on representative
    latent statistics OR a real Z3HP1 sidecar from a smoke artifact
  - The C6 IBPS1 archive byte count (from a real C6 smoke archive OR a
    synthetic IBPS1 with representative latent sizes)
  - The composition byte count via `(Z3HP1 sidecar) + (C6 IBPS1 latent_blob)`
    OR a real stacked artifact

Output:
  JSON to stdout (or --output) with the verdict + α + all four byte sizes.

NO score claim. NO promotion. NO exact-eval dispatch.
Tagged `research_only=true`. Pure CPU; $0 GPU.

Citations
=========

- Ballé et al. 2018 ICLR "Variational Image Compression with a Scale
  Hyperprior" — §IV.A sub-additivity factor on saturated substrates.
- Atick & Redlich 1990 "Towards a Theory of Early Visual Processing" —
  cooperative-receiver framing; substrate-conditional H(payload | scorer).
- MacKay 2003 "Information Theory, Inference, and Learning Algorithms" —
  §6.7 conditional Gaussian rate-distortion.
- Rissanen 1978 "Modeling by Shortest Data Description" — MDL framework.

Hook wire-in (Catalog #125, the 6 mandatory hooks per CLAUDE.md
"Subagent coherence-by-default"):

1. Sensitivity-map: α IS the per-substrate-pair composition sensitivity signal.
2. Pareto constraint: composition feasibility = intersection of Z3 + C6
   feasibility regions (Pareto-Z3 ∩ Pareto-C6 ⊂ Pareto-stacked).
3. Bit-allocator: N/A — this is the disambiguator, not the allocator.
4. Cathedral autopilot dispatch hook: YES — α verdict feeds the v2 ranker
   (Catalog #219) by updating stacking candidates' predicted ΔS.
5. Continual-learning posterior: YES — every probe run is an empirical
   anchor for the composition matrix posterior at
   `.omx/state/substrate_composition_matrix.json`.
6. Probe-disambiguator: THIS IS the probe-disambiguator (hook #6 canonical).
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError as e:  # pragma: no cover — torch is a hard dep
    raise RuntimeError(
        "torch is required for the Z3xC6 composition probe-disambiguator. "
        "This module exercises the Z3 hyperprior architecture on CPU."
    ) from e

from tac.substrates.z3_balle_hyperprior_bolton.architecture import (
    A1_LATENT_DIM,
    A1_N_PAIRS,
    Z3HyperpriorConfig,
    Z3HyperpriorMLP,
    total_balle_rate_bits,
)
from tac.substrates.z3_balle_hyperprior_bolton.archive import (
    Z3HP1_HEADER_STRUCT,
    Z3HP1_MAGIC,
    decode_z3hp1_sidecar,
    encode_z3hp1_sidecar,
    quantize_int8_with_scale,
)


# Council-pinned verdict thresholds.
ALPHA_ADDITIVE_THRESHOLD: float = 0.7
ALPHA_SATURATING_THRESHOLD: float = 0.3

# Default A1 baseline (frozen public anchor; refusable via CLI).
A1_BASELINE_ARCHIVE = Path("submissions/a1/archive.zip")
A1_INNER_MEMBER_NAME = "x"

# Default canonical composition matrix path.
DEFAULT_COMPOSITION_MATRIX_PATH = Path(".omx/state/substrate_composition_matrix.json")


@dataclass(frozen=True)
class ProbeInputs:
    """Inputs to the Z3 × C6 composition probe.

    Each archive size is in BYTES. Score-axis tags + custody metadata
    are intentionally optional here because the probe is research-only;
    promotion is gated by the canonical custody validators downstream.
    """

    base_a1_archive_bytes: int
    """B_A1: frozen A1 public anchor archive bytes."""

    z3_only_archive_bytes: int
    """B_Z3: A1 + Z3HP1 sidecar archive bytes (or projected estimate)."""

    c6_only_archive_bytes: int
    """B_C6: C6 IBPS1 archive bytes."""

    stacked_archive_bytes: int
    """B_stack: C6 IBPS1 + Z3HP1 sidecar archive bytes (or estimate)."""

    base_source: str = "unspecified"
    z3_source: str = "unspecified"
    c6_source: str = "unspecified"
    stacked_source: str = "unspecified"


@dataclass(frozen=True)
class ProbeVerdict:
    """Output verdict from the Z3 × C6 composition probe."""

    alpha: float
    """Composition additivity factor (see module docstring)."""

    verdict: str
    """One of: ``additive`` / ``sub_additive`` / ``saturating`` /
    ``predicted_additive_le_zero`` / ``degenerate_both_negative``."""

    delta_b_z3: int
    """Δb_Z3 = B_A1 - B_Z3 (positive if Z3 saves bytes)."""

    delta_b_c6: int
    """Δb_C6 = B_A1 - B_C6 (positive if C6 saves bytes)."""

    delta_b_stack: int
    """Δb_stack = B_A1 - B_stack (positive if stack saves bytes)."""

    predicted_additive_savings: int
    """Δb_Z3 + Δb_C6 (the optimistic upper bound)."""

    realized_savings: int
    """Δb_stack (the actual stack savings)."""

    predicted_delta_s_additive: float
    """Predicted ΔS under additive composition = -25 · predicted_additive / N."""

    predicted_delta_s_realized: float
    """Predicted ΔS at realized α = -25 · realized / N."""

    score_claim: bool = False
    """Always False — probe is research_only."""

    promotion_eligible: bool = False
    """Always False — probe is research_only."""

    ready_for_exact_eval_dispatch: bool = False
    """Always False — probe is synthetic CPU."""

    result_review_blockers: tuple[str, ...] = ()
    """Fail-closed blockers per CLAUDE.md custody discipline."""

    inputs: ProbeInputs = field(
        default_factory=lambda: ProbeInputs(0, 0, 0, 0)
    )

    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-safe manifest."""
        d = asdict(self)
        d["inputs"] = asdict(self.inputs)
        d["result_review_blockers"] = list(self.result_review_blockers)
        return d


def compute_alpha(
    *,
    delta_b_z3: int,
    delta_b_c6: int,
    delta_b_stack: int,
) -> tuple[float, str]:
    """Compute the composition factor α and the verdict band.

    Returns ``(alpha, verdict_str)`` where verdict_str is one of:
      ``additive`` / ``sub_additive`` / ``saturating`` /
      ``predicted_additive_le_zero`` / ``degenerate_both_negative``.

    For predicted_additive ≤ 0 we report a special verdict because α
    becomes ill-defined (we'd divide by zero or get the wrong sign).
    """
    predicted_additive = delta_b_z3 + delta_b_c6
    realized = delta_b_stack
    # Special-case degenerate regimes first.
    if delta_b_z3 < 0 and delta_b_c6 < 0:
        # Both substrates ADD bytes (class-shift trade); α is ill-defined
        # because there's no savings to share.
        return float("nan"), "degenerate_both_negative"
    if predicted_additive <= 0:
        # E.g. Z3 saves 1 KB but C6 adds 5 KB → predicted_additive < 0.
        # The "stacked saves more than the sum" framing breaks; punt
        # the verdict to the operator.
        return float("nan"), "predicted_additive_le_zero"
    alpha = realized / predicted_additive
    if alpha > ALPHA_ADDITIVE_THRESHOLD:
        verdict = "additive"
    elif alpha > ALPHA_SATURATING_THRESHOLD:
        verdict = "sub_additive"
    else:
        verdict = "saturating"
    return float(alpha), verdict


def run_probe(inputs: ProbeInputs, *, n_pairs: int = A1_N_PAIRS) -> ProbeVerdict:
    """Run the Z3 × C6 composition probe-disambiguator on byte-size inputs.

    Pure function; no I/O. Tests can drive this directly.
    """
    delta_b_z3 = inputs.base_a1_archive_bytes - inputs.z3_only_archive_bytes
    delta_b_c6 = inputs.base_a1_archive_bytes - inputs.c6_only_archive_bytes
    delta_b_stack = inputs.base_a1_archive_bytes - inputs.stacked_archive_bytes
    alpha, verdict = compute_alpha(
        delta_b_z3=delta_b_z3,
        delta_b_c6=delta_b_c6,
        delta_b_stack=delta_b_stack,
    )
    # Predicted ΔS uses the canonical -25 · Δb / N upstream evaluator rate
    # term where N = frames_in_video (37545489 bytes per ~600 pairs is the
    # contest normalizer). We just use the magnitude for the prediction.
    # Contest formula: rate = 25 * archive_bytes / video_bytes
    # So ΔS_rate = -25 · Δb / N.
    # For research-only probe the constant `25 / N` is folded into a
    # single coefficient; we report the raw byte deltas so the caller
    # can multiply by their own N if they want a ΔS estimate.
    # N = 37545489 (the canonical contest video bytes).
    N_contest = 37_545_489
    predicted_delta_s_additive = -25.0 * (delta_b_z3 + delta_b_c6) / N_contest
    predicted_delta_s_realized = -25.0 * delta_b_stack / N_contest

    blockers: list[str] = []
    if verdict in ("predicted_additive_le_zero", "degenerate_both_negative"):
        blockers.append(
            "composition_additivity_indeterminate_inputs_yield_no_predicted_savings"
        )
    if inputs.base_a1_archive_bytes <= 0:
        blockers.append("invalid_base_a1_archive_bytes_must_be_positive")
    if any(
        b <= 0
        for b in (
            inputs.z3_only_archive_bytes,
            inputs.c6_only_archive_bytes,
            inputs.stacked_archive_bytes,
        )
    ):
        blockers.append("invalid_archive_bytes_must_be_positive")
    # Surface the additive-estimator-by-construction caveat: if the stacked
    # source string declares "additive_estimate", the α=1.0 verdict is
    # STRUCTURAL not EMPIRICAL — operator must train a real stacked
    # substrate to observe the realistic α.
    if "additive_estimate" in inputs.stacked_source:
        blockers.append(
            "alpha_is_structural_not_empirical_real_stacked_archive_required_for_empirical_alpha"
        )
    # Surface the smoke-num-pairs caveat: if C6 archive bytes < 50KB it's
    # likely a reduced num_pairs smoke artifact whose savings are not
    # representative of the full 600-pair production substrate.
    if 0 < inputs.c6_only_archive_bytes < 50_000:
        blockers.append(
            "c6_archive_bytes_below_50kb_likely_reduced_smoke_num_pairs_not_production"
        )
    # Custody-discipline: probe is synthetic / research-only.
    blockers.append("probe_is_research_only_not_a_score_claim")
    blockers.append(
        "probe_is_cpu_synthetic_not_contest_cuda_or_contest_cpu_authority"
    )

    return ProbeVerdict(
        alpha=alpha,
        verdict=verdict,
        delta_b_z3=delta_b_z3,
        delta_b_c6=delta_b_c6,
        delta_b_stack=delta_b_stack,
        predicted_additive_savings=delta_b_z3 + delta_b_c6,
        realized_savings=delta_b_stack,
        predicted_delta_s_additive=float(predicted_delta_s_additive),
        predicted_delta_s_realized=float(predicted_delta_s_realized),
        inputs=inputs,
        result_review_blockers=tuple(blockers),
        notes=f"alpha={alpha:.4f}; verdict={verdict}; n_pairs={n_pairs}",
    )


def synthesize_z3_sidecar_bytes(
    *,
    seed: int = 0,
    hyper_dim: int = 8,
    latent_dim: int = A1_LATENT_DIM,
    n_pairs: int = A1_N_PAIRS,
) -> int:
    """Synthesize a representative Z3HP1 sidecar and return its size in bytes.

    Builds a Z3HyperpriorMLP with the canonical Z3HyperpriorConfig, feeds
    a deterministic synthetic A1-shaped latent tensor through it, quantizes
    weights + hyper-latents to int8, and packs the sidecar via the canonical
    ``encode_z3hp1_sidecar`` API. Returns the byte size of the packed sidecar.

    NOTE: this is a SYNTHETIC reproduction. The real sidecar size depends
    on the actual A1 latent statistics and the trained hyperprior. For the
    probe we want a representative ESTIMATE; the empirical α is computed
    against this estimate so callers should pass real sidecar bytes when
    they're available.

    Pure CPU. Deterministic for a given seed.
    """
    torch.manual_seed(seed)
    cfg = Z3HyperpriorConfig(hyper_latent_dim=hyper_dim)
    mlp = Z3HyperpriorMLP(cfg)
    mlp.eval()

    # Synthesize representative A1-shaped latents.
    # Use heavy-tailed N(0, 4) — A1 latents are approximately Gaussian
    # post-quantization-aware training.
    y = torch.randn(n_pairs, latent_dim) * 4.0

    with torch.no_grad():
        sigma, w_hat = mlp(y, quantize=True)
        # Residual: y minus its conditional mean (here 0 under Gaussian prior).
        residual = (y / cfg.quantization_step).round().to(torch.int8)
        residual_clipped = residual.clamp(-128, 127)
        # w_hat: round + clip to int8.
        w_hat_int8 = w_hat.round().clamp(-128, 127).to(torch.int8)

    # Quantize all MLP weights to int8.
    weights_blobs: list[bytes] = []
    scale_values: list[float] = []
    for p in mlp.parameters():
        blob, scale = quantize_int8_with_scale(
            p, scale_clip_range=cfg.int8_scale_clip
        )
        weights_blobs.append(blob)
        scale_values.append(scale)
    weights_int8 = b"".join(weights_blobs)
    # Use the maximum scale as the canonical sidecar scale.
    overall_scale = float(max(scale_values))

    sidecar_bytes = encode_z3hp1_sidecar(
        hyperprior_weights_int8=weights_int8,
        w_hat_int8=w_hat_int8.cpu().numpy().tobytes(),
        residual_int8=residual_clipped.cpu().numpy().tobytes(),
        hyper_dim=hyper_dim,
        int8_w_scale=overall_scale,
        quant_step=float(cfg.quantization_step),
        min_sigma=float(cfg.min_sigma),
        max_sigma=float(cfg.max_sigma),
        n_pairs=n_pairs,
        latent_dim=latent_dim,
    )
    return len(sidecar_bytes)


def read_a1_baseline_bytes(path: Path = A1_BASELINE_ARCHIVE) -> int:
    """Read the A1 baseline archive size from the canonical public anchor.

    Returns the OUTER archive.zip byte size (the contest scorer sees this
    via the ZIP file size).
    """
    if not path.exists():
        raise FileNotFoundError(f"A1 baseline archive not found: {path}")
    return path.stat().st_size


def read_c6_archive_bytes(path: Path) -> int:
    """Read a C6 IBPS1 archive size from disk.

    Accepts either a raw 0.bin (IBPS1 monolithic) OR a zip containing it.
    """
    if not path.exists():
        raise FileNotFoundError(f"C6 archive not found: {path}")
    if path.suffix == ".zip":
        return path.stat().st_size
    # 0.bin — validate magic + return size.
    data = path.read_bytes()
    if data[:4] != b"IBPS":
        raise ValueError(
            f"C6 archive {path} does not start with IBPS magic; got {data[:4]!r}"
        )
    return len(data)


def estimate_stacked_archive_bytes(
    *,
    c6_archive_bytes: int,
    z3_sidecar_bytes: int,
) -> int:
    """Estimate stacked archive bytes (C6 IBPS1 + Z3HP1 sidecar appended).

    This is a FIRST-ORDER estimate: in the realistic stacked layout the
    Z3 sidecar replaces a portion of the C6 IBPS1 latent_blob (if Z3's
    learned hyperprior amortizes the C6 latents better than C6's native
    factorized prior). The naive estimate is additive (worst-case for
    stacking), so we use:

      B_stack ≈ B_C6 + B_Z3HP1_sidecar

    Real stacked archives can be smaller if the Z3 sidecar genuinely
    replaces C6's latent_blob — the empirical α captures this.
    """
    return c6_archive_bytes + z3_sidecar_bytes


def estimate_z3_archive_bytes(
    *,
    a1_archive_bytes: int,
    z3_sidecar_bytes: int,
) -> int:
    """Estimate Z3-only archive bytes (A1 + Z3HP1 sidecar appended)."""
    return a1_archive_bytes + z3_sidecar_bytes


def append_composition_matrix_entry(
    verdict: ProbeVerdict,
    *,
    matrix_path: Path = DEFAULT_COMPOSITION_MATRIX_PATH,
) -> None:
    """Append the probe's verdict to the canonical composition matrix.

    The matrix file is a JSON object keyed by (substrate_a, substrate_b)
    tuple-as-string. Each entry records the α + verdict + timestamp.
    This is the continual-learning posterior surface for stacking
    candidates per Catalog #125 hook #5.
    """
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    if matrix_path.exists():
        try:
            matrix = json.loads(matrix_path.read_text())
        except json.JSONDecodeError:
            matrix = {}
    else:
        matrix = {}
    key = "z3_balle_hyperprior_bolton__x__c6_e4_mdl_ibps"
    if "entries" not in matrix:
        matrix["entries"] = {}
    if key not in matrix["entries"]:
        matrix["entries"][key] = []
    matrix["entries"][key].append(
        {
            "alpha": verdict.alpha if not math.isnan(verdict.alpha) else None,
            "verdict": verdict.verdict,
            "delta_b_z3": verdict.delta_b_z3,
            "delta_b_c6": verdict.delta_b_c6,
            "delta_b_stack": verdict.delta_b_stack,
            "predicted_additive_savings": verdict.predicted_additive_savings,
            "realized_savings": verdict.realized_savings,
            "predicted_delta_s_additive": verdict.predicted_delta_s_additive,
            "predicted_delta_s_realized": verdict.predicted_delta_s_realized,
            "score_claim": verdict.score_claim,
            "promotion_eligible": verdict.promotion_eligible,
            "ready_for_exact_eval_dispatch": verdict.ready_for_exact_eval_dispatch,
            "result_review_blockers": list(verdict.result_review_blockers),
            "base_source": verdict.inputs.base_source,
            "z3_source": verdict.inputs.z3_source,
            "c6_source": verdict.inputs.c6_source,
            "stacked_source": verdict.inputs.stacked_source,
            "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    matrix["schema_version"] = 1
    matrix["last_updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    matrix_path.write_text(json.dumps(matrix, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Z3 × C6 composition probe-disambiguator. Estimates the additivity "
            "factor α for stacking the Z3 Ballé hyperprior bolt-on with the "
            "C6 MDL-IBPS substrate. CPU-only; $0; research_only."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--a1-archive",
        type=Path,
        default=A1_BASELINE_ARCHIVE,
        help="Path to A1 baseline archive.zip (default: submissions/a1/archive.zip)",
    )
    parser.add_argument(
        "--c6-archive",
        type=Path,
        default=None,
        help="Path to C6 IBPS1 archive (0.bin OR archive.zip). If not provided, "
        "a synthetic estimate is used.",
    )
    parser.add_argument(
        "--z3-sidecar-bytes",
        type=int,
        default=None,
        help="Override the Z3HP1 sidecar byte size with a real value. "
        "If not provided, a synthetic estimate is computed from the canonical "
        "Z3HyperpriorMLP.",
    )
    parser.add_argument(
        "--stacked-archive",
        type=Path,
        default=None,
        help="Path to a real stacked (C6 + Z3) archive bytes. If not provided, "
        "an additive estimate (B_C6 + B_Z3HP1_sidecar) is used.",
    )
    parser.add_argument(
        "--c6-archive-bytes-override",
        type=int,
        default=None,
        help="Override C6 archive byte size (skip reading any file).",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=A1_N_PAIRS,
        help=f"Number of A1 pairs in the synthesis (default: {A1_N_PAIRS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic seed for the synthetic Z3 sidecar reproduction.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path (default: stdout).",
    )
    parser.add_argument(
        "--update-composition-matrix",
        action="store_true",
        help="Append the probe verdict to .omx/state/substrate_composition_matrix.json",
    )
    parser.add_argument(
        "--composition-matrix-path",
        type=Path,
        default=DEFAULT_COMPOSITION_MATRIX_PATH,
        help=f"Composition matrix JSON path (default: {DEFAULT_COMPOSITION_MATRIX_PATH})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # 1. Read A1 baseline.
    a1_bytes = read_a1_baseline_bytes(args.a1_archive)
    base_source = str(args.a1_archive)

    # 2. Compute / read Z3HP1 sidecar bytes.
    if args.z3_sidecar_bytes is not None:
        z3_sidecar_bytes = int(args.z3_sidecar_bytes)
        z3_source = f"override_user_z3_sidecar_bytes={z3_sidecar_bytes}"
    else:
        z3_sidecar_bytes = synthesize_z3_sidecar_bytes(
            seed=args.seed, n_pairs=args.n_pairs
        )
        z3_source = (
            f"synthetic_z3hp1_sidecar_seed={args.seed}_n_pairs={args.n_pairs}"
        )
    z3_only_bytes = estimate_z3_archive_bytes(
        a1_archive_bytes=a1_bytes, z3_sidecar_bytes=z3_sidecar_bytes
    )

    # 3. Read / estimate C6 archive bytes.
    if args.c6_archive_bytes_override is not None:
        c6_bytes = int(args.c6_archive_bytes_override)
        c6_source = f"override_user_c6_archive_bytes={c6_bytes}"
    elif args.c6_archive is not None:
        c6_bytes = read_c6_archive_bytes(args.c6_archive)
        c6_source = str(args.c6_archive)
    else:
        # Synthetic C6 estimate: per the C6 smoke logs at full N_pairs=600,
        # decoder ~33075B + latent ~600*8B=4800B + encoder ~61000B + meta ~500B
        # ≈ 99,000 bytes. We use the canonical estimate.
        c6_bytes = 99_000  # smoke-anchor estimate; can be overridden.
        c6_source = "synthetic_smoke_anchor_estimate_99000_bytes"

    # 4. Compute / read stacked archive bytes.
    if args.stacked_archive is not None:
        stacked_bytes = args.stacked_archive.stat().st_size
        stacked_source = str(args.stacked_archive)
    else:
        stacked_bytes = estimate_stacked_archive_bytes(
            c6_archive_bytes=c6_bytes, z3_sidecar_bytes=z3_sidecar_bytes
        )
        stacked_source = (
            f"additive_estimate_c6={c6_bytes}+z3_sidecar={z3_sidecar_bytes}"
        )

    inputs = ProbeInputs(
        base_a1_archive_bytes=a1_bytes,
        z3_only_archive_bytes=z3_only_bytes,
        c6_only_archive_bytes=c6_bytes,
        stacked_archive_bytes=stacked_bytes,
        base_source=base_source,
        z3_source=z3_source,
        c6_source=c6_source,
        stacked_source=stacked_source,
    )

    verdict = run_probe(inputs, n_pairs=args.n_pairs)

    # 5. Emit verdict.
    output_json = json.dumps(verdict.as_dict(), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_json)
    else:
        print(output_json)

    # 6. Optionally append to composition matrix.
    if args.update_composition_matrix:
        append_composition_matrix_entry(
            verdict, matrix_path=args.composition_matrix_path
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
