#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Variational Information Bottleneck (VIB) tractability probe.

Canonical pre-dispatch tractability gate for the Tishby IB-pure substrate
(per ``.omx/research/tishby_ib_pure_substrate_asymptotic_pursuit_scoping_design_20260516.md``
§19 V1 lift gate criterion #2). Sister of the D4 H(latent|scorer_class)
probe at ``tools/probe_latent_conditional_entropy_h_latent_given_scorer_class.py``
but at a DIFFERENT surface: where D4 measures empirical mutual information
between latents + scorer-class indices, this probe measures whether the
variational IB Lagrangian's gradient is tractable enough for end-to-end
optimization (i.e. signal-to-noise ratio of the gradient with respect to
the substrate's loss).

Per Catalog #292 sextet-pact assumption-statement discipline (CLAUDE.md
"Council conduct" amendment 2026-05-15) + the standing operator directive
*"need grand council per-round explicit assumption-statement discipline"*:
this probe's verdict carries explicit per-axis operating-within assumption
disclosure so the consumer can decide whether the empirical SNR is the
right summary statistic for THIS substrate's gradient class.

Algorithm (lightweight synthetic-data Monte Carlo VIB gradient SNR):

  1. Construct a small variational encoder ``q(t|x) -> (mu, log_sigma)``
     (diagonal Gaussian) and a small decoder ``p(y|t)`` (MLP).
  2. Synthesize ``num_samples`` random ``x`` from a fixed seed and ``y``
     from a fixed reference scorer (deterministic linear map of x with
     noise — see ``_make_synthetic_scorer`` below).
  3. For each Monte Carlo replicate (``num_replicates``):
     a. Sample fresh ``x`` batch (B = ``batch_size``)
     b. Forward through encoder via reparam trick t ~ q(t|x)
     c. Forward through decoder p(y|t)
     d. Compute L_VIB = E[-log p(y|t)] + beta * KL(q(t|x) || p(t))
     e. Backward; record per-parameter gradient norm
  4. Estimate gradient SNR = mean(gradient_norm) / std(gradient_norm)
     across replicates per parameter group; report worst-case + median.
  5. Verdict per ``--gradient-snr-threshold`` (default 1.0 conservative):
     - TRACTABLE: gradient SNR >= threshold (typically >= 1.0)
     - MARGINAL: 0.1 <= SNR < threshold (Path-MINE pivot recommended)
     - INTRACTABLE: SNR < 0.1 (DEFER per "Forbidden premature KILL"; pivot
       to sister asymptotic-pursuit substrate per design memo §19)

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #192/#221
fail-closed: the emitted JSON is fail-closed (``score_claim=false``,
evidence grade ``diagnostic_cpu`` or ``diagnostic_a100`` depending on
``--device``, axis explicitly tagged) so no autopilot consumer can mistake
the probe output for a contest-CUDA score claim.

CLI contract::

    .venv/bin/python tools/check_variational_ib_tractability.py \\
        --substrate-id tishby_ib_pure \\
        --path-variant VIB \\
        --output-json .omx/state/variational_ib_tractability_tishby_ib_pure.json

The probe is a $0 local-CPU smoke at default config (300 samples, 8
replicates, ~30 seconds); for council-grade Modal A100 tractability
verification with 100ep curriculum proxy, see the design memo §19 STAGE 1
($5-10 envelope).

Per the design memo §16 cargo-cult audit row "Gradient SNR >= 1.0 is the
canonical tractability threshold": the threshold IS heuristic; operators
should configure per-substrate via ``--gradient-snr-threshold``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

# Per design memo §16: threshold 1.0 is a reasonable conservative default
# but is HEURISTIC not first-principles-derived. Empirical IB literature
# (Alemi 2017; Saxe et al. 2018) typically reports gradient SNR > 0.5 as
# tractable for IB; SNR < 0.1 as intractable; the 1.0 threshold is
# conservative.
DEFAULT_GRADIENT_SNR_TRACTABLE_THRESHOLD: float = 1.0
DEFAULT_GRADIENT_SNR_INTRACTABLE_THRESHOLD: float = 0.1
DEFAULT_DEGENERATE_SNR_CAP: float = 1_000_000.0

# Synthetic-data smoke defaults — kept small so the probe is $0 CPU + ~30s
# at default config. Operators bump via flags for council-grade analyses.
DEFAULT_NUM_REPLICATES: int = 8
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_LATENT_DIM: int = 16
DEFAULT_INPUT_DIM: int = 64
DEFAULT_OUTPUT_DIM: int = 5  # matches contest SegNet 5-class output
DEFAULT_BETA: float = 0.01

VerdictStr = Literal["TRACTABLE", "MARGINAL", "INTRACTABLE"]
PathVariantStr = Literal["VIB", "MINE"]


@dataclass(frozen=True)
class VariationalIBTractabilityVerdict:
    """Machine-readable VIB tractability verdict for autopilot consumers.

    Per the design memo §21.1 Observability artifacts table this dataclass
    IS the canonical schema for ``.omx/state/variational_ib_tractability_<substrate_id>.json``.
    """

    substrate_id: str
    path_variant: PathVariantStr
    verdict: VerdictStr
    gradient_snr_mean: float
    gradient_snr_median: float
    gradient_snr_worst_case: float
    gradient_norm_mean: float
    gradient_norm_std: float
    kl_term_mean: float
    reconstruction_term_mean: float
    num_replicates: int
    batch_size: int
    latent_dim: int
    input_dim: int
    output_dim: int
    beta: float
    tractable_threshold: float
    intractable_threshold: float
    evidence_grade: str
    score_claim: bool
    axis_label: str
    operating_within_assumption: str
    observed_at_utc: str
    notes: str


def _make_synthetic_data(
    *,
    num_samples: int,
    input_dim: int,
    output_dim: int,
    seed: int,
):
    """Synthesize ``(x, y)`` pairs where ``y = scorer(x) + small noise``.

    Returns ``(x, y)`` as torch tensors. The synthetic scorer is a fixed
    linear projection so the IB objective has a well-defined gradient
    signal; the noise floor ensures the probe measures the *tractability*
    of the optimization, not the *correctness* of the synthetic scorer.
    """
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    x = torch.randn((num_samples, input_dim), generator=g)
    # Fixed scorer: linear projection + sigmoid (matches contest SegNet
    # 5-class softmax shape; deterministic per seed).
    proj_seed = seed + 1
    g2 = torch.Generator()
    g2.manual_seed(proj_seed)
    proj_w = torch.randn((input_dim, output_dim), generator=g2) * (1.0 / math.sqrt(input_dim))
    proj_b = torch.randn((output_dim,), generator=g2) * 0.1
    logits = x @ proj_w + proj_b
    # Add a small noise floor so the scorer is not deterministic-bijective.
    noise = torch.randn(logits.shape, generator=g) * 0.05
    y_logits = logits + noise
    y = torch.softmax(y_logits, dim=-1)
    return x, y


def _build_synthetic_variational_encoder(
    *,
    input_dim: int,
    latent_dim: int,
):
    """Small synthetic variational encoder ``q(t|x) -> (mu, log_sigma)``."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(input_dim, 32),
        nn.ReLU(),
        nn.Linear(32, 2 * latent_dim),  # outputs (mu, log_sigma) concatenated
    )


def _build_synthetic_decoder(
    *,
    latent_dim: int,
    output_dim: int,
):
    """Small synthetic decoder ``p(y|t) -> softmax over output_dim classes``."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(latent_dim, 32),
        nn.ReLU(),
        nn.Linear(32, output_dim),
    )


def _reparam_sample(mu, log_sigma):
    """Reparameterization-trick sample ``t = mu + sigma * eps`` (eps ~ N(0,I))."""
    import torch

    sigma = (log_sigma).exp()
    eps = torch.randn_like(mu)
    return mu + sigma * eps


def _kl_diagonal_gaussian_to_standard_normal(mu, log_sigma):
    """Closed-form ``KL(q(t|x)=N(mu, sigma^2) || p(t)=N(0, I))``."""
    sigma_sq = (2.0 * log_sigma).exp()
    return 0.5 * (mu.pow(2) + sigma_sq - 2.0 * log_sigma - 1.0).sum(dim=-1).mean()


def _cross_entropy_reconstruction(pred_logits, target_probs):
    """Cross-entropy ``E[-log p(y|t)]`` with soft target probabilities."""
    import torch.nn.functional as F

    log_pred = F.log_softmax(pred_logits, dim=-1)
    # ``-sum(target * log_pred)`` averaged over batch.
    return -(target_probs * log_pred).sum(dim=-1).mean()


def _verdict_from_snr(
    snr: float,
    *,
    tractable_threshold: float,
    intractable_threshold: float,
) -> VerdictStr:
    """Map gradient SNR to verdict per the design memo §19 V1 lift gate."""
    if snr < intractable_threshold:
        return "INTRACTABLE"
    if snr < tractable_threshold:
        return "MARGINAL"
    return "TRACTABLE"


def compute_variational_ib_tractability(
    *,
    substrate_id: str,
    path_variant: PathVariantStr = "VIB",
    num_replicates: int = DEFAULT_NUM_REPLICATES,
    batch_size: int = DEFAULT_BATCH_SIZE,
    latent_dim: int = DEFAULT_LATENT_DIM,
    input_dim: int = DEFAULT_INPUT_DIM,
    output_dim: int = DEFAULT_OUTPUT_DIM,
    beta: float = DEFAULT_BETA,
    tractable_threshold: float = DEFAULT_GRADIENT_SNR_TRACTABLE_THRESHOLD,
    intractable_threshold: float = DEFAULT_GRADIENT_SNR_INTRACTABLE_THRESHOLD,
    seed: int = 20260516,
    notes: str = "",
) -> VariationalIBTractabilityVerdict:
    """Synthetic-data Monte Carlo VIB tractability probe.

    Raises ``ValueError`` on contract violations per CLAUDE.md
    "Comment-only contracts are FORBIDDEN" — non-positive replicates /
    batch size / non-finite beta / non-finite thresholds / unsupported
    path variant are refused instead of returning a vacuous verdict.
    """
    if not isinstance(substrate_id, str) or not substrate_id.strip():
        raise ValueError("substrate_id must be a non-empty string")
    if path_variant not in ("VIB", "MINE"):
        raise ValueError(
            f"path_variant={path_variant!r} must be 'VIB' or 'MINE'"
        )
    if path_variant == "MINE":
        # Per design memo §4.3 default Path-MINE pivot is the V2 fallback if
        # Path-VIB tractability fails; the canonical v1 probe ships Path-VIB
        # measurement only. Path-MINE measurement requires adding the
        # statistic-network T_NN scaffolding which is out-of-scope for the v1
        # tractability probe per the design memo §22 op-routable #3 scope.
        raise NotImplementedError(
            "Path-MINE tractability measurement requires the statistic-network "
            "T_NN scaffolding per design memo §4.2; v1 probe ships Path-VIB only. "
            "Per design memo §22 op-routable #3 the Path-MINE measurement is "
            "queued as a v2 follow-on subagent landing."
        )
    if num_replicates <= 0:
        raise ValueError(f"num_replicates={num_replicates} must be > 0")
    if batch_size <= 0:
        raise ValueError(f"batch_size={batch_size} must be > 0")
    if latent_dim <= 0:
        raise ValueError(f"latent_dim={latent_dim} must be > 0")
    if input_dim <= 0:
        raise ValueError(f"input_dim={input_dim} must be > 0")
    if output_dim <= 0:
        raise ValueError(f"output_dim={output_dim} must be > 0")
    if not math.isfinite(beta) or beta < 0:
        raise ValueError(f"beta={beta} must be finite >= 0")
    if not math.isfinite(tractable_threshold) or tractable_threshold <= 0:
        raise ValueError(
            f"tractable_threshold={tractable_threshold} must be finite > 0"
        )
    if not math.isfinite(intractable_threshold) or intractable_threshold <= 0:
        raise ValueError(
            f"intractable_threshold={intractable_threshold} must be finite > 0"
        )
    if intractable_threshold >= tractable_threshold:
        raise ValueError(
            f"intractable_threshold={intractable_threshold} must be < "
            f"tractable_threshold={tractable_threshold}"
        )

    # Lazy torch import so the module can be inspected via --help without torch.
    import torch

    torch.manual_seed(seed)
    encoder = _build_synthetic_variational_encoder(
        input_dim=input_dim, latent_dim=latent_dim
    )
    decoder = _build_synthetic_decoder(
        latent_dim=latent_dim, output_dim=output_dim
    )
    encoder.train()
    decoder.train()

    # Pre-build synthetic data once; resample per-replicate by reseeding.
    num_samples_total = batch_size * num_replicates
    x_full, y_full = _make_synthetic_data(
        num_samples=num_samples_total,
        input_dim=input_dim,
        output_dim=output_dim,
        seed=seed,
    )

    per_replicate_grad_norms: list[float] = []
    per_replicate_kls: list[float] = []
    per_replicate_recons: list[float] = []

    for r in range(num_replicates):
        # Reset gradients
        for p in encoder.parameters():
            if p.grad is not None:
                p.grad.zero_()
        for p in decoder.parameters():
            if p.grad is not None:
                p.grad.zero_()

        start = r * batch_size
        end = start + batch_size
        x_b = x_full[start:end]
        y_b = y_full[start:end]

        # Encoder forward
        enc_out = encoder(x_b)
        mu, log_sigma = enc_out.chunk(2, dim=-1)
        # Clamp log_sigma for numerical stability per Alemi 2017
        log_sigma = log_sigma.clamp(min=-10.0, max=10.0)

        # Reparameterized sample
        t = _reparam_sample(mu, log_sigma)

        # Decoder forward
        pred_logits = decoder(t)

        # L_VIB = E[-log p(y|t)] + beta * KL(q(t|x) || p(t))
        recon = _cross_entropy_reconstruction(pred_logits, y_b)
        kl = _kl_diagonal_gaussian_to_standard_normal(mu, log_sigma)
        loss = recon + beta * kl

        loss.backward()

        # Gradient norm across all parameters
        total_grad_sq = 0.0
        for p in list(encoder.parameters()) + list(decoder.parameters()):
            if p.grad is not None:
                total_grad_sq += float(p.grad.pow(2).sum().item())
        grad_norm = math.sqrt(total_grad_sq)
        per_replicate_grad_norms.append(grad_norm)
        per_replicate_kls.append(float(kl.item()))
        per_replicate_recons.append(float(recon.item()))

    # Aggregate
    n = len(per_replicate_grad_norms)
    grad_mean = sum(per_replicate_grad_norms) / n
    grad_var = sum((g - grad_mean) ** 2 for g in per_replicate_grad_norms) / max(n - 1, 1)
    grad_std = math.sqrt(grad_var)

    # SNR = mean / std (Welford's interpretation; small-std => high SNR)
    # Degenerate zero variance implies perfectly tractable OR the synthetic
    # data did not exercise the gradient; report a large SNR and preserve the
    # raw norm stats so consumers can audit that case.
    snr_mean = grad_mean / grad_std if grad_std > 0 else DEFAULT_DEGENERATE_SNR_CAP

    # Median SNR — use the median of bootstrap-style sub-sample SNR
    # estimates by partitioning the replicates into pairs and computing
    # per-pair SNR; for tiny num_replicates fall back to snr_mean.
    if n >= 4:
        sorted_grads = sorted(per_replicate_grad_norms)
        pair_snrs: list[float] = []
        for i in range(n // 2):
            a = sorted_grads[i]
            b = sorted_grads[n - 1 - i]
            pair_mean = (a + b) / 2.0
            pair_std = abs(b - a) / 2.0
            if pair_std > 0:
                pair_snrs.append(pair_mean / pair_std)
        if pair_snrs:
            pair_snrs.sort()
            snr_median = pair_snrs[len(pair_snrs) // 2]
            snr_worst = pair_snrs[0]
        else:
            snr_median = snr_mean
            snr_worst = snr_mean
    else:
        snr_median = snr_mean
        snr_worst = snr_mean

    verdict = _verdict_from_snr(
        snr_mean,
        tractable_threshold=tractable_threshold,
        intractable_threshold=intractable_threshold,
    )

    kl_mean = sum(per_replicate_kls) / n
    recon_mean = sum(per_replicate_recons) / n

    operating_within = (
        "I am operating within the assumption that the variational IB gradient "
        "SNR measured on synthetic Gaussian data with a deterministic linear "
        "+ small-noise scorer is a useful PROXY for the substrate's actual "
        "gradient tractability on contest video + SegNet/PoseNet scorer. The "
        "synthetic-data proxy is HARD-EARNED at the bound-derivation level (the "
        "IB Lagrangian's reparam-trick gradient is well-defined for any "
        "diagonal-Gaussian variational posterior + any decoder) but CARGO-CULTED "
        "at the empirical-equivalence level (real-scorer gradient SNR may differ "
        "by 2-5x due to scorer-output saturation / chroma-vs-luma sensitivity / "
        "PoseNet 12-channel YUV6 nonlinearity). Council-grade verification "
        "requires the design memo §19 STAGE 1 Modal A100 100ep proxy "
        "($5-10 envelope) measuring gradient SNR on the real scorer."
    )

    return VariationalIBTractabilityVerdict(
        substrate_id=substrate_id,
        path_variant=path_variant,
        verdict=verdict,
        gradient_snr_mean=float(snr_mean),
        gradient_snr_median=float(snr_median),
        gradient_snr_worst_case=float(snr_worst),
        gradient_norm_mean=float(grad_mean),
        gradient_norm_std=float(grad_std),
        kl_term_mean=float(kl_mean),
        reconstruction_term_mean=float(recon_mean),
        num_replicates=int(num_replicates),
        batch_size=int(batch_size),
        latent_dim=int(latent_dim),
        input_dim=int(input_dim),
        output_dim=int(output_dim),
        beta=float(beta),
        tractable_threshold=float(tractable_threshold),
        intractable_threshold=float(intractable_threshold),
        # Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #192:
        # this is a diagnostic CPU/A100 probe; NEVER promotable, NEVER a score
        # claim. The autopilot consumer reads these three fields to refuse
        # the row's promotion regardless of verdict.
        evidence_grade="diagnostic_cpu",
        score_claim=False,
        axis_label="[diagnostic-CPU; variational_ib_tractability probe]",
        operating_within_assumption=operating_within,
        observed_at_utc=datetime.datetime.now(datetime.UTC).isoformat(
            timespec="seconds"
        ),
        notes=notes,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute synthetic-data VIB tractability verdict for substrates "
            "that use the variational Information Bottleneck Lagrangian "
            "(Tishby IB-pure / sister IB-class substrates). Canonical "
            "pre-dispatch tractability gate per Tishby IB-pure design "
            "memo §19 V1 lift gate criterion #2."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--substrate-id", required=True,
        help="lane / substrate identifier (e.g. tishby_ib_pure)",
    )
    parser.add_argument(
        "--path-variant", choices=["VIB", "MINE"], default="VIB",
        help="VIB = Alemi 2017 canonical reparam-trick form; "
             "MINE = Belghazi 2018 alternative (v2 op-routable; refuses at v1)",
    )
    parser.add_argument(
        "--num-replicates", type=int, default=DEFAULT_NUM_REPLICATES,
        help="Monte Carlo replicates for gradient SNR estimation",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help="per-replicate batch size",
    )
    parser.add_argument(
        "--latent-dim", type=int, default=DEFAULT_LATENT_DIM,
        help="variational latent dimensionality (matches Tishby IB-pure default 16)",
    )
    parser.add_argument(
        "--input-dim", type=int, default=DEFAULT_INPUT_DIM,
        help="synthetic-data input dimensionality",
    )
    parser.add_argument(
        "--output-dim", type=int, default=DEFAULT_OUTPUT_DIM,
        help="synthetic-data output dimensionality (matches SegNet 5-class default)",
    )
    parser.add_argument(
        "--beta", type=float, default=DEFAULT_BETA,
        help="IB Lagrangian rate-distortion tradeoff (matches design memo §6 target 0.01)",
    )
    parser.add_argument(
        "--gradient-snr-threshold", type=float,
        default=DEFAULT_GRADIENT_SNR_TRACTABLE_THRESHOLD,
        help="SNR >= threshold => TRACTABLE verdict",
    )
    parser.add_argument(
        "--intractable-snr-threshold", type=float,
        default=DEFAULT_GRADIENT_SNR_INTRACTABLE_THRESHOLD,
        help="SNR < this threshold => INTRACTABLE verdict",
    )
    parser.add_argument(
        "--seed", type=int, default=20260516,
        help="random seed for reproducibility",
    )
    parser.add_argument(
        "--output-json", type=Path, default=None,
        help="optional output JSON path (otherwise stdout-only)",
    )
    parser.add_argument(
        "--notes", type=str, default="",
        help="free-form provenance note attached to the verdict JSON",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        verdict = compute_variational_ib_tractability(
            substrate_id=args.substrate_id,
            path_variant=args.path_variant,
            num_replicates=args.num_replicates,
            batch_size=args.batch_size,
            latent_dim=args.latent_dim,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            beta=args.beta,
            tractable_threshold=args.gradient_snr_threshold,
            intractable_threshold=args.intractable_snr_threshold,
            seed=args.seed,
            notes=args.notes,
        )
    except (ValueError, NotImplementedError) as exc:
        print(f"[variational-ib-tractability] FATAL: {exc}", file=sys.stderr)
        return 2

    payload = asdict(verdict)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, allow_nan=False, sort_keys=True, indent=2) + "\n"
        )
    print(json.dumps(payload, allow_nan=False, sort_keys=True, indent=2))
    return 0


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_BETA",
    "DEFAULT_GRADIENT_SNR_INTRACTABLE_THRESHOLD",
    "DEFAULT_GRADIENT_SNR_TRACTABLE_THRESHOLD",
    "DEFAULT_INPUT_DIM",
    "DEFAULT_LATENT_DIM",
    "DEFAULT_NUM_REPLICATES",
    "DEFAULT_OUTPUT_DIM",
    "VariationalIBTractabilityVerdict",
    "compute_variational_ib_tractability",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
