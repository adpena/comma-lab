#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PR101 kalle-fold mixture codec — FAITHFUL NN-WEIGHT-LITERATURE SHAPES.

This is the 1:1 faithful re-implementation of the
``kalle_fold_mixture_canonical_shapes`` catalog row, replacing the prior
``pr101_kalle_fold_8comp_hierarchical_codec.py`` whose canonical shapes
were author-picked (generic Gaussian / Laplace / Cauchy) rather than
drawn from the empirical NN-weight-distribution literature.

Per ``feedback_implementation_vs_model_gap_audit_20260508.md``: the
prior tool's "FALSIFIED" verdict applied to author-picked shapes, NOT
to the technique class. This tool tests the model the catalog row
actually claims by drawing canonical shapes from the literature on
neural-network weight distributions:

NN-weight-literature canonical shapes (cited per shape):

1. **Kaiming/He truncated normal** (He et al. 2015,
   "Delving Deep into Rectifiers"). Std = sqrt(2 / fan_in). The
   variance-preserving initializer for ReLU convolutions; the symbol
   PMF after symmetric INT8 quantization is approximately a truncated
   normal with a fan-in-derived sigma.

2. **Xavier/Glorot truncated normal** (Glorot & Bengio 2010,
   "Understanding the difficulty of training deep feedforward neural
   networks"). Std = sqrt(2 / (fan_in + fan_out)). The variance-
   preserving initializer for tanh/sigmoid layers; many trained
   networks retain a Glorot-shaped envelope on layers that did not
   move much during training.

3. **Trained-Laplace-with-heavy-tails** (Han, Mao, Dally 2016,
   "Deep Compression: Compressing Deep Neural Networks with Pruning,
   Trained Quantization and Huffman Coding"). Trained weights are
   well-modelled as Laplace(0, b) with b set per-layer, plus a small
   fraction of heavy-tail outliers (modelled here as a wide
   secondary Laplace).

4. **Spike-and-slab** (Mitchell & Beauchamp 1988, "Bayesian Variable
   Selection in Linear Regression"; Louizos et al. 2017, "Bayesian
   Compression for Deep Learning"). For sparse models, weights are a
   mixture of (a) a delta spike at zero and (b) a slab (Gaussian or
   Laplace) for the surviving weights.

5. **Truncated-normal-with-outliers** (Dettmers et al. 2022,
   "LLM.int8()"; Xiao et al. 2023, "SmoothQuant"). Modern transformer
   post-training distributions show a Gaussian core with a small but
   significant outlier fraction in the heavy tails. Modelled as a
   narrow truncated normal plus a wide uniform tail.

6. **Symmetric quantization-clip mass** (Banner et al. 2019,
   "Post-training 4-bit quantization of convolution networks for
   rapid-deployment"). Symmetric INT8 quantization induces small but
   non-zero mass at the saturation extremes (-127, +127). Modelled
   as a delta mixture at the two endpoints.

The model is an over-complete mixture: each per-tensor PMF is fit as
a softmax-weighted sum of all 7 canonical shapes (5 literature shapes
counted as 1, 2, 3a+3b, 4, 5 = 6 components; 6 above is component 7).
A faithful test must let the per-tensor optimizer choose the mixture
weights. Layers that did not deviate far from initialization should
prefer Kaiming/Xavier shapes; deeper, more-trained layers should
prefer Laplace; sparse layers should prefer spike-and-slab.

Per-tensor parameters:
  - 7 mixture weight logits (softmax)
  - 1 fan_in scale (already implied by tensor shape; included as
    optimizable scale on Kaiming shape)
  - 2 Laplace scales (b1 narrow, b2 wide-tail)
  - 1 slab Gaussian scale (for spike-and-slab)
  - 1 outlier-uniform width (for truncated-normal-with-outliers)
  Total: 12 floats per tensor = 24 fp16 bytes per tensor.
  28 tensors x 24 B = 672 B mixture metadata + 28 * 2 B scale = 728 B.

Compliance with ``feedback_premature_falsification_metacognitive_failure_mode``:
  - This tool tests EXACTLY ONE configuration of one shape-family.
  - The full config space is enumerated below.
  - If archive bytes > brotli baseline, verdict is
    ``MEASURED_CONFIG_NOT_DISPATCHABLE`` not "FALSIFIED".
  - ``family_falsified=False`` always; class-level KILL requires
    consensus + exhaustion of all configs in the enumeration.

## Config space enumeration (canonical-shape-mixture)

Technique class: kalle_fold_mixture_canonical_shapes
Full config space:
  { author_picked_4comp,                  # prior: pr101_kalle_fold_mixture_codec.py
    author_picked_8comp_hierarchical,     # prior: pr101_kalle_fold_8comp_hierarchical_codec.py
    nn_literature_shapes,                  # this tool
    learned_dictionary_basis,             # van den Oord style trained codebook of canonical PMFs (untested)
    sparse_coding_basis,                   # NMF/dictionary on PR101 PMFs (untested)
    tensor_class_conditioned_mixture,      # weights conditioned on layer-class label (untested)
    rate_distortion_optimal_basis,         # Ballé-style learned analytical basis (untested)
  }

This tool tests: { nn_literature_shapes }
Falsification scope: this tool can only falsify nn_literature_shapes
configuration. The other 6 configs remain untested.

CLAUDE.md compliance: pure CPU + numpy + scipy + brotli + constriction;
no scorer load; no contest score claims; output tagged
``[CPU-prep empirical reactivation]``.
``ready_for_exact_eval_dispatch`` is ALWAYS False from this tool
(CPU-derived; per ``feedback_premature_falsification_metacognitive_failure_mode``
Rule 4).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import struct
import sys
from collections import Counter
from pathlib import Path

import brotli
import constriction
import numpy as np
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
)

TOOL_NAME = "tools/pr101_kalle_fold_nn_literature_shapes.py"
SCHEMA_VERSION = "pr101_kalle_fold_nn_literature_shapes.v1"
ARCHIVE_OVERHEAD_BYTES = 16_094
N_SYMBOLS = 2 * N_QUANT + 1  # 255
SYMBOL_AXIS = np.arange(-N_QUANT, N_QUANT + 1, dtype=np.float64)
EVIDENCE_GRADE = "[CPU-prep empirical reactivation]"
EVIDENCE_SEMANTICS = (
    "cpu_kalle_fold_nn_literature_shapes_byte_anchor_no_decoder_no_score"
)

# Per ``feedback_premature_falsification_metacognitive_failure_mode``
# Rule 4: ``ready_for_exact_eval_dispatch`` MUST be False from any CPU
# measurement. The dispatch_blockers list documents what's missing.
DISPATCH_BLOCKERS = (
    "nn_literature_mixture_decoder_not_wired_into_runtime_packet",
    "no_archive_substitution_performed",
    "no_decode_roundtrip_fixture_for_full_packet",
    "missing_exact_cuda_auth_eval",
    "requires_exact_cuda_auth_eval_before_any_score_use",
)

# Reference baselines for delta accounting only (not "win/lose" verdict
# strings; per Rule 2 we use empirical-band language).
REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES = 178_144
REFERENCE_PRIOR_8COMP_ARCHIVE_BYTES = 206_354

# 7-component literature mixture: indices map to component identity.
# Comp 0: Kaiming truncated normal (He 2015)
# Comp 1: Xavier truncated normal (Glorot & Bengio 2010)
# Comp 2a: Trained-Laplace narrow (Han et al. 2016)
# Comp 3: Trained-Laplace wide-tail (Han et al. 2016)
# Comp 4: Spike-and-slab spike (Mitchell & Beauchamp 1988)
# Comp 5: Truncated-normal-with-outliers core (Dettmers 2022)
# Comp 6: Symmetric clip-mass at endpoints (Banner et al. 2019)
N_COMPONENTS = 7
N_PARAMS_PER_TENSOR = N_COMPONENTS + 5  # 7 weight logits + 5 scales = 12 floats


# ---------------------------------------------------------------------------
# Canonical shape constructors (each cites paper + provides a per-symbol PMF)
# ---------------------------------------------------------------------------


def kaiming_truncnormal_pmf(sigma: float) -> np.ndarray:
    """Kaiming/He 2015 truncated normal, std=sigma. Truncated to symbol grid."""
    sigma = max(float(sigma), 1e-3)
    pmf = np.exp(-0.5 * (SYMBOL_AXIS / sigma) ** 2)
    pmf /= pmf.sum()
    return pmf


def xavier_truncnormal_pmf(sigma: float) -> np.ndarray:
    """Glorot & Bengio 2010 truncated normal, std=sigma.

    Functionally a truncated normal but with a different sigma derivation
    (sqrt(2/(fan_in+fan_out))). The fitter chooses sigma per-tensor so
    this shape's contribution is parameterized identically; the
    distinction is in INTERPRETATION of which sigma family the
    optimizer chose.
    """
    sigma = max(float(sigma), 1e-3)
    pmf = np.exp(-0.5 * (SYMBOL_AXIS / sigma) ** 2)
    pmf /= pmf.sum()
    return pmf


def trained_laplace_narrow_pmf(b: float) -> np.ndarray:
    """Han, Mao, Dally 2016 "Deep Compression" — narrow Laplace core."""
    b = max(float(b), 1e-3)
    pmf = np.exp(-np.abs(SYMBOL_AXIS) / b)
    pmf /= pmf.sum()
    return pmf


def trained_laplace_wide_tail_pmf(b: float) -> np.ndarray:
    """Han et al. 2016 — heavy-tail Laplace component for outlier mass."""
    b = max(float(b), 1e-3)
    pmf = np.exp(-np.abs(SYMBOL_AXIS) / b)
    pmf /= pmf.sum()
    return pmf


def spike_and_slab_pmf(slab_sigma: float, spike_mass: float = 1.0) -> np.ndarray:
    """Mitchell & Beauchamp 1988 spike-and-slab — delta at zero plus
    Gaussian slab. ``spike_mass`` is the relative weight of the spike
    inside this component (the OUTER mixture weight handles selection
    between this and other components)."""
    slab_sigma = max(float(slab_sigma), 1e-3)
    spike_mass = float(np.clip(spike_mass, 0.0, 1.0))
    spike = np.zeros(N_SYMBOLS, dtype=np.float64)
    spike[N_QUANT] = 1.0
    slab = np.exp(-0.5 * (SYMBOL_AXIS / slab_sigma) ** 2)
    slab /= slab.sum()
    pmf = spike_mass * spike + (1.0 - spike_mass) * slab
    pmf /= pmf.sum()
    return pmf


def trunc_normal_with_outliers_pmf(
    core_sigma: float, outlier_width: float
) -> np.ndarray:
    """Dettmers 2022 LLM.int8() / Xiao 2023 SmoothQuant — Gaussian
    core plus uniform-outlier tail. ``outlier_width`` controls the
    fraction of the symbol grid covered by the uniform component
    (0.0 = no outliers, 1.0 = full grid). The internal mixture of
    core/uniform is fixed at 0.95/0.05 per the LLM.int8() empirical
    finding that ~5% of weights are outliers."""
    core_sigma = max(float(core_sigma), 1e-3)
    outlier_width = float(np.clip(outlier_width, 0.0, 1.0))
    core = np.exp(-0.5 * (SYMBOL_AXIS / core_sigma) ** 2)
    core /= core.sum()
    if outlier_width <= 1e-6:
        return core
    half_width = max(int(round((N_SYMBOLS - 1) / 2 * outlier_width)), 1)
    tail = np.zeros(N_SYMBOLS, dtype=np.float64)
    tail[N_QUANT - half_width: N_QUANT + half_width + 1] = 1.0
    tail /= tail.sum()
    pmf = 0.95 * core + 0.05 * tail
    pmf /= pmf.sum()
    return pmf


def clip_mass_endpoints_pmf() -> np.ndarray:
    """Banner et al. 2019 post-training INT8 — saturation mass at the
    symmetric quantization endpoints (-127, +127). Parameter-free; both
    endpoints get equal mass."""
    pmf = np.zeros(N_SYMBOLS, dtype=np.float64)
    pmf[0] = 0.5
    pmf[-1] = 0.5
    return pmf


# ---------------------------------------------------------------------------
# Mixture construction and fitting
# ---------------------------------------------------------------------------


def mixture_pmf(params: np.ndarray) -> np.ndarray:
    """Build the 7-component literature-shape mixture from a 12-param vector.

    params[0:7]  = mixture weight logits (softmax over all 7 components)
    params[7]    = log(kaiming_sigma)  — Kaiming/He 2015
    params[8]    = log(xavier_sigma)   — Glorot & Bengio 2010
    params[9]    = log(laplace_b1)     — Han 2016 narrow
    params[10]   = log(laplace_b2)     — Han 2016 wide-tail
    params[11]   = log(slab_sigma)     — Mitchell & Beauchamp 1988

    Truncated-normal-with-outliers reuses the kaiming sigma as core
    sigma (they're both Gaussian-family shapes; the fitter doesn't
    need an extra independent sigma) and a fixed outlier_width=0.6.
    Clip-mass component is parameter-free.
    """
    if params.shape[0] != N_PARAMS_PER_TENSOR:
        raise ValueError(
            f"expected {N_PARAMS_PER_TENSOR} params, got {params.shape[0]}"
        )
    w_logits = params[:N_COMPONENTS]
    w = np.exp(w_logits - np.max(w_logits))
    w /= w.sum()
    # Clip log-scales to fp16-safe range; this also tames optimizer trial
    # points that probe extreme exp() values during L-BFGS-B linesearch.
    log_scales = np.clip(params[N_COMPONENTS:], -15.0, 11.0)
    kaiming_sigma = float(np.exp(log_scales[0]))
    xavier_sigma = float(np.exp(log_scales[1]))
    laplace_b1 = float(np.exp(log_scales[2]))
    laplace_b2 = float(np.exp(log_scales[3]))
    slab_sigma = float(np.exp(log_scales[4]))

    shapes = [
        kaiming_truncnormal_pmf(kaiming_sigma),
        xavier_truncnormal_pmf(xavier_sigma),
        trained_laplace_narrow_pmf(laplace_b1),
        trained_laplace_wide_tail_pmf(laplace_b2),
        spike_and_slab_pmf(slab_sigma=slab_sigma, spike_mass=0.6),
        trunc_normal_with_outliers_pmf(
            core_sigma=kaiming_sigma, outlier_width=0.6
        ),
        clip_mass_endpoints_pmf(),
    ]
    pmf = np.zeros(N_SYMBOLS, dtype=np.float64)
    for w_i, shape_i in zip(w, shapes):
        pmf += w_i * shape_i
    pmf = np.maximum(pmf, 1e-12)
    pmf /= pmf.sum()
    return pmf


def empirical_pmf(symbols_i8: np.ndarray) -> np.ndarray:
    """Return a 255-bin PMF over symmetric int8 symbols (zero-centered)."""
    counts = Counter(int(s) for s in symbols_i8.flatten().tolist())
    pmf = np.zeros(N_SYMBOLS, dtype=np.float64)
    for sym, c in counts.items():
        idx = sym + N_QUANT
        if 0 <= idx < N_SYMBOLS:
            pmf[idx] = float(c)
    total = pmf.sum()
    if total > 0:
        pmf /= total
    return pmf


def _fit_neg_log_lik(target: np.ndarray):
    def _loss(params: np.ndarray) -> float:
        pmf = mixture_pmf(params)
        return float(-np.sum(target * np.log(pmf)))

    return _loss


def fit_mixture_multi_start(
    target_pmf: np.ndarray, *, seed: int = 0
) -> tuple[np.ndarray, float, list[float]]:
    """Fit literature-shape mixture via L-BFGS-B with multiple inits.

    Returns ``(best_params, best_kl_bits_per_element, all_kl_bits_per_element)``.

    Multi-start strategy: 5 inits drawn from the literature parameter
    ranges. The all_kl returned should be monotonic-or-equal (the test
    checks the optimizer respects the multi-start contract: the best
    seen at each cumulative step does not increase).
    """
    target = np.maximum(target_pmf, 1e-12)
    target /= target.sum()
    loss = _fit_neg_log_lik(target)

    rng = np.random.default_rng(seed)
    # 5 inits: each draws scale parameters from literature-realistic ranges.
    # Logit jitter is small (the 7-comp mixture is fit by gradient).
    inits = []
    for sigma_kaiming, sigma_xavier, b1, b2, slab in [
        (4.0, 8.0, 2.0, 16.0, 4.0),
        (8.0, 16.0, 4.0, 32.0, 8.0),
        (2.0, 4.0, 1.0, 8.0, 2.0),
        (16.0, 32.0, 8.0, 64.0, 16.0),
        (1.0, 2.0, 0.5, 4.0, 1.0),
    ]:
        x0 = np.zeros(N_PARAMS_PER_TENSOR, dtype=np.float64)
        x0[:N_COMPONENTS] = rng.normal(0.0, 0.1, size=N_COMPONENTS)
        x0[7] = np.log(sigma_kaiming)
        x0[8] = np.log(sigma_xavier)
        x0[9] = np.log(b1)
        x0[10] = np.log(b2)
        x0[11] = np.log(slab)
        inits.append(x0)

    best_params = inits[0]
    best_loss = float("inf")
    cumulative_best_kl_bits: list[float] = []
    for x0 in inits:
        result = minimize(
            loss, x0, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-9}
        )
        if result.fun < best_loss:
            best_loss = float(result.fun)
            best_params = result.x
        # Compute current cumulative-best KL in bits/element.
        cur_pmf = mixture_pmf(best_params)
        kl_nats = float(
            np.sum(target * (np.log(target + 1e-12) - np.log(cur_pmf)))
        )
        cumulative_best_kl_bits.append(kl_nats / np.log(2))

    final_pmf = mixture_pmf(best_params)
    final_kl_nats = float(
        np.sum(target * (np.log(target + 1e-12) - np.log(final_pmf)))
    )
    final_kl_bits = final_kl_nats / np.log(2)
    return best_params, final_kl_bits, cumulative_best_kl_bits


# ---------------------------------------------------------------------------
# Encode / decode
# ---------------------------------------------------------------------------


def serialize_mixture_params(all_params: list[np.ndarray]) -> bytes:
    """Pack mixture params as fp16: 12 fp16 per tensor = 24 bytes per tensor.

    Log-scales are clipped to fp16-safe range [-15, 11] before storage to
    avoid silent overflow at deserialize time (fp16 max ~65504, exp(11)
    is already ~59874; values larger than that would round to inf and
    produce a degenerate mixture)."""
    payload = bytearray()
    for p in all_params:
        if p.shape[0] != N_PARAMS_PER_TENSOR:
            raise ValueError(
                f"each tensor's params must be {N_PARAMS_PER_TENSOR} floats"
            )
        clipped = p.copy()
        # Logits stay in fp16 dynamic range fine. Log-scales (indices 7..)
        # may blow up; clip them defensively.
        clipped[N_COMPONENTS:] = np.clip(clipped[N_COMPONENTS:], -15.0, 11.0)
        for v in clipped:
            payload += np.float16(v).tobytes()
    return bytes(payload)


def deserialize_mixture_params(blob: bytes, n_tensors: int) -> list[np.ndarray]:
    """Inverse of serialize_mixture_params; recovers per-tensor params."""
    expected = n_tensors * N_PARAMS_PER_TENSOR * 2
    if len(blob) != expected:
        raise ValueError(
            f"expected {expected} bytes for {n_tensors} tensors of "
            f"{N_PARAMS_PER_TENSOR} fp16 each; got {len(blob)}"
        )
    arr = np.frombuffer(blob, dtype=np.float16).astype(np.float64)
    return [
        arr[i * N_PARAMS_PER_TENSOR : (i + 1) * N_PARAMS_PER_TENSOR].copy()
        for i in range(n_tensors)
    ]


def encode_tensor_with_mixture(
    symbols_i8: np.ndarray, params: np.ndarray
) -> bytes:
    """AC-encode the symbols using the per-tensor mixture PMF."""
    pmf = mixture_pmf(params)
    encoder = constriction.stream.queue.RangeEncoder()
    model = constriction.stream.model.Categorical(pmf, perfect=False)
    biased = (symbols_i8.astype(np.int32) + N_QUANT).flatten()
    encoder.encode(biased, model)
    return bytes(encoder.get_compressed())


def decode_tensor_with_mixture(
    payload: bytes, params: np.ndarray, n_symbols: int
) -> np.ndarray:
    """AC-decode using the same per-tensor mixture PMF; inverse of encode."""
    pmf = mixture_pmf(params)
    decoder = constriction.stream.queue.RangeDecoder(np.frombuffer(payload, dtype=np.uint32))
    model = constriction.stream.model.Categorical(pmf, perfect=False)
    biased = decoder.decode(model, n_symbols)
    return (np.asarray(biased, dtype=np.int32) - N_QUANT).astype(np.int8)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def proxy_evidence_contract() -> dict[str, object]:
    """Per ``feedback_premature_falsification_metacognitive_failure_mode``
    Rule 4: ANY CPU-derived evidence row sets ``ready_for_exact_eval_dispatch``
    to False and ``family_falsified`` to False. Class-level falsification
    requires consensus + exhaustion."""
    return {
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_marker": EVIDENCE_GRADE,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
        "proxy_row": True,
        "family_falsified": False,
        "falsification_scope": "measured_configuration_only",
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
    }


# ---------------------------------------------------------------------------
# Top-level codec runner
# ---------------------------------------------------------------------------


def run_codec(state_dict_path: Path, *, seed: int = 0) -> dict:
    """Quantize, fit literature-shape mixtures per-tensor, AC-encode,
    brotli-wrap, measure."""
    import torch

    input_sha256 = sha256_file(state_dict_path)
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        raise SystemExit(f"state_dict at {state_dict_path} is not a dict")

    per_tensor: list[dict] = []
    all_params: list[np.ndarray] = []
    all_payloads: list[bytes] = []
    scales: list[float] = []
    total_elements = 0
    total_kl_bits = 0.0

    for name, _shape in FIXED_STATE_SCHEMA:
        if name not in sd:
            raise SystemExit(f"missing tensor {name!r} in state_dict")
        qt = _quantize_tensor(name, sd[name], n_quant=N_QUANT)
        scales.append(float(qt.scale))
        pmf = empirical_pmf(qt.q_i8)
        params, kl_bits, cum_kl_trace = fit_mixture_multi_start(pmf, seed=seed)
        all_params.append(params)
        ac_payload = encode_tensor_with_mixture(qt.q_i8, params)
        all_payloads.append(ac_payload)
        n = int(qt.q_i8.size)
        total_elements += n
        total_kl_bits += kl_bits * n
        # Resolved mixture weights for forensic.
        w_logits = params[:N_COMPONENTS]
        w = np.exp(w_logits - np.max(w_logits))
        w /= w.sum()
        per_tensor.append({
            "name": name,
            "n_elements": n,
            "kl_bits_per_element": float(kl_bits),
            "ac_payload_bytes": len(ac_payload),
            "mixture_weights_by_component": {
                "kaiming_he2015": float(w[0]),
                "xavier_glorot2010": float(w[1]),
                "trained_laplace_narrow_han2016": float(w[2]),
                "trained_laplace_wide_tail_han2016": float(w[3]),
                "spike_slab_mitchell1988": float(w[4]),
                "trunc_normal_outliers_dettmers2022": float(w[5]),
                "clip_mass_endpoints_banner2019": float(w[6]),
            },
            "resolved_scales": {
                "kaiming_sigma": float(np.exp(params[7])),
                "xavier_sigma": float(np.exp(params[8])),
                "laplace_b_narrow": float(np.exp(params[9])),
                "laplace_b_wide_tail": float(np.exp(params[10])),
                "slab_sigma": float(np.exp(params[11])),
            },
            "multi_start_cum_kl_bits_trace": [float(v) for v in cum_kl_trace],
        })

    scales_arr = np.array(scales, dtype=np.float16)
    metadata_blob = (
        b"KLIT"  # tag: Kalle-LITerature shapes
        + struct.pack("<I", len(FIXED_STATE_SCHEMA))
        + scales_arr.tobytes()
        + serialize_mixture_params(all_params)
    )
    metadata_brotli = brotli.compress(
        metadata_blob, quality=11, lgwin=16, lgblock=19
    )
    payload_concat = b"".join(
        struct.pack("<I", len(p)) + p for p in all_payloads
    )
    payload_brotli = brotli.compress(
        payload_concat, quality=11, lgwin=16, lgblock=19
    )
    decoder_blob_bytes = len(metadata_brotli) + len(payload_brotli)
    archive_bytes = decoder_blob_bytes + ARCHIVE_OVERHEAD_BYTES

    contract = proxy_evidence_contract()
    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        **contract,
        "input_state_dict": repo_relative(state_dict_path),
        "input_state_dict_sha256": input_sha256,
        "n_components": N_COMPONENTS,
        "n_params_per_tensor": N_PARAMS_PER_TENSOR,
        "literature_citations": [
            "He et al. 2015 — Kaiming truncated normal",
            "Glorot & Bengio 2010 — Xavier truncated normal",
            "Han, Mao, Dally 2016 — Deep Compression (Laplace narrow + wide-tail)",
            "Mitchell & Beauchamp 1988 — spike-and-slab",
            "Dettmers et al. 2022 LLM.int8() — truncated normal with outliers",
            "Banner et al. 2019 — symmetric quantization clip-mass at endpoints",
        ],
        "n_tensors": len(FIXED_STATE_SCHEMA),
        "total_elements": total_elements,
        "metadata_blob_bytes": len(metadata_blob),
        "metadata_brotli_bytes": len(metadata_brotli),
        "ac_payload_concat_bytes": len(payload_concat),
        "payload_brotli_bytes": len(payload_brotli),
        "decoder_blob_bytes": decoder_blob_bytes,
        "archive_overhead_bytes": ARCHIVE_OVERHEAD_BYTES,
        "archive_bytes": archive_bytes,
        "comparison_brotli_optuna_archive_bytes": REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES,
        "delta_vs_brotli_optuna_archive_bytes": (
            archive_bytes - REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES
        ),
        "comparison_prior_8comp_archive_bytes": REFERENCE_PRIOR_8COMP_ARCHIVE_BYTES,
        "delta_vs_prior_8comp_archive_bytes": (
            archive_bytes - REFERENCE_PRIOR_8COMP_ARCHIVE_BYTES
        ),
        "weighted_kl_bits_per_element": total_kl_bits / max(total_elements, 1),
        "per_tensor": per_tensor,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--state-dict",
        type=Path,
        default=REPO_ROOT
        / "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt",
    )
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-evidence", type=Path, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict}")

    manifest = run_codec(args.state_dict, seed=args.seed)

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_json is None:
        out_dir = REPO_ROOT / f"reports/raw/pr101_kalle_fold_nn_lit_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_json = out_dir / "manifest.json"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"manifest: {args.output_json}\n")
    print(f"archive_bytes: {manifest['archive_bytes']:,} B")
    print(f"  vs prior 8-comp result: {REFERENCE_PRIOR_8COMP_ARCHIVE_BYTES:,} B")
    delta_8c = manifest["delta_vs_prior_8comp_archive_bytes"]
    print(f"  delta vs 8-comp: {delta_8c:+,} B")
    print(f"  vs brotli baseline: {REFERENCE_BROTLI_OPTUNA_ARCHIVE_BYTES:,} B")
    delta_brotli = manifest["delta_vs_brotli_optuna_archive_bytes"]
    print(f"  delta vs brotli: {delta_brotli:+,} B")
    print(
        f"\n  metadata: {manifest['metadata_blob_bytes']:,} raw -> "
        f"{manifest['metadata_brotli_bytes']:,} B brotli"
    )
    print(
        f"  ac payload: {manifest['ac_payload_concat_bytes']:,} -> "
        f"{manifest['payload_brotli_bytes']:,} B brotli"
    )
    print(f"  weighted KL: {manifest['weighted_kl_bits_per_element']:.4f} bits/element")

    # Per ``feedback_premature_falsification_metacognitive_failure_mode``:
    # use neutral verdict language. Never "FALSIFIED" / "BEAT" / "WIN".
    if delta_brotli < 0:
        verdict = "EMPIRICAL_BAND_BELOW_BROTLI_BASELINE"
    elif abs(delta_brotli) < 100:
        verdict = "EMPIRICAL_BAND_AT_BROTLI_BASELINE"
    else:
        verdict = "MEASURED_CONFIG_NOT_DISPATCHABLE"
    print(f"\n  verdict (neutral): {verdict}")
    print(
        "  (per metacognitive protocol: FALSIFIED / KILL forbidden; "
        "class falsification requires consensus + config-space exhaustion)"
    )

    if args.output_evidence:
        evidence_row = {
            "technique": "kalle_fold_mixture_canonical_shapes",
            "configuration": "nn_literature_shapes",
            "empirical_archive_bytes": manifest["archive_bytes"],
            **proxy_evidence_contract(),
            "source": (
                f"{EVIDENCE_GRADE} {repo_relative(args.output_json)} "
                "7-component literature mixture (Kaiming + Xavier + "
                "trained-Laplace x2 + spike-slab + outlier-normal + clip-mass)"
            ),
            "timestamp": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "contest_dispatch_verdict": verdict,
            "reactivation_criteria_tested": ["nn_literature_shapes"],
            "reactivation_criteria_remaining": [
                "learned_dictionary_basis",  # van den Oord style codebook
                "sparse_coding_basis",  # NMF on PR101 PMFs
                "tensor_class_conditioned_mixture",  # weights conditioned on layer-class
                "rate_distortion_optimal_basis",  # Ballé-style learned analytical
                "12_or_more_component_mixture",
            ],
        }
        args.output_evidence.parent.mkdir(parents=True, exist_ok=True)
        with args.output_evidence.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evidence_row) + "\n")
        print(f"\nevidence row appended: {args.output_evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
