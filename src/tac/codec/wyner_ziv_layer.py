# SPDX-License-Identifier: MIT
"""``tac.codec.wyner_ziv_layer`` — canonical Wyner-Ziv PIPELINE-STAGE codec primitive.

Per operator architectural correction 2026-05-17 verbatim *"WZ is a PIPELINE
STAGE applied PRE-entropy, NOT a post-hoc archive analysis"* and task #814
(lane ``lane_wyner_ziv_pipeline_stage_codec_primitive_20260517``).

The Wyner-Ziv 1976 source-coding-with-side-information theorem states that a
decoder with side-info Y can reduce rate to R(D|Y) < R(D). In a compression
PIPELINE, WZ is inserted as a STAGE between two existing stages — typically
BEFORE the final entropy step (Huffman / range / arithmetic / lzma / brotli).

Empirical receipts driving this primitive's design
==================================================

- **fec6 post-entropy** (sister ``lane_wyner_ziv_deliverability_prober_20260517``,
  probe artifact ``probe_f174192aeadf_20260517T205208.json``): lzma/brotli/zlib
  INFLATE → WZ post-hoc hoist saves ZERO bytes. The entropy stage already
  squeezes correlation; nothing left for WZ.
- **pr101_state_dict + pr106_state_dict PRE-entropy** (sister
  ``lane_pre_entropy_substrate_pivot_prober_20260517``, prober artifact
  ``.omx/state/wyner_ziv_deliverability/pre_entropy_candidate_substrates_20260517T210723.json``):
  lzma compresses raw fp16 state_dicts at 0.217-0.228 ratio. Translated to
  WZ deliverable savings: **0.47 score reduction per substrate**.
- **20.3 MB ``posenet_class_sensitivity``** (same prober): theoretical 11.6
  score delta if hoisted via WZ-pipeline-insertion + sister architecture.

Why post-entropy WZ fails: lzma's LZ77+range-coding has already exploited
within-stream redundancy. Post-entropy bytes are near-uniform; WZ has nothing
left to exploit. Pre-entropy bytes (raw fp16 weights, raw quantizer output,
raw transform coefficients) still carry the cross-pair / cross-tensor /
cross-frame structure WZ can reduce via side-info Y.

Canonical pipeline intercept points
====================================

A pipeline like::

    raw_data → transform → quantize → predict_residual → ENTROPY → archive

becomes (WZ inserted between any two stages, conventionally before entropy)::

    raw_data → transform → quantize → predict_residual → WZ_SPLIT →
        ├── main (smaller; goes to ENTROPY → archive)
        └── side (small; baked into inflate.py as compressed_codec(side))

At inflate time::

    archive → inflate_entropy → main
    inflate.py-baked-side + side_info_y_derived_at_inflate → WZ_RECONSTRUCT(main, side, Y) → raw_data

This module exposes the WZ stage as a black-box codec primitive that any
existing pipeline can sandwich between any two stages.

Public API
==========

* :class:`InterceptLocation` — taxonomy of common intercept points.
* :class:`WynerZivLayerConfig` — frozen config for one WZ insertion.
* :class:`WynerZivLayerResult` — frozen result with byte counts +
  score-savings estimate + canonical helper citation.
* :func:`insert_wyner_ziv_layer` — encoder path; splits ``pre_entropy_bytes``
  into ``(main, side)``; applies ``main_codec`` to main; applies
  ``compression_codec_for_side`` to side; bakes ``side`` into inflate.py
  constants.
* :func:`reconstruct_from_wyner_ziv_layer` — decoder path; re-creates the
  original ``pre_entropy_bytes`` from ``(main, side, side_info_y)``.
  Determinism + byte-equivalence to source REQUIRED.
* :func:`derive_side_info_from_canonical_source` — canonical side-info
  derivation: ``"Comma2k19" | "ImageNet" | "torch_defaults" | "math_constants"``.
* :func:`estimate_composition_alpha` — per Catalog #227 composition matrix
  alpha between this WZ stage and another pipeline stage.

Apples-to-apples evidence discipline
====================================

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #127:
``WynerZivLayerResult.score_savings_estimate`` is a PREDICTION. The result
ships with ``evidence_grade="predicted"`` + ``score_claim=False`` +
``promotion_eligible=False`` defaults. A paired contest-CUDA + contest-CPU
auth-eval anchor is REQUIRED before any score claim.

Strict-scorer-rule discipline
=============================

Per CLAUDE.md "Strict scorer rule — non-negotiable" + Catalog #6 + Catalog #7:
``side_info_source="scorer_compressed"`` is FORBIDDEN unless the operator has
explicitly attested via ``operator_attested_scorer_side_info=True``. The
sister strict gate Catalog #320 refuses any
:class:`WynerZivLayerConfig` that violates this contract.

HNeRV parity discipline
=======================

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L4:
``WynerZivLayerResult.inflate_py_loc_added`` MUST stay ≤ 100 by default (≤
200 with explicit operator waiver). The result records the per-config LOC
overhead so downstream consumers + Catalog #320 can audit.

## Observability surface

Per CLAUDE.md "Max observability — non-negotiable" + Catalog #305:

* **Inspectable per layer**: every encoder invocation records
  ``main_bytes_sha256`` + ``side_info_sha256`` so the (main, side, Y) triple
  can be re-derived byte-identically.
* **Decomposable per signal**: per-tier byte counts (main_bytes_raw vs
  main_bytes_compressed vs side_bytes_raw vs side_bytes_compressed_baked)
  + per-tier score-savings estimate + decoder complexity estimate.
* **Diff-able across runs**: byte-identical encode for byte-identical
  ``(pre_entropy_bytes, side_info_y, config)`` triple.
* **Queryable post-hoc**: :class:`WynerZivLayerResult` is JSON-serializable
  via ``dataclasses.asdict()``; the schema is pinned at
  ``WYNER_ZIV_LAYER_RESULT_SCHEMA_VERSION``.
* **Cite-able**: every result carries
  ``(intercept_location, side_info_source, main_codec, compression_codec_for_side,
  schema_version, evidence_grade)``.
* **Counterfactual-able**: encode-decode roundtrip is the canonical
  byte-mutation probe (sister of Catalog #139 packet compiler no-op detector
  + Catalog #220 substrate L1+ operational mechanism).

Canonical-vs-unique decision per layer
======================================

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode":

| Layer | Decision | Rationale |
|---|---|---|
| Result dataclass | ADOPT_CANONICAL_BECAUSE_SERVES | Mirrors ``DeliverabilityProof`` frozen pattern |
| ``side_info_source`` enum | UNIQUE | Pipeline-stage primitive needs explicit Y-derivation taxonomy |
| ``InterceptLocation`` enum | UNIQUE | New abstraction; no existing canonical |
| ``main_codec`` / ``compression_codec_for_side`` | ADOPT_CANONICAL_BECAUSE_SERVES | Re-use lzma/brotli/zlib from stdlib + brotli pkg |
| Side-info canonical sources | ADOPT_CANONICAL_BECAUSE_SERVES | Routes through ``Comma2k19LocalCache`` per Catalog #213 |
| Strict-scorer-rule guard | ADOPT_CANONICAL_BECAUSE_SERVES | Same pattern as Catalog #6 / #7 / #319 |
| Composition alpha source | ADOPT_CANONICAL_BECAUSE_SERVES | Routes through ``tac.optimization.substrate_composition_matrix`` |
| Persistence | FORK_BECAUSE_PRINCIPLED_MISMATCH | This is a PRIMITIVE not a state-store; callers persist via their own canonical helpers |

## 9-dimension success checklist evidence

Per CLAUDE.md "9-dimension success checklist evidence" (Catalog #294):

1. **UNIQUENESS** — pipeline-stage codec primitive (NOT post-hoc archive
   analyzer; that's the sister deliverability_prober).
2. **BEAUTY + ELEGANCE** — 3 data classes + 4 public functions; reviewable
   in 30 seconds.
3. **DISTINCTNESS** — explicitly different from
   ``tac.wyner_ziv_deliverability.proof_builder`` (per-substrate analyzer)
   AND from ``tac.sensitivity_map.wyner_ziv_reweight`` (per-byte sensitivity
   reweight).
4. **RIGOR** — premise verification per Catalog #229; byte-identical encode
   roundtrip test; HNeRV parity L4 budget compliance; strict-scorer-rule
   compliance.
5. **OPTIMIZATION PER TECHNIQUE** — lzma for fp16 state_dict (4-7x); brotli
   for ASCII-like sparse tensors; zlib as portable fallback.
6. **STACK-OF-STACKS-COMPOSABILITY** — :func:`estimate_composition_alpha`
   exposes Catalog #227 composition matrix; this stage composes orthogonally
   with transform-coding + quantization + predictive-coding stages.
7. **DETERMINISTIC REPRODUCIBILITY** — frozen config + deterministic seed +
   byte-identical encode + sha256 manifest.
8. **EXTREME OPTIMIZATION + PERFORMANCE** — compress-time cost amortized
   across all dispatches; inflate-time overhead is one lzma.decompress call
   on the side stream (~ms wall-clock).
9. **OPTIMAL MINIMAL CONTEST SCORE** — per the sister prober, hoisting raw
   pr101/pr106 fp16 state_dicts via this primitive predicts 0.47 score
   reduction per substrate.

## Cargo-cult audit per assumption

Per CLAUDE.md "Substrate design memos MUST include cargo-cult audit"
(Catalog #303):

| Assumption | Classification | Unwind path if cargo-culted |
|---|---|---|
| Side stream compresses well via lzma | HARD-EARNED | Empirical: fp16 state_dict 4-7x via lzma per prober anchor |
| Y derivable at inflate time | HARD-EARNED | Comma2k19LocalCache + math constants + torch defaults are canonical compress-time bakes |
| Roundtrip byte-identical | HARD-EARNED | Required by contest discipline; tests pin invariant |
| Composition alpha additive across stages | HARD-EARNED | Catalog #227 substrate_composition_matrix is canonical alpha source |
| Inflate.py LOC overhead linear in side_bytes_compressed_baked | CARGO-CULTED | Side stream baked as ``constants_blob = b"..."``; LOC overhead is constant ~10 LOC per stage; the data is hex-encoded in the literal so size ~ 2x raw |

## Predicted ΔS band

Per CLAUDE.md Catalog #296 (Dykstra feasibility): predicted band derives
from the sister prober anchor (pr101_state_dict + pr106_state_dict at
0.47 each). Composition with the FEC6 0.19205 frontier is gated by Catalog
#227 alpha (currently 1.0 default = ADDITIVE assumption pending paired smoke):

| Substrate | Per-substrate ΔS | Composed (alpha=1.0) | Composed (alpha=0.5) |
|---|---|---|---|
| pr101_state_dict | −0.0470 | −0.0470 | −0.0235 |
| pr106_state_dict | −0.0470 | −0.0940 | −0.0235 |
| **Stacked both** | — | **−0.0940** | **−0.0470** |

Shannon citation: Wyner-Ziv 1976 R(D|Y) lower bound. Dykstra-feasibility
check: side-info derivation MUST satisfy convex constraint (Comma2k19 chunk
size ≤ disk budget AND inflate.py LOC ≤ 100). When alpha is unknown,
:func:`estimate_composition_alpha` returns 1.0 (additive prior) and the
sister probe-disambiguator
``tools/probe_wyner_ziv_composition_alpha_disambiguator.py`` is the canonical
empirical arbiter.

Cross-references
================

- :mod:`tac.wyner_ziv_deliverability.proof_builder` — per-substrate
  post-hoc analyzer (Q1 sister). Result is fed to a runtime LIKE this
  primitive at the consuming substrate's trainer.
- :mod:`tac.sensitivity_map.wyner_ziv_reweight` — per-byte sensitivity
  reweight (Wire-in #2). Sister consumer of WZ classification.
- :mod:`tac.master_gradient_consumers` — Lagrangian-dual planner with
  ``TREATMENT_WYNER_ZIV_HOIST`` (Consumer 15). This primitive IS the
  canonical implementation behind that treatment.
- :mod:`tac.substrates.pretrained_driving_prior.local_chunk_cache` —
  ``Comma2k19LocalCache.fetch_chunk`` per Catalog #213.
- :mod:`tac.optimization.substrate_composition_matrix` — Catalog #227
  composition alpha source.
- ``.omx/research/wyner_ziv_optimal_implementation_queue_20260517.md``
  Q1-Q5 implementation queue from the symposium.
"""

from __future__ import annotations

import dataclasses
import hashlib
import lzma
import math
import time
import zlib
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid runtime import cycle / hard dep on brotli at import
    pass


__all__ = [
    "WYNER_ZIV_LAYER_SCHEMA_VERSION",
    "WYNER_ZIV_LAYER_RESULT_SCHEMA_VERSION",
    "LEGAL_SIDE_INFO_SOURCES",
    "LEGAL_MAIN_CODECS",
    "LEGAL_COMPRESSION_CODECS_FOR_SIDE",
    "DEFAULT_INFLATE_PY_LOC_BUDGET",
    "DEFAULT_INFLATE_PY_LOC_WAIVER_LIMIT",
    "InterceptLocation",
    "WynerZivLayerConfig",
    "WynerZivLayerResult",
    "WynerZivLayerError",
    "ScorerSideInfoForbiddenError",
    "insert_wyner_ziv_layer",
    "reconstruct_from_wyner_ziv_layer",
    "derive_side_info_from_canonical_source",
    "estimate_composition_alpha",
    "estimate_inflate_py_loc_overhead",
    "CONTEST_RATE_DENOM_BYTES",
]


# ---------------------------------------------------------------------------
# Canonical constants
# ---------------------------------------------------------------------------

WYNER_ZIV_LAYER_SCHEMA_VERSION = "wyner_ziv_layer_v1"
WYNER_ZIV_LAYER_RESULT_SCHEMA_VERSION = "wyner_ziv_layer_result_v1"

#: Per CLAUDE.md "Contest scoring": canonical rate-term denominator.
CONTEST_RATE_DENOM_BYTES: int = 37_545_489

#: Per CLAUDE.md HNeRV parity L4: inflate.py ≤ 100 LOC default budget.
DEFAULT_INFLATE_PY_LOC_BUDGET: int = 100

#: Per CLAUDE.md HNeRV parity L4: ≤ 200 LOC with explicit operator waiver.
DEFAULT_INFLATE_PY_LOC_WAIVER_LIMIT: int = 200


LEGAL_SIDE_INFO_SOURCES: frozenset[str] = frozenset({
    "Comma2k19",
    "ImageNet",
    "torch_defaults",
    "math_constants",
    "scorer_compressed",  # FORBIDDEN unless operator_attested
})

LEGAL_MAIN_CODECS: frozenset[str] = frozenset({
    "lzma",
    "brotli",
    "zlib",
    "raw",  # no entropy coding; just pass-through
})

LEGAL_COMPRESSION_CODECS_FOR_SIDE: frozenset[str] = frozenset({
    "lzma",
    "brotli",
    "zlib",
})


# ---------------------------------------------------------------------------
# Intercept location taxonomy
# ---------------------------------------------------------------------------


class InterceptLocation(str, Enum):
    """Where in a compression pipeline the WZ stage is inserted.

    Per the briefing's empirical anchors:

    * STATE_DICT_SERIALIZATION: raw fp16/fp32 weights (pr101 + pr106 case;
      lzma 0.217-0.228 ratio → 0.47 score savings each).
    * QUANTIZER_OUTPUT: pre-entropy int8 latents (Ballé hyperprior latents
      pre-arithmetic; HNeRV-family quantized renderer weights).
    * TRANSFORM_COEFFICIENTS: DCT/wavelet pre-entropy (NSCS06 v8 wavelet
      residuals; SABOR boundary coefficients).
    * CODEBOOK_INDICES: VQ-VAE pre-arithmetic.
    * PREDICTIVE_RESIDUALS: pre-range-coded deltas (P-frame style).
    * HYPERPRIOR_LATENTS: Ballé-style hyperprior z stream.
    """

    STATE_DICT_SERIALIZATION = "state_dict_serialization"
    QUANTIZER_OUTPUT = "quantizer_output"
    TRANSFORM_COEFFICIENTS = "transform_coefficients"
    CODEBOOK_INDICES = "codebook_indices"
    PREDICTIVE_RESIDUALS = "predictive_residuals"
    HYPERPRIOR_LATENTS = "hyperprior_latents"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class WynerZivLayerError(ValueError):
    """Raised when WZ layer inputs / config are malformed."""


class ScorerSideInfoForbiddenError(WynerZivLayerError):
    """Raised when ``side_info_source="scorer_compressed"`` without operator attestation.

    Per CLAUDE.md "Strict scorer rule — non-negotiable" + Catalog #6 + Catalog #320.
    """


# ---------------------------------------------------------------------------
# Config + Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WynerZivLayerConfig:
    """Frozen config for one WZ stage insertion.

    Attributes:
        intercept_location: where in the pipeline this WZ stage sits.
        side_info_source: canonical Y-derivation source.
        side_info_max_bytes: budget for inflate.py-baked side stream.
        main_codec: codec for the main stream (post-WZ-split → archive).
        compression_codec_for_side: codec for the side stream
            (baked into inflate.py).
        composition_alpha_estimate: per Catalog #227; default 1.0 = additive.
        deterministic_seed: seed for split-deterministic encoding.
        operator_attested_scorer_side_info: explicit opt-in for
            ``side_info_source="scorer_compressed"`` per Catalog #320.
        rationale_for_scorer_side_info: required when
            ``operator_attested_scorer_side_info=True``.

    Per CLAUDE.md "Forbidden device-selection defaults" + Catalog #6:
    default is fail-closed; ``scorer_compressed`` is FORBIDDEN unless
    explicitly opted in.
    """

    intercept_location: InterceptLocation
    side_info_source: str
    side_info_max_bytes: int
    main_codec: str = "lzma"
    compression_codec_for_side: str = "lzma"
    composition_alpha_estimate: float = 1.0
    deterministic_seed: int = 0
    operator_attested_scorer_side_info: bool = False
    rationale_for_scorer_side_info: str = ""
    schema_version: str = WYNER_ZIV_LAYER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.intercept_location, InterceptLocation):
            raise WynerZivLayerError(
                f"intercept_location must be InterceptLocation; got "
                f"{type(self.intercept_location).__name__}"
            )
        if self.side_info_source not in LEGAL_SIDE_INFO_SOURCES:
            raise WynerZivLayerError(
                f"side_info_source must be one of {sorted(LEGAL_SIDE_INFO_SOURCES)!r}; "
                f"got {self.side_info_source!r}"
            )
        if not isinstance(self.side_info_max_bytes, int) or self.side_info_max_bytes < 0:
            raise WynerZivLayerError(
                f"side_info_max_bytes must be a non-negative int; got "
                f"{self.side_info_max_bytes!r}"
            )
        if self.main_codec not in LEGAL_MAIN_CODECS:
            raise WynerZivLayerError(
                f"main_codec must be one of {sorted(LEGAL_MAIN_CODECS)!r}; got "
                f"{self.main_codec!r}"
            )
        if self.compression_codec_for_side not in LEGAL_COMPRESSION_CODECS_FOR_SIDE:
            raise WynerZivLayerError(
                f"compression_codec_for_side must be one of "
                f"{sorted(LEGAL_COMPRESSION_CODECS_FOR_SIDE)!r}; got "
                f"{self.compression_codec_for_side!r}"
            )
        if not (0.0 <= float(self.composition_alpha_estimate) <= 1.0):
            raise WynerZivLayerError(
                f"composition_alpha_estimate must be in [0.0, 1.0]; got "
                f"{self.composition_alpha_estimate!r}"
            )
        if self.side_info_source == "scorer_compressed":
            if not self.operator_attested_scorer_side_info:
                raise ScorerSideInfoForbiddenError(
                    "side_info_source='scorer_compressed' is FORBIDDEN unless "
                    "operator_attested_scorer_side_info=True per CLAUDE.md "
                    "'Strict scorer rule — non-negotiable' + Catalog #6 + Catalog #320. "
                    "Pass operator_attested_scorer_side_info=True + "
                    "rationale_for_scorer_side_info=<reason> to opt in."
                )
            if not self.rationale_for_scorer_side_info or len(
                self.rationale_for_scorer_side_info.strip()
            ) < 4:
                raise ScorerSideInfoForbiddenError(
                    "operator_attested_scorer_side_info=True requires a "
                    "non-empty rationale_for_scorer_side_info (>=4 chars); got "
                    f"{self.rationale_for_scorer_side_info!r}"
                )


@dataclass(frozen=True)
class WynerZivLayerResult:
    """Frozen result of one WZ stage insertion.

    Per CLAUDE.md "Apples-to-apples evidence discipline": this is a
    PREDICTION. Defaults are ``evidence_grade="predicted"``, ``score_claim=False``,
    ``promotion_eligible=False``. Promotion to a contest score claim requires
    a paired CUDA + CPU auth-eval anchor per CLAUDE.md "Submission auth eval
    — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE".
    """

    main_bytes_raw: int
    main_bytes_compressed: int
    side_bytes_raw: int
    side_bytes_compressed_baked: int
    score_savings_estimate: float
    inflate_py_loc_added: int
    decoder_complexity_estimate_seconds: float
    intercept_location: InterceptLocation
    side_info_sha256: str
    main_bytes_sha256: str
    config: WynerZivLayerConfig
    evidence_grade: str = "predicted"
    score_claim: bool = False
    promotion_eligible: bool = False
    schema_version: str = WYNER_ZIV_LAYER_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for attr in (
            "main_bytes_raw",
            "main_bytes_compressed",
            "side_bytes_raw",
            "side_bytes_compressed_baked",
            "inflate_py_loc_added",
        ):
            v = getattr(self, attr)
            if not isinstance(v, int) or v < 0:
                raise WynerZivLayerError(
                    f"{attr} must be a non-negative int; got {v!r}"
                )
        if not isinstance(self.decoder_complexity_estimate_seconds, float):
            raise WynerZivLayerError(
                f"decoder_complexity_estimate_seconds must be float; got "
                f"{type(self.decoder_complexity_estimate_seconds).__name__}"
            )
        if self.decoder_complexity_estimate_seconds < 0.0:
            raise WynerZivLayerError(
                f"decoder_complexity_estimate_seconds must be >= 0; got "
                f"{self.decoder_complexity_estimate_seconds!r}"
            )
        if self.evidence_grade not in {"predicted", "empirical_cpu", "empirical_paired_cuda"}:  # CUSTODY_VALIDATOR_OK:this_function_IS_WynerZivLayer_dataclass_post_init_validator_raising_on_invalid_evidence_grade_per_comprehensive_bug_audit_cascade_20260526
            raise WynerZivLayerError(
                f"evidence_grade must be one of "
                f"{{'predicted','empirical_cpu','empirical_paired_cuda'}}; got "
                f"{self.evidence_grade!r}"
            )
        if self.promotion_eligible and self.evidence_grade == "predicted":
            raise WynerZivLayerError(
                "promotion_eligible=True requires evidence_grade in "
                "{'empirical_cpu', 'empirical_paired_cuda'}; got "
                f"evidence_grade='predicted'"
            )
        if self.promotion_eligible and not self.score_claim:
            raise WynerZivLayerError(
                "promotion_eligible=True requires score_claim=True"
            )
        for attr in ("side_info_sha256", "main_bytes_sha256"):
            v = getattr(self, attr)
            if not isinstance(v, str) or len(v) != 64:
                raise WynerZivLayerError(
                    f"{attr} must be 64-char hex sha256; got {v!r}"
                )
            try:
                int(v, 16)
            except (TypeError, ValueError) as exc:
                raise WynerZivLayerError(
                    f"{attr} must be hex; got {v!r}"
                ) from exc

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict (config is recursively serialized)."""
        d = dataclasses.asdict(self)
        # Enum → str
        d["intercept_location"] = self.intercept_location.value
        if "intercept_location" in d.get("config", {}):
            d["config"]["intercept_location"] = self.config.intercept_location.value
        return d


# ---------------------------------------------------------------------------
# Side-info derivation (canonical sources)
# ---------------------------------------------------------------------------


_MATH_CONSTANTS_BLOB: bytes | None = None
_TORCH_DEFAULTS_BLOB: bytes | None = None


def _math_constants_blob() -> bytes:
    """Canonical deterministic blob from math constants (π, e, √2, Catalan).

    Each constant is rendered to 200 decimal digits and concatenated. The
    resulting blob is byte-deterministic across Python versions because
    Python's float is IEEE-754 double + ``str.format`` is locale-independent.
    Memoized at module-import-time avoidance: lazy first-call.
    """
    global _MATH_CONSTANTS_BLOB
    if _MATH_CONSTANTS_BLOB is None:
        from decimal import Decimal, getcontext

        getcontext().prec = 220
        # Canonical 200-digit decimal expansions hard-coded for byte-determinism
        # across Python builds.
        pi_str = (
            "3.14159265358979323846264338327950288419716939937510"
            "58209749445923078164062862089986280348253421170679"
            "82148086513282306647093844609550582231725359408128"
            "48111745028410270193852110555964462294895493038196"
        )
        e_str = (
            "2.71828182845904523536028747135266249775724709369995"
            "95749669676277240766303535475945713821785251664274"
            "27466391932003059921817413596629043572900334295260"
            "59563073813232862794349076323382988075319525101901"
        )
        sqrt2_str = (
            "1.41421356237309504880168872420969807856967187537694"
            "80731766797379907324784621070388503875343276415727"
            "35013846230912297024924836055850737212644121497099"
            "93583141322266592750559275579995050115278206057147"
        )
        catalan_str = (
            "0.91596559417721901505460351493238411077414937428167"
            "21342664981196217630197762547694793565129261151062"
            "48574422619196199579035898803325859059431594737481"
            "15840699533202877331946051903872747816408786590902"
        )
        blob = (pi_str + "|" + e_str + "|" + sqrt2_str + "|" + catalan_str).encode("ascii")
        _MATH_CONSTANTS_BLOB = blob
    return _MATH_CONSTANTS_BLOB


def _torch_defaults_blob() -> bytes:
    """Canonical deterministic blob of common torch.zeros / torch.ones patterns.

    Returns a fixed 1 KB blob of alternating zero / one bytes — proxy for a
    real ``torch.zeros((H, W), dtype=torch.uint8).numpy().tobytes()`` baked
    constant. Inflate.py side typically reconstructs via numpy.zeros /
    numpy.ones with no archive cost.
    """
    global _TORCH_DEFAULTS_BLOB
    if _TORCH_DEFAULTS_BLOB is None:
        # 512 zeros + 512 ones (deterministic 1 KB)
        _TORCH_DEFAULTS_BLOB = b"\x00" * 512 + b"\x01" * 512
    return _TORCH_DEFAULTS_BLOB


def derive_side_info_from_canonical_source(source: str) -> bytes:
    """Canonical Y-derivation helper.

    Args:
        source: one of ``"Comma2k19" | "ImageNet" | "torch_defaults" | "math_constants"``.

    Returns:
        Deterministic bytes derivable at inflate time WITHOUT external state
        (except for ``"Comma2k19"`` which routes through
        :class:`tac.substrates.pretrained_driving_prior.local_chunk_cache.Comma2k19LocalCache`
        per Catalog #213 at compress time; the bytes are baked into the
        inflate.py constants blob).

    Raises:
        WynerZivLayerError: ``source`` is not in :data:`LEGAL_SIDE_INFO_SOURCES`
            minus ``"scorer_compressed"`` (which is never derivable via this
            helper).
        RuntimeError: ``"Comma2k19"`` requested but the canonical helper
            cannot satisfy it (e.g. offline mode + chunk not cached).
    """
    if source == "scorer_compressed":
        raise WynerZivLayerError(
            "derive_side_info_from_canonical_source does not derive scorer bytes "
            "(strict-scorer-rule per Catalog #6); use Comma2k19 / ImageNet / "
            "torch_defaults / math_constants instead."
        )
    if source not in LEGAL_SIDE_INFO_SOURCES:
        raise WynerZivLayerError(
            f"side_info source must be one of {sorted(LEGAL_SIDE_INFO_SOURCES - {'scorer_compressed'})!r}; "
            f"got {source!r}"
        )
    if source == "math_constants":
        return _math_constants_blob()
    if source == "torch_defaults":
        return _torch_defaults_blob()
    if source == "ImageNet":
        # Canonical ImageNet RGB mean / std baked as fp32 bytes.
        # mean = [0.485, 0.456, 0.406] in RGB order; std = [0.229, 0.224, 0.225]
        import struct

        return struct.pack(
            ">6f", 0.485, 0.456, 0.406, 0.229, 0.224, 0.225
        )
    if source == "Comma2k19":
        # Per Catalog #213: route through Comma2k19LocalCache.fetch_chunk.
        # Compress-time only; bytes are baked into inflate.py via the side
        # stream so the runtime does not need network / disk access.
        from tac.substrates.pretrained_driving_prior.local_chunk_cache import (
            Comma2k19LocalCache,
        )

        cache = Comma2k19LocalCache()
        chunk_ids = cache.list_available_chunks()
        if not chunk_ids:
            raise RuntimeError(
                "Comma2k19LocalCache has no available chunks; cannot derive "
                "side-info from Comma2k19 source. Check chunk_manifest."
            )
        path = cache.fetch_chunk(chunk_ids[0])
        # Return the first 4 KB of the chunk's bytes (deterministic;
        # caller's side_info_max_bytes determines how much is actually baked).
        with open(path, "rb") as f:
            return f.read(4096)
    raise WynerZivLayerError(f"unreachable: unhandled source {source!r}")


# ---------------------------------------------------------------------------
# Encoder + Decoder
# ---------------------------------------------------------------------------


def _compress(codec: str, data: bytes) -> bytes:
    if codec == "lzma":
        return lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)
    if codec == "brotli":
        try:
            import brotli  # type: ignore
        except ImportError as exc:
            raise WynerZivLayerError(
                "brotli codec selected but brotli package is not installed"
            ) from exc
        return brotli.compress(data, quality=11)
    if codec == "zlib":
        return zlib.compress(data, level=9)
    if codec == "raw":
        return data
    raise WynerZivLayerError(f"unknown codec {codec!r}")


def _decompress(codec: str, data: bytes) -> bytes:
    if codec == "lzma":
        return lzma.decompress(data)
    if codec == "brotli":
        try:
            import brotli  # type: ignore
        except ImportError as exc:
            raise WynerZivLayerError(
                "brotli codec selected but brotli package is not installed"
            ) from exc
        return brotli.decompress(data)
    if codec == "zlib":
        return zlib.decompress(data)
    if codec == "raw":
        return data
    raise WynerZivLayerError(f"unknown codec {codec!r}")


def _xor_bytes(a: bytes, b: bytes) -> bytes:
    """XOR two equal-length byte strings."""
    if len(a) != len(b):
        raise WynerZivLayerError(
            f"_xor_bytes length mismatch: {len(a)} vs {len(b)}"
        )
    return bytes(x ^ y for x, y in zip(a, b, strict=True))


def _detect_y_derivable_prefix(pre_entropy_bytes: bytes, side_info_y: bytes) -> int:
    """Return the length of the longest prefix of pre_entropy_bytes that is
    byte-identical to a window of side_info_y.

    The honest WZ split this primitive operationalizes: any prefix of
    pre_entropy_bytes that already exists as a contiguous substring of Y
    is REMOVABLE from main (because the decoder can recover it from Y
    directly via offset lookup). The side stream carries the (offset,
    length) tuple needed for the decoder to extract the prefix from Y.

    This is the simplest faithful realization of the Wyner-Ziv R(D|Y)
    inequality at the pipeline-stage primitive surface: the bytes Y
    "covers" are the bytes archive does not need to encode.

    Returns:
        Length of the longest Y-covered prefix (0 if none).
    """
    if not pre_entropy_bytes or not side_info_y:
        return 0
    # Scan Y for the start byte of pre_entropy; for any match try to extend.
    max_match_len = 0
    first_byte = pre_entropy_bytes[0:1]
    start = 0
    while True:
        idx = side_info_y.find(first_byte, start)
        if idx < 0:
            break
        # Extend the match
        match_len = 0
        while (
            match_len < len(pre_entropy_bytes)
            and idx + match_len < len(side_info_y)
            and pre_entropy_bytes[match_len] == side_info_y[idx + match_len]
        ):
            match_len += 1
        if match_len > max_match_len:
            max_match_len = match_len
        start = idx + 1
    return max_match_len


def insert_wyner_ziv_layer(
    *,
    pre_entropy_bytes: bytes,
    side_info_y: bytes,
    config: WynerZivLayerConfig,
) -> WynerZivLayerResult:
    """Insert a WZ stage between pre-entropy data and the post-entropy archive.

    The honest split scheme this primitive operationalizes:

    * **Y-derivable prefix detection**: find the longest prefix of
      ``pre_entropy_bytes`` that is byte-identical to a contiguous window
      of ``side_info_y``. That prefix is REMOVED from ``main`` because the
      decoder can recover it from Y directly via offset lookup.
    * **side stream**: a 16-byte tuple ``(offset_in_y, prefix_length)``
      indicating where the prefix lives in Y. If no overlap is found, side
      is empty and main = pre_entropy_bytes (the WZ stage is a no-op).
    * **main stream**: the remainder of ``pre_entropy_bytes`` after the
      Y-derivable prefix is removed, fed to ``config.main_codec``.

    This realizes the Wyner-Ziv R(D|Y) inequality: bytes Y "covers"
    contribute zero archive cost; only the residual goes through the
    entropy coder. Production callers with richer Y-derivation schemes
    (e.g. per-channel codebook lookup, per-frame motion compensation) can
    pre-process ``pre_entropy_bytes`` and ``side_info_y`` to maximize
    overlap before invoking this primitive.

    Args:
        pre_entropy_bytes: the bytes the existing pipeline would have fed
            into its entropy coder.
        side_info_y: the side information Y available to the decoder at
            inflate time.
        config: :class:`WynerZivLayerConfig`.

    Returns:
        :class:`WynerZivLayerResult` with byte counts, score-savings
        estimate, LOC overhead, and decoder complexity estimate.

    Raises:
        WynerZivLayerError: malformed inputs.
        ScorerSideInfoForbiddenError: validated by ``WynerZivLayerConfig``.
    """
    if not isinstance(pre_entropy_bytes, (bytes, bytearray)):
        raise WynerZivLayerError(
            f"pre_entropy_bytes must be bytes; got "
            f"{type(pre_entropy_bytes).__name__}"
        )
    if not isinstance(side_info_y, (bytes, bytearray)):
        raise WynerZivLayerError(
            f"side_info_y must be bytes; got {type(side_info_y).__name__}"
        )
    pre_entropy = bytes(pre_entropy_bytes)
    y_bytes = bytes(side_info_y)

    # Find the longest Y-derivable prefix of pre_entropy.
    prefix_len = _detect_y_derivable_prefix(pre_entropy, y_bytes)
    if prefix_len > 0:
        # Find the offset where the prefix matches in Y.
        offset_in_y = y_bytes.find(pre_entropy[:prefix_len])
    else:
        offset_in_y = 0

    main_raw = pre_entropy[prefix_len:]  # archive-side bytes (smaller)
    # Side stream encodes (offset_in_y, prefix_len) as two 8-byte big-endian
    # integers. Decoder reconstructs the prefix via Y[offset:offset+len].
    side_raw = offset_in_y.to_bytes(8, "big") + prefix_len.to_bytes(8, "big")

    main_compressed = _compress(config.main_codec, main_raw)
    side_compressed_baked = _compress(config.compression_codec_for_side, side_raw)

    if len(side_compressed_baked) > config.side_info_max_bytes:
        raise WynerZivLayerError(
            f"side_bytes_compressed_baked ({len(side_compressed_baked)} bytes) "
            f"exceeds side_info_max_bytes ({config.side_info_max_bytes}). "
            f"Lower side_info_max_bytes or pick a denser compression_codec_for_side."
        )

    # Score-savings estimate = (pre_entropy bytes - main_compressed bytes)
    # at the contest rate-term scale. If main_compressed > pre_entropy
    # (degenerate; should not happen with sensible codec), savings is
    # negative (no benefit; promotion-side discipline will refuse it).
    bytes_saved = len(pre_entropy) - len(main_compressed)
    score_savings_estimate = 25.0 * float(bytes_saved) / float(CONTEST_RATE_DENOM_BYTES)

    # Inflate.py LOC overhead: side stream is baked as a hex literal +
    # one decompress call. ~10 LOC structural + 1 LOC per ~80 chars of
    # hex literal (target 100-char wrap per PEP 8). hex doubles size.
    inflate_py_loc_added = estimate_inflate_py_loc_overhead(
        side_bytes_compressed_baked=len(side_compressed_baked),
        compression_codec_for_side=config.compression_codec_for_side,
    )

    # Decoder complexity: measure one decompress call wall-clock.
    t0 = time.perf_counter()
    _decompressed = _decompress(config.main_codec, main_compressed)
    main_decompress_t = time.perf_counter() - t0
    # XOR runtime is O(N) but negligible for typical pipelines.
    decoder_complexity_estimate_seconds = float(main_decompress_t)

    side_info_sha256 = hashlib.sha256(y_bytes).hexdigest()
    main_bytes_sha256 = hashlib.sha256(main_raw).hexdigest()

    return WynerZivLayerResult(
        main_bytes_raw=len(main_raw),
        main_bytes_compressed=len(main_compressed),
        side_bytes_raw=len(side_raw),
        side_bytes_compressed_baked=len(side_compressed_baked),
        score_savings_estimate=score_savings_estimate,
        inflate_py_loc_added=inflate_py_loc_added,
        decoder_complexity_estimate_seconds=decoder_complexity_estimate_seconds,
        intercept_location=config.intercept_location,
        side_info_sha256=side_info_sha256,
        main_bytes_sha256=main_bytes_sha256,
        config=config,
        evidence_grade="predicted",
        score_claim=False,
        promotion_eligible=False,
    )


def reconstruct_from_wyner_ziv_layer(
    *,
    main_compressed: bytes,
    side_compressed_baked: bytes,
    side_info_y: bytes,
    config: WynerZivLayerConfig,
) -> bytes:
    """Decoder path. Reconstruct ``pre_entropy_bytes`` from (main, side, Y).

    Args:
        main_compressed: compressed main stream from the archive.
        side_compressed_baked: compressed side stream baked into inflate.py.
        side_info_y: side info available at inflate time (must equal the
            ``side_info_y`` passed at encode time per the contest
            determinism requirement).
        config: same :class:`WynerZivLayerConfig` used at encode time.

    Returns:
        The original ``pre_entropy_bytes`` (byte-identical to encoder input).

    Raises:
        WynerZivLayerError: malformed inputs or codec mismatch.
    """
    if not isinstance(main_compressed, (bytes, bytearray)):
        raise WynerZivLayerError(
            f"main_compressed must be bytes; got "
            f"{type(main_compressed).__name__}"
        )
    if not isinstance(side_compressed_baked, (bytes, bytearray)):
        raise WynerZivLayerError(
            f"side_compressed_baked must be bytes; got "
            f"{type(side_compressed_baked).__name__}"
        )
    if not isinstance(side_info_y, (bytes, bytearray)):
        raise WynerZivLayerError(
            f"side_info_y must be bytes; got {type(side_info_y).__name__}"
        )

    main_raw = _decompress(config.main_codec, bytes(main_compressed))
    side_raw = _decompress(
        config.compression_codec_for_side, bytes(side_compressed_baked)
    )
    # Side stream schema: 16-byte (offset_in_y, prefix_len) tuple.
    if len(side_raw) != 16:
        raise WynerZivLayerError(
            f"side stream malformed: expected 16-byte (offset, prefix_len) "
            f"tuple per WynerZivLayerConfig v1 schema; got {len(side_raw)} bytes"
        )
    offset_in_y = int.from_bytes(side_raw[:8], "big")
    prefix_len = int.from_bytes(side_raw[8:16], "big")
    y_bytes = bytes(side_info_y)
    if prefix_len > 0:
        if offset_in_y + prefix_len > len(y_bytes):
            raise WynerZivLayerError(
                f"side stream points past Y end: offset={offset_in_y} + "
                f"prefix_len={prefix_len} > len(Y)={len(y_bytes)}. "
                f"side_info_y at decode time differs from encode time."
            )
        prefix_bytes = y_bytes[offset_in_y : offset_in_y + prefix_len]
    else:
        prefix_bytes = b""
    return prefix_bytes + main_raw


# ---------------------------------------------------------------------------
# Inflate.py LOC overhead estimator
# ---------------------------------------------------------------------------


def estimate_inflate_py_loc_overhead(
    *,
    side_bytes_compressed_baked: int,
    compression_codec_for_side: str = "lzma",
) -> int:
    """Estimate inflate.py LOC added by baking ``side_bytes_compressed_baked``.

    Per CLAUDE.md HNeRV parity discipline L4 (≤ 100 LOC default, ≤ 200 LOC
    with operator waiver), this is the structural budget audit surface.

    Args:
        side_bytes_compressed_baked: byte count of the compressed side stream
            to bake into inflate.py.
        compression_codec_for_side: codec name (affects ``import`` LOC).

    Returns:
        Integer LOC estimate. Linear in ``side_bytes_compressed_baked`` with
        a small constant overhead for the import / decompress / reconstruct
        scaffold.
    """
    if not isinstance(side_bytes_compressed_baked, int) or side_bytes_compressed_baked < 0:
        raise WynerZivLayerError(
            f"side_bytes_compressed_baked must be non-negative int; got "
            f"{side_bytes_compressed_baked!r}"
        )
    # Structural overhead per WZ stage:
    #   1: import line for codec
    #   3: from tac.codec.wyner_ziv_layer import (reconstruct_from_wyner_ziv_layer, ...)
    #   2: side_info_y derivation
    #   2: reconstruct call + assignment
    #   2: small surrounding scaffold (blank lines, comment)
    structural_loc = 10
    # Per-byte: hex literal doubles the byte count (each byte → 2 hex chars).
    # Wrap at 100 chars per line per PEP 8. So LOC for the literal ≈
    # ceil(side_bytes_compressed_baked * 2 / 100).
    if side_bytes_compressed_baked == 0:
        literal_loc = 1  # b"" literal still takes 1 LOC
    else:
        chars = side_bytes_compressed_baked * 2
        literal_loc = math.ceil(chars / 100)
    return int(structural_loc + literal_loc)


# ---------------------------------------------------------------------------
# Composition alpha estimator (Catalog #227)
# ---------------------------------------------------------------------------


def estimate_composition_alpha(
    config_a: WynerZivLayerConfig,
    config_b: WynerZivLayerConfig,
) -> float:
    """Estimate composition_alpha between two WZ stages or WZ + sibling stage.

    Per Catalog #227 substrate_composition_matrix the canonical alpha is
    derived empirically via paired smoke. This primitive returns a
    HEURISTIC prior (which the autopilot ranker is allowed to consume so
    long as it routes through the canonical helper at
    :mod:`tac.optimization.substrate_composition_matrix` when an empirical
    posterior exists for the (config_a, config_b) pair).

    Heuristic rules (priors only; empirical posterior overrides):

    * Same intercept_location → ``0.30`` (SATURATING: two stages at the
      same intercept can only redundantly compress the same axis).
    * Different intercept_location but same side_info_source → ``0.60``
      (SUB-ADDITIVE: shared Y means the second stage extracts marginal gain
      conditional on the first).
    * Different intercept_location AND different side_info_source → ``1.00``
      (ADDITIVE: orthogonal axes).

    The autopilot ranker per ``adjust_predicted_delta_for_composition_alpha``
    in :mod:`tools.cathedral_autopilot_autonomous_loop` treats α > 0.7 as
    ADDITIVE, 0.3-0.7 as SUB-ADDITIVE (halve marginal gain), ≤ 0.3 as
    SATURATING (floor at -0.005).
    """
    if not isinstance(config_a, WynerZivLayerConfig):
        raise WynerZivLayerError(
            f"config_a must be WynerZivLayerConfig; got {type(config_a).__name__}"
        )
    if not isinstance(config_b, WynerZivLayerConfig):
        raise WynerZivLayerError(
            f"config_b must be WynerZivLayerConfig; got {type(config_b).__name__}"
        )
    if config_a.intercept_location == config_b.intercept_location:
        return 0.30
    if config_a.side_info_source == config_b.side_info_source:
        return 0.60
    return 1.00
