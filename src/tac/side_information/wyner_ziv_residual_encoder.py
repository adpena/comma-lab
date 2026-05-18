# SPDX-License-Identifier: MIT
"""WynerZivResidualEncoder — encode the RESIDUAL X - f(Y) where f is a
function the decoder can compute from side info Y (Wyner-Ziv 1976
classical compression gain).

Per ``.omx/research/meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md``
§J row 2 (Optical-flow side-information — RAFT-derived per-pair flow
baked into archive; decoder uses to predict frame_1 from frame_0,
Wyner-Ziv classical) + CLAUDE.md "Substrate scaffolds MUST be COMPLETE
or RESEARCH-ONLY" + Wyner-Ziv 1976 theorem.

The Wyner-Ziv 1976 theorem (verbatim):
  Rate(X|Y) <= R(D) where R(D) is the conditional rate-distortion
  function and Y is side information available at the decoder. The
  compression gain over the unconditional R(D|nothing) is
  Rate(X) - Rate(X|Y) = I(X;Y) - I(X;Y|D).

In practice for the contest:
  - Y = a SHARED PRIOR computable by the decoder from baked-in inflate.py
    constants (scorer-features / Comma2k19-palette / ImageNet-statistics
    / dashcam-priors — i.e. ALL THE OTHER 4 BAKERS in this namespace).
  - f(Y) = a reconstruction the decoder can produce from Y alone.
  - X - f(Y) = the residual the encoder must transmit in the archive.
  - The bytes saved = (original X bytes) - (residual encoding bytes).

This baker is UNIQUE because it actually DOES add archive bytes (unlike
the 4 shared-prior bakers which add zero archive bytes). The residual
encoding is in the archive; the f(Y) function lives in inflate.py +
consumes the shared-prior bytes ALSO in inflate.py.

The unique-per-method engineering surface is:
  - WHICH shared-prior is consumed (which sister baker emits Y)
  - HOW f(Y) is computed (linear predictor, neural network, geometric
    transform, optical flow warp, etc.)
  - HOW the residual is encoded (entropy code, scalar quantization, etc.)

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L7:
the residual encoder is a CODEC bolt-on (≤ 350 LOC) on top of the
shared-prior bake (substrate engineering). The two-layer split is
canonical.

This baker is the canonical case of:
  - stage_phase="compress" (the residual encoding happens at compress;
    the inflate-side reconstruction f(Y) + residual decode happens in
    inflate.py — but inflate.py SHOULD NOT need a separate baker for
    that since the reconstruction is just the inverse of the encoding)
  - side_info_source="wyner_ziv_residual"
  - side_info_reproducible=True (the SHARED PRIOR Y is derived from a
    sister-baker that already enforces reproducibility)
  - scorer_free=True (residual encoding does NOT load contest scorer)
  - archive_bytes_added > 0 (the residual IS in the archive)
  - inflate_runtime_bytes_added > 0 (the reconstruction function f(Y) +
    residual decoder live in inflate.py)
"""

from __future__ import annotations

from dataclasses import dataclass

from tac.side_information.contract import (
    SideInfoBakerContract,
)

__all__ = [
    "LEGAL_RECONSTRUCTION_FN",
    "LEGAL_RESIDUAL_CODE",
    "WynerZivResidualEncoder",
    "WynerZivResidualEncoderSpec",
]

LEGAL_RECONSTRUCTION_FN: frozenset[str] = frozenset(
    {
        # Linear predictor: f(Y) = A @ Y where A is a baked matrix.
        "linear_predictor",
        # Palette quantization: f(Y) = palette[nearest_index(X | Y)].
        "palette_quantization",
        # Optical-flow warp: f(Y) = warp(frame_0, optical_flow_from_Y).
        "optical_flow_warp",
        # Geometric transform: f(Y) = transform(X | pose_from_Y).
        "geometric_transform",
        # Per-class detail: f(Y) = class_template[seg_class_from_Y].
        "per_class_detail",
        # Custom: f(Y) is operator-attested (substrate-specific).
        "custom",
    }
)

LEGAL_RESIDUAL_CODE: frozenset[str] = frozenset(
    {
        # Arithmetic coding over residual symbols.
        "arithmetic",
        # Range coding.
        "range",
        # Asymmetric numeral systems (ANS / FSE).
        "ans",
        # Brotli compression of residual bytes.
        "brotli",
        # Huffman coding (e.g. fec6 family).
        "huffman",
        # Block-FP scalar quantization + brotli.
        "block_fp_brotli",
        # Custom: operator-attested encoding.
        "custom",
    }
)


@dataclass(frozen=True)
class WynerZivResidualEncoderSpec:
    """Specification for a Wyner-Ziv residual-encoder baker.

    Frozen so spec composition is structurally immutable. The
    reconstruction function + residual code + size estimates are pinned
    at decoration time for byte-stable reproducibility per Catalog #158.

    The ``shared_prior_baker_id`` field NAMES the sister baker (one of
    the 4 shared-prior bakers in this namespace) whose Y this residual
    encoder consumes. The pipeline validator confirms the named baker is
    registered AND that this baker is positioned after it.

    The ``wyner_ziv_correlation_estimate`` is the predicted I(X;Y)/H(X)
    ratio — the higher it is, the more compression gain this encoder
    delivers. The cathedral autopilot ranker uses this estimate to
    prioritize residual encoders with higher predicted gain.
    """

    baker_id: str
    shared_prior_baker_id: str  # name of the sister baker emitting Y
    reconstruction_fn: str  # one of LEGAL_RECONSTRUCTION_FN
    residual_code: str  # one of LEGAL_RESIDUAL_CODE
    archive_bytes_added: int  # estimated residual bytes per archive
    inflate_runtime_bytes_added: int = 512  # f(Y) + decoder code size
    wyner_ziv_correlation_estimate: float | None = None
    sensitivity_weighted: bool = False
    seed: int = 42
    correction_resolution: str = "per_pair"
    description: str = ""
    lane_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.shared_prior_baker_id, str
        ) or not self.shared_prior_baker_id.strip():
            raise ValueError(
                f"shared_prior_baker_id={self.shared_prior_baker_id!r} "
                f"must be a non-empty string (the id of the sister baker "
                f"emitting the shared-prior Y this residual encoder "
                f"consumes)"
            )
        if self.reconstruction_fn not in LEGAL_RECONSTRUCTION_FN:
            raise ValueError(
                f"reconstruction_fn={self.reconstruction_fn!r} not in "
                f"{sorted(LEGAL_RECONSTRUCTION_FN)}"
            )
        if self.residual_code not in LEGAL_RESIDUAL_CODE:
            raise ValueError(
                f"residual_code={self.residual_code!r} not in "
                f"{sorted(LEGAL_RESIDUAL_CODE)}"
            )
        if self.archive_bytes_added < 1:
            raise ValueError(
                f"archive_bytes_added={self.archive_bytes_added} must be "
                f">= 1 (Wyner-Ziv residual encoder MUST contribute >= 1 "
                f"archive byte; encoders with zero residual are degenerate)"
            )
        if self.inflate_runtime_bytes_added < 0:
            raise ValueError(
                f"inflate_runtime_bytes_added="
                f"{self.inflate_runtime_bytes_added} must be >= 0"
            )
        if self.wyner_ziv_correlation_estimate is not None:
            if not 0.0 <= self.wyner_ziv_correlation_estimate <= 1.0:
                raise ValueError(
                    f"wyner_ziv_correlation_estimate="
                    f"{self.wyner_ziv_correlation_estimate} must be in "
                    f"[0.0, 1.0]"
                )
        if self.seed < 0:
            raise ValueError(f"seed={self.seed} must be >= 0")


class WynerZivResidualEncoder:
    """Builder for a Wyner-Ziv-residual-encoder baker contract.

    The canonical compress-time + inflate-time cycle:
      1. COMPRESS:
         a. Consume shared-prior Y from the named sister baker
            (``shared_prior_baker_id``).
         b. Compute f(Y) — the decoder's predicted reconstruction.
         c. Compute residual = X - f(Y).
         d. Encode residual via ``residual_code``.
         e. Append residual bytes to the archive.
      2. INFLATE: inflate.py computes f(Y) using the baked shared-prior
         constants (already in inflate.py from the sister baker) +
         decodes the residual + reconstructs X = f(Y) + decoded_residual.

    Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline"
    L1 (score-aware) + Wyner-Ziv 1976: the residual encoder's loss
    SHOULD be aware of the contest scorer's sensitivity — bits allocated
    to score-relevant pixels (high dseg/dpose) yield more compression
    gain than bits on score-irrelevant pixels. The
    ``sensitivity_weighted=True`` flag opts into this; the
    master_gradient hook is auto-threaded by the pipeline.

    Usage::

        from tac.side_information import (
            WynerZivResidualEncoder, WynerZivResidualEncoderSpec,
            side_info_baker,
        )

        spec = WynerZivResidualEncoderSpec(
            baker_id="wz_residual_optical_flow_warp_arithmetic",
            shared_prior_baker_id="comma2k19_optical_flow_palette",
            reconstruction_fn="optical_flow_warp",
            residual_code="arithmetic",
            archive_bytes_added=2048,
            inflate_runtime_bytes_added=512,
            wyner_ziv_correlation_estimate=0.65,
            sensitivity_weighted=True,
            seed=42,
            correction_resolution="per_pair",
            description=(
                "Per-pair Wyner-Ziv residual: f(Y) = optical-flow-warp from "
                "frame_0 using flow estimated via comma2k19-distilled flow "
                "palette; residual = X - f(Y) encoded via arithmetic code."
            ),
            lane_id="lane_wz_optical_flow_residual_20260601",
        )
        contract = WynerZivResidualEncoder(spec=spec).build_contract()

        @side_info_baker(contract)
        def wz_residual_optical_flow_warp_arithmetic(
            state, *, policy, seed=42, master_gradient=None,
        ):
            shared_prior = state["side_info_palette_v1"]
            X = state["per_pair_frames"]
            # f(Y): optical-flow warp using the shared-prior flow palette
            flow = lookup_flow_palette(shared_prior, X)
            f_Y = warp(X[:, 0], flow)
            residual = X[:, 1] - f_Y
            encoded = arithmetic_code(
                residual, weights=master_gradient, seed=seed,
            )
            return {
                "archive_residual_bytes_v1": encoded,
                "archive_bytes_added": len(encoded),
            }
    """

    def __init__(self, *, spec: WynerZivResidualEncoderSpec) -> None:
        if not isinstance(spec, WynerZivResidualEncoderSpec):
            raise TypeError(
                f"spec must be WynerZivResidualEncoderSpec; got "
                f"{type(spec).__name__}"
            )
        self.spec = spec

    def build_contract(self) -> SideInfoBakerContract:
        """Build the SideInfoBakerContract for this Wyner-Ziv encoder baker.

        Emits the canonical pattern:
          - stage_phase="compress" (the residual encode happens at
            compress; the inflate-side reconstruction is performed by the
            substrate's inflate.py decoder, NOT by a separate baker)
          - correction_kind="wyner_ziv_residual_encode"
          - side_info_source="wyner_ziv_residual"
          - side_info_reproducible=True (the shared-prior Y is itself
            reproducibly derived by the sister baker)
          - scorer_free=True (residual encoding does NOT load scorer at
            inflate; sensitivity_weighted MAY consume master_gradient at
            compress but that is NOT a scorer load)
          - archive_bytes_added=spec.archive_bytes_added (>0)
          - inflate_runtime_bytes_added=spec.inflate_runtime_bytes_added
        """
        # The consume key is the shared prior emitted by the sister baker.
        # We use the canonical name "side_info_*_v1" to match how the 4
        # shared-prior bakers emit. The actual key name depends on which
        # sister baker is named — the pipeline validator catches a
        # mismatch when the spec's shared_prior_baker_id is registered.
        consumes: frozenset[str] = frozenset(
            {
                # Generic shared-prior key; specific naming depends on
                # which sister baker is named via shared_prior_baker_id.
                "shared_prior_for_residual_v1",
                "per_pair_frames",
            }
        )
        if self.spec.sensitivity_weighted:
            consumes = consumes | frozenset({"master_gradient"})

        # Probe disambiguator if both linear_predictor + arithmetic vs
        # palette_quantization + brotli (etc.) are defensible
        # interpretations for the substrate's reconstruction.
        probe = (
            "tools/probe_wyner_ziv_reconstruction_fn_disambiguator.py"
            if self.spec.reconstruction_fn != "custom"
            else None
        )
        hook_not_applicable_rationale: dict[str, str] = {}
        if probe is None:
            hook_not_applicable_rationale["hook_probe_disambiguator"] = (
                "Custom reconstruction_fn is operator-attested with single "
                "interpretation; no probe disambiguator path required."
            )

        return SideInfoBakerContract(
            id=self.spec.baker_id,
            parent_baker_id=self.spec.shared_prior_baker_id,
            stage_phase="compress",
            description=(
                self.spec.description
                or (
                    f"WynerZivResidualEncoder; shared_prior="
                    f"{self.spec.shared_prior_baker_id!r}; reconstruction="
                    f"{self.spec.reconstruction_fn!r}; residual_code="
                    f"{self.spec.residual_code!r}; archive_bytes_added="
                    f"{self.spec.archive_bytes_added}; "
                    f"inflate_runtime_bytes_added="
                    f"{self.spec.inflate_runtime_bytes_added}; "
                    f"seed={self.spec.seed}."
                )
            ),
            consumes=consumes,
            emits=frozenset({"archive_residual_bytes_v1"}),
            correction_kind="wyner_ziv_residual_encode",
            correction_resolution=self.spec.correction_resolution,
            side_info_source="wyner_ziv_residual",
            side_info_reproducible=True,
            requires_canonical_comma2k19_cache=False,
            wyner_ziv_correlation_estimate=(
                self.spec.wyner_ziv_correlation_estimate
            ),
            deterministic=True,
            scorer_free=True,
            archive_bytes_added=self.spec.archive_bytes_added,
            inflate_runtime_bytes_added=(
                self.spec.inflate_runtime_bytes_added
            ),
            seed=self.spec.seed,
            merge_policy="last_writer_wins",
            hook_sensitivity_contribution=(
                "master_gradient_v1"
                if self.spec.sensitivity_weighted
                else "not_applicable_with_rationale"
            ),
            hook_pareto_constraint="wyner_ziv_rate_distortion_v1",
            hook_bit_allocator_class="wyner_ziv_residual_allocator",
            hook_autopilot_ranker="cathedral_autopilot_v1",
            hook_continual_learning_anchor_kind=(
                "side_information_baker_outcomes_v1"
            ),
            hook_probe_disambiguator=probe,
            hook_not_applicable_rationale={
                **(
                    {}
                    if self.spec.sensitivity_weighted
                    else {
                        "hook_sensitivity_contribution": (
                            "Uniform per-pixel residual encoding (no "
                            "master_gradient weighting); operator opted "
                            "for uniform bits across all residual symbols."
                        )
                    }
                ),
                **hook_not_applicable_rationale,
            },
            lane_id=self.spec.lane_id,
            design_memo=(
                ".omx/research/"
                "meat_on_the_bone_inventory_and_canonical_helpers_design_"
                "20260517.md"
            ),
            canonical_vs_unique_decision=(
                "ADOPT_CANONICAL for the Wyner-Ziv 1976 residual-encoding "
                "framework (encode X - f(Y) instead of X); "
                "FORK_BECAUSE_PRINCIPLED_MISMATCH for the per-substrate "
                "reconstruction_fn and residual_code (linear_predictor + "
                "arithmetic vs optical_flow_warp + ANS vs palette_quant + "
                "brotli each yield different compression gain for "
                "different X / Y joint distributions). The "
                "sensitivity_weighted opt-in routes master_gradient as "
                "the per-symbol bit-allocation weight; without it the "
                "residual code allocates uniform bits."
            ),
        )
