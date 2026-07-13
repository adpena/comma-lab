# SPDX-License-Identifier: MIT
"""Canonical lossless joint-coding law for fixed witness int8 symbols."""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "witness_lossless_cross_tensor_storage_law_v1"
_UTC = "2026-07-13T04:05:00Z"
_RECEIPT = "experiments/results/witness_crosstensor_structure_rate_20260713/measurement_receipt.json"
_MEMO = ".omx/research/witness_crosstensor_structure_rate_20260713.md"
_AXIS = "[macOS-CPU/numpy-fp32 advisory] NON-PROMOTABLE"


def lossless_joint_storage_gain_bytes(identity_archive_bytes: int, joint_archive_bytes: int) -> int:
    """Return the exact archive-byte gain; negative means the joint chart loses."""

    identity = int(identity_archive_bytes)
    joint = int(joint_archive_bytes)
    if identity < 0 or joint < 0:
        raise ValueError("archive byte counts must be nonnegative")
    return identity - joint


def build_witness_lossless_cross_tensor_storage_law_v1() -> CanonicalEquation:
    anchor = EmpiricalAnchor(
        anchor_id="v9_ep150_n600_lossless_joint_storage_20260713",
        measurement_utc=_UTC,
        inputs={
            "checkpoint_sha256": "2599ad8b396af2af220a3bdbeee2ade92f194771ae6ef01a6faa15d39333484c",
            "eval_pairs": 600,
            "base_tensors": 17,
            "base_weight_symbols": 61_175,
            "pair_code_symbols": 22_800,
            "fixed_quantizer": "per-tensor symmetric int8; scales unchanged",
        },
        predicted_output={
            "storage_law": ("min over bijective axis charts and raw-vs-frame-delta pair charts of exact archive bytes"),
            "distortion_invariance": "D(P(q), Delta(q)) = D(q) because both maps are bijections",
        },
        empirical_output={
            "identity_archive_bytes": 63_659,
            "weight_permutation_only_archive_bytes": 63_510,
            "pair_delta_only_archive_bytes": 63_408,
            "joint_archive_bytes": 63_242,
            "joint_archive_bytes_saved": 417,
            "base_brotli_bytes_saved_before_metadata": 180,
            "code_brotli_bytes_saved_before_metadata": 268,
            "advisory_delta_S": -0.0002776631834519455,
            "decoded_quantized_state_sha256": ("c67830f51d58291c7e6f92ef6140fc3599e872b7fb1436874577ea66f32e14fb"),
            "decoded_state_exact_equal": True,
            "n600_component_delta_d_seg": 0.0,
            "n600_component_delta_d_pose": 0.0,
            "component_delta_evidence": ("DERIVED from MEASURED exact equality of every decoded base/code int8 symbol"),
            "receiver_smoke": "MEASURED BIT-EXACT on identity and joint packets, 1 pair/2 frames",
            "shared_codebook_verdict": "NULL_POSTHOC_EXACT_SHARED_VALUE_CODEBOOK",
            "verdict_scope": (
                "INSTANCE x FORMULATION: this n600 checkpoint and fixed symmetric-int8 grid; "
                "not a negative on training-induced tying, low-rank, or VQ-in-loop"
            ),
        },
        residual=0.0,
        source_artifact=_RECEIPT,
        measurement_method="n600_full_quantized_state_parseback_plus_exact_zip_bytes",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_RECEIPT,
            reactivation_criteria=("rederive the chart on any new checkpoint or after a quantizer/grammar change"),
            measurement_axis=_AXIS,
            hardware_substrate="macos_arm64_cpu",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Lossless cross-tensor witness storage over a fixed quantized state",
        one_line_summary=(
            "Choose exact-byte-minimizing bijective storage charts only after fixing q; decoded state and scored components remain invariant."
        ),
        latex_form=(
            r"(P^*,c^*)=\arg\min_{P,c}|\operatorname{ZIP}(M(P,c),"
            r"\operatorname{Br}(\Vert_t P_tq_t),\operatorname{Br}(\Delta_c q^{code}))|;\quad "
            r"P_t^{-1}P_tq_t=q_t,\quad \Delta_c^{-1}\Delta_cq^{code}=q^{code};\quad "
            r"\Delta S=25(B^*-B_0)/37{,}545{,}489"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.witness_crosstensor_coding_20260713:lossless_joint_storage_gain_bytes"
        ),
        domain_of_validity={
            "fixed_surface": "post-quantization fixed per-tensor int8 symbols and scales",
            "admitted_transforms": [
                "bijective 2-D storage-axis permutation",
                "bijective frame-separated modulo-256 temporal delta",
            ],
            "excluded_ownership": "per-tensor bit allocation or precision changes (task #336)",
            "negative_scope": ("post-hoc exact shared value-codebook and exact-row dedup on the named checkpoint only"),
            "not_killed": [
                "training-time weight tying or low-rank factorization",
                "VQ-in-the-loop",
                "task #110 latent-structure regularization",
                "task #242 ideal-config rate objective",
                "other checkpoints or quantizer families",
            ],
            "measurement_axis": _AXIS,
            "score_claim": False,
            "promotion_eligible": False,
        },
        units_in={
            "identity_archive_bytes": "exact_ZIP_bytes",
            "joint_archive_bytes": "exact_ZIP_bytes",
            "q_t": "signed_int8_symbols",
        },
        units_out={
            "gain": "exact_ZIP_bytes",
            "delta_S": "advisory_contest_score_units",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "decoded_state_identity": 0.0,
            "component_delta_from_bijection": 0.0,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.gauge.WitnessCrossTensorCoderGauge",
            "tools.levelset_byte_close_and_eval",
            ".omx.research.sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611",
            "task#110",
            "task#242",
        ),
        canonical_producers=(
            "tools.measure_witness_crosstensor_structure",
            "tac.boundary_math.witness_crosstensor_codec",
            _RECEIPT,
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_RECEIPT,
            reactivation_criteria=(
                "append a superseding anchor after any new witness checkpoint, quantizer, or archive grammar"
            ),
            measurement_axis=_AXIS,
            hardware_substrate="macos_arm64_cpu",
        ),
    )


def populate_witness_lossless_cross_tensor_storage_law_v1(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Append the law through the fcntl-locked canonical registry writer."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_witness_lossless_cross_tensor_storage_law_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="FEED-witness-xcodec: exact n600 state identity; 417 measured archive bytes saved",
    )
    return equation


__all__ = [
    "EQUATION_ID",
    "build_witness_lossless_cross_tensor_storage_law_v1",
    "lossless_joint_storage_gain_bytes",
    "populate_witness_lossless_cross_tensor_storage_law_v1",
]
