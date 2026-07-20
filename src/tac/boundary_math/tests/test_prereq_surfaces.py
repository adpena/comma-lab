from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.boundary_math.prereq_surfaces import (
    PrerequisiteSurfaceError,
    audit_m1_positive_band_prerequisites,
    build_frozen_rank4_prototype_bank,
    compare_affine_cell_representatives_same_coder,
    matched_continuous_to_uint8_hard_accept,
    serialize_affine_cell_candidate_same_coder,
)
from tac.canonical_equations.segnet_head_rank4_flipdist_20260715 import (
    SEGNET_WEIGHTS_SHA256,
)
from tac.optimization.uint8_lattice_feasibility import (
    BlockSolveStatus,
    DisjointResizeOperator,
    HardOracleEvaluation,
    Uint8LatticeError,
)

_WORKTREE_ROOT = Path(__file__).resolve().parents[4]
_WEIGHTS_CANDIDATES = (
    _WORKTREE_ROOT / "upstream/models/segnet.safetensors",
    Path("/Users/adpena/Projects/pact/upstream/models/segnet.safetensors"),
)
_FROZEN_WEIGHTS = next((path for path in _WEIGHTS_CANDIDATES if path.is_file()), None)


def _operator() -> DisjointResizeOperator:
    return DisjointResizeOperator.build(
        camera_h=8,
        camera_w=10,
        scorer_h=3,
        scorer_w=4,
    )


def _exact_target(operator: DisjointResizeOperator) -> tuple[np.ndarray, np.ndarray]:
    source = np.full((operator.camera_h, operator.camera_w, 1), 127, dtype=np.uint8)
    numerators, denominator = operator.apply_numerators(source)
    return numerators / denominator, numerators


def test_matched_adapter_uses_shared_exact_solver_and_fresh_parsed_oracle() -> None:
    operator = _operator()
    target, numerators = _exact_target(operator)
    calls: list[np.ndarray] = []

    def exact_oracle(frame: np.ndarray) -> HardOracleEvaluation:
        calls.append(frame.copy())
        exact = bool(np.array_equal(operator.apply_numerators(frame)[0], numerators))
        return HardOracleEvaluation(
            satisfied=np.array([exact], dtype=bool),
            margins=np.array([1.0 if exact else -1.0], dtype=np.float64),
        )

    result = matched_continuous_to_uint8_hard_accept(
        operator,
        np.linspace(-10.0, 265.0, 3 * 4).reshape(3, 4, 1),
        target,
        numerators,
        exact_oracle,
        pre_step_family="entropy_hopfield",
        max_nodes_per_block=50_000,
    )
    assert result.lattice.aggregate_status is BlockSolveStatus.FEASIBLE_EXACT
    assert result.lattice.certified_exact is True
    assert result.receipt["hard_accept"] is True
    assert result.receipt["fresh_decoded_uint8_hard_oracle_calls"] == 1
    assert result.receipt["proposal_space"] == "scorer_plane_lift"
    assert result.receipt["continuous_values_clipped_to_uint8_box"] == 2
    assert len(calls) == 1
    assert np.array_equal(calls[0], result.frame)
    assert result.frame.flags.writeable is False


def test_preferred_preimage_changes_only_tie_break_not_exact_proof() -> None:
    operator = _operator()
    target, numerators = _exact_target(operator)
    low = operator.solve_uint8(
        target,
        target_numerators=numerators,
        preferred_preimage=np.zeros((8, 10, 1), dtype=np.float64),
        max_nodes_per_block=50_000,
    )
    high = operator.solve_uint8(
        target,
        target_numerators=numerators,
        preferred_preimage=np.full((8, 10, 1), 255.0, dtype=np.float64),
        max_nodes_per_block=50_000,
    )
    assert low.certified_exact and high.certified_exact
    np.testing.assert_array_equal(operator.apply_numerators(low.frame)[0], numerators)
    np.testing.assert_array_equal(operator.apply_numerators(high.frame)[0], numerators)
    assert not np.array_equal(low.frame, high.frame)


def test_preferred_preimage_fails_closed_outside_uint8_box() -> None:
    operator = _operator()
    target, numerators = _exact_target(operator)
    bad = np.zeros((8, 10, 1), dtype=np.float64)
    bad[0, 0, 0] = -1.0
    with pytest.raises(Uint8LatticeError, match="uint8 box"):
        operator.solve_uint8(
            target,
            target_numerators=numerators,
            preferred_preimage=bad,
        )


def test_matched_adapter_rejects_non_numeric_continuous_output() -> None:
    operator = _operator()
    target, numerators = _exact_target(operator)

    def unused(_: np.ndarray) -> HardOracleEvaluation:
        raise AssertionError("oracle must not run")

    with pytest.raises(PrerequisiteSurfaceError, match="real numeric"):
        matched_continuous_to_uint8_hard_accept(
            operator,
            np.full((3, 4, 1), "0", dtype="U1"),
            target,
            numerators,
            unused,
            pre_step_family="sparsemax",
        )


@pytest.mark.skipif(_FROZEN_WEIGHTS is None, reason="real frozen SegNet weights absent")
def test_real_frozen_rank4_prototypes_are_deterministic_strict_cells() -> None:
    assert _FROZEN_WEIGHTS is not None
    first = build_frozen_rank4_prototype_bank(_FROZEN_WEIGHTS)
    second = build_frozen_rank4_prototype_bank(_FROZEN_WEIGHTS)
    assert first.receipt["frozen_weights_sha256"] == SEGNET_WEIGHTS_SHA256
    assert first.receipt["rank"] == 4
    assert first.receipt["prototype_labels"] == [0, 1, 2, 3, 4]
    assert first.receipt["rank4_reconstruction_maxabs_fp32"] <= 5.960464477539063e-08
    assert min(first.receipt["prototype_margins"]) >= 0.99998
    assert first.receipt["gauge"] == "PDW2_REFERENCE_CLASS_AFFINE_GAUGE"
    assert first.receipt["reference_class"] == 0
    assert first.pdw2_packet == second.pdw2_packet
    np.testing.assert_array_equal(first.prototypes, second.prototypes)
    with pytest.raises(ValueError, match="read-only"):
        first.prototypes[0, 0] = 0.0


@pytest.mark.skipif(_FROZEN_WEIGHTS is None, reason="real frozen SegNet weights absent")
def test_same_coder_comparator_preserves_exact_cell_identity_and_bytes() -> None:
    assert _FROZEN_WEIGHTS is not None
    bank = build_frozen_rank4_prototype_bank(_FROZEN_WEIGHTS)
    receipt = compare_affine_cell_representatives_same_coder(bank)
    assert receipt["same_coder"] is True
    assert receipt["same_packet_serializer"] is True
    assert receipt["exact_cell_identity"] is True
    assert set(receipt["representatives"]) == {
        "aurenhammer_min_generator_lp",
        "tropical_residuation_principal",
        "zero_sum_min_norm",
    }
    for row in receipt["representatives"].values():
        assert row["packet_bytes"] > 0
        assert row["coded_bytes"] > 0
        assert len(row["packet_sha256"]) == 64
        assert len(row["coded_sha256"]) == 64
        assert row["prototype_cell_labels"] == [0, 1, 2, 3, 4]


@pytest.mark.skipif(_FROZEN_WEIGHTS is None, reason="real frozen SegNet weights absent")
def test_same_coder_generic_interface_accepts_arbitrary_shared_affine_gauge() -> None:
    assert _FROZEN_WEIGHTS is not None
    bank = build_frozen_rank4_prototype_bank(_FROZEN_WEIGHTS)
    shift = np.array([2.0, -3.0, 4.0, -5.0], dtype=np.float32)
    row = serialize_affine_cell_candidate_same_coder(
        "external_candidate",
        bank.affine_weight - shift[None, :],
        bank.affine_bias - np.float32(7.0),
        bank.prototypes,
    )
    assert row["exact_cell_identity"] is True
    assert row["prototype_cell_labels"] == [0, 1, 2, 3, 4]
    assert row["cell_identity_scope"] == "caller-supplied strict prototype witnesses"


@pytest.mark.skipif(_FROZEN_WEIGHTS is None, reason="real frozen SegNet weights absent")
def test_real_prototype_builder_refuses_wrong_weight_hash() -> None:
    assert _FROZEN_WEIGHTS is not None
    with pytest.raises(PrerequisiteSurfaceError, match="SHA mismatch"):
        build_frozen_rank4_prototype_bank(
            _FROZEN_WEIGHTS,
            expected_weights_sha256="1" * 64,
        )


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, sort_keys=True), encoding="ascii")
    return path


def _m1_audit_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    sidecar = tmp_path / "pair_0000.vjp.npz"
    sidecar.write_bytes(b"real-sidecar-fixture")
    sidecar_sha = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    tensor_hashes = {
        name: hashlib.sha256(name.encode("ascii")).hexdigest()
        for name in (
            "winner",
            "rival",
            "seg_q",
            "seg_local_lipschitz",
            "head_pair_norms",
            "pose_j_x",
            "pose_j_y",
        )
    }
    manifest = _write_json(
        tmp_path / "vjp_manifest.json",
        {
            "schema": "vjp_custody_manifest.v1",
            "pair_ids": [0],
            "sidecars": [
                {
                    "pair_id": 0,
                    "path": str(sidecar),
                    "sha256": sidecar_sha,
                    "tensor_hashes": tensor_hashes,
                }
            ],
            "source_hashes": {
                "cache_sha256": (
                    "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
                ),
                "segnet_weights_sha256": SEGNET_WEIGHTS_SHA256,
            },
            "authority": {"score_claim": False},
        },
    )
    prototype = _write_json(
        tmp_path / "prototype.json",
        {
            "schema": "rank4_valid_cell_prototypes_v1",
            "frozen_weights_sha256": SEGNET_WEIGHTS_SHA256,
            "rank": 4,
            "affine_weight_sha256": "a" * 64,
        },
    )
    candidate = _write_json(
        tmp_path / "candidate.json",
        {
            "schema": "r2b_sparse_target_selection_receipt.v1",
            "gt_cache": {
                "sha256": (
                    "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
                )
            },
            "baseline": {"flip_count": 17_926},
            "candidate_evaluation_decisions": 16_751,
            "ranking": "target top1-top2 Fisher-margin",
            "hard_gate_pass": False,
            "score_claim": False,
            "promotion_eligible": False,
        },
    )
    return manifest, prototype, candidate


def test_m1_readiness_audit_sha_checks_real_fields_and_refuses_partial_n600(
    tmp_path: Path,
) -> None:
    manifest, prototype, candidate = _m1_audit_fixture(tmp_path)
    receipt = audit_m1_positive_band_prerequisites(
        (manifest,),
        prototype,
        candidate,
    )
    assert receipt["ready_to_assemble"] is False
    assert receipt["artifact_materialized"] is False
    assert receipt["vjp_custody"]["covered_pair_count"] == 1
    assert receipt["vjp_custody"]["missing_pair_count"] == 599
    assert receipt["vjp_custody"]["sidecar_bytes_rehashed"] is True
    assert [row["code"] for row in receipt["blockers"]] == [
        "INCOMPLETE_PAIR_LOCAL_VJP_CUSTODY",
        "EXACT_38077_CANDIDATE_EV_FIELD_ABSENT",
    ]
    assert receipt["score_claim"] is False
    assert receipt["pointer"].endswith("UNMOVED")


def test_m1_readiness_audit_refuses_sidecar_sha_drift(tmp_path: Path) -> None:
    manifest, prototype, candidate = _m1_audit_fixture(tmp_path)
    sidecar = tmp_path / "pair_0000.vjp.npz"
    sidecar.write_bytes(b"drifted")
    with pytest.raises(PrerequisiteSurfaceError, match="sidecar SHA mismatch"):
        audit_m1_positive_band_prerequisites(
            (manifest,),
            prototype,
            candidate,
        )
