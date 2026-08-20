# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

import pytest

from tac.witness_control.verified_continuation_certificate_v1 import (
    BASE_STOP_ACTION_ID,
    CANONICAL_SCORE_IMPLEMENTATION_SHA256,
    FINITE_MEASUREMENT_RECEIPT_SCHEMA,
    GENERATOR_DOMAIN_MANIFEST_SCHEMA,
    HARD_CONSTRAINT_RECEIPT_SCHEMA,
    PINNED_UPSTREAM_EVALUATE_SHA256,
    ConservativePrimitiveIntervalLeafEvidenceV1,
    ContinuationCertificateVerificationError,
    ContinuationProofBindingV1,
    ContinuationVerificationContextV1,
    FiniteTerminalLeafEvidenceV1,
    VerifiedContinuationCertificateV1,
    assert_registered_verified_certificate,
    base_stop_support_hyperedge_sha256,
    base_stop_terminal_leaf_id,
    build_base_stop_proof_bytes,
    build_conservative_primitive_interval_proof_bytes,
    build_finite_terminal_leaf_proof_bytes,
    reverify_continuation_certificate,
    terminal_leaf_set_root_sha256,
    upstream_binary64_contest_score,
    verify_base_stop_proof_bytes,
    verify_continuation_proof_bytes,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _reseal(envelope: dict[str, object]) -> bytes:
    body = envelope["proof_body"]
    envelope["proof_body_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    return _canonical(envelope)


def _binding(**changes: object) -> ContinuationProofBindingV1:
    values: dict[str, object] = {
        "exact_base_identity_sha256": _sha("base"),
        "epoch_id": "epoch.7",
        "generator_domain_manifest_sha256": _sha("manifest"),
        "first_action_id": "prune.first",
        "support_hyperedge_sha256": _sha("support"),
        "continuation_equivalence_sha256": _sha("continuation"),
        "horizon": 3,
        "authority_mode": "RESEARCH_ADVISORY",
        "axis": "[research-only structural verifier]",
        "hard_constraint_contract_sha256": _sha("hard-constraints"),
        "score_implementation_sha256": CANONICAL_SCORE_IMPLEMENTATION_SHA256,
    }
    values.update(changes)
    return ContinuationProofBindingV1(**values)


def _support_sha() -> str:
    return hashlib.sha256(
        _canonical({"members": ["PRUNE:MICRO"], "schema": "tac.action_support_hyperedge.v1"})
    ).hexdigest()


def _manifest_bytes(
    *,
    base_sha256: str,
    epoch_id: str,
    horizon: int,
    leaves: tuple[tuple[str, str, tuple[str, ...]], ...],
) -> bytes:
    serialized_leaves: list[dict[str, object]] = []
    leaf_identity_rows: list[dict[str, str]] = []
    for leaf_id, first_action, action_path in sorted(leaves):
        leaf: dict[str, object] = {
            "action_path": list(action_path),
            "chronology_context_sha256": _sha("chronology-" + leaf_id),
            "exact_base_identity_sha256": base_sha256,
            "first_action_id": first_action,
            "parameter_identity_sha256": _sha("parameter-" + leaf_id),
            "parent_action_ids": list(action_path[:-1]),
            "support_hyperedge": [{"family": "PRUNE", "scale": "MICRO"}],
            "terminal_leaf_id": leaf_id,
            "universe_epoch_id": epoch_id,
        }
        serialized_leaves.append(leaf)
        leaf_identity_rows.append(
            {
                "identity_sha256": hashlib.sha256(
                    _canonical({"schema": "tac.generated_terminal_leaf.v1", **leaf})
                ).hexdigest(),
                "terminal_leaf_id": leaf_id,
            }
        )
    leaf_ids = tuple(row[0] for row in sorted(leaves))
    payload = {
        "chronology_domain_manifest_sha256": _sha("chronology-domain"),
        "deterministic_seed": 7,
        "enumeration_receipt_sha256": _sha("enumeration"),
        "epoch_id": epoch_id,
        "exact_base_identity_sha256": base_sha256,
        "families": [
            "PRUNE",
            "MERGE",
            "MACRO_EVICT",
            "MIGRATE",
            "INVERSE_REPAIR",
            "FIT_QUOTIENT",
            "TRAIN_QUOTIENT",
            "REQUANTIZE_STORAGE",
            "JOINT_DESCENT",
        ],
        "generated_leaf_cardinality": len(serialized_leaves),
        "generated_leaf_merkle_root_sha256": hashlib.sha256(
            _canonical({"leaves": leaf_identity_rows, "schema": "tac.generated_terminal_leaf_set.v1"})
        ).hexdigest(),
        "generated_terminal_leaf_id_root_sha256": (None if not leaf_ids else terminal_leaf_set_root_sha256(leaf_ids)),
        "generator_program_sha256": _sha("generator-program"),
        "horizon": horizon,
        "manifest_id": "manifest.test",
        "maximum_interaction_order": 1,
        "parameter_domain_manifest_sha256": _sha("parameter-domain"),
        "scales": ["MICRO", "MESO", "MACRO", "SYSTEM"],
        "schema": GENERATOR_DOMAIN_MANIFEST_SCHEMA,
        "terminal_leaves": serialized_leaves,
    }
    return _canonical(payload)


def _typed_receipts(
    *,
    binding: ContinuationProofBindingV1,
    leaf_id: str,
    endpoint_sha256: str,
    action_path: tuple[str, ...],
    d_seg: float,
    d_pose: float,
    archive_nbytes: int,
    base_stop: bool = False,
) -> tuple[str, bytes, str, bytes]:
    common: dict[str, object] = {
        "action_path": list(action_path),
        "authority_mode": binding.authority_mode,
        "axis": binding.axis,
        "continuation_equivalence_sha256": binding.continuation_equivalence_sha256,
        "epoch_id": binding.epoch_id,
        "exact_base_identity_sha256": binding.exact_base_identity_sha256,
        "first_action_id": binding.first_action_id,
        "generator_domain_manifest_sha256": binding.generator_domain_manifest_sha256,
        "horizon": binding.horizon,
        "research_only": True,
        "support_hyperedge_sha256": binding.support_hyperedge_sha256,
        "terminal_endpoint_identity_sha256": endpoint_sha256,
        "terminal_leaf_id": leaf_id,
    }
    measurement = _canonical(
        {
            **common,
            "archive_nbytes": archive_nbytes,
            "archive_sha256": _sha("archive-" + leaf_id),
            "d_pose_binary64_hex": float(d_pose).hex(),
            "d_seg_binary64_hex": float(d_seg).hex(),
            "evaluator_measurement_receipt_sha256": _sha("evaluator-" + leaf_id),
            "public_auth_eval_receipt_sha256": _sha("public-" + leaf_id),
            "realized_output_sha256": _sha("output-" + leaf_id),
            "receipt_role": (
                "G36_VERIFIED_WHOLE_OBJECT_BASE" if base_stop else "G36_VERIFIED_WHOLE_OBJECT_ACTION_ENDPOINT"
            ),
            "schema": FINITE_MEASUREMENT_RECEIPT_SCHEMA,
            "score_implementation_sha256": binding.score_implementation_sha256,
            "upstream_evaluate_sha256": PINNED_UPSTREAM_EVALUATE_SHA256,
        }
    )
    constraints = _canonical(
        {
            **common,
            "blocker_codes": [],
            "decode_constraint_receipt_sha256": _sha("decode-constraints-" + leaf_id),
            "hard_constraint_contract_sha256": binding.hard_constraint_contract_sha256,
            "hard_constraints_satisfied": True,
            "receipt_role": "G36_VERIFIED_DECODE_CONSTRAINTS",
            "schema": HARD_CONSTRAINT_RECEIPT_SCHEMA,
        }
    )
    return (
        hashlib.sha256(measurement).hexdigest(),
        measurement,
        hashlib.sha256(constraints).hexdigest(),
        constraints,
    )


def _finite_fixture() -> tuple[
    ContinuationProofBindingV1,
    tuple[FiniteTerminalLeafEvidenceV1, ...],
    dict[str, bytes],
    bytes,
]:
    base = _sha("base")
    epoch = "epoch.7"
    manifest = _manifest_bytes(
        base_sha256=base,
        epoch_id=epoch,
        horizon=3,
        leaves=(
            ("leaf.a", "prune.first", ("prune.first", "requantize.second")),
            ("leaf.b", "prune.first", ("prune.first", "merge.second")),
        ),
    )
    binding = _binding(
        exact_base_identity_sha256=base,
        epoch_id=epoch,
        generator_domain_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        support_hyperedge_sha256=_support_sha(),
    )
    m1_sha, m1, h1_sha, h1 = _typed_receipts(
        binding=binding,
        leaf_id="leaf.b",
        endpoint_sha256=_sha("endpoint-b"),
        action_path=("prune.first", "merge.second"),
        d_seg=0.01,
        d_pose=0.04,
        archive_nbytes=500,
    )
    m2_sha, m2, h2_sha, h2 = _typed_receipts(
        binding=binding,
        leaf_id="leaf.a",
        endpoint_sha256=_sha("endpoint-a"),
        action_path=("prune.first", "requantize.second"),
        d_seg=0.001,
        d_pose=0.0001,
        archive_nbytes=50,
    )
    leaves = (
        FiniteTerminalLeafEvidenceV1(
            terminal_leaf_id="leaf.b",
            terminal_endpoint_identity_sha256=_sha("endpoint-b"),
            action_path=("prune.first", "merge.second"),
            d_seg=0.01,
            d_pose=0.04,
            archive_nbytes=500,
            measurement_receipt_sha256=m1_sha,
            hard_constraint_receipt_sha256=h1_sha,
        ),
        FiniteTerminalLeafEvidenceV1(
            terminal_leaf_id="leaf.a",
            terminal_endpoint_identity_sha256=_sha("endpoint-a"),
            action_path=("prune.first", "requantize.second"),
            d_seg=0.001,
            d_pose=0.0001,
            archive_nbytes=50,
            measurement_receipt_sha256=m2_sha,
            hard_constraint_receipt_sha256=h2_sha,
        ),
    )
    dependencies = {m1_sha: m1, h1_sha: h1, m2_sha: m2, h2_sha: h2}
    return binding, leaves, dependencies, manifest


def _context(
    *,
    binding: ContinuationProofBindingV1,
    leaf_ids: tuple[str, ...],
    dependency_sha256s: tuple[str, ...],
    proof: bytes | None = None,
) -> ContinuationVerificationContextV1:
    ordered_leaf_ids = tuple(sorted(leaf_ids))
    return ContinuationVerificationContextV1(
        binding=binding,
        expected_subtree_root_sha256=terminal_leaf_set_root_sha256(ordered_leaf_ids),
        expected_subtree_cardinality=len(ordered_leaf_ids),
        expected_covered_terminal_leaf_ids=ordered_leaf_ids,
        expected_dependency_sha256s=tuple(sorted(dependency_sha256s)),
        expected_proof_sha256=None if proof is None else hashlib.sha256(proof).hexdigest(),
    )


def test_valid_finite_leaf_proof_derives_exact_lower_upper_and_partition() -> None:
    binding, leaves, dependencies, manifest = _finite_fixture()
    proof = build_finite_terminal_leaf_proof_bytes(binding=binding, leaves=leaves)
    context = _context(
        binding=binding,
        leaf_ids=tuple(row.terminal_leaf_id for row in leaves),
        dependency_sha256s=tuple(dependencies),
        proof=proof,
    )

    certificate = verify_continuation_proof_bytes(
        proof,
        context=context,
        generator_domain_manifest_payload_bytes=manifest,
        dependency_payloads_by_sha256=dependencies,
    )

    expected = upstream_binary64_contest_score(0.001, 0.0001, 50)
    assert certificate.verifier_status == "VERIFIED_FINITE_TERMINAL_ENUMERATION"
    assert certificate.terminal_lower_bound == expected
    assert certificate.feasible_terminal_upper_bound == expected
    assert certificate.feasible_terminal_action_path == (
        "prune.first",
        "requantize.second",
    )
    assert certificate.feasible_terminal_endpoint_identity_sha256 == _sha("endpoint-a")
    assert certificate.covered_terminal_leaf_ids == ("leaf.a", "leaf.b")
    assert certificate.covered_terminal_leaf_root_sha256 == context.expected_subtree_root_sha256
    assert certificate.subtree_cardinality == 2
    assert certificate.proof_sha256 == hashlib.sha256(proof).hexdigest()
    assert certificate.research_only is True
    assert_registered_verified_certificate(certificate)
    assert (
        reverify_continuation_certificate(
            certificate,
            proof,
            context=context,
            generator_domain_manifest_payload_bytes=manifest,
            dependency_payloads_by_sha256=dependencies,
        ).proof_sha256
        == certificate.proof_sha256
    )


def test_certificate_has_no_public_constructor() -> None:
    with pytest.raises(ContinuationCertificateVerificationError, match="no public constructor"):
        VerifiedContinuationCertificateV1()
    assert not hasattr(VerifiedContinuationCertificateV1, "_construct")


def test_forged_or_mutated_python_object_fails_registry_fingerprint() -> None:
    binding, leaves, dependencies, manifest = _finite_fixture()
    proof = build_finite_terminal_leaf_proof_bytes(binding=binding, leaves=leaves)
    context = _context(
        binding=binding,
        leaf_ids=tuple(row.terminal_leaf_id for row in leaves),
        dependency_sha256s=tuple(dependencies),
    )
    valid = verify_continuation_proof_bytes(
        proof,
        context=context,
        generator_domain_manifest_payload_bytes=manifest,
        dependency_payloads_by_sha256=dependencies,
    )
    forged = object.__new__(VerifiedContinuationCertificateV1)
    for name, value in asdict(valid).items():
        object.__setattr__(forged, name, value)
    with pytest.raises(ContinuationCertificateVerificationError, match="forged, unregistered, or mutated"):
        assert_registered_verified_certificate(forged)

    object.__setattr__(valid, "terminal_lower_bound", 1234.5)
    with pytest.raises(ContinuationCertificateVerificationError, match="forged, unregistered, or mutated"):
        assert_registered_verified_certificate(valid)


def test_random_scalar_and_sha_are_not_a_proof_envelope() -> None:
    binding, leaves, dependencies, manifest = _finite_fixture()
    fake = _canonical(
        {
            "evidence_sha256": _sha("random"),
            "optimistic_score_lower_bound": 0.0,
        }
    )
    context = _context(
        binding=binding,
        leaf_ids=tuple(row.terminal_leaf_id for row in leaves),
        dependency_sha256s=tuple(dependencies),
    )

    with pytest.raises(ContinuationCertificateVerificationError, match="proof envelope fields"):
        verify_continuation_proof_bytes(
            fake,
            context=context,
            generator_domain_manifest_payload_bytes=manifest,
            dependency_payloads_by_sha256={},
        )


def test_tampered_body_refuses_before_bound_derivation() -> None:
    binding, leaves, dependencies, manifest = _finite_fixture()
    proof = build_finite_terminal_leaf_proof_bytes(binding=binding, leaves=leaves)
    envelope = json.loads(proof)
    envelope["proof_body"]["finite_terminal_leaves"][0]["d_seg"] = 0.0
    tampered = _canonical(envelope)
    context = _context(
        binding=binding,
        leaf_ids=tuple(row.terminal_leaf_id for row in leaves),
        dependency_sha256s=tuple(dependencies),
    )

    with pytest.raises(ContinuationCertificateVerificationError, match="self-hash"):
        verify_continuation_proof_bytes(
            tampered,
            context=context,
            generator_domain_manifest_payload_bytes=manifest,
            dependency_payloads_by_sha256=dependencies,
        )


def test_resealed_tamper_refuses_expected_proof_identity() -> None:
    binding, leaves, dependencies, manifest = _finite_fixture()
    proof = build_finite_terminal_leaf_proof_bytes(binding=binding, leaves=leaves)
    envelope = json.loads(proof)
    envelope["proof_body"]["finite_terminal_leaves"][0]["d_seg"] = 0.0
    resealed = _reseal(envelope)
    context = _context(
        binding=binding,
        leaf_ids=tuple(row.terminal_leaf_id for row in leaves),
        dependency_sha256s=tuple(dependencies),
        proof=proof,
    )

    with pytest.raises(ContinuationCertificateVerificationError, match="expected proof identity"):
        verify_continuation_proof_bytes(
            resealed,
            context=context,
            generator_domain_manifest_payload_bytes=manifest,
            dependency_payloads_by_sha256=dependencies,
        )


def test_resealed_scalar_tamper_without_expected_proof_sha_refuses_typed_receipt_bits() -> None:
    binding, leaves, dependencies, manifest = _finite_fixture()
    proof = build_finite_terminal_leaf_proof_bytes(binding=binding, leaves=leaves)
    envelope = json.loads(proof)
    envelope["proof_body"]["finite_terminal_leaves"][0]["d_seg"] = 0.0
    resealed = _reseal(envelope)
    context = _context(
        binding=binding,
        leaf_ids=tuple(row.terminal_leaf_id for row in leaves),
        dependency_sha256s=tuple(dependencies),
    )

    with pytest.raises(ContinuationCertificateVerificationError, match="receipt-bound binary64 bits"):
        verify_continuation_proof_bytes(
            resealed,
            context=context,
            generator_domain_manifest_payload_bytes=manifest,
            dependency_payloads_by_sha256=dependencies,
        )


def test_opaque_hashed_dependency_cannot_be_admitted_as_measurement_receipt() -> None:
    binding, leaves, dependencies, manifest = _finite_fixture()
    opaque = _canonical({"scalar": 0.0, "schema": "tac.opaque.v1"})
    opaque_sha = hashlib.sha256(opaque).hexdigest()
    first = leaves[0]
    replaced = replace(first, measurement_receipt_sha256=opaque_sha)
    proof_leaves = (replaced, leaves[1])
    proof = build_finite_terminal_leaf_proof_bytes(binding=binding, leaves=proof_leaves)
    changed_dependencies = dict(dependencies)
    del changed_dependencies[first.measurement_receipt_sha256]
    changed_dependencies[opaque_sha] = opaque
    context = _context(
        binding=binding,
        leaf_ids=tuple(row.terminal_leaf_id for row in proof_leaves),
        dependency_sha256s=tuple(changed_dependencies),
    )

    with pytest.raises(ContinuationCertificateVerificationError, match="measurement receipt fields"):
        verify_continuation_proof_bytes(
            proof,
            context=context,
            generator_domain_manifest_payload_bytes=manifest,
            dependency_payloads_by_sha256=changed_dependencies,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("derivation_method", []),
        ("authority_mode", []),
        ("proof_dependency_sha256s", [{}]),
        ("covered_terminal_leaf_ids", [{}]),
        ("subtree_cardinality", True),
    ],
)
def test_malformed_container_and_bool_types_fail_closed(
    field: str,
    replacement: object,
) -> None:
    binding, leaves, dependencies, manifest = _finite_fixture()
    proof = build_finite_terminal_leaf_proof_bytes(binding=binding, leaves=leaves)
    envelope = json.loads(proof)
    envelope["proof_body"][field] = replacement
    malformed = _reseal(envelope)
    context = _context(
        binding=binding,
        leaf_ids=tuple(row.terminal_leaf_id for row in leaves),
        dependency_sha256s=tuple(dependencies),
    )

    with pytest.raises(ContinuationCertificateVerificationError):
        verify_continuation_proof_bytes(
            malformed,
            context=context,
            generator_domain_manifest_payload_bytes=manifest,
            dependency_payloads_by_sha256=dependencies,
        )


def test_finite_leaf_path_must_begin_with_bound_first_action() -> None:
    binding, leaves, dependencies, manifest = _finite_fixture()
    proof = build_finite_terminal_leaf_proof_bytes(binding=binding, leaves=leaves)
    envelope = json.loads(proof)
    envelope["proof_body"]["finite_terminal_leaves"][0]["action_path"][0] = "foreign.first"
    malformed = _reseal(envelope)
    context = _context(
        binding=binding,
        leaf_ids=tuple(row.terminal_leaf_id for row in leaves),
        dependency_sha256s=tuple(dependencies),
    )

    with pytest.raises(ContinuationCertificateVerificationError, match="first action or horizon"):
        verify_continuation_proof_bytes(
            malformed,
            context=context,
            generator_domain_manifest_payload_bytes=manifest,
            dependency_payloads_by_sha256=dependencies,
        )


@pytest.mark.parametrize(
    ("context_binding", "message"),
    [
        (_binding(generator_domain_manifest_sha256=_sha("other-manifest")), "foreign keys"),
        (_binding(first_action_id="other.first"), "foreign keys"),
    ],
)
def test_manifest_or_first_action_mismatch_refuses(
    context_binding: ContinuationProofBindingV1,
    message: str,
) -> None:
    proof_binding, leaves, dependencies, manifest = _finite_fixture()
    proof = build_finite_terminal_leaf_proof_bytes(
        binding=proof_binding,
        leaves=leaves,
    )
    context = _context(
        binding=context_binding,
        leaf_ids=tuple(row.terminal_leaf_id for row in leaves),
        dependency_sha256s=tuple(dependencies),
    )

    with pytest.raises(ContinuationCertificateVerificationError, match=message):
        verify_continuation_proof_bytes(
            proof,
            context=context,
            generator_domain_manifest_payload_bytes=manifest,
            dependency_payloads_by_sha256=dependencies,
        )


def test_manifest_subtree_root_mismatch_refuses() -> None:
    binding, leaves, dependencies, manifest = _finite_fixture()
    proof = build_finite_terminal_leaf_proof_bytes(binding=binding, leaves=leaves)
    foreign_leaf_ids = ("leaf.a", "leaf.foreign")
    context = ContinuationVerificationContextV1(
        binding=binding,
        expected_subtree_root_sha256=terminal_leaf_set_root_sha256(foreign_leaf_ids),
        expected_subtree_cardinality=2,
        expected_covered_terminal_leaf_ids=foreign_leaf_ids,
        expected_dependency_sha256s=tuple(sorted(dependencies)),
    )

    with pytest.raises(ContinuationCertificateVerificationError, match="generator-domain manifest"):
        verify_continuation_proof_bytes(
            proof,
            context=context,
            generator_domain_manifest_payload_bytes=manifest,
            dependency_payloads_by_sha256=dependencies,
        )


def test_dependency_sha_string_without_reopened_bytes_refuses() -> None:
    binding, leaves, dependencies, manifest = _finite_fixture()
    proof = build_finite_terminal_leaf_proof_bytes(binding=binding, leaves=leaves)
    context = _context(
        binding=binding,
        leaf_ids=tuple(row.terminal_leaf_id for row in leaves),
        dependency_sha256s=tuple(dependencies),
    )

    with pytest.raises(ContinuationCertificateVerificationError, match="dependency payload set"):
        verify_continuation_proof_bytes(
            proof,
            context=context,
            generator_domain_manifest_payload_bytes=manifest,
            dependency_payloads_by_sha256={},
        )


def test_primitive_intervals_fail_closed_until_typed_theorem_verifier_exists() -> None:
    base = _sha("primitive-base")
    epoch = "epoch.primitive"
    manifest = _manifest_bytes(
        base_sha256=base,
        epoch_id=epoch,
        horizon=3,
        leaves=(
            ("interval.a", "fit.first", ("fit.first", "fit.a")),
            ("interval.b", "fit.first", ("fit.first", "fit.b")),
        ),
    )
    binding = _binding(
        exact_base_identity_sha256=base,
        epoch_id=epoch,
        first_action_id="fit.first",
        generator_domain_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        support_hyperedge_sha256=_support_sha(),
    )
    i1, c1, i2, c2 = (b"interval-1", b"coverage-1", b"interval-2", b"coverage-2")
    i1_sha, c1_sha, i2_sha, c2_sha = (hashlib.sha256(value).hexdigest() for value in (i1, c1, i2, c2))
    leaves = (
        ConservativePrimitiveIntervalLeafEvidenceV1(
            terminal_leaf_id="interval.a",
            d_seg_lower=0.002,
            d_seg_upper=0.01,
            d_pose_lower=0.0004,
            d_pose_upper=0.03,
            archive_nbytes_lower=100,
            archive_nbytes_upper=1000,
            interval_derivation_receipt_sha256=i1_sha,
            leaf_coverage_receipt_sha256=c1_sha,
        ),
        ConservativePrimitiveIntervalLeafEvidenceV1(
            terminal_leaf_id="interval.b",
            d_seg_lower=0.001,
            d_seg_upper=0.02,
            d_pose_lower=0.0001,
            d_pose_upper=0.05,
            archive_nbytes_lower=80,
            archive_nbytes_upper=2000,
            interval_derivation_receipt_sha256=i2_sha,
            leaf_coverage_receipt_sha256=c2_sha,
        ),
    )
    dependencies = {i1_sha: i1, c1_sha: c1, i2_sha: i2, c2_sha: c2}
    proof = build_conservative_primitive_interval_proof_bytes(
        binding=binding,
        leaves=leaves,
    )
    context = _context(
        binding=binding,
        leaf_ids=tuple(row.terminal_leaf_id for row in leaves),
        dependency_sha256s=tuple(dependencies),
        proof=proof,
    )

    with pytest.raises(ContinuationCertificateVerificationError, match="typed theorem verification"):
        verify_continuation_proof_bytes(
            proof,
            context=context,
            generator_domain_manifest_payload_bytes=manifest,
            dependency_payloads_by_sha256=dependencies,
        )


def test_base_stop_factory_binds_exact_base_score_and_controller_owned_leaf() -> None:
    base_identity = _sha("base-stop-base")
    epoch = "epoch.stop"
    manifest = _manifest_bytes(
        base_sha256=base_identity,
        epoch_id=epoch,
        horizon=3,
        leaves=(),
    )
    binding = _binding(
        exact_base_identity_sha256=base_identity,
        epoch_id=epoch,
        generator_domain_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
        first_action_id=BASE_STOP_ACTION_ID,
        support_hyperedge_sha256=base_stop_support_hyperedge_sha256(
            exact_base_identity_sha256=base_identity,
            epoch_id=epoch,
        ),
    )
    leaf_id = base_stop_terminal_leaf_id(
        exact_base_identity_sha256=base_identity,
        epoch_id=epoch,
    )
    measurement_sha, measurement, constraints_sha, constraints = _typed_receipts(
        binding=binding,
        leaf_id=leaf_id,
        endpoint_sha256=_sha("base-endpoint"),
        action_path=(BASE_STOP_ACTION_ID,),
        d_seg=0.003,
        d_pose=0.0009,
        archive_nbytes=123,
        base_stop=True,
    )
    proof = build_base_stop_proof_bytes(
        binding=binding,
        base_endpoint_identity_sha256=_sha("base-endpoint"),
        d_seg=0.003,
        d_pose=0.0009,
        archive_nbytes=123,
        measurement_receipt_sha256=measurement_sha,
        hard_constraint_receipt_sha256=constraints_sha,
    )
    dependencies = {
        measurement_sha: measurement,
        constraints_sha: constraints,
    }
    context = _context(
        binding=binding,
        leaf_ids=(leaf_id,),
        dependency_sha256s=tuple(dependencies),
        proof=proof,
    )

    certificate = verify_base_stop_proof_bytes(
        proof,
        context=context,
        generator_domain_manifest_payload_bytes=manifest,
        dependency_payloads_by_sha256=dependencies,
    )

    expected = upstream_binary64_contest_score(0.003, 0.0009, 123)
    assert certificate.verifier_status == "VERIFIED_BASE_STOP_EXACT_SCORE"
    assert certificate.first_action_id == BASE_STOP_ACTION_ID
    assert certificate.terminal_lower_bound == expected
    assert certificate.feasible_terminal_upper_bound == expected
    assert certificate.feasible_terminal_action_path == (BASE_STOP_ACTION_ID,)
    assert certificate.covered_terminal_leaf_ids == (leaf_id,)


def test_score_implementation_cannot_be_relabeled() -> None:
    with pytest.raises(
        ContinuationCertificateVerificationError,
        match="canonical contest-score arithmetic",
    ):
        replace(_binding(), score_implementation_sha256=_sha("fake-score-helper"))


@pytest.mark.parametrize(
    ("authority_mode", "axis"),
    [
        ("PRODUCTION_CONTEST_CPU", "[contest-CUDA]"),
        ("PRODUCTION_CONTEST_CUDA", "anything"),
        ("RESEARCH_ADVISORY", "[contest-CPU]"),
    ],
)
def test_authority_axis_mismatch_refuses(authority_mode: str, axis: str) -> None:
    with pytest.raises(ContinuationCertificateVerificationError, match="authority mode and hardware axis"):
        _binding(authority_mode=authority_mode, axis=axis)


def test_production_admission_fails_closed_pending_governed_adapter() -> None:
    research_binding, leaves, dependencies, manifest = _finite_fixture()
    binding = replace(
        research_binding,
        authority_mode="PRODUCTION_CONTEST_CPU",
        axis="[contest-CPU]",
    )
    proof = build_finite_terminal_leaf_proof_bytes(binding=binding, leaves=leaves)
    context = _context(
        binding=binding,
        leaf_ids=tuple(row.terminal_leaf_id for row in leaves),
        dependency_sha256s=tuple(dependencies),
    )
    with pytest.raises(ContinuationCertificateVerificationError, match="production continuation admission"):
        verify_continuation_proof_bytes(
            proof,
            context=context,
            generator_domain_manifest_payload_bytes=manifest,
            dependency_payloads_by_sha256=dependencies,
        )


def test_manifest_bytes_are_reopened_not_accepted_by_sha_string_only() -> None:
    binding, leaves, dependencies, manifest = _finite_fixture()
    proof = build_finite_terminal_leaf_proof_bytes(binding=binding, leaves=leaves)
    context = _context(
        binding=binding,
        leaf_ids=tuple(row.terminal_leaf_id for row in leaves),
        dependency_sha256s=tuple(dependencies),
    )
    tampered_manifest = json.loads(manifest)
    tampered_manifest["terminal_leaves"][0]["action_path"][-1] = "foreign.second"
    with pytest.raises(ContinuationCertificateVerificationError, match="manifest bytes differ from bound identity"):
        verify_continuation_proof_bytes(
            proof,
            context=context,
            generator_domain_manifest_payload_bytes=_canonical(tampered_manifest),
            dependency_payloads_by_sha256=dependencies,
        )


@pytest.mark.parametrize("bad_d_seg", [0, -0.0])
def test_noncanonical_numeric_component_encodings_refuse(bad_d_seg: int | float) -> None:
    with pytest.raises(ContinuationCertificateVerificationError, match=r"canonical binary64|negative zero"):
        FiniteTerminalLeafEvidenceV1(
            terminal_leaf_id="leaf.numeric",
            terminal_endpoint_identity_sha256=_sha("endpoint.numeric"),
            action_path=("numeric.first",),
            d_seg=bad_d_seg,
            d_pose=0.0,
            archive_nbytes=1,
            measurement_receipt_sha256=_sha("measurement.numeric"),
            hard_constraint_receipt_sha256=_sha("constraints.numeric"),
        )


def test_duplicate_key_nan_and_huge_integer_proofs_fail_closed() -> None:
    binding, leaves, dependencies, manifest = _finite_fixture()
    context = _context(
        binding=binding,
        leaf_ids=tuple(row.terminal_leaf_id for row in leaves),
        dependency_sha256s=tuple(dependencies),
    )
    for malformed in (
        b'{"schema":"x","schema":"y"}',
        b'{"schema":NaN}',
        ('{"schema":' + "9" * 10000 + "}").encode("ascii"),
    ):
        with pytest.raises(ContinuationCertificateVerificationError):
            verify_continuation_proof_bytes(
                malformed,
                context=context,
                generator_domain_manifest_payload_bytes=manifest,
                dependency_payloads_by_sha256=dependencies,
            )


def test_upstream_binary64_operation_order_is_pinned_at_one_byte_counterexample() -> None:
    d_seg = 0.0
    d_pose = 0.0
    rate = 1 / 37_545_489
    upstream = 100 * d_seg + (d_pose * 10) ** 0.5 + 25 * rate
    assert float(upstream_binary64_contest_score(d_seg, d_pose, 1)).hex() == float(upstream).hex()
