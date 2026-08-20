from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

import pytest

from tac.witness_control.taskspace_receding_horizon_controller_v1 import (
    ACTION_FAMILIES,
    ACTION_SCALES,
    ActionFamilyScaleCoverageV1,
    ActionSupportCellV1,
    CompleteActionUniverseV1,
    ContinuationCertificateEvidenceBundleV1,
    ContinuationProofDependencyPayloadV1,
    DecodeConstraintEvidenceV1,
    GeneratedTerminalLeafV1,
    GeneratorDomainManifestV1,
    TaskspaceRecedingHorizonError,
    WholeObjectActionEndpointV1,
    WholeObjectBaseV1,
    select_receding_horizon_action_v1,
)
from tac.witness_control.verified_continuation_certificate_v1 import (
    CANONICAL_SCORE_IMPLEMENTATION_SHA256,
    FINITE_MEASUREMENT_RECEIPT_SCHEMA,
    HARD_CONSTRAINT_RECEIPT_SCHEMA,
    PINNED_UPSTREAM_EVALUATE_SHA256,
    ConservativePrimitiveIntervalLeafEvidenceV1,
    ContinuationCertificateVerificationError,
    ContinuationProofBindingV1,
    ContinuationVerificationContextV1,
    FiniteTerminalLeafEvidenceV1,
    VerifiedContinuationCertificateV1,
    build_conservative_primitive_interval_proof_bytes,
    build_finite_terminal_leaf_proof_bytes,
    terminal_leaf_set_root_sha256,
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


def _constraints(**changes: object) -> DecodeConstraintEvidenceV1:
    values = {
        "receipt_sha256": _sha("decode-constraints"),
        "measurement_scope": "PUBLIC_EVALUATE_SH_WHOLE_WORKFLOW",
        "measured_wall_seconds": 120.0,
        "wall_seconds_limit": 1_800.0,
        "measured_peak_rss_bytes": 2_000_000_000,
        "peak_rss_limit_bytes": 8_000_000_000,
        "double_decode_bit_identical": True,
        "public_inflate_video_closed": True,
        "recursive_auth_eval_closed": True,
        "hidden_data_absence_proved": True,
        "network_access_absence_proved": True,
        "source_video_access_absence_proved": True,
        "scorer_access_absence_proved": True,
        "deterministic_portability_proved": True,
        "video_specific_state_fully_counted": True,
    }
    values.update(changes)
    return DecodeConstraintEvidenceV1(**values)  # type: ignore[arg-type]


def _base(*, advisory: bool = False) -> WholeObjectBaseV1:
    return WholeObjectBaseV1(
        epoch_id="epoch.0",
        authority_mode="RESEARCH_ADVISORY" if advisory else "PRODUCTION_CONTEST_CPU",
        axis="[macOS-CPU advisory]" if advisory else "[contest-CPU]",
        archive_sha256=_sha("base-archive"),
        archive_nbytes=100_000,
        realized_output_sha256=_sha("base-output"),
        evaluator_cell_identity_sha256=_sha("base-cells"),
        evaluator_output_cell_ledger_sha256=_sha("base-output-cell-ledger"),
        continuation_equivalence_sha256=_sha("base-continuation"),
        continuation_equivalence_receipt_sha256=_sha("base-continuation-receipt"),
        decoder_runtime_sha256=_sha("runtime"),
        payload_placement_manifest_sha256=_sha("base-placement"),
        measurement_receipt_sha256=_sha("base-measurement"),
        public_auth_eval_receipt_sha256=_sha("base-auth"),
        dynamic_pointer_artifact_sha256=_sha("pointer"),
        dynamic_pointer_artifact_nbytes=17_319,
        dynamic_target_score=0.172,
        d_seg=0.002,
        d_pose=0.001,
        decode_constraints=_constraints(),
    )


def _support(*pairs: tuple[str, str]) -> tuple[ActionSupportCellV1, ...]:
    return tuple(
        sorted(
            {
                ActionSupportCellV1(family, scale)  # type: ignore[arg-type]
                for family, scale in pairs
            }
        )
    )


def _endpoint(
    base: WholeObjectBaseV1,
    terminal_leaf_id: str,
    *,
    first_action_id: str | None = None,
    action_path: tuple[str, ...] | None = None,
    support: tuple[ActionSupportCellV1, ...] | None = None,
    d_seg: float = 0.0018,
    d_pose: float = 0.001,
    archive_nbytes: int = 99_000,
    cells: str | None = None,
    continuation: str | None = None,
    proof_label: str | None = None,
    object_label: str | None = None,
    constraints: DecodeConstraintEvidenceV1 | None = None,
) -> WholeObjectActionEndpointV1:
    first = first_action_id or terminal_leaf_id
    path = action_path or (first,)
    support_cells = support or _support(("PRUNE", "MICRO"))
    leaf = GeneratedTerminalLeafV1(
        terminal_leaf_id=terminal_leaf_id,
        first_action_id=first,
        action_path=path,
        support_hyperedge=support_cells,
        parent_action_ids=path[:-1],
        parameter_identity_sha256=_sha(f"{terminal_leaf_id}-parameters"),
        chronology_context_sha256=_sha(f"{terminal_leaf_id}-chronology"),
        exact_base_identity_sha256=base.identity_sha256,
        universe_epoch_id=base.epoch_id,
    )
    proof = proof_label or terminal_leaf_id
    obj = object_label or terminal_leaf_id
    return WholeObjectActionEndpointV1(
        generated_leaf=leaf,
        authority_mode=base.authority_mode,
        axis=base.axis,
        archive_sha256=_sha(f"{obj}-archive"),
        archive_nbytes=archive_nbytes,
        realized_output_sha256=_sha(f"{obj}-output"),
        evaluator_cell_identity_sha256=_sha(cells or f"{obj}-cells"),
        evaluator_output_cell_ledger_sha256=_sha(f"{proof}-output-cell-ledger"),
        continuation_equivalence_sha256=_sha(continuation or f"{obj}-continuation"),
        continuation_equivalence_receipt_sha256=_sha(f"{proof}-continuation-receipt"),
        decoder_runtime_sha256=_sha(f"{proof}-runtime"),
        payload_placement_manifest_sha256=_sha(f"{proof}-placement"),
        measurement_receipt_sha256=_sha(f"{proof}-measurement"),
        public_auth_eval_receipt_sha256=_sha(f"{proof}-public-auth"),
        d_seg=d_seg,
        d_pose=d_pose,
        decode_constraints=constraints or _constraints(),
        proposal_evidence_sha256=_sha(f"{terminal_leaf_id}-proposal"),
        interaction_hypergraph_sha256=_sha(f"{terminal_leaf_id}-hypergraph"),
    )


def _manifest(
    base: WholeObjectBaseV1,
    endpoints: tuple[WholeObjectActionEndpointV1, ...],
    *,
    horizon: int = 2,
    families: tuple[str, ...] = ACTION_FAMILIES,
) -> GeneratorDomainManifestV1:
    return GeneratorDomainManifestV1(
        manifest_id="manifest.0",
        epoch_id=base.epoch_id,
        exact_base_identity_sha256=base.identity_sha256,
        generator_program_sha256=_sha("generator-program"),
        enumeration_receipt_sha256=_sha("generator-enumeration"),
        parameter_domain_manifest_sha256=_sha("generator-parameter-domain"),
        chronology_domain_manifest_sha256=_sha("generator-chronology-domain"),
        deterministic_seed=17,
        horizon=horizon,
        maximum_interaction_order=max(
            (endpoint.generated_leaf.interaction_order for endpoint in endpoints),
            default=1,
        ),
        families=families,  # type: ignore[arg-type]
        scales=ACTION_SCALES,
        terminal_leaves=tuple(
            sorted(
                (endpoint.generated_leaf for endpoint in endpoints),
                key=lambda leaf: leaf.terminal_leaf_id,
            )
        ),
    )


def _coverage(
    terminal_leaves: tuple[GeneratedTerminalLeafV1, ...],
    *,
    blocking_cell: tuple[str, str] | None = None,
) -> tuple[ActionFamilyScaleCoverageV1, ...]:
    rows: list[ActionFamilyScaleCoverageV1] = []
    for family in ACTION_FAMILIES:
        for scale in ACTION_SCALES:
            cell = ActionSupportCellV1(family, scale)
            action_ids = tuple(
                sorted(leaf.terminal_leaf_id for leaf in terminal_leaves if cell in leaf.support_hyperedge)
            )
            blocked = blocking_cell == (family, scale)
            rows.append(
                ActionFamilyScaleCoverageV1(
                    family=family,
                    scale=scale,
                    status=(
                        "GENERATED_MANIFEST_ACTIONS"
                        if action_ids
                        else "EXPLICITLY_BLOCKED"
                        if blocked
                        else "EMPTY_IN_MANIFEST"
                    ),
                    action_ids=action_ids,
                    evidence_sha256=_sha(f"coverage-{family}-{scale}"),
                    blocks_global_selection=blocked,
                    verdict_scope="audit projection of the exact finite generator manifest",
                )
            )
    return tuple(rows)


def _universe(
    base: WholeObjectBaseV1,
    endpoints: tuple[WholeObjectActionEndpointV1, ...],
    *,
    manifest: GeneratorDomainManifestV1 | None = None,
    blocking_cell: tuple[str, str] | None = None,
    bundles: tuple[ContinuationCertificateEvidenceBundleV1, ...] = (),
) -> CompleteActionUniverseV1:
    exact_manifest = manifest or _manifest(base, endpoints)
    return CompleteActionUniverseV1(
        universe_id="universe.0",
        generator_manifest=exact_manifest,
        coverage=_coverage(exact_manifest.terminal_leaves, blocking_cell=blocking_cell),
        endpoints=endpoints,
        continuation_evidence_bundles=bundles,
    )


def _three_endpoints(base: WholeObjectBaseV1) -> tuple[WholeObjectActionEndpointV1, ...]:
    return (
        _endpoint(
            base,
            "prune.modest.terminal",
            first_action_id="prune.route",
            action_path=("prune.route", "repair.modest"),
        ),
        _endpoint(
            base,
            "prune.best.terminal",
            first_action_id="prune.route",
            action_path=("prune.route", "repair.best"),
            support=_support(("PRUNE", "MICRO"), ("INVERSE_REPAIR", "MESO")),
            d_seg=0.001,
            d_pose=0.0005,
            archive_nbytes=98_000,
        ),
        _endpoint(
            base,
            "macro.evict.terminal",
            first_action_id="macro.route",
            action_path=("macro.route", "requantize.followup"),
            support=_support(("MACRO_EVICT", "MACRO"), ("REQUANTIZE_STORAGE", "SYSTEM")),
            d_seg=0.0015,
            d_pose=0.0008,
            archive_nbytes=90_000,
        ),
    )


def _binding(
    base: WholeObjectBaseV1,
    manifest: GeneratorDomainManifestV1,
    leaf: GeneratedTerminalLeafV1,
    *,
    continuation: str,
) -> ContinuationProofBindingV1:
    return ContinuationProofBindingV1(
        exact_base_identity_sha256=base.identity_sha256,
        epoch_id=base.epoch_id,
        generator_domain_manifest_sha256=manifest.identity_sha256,
        first_action_id=leaf.first_action_id,
        support_hyperedge_sha256=leaf.support_hyperedge_sha256,
        continuation_equivalence_sha256=_sha(continuation),
        horizon=manifest.horizon,
        authority_mode=base.authority_mode,
        axis=base.axis,
        hard_constraint_contract_sha256=base.decode_constraints.receipt_sha256,
        score_implementation_sha256=CANONICAL_SCORE_IMPLEMENTATION_SHA256,
    )


def _typed_receipts(
    binding: ContinuationProofBindingV1,
    endpoint: WholeObjectActionEndpointV1,
) -> tuple[str, bytes, str, bytes]:
    common: dict[str, object] = {
        "action_path": list(endpoint.generated_leaf.action_path),
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
        "terminal_endpoint_identity_sha256": endpoint.endpoint_identity_sha256,
        "terminal_leaf_id": endpoint.action_id,
    }
    measurement = _canonical(
        {
            **common,
            "archive_nbytes": endpoint.archive_nbytes,
            "archive_sha256": endpoint.archive_sha256,
            "d_pose_binary64_hex": float(endpoint.d_pose).hex(),
            "d_seg_binary64_hex": float(endpoint.d_seg).hex(),
            "evaluator_measurement_receipt_sha256": endpoint.measurement_receipt_sha256,
            "public_auth_eval_receipt_sha256": endpoint.public_auth_eval_receipt_sha256,
            "realized_output_sha256": endpoint.realized_output_sha256,
            "receipt_role": "G36_VERIFIED_WHOLE_OBJECT_ACTION_ENDPOINT",
            "schema": FINITE_MEASUREMENT_RECEIPT_SCHEMA,
            "score_implementation_sha256": binding.score_implementation_sha256,
            "upstream_evaluate_sha256": PINNED_UPSTREAM_EVALUATE_SHA256,
        }
    )
    constraints = _canonical(
        {
            **common,
            "blocker_codes": [],
            "decode_constraint_receipt_sha256": endpoint.decode_constraints.receipt_sha256,
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


def _verified_finite_bundle(
    base: WholeObjectBaseV1,
    manifest: GeneratorDomainManifestV1,
    endpoint: WholeObjectActionEndpointV1,
) -> ContinuationCertificateEvidenceBundleV1:
    binding = _binding(
        base,
        manifest,
        endpoint.generated_leaf,
        continuation=f"{endpoint.action_id}-certificate-continuation",
    )
    measurement_sha, measurement, constraint_sha, constraints = _typed_receipts(
        binding,
        endpoint,
    )
    evidence = FiniteTerminalLeafEvidenceV1(
        terminal_leaf_id=endpoint.action_id,
        terminal_endpoint_identity_sha256=endpoint.endpoint_identity_sha256,
        action_path=endpoint.generated_leaf.action_path,
        d_seg=endpoint.d_seg,
        d_pose=endpoint.d_pose,
        archive_nbytes=endpoint.archive_nbytes,
        measurement_receipt_sha256=measurement_sha,
        hard_constraint_receipt_sha256=constraint_sha,
    )
    proof = build_finite_terminal_leaf_proof_bytes(binding=binding, leaves=(evidence,))
    leaf_ids = (endpoint.action_id,)
    dependencies = (constraint_sha, measurement_sha)
    context = ContinuationVerificationContextV1(
        binding=binding,
        expected_subtree_root_sha256=terminal_leaf_set_root_sha256(leaf_ids),
        expected_subtree_cardinality=1,
        expected_covered_terminal_leaf_ids=leaf_ids,
        expected_dependency_sha256s=tuple(sorted(dependencies)),
        expected_proof_sha256=hashlib.sha256(proof).hexdigest(),
    )
    dependency_payloads = {
        measurement_sha: measurement,
        constraint_sha: constraints,
    }
    certificate = verify_continuation_proof_bytes(
        proof,
        context=context,
        generator_domain_manifest_payload_bytes=manifest.canonical_payload_bytes,
        dependency_payloads_by_sha256=dependency_payloads,
    )
    return ContinuationCertificateEvidenceBundleV1(
        certificate=certificate,
        proof_payload_bytes=proof,
        verification_context=context,
        generator_domain_manifest_payload_bytes=manifest.canonical_payload_bytes,
        dependency_payloads=tuple(
            ContinuationProofDependencyPayloadV1(sha256=sha256, payload_bytes=payload)
            for sha256, payload in sorted(dependency_payloads.items())
        ),
    )


def _attempt_primitive_interval_verification(
    base: WholeObjectBaseV1,
    manifest: GeneratorDomainManifestV1,
    leaf: GeneratedTerminalLeafV1,
) -> VerifiedContinuationCertificateV1:
    derivation = f"{leaf.terminal_leaf_id}-interval".encode("ascii")
    coverage = f"{leaf.terminal_leaf_id}-coverage".encode("ascii")
    derivation_sha = hashlib.sha256(derivation).hexdigest()
    coverage_sha = hashlib.sha256(coverage).hexdigest()
    binding = _binding(
        base,
        manifest,
        leaf,
        continuation=f"{leaf.terminal_leaf_id}-interval-continuation",
    )
    evidence = ConservativePrimitiveIntervalLeafEvidenceV1(
        terminal_leaf_id=leaf.terminal_leaf_id,
        d_seg_lower=0.0001,
        d_seg_upper=0.004,
        d_pose_lower=0.0001,
        d_pose_upper=0.01,
        archive_nbytes_lower=60_000,
        archive_nbytes_upper=120_000,
        interval_derivation_receipt_sha256=derivation_sha,
        leaf_coverage_receipt_sha256=coverage_sha,
    )
    proof = build_conservative_primitive_interval_proof_bytes(
        binding=binding,
        leaves=(evidence,),
    )
    leaf_ids = (leaf.terminal_leaf_id,)
    context = ContinuationVerificationContextV1(
        binding=binding,
        expected_subtree_root_sha256=terminal_leaf_set_root_sha256(leaf_ids),
        expected_subtree_cardinality=1,
        expected_covered_terminal_leaf_ids=leaf_ids,
        expected_dependency_sha256s=tuple(sorted((coverage_sha, derivation_sha))),
        expected_proof_sha256=hashlib.sha256(proof).hexdigest(),
    )
    return verify_continuation_proof_bytes(
        proof,
        context=context,
        generator_domain_manifest_payload_bytes=manifest.canonical_payload_bytes,
        dependency_payloads_by_sha256={
            derivation_sha: derivation,
            coverage_sha: coverage,
        },
    )


def test_selects_strictly_dominant_first_action_independent_of_caller_order() -> None:
    base = _base()
    endpoints = _three_endpoints(base)
    first = select_receding_horizon_action_v1(base, _universe(base, endpoints))
    second_endpoints = tuple(reversed(endpoints))
    second = select_receding_horizon_action_v1(
        base,
        _universe(base, second_endpoints, manifest=_manifest(base, second_endpoints)),
    )
    assert first == second
    assert first["decision"]["selected_first_action_id"] == "prune.route"
    assert first["decision"]["selected_terminal_leaf_id"] == "prune.best.terminal"
    assert first["decision"]["strict_finite_horizon_dominance_proved"] is True
    assert first["decision"]["globally_lowest_exact_endpoint_selected"] is False
    assert first["action_universe"]["global_optimality_claim"] is False
    assert first["action_universe"]["family_scale_projection_is_load_bearing"] is False
    assert first["finite_horizon_value"]["horizon"] == 2
    assert first["decision"]["production_commit_authorized"] is False
    assert first["decision"]["pointer_promotion_authorized"] is False


def test_rich_future_beats_lexically_earlier_dead_end_by_terminal_value() -> None:
    base = _base()
    dead = _endpoint(
        base,
        "dead.terminal",
        first_action_id="a.dead_end",
        action_path=("a.dead_end", "dead.noop"),
        d_seg=0.0016,
        d_pose=0.001,
        archive_nbytes=99_000,
    )
    rich = _endpoint(
        base,
        "rich.terminal",
        first_action_id="z.rich_future",
        action_path=("z.rich_future", "inverse.repair"),
        support=_support(("PRUNE", "MICRO"), ("INVERSE_REPAIR", "MESO")),
        d_seg=0.0004,
        d_pose=0.0002,
        archive_nbytes=88_000,
    )
    result = select_receding_horizon_action_v1(base, _universe(base, (dead, rich)))
    assert result["decision"]["selected_first_action_id"] == "z.rich_future"
    branches = {row["first_action_id"]: row for row in result["finite_horizon_value"]["first_action_branches"]}
    assert branches["z.rich_future"]["feasible_terminal_upper_bound"] < branches["a.dead_end"]["terminal_lower_bound"]


def test_equal_terminal_values_with_distinct_continuations_block_instead_of_lexical_tie_break() -> None:
    base = _base()
    one = _endpoint(
        base,
        "one.terminal",
        first_action_id="a.dead_end",
        action_path=("a.dead_end", "one.followup"),
        d_seg=0.001,
        d_pose=0.0005,
        archive_nbytes=98_000,
        continuation="dead-continuation",
    )
    two = _endpoint(
        base,
        "two.terminal",
        first_action_id="z.rich_future",
        action_path=("z.rich_future", "two.followup"),
        d_seg=0.001,
        d_pose=0.0005,
        archive_nbytes=98_000,
        continuation="rich-continuation",
    )
    result = select_receding_horizon_action_v1(base, _universe(base, (one, two)))
    assert result["decision"]["verdict"] == "BLOCKED_CONTINUATION_DOMINANCE_UNPROVED"
    assert result["decision"]["selected_first_action_id"] is None
    assert set(result["finite_horizon_value"]["dominance_blocking_first_action_ids"]) == {"z.rich_future"}


def test_generator_manifest_refuses_caller_selected_family_subset() -> None:
    base = _base()
    endpoints = _three_endpoints(base)
    with pytest.raises(TaskspaceRecedingHorizonError, match="exact full action family"):
        _manifest(base, endpoints, families=("PRUNE",))


def test_missing_generated_leaf_refuses_universe_even_when_every_cell_has_a_row() -> None:
    base = _base()
    endpoints = _three_endpoints(base)
    manifest = _manifest(base, endpoints)
    with pytest.raises(TaskspaceRecedingHorizonError, match="manifest leaf set"):
        _universe(base, endpoints[:-1], manifest=manifest)


def test_omitted_joint_hyperedge_leaf_refuses_universe() -> None:
    base = _base()
    atomic = _endpoint(base, "atomic.terminal")
    joint = _endpoint(
        base,
        "joint.terminal",
        first_action_id="joint.route",
        action_path=("joint.route", "joint.repair"),
        support=_support(("PRUNE", "MICRO"), ("INVERSE_REPAIR", "MESO")),
    )
    manifest = _manifest(base, (atomic, joint))
    with pytest.raises(TaskspaceRecedingHorizonError, match="manifest leaf set"):
        _universe(base, (atomic,), manifest=manifest)


def test_family_scale_projection_must_cover_full_fixed_vocabulary() -> None:
    base = _base()
    universe = _universe(base, _three_endpoints(base))
    assert len(universe.coverage) == len(ACTION_FAMILIES) * len(ACTION_SCALES)
    with pytest.raises(TaskspaceRecedingHorizonError, match="exact full family x scale"):
        replace(universe, coverage=universe.coverage[:-1])


def test_joint_leaf_is_projected_into_every_support_cell() -> None:
    base = _base()
    joint = _endpoint(
        base,
        "joint.terminal",
        support=_support(("PRUNE", "MICRO"), ("INVERSE_REPAIR", "MESO")),
    )
    universe = _universe(base, (joint,))
    rows = {(row.family, row.scale): row for row in universe.coverage}
    assert rows[("PRUNE", "MICRO")].action_ids == ("joint.terminal",)
    assert rows[("INVERSE_REPAIR", "MESO")].action_ids == ("joint.terminal",)


def test_random_scalar_plus_sha_cannot_construct_certified_dominance() -> None:
    with pytest.raises(TaskspaceRecedingHorizonError, match="typed verifier proof"):
        ActionFamilyScaleCoverageV1(
            family="MERGE",
            scale="MICRO",
            status="CERTIFIED_DOMINATED",
            action_ids=(),
            evidence_sha256=_sha("random-sha"),
            blocks_global_selection=False,
            verdict_scope="caller-authored scalar",
            optimistic_score_lower_bound=0.30,
        )


def test_verified_finite_subtree_can_close_manifest_partition_and_win_by_terminal_value() -> None:
    base = _base(advisory=True)
    certified_endpoint = _endpoint(
        base,
        "certified.terminal",
        first_action_id="certified.route",
        action_path=("certified.route", "inverse.finish"),
        support=_support(("PRUNE", "MICRO"), ("INVERSE_REPAIR", "MESO")),
        d_seg=0.0003,
        d_pose=0.0001,
        archive_nbytes=80_000,
    )
    measured = _endpoint(
        base,
        "measured.terminal",
        first_action_id="measured.route",
        action_path=("measured.route", "requantize.finish"),
        d_seg=0.0014,
        d_pose=0.0008,
        archive_nbytes=92_000,
    )
    manifest = _manifest(base, (certified_endpoint, measured))
    bundle = _verified_finite_bundle(base, manifest, certified_endpoint)
    universe = _universe(
        base,
        (measured,),
        manifest=manifest,
        bundles=(bundle,),
    )
    result = select_receding_horizon_action_v1(base, universe)
    assert result["decision"]["selected_first_action_id"] == "certified.route"
    assert result["decision"]["selected_terminal_leaf_id"] == "certified.terminal"
    assert result["decision"]["selected_terminal_endpoint"]["source_kind"] == ("VERIFIED_CONTINUATION_CERTIFICATE")
    assert result["action_universe"]["verified_continuation_certificate_count"] == 1
    assert result["finite_horizon_value"]["all_bounds_exact"] is True


def test_primitive_interval_proof_fails_closed_before_universe_admission() -> None:
    base = _base(advisory=True)
    placeholder = _endpoint(
        base,
        "interval.terminal",
        first_action_id="interval.route",
        action_path=("interval.route", "unknown.finish"),
    )
    manifest = _manifest(base, (placeholder,))
    with pytest.raises(ContinuationCertificateVerificationError, match="typed theorem verification"):
        _attempt_primitive_interval_verification(base, manifest, placeholder.generated_leaf)


def test_bare_certificate_cannot_close_manifest_partition() -> None:
    base = _base(advisory=True)
    certified = _endpoint(base, "certified.terminal")
    manifest = _manifest(base, (certified,))
    bundle = _verified_finite_bundle(base, manifest, certified)
    with pytest.raises(TaskspaceRecedingHorizonError, match="bare continuation certificates"):
        CompleteActionUniverseV1(
            universe_id="universe.bare",
            generator_manifest=manifest,
            coverage=_coverage(manifest.terminal_leaves),
            endpoints=(),
            continuation_evidence_bundles=(bundle.certificate,),  # type: ignore[arg-type]
        )


def test_mutated_certificate_is_rejected_from_valid_proof_bundle() -> None:
    base = _base(advisory=True)
    certified = _endpoint(base, "certified.terminal")
    manifest = _manifest(base, (certified,))
    bundle = _verified_finite_bundle(base, manifest, certified)
    forged = object.__new__(VerifiedContinuationCertificateV1)
    for name, value in asdict(bundle.certificate).items():
        object.__setattr__(forged, name, value)
    object.__setattr__(forged, "terminal_lower_bound", 0.0)
    forged_bundle = replace(bundle, certificate=forged)
    with pytest.raises(ContinuationCertificateVerificationError, match="differs from reverified"):
        _universe(base, (), manifest=manifest, bundles=(forged_bundle,))


def test_certificate_mutated_after_universe_closure_is_reverified_before_selection() -> None:
    base = _base(advisory=True)
    certified = _endpoint(base, "certified.terminal")
    manifest = _manifest(base, (certified,))
    bundle = _verified_finite_bundle(base, manifest, certified)
    universe = _universe(base, (), manifest=manifest, bundles=(bundle,))
    object.__setattr__(bundle.certificate, "terminal_lower_bound", 0.0)
    with pytest.raises(ContinuationCertificateVerificationError, match="differs from reverified"):
        select_receding_horizon_action_v1(base, universe)


def test_production_continuation_bundle_fails_closed_pending_governed_adapter() -> None:
    base = _base()
    certified = _endpoint(base, "certified.terminal")
    manifest = _manifest(base, (certified,))
    with pytest.raises(ContinuationCertificateVerificationError, match="production continuation admission"):
        _verified_finite_bundle(base, manifest, certified)


def test_dependency_payload_set_drift_is_rejected_at_universe_boundary() -> None:
    base = _base(advisory=True)
    certified = _endpoint(base, "certified.terminal")
    manifest = _manifest(base, (certified,))
    bundle = _verified_finite_bundle(base, manifest, certified)
    replacement_payload = b"different typed receipt bytes"
    replacement = ContinuationProofDependencyPayloadV1(
        sha256=hashlib.sha256(replacement_payload).hexdigest(),
        payload_bytes=replacement_payload,
    )
    drifted_bundle = replace(
        bundle,
        dependency_payloads=tuple(sorted((replacement, *bundle.dependency_payloads[1:]), key=lambda row: row.sha256)),
    )
    with pytest.raises(ContinuationCertificateVerificationError, match="proof dependencies"):
        _universe(base, (), manifest=manifest, bundles=(drifted_bundle,))


def test_generator_manifest_payload_drift_is_rejected_before_bound_use() -> None:
    base = _base(advisory=True)
    certified = _endpoint(base, "certified.terminal")
    manifest = _manifest(base, (certified,))
    bundle = _verified_finite_bundle(base, manifest, certified)
    manifest_payload = json.loads(bundle.generator_domain_manifest_payload_bytes)
    manifest_payload["deterministic_seed"] += 1
    drifted_bundle = replace(
        bundle,
        generator_domain_manifest_payload_bytes=_canonical(manifest_payload),
    )
    with pytest.raises(TaskspaceRecedingHorizonError, match="manifest bytes differ"):
        _universe(base, (), manifest=manifest, bundles=(drifted_bundle,))


def test_proof_payload_mutation_is_rejected_at_universe_boundary() -> None:
    base = _base(advisory=True)
    certified = _endpoint(base, "certified.terminal")
    manifest = _manifest(base, (certified,))
    bundle = _verified_finite_bundle(base, manifest, certified)
    proof = json.loads(bundle.proof_payload_bytes)
    proof["proof_body"]["finite_terminal_leaves"][0]["d_seg"] = 0.0
    drifted_bundle = replace(bundle, proof_payload_bytes=_canonical(proof))
    with pytest.raises(ContinuationCertificateVerificationError, match="self-hash"):
        _universe(base, (), manifest=manifest, bundles=(drifted_bundle,))


def test_explicitly_blocked_domain_prevents_finite_manifest_selection() -> None:
    base = _base()
    endpoints = _three_endpoints(base)
    result = select_receding_horizon_action_v1(
        base,
        _universe(base, endpoints, blocking_cell=("TRAIN_QUOTIENT", "SYSTEM")),
    )
    assert result["decision"]["verdict"] == "BLOCKED_GENERATOR_DOMAIN_NOT_CLOSED"
    assert result["decision"]["selected_first_action_id"] is None
    assert result["action_universe"]["blocking_coverage_cells"] == ["TRAIN_QUOTIENT:SYSTEM"]


def test_hard_decode_constraints_exclude_lower_score_terminal() -> None:
    base = _base()
    unsafe = _endpoint(
        base,
        "unsafe.terminal",
        first_action_id="unsafe.route",
        d_seg=0.0001,
        d_pose=0.0001,
        archive_nbytes=70_000,
        constraints=_constraints(source_video_access_absence_proved=False),
    )
    safe = _endpoint(
        base,
        "safe.terminal",
        first_action_id="safe.route",
        d_seg=0.001,
        d_pose=0.0005,
        archive_nbytes=98_000,
    )
    result = select_receding_horizon_action_v1(base, _universe(base, (unsafe, safe)))
    assert result["decision"]["selected_first_action_id"] == "safe.route"
    unsafe_row = next(row for row in result["endpoint_measurements"] if row["action_id"] == "unsafe.terminal")
    assert unsafe_row["hard_constraint_blockers"] == ["SOURCE_VIDEO_ABSENCE_UNPROVED"]


def test_functional_quotient_requires_equal_proof_dependencies() -> None:
    base = _base()
    one = _endpoint(
        base,
        "one.terminal",
        cells="same-cells",
        continuation="same-continuation",
        proof_label="proof-one",
    )
    two = _endpoint(
        base,
        "two.terminal",
        cells="same-cells",
        continuation="same-continuation",
        proof_label="proof-two",
    )
    result = select_receding_horizon_action_v1(base, _universe(base, (one, two)))
    assert result["functional_quotient"]["representative_count"] == 2
    distinctions = result["functional_quotient"]["preserved_future_distinctions"]
    assert len(distinctions) == 1
    assert len(distinctions[0]["continuation_and_proof_dependency_identities"]) == 2


def test_equal_cell_identity_with_different_measurements_is_refused() -> None:
    base = _base()
    one = _endpoint(base, "one.terminal", cells="same-cells", d_seg=0.001)
    two = _endpoint(base, "two.terminal", cells="same-cells", d_seg=0.0011)
    with pytest.raises(TaskspaceRecedingHorizonError, match="equal evaluator-cell identities"):
        select_receding_horizon_action_v1(base, _universe(base, (one, two)))


def test_generated_action_path_cannot_exceed_manifest_horizon() -> None:
    base = _base()
    endpoint = _endpoint(
        base,
        "long.terminal",
        first_action_id="first.action",
        action_path=("first.action", "second.action", "third.action"),
    )
    with pytest.raises(TaskspaceRecedingHorizonError, match="exceeds the finite control horizon"):
        _manifest(base, (endpoint,), horizon=2)


def test_base_stop_is_exact_incumbent_and_no_improvement_is_scoped_fixed_point() -> None:
    base = _base()
    worse = _endpoint(
        base,
        "worse.terminal",
        d_seg=0.003,
        d_pose=0.002,
        archive_nbytes=110_000,
    )
    result = select_receding_horizon_action_v1(base, _universe(base, (worse,)))
    assert result["decision"]["verdict"] == "SCOPED_FINITE_GENERATOR_FIXED_POINT"
    assert result["finite_horizon_value"]["base_stop"]["controller_owned"] is True
    assert result["finite_horizon_value"]["base_stop"]["terminal_lower_bound"] == base.score
    assert result["fixed_point_scope"]["family_level_negative_claimed"] is False


def test_nonselected_exact_branch_evidence_is_retained_after_commit() -> None:
    base = _base()
    result = select_receding_horizon_action_v1(base, _universe(base, _three_endpoints(base)))
    transition = result["receding_horizon_transition"]
    assert transition["nonselected_exact_branch_receipts_discarded"] is False
    assert set(transition["retained_nonselected_exact_terminal_evidence_ids"]) == {
        "prune.modest.terminal",
        "macro.evict.terminal",
    }
    assert set(transition["stale_local_marginal_action_ids"]) == {
        "prune.modest.terminal",
        "prune.best.terminal",
        "macro.evict.terminal",
    }


def test_wall_time_scope_must_cover_complete_public_evaluate_workflow() -> None:
    with pytest.raises(TaskspaceRecedingHorizonError, match=r"complete upstream/evaluate\.sh"):
        _constraints(measurement_scope="INFLATE_ONLY")


def test_infeasible_base_cannot_be_used_as_base_stop_incumbent() -> None:
    with pytest.raises(TaskspaceRecedingHorizonError, match="not a feasible public-workflow incumbent"):
        replace(_base(), decode_constraints=_constraints(recursive_auth_eval_closed=False))


def test_stale_base_is_refused_before_terminal_value_comparison() -> None:
    base = _base()
    endpoints = _three_endpoints(base)
    universe = _universe(base, endpoints)
    changed = replace(base, archive_sha256=_sha("different-base"))
    with pytest.raises(TaskspaceRecedingHorizonError, match="stale"):
        select_receding_horizon_action_v1(changed, universe)


def test_endpoint_axis_cannot_be_laundered_through_production_base() -> None:
    base = _base()
    endpoints = _three_endpoints(base)
    mismatched = (
        replace(endpoints[0], authority_mode="RESEARCH_ADVISORY", axis="[macOS-CPU advisory]"),
        *endpoints[1:],
    )
    universe = _universe(base, mismatched)
    with pytest.raises(TaskspaceRecedingHorizonError, match="authority axis"):
        select_receding_horizon_action_v1(base, universe)


def test_advisory_axis_can_recommend_but_never_authorize_production_commit() -> None:
    base = _base(advisory=True)
    result = select_receding_horizon_action_v1(base, _universe(base, _three_endpoints(base)))
    assert result["decision"]["verdict"] == "ADVISORY_FINITE_HORIZON_FIRST_ACTION_NO_PRODUCTION_COMMIT"
    assert result["decision"]["selected_first_action_id"] == "prune.route"
    assert result["decision"]["production_commit_authorized"] is False
    assert result["truth"]["score_claim"] is False
