from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tac.witness_dsl.coupled_witness_state import (
    DECODER_PAYLOAD_POLICY,
    SCIENTIFIC_STREAM_ORDER,
    CodecObjectManifest,
    CompileStreamPolicy,
    ContentAddress,
    CoupledWitnessState,
    CoupledWitnessStateError,
    FrozenSpaceIdentity,
    ScientificStream,
    ScientificStreamDependency,
    ScientificStreamRole,
    StatePatch,
    StateTransitionReceipt,
    WitnessCompileConfig,
    apply_state_patch,
    canonical_json_bytes,
    canonical_sha256,
    decode_canonical_json,
)


def _address(name: str, payload: bytes | None = None) -> ContentAddress:
    data = payload if payload is not None else name.encode("ascii")
    return ContentAddress.from_payload(
        artifact_id=name,
        artifact_schema=f"test.{name}.v1",
        payload=data,
    )


def _frozen_space() -> FrozenSpaceIdentity:
    pair_count = 6
    return FrozenSpaceIdentity(
        source_video=_address("source-video"),
        evaluator_artifacts=tuple(
            sorted(
                (
                    _address("upstream/evaluate.py"),
                    _address("upstream/models/posenet.safetensors"),
                    _address("upstream/models/segnet.safetensors"),
                    _address("upstream/modules.py"),
                ),
                key=lambda item: item.artifact_id,
            )
        ),
        pair_count=pair_count,
        pair_order_id="canonical-contiguous-pairs.v1",
        pair_order_sha256=canonical_sha256(list(range(pair_count))),
        scorer_height=384,
        scorer_width=512,
    )


def _stream(
    role: ScientificStreamRole,
    suffix: str = "a",
    *,
    dependencies: tuple[ScientificStream, ...] = (),
) -> ScientificStream:
    dependency_roles = {
        ScientificStreamRole.TOPOLOGY_WORLDSHEET: (),
        ScientificStreamRole.BULK_BOUNDARY: (ScientificStreamRole.TOPOLOGY_WORLDSHEET,),
        ScientificStreamRole.LANE_CHART: (ScientificStreamRole.TOPOLOGY_WORLDSHEET,),
        ScientificStreamRole.MOVABLE_MYCAR: (ScientificStreamRole.TOPOLOGY_WORLDSHEET,),
        ScientificStreamRole.CELL_VALUE_PREIMAGE: (
            ScientificStreamRole.TOPOLOGY_WORLDSHEET,
            ScientificStreamRole.BULK_BOUNDARY,
            ScientificStreamRole.LANE_CHART,
            ScientificStreamRole.MOVABLE_MYCAR,
        ),
        ScientificStreamRole.POSE_TRANSPORT_FRAME0: (
            ScientificStreamRole.TOPOLOGY_WORLDSHEET,
            ScientificStreamRole.CELL_VALUE_PREIMAGE,
        ),
        ScientificStreamRole.IRREDUCIBLE_QUOTIENT: SCIENTIFIC_STREAM_ORDER[:-1],
    }
    by_role = {stream.role: stream for stream in dependencies}
    dependency_rows = tuple(
        ScientificStreamDependency(
            role=dependency_role,
            content_sha256=by_role[dependency_role].content.sha256,
        )
        for dependency_role in dependency_roles[role]
    )
    return ScientificStream(
        role=role,
        content=_address(f"{role.value}-{suffix}"),
        provenance_manifest=_address(f"{role.value}-{suffix}-provenance"),
        dependencies=dependency_rows,
    )


def _full_state() -> CoupledWitnessState:
    root = CoupledWitnessState.empty(
        _frozen_space(),
        generation_seed=0,
        generation_rng_id="numpy-pcg64-derived.v1",
    )
    current = root
    for role in SCIENTIFIC_STREAM_ORDER:
        patch = StatePatch(
            patch_id=f"set-{role.value}",
            expected_parent_state_sha256=current.state_sha256,
            set_streams=(_stream(role, dependencies=current.streams),),
            remove_roles=(),
            rationale="test the canonical dependency order",
            provenance_ref="src/tac/tests/test_coupled_witness_state.py",
        )
        current, _ = apply_state_patch(current, patch)
    return current


def _compile_config(state: CoupledWitnessState) -> WitnessCompileConfig:
    return WitnessCompileConfig(
        container_id="test-container.v1",
        receiver_contract_id="factor2-test-receiver.v1",
        receiver_artifacts=tuple(
            sorted(
                (
                    _address("receiver.py"),
                    _address("uint8-lattice.py"),
                ),
                key=lambda item: item.artifact_id,
            )
        ),
        r_chain_id="factor2-half-pixel-uint8.v1",
        tie_policy_id="native-f32-first-max.v1",
        camera_height=768,
        camera_width=1024,
        scorer_height=384,
        scorer_width=512,
        decoder_seed=0,
        stream_policies=tuple(
            CompileStreamPolicy(
                role=role,
                coder_id="brotli-v1",
                precision_id="exact-test",
                section_id=f"section-{role.value}",
            )
            for role in state.present_roles
        ),
        decoder_payload_policy=DECODER_PAYLOAD_POLICY,
    )


def test_empty_state_compile_and_codec_object_roundtrip_is_deterministic() -> None:
    state = CoupledWitnessState.empty(
        _frozen_space(),
        generation_seed=0,
        generation_rng_id="numpy-pcg64-derived.v1",
    )
    config = _compile_config(state)
    codec_object = CodecObjectManifest.bind(state, config)

    assert state.next_missing_role is ScientificStreamRole.TOPOLOGY_WORLDSHEET
    assert codec_object.stage == "C0_IDENTITY_SCAFFOLD"
    assert CoupledWitnessState.from_bytes(state.to_bytes()) == state
    assert WitnessCompileConfig.from_bytes(config.to_bytes()) == config
    assert CodecObjectManifest.from_bytes(codec_object.to_bytes()) == codec_object
    assert state.to_bytes() == CoupledWitnessState.from_bytes(state.to_bytes()).to_bytes()
    assert config.to_bytes() == WitnessCompileConfig.from_bytes(config.to_bytes()).to_bytes()
    assert codec_object.to_bytes() == CodecObjectManifest.from_bytes(codec_object.to_bytes()).to_bytes()


def test_pair_order_is_derived_not_caller_asserted() -> None:
    frozen = _frozen_space()
    payload = frozen.as_dict()
    payload["pair_order_sha256"] = "0" * 64
    with pytest.raises(CoupledWitnessStateError, match="canonical contiguous pair order"):
        FrozenSpaceIdentity.from_dict(payload)


def test_evaluator_artifact_order_and_identity_are_part_of_frozen_space() -> None:
    frozen = _frozen_space()
    with pytest.raises(CoupledWitnessStateError, match="uniquely sorted"):
        FrozenSpaceIdentity(
            source_video=frozen.source_video,
            evaluator_artifacts=tuple(reversed(frozen.evaluator_artifacts)),
            pair_count=frozen.pair_count,
            pair_order_id=frozen.pair_order_id,
            pair_order_sha256=frozen.pair_order_sha256,
            scorer_height=frozen.scorer_height,
            scorer_width=frozen.scorer_width,
        )


def test_content_address_verifies_exact_payload() -> None:
    address = _address("payload", b"exact")
    address.verify_payload(b"exact")
    with pytest.raises(CoupledWitnessStateError, match="differs from content address"):
        address.verify_payload(b"inexact")


def test_content_address_refuses_noncanonical_uppercase_sha256() -> None:
    with pytest.raises(CoupledWitnessStateError, match="lowercase SHA-256"):
        ContentAddress(
            artifact_id="uppercase",
            artifact_schema="test.uppercase.v1",
            sha256="A" * 64,
            byte_length=0,
        )


@pytest.mark.parametrize(
    ("lineage", "borrowed", "match"),
    [
        ("public-pr-archive", 0, "original source-derived lineage"),
        ("source-video-derived-our-original-build", 1, "borrowed candidate bytes"),
    ],
)
def test_scientific_stream_rejects_foreign_candidate_lineage(
    lineage: str,
    borrowed: int,
    match: str,
) -> None:
    with pytest.raises(CoupledWitnessStateError, match=match):
        ScientificStream(
            role=ScientificStreamRole.TOPOLOGY_WORLDSHEET,
            content=_address("foreign"),
            provenance_manifest=_address("foreign-provenance"),
            dependencies=(),
            lineage=lineage,
            borrowed_candidate_bytes=borrowed,
        )


def test_v10_preimage_cannot_exist_before_complete_v9_semantic_state() -> None:
    with pytest.raises(CoupledWitnessStateError, match="missing dependencies"):
        dependencies = tuple(
            ScientificStreamDependency(role=role, content_sha256="a" * 64)
            for role in (
                ScientificStreamRole.TOPOLOGY_WORLDSHEET,
                ScientificStreamRole.BULK_BOUNDARY,
                ScientificStreamRole.LANE_CHART,
                ScientificStreamRole.MOVABLE_MYCAR,
            )
        )
        CoupledWitnessState(
            frozen_space=_frozen_space(),
            generation_seed=0,
            generation_rng_id="numpy-pcg64-derived.v1",
            parent_state_sha256=None,
            transition_index=0,
            streams=(
                ScientificStream(
                    role=ScientificStreamRole.CELL_VALUE_PREIMAGE,
                    content=_address("cell-value"),
                    provenance_manifest=_address("cell-value-provenance"),
                    dependencies=dependencies,
                ),
            ),
        )


def test_patch_is_parent_bound_and_emits_identity_transition_receipt() -> None:
    root = CoupledWitnessState.empty(
        _frozen_space(),
        generation_seed=0,
        generation_rng_id="numpy-pcg64-derived.v1",
    )
    patch = StatePatch(
        patch_id="topology-seed",
        expected_parent_state_sha256=root.state_sha256,
        set_streams=(_stream(ScientificStreamRole.TOPOLOGY_WORLDSHEET),),
        remove_roles=(),
        rationale="create the first finite topology stream",
        provenance_ref="test",
    )
    child, receipt = apply_state_patch(root, patch)

    assert child.parent_state_sha256 == root.state_sha256
    assert child.transition_index == 1
    assert child.state_sha256 != root.state_sha256
    assert receipt.from_state_sha256 == root.state_sha256
    assert receipt.to_state_sha256 == child.state_sha256
    assert receipt.patch_sha256 == patch.patch_sha256
    assert receipt.changed_roles == (ScientificStreamRole.TOPOLOGY_WORLDSHEET,)
    assert StatePatch.from_bytes(patch.to_bytes()) == patch
    assert StateTransitionReceipt.from_bytes(receipt.to_bytes()) == receipt
    receipt.validate_against(root, patch, child)

    forged_receipts = (
        replace(receipt, patch_id="forged-patch-id"),
        replace(receipt, patch_sha256="f" * 64),
        replace(receipt, from_state_sha256="e" * 64),
        replace(receipt, to_state_sha256="d" * 64),
        replace(receipt, transition_index=receipt.transition_index + 1),
        replace(receipt, changed_roles=(ScientificStreamRole.BULK_BOUNDARY,)),
    )
    for forged in forged_receipts:
        with pytest.raises(
            CoupledWitnessStateError,
            match="foreign keys or changed-role semantics",
        ):
            forged.validate_against(root, patch, child)

    stale_patch = StatePatch(
        patch_id="stale",
        expected_parent_state_sha256=root.state_sha256,
        set_streams=(_stream(ScientificStreamRole.TOPOLOGY_WORLDSHEET, "b"),),
        remove_roles=(),
        rationale="must fail against a different parent",
        provenance_ref="test",
    )
    with pytest.raises(CoupledWitnessStateError, match="parent state identity"):
        apply_state_patch(child, stale_patch)


def test_transition_validator_refuses_child_endpoint_mutations() -> None:
    root = CoupledWitnessState.empty(
        _frozen_space(),
        generation_seed=0,
        generation_rng_id="numpy-pcg64-derived.v1",
    )
    patch = StatePatch(
        patch_id="topology-seed",
        expected_parent_state_sha256=root.state_sha256,
        set_streams=(_stream(ScientificStreamRole.TOPOLOGY_WORLDSHEET),),
        remove_roles=(),
        rationale="exercise durable transition validation",
        provenance_ref="test",
    )
    child, receipt = apply_state_patch(root, patch)

    wrong_generation = replace(child, generation_seed=1)
    with pytest.raises(CoupledWitnessStateError, match="generation identity"):
        receipt.validate_against(root, patch, wrong_generation)

    wrong_parent = replace(child, parent_state_sha256="f" * 64)
    with pytest.raises(CoupledWitnessStateError, match="parent-state foreign key"):
        receipt.validate_against(root, patch, wrong_parent)

    wrong_index = replace(child, transition_index=child.transition_index + 1)
    with pytest.raises(CoupledWitnessStateError, match="transition index"):
        receipt.validate_against(root, patch, wrong_index)


def test_patch_refuses_noop_and_dependency_breaking_delete() -> None:
    full = _full_state()
    existing = next(stream for stream in full.streams if stream.role is ScientificStreamRole.TOPOLOGY_WORLDSHEET)
    noop = StatePatch(
        patch_id="noop",
        expected_parent_state_sha256=full.state_sha256,
        set_streams=(existing,),
        remove_roles=(),
        rationale="must be rejected",
        provenance_ref="test",
    )
    with pytest.raises(CoupledWitnessStateError, match="does not change"):
        apply_state_patch(full, noop)

    breaking = StatePatch(
        patch_id="remove-topology",
        expected_parent_state_sha256=full.state_sha256,
        set_streams=(),
        remove_roles=(ScientificStreamRole.TOPOLOGY_WORLDSHEET,),
        rationale="must break downstream dependencies",
        provenance_ref="test",
    )
    with pytest.raises(CoupledWitnessStateError, match="missing dependencies"):
        apply_state_patch(full, breaking)


def test_upstream_replacement_cannot_leave_stale_descendants_attached() -> None:
    full = _full_state()
    stale = StatePatch(
        patch_id="replace-topology-only",
        expected_parent_state_sha256=full.state_sha256,
        set_streams=(_stream(ScientificStreamRole.TOPOLOGY_WORLDSHEET, "changed"),),
        remove_roles=(),
        rationale="must not preserve descendants derived from the old topology",
        provenance_ref="test",
    )
    with pytest.raises(CoupledWitnessStateError, match="stale dependency"):
        apply_state_patch(full, stale)


def test_compile_policy_must_exactly_cover_state_and_match_frozen_geometry() -> None:
    state = _full_state()
    config = _compile_config(state)
    config.validate_for_state(state)

    root = CoupledWitnessState.empty(
        state.frozen_space,
        generation_seed=0,
        generation_rng_id="numpy-pcg64-derived.v1",
    )
    with pytest.raises(CoupledWitnessStateError, match="exactly cover"):
        config.validate_for_state(root)

    payload = config.as_dict()
    payload["scorer_height"] = 383
    wrong_geometry = WitnessCompileConfig.from_dict(payload)
    with pytest.raises(CoupledWitnessStateError, match="geometry differs"):
        wrong_geometry.validate_for_state(state)


def test_coder_race_changes_compile_and_object_but_not_scientific_state() -> None:
    state = _full_state()
    config_a = _compile_config(state)
    payload = config_a.as_dict()
    payload["stream_policies"][0]["coder_id"] = "range-v2"
    config_b = WitnessCompileConfig.from_dict(payload)

    assert state.state_sha256 == state.state_sha256
    assert config_a.config_sha256 != config_b.config_sha256
    assert (
        CodecObjectManifest.bind(state, config_a).object_sha256
        != CodecObjectManifest.bind(state, config_b).object_sha256
    )


def test_logical_streams_may_share_one_joint_physical_entropy_section() -> None:
    state = _full_state()
    payload = _compile_config(state).as_dict()
    for policy in payload["stream_policies"]:
        policy["section_id"] = "joint-conditioned-semantic-stream"
    joint = WitnessCompileConfig.from_dict(payload)
    joint.validate_for_state(state)
    assert {policy.section_id for policy in joint.stream_policies} == {"joint-conditioned-semantic-stream"}


def test_codec_object_is_metadata_scaffold_not_archive_or_score_receipt() -> None:
    state = _full_state()
    config = _compile_config(state)
    manifest = CodecObjectManifest.bind(state, config)
    payload = manifest.as_dict()
    payload["archive"] = _address("archive.zip").as_dict()
    with pytest.raises(CoupledWitnessStateError, match="fields differ"):
        CodecObjectManifest.from_dict(payload)

    with pytest.raises(CoupledWitnessStateError, match="never score or promotion authority"):
        CodecObjectManifest(
            state_sha256=state.state_sha256,
            frozen_space_sha256=state.frozen_space.identity_sha256,
            compile_config_sha256=config.config_sha256,
            receiver_bundle_sha256=config.receiver_bundle_sha256,
            score_claim=True,
        )


def test_stream_provenance_manifest_changes_scientific_state_identity() -> None:
    root = CoupledWitnessState.empty(
        _frozen_space(),
        generation_seed=0,
        generation_rng_id="numpy-pcg64-derived.v1",
    )
    stream_a = _stream(ScientificStreamRole.TOPOLOGY_WORLDSHEET, "a")
    stream_b = ScientificStream(
        role=stream_a.role,
        content=stream_a.content,
        provenance_manifest=_address("different-provenance"),
        dependencies=(),
    )
    state_a = CoupledWitnessState(
        frozen_space=root.frozen_space,
        generation_seed=root.generation_seed,
        generation_rng_id=root.generation_rng_id,
        parent_state_sha256=None,
        transition_index=0,
        streams=(stream_a,),
    )
    state_b = CoupledWitnessState(
        frozen_space=root.frozen_space,
        generation_seed=root.generation_seed,
        generation_rng_id=root.generation_rng_id,
        parent_state_sha256=None,
        transition_index=0,
        streams=(stream_b,),
    )
    assert state_a.state_sha256 != state_b.state_sha256


def test_codec_object_foreign_keys_detect_state_or_receiver_transplant() -> None:
    state = _full_state()
    config = _compile_config(state)
    manifest = CodecObjectManifest.bind(state, config)

    changed_streams = list(state.streams)
    changed_streams[-1] = _stream(
        ScientificStreamRole.IRREDUCIBLE_QUOTIENT,
        "changed",
        dependencies=tuple(changed_streams[:-1]),
    )
    changed_state = CoupledWitnessState(
        frozen_space=state.frozen_space,
        generation_seed=state.generation_seed,
        generation_rng_id=state.generation_rng_id,
        parent_state_sha256=state.parent_state_sha256,
        transition_index=state.transition_index,
        streams=tuple(changed_streams),
    )
    with pytest.raises(CoupledWitnessStateError, match="foreign keys differ"):
        manifest.validate_against(changed_state, _compile_config(changed_state))

    config_payload = config.as_dict()
    config_payload["receiver_artifacts"][0]["sha256"] = "f" * 64
    config_payload["receiver_bundle_sha256"] = canonical_sha256(config_payload["receiver_artifacts"])
    changed_receiver = WitnessCompileConfig.from_dict(config_payload)
    with pytest.raises(CoupledWitnessStateError, match="foreign keys differ"):
        manifest.validate_against(state, changed_receiver)


def test_serialization_rejects_duplicate_keys_and_noncanonical_spelling() -> None:
    with pytest.raises(CoupledWitnessStateError, match="duplicate JSON key"):
        decode_canonical_json(b'{"a":1,"a":2}')
    with pytest.raises(CoupledWitnessStateError, match="not canonical"):
        decode_canonical_json(b'{"a": 1}')
    with pytest.raises(CoupledWitnessStateError, match=r"mapping key.*must be a string"):
        canonical_json_bytes({1: "integer key collides with JSON stringification"})
    assert decode_canonical_json(canonical_json_bytes({"a": 1})) == {"a": 1}


def test_envelope_hash_tampering_fails_closed() -> None:
    state = CoupledWitnessState.empty(
        _frozen_space(),
        generation_seed=0,
        generation_rng_id="numpy-pcg64-derived.v1",
    )
    envelope = json.loads(state.to_bytes())
    envelope["body_sha256"] = "0" * 64
    with pytest.raises(CoupledWitnessStateError, match="body hash differs"):
        CoupledWitnessState.from_bytes(canonical_json_bytes(envelope))
