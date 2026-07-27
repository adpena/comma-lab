# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.direct_description_carrier_compose import BoundaryShearletAtomV1
from tac.optimization.uint8_lattice_feasibility import (
    realize_factor2_uint8_scorer_plane,
)
from tac.witness_dsl import (
    taskspace_selected_preimage_operand_adapter_v1 as selected_adapter,
)
from tac.witness_dsl.taskspace_fresh_selected_plane_codec_v1 import (
    FreshOperandProviderV1,
)
from tac.witness_dsl.taskspace_selected_preimage_operand_adapter_v1 import (
    BLOCKED_REPRESENTATION_STATUS,
    FIXTURE_AUXILIARY_AGGREGATE_SCHEMA,
    NEXT_PRECLOSURE_GATE,
    PROGRAM_RESIDUAL_MODE,
    AuxiliaryOperandCustodyV1,
    LearnedDecoderSourceCustodyV1,
    ReopenableRegularFileIdentityV1,
    SelectedPreimageOperandAdapterError,
    SelectedPreimageProductionCustodyV1,
    TaskspaceSelectedPreimageFreshOperandAdapterV1,
    publish_program_residual_outer_archive_proof,
    reopen_program_residual_outer_archive,
    validate_pre_encode_stage,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v1 import (
    GENERIC_V10_FACTOR2_DECODER_ID,
    V15_SEMANTIC_COMPILE_DERIVATION,
    V15_SEMANTIC_COMPILE_RECEIPT_SCHEMA,
    GenericV10Factor2DecoderIdentityV1,
    ScorerTargetCustodyIdentityV1,
    SelectedPreimageCompileConfigV1,
    SelectedPreimageFactor2PairV1,
    SelectedPreimageFrameSelectorV1,
    TaskspaceSelectedPreimageFactorV1,
    TaskspaceSelectedPreimageProgramV1,
    V15SemanticProgramIdentityV1,
    build_analytic_shearlet_residual_factor,
    build_learned_irreducible_quotient_factor,
)

PAIR_COUNT = 4
PAIRS_PER_STAGE = 2
HEIGHT = 384
WIDTH = 512
LEARNED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


@dataclass(frozen=True, slots=True)
class _AuxStage:
    pair_range: tuple[int, int]
    pair_ids: np.ndarray
    y0_u8: np.ndarray
    y1_u8: np.ndarray
    target_labels_u8: np.ndarray
    gt_poses_f32: np.ndarray
    pose_authority: str = "SEALED_SOURCE_CACHE_ADVISORY_ONLY"


class _AuxProvider:
    def __init__(
        self,
        *,
        labels: np.ndarray,
        poses: np.ndarray,
        direct_y0: np.ndarray | None = None,
        direct_y1: np.ndarray | None = None,
    ) -> None:
        self.labels = labels
        self.poses = poses
        self.direct_y0 = (
            np.full((PAIR_COUNT, HEIGHT, WIDTH, 3), 203, dtype=np.uint8) if direct_y0 is None else direct_y0
        )
        self.direct_y1 = (
            np.full((PAIR_COUNT, HEIGHT, WIDTH, 3), 197, dtype=np.uint8) if direct_y1 is None else direct_y1
        )

    def iter_stages(self, *, max_pairs: int = 120):
        assert max_pairs >= PAIRS_PER_STAGE
        for start in range(0, PAIR_COUNT, PAIRS_PER_STAGE):
            stop = start + PAIRS_PER_STAGE
            yield _AuxStage(
                pair_range=(start, stop),
                pair_ids=np.arange(start, stop, dtype=np.int64),
                y0_u8=self.direct_y0[start:stop],
                y1_u8=self.direct_y1[start:stop],
                target_labels_u8=self.labels[start:stop],
                gt_poses_f32=self.poses[start:stop],
            )


class _Decoder:
    decoder_id = GENERIC_V10_FACTOR2_DECODER_ID

    def __init__(
        self,
        *,
        semantic_identity: V15SemanticProgramIdentityV1,
        target_identity: ScorerTargetCustodyIdentityV1,
        inert: bool = False,
        analytic_base: bool = False,
    ) -> None:
        self.semantic_identity = semantic_identity
        self.target_identity = target_identity
        self.inert = inert
        self.analytic_base = analytic_base

    @property
    def implementation_source_sha256(self) -> str:
        return GenericV10Factor2DecoderIdentityV1.current().implementation_source_sha256

    def verify_semantic_program_identity(
        self,
        identity: V15SemanticProgramIdentityV1,
    ) -> bool:
        return identity == self.semantic_identity

    def verify_target_custody_identity(
        self,
        identity: ScorerTargetCustodyIdentityV1,
    ) -> bool:
        return identity == self.target_identity

    def decode_semantic_base_pair(
        self,
        source_pair_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        y0 = np.full((HEIGHT, WIDTH, 3), source_pair_id + 3, dtype=np.uint8)
        y1 = np.full((HEIGHT, WIDTH, 3), source_pair_id + 7, dtype=np.uint8)
        if self.analytic_base and source_pair_id == 0:
            y0[:] = 0
            y0[160:220, 120:392] = (10, 20, 30)
        return y0, y1

    def learned_quotient_decoder_contract_id(
        self,
        factor: TaskspaceSelectedPreimageFactorV1,
    ) -> str:
        del factor
        return "tac.test.generic_selected_preimage_factor_decoder.v1"

    def learned_quotient_decoder_implementation_source_sha256(
        self,
        factor: TaskspaceSelectedPreimageFactorV1,
    ) -> str:
        del factor
        return LEARNED_SOURCE_SHA256

    def apply_learned_irreducible_quotient(
        self,
        *,
        factor: TaskspaceSelectedPreimageFactorV1,
        source_pair_id: int,
        scorer_y0: np.ndarray,
        scorer_y1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        del factor
        output0 = scorer_y0.copy()
        output1 = scorer_y1.copy()
        if not self.inert:
            output1[0, 0] = (source_pair_id + 101, 17, 23)
        return output0, output1

    def realize_factor2_pair(
        self,
        scorer_y0: np.ndarray,
        scorer_y1: np.ndarray,
    ) -> SelectedPreimageFactor2PairV1:
        del scorer_y0, scorer_y1
        raise AssertionError("factor-2 realization is outside this structural fixture")


@dataclass(frozen=True, slots=True)
class _Fixture:
    semantic_bytes: bytes
    labels: np.ndarray
    poses: np.ndarray
    auxiliary: _AuxProvider
    custody: AuxiliaryOperandCustodyV1
    program: TaskspaceSelectedPreimageProgramV1
    decoder: _Decoder


def _fixture(
    tmp_path: Path,
    *,
    inert: bool = False,
    direct_y0: np.ndarray | None = None,
    direct_y1: np.ndarray | None = None,
    analytic_only: bool = False,
) -> _Fixture:
    semantic_bytes = b"fresh-own-lineage-semantic-program"
    labels = (
        np.arange(PAIR_COUNT * HEIGHT * WIDTH, dtype=np.uint32).reshape(PAIR_COUNT, HEIGHT, WIDTH).astype(np.uint8) % 5
    )
    poses = np.arange(PAIR_COUNT * 6, dtype=np.float32).reshape(PAIR_COUNT, 6)
    labels_sha256 = _sha256(labels.tobytes(order="C"))
    poses_sha256 = _sha256(poses.tobytes(order="C"))
    auxiliary = _AuxProvider(
        labels=labels,
        poses=poses,
        direct_y0=direct_y0,
        direct_y1=direct_y1,
    )
    aggregate = tmp_path / "fresh_operand_aggregate_receipt.json"
    aggregate.parent.mkdir(parents=True, exist_ok=True)
    aggregate_body = {
        "gt_poses_f32_sha256": poses_sha256,
        "historical_payload_reused": False,
        "pair_count": PAIR_COUNT,
        "pose_authority": "SEALED_SOURCE_CACHE_ADVISORY_ONLY",
        "schema": FIXTURE_AUXILIARY_AGGREGATE_SCHEMA,
        "scorer_batch_size": 16,
        "stage_count": PAIR_COUNT // PAIRS_PER_STAGE,
        "stage_pairs": PAIRS_PER_STAGE,
        "stages": [
            {
                "direct_source_y0_sha256": _sha256(auxiliary.direct_y0[start:stop].tobytes(order="C")),
                "direct_source_y1_sha256": _sha256(auxiliary.direct_y1[start:stop].tobytes(order="C")),
                "gt_poses_f32_sha256": _sha256(poses[start:stop].tobytes(order="C")),
                "pair_range": [start, stop],
                "target_labels_sha256": _sha256(labels[start:stop].tobytes(order="C")),
            }
            for start in range(0, PAIR_COUNT, PAIRS_PER_STAGE)
            for stop in (start + PAIRS_PER_STAGE,)
        ],
        "target_labels_sha256": labels_sha256,
    }
    aggregate_payload = {
        **aggregate_body,
        "aggregate_receipt_sha256": _sha256(_canonical_json(aggregate_body)),
    }
    aggregate.write_bytes(_canonical_json(aggregate_payload) + b"\n")
    custody = AuxiliaryOperandCustodyV1(
        aggregate_receipt_path=str(aggregate.resolve()),
        aggregate_receipt_sha256=_sha256(aggregate.read_bytes()),
    )
    semantic_identity = V15SemanticProgramIdentityV1(
        fresh_compile_receipt_schema=V15_SEMANTIC_COMPILE_RECEIPT_SCHEMA,
        fresh_compile_receipt_sha256="1" * 64,
        compile_proof_dependency_sha256="2" * 64,
        typed_compile_config_sha256="3" * 64,
        compiler_source_sha256="4" * 64,
        receiver_source_sha256="5" * 64,
        compiled_semantic_archive_sha256=_sha256(semantic_bytes),
        compiled_semantic_archive_bytes=len(semantic_bytes),
        source_pair_start=0,
        pair_count=PAIR_COUNT,
        declared_compile_dependency_sha256s=("6" * 64, "7" * 64),
        compile_derivation=V15_SEMANTIC_COMPILE_DERIVATION,
    )
    target_identity = ScorerTargetCustodyIdentityV1(
        target_custody_receipt_sha256="8" * 64,
        target_bank_sha256=labels_sha256,
    )
    if analytic_only:
        factor = build_analytic_shearlet_residual_factor(
            section_id="analytic.compact.factor",
            source_pair_start=0,
            source_pair_stop_exclusive=1,
            frame_selector=SelectedPreimageFrameSelectorV1.Y0,
            source_rgb_u8=(10, 20, 30),
            added_rgb_u8=(80, 81, 82),
            removed_rgb_u8=(1, 1, 1),
            atoms=(
                BoundaryShearletAtomV1(
                    pair_index=0,
                    role="Road",
                    center_y=160,
                    center_x=256,
                    scale_y=24,
                    scale_x=96,
                    shear_q4=0,
                    amplitude_q4=64,
                ),
            ),
            source_receipt_sha256="a" * 64,
        )
    else:
        factor = build_learned_irreducible_quotient_factor(
            section_id="learned.compact.factor",
            source_pair_start=0,
            source_pair_stop_exclusive=PAIR_COUNT,
            decoder_contract_id=("tac.test.generic_selected_preimage_factor_decoder.v1"),
            decoder_implementation_source_sha256=LEARNED_SOURCE_SHA256,
            model_family_id="fixture_compact_pair_delta",
            latent_codec_id="fixture_delta_i8",
            parameter_codec_id="fixture_weights_i8",
            latent_dtype="int8",
            parameter_dtype="int8",
            latent_payload=b"\x01\x02\x03\x04",
            parameter_payload=b"\x05\x06\x07",
            source_receipt_sha256="a" * 64,
        )
    program = TaskspaceSelectedPreimageProgramV1(
        semantic_program_identity=semantic_identity,
        target_custody_identity=target_identity,
        decoder_identity=GenericV10Factor2DecoderIdentityV1.current(),
        compile_config=SelectedPreimageCompileConfigV1(
            source_pair_start=0,
            pair_count=PAIR_COUNT,
            maximum_packet_bytes=1 << 20,
            score_budget_receipt_sha256="b" * 64,
            budget_rule_id="fixture_dynamic_frontier_budget",
        ),
        factors=(factor,),
    )
    return _Fixture(
        semantic_bytes=semantic_bytes,
        labels=labels,
        poses=poses,
        auxiliary=auxiliary,
        custody=custody,
        program=program,
        decoder=_Decoder(
            semantic_identity=semantic_identity,
            target_identity=target_identity,
            inert=inert,
            analytic_base=analytic_only,
        ),
    )


def _adapter(fixture: _Fixture) -> TaskspaceSelectedPreimageFreshOperandAdapterV1:
    return TaskspaceSelectedPreimageFreshOperandAdapterV1(
        program=fixture.program,
        decoder=fixture.decoder,
        auxiliary_provider=fixture.auxiliary,
        auxiliary_custody=fixture.custody,
        pairs_per_stage=PAIRS_PER_STAGE,
        test_only_small_fixture=True,
    )


def test_adapter_is_g52_provider_and_recurrently_proves_g49_planes(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    adapter = _adapter(fixture)

    assert isinstance(adapter, FreshOperandProviderV1)
    identity = adapter.pre_encode_identity_receipt
    assert identity["representation_source"].startswith("G49_")
    assert identity["program_packet"]["sha256"] == fixture.program.packet_sha256
    assert identity["semantic_program"]["sha256"] == _sha256(fixture.semantic_bytes)
    assert identity["target_custody"]["bank_sha256"] == _sha256(fixture.labels.tobytes())
    assert identity["auxiliary_planes_forwarded"] is False
    assert identity["g49_decode_custody_verified"] is True
    assert identity["auxiliary_custody"]["receipt_schema"] == FIXTURE_AUXILIARY_AGGREGATE_SCHEMA
    assert identity["representation_status"] == BLOCKED_REPRESENTATION_STATUS
    assert identity["next_preclosure_gate"] == NEXT_PRECLOSURE_GATE
    published = adapter.publish_pre_encode_identity_receipt(tmp_path / "g58_pre_encode_identity.json")
    assert published["sha256"] == adapter.pre_encode_identity_sha256
    assert Path(published["path"]).read_bytes() == _canonical_json(identity)
    assert adapter.publish_pre_encode_identity_receipt(tmp_path / "g58_pre_encode_identity.json") == published

    prior_chain = "0" * 64
    stages = list(adapter.iter_stages(max_pairs=PAIRS_PER_STAGE))
    assert [stage.pair_range for stage in stages] == [(0, 2), (2, 4)]
    for stage in stages:
        direct_y0 = fixture.auxiliary.direct_y0[stage.pair_range[0] : stage.pair_range[1]]
        direct_y1 = fixture.auxiliary.direct_y1[stage.pair_range[0] : stage.pair_range[1]]
        assert not np.shares_memory(stage.y0_u8, direct_y0)
        assert not np.shares_memory(stage.y1_u8, direct_y1)
        assert stage.pre_encode_admission.auxiliary_planes_forwarded is False
        assert stage.pre_encode_admission.g49_decode_custody_verified is True
        assert stage.pre_encode_admission.behavior_changing_factor_section_ids
        prior_chain = validate_pre_encode_stage(
            stage,
            expected_prior_stage_chain_sha256=prior_chain,
        )
    assert adapter.last_complete_stage_chain_sha256 == prior_chain

    tampered_y1 = stages[0].y1_u8.copy()
    tampered_y1[0, 0, 0, 0] ^= np.uint8(1)
    with pytest.raises(
        SelectedPreimageOperandAdapterError,
        match="stage admission failed",
    ):
        validate_pre_encode_stage(
            replace(stages[0], y1_u8=tampered_y1),
            expected_prior_stage_chain_sha256="0" * 64,
        )
    for malformed in (
        replace(
            stages[0],
            y1_u8=stages[0].y1_u8.reshape(
                PAIRS_PER_STAGE,
                HEIGHT,
                WIDTH * 3,
            ),
        ),
        replace(
            stages[0],
            y1_u8=stages[0].y1_u8.view(np.int8),
        ),
        replace(
            stages[0],
            target_labels_u8=stages[0].target_labels_u8.reshape(
                PAIRS_PER_STAGE,
                HEIGHT * WIDTH,
            ),
        ),
        replace(
            stages[0],
            gt_poses_f32=stages[0].gt_poses_f32.view(np.uint32),
        ),
    ):
        with pytest.raises(
            SelectedPreimageOperandAdapterError,
            match="stage admission failed",
        ):
            validate_pre_encode_stage(
                malformed,
                expected_prior_stage_chain_sha256="0" * 64,
            )


def test_production_lattice_inert_factor_and_equal_value_auxiliary_custody(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(
        SelectedPreimageOperandAdapterError,
        match="production adapter requires n600",
    ):
        TaskspaceSelectedPreimageFreshOperandAdapterV1(
            program=fixture.program,
            decoder=fixture.decoder,
            auxiliary_provider=fixture.auxiliary,
            auxiliary_custody=fixture.custody,
            pairs_per_stage=PAIRS_PER_STAGE,
        )

    inert = _fixture(tmp_path / "inert", inert=True)
    with pytest.raises(
        SelectedPreimageOperandAdapterError,
        match="behavior probe failed",
    ):
        _adapter(inert)

    direct_y0 = np.stack([fixture.decoder.decode_semantic_base_pair(pair_id)[0] for pair_id in range(PAIR_COUNT)])
    equal_output = _fixture(tmp_path / "equal", direct_y0=direct_y0)
    stages = list(_adapter(equal_output).iter_stages(max_pairs=PAIRS_PER_STAGE))
    assert np.array_equal(stages[0].y0_u8, direct_y0[:PAIRS_PER_STAGE])
    assert not np.shares_memory(
        stages[0].y0_u8,
        equal_output.auxiliary.direct_y0[:PAIRS_PER_STAGE],
    )


def test_every_admission_field_is_in_the_recurrent_chain(tmp_path: Path) -> None:
    stage = next(_adapter(_fixture(tmp_path)).iter_stages(max_pairs=PAIRS_PER_STAGE))
    admission = stage.pre_encode_admission
    for field_info in fields(admission):
        value = getattr(admission, field_info.name)
        if type(value) is bool:
            mutated_value = not value
        elif type(value) is int:
            mutated_value = value + 1
        elif type(value) is tuple:
            mutated_value = (*value, "tampered") if not value or type(value[0]) is str else (value[0] + 1, *value[1:])
        elif field_info.name.endswith("sha256"):
            mutated_value = "e" * 64 if value != "e" * 64 else "d" * 64
        else:
            mutated_value = f"{value}.tampered"
        mutated = replace(
            stage,
            pre_encode_admission=replace(
                admission,
                **{field_info.name: mutated_value},
            ),
        )
        with pytest.raises(SelectedPreimageOperandAdapterError):
            validate_pre_encode_stage(
                mutated,
                expected_prior_stage_chain_sha256="0" * 64,
            )


def test_production_g51_aggregate_shape_and_batch_are_structurally_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        selected_adapter.FreshScorerPlaneOperandLoaderV1,
        "open",
        classmethod(lambda cls, path, expected_sha256=None: object()),
    )
    stages = []
    for stage_index in range(5):
        start = stage_index * 120
        stop = start + 120
        manifest = tmp_path / f"stage_{stage_index}.json"
        manifest.write_bytes(
            _canonical_json(
                {
                    "files": {
                        "gt_poses_f32": {"sha256": f"{stage_index + 1:x}" * 64},
                        "y0_u8": {"sha256": f"{stage_index + 6:x}" * 64},
                        "y1_u8": {"sha256": f"{stage_index + 11:x}" * 64},
                    },
                    "pair_range": [start, stop],
                    "pose_authority": "SEALED_SOURCE_CACHE_ADVISORY_ONLY",
                }
            )
        )
        stages.append({"path": str(manifest)})
    body = {
        "aggregate_receipt_sha256": "a" * 64,
        "fresh_teacher_receipt": {"scorer_pair_batch_size": 16},
        "pair_count": 600,
        "pose_authority": "SEALED_SOURCE_CACHE_ADVISORY_ONLY",
        "schema": selected_adapter.FRESH_SCORER_PLANE_AGGREGATE_SCHEMA,
        "stage_pairs": 120,
        "stages": stages,
        "target_labels": {
            "dtype": "uint8",
            "sha256": "b" * 64,
            "shape": [600, 384, 512],
        },
    }
    aggregate = tmp_path / "g51_aggregate.json"
    aggregate.write_bytes(_canonical_json(body))
    custody = AuxiliaryOperandCustodyV1(
        aggregate_receipt_path=str(aggregate.resolve()),
        aggregate_receipt_sha256=_sha256(aggregate.read_bytes()),
    )
    assert custody.pair_count == 600
    assert custody.stage_pairs == 120
    assert custody.stage_count == 5
    assert custody.scorer_batch_size == 16
    assert custody.target_labels_sha256 == "b" * 64
    assert len(custody.stage_bindings) == 5

    mutated = {**body, "fresh_teacher_receipt": {"scorer_pair_batch_size": 32}}
    aggregate.write_bytes(_canonical_json(mutated))
    with pytest.raises(
        SelectedPreimageOperandAdapterError,
        match="not n600/five-stage/batch16",
    ):
        AuxiliaryOperandCustodyV1(
            aggregate_receipt_path=str(aggregate.resolve()),
            aggregate_receipt_sha256=_sha256(aggregate.read_bytes()),
        )


def test_terminal_stage_chain_is_separate_sealed_lifecycle_artifact(
    tmp_path: Path,
) -> None:
    adapter = _adapter(_fixture(tmp_path))
    identity_path = tmp_path / "identity.json"
    adapter.publish_pre_encode_identity_receipt(identity_path)
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_bytes(b'{"schema":"fixture.campaign.v1"}')
    with pytest.raises(
        SelectedPreimageOperandAdapterError,
        match="unavailable before full iteration",
    ):
        adapter.publish_terminal_stage_chain_receipt(
            tmp_path / "early_terminal.json",
            campaign_receipt=ReopenableRegularFileIdentityV1.from_path(campaign_path),
            pre_encode_identity_receipt=(ReopenableRegularFileIdentityV1.from_path(identity_path)),
        )
    list(adapter.iter_stages(max_pairs=PAIRS_PER_STAGE))
    published = adapter.publish_terminal_stage_chain_receipt(
        tmp_path / "terminal.json",
        campaign_receipt=ReopenableRegularFileIdentityV1.from_path(campaign_path),
        pre_encode_identity_receipt=ReopenableRegularFileIdentityV1.from_path(identity_path),
    )
    receipt = json.loads(Path(published["path"]).read_bytes())
    seal = receipt.pop("receipt_sha256")
    assert seal == _sha256(_canonical_json(receipt))
    assert len(receipt["stages"]) == PAIR_COUNT // PAIRS_PER_STAGE
    assert receipt["terminal_stage_chain_sha256"] == adapter.last_complete_stage_chain_sha256


def test_reopenable_identity_and_publishers_refuse_symlinks(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"{}")
    symlink = tmp_path / "target-link.json"
    symlink.symlink_to(target)
    with pytest.raises(SelectedPreimageOperandAdapterError):
        ReopenableRegularFileIdentityV1.from_path(symlink)

    adapter = _adapter(_fixture(tmp_path / "fixture"))
    identity_target = tmp_path / "identity-target.json"
    identity_target.write_bytes(b"not-the-identity")
    identity_link = tmp_path / "identity-link.json"
    identity_link.symlink_to(identity_target)
    with pytest.raises(SelectedPreimageOperandAdapterError):
        adapter.publish_pre_encode_identity_receipt(identity_link)
    assert identity_target.read_bytes() == b"not-the-identity"


def test_canonical_receipt_identity_and_payload_use_one_stable_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(_canonical_json({"schema": "fixture.single-reopen.v1"}))
    original = selected_adapter._stable_regular_file_identity_and_bytes
    calls = 0

    def counted(path: object, *, label: str):
        nonlocal calls
        calls += 1
        return original(path, label=label)

    monkeypatch.setattr(
        selected_adapter,
        "_stable_regular_file_identity_and_bytes",
        counted,
    )
    record, value = selected_adapter._load_canonical_receipt(
        receipt,
        label="single-reopen fixture",
    )
    assert calls == 1
    assert record["sha256"] == _sha256(receipt.read_bytes())
    assert value["schema"] == "fixture.single-reopen.v1"


def test_learned_decoder_custody_reopens_exact_factor_contract_source(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path / "fixture")
    semantic_archive_path = tmp_path / "semantic.bin"
    semantic_archive_path.write_bytes(fixture.semantic_bytes)
    compile_receipt_path = tmp_path / "compile_receipt.json"
    compile_receipt_path.write_bytes(b'{"schema":"fixture.compile.v1"}')
    compiler_source_path = Path(__file__).resolve()
    generic_source_path = Path(
        selected_adapter.inspect.getsourcefile(realize_factor2_uint8_scorer_plane) or ""
    ).resolve()
    target_body = {
        "batch_geometry_matches_upstream_default": True,
        "candidate_payload_allowed": False,
        "encoder_only": True,
        "pair_count": 600,
        "schema": "tac.taskspace_fresh_teacher_materialization.v1",
        "scorer_pair_batch_size": 16,
        "target_labels": {
            "dtype": "uint8",
            "sha256": _sha256(fixture.labels.tobytes()),
            "shape": [600, 384, 512],
        },
    }
    target_seal = _sha256(_canonical_json(target_body))
    target_receipt_path = tmp_path / "target_receipt.json"
    target_receipt_path.write_bytes(
        _canonical_json(
            {
                **target_body,
                "receipt_sha256": target_seal,
            }
        )
    )
    semantic_identity = replace(
        fixture.program.semantic_program_identity,
        fresh_compile_receipt_sha256=_sha256(compile_receipt_path.read_bytes()),
        compiler_source_sha256=_sha256(compiler_source_path.read_bytes()),
        compiled_semantic_archive_sha256=_sha256(fixture.semantic_bytes),
        compiled_semantic_archive_bytes=len(fixture.semantic_bytes),
    )
    target_identity = replace(
        fixture.program.target_custody_identity,
        target_custody_receipt_sha256=target_seal,
    )
    program = replace(
        fixture.program,
        semantic_program_identity=semantic_identity,
        target_custody_identity=target_identity,
    )
    decoder = _Decoder(
        semantic_identity=semantic_identity,
        target_identity=target_identity,
    )
    production = SelectedPreimageProductionCustodyV1(
        semantic_archive=ReopenableRegularFileIdentityV1.from_path(semantic_archive_path),
        semantic_compile_receipt=(ReopenableRegularFileIdentityV1.from_path(compile_receipt_path)),
        target_custody_receipt=(ReopenableRegularFileIdentityV1.from_path(target_receipt_path)),
        target_custody_receipt_seal_sha256=target_seal,
        compiler_source=ReopenableRegularFileIdentityV1.from_path(compiler_source_path),
        generic_v10_source=ReopenableRegularFileIdentityV1.from_path(generic_source_path),
        decoder_callable_source_sha256=(program.decoder_identity.implementation_source_sha256),
        learned_decoder_sources=(
            LearnedDecoderSourceCustodyV1(
                section_id="learned.compact.factor",
                decoder_contract_id=("tac.test.generic_selected_preimage_factor_decoder.v1"),
                implementation_source=(ReopenableRegularFileIdentityV1.from_path(compiler_source_path)),
            ),
        ),
    )
    production.verify_against(program, decoder)

    wrong_source = tmp_path / "wrong_learned_decoder.py"
    wrong_source.write_bytes(b"def wrong():\\n    return None\\n")
    wrong = replace(
        production,
        learned_decoder_sources=(
            replace(
                production.learned_decoder_sources[0],
                implementation_source=(ReopenableRegularFileIdentityV1.from_path(wrong_source)),
            ),
        ),
    )
    with pytest.raises(
        SelectedPreimageOperandAdapterError,
        match="learned decoder source custody differs",
    ):
        wrong.verify_against(program, decoder)


def _write_outer_archive(
    path: Path,
    *,
    semantic: bytes,
    packet: bytes,
    extra: tuple[tuple[str, bytes], ...] = (),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("semantic.bin", semantic)
        archive.writestr("program.bin", packet)
        for name, payload in extra:
            archive.writestr(name, payload)


def test_program_residual_name_requires_physical_counted_members_and_decoder(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    archive_path = tmp_path / "counted_outer.zip"
    _write_outer_archive(
        archive_path,
        semantic=fixture.semantic_bytes,
        packet=fixture.program.packet_bytes,
    )
    proof = reopen_program_residual_outer_archive(
        archive_path=archive_path,
        semantic_archive_bytes=fixture.semantic_bytes,
        semantic_member_name="semantic.bin",
        program_member_name="program.bin",
        program=fixture.program,
        decoder=fixture.decoder,
        forbidden_payload_sha256s=(
            _sha256(fixture.labels.tobytes()),
            _sha256(fixture.poses.tobytes()),
        ),
    )
    assert proof.representation_mode == PROGRAM_RESIDUAL_MODE
    assert proof.archive_bytes == archive_path.stat().st_size
    assert proof.archive_sha256 == _sha256(archive_path.read_bytes())
    assert proof.outer_member_names == ("semantic.bin", "program.bin")
    assert proof.outer_members_partition_exactly is True
    assert proof.program_packet_sha256 == fixture.program.packet_sha256
    assert proof.generic_learned_decoder_source_bound is True
    assert proof.learned_decoder_contracts == (
        (
            "learned.compact.factor",
            "tac.test.generic_selected_preimage_factor_decoder.v1",
            LEARNED_SOURCE_SHA256,
        ),
    )
    assert proof.factor_payload_bytes_inside_packet > 0
    assert proof.precontainer_counted_source_bytes == (len(fixture.semantic_bytes) + len(fixture.program.packet_bytes))
    assert proof.next_preclosure_gate == NEXT_PRECLOSURE_GATE
    outer_receipt = publish_program_residual_outer_archive_proof(
        proof,
        tmp_path / "outer_proof.json",
    )
    outer_payload = json.loads(Path(outer_receipt["path"]).read_bytes())
    outer_seal = outer_payload.pop("receipt_sha256")
    assert outer_seal == _sha256(_canonical_json(outer_payload))
    evidence = _adapter(fixture).program_residual_pre_encode_gate_evidence(proof)
    assert evidence == {
        "actual_representation": "PROGRAM_RESIDUAL_LAYERED",
        "pair_count": PAIR_COUNT,
        "scorer_batch_size": 16,
        "provider_kind": "G49_SELECTED_PREIMAGE_PROGRAM",
        "source_plane_definition": "G49_DECODE_SELECTED_PREIMAGE_PAIR",
        "semantic_archive_bytes": len(fixture.semantic_bytes),
        "semantic_archive_sha256": _sha256(fixture.semantic_bytes),
        "semantic_archive_counted": True,
        "semantic_archive_reopened": True,
        "program_packet_bytes": len(fixture.program.packet_bytes),
        "program_packet_sha256": fixture.program.packet_sha256,
        "factor_count": 1,
        "behavior_changing_factor_count": 1,
        "target_payload_embedded": False,
        "historical_payload_reused": False,
    }
    mutated = tmp_path / "mutated_outer.zip"
    _write_outer_archive(
        mutated,
        semantic=fixture.semantic_bytes,
        packet=fixture.program.packet_bytes + b"x",
    )
    with pytest.raises(
        SelectedPreimageOperandAdapterError,
        match="program member differs",
    ):
        reopen_program_residual_outer_archive(
            archive_path=mutated,
            semantic_archive_bytes=fixture.semantic_bytes,
            semantic_member_name="semantic.bin",
            program_member_name="program.bin",
            program=fixture.program,
            decoder=fixture.decoder,
        )

    forbidden = tmp_path / "forbidden_outer.zip"
    _write_outer_archive(
        forbidden,
        semantic=fixture.semantic_bytes,
        packet=fixture.program.packet_bytes,
        extra=(("target_labels.bin", fixture.labels.tobytes()),),
    )
    with pytest.raises(
        SelectedPreimageOperandAdapterError,
        match="forbidden target/pose/scorer",
    ):
        reopen_program_residual_outer_archive(
            archive_path=forbidden,
            semantic_archive_bytes=fixture.semantic_bytes,
            semantic_member_name="semantic.bin",
            program_member_name="program.bin",
            program=fixture.program,
            decoder=fixture.decoder,
        )

    neutral_smuggle = tmp_path / "neutral_smuggle_outer.zip"
    _write_outer_archive(
        neutral_smuggle,
        semantic=fixture.semantic_bytes,
        packet=fixture.program.packet_bytes,
        extra=(("aux.bin", fixture.labels.tobytes()),),
    )
    with pytest.raises(
        SelectedPreimageOperandAdapterError,
        match="unexpected untyped member",
    ):
        reopen_program_residual_outer_archive(
            archive_path=neutral_smuggle,
            semantic_archive_bytes=fixture.semantic_bytes,
            semantic_member_name="semantic.bin",
            program_member_name="program.bin",
            program=fixture.program,
            decoder=fixture.decoder,
            forbidden_payload_sha256s=(),
        )


def test_analytic_only_outer_program_is_legal_and_vacuously_source_bound(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, analytic_only=True)
    archive_path = tmp_path / "analytic_only_outer.zip"
    _write_outer_archive(
        archive_path,
        semantic=fixture.semantic_bytes,
        packet=fixture.program.packet_bytes,
    )
    proof = reopen_program_residual_outer_archive(
        archive_path=archive_path,
        semantic_archive_bytes=fixture.semantic_bytes,
        semantic_member_name="semantic.bin",
        program_member_name="program.bin",
        program=fixture.program,
        decoder=fixture.decoder,
    )
    assert proof.learned_decoder_contracts == ()
    assert proof.generic_learned_decoder_source_bound is True
    assert proof.factor_payload_bytes_inside_packet > 0
    adapter = _adapter(fixture)
    stages = list(adapter.iter_stages(max_pairs=PAIRS_PER_STAGE))
    prior = "0" * 64
    for stage in stages:
        prior = validate_pre_encode_stage(
            stage,
            expected_prior_stage_chain_sha256=prior,
        )
