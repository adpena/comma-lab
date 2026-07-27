# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import inspect
import json
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.direct_description_carrier_compose import BoundaryShearletAtomV1
from tac.optimization.uint8_lattice_feasibility import (
    realize_factor2_uint8_scorer_plane,
)
from tac.witness_control.taskspace_codec_adversarial_gate_v2 import (
    _validate_program_producer_config,
)
from tac.witness_dsl.taskspace_program_residual_producer_v1 import (
    CONFIG_SCHEMA,
    EXAMPLE_CONFIG_SCHEMA,
    MISSING_PRIMARY_LAYERS,
    PRIMARY_CODEC_BLOCKERS,
    PRIMARY_CODEC_STATUS,
    PRODUCER_ROLE,
    ProgramResidualProducerError,
    build_counted_outer_archive_bytes,
    canonical_json,
    factor_operand_inventory,
    load_config,
    publish_write_once,
    run_stage_lattice,
    stable_file_identity,
    stable_read_regular_file,
    validate_program_residual_producer_config_for_g59,
)
from tac.witness_dsl.taskspace_selected_preimage_operand_adapter_v1 import (
    FIXTURE_AUXILIARY_AGGREGATE_SCHEMA,
    PRODUCTION_PRE_ENCODE_EVIDENCE_SCHEMA,
    AuxiliaryOperandCustodyV1,
    TaskspaceSelectedPreimageFreshOperandAdapterV1,
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

HEIGHT = 384
WIDTH = 512


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": _sha(payload),
    }


def _primary_codec() -> dict[str, object]:
    return {
        "status": PRIMARY_CODEC_STATUS,
        "g49_role": PRODUCER_ROLE,
        "closed_g49_factor_vocabulary": [
            "SHEARLET_BOUNDARY_TRANSPORT_Q4",
            "COMPACT_LATENT_QUOTIENT_PLUGIN",
        ],
        "missing_required_layers": list(MISSING_PRIMARY_LAYERS),
        "canonical_g17_ontology_bound": True,
        "candidate_ready": False,
    }


def _truth() -> dict[str, bool]:
    return {
        "research_only": True,
        "score_claim": False,
        "candidate_claim": False,
        "evaluation_claim": False,
        "promotion_eligible": False,
        "historical_payload_reused": False,
        "direct_source_plane_fallback_allowed": False,
        "raw_labels_embedded": False,
        "source_planes_embedded": False,
        "generic_algorithm_code_free": True,
        "all_video_derived_operands_counted": True,
    }


@dataclass(frozen=True, slots=True)
class _ConfigFixture:
    config_path: Path
    raw: dict[str, object]
    semantic: dict[str, object]
    semantic_receipt: dict[str, object]
    target_receipt: dict[str, object]
    auxiliary: dict[str, object]
    compiler: dict[str, object]
    generic: dict[str, object]
    campaign: dict[str, object]
    packet: dict[str, object]
    program: TaskspaceSelectedPreimageProgramV1


def _make_program(
    root: Path,
    *,
    pair_count: int,
    target_bank_sha256: str = "8" * 64,
    include_learned: bool = False,
) -> tuple[
    TaskspaceSelectedPreimageProgramV1,
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    semantic = _write(root / "semantic.zip", b"fresh-own-lineage-semantic")
    semantic_receipt = _write(root / "semantic_receipt.json", b'{"fresh":true}\n')
    compiler = _write(root / "compiler.py", b"def compile_fresh_program():\n    return None\n")
    generic_path = Path(inspect.getsourcefile(realize_factor2_uint8_scorer_plane) or "")
    generic_identity = stable_file_identity(generic_path, label="generic V10 source")
    generic = generic_identity.to_mapping()
    semantic_identity = V15SemanticProgramIdentityV1(
        fresh_compile_receipt_schema=V15_SEMANTIC_COMPILE_RECEIPT_SCHEMA,
        fresh_compile_receipt_sha256=str(semantic_receipt["sha256"]),
        compile_proof_dependency_sha256="2" * 64,
        typed_compile_config_sha256="3" * 64,
        compiler_source_sha256=str(compiler["sha256"]),
        receiver_source_sha256="5" * 64,
        compiled_semantic_archive_sha256=str(semantic["sha256"]),
        compiled_semantic_archive_bytes=int(semantic["bytes"]),
        source_pair_start=0,
        pair_count=pair_count,
        declared_compile_dependency_sha256s=("6" * 64, "7" * 64),
        compile_derivation=V15_SEMANTIC_COMPILE_DERIVATION,
    )
    target = ScorerTargetCustodyIdentityV1(
        target_custody_receipt_sha256="9" * 64,
        target_bank_sha256=target_bank_sha256,
    )
    analytic = build_analytic_shearlet_residual_factor(
        section_id="analytic.boundary.transport",
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
    factors: list[TaskspaceSelectedPreimageFactorV1] = [analytic]
    learned_source: dict[str, object] | None = None
    if include_learned:
        learned_source = _write(
            root / "learned_decoder.py",
            b"def decode_compact_quotient(*args):\n    return args[-2:]\n",
        )
        factors.append(
            build_learned_irreducible_quotient_factor(
                section_id="learned.irreducible.quotient",
                source_pair_start=1,
                source_pair_stop_exclusive=pair_count,
                active_pair_ranges=((1, min(pair_count, 2)),),
                decoder_contract_id="tac.test.compact_quotient.v1",
                decoder_implementation_source_sha256=str(learned_source["sha256"]),
                model_family_id="compact_test_quotient",
                latent_codec_id="delta_i8",
                parameter_codec_id="weights_i8",
                latent_dtype="int8",
                parameter_dtype="int8",
                latent_payload=b"\x01\x02",
                parameter_payload=b"\x03",
                source_receipt_sha256="b" * 64,
            )
        )
    program = TaskspaceSelectedPreimageProgramV1(
        semantic_program_identity=semantic_identity,
        target_custody_identity=target,
        decoder_identity=GenericV10Factor2DecoderIdentityV1.current(),
        compile_config=SelectedPreimageCompileConfigV1(
            source_pair_start=0,
            pair_count=pair_count,
            maximum_packet_bytes=1 << 20,
            score_budget_receipt_sha256="c" * 64,
            budget_rule_id="test_dynamic_frontier_control",
        ),
        factors=tuple(factors),
    )
    return (
        program,
        semantic,
        semantic_receipt,
        compiler,
        {
            "generic": generic,
            "learned_source": learned_source,
        },
    )


def _config_fixture(
    tmp_path: Path,
    *,
    pair_count: int = 600,
    pairs_per_stage: int = 120,
    test_only: bool = False,
    include_learned: bool = False,
    target_bank_sha256: str = "8" * 64,
) -> _ConfigFixture:
    source_root = tmp_path / "sources"
    (
        program,
        semantic,
        semantic_receipt,
        compiler,
        source_rows,
    ) = _make_program(
        source_root,
        pair_count=pair_count,
        target_bank_sha256=target_bank_sha256,
        include_learned=include_learned,
    )
    packet = _write(source_root / "program.tspp", program.packet_bytes)
    target_receipt = _write(source_root / "target_receipt.json", b'{"encoder_only":true}\n')
    auxiliary = _write(source_root / "auxiliary_aggregate.json", b'{"fresh":true}\n')
    campaign = _write(source_root / "campaign_seal.json", b'{"stage":"CAMPAIGN_SEAL"}\n')
    stage_count = pair_count // pairs_per_stage
    output_root = (
        tmp_path / "output" if test_only else Path("/Volumes/VertigoDataTier/pact/g63_schema_test_nonexecuted")
    )
    outputs = {
        "output_root": str(output_root),
        "stage_checkpoint_dir": str(output_root / "stages"),
        "g58_identity_receipt": str(output_root / "g58_identity.json"),
        "g58_terminal_stage_chain_receipt": str(output_root / "g58_terminal.json"),
        "outer_archive": str(output_root / "archive.zip"),
        "g58_outer_proof_receipt": str(output_root / "g58_outer_proof.json"),
        "g59_pre_encode_receipt": str(output_root / "g59_pre_encode.json"),
        "run_receipt": str(output_root / "run_receipt.json"),
    }
    learned_rows: list[dict[str, object]] = []
    if include_learned:
        assert source_rows["learned_source"] is not None
        learned_rows.append(
            {
                "section_id": "learned.irreducible.quotient",
                "decoder_contract_id": "tac.test.compact_quotient.v1",
                "implementation_source": source_rows["learned_source"],
            }
        )
    raw: dict[str, object] = {
        "schema": CONFIG_SCHEMA,
        "run_id": "g63_test_run",
        "campaign_id": "g63_test_campaign",
        "requested_representation": "PROGRAM_RESIDUAL_LAYERED",
        "producer_role": PRODUCER_ROLE,
        "execution_ready": True,
        "pair_count": pair_count,
        "pairs_per_stage": pairs_per_stage,
        "stage_count": stage_count,
        "scorer_batch_size": 16,
        "program_packet": packet,
        "semantic_archive": semantic,
        "semantic_compile_receipt": semantic_receipt,
        "target_custody_receipt": target_receipt,
        "auxiliary_aggregate_receipt": auxiliary,
        "compiler_source": compiler,
        "generic_v10_source": source_rows["generic"],
        "learned_decoder_sources": learned_rows,
        "campaign_seal_receipt": campaign,
        "factor_operands": list(factor_operand_inventory(program)),
        "primary_program_codec": _primary_codec(),
        "outer_archive_members": {
            "semantic_member_name": "semantic/program.zip",
            "program_member_name": "residual/program.tspp",
        },
        "output_paths": outputs,
        "required_free_bytes": 1,
        "resume_required": True,
        "test_only_small_fixture": test_only,
        "truth": _truth(),
    }
    config_path = tmp_path / "config.json"
    config_path.write_bytes(canonical_json(raw))
    return _ConfigFixture(
        config_path=config_path,
        raw=raw,
        semantic=semantic,
        semantic_receipt=semantic_receipt,
        target_receipt=target_receipt,
        auxiliary=auxiliary,
        compiler=compiler,
        generic=source_rows["generic"],
        campaign=campaign,
        packet=packet,
        program=program,
    )


def _strict_g58_evidence(fixture: _ConfigFixture) -> dict[str, object]:
    outputs = fixture.raw["output_paths"]
    assert isinstance(outputs, dict)

    def row(path_key: str) -> dict[str, object]:
        return {"path": outputs[path_key], "bytes": 1, "sha256": "d" * 64}

    learned = sorted(
        factor.section_id for factor in fixture.program.factors if factor.role.value == "LEARNED_IRREDUCIBLE_QUOTIENT"
    )
    return {
        "strict_production_evidence": {
            "schema": PRODUCTION_PRE_ENCODE_EVIDENCE_SCHEMA,
            "status": "ADMIT",
            "representation_mode": "PROGRAM_RESIDUAL_LAYERED",
            "identity_receipt": row("g58_identity_receipt"),
            "terminal_stage_chain_receipt": row("g58_terminal_stage_chain_receipt"),
            "outer_proof_receipt": row("g58_outer_proof_receipt"),
            "campaign_receipt": fixture.campaign,
            "auxiliary_aggregate_receipt": {
                **fixture.auxiliary,
                "self_seal_sha256": "e" * 64,
            },
            "semantic_archive": fixture.semantic,
            "semantic_compile_receipt": fixture.semantic_receipt,
            "target_custody_receipt": fixture.target_receipt,
            "generic_v10_source": fixture.generic,
            "outer_archive": row("outer_archive"),
            "pair_count": 600,
            "pairs_per_stage": 120,
            "stage_count": 5,
            "scorer_batch_size": 16,
            "program_packet_sha256": fixture.program.packet_sha256,
            "factor_section_ids": [factor.section_id for factor in fixture.program.factors],
            "behavior_changing_factor_section_ids": [factor.section_id for factor in fixture.program.factors],
            "learned_factor_section_ids": learned,
            "analytic_only": not learned,
            "target_payload_embedded": False,
            "historical_payload_reused": False,
        }
    }


def test_example_is_typed_but_deliberately_non_executable() -> None:
    path = (
        Path(__file__).resolve().parents[4]
        / ".omx/research/configs/taskspace_program_residual_n600_example_20260726.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["schema"] == EXAMPLE_CONFIG_SCHEMA
    assert value["execution_ready"] is False
    assert value["program_packet"] is None
    with pytest.raises(ProgramResidualProducerError, match="deliberately non-executable"):
        load_config(path)


def test_factor_inventory_separates_free_algorithm_from_both_counted_operands(
    tmp_path: Path,
) -> None:
    fixture = _config_fixture(tmp_path, include_learned=True)
    config = load_config(fixture.config_path)
    rows = config.factor_operands
    assert [row["role"] for row in rows] == [
        "ANALYTIC_RESIDUAL",
        "LEARNED_IRREDUCIBLE_QUOTIENT",
    ]
    assert [row["operand_byte_home"] for row in rows] == [
        "COUNTED_ANALYTIC_OPERAND",
        "COUNTED_LEARNED_OPERAND",
    ]
    assert all(row["generic_algorithm_byte_home"] == "GENERIC_DECODER_CODE_FREE" for row in rows)
    assert all(row["operand_payload_counted"] is True for row in rows)
    assert all(str(row["operand_lineage_class"]).startswith("VIDEO_DERIVED_") for row in rows)


def test_descriptor_stable_reads_and_atomic_write_once_refuse_alias_or_drift(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"exact")
    identity, payload = stable_read_regular_file(source, label="source")
    assert payload == b"exact"
    assert identity.sha256 == _sha(b"exact")
    alias = tmp_path / "alias.bin"
    alias.symlink_to(source)
    with pytest.raises(ProgramResidualProducerError, match="no-follow"):
        stable_read_regular_file(alias, label="alias")

    output = tmp_path / "receipt.json"
    first = publish_write_once(output, b"one\n", label="receipt")
    assert publish_write_once(output, b"one\n", label="receipt") == first
    with pytest.raises(ProgramResidualProducerError, match="drifted"):
        publish_write_once(output, b"two\n", label="receipt")


def test_config_reopens_source_lineage_and_rejects_hidden_fields_or_mutation(
    tmp_path: Path,
) -> None:
    fixture = _config_fixture(tmp_path)
    config = load_config(fixture.config_path)
    assert config.program.packet_sha256 == fixture.program.packet_sha256
    assert config.semantic_archive.sha256 == fixture.semantic["sha256"]

    malformed = dict(fixture.raw)
    malformed["source_planes"] = {"y0": "hidden"}
    fixture.config_path.write_bytes(canonical_json(malformed))
    with pytest.raises(ProgramResidualProducerError, match=r"extra=.*source_planes"):
        load_config(fixture.config_path)

    fixture.config_path.write_bytes(canonical_json(fixture.raw))
    Path(str(fixture.semantic["path"])).write_bytes(b"mutated")
    with pytest.raises(ProgramResidualProducerError, match="identity differs"):
        load_config(fixture.config_path)


def test_outer_archive_is_exact_two_member_partition_and_names_fail_closed(
    tmp_path: Path,
) -> None:
    fixture = _config_fixture(tmp_path, test_only=True)
    config = load_config(fixture.config_path)
    payload = build_counted_outer_archive_bytes(config)
    archive_path = tmp_path / "outer.zip"
    archive_path.write_bytes(payload)
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.namelist() == [
            "semantic/program.zip",
            "residual/program.tspp",
        ]
        assert archive.read("semantic/program.zip") == config.semantic_archive_bytes
        assert archive.read("residual/program.tspp") == config.program_packet_bytes
    assert payload == build_counted_outer_archive_bytes(config)

    malformed = dict(fixture.raw)
    malformed["outer_archive_members"] = {
        "semantic_member_name": "semantic/program.zip",
        "program_member_name": "hidden/labels.bin",
    }
    fixture.config_path.write_bytes(canonical_json(malformed))
    with pytest.raises(ProgramResidualProducerError, match="forbidden payload"):
        load_config(fixture.config_path)


def test_g59_accepts_exact_schema_custody_then_refuses_primary_codec(
    tmp_path: Path,
) -> None:
    fixture = _config_fixture(tmp_path)
    identity = stable_file_identity(fixture.config_path, label="producer config")
    evidence = _strict_g58_evidence(fixture)
    assert (
        validate_program_residual_producer_config_for_g59(
            identity.to_mapping(),
            evidence,
        )
        == PRIMARY_CODEC_BLOCKERS
    )
    assert _validate_program_producer_config(identity.to_mapping(), evidence) == list(PRIMARY_CODEC_BLOCKERS)

    malformed = json.loads(fixture.config_path.read_text(encoding="utf-8"))
    malformed["factor_operands"][0]["operand_payload_counted"] = False
    fixture.config_path.write_bytes(canonical_json(malformed))
    mutated_identity = stable_file_identity(fixture.config_path, label="mutated config")
    refusals = _validate_program_producer_config(
        mutated_identity.to_mapping(),
        evidence,
    )
    assert len(refusals) == 1
    assert refusals[0].startswith("PROGRAM_PRODUCER_CONFIG_REFUSED:")
    assert "factor operand inventory differs" in refusals[0]

    drift_refusals = _validate_program_producer_config(
        identity.to_mapping(),
        evidence,
    )
    assert "identity drifted" in drift_refusals[0]


def test_primary_blockers_name_g17_vertical_debts_not_a_duplicate_schema() -> None:
    assert PRIMARY_CODEC_STATUS == "OWED_G17_VERTICAL_LINKER"
    assert PRIMARY_CODEC_BLOCKERS == (
        "G17_PRIMARY_ARCHIVE_PRODUCER_OWED",
        "G17_PUBLIC_RECEIVER_OPERATION_REGISTRY_OWED",
    )
    assert all("VOCABULARY_V2" not in blocker for blocker in PRIMARY_CODEC_BLOCKERS)


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
    def __init__(self, labels: np.ndarray, poses: np.ndarray) -> None:
        self.labels = labels
        self.poses = poses
        self.y0 = np.full((4, HEIGHT, WIDTH, 3), 203, dtype=np.uint8)
        self.y1 = np.full((4, HEIGHT, WIDTH, 3), 197, dtype=np.uint8)

    def iter_stages(self, *, max_pairs: int = 120):
        assert max_pairs >= 2
        for start in (0, 2):
            stop = start + 2
            yield _AuxStage(
                pair_range=(start, stop),
                pair_ids=np.arange(start, stop, dtype=np.int64),
                y0_u8=self.y0[start:stop],
                y1_u8=self.y1[start:stop],
                target_labels_u8=self.labels[start:stop],
                gt_poses_f32=self.poses[start:stop],
            )


class _CountingDecoder:
    decoder_id = GENERIC_V10_FACTOR2_DECODER_ID

    def __init__(
        self,
        semantic: V15SemanticProgramIdentityV1,
        target: ScorerTargetCustodyIdentityV1,
    ) -> None:
        self.semantic = semantic
        self.target = target
        self.calls: Counter[int] = Counter()

    @property
    def implementation_source_sha256(self) -> str:
        return GenericV10Factor2DecoderIdentityV1.current().implementation_source_sha256

    def verify_semantic_program_identity(self, identity: object) -> bool:
        return identity == self.semantic

    def verify_target_custody_identity(self, identity: object) -> bool:
        return identity == self.target

    def decode_semantic_base_pair(self, source_pair_id: int):
        self.calls[source_pair_id] += 1
        y0 = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        y1 = np.full((HEIGHT, WIDTH, 3), source_pair_id + 7, dtype=np.uint8)
        if source_pair_id == 0:
            y0[160:220, 120:392] = (10, 20, 30)
        return y0, y1

    def learned_quotient_decoder_contract_id(
        self,
        factor: TaskspaceSelectedPreimageFactorV1,
    ) -> str:
        del factor
        raise AssertionError("analytic-only fixture")

    def learned_quotient_decoder_implementation_source_sha256(
        self,
        factor: TaskspaceSelectedPreimageFactorV1,
    ) -> str:
        del factor
        raise AssertionError("analytic-only fixture")

    def apply_learned_irreducible_quotient(self, **kwargs):
        del kwargs
        raise AssertionError("analytic-only fixture")

    def realize_factor2_pair(
        self,
        scorer_y0: np.ndarray,
        scorer_y1: np.ndarray,
    ) -> SelectedPreimageFactor2PairV1:
        del scorer_y0, scorer_y1
        raise AssertionError("factor-2 is outside stage custody")


def _small_adapter_fixture(
    tmp_path: Path,
) -> tuple[object, TaskspaceSelectedPreimageFreshOperandAdapterV1, _CountingDecoder]:
    labels = np.zeros((4, HEIGHT, WIDTH), dtype=np.uint8)
    poses = np.arange(24, dtype=np.float32).reshape(4, 6)
    labels_sha = _sha(labels.tobytes())
    fixture = _config_fixture(
        tmp_path,
        pair_count=4,
        pairs_per_stage=2,
        test_only=True,
        target_bank_sha256=labels_sha,
    )
    config = load_config(fixture.config_path)
    auxiliary = _AuxProvider(labels, poses)
    aggregate_body = {
        "schema": FIXTURE_AUXILIARY_AGGREGATE_SCHEMA,
        "pair_count": 4,
        "stage_pairs": 2,
        "stage_count": 2,
        "scorer_batch_size": 16,
        "target_labels_sha256": labels_sha,
        "gt_poses_f32_sha256": _sha(poses.tobytes()),
        "pose_authority": "SEALED_SOURCE_CACHE_ADVISORY_ONLY",
        "historical_payload_reused": False,
        "stages": [
            {
                "pair_range": [start, start + 2],
                "direct_source_y0_sha256": _sha(auxiliary.y0[start : start + 2].tobytes()),
                "direct_source_y1_sha256": _sha(auxiliary.y1[start : start + 2].tobytes()),
                "target_labels_sha256": _sha(labels[start : start + 2].tobytes()),
                "gt_poses_f32_sha256": _sha(poses[start : start + 2].tobytes()),
            }
            for start in (0, 2)
        ],
    }
    aggregate = {
        **aggregate_body,
        "aggregate_receipt_sha256": _sha(
            json.dumps(
                aggregate_body,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ),
    }
    aggregate_path = Path(str(config.auxiliary_aggregate_receipt.path))
    aggregate_path.write_bytes(canonical_json(aggregate))
    # Rebind the typed config to the exact fixture aggregate identity.
    fixture.raw["auxiliary_aggregate_receipt"] = _write(
        aggregate_path,
        canonical_json(aggregate),
    )
    fixture.config_path.write_bytes(canonical_json(fixture.raw))
    config = load_config(fixture.config_path)
    custody = AuxiliaryOperandCustodyV1(
        aggregate_receipt_path=str(aggregate_path.resolve()),
        aggregate_receipt_sha256=_sha(aggregate_path.read_bytes()),
    )
    decoder = _CountingDecoder(
        config.program.semantic_program_identity,
        config.program.target_custody_identity,
    )
    adapter = TaskspaceSelectedPreimageFreshOperandAdapterV1(
        program=config.program,
        decoder=decoder,
        auxiliary_provider=auxiliary,
        auxiliary_custody=custody,
        pairs_per_stage=2,
        test_only_small_fixture=True,
    )
    return config, adapter, decoder


def test_stage_resume_reopens_checkpoint_without_redecoding_prior_segment(
    tmp_path: Path,
) -> None:
    config, adapter, decoder = _small_adapter_fixture(tmp_path)
    first = run_stage_lattice(config, adapter, stop_after_stage=0)
    assert len(first) == 1
    calls_after_first = decoder.calls.copy()
    stage0 = config.output_paths["stage_checkpoint_dir"] / "stage_00.json"
    stage0_bytes = stage0.read_bytes()
    receipt = json.loads(stage0_bytes)
    assert receipt["decoded_planes_persisted"] is False
    assert receipt["direct_source_planes_persisted"] is False
    assert receipt["raw_labels_persisted"] is False

    completed = run_stage_lattice(config, adapter)
    assert len(completed) == 2
    assert stage0.read_bytes() == stage0_bytes
    assert decoder.calls[0] == calls_after_first[0]
    assert decoder.calls[1] == calls_after_first[1]
    assert decoder.calls[2] > 0 and decoder.calls[3] > 0
