# SPDX-License-Identifier: MIT
from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import struct
import subprocess
import sys
import zipfile
from dataclasses import fields, replace
from pathlib import Path

import pytest

import tac.witness_dsl.taskspace_public_auth_eval_closure as closure
from tac.witness_dsl.ep725_lossless_xcodec_recode import parse_ep725_lvls1
from tac.witness_dsl.ep725_population_global_recode_v2 import (
    MEMBER_NAME,
    parse_population_global_member,
)
from tac.witness_dsl.taskspace_g17_production_envelope import (
    build_g17_a_packet,
    build_g17_g_packet,
    build_g17_production_archive,
    build_g17_terminal_envelope,
)
from tac.witness_dsl.taskspace_lvpg2_public_inverse import lvpg2_to_lvls1
from tac.witness_dsl.taskspace_public_auth_eval_closure import (
    AuthClosureCheckpointArtifactV1,
    AuthClosureStageCheckpointV1,
    AuthClosureStageV1,
    CompiledPublicRuntimeV1,
    ContentOriginV1,
    GenericVMFacilityV1,
    InterpreterDistributionABIClosureV1,
    OfficialEvaluationRunReceiptV1,
    PayloadPlacementItemV1,
    PlacementLocationV1,
    PublicAuthClosureError,
    PublicDecodeEqualityReceiptV1,
    PublicEvaluatorExecutionReceiptV1,
    PublicRuntimeCompileReceiptV1,
    PublicRuntimeFileDigestV1,
    VideoDerivedPayloadClassV1,
    assess_auth_eval_execution_readiness,
    audit_generic_runtime_source,
    compile_lvpg2_public_runtime,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
G25_ARCHIVE = (
    REPO_ROOT
    / ".omx/research/original_taskspace_inverse_witness_codec_20260725"
    / "ep725_population_global_recode_v2_20260726_r2"
    / "ep725_population_global_recode_v2.not_a_candidate.zip"
)
LVLS1_RUNTIME = Path("/Volumes/VertigoDataTier/pact/yhat_rd_ladder_20260719/prepare/full_n600_packet/inflate.py")
SHA = "0" * 64


def _minimal_abi() -> InterpreterDistributionABIClosureV1:
    return InterpreterDistributionABIClosureV1(
        interpreter_implementation="cpython",
        interpreter_version="3.13.0",
        interpreter_cache_tag="cpython-313",
        interpreter_executable_name="python",
        interpreter_executable_realpath="/usr/bin/python",
        interpreter_executable_sha256=SHA,
        interpreter_prefix_realpaths=("/usr",),
        interpreter_prefix_tree_sha256=None,
        interpreter_prefix_tree_nbytes=0,
        soabi="cpython-313-x86_64-linux-gnu",
        multiarch="x86_64-linux-gnu",
        platform_system="Linux",
        platform_machine="x86_64",
        distributions=(),
        unresolved_import_roots=(),
    )


def _minimal_compiled() -> CompiledPublicRuntimeV1:
    runtime_file = PublicRuntimeFileDigestV1(
        relative_path="inflate.py",
        content_sha256=SHA,
        nbytes=1,
        executable=True,
    )
    receipt = PublicRuntimeCompileReceiptV1(
        archive_sha256=SHA,
        archive_nbytes=1,
        member_sha256=SHA,
        member_nbytes=1,
        decoded_state_sha256=SHA,
        materialized_lvls1_sha256=SHA,
        runtime_tree_sha256=SHA,
        runtime_files=(runtime_file,),
        source_audit_receipt_sha256s=(SHA,),
        placement_identity_sha256=SHA,
        abi_identity_sha256=_minimal_abi().identity_sha256,
        inverse_process_argv_sha256=SHA,
        parseback_full_state_equal=True,
        bytecode_contamination_paths=(),
        research_only=True,
        public_n600_output_equality_owed=True,
    )
    return CompiledPublicRuntimeV1(
        compile_receipt=receipt,
        placement=object(),  # readiness reads only byte-bound compile/ABI parents
        abi_closure=_minimal_abi(),
        source_audits=(),
    )


def test_raw_contract_is_derived_from_frozen_frame_utils() -> None:
    tree = ast.parse((REPO_ROOT / "upstream/frame_utils.py").read_text())
    assignments: dict[str, object] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target = node.targets[0]
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            target = node.target
        else:
            continue
        if target.id in {"seq_len", "camera_size"}:
            assignments[target.id] = ast.literal_eval(node.value)
    width, height = assignments["camera_size"]
    frames = closure.EXPECTED_N_PAIRS * assignments["seq_len"]
    assert frames == closure.EXPECTED_N_FRAMES == 1200
    assert width * height * frames * 3 == closure.EXPECTED_RAW_NBYTES == 3_662_409_600


@pytest.mark.parametrize(
    "payload_class",
    [
        VideoDerivedPayloadClassV1.WEIGHT,
        VideoDerivedPayloadClassV1.LATENT,
        VideoDerivedPayloadClassV1.SELECTOR,
        VideoDerivedPayloadClassV1.THRESHOLD,
        VideoDerivedPayloadClassV1.EXCEPTION,
    ],
)
def test_every_instance_derived_state_must_be_counted(payload_class: VideoDerivedPayloadClassV1) -> None:
    with pytest.raises(PublicAuthClosureError, match="must be counted archive payload"):
        PayloadPlacementItemV1(
            item_id="derived.state.item",
            origin=ContentOriginV1.VIDEO_DERIVED_STATE,
            location=PlacementLocationV1.BOUNDED_PUBLIC_VM,
            content_sha256=SHA,
            object_nbytes=4,
            charged_archive_nbytes=0,
            video_payload_class=payload_class,
        )


@pytest.mark.parametrize("facility", tuple(GenericVMFacilityV1))
def test_arbitrary_generic_public_vm_facilities_are_free(facility: GenericVMFacilityV1) -> None:
    item = PayloadPlacementItemV1(
        item_id=f"generic.{facility.value.lower()}",
        origin=ContentOriginV1.GENERIC_ALGORITHM,
        location=PlacementLocationV1.BOUNDED_PUBLIC_VM,
        content_sha256=SHA,
        object_nbytes=9,
        charged_archive_nbytes=0,
        vm_facility=facility,
    )
    assert item.charged_archive_nbytes == 0


def test_generic_source_audit_rejects_hidden_state_escape() -> None:
    for source in (
        b"import os\nos.system('true')\n",
        b"import torch\ntorch.load('teacher.pt')\n",
        b"import os\nprint(os.getenv('HIDDEN_VIDEO_SELECTOR'))\n",
        b"SCORER_WEIGHT = b'not generic'\n",
    ):
        with pytest.raises(PublicAuthClosureError):
            audit_generic_runtime_source(source, source_name="inflate.py", lineage_attested_generic=True)


def test_generic_source_audit_accepts_deterministic_generic_optimizer() -> None:
    receipt = audit_generic_runtime_source(
        b"import math\ndef solve(x):\n    return min(range(8), key=lambda i: abs(x-math.sin(i)))\n",
        source_name="inflate.py",
        lineage_attested_generic=True,
    )
    assert receipt.passed


def test_real_g25_lvpg2_inverse_preserves_full_quantized_state() -> None:
    if not G25_ARCHIVE.is_file():
        pytest.skip("real G25 archive is not mounted")
    with zipfile.ZipFile(G25_ARCHIVE, mode="r") as archive:
        selected = parse_population_global_member(archive.read(MEMBER_NAME))
    logical = lvpg2_to_lvls1(selected.member_bytes)
    reopened = parse_ep725_lvls1(logical, require_source_form=True)
    assert closure._decoded_state_sha256(selected) == closure._decoded_state_sha256(reopened)


def test_lvpg2_compiler_refuses_current_g17_production_archive_at_typed_boundary(
    tmp_path: Path,
) -> None:
    p_section = b"g66-current-g17-production-p"
    g_section = build_g17_g_packet(p_section=p_section, pair_start=0, pair_count=600)
    a_section = build_g17_a_packet(
        p_section=p_section,
        g_section=g_section,
        pair_start=0,
        pair_count=600,
    )
    terminal = build_g17_terminal_envelope(
        p_section=p_section,
        g_section=g_section,
        a_section=a_section,
    )
    archive = build_g17_production_archive(
        p_section=p_section,
        g_section=g_section,
        a_section=a_section,
        terminal_section=terminal,
    ).selected.outer.archive_bytes
    archive_path = tmp_path / "archive.zip"
    archive_path.write_bytes(archive)

    with pytest.raises(PublicAuthClosureError, match="failed strict packet parse"):
        closure._inspect_exact_lvpg2_archive(archive_path)


def test_real_compile_executes_emitted_inverse_parseback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not G25_ARCHIVE.is_file() or not LVLS1_RUNTIME.is_file():
        pytest.skip("real G25 archive or generic LVLS1 runtime is not mounted")
    monkeypatch.setattr(closure, "capture_interpreter_distribution_abi", lambda _roots: _minimal_abi())
    compiled = compile_lvpg2_public_runtime(
        archive_path=G25_ARCHIVE,
        lvls1_runtime_source_path=LVLS1_RUNTIME,
        runtime_dir=tmp_path / "runtime",
        lineage_attested_generic=True,
    )
    assert compiled.compile_receipt.parseback_full_state_equal
    assert compiled.compile_receipt.public_n600_output_equality_owed
    assert compiled.compile_receipt.research_only


def test_decoder_policy_is_installed_before_lvpg2_inverse_executes(tmp_path: Path) -> None:
    if not G25_ARCHIVE.is_file():
        pytest.skip("real G25 archive is not mounted")
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    inverse_source = REPO_ROOT / "src/tac/witness_dsl/taskspace_lvpg2_public_inverse.py"
    (runtime_dir / "inflate.py").write_bytes(inverse_source.read_bytes())
    (runtime_dir / "inflate.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    (runtime_dir / "lvls1_runtime.py").write_text("raise RuntimeError('must not execute')\n")
    source_member = tmp_path / "submission/archive/0.bin"
    source_member.parent.mkdir(parents=True)
    with zipfile.ZipFile(G25_ARCHIVE, mode="r") as archive:
        source_member.write_bytes(archive.read(MEMBER_NAME))
    destination = tmp_path / "submission/inflated/0.raw"
    forbidden = tmp_path / "forbidden_secret.bin"
    forbidden.write_bytes(b"must remain unread")
    script = """
import importlib.util
import os
from pathlib import Path

runtime = Path(os.environ["G29_TEST_RUNTIME"])
spec = importlib.util.spec_from_file_location("isolated_public_inverse", runtime / "inflate.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
def malicious_inverse(_payload):
    Path(os.environ["G29_TEST_FORBIDDEN"]).read_bytes()
    raise AssertionError("forbidden read unexpectedly succeeded")
module.lvpg2_to_lvls1 = malicious_inverse
module.inflate(Path(os.environ["G29_TEST_SOURCE"]), Path(os.environ["G29_TEST_DESTINATION"]))
"""
    environment = {
        **os.environ,
        "G29_TEST_DESTINATION": destination.as_posix(),
        "G29_TEST_FORBIDDEN": forbidden.as_posix(),
        "G29_TEST_RUNTIME": runtime_dir.as_posix(),
        "G29_TEST_SOURCE": source_member.as_posix(),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "public decoder policy denied external read" in completed.stderr


def test_authority_receipts_have_no_public_constructor() -> None:
    for receipt_type in (
        OfficialEvaluationRunReceiptV1,
        PublicDecodeEqualityReceiptV1,
        PublicEvaluatorExecutionReceiptV1,
    ):
        with pytest.raises(PublicAuthClosureError, match="no public constructor"):
            receipt_type()


def test_authority_receipt_key_schemas_cover_every_dataclass_field() -> None:
    for receipt_type in (
        OfficialEvaluationRunReceiptV1,
        PublicDecodeEqualityReceiptV1,
        PublicEvaluatorExecutionReceiptV1,
    ):
        assert frozenset(item.name for item in fields(receipt_type)) == receipt_type._KEYS


def test_eight_decimal_components_use_rounding_interval_not_adjacent_cent() -> None:
    assert closure._reported_score_interval_consistent(
        displayed_two_decimal="0.08",
        avg_segnet_dist_8dec=0.0005,
        avg_posenet_dist_8dec=0.0001,
        archive_nbytes=2_000,
        original_uncompressed_nbytes=closure.SCORE_RATE_DENOMINATOR,
    )
    assert not closure._reported_score_interval_consistent(
        displayed_two_decimal="0.09",
        avg_segnet_dist_8dec=0.0005,
        avg_posenet_dist_8dec=0.0001,
        archive_nbytes=2_000,
        original_uncompressed_nbytes=closure.SCORE_RATE_DENOMINATOR,
    )


def test_axis_specific_evaluator_abi_rejects_decoder_only_closure() -> None:
    with pytest.raises(PublicAuthClosureError, match="lacks required roots"):
        closure._require_axis_specific_evaluator_abi(
            _minimal_abi(),
            execution_axis=closure.ExecutionAxisV1.CPU,
        )


def test_macos_readiness_is_advisory_and_evaluator_abi_remains_owed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "upstream").mkdir()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    for name in ("inflate.sh", "inflate.py", "lvls1_runtime.py"):
        (runtime / name).write_bytes(name.encode("ascii"))
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        closure.shutil, "which", lambda name: "/usr/bin/sandbox-exec" if name == "sandbox-exec" else None
    )
    readiness = assess_auth_eval_execution_readiness(
        repo_root=tmp_path,
        runtime_dir=runtime,
        compiled=_minimal_compiled(),
        discovery=None,
    )
    assert not readiness.ready_to_execute
    assert "OFFICIAL_AUTHORITY_REQUIRES_LINUX_X86_64_CONTEST_HARDWARE" in readiness.preflight_blockers
    assert "AXIS_SPECIFIC_EVALUATOR_INTERPRETER_PACKAGE_NATIVE_ABI_CAPTURE_OWED" in readiness.preflight_blockers


def test_checkpoint_roundtrip_binds_stage_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "receipt.json"
    artifact_path.write_bytes(b"{}")
    artifact = AuthClosureCheckpointArtifactV1(
        artifact_kind="compile.receipt",
        relative_path="receipt.json",
        content_sha256=hashlib.sha256(b"{}").hexdigest(),
        nbytes=2,
    )
    checkpoint = AuthClosureStageCheckpointV1(
        run_id="g29.test.run",
        stage=AuthClosureStageV1.COMPILE_PUBLIC_RUNTIME,
        stage_ordinal=0,
        previous_checkpoint_sha256=None,
        artifacts=(artifact,),
        blockers=(),
        completed=True,
        research_only=True,
        cleanup_certification_sha256=None,
    )
    assert AuthClosureStageCheckpointV1.from_receipt_bytes(checkpoint.to_receipt_bytes()) == checkpoint
    pretty = json.dumps(json.loads(checkpoint.to_receipt_bytes()), indent=2).encode()
    with pytest.raises(PublicAuthClosureError, match="canonical"):
        AuthClosureStageCheckpointV1.from_receipt_bytes(pretty)


def test_scorer_mirror_ledgers_roundtrip_and_separate_candidate_identity() -> None:
    archive_sha = "1" * 64
    raw_sha = "2" * 64
    static_sha = "3" * 64
    mirror_trace_sha = "4" * 64
    input_entries = tuple(
        closure.ScorerInputBatchEntryV1(
            batch_index=batch_index,
            pair_start_index=batch_index * closure.OFFICIAL_EVALUATOR_BATCH_SIZE,
            pair_count=(
                closure.OFFICIAL_EVALUATOR_BATCH_SIZE
                if batch_index < closure.EXPECTED_EVALUATOR_BATCH_COUNT - 1
                else closure.EXPECTED_N_PAIRS - batch_index * closure.OFFICIAL_EVALUATOR_BATCH_SIZE
            ),
            gt_decoded_uint8_sha256="5" * 64,
            candidate_decoded_uint8_sha256="6" * 64,
            gt_posenet_preprocessed_fp32_sha256="7" * 64,
            candidate_posenet_preprocessed_fp32_sha256="8" * 64,
            gt_segnet_preprocessed_fp32_sha256="9" * 64,
            candidate_segnet_preprocessed_fp32_sha256="a" * 64,
        )
        for batch_index in range(closure.EXPECTED_EVALUATOR_BATCH_COUNT)
    )
    input_content_sha = closure._sha256(
        closure._canonical_json(
            {
                "archive_sha256": archive_sha,
                "candidate_raw_sha256": raw_sha,
                "entries": [item.to_dict() for item in input_entries],
                "execution_axis": closure.ExecutionAxisV1.CPU.value,
                "static_authority_input_file_manifest_sha256": static_sha,
                "upstream_snapshot_sha256": closure.EXPECTED_UPSTREAM_SNAPSHOT_SHA256,
            }
        )
    )
    output_entries = tuple(
        closure.ScorerOutputCellEntryV1(
            pair_index=pair_index,
            target_seg_argmax_u8_sha256="b" * 64,
            candidate_seg_argmax_u8_sha256="c" * 64,
            seg_mismatch_pixels=0,
            seg_dist_fp32_hex=struct.pack("<f", 0.0).hex(),
            target_pose6_fp32_sha256="d" * 64,
            candidate_pose6_fp32_sha256="e" * 64,
            pose_mse_fp32_hex=struct.pack("<f", 0.0).hex(),
        )
        for pair_index in range(closure.EXPECTED_N_PAIRS)
    )
    output_rows_sha = closure._sha256(closure._canonical_json([item.to_dict() for item in output_entries]))
    mirror = closure.ScorerOutputMirrorEquivalenceReceiptV1(
        run_label="A",
        execution_axis=closure.ExecutionAxisV1.CPU,
        archive_sha256=archive_sha,
        candidate_raw_sha256=raw_sha,
        upstream_snapshot_sha256=closure.EXPECTED_UPSTREAM_SNAPSHOT_SHA256,
        static_authority_input_file_manifest_sha256=static_sha,
        scorer_input_batch_content_sha256=input_content_sha,
        official_process_trace_sha256="f" * 64,
        official_report_sha256="0" * 64,
        mirror_process_trace_sha256=mirror_trace_sha,
        mirror_source_sha256="1" * 64,
        reviewed_observation_patch_sha256="2" * 64,
        scorer_output_cell_rows_sha256=output_rows_sha,
        exact_raw_inputs_equal=True,
        exact_preprocessed_inputs_equal=True,
        exact_report_bytes_equal=True,
        scorer_execution_unmodified_except_observation_serialization=True,
        instrumented_mirror_not_official_authority=True,
        capture_method="REVIEWED_INSTRUMENTED_OBSERVATION_MIRROR_V1",
    )
    mirror_ascii = mirror.to_receipt_bytes().decode("ascii")
    input_ledger = closure.ScorerInputBatchLedgerV1(
        archive_sha256=archive_sha,
        execution_axis=closure.ExecutionAxisV1.CPU,
        candidate_raw_sha256=raw_sha,
        upstream_snapshot_sha256=closure.EXPECTED_UPSTREAM_SNAPSHOT_SHA256,
        static_authority_input_file_manifest_sha256=static_sha,
        capture_trace_sha256=mirror_trace_sha,
        entries=input_entries,
        capture_method="INSTRUMENTED_OBSERVATION_MIRROR_DISTORTIONNET_PREPROCESS_CAPTURE_V1",
        observation_mirror_equivalence_receipt_ascii=mirror_ascii,
    )
    output_ledger = closure.ScorerOutputCellLedgerV1(
        archive_sha256=archive_sha,
        execution_axis=closure.ExecutionAxisV1.CPU,
        candidate_raw_sha256=raw_sha,
        upstream_snapshot_sha256=closure.EXPECTED_UPSTREAM_SNAPSHOT_SHA256,
        static_authority_input_file_manifest_sha256=static_sha,
        scorer_input_batch_content_sha256=input_content_sha,
        capture_trace_sha256=mirror_trace_sha,
        entries=output_entries,
        capture_method="INSTRUMENTED_OBSERVATION_MIRROR_DISTORTIONNET_OUTPUT_CAPTURE_V1",
        evaluator_target_cells_evidence_only_not_payload=True,
        observation_mirror_equivalence_receipt_ascii=mirror_ascii,
    )
    assert closure.ScorerInputBatchLedgerV1.from_receipt_bytes(input_ledger.to_receipt_bytes()) == input_ledger
    reopened = closure.ScorerOutputCellLedgerV1.from_receipt_bytes(output_ledger.to_receipt_bytes())
    assert reopened == output_ledger
    assert reopened.candidate_cell_content_sha256 != reopened.content_sha256
    assert not hasattr(reopened, "derived_avg_segnet_dist")
    mirror_other_context = replace(
        mirror,
        run_label="B",
        archive_sha256="3" * 64,
        candidate_raw_sha256="4" * 64,
        official_process_trace_sha256="5" * 64,
        mirror_process_trace_sha256="6" * 64,
    )
    other_context = replace(
        output_ledger,
        archive_sha256=mirror_other_context.archive_sha256,
        candidate_raw_sha256=mirror_other_context.candidate_raw_sha256,
        capture_trace_sha256=mirror_other_context.mirror_process_trace_sha256,
        observation_mirror_equivalence_receipt_ascii=(mirror_other_context.to_receipt_bytes().decode("ascii")),
    )
    assert other_context.candidate_cell_content_sha256 == reopened.candidate_cell_content_sha256
    assert other_context.content_sha256 != reopened.content_sha256
