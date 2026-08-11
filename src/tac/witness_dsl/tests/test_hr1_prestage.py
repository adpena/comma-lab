# SPDX-License-Identifier: MIT
"""Scorer-free apparatus tests for the HR1 SAFE-TO-PREPARE build."""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tac.payload_retention_gate import check_no_measure_and_discard_payload
from tac.witness_dsl.hr1_prestage import (
    BindingState,
    CheckpointManifest,
    FileBindingRequest,
    Hr1Arm,
    Hr1PrestageError,
    MemoryConfiguration,
    MemoryDisposition,
    MemoryProbeReceipt,
    PayloadRecord,
    ResumeManifest,
    TensorDType,
    TensorShapeSpec,
    atomic_write_json,
    bind_existing_file,
    compile_memory_configuration,
    make_four_arm_race_programs,
    make_shape_only_memory_configuration,
    payload_manifest_for_tree,
    stream_sha256,
    unresolved_terminal_binding,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _payload(path: Path, role: str = "checkpoint_state") -> PayloadRecord:
    path.write_bytes(b"real-apparatus-state")
    digest, size = stream_sha256(path)
    return PayloadRecord(role=role, path=str(path), bytes=size, sha256=digest)


def _trained_checkpoint(tmp_path: Path) -> CheckpointManifest:
    return CheckpointManifest(
        arm=Hr1Arm.FULL_RENDERER_FINETUNE,
        event="ce_cell_entry",
        step=17,
        config_sha256="a" * 64,
        root_seed=20260811,
        live_state_roles=("renderer", "hard_tokens"),
        ema_state_roles=("renderer_ema",),
        optimizer_state_roles=("adamw_moments",),
        rng_state_roles=("python", "numpy", "torch"),
        guard_state_roles=("seg_floor", "pose_gate", "rollback_parent"),
        payloads=(_payload(tmp_path / "state.bin"),),
    )


def test_four_arm_factory_is_complete_and_not_enum_padded():
    programs = make_four_arm_race_programs()
    assert tuple(program.arm for program in programs) == tuple(Hr1Arm)
    fingerprints = {
        (program.trainable_state_roles, program.event_graph, program.counted_payload_roles)
        for program in programs
    }
    assert len(fingerprints) == 4


@pytest.mark.parametrize("program", make_four_arm_race_programs())
def test_every_arm_compiles_without_raw_argv_or_consumer(program):
    compiled = program.compile()
    assert compiled.argv == ()
    assert compiled.consumer_bound is False
    assert compiled.execution_allowed is False
    assert "REAL_CONSUMERS_REQUIRED" in compiled.refusal_reasons
    assert compiled.typed_config["consumer_bindings"] == []


def test_frozen_arm_has_no_fake_optimizer_or_ema_state():
    frozen = make_four_arm_race_programs()[0]
    assert frozen.arm is Hr1Arm.FROZEN_DECODE
    assert frozen.trainable_state_roles == ()
    assert frozen.initialization_invariant == "no_optimizer_no_ema_no_trainable_state"


def test_adapter_arm_counts_all_adapter_wire_roles():
    adapter = make_four_arm_race_programs()[2]
    assert {"adapter_factors", "adapter_scales", "adapter_schema"}.issubset(
        adapter.counted_payload_roles
    )
    assert "terminal_renderer_adapter_hooks" in adapter.required_consumer_roles


def test_joint_arm_declares_pair_local_not_full_population_relaxation():
    joint = make_four_arm_race_programs()[3]
    assert "pair_chunk_local_token_proposals" in joint.trainable_state_roles
    assert all("n600_logits" not in role for role in joint.trainable_state_roles)


def test_program_config_hash_is_deterministic():
    first = make_four_arm_race_programs()[1].compile()
    second = make_four_arm_race_programs()[1].compile()
    assert first.typed_config_sha256 == second.typed_config_sha256
    assert len(first.typed_config_sha256) == 64


def test_compiled_program_returns_fresh_config_copy_not_mutable_hash_state():
    compiled = make_four_arm_race_programs()[1].compile()
    mutated = compiled.typed_config
    mutated["execution_allowed"] = True
    assert compiled.typed_config["execution_allowed"] is False
    assert compiled.typed_config_sha256 == make_four_arm_race_programs()[1].compile().typed_config_sha256


def test_checkpoint_schema_requires_real_state_groups(tmp_path):
    checkpoint = _trained_checkpoint(tmp_path)
    assert checkpoint.step == 17
    assert checkpoint.payloads[0].bytes > 0


def test_frozen_checkpoint_refuses_fabricated_optimizer(tmp_path):
    trained = _trained_checkpoint(tmp_path)
    with pytest.raises(Hr1PrestageError, match="cannot fabricate EMA/optimizer"):
        replace(
            trained,
            arm=Hr1Arm.FROZEN_DECODE,
            ema_state_roles=("fake_ema",),
            optimizer_state_roles=("fake_optimizer",),
        )


def test_trained_checkpoint_refuses_missing_ema(tmp_path):
    trained = _trained_checkpoint(tmp_path)
    with pytest.raises(Hr1PrestageError, match="must carry EMA and optimizer"):
        replace(trained, ema_state_roles=())


def test_resume_manifest_requires_checkpoint_custody(tmp_path):
    checkpoint = _trained_checkpoint(tmp_path)
    path = tmp_path / "checkpoint.json"
    record = atomic_write_json(path, checkpoint.to_dict())
    resume = ResumeManifest(
        checkpoint_path=str(path),
        checkpoint_sha256=record.sha256,
        resume_event="ce_cell_entry",
        next_step=18,
        max_recovery_loss_steps=1,
    )
    assert resume.next_step == checkpoint.step + 1


def test_atomic_write_json_replaces_complete_document_and_leaves_no_tmp(tmp_path):
    path = tmp_path / "manifest.json"
    first = atomic_write_json(path, {"generation": 1, "payload": "kept"})
    second = atomic_write_json(path, {"generation": 2, "payload": "kept-again"})
    assert json.loads(path.read_text())["generation"] == 2
    assert first.sha256 != second.sha256
    assert list(tmp_path.glob(".*.tmp")) == []


def test_stream_hash_is_chunk_size_independent(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(bytes(range(256)) * 8193)
    assert stream_sha256(path, chunk_bytes=97) == stream_sha256(path, chunk_bytes=4096)


def test_file_binder_checks_expected_bytes_and_sha(tmp_path):
    path = tmp_path / "object.bin"
    path.write_bytes(b"content-bound")
    digest, size = stream_sha256(path)
    binding = bind_existing_file(
        FileBindingRequest("object", path, expected_bytes=size, expected_sha256=digest)
    )
    assert binding.state is BindingState.BOUND
    assert binding.bytes == size
    assert binding.sha256 == digest


def test_file_binder_refuses_wrong_sha(tmp_path):
    path = tmp_path / "object.bin"
    path.write_bytes(b"content-bound")
    with pytest.raises(Hr1PrestageError, match="SHA-256 mismatch"):
        bind_existing_file(FileBindingRequest("object", path, expected_sha256="0" * 64))


def test_public_intake_binding_is_explicitly_read_only(tmp_path):
    path = tmp_path / "public_source.py"
    path.write_text("VALUE = 1\n")
    binding = bind_existing_file(
        FileBindingRequest("public", path, public_intake_read_only=True)
    )
    assert binding.access == "read_only"


def test_terminal_binding_uses_typed_unresolved_state_not_placeholder_string():
    binding = unresolved_terminal_binding(
        "terminal_archive",
        resolution_trigger="ps135 terminal receipt lands",
    )
    assert binding.state is BindingState.UNRESOLVED_TERMINAL
    assert binding.path is None
    assert binding.sha256 is None
    assert "placeholder" not in json.dumps(binding.to_dict()).lower()


@pytest.mark.parametrize("arm", list(Hr1Arm))
def test_shape_only_compiler_refuses_every_arm_without_fresh_probe(arm):
    config = make_shape_only_memory_configuration(arm)
    decision = compile_memory_configuration(config)
    assert decision.disposition is MemoryDisposition.REFUSE
    assert "FRESH_REAL_CONFIG_MEMORY_PROBE_REQUIRED" in decision.reasons
    assert decision.tensor_storage_lower_bound_bytes > 0
    assert decision.measured_peak_bytes is None


def test_joint_memory_config_is_pair_local_and_not_dense_n600_logits():
    config = make_shape_only_memory_configuration(Hr1Arm.JOINT_TOKEN_RENDERER)
    token_logits = next(tensor for tensor in config.tensors if tensor.role == "pair_local_token_logits")
    assert token_logits.shape == (1, 384, 512, 5)
    assert token_logits.shape[0] != 600


def test_memory_compiler_refuses_config_mismatched_receipt(tmp_path):
    config = make_shape_only_memory_configuration(Hr1Arm.FROZEN_DECODE)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{}\n")
    digest, _ = stream_sha256(receipt_path)
    receipt = MemoryProbeReceipt(
        config_sha256="0" * 64,
        measured_peak_bytes=1234,
        measured_at_utc=datetime.now(UTC).isoformat(),
        measurement_kind="real_config",
        command="python governed_probe.py",
        receipt_path=str(receipt_path),
        receipt_sha256=digest,
    )
    decision = compile_memory_configuration(config, receipt)
    assert decision.disposition is MemoryDisposition.REFUSE
    assert "MEMORY_PROBE_CONFIG_HASH_MISMATCH" in decision.reasons


def test_memory_compiler_refuses_stale_receipt(tmp_path):
    base = make_shape_only_memory_configuration(Hr1Arm.FROZEN_DECODE)
    config = replace(base, unresolved_shape_roles=())
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{}\n")
    digest, _ = stream_sha256(receipt_path)
    now = datetime.now(UTC)
    receipt = MemoryProbeReceipt(
        config_sha256=config.config_sha256,
        measured_peak_bytes=1234,
        measured_at_utc=(now - timedelta(days=2)).isoformat(),
        measurement_kind="real_config",
        command="python governed_probe.py",
        receipt_path=str(receipt_path),
        receipt_sha256=digest,
    )
    decision = compile_memory_configuration(config, receipt, now_utc=now)
    assert decision.disposition is MemoryDisposition.REFUSE
    assert "MEMORY_PROBE_STALE" in decision.reasons


def test_memory_compiler_pass_requires_resolved_shapes_and_fresh_real_receipt(tmp_path):
    base = make_shape_only_memory_configuration(Hr1Arm.FROZEN_DECODE)
    config = replace(base, unresolved_shape_roles=())
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text("{\"measured\": true}\n")
    digest, _ = stream_sha256(receipt_path)
    now = datetime.now(UTC)
    receipt = MemoryProbeReceipt(
        config_sha256=config.config_sha256,
        measured_peak_bytes=987_654_321,
        measured_at_utc=now.isoformat(),
        measurement_kind="real_config",
        command="python governed_probe.py --typed-config exact.json",
        receipt_path=str(receipt_path),
        receipt_sha256=digest,
    )
    decision = compile_memory_configuration(config, receipt, now_utc=now)
    assert decision.disposition is MemoryDisposition.PASS
    assert decision.reasons == ()
    assert decision.measured_peak_bytes == 987_654_321


def test_tensor_shape_storage_lower_bound_handles_packed_int12():
    tensor = TensorShapeSpec(
        role="int12_values",
        shape=(3,),
        dtype=TensorDType.INT12_PACKED_LOWER_BOUND,
        persistence="retained",
    )
    assert tensor.storage_lower_bound_bytes == 5


def test_memory_configuration_rejects_pair_chunk_over_contract_limit():
    tensor = TensorShapeSpec("x", (1,), TensorDType.UINT8, "ephemeral")
    with pytest.raises(Hr1PrestageError, match="pair_chunk"):
        MemoryConfiguration(
            arm=Hr1Arm.FROZEN_DECODE,
            pair_chunk=121,
            verdict_batch=32,
            tensors=(tensor,),
        )


def test_payload_tree_manifest_hashes_every_retained_output(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"a")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/b.bin").write_bytes(b"bb")
    (tmp_path / "payload_manifest.json").write_text("self excluded\n")
    manifest = payload_manifest_for_tree(tmp_path, exclude_names={"payload_manifest.json"})
    assert [row["relative_path"] for row in manifest["records"]] == ["a.bin", "nested/b.bin"]
    assert manifest["total_bytes"] == 3
    assert len(manifest["records_sha256"]) == 64


def test_hr2_materializers_pass_always_keep_payload_gate():
    findings = check_no_measure_and_discard_payload(
        repo_root=_REPO_ROOT,
        strict=False,
        roots=(
            "experiments/ddm_hr2_prestage_build.py",
            "src/tac/differentiable_eval_roundtrip.py",
            "src/tac/witness_dsl/hr1_prestage.py",
        ),
    )
    assert findings == []


def test_hr2_prestage_sources_have_no_scorer_or_model_imports():
    forbidden_prefixes = (
        "modules",
        "tac.scorer",
        "tac.local_acceleration.mlx_scorer_adapters",
    )
    for relative in (
        "experiments/ddm_hr2_prestage_build.py",
        "src/tac/witness_dsl/hr1_prestage.py",
    ):
        tree = ast.parse((_REPO_ROOT / relative).read_text())
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not any(
            module == prefix or module.startswith(prefix + ".")
            for module in imported
            for prefix in forbidden_prefixes
        ), (relative, imported)
