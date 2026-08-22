from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import ddm_jo1_joint_objective_design as design
from experiments import ddm_jo1_joint_objective_worker as worker
from experiments import ddm_jo1_modal_joint_objective as dispatcher
from experiments import ddm_jo1_payload_materializer_worker as materializer_worker
from experiments import ddm_jo2_receiver_close as receiver_close
from experiments import ddm_jo2_residual_runtime as residual_runtime
from experiments import ddm_jo3_joint_objective_entrypoint as entrypoint


def _write(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _record(
    path: Path,
    *,
    source: str = "1" * 64,
    shape: tuple[int, ...] | None = None,
    dtype: str | None = None,
) -> design.ArtifactRef:
    return design.artifact_ref(
        path,
        axis="[unit-test non-authority]",
        source_object_sha256=source,
        shape=shape,
        dtype=dtype,
    )


def _knob(value: float) -> dict[str, object]:
    return {
        "value": value,
        "unit": "unitless",
        "provenance_class": "HYPOTHESIS",
        "source_citation": ".omx/tmp/codex_runs/ddm_jo1_joint_objective_design_spec.md",
        "rederivation_trigger": "first exact JO1 stage field pass",
    }


def _stage(stage_id: str) -> dict[str, object]:
    return {
        "stage_id": stage_id,
        "boundary_event": f"{stage_id}_complete",
        "fail_safe_steps": 10,
        "learning_rate": _knob(1e-3),
        "benefit_weight": _knob(1.0),
        "harm_weight": _knob(1.0),
        "pose_weight": _knob(1.0),
        "rate_proxy_weight": _knob(0.0),
        "field_pass": {
            "at_end": True,
            "pair_start": 0,
            "pair_count": 600,
            "batch_pairs": 16,
            "retained_outputs": sorted(design.REQUIRED_FIELD_OUTPUTS),
        },
        "checkpoint_every_steps": 5,
    }


def _stage_model(stage_id: str = "target_birth") -> design.StageConfig:
    return design.StageConfig.model_validate(_stage(stage_id))


def _config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    complete_inputs: bool = False,
    memory_receipt: design.ArtifactRef | None = None,
    action: str = "prepare",
    include_materializer: bool = False,
) -> design.CompiledConfig:
    monkeypatch.setattr(design, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(design, "MATERIALIZER_OUTPUT_ROOT", tmp_path)
    source = _write(tmp_path / "source.raw", b"source")
    source_sha = design._sha256_path(source)
    archive = _write(tmp_path / "archive.zip", b"rc2-test")
    runtime = tmp_path / "runtime"
    _write(runtime / "inflate.py", b"runtime")
    monkeypatch.setattr(design, "BASE_ARCHIVE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(design, "RC2_ARCHIVE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(design, "RC2_ARCHIVE_SHA256", design._sha256_path(archive))
    monkeypatch.setattr(design, "FX5_ARCHIVE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(design, "FX5_ARCHIVE_SHA256", design._sha256_path(archive))
    monkeypatch.setattr(design, "FX5_RUNTIME_TREE_SHA256", "f" * 64)
    monkeypatch.setattr(design, "MATERIALIZER_EXPECTED_RETAINED_PAYLOAD_BYTES", 1024)
    monkeypatch.setattr(design, "MATERIALIZER_STORAGE_RESERVE_BYTES", 1024)
    monkeypatch.setattr(design, "TRAINING_MIN_AP_FREE_BYTES", 1)
    files = {
        name: _write(tmp_path / f"{name}.bin", name.encode())
        for name in (
            "tokens",
            "gt_field",
            "base_field",
            "pose6",
            "base_pose6",
            "segnet",
            "posenet",
            "compiler",
            "worker",
            "dispatcher",
            "materializer_worker",
            "receiver_close",
            "residual_runtime",
        )
    }
    optional = (
        {
            "rc2_decoded_semantic_tokens": _record(
                files["tokens"], source=source_sha, shape=(600, 384, 512), dtype="uint8"
            ),
            "gt_argmax_field": _record(files["gt_field"], source=source_sha, shape=(600, 384, 512), dtype="uint8"),
            "rc2_base_argmax_field": _record(
                files["base_field"], source=source_sha, shape=(600, 384, 512), dtype="uint8"
            ),
            "source_pose6_targets": _record(files["pose6"], source=source_sha, shape=(600, 6), dtype="float32"),
            "fx5_base_pose6": _record(files["base_pose6"], source=source_sha, shape=(600, 6), dtype="float32"),
        }
        if complete_inputs
        else {
            "rc2_decoded_semantic_tokens": None,
            "gt_argmax_field": None,
            "rc2_base_argmax_field": None,
            "source_pose6_targets": None,
            "fx5_base_pose6": None,
        }
    )
    return design.CompiledConfig.model_validate(
        {
            "schema": design.SCHEMA,
            "action": action,
            "run_id": "ddm_jo1_test",
            "output_root": str(tmp_path),
            "seed": 20260821,
            "retain_all_payloads": True,
            "authority": design.authority_constants().model_dump(mode="json"),
            "inputs": {
                "rc2_archive": _record(archive, source=source_sha).model_dump(mode="json"),
                "rc2_runtime": _record(runtime, source=source_sha).model_dump(mode="json"),
                **{
                    name: None if record is None else record.model_dump(mode="json")
                    for name, record in optional.items()
                },
                "source_object": _record(source, source=source_sha).model_dump(mode="json"),
                "segnet_weights": _record(files["segnet"], source=source_sha).model_dump(mode="json"),
                "posenet_weights": _record(files["posenet"], source=source_sha).model_dump(mode="json"),
                "compiler_source": _record(files["compiler"], source=source_sha).model_dump(mode="json"),
                "worker_source": _record(files["worker"], source=source_sha).model_dump(mode="json"),
                "dispatcher_source": _record(files["dispatcher"], source=source_sha).model_dump(mode="json"),
                "materializer_worker_source": (
                    _record(files["materializer_worker"], source=source_sha).model_dump(mode="json")
                    if include_materializer
                    else None
                ),
                "receiver_close_source": _record(files["receiver_close"], source=source_sha).model_dump(mode="json"),
                "residual_runtime_source": _record(files["residual_runtime"], source=source_sha).model_dump(
                    mode="json"
                ),
                "memory_preflight_receipt": (
                    None if memory_receipt is None else memory_receipt.model_dump(mode="json")
                ),
            },
            "actuation": {
                "family": "hybrid_oriented_context_output_rgb_residual",
                "injection_point": "semantic_renderer_output_before_exact_R",
                "hidden_channels": 8,
                "max_rgb_delta": _knob(3.0),
                "derive_context_from_tokens": True,
                "token_blocks_after_actuation": 0,
                "exact_roundtrip_in_loop": True,
            },
            "objective": {
                "benefit_on_base_errors": True,
                "harm_on_base_correct": True,
                "collateral_rho": 0.89,
                "collateral_augmented_lagrangian": True,
                "pose_augmented_lagrangian": True,
                "pose_hard_cap": design.BASE_DPOSE,
                "rate_proxy_name": "rate_proxy",
                "realized_rate_at_stage_boundary": True,
                "dual_updates": "stage_boundary_only",
                "exact_bhw_admission": True,
            },
            "stages": [_stage(name) for name in design.REQUIRED_STAGE_IDS],
            "checkpoint": {
                "schema": design.CHECKPOINT_SCHEMA,
                "atomic_replace": True,
                "distinct_stage_paths": True,
                "required_state": [
                    "stage_id",
                    "step",
                    "field_pass_cursor",
                    "package_cursor",
                    "live",
                    "ema",
                    "optimizer",
                    "rng",
                    "duals",
                    "config_sha256",
                ],
            },
            "memory_preflight": {
                "required": True,
                "device_class": "NVIDIA T4",
                "real_config": True,
                "requested_memory_bytes": 16 * 1024**3,
                "minimum_headroom_bytes": 512 * 1024**2,
                "minimum_ap_free_bytes": 1,
                "max_age_hours": 24,
            },
            "dispatch": {
                "lane_id": ("ddm_jo1_payload_unblock" if include_materializer else "ddm_jo1_joint_objective"),
                "claim_agent": "MAIN",
                "platform": "modal",
                "gpu": "T4",
                "detach_required": True,
                "provider_detach_ack_required": True,
                "single_flight": True,
                "durable_call_id": True,
                "automatic_terminal_closure": True,
            },
            "materializer": (
                {
                    "vehicle_id": "fx5_e1",
                    "archive": _record(archive, source=source_sha).model_dump(mode="json"),
                    "runtime": _record(runtime, source=source_sha).model_dump(mode="json"),
                    "expected_runtime_tree_sha256": "f" * 64,
                    "batch_pairs": 16,
                    "chunk_pair_limit": 120,
                    "remote_volume_name": "comma-auth-eval-cache-artifacts",
                    "remote_volume_run_id": "ddm_jo1u_fx5_e1_n600_test",
                    "harvest_root": str(tmp_path),
                    "rc2_fallback_reason": None,
                }
                if include_materializer
                else None
            ),
            "workload_config_sha256": None,
        }
    )


def test_exact_score_arithmetic_and_preregistered_bands() -> None:
    observed = design.delta_score(
        fixed=1000,
        introduced=20,
        d_pose_candidate=design.BASE_DPOSE,
        candidate_archive_bytes=design.BASE_ARCHIVE_BYTES + 1176,
    )
    expected = 100 * (20 - 1000) / design.SEG_DENOMINATOR + 25 * 1176 / design.SCORE_DENOMINATOR
    assert observed == pytest.approx(expected)
    assert design.preregistered_band(965) == "LIVE"
    assert design.preregistered_band(964) == "MARGINAL"
    assert design.preregistered_band(923) == "CLOSED-neutral"
    assert design.preregistered_band(-1) == "CLOSED-harmful"
    assert design.PREREGISTERED_LIVE_FLIPS == 965
    assert design.STRICT_TEN_X_FLIPS == 966


def test_prior_law_preserves_exact_and_registered_diagnostics() -> None:
    row = design.prior_law_diagnostics()
    assert row["transferred_gross_recovery_fraction"] == pytest.approx(12_075 / 34_970)
    assert row["predicted_fixes"] == pytest.approx(8203.196311)
    assert row["introduced_per_fixed_required_exact"] == pytest.approx(0.882362927)
    assert row["measured_old_introduced_per_fixed"] == pytest.approx(52_854 / 12_075)
    assert row["suppression_from_measured_ratio_required_exact"] == pytest.approx(4.960705761)
    assert row["registered_collateral_cap"] == 0.89
    assert row["registered_suppression_label"] == 4.93


def test_unknown_missing_config_fields_and_reduced_field_pass_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    body = config.model_dump(mode="json")
    body["unknown"] = 1
    with pytest.raises(ValueError):
        design.CompiledConfig.model_validate(body)
    del body["unknown"]
    del body["seed"]
    with pytest.raises(ValueError):
        design.CompiledConfig.model_validate(body)
    body = config.model_dump(mode="json")
    body["stages"][0]["field_pass"]["pair_count"] = 120
    with pytest.raises(ValueError):
        design.CompiledConfig.model_validate(body)
    body = config.model_dump(mode="json")
    body["stages"][0]["field_pass"]["at_end"] = False
    with pytest.raises(ValueError):
        design.CompiledConfig.model_validate(body)

    complete = _config(tmp_path, monkeypatch, complete_inputs=True)
    body = complete.model_dump(mode="json")
    body["inputs"]["gt_argmax_field"]["shape"] = [120, 384, 512]
    with pytest.raises(ValueError, match="gt_argmax_field"):
        design.CompiledConfig.model_validate(body)


def test_missing_fields_and_memory_receipt_block_readiness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = design.attach_workload_sha256(_config(tmp_path, monkeypatch))
    result = design.readiness(config)
    assert result["status"] == "BLOCKED"
    assert "RC2_BASE_ARGMAX_FIELD_MISSING" in result["blockers"]
    assert any(value.startswith("MEMORY_PREFLIGHT_BLOCKED") for value in result["blockers"])


def test_matching_real_memory_receipt_clears_payload_gates_but_not_implementation_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = design.attach_workload_sha256(_config(tmp_path, monkeypatch, complete_inputs=True))
    receipt_path = tmp_path / "memory.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema": design.MEMORY_RECEIPT_SCHEMA,
                "passed": True,
                "device": "NVIDIA T4",
                "training_batch_pairs": 1,
                "field_batch_pairs": 16,
                "geometry": [600, 384, 512],
                "measured_peak_rss_bytes": 8 * 1024**3,
                "projected_n600_peak_rss_bytes": 10 * 1024**3,
                "projection_method": "measured real pair plus bounded chunk projection",
                "chunk_pair_limit": 120,
                "chunked_verdict": "PASS",
                "requested_memory_bytes": 16 * 1024**3,
                "headroom_bytes": 6 * 1024**3,
                "workload_config_sha256": base.workload_config_sha256,
                "producer_command": ["modal", "run", "memory_preflight"],
                "wall_clock_projection": {"lower_seconds": 1, "upper_seconds": 2},
                "receiver_scale_preflight": {
                    "schema": "ddm_jo3_receiver_scale_preflight.v1",
                    "passed": True,
                    "blockers": [],
                    "endpoint_coordinates": [],
                    "endpoint_pair_denominator": 0,
                    "endpoint_one_sided_coordinate_denominator": 0,
                    "endpoint_blocked_coordinate_denominator": 0,
                    "derivative_mode_denominators": {
                        "central_second_order": 7200,
                        "forward_one_sided_first_order": 0,
                        "backward_one_sided_first_order": 0,
                    },
                    "full_winner_bytes_per_stage": 1,
                    "certified_rebuild_bytes_per_stage": 1,
                    "one_stage_projected_retained_bytes": 10,
                    "all_stage_projected_retained_bytes": 30,
                    "non_solver_and_extra_pass_reserve_bytes": 10,
                    "all_stage_plus_reserve_projected_bytes": 40,
                    "available_free_bytes": 100,
                },
                "retained_payloads": {"real_pair": {"sha256": "a" * 64}},
                "created_at_utc": datetime.now(UTC).isoformat(),
            },
            sort_keys=True,
        )
    )
    complete = base.model_copy(
        update={"inputs": base.inputs.model_copy(update={"memory_preflight_receipt": _record(receipt_path)})}
    )
    complete = design.attach_workload_sha256(complete)
    assert complete.workload_config_sha256 == base.workload_config_sha256
    ready = design.readiness(complete)
    assert ready["status"] == "BLOCKED"
    assert ready["blockers"] == [design.REMOTE_TRAINER_BLOCKER]
    receipt_path.write_text(receipt_path.read_text().replace("NVIDIA T4", "fake"))
    assert design.readiness(complete)["status"] == "BLOCKED"


def test_pose_cap_nonnegative_delta_and_collateral_reject_stage() -> None:
    pose = design.stage_admission(
        fixed=2000,
        introduced=0,
        wrong_to_wrong=0,
        d_pose_candidate=design.BASE_DPOSE + 1e-8,
        candidate_archive_bytes=design.BASE_ARCHIVE_BYTES,
        single_p=True,
        package_parseback_identity=True,
    )
    assert not pose["admissible"] and "POSE_CAP_EXCEEDED" in pose["blockers"]
    neutral = design.stage_admission(
        fixed=0,
        introduced=0,
        wrong_to_wrong=0,
        d_pose_candidate=design.BASE_DPOSE,
        candidate_archive_bytes=design.BASE_ARCHIVE_BYTES,
        single_p=True,
        package_parseback_identity=True,
    )
    assert not neutral["admissible"] and "EXACT_DELTA_NONNEGATIVE" in neutral["blockers"]
    collateral = design.stage_admission(
        fixed=1000,
        introduced=900,
        wrong_to_wrong=0,
        d_pose_candidate=design.BASE_DPOSE,
        candidate_archive_bytes=design.BASE_ARCHIVE_BYTES,
        single_p=True,
        package_parseback_identity=True,
    )
    assert "COLLATERAL_CAP_EXCEEDED" in collateral["blockers"]


def test_hybrid_objective_is_real_and_backpropagates() -> None:
    torch.manual_seed(4)
    actuator = worker.HybridOutputResidual(4, 3.0)
    tokens = torch.randint(0, 5, (2, 8, 12))
    residual = actuator(tokens)
    assert residual.shape == (2, 3, 8, 12)
    assert torch.count_nonzero(residual) == 0
    with torch.no_grad():
        actuator.head.weight[0, 0, 0, 0] = 0.5
    residual = actuator(tokens)
    assert torch.count_nonzero(residual)
    logits = torch.randn(2, 5, 8, 12, requires_grad=True)
    target = torch.randint(0, 5, (2, 8, 12))
    base = target.clone()
    base[:, :2] = (base[:, :2] + 1) % 5
    pose = torch.zeros(2, 6, requires_grad=True)
    loss, metrics = worker.joint_inner_objective(
        seg_logits=logits,
        target=target,
        retained_base_argmax=base,
        pose6_candidate=pose,
        pose6_target=torch.ones(2, 6) * 1e-3,
        rate_proxy=logits.sum() * 0.0,
        duals=worker.DualState(collateral=1.0, pose=1.0),
        stage=_stage_model(),
    )
    loss.backward()
    assert torch.isfinite(loss) and torch.count_nonzero(logits.grad)
    assert set(metrics) == {
        "loss",
        "soft_benefit",
        "soft_harm",
        "collateral_violation",
        "pose_mse",
        "pose_violation",
        "rate_proxy",
    }


def test_jo3_quantized_training_forward_matches_counted_receiver() -> None:
    torch.manual_seed(23)
    model = worker.HybridOutputResidual(3, 2.5)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.1)
    tokens = torch.randint(0, 5, (1, 7, 9))
    trained = entrypoint.quantized_residual(model, tokens)
    payload = residual_runtime.encode_residual_state(model.state_dict(), hidden_channels=3, max_rgb_delta=2.5)
    received = residual_runtime.residual_from_payload(payload)(tokens)
    assert torch.equal(trained, received)
    trained.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert sum(int(torch.count_nonzero(parameter.grad)) for parameter in model.parameters()) > 0


def test_jo3_checkpoint_pointer_restores_live_ema_optimizer_rng_and_cursor(
    tmp_path: Path,
) -> None:
    torch.manual_seed(7)
    model = worker.HybridOutputResidual(2, 1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss = model(torch.randint(0, 5, (1, 6, 8))).square().mean()
    loss.backward()
    optimizer.step()
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    ema = {name: value.detach().clone() + 0.25 for name, value in model.state_dict().items()}
    entrypoint.save_checkpoint(
        checkpoint_root=tmp_path,
        stage_id="joint_balance",
        step=100,
        field_cursor=0,
        package_cursor=0,
        model=model,
        ema=ema,
        optimizer=optimizer,
        duals=worker.DualState(collateral=2.0, pose=3.0),
        config_sha256="d" * 64,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    restored = entrypoint.restore_checkpoint(
        tmp_path,
        model=model,
        optimizer=optimizer,
        expected_config_sha256="d" * 64,
    )
    assert restored is not None
    restored_ema, duals, cursor = restored
    assert cursor == worker.ResumeCursor("joint_balance", 100, 0, 0)
    assert duals == worker.DualState(collateral=2.0, pose=3.0)
    assert all(torch.equal(model.state_dict()[name], value) for name, value in expected.items())
    assert all(torch.equal(restored_ema[name], value) for name, value in ema.items())


def test_jo5_checkpoint_migration_preserves_payload_bytes_and_refuses_state_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _config(tmp_path, monkeypatch, complete_inputs=True)
    rebound = design.rebind_jo3_inputs(
        base,
        rc2_base_argmax_field=Path(base.inputs.rc2_base_argmax_field.path),
        fx5_base_pose6=Path(base.inputs.fx5_base_pose6.path),
        local_entrypoint_source=Path(entrypoint.__file__),
        memory_preflight_receipt=None,
        dispatch_local=True,
        run_id="jo5_source_r7",
    )
    source_config = design.attach_workload_sha256(rebound)
    destination_config = design.attach_workload_sha256(
        rebound.model_copy(
            update={"run_id": "jo5_destination_r8", "workload_config_sha256": None}
        )
    )
    source_config_path = _write(
        tmp_path / "source_config.json",
        design.canonical_json_bytes(source_config.model_dump(mode="json")),
    )
    destination_config_path = _write(
        tmp_path / "destination_config.json",
        design.canonical_json_bytes(destination_config.model_dump(mode="json")),
    )
    source_config_sha256 = design._sha256_path(source_config_path)
    destination_config_sha256 = design._sha256_path(destination_config_path)

    model = worker.HybridOutputResidual(8, 3.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.zeros((1, 4, 5), dtype=torch.long)).square().mean()
    loss.backward()
    optimizer.step()
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    source_resume = tmp_path / "source_checkpoints"
    source_checkpoint = source_resume / "target_birth/source"
    source_manifest = worker.save_checkpoint_bundle(
        source_checkpoint,
        model=model,
        ema_state=ema,
        optimizer=optimizer,
        duals=worker.DualState(collateral=0.5, pose=0.25),
        cursor=worker.ResumeCursor("target_birth", 600, 0, 0),
        config_sha256=source_config_sha256,
    )
    entrypoint.save_resume_pointer(source_resume, source_checkpoint, source_manifest)

    destination_resume = tmp_path / "jo5_destination_r8/checkpoints"
    receipt_path = tmp_path / "migration.json"
    result = entrypoint.migrate_checkpoint(
        source_compiled_config=source_config_path,
        source_config_sha256=source_config_sha256,
        source_resume_from=source_resume,
        destination_compiled_config=destination_config_path,
        destination_config_sha256=destination_config_sha256,
        destination_resume_from=destination_resume,
        output_receipt=receipt_path,
    )
    assert result["training_restarted_from_scratch"] is False
    assert result["cursor"] == {
        "stage_id": "target_birth",
        "step": 600,
        "field_pass_cursor": 0,
        "package_cursor": 0,
    }
    assert all(
        row["byte_identical"]
        and row["source"]["sha256"] == row["destination"]["sha256"]
        for row in result["payload_byte_identity"].values()
    )
    destination_pointer = json.loads(
        (destination_resume / "RESUME_LATEST.json").read_text()
    )
    worker.validate_checkpoint_bundle(
        Path(destination_pointer["checkpoint"]), destination_config_sha256
    )
    worker.validate_checkpoint_bundle(source_checkpoint, source_config_sha256)

    escaped_manifest = json.loads((source_checkpoint / "CHECKPOINT.json").read_text())
    escaped_manifest["payloads"]["duals"] = entrypoint.atomic_json(
        source_checkpoint.parent / "outside_duals.json",
        {"collateral": 0.5, "pose": 0.25},
    )
    escaped_source = tmp_path / "escaped_checkpoints"
    escaped_checkpoint = escaped_source / "target_birth/source"
    escaped_checkpoint.mkdir(parents=True)
    entrypoint.atomic_json(escaped_checkpoint / "CHECKPOINT.json", escaped_manifest)
    entrypoint.save_resume_pointer(escaped_source, escaped_checkpoint, escaped_manifest)
    escaped_destination = design.attach_workload_sha256(
        destination_config.model_copy(
            update={
                "run_id": "jo5_escaped_destination",
                "workload_config_sha256": None,
            }
        )
    )
    escaped_destination_path = _write(
        tmp_path / "escaped_destination_config.json",
        design.canonical_json_bytes(escaped_destination.model_dump(mode="json")),
    )
    with pytest.raises(entrypoint.JO3EntrypointError, match="canonical slot"):
        entrypoint.migrate_checkpoint(
            source_compiled_config=source_config_path,
            source_config_sha256=source_config_sha256,
            source_resume_from=escaped_source,
            destination_compiled_config=escaped_destination_path,
            destination_config_sha256=design._sha256_path(escaped_destination_path),
            destination_resume_from=tmp_path / "jo5_escaped_destination/checkpoints",
            output_receipt=tmp_path / "escaped.json",
        )

    drift_body = destination_config.model_dump(mode="json")
    drift_body["stages"][0]["fail_safe_steps"] += 1
    drift_body["stages"][0]["checkpoint_every_steps"] = 1
    drift_body["workload_config_sha256"] = None
    drift = design.attach_workload_sha256(
        design.CompiledConfig.model_validate(drift_body)
    )
    drift_path = _write(
        tmp_path / "drift_config.json",
        design.canonical_json_bytes(drift.model_dump(mode="json")),
    )
    with pytest.raises(entrypoint.JO3EntrypointError, match="load-bearing"):
        entrypoint.migrate_checkpoint(
            source_compiled_config=source_config_path,
            source_config_sha256=source_config_sha256,
            source_resume_from=source_resume,
            destination_compiled_config=drift_path,
            destination_config_sha256=design._sha256_path(drift_path),
            destination_resume_from=tmp_path / "drift/checkpoints",
            output_receipt=tmp_path / "drift.json",
        )


def test_jo3_stage_materialization_keeps_live_weights_and_binds_partial_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(design, "N_PAIRS", 1)

    class _Semantic(torch.nn.Module):
        def forward(self, tokens: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
            del pair
            return torch.zeros(
                tokens.shape[0], 3, tokens.shape[1], tokens.shape[2]
            )

    torch.manual_seed(17)
    model = worker.HybridOutputResidual(2, 1.5)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.1)
    live = {name: value.detach().clone() for name, value in model.state_dict().items()}
    ema = {name: value.detach().clone() + 0.125 for name, value in model.state_dict().items()}
    entrypoint.materialize_candidate_master(
        stage_root=tmp_path,
        semantic=_Semantic(),
        model=model,
        ema=ema,
        tokens=np.zeros((1, design.SEG_H, design.SEG_W), dtype=np.uint8),
        workload_config_sha256="a" * 64,
        stage_id="target_birth",
    )
    assert all(torch.equal(model.state_dict()[name], value) for name, value in live.items())
    changed_ema = {name: value + 0.25 for name, value in ema.items()}
    with pytest.raises(entrypoint.JO3EntrypointError, match="another object"):
        entrypoint.materialize_candidate_master(
            stage_root=tmp_path,
            semantic=_Semantic(),
            model=model,
            ema=changed_ema,
            tokens=np.zeros((1, design.SEG_H, design.SEG_W), dtype=np.uint8),
            workload_config_sha256="a" * 64,
            stage_id="target_birth",
        )


def test_jo3_receiver_compile_retry_preserves_failed_attempt_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    residual_path = _write(tmp_path / "residual.j2s1", b"residual")
    master_path = _write(tmp_path / "master.npy", b"master")
    codes_path = _write(tmp_path / "codes.npy", b"codes")
    calls: list[Path] = []

    def _compile(**kwargs: object) -> dict[str, object]:
        output = Path(kwargs["output"])
        calls.append(output)
        output.mkdir(parents=True)
        if len(calls) == 1:
            _write(output / "retained_partial.bin", b"keep")
            raise receiver_close.JO2ReceiverCloseError("simulated crash")
        archive = entrypoint.file_record(_write(output / "archive.zip", b"archive"))
        repeat = entrypoint.file_record(_write(output / "archive.repeat.zip", b"archive"))
        parseback = entrypoint.file_record(_write(output / "RECEIVER_PARSEBACK.json", b"{}"))
        result: dict[str, object] = {
            "status": "COMPLETE",
            "archive": archive,
            "archive_repeat": repeat,
            "receiver_parseback": parseback,
        }
        entrypoint.atomic_json(output / "RECEIVER_CLOSE_RESULT.json", result)
        return result

    monkeypatch.setattr(receiver_close, "compile_receiver_closed_stage", _compile)
    kwargs = {
        "stage_root": tmp_path / "stage",
        "residual": entrypoint.file_record(residual_path),
        "solve": {
            "semantic_object_sha256": "c" * 64,
            "candidate_codes": entrypoint.file_record(codes_path),
        },
        "candidate_master": entrypoint.file_record(master_path),
        "archive": tmp_path / "base.zip",
        "runtime_root": tmp_path / "runtime",
        "workload_config_sha256": "d" * 64,
    }
    with pytest.raises(receiver_close.JO2ReceiverCloseError, match="simulated crash"):
        entrypoint.compile_receiver_closed_resumable(**kwargs)
    assert (calls[0] / "retained_partial.bin").read_bytes() == b"keep"
    completed = entrypoint.compile_receiver_closed_resumable(**kwargs)
    assert completed["status"] == "COMPLETE"
    assert calls[1].name == "receiver_close_attempt_0001"
    resumed = entrypoint.compile_receiver_closed_resumable(**kwargs)
    assert resumed == completed
    assert len(calls) == 2


def test_jo2_stage_runtime_unwraps_j2r1_before_f26_semantic_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    _write(
        source / "runtime/residual_archive.py",
        b'    tagged_semantic = semantic_body.startswith((b"SD1M", b"SM3R"))\n',
    )
    _write(
        source / "runtime/f26_inflate.py",
        b"from .compensation_overlay import apply_compensation_overlay\n"
        b'    if not parts.semantic_blob.startswith((WANS1_MAGIC, b"SD1M", b"SM3R")):\n'
        b'        raise InflationError("F26 requires WANS1, SD1M, or SM3R semantic weights")\n'
        b"    semantic = renderer.SemanticTokenRenderer(96)\n"
        b"    tagged_state = renderer.unpack_variant_semantic_or_none(\n"
        b"        parts.semantic_blob,\n"
        b"        semantic.state_dict(),\n"
        b"    )\n"
        b"        records = decode_wans1(parts.semantic_blob)\n"
        b"    semantic.load_state_dict(tagged_state, strict=True)\n"
        b"    setup_seconds = time.perf_counter() - setup_started\n",
    )
    _write(
        source / "inflate.py",
        b'ARCHIVE_SHA256 = "'
        + receiver_close.FX5_ARCHIVE_SHA256.encode()
        + b'"\nARCHIVE_BYTES = 180_386\n',
    )
    monkeypatch.setattr(receiver_close, "FX5_RUNTIME", source)
    staged = tmp_path / "staged"
    receiver_close.stage_runtime(staged, b"archive")
    runtime = (staged / "runtime/f26_inflate.py").read_text()
    split = runtime.index(
        "base_semantic_blob, jo2_residual_payload = split_semantic_blob(parts.semantic_blob)"
    )
    guard = runtime.index("if not base_semantic_blob.startswith")
    load = runtime.index("semantic = renderer.SemanticTokenRenderer(96)")
    assert split < guard < load
    assert runtime.count("split_semantic_blob(parts.semantic_blob)") == 1
    assert "decode_wans1(base_semantic_blob)" in runtime
    assert 'startswith((b"SD1M", b"SM3R", b"J2R1"))' in (
        staged / "runtime/residual_archive.py"
    ).read_text()


def test_jo2_receiver_close_requires_full_staged_decode_before_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(receiver_close, "RAW_BYTES", 4)

    def _runtime(name: str, raw: bytes) -> Path:
        runtime = tmp_path / name
        _write(
            runtime / "archive.zip",
            design.deterministic_single_p_archive(b"member"),
        )
        inflate = _write(
            runtime / "inflate.sh",
            (
                "#!/bin/sh\n"
                "set -eu\n"
                "mkdir -p \"$2\"\n"
                f"printf '{raw.decode()}' > \"$2/0.raw\"\n"
            ).encode(),
        )
        inflate.chmod(0o755)
        return runtime

    passed = receiver_close.validate_staged_receiver(
        _runtime("passing", b"raw!"), tmp_path / "validation_pass"
    )
    assert passed["status"] == "COMPLETE"
    assert passed["raw"]["bytes"] == 4
    assert Path(passed["receipt"]["path"]).is_file()
    assert passed["all_materialized_payloads_retained"] is True

    failed_root = tmp_path / "validation_fail"
    with pytest.raises(
        receiver_close.JO2ReceiverCloseError,
        match="failed before closure",
    ):
        receiver_close.validate_staged_receiver(
            _runtime("failing", b"bad"), failed_root
        )
    assert (failed_root / "output/0.raw").read_bytes() == b"bad"
    assert (failed_root / "inflate.log").is_file()
    assert not (failed_root / "RECEIVER_EXECUTION.json").exists()


def test_endpoint_derivative_stencils_are_in_domain_and_match_unit_step() -> None:
    assert receiver_close.jacobian_probe_offsets(-2048) == (
        (0, 1),
        1.0,
        "forward_one_sided_first_order",
    )
    assert receiver_close.jacobian_probe_offsets(2047) == (
        (-1, 0),
        1.0,
        "backward_one_sided_first_order",
    )
    assert receiver_close.jacobian_probe_offsets(0) == ((-1, 1), 2.0, "central_second_order")
    with pytest.raises(receiver_close.JO2ReceiverCloseError, match="outside int12"):
        receiver_close.jacobian_probe_offsets(2048)


def test_jo3_receiver_scale_preflight_clears_endpoints_and_prices_two_tier_retention() -> None:
    codes = np.zeros((600, 12), dtype=np.int32)
    for row, column in ((63, 10), (67, 10), (150, 0), (150, 7), (162, 6), (214, 8), (252, 11), (450, 9), (543, 4)):
        codes[row, column] = 2047
    surface = type("Surface", (), {"codes": codes})()
    result = entrypoint.receiver_scale_preflight(
        surface=surface,
        free_bytes=603_356_557_312,
        stage_denominator=3,
    )
    assert result["passed"] is True
    assert result["endpoint_pair_denominator"] == 8
    assert result["endpoint_one_sided_coordinate_denominator"] == 9
    assert result["endpoint_blocked_coordinate_denominator"] == 0
    assert result["derivative_mode_denominators"] == {
        "central_second_order": 7191,
        "forward_one_sided_first_order": 0,
        "backward_one_sided_first_order": 9,
    }
    assert result["blockers"] == []
    assert result["full_winner_bytes_per_stage"] > 5 * 1024**3
    assert result["certified_rebuild_bytes_per_stage"] < 100 * 1024**2
    assert result["all_stage_projected_retained_bytes"] == (
        3 * result["one_stage_projected_retained_bytes"]
    )
    assert result["all_stage_plus_reserve_projected_bytes"] < result["available_free_bytes"]


def test_certified_retention_keeps_rebuild_rows_and_full_winner_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    solve_root = tmp_path / "solve"
    retention = entrypoint.CertifiedCandidateRetention(
        solve_root=solve_root,
        stage_id="target_birth",
        workload_config_sha256="a" * 64,
        base_archive_sha256="b" * 64,
    )
    base = np.zeros(receiver_close.D, dtype=np.int32)
    codes = np.stack((base, np.ones(receiver_close.D, dtype=np.int32)))
    slaves = np.arange(2 * 2 * 3 * 3, dtype=np.uint8).reshape(2, 2, 3, 3)
    pose_inputs = np.stack((np.stack((slaves[0], slaves[0])), np.stack((slaves[1], slaves[0]))))
    pose_vectors = np.arange(2 * receiver_close.POSE_DIMS, dtype=np.float32).reshape(
        2, receiver_close.POSE_DIMS
    )
    explored = retention.retain_explored(
        root=solve_root / "pairs/pair_0000/stage_20_jacobian/batch_0000_0002",
        pair=0,
        base_codes=base,
        codes=codes,
        slave_camera=slaves,
        pose_input=pose_inputs,
        pose_vectors=pose_vectors,
    )
    retention.verify_explored_result(explored)
    certificate = json.loads(Path(explored["certified_rebuild_manifest"]["path"]).read_text())
    assert certificate["candidate_denominator"] == 2
    assert certificate["regeneration_context"] == {
        "entrypoint_sha256": design._sha256_path(Path(entrypoint.__file__)),
        "workload_identity_sha256": "a" * 64,
        "base_archive_sha256": "b" * 64,
        "stage_id": "target_birth",
        "solve_phase": "pairs/pair_0000/stage_20_jacobian/batch_0000_0002",
    }
    winner = retention.retain_winner(
        root=solve_root / "pairs/pair_0000/stage_50_winner_full",
        pair=0,
        base_codes=base,
        codes=codes[1],
        slave_camera=slaves[1],
        pose_input=pose_inputs[1],
        pose_vector=pose_vectors[1],
    )
    retention.verify_winner(winner)
    winner_receipt = json.loads(Path(winner["receipt"]["path"]).read_text())
    assert winner_receipt["deterministic_repeat_byte_identical"] is True
    assert Path(winner_receipt["payloads"]["pose_input"]["path"]).is_file()
    monkeypatch.setattr(design, "N_PAIRS", 1)
    inventory = retention.finalize()
    retention.verify_inventory(inventory)


def test_winner_repeat_reuses_exact_exploration_batch_shape_and_retains_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    solve_root = tmp_path / "solve"
    retention = entrypoint.CertifiedCandidateRetention(
        solve_root=solve_root,
        stage_id="target_birth",
        workload_config_sha256="a" * 64,
        base_archive_sha256="b" * 64,
    )
    base = np.zeros(receiver_close.D, dtype=np.int32)
    codes = np.stack((base, np.ones(receiver_close.D, dtype=np.int32)))
    master = np.full((2, 3, 3), 17, dtype=np.uint8)
    calls: list[int] = []

    def _render(_surface: object, _modules: object, values: np.ndarray, _pair: int) -> np.ndarray:
        calls.append(len(values))
        return np.repeat(values[:, :1, None, None], 2 * 3 * 3, axis=1).reshape(
            len(values), 2, 3, 3
        ).astype(np.uint8)

    def _pose(_posenet: object, inputs: np.ndarray) -> np.ndarray:
        # Deliberately batch-shape-dependent: a singleton repeat would differ.
        return np.full(
            (len(inputs), receiver_close.POSE_DIMS), len(inputs), dtype=np.float32
        )

    monkeypatch.setattr(receiver_close, "render_frame0", _render)
    monkeypatch.setattr(receiver_close, "pose_vectors", _pose)
    explored_slaves = _render(None, None, codes, 0)
    explored_inputs = np.stack(
        (explored_slaves, np.repeat(master[None], len(codes), axis=0)), axis=1
    )
    retention.retain_explored(
        root=solve_root / "pairs/pair_0000/stage_40_descent/pass_0000/batch_0000_0002",
        pair=0,
        base_codes=base,
        codes=codes,
        slave_camera=explored_slaves,
        pose_input=explored_inputs,
        pose_vectors=_pose(None, explored_inputs),
    )
    repeated = retention.recompute_selected_winner(
        root=solve_root / "pairs/pair_0000/stage_50_winner_repeat_batch",
        pair=0,
        base_codes=base,
        candidate_codes=tuple(codes),
        selected_index=1,
        master=master,
        surface=None,
        modules=None,
        posenet=None,
    )
    assert calls == [2, 2]
    assert np.array_equal(
        repeated["pose_vector"], np.full(receiver_close.POSE_DIMS, 2, dtype=np.float32)
    )
    assert Path(repeated["repeat_receipt"]["path"]).is_file()
    winner = retention.retain_winner(
        root=solve_root / "pairs/pair_0000/stage_50_winner_full",
        pair=0,
        base_codes=base,
        codes=repeated["codes"],
        slave_camera=repeated["slave_camera"],
        pose_input=repeated["pose_input"],
        pose_vector=repeated["pose_vector"],
    )
    retention.verify_winner(winner)


def test_certified_retention_refuses_candidate_when_certificate_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    retention = entrypoint.CertifiedCandidateRetention(
        solve_root=tmp_path,
        stage_id="target_birth",
        workload_config_sha256="a" * 64,
        base_archive_sha256="b" * 64,
    )
    monkeypatch.setattr(
        entrypoint,
        "atomic_compact_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("simulated full tier")),
    )
    with pytest.raises(entrypoint.JO3EntrypointError, match="refusing candidate"):
        retention.retain_explored(
            root=tmp_path / "phase/batch_0000_0001",
            pair=0,
            base_codes=np.zeros(receiver_close.D, dtype=np.int32),
            codes=np.zeros((1, receiver_close.D), dtype=np.int32),
            slave_camera=np.zeros((1, 2, 3, 3), dtype=np.uint8),
            pose_input=np.zeros((1, 2, 2, 3, 3), dtype=np.uint8),
            pose_vectors=np.zeros((1, receiver_close.POSE_DIMS), dtype=np.float32),
        )


def test_jo3_training_rechecks_scale_storage_before_materializing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = design.attach_workload_sha256(_config(tmp_path, monkeypatch, complete_inputs=True))
    monkeypatch.setattr(
        design,
        "verify_memory_receipt",
        lambda _config: {
            "receiver_scale_preflight": {"all_stage_plus_reserve_projected_bytes": 100}
        },
    )
    disk_usage = type("DiskUsage", (), {"free": 99})()
    monkeypatch.setattr(entrypoint.shutil, "disk_usage", lambda _path: disk_usage)
    with pytest.raises(entrypoint.JO3EntrypointError, match="storage changed after scale preflight"):
        entrypoint.write_storage_policy(tmp_path / "run", config)
    disk_usage.free = 101
    record = entrypoint.write_storage_policy(tmp_path / "run", config)
    policy = json.loads(Path(record["path"]).read_text())
    assert policy["minimum_free_bytes"] == 100
    assert policy["observed_free_bytes"] == 101


def test_jo3_rebind_exposes_only_memory_gate_and_exact_local_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch, complete_inputs=True)
    rebound = design.rebind_jo3_inputs(
        config,
        rc2_base_argmax_field=Path(config.inputs.rc2_base_argmax_field.path),
        fx5_base_pose6=Path(config.inputs.fx5_base_pose6.path),
        local_entrypoint_source=Path(entrypoint.__file__),
        memory_preflight_receipt=None,
        dispatch_local=True,
        run_id="ddm_jo2_joint_objective_r6_test",
    )
    rebound = design.attach_workload_sha256(rebound)
    readiness = design.readiness(rebound)
    assert readiness["blockers"] == ["MEMORY_PREFLIGHT_BLOCKED:memory preflight receipt is absent"]
    seal = tmp_path / "local_seal"
    result = design.prepare(rebound, destination=seal)
    order = json.loads(Path(result["fire_order"]["path"]).read_text())
    memory = next(command for command in order["commands"] if command["ordinal"] == 2)
    train = next(command for command in order["commands"] if command["ordinal"] == 3)
    assert memory["argv"] is not None
    assert "tools/safe_run.py" in memory["argv"]
    assert "TAC_GOVERNED_ADMISSION=1" in memory["argv"]
    assert "experiments.ddm_jo3_joint_objective_entrypoint" in memory["argv"]
    assert train["argv"] is None
    assert order["owner"] == "MAIN"
    assert rebound.run_id == "ddm_jo2_joint_objective_r6_test"


def test_joint_objective_prices_benefit_and_harm_on_one_field_denominator() -> None:
    logits = torch.zeros(1, 5, 1, 4, requires_grad=True)
    target = torch.zeros(1, 1, 4, dtype=torch.long)
    base = target.clone()
    base[..., 0] = 1
    _, metrics = worker.joint_inner_objective(
        seg_logits=logits,
        target=target,
        retained_base_argmax=base,
        pose6_candidate=torch.zeros(1, 6),
        pose6_target=torch.zeros(1, 6),
        rate_proxy=logits.sum() * 0.0,
        duals=worker.DualState(),
        stage=_stage_model(),
    )
    assert metrics["soft_benefit"].item() == pytest.approx(0.2 / 4)
    assert metrics["soft_harm"].item() == pytest.approx(3 * 0.8 / 4)


def test_checkpoint_bundle_carries_complete_resume_state(tmp_path: Path) -> None:
    model = worker.HybridOutputResidual(2, 1.0)

    class _OptimizerState:
        def state_dict(self) -> dict[str, object]:
            return {"state": {}, "param_groups": [{"lr": 1e-3}]}

    optimizer = _OptimizerState()
    manifest = worker.save_checkpoint_bundle(
        tmp_path / "checkpoint",
        model=model,
        ema_state=model.state_dict(),
        optimizer=optimizer,
        duals=worker.DualState(collateral=2.0, pose=3.0),
        cursor=worker.ResumeCursor("target_birth", 3, 32, 1),
        config_sha256="a" * 64,
    )
    assert set(manifest["payloads"]) == {
        "live",
        "ema",
        "optimizer",
        "rng",
        "duals",
        "resume_cursor",
    }
    loaded = worker.validate_checkpoint_bundle(tmp_path / "checkpoint", "a" * 64)
    assert loaded["field_pass_cursor"] == 32
    assert loaded["package_cursor"] == 1


def test_single_p_package_and_retained_payload_completeness(tmp_path: Path) -> None:
    archive = _write(tmp_path / "archive.zip", design.deterministic_single_p_archive(b"body"))
    repeat = _write(tmp_path / "archive.repeat.zip", archive.read_bytes())
    payloads = {}
    for name in sorted(design.REQUIRED_FIELD_OUTPUTS):
        payloads[name] = design._atomic_bytes(tmp_path / f"{name}.bin", name.encode())
    result = worker.validate_stage_package(
        archive=archive,
        repeat_archive=repeat,
        retained_payloads=payloads,
        receiver_parseback_identity=True,
        compensation_object_sha256="b" * 64,
        expected_object_sha256="b" * 64,
    )
    assert result["single_p"] and result["fresh_same_object_compensation"]
    bad = tmp_path / "bad.zip"
    import zipfile

    with zipfile.ZipFile(bad, "w") as output:
        output.writestr("p", b"body")
        output.writestr("extra", b"forbidden")
    with pytest.raises(design.JO1Error):
        design.read_single_p_archive(bad)
    with pytest.raises(worker.JO1WorkerError, match="different object"):
        worker.validate_stage_package(
            archive=archive,
            repeat_archive=repeat,
            retained_payloads=payloads,
            receiver_parseback_identity=True,
            compensation_object_sha256="c" * 64,
            expected_object_sha256="b" * 64,
        )


def test_local_prepare_emits_blocked_ticket_without_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path, monkeypatch)
    result = design.prepare(config, destination=tmp_path / "prepared")
    assert result["status"] == "BLOCKED"
    assert result["dispatch_performed"] is False
    assert Path(result["compiled_config"]["path"]).is_file()
    assert Path(result["fire_order"]["path"]).is_file()
    order = json.loads(Path(result["fire_order"]["path"]).read_text())
    assert order["current_disposition"] == "BLOCKED"
    assert order["commands"][0]["purpose"] == "recover_existing_fx5_base_argmax"
    assert order["commands"][1]["purpose"] == "recover_existing_fx5_base_pose6"
    assert order["commands"][0]["argv"] is not None
    assert order["commands"][1]["argv"] is not None
    assert all(command["argv"] is None for command in order["commands"][2:])


def test_refresh_local_source_pins_updates_all_three_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path, monkeypatch)
    source = Path(config.inputs.receiver_close_source.path)
    source.write_bytes(b"changed receiver source")
    refreshed = design.refresh_local_source_pins(config)
    record = refreshed.inputs.receiver_close_source
    assert record.bytes == source.stat().st_size
    assert record.sha256 == design._sha256_path(source)
    assert record.source_object_sha256 == record.sha256
    assert refreshed.workload_config_sha256 is None


def test_materializer_readiness_is_independent_of_training_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = design.attach_workload_sha256(
        _config(
            tmp_path,
            monkeypatch,
            complete_inputs=True,
            action="materialize_scorer_payloads",
            include_materializer=True,
        )
    )
    materializer = design.materializer_readiness(config)
    assert materializer["status"] == "READY_TO_FIRE"
    assert materializer["blockers"] == []
    assert materializer["storage_probe"]["training_requirement_applied"] is False
    assert materializer["storage_probe"]["required_free_bytes"] == (
        materializer["storage_probe"]["remaining_payload_bytes"] + 1024
    )

    training = design.readiness(config)
    assert training["status"] == "BLOCKED"
    assert design.REMOTE_TRAINER_BLOCKER in training["blockers"]
    assert any(value.startswith("MEMORY_PREFLIGHT_BLOCKED") for value in training["blockers"])

    prepared = design.prepare(config, destination=tmp_path / "materializer_seal")
    assert prepared["status"] == "READY_TO_FIRE"
    order = json.loads(Path(prepared["fire_order"]["path"]).read_text())
    assert order["current_disposition"] == "READY"
    assert order["current_blocker"] is None
    assert order["commands"][0]["requires_reseal_after_harvest"] is True
    assert order["commands"][1]["argv"] is not None
    assert order["commands"][0]["argv"][2].endswith("ddm_jo1_modal_joint_objective.py::materialize_scorer_payloads")
    assert "no active n600 scorer job" in order["commands"][0]["fire_trigger"]


def test_materializer_dispatch_request_is_ready_but_other_entrypoints_stay_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        complete_inputs=True,
        action="materialize_scorer_payloads",
        include_materializer=True,
    )
    prepared = design.prepare(config, destination=tmp_path / "seal")
    request = dispatcher.dispatch_request(
        entrypoint="materialize_scorer_payloads",
        compiled_config=Path(prepared["compiled_config"]["path"]),
        expected_config_sha256=str(prepared["compiled_config"]["sha256"]),
        main_owned_dispatch_authorization=True,
        detach=True,
        provider_detach_ack=True,
    )
    assert request["readiness"]["status"] == "READY_TO_FIRE"

    umbrella = _config(
        tmp_path / "umbrella",
        monkeypatch,
        complete_inputs=True,
        action="prepare",
        include_materializer=True,
    )
    umbrella_prepared = design.prepare(umbrella, destination=tmp_path / "umbrella_seal")
    kwargs = {
        "compiled_config": str(umbrella_prepared["compiled_config"]["path"]),
        "expected_config_sha256": str(umbrella_prepared["compiled_config"]["sha256"]),
        "main_owned_dispatch_authorization": True,
        "detach": True,
        "provider_detach_ack": True,
    }
    with pytest.raises(dispatcher.JO1DispatchError, match=design.REMOTE_TRAINER_BLOCKER):
        dispatcher.memory_preflight(**kwargs)
    with pytest.raises(dispatcher.JO1DispatchError, match="training readiness is blocked"):
        dispatcher.train(**kwargs)


def test_jo2_counted_residual_roundtrip_and_fresh_object_binding() -> None:
    model = residual_runtime.OutputResidual(hidden_channels=4, max_rgb_delta=2.0)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    payload = residual_runtime.encode_residual_state(model.state_dict(), hidden_channels=4, max_rgb_delta=2.0)
    semantic = residual_runtime.pack_semantic_blob(b"base-semantic", payload)
    base, decoded = residual_runtime.split_semantic_blob(semantic)
    assert base == b"base-semantic"
    assert decoded == payload
    restored = residual_runtime.residual_from_payload(payload)
    tokens = torch.zeros((1, 3, 4), dtype=torch.long)
    assert torch.count_nonzero(restored(tokens)) == 0
    corrupted = bytearray(semantic)
    corrupted[-1] ^= 1
    with pytest.raises(residual_runtime.JO2ResidualError, match="digest"):
        residual_runtime.split_semantic_blob(bytes(corrupted))

    camera_a = {"path": "/retained/a.npy", "bytes": 10, "sha256": "a" * 64}
    camera_b = {"path": "/retained/b.npy", "bytes": 10, "sha256": "b" * 64}
    base_pose = {"path": "/retained/base.npy", "bytes": 20, "sha256": "d" * 64}
    first = receiver_close.candidate_object_fingerprint(
        pair=0,
        semantic_object_sha256="c" * 64,
        candidate_master=camera_a,
        base_pose6=base_pose,
    )
    assert first != receiver_close.candidate_object_fingerprint(
        pair=0,
        semantic_object_sha256="c" * 64,
        candidate_master=camera_b,
        base_pose6=base_pose,
    )
    assert first != receiver_close.candidate_object_fingerprint(
        pair=1,
        semantic_object_sha256="c" * 64,
        candidate_master=camera_a,
        base_pose6=base_pose,
    )
    assert first != receiver_close.candidate_object_fingerprint(
        pair=0,
        semantic_object_sha256="c" * 64,
        candidate_master=camera_a,
        base_pose6={**base_pose, "sha256": "e" * 64},
    )


def test_uploaded_materializer_inputs_are_retained_and_resume_exact(
    tmp_path: Path,
) -> None:
    archive = b"archive-payload"
    runtime = b"runtime-transport-payload"
    request = {
        "schema": materializer_worker.REQUEST_SCHEMA,
        "uploads": {
            "archive.zip": {
                "bytes": len(archive),
                "sha256": materializer_worker._sha256_bytes(archive),
            },
            "submission_dir.zip": {
                "bytes": len(runtime),
                "sha256": materializer_worker._sha256_bytes(runtime),
            },
        },
    }
    first = materializer_worker.stage_uploaded_inputs(
        run_root=tmp_path,
        request=request,
        archive_bytes=archive,
        runtime_zip_bytes=runtime,
    )
    second = materializer_worker.stage_uploaded_inputs(
        run_root=tmp_path,
        request=request,
        archive_bytes=archive,
        runtime_zip_bytes=runtime,
    )
    assert first == second
    assert (tmp_path / "inputs/archive.zip").read_bytes() == archive
    assert (tmp_path / "inputs/submission_dir.zip").read_bytes() == runtime

    changed = json.loads(json.dumps(request))
    changed["new_field"] = True
    with pytest.raises(materializer_worker.JO1MaterializerError, match="request differs"):
        materializer_worker.stage_uploaded_inputs(
            run_root=tmp_path,
            request=changed,
            archive_bytes=archive,
            runtime_zip_bytes=runtime,
        )


def test_materializer_and_training_pins_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(
        tmp_path,
        monkeypatch,
        complete_inputs=True,
        action="materialize_scorer_payloads",
        include_materializer=True,
    )
    body = config.model_dump(mode="json")
    body["materializer"]["vehicle_id"] = "rc2"
    with pytest.raises(ValueError, match="fallback requires a written reason"):
        design.CompiledConfig.model_validate(body)
    body["materializer"]["rc2_fallback_reason"] = "live fx5 custody unavailable"
    assert design.CompiledConfig.model_validate(body).materializer is not None

    body = config.model_dump(mode="json")
    body["materializer"]["archive"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fx5_e1 materializer archive pin differs"):
        design.CompiledConfig.model_validate(body)

    monkeypatch.setattr(design, "TRAINING_MIN_AP_FREE_BYTES", 100)
    body = config.model_dump(mode="json")
    body["memory_preflight"]["minimum_ap_free_bytes"] = 99
    with pytest.raises(ValueError, match="training storage bar"):
        design.CompiledConfig.model_validate(body)


def test_materializer_validates_every_retained_batch_payload(
    tmp_path: Path,
) -> None:
    payload = _write(tmp_path / "payload.npy", b"retained-payload")
    record = materializer_worker.retained.file_record(payload)
    rows = []
    cursor = 0
    ordinal = 0
    while cursor < materializer_worker.retained.N_PAIRS:
        pair_end = min(
            cursor + materializer_worker.retained.BATCH_SIZE,
            materializer_worker.retained.N_PAIRS,
        )
        rows.append(
            {
                "ordinal": ordinal,
                "pair_start": cursor,
                "pair_end": pair_end,
                "source_payload": record,
                "seg_input": record,
                "logits": record,
            }
        )
        cursor = pair_end
        ordinal += 1
    receipt_path = tmp_path / "BATCH_RESULTS.jsonl"
    receipt_path.write_bytes(b"".join(materializer_worker.retained.canonical_json_bytes(row) for row in rows))
    scorer = {
        "batch_size": materializer_worker.retained.BATCH_SIZE,
        "retained_batch_receipts": materializer_worker.retained.file_record(receipt_path),
    }
    materializer_worker._validate_scorer_batches(scorer, ("source_payload", "seg_input", "logits"))
    payload.write_bytes(b"drifted")
    with pytest.raises(materializer_worker.retained.WorkerError, match="retained payload"):
        materializer_worker._validate_scorer_batches(scorer, ("source_payload", "seg_input", "logits"))


def test_dispatcher_uses_one_explicit_package_mount_topology() -> None:
    source = Path(dispatcher.__file__).read_text(encoding="utf-8")
    assert 'importlib.import_module("experiments.modal_auth_eval")' in source
    assert '"experiments/__init__.py"' in source
    assert '"experiments/modal_auth_eval.py"' in source
    assert 'remote_path="/workspace/pact/experiments/modal_auth_eval.py"' in source
    assert "from experiments import modal_auth_eval" not in source


def test_remote_failure_recorder_retains_traceback_stage_inputs_and_reraises_builtin(
    tmp_path: Path,
) -> None:
    commits: list[bool] = []

    def fail(stage: dict[str, str]) -> dict[str, object]:
        stage["name"] = "unit_test_worker_stage"
        raise materializer_worker.JO1MaterializerError("unit-test custom remote error")

    with pytest.raises(RuntimeError, match="unit_test_worker_stage") as captured:
        dispatcher._execute_with_remote_failure_receipt(
            run_root=tmp_path,
            inputs_seen={"archive": {"bytes": 7, "sha256": "a" * 64}},
            operation=fail,
            commit=lambda: commits.append(True),
        )
    assert str(captured.value)
    assert commits == [True, True, True]
    start = json.loads((tmp_path / "REMOTE_START.json").read_text(encoding="utf-8"))
    failure = json.loads((tmp_path / "REMOTE_FAILURE.json").read_text(encoding="utf-8"))
    assert start["stage"] == "entrypoint_entered"
    assert failure["stage"] == "unit_test_worker_stage"
    assert failure["inputs_seen"]["archive"]["bytes"] == 7
    assert failure["error_type"] == "JO1MaterializerError"
    assert failure["error_message"] == "unit-test custom remote error"
    assert "JO1MaterializerError" in failure["traceback"]
    immutable = Path(failure["immutable_failure_receipt"]["path"])
    assert immutable.is_file()
    assert dispatcher._file_sha256(immutable) == failure["immutable_failure_receipt"]["sha256"]


def test_materializer_entrypoint_records_failure_before_custom_exception_crosses_modal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Volume:
        def __init__(self) -> None:
            self.commits = 0

        def commit(self) -> None:
            self.commits += 1

    volume = Volume()
    monkeypatch.setattr(dispatcher.auth_eval, "AUTH_CACHE_VOLUME_ROOT", tmp_path)
    monkeypatch.setattr(dispatcher.auth_eval, "auth_cache_vol", volume)

    def fail_stage(**_kwargs: object) -> dict[str, object]:
        raise materializer_worker.JO1MaterializerError("staging failed before first payload")

    monkeypatch.setattr(materializer_worker, "stage_uploaded_inputs", fail_stage)
    archive = b"archive"
    runtime = b"runtime"
    request = {
        "schema": materializer_worker.REQUEST_SCHEMA,
        "remote_volume_run_id": "entrypoint_failure_test",
        "resume_from": "b" * 64,
    }
    with pytest.raises(RuntimeError, match="stage_uploaded_inputs"):
        dispatcher.run_payload_materializer.get_raw_f()(
            request=request,
            archive_bytes=archive,
            runtime_zip_bytes=runtime,
        )
    failure = json.loads((tmp_path / "entrypoint_failure_test/REMOTE_FAILURE.json").read_text(encoding="utf-8"))
    assert failure["inputs_seen"]["archive"] == {
        "bytes": len(archive),
        "sha256": dispatcher.hashlib.sha256(archive).hexdigest(),
    }
    assert failure["inputs_seen"]["runtime_bundle"]["bytes"] == len(runtime)
    assert failure["stage"] == "stage_uploaded_inputs"
    assert volume.commits == 3


def test_malformed_remote_request_still_gets_a_failure_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Volume:
        def commit(self) -> None:
            return None

    monkeypatch.setattr(dispatcher.auth_eval, "AUTH_CACHE_VOLUME_ROOT", tmp_path)
    monkeypatch.setattr(dispatcher.auth_eval, "auth_cache_vol", Volume())
    request = {"schema": materializer_worker.REQUEST_SCHEMA, "resume_from": "c" * 64}
    with pytest.raises(RuntimeError, match="validate_remote_request"):
        dispatcher.run_payload_materializer.get_raw_f()(
            request=request,
            archive_bytes=b"archive",
            runtime_zip_bytes=b"runtime",
        )
    failure_paths = list(tmp_path.glob("invalid_jo1_remote_request_*/REMOTE_FAILURE.json"))
    assert len(failure_paths) == 1
    failure = json.loads(failure_paths[0].read_text(encoding="utf-8"))
    assert failure["stage"] == "validate_remote_request"
    assert "remote_volume_run_id is invalid" in failure["error_message"]


def test_control_plane_probe_requires_explicit_diagnostic_authorization(tmp_path: Path) -> None:
    with pytest.raises(dispatcher.JO1DispatchError, match="diagnostic authorization"):
        dispatcher.probe_control_plane(
            output_receipt=str(tmp_path / "probe.json"),
            diagnostic_authorization=False,
        )
