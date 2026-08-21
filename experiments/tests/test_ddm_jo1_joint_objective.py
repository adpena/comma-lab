from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch

from experiments import ddm_jo1_joint_objective_design as design
from experiments import ddm_jo1_joint_objective_worker as worker
from experiments import ddm_jo1_modal_joint_objective as dispatcher
from experiments import ddm_jo1_payload_materializer_worker as materializer_worker


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
            "segnet",
            "posenet",
            "compiler",
            "worker",
            "dispatcher",
            "materializer_worker",
        )
    }
    optional = (
        {
            "rc2_decoded_semantic_tokens": _record(
                files["tokens"], source=source_sha, shape=(600, 384, 512), dtype="uint8"
            ),
            "gt_argmax_field": _record(
                files["gt_field"], source=source_sha, shape=(600, 384, 512), dtype="uint8"
            ),
            "rc2_base_argmax_field": _record(
                files["base_field"], source=source_sha, shape=(600, 384, 512), dtype="uint8"
            ),
            "source_pose6_targets": _record(
                files["pose6"], source=source_sha, shape=(600, 6), dtype="float32"
            ),
        }
        if complete_inputs
        else {
            "rc2_decoded_semantic_tokens": None,
            "gt_argmax_field": None,
            "rc2_base_argmax_field": None,
            "source_pose6_targets": None,
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
                "lane_id": (
                    "ddm_jo1_payload_unblock"
                    if include_materializer
                    else "ddm_jo1_joint_objective"
                ),
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


def test_missing_fields_and_memory_receipt_block_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
                "max_memory_allocated_bytes": 8 * 1024**3,
                "max_memory_reserved_bytes": 10 * 1024**3,
                "requested_memory_bytes": 16 * 1024**3,
                "headroom_bytes": 6 * 1024**3,
                "workload_config_sha256": base.workload_config_sha256,
                "producer_command": ["modal", "run", "memory_preflight"],
                "created_at_utc": datetime.now(UTC).isoformat(),
            },
            sort_keys=True,
        )
    )
    complete = base.model_copy(
        update={
            "inputs": base.inputs.model_copy(
                update={"memory_preflight_receipt": _record(receipt_path)}
            )
        }
    )
    complete = design.attach_workload_sha256(complete)
    assert complete.workload_config_sha256 == base.workload_config_sha256
    ready = design.readiness(complete)
    assert ready["status"] == "BLOCKED"
    assert ready["blockers"] == [design.IMPLEMENTATION_BLOCKER]
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


def test_local_prepare_emits_blocked_ticket_without_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, monkeypatch)
    result = design.prepare(config, destination=tmp_path / "prepared")
    assert result["status"] == "BLOCKED"
    assert result["dispatch_performed"] is False
    assert Path(result["compiled_config"]["path"]).is_file()
    assert Path(result["fire_order"]["path"]).is_file()
    order = json.loads(Path(result["fire_order"]["path"]).read_text())
    assert order["current_disposition"] == "BLOCKED"
    assert order["commands"][0]["argv"] is not None
    assert order["commands"][1]["argv"] is None
    assert order["commands"][2]["argv"] is None


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
    assert design.IMPLEMENTATION_BLOCKER in training["blockers"]
    assert any(value.startswith("MEMORY_PREFLIGHT_BLOCKED") for value in training["blockers"])

    prepared = design.prepare(config, destination=tmp_path / "materializer_seal")
    assert prepared["status"] == "READY_TO_FIRE"
    order = json.loads(Path(prepared["fire_order"]["path"]).read_text())
    assert order["current_disposition"] == "READY"
    assert order["current_blocker"] is None
    assert order["commands"][0]["requires_reseal_after_harvest"] is True
    assert order["commands"][1]["argv"] is not None
    assert order["commands"][0]["argv"][2].endswith(
        "ddm_jo1_modal_joint_objective.py::materialize_scorer_payloads"
    )
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
    with pytest.raises(dispatcher.JO1DispatchError, match=design.IMPLEMENTATION_BLOCKER):
        dispatcher.memory_preflight(**kwargs)
    with pytest.raises(dispatcher.JO1DispatchError, match="training readiness is blocked"):
        dispatcher.train(**kwargs)


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


def test_materializer_and_training_pins_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    receipt_path.write_bytes(
        b"".join(materializer_worker.retained.canonical_json_bytes(row) for row in rows)
    )
    scorer = {
        "batch_size": materializer_worker.retained.BATCH_SIZE,
        "retained_batch_receipts": materializer_worker.retained.file_record(receipt_path),
    }
    materializer_worker._validate_scorer_batches(
        scorer, ("source_payload", "seg_input", "logits")
    )
    payload.write_bytes(b"drifted")
    with pytest.raises(materializer_worker.retained.WorkerError, match="retained payload"):
        materializer_worker._validate_scorer_batches(
            scorer, ("source_payload", "seg_input", "logits")
        )
