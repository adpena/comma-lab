from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import ddm_ds1_cheap_to_shrink as ds1
from experiments import ddm_wd3_scorer_aware_width_distillation as wd3

ROOT = Path("/Volumes/APDataStore/pact/ddm_s1a_stage_a_adapter")
SEAL = ROOT / "S1A_CHAIN_SEAL.json"
pytestmark = pytest.mark.skipif(not SEAL.is_file(), reason="S1A retained seal is not mounted")


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def assert_record(record: dict) -> Path:
    path = Path(record["path"])
    assert path.is_file()
    assert path.stat().st_size == record["bytes"]
    assert sha256(path) == record["sha256"]
    return path


def test_seal_is_scorer_free_and_main_gated() -> None:
    seal = load("S1A_CHAIN_SEAL.json")
    assert seal["status"] == "APPARATUS_COMPLETE_MAIN_GATES_PENDING"
    assert seal["axis"] == "[macOS-CPU scorer-free exact byte/container apparatus]"
    assert seal["score_claim"] is False
    assert seal["frontier_moved"] is False
    assert seal["training_launched"] is False
    assert seal["scorer_invocations"] == 0
    assert seal["metal_invocations"] == 0
    assert seal["compensation_binding_pass"] is True
    assert seal["carrier_binding_pass"] is True


def test_rj1_custody_and_initializer_are_reproved() -> None:
    receipt = load("RJ1_CUSTODY_REPROOF.json")
    assert receipt["status"] == "PASS"
    assert receipt["verified_current_records_numerator"] == 189
    assert receipt["current_payload_records_denominator"] == 189
    assert len(receipt["voided_metadata_records"]) == 3
    assert receipt["source_tree_read_only"] is True
    initializer = receipt["initializer"]
    assert initializer["bytes"] == 253_955
    assert initializer["sha256"] == "e74ba046af251808ef105cf0a2295f6133efa194360148f3110762765b9db434"


def test_exact_gb1_body_and_every_untouched_section_are_byte_identical() -> None:
    identity = load("GB1_IDENTITY_REPROOF.json")
    assert identity["archive"]["bytes"] == identity["runtime_archive"]["bytes"]
    assert identity["archive"]["sha256"] == identity["runtime_archive"]["sha256"]
    assert identity["archive"]["bytes"] == 180_215
    assert identity["archive"]["sha256"] == ("ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4")
    assert identity["archive_equals_runtime_archive"] is True
    assert identity["identity_rebuild_byte_identical"] is True

    preserved = load("retained/initializer_candidate/SECTION_PRESERVATION.json")
    assert preserved["semantic_length_field_is_the_only_expected_header_byte_change"] is True
    assert preserved["assertions"] == {
        "carrier": True,
        "fixed_residual": True,
        "framing": True,
        "hpac": True,
        "renderer_is_only_mutable_section": True,
        "token_stream": True,
    }
    for section in ("hpac", "carrier", "fixed_residual", "token_stream"):
        assert preserved["candidate_sections"][section] == preserved["source_sections"][section]
    assert preserved["candidate_sections"]["semantic"] != preserved["source_sections"]["semantic"]

    runtime = load("S1A_CHAIN_SEAL.json")["initializer_candidate"]["runtime_patch"]
    assert runtime["source_archive_excluded_on_atomic_copy"] is True
    assert runtime["source_archive_absent_after_candidate_bind"] is True
    assert runtime["runtime_archive_equals_candidate"] is True
    assert runtime["candidate_archive"]["sha256"] != identity["archive"]["sha256"]


def test_rj2_adapters_are_rebound_to_gb1_with_retained_payloads() -> None:
    receipt = load("RJ2_ADAPTER_REPROOF.json")
    assert receipt["carrier_production_chain"] == "CAP1 then DX2 then RR5 then Brotli q9/lgwin16"
    assert receipt["gb1_carrier_equals_rj2_dx2_carrier"] is True
    assert receipt["carrier_stream_byte_identical"] is True
    assert receipt["carrier_state_codes_exact"] is True
    assert receipt["compensation_int12_codes_exact"] is True
    assert receipt["compensation_binding_pass"] is True
    assert receipt["compensation_float_replay_max_abs"] <= receipt["compensation_float_replay_atol"]
    for record in receipt["encoder_retained"].values():
        assert_record(record)
    for record in receipt["retained"].values():
        assert_record(record)


def test_both_births_share_initializer_weights_but_not_rng() -> None:
    variation = load("SEED_VARIATION_REPROOF.json")
    assert variation["weights_identical_numerator"] == variation["weights_identical_denominator"] == 32
    assert variation["optimizer_state_identical"] is True
    assert variation["rng_generator_differs"] is True
    assert variation["overwrite_after_knob"] is False

    births = [
        torch.load(assert_record(row["birth"]), map_location="cpu", weights_only=False) for row in variation["births"]
    ]
    initializer = torch.load(
        "/Volumes/VertigoDataTier/pact/ddm_rj1_renderer_joint_move/precompile_r1/"
        "rungs/film_amortized_flat_w96/renderer_initialization.pt",
        map_location="cpu",
        weights_only=False,
    )
    for birth in births:
        assert birth["initializer_loaded_strict"] is True
        assert birth["training_launched"] is False
        assert birth["scorer_invocations"] == 0
        assert birth["metal_invocations"] == 0
        assert set(birth["live_state_dict"]) == set(initializer)
        assert all(torch.equal(birth["live_state_dict"][name], value) for name, value in initializer.items())
    assert not torch.equal(births[0]["rng"]["generator"], births[1]["rng"]["generator"])


def test_launch_order_is_two_off_then_one_on_with_only_main_blockers() -> None:
    launch = load("MAIN_LAUNCH_ORDER.json")
    assert launch["disposition"] == "QUEUED-WITH-A-FIRE-ORDER"
    assert launch["owner"] == "MAIN"
    assert launch["order"] == ["off_seed_20260815", "off_seed_20260816", "on_seed_20260815"]
    assert launch["training_launched"] is False
    expected_blockers = {
        "global scorer lane is unclaimed",
        "global metal lane is unclaimed",
        "charter launch authorization remains false",
        "r5 PID 63183 exit is not verified in the compiled config",
    }
    for row in launch["runs"]:
        request = json.loads(assert_record(row["request"]).read_text(encoding="utf-8"))
        assert row["apparatus_blockers"] == []
        assert set(row["main_gate_blockers"]) == expected_blockers
        assert request["checkpoint_every_epochs"] == 5
        assert request["retain_all_payloads"] is True
        assert request["resume_from"] == row["resume_from"]["path"]
    off15, off16, on15 = [
        json.loads(assert_record(row["request"]).read_text(encoding="utf-8")) for row in launch["runs"]
    ]
    assert off15["cheap_to_shrink"]["mode"] == off16["cheap_to_shrink"]["mode"] == "off"
    assert on15["cheap_to_shrink"] == {
        "allocation_family": "uniform_bits",
        "base_weight": 1.0,
        "mode": "sampled",
        "rung_weights": [1.0, 1.0],
        "sampler_seed": 20260815,
        "uniform_bits": [3, 2],
    }
    assert off15["resume_from"] == on15["resume_from"]


def test_memory_and_stage_b_contracts_are_typed() -> None:
    memory = load("MEMORY_WALL_PREFLIGHT.json")
    assert memory["pass"] is True
    assert memory["sequential_only"] is True
    assert memory["available_memory_bytes_numerator"] > memory["planned_launcher_limit_bytes_denominator"]
    assert memory["wall_projection"]["total_projected_hours"] == pytest.approx(6.7332777777777775)

    stage_b = load("STAGE_B_FINGERPRINT_CONTRACT.json")
    assert stage_b["status"] == "AWAITING_STAGE_A_TRAINING_OUTPUT"
    assert stage_b["disposition"] == "QUEUED-WITH-A-FIRE-ORDER"
    assert stage_b["owner"] == "MAIN-designated Stage-B moved-field producer"
    assert stage_b["jg2_mirror_consistency"] == {
        "control_600_must_reencode_receiver_consumed_field_byte_identically": True,
        "edited_encode_delta_trustworthy_only_after_control": True,
        "expect_pointer_sha256_must_equal_moved_archive_sha256": True,
        "pointer_archive_member_must_equal_runtime_archive_member": True,
        "runtime_root_must_equal_moved_runtime": True,
    }
    pose = stage_b["pose6_target_custody"]
    assert pose["shape"] == [600, 6]
    assert pose["dtype"] == "<f8"
    assert pose["member_sha256"] == "f73ec194b379a7c04ecf208ac80ab3b1855fe7466ea6eeb7366edafcd824f6a2"


def test_stage_a_uniform_ladder_uses_real_wd3_packets() -> None:
    initializer = torch.load(
        "/Volumes/VertigoDataTier/pact/ddm_rj1_renderer_joint_move/precompile_r1/"
        "rungs/film_amortized_flat_w96/renderer_initialization.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = wd3.receiver.StudentSemanticRenderer(wd3.ARM_SPECS["W96_flattened"])
    model.load_state_dict(initializer, strict=True)
    base = wd3.receiver.uniform_allocation(model, 4)
    config = ds1.CheapToShrinkConfig(
        mode="sampled",
        allocation_family="uniform_bits",
        uniform_bits=(3, 2),
        seed=20260815,
    )
    rungs, receipt = wd3._training_rung_allocations(model, base, config)
    assert len(rungs) == 2
    assert receipt["allocation_family"] == "uniform_bits"
    assert receipt["active"] is True
    assert receipt["base_bytes"] > receipt["rung_bytes"][0] > receipt["rung_bytes"][1]
    assert receipt["rung_byte_savings"] == [
        receipt["base_bytes"] - receipt["rung_bytes"][0],
        receipt["base_bytes"] - receipt["rung_bytes"][1],
    ]


def test_batch_objective_evaluates_the_selected_real_allocation_path(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[float] = []

    def fake_pair(*, allocation: float, **_: object) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(allocation)
        value = torch.tensor(float(allocation), requires_grad=True)
        pair = value.expand(1, 2, 3, 1, 1)
        return pair, value.expand(1, 3, 1, 1)

    def fake_scorer(pair: torch.Tensor, *_: object) -> tuple[torch.Tensor, torch.Tensor]:
        value = pair[:, 0, 0, 0, 0]
        return value[:, None].expand(1, 6), value[:, None, None, None].expand(1, 5, 1, 1)

    def fake_objective(*, student_pose6: torch.Tensor, **_: object) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss = student_pose6[:, 0].sum()
        return loss, {"base_component": loss}

    monkeypatch.setattr(wd3, "paired_receiver_tensor", fake_pair)
    monkeypatch.setattr(wd3, "scorer_forward", fake_scorer)
    monkeypatch.setattr(wd3, "score_native_objective", fake_objective)
    monkeypatch.setattr(wd3, "_load_fixed_frames", lambda *_: torch.zeros(1, 3, 1, 1))
    monkeypatch.setattr(wd3, "_load_teacher_frames", lambda *_: torch.zeros(1, 3, 1, 1))

    cache = {
        "teacher_segnet_logits_f16": np.zeros((1, 5, 1, 1), dtype=np.float16),
        "teacher_segnet_argmax_u8": np.zeros((1, 1, 1), dtype=np.uint8),
        "teacher_top1_runnerup_margin_f16": np.zeros((1, 1, 1), dtype=np.float16),
        "teacher_posenet_first6_f32": np.zeros((1, 6), dtype=np.float32),
        "original_gt_segnet_argmax_u8": np.zeros((1, 1, 1), dtype=np.uint8),
        "original_gt_posenet_first6_f32": np.zeros((1, 6), dtype=np.float32),
    }
    config = ds1.CheapToShrinkConfig(
        mode="sampled",
        allocation_family="uniform_bits",
        uniform_bits=(3, 2),
        seed=20260815,
    )
    selected = ds1.select_rung_for_step(config, step=7, rung_count=2)
    total, components = wd3._batch_objective(
        model=object(),
        allocation=1.0,
        ids=np.array([0], dtype=np.int64),
        tokens=torch.zeros(1, 1),
        cache=cache,
        selection=np.zeros((1, 1, 1), dtype=np.uint8),
        posenet=object(),
        segnet=object(),
        device=torch.device("cpu"),
        thresholds=object(),
        duals=object(),
        cheap_to_shrink=config,
        rung_allocations=(3.0, 5.0),
        step=7,
    )
    selected_value = (3.0, 5.0)[selected]
    assert calls == [1.0, selected_value]
    assert total.item() == 1.0 + 2.0 * selected_value
    assert components["cheap_to_shrink_active"].item() == 1.0
    assert components["cheap_to_shrink_rungs_evaluated"].item() == 1.0
