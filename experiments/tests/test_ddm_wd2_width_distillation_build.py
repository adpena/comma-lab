from __future__ import annotations

import brotli
import torch

from experiments import ddm_wd2_student_receiver as receiver
from experiments import ddm_wd2_width_distillation_build as build


def test_design_is_derived_from_exact_sub015_packet_ceiling() -> None:
    receipt = build.design_receipt()
    rows = receipt["candidates"]
    assert rows
    assert receipt["budget_derivation"]["rung_semantic_stream_ceiling_bytes"] == 19_610
    assert (
        receipt["budget_derivation"][
            "strict_sub015_rate_only_semantic_stream_ceiling_bytes"
        ]
        == 19_606
    )
    assert {row["form"] for row in rows} == {"dense", "factorized", "flattened"}
    assert all(row["exact_uncompressed_packet_bytes"] <= 19_606 for row in rows)
    assert len({row["candidate_id"] for row in rows}) == len(rows)


def test_packet_roundtrip_matches_exact_deployment_quantization() -> None:
    torch.manual_seed(20260815)
    spec = receiver.StudentSpec("dense_fixture", "dense", 16, 2)
    model = receiver.StudentSemanticRenderer(spec)
    packet = receiver.pack_student(model)
    restored = receiver.unpack_student(packet)
    assert len(packet) == receiver.serialized_bytes_for_spec(spec)
    assert receiver.pack_student(restored) == packet
    for name, value in model.state_dict().items():
        expected = receiver.quantize_tensor(
            value, embedding=name.endswith("embed.weight")
        )
        torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)


def test_qat_ste_forward_values_are_packet_values_and_gradients_flow() -> None:
    torch.manual_seed(20260815)
    spec = receiver.StudentSpec("flat_fixture", "flattened", 8, 1)
    model = receiver.StudentSemanticRenderer(spec)
    state = receiver.fake_quantize_state(model)
    restored = receiver.unpack_student(receiver.pack_student(model))
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(state[name].detach(), value, rtol=0, atol=0)
    sum(item.sum() for item in state.values() if item.dtype.is_floating_point).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_patched_runtime_is_inactive_identical_and_consumes_student(tmp_path) -> None:
    runtime = tmp_path / "runtime"
    receiver.patch_runtime_tree(build.SOURCE_RUNTIME, runtime)
    base_parts = build._load_residual_parts(build.SOURCE_RUNTIME, build.SOURCE_ARCHIVE)
    patched_parts = build._load_residual_parts(runtime, build.SOURCE_ARCHIVE)
    assert patched_parts == base_parts

    torch.manual_seed(20260815)
    student = receiver.StudentSemanticRenderer(
        receiver.StudentSpec("student_fixture", "factorized", 16, 1, 4)
    )
    packet = receiver.pack_student(student)
    semantic_stream = brotli.compress(packet, mode=brotli.MODE_GENERIC, quality=11)
    hpac, _, carrier = build._source_streams()
    model = build._pack_rx1_model(hpac, semantic_stream, carrier)
    member = model + build.SOURCE_RESIDUAL.read_bytes() + build.SOURCE_TOKEN.read_bytes()
    archive = build.deterministic_zip(member)
    candidate_path = tmp_path / "student.zip"
    candidate_path.write_bytes(archive)
    student_parts = build._load_residual_parts(runtime, candidate_path)
    assert student_parts["semantic_blob"] == packet
    assert receiver.pack_student(receiver.unpack_student(student_parts["semantic_blob"])) == packet


def test_archive_arithmetic_uses_real_serialized_container() -> None:
    torch.manual_seed(20260815)
    spec = build._spec_by_id("flattened_d4_w64")
    model = receiver.StudentSemanticRenderer(spec)
    packet = receiver.pack_student(model)
    semantic_stream = brotli.compress(packet, mode=brotli.MODE_GENERIC, quality=11)
    hpac, _, carrier = build._source_streams()
    model_blob = build._pack_rx1_model(hpac, semantic_stream, carrier)
    member = model_blob + build.SOURCE_RESIDUAL.read_bytes() + build.SOURCE_TOKEN.read_bytes()
    archive = build.deterministic_zip(member)
    assert len(packet) == receiver.serialized_bytes_for_spec(spec)
    assert len(archive) == len(member) + 100
    assert len(model_blob) == build.RX1_WRAPPER_BYTES + len(hpac) + len(semantic_stream) + len(carrier)  # MEASURE_ONLY_OK:seeded unit arithmetic fixture with no empirical measurement


def test_retained_candidate_is_receiver_closed_with_repeat_archive(tmp_path) -> None:
    torch.manual_seed(20260815)
    model = receiver.StudentSemanticRenderer(
        receiver.StudentSpec("fixture", "dense", 16, 1)
    )
    rendered = tmp_path / "fixture.rgb.u8"
    rendered.write_bytes(b"unit-test-render-placeholder")
    result = build.retain_candidate(tmp_path, "fixture/attempt_0000", model, rendered)
    assert result["receiver_parse_back"] is True
    assert result["repeat_byte_identical"] is True
    assert result["payloads"]["archive"]["sha256"] == result["payloads"]["archive_repeat"]["sha256"]
    packet = (tmp_path / "retained/candidates/fixture/attempt_0000/semantic.wd2s").read_bytes()
    assert result["receiver_parse_back_semantic_sha256"] == build.sha256_bytes(packet)
