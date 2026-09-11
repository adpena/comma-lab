# SPDX-License-Identifier: MIT
"""Behaviour tests for the OBX2 base+lattice trainer.

These run on synthetic lattices and small grids so they stay fast; the pinned
born packet and the frozen scorers are exercised by the governed stages, not
here.  The load-bearing check is that the torch training twin and the NumPy
receiver compute the SAME lattice, because the receiver is the verdict authority
and the twin is what the gradients see.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
for _root in (REPO, REPO / "src"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from typing import Any  # noqa: E402

from experiments import ddm_obx2_trainer as tr  # noqa: E402
from experiments import ddm_qbt1_qbflow_trainer as qbt1  # noqa: E402
from tac import obx2_lattice_packet as lat  # noqa: E402


def _stub_base() -> Any:
    """Smallest object with the base surface the module and checkpoint touch."""

    params, boundary, interior = tr.born_state(tr.born_packet())
    return qbt1.QBFLOWTorch(params, boundary, interior)


def _small_spec() -> lat.LatticeSpec:
    return lat.LatticeSpec(
        levels=((5, 3, 4), (9, 6, 8)),
        channels=3,
        bits=(8, 8),
        gate_kind=1,
        quantizer_kind=0,
        entropy_model=0,
        condition_channels=tr.CONDITION_CHANNELS,
        hidden=5,
        outputs=tr.head_width(1),
    )


def _trained_lattice(spec: lat.LatticeSpec, *, seed: int = 4) -> tr.LatticeTorch:
    lattice = tr.LatticeTorch(spec, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for grid in lattice.grids:
            grid.copy_(torch.randn(grid.shape, generator=generator) * 0.2)
        lattice.out_w.copy_(torch.randn(lattice.out_w.shape, generator=generator) * 0.3)
        lattice.out_b.copy_(torch.randn(lattice.out_b.shape, generator=generator) * 0.1)
        lattice.hidden_b.copy_(torch.randn(lattice.hidden_b.shape, generator=generator) * 0.1)
    return lattice


def test_axis_weights_bracket_and_clamp_every_coordinate() -> None:
    coordinate = torch.tensor([-2.0, -1.0, 0.0, 1.0, 5.0])
    low, high, fraction = tr._axis_weights(coordinate, 5)
    assert low.tolist() == [0, 0, 2, 4, 4]
    assert high.tolist() == [1, 1, 3, 4, 4]
    assert torch.all(fraction >= 0.0) and torch.all(fraction <= 1.0)


def test_torch_lattice_sampler_matches_the_numpy_receiver() -> None:
    spec = _small_spec()
    lattice = _trained_lattice(spec)
    codes, scales, fusion = lattice.export()
    grids = lat.lattice_grids(spec, codes, scales)
    rng = np.random.default_rng(9)
    points = 37
    t = rng.uniform(-1.0, 1.0, points).astype(np.float32)
    y = rng.uniform(-1.0, 1.0, points).astype(np.float32)
    x = rng.uniform(-1.0, 1.0, points).astype(np.float32)
    condition = rng.standard_normal((points, spec.condition_channels)).astype(np.float32)
    want = lat.query_lattice_numpy(spec, grids, fusion, t=t, y=y, x=x, condition=condition)
    with torch.no_grad():
        got = lattice(
            t=torch.from_numpy(t),
            y=torch.from_numpy(y),
            x=torch.from_numpy(x),
            condition=torch.from_numpy(condition),
        ).numpy()
    assert got.shape == want.shape
    assert float(np.abs(got - want).max()) < 2.0e-5


def test_quantized_export_round_trips_through_the_packet() -> None:
    spec = _small_spec()
    lattice = _trained_lattice(spec)
    codes, scales, fusion = lattice.export()
    raw = lat.encode_lattice_section(spec, codes=codes, scales=scales, fusion=fusion)
    got_spec, got_codes, got_scales, got_fusion = lat.decode_lattice_section(raw)
    assert got_spec == spec
    for want, have in zip(codes, got_codes, strict=True):
        assert np.array_equal(want.reshape(-1), have.reshape(-1))
    assert got_scales == pytest.approx(scales)
    assert float(got_fusion["gate_tau"][0]) == pytest.approx(tr.DEFAULT_GATE_TAU)


def test_quantize_ste_is_the_identity_in_value_and_passes_gradient_through() -> None:
    values = torch.tensor([0.0, 0.4, -0.9, 1.0], requires_grad=True)
    scale = tr.symmetric_scale(values, 8)
    quantized = tr.quantize_ste(values, scale, 8)
    quantized.sum().backward()
    assert torch.allclose(values.grad, torch.ones_like(values))
    assert float((quantized - values).detach().abs().max()) <= float(scale)


def test_symmetric_scale_never_returns_zero() -> None:
    assert float(tr.symmetric_scale(torch.zeros(4), 8)) > 0.0
    assert float(tr.symmetric_scale(torch.tensor([2.0, -3.0]), 8)) == pytest.approx(3.0 / 127.0)


def test_condition_matches_the_receiver_gate_and_layout() -> None:
    signed = torch.tensor([[[0.0, 2.0, 1.0], [0.4, 3.0, 5.0]]])
    tau = torch.tensor([0.5])
    condition, gate = tr.condition_from_interfaces(signed, tau)
    want_gate = lat.interface_gate(signed.numpy(), 0.5)
    assert condition.shape[-1] == signed.shape[-1] + 1
    assert np.allclose(gate.numpy(), want_gate, atol=1e-6)
    assert np.allclose(condition.numpy()[..., :-1], np.tanh(signed.numpy()), atol=1e-6)
    assert np.allclose(condition.numpy()[..., -1], want_gate, atol=1e-6)


def test_operating_point_pose_weight_is_the_exact_contest_derivative() -> None:
    for d_pose in (1.0e-3, 4.59e-6, 1.0e-5):
        want = 5.0 / math.sqrt(10.0 * d_pose)
        assert tr.operating_point_pose_weight(d_pose) == pytest.approx(want)
    assert tr.operating_point_pose_weight(0.0) > 0.0


def test_pair_order_is_a_seeded_permutation_of_the_full_population() -> None:
    first = tr.pair_order(11, 0)
    again = tr.pair_order(11, 0)
    later = tr.pair_order(11, 1)
    assert sorted(first) == list(range(tr.N))
    assert first == again
    assert first != later


def test_default_spec_fits_the_declared_byte_budget() -> None:
    spec = tr.default_spec()
    total_codes = sum(spec.codes_per_level())
    assert total_codes == 15_840
    # Measured 7.11 bits per code on the first trained geometry; the budget is
    # the 122,000 B gate minus the born model, latents, config, metadata, frame.
    assert total_codes * 7.11 / 8.0 < 15_400


def test_lattice_spec_refuses_a_render_contract_mismatch() -> None:
    bad = lat.LatticeSpec(
        levels=((4, 2, 2),),
        channels=2,
        bits=(8,),
        gate_kind=1,
        quantizer_kind=0,
        entropy_model=0,
        condition_channels=tr.CONDITION_CHANNELS,
        hidden=4,
        outputs=tr.head_width(1) + 1,
    )
    with pytest.raises(tr.OBX2TrainerError):
        tr.LatticeTorch(bad, seed=1)


def test_prequantized_receiver_lattice_does_not_requantize() -> None:
    spec = _small_spec()
    trained = _trained_lattice(spec)
    codes, scales, _ = trained.export()
    grids = lat.lattice_grids(spec, codes, scales)
    receiver = tr.LatticeTorch(spec, seed=0, prequantized=True)
    with torch.no_grad():
        for parameter, grid in zip(receiver.grids, grids, strict=True):
            parameter.copy_(torch.from_numpy(np.ascontiguousarray(grid)))
    used, _ = receiver.quantized_grids()
    for have, want in zip(used, grids, strict=True):
        assert np.array_equal(have.detach().numpy(), want)
    # The training lattice, given the SAME dequantized values, runs them through
    # the straight-through quantizer again; that second pass is what the
    # receiver must not do.
    training_copy = tr.LatticeTorch(spec, seed=0)
    with torch.no_grad():
        for parameter, grid in zip(training_copy.grids, grids, strict=True):
            parameter.copy_(torch.from_numpy(np.ascontiguousarray(grid)))
    requantized, _ = training_copy.quantized_grids()
    for tensor, want in zip(requantized, grids, strict=True):
        assert tensor.shape == want.shape


def test_score_parsed_object_refuses_an_unknown_receiver() -> None:
    with pytest.raises(tr.OBX2TrainerError):
        tr.score_parsed_object(
            b"",
            pair_ids=[0],
            gt=np.zeros((1, 4, 4), dtype=np.uint8),
            pose_target=np.zeros((1, 6), dtype=np.float32),
            workers=1,
            receiver="mps",
        )


def test_head_width_doubles_only_for_the_blend_gate() -> None:
    assert tr.head_width(0) == tr.RENDER_OUTPUTS
    assert tr.head_width(1) == tr.RENDER_OUTPUTS
    assert tr.head_width(2) == 2 * tr.RENDER_OUTPUTS
    with pytest.raises(tr.OBX2TrainerError):
        tr.head_width(7)


def test_blend_correction_matches_each_gate_kind() -> None:
    outputs = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    gate = np.asarray([0.25], dtype=np.float32)
    assert np.allclose(lat.blend_correction(outputs, gate, 0), outputs)
    assert np.allclose(lat.blend_correction(outputs, gate, 1), outputs * 0.25)
    blended = lat.blend_correction(outputs, gate, 2)
    assert blended.shape == (1, 2)
    assert np.allclose(blended, [[1.0 * 0.25 + 3.0 * 0.75, 2.0 * 0.25 + 4.0 * 0.75]])
    with pytest.raises(lat.OBX2PacketError):
        lat.blend_correction(np.zeros((1, 3), dtype=np.float32), gate, 2)
    with pytest.raises(lat.OBX2PacketError):
        lat.blend_correction(outputs, gate, 9)


def test_blend_spec_builds_a_double_width_head() -> None:
    spec = tr.default_spec(gate_kind=2)
    assert spec.outputs == 2 * tr.RENDER_OUTPUTS
    lattice = tr.LatticeTorch(spec, seed=3)
    assert lattice.out_w.shape[1] == 2 * tr.RENDER_OUTPUTS
    assert "gate_tau" in lat.fusion_parameter_order(spec)


def test_assert_resume_compatible_refuses_every_binding_difference() -> None:
    saved = {
        "stage": "joint",
        "seed": 11,
        "lattice_enabled": False,
        "chunk_pairs": 4,
        "lattice_spec": {"channels": 4},
        "learning_rate": 3.0e-4,
    }
    tr.assert_resume_compatible(saved, dict(saved), Path("x"))
    # a non-binding key may differ without refusing
    tr.assert_resume_compatible(saved, {**saved, "learning_rate": 1.0e-4}, Path("x"))
    for key, value in (
        ("stage", "distill"),
        ("seed", 12),
        ("lattice_enabled", True),
        ("chunk_pairs", 8),
        ("lattice_spec", {"channels": 8}),
    ):
        with pytest.raises(tr.OBX2TrainerError):
            tr.assert_resume_compatible(saved, {**saved, key: value}, Path("x"))


def test_resume_restores_the_ema_shadow_onto_the_module_device(tmp_path: Path) -> None:
    from tac.training import EMA

    spec = _small_spec()
    module = tr.OBX2Module(_stub_base(), tr.LatticeTorch(spec, seed=2))
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-4)
    ema = EMA(module, decay=0.997)
    path = tmp_path / "ckpt.pt"
    tr.save_stage_checkpoint(
        path, module=module, ema=ema, optimizer=optimizer, config={"stage": "joint"}, step=1, history=[]
    )
    fresh = tr.OBX2Module(_stub_base(), tr.LatticeTorch(spec, seed=2))
    fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=1e-4)
    fresh_ema = EMA(fresh, decay=0.997)
    tr.load_stage_checkpoint(path, fresh, fresh_optimizer, fresh_ema)
    live = dict(fresh.state_dict())
    assert fresh_ema.shadow, "resume restored an empty EMA shadow"
    for name, shadow in fresh_ema.shadow.items():
        if name in live:
            assert shadow.device == live[name].device, name
    # the invariant the bug broke: one update after a resume must not raise
    fresh_ema.update(fresh)


def test_seg_error_locality_separates_boundary_jitter_from_a_region_error() -> None:
    target = np.zeros((40, 40), dtype=np.uint8)
    target[:, 20:] = 1
    # one wrong pixel ON the class boundary
    jitter = target.copy()
    jitter[10, 19] = 1
    histogram = tr.seg_error_distance_histogram(jitter, target)
    assert histogram[0] == 1
    assert sum(v for k, v in histogram.items() if k > 0) == 0
    # a wrong block deep inside one region
    region = target.copy()
    region[2:8, 2:8] = 1
    histogram = tr.seg_error_distance_histogram(region, target)
    assert histogram[0] == 0
    assert histogram[tr.SEG_LOCALITY_MAX_DISTANCE + 1] > 0
    summary = tr.summarize_locality(histogram, sum(histogram.values()))
    assert summary["boundary_local_fraction_within_1_cell"] == 0.0
    assert summary["region_level_fraction_beyond_4_cells"] > 0.0


def test_seg_error_locality_totals_match_the_error_count_or_refuse() -> None:
    target = np.zeros((16, 16), dtype=np.uint8)
    target[:, 8:] = 2
    got = target.copy()
    got[0, 0] = 1
    got[5, 7] = 2
    histogram = tr.seg_error_distance_histogram(got, target)
    assert sum(histogram.values()) == int((got != target).sum())
    summary = tr.summarize_locality(histogram, sum(histogram.values()))
    assert summary["total_seg_errors"] == sum(histogram.values())
    assert sum(summary["fractions_by_distance"].values()) == pytest.approx(1.0)
    with pytest.raises(tr.OBX2TrainerError):
        tr.summarize_locality(histogram, sum(histogram.values()) + 1)


def test_seg_error_locality_refuses_a_shape_mismatch() -> None:
    with pytest.raises(tr.OBX2TrainerError):
        tr.seg_error_distance_histogram(np.zeros((4, 4), np.uint8), np.zeros((4, 5), np.uint8))
    with pytest.raises(tr.OBX2TrainerError):
        tr.seg_error_distance_histogram(np.zeros((2, 4, 4), np.uint8), np.zeros((2, 4, 4), np.uint8))
