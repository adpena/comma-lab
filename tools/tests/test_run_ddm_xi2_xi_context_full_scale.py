from __future__ import annotations

import hashlib
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "tools/run_ddm_xi2_xi_context_full_scale.py"


def _load_runner():
    name = "test_ddm_xi2_xi_context_full_scale"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _context_dependencies(runner):
    xi1 = runner.load_xi1()
    pose_warp = xi1.import_path(xi1.POSE_WARP_PATH, "test_ddm_xi2_pose_warp")
    return xi1, pose_warp


def test_context_builder_is_deterministic_and_zero_screw_is_identity() -> None:
    runner = _load_runner()
    xi1, pose_warp = _context_dependencies(runner)
    rows, columns = np.indices((runner.H, runner.W))
    previous = ((rows // 31 + columns // 47) % runner.CLASSES).astype(np.uint8)
    pose = np.zeros(6, dtype=np.float64)
    calibration = np.asarray((0.01, 0.02, -0.03), dtype=np.float64)
    first = runner.derive_xi_context(
        previous,
        pose,
        calibration,
        xi1=xi1,
        pose_warp=pose_warp,
    )
    second = runner.derive_xi_context(
        previous,
        pose,
        calibration,
        xi1=xi1,
        pose_warp=pose_warp,
    )
    assert np.array_equal(first, previous)
    assert np.array_equal(second, previous)
    assert hashlib.sha256(first.tobytes()).hexdigest() == hashlib.sha256(second.tobytes()).hexdigest()

    poses = np.zeros((2, 6), dtype=np.float64)
    frame_zero = runner.causal_context_for_frame(
        0,
        None,
        poses,
        calibration,
        xi1=xi1,
        pose_warp=pose_warp,
    )
    assert not frame_zero.any()


def test_causal_builder_uses_supplied_previous_decode_not_future_plane() -> None:
    runner = _load_runner()
    xi1, pose_warp = _context_dependencies(runner)
    poses = np.zeros((2, 6), dtype=np.float64)
    calibration = np.asarray((0.01, 0.02, -0.03), dtype=np.float64)
    previous_a = np.zeros((runner.H, runner.W), dtype=np.uint8)
    previous_b = np.full((runner.H, runner.W), 3, dtype=np.uint8)
    context_a = runner.causal_context_for_frame(
        1,
        previous_a,
        poses,
        calibration,
        xi1=xi1,
        pose_warp=pose_warp,
    )
    context_b = runner.causal_context_for_frame(
        1,
        previous_b,
        poses,
        calibration,
        xi1=xi1,
        pose_warp=pose_warp,
    )
    assert np.array_equal(context_a, previous_a)
    assert np.array_equal(context_b, previous_b)
    assert not np.array_equal(context_a, context_b)


def test_real_self_compressed_pack_round_trip_retains_payloads() -> None:
    runner = _load_runner()
    xi1 = runner.load_xi1()
    integer, compression, packer, _ = xi1.configure_hpac()
    trained = xi1.build_train_model(integer, compression, torch.device("cpu"))
    with torch.no_grad():
        for index, name in enumerate(xi1.EXPECTED_BIT_DEPTH_NAMES):
            dict(trained.named_parameters())[name].fill_(float(2 + index % 6))
    ema = xi1.EMA(trained, decay=0.9, warmup=True)
    terminal = {
        "schema": runner.CHECKPOINT_SCHEMA,
        "epoch": 60,
        "ema_shadow": runner._cpu_tree(ema.state_dict()),
        "deployment_weights": "terminal ema_shadow",
    }
    checkpoint_buffer = io.BytesIO()
    torch.save(terminal, checkpoint_buffer)
    checkpoint_record = runner.retain_payload(
        runner.RETAINED / "tests/synthetic_terminal_checkpoint.pt",
        checkpoint_buffer.getvalue(),
    )
    assert checkpoint_record["bytes"] > 0

    source = packer.model_from_args(xi1.model_args(), True).eval()
    source.load_state_dict(terminal["ema_shadow"], strict=True)
    packer.set_deployed_bit_depths(source, True)
    raw = packer.serialize_self_compressed(source)
    primary = runner.retain_payload(runner.RETAINED / "tests/synthetic_hpac.raw", raw)
    repeat = runner.retain_payload(runner.RETAINED / "tests/synthetic_hpac.repeat.raw", raw)
    assert primary["sha256"] == repeat["sha256"]

    restored = packer.model_from_args(xi1.model_args(), False).eval()
    packer.deserialize_self_compressed(restored, raw)
    generator = torch.Generator(device="cpu").manual_seed(runner.SEED)
    current = torch.randint(0, runner.CLASSES, (2, 64, 64), generator=generator)
    previous = torch.randint(0, runner.CLASSES, (2, 64, 64), generator=generator)
    ids = torch.tensor([0, runner.FRAME_COUNT - 1])
    with torch.no_grad():
        maximum = float((source(current, ids, previous) - restored(current, ids, previous)).abs().max())
    assert maximum == 0.0
    runner.atomic_json(
        runner.RETAINED / "tests/CPU_PACK_ROUNDTRIP.json",
        {
            "schema": "ddm_xi2_cpu_pack_roundtrip.v1",
            "checkpoint": checkpoint_record,
            "packed": primary,
            "packed_repeat": repeat,
            "max_logit_abs_diff": maximum,
            "verified_exact": True,
            "axis": "[macOS-CPU unit test; synthetic checkpoint; scorer-free]",
            "score_claim": False,
        },
    )


def test_parser_declares_stage_leg_and_auto_resume() -> None:
    runner = _load_runner()
    args = runner.parse_args(["--leg", "prepare", "--resume-from", "auto"])
    assert args.leg == "prepare"
    assert args.resume_from == "auto"


def test_strict_promotion_boundary_is_more_than_two_percent() -> None:
    runner = _load_runner()
    assert runner.CONTROL_RANGE_BYTES == 116_716
    assert runner.PROMOTION_MAX_BYTES == 114_381
    assert runner.PROMOTION_MAX_BYTES < 0.98 * runner.CONTROL_RANGE_BYTES
    assert runner.promotion_passes(114_381)
    assert not runner.promotion_passes(114_382)
