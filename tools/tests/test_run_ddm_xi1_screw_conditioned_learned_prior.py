from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "tools/run_ddm_xi1_screw_conditioned_learned_prior.py"


def _load_runner():
    name = "test_ddm_xi1_screw_conditioned_learned_prior"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_pack_load_round_trips_synthetic_checkpoint_with_trained_bit_depths_cpu() -> None:
    runner = _load_runner()
    integer, compression, packer, _ = runner.configure_hpac()
    trained = runner.build_train_model(integer, compression, torch.device("cpu"))
    with torch.no_grad():
        for index, name in enumerate(runner.EXPECTED_BIT_DEPTH_NAMES):
            parameter = dict(trained.named_parameters())[name]
            parameter.fill_(float(2 + index % 6))

    ema = runner.EMA(trained, decay=0.9, warmup=True)
    terminal = {
        "schema": runner.CHECKPOINT_SCHEMA,
        "ema_shadow": runner._cpu_tree(ema.state_dict()),
    }
    checkpoint_buffer = io.BytesIO()
    torch.save(terminal, checkpoint_buffer)
    checkpoint_payload = checkpoint_buffer.getvalue()
    checkpoint_sha256 = hashlib.sha256(checkpoint_payload).hexdigest()
    checkpoint_record = runner.retain_payload(
        runner.FIX_OUTPUT / "tests" / f"synthetic_checkpoint_{checkpoint_sha256}.pt",
        checkpoint_payload,
    )

    source = runner.load_terminal_ema_for_pack(terminal=terminal, packer=packer, args=runner.model_args())
    assert runner._bit_depth_names(source) == runner.EXPECTED_BIT_DEPTH_NAMES
    for name in runner.EXPECTED_BIT_DEPTH_NAMES:
        assert torch.equal(dict(source.named_parameters())[name], terminal["ema_shadow"][name])

    raw = packer.serialize_self_compressed(source)
    repeat = packer.serialize_self_compressed(source)
    raw_record = runner.retain_payload(runner.FIX_OUTPUT / "tests/synthetic_hpac.raw", raw)
    repeat_record = runner.retain_payload(runner.FIX_OUTPUT / "tests/synthetic_hpac.repeat.raw", repeat)
    assert raw_record["sha256"] == repeat_record["sha256"]

    restored = packer.model_from_args(runner.model_args(), False).eval()
    packer.deserialize_self_compressed(restored, raw)
    generator = torch.Generator(device="cpu").manual_seed(runner.SEED)
    current = torch.randint(0, runner.CLASSES, (2, 64, 64), generator=generator)
    previous = torch.randint(0, runner.CLASSES, (2, 64, 64), generator=generator)
    frame_ids = torch.tensor([0, runner.FRAME_COUNT - 1])
    with torch.no_grad():
        expected = source(current, frame_ids, previous)
        actual = restored(current, frame_ids, previous)
    max_abs_diff = float((expected - actual).abs().max())
    assert max_abs_diff == 0.0

    receipt = {
        "schema": "ddm_xi1f_cpu_pack_roundtrip.v1",
        "axis": "[macOS-CPU unit test; synthetic checkpoint; scorer-free]",
        "checkpoint": checkpoint_record,
        "packed_model": raw_record,
        "packed_model_repeat": repeat_record,
        "bit_depth_parameter_names": list(runner.EXPECTED_BIT_DEPTH_NAMES),
        "max_abs_logit_diff": max_abs_diff,
        "verified_exact": True,
        "score_claim": False,
    }
    runner.atomic_json(runner.FIX_OUTPUT / "tests/CPU_PACK_ROUNDTRIP.json", receipt, replace=True)
    assert json.loads((runner.FIX_OUTPUT / "tests/CPU_PACK_ROUNDTRIP.json").read_text())["verified_exact"]


def test_pack_load_rejects_legacy_checkpoint_without_bit_depth_history() -> None:
    runner = _load_runner()
    integer, compression, packer, _ = runner.configure_hpac()
    trained = runner.build_train_model(integer, compression, torch.device("cpu"))
    legacy_ema = {
        name: value
        for name, value in runner._cpu_tree(trained.state_dict()).items()
        if not name.endswith(".bit_depth")
    }
    terminal = {"schema": runner.LEGACY_CHECKPOINT_SCHEMA, "ema_shadow": legacy_ema}
    with pytest.raises(runner.XI1Error, match="fresh 20-epoch rerun required"):
        runner.load_terminal_ema_for_pack(terminal=terminal, packer=packer, args=runner.model_args())
