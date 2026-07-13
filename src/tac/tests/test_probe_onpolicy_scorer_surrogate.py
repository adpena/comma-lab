from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[3]


def _load():
    path = REPO / "tools/probe_onpolicy_scorer_surrogate.py"
    spec = importlib.util.spec_from_file_location("_probe_onpolicy_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_candidate_at_norm_preserves_requested_norm() -> None:
    module = _load()
    theta = torch.tensor([3.0, 4.0])
    gradient = torch.tensor([1.0, 2.0])
    candidate = module._candidate_at_norm(theta, gradient, 0.25)
    assert candidate is not None
    assert float(torch.linalg.vector_norm(candidate - theta)) == pytest.approx(0.25)


def test_atomic_json_replaces_complete_payload(tmp_path: Path) -> None:
    module = _load()
    path = tmp_path / "receipt.json"
    module._atomic_json(path, {"a": 1})
    module._atomic_json(path, {"b": [2]})
    assert path.read_text() == '{\n  "b": [\n    2\n  ]\n}\n'


def test_negative_direction_is_opposite() -> None:
    module = _load()
    theta = torch.zeros(4)
    gradient = torch.tensor([1.0, -2.0, 3.0, -4.0])
    forward = module._candidate_at_norm(theta, gradient, 1.0)
    reverse = module._candidate_at_norm(theta, -gradient, 1.0)
    np.testing.assert_allclose(forward.numpy(), -reverse.numpy(), rtol=0, atol=0)


def test_steady_k20_economics_uses_second_anchor_fit() -> None:
    module = _load()
    steps = [
        {
            "step": step,
            "refresh": step in (1, 21),
            "timing_seconds": {"operational_whole_step": 1.0 if step == 1 else (3.0 if step == 21 else 0.1)},
        }
        for step in range(1, 41)
    ]
    economics = module._steady_economics(20, steps)
    assert economics["t_exact_seconds"] == 3.0
    assert economics["t_surrogate_seconds"] == pytest.approx(0.1)
    assert economics["speedup"] == pytest.approx(60.0 / 4.9)


def test_checkpoint_is_content_addressed_and_contract_bound(tmp_path: Path) -> None:
    module = _load()
    path = tmp_path / "step.pt"
    payload = {
        "schema": module.SCHEMA,
        "run_contract_sha256": "a" * 64,
        "regime": "early",
        "cadence": 20,
        "completed_step": 7,
        "tensor": torch.arange(3),
    }
    meta = module._write_checkpoint(path, payload)
    loaded = module._load_checkpoint(
        meta,
        output_dir=tmp_path,
        expected={
            "schema": module.SCHEMA,
            "run_contract_sha256": "a" * 64,
            "regime": "early",
            "cadence": 20,
            "completed_step": 7,
        },
    )
    torch.testing.assert_close(loaded["tensor"], payload["tensor"])
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="custody"):
        module._load_checkpoint(meta, output_dir=tmp_path, expected={})


def test_source_bundle_preserves_and_authenticates_launch_bytes(tmp_path: Path) -> None:
    module = _load()
    bundle = module._materialize_source_bundle(tmp_path)
    assert set(bundle) == set(module.SOURCE_FILES)
    expected = module._source_custody()
    module._verify_source_bundle(tmp_path, bundle, expected)
    first = next(iter(bundle.values()))
    (tmp_path / first["path"]).write_bytes(b"tamper")
    with pytest.raises(RuntimeError, match="source bundle custody"):
        module._verify_source_bundle(tmp_path, bundle, expected)


def test_source_bundle_must_match_run_contract_custody(tmp_path: Path) -> None:
    module = _load()
    bundle = module._materialize_source_bundle(tmp_path)
    expected = module._source_custody()
    first_name = next(iter(expected))
    expected[first_name] = {**expected[first_name], "sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="run-contract custody"):
        module._verify_source_bundle(tmp_path, bundle, expected)


def test_verdict_requires_valid_matched_exact_control() -> None:
    module = _load()
    good = {
        "arms": {
            "K1": {"sequence_holds_exact_dseg_dpose_descent": True},
            "K20": {
                "sequence_holds_exact_dseg_dpose_descent": True,
                "all_nonrefresh_cycle_validations_hold_teacher_relaxation_descent": True,
            },
        }
    }
    assert module._classify_verdict([good], 1)[0] == "GO"
    bad_control = {
        "arms": {
            "K1": {"sequence_holds_exact_dseg_dpose_descent": False},
            "K20": good["arms"]["K20"],
        }
    }
    assert module._classify_verdict([bad_control], 1)[0] == "NEEDS-MORE"
    bad_target = {
        "arms": {
            "K1": good["arms"]["K1"],
            "K20": {
                "sequence_holds_exact_dseg_dpose_descent": False,
                "all_nonrefresh_cycle_validations_hold_teacher_relaxation_descent": False,
            },
        }
    }
    assert module._classify_verdict([bad_target], 1)[0] == "NO-GO"
