from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tac.pr130_lift.pose import mps_port
from tac.pr130_lift.pose.source_loader import load_lifted_module


def _state(optimizer: torch.optim.Optimizer, parameter: torch.Tensor) -> dict[str, torch.Tensor]:
    return optimizer.state[parameter]


def test_dense_adapter_matches_reference_over_64_step_cpu_trajectory() -> None:
    lifted = load_lifted_module("train_pose_carrier_full")
    initial = torch.linspace(-0.4, 0.6, 10 * 4).reshape(10, 4)
    sparse = torch.nn.Embedding(10, 4, sparse=True)
    dense = torch.nn.Embedding(10, 4, sparse=False)
    with torch.no_grad():
        sparse.weight.copy_(initial)
        dense.weight.copy_(initial)
    sparse_optimizer = lifted.RowLocalSparseAdam([sparse.weight], lr=0.037)
    dense_optimizer = mps_port.RowLocalDenseAdam([dense.weight], lr=0.037)
    sparse_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        sparse_optimizer, T_max=64, eta_min=0.00037
    )
    dense_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        dense_optimizer, T_max=64, eta_min=0.00037
    )
    generator = torch.Generator().manual_seed(130)
    steps = []
    for _ in range(64):
        sampled = torch.randint(0, 10, (6,), generator=generator)
        steps.append(torch.cat((sampled, sampled[:3])))

    for step, row_ids in enumerate(steps, start=1):
        sparse_before = sparse.weight.detach().clone()
        dense_before = dense.weight.detach().clone()
        sparse_optimizer.zero_grad(set_to_none=True)
        dense_optimizer.zero_grad(set_to_none=True)
        sparse(row_ids).square().sum().backward()
        dense(row_ids).square().sum().backward()
        sparse_rows = mps_port.prepare_row_local_step(
            sparse_optimizer, sparse.weight, row_ids, 0.75
        )
        dense_rows = mps_port.prepare_row_local_step(
            dense_optimizer, dense.weight, row_ids, 0.75
        )
        expected_rows = torch.unique(row_ids, sorted=True)
        assert torch.equal(sparse_rows, expected_rows)
        assert torch.equal(dense_rows, expected_rows)
        sparse_optimizer.step()
        dense_optimizer.step()
        sparse_scheduler.step()
        dense_scheduler.step()

        assert torch.equal(sparse.weight, dense.weight)
        sparse_state = _state(sparse_optimizer, sparse.weight)
        dense_state = _state(dense_optimizer, dense.weight)
        for key in ("row_step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(sparse_state[key], dense_state[key])
        untouched = torch.ones(10, dtype=torch.bool)
        untouched.index_fill_(0, expected_rows, False)
        assert torch.equal(sparse.weight[untouched], sparse_before[untouched])
        assert torch.equal(dense.weight[untouched], dense_before[untouched])
        assert sparse_optimizer.param_groups[0]["lr"] == dense_optimizer.param_groups[0]["lr"]
        assert int(sparse_state["row_step"].sum()) >= step


def test_reference_sparse_is_default_and_dense_adapter_requires_opt_in() -> None:
    lifted = load_lifted_module("train_pose_carrier_full")
    sparse, sparse_optimizer = mps_port.build_row_local_coefficients(
        num_embeddings=8,
        embedding_dim=3,
        device=torch.device("cpu"),
        lr=0.02,
        sparse_optimizer_type=lifted.RowLocalSparseAdam,
    )
    dense, dense_optimizer = mps_port.build_row_local_coefficients(
        num_embeddings=8,
        embedding_dim=3,
        device=torch.device("cpu"),
        lr=0.02,
        sparse_optimizer_type=lifted.RowLocalSparseAdam,
        mode=mps_port.DENSE_ADAPTER_MODE,
    )

    assert sparse.sparse is True
    assert type(sparse_optimizer) is lifted.RowLocalSparseAdam
    assert dense.sparse is False
    assert isinstance(dense_optimizer, mps_port.RowLocalDenseAdam)


def test_reference_sparse_mps_refuses_unpinned_torch(monkeypatch) -> None:
    lifted = load_lifted_module("train_pose_carrier_full")
    monkeypatch.setattr(torch, "__version__", "2.9.0")

    with pytest.raises(RuntimeError, match=r"receipt-pinned Torch 2\.10\.0"):
        mps_port.build_row_local_coefficients(
            num_embeddings=8,
            embedding_dim=3,
            device=torch.device("mps"),
            lr=0.02,
            sparse_optimizer_type=lifted.RowLocalSparseAdam,
        )


def test_dense_adapter_refuses_undeclared_gradient_rows() -> None:
    embedding = torch.nn.Embedding(6, 2, sparse=False)
    optimizer = mps_port.RowLocalDenseAdam([embedding.weight], lr=0.1)
    embedding(torch.tensor([1, 1, 4])).square().sum().backward()
    embedding.weight.grad[3, 0] = 1.0

    with pytest.raises(RuntimeError, match="undeclared rows"):
        mps_port.prepare_row_local_step(
            optimizer, embedding.weight, torch.tensor([1, 1, 4]), 1.0
        )


def test_dense_adapter_requires_fresh_row_declaration_each_step() -> None:
    embedding = torch.nn.Embedding(4, 2, sparse=False)
    optimizer = mps_port.RowLocalDenseAdam([embedding.weight], lr=0.1)
    embedding(torch.tensor([2, 2])).square().sum().backward()
    mps_port.prepare_row_local_step(
        optimizer, embedding.weight, torch.tensor([2, 2]), 1.0
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    embedding(torch.tensor([2])).square().sum().backward()

    with pytest.raises(RuntimeError, match="requires rows"):
        optimizer.step()


def test_device_cache_dispatch_without_accelerator_access(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append("mps"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("cuda"))

    mps_port.clear_device_cache(torch.device("cpu"))
    mps_port.clear_device_cache(torch.device("mps"))
    mps_port.clear_device_cache(torch.device("cuda"))

    assert calls == ["mps", "cuda"]
    with pytest.raises(ValueError, match="unsupported"):
        mps_port.clear_device_cache(torch.device("meta"))


def test_safetensors_load_is_cpu_first(monkeypatch) -> None:
    events: list[tuple[str, str]] = []

    class RecordingLinear(torch.nn.Linear):
        def load_state_dict(self, state_dict, *args, **kwargs):
            events.append(("load_state_dict", self.weight.device.type))
            return super().load_state_dict(state_dict, *args, **kwargs)

        def to(self, device, *args, **kwargs):
            events.append(("to", torch.device(device).type))
            return super().to(device, *args, **kwargs)

    module = RecordingLinear(3, 2)
    state = {key: value.detach().clone() for key, value in module.state_dict().items()}

    def fake_load_file(path: str, *, device: str):
        assert path == "pose.safetensors"
        events.append(("load_file", device))
        return state

    monkeypatch.setattr(mps_port, "load_file", fake_load_file)
    result = mps_port.load_safetensors_cpu_then_move(
        module, "pose.safetensors", torch.device("cpu")
    )

    assert result is module
    assert events == [
        ("load_file", "cpu"),
        ("load_state_dict", "cpu"),
        ("to", "cpu"),
    ]


def test_probe_cpu_worker_exercises_sparse_reference_path() -> None:
    completed = subprocess.run(
        [sys.executable, "tools/probe_sparse_mps.py", "--worker-device", "cpu"],
        check=True,
        text=True,
        capture_output=True,
    )
    line = next(
        value
        for value in completed.stdout.splitlines()
        if value.startswith("PQ1_SPARSE_MPS_WORKER=")
    )
    payload = json.loads(line.split("=", 1)[1])

    assert payload["embedding"] == {"rows": 600, "sparse": True, "width": 12}
    assert payload["fallback_env"] == "0"
    assert [step["selected_rows"] for step in payload["steps"]] == [
        [2, 5, 19],
        [5, 7, 19],
    ]
    assert all(step["grad_is_coalesced"] for step in payload["steps"])
    assert all(step["untouched_rows_bit_identical"] for step in payload["steps"])
    assert payload["nonzero_clock_rows"] == [2, 5, 7, 19]


def test_wrapper_uses_port_adapter_not_direct_device_assumptions() -> None:
    source = Path(
        "src/tac/pr130_lift/pose/train_pose_carrier_full_resumable.py"
    ).read_text()

    assert "clear_device_cache(device)" in source
    assert "load_safetensors_cpu_then_move(" in source
    assert "prepare_row_local_step(" in source
    assert "load_file(" not in source
    assert "torch.cuda.empty_cache()" not in source
