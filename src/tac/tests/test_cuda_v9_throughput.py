import pytest

torch = pytest.importorskip("torch")

from tac.cuda_levelset_training import TorchExecutionPolicy
from tac.cuda_v9_throughput import CudaGraphForwardBackward, adopt_compiled_training_region


def _cpu_policy():
    return TorchExecutionPolicy(
        device_type="cpu",
        amp_dtype=None,
        grad_scaler=False,
        tf32=False,
        cudnn_benchmark=False,
        compile_mode=None,
        cuda_graphs=False,
        execution_label="eager_fallback",
    )


def test_compile_adoption_requires_functional_oracle():
    with pytest.raises(RuntimeError, match="functional oracle"):
        adopt_compiled_training_region(
            lambda x: x,
            _cpu_policy(),
            {"argmax_equal": True, "cosine_phi": 0.9996},
        )


def test_cpu_compile_receipt_is_honest_eager_fallback():
    fn, receipt = adopt_compiled_training_region(
        lambda x: x + 1,
        _cpu_policy(),
        {"argmax_equal": True, "cosine_phi": 1.0},
    )
    assert torch.equal(fn(torch.tensor(2)), torch.tensor(3))
    assert not receipt.adopted
    assert receipt.throughput == "UNMEASURED-pending-CUDA-dispatch"


def test_cuda_graph_runner_cpu_fallback_executes_every_real_step():
    calls = []

    def step(x):
        calls.append(float(x))
        return x.square()

    runner = CudaGraphForwardBackward(step, enabled=True, warmup_real_steps=2)
    assert float(runner.run(torch.tensor(2.0))) == 4.0
    assert float(runner.run(torch.tensor(3.0))) == 9.0
    assert calls == [2.0, 3.0]
    assert not runner.receipt()["captured"]
