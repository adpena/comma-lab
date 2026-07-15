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


def test_source_requests_inductor_cuda_graph_trees_for_compiled_regions():
    import inspect

    source = inspect.getsource(adopt_compiled_training_region)
    assert '"triton.cudagraphs": bool(policy.cuda_graphs)' in source
    assert '"triton.cudagraph_trees": bool(policy.cuda_graphs)' in source


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


def _cuda_shaped_policy():
    return TorchExecutionPolicy(
        device_type="cuda",
        amp_dtype="bfloat16",
        grad_scaler=False,
        tf32=True,
        cudnn_benchmark=True,
        compile_mode="max-autotune",
        cuda_graphs=True,
        execution_label="megakernel_candidate",
    )


def test_compile_kwargs_accepted_by_torch():
    """$0 wrap-time guard for the 2026-07-15 r3 H100 rc=1 (@323.8s, ~$0.45):
    torch.compile refuses mode= and options= together, and it validates kwargs
    at WRAP time on any device — so this failure class never needed a paid
    dispatch to surface. The prior source-token test verified constants, not
    behavior (NO-FAKE class #2); this one actually wraps through the real
    adoption path with a CUDA-shaped policy."""
    fn, receipt = adopt_compiled_training_region(
        lambda x: x * 2,
        _cuda_shaped_policy(),
        {"argmax_equal": True, "cosine_phi": 1.0},
    )
    assert receipt.adopted
    assert receipt.mode == "max-autotune"
    assert callable(fn)


def test_compile_refuses_unregistered_mode():
    policy = TorchExecutionPolicy(
        device_type="cuda",
        amp_dtype="bfloat16",
        grad_scaler=False,
        tf32=True,
        cudnn_benchmark=True,
        compile_mode="not-a-mode",
        cuda_graphs=True,
        execution_label="megakernel_candidate",
    )
    with pytest.raises(RuntimeError, match="no options-equivalent"):
        adopt_compiled_training_region(
            lambda x: x,
            policy,
            {"argmax_equal": True, "cosine_phi": 1.0},
        )
