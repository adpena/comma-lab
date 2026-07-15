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


class _FakeCudaGraph:
    """Records replay() calls; the fake capture context runs fn eagerly."""

    def __init__(self):
        self.replay_calls = 0

    def replay(self):
        self.replay_calls += 1


class _FakeGraphContext:
    def __init__(self, graph):
        self.graph = graph

    def __enter__(self):
        return self.graph

    def __exit__(self, *exc):
        return False


def test_cuda_graph_capture_step_replays_and_counts_as_training_step(monkeypatch):
    """F8 (fresh-eyes 2026-07-15): torch.cuda.graph capture RECORDS kernels
    without executing them — the pre-fix code returned undefined static outputs
    from the capture call AND silently consumed the caller's real training step
    into the recording. The capture call must replay() immediately so it still
    performs its training step, and every subsequent call replays again."""
    graphs = []

    def fake_cuda_graph_cls():
        g = _FakeCudaGraph()
        graphs.append(g)
        return g

    monkeypatch.setattr(torch.cuda, "CUDAGraph", fake_cuda_graph_cls, raising=False)
    monkeypatch.setattr(
        torch.cuda, "graph", lambda g: _FakeGraphContext(g), raising=False
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None, raising=False)

    calls = []

    def step(x):
        calls.append(float(x[0]))
        return x * 2

    runner = CudaGraphForwardBackward(step, enabled=True, warmup_real_steps=1)
    # CPU tensors would normally fall back; force the CUDA-ready branch so the
    # capture/replay control flow is exercised at $0 with the fake graph.
    monkeypatch.setattr(runner, "_cuda_ready", lambda inputs: True)

    # Warmup step: eager, real.
    out0 = runner.run(torch.tensor([1.0]))
    assert float(out0[0]) == 2.0 and calls == [1.0]

    # Capture step: fn recorded once with the static copy of THIS call's input,
    # then the graph MUST replay immediately (the F8 fix) so the caller's step
    # still executes; the guard must be in a replayable state afterwards.
    out1 = runner.run(torch.tensor([5.0]))
    assert calls == [1.0, 5.0]  # capture consumed exactly this call's values
    assert len(graphs) == 1 and graphs[0].replay_calls == 1
    assert runner.receipt()["captured"]
    assert float(out1[0]) == 10.0  # defined because the fake context ran eagerly

    # Subsequent step: pure replay with copied inputs, no new fn invocation.
    runner.run(torch.tensor([7.0]))
    assert graphs[0].replay_calls == 2
    assert calls == [1.0, 5.0]  # no eager re-invocation after capture
    assert float(runner._static_inputs[0][0]) == 7.0  # inputs copied for replay


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
