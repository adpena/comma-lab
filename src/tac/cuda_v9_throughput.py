# SPDX-License-Identifier: MIT
"""Backend-only Torch execution accelerators for V9 CGauge training.

Scientific configuration never enters this module.  Adoption is gated by the
portable forward oracle, while the operator-authorized training path may use
AMP, Inductor/Triton fusion, TF32, and a CUDA graph despite floating-point
reordering.  CUDA timing is intentionally absent: it must be measured on the
first authorized CUDA dispatch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

from tac.cuda_levelset_training import (
    NUMPY_FP32_PARITY_COSINE_BAR,
    CudaGraphRecaptureGuard,
    TorchExecutionPolicy,
)


@dataclass(frozen=True)
class CompiledRegionReceipt:
    adopted: bool
    engine: str
    mode: str | None
    functional_argmax_equal: bool
    functional_cosine_phi: float
    throughput: str = "UNMEASURED-pending-CUDA-dispatch"
    # F7 (fresh-eyes 2026-07-15) — honest receipt scope: the adoption probe that
    # feeds this receipt is an INSTANCE receipt (r3: feats[:64] + synthetic
    # rgb^2+phi^2 closure), not a full-config functional proof of every compiled
    # region. Upgrade path: a full-config probe on the real loss, nearly free
    # from the r6 harvest's own outputs.
    functional_probe_scope: str = (
        "INSTANCE receipt: 64-feat synthetic-loss closure probe; NOT a "
        "full-config functional proof — full-config real-loss probe owed"
    )
    adoption_rule: str = (
        "argmax_equal && cosine_phi>=0.9997 (NUMPY_FP32_PARITY_COSINE_BAR: "
        "CLAUDE.md deterministic-reproducibility item 3, numpy-fp32 parity bar)"
    )


def adopt_compiled_training_region(
    fn: Callable[..., Any],
    policy: TorchExecutionPolicy,
    functional_probe: Mapping[str, Any],
) -> tuple[Callable[..., Any], CompiledRegionReceipt]:
    """Compile the full forward/loss/backward region after functional adoption.

    ``fn`` may call ``backward``; AOTAutograd then compiles both the forward and
    backward graphs.  ``fullgraph=False`` keeps frozen third-party scorer ops
    correct when an unsupported operator needs a bounded graph break.
    """
    import torch

    argmax_equal = bool(functional_probe.get("argmax_equal", False))
    cosine_phi = float(functional_probe.get("cosine_phi", float("nan")))
    # F7 provenance: NUMPY_FP32_PARITY_COSINE_BAR is the CLAUDE.md
    # deterministic-reproducibility item (3) numpy-fp32 authority parity bar.
    functional_ok = argmax_equal and cosine_phi >= NUMPY_FP32_PARITY_COSINE_BAR
    if not functional_ok:
        raise RuntimeError(
            "compiled training refused: functional oracle requires "
            f"argmax_equal and cosine_phi>={NUMPY_FP32_PARITY_COSINE_BAR}, "
            f"got {dict(functional_probe)}"
        )
    if policy.device_type != "cuda" or policy.compile_mode is None:
        return fn, CompiledRegionReceipt(
            adopted=False,
            engine="eager",
            mode=None,
            functional_argmax_equal=argmax_equal,
            functional_cosine_phi=cosine_phi,
        )
    # torch.compile REFUSES mode= and options= together (measured on the H100
    # container's torch, 2026-07-15 r3 smoke, rc=1 @323.8s). Express the mode
    # via its documented option equivalents so the cudagraph toggles and the
    # mode live in ONE options dict. Wrap-time kwarg validation happens on any
    # device, so test_compile_kwargs_accepted_by_torch guards this at $0.
    mode_option_equivalents: dict[str, dict[str, Any]] = {
        "default": {},
        "reduce-overhead": {},
        "max-autotune": {"max_autotune": True},
    }
    mode = str(policy.compile_mode)
    if mode not in mode_option_equivalents:
        raise RuntimeError(
            f"compiled training refused: no options-equivalent registered for "
            f"torch.compile mode {mode!r}"
        )
    options: dict[str, Any] = {
        "triton.cudagraphs": bool(policy.cuda_graphs),
        "triton.cudagraph_trees": bool(policy.cuda_graphs),
        **mode_option_equivalents[mode],
    }
    compiled = torch.compile(
        fn,
        fullgraph=False,
        dynamic=False,
        options=options,
    )
    return compiled, CompiledRegionReceipt(
        adopted=True,
        engine="torch-inductor-triton-regional-megakernel",
        mode=str(policy.compile_mode),
        functional_argmax_equal=argmax_equal,
        functional_cosine_phi=cosine_phi,
    )


class CudaGraphForwardBackward:
    """Capture and replay a fixed-shape forward/loss/backward callable.

    The callable must zero persistent gradients and run backward, but must leave
    unscale/gradient clipping/optimizer.step to the caller.  Keeping optimizer
    actuation outside the graph supports the typed AdamW-to-Muon transition even
    though native Torch Muon is not graph-capturable.  Warmup calls are real
    training steps supplied by the caller; no unrecorded dummy update occurs.
    The capture call replays immediately after recording (F8, 2026-07-15), so
    it too performs exactly one real training step and returns defined outputs.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        enabled: bool,
        warmup_real_steps: int = 3,
    ) -> None:
        if int(warmup_real_steps) < 0:
            raise ValueError("warmup_real_steps must be non-negative")
        self.fn = fn
        self.enabled = bool(enabled)
        self.warmup_real_steps = int(warmup_real_steps)
        self.warmup_completed = 0
        self.guard = CudaGraphRecaptureGuard()
        self._graph = None
        self._static_inputs: tuple[Any, ...] | None = None
        self._static_outputs: Any = None
        self._capture_failure: str | None = None

    def _cuda_ready(self, inputs: tuple[Any, ...]) -> bool:
        import torch

        return (
            self.enabled
            and bool(inputs)
            and torch.cuda.is_available()
            and all(isinstance(x, torch.Tensor) and x.is_cuda for x in inputs)
        )

    @staticmethod
    def _copy_inputs_(static_inputs: tuple[Any, ...], inputs: tuple[Any, ...]) -> None:
        for static, current in zip(static_inputs, inputs, strict=True):
            if static.shape != current.shape or static.dtype != current.dtype:
                raise ValueError("CUDA graph input shape/dtype changed; invalidate and recapture")
            static.copy_(current)

    def invalidate_control_layout(self) -> None:
        self.guard.invalidate_control_layout()
        self._graph = None
        self._static_inputs = None
        self._static_outputs = None
        self.warmup_completed = 0

    def run(self, *inputs: Any) -> Any:
        import torch

        values = tuple(inputs)
        if not self._cuda_ready(values) or self._capture_failure is not None:
            return self.fn(*values)
        if self._graph is None and self.warmup_completed < self.warmup_real_steps:
            self.warmup_completed += 1
            return self.fn(*values)
        if self._graph is None:
            self._static_inputs = tuple(torch.empty_like(x).copy_(x) for x in values)
            graph = torch.cuda.CUDAGraph()
            try:
                torch.cuda.synchronize()
                with torch.cuda.graph(graph):
                    self._static_outputs = self.fn(*self._static_inputs)
            except Exception as exc:
                self._capture_failure = f"{type(exc).__name__}: {exc}"
                self._graph = None
                self._static_inputs = None
                self._static_outputs = None
                return self.fn(*values)
            self._graph = graph
            self.guard.mark_captured()
            # F8 (fresh-eyes 2026-07-15): torch.cuda.graph capture RECORDS the
            # kernels without executing them — at this point the static outputs
            # hold UNDEFINED values and the caller's real training step was
            # consumed into the recording. Replay immediately so this call still
            # performs its training step and returns defined outputs (the static
            # inputs already hold this call's `values` by construction).
            self._graph.replay()
            self.guard.mark_replayed()
            return self._static_outputs
        if not self.guard.may_replay():
            raise RuntimeError("stale CUDA graph requires explicit invalidation/recapture")
        assert self._static_inputs is not None
        self._copy_inputs_(self._static_inputs, values)
        self._graph.replay()
        self.guard.mark_replayed()
        return self._static_outputs

    def receipt(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "captured": self._graph is not None,
            "capture_failure": self._capture_failure,
            "warmup_real_steps": self.warmup_real_steps,
            "warmup_completed": self.warmup_completed,
            "guard": asdict(self.guard),
            "scope": "fixed-shape forward+loss+backward; optimizer actuation outside graph",
            "throughput": "UNMEASURED-pending-CUDA-dispatch",
        }


__all__ = [
    "CompiledRegionReceipt",
    "CudaGraphForwardBackward",
    "adopt_compiled_training_region",
]
