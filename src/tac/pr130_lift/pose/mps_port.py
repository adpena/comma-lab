"""Device-port adapters for the lifted PR130 pose-carrier trainer.

The borrowed sparse embedding plus ``RowLocalSparseAdam`` mechanism is the
reference on every device.  It is admitted on MPS only under the Torch runtime
covered by the pinned native-sparse receipt.  ``RowLocalDenseAdam`` remains an
explicit portability adapter; selecting MPS never activates it implicitly.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors.torch import load_file

REFERENCE_SPARSE_MODE = "reference-sparse"
DENSE_ADAPTER_MODE = "dense-adapter"
ROW_LOCAL_MODES = (REFERENCE_SPARSE_MODE, DENSE_ADAPTER_MODE)
PINNED_MPS_TORCH_VERSION = "2.10.0"


def torch_public_version() -> str:
    """Return the public Torch version without a local build suffix."""

    return torch.__version__.split("+", 1)[0]


def assert_reference_runtime_compatible(device: torch.device) -> None:
    """Fail closed when MPS reference mode leaves its validated runtime."""

    if device.type != "mps":
        return
    actual = torch_public_version()
    if actual != PINNED_MPS_TORCH_VERSION:
        raise RuntimeError(
            "reference sparse MPS mode requires the receipt-pinned Torch "
            f"{PINNED_MPS_TORCH_VERSION}; found {actual}. "
            "Use the pinned runtime or explicitly opt into --row-local-mode "
            f"{DENSE_ADAPTER_MODE}."
        )


def clear_device_cache(device: torch.device) -> None:
    """Clear the selected accelerator cache, or do nothing for CPU."""

    if device.type == "cpu":
        return
    if device.type == "mps":
        torch.mps.empty_cache()
        return
    if device.type == "cuda":
        torch.cuda.empty_cache()
        return
    raise ValueError(f"unsupported pose-carrier device type: {device.type!r}")


def load_safetensors_cpu_then_move(
    module: torch.nn.Module,
    path: str | Path,
    device: torch.device,
) -> torch.nn.Module:
    """Load a module's safetensors state on CPU before device placement."""

    tensors = [*module.parameters(), *module.buffers()]
    non_cpu = sorted({tensor.device.type for tensor in tensors} - {"cpu"})
    if non_cpu:
        raise ValueError(
            "CPU-first safetensors loading requires a CPU module; "
            f"found parameter devices {non_cpu}"
        )
    state_dict = load_file(str(path), device="cpu")
    module.load_state_dict(state_dict)
    return module.to(device)


def _unique_rows(
    row_ids: torch.Tensor,
    parameter: torch.Tensor,
) -> torch.Tensor:
    if row_ids.ndim != 1:
        raise ValueError("row ids must be one-dimensional")
    rows_cpu = torch.unique(
        row_ids.detach().to(device="cpu", dtype=torch.long), sorted=True
    )
    if rows_cpu.numel() == 0:
        raise ValueError("at least one coefficient row must be selected")
    if int(rows_cpu[0]) < 0 or int(rows_cpu[-1]) >= parameter.shape[0]:
        raise IndexError("selected coefficient row is out of range")
    return rows_cpu.to(parameter.device)


def _dense_gradient_touched_undeclared_rows(
    gradient: torch.Tensor,
    rows: torch.Tensor,
) -> bool:
    """Validate the tiny 600-row table on CPU without adding an MPS op family."""

    gradient_cpu = gradient.detach().to("cpu")
    rows_cpu = rows.detach().to("cpu")
    mask = torch.ones(gradient_cpu.shape[0], dtype=torch.bool)
    mask.index_fill_(0, rows_cpu, False)
    return bool(torch.count_nonzero(gradient_cpu[mask]).item())


class RowLocalDenseAdam(torch.optim.Optimizer):
    """Dense-gradient adapter preserving PR130's row-local Adam mechanism.

    The caller must declare the rows for every step.  The declaration is
    consumed before the update, so a stale row set cannot affect a later step.
    Optimizer state uses the same keys and shapes as ``RowLocalSparseAdam``.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps})
        self._selected_rows: dict[torch.Tensor, torch.Tensor] = {}

    def select_rows(self, parameter: torch.Tensor, row_ids: torch.Tensor) -> torch.Tensor:
        """Declare and return the unique sorted rows for the next step."""

        if parameter in self._selected_rows:
            raise RuntimeError("coefficient rows were already declared for this step")
        rows = _unique_rows(row_ids, parameter)
        self._selected_rows[parameter] = rows
        return rows

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                rows = self._selected_rows.pop(parameter, None)
                if parameter.grad is None:
                    if rows is not None:
                        raise RuntimeError("selected coefficient rows have no gradient")
                    continue
                if rows is None:
                    raise RuntimeError("dense row-local Adam requires rows for every step")
                if parameter.grad.is_sparse:
                    raise TypeError("RowLocalDenseAdam requires a dense gradient")

                if _dense_gradient_touched_undeclared_rows(parameter.grad, rows):
                    raise RuntimeError("dense coefficient gradient touched undeclared rows")
                values = parameter.grad.index_select(0, rows)
                state = self.state[parameter]
                if not state:
                    state["row_step"] = torch.zeros(
                        parameter.shape[0], dtype=torch.int64, device=parameter.device
                    )
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                row_step = state["row_step"].index_select(0, rows).add_(1)
                exp_avg = state["exp_avg"].index_select(0, rows)
                exp_avg_sq = state["exp_avg_sq"].index_select(0, rows)
                exp_avg.mul_(beta1).add_(values, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    values, values, value=1.0 - beta2
                )
                state["row_step"].index_copy_(0, rows, row_step)
                state["exp_avg"].index_copy_(0, rows, exp_avg)
                state["exp_avg_sq"].index_copy_(0, rows, exp_avg_sq)

                step_float = row_step.to(values.dtype)
                bias_correction1 = 1.0 - beta1**step_float
                bias_correction2 = 1.0 - beta2**step_float
                denominator = (
                    exp_avg_sq.sqrt() / bias_correction2.sqrt().unsqueeze(1)
                ).add_(group["eps"])
                update = exp_avg / bias_correction1.unsqueeze(1) / denominator
                parameter.index_add_(0, rows, update, alpha=-group["lr"])
        if self._selected_rows:
            raise RuntimeError("declared rows were not consumed by an optimizer parameter")
        return loss


def build_row_local_coefficients(
    *,
    num_embeddings: int,
    embedding_dim: int,
    device: torch.device,
    lr: float,
    sparse_optimizer_type: type[torch.optim.Optimizer],
    mode: str = REFERENCE_SPARSE_MODE,
) -> tuple[torch.nn.Embedding, torch.optim.Optimizer]:
    """Build the reference sparse path or the explicitly selected adapter."""

    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported pose-carrier device type: {device.type!r}")
    if mode not in ROW_LOCAL_MODES:
        raise ValueError(f"unsupported row-local optimizer mode: {mode!r}")
    use_sparse = mode == REFERENCE_SPARSE_MODE
    if use_sparse:
        assert_reference_runtime_compatible(device)
    coefficients = torch.nn.Embedding(
        num_embeddings, embedding_dim, sparse=use_sparse
    ).to(device)
    if use_sparse:
        optimizer = sparse_optimizer_type([coefficients.weight], lr=lr)
    else:
        optimizer = RowLocalDenseAdam([coefficients.weight], lr=lr)
    return coefficients, optimizer


def prepare_row_local_step(
    optimizer: torch.optim.Optimizer,
    parameter: torch.Tensor,
    row_ids: torch.Tensor,
    max_norm: float,
) -> torch.Tensor:
    """Validate selected rows and clip the active coefficient gradient."""

    if parameter.grad is None:
        raise RuntimeError("coefficient parameter has no gradient")
    expected_rows = _unique_rows(row_ids, parameter)
    if parameter.grad.is_sparse:
        if isinstance(optimizer, RowLocalDenseAdam):
            raise TypeError("dense row-local optimizer received a sparse gradient")
        gradient = parameter.grad.coalesce()
        rows = gradient.indices()[0]
        if not torch.equal(rows, expected_rows):
            raise RuntimeError(
                "sparse coefficient gradient rows do not match the selected batch rows"
            )
        values = gradient.values()
        norm = values.norm()
        if torch.isfinite(norm) and norm > max_norm:
            values.mul_(max_norm / norm)
        parameter.grad = gradient
        return rows

    if not isinstance(optimizer, RowLocalDenseAdam):
        raise TypeError("borrowed sparse optimizer received a dense gradient")
    rows = optimizer.select_rows(parameter, row_ids)
    if _dense_gradient_touched_undeclared_rows(parameter.grad, rows):
        raise RuntimeError("dense coefficient gradient touched undeclared rows")
    values = parameter.grad.index_select(0, rows)
    norm = values.norm()
    if torch.isfinite(norm) and norm > max_norm:
        clipped = values * (max_norm / norm)
        parameter.grad.index_copy_(0, rows, clipped)
    return rows
