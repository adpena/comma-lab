# SPDX-License-Identifier: MIT
"""Portable optimizer (Adam) with MLX + PyTorch sister implementations.

Per OVERNIGHT-WW: minimal Adam optimizer wrapper that operates on
backend-native parameter dicts. Selfcomp grayscale_lut uses Adam at lr=1e-3
per the existing PyTorch trainer; the MLX variant must match.

For backend swapping the optimizer state is intentionally NOT exported via
the canonical weight-export pipeline — only model weights are exported. A
fresh optimizer (in either backend) starts at step 0 with zero
first/second moments. This is the canonical pattern because MLX-to-PyTorch
weight export targets EVAL-TIME deployment, not mid-training continuation.
"""

from __future__ import annotations

from typing import Any

from tac.portable_primitives.backend import Backend, resolve_backend

__all__ = [
    "PortableAdam",
]


class PortableAdam:
    """Canonical Adam optimizer.

    Constructor: ``PortableAdam(parameters, lr=1e-3, backend=...)``
    Step: ``optimizer.step(gradients)`` where ``gradients`` is a dict
    matching the parameter dict structure.

    Both backends use the standard Adam update rule:
        m_t = beta1 * m_{t-1} + (1-beta1) * g_t
        v_t = beta2 * v_{t-1} + (1-beta2) * g_t^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(
        self,
        parameters: dict[str, Any],
        *,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        backend: Backend | str,
    ) -> None:
        self.parameters = parameters
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.backend = resolve_backend(backend)
        self.step_count = 0

        # Initialize moment buffers per parameter.
        if self.backend is Backend.MLX:
            import mlx.core as mx

            self._m = {k: mx.zeros_like(p) for k, p in parameters.items()}
            self._v = {k: mx.zeros_like(p) for k, p in parameters.items()}
        else:
            import torch

            self._m = {k: torch.zeros_like(p) for k, p in parameters.items()}
            self._v = {k: torch.zeros_like(p) for k, p in parameters.items()}

    def step(self, gradients: dict[str, Any]) -> dict[str, Any]:
        """Apply one Adam step. Returns the updated parameter dict.

        Both MLX and PyTorch backends compute the update in fp32; the result
        is byte-stable when fed the same gradients across backends (within
        ε per the test contract).
        """
        self.step_count += 1
        t = self.step_count
        bc1 = 1.0 - (self.beta1 ** t)
        bc2 = 1.0 - (self.beta2 ** t)

        if self.backend is Backend.MLX:
            import mlx.core as mx

            updated: dict[str, Any] = {}
            for name, param in self.parameters.items():
                if name not in gradients:
                    updated[name] = param
                    continue
                g = gradients[name]
                m = self.beta1 * self._m[name] + (1.0 - self.beta1) * g
                v = self.beta2 * self._v[name] + (1.0 - self.beta2) * (g * g)
                self._m[name] = m
                self._v[name] = v
                m_hat = m / bc1
                v_hat = v / bc2
                updated[name] = param - self.lr * m_hat / (mx.sqrt(v_hat) + self.eps)
            self.parameters = updated
            return updated

        import torch

        updated: dict[str, Any] = {}
        for name, param in self.parameters.items():
            if name not in gradients:
                updated[name] = param
                continue
            g = gradients[name]
            m = self.beta1 * self._m[name] + (1.0 - self.beta1) * g
            v = self.beta2 * self._v[name] + (1.0 - self.beta2) * (g * g)
            self._m[name] = m
            self._v[name] = v
            m_hat = m / bc1
            v_hat = v / bc2
            updated[name] = param - self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        self.parameters = updated
        return updated
