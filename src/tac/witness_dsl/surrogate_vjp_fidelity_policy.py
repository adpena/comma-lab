# SPDX-License-Identifier: MIT
"""Typed research-only policy for the argmax-native surrogate-VJP metric.

The policy deliberately emits no trainer argv.  It exists so standalone custody checks can
authenticate the metric knobs and receipt paths without silently turning a diagnostic into a
live optimization actuator.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SURROGATE_VJP_POLICY_SCHEMA = "surrogate_vjp_fidelity_policy.v1"
_REQUIRED_BINDING_KEYS = frozenset(
    {
        "schema",
        "enabled",
        "research_only",
        "metric_mode",
        "low_margin_annulus",
        "importance_clip",
        "directional_probe_seed",
        "directional_probe_count",
        "anchor_k",
        "measurement_receipt",
        "functional_gate_receipt",
        "terminality_receipt",
    }
)


class SurrogateVJPFidelityPolicyError(ValueError):
    """Raised when the standalone policy binding is incomplete or unsafe."""


def _nullable_receipt(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise SurrogateVJPFidelityPolicyError(f"{field} must be explicit null or a non-empty path")
    return str(value)


@dataclass(frozen=True, slots=True)
class SurrogateVJPFidelityPolicy:
    """Closed set of research-only metric knobs and receipt custody references."""

    enabled: bool = False
    research_only: bool = True
    metric_mode: str = "winner_rival_fisher_directional"
    low_margin_annulus: float = 0.05
    importance_clip: float = 10.0
    directional_probe_seed: int = 0
    directional_probe_count: int = 16
    anchor_k: int = 120
    measurement_receipt: str | None = None
    functional_gate_receipt: str | None = None
    terminality_receipt: str | None = None

    def validate(self) -> None:
        if not self.research_only:
            raise SurrogateVJPFidelityPolicyError("surrogate-VJP policy must remain research_only")
        if not isinstance(self.enabled, bool):
            raise SurrogateVJPFidelityPolicyError("enabled must be bool")
        if self.metric_mode not in {
            "winner_rival_fisher_directional",
            "centered_logit_quotient",
        }:
            raise SurrogateVJPFidelityPolicyError(f"unsupported metric_mode={self.metric_mode!r}")
        for field, value in (
            ("low_margin_annulus", self.low_margin_annulus),
            ("importance_clip", self.importance_clip),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise SurrogateVJPFidelityPolicyError(f"{field} must be finite and positive")
        if not isinstance(self.directional_probe_seed, int):
            raise SurrogateVJPFidelityPolicyError("directional_probe_seed must be int")
        if not isinstance(self.directional_probe_count, int) or self.directional_probe_count <= 0:
            raise SurrogateVJPFidelityPolicyError("directional_probe_count must be positive int")
        if not isinstance(self.anchor_k, int) or self.anchor_k <= 0:
            raise SurrogateVJPFidelityPolicyError("anchor_k must be positive int")
        receipt_values = {
            "measurement_receipt": _nullable_receipt(
                self.measurement_receipt, field="measurement_receipt"
            ),
            "functional_gate_receipt": _nullable_receipt(
                self.functional_gate_receipt, field="functional_gate_receipt"
            ),
            "terminality_receipt": _nullable_receipt(
                self.terminality_receipt, field="terminality_receipt"
            ),
        }
        if self.enabled and any(value is None for value in receipt_values.values()):
            missing = sorted(name for name, value in receipt_values.items() if value is None)
            raise SurrogateVJPFidelityPolicyError(
                f"enabled policy requires all sealed receipts; explicit null remains unready: {missing}"
            )

    def to_binding(self) -> dict[str, Any]:
        self.validate()
        binding = {"schema": SURROGATE_VJP_POLICY_SCHEMA, **asdict(self)}
        canonical = json.dumps(binding, sort_keys=True, separators=(",", ":"))
        binding["binding_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
        return binding

    def to_trainer_overrides(self) -> dict[str, Any]:
        """Make the no-live-argv boundary executable and reviewable."""

        self.validate()
        return {}


def validate_surrogate_vjp_fidelity_binding(
    binding: Mapping[str, Any],
) -> SurrogateVJPFidelityPolicy:
    """Validate a standalone binding, distinguishing absent keys from explicit null.

    ``measurement_receipt`` is nullable while the policy is off, but the key itself is
    mandatory.  Indexing after an explicit membership check is intentional: ``binding.get``
    would collapse a malformed absent key and a valid explicit-null declaration.
    """

    keys = frozenset(binding)
    missing = sorted(_REQUIRED_BINDING_KEYS - keys)
    if missing:
        raise SurrogateVJPFidelityPolicyError(f"policy binding missing required keys: {missing}")
    allowed = _REQUIRED_BINDING_KEYS | {"binding_sha256"}
    unknown = sorted(keys - allowed)
    if unknown:
        raise SurrogateVJPFidelityPolicyError(f"policy binding has unknown keys: {unknown}")
    if binding["schema"] != SURROGATE_VJP_POLICY_SCHEMA:
        raise SurrogateVJPFidelityPolicyError("surrogate-VJP policy schema drift")
    policy = SurrogateVJPFidelityPolicy(
        enabled=binding["enabled"],
        research_only=binding["research_only"],
        metric_mode=binding["metric_mode"],
        low_margin_annulus=binding["low_margin_annulus"],
        importance_clip=binding["importance_clip"],
        directional_probe_seed=binding["directional_probe_seed"],
        directional_probe_count=binding["directional_probe_count"],
        anchor_k=binding["anchor_k"],
        measurement_receipt=_nullable_receipt(
            binding["measurement_receipt"], field="measurement_receipt"
        ),
        functional_gate_receipt=_nullable_receipt(
            binding["functional_gate_receipt"], field="functional_gate_receipt"
        ),
        terminality_receipt=_nullable_receipt(
            binding["terminality_receipt"], field="terminality_receipt"
        ),
    )
    policy.validate()
    if "binding_sha256" in binding:
        unhashed = {key: binding[key] for key in binding if key != "binding_sha256"}
        canonical = json.dumps(unhashed, sort_keys=True, separators=(",", ":"))
        expected = hashlib.sha256(canonical.encode()).hexdigest()
        if binding["binding_sha256"] != expected:
            raise SurrogateVJPFidelityPolicyError("surrogate-VJP policy binding hash mismatch")
    return policy


__all__ = [
    "SURROGATE_VJP_POLICY_SCHEMA",
    "SurrogateVJPFidelityPolicy",
    "SurrogateVJPFidelityPolicyError",
    "validate_surrogate_vjp_fidelity_binding",
]
