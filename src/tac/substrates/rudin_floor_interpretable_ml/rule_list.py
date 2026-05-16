# SPDX-License-Identifier: MIT
"""Minimal falling-rule-list primitives for the Rudin floor scaffold."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

RGB = tuple[int, int, int]


@dataclass(frozen=True)
class RudinFallingRule:
    """One first-match-wins rule with a compact predicate string."""

    predicate: str
    action_rgb: RGB
    slim_coefficients: tuple[int, ...] = ()

    def matches(self, features: Mapping[str, object]) -> bool:
        """Return whether this rule matches a simple feature dictionary."""

        predicate = self.predicate.strip()
        if predicate in {"*", "always", "true"}:
            return True
        if "==" in predicate:
            key, value = predicate.split("==", 1)
        elif "=" in predicate:
            key, value = predicate.split("=", 1)
        else:
            return False
        return str(features.get(key.strip())) == value.strip()

    def to_json_obj(self) -> dict[str, object]:
        """Serialize to a deterministic JSON-compatible object."""

        return {
            "predicate": self.predicate,
            "action_rgb": list(self.action_rgb),
            "slim_coefficients": list(self.slim_coefficients),
        }

    @classmethod
    def from_json_obj(cls, obj: Mapping[str, object]) -> RudinFallingRule:
        """Deserialize from :meth:`to_json_obj` output."""

        rgb_raw = obj.get("action_rgb")
        if not isinstance(rgb_raw, list) or len(rgb_raw) != 3:
            raise ValueError("RudinFallingRule action_rgb must be a 3-item list")
        coeffs_raw = obj.get("slim_coefficients", [])
        if not isinstance(coeffs_raw, list):
            raise ValueError("RudinFallingRule slim_coefficients must be a list")
        return cls(
            predicate=str(obj.get("predicate", "always")),
            action_rgb=tuple(int(v) for v in rgb_raw),  # type: ignore[arg-type]
            slim_coefficients=tuple(int(v) for v in coeffs_raw),
        )


@dataclass(frozen=True)
class RudinRuleList:
    """First-match-wins falling rule list."""

    rules: tuple[RudinFallingRule, ...]
    default_rgb: RGB = (0, 0, 0)

    def evaluate(self, features: Mapping[str, object] | None = None) -> RGB:
        """Evaluate the first matching rule and return an RGB action."""

        feature_map = features or {}
        for rule in self.rules:
            if rule.matches(feature_map):
                return rule.action_rgb
        return self.default_rgb

    def to_json_bytes(self) -> bytes:
        """Serialize deterministically for RDIF v1 archives."""

        payload = {
            "default_rgb": list(self.default_rgb),
            "rules": [rule.to_json_obj() for rule in self.rules],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )

    @classmethod
    def from_json_bytes(cls, blob: bytes) -> RudinRuleList:
        """Parse deterministic RDIF rule-list bytes."""

        obj = json.loads(blob.decode("utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("RudinRuleList payload must be a JSON object")
        default_raw = obj.get("default_rgb", [0, 0, 0])
        if not isinstance(default_raw, list) or len(default_raw) != 3:
            raise ValueError("RudinRuleList default_rgb must be a 3-item list")
        rules_raw = obj.get("rules", [])
        if not isinstance(rules_raw, list):
            raise ValueError("RudinRuleList rules must be a list")
        return cls(
            rules=tuple(
                RudinFallingRule.from_json_obj(rule)
                for rule in rules_raw
                if isinstance(rule, dict)
            ),
            default_rgb=tuple(int(v) for v in default_raw),  # type: ignore[arg-type]
        )


__all__ = ["RGB", "RudinFallingRule", "RudinRuleList"]
