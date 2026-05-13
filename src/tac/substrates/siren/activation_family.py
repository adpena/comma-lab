"""Typed activation-family contract for SIREN/INR substrate variants.

The full SIREN substrate keeps one archive/trainer contract while allowing
small activation-family probes. These modes are not score claims and do not
imply that FINER, WIRE, or BACON have been faithfully reimplemented as complete
paper architectures. They are byte-closed INR activation modes: the selected
family is serialized in SRV1 metadata and consumed by inflate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

ActivationFamilyId = Literal["siren", "finer", "wire", "bacon"]

DEFAULT_ACTIVATION_FAMILY: ActivationFamilyId = "siren"
ACTIVATION_FAMILY_IDS: tuple[ActivationFamilyId, ...] = (
    "siren",
    "finer",
    "wire",
    "bacon",
)


@dataclass(frozen=True)
class ActivationFamilySpec:
    """Human-readable activation-family metadata for manifests."""

    activation_id: ActivationFamilyId
    summary: str
    archive_note: str
    full_paper_architecture: bool


SIREN_ACTIVATION_FAMILIES: dict[ActivationFamilyId, ActivationFamilySpec] = {
    "siren": ActivationFamilySpec(
        activation_id="siren",
        summary="Canonical SIREN sine activation with Sitzmann omega schedule.",
        archive_note="Default; preserves naked_siren_replacement behavior.",
        full_paper_architecture=True,
    ),
    "finer": ActivationFamilySpec(
        activation_id="finer",
        summary="FINER-style variable-periodic sine activation.",
        archive_note="Drop-in activation probe under the same SRV1 payload contract.",
        full_paper_architecture=False,
    ),
    "wire": ActivationFamilySpec(
        activation_id="wire",
        summary="WIRE-style Gabor/windowed periodic activation.",
        archive_note="Drop-in activation probe; window scale is carried in metadata.",
        full_paper_architecture=False,
    ),
    "bacon": ActivationFamilySpec(
        activation_id="bacon",
        summary="BACON-style band-limited sine schedule.",
        archive_note=(
            "Band-limited activation probe; not BACON's full multi-output "
            "network architecture."
        ),
        full_paper_architecture=False,
    ),
}


def normalize_activation_family(value: str) -> ActivationFamilyId:
    """Normalize an activation-family id or raise ``ValueError``."""

    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "naked_siren": "siren",
        "base_siren": "siren",
        "canonical_siren": "siren",
        "finer_style": "finer",
        "wire_style": "wire",
        "bacon_style": "bacon",
        "bandlimited": "bacon",
        "band_limited": "bacon",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SIREN_ACTIVATION_FAMILIES:
        valid = ", ".join(ACTIVATION_FAMILY_IDS)
        raise ValueError(f"unknown SIREN activation family {value!r}; valid: {valid}")
    return normalized  # type: ignore[return-value]


def effective_layer_omega(
    *,
    activation_family: ActivationFamilyId,
    base_omega: float,
    layer_index: int,
    bacon_bandwidth_scale: float,
) -> float:
    """Return the activation omega used by a layer."""

    if activation_family == "bacon":
        return float(base_omega) * float(bacon_bandwidth_scale) * float(layer_index + 1)
    return float(base_omega)


def apply_activation_family(
    x: torch.Tensor,
    *,
    activation_family: ActivationFamilyId,
    omega: float,
    wire_scale: float,
) -> torch.Tensor:
    """Apply a supported INR activation family to preactivation values."""

    if activation_family == "siren":
        return torch.sin(omega * x)
    if activation_family == "finer":
        return torch.sin(omega * (torch.abs(x) + 1.0) * x)
    if activation_family == "wire":
        window = torch.exp(-0.5 * torch.square(wire_scale * x))
        return torch.sin(omega * x) * window
    if activation_family == "bacon":
        return torch.sin(omega * x)
    raise ValueError(f"unsupported activation family {activation_family!r}")


def activation_family_manifest() -> dict[str, dict[str, object]]:
    """Return JSON-serializable activation-family metadata."""

    return {
        key: {
            "summary": spec.summary,
            "archive_note": spec.archive_note,
            "full_paper_architecture": spec.full_paper_architecture,
        }
        for key, spec in SIREN_ACTIVATION_FAMILIES.items()
    }


__all__ = [
    "ACTIVATION_FAMILY_IDS",
    "DEFAULT_ACTIVATION_FAMILY",
    "SIREN_ACTIVATION_FAMILIES",
    "ActivationFamilyId",
    "ActivationFamilySpec",
    "activation_family_manifest",
    "apply_activation_family",
    "effective_layer_omega",
    "normalize_activation_family",
]
