"""Explicit SIREN/INR dispatch-contract taxonomy.

The SIREN family can enter the contest packet through three materially
different routes. Keeping those routes named prevents a residual sidecar,
domain-prior hybrid, or pure substrate replacement from being dispatched under
the wrong archive/eval contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SirenDispatchContractId = Literal[
    "naked_siren_replacement",
    "siren_residual_on_hnerv_a1",
    "hybrid_siren_domain_prior",
]

NAKED_SIREN_REPLACEMENT: SirenDispatchContractId = "naked_siren_replacement"
SIREN_RESIDUAL_ON_HNERV_A1: SirenDispatchContractId = "siren_residual_on_hnerv_a1"
HYBRID_SIREN_DOMAIN_PRIOR: SirenDispatchContractId = "hybrid_siren_domain_prior"


@dataclass(frozen=True)
class SirenDispatchContract:
    """Dispatch contract metadata recorded in recipes, archives, and manifests."""

    contract_id: SirenDispatchContractId
    summary: str
    archive_role: str
    hnerv_a1_relationship: str
    builder_surface: str
    train_substrate_siren_supported: bool
    required_inputs: tuple[str, ...]


SIREN_DISPATCH_CONTRACTS: dict[SirenDispatchContractId, SirenDispatchContract] = {
    NAKED_SIREN_REPLACEMENT: SirenDispatchContract(
        contract_id=NAKED_SIREN_REPLACEMENT,
        summary="SIREN/INR is the full RGB renderer substrate.",
        archive_role="replaces_hnerv_a1_substrate",
        hnerv_a1_relationship=(
            "No HNeRV/A1 base archive is consumed; SRV1 0.bin is the scored "
            "renderer payload."
        ),
        builder_surface="experiments/train_substrate_siren.py",
        train_substrate_siren_supported=True,
        required_inputs=("upstream/videos/0.mkv",),
    ),
    SIREN_RESIDUAL_ON_HNERV_A1: SirenDispatchContract(
        contract_id=SIREN_RESIDUAL_ON_HNERV_A1,
        summary="SIREN/INR carries residual atoms over a frozen HNeRV/A1 base.",
        archive_role="residual_sidecar_on_hnerv_a1_base",
        hnerv_a1_relationship=(
            "A byte-verified HNeRV/A1 archive remains the base payload; the "
            "SIREN residual payload must prove old/new archive SHA custody and "
            "runtime consumption."
        ),
        builder_surface="future SIREN residual materializer over HNeRV/A1 custody",
        train_substrate_siren_supported=False,
        required_inputs=("base_hnerv_a1_archive.zip", "base_runtime_tree_sha256"),
    ),
    HYBRID_SIREN_DOMAIN_PRIOR: SirenDispatchContract(
        contract_id=HYBRID_SIREN_DOMAIN_PRIOR,
        summary="SIREN/INR renderer plus explicit domain-prior payload.",
        archive_role="hybrid_replacement_with_domain_prior",
        hnerv_a1_relationship=(
            "May replace HNeRV/A1 or branch from it, but the prior payload must "
            "be declared in the parser manifest and scored as one packet."
        ),
        builder_surface="future hybrid SIREN+domain-prior packet builder",
        train_substrate_siren_supported=False,
        required_inputs=("domain_prior_manifest.json",),
    ),
}


def normalize_siren_dispatch_contract(value: str) -> SirenDispatchContractId:
    """Return a canonical contract id or raise ``ValueError``."""

    normalized = value.strip().lower().replace("-", "_")
    if normalized == "siren_replacement":
        normalized = NAKED_SIREN_REPLACEMENT
    if normalized not in SIREN_DISPATCH_CONTRACTS:
        valid = ", ".join(sorted(SIREN_DISPATCH_CONTRACTS))
        raise ValueError(f"unknown SIREN dispatch contract {value!r}; valid: {valid}")
    return normalized  # type: ignore[return-value]


def require_train_substrate_siren_contract(value: str) -> SirenDispatchContract:
    """Fail closed when this trainer is asked to build a non-naked contract."""

    contract_id = normalize_siren_dispatch_contract(value)
    contract = SIREN_DISPATCH_CONTRACTS[contract_id]
    if not contract.train_substrate_siren_supported:
        raise NotImplementedError(
            "experiments/train_substrate_siren.py only builds the naked SIREN "
            f"replacement archive contract ({NAKED_SIREN_REPLACEMENT}); "
            f"requested {contract_id} would otherwise produce a wrong archive. "
            f"Use/land the byte-closed builder surface: {contract.builder_surface}."
        )
    return contract


def siren_dispatch_contract_manifest() -> dict[str, dict[str, object]]:
    """Return JSON-serializable contract metadata."""

    return {
        key: {
            "summary": value.summary,
            "archive_role": value.archive_role,
            "hnerv_a1_relationship": value.hnerv_a1_relationship,
            "builder_surface": value.builder_surface,
            "train_substrate_siren_supported": value.train_substrate_siren_supported,
            "required_inputs": list(value.required_inputs),
        }
        for key, value in SIREN_DISPATCH_CONTRACTS.items()
    }


__all__ = [
    "HYBRID_SIREN_DOMAIN_PRIOR",
    "NAKED_SIREN_REPLACEMENT",
    "SIREN_DISPATCH_CONTRACTS",
    "SIREN_RESIDUAL_ON_HNERV_A1",
    "SirenDispatchContract",
    "SirenDispatchContractId",
    "normalize_siren_dispatch_contract",
    "require_train_substrate_siren_contract",
    "siren_dispatch_contract_manifest",
]
