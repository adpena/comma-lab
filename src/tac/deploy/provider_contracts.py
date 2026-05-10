"""Provider-agnostic deployment contracts and static guardrails.

This module is deliberately small: provider actuators own their platform API
calls, while this registry owns the cross-provider invariants that decide
whether a provider surface is safe to wire into score-moving work.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path


PROVIDER_NAMES: tuple[str, ...] = ("modal", "kaggle", "aws", "azure", "gcp")


@dataclass(frozen=True)
class ProviderDeployContract:
    """Static deployment contract shared by all cloud provider surfaces."""

    provider: str
    module: str
    canonical_entrypoints: tuple[str, ...]
    status: str
    plan_only_default: bool
    execution_flag: str | None
    requires_lane_claim_before_dispatch: bool
    terminal_claim_required: bool
    custody_manifest_required: bool
    exact_cuda_eval_supported: bool
    proxy_score_claim_allowed: bool = False
    mps_auth_eval_allowed: bool = False
    provider_proxy_promotes_score: bool = False
    setup_blockers: tuple[str, ...] = ()
    notes: str = ""

    @property
    def implemented(self) -> bool:
        return self.status == "implemented"

    @property
    def scaffold_only(self) -> bool:
        return self.status == "scaffold"

    def validate_static(self, repo_root: Path | None = None) -> list[str]:
        """Return static contract violations without touching provider APIs."""
        violations: list[str] = []
        if self.provider not in PROVIDER_NAMES:
            violations.append(f"{self.provider}: provider not in canonical registry")
        if self.status not in {"implemented", "scaffold"}:
            violations.append(f"{self.provider}: invalid status {self.status!r}")
        if not self.plan_only_default:
            violations.append(f"{self.provider}: dispatch path must be plan-only by default")
        if self.implemented and not self.execution_flag:
            violations.append(f"{self.provider}: implemented dispatch requires explicit execution flag")
        if not self.requires_lane_claim_before_dispatch:
            violations.append(f"{self.provider}: lane claim required before remote dispatch")
        if not self.terminal_claim_required:
            violations.append(f"{self.provider}: terminal lane claim row required after dispatch")
        if not self.custody_manifest_required:
            violations.append(f"{self.provider}: custody manifest required for remote artifacts")
        if self.proxy_score_claim_allowed:
            violations.append(f"{self.provider}: provider proxy score claims must stay forbidden")
        if self.provider_proxy_promotes_score:
            violations.append(f"{self.provider}: provider proxy must not promote/rank/kill")
        if self.mps_auth_eval_allowed:
            violations.append(f"{self.provider}: MPS auth eval must stay forbidden")
        if self.exact_cuda_eval_supported and "cuda" not in self.notes.lower():
            violations.append(f"{self.provider}: exact eval support must name CUDA in notes")
        if find_spec(self.module) is None:
            violations.append(f"{self.provider}: module is not importable: {self.module}")

        if repo_root is not None:
            for entrypoint in self.canonical_entrypoints:
                if entrypoint.startswith("tac."):
                    continue
                if not (repo_root / entrypoint).exists():
                    violations.append(f"{self.provider}: missing entrypoint {entrypoint}")
        return violations


PROVIDER_CONTRACTS: dict[str, ProviderDeployContract] = {
    "modal": ProviderDeployContract(
        provider="modal",
        module="tac.deploy.modal.runtime",
        canonical_entrypoints=(
            "experiments/modal_auth_eval.py",
            "src/tac/deploy/modal/modal_asymmetric_warp_deploy.py",
        ),
        status="implemented",
        plan_only_default=True,
        execution_flag="modal run",
        requires_lane_claim_before_dispatch=True,
        terminal_claim_required=True,
        custody_manifest_required=True,
        exact_cuda_eval_supported=True,
        setup_blockers=("modal login", "billing context", "CUDA image import probe"),
        notes="Modal can host claimed CUDA exact eval, but harvested artifacts still need adjudication.",
    ),
    "kaggle": ProviderDeployContract(
        provider="kaggle",
        module="tac.deploy.kaggle.runner",
        canonical_entrypoints=(
            "experiments/kaggle_asym_warp_launcher.py",
            "src/tac/deploy/kaggle/runner.py",
        ),
        status="implemented",
        plan_only_default=True,
        execution_flag="kaggle kernels push",
        requires_lane_claim_before_dispatch=True,
        terminal_claim_required=True,
        custody_manifest_required=True,
        exact_cuda_eval_supported=False,
        setup_blockers=("Kaggle API credentials", "tac wheel dataset", "GPU session quota"),
        notes="Kaggle is proxy/free-GPU capacity; it must not create score truth.",
    ),
    "aws": ProviderDeployContract(
        provider="aws",
        module="tac.deploy.aws.ec2_client",
        canonical_entrypoints=("src/tac/deploy/aws/ec2_client.py",),
        status="scaffold",
        plan_only_default=True,
        execution_flag=None,
        requires_lane_claim_before_dispatch=True,
        terminal_claim_required=True,
        custody_manifest_required=True,
        exact_cuda_eval_supported=True,
        setup_blockers=("boto3", "AWS credentials", "region-specific DLAMI", "SSH key/security group"),
        notes="AWS scaffold targets claimed CUDA hosts after EC2 lifecycle implementation lands.",
    ),
    "azure": ProviderDeployContract(
        provider="azure",
        module="tac.deploy.azure.azure_dispatch",
        canonical_entrypoints=("scripts/launch_lane_azure.py", "src/tac/deploy/azure/azure_dispatch.py"),
        status="implemented",
        plan_only_default=True,
        execution_flag="--no-dry-run",
        requires_lane_claim_before_dispatch=True,
        terminal_claim_required=True,
        custody_manifest_required=True,
        exact_cuda_eval_supported=True,
        setup_blockers=("az login", "quota/spot availability", "SSH public key", "lane tarball wiring"),
        notes="Azure can host claimed CUDA remote lanes; dry-run remains the default.",
    ),
    "gcp": ProviderDeployContract(
        provider="gcp",
        module="tac.deploy.gcp.gcp_dispatch",
        canonical_entrypoints=("src/tac/deploy/gcp/gcp_dispatch.py",),
        status="scaffold",
        plan_only_default=True,
        execution_flag=None,
        requires_lane_claim_before_dispatch=True,
        terminal_claim_required=True,
        custody_manifest_required=True,
        exact_cuda_eval_supported=True,
        setup_blockers=("gcloud auth", "GPU quota", "project/zone selection", "GCS harvest bucket"),
        notes="GCP scaffold targets claimed CUDA hosts after lifecycle implementation lands.",
    ),
}


def provider_contracts() -> dict[str, ProviderDeployContract]:
    """Return a copy of the canonical provider contract registry."""
    return dict(PROVIDER_CONTRACTS)


def validate_provider_contracts(repo_root: Path | None = None) -> list[str]:
    """Validate registry coverage and static provider-safety invariants."""
    violations: list[str] = []
    missing = sorted(set(PROVIDER_NAMES) - set(PROVIDER_CONTRACTS))
    extra = sorted(set(PROVIDER_CONTRACTS) - set(PROVIDER_NAMES))
    if missing:
        violations.append(f"missing provider contracts: {', '.join(missing)}")
    if extra:
        violations.append(f"unexpected provider contracts: {', '.join(extra)}")

    for provider in PROVIDER_NAMES:
        contract = PROVIDER_CONTRACTS.get(provider)
        if contract is None:
            continue
        violations.extend(contract.validate_static(repo_root=repo_root))
    return violations
