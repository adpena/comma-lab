"""AWS EC2 spot instance lifecycle management (stub).

This module will manage EC2 spot instances for tac experiments.
It follows the same patterns as :mod:`tac.deploy.vastai.client`:

- Context manager for automatic instance termination
- SSH/rsync for code upload and result download
- Budget tracking via :class:`tac.deploy.base.BudgetState`

Pricing reference (us-east-1, April 2026):
    g4dn.xlarge (T4 16GB):   ~$0.53 on-demand, ~$0.16-0.22 spot
    g5.xlarge   (A10G 24GB): ~$1.01 on-demand, ~$0.30-0.40 spot
"""
from __future__ import annotations

from dataclasses import dataclass

from tac.deploy.base import ExperimentConfig, InstanceSpec


# ── AWS-specific constants ────────────────────────────────────────────────────

# Instance type mapping from generic GPU names to AWS instance types
GPU_TO_INSTANCE_TYPE: dict[str, str] = {
    "T4": "g4dn.xlarge",
    "A10G": "g5.xlarge",
}

# Spot pricing estimates (USD/hr, us-east-1)
SPOT_PRICE_ESTIMATES: dict[str, float] = {
    "g4dn.xlarge": 0.19,   # T4, typical spot
    "g5.xlarge": 0.35,     # A10G, typical spot
}

# On-demand pricing (USD/hr, us-east-1)
ON_DEMAND_PRICES: dict[str, float] = {
    "g4dn.xlarge": 0.526,
    "g5.xlarge": 1.006,
}

# Default AMI: Deep Learning AMI (Ubuntu) with PyTorch pre-installed
DEFAULT_AMI_ID = "ami-0placeholder"  # TODO: resolve actual AMI ID at launch time

DEFAULT_REGION = "us-east-1"
DEFAULT_SECURITY_GROUP = "tac-experiments"
DEFAULT_KEY_NAME = "tac-deploy"


@dataclass(frozen=True)
class EC2InstanceSpec(InstanceSpec):
    """AWS-specific instance specification.

    Extends the base InstanceSpec with AWS-specific fields.
    """

    region: str = DEFAULT_REGION
    """AWS region for the instance."""

    use_spot: bool = True
    """Whether to use spot instances (cheaper but can be interrupted)."""

    max_spot_price: float | None = None
    """Maximum spot price in USD/hr. None means use on-demand price as cap."""

    ami_id: str = DEFAULT_AMI_ID
    """AMI ID for the instance."""

    key_name: str = DEFAULT_KEY_NAME
    """EC2 key pair name for SSH access."""

    security_group: str = DEFAULT_SECURITY_GROUP
    """Security group name (must allow SSH inbound)."""

    @property
    def instance_type(self) -> str:
        """Map gpu_type to AWS instance type."""
        return GPU_TO_INSTANCE_TYPE.get(self.gpu_type, "g4dn.xlarge")

    @property
    def estimated_hourly_cost(self) -> float:
        """Estimated hourly cost based on spot/on-demand."""
        itype = self.instance_type
        if self.use_spot:
            return SPOT_PRICE_ESTIMATES.get(itype, 0.20)
        return ON_DEMAND_PRICES.get(itype, 0.53)


class EC2Client:
    """EC2 spot instance lifecycle manager (stub).

    Will use boto3 for:
    - ``request_spot_instances`` / ``run_instances``
    - ``describe_instances`` for status polling
    - ``terminate_instances`` for cleanup
    - SSH/rsync for code transfer (same pattern as VastClient)

    Not yet implemented. Requires:
    - boto3 dependency
    - AWS credentials configured (``~/.aws/credentials`` or env vars)
    - VPC + security group + key pair setup
    """

    def __init__(self) -> None:
        raise NotImplementedError(
            "EC2Client is a stub. AWS deployment is not yet implemented. "
            "See tac.deploy.aws.ec2_client for the planned interface."
        )
