"""AWS-specific experiment config overrides (stub).

Reuses the base ExperimentConfig from tac.deploy.base with AWS pricing.
The same experiments from the Vast.ai registry can be adapted by
overriding ``gpu_type`` and ``estimated_cost_per_hour``.
"""
from __future__ import annotations

from tac.deploy.base import ExperimentConfig
from tac.deploy.aws.ec2_client import SPOT_PRICE_ESTIMATES


def adapt_for_aws(
    config: ExperimentConfig,
    use_spot: bool = True,
) -> ExperimentConfig:
    """Create an AWS-adapted copy of an ExperimentConfig.

    Remaps gpu_type to T4 (g4dn.xlarge) and adjusts cost estimates
    for AWS spot or on-demand pricing.

    Parameters
    ----------
    config:
        Base experiment config (typically from the Vast.ai registry).
    use_spot:
        If True, use spot pricing estimates. Otherwise, on-demand.
    """
    # AWS T4 instances are g4dn.xlarge
    aws_gpu = "T4"
    instance_type = "g4dn.xlarge"

    if use_spot:
        cost = SPOT_PRICE_ESTIMATES.get(instance_type, 0.19)
    else:
        from tac.deploy.aws.ec2_client import ON_DEMAND_PRICES
        cost = ON_DEMAND_PRICES.get(instance_type, 0.53)

    return ExperimentConfig(
        name=config.name,
        script=config.script,
        args=list(config.args),
        needs_upstream=config.needs_upstream,
        needs_checkpoint=config.needs_checkpoint,
        timeout_hours=config.timeout_hours,
        gpu_type=aws_gpu,
        estimated_cost_per_hour=cost,
    )
