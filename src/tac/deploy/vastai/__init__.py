# SPDX-License-Identifier: MIT
"""Vast.ai deployment module for tac experiments.

Public API:
    VastClient      — SSH, rsync, instance lifecycle management
    BudgetTracker   — Cost tracking with hard caps and warnings
    ExperimentConfig — Re-exported from tac.deploy.base for convenience
"""
from tac.deploy.base import ExperimentConfig
from tac.deploy.vastai.budget import BudgetTracker
from tac.deploy.vastai.client import VastClient

__all__ = [
    "BudgetTracker",
    "ExperimentConfig",
    "VastClient",
]
