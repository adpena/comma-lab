# SPDX-License-Identifier: MIT
"""Azure GPU dispatch — VM-spot pattern (mirrors Vast.ai).

This package wires Azure as a 4th dispatch platform alongside Vast.ai +
Modal + Lightning.ai. The user has $200 free Azure credits per CLAUDE.md
"GPU budget and compute resources" section.

Architecture: provision spot VM → SSH in → run lane script (Pattern A
nohup) → harvest results → deprovision. This mirrors the Vast.ai pattern
because for single-instance lane scripts, lightweight VM-spot is a better
fit than the heavyweight Azure ML SDK.

Modules:
    azure_dispatch: provisioning, SSH, run_lane, harvest, deprovision

Usage from CLI:
    See ``scripts/launch_lane_azure.py`` for the dispatch wrapper analog
    to ``scripts/launch_lane_with_retry.py``.

User pre-action required: ``az login`` to authenticate before the first
dispatch. This module does NOT spawn VMs at import time and refuses to
run any Azure operation if ``az account list`` returns empty.
"""
