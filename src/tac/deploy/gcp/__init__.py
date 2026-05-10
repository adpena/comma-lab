"""GCP deployment scaffold.

The package exists so provider-agnostic deploy checks can reason about GCP
without importing ad-hoc experiment code or touching Google Cloud APIs.
"""
from __future__ import annotations

from tac.deploy.gcp.gcp_dispatch import GCPDispatchPlan, plan_gcp_dispatch

__all__ = ["GCPDispatchPlan", "plan_gcp_dispatch"]
