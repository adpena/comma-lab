# SPDX-License-Identifier: MIT
"""Shared deployment primitives for all cloud platforms.

Defines ExperimentConfig and other dataclasses that are platform-agnostic.
Importable by Modal, Vast.ai, AWS, and any future deploy modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    """Platform-agnostic experiment configuration.

    Describes what to run and what resources it needs, without specifying
    how any particular platform provisions those resources.
    """

    name: str
    """Unique experiment identifier (e.g., 'tto_v1')."""

    script: str
    """Script path relative to repo root (e.g., 'experiments/renderer_tto.py')."""

    args: list[str] = field(default_factory=list)
    """CLI arguments passed to the script."""

    needs_upstream: bool = True
    """Whether the experiment requires the upstream scorer repo."""

    needs_checkpoint: str | None = None
    """Checkpoint filename required by the experiment (or None)."""

    timeout_hours: float = 4.0
    """Maximum wall-clock hours before auto-kill."""

    gpu_type: str = "T4"
    """Requested GPU type. Platforms map this to their own instance types."""

    estimated_cost_per_hour: float = 0.59
    """Default cost estimate in USD/hr. Platforms override with actual pricing."""

    @property
    def timeout_seconds(self) -> int:
        """Timeout converted to seconds for subprocess/timeout commands."""
        return int(self.timeout_hours * 3600)

    @property
    def estimated_total_cost(self) -> float:
        """Estimated total cost for the full timeout duration."""
        return self.estimated_cost_per_hour * self.timeout_hours


@dataclass(frozen=True)
class InstanceSpec:
    """Desired instance specification for provisioning.

    Platform-specific clients translate this into their own API calls.
    """

    gpu_type: str = "RTX 4090"
    """GPU model name."""

    num_gpus: int = 1
    """Number of GPUs."""

    disk_gb: float = 60.0
    """Minimum disk space in GB.

    2026-05-01 (Bug Class #6): bumped from 40 → 60. Multi-candidate eval
    chains (e.g. 6 archive eval) need ~5 GB uv-torch wheels + 6 × 3.6 GB
    inflated frames = 27 GB just for the work. 40 GB left no margin for
    OS + container overhead and crashed mid-chain. 60 GB is the safe floor.
    Reference: feedback_loop_session_permanent_bug_class_extinction_20260501.md.
    """

    min_reliability: float = 0.95
    """Minimum reliability score (0.0-1.0). Vast.ai-specific but harmless elsewhere."""

    min_download_mbps: float = 200.0
    """Minimum download bandwidth in Mbps."""

    docker_image: str = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
    """Docker image to use."""


@dataclass
class BudgetState:
    """Serializable budget tracking state.

    Shared across platforms. Each platform writes to its own budget file.
    """

    total_spent: float = 0.0
    """Cumulative USD spent."""

    sessions: list[dict] = field(default_factory=list)
    """List of spending events with timestamp, amount, description."""

    @classmethod
    def from_dict(cls, data: dict) -> BudgetState:
        """Deserialize from a JSON-compatible dict."""
        return cls(
            total_spent=data.get("total_spent", 0.0),
            sessions=data.get("sessions", []),
        )

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "total_spent": self.total_spent,
            "sessions": self.sessions,
        }


# ── Repo layout constants ────────────────────────────────────────────────────

def repo_root() -> Path:
    """Locate the repository root by searching upward for pyproject.toml."""
    candidate = Path(__file__).resolve().parent
    for _ in range(10):
        if (candidate / "pyproject.toml").exists():
            return candidate
        candidate = candidate.parent
    raise RuntimeError("Cannot locate repository root (no pyproject.toml found)")
