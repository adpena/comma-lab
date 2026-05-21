# SPDX-License-Identifier: MIT
"""MLX framework canonical integration scaffold for local substrate prototyping.

Per operator directive 2026-05-21 verbatim *"Let's make sure we are leveraging
local cpu and mps and metal and mlx as much as possible"*: this module is the
canonical MLX-framework-on-Apple-Silicon training scaffold for substrate
paradigm prototyping BEFORE paid Modal/Vast.ai/Lightning dispatch.

MLX (Apple's machine learning research framework) targets Apple Silicon's
unified memory + Metal GPU directly. On M5 Max + 128GB unified memory MLX
typically runs 2-3x faster than PyTorch MPS for small-to-medium models
(< 1B params) due to (a) zero CPU-GPU memory copy cost (unified memory),
(b) lazy-evaluation graph optimization, (c) Metal-native kernel selection.

Per CLAUDE.md non-negotiables PRESERVED:
- **MPS auth eval is NOISE** (Catalog #1): MLX inherits this property —
  MLX-derived scores are NEVER authoritative for promotion / falsification
  / strategy decisions. MLX is for paradigm prototyping + premise
  verification + smoke validation per Carmack MVP-first 5-step Step 1.
- **macOS-MLX is `[macOS-MLX research-signal]`** per Catalog #192 sister
  discipline at the MLX surface; non-promotable without paired Linux
  x86_64 + NVIDIA per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA".
- Every persisted MLX manifest row carries `score_claim=False`,
  `promotion_eligible=False`, `ready_for_exact_eval_dispatch=False`,
  `evidence_grade="macOS-MLX-research-signal"` per Catalog #287/#323
  canonical Provenance.

Canonical use cases (per Carmack MVP-first 5-step amendment proposal):

1. **FREE local MLX substrate prototype training** (NEW step between
   "FREE local macOS-CPU smoke" + "paid dispatch"): for any L1+ substrate
   in LOCAL_MLX_TRAINABLE class per :mod:`routability_audit`, train the
   first ~10-50 epochs on M5 Max MLX to verify (a) gradient flow,
   (b) loss decreases, (c) intermediate checkpoint can produce a valid
   archive, (d) inflate.py round-trips correctly. If smoke fails, the
   bug is structural and paid GPU dispatch would have wasted $2-15.

2. **Premise verification at recipe-exact fidelity** per Catalog #229
   premise-verification-before-edit discipline: load actual contest video
   pyav-decoded + actual archive grammar + actual scorer preprocess
   (canonical sister patterns from `tac.substrates._shared.score_aware_common`
   work on MLX with conversion shims).

3. **Cargo-cult-unwind iteration** per Carmack MVP-first 5-step Step 2
   (smoke MUST falsifiably challenge cargo-cult): when a CARGO-CULTED
   assumption is identified per Catalog #303, unwind on MLX first (free,
   fast iteration cycle) before paying for the verified-unwind dispatch.
   Empirical anchor: NSCS06 v6→v7 cargo-cult-unwind achieved 44%
   improvement in ONE iteration; if M5 Max MLX had been the iteration
   surface, the cycle would have been ~30 min vs hours.

NOT canonical use cases (explicitly forbidden):
- Auth-eval score claims (per Catalog #1 + Catalog #192 + this module's
  evidence_grade contract — MLX is for proxy curve discovery, never the
  authoritative axis)
- Promotion decisions / KILL/FALSIFY verdicts on a single MLX run
- Frontier-pointer updates (per CLAUDE.md "Frontier scores are
  pointer-only" Catalog #343 sister — only paired Linux x86_64 + NVIDIA
  qualifies)

Per CLAUDE.md Catalog #335 cathedral consumer canonical contract sister:
the MLX-training output should be routed through a sister cathedral
consumer (e.g. `mlx_prototype_consumer`; queued for sister-subagent
landing) that surfaces non-promotable `[macOS-MLX]`-tagged rankings to
the autopilot ranker per Catalog #341 sister routing pattern.

This module's API is intentionally minimal — operators / sister subagents
build substrate-specific training loops on top using the standard MLX
patterns. The shared utilities here cover the canonical apparatus
discipline (Provenance emission, manifest persistence, fail-closed
score-claim) so substrate-specific code doesn't reinvent it.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX

__all__ = [
    "MLX_AVAILABLE",
    "MLXTrainingResult",
    "is_mlx_available",
    "build_mlx_training_result",
    "mlx_smoke_test",
]


def is_mlx_available() -> bool:
    """Check whether MLX framework is importable + Metal device available.

    Returns False on non-Apple-Silicon hosts, on Apple Silicon without MLX
    installed, or when Metal compute is unavailable (rare; would indicate
    a system issue). Callers must check this before invoking MLX-routed
    helpers to avoid ImportError on Linux dispatchers that share this
    module.
    """

    if importlib.util.find_spec("mlx") is None:
        return False
    if importlib.util.find_spec("mlx.core") is None:
        return False
    try:
        import mlx.core as mx

        return bool(mx.metal.is_available())
    except (ImportError, AttributeError, RuntimeError):
        return False


MLX_AVAILABLE = is_mlx_available()


@dataclass(frozen=True, slots=True)
class MLXTrainingResult:
    """Typed result of an MLX local training run.

    Carries canonical Provenance triple per Catalog #287/#323 so every
    persisted row makes the non-promotable axis explicit.
    """

    substrate_id: str
    run_id: str
    epochs_completed: int
    final_proxy_loss: float
    wall_seconds: float
    archive_bytes_estimate: int | None
    # Canonical Provenance per Catalog #287/#323/#192/#1.
    evidence_grade: str = EVIDENCE_GRADE_MLX
    evidence_tag: str = EVIDENCE_TAG_MLX
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False
    # Non-promotable blockers per Catalog #1/#192/#317 sister discipline.
    blockers: tuple[str, ...] = field(
        default=(
            "macos_mlx_research_signal_not_score_evidence",
            "not_cuda_auth_eval",
            "not_a_11_contest_compliant_axis",
            "requires_paired_linux_x86_64_nvidia_for_promotion",
            "requires_paired_contest_cpu_gha_linux_x86_64_for_cpu_axis_promotion",
        )
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "substrate_id": self.substrate_id,
            "run_id": self.run_id,
            "epochs_completed": self.epochs_completed,
            "final_proxy_loss": self.final_proxy_loss,
            "wall_seconds": self.wall_seconds,
            "archive_bytes_estimate": self.archive_bytes_estimate,
            "evidence_grade": self.evidence_grade,
            "evidence_tag": self.evidence_tag,
            "score_claim": self.score_claim,
            "promotion_eligible": self.promotion_eligible,
            "ready_for_exact_eval_dispatch": self.ready_for_exact_eval_dispatch,
            "blockers": list(self.blockers),
        }


def build_mlx_training_result(
    *,
    substrate_id: str,
    run_id: str,
    epochs_completed: int,
    final_proxy_loss: float,
    wall_seconds: float,
    archive_bytes_estimate: int | None = None,
) -> MLXTrainingResult:
    """Construct a canonical MLX training result with non-promotable markers.

    Sister of :func:`tac.optimization.macos_cpu_advisory_signal.build_macos_cpu_advisory_signal_manifest`
    and :func:`tac.optimization.mps_research_signal.build_mps_research_signal_manifest`.
    Every result is non-promotable by construction; callers cannot override
    the score_claim / promotion_eligible / ready_for_exact_eval_dispatch
    fields without re-deriving via a different evidence_grade per
    Catalog #287/#323/#192.
    """

    if not substrate_id or not isinstance(substrate_id, str):
        raise ValueError("substrate_id must be a non-empty string")
    if not run_id or not isinstance(run_id, str):
        raise ValueError("run_id must be a non-empty string")
    if epochs_completed < 0:
        raise ValueError("epochs_completed must be >= 0")
    if wall_seconds < 0:
        raise ValueError("wall_seconds must be >= 0")

    return MLXTrainingResult(
        substrate_id=substrate_id,
        run_id=run_id,
        epochs_completed=epochs_completed,
        final_proxy_loss=float(final_proxy_loss),
        wall_seconds=float(wall_seconds),
        archive_bytes_estimate=archive_bytes_estimate,
    )


def mlx_smoke_test() -> dict[str, Any]:
    """One-shot MLX availability + Metal device smoke test.

    Returns structured device_info dict for operator-facing diagnostics +
    routability classification. Safe to call on non-Apple-Silicon hosts
    (returns ``available=False``).
    """

    if not MLX_AVAILABLE:
        return {
            "available": False,
            "reason": "MLX framework not installed or Metal device unavailable",
            "remediation": "Install MLX via `pip install mlx` on Apple Silicon host",
        }

    import mlx.core as mx

    try:
        # New API per MLX 0.x: device_info() at module level (mx.metal.device_info
        # is deprecated).
        info = mx.metal.device_info()
    except Exception:
        try:
            info = mx.device_info()
        except Exception as exc:
            return {
                "available": True,
                "reason": f"device_info introspection failed: {exc}",
                "remediation": "MLX functional but introspection API drift; report",
            }

    return {
        "available": True,
        "device_name": info.get("device_name", "unknown"),
        "memory_size_bytes": info.get("memory_size", 0),
        "max_recommended_working_set_size_bytes": info.get(
            "max_recommended_working_set_size", 0
        ),
        "architecture": info.get("architecture", "unknown"),
        "default_device": str(mx.default_device()),
    }
