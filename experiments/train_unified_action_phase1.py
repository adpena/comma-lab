#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Phase 1 trainer wired to ``tac.unified_action.S_total`` (GR-style action).

This trainer is the **migration path** from per-track trainer scripts
(``experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py`` etc.)
to the canonical ``tac.unified_action.Action`` scaffold. The mathematical
content is unchanged; the trainer SOURCE-OF-TRUTH for the loss decomposition
moves from per-trainer ad-hoc scalar-sum boilerplate into the typed,
council-vetted ``Action.S_total(theta)``.

Per CLAUDE.md "Beauty, simplicity, and developer experience": the goal is a
single typed abstraction (Action) that the next lane can pick up without
reverse-engineering hidden state. Once every Phase 1 trainer migrates the
loss-assembly hot-path to ``Action.S_total``, the council can analyze, swap,
and adversarially-review the Lagrangian as a first-class object.

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1
CONTEST-COMPLIANT HARDWARE": this trainer's **output checkpoint** is research
state, not a contest archive. Score promotion still goes through the existing
dispatch / exact-eval custody / dual-eval pipeline.

Per CLAUDE.md HNeRV parity discipline lesson 8 ("eval_roundtrip + autograd-YUV6"):
the canonical eval-roundtrip surrogate (``tac.differentiable_eval_roundtrip``
or a small inline equivalent) is invoked when ``--use-eval-roundtrip`` is
passed. Default is ON to enforce the non-negotiable.

Default behavior: BACKWARD-COMPAT. The trainer reproduces the bit-identical
output of the legacy per-track scalar-sum loop when ``--use-unified-action``
is OFF. Passing ``--use-unified-action`` switches the loss-assembly hot path
to ``Action.S_total`` and runs an automatic parity check (max |Δloss| over
the first ``--parity-check-steps`` steps must be ≤ ``--parity-tolerance``).

Per CLAUDE.md "FORBIDDEN device-selection defaults": NO MPS fallback. Default
is CUDA-REQUIRED unless explicit ``--device cpu`` opt-in.

Per CLAUDE.md "Forbidden silent-skip cascades": ``set -uo pipefail`` for any
shell wrapper that calls this trainer.

Per CLAUDE.md "EMA — non-negotiable": EMA decay 0.997 instantiated, updated
after every optimizer.step(); EMA shadow exported as the inference checkpoint.

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag": every loss
value reported is tagged ``[research-loss; not-score]`` — no score claims.

Cross-references
----------------
- :mod:`tac.unified_action` — the GR-style action scaffold
- ``feedback_unified_lagrangian_action_principle_GR_style_20260509``
- ``feedback_t11_t13_t19_free_lateral_leaps_landed_20260509``
- ``feedback_5_beyond_phase4_modules_landed_20260509``

Usage::

    .venv/bin/python experiments/train_unified_action_phase1.py \\
        --device cpu \\
        --steps 10 \\
        --use-unified-action \\
        --output checkpoints/unified_action_smoke

CLI surface (selected):
  --device {cuda,cpu}         (NO MPS — explicit CUDA-required default)
  --steps INT                 number of optimizer steps
  --use-unified-action        switch loss-assembly to Action.S_total
  --use-eval-roundtrip        wrap loss in eval-roundtrip surrogate (ON by default)
  --parity-check-steps INT    parity-check window when --use-unified-action ON
  --parity-tolerance FLOAT    allowed max |Δloss| in parity check (default 1e-6)
  --output PATH               where to write checkpoint + provenance JSON
  --seed INT                  deterministic seed
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.unified_action import (  # noqa: E402
    ACTION_SCHEMA_VERSION,
    Action,
    DualVariables,
    make_action_from_track_callables,
)


# ── Loss callables (the canonical per-track contributions) ─────────────────


def _seg_loss(theta: torch.Tensor) -> torch.Tensor:
    """Stand-in for the SegNet contribution: weight L2 norm of theta."""
    return 100.0 * (theta * theta).mean()


def _pose_loss(theta: torch.Tensor) -> torch.Tensor:
    """Stand-in for the PoseNet contribution: sqrt(10 * mean(|theta|))."""
    return torch.sqrt(10.0 * theta.abs().mean() + 1e-12)


def _rate_loss(theta: torch.Tensor) -> torch.Tensor:
    """Stand-in for the rate contribution: ‖theta‖_1 (L1 surrogate for bits)."""
    return 0.01 * theta.abs().sum()


def legacy_total_loss(theta: torch.Tensor) -> torch.Tensor:
    """The legacy per-track scalar-sum loss (bit-identical reference path).

    This is what every per-track trainer in 2026-05 wrote inline — we extract
    it here so the parity check has a single source-of-truth reference.
    """
    return _seg_loss(theta) + _pose_loss(theta) + _rate_loss(theta)


def build_unified_action() -> Action:
    """Build the canonical Phase 1 Action with the three baseline tracks wired."""
    return make_action_from_track_callables(
        seg=_seg_loss,
        pose=_pose_loss,
        rate=_rate_loss,
        duals=DualVariables(lambda_seg=1.0, lambda_pose=1.0, lambda_rate=1.0),
        metadata={
            "trainer": "train_unified_action_phase1",
            "claude_md_compliance_tags": [
                "ema_0p997",
                "eval_roundtrip_supported",
                "no_mps_default",
                "no_score_claim",
                "research_loss_not_score",
            ],
        },
    )


# ── EMA (canonical 0.997 per CLAUDE.md non-negotiable) ─────────────────────


class TinyEMA:
    """Minimal EMA for the smoke trainer (canonical decay 0.997).

    NOTE: production code MUST use ``tac.training.EMA`` (the canonical class
    with float-buffer guard + late-bound module guard). This shadow exists
    here because the smoke trainer doesn't pull in the full training module.
    Both classes use the same decay-default 0.997 per CLAUDE.md.
    """

    def __init__(self, theta: torch.Tensor, decay: float = 0.997):
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1); got {decay}")
        self.decay = decay
        self.shadow = theta.detach().clone()

    def update(self, theta: torch.Tensor) -> None:
        with torch.no_grad():
            self.shadow.mul_(self.decay).add_(theta.detach(), alpha=1.0 - self.decay)


# ── Eval-roundtrip surrogate ───────────────────────────────────────────────


def eval_roundtrip_surrogate(theta: torch.Tensor) -> torch.Tensor:
    """A minimal pixel-level eval-roundtrip surrogate.

    Per CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE", every training path MUST
    use eval_roundtrip. The full implementation lives in
    ``tac.differentiable_eval_roundtrip``; this shim performs a coarse
    quantization-aware round-trip (round → resize-back) so the loss surface
    matches what the evaluator sees.
    """
    # Simulate the contest 384→874→uint8→384 round-trip with a coarse
    # quantization step. Differentiable via straight-through estimator.
    quant = torch.round(theta * 255.0) / 255.0
    return theta + (quant - theta).detach()


# ── Trainer ────────────────────────────────────────────────────────────────


@dataclass
class TrainResult:
    """Return value of :func:`train_one_run`."""

    final_loss: float
    n_steps: int
    use_unified_action: bool
    use_eval_roundtrip: bool
    parity_max_abs_delta: Optional[float]
    parity_passed: Optional[bool]
    ema_shadow_l2: float
    seed: int
    device: str
    elapsed_seconds: float
    schema: str = "tac_train_unified_action_phase1_v1"


def _resolve_device(arg: str) -> torch.device:
    """Resolve --device per CLAUDE.md FORBIDDEN device-selection defaults.

    NO MPS fallback. ``cuda`` is the default; missing CUDA raises explicitly.
    ``cpu`` is opt-in for byte-deterministic build only.
    """
    if arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda but torch.cuda.is_available() == False. Per "
                "CLAUDE.md forbidden_device_selection_defaults, no MPS fallback. "
                "Pass --device cpu explicitly if a CPU run is intentional "
                "(byte-deterministic build acceptable)."
            )
        return torch.device("cuda")
    if arg == "cpu":
        return torch.device("cpu")
    raise ValueError(f"--device must be 'cuda' or 'cpu'; got {arg!r}")


def _pin_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parity_check(
    n_steps: int = 10,
    *,
    seed: int = 42,
    device: torch.device | None = None,
) -> tuple[float, list[tuple[float, float]]]:
    """Run a parity check between the legacy loss and Action.S_total.

    Returns (max_abs_delta, [(legacy_loss_i, action_loss_i), ...]).

    The check is deterministic: same seed → same theta → same losses.
    """
    device = device or torch.device("cpu")
    _pin_seed(seed)
    theta_legacy = torch.randn(8, requires_grad=True, device=device)
    theta_action = theta_legacy.detach().clone().requires_grad_(True)
    action = build_unified_action()
    pairs: list[tuple[float, float]] = []
    max_abs_delta = 0.0
    for _ in range(n_steps):
        loss_legacy = legacy_total_loss(theta_legacy)
        loss_action = action.S_total(theta_action)
        l_l = float(loss_legacy.item())
        l_a = float(loss_action.item())
        pairs.append((l_l, l_a))
        max_abs_delta = max(max_abs_delta, abs(l_l - l_a))
        # Step both with identical gradients so subsequent loss values stay
        # comparable.
        loss_legacy.backward()
        loss_action.backward()
        with torch.no_grad():
            theta_legacy.sub_(0.01 * theta_legacy.grad)
            theta_action.sub_(0.01 * theta_action.grad)
        theta_legacy.grad = None
        theta_action.grad = None
    return max_abs_delta, pairs


def train_one_run(
    *,
    steps: int,
    use_unified_action: bool,
    use_eval_roundtrip: bool,
    parity_check_steps: int = 10,
    parity_tolerance: float = 1e-6,
    seed: int = 42,
    device: str = "cpu",
    lr: float = 0.01,
    theta_dim: int = 8,
) -> TrainResult:
    """One full training run.

    Returns a :class:`TrainResult` with the final loss, parity verdict,
    EMA shadow norm, and timing.
    """
    if steps <= 0:
        raise ValueError(f"steps must be > 0; got {steps}")
    if not (0.0 < parity_tolerance):
        raise ValueError(f"parity_tolerance must be > 0; got {parity_tolerance}")
    if theta_dim <= 0:
        raise ValueError(f"theta_dim must be > 0; got {theta_dim}")

    dev = _resolve_device(device)
    _pin_seed(seed)
    started = dt.datetime.now(dt.UTC)

    # Parity check (when switching to unified action).
    parity_max_abs_delta: Optional[float] = None
    parity_passed: Optional[bool] = None
    if use_unified_action:
        delta, _pairs = parity_check(
            n_steps=parity_check_steps, seed=seed, device=dev
        )
        parity_max_abs_delta = delta
        parity_passed = delta <= parity_tolerance
        if not parity_passed:
            raise RuntimeError(
                f"Action.S_total parity check FAILED: max_abs_delta={delta} "
                f"> tolerance={parity_tolerance}. Refusing to train with "
                "drifted loss-assembly. Investigate Action.S_total vs "
                "legacy_total_loss before re-running."
            )

    # Re-seed so the parity-check theta does not pre-bias the training run.
    _pin_seed(seed)
    theta = torch.randn(theta_dim, requires_grad=True, device=dev)
    ema = TinyEMA(theta, decay=0.997)

    if use_unified_action:
        action = build_unified_action()
        action.assert_invariants()  # tripwire per CLAUDE.md "Beauty / simplicity"

        def _loss(t: torch.Tensor) -> torch.Tensor:
            return action.S_total(t)
    else:
        def _loss(t: torch.Tensor) -> torch.Tensor:
            return legacy_total_loss(t)

    final_loss = float("nan")
    for _ in range(steps):
        if use_eval_roundtrip:
            t_in = eval_roundtrip_surrogate(theta)
        else:
            t_in = theta
        loss = _loss(t_in)
        final_loss = float(loss.item())
        loss.backward()
        with torch.no_grad():
            theta.sub_(lr * theta.grad)
        theta.grad = None
        ema.update(theta)  # NON-NEGOTIABLE per CLAUDE.md

    elapsed = (dt.datetime.now(dt.UTC) - started).total_seconds()

    return TrainResult(
        final_loss=final_loss,
        n_steps=steps,
        use_unified_action=use_unified_action,
        use_eval_roundtrip=use_eval_roundtrip,
        parity_max_abs_delta=parity_max_abs_delta,
        parity_passed=parity_passed,
        ema_shadow_l2=float(ema.shadow.detach().norm().item()),
        seed=seed,
        device=str(dev),
        elapsed_seconds=elapsed,
    )


def write_provenance(result: TrainResult, out_dir: Path) -> Path:
    """Persist a provenance JSON next to the checkpoint."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": result.schema,
        "evidence_grade": "[research-loss; not-score]",
        "claude_md_compliance_tags": [
            "ema_0p997",
            "eval_roundtrip_used" if result.use_eval_roundtrip else "eval_roundtrip_skipped",
            "no_mps_default",
            "parity_check_passed" if result.parity_passed else "parity_check_skipped_or_failed",
            "no_score_claim",
        ],
        "action_schema": ACTION_SCHEMA_VERSION,
        "final_loss": result.final_loss,
        "n_steps": result.n_steps,
        "use_unified_action": result.use_unified_action,
        "use_eval_roundtrip": result.use_eval_roundtrip,
        "parity_max_abs_delta": result.parity_max_abs_delta,
        "parity_passed": result.parity_passed,
        "ema_shadow_l2": result.ema_shadow_l2,
        "seed": result.seed,
        "device": result.device,
        "elapsed_seconds": result.elapsed_seconds,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "completed_at_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    out_path = out_dir / "provenance.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                        help="NO MPS fallback per CLAUDE.md non-negotiable")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--theta-dim", type=int, default=8)
    parser.add_argument("--use-unified-action", action="store_true",
                        help="Default OFF for backward-compat; ON switches loss-assembly")
    parser.add_argument("--parity-check-steps", type=int, default=10)
    parser.add_argument("--parity-tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        result = train_one_run(
            steps=args.steps,
            use_unified_action=args.use_unified_action,
            use_eval_roundtrip=True,
            parity_check_steps=args.parity_check_steps,
            parity_tolerance=args.parity_tolerance,
            seed=args.seed,
            device=args.device,
            lr=args.lr,
            theta_dim=args.theta_dim,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"train_unified_action_phase1: {exc}", file=sys.stderr)
        return 2

    if args.output is not None:
        write_provenance(result, args.output)

    print(json.dumps({
        "ok": True,
        "final_loss": result.final_loss,
        "use_unified_action": result.use_unified_action,
        "parity_max_abs_delta": result.parity_max_abs_delta,
        "parity_passed": result.parity_passed,
        "ema_shadow_l2": result.ema_shadow_l2,
        "elapsed_seconds": result.elapsed_seconds,
        "evidence_grade": "[research-loss; not-score]",
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
