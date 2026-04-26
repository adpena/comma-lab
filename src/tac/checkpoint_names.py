"""Canonical checkpoint name registry.

ONE source of truth for which checkpoint filenames the deploy/inflate paths
will look for, in priority order. All consumers (scripts/deploy_vastai.py,
scripts/remote_train_bootstrap.sh, preflight) must reference this list.

Adding a new producer that emits a different filename? Add the pattern here
AND verify it shows up in deploy_canonical_checkpoint_names() output.

Canonical naming guidance:
  - Profile-specific: ``renderer_<profile>_best_fp32.pt`` (preferred for
    train_renderer.py outputs — embeds the profile name).
  - Stage-numbered: ``distill_phase<N>_best.pt`` (legacy, train_distill.py).
  - Generic best: ``qat_best_float.pt`` / ``distill_latest.pt``.

Why this exists: 2026-04-26 SHIRAZ + DEN deploys both crashed in deploy
Stage 4 because the training script emitted ``renderer_den_best_fp32.pt``
and the probe list only had ``distill_*``. We wasted a full DEN training
run before the compress step could probe. Centralising kills that mode.
"""
from __future__ import annotations


def canonical_checkpoint_names(profile: str | None = None) -> tuple[str, ...]:
    """Return the priority-ordered checkpoint name list to probe.

    The first existing file at any consumer's checkpoint directory wins.

    Args:
        profile: profile name. If given, profile-specific patterns are
            included. If None, only profile-agnostic patterns.

    Returns:
        Tuple of filenames (no path), priority order.
    """
    base: list[str] = []
    if profile:
        # train_renderer.py canonical output (most specific — wins).
        base.append(f"renderer_{profile}_best_fp32.pt")
        base.append(f"renderer_{profile}_best.pt")
    # Legacy + general patterns from older training scripts.
    base.extend([
        "distill_phase3_best.pt",
        "distill_phase2_best.pt",
        "qat_best_float.pt",
        "distill_latest.pt",
    ])
    return tuple(base)


# Producer → primary output name. Used by preflight to check that every
# training entry-point's emitted filename appears in canonical_checkpoint_names.
PRODUCER_OUTPUTS: dict[str, str] = {
    "src/tac/experiments/train_renderer.py": "renderer_<profile>_best_fp32.pt",
    "experiments/train_distill.py": "distill_phase3_best.pt",
    "src/tac/experiments/qat_finetune.py": "qat_best_float.pt",
}
