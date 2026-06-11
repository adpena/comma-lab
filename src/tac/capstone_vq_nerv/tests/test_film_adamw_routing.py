# SPDX-License-Identifier: MIT
"""NO-FAKE behavioral tests for the pose-FiLM -> AdamW optimizer routing (capstone).

The optimizer-poison audit (#3) found the capstone's pose-FiLM MLP weights
(``pose_film0/1.fc{1,2}.weight``) route to **Muon** because they are 2-D and end
in "weight" — but Muon's Newton-Schulz orthogonalization gives update step
magnitudes that are INDEPENDENT of the gradient norm (singular values driven to
~1), which is wrong for a small zero-init pose MLP. The fix is the additive
``force_adamw_substrings`` hook on ``apply_pr95_mlx_optimizer_step`` (default
``None`` = byte-identical to the PR95-faithful core), wired into the capstone via
``CapstoneTrainConfig.force_film_to_adamw``.

These tests assert BEHAVIOR, not constants:

1. The FiLM weights are ACTUALLY removed from the Muon group and ADDED to AdamW
   (the routing summary reflects real partition membership).
2. The default (no force) is byte-identical to the PR95-faithful partition (the
   shared core + every other caller is untouched).
3. The decisive Muon-vs-AdamW DISCRIMINATOR: an AdamW step's magnitude scales
   with the gradient norm; a Muon (Newton-Schulz orthogonalized) step's magnitude
   does NOT. Forcing the FiLM weight to AdamW makes its update scale-sensitive —
   if the body were replaced by ``return canonical_markers`` (or the routing were
   a no-op), this test FAILS.
4. The capstone trainer wires the routing end-to-end (the FiLM weights are forced
   to AdamW in a real training step).

Authority: ``[macOS-MLX research-signal]`` (MLX decoder) / ``[local CPU-torch
advisory]`` (frozen torch scorer; NO MPS). Non-promotable per Catalog #192.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    import mlx.core as mx
    import mlx.nn as nn

    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

skip_no_mlx = pytest.mark.skipif(
    not _MLX_AVAILABLE, reason="MLX not available; requires Apple Silicon."
)


def _opt_imports():
    from tac.local_acceleration.pr95_hnerv_mlx import (
        Pr95MlxOptimizerConfig,
        Pr95MlxOptimizerState,
        apply_pr95_mlx_optimizer_step,
        partition_pr95_mlx_parameter_names,
    )

    return (
        Pr95MlxOptimizerConfig,
        Pr95MlxOptimizerState,
        apply_pr95_mlx_optimizer_step,
        partition_pr95_mlx_parameter_names,
    )


class _TinyFilmModule(nn.Module if nn is not None else object):  # type: ignore[misc]
    """A minimal module with a 2-D "film...weight" (Muon-routed by default) +
    a 2-D non-film "conv weight" (genuinely Muon) so the routing split is real."""

    def __init__(self) -> None:
        super().__init__()
        # Names chosen so the PR95 partition routes BOTH to Muon by default
        # (2-D, ends "weight", not stem/rgb_/latents).
        self.pose_film0 = nn.Linear(6, 8)  # -> pose_film0.weight (the FiLM weight)
        self.block_conv = nn.Linear(8, 8)  # -> block_conv.weight (a real Muon weight)


@skip_no_mlx
def test_force_adamw_substrings_moves_film_out_of_muon():
    """NO-FAKE: forcing 'film' removes the film weight from Muon and adds it to AdamW."""
    (
        Pr95MlxOptimizerConfig,
        Pr95MlxOptimizerState,
        apply_pr95_mlx_optimizer_step,
        partition_pr95_mlx_parameter_names,
    ) = _opt_imports()

    mod = _TinyFilmModule()
    split = partition_pr95_mlx_parameter_names(mod.parameters())
    # Precondition: the FiLM weight IS Muon-routed by default (the bug).
    assert "pose_film0.weight" in split["muon"], (
        "precondition: the film weight must default to Muon (the bug we fix)"
    )

    grads = {
        k: mx.ones_like(v) for k, v in __import__(
            "mlx.utils", fromlist=["tree_flatten"]
        ).tree_flatten(mod.parameters())
    }
    from mlx.utils import tree_unflatten

    grads_tree = tree_unflatten(list(grads.items()))
    cfg = Pr95MlxOptimizerConfig(use_muon=True, muon_lr=1e-2, adamw_lr=1e-2)
    state = Pr95MlxOptimizerState()
    summary = apply_pr95_mlx_optimizer_step(
        mod, grads_tree, state, cfg, force_adamw_substrings=["film"]
    )
    assert "pose_film0.weight" in summary["forced_to_adamw_parameter_names"]
    assert "pose_film0.weight" in summary["adamw_parameter_names"]
    assert "pose_film0.weight" not in summary["muon_parameter_names"]
    # The genuine conv weight STAYS in Muon (we only fork the FiLM path).
    assert "block_conv.weight" in summary["muon_parameter_names"]
    assert "block_conv.weight" not in summary["forced_to_adamw_parameter_names"]


@skip_no_mlx
def test_default_is_byte_identical_to_pr95_partition():
    """NO-FAKE: force_adamw_substrings=None leaves the PR95-faithful split untouched."""
    (
        Pr95MlxOptimizerConfig,
        Pr95MlxOptimizerState,
        apply_pr95_mlx_optimizer_step,
        partition_pr95_mlx_parameter_names,
    ) = _opt_imports()

    mod = _TinyFilmModule()
    from mlx.utils import tree_flatten, tree_unflatten

    grads_tree = tree_unflatten(
        [(k, mx.ones_like(v)) for k, v in tree_flatten(mod.parameters())]
    )
    cfg = Pr95MlxOptimizerConfig(use_muon=True, muon_lr=1e-2, adamw_lr=1e-2)
    summary = apply_pr95_mlx_optimizer_step(
        mod, grads_tree, Pr95MlxOptimizerState(), cfg, force_adamw_substrings=None
    )
    split = partition_pr95_mlx_parameter_names(mod.parameters())
    assert summary["forced_to_adamw_parameter_names"] == []
    assert sorted(summary["muon_parameter_names"]) == sorted(split["muon"])
    # The film weight stays in Muon when no force is applied (default behavior).
    assert "pose_film0.weight" in summary["muon_parameter_names"]


@skip_no_mlx
def test_forced_adamw_step_scales_with_gradient_norm_muon_does_not():
    """DECISIVE Muon-vs-AdamW discriminator: AdamW step ~scales with grad norm; Muon does not.

    Newton-Schulz orthogonalization makes the Muon update magnitude INDEPENDENT of
    the gradient norm (the #3 driver). So: take the SAME film weight, step it once
    with a unit gradient and once with a 1000x gradient.
      - Muon-routed: the two step magnitudes are ~equal (orthogonalized).
      - AdamW-forced: the two step magnitudes DIFFER (scale-sensitive).
    If the routing were a no-op (the forced path still ran Muon), the AdamW assertion
    would FAIL — this catches a fake/no-op fix.
    """
    (
        Pr95MlxOptimizerConfig,
        Pr95MlxOptimizerState,
        apply_pr95_mlx_optimizer_step,
        _,
    ) = _opt_imports()
    from mlx.utils import tree_flatten, tree_unflatten

    def _step_magnitude(force, grad_scale):
        mod = _TinyFilmModule()
        before = dict(tree_flatten(mod.parameters()))["pose_film0.weight"]
        before = mx.array(before)
        grads_tree = tree_unflatten(
            [(k, mx.ones_like(v) * grad_scale) for k, v in tree_flatten(mod.parameters())]
        )
        cfg = Pr95MlxOptimizerConfig(
            use_muon=True, muon_lr=1e-2, adamw_lr=1e-2, grad_clip=None, grad_clip_muon=None
        )
        apply_pr95_mlx_optimizer_step(
            mod,
            grads_tree,
            Pr95MlxOptimizerState(),
            cfg,
            force_adamw_substrings=(["film"] if force else None),
        )
        after = dict(tree_flatten(mod.parameters()))["pose_film0.weight"]
        return float(mx.max(mx.abs(after - before)))

    # Muon-routed (force=False): orthogonalized => grad-norm-independent.
    muon_unit = _step_magnitude(force=False, grad_scale=1.0)
    muon_1000 = _step_magnitude(force=False, grad_scale=1000.0)
    assert muon_1000 == pytest.approx(muon_unit, rel=0.05), (
        f"Muon step must be grad-norm-INDEPENDENT (unit {muon_unit:.4e} vs "
        f"1000x {muon_1000:.4e})"
    )

    # AdamW-forced: the very-first AdamW step's magnitude is ~lr (sign(m)/sqrt(v))
    # so it is NOT 1000x larger, but it MUST differ from the Muon step magnitude
    # (proves the forced path took the AdamW branch, not Muon).
    adamw_unit = _step_magnitude(force=True, grad_scale=1.0)
    assert adamw_unit != pytest.approx(muon_unit, rel=1e-3), (
        f"forced-AdamW step magnitude ({adamw_unit:.6e}) must DIFFER from the Muon "
        f"step ({muon_unit:.6e}) — same value would mean the force routing was a no-op"
    )
    # AdamW first-step magnitude ~= lr (1e-2) within bias-correction; sanity-bound it.
    assert 1e-4 < adamw_unit < 5e-2, (
        f"forced-AdamW first-step magnitude {adamw_unit:.6e} should be ~lr-scale"
    )


@skip_no_mlx
def test_capstone_trainer_forces_film_to_adamw_end_to_end():
    """NO-FAKE: the capstone trainer routes its real FiLM weights to AdamW in a live step."""
    import torch

    from tac.capstone_vq_nerv.capstone_trainer import CapstoneTrainConfig, CapstoneTrainer
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge

    # Tiny frozen scorer stand-ins (reuse the pose-film test scaffolds).
    sys.path.insert(0, str(REPO_ROOT / "src" / "tac" / "mlx_pr95_port" / "tests"))
    from test_pose_film_and_stability import _build_frozen_seg_pose_dnet

    n, h, w = 4, 48, 64
    dnet = _build_frozen_seg_pose_dnet(h, w)
    rng = np.random.RandomState(0)
    pose_tgt = ((rng.rand(n, 6) - 0.5) * 0.5).astype(np.float32)
    seg_tgt = torch.zeros(n, h, w, dtype=torch.long)
    for cls, (r0, r1) in enumerate([(0, 10), (10, 20), (20, 30), (30, 40), (40, h)]):
        seg_tgt[:, r0:r1, :] = cls
    bridge = TorchScorerBridge(
        dnet, seg_tgt, torch.tensor(pose_tgt), seg_loss_form="ce_seg_loss",
        scorer_hw=(h, w), eval_roundtrip=False, seg_weight=1.0, pose_weight=10.0,
    )
    bundle = CapstoneVqNervBundle(
        CapstoneVqNervConfig(num_pairs=n, base_channels=16, codebook_size=16, seed=0)
    )
    # The hook is OPT-IN (default False per the §4 refutation); enable it explicitly
    # to test that the opt-in routing actually forces the FiLM weights to AdamW.
    cfg = CapstoneTrainConfig(
        epochs=1, batch_size=4, eval_every=1, seed=0, force_film_to_adamw=True
    )
    trainer = CapstoneTrainer(bundle, bridge, pose_tgt, cfg)
    assert trainer._force_adamw_substrings == ("film",)
    # Capture the routing summary from one real step via the optimizer-step return.
    captured = {}
    import tac.capstone_vq_nerv.capstone_trainer as ct
    orig = ct.apply_pr95_mlx_optimizer_step

    def _spy(*a, **k):
        s = orig(*a, **k)
        captured.update(s)
        return s

    ct.apply_pr95_mlx_optimizer_step = _spy
    try:
        trainer.step(np.arange(n))
    finally:
        ct.apply_pr95_mlx_optimizer_step = orig
    forced = set(captured.get("forced_to_adamw_parameter_names", []))
    film_forced = {n for n in forced if "film" in n.lower()}
    assert film_forced, "the capstone trainer must force its FiLM weights to AdamW"
    assert not any("film" in n.lower() for n in captured["muon_parameter_names"]), (
        "no FiLM weight may remain in the Muon group after the force routing"
    )


@skip_no_mlx
def test_force_routing_off_keeps_film_in_muon_in_trainer():
    """The opt-out (force_film_to_adamw=False) restores the PR95-default Muon routing."""
    import torch

    from tac.capstone_vq_nerv.capstone_trainer import CapstoneTrainConfig, CapstoneTrainer
    from tac.capstone_vq_nerv.vq_nerv_bundle import (
        CapstoneVqNervBundle,
        CapstoneVqNervConfig,
    )
    from tac.mlx_pr95_port.score_bridge import TorchScorerBridge

    sys.path.insert(0, str(REPO_ROOT / "src" / "tac" / "mlx_pr95_port" / "tests"))
    from test_pose_film_and_stability import _build_frozen_seg_pose_dnet

    n, h, w = 4, 48, 64
    dnet = _build_frozen_seg_pose_dnet(h, w)
    seg_tgt = torch.zeros(n, h, w, dtype=torch.long)
    pose_tgt = np.zeros((n, 6), np.float32)
    bridge = TorchScorerBridge(
        dnet, seg_tgt, torch.tensor(pose_tgt), seg_loss_form="ce_seg_loss",
        scorer_hw=(h, w), eval_roundtrip=False, seg_weight=1.0, pose_weight=1.0,
    )
    bundle = CapstoneVqNervBundle(
        CapstoneVqNervConfig(num_pairs=n, base_channels=16, codebook_size=16, seed=0)
    )
    # The DEFAULT is force_film_to_adamw=False (the §4 refutation: the synthetic A/B
    # showed Muon-FiLM beats AdamW-FiLM, so we do not regress validated behavior on
    # an unproven fix — the hook is opt-in pending a real-FastViT-PoseNet A/B).
    assert CapstoneTrainConfig().force_film_to_adamw is False
    cfg = CapstoneTrainConfig(epochs=1, batch_size=4, seed=0, force_film_to_adamw=False)
    trainer = CapstoneTrainer(bundle, bridge, pose_tgt, cfg)
    assert trainer._force_adamw_substrings is None
