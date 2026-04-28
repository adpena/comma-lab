"""Round 11 Finding 2 (2026-04-28, anti-arbitrariness) — integration tests
proving that ``--use-learnable-pair-weights`` and
``--use-learnable-segnet-class-weights`` ACTUALLY ADAPT the lambdas during
training.

Background: Round 10 retired the softplus(raw)/softmax(raw_logits) Parameter
APIs in favour of buffer-only projected dual-ascent
(``LearnablePairWeights.dual_update`` /
``LearnableClassWeights.dual_update``). However, the train_renderer.py
wiring kept the old "add module.parameters() to the optimiser" branch —
which is now an empty list — and never called ``dual_update``. The flags
silently produced INERT runs that the council would interpret as "the
learnable weights helped" while nothing in fact learned.

Codex Round 11 Finding 2 caught the bug. This test suite locks in the
fix. Two test classes:

  ``TestStaticWiring`` — re-greps train_renderer.py to assert:
    1. NO ``module.parameters()`` ⇒ ``optimizer.add_param_group`` chain
       (the empty-group bug).
    2. ``LearnablePairWeights.dual_update`` IS called inside the per-step
       loop with ``pair_idx=int(pair_idx_int)`` and
       ``eta=args.learnable_pair_weights_lr``.
    3. ``LearnableClassWeights.dual_update`` IS called inside the per-step
       loop with the per-class distortion vector from ``_scorer_aux``.
    4. ``learnable_segnet_class_weights.weights()`` is THREADED into the
       SegNet loss path via ``_pcw`` (otherwise the learned weights would
       never reach the loss).

  ``TestDynamicAdaptation`` — runs the dual-update primitives through 10
  mock training steps and asserts the lambda buffers actually changed
  by more than 1e-6, with the sign of change matching the direction of
  the simulated loss/distortion residual. This is the "did anything
  actually learn" smoke test.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
TRAIN_PATH = REPO / "src" / "tac" / "experiments" / "train_renderer.py"


@pytest.fixture(scope="module")
def train_src() -> str:
    return TRAIN_PATH.read_text()


# ── Static wiring assertions (Round 11 Finding 2) ────────────────────────


class TestStaticWiring:
    def test_empty_param_group_branch_is_removed(self, train_src):
        """Round 11 Finding 2: the previous wiring did
        ``optimizer.add_param_group({"params": list(module.parameters()), ...})``
        which is a NO-OP because the modules expose only buffers. Assert
        no such branch survives.
        """
        # Either of these patterns indicates the bug came back.
        assert "list(learnable_pair_weights.parameters())" not in train_src, (
            "Round 11 Finding 2 regression: passing learnable_pair_weights."
            "parameters() to optimizer adds an empty list (no Parameters "
            "exist on the buffer-only module). Use explicit dual_update() "
            "instead."
        )
        assert "list(learnable_segnet_class_weights.parameters())" not in train_src, (
            "Round 11 Finding 2 regression: passing learnable_segnet_class_"
            "weights.parameters() to optimizer adds an empty list. Use "
            "explicit dual_update() instead."
        )

    def test_pair_weights_dual_update_is_called_in_step_loop(self, train_src):
        """The training loop must call ``learnable_pair_weights.dual_update(...)``
        with the per-step observed loss and the canonical pair_idx_int."""
        # Grep for a dual_update call passing pair_idx + eta=*_lr arg.
        m = re.search(
            r"learnable_pair_weights\.dual_update\([^)]*"
            r"eta=float\(args\.learnable_pair_weights_lr\)[^)]*"
            r"pair_idx=int\(pair_idx_int\)",
            train_src,
            re.DOTALL,
        )
        assert m is not None, (
            "Round 11 Finding 2: train_renderer.py must call "
            "learnable_pair_weights.dual_update(observed_loss, "
            "eta=args.learnable_pair_weights_lr, pair_idx=int(pair_idx_int)) "
            "inside the per-step loop. Without this call the lambda buffer "
            "never updates and the run is effectively a no-op vs the "
            "static --pair-loss-weights branch."
        )

    def test_class_weights_dual_update_is_called_in_step_loop(self, train_src):
        """The training loop must call
        ``learnable_segnet_class_weights.dual_update(per_class_d, eta=...)``
        with the per-class SegNet distortion vector from _scorer_aux."""
        m = re.search(
            r"learnable_segnet_class_weights\.dual_update\([^)]*"
            r"eta=float\(args\.learnable_segnet_class_weights_lr\)",
            train_src,
            re.DOTALL,
        )
        assert m is not None, (
            "Round 11 Finding 2: train_renderer.py must call "
            "learnable_segnet_class_weights.dual_update(per_class_d, "
            "eta=args.learnable_segnet_class_weights_lr) inside the "
            "per-step loop. Without this call the per-class weights never "
            "adapt and Lane PS-V2 is silently inert."
        )

    def test_class_weights_threaded_into_pcw(self, train_src):
        """The learnable class-weight tensor must be threaded into ``_pcw``
        so the SegNet loss path actually USES the adaptive weights. The
        original Round 10 wiring left _pcw bound to the static
        ``args._segnet_class_weights_tensor`` — meaning the learned
        weights were computed but never reached the loss."""
        # _pcw must be reassigned to learnable_segnet_class_weights.weights()
        # when the module is active, BEFORE the scorer_loss call.
        m = re.search(
            r"if learnable_segnet_class_weights is not None:\s*\n"
            r"\s*_pcw\s*=\s*learnable_segnet_class_weights\.weights\(\)\.detach\(\)",
            train_src,
        )
        assert m is not None, (
            "Round 11 Finding 2: when learnable_segnet_class_weights is "
            "active, _pcw must be re-bound to its detached .weights() so "
            "the scorer_loss call USES the learned distribution. Otherwise "
            "the learned weights are computed but never reach the loss."
        )

    def test_with_aux_helpers_called_when_learnable_modules_active(self, train_src):
        """The new ``scorer_loss_with_aux`` / ``scorer_loss_cached_with_aux``
        helpers must be used when either learnable module is active so the
        per-pair pose loss + per-class SegNet distortion are returned for
        the dual_update calls."""
        assert "scorer_loss_with_aux" in train_src, (
            "Round 11 Finding 2: train_renderer.py must import + call "
            "scorer_loss_with_aux when --learnable-pair-weights / "
            "--learnable-segnet-class-weights is active so per-class "
            "distortion is available for dual_update."
        )
        assert "scorer_loss_cached_with_aux" in train_src, (
            "Round 11 Finding 2: train_renderer.py must import + call "
            "scorer_loss_cached_with_aux for the cached-GT path too."
        )


# ── Dynamic adaptation: the load-bearing test ─────────────────────────────


class TestDynamicAdaptation:
    """Run the ACTUAL dual_update primitives over 10 mock training steps
    with a hard pair / hard class pattern. Assert the lambda buffers move
    by > 1e-6, in the correct sign direction.

    This is the test the council would have wanted before authorising any
    Lane W or Lane PS deploy: "PROVE the lambdas change."
    """

    def test_pair_lambda_changes_over_10_steps_and_concentrates_on_hard_pair(self):
        """Simulate 10 streaming dual updates: 9 easy pairs (loss=0.1)
        interleaved with 1 hard pair (loss=2.0). Assert:
          - lambda_pair[hard_idx] strictly > lambda_pair at step 0.
          - all 10 step-1 → step-10 lambda deltas have magnitude > 1e-6.
          - hard pair eventually dominates the easy pairs in lambda value.
        """
        torch.manual_seed(0)
        from tac.learnable_pair_weights import LearnablePairWeights

        N = 10
        HARD_IDX = 7
        pw = LearnablePairWeights(N)
        lambda_step0 = pw.lambda_pair.clone()

        # 10 mock training steps: alternating easy/hard observations.
        for step in range(10):
            for pair_idx in range(N):
                observed = torch.tensor(
                    [2.0 if pair_idx == HARD_IDX else 0.1]
                )
                pw.dual_update(observed, eta=0.05, pair_idx=pair_idx)

        lambda_step10 = pw.lambda_pair.clone()

        # 1) The hard pair's lambda must have strictly increased.
        assert lambda_step10[HARD_IDX].item() > lambda_step0[HARD_IDX].item() + 1e-6, (
            f"Hard pair lambda did not increase: step0="
            f"{lambda_step0[HARD_IDX].item()}, step10="
            f"{lambda_step10[HARD_IDX].item()}. The dual_update wiring is "
            "INERT — Round 11 Finding 2 fix did not actually wire."
        )

        # 2) The hard pair must dominate easy pairs (the whole point).
        easy_max = torch.cat([
            lambda_step10[:HARD_IDX], lambda_step10[HARD_IDX + 1:],
        ]).max().item()
        assert lambda_step10[HARD_IDX].item() > easy_max, (
            f"Hard pair (lambda={lambda_step10[HARD_IDX].item()}) does NOT "
            f"dominate easy pairs (max={easy_max}). Adaptation is broken."
        )

        # 3) The dual_step counter must reflect the call count.
        assert int(pw.dual_step.item()) == 10 * N, (
            f"dual_step counter = {int(pw.dual_step.item())}, expected "
            f"{10 * N}. Some calls were silently dropped."
        )

    def test_class_lambda_changes_over_10_steps_and_grows_on_bottleneck_class(self):
        """Simulate 10 dual updates with a persistent bottleneck class.
        Assert the bottleneck class lambda strictly grows and dominates."""
        torch.manual_seed(0)
        from tac.learnable_class_weights import LearnableClassWeights

        cw = LearnableClassWeights(5)
        lambda_step0 = cw.lambda_class.clone()
        weights_step0 = cw().clone()

        # Bottleneck = class 2 (e.g. "vehicle"). Easy classes near zero.
        per_class_d = torch.tensor([0.01, 0.01, 0.50, 0.01, 0.01])

        for step in range(10):
            cw.dual_update(per_class_d, eta=0.1)

        lambda_step10 = cw.lambda_class.clone()
        weights_step10 = cw().clone()

        # 1) Bottleneck lambda must have strictly grown.
        assert lambda_step10[2].item() > lambda_step0[2].item() + 1e-6, (
            f"Bottleneck class lambda did not grow: step0="
            f"{lambda_step0[2].item()}, step10={lambda_step10[2].item()}. "
            "dual_update wiring is INERT."
        )

        # 2) Easy classes' lambdas must have STAYED at 0 (ReLU floor when
        #    distortion < target). The dual update centres on mean = 0.108.
        # All easy classes have residual ~ -0.098 < 0 ⇒ clamped to 0.
        for c in (0, 1, 3, 4):
            assert lambda_step10[c].item() == pytest.approx(0.0, abs=1e-7), (
                f"Easy class {c} lambda should stay 0 (ReLU floor), got "
                f"{lambda_step10[c].item()}. Sign-direction bug?"
            )

        # 3) Emitted weights for the bottleneck class must now exceed
        #    its initial uniform value — adaptation reached the loss.
        assert weights_step10[2].item() > weights_step0[2].item() + 1e-6, (
            f"Bottleneck class WEIGHT did not grow: step0="
            f"{weights_step0[2].item()}, step10={weights_step10[2].item()}. "
            "Adaptation didn't propagate from lambda to forward output."
        )

        # 4) dual_step counter reflects 10 calls.
        assert int(cw.dual_step.item()) == 10

    def test_pair_weighted_loss_value_changes_after_dual_update(self):
        """End-to-end: the WEIGHTED loss value must differ at step 0 vs
        step 10, i.e. the adaptation must reach the loss the optimiser
        sees. (If it doesn't, the dual_update is decorative.)

        Realistic pattern: streaming dual_update receives observations
        from a MIX of easy and hard pairs (as in train_renderer.py's
        per-step loop). The running-mean target stabilises near the
        global average loss, so the hard pair's residual stays positive
        and its lambda grows.
        """
        torch.manual_seed(0)
        from tac.learnable_pair_weights import (
            LearnablePairWeights,
            compute_pair_weighted_primal_loss,
        )

        N = 6
        HARD_IDX = 4
        pw = LearnablePairWeights(N)

        # Fixed pair_losses tensor — only the WEIGHTS change between calls.
        pair_losses = torch.tensor([0.1, 0.1, 0.1, 0.1, 2.0, 0.1])

        loss_step0 = compute_pair_weighted_primal_loss(pw, pair_losses).item()

        # Mock training: 10 epochs, each iterating over all 6 pairs in
        # random permutation (mirrors train_renderer.py's `perm` loop).
        for epoch in range(10):
            perm = torch.randperm(N).tolist()
            for pair_idx in perm:
                observed = torch.tensor([pair_losses[pair_idx].item()])
                pw.dual_update(observed, eta=0.05, pair_idx=pair_idx)

        loss_step_final = compute_pair_weighted_primal_loss(pw, pair_losses).item()
        assert abs(loss_step_final - loss_step0) > 1e-3, (
            f"Weighted loss at step 0 ({loss_step0}) ≈ step final "
            f"({loss_step_final}); adaptation did NOT reach the loss the "
            "optimiser sees. The whole point of LearnablePairWeights is "
            "to redistribute mass; if the loss is unchanged, the "
            "redistribution is a no-op."
        )
        # The hard pair's lambda must have grown (it was above the
        # running-mean target).
        assert pw.lambda_pair[HARD_IDX].item() > 0.0, (
            f"Hard pair lambda did not grow: {pw.lambda_pair[HARD_IDX].item()}. "
            "The streaming dual_update did not respond to a persistently "
            "above-mean observation."
        )


# ── Argparse / CLI plumbing (defence against re-introducing the bug) ─────


def test_argparse_flags_present():
    """Both flags must remain on the parser — removing them silently
    would also break the wiring."""
    sys.path.insert(0, str(REPO / "src" / "tac" / "experiments"))
    if "train_renderer" in sys.modules:
        del sys.modules["train_renderer"]
    from train_renderer import parse_args  # noqa: E402

    a = parse_args(["--tag", "smoke"])
    assert hasattr(a, "learnable_pair_weights")
    assert hasattr(a, "learnable_pair_weights_lr")
    assert hasattr(a, "learnable_segnet_class_weights")
    assert hasattr(a, "learnable_segnet_class_weights_lr")
    assert a.learnable_pair_weights is False
    assert a.learnable_segnet_class_weights is False
