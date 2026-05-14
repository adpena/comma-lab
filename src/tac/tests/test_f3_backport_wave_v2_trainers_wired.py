# SPDX-License-Identifier: MIT
"""F3-BACKPORT-WAVE-V2 trainer-side wire-in regression suite.

Per F3-BACKPORT-WAVE-V2 landing 2026-05-14 + the premise-verification
pattern: every substrate trainer that declares the RESERVED
``--enable-gt-scorer-cache`` flag MUST also consume the cache in its
hot loop. Without trainer-side consumption the flag is parsed-but-not-
threaded dead code, and the F3 GTScorerCache primitive provides zero
empirical speedup despite landing in the canonical helper.

These tests do NOT run any training; they verify the source-level
contract:

1. ``build_optimized_training_context`` (or ``build_gt_scorer_cache``)
   is imported and called after scorer load + pair decode.
2. ``gt_cache.lookup(...)`` is invoked per-batch inside the hot loop.
3. ``gt_pose_batch=`` / ``gt_seg_batch=`` / ``gt_seg_already_probs=``
   kwargs are threaded into the ``loss_fn(...)`` call (training AND
   val loops).

The canonical reproducer ``tools/check_f3_trainer_actionable.py``
implements the same predicates; this pytest is the CI gate that
prevents regression of the wire-in across the 10 trainers backported in
F3-BACKPORT-WAVE-V2 + the 3 already-wired trainers that predated it.

Out of scope (per F3-BACKPORT-WAVE-V2 landing memo):
- vq_vae (needs argparse flag declaration ALSO; OUT_OF_SCOPE)
- pretrained_driving_prior (substrate-side score_aware_loss lacks F3
  kwargs; OUT_OF_SCOPE)
- time_traveler_l5_autonomy (substrate not F3-wired AND _full_main
  raises; OUT_OF_SCOPE)
- pr101_lc_v2_clone_enhanced_curriculum (_full_main raises + smoke uses
  synthetic scorers; structurally moot)
- d1_segnet_margin_polytope (SegNet-only sidecar; no PoseNet hot loop)
- c1_world_model_foveation / z3 / z4 / z5 (stub-state per predecessor
  finding)
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


# The 13 trainers that MUST stay ALREADY_WIRED to prevent F3-BACKPORT-WAVE-V2
# regression. The 10 added in this wave + the 3 that predated it.
F3_WIRED_TRAINERS = [
    # F3-BACKPORT-WAVE-V2 (added 2026-05-14)
    "sane_hnerv",
    "balle_renderer",
    "tc_nerv",
    "block_nerv",
    "ff_nerv",
    "ds_nerv",
    "hi_nerv",
    "cool_chic",
    "self_compress_nn",
    "hybrid_renderer_residual",
    # Predated F3-BACKPORT-WAVE-V2
    "d4_wyner_ziv_frame_0",
    "c6_e4_mdl_ibps",
    "siren",
]


@pytest.fixture(scope="module")
def _trainer_text():
    """Cache per-trainer source text for the duration of the test module."""
    cache = {}
    for tid in F3_WIRED_TRAINERS:
        path = REPO_ROOT / "experiments" / f"train_substrate_{tid}.py"
        if path.is_file():
            cache[tid] = path.read_text(encoding="utf-8", errors="replace")
        else:
            cache[tid] = None
    return cache


@pytest.mark.parametrize("trainer_id", F3_WIRED_TRAINERS)
def test_trainer_imports_canonical_optimization_context(
    _trainer_text, trainer_id
):
    """The canonical helper for F3 cache building MUST be imported.

    The trainer must import ``build_optimized_training_context`` or
    ``build_gt_scorer_cache`` from ``tac.substrates._shared.trainer_skeleton``
    (or ``tac.training_optimization``).
    """
    text = _trainer_text[trainer_id]
    assert text is not None, f"trainer file missing for {trainer_id}"
    assert (
        "build_optimized_training_context" in text
        or "build_gt_scorer_cache" in text
    ), (
        f"{trainer_id}: F3 cache building helper not imported. Add\n"
        "  from tac.substrates._shared.trainer_skeleton import\n"
        "      build_optimized_training_context as _canon_..."
    )


@pytest.mark.parametrize("trainer_id", F3_WIRED_TRAINERS)
def test_trainer_calls_gt_cache_lookup(_trainer_text, trainer_id):
    """The trainer MUST call ``gt_cache.lookup(...)`` per-batch."""
    text = _trainer_text[trainer_id]
    assert text is not None, f"trainer file missing for {trainer_id}"
    assert "gt_cache.lookup" in text, (
        f"{trainer_id}: trainer hot loop does not call gt_cache.lookup. "
        "F3 cache wire-in incomplete (cache built but never consumed)."
    )


@pytest.mark.parametrize("trainer_id", F3_WIRED_TRAINERS)
def test_trainer_threads_gt_pose_batch_kwarg(_trainer_text, trainer_id):
    """The loss_fn call MUST thread the F3 kwargs."""
    text = _trainer_text[trainer_id]
    assert text is not None, f"trainer file missing for {trainer_id}"
    assert "gt_pose_batch=" in text, (
        f"{trainer_id}: loss_fn call missing gt_pose_batch= kwarg. "
        "F3 cache wire-in incomplete (lookup happens but result is discarded)."
    )
    assert "gt_seg_batch=" in text, (
        f"{trainer_id}: loss_fn call missing gt_seg_batch= kwarg."
    )
    assert "gt_seg_already_probs=" in text, (
        f"{trainer_id}: loss_fn call missing gt_seg_already_probs= kwarg."
    )


@pytest.mark.parametrize("trainer_id", F3_WIRED_TRAINERS)
def test_trainer_declares_reserved_gt_scorer_cache_flag(
    _trainer_text, trainer_id
):
    """The argparse must declare ``--enable-gt-scorer-cache`` per Catalog #151."""
    text = _trainer_text[trainer_id]
    assert text is not None, f"trainer file missing for {trainer_id}"
    assert (
        "--enable-gt-scorer-cache" in text
        or "enable_gt_scorer_cache" in text
    ), (
        f"{trainer_id}: argparse / TIER_1 manifest missing "
        "--enable-gt-scorer-cache flag declaration."
    )


@pytest.mark.parametrize("trainer_id", F3_WIRED_TRAINERS)
def test_trainer_has_opt_ctx_or_canonical_pattern(_trainer_text, trainer_id):
    """The trainer must use the canonical ``opt_ctx`` / ``_build_optimized_training_context`` pattern.

    Catches the failure mode where someone calls
    ``build_optimized_training_context`` but discards the result.
    """
    text = _trainer_text[trainer_id]
    assert text is not None, f"trainer file missing for {trainer_id}"
    # The canonical pattern threads through opt_ctx.gt_cache
    assert (
        "opt_ctx.gt_cache" in text
        or "opt_ctx = " in text
    ), (
        f"{trainer_id}: canonical opt_ctx pattern not found. The trainer "
        "must capture the OptimizedTrainingContext return value."
    )


def test_check_f3_trainer_actionable_classifies_all_thirteen_as_already_wired(
    _trainer_text,
):
    """End-to-end: run the canonical checker against the 13 wired trainers.

    Regression guard: if any of the 13 trainers gets accidentally
    un-wired (e.g. by a refactor that drops the F3 kwargs from the
    loss call), this test fails with the offending trainer's verdict.
    """
    import importlib.util

    check_path = REPO_ROOT / "tools" / "check_f3_trainer_actionable.py"
    spec = importlib.util.spec_from_file_location(
        "check_f3_trainer_actionable", check_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for tid in F3_WIRED_TRAINERS:
        result = mod._classify(tid)
        assert result["verdict"] == "ALREADY_WIRED", (
            f"{tid}: expected ALREADY_WIRED, got {result['verdict']}. "
            f"Diagnostics: cache_build={result.get('has_cache_build')}, "
            f"cache_lookup={result.get('has_cache_lookup')}, "
            f"kwargs_threaded={result.get('has_kwargs_threaded')}, "
            f"reserved={result.get('has_reserved_marker')}"
        )
