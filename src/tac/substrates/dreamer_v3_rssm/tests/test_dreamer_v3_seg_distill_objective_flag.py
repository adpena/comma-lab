# SPDX-License-Identifier: MIT
"""DreamerV3 ``--seg-distill-objective`` CLI flag wire-in tests (NO FAKE).

WAVE-1 SUBAGENT A (2026-05-31): the sister "optimal teacher" wave (commits
``9c4f30b47`` + ``72cf191c5``) landed the four MLX seg-distill kernels
(``kl_t2`` / ``boundary_tckd`` / ``boundary_decision_tckd`` /
``boundary_argmax_hinge``) in
``tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss`` and the
``RendererBundle.segnet_distillation_objective`` field that threads the choice
through ``adapter`` -> the loss kernel. The real-SegNet A/B
(``.omx/research/ab_boundary_four_arm_nearcorrect_20260531.json``) showed
``boundary_argmax_hinge`` (Crammer-Singer impostor-complete) drives
``d_seg -> 0.0`` vs the KL-T2 soft-loss floor ``0.0065`` at ``init_d_seg=0.30``.

But the empirically-winning objective was UNREACHABLE from any substrate
trainer CLI — every DreamerV3 / Z6 / pact_nerv ``--full`` run was hardcoded to
the default ``kl_t2`` (the inferior soft objective per the A/B). Per CLAUDE.md
"Forbidden fix-lands-in-helper-but-not-callsite (the dangling-helper trap)" the
kernel + bundle field landed but no trainer caller threaded them. This module's
companion edit adds ``--seg-distill-objective`` / ``--seg-tau-boundary`` /
``--seg-hinge-margin`` to ``experiments/train_substrate_dreamer_v3_rssm.py`` and
threads them into the ``_full_main`` ``RendererBundle``.

Per CLAUDE.md "NO FAKE IMPLEMENTATIONS" (Slot EEE 5 forbidden classes):

- Class 1 protection: the headline test exercises the ACTUAL canonical loss
  surface (``tac.substrates._shared.mlx_score_aware.loss.score_aware_loss``)
  with the DreamerV3 renderer + a real-shaped SegNet teacher cache + the real
  learnable student head, and verifies a BEHAVIORAL consequence (the seg-distill
  term ``parts["distill"]`` differs between ``kl_t2`` and the boundary
  objectives).
- Class 2 protection: the headline guard would FAIL if the flag no-op'd (i.e. if
  the bundle ignored ``segnet_distillation_objective`` and always ran ``kl_t2``)
  OR if the loss were replaced by a constant (all four objectives would then be
  equal). Each boundary objective is asserted DISTINCT from ``kl_t2`` and from
  each other on identical inputs.
- Class 4 protection (no enum padding): the four objectives are verified to
  dispatch to STRUCTURALLY DISTINCT kernels (different real loss values), not
  three branches into the same code.

Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #192/#341:
every value computed here is ``[macOS-MLX research-signal]`` and carries NO
contest-score authority; this is a training-objective wire-in correctness test,
not a score claim.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

try:  # pragma: no cover - import guard for non-Apple CI
    import mlx.core as mx

    MLX_AVAILABLE = True
except Exception:  # pragma: no cover
    MLX_AVAILABLE = False

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not MLX_AVAILABLE, reason="MLX required (Apple Silicon)"
)

if MLX_AVAILABLE:
    from tac.substrates._shared.mlx_score_aware.bundle import RendererBundle
    from tac.substrates._shared.mlx_score_aware.loss import score_aware_loss
    from tac.substrates.dreamer_v3_rssm import (
        DreamerV3RSSMConfig,
        DreamerV3RSSMSubstrateMLX,
    )
    from tac.substrates.hinton_distilled_scorer_surrogate import (
        DEFAULT_POSE_DIMS,
        DEFAULT_SEGNET_CLASSES,
        RealPoseNetTeacherCache,
        RealSegNetTeacherLogitsCache,
        build_learnable_pose_student_head,
        build_learnable_student_head,
    )

_REPO_ROOT = Path(__file__).resolve().parents[5]
_TRAINER_PATH = (
    _REPO_ROOT / "experiments" / "train_substrate_dreamer_v3_rssm.py"
)

_VALID_OBJECTIVES = (
    "kl_t2",
    "boundary_tckd",
    "boundary_decision_tckd",
    "boundary_argmax_hinge",
)
_BOUNDARY_OBJECTIVES = (
    "boundary_tckd",
    "boundary_decision_tckd",
    "boundary_argmax_hinge",
)

_H, _W = 384, 512


# ---------------------------------------------------------------------------
# Trainer-module loading (the trainer is an experiments/ script, not a package
# module; load it via importlib so we can exercise its _build_parser + the
# RendererBundle threading without invoking _full_main).
# ---------------------------------------------------------------------------


_REQUIRED_ARGV = ["--output-dir", "/tmp/_dreamer_seg_distill_test_outdir"]


def _load_trainer_module():
    spec = importlib.util.spec_from_file_location(
        "_train_substrate_dreamer_v3_rssm_under_test", _TRAINER_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Real-shaped fixtures (real-provider CLASSES + varied data, no mock constants).
# ---------------------------------------------------------------------------


def _real_dreamer_renderer(num_pairs: int = 4) -> DreamerV3RSSMSubstrateMLX:
    cfg = DreamerV3RSSMConfig(
        num_groups=24,
        num_categories=256,
        base_channels=24,
        num_pairs=num_pairs,
        gumbel_temperature=1.0,
        use_straight_through=True,
    )
    return DreamerV3RSSMSubstrateMLX(cfg)


def _toy_targets(num_pairs: int = 4):
    rng = np.random.default_rng(0)
    t0 = mx.array(rng.uniform(0, 1, (num_pairs, _H, _W, 3)).astype(np.float32))
    t1 = mx.array(rng.uniform(0, 1, (num_pairs, _H, _W, 3)).astype(np.float32))
    return t0, t1


def _real_shaped_segnet_cache(num_pairs: int = 4) -> RealSegNetTeacherLogitsCache:
    rng = np.random.default_rng(1)
    logits = mx.array(
        rng.uniform(-2, 2, (num_pairs, _H, _W, DEFAULT_SEGNET_CLASSES)).astype(
            np.float32
        )
    )
    return RealSegNetTeacherLogitsCache(
        teacher_logits_thwk=logits,
        frame_count=num_pairs,
        height=_H,
        width=_W,
        num_classes=DEFAULT_SEGNET_CLASSES,
    )


def _real_shaped_posenet_cache(num_pairs: int = 4) -> RealPoseNetTeacherCache:
    rng = np.random.default_rng(2)
    pose = rng.uniform(-1, 1, (num_pairs, DEFAULT_POSE_DIMS)).astype(np.float32)
    raw_std = np.std(pose, axis=0).astype(np.float32)
    scale_floor = max(float(raw_std.max()) * 0.1, 1.0e-3)
    per_dim_scale = np.maximum(raw_std, scale_floor)
    return RealPoseNetTeacherCache(
        teacher_pose_np=mx.array(pose),
        num_pairs=num_pairs,
        pose_dims=DEFAULT_POSE_DIMS,
        per_dim_scale=mx.array(per_dim_scale),
    )


def _bundle_for_objective(
    objective: str,
    *,
    num_pairs: int = 4,
    tau_boundary: float = 1.0,
    hinge_margin: float = 1.0,
) -> RendererBundle:
    """Canonical DreamerV3 real-teacher bundle with the given seg objective.

    Mirrors the trainer ``_full_main`` wiring EXACTLY except for the
    ``segnet_distillation_objective`` (+ companion tau/margin). The model + all
    targets + teacher caches + heads are built from FIXED seeds so two bundles
    that differ only in the objective are pixel-identical on every other axis —
    so any difference in ``parts["distill"]`` is attributable to the objective.
    """
    model = _real_dreamer_renderer(num_pairs)
    t0, t1 = _toy_targets(num_pairs)
    return RendererBundle(
        model=model,
        target_rgb_0=t0,
        target_rgb_1=t1,
        num_pairs=num_pairs,
        forward_convention="call_b2chw_255",
        distillation_weight=0.5,
        scorer_teacher=_real_shaped_segnet_cache(num_pairs),
        learnable_student_head=build_learnable_student_head(
            num_classes=DEFAULT_SEGNET_CLASSES, in_channels=3, seed=0
        ),
        segnet_distillation_objective=objective,
        segnet_tau_boundary=tau_boundary,
        segnet_hinge_margin=hinge_margin,
        pose_distillation_weight=1.0,
        pose_scorer_teacher=_real_shaped_posenet_cache(num_pairs),
        learnable_pose_student_head=build_learnable_pose_student_head(
            pose_dims=DEFAULT_POSE_DIMS, seed=0
        ),
        pose_dims=DEFAULT_POSE_DIMS,
        allow_mock_scorer_teacher=False,
    )


def _distill_value(bundle: RendererBundle, num_pairs: int = 4) -> float:
    idx = mx.arange(num_pairs).astype(mx.int32)
    _total, parts = score_aware_loss(bundle, idx)
    assert "distill" in parts, "score_aware_loss must emit the seg-distill term"
    return float(parts["distill"])


# ---------------------------------------------------------------------------
# (A) Trainer argparse wire-in — the flag exists, parses, defaults preserved.
# ---------------------------------------------------------------------------


def test_parser_exposes_seg_distill_objective_flag():
    mod = _load_trainer_module()
    parser = mod._build_parser()
    ns = parser.parse_args(list(_REQUIRED_ARGV))
    assert hasattr(ns, "seg_distill_objective")
    assert hasattr(ns, "seg_tau_boundary")
    assert hasattr(ns, "seg_hinge_margin")


def test_seg_distill_objective_default_is_kl_t2_legacy_preserving():
    """Default MUST stay kl_t2 so existing runs are byte-for-byte unchanged."""
    mod = _load_trainer_module()
    ns = mod._build_parser().parse_args(list(_REQUIRED_ARGV))
    assert ns.seg_distill_objective == "kl_t2"
    assert ns.seg_tau_boundary == pytest.approx(1.0)
    assert ns.seg_hinge_margin == pytest.approx(1.0)


@pytest.mark.parametrize("objective", _VALID_OBJECTIVES)
def test_parser_accepts_each_valid_objective(objective: str):
    mod = _load_trainer_module()
    ns = mod._build_parser().parse_args(
        [*_REQUIRED_ARGV, "--seg-distill-objective", objective]
    )
    assert ns.seg_distill_objective == objective


def test_parser_rejects_invalid_objective():
    mod = _load_trainer_module()
    with pytest.raises(SystemExit):
        mod._build_parser().parse_args(
            [*_REQUIRED_ARGV, "--seg-distill-objective", "not_a_real_objective"]
        )


def test_parser_threads_tau_and_margin_values():
    mod = _load_trainer_module()
    ns = mod._build_parser().parse_args(
        [
            *_REQUIRED_ARGV,
            "--seg-distill-objective",
            "boundary_argmax_hinge",
            "--seg-tau-boundary",
            "2.0",
            "--seg-hinge-margin",
            "0.5",
        ]
    )
    assert ns.seg_tau_boundary == pytest.approx(2.0)
    assert ns.seg_hinge_margin == pytest.approx(0.5)


def test_trainer_source_threads_objective_into_renderer_bundle():
    """Catalog #229 dangling-helper guard: the trainer source MUST pass the flag
    into the main RendererBundle (not just parse it). A flag that parses but is
    never threaded is the no-op trap. We assert the source text wires all three
    bundle kwargs from the parsed args."""
    src = _TRAINER_PATH.read_text(encoding="utf-8")
    assert "segnet_distillation_objective=str(args.seg_distill_objective)" in src
    assert "segnet_tau_boundary=float(args.seg_tau_boundary)" in src
    assert "segnet_hinge_margin=float(args.seg_hinge_margin)" in src


def test_trainer_threads_objective_into_archive_export_and_metadata():
    """No signal loss: full training must export the selected objective into both
    the archive-bound bridge and the canonical training artifact metadata."""
    src = _TRAINER_PATH.read_text(encoding="utf-8")
    assert "export_archive_fn=_export_dreamer_archive" in src
    assert "export_dreamer_v3_rssm_mlx_archive(" in src
    assert '"score_aware_training": score_aware_training_metadata' in src
    assert "mlx_triage_argv=replay_argv" in src
    assert "archive_bound_candidate_adapter_package.json" in src


def test_full_replay_argv_preserves_boundary_hinge_flags():
    """Replay bundles must preserve the mathematically selected objective."""
    mod = _load_trainer_module()
    ns = mod._build_parser().parse_args(
        [
            *_REQUIRED_ARGV,
            "--seg-distill-objective",
            "boundary_argmax_hinge",
            "--seg-tau-boundary",
            "2.0",
            "--seg-hinge-margin",
            "0.5",
            "--tau-anneal-enabled",
            "--cosine-decay-enabled",
        ]
    )

    argv = mod._full_replay_argv(ns)
    assert "--smoke" not in argv
    assert argv[argv.index("--seg-distill-objective") + 1] == "boundary_argmax_hinge"
    assert argv[argv.index("--seg-tau-boundary") + 1] == "2.0"
    assert argv[argv.index("--seg-hinge-margin") + 1] == "0.5"
    assert "--tau-anneal-enabled" in argv
    assert "--cosine-decay-enabled" in argv


# ---------------------------------------------------------------------------
# (B) HEADLINE behavioral NO-FAKE guard — objective selection is NOT a no-op.
# Identical bundles (same renderer / targets / teacher / heads / seeds) except
# the objective MUST produce DIFFERENT real seg-distill loss. This would FAIL if
# the flag no-op'd OR the loss were a constant (Class 1 + 2 + 4 protection).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("boundary_objective", _BOUNDARY_OBJECTIVES)
def test_boundary_objective_changes_seg_distill_loss_vs_kl_t2(
    boundary_objective: str,
):
    kl = _distill_value(_bundle_for_objective("kl_t2"))
    boundary = _distill_value(_bundle_for_objective(boundary_objective))
    # Both must be finite real losses (not NaN / inf / a constant sentinel).
    assert np.isfinite(kl) and np.isfinite(boundary)
    # The HEADLINE guard: a different objective MUST move the seg-distill term.
    # Replacing the kernel with a constant (or ignoring the flag) makes these
    # equal -> this assertion is the structural NO-FAKE protection.
    # Inter-objective gaps are O(0.17-0.62); the DreamerV3 Gumbel-Softmax
    # run-to-run spread is <=1.6e-4. atol=1e-2 is ~60x the noise floor and far
    # below the smallest real inter-objective gap, so this guard fires on a
    # no-op / constant-collapse and NOT on Gumbel stochasticity.
    assert abs(kl - boundary) > 1.0e-2, (
        f"{boundary_objective} produced ~the SAME seg-distill loss as kl_t2 "
        f"({boundary} ~= {kl}); the --seg-distill-objective flag is a no-op or "
        f"the kernel collapsed to a constant (NO FAKE Class 2/4 violation)"
    )


def test_all_four_objectives_are_mutually_distinct():
    """Class 4 (no enum padding): each objective dispatches to a STRUCTURALLY
    DISTINCT kernel, so the four real loss values are mutually distinct."""
    values = {obj: _distill_value(_bundle_for_objective(obj)) for obj in _VALID_OBJECTIVES}
    for v in values.values():
        assert np.isfinite(v)
    seen: list[float] = []
    for obj, v in values.items():
        for prev in seen:
            assert abs(v - prev) > 1.0e-2, (
                f"objective {obj} collapsed onto a sister kernel value {v}; "
                f"the four objectives must be structurally distinct (gaps are "
                f"O(0.17); Gumbel noise <=1.6e-4)"
            )
        seen.append(v)


def test_argmax_hinge_responds_to_margin():
    """The hinge margin is a real hyperparameter, not a dead kwarg: two margins
    on boundary_argmax_hinge produce different real losses."""
    lo = _distill_value(
        _bundle_for_objective("boundary_argmax_hinge", hinge_margin=0.5)
    )
    hi = _distill_value(
        _bundle_for_objective("boundary_argmax_hinge", hinge_margin=2.0)
    )
    assert np.isfinite(lo) and np.isfinite(hi)
    assert abs(lo - hi) > 1.0e-2, (
        "boundary_argmax_hinge ignored --seg-hinge-margin (dead kwarg); "
        "margin 0.5 vs 2.0 must move the hinge by >> the 1.6e-4 Gumbel noise"
    )


def test_boundary_tckd_responds_to_tau_boundary():
    """tau_boundary is a real hyperparameter for the boundary-band weight: two
    tau values on boundary_tckd produce different real losses."""
    tight = _distill_value(
        _bundle_for_objective("boundary_tckd", tau_boundary=0.5)
    )
    wide = _distill_value(
        _bundle_for_objective("boundary_tckd", tau_boundary=4.0)
    )
    assert np.isfinite(tight) and np.isfinite(wide)
    assert abs(tight - wide) > 1.0e-2, (
        "boundary_tckd ignored --seg-tau-boundary (dead kwarg); tau 0.5 vs 4.0 "
        "must reweight the boundary-band TCKD by >> the 1.6e-4 Gumbel noise"
    )


def test_kl_t2_run_to_run_spread_is_below_inter_objective_gap():
    """RNG-floor guard: the DreamerV3 renderer uses Gumbel-Softmax sampling, so
    the same objective on two identically-seeded bundles is NOT bit-identical —
    but the run-to-run spread (<=1.6e-4 empirically) is far below the smallest
    inter-objective gap (~0.17). This bounds the noise so the distinctness guards
    above measure the OBJECTIVE, not RNG."""
    a = _distill_value(_bundle_for_objective("kl_t2"))
    b = _distill_value(_bundle_for_objective("kl_t2"))
    assert abs(a - b) < 1.0e-3, (
        f"kl_t2 run-to-run spread {abs(a - b):.2e} exceeded the 1e-3 noise "
        f"bound; the inter-objective distinctness guards (atol 1e-2) would no "
        f"longer cleanly separate signal from RNG"
    )
