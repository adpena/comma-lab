# SPDX-License-Identifier: MIT
"""Slot CCC HUGO real per-pixel SPAM-delta via canonical helper migration tests.

Per the operator binding 5-invariant standing directive 2026-05-29 +
Slot EEE fake-implementation audit verdict on Slot CCC (PARTIAL: per-pixel
SPAM-delta is simplified to cell-counting heuristic; smoke is synthetic
random noise NOT real video): these tests verify the new bind helper
``apply_hugo_canonical_real_per_pixel_spam_delta_via_canonical_real_video_mlx_to_pr110_archive``
routes through the canonical shared helper
``tac.inverse_steganalysis_real_video_mlx.compute_hugo_per_pixel_spam_delta_mlx``
on REAL ``upstream/videos/0.mkv`` decoded frames per Pevný-Filler-Bas 2010.

Sister of the canonical Slot YY HILL bind helper test pattern at
``src/tac/composition/hill_canonical_inverse_steganalysis_li_wang_li_huang_2014/tests/test_canonical_helpers.py``
(commit ``32a70c051``).

Catalog discipline
==================

- Catalog #192: NEVER promotable (axis_tag, promotable, score_claim, evidence_grade).
- Catalog #213: REAL ``upstream/videos/0.mkv`` decoded frames (NOT synthetic).
- Catalog #287: every claim carries an evidence tag; no placeholder rationales.
- Catalog #323: canonical Provenance is present + valid per the canonical contract.
- Catalog #341: Tier A canonical-routing markers present.
- Catalog #356: canonical AxisDecomposition emitted.
- Catalog #305: smoke surfaces canonical 6-facet observability stats.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# Skip cleanly when pyav is not installed (e.g. minimal CI environments).
av = pytest.importorskip("av")

from tac.composition.hugo_canonical_inverse_steganalysis_pevny_filler_bas_2010 import (
    CANONICAL_SPAM_TRUNCATION_T,
    STRATEGY_PER_PIXEL_REAL_SPAM_DELTA_MLX,
    apply_hugo_canonical_real_per_pixel_spam_delta_via_canonical_real_video_mlx_to_pr110_archive as apply_real,
)


_REAL_VIDEO_PATH = Path("upstream/videos/0.mkv")


def _real_video_available() -> bool:
    return _REAL_VIDEO_PATH.exists()


_SKIP_NO_REAL_VIDEO = pytest.mark.skipif(
    not _real_video_available(),
    reason=(
        "Canonical real upstream/videos/0.mkv not available; Slot CCC real-video "
        "bind helper requires Catalog #213 canonical real-video frames."
    ),
)


# -----------------------------------------------------------------------------
# CANONICAL: bind helper produces per-pixel SPAM-delta on real frames
# -----------------------------------------------------------------------------


@_SKIP_NO_REAL_VIDEO
def test_bind_helper_produces_per_pixel_spam_delta_on_real_frames():
    """The bind helper decodes REAL upstream frames and emits a per-pixel cost matrix."""
    result = apply_real(num_frames=4, target_resolution=(128, 96), use_mlx=False)

    smoke = result["smoke_result"]
    assert smoke["n_frames_decoded"] == 4
    # frame_resolution serialized as [H, W]
    assert tuple(smoke["frame_resolution"]) == (96, 128)
    assert tuple(smoke["cost_matrix_shape"]) == (96, 128)
    # Cost values are bounded by the canonical 4-direction SPAM weighting:
    # at most 2 * 4 = 8 weight units per pixel, never negative.
    assert smoke["cost_matrix_min"] >= 0.0
    assert smoke["cost_matrix_max"] <= 8.0 + 1e-6
    # The smoke should produce a finite elapsed time.
    assert smoke["elapsed_seconds"] > 0.0


@_SKIP_NO_REAL_VIDEO
def test_bind_helper_default_arguments_run_clean():
    """The bind helper runs with default arguments without raising."""
    result = apply_real()
    assert result["verdict"].startswith("PER_PIXEL_MLX_REAL_VIDEO_SMOKE_GREEN")


# -----------------------------------------------------------------------------
# CANONICAL: bind helper differs from existing cell-counting heuristic
# -----------------------------------------------------------------------------


@_SKIP_NO_REAL_VIDEO
def test_bind_helper_differs_from_existing_cell_counting_heuristic():
    """Per Slot EEE Axis A: the REAL per-pixel SPAM-delta is NOT identical to
    the existing cell-counting heuristic.

    The existing ``_compute_spam_feature_delta_per_pixel`` returns either {0,
    1, 2} per cell counted (canonical integer-step weighting), while the
    canonical helper's per-pixel SPAM-delta multiplies by
    ``perturbation_magnitude * 255`` (default 1.0) and returns floats in a
    different numerical band. The mean values across real frames should
    differ measurably; if they are byte-identical the migration would be a
    no-op and Slot EEE Axis A would not be remediated.
    """
    import numpy as np

    from tac.composition.hugo_canonical_inverse_steganalysis_pevny_filler_bas_2010 import (
        _compute_canonical_residual_per_direction,
        _canonical_truncate_residual,
        _compute_spam_feature_delta_per_pixel,
        CANONICAL_4_DIRECTION_OFFSETS,
    )
    from tac.inverse_steganalysis_real_video_mlx import (
        compute_hugo_per_pixel_spam_delta_mlx,
        decode_upstream_video_frames,
    )

    luma_batch = decode_upstream_video_frames(
        video_path=_REAL_VIDEO_PATH,
        num_frames=2,
        target_resolution=(64, 48),
        return_format="luma_fp32",
    )
    frame = luma_batch[0]

    real_cost = compute_hugo_per_pixel_spam_delta_mlx(
        frame,
        truncation_t=CANONICAL_SPAM_TRUNCATION_T,
        perturbation_magnitude=1.0 / 255.0,
        use_mlx=False,
    )

    cover_seq = [list(row) for row in frame.tolist()]
    cell_cost = _compute_spam_feature_delta_per_pixel(
        cover_seq,
        direction_offsets=list(CANONICAL_4_DIRECTION_OFFSETS),
        T=CANONICAL_SPAM_TRUNCATION_T,
        cooccurrence_order=1,
    )
    cell_cost_arr = np.asarray(cell_cost, dtype=np.float32)

    # Both must be the same shape so they are diffable.
    assert real_cost.shape == cell_cost_arr.shape
    # The mean values are computed on different numerical scales so they
    # must differ measurably; if they are within 1e-6 the migration is a
    # no-op and the Slot EEE Axis A verdict is NOT remediated.
    assert abs(float(real_cost.mean()) - float(cell_cost_arr.mean())) > 1e-6


# -----------------------------------------------------------------------------
# CANONICAL: Tier A canonical-routing markers per Catalog #341
# -----------------------------------------------------------------------------


@_SKIP_NO_REAL_VIDEO
def test_bind_helper_emits_tier_a_canonical_routing_markers():
    result = apply_real(num_frames=2, target_resolution=(64, 48), use_mlx=False)
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["promotable"] is False
    assert result["score_claim"] is False
    assert result["axis_tag"] == "[predicted]"


@_SKIP_NO_REAL_VIDEO
def test_smoke_result_carries_canonical_routing_markers():
    """Per Catalog #341: the canonical smoke result also carries Tier A markers."""
    result = apply_real(num_frames=2, target_resolution=(64, 48), use_mlx=False)
    smoke = result["smoke_result"]
    markers = smoke["canonical_routing_markers"]
    assert markers["predicted_delta_adjustment"] == 0.0
    assert markers["promotable"] is False
    assert markers["score_claim"] is False
    assert markers["axis_tag"] == "[predicted]"
    assert markers["evidence_grade"] == "predicted"


# -----------------------------------------------------------------------------
# CANONICAL: Provenance per Catalog #323
# -----------------------------------------------------------------------------


@_SKIP_NO_REAL_VIDEO
def test_bind_helper_emits_canonical_provenance_per_catalog_323():
    """The AxisDecomposition's canonical_provenance is emitted via the
    canonical helper :func:`tac.provenance.builders.build_provenance_for_predicted`,
    which serializes to the canonical Provenance schema per Catalog #323."""
    result = apply_real(num_frames=2, target_resolution=(64, 48), use_mlx=False)
    decomp = result["predicted_axis_decomposition"]
    prov = decomp["canonical_provenance"]
    # Required Provenance fields per the canonical helper-emitted dict shape.
    assert "artifact_kind" in prov
    assert "measurement_axis" in prov
    assert "evidence_grade" in prov
    # The bind helper produces predicted-grade evidence per Catalog #192.
    assert prov["evidence_grade"] == "predicted"
    assert prov["measurement_axis"] == "[predicted]"
    # Per the canonical Catalog #192 NEVER promotable contract.
    assert prov["promotion_eligible"] is False
    assert prov["score_claim_valid"] is False
    # The canonical helper invocation must be cited per Catalog #305.
    assert (
        prov["canonical_helper_invocation"]
        == "tac.provenance.builders.build_provenance_for_predicted"
    )
    # source_sha256 should be the lowercase 64-char hex digest computed
    # from the canonical bind-input signature.
    assert "source_sha256" in prov
    sha = prov["source_sha256"]
    assert isinstance(sha, str)
    assert len(sha) == 64
    assert sha == sha.lower()
    assert all(c in "0123456789abcdef" for c in sha)


@_SKIP_NO_REAL_VIDEO
def test_bind_helper_smoke_carries_canonical_provenance_per_catalog_323():
    """Per Catalog #323: the smoke result also carries canonical Provenance
    (built directly by the canonical smoke runner via the
    :mod:`tac.inverse_steganalysis_real_video_mlx` ad-hoc dict shape)."""
    result = apply_real(num_frames=2, target_resolution=(64, 48), use_mlx=False)
    smoke = result["smoke_result"]
    prov = smoke["canonical_provenance"]
    # The smoke runner uses an ad-hoc Provenance dict with `kind` rather than
    # the canonical helper's `artifact_kind`; both shapes are valid per the
    # macOS-CPU advisory canonical surface.
    assert prov["kind"] == "predicted"
    assert prov["axis_tag"] == "[macOS-CPU advisory]"
    assert prov["evidence_grade"] == "predicted"
    # Per Catalog #192: hardware_substrate identifies macOS Apple Silicon.
    assert "macos" in prov["hardware_substrate"].lower()


# -----------------------------------------------------------------------------
# CANONICAL: AxisDecomposition per Catalog #356
# -----------------------------------------------------------------------------


@_SKIP_NO_REAL_VIDEO
def test_bind_helper_emits_axis_decomposition_per_catalog_356():
    result = apply_real(num_frames=2, target_resolution=(64, 48), use_mlx=False)
    decomp = result["predicted_axis_decomposition"]
    assert decomp["axis_tag"] == "[predicted]"
    # L0 SCAFFOLD: smoke-only, no actual perturbation, so deltas are 0.
    assert decomp["predicted_d_seg_delta"] == 0.0
    assert decomp["predicted_d_pose_delta"] == 0.0
    assert decomp["predicted_archive_bytes_delta"] == 0


# -----------------------------------------------------------------------------
# CANONICAL: macOS-CPU advisory tagging per Catalog #192 NEVER promotable
# -----------------------------------------------------------------------------


@_SKIP_NO_REAL_VIDEO
def test_bind_helper_smoke_axis_tag_is_macos_cpu_advisory():
    """Per Catalog #192: smoke results carry the [macOS-CPU advisory] axis tag."""
    result = apply_real(num_frames=2, target_resolution=(64, 48), use_mlx=False)
    smoke = result["smoke_result"]
    prov = smoke["canonical_provenance"]
    assert prov["axis_tag"] == "[macOS-CPU advisory]"
    assert prov["score_claim_valid"] is False


# -----------------------------------------------------------------------------
# CANONICAL: existing 112-test per-pair surface still callable (regression)
# -----------------------------------------------------------------------------


def test_existing_per_pair_apply_helper_still_importable():
    """Regression: the existing apply helper at the PR110 per-pair surface
    remains importable for backward compatibility."""
    from tac.composition.hugo_canonical_inverse_steganalysis_pevny_filler_bas_2010 import (
        apply_hugo_canonical_spam_feature_cost_matrix_to_pr110_archive,
        HUGOConfig,
        HUGOSPAMFeatureStrategy,
    )

    assert callable(apply_hugo_canonical_spam_feature_cost_matrix_to_pr110_archive)
    cfg = HUGOConfig(strategy=HUGOSPAMFeatureStrategy.CANONICAL_4_DIRECTION_SPAM)
    # Tiny image — just to confirm the surface still answers a real call.
    image = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    out = apply_hugo_canonical_spam_feature_cost_matrix_to_pr110_archive(image, cfg)
    assert out["axis_tag"] == "[predicted]"
    assert out["promotable"] is False


def test_existing_compute_helper_still_importable():
    """Regression: the existing compute helper at the PR110 per-pair surface
    remains importable for backward compatibility."""
    from tac.composition.hugo_canonical_inverse_steganalysis_pevny_filler_bas_2010 import (
        compute_hugo_canonical_spam_feature_cost_matrix_for_pr110_catalog,
    )

    assert callable(compute_hugo_canonical_spam_feature_cost_matrix_for_pr110_catalog)


# -----------------------------------------------------------------------------
# CANONICAL: dynamic-range >= Slot EEE baseline indicator
# -----------------------------------------------------------------------------


@_SKIP_NO_REAL_VIDEO
def test_smoke_result_carries_dynamic_range_db_observability():
    """Per Catalog #305: the smoke result surfaces the dynamic-range-dB
    cost-discrimination indicator (Slot EEE baseline anchor 6.02 dB on real
    upstream/videos/0.mkv per the canonical Slot YY HILL reference)."""
    result = apply_real(num_frames=4, target_resolution=(128, 96), use_mlx=False)
    smoke = result["smoke_result"]
    # The canonical truncated-residual 4-direction SPAM-delta on real video
    # produces a measurable dynamic range. The exact value depends on the
    # decoded frame content; verify it is finite and >= 0.
    dr = smoke["cost_matrix_dynamic_range_db"]
    assert dr >= 0.0
    # The canonical Slot EEE baseline anchor on real video is 6.02 dB; the
    # bind helper on the canonical 4-direction SPAM-delta should land in
    # that band (or higher when textured regions dominate).
    assert dr == pytest.approx(6.02, abs=0.5) or dr > 6.02


# -----------------------------------------------------------------------------
# CANONICAL: strategy sentinel echo
# -----------------------------------------------------------------------------


@_SKIP_NO_REAL_VIDEO
def test_bind_helper_strategy_sentinel_echoes_default():
    result = apply_real(num_frames=2, target_resolution=(64, 48), use_mlx=False)
    assert result["strategy"] == STRATEGY_PER_PIXEL_REAL_SPAM_DELTA_MLX
    assert (
        result["strategy"]
        == "per_pixel_real_spam_delta_via_canonical_real_video_mlx"
    )


@_SKIP_NO_REAL_VIDEO
def test_bind_helper_strategy_sentinel_echoes_custom():
    custom = "custom_research_sentinel_for_test"
    result = apply_real(
        num_frames=2,
        target_resolution=(64, 48),
        use_mlx=False,
        strategy=custom,
    )
    assert result["strategy"] == custom


# -----------------------------------------------------------------------------
# CANONICAL: verdict + remediation anchor
# -----------------------------------------------------------------------------


@_SKIP_NO_REAL_VIDEO
def test_bind_helper_verdict_is_deferred_pending_paired_cuda_per_catalog_325():
    result = apply_real(num_frames=2, target_resolution=(64, 48), use_mlx=False)
    assert (
        result["verdict"]
        == "PER_PIXEL_MLX_REAL_VIDEO_SMOKE_GREEN_DEFERRED_PENDING_PAIRED_CUDA_EMPIRICAL_ANCHOR"
    )


@_SKIP_NO_REAL_VIDEO
def test_bind_helper_remediation_anchor_cites_slot_eee_and_canonical_helper():
    result = apply_real(num_frames=2, target_resolution=(64, 48), use_mlx=False)
    anchor = result["per_pixel_real_video_remediation_anchor"]
    assert anchor["slot_eee_audit_axis_a_verdict"] == "PARTIAL_remediated"
    assert anchor["slot_eee_audit_axis_c_verdict"] == "FAIL_remediated"
    assert (
        anchor["canonical_helper_module"]
        == "tac.inverse_steganalysis_real_video_mlx"
    )
    assert (
        anchor["canonical_helper_function"]
        == "compute_hugo_per_pixel_spam_delta_mlx"
    )
    # Slot EEE audit anchor must be cited per CLAUDE.md "Apples-to-apples".
    assert "slot_eee" in anchor["slot_eee_audit_anchor"].lower()
    # Sister Slot YY HILL reference pattern commit must be cited.
    assert anchor["sister_slot_yy_hill_reference_pattern_commit"] == "32a70c051"


# -----------------------------------------------------------------------------
# CANONICAL: truncation_t variants (canonical 2 / 3 / 4 per Pevný-Bas-Fridrich)
# -----------------------------------------------------------------------------


@_SKIP_NO_REAL_VIDEO
@pytest.mark.parametrize("truncation_t", [2, 3, 4])
def test_bind_helper_truncation_t_variants_run_clean(truncation_t):
    result = apply_real(
        num_frames=2,
        target_resolution=(64, 48),
        use_mlx=False,
        truncation_t=truncation_t,
    )
    assert result["smoke_result"]["n_frames_decoded"] == 2


# -----------------------------------------------------------------------------
# CANONICAL: __all__ exports the new bind helper
# -----------------------------------------------------------------------------


def test_module_all_exports_new_bind_helper():
    import tac.composition.hugo_canonical_inverse_steganalysis_pevny_filler_bas_2010 as mod
    assert (
        "apply_hugo_canonical_real_per_pixel_spam_delta_via_canonical_real_video_mlx_to_pr110_archive"
        in mod.__all__
    )
    assert "STRATEGY_PER_PIXEL_REAL_SPAM_DELTA_MLX" in mod.__all__


def test_module_all_preserves_existing_per_pair_exports():
    """Regression: existing exports remain in __all__."""
    import tac.composition.hugo_canonical_inverse_steganalysis_pevny_filler_bas_2010 as mod
    assert (
        "apply_hugo_canonical_spam_feature_cost_matrix_to_pr110_archive"
        in mod.__all__
    )
    assert (
        "compute_hugo_canonical_spam_feature_cost_matrix_for_pr110_catalog"
        in mod.__all__
    )
    assert "HUGOConfig" in mod.__all__
    assert "HUGOSPAMFeatureStrategy" in mod.__all__
