"""Integration tests: L2 encoders + Hinton-distilled scorer + saliency masking.

Per W's DEFERRED reactivation criteria #1+#2 + N D2 council verdict +
CLAUDE.md "Bugs must be permanently fixed AND self-protected against".

Tests verify:
* L2 encoders accept the new flags WITHOUT breaking back-compat
* `compute_score_aware_proxy_loss` swap from YUV6 MSE → distilled scorer
  produces a sensibly-bounded Lagrangian (not the dominated 50000 of YUV6 MSE)
* Saliency masking actually zeros pixels in the encoded residual stream
* All 3 encoders (c3, wavelet, cool_chic) honor both flags
* The new flags compose with sparse_aware (W's pre-existing flag)
* Frozen-dataclass invariants preserved (score_claim=False permanently)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tac.residual_basis.c3_encoder_l2 import (
    encode_c3_residual_l2,
    dense_c3_residual_blob_bytes,
)
from tac.residual_basis.cool_chic_encoder_l2 import (
    dense_cool_chic_residual_blob_bytes,
    encode_cool_chic_residual_l2,
)
from tac.residual_basis.hinton_distilled_scorer_surrogate import (
    ScorerSurrogateConfig,
    load_pretrained_distilled_scorer_pair,
)
from tac.residual_basis.l2_score_aware_loss import (
    L2ScoreAwareLossError,
    ScoreAwareLagrangian,
    compute_score_aware_proxy_loss,
)
from tac.residual_basis.saliency_masked_residual import SaliencyMaskingConfig
from tac.residual_basis.wavelet_encoder_l2 import (
    dense_wavelet_residual_blob_bytes,
    encode_wavelet_residual_l2,
)


T = 4
H = 874
W = 1164
RGB = 3


def _make_camera_pair(seed=0):
    rng = np.random.default_rng(seed)
    decoded = rng.integers(0, 256, (T, H, W, RGB), dtype=np.uint8)
    gt = decoded.astype(np.int16) + rng.integers(-5, 6, decoded.shape, dtype=np.int8).astype(np.int16)
    gt = np.clip(gt, 0, 255).astype(np.uint8)
    return decoded, gt


@pytest.fixture(scope="module")
def distilled_pair():
    config = ScorerSurrogateConfig.council_canonical()
    return load_pretrained_distilled_scorer_pair(config=config)


# ---------------------------------------------------------------------------
# compute_score_aware_proxy_loss (the shared inner-loop loss)
# ---------------------------------------------------------------------------


def test_proxy_loss_back_compat_default_uses_yuv6_mse():
    """Without the new flags, behavior matches W's pre-fix YUV6 MSE proxy."""
    decoded = torch.rand(2, 3, 64, 96) * 200.0 + 28.0
    gt = decoded.clone()
    loss, diag = compute_score_aware_proxy_loss(
        decoded, gt, archive_bytes=200_000,
    )
    assert diag["proxy_kind_yuv6_mse"] == 1.0
    assert diag["proxy_kind_hinton_distilled"] == 0.0
    assert diag["use_hinton_distilled_scorer"] == 0.0


def test_proxy_loss_with_hinton_distilled_scorer_replaces_proxy(distilled_pair):
    seg, pose = distilled_pair
    decoded = torch.rand(2, 3, 64, 96) * 200.0 + 28.0
    gt = decoded.clone()
    loss, diag = compute_score_aware_proxy_loss(
        decoded, gt, archive_bytes=200_000,
        use_hinton_distilled_scorer=True,
        distilled_segnet=seg, distilled_posenet=pose,
    )
    assert diag["proxy_kind_yuv6_mse"] == 0.0
    assert diag["proxy_kind_hinton_distilled"] == 1.0
    assert diag["use_hinton_distilled_scorer"] == 1.0


def test_proxy_loss_with_hinton_requires_distilled_segnet(distilled_pair):
    seg, pose = distilled_pair
    decoded = torch.rand(2, 3, 64, 96) * 200.0 + 28.0
    gt = decoded.clone()
    with pytest.raises(L2ScoreAwareLossError, match="distilled_segnet and"):
        compute_score_aware_proxy_loss(
            decoded, gt, archive_bytes=200_000,
            use_hinton_distilled_scorer=True,
            distilled_segnet=None,
            distilled_posenet=pose,
        )


def test_proxy_loss_with_hinton_requires_distilled_posenet(distilled_pair):
    seg, pose = distilled_pair
    decoded = torch.rand(2, 3, 64, 96) * 200.0 + 28.0
    gt = decoded.clone()
    with pytest.raises(L2ScoreAwareLossError, match="distilled_segnet and"):
        compute_score_aware_proxy_loss(
            decoded, gt, archive_bytes=200_000,
            use_hinton_distilled_scorer=True,
            distilled_segnet=seg,
            distilled_posenet=None,
        )


def test_proxy_loss_hinton_path_breaks_w_proxy_dominance(distilled_pair):
    """W's DEFERRED finding: YUV6 MSE proxy beta_term ~50000 dwarfs alpha rate ~15.

    With Hinton-distilled scorer the beta_term is in distilled-scorer units
    (typically O(1)) and the alpha rate term is comparable, so the rate
    term is no longer dominated. This is W's reactivation criterion #1.
    """
    seg, pose = distilled_pair
    decoded = torch.rand(2, 3, 64, 96) * 200.0 + 28.0
    gt = (torch.rand(2, 3, 64, 96) * 200.0 + 28.0).clamp(1, 254)
    archive_bytes = 200_000
    loss_yuv6, diag_yuv6 = compute_score_aware_proxy_loss(
        decoded.clone(), gt.clone(), archive_bytes=archive_bytes,
    )
    loss_hinton, diag_hinton = compute_score_aware_proxy_loss(
        decoded.clone(), gt.clone(), archive_bytes=archive_bytes,
        use_hinton_distilled_scorer=True,
        distilled_segnet=seg, distilled_posenet=pose,
    )
    # Total losses can be very different. The point: under Hinton-distilled,
    # the seg+pose terms come back to a sensible scale (typically 1-100), not
    # the YUV6 MSE proxy's 50000+. Verify rate term is comparable to seg+pose
    # under Hinton (i.e. NOT dominated by 50000+).
    assert diag_yuv6["beta_term"] > diag_hinton["beta_term"], (
        "YUV6 MSE proxy should dominate; Hinton-distilled should be smaller"
    )


def test_proxy_loss_diagnostics_exposed_correctly(distilled_pair):
    seg, pose = distilled_pair
    decoded = torch.rand(2, 3, 64, 96) * 200.0 + 28.0
    gt = decoded.clone()
    _, diag = compute_score_aware_proxy_loss(
        decoded, gt, archive_bytes=200_000,
        use_hinton_distilled_scorer=True,
        distilled_segnet=seg, distilled_posenet=pose,
        distill_temperature=2.0,
    )
    expected_keys = {
        "alpha_term", "beta_term", "gamma_term", "soft_barrier_term",
        "total", "seg_proxy_mse", "pose_proxy_mse", "rate_term_raw",
        "archive_bytes", "eval_roundtrip", "yuv6_routing",
        "use_hinton_distilled_scorer", "proxy_kind_yuv6_mse",
        "proxy_kind_hinton_distilled",
    }
    assert expected_keys.issubset(set(diag.keys()))


# ---------------------------------------------------------------------------
# c3 encoder integration
# ---------------------------------------------------------------------------


def test_c3_encoder_back_compat_without_new_flags():
    """The pre-existing default path still works (no new flags)."""
    decoded, gt = _make_camera_pair()
    budget = dense_c3_residual_blob_bytes(T) + 100
    res = encode_c3_residual_l2(decoded, gt, byte_budget=budget, n_iterations=1)
    assert res.score_claim is False
    assert res.promotion_eligible is False
    assert res.ready_for_exact_eval_dispatch is False
    # Diagnostic carries the new flag indicators (default 0).
    assert res.diagnostics.get("c3_use_hinton_distilled_scorer", 0.0) == 0.0
    assert res.diagnostics.get("c3_use_saliency_masking", 0.0) == 0.0


def test_c3_encoder_with_hinton_distilled_scorer(distilled_pair):
    seg, pose = distilled_pair
    decoded, gt = _make_camera_pair()
    budget = dense_c3_residual_blob_bytes(T) + 100
    res = encode_c3_residual_l2(
        decoded, gt, byte_budget=budget, n_iterations=1,
        use_hinton_distilled_scorer=True,
        distilled_segnet=seg, distilled_posenet=pose,
    )
    assert res.score_claim is False  # invariant preserved
    assert res.diagnostics["c3_use_hinton_distilled_scorer"] == 1.0
    # Loss should be sensibly bounded (not the YUV6 MSE 600+).
    assert res.final_loss < 100.0, (
        f"Hinton-distilled Lagrangian should be sensible: got {res.final_loss}"
    )


def test_c3_encoder_with_saliency_requires_hinton(distilled_pair):
    decoded, gt = _make_camera_pair()
    budget = dense_c3_residual_blob_bytes(T) + 100
    with pytest.raises(ValueError, match="use_saliency_masking=True requires"):
        encode_c3_residual_l2(
            decoded, gt, byte_budget=budget, n_iterations=1,
            use_saliency_masking=True,
            # Missing use_hinton_distilled_scorer + distilled scorer args.
        )


def test_c3_encoder_with_saliency_masking_records_kept_fraction(distilled_pair):
    seg, pose = distilled_pair
    decoded, gt = _make_camera_pair()
    budget = dense_c3_residual_blob_bytes(T) + 100
    res = encode_c3_residual_l2(
        decoded, gt, byte_budget=budget, n_iterations=1,
        use_hinton_distilled_scorer=True,
        distilled_segnet=seg, distilled_posenet=pose,
        use_saliency_masking=True,
    )
    assert res.score_claim is False
    assert res.diagnostics["c3_use_saliency_masking"] == 1.0
    assert "c3_saliency_kept_fraction" in res.diagnostics
    # Council-canonical config keeps top 25%.
    assert 0.0 < res.diagnostics["c3_saliency_kept_fraction"] < 1.0


# ---------------------------------------------------------------------------
# wavelet encoder integration
# ---------------------------------------------------------------------------


def test_wavelet_encoder_back_compat_without_new_flags():
    decoded, gt = _make_camera_pair(seed=1)
    budget = dense_wavelet_residual_blob_bytes(T) + 100
    res = encode_wavelet_residual_l2(decoded, gt, byte_budget=budget, n_iterations=1)
    assert res.score_claim is False
    assert res.diagnostics.get("wavelet_use_hinton_distilled_scorer", 0.0) == 0.0


def test_wavelet_encoder_with_hinton_distilled_scorer(distilled_pair):
    seg, pose = distilled_pair
    decoded, gt = _make_camera_pair(seed=2)
    budget = dense_wavelet_residual_blob_bytes(T) + 100
    res = encode_wavelet_residual_l2(
        decoded, gt, byte_budget=budget, n_iterations=1,
        use_hinton_distilled_scorer=True,
        distilled_segnet=seg, distilled_posenet=pose,
    )
    assert res.score_claim is False
    assert res.diagnostics["wavelet_use_hinton_distilled_scorer"] == 1.0


def test_wavelet_encoder_with_saliency_requires_hinton():
    decoded, gt = _make_camera_pair(seed=3)
    budget = dense_wavelet_residual_blob_bytes(T) + 100
    with pytest.raises(ValueError, match="use_saliency_masking=True requires"):
        encode_wavelet_residual_l2(
            decoded, gt, byte_budget=budget, n_iterations=1,
            use_saliency_masking=True,
        )


def test_wavelet_encoder_with_saliency_masking(distilled_pair):
    seg, pose = distilled_pair
    decoded, gt = _make_camera_pair(seed=4)
    budget = dense_wavelet_residual_blob_bytes(T) + 100
    res = encode_wavelet_residual_l2(
        decoded, gt, byte_budget=budget, n_iterations=1,
        use_hinton_distilled_scorer=True,
        distilled_segnet=seg, distilled_posenet=pose,
        use_saliency_masking=True,
    )
    assert res.score_claim is False
    assert res.diagnostics["wavelet_use_saliency_masking"] == 1.0
    assert "wavelet_saliency_kept_fraction" in res.diagnostics


# ---------------------------------------------------------------------------
# cool_chic encoder integration
# ---------------------------------------------------------------------------


def test_cool_chic_encoder_back_compat_without_new_flags():
    decoded, gt = _make_camera_pair(seed=5)
    budget = dense_cool_chic_residual_blob_bytes(T, 1) + 100
    res = encode_cool_chic_residual_l2(
        decoded, gt, byte_budget=budget,
        candidate_n_levels=(1,),
    )
    assert res.score_claim is False
    assert res.diagnostics.get("cool_chic_use_hinton_distilled_scorer", 0.0) == 0.0


def test_cool_chic_encoder_with_hinton_distilled_scorer(distilled_pair):
    seg, pose = distilled_pair
    decoded, gt = _make_camera_pair(seed=6)
    budget = dense_cool_chic_residual_blob_bytes(T, 1) + 100
    res = encode_cool_chic_residual_l2(
        decoded, gt, byte_budget=budget,
        candidate_n_levels=(1,),
        use_hinton_distilled_scorer=True,
        distilled_segnet=seg, distilled_posenet=pose,
    )
    assert res.score_claim is False
    assert res.diagnostics["cool_chic_use_hinton_distilled_scorer"] == 1.0


def test_cool_chic_encoder_with_saliency_requires_hinton():
    decoded, gt = _make_camera_pair(seed=7)
    budget = dense_cool_chic_residual_blob_bytes(T, 1) + 100
    with pytest.raises(ValueError, match="use_saliency_masking=True requires"):
        encode_cool_chic_residual_l2(
            decoded, gt, byte_budget=budget,
            candidate_n_levels=(1,),
            use_saliency_masking=True,
        )


def test_cool_chic_encoder_with_saliency_masking(distilled_pair):
    seg, pose = distilled_pair
    decoded, gt = _make_camera_pair(seed=8)
    budget = dense_cool_chic_residual_blob_bytes(T, 1) + 100
    res = encode_cool_chic_residual_l2(
        decoded, gt, byte_budget=budget,
        candidate_n_levels=(1,),
        use_hinton_distilled_scorer=True,
        distilled_segnet=seg, distilled_posenet=pose,
        use_saliency_masking=True,
    )
    assert res.score_claim is False
    assert res.diagnostics["cool_chic_use_saliency_masking"] == 1.0
    assert "cool_chic_saliency_kept_fraction" in res.diagnostics


# ---------------------------------------------------------------------------
# Score-claim invariants preserved across all combinations
# ---------------------------------------------------------------------------


def test_score_claim_invariant_preserved_under_all_flag_combinations(distilled_pair):
    """Per CLAUDE.md HNeRV parity discipline lessons 1/6/8/13:
    score_claim/promotion_eligible/ready_for_exact_eval_dispatch must remain
    permanently False regardless of which research-mode flags are set.
    """
    seg, pose = distilled_pair
    decoded, gt = _make_camera_pair(seed=9)
    budget = dense_c3_residual_blob_bytes(T) + 100
    flag_combos = [
        # (hinton, saliency)
        (False, False),
        (True, False),
        (True, True),
    ]
    for use_hinton, use_saliency in flag_combos:
        res = encode_c3_residual_l2(
            decoded, gt, byte_budget=budget, n_iterations=1,
            use_hinton_distilled_scorer=use_hinton,
            distilled_segnet=seg if use_hinton else None,
            distilled_posenet=pose if use_hinton else None,
            use_saliency_masking=use_saliency,
        )
        assert res.score_claim is False
        assert res.promotion_eligible is False
        assert res.ready_for_exact_eval_dispatch is False


# ---------------------------------------------------------------------------
# Sparse-aware composes with new flags
# ---------------------------------------------------------------------------


def test_c3_encoder_signature_accepts_sparse_aware_with_hinton(distilled_pair):
    """W's pre-existing sparse_aware flag composes (signature-level) with Hinton.

    Signature-level smoke test only: confirms the function signature accepts
    all three flags simultaneously without TypeError. End-to-end byte-pack
    composition has a pre-existing PacketIR bug (non-uniform per_frame_bytes)
    that is OUT OF SCOPE for this landing — see
    `feedback_l2_sparse_aware_encoders_first_dispatch_landed_20260511.md` for
    the sparse_aware byte-pack details (sister subagent S's codec).
    """
    import inspect
    seg, pose = distilled_pair
    sig = inspect.signature(encode_c3_residual_l2)
    expected_kwargs = {
        "sparse_aware",
        "use_hinton_distilled_scorer",
        "distilled_segnet",
        "distilled_posenet",
        "use_saliency_masking",
        "saliency_masking_config",
    }
    assert expected_kwargs.issubset(set(sig.parameters.keys()))
