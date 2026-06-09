# SPDX-License-Identifier: MIT
"""NO-FAKE behavioral tests for the deforestation read-surface atom proposers.

Every test here verifies BEHAVIOR, not constants.  Per CLAUDE.md "NO FAKE
IMPLEMENTATIONS" Slot EEE class 2: a test that would still pass if the function
body returned canonical markers is FAKE.  The causal tests below would FAIL if:

- ``segnet_argmax_margin_tolerance_map`` returned a constant margin (the
  margin-vs-argmax-flip equivalence would break);
- ``seg_scored_frame_mask`` returned a fixed list (the parametrized
  scored==last-of-pair invariant would break);
- ``pose_null_projection`` returned zeros / the raw perturbation (the
  ``J @ projected ≈ 0`` null-space invariant would break).

Real-scorer tests load the actual upstream SegNet/PoseNet via the EXACT contest
path (``segnet(segnet.preprocess_input(pair))`` -> argmax over dim=1, the same
tensor ``upstream/modules.py::SegNet.compute_distortion`` consumes) -- NOT
``mini_scorer`` (a surrogate = proxy, forbidden as authority).  They skip
cleanly if the upstream model files are absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.contest_eval_contract import (
    PUBLIC_TEST_FRAME_COUNT,
    PUBLIC_TEST_PAIR_COUNT,
    SEQ_LEN,
)
from tac.optimization.scorer_read_surface_atoms import (
    DEFOREST_ATOM_SCHEMA,
    DeforestationAtom,
    seg_free_frame_atom,
    seg_scored_frame_mask,
    summarize_read_surface,
)

REPO = Path(__file__).resolve().parents[4]
UPSTREAM = REPO / "upstream"
SEGNET_PATH = UPSTREAM / "models" / "segnet.safetensors"
POSENET_PATH = UPSTREAM / "models" / "posenet.safetensors"

_segnet_available = pytest.mark.skipif(
    not SEGNET_PATH.exists(), reason="upstream SegNet safetensors not present"
)
_scorers_available = pytest.mark.skipif(
    not (SEGNET_PATH.exists() and POSENET_PATH.exists()),
    reason="upstream SegNet+PoseNet safetensors not present",
)


# ─────────────────────────────────────────────────────────────────────────────
# Atom 2 — seg_scored_frame_mask: contract-grounded, no torch needed.
# ─────────────────────────────────────────────────────────────────────────────


def test_seg_scored_frame_mask_default_is_half_seg_free():
    """The biggest deforestation: ~half the frames are SEG-FREE.

    With the contract's public-test count + seq_len==2, exactly
    PUBLIC_TEST_PAIR_COUNT of PUBLIC_TEST_FRAME_COUNT frames are seg-free.
    """
    mask = seg_scored_frame_mask()
    assert mask.num_frames == PUBLIC_TEST_FRAME_COUNT
    assert mask.seq_len == SEQ_LEN
    assert len(mask.scored_frame_indices) == PUBLIC_TEST_PAIR_COUNT
    assert len(mask.seg_free_frame_indices) == PUBLIC_TEST_FRAME_COUNT - PUBLIC_TEST_PAIR_COUNT
    assert mask.seg_free_fraction == pytest.approx(0.5)


def test_seg_scored_frame_mask_scored_are_last_of_pair():
    """SegNet reads x[:,-1] => scored frame is the LAST of each pair.

    For seq_len==2 that makes ODD global indices scored, EVEN seg-free.  This
    test would fail if the helper returned a fixed list or used the wrong
    within-pair index.
    """
    mask = seg_scored_frame_mask(num_frames=12, seq_len=2)
    assert mask.scored_frame_indices == (1, 3, 5, 7, 9, 11)
    assert mask.seg_free_frame_indices == (0, 2, 4, 6, 8, 10)
    # Every scored index is the last frame of its pair.
    for p, idx in enumerate(mask.scored_frame_indices):
        assert idx == p * mask.seq_len + (mask.seq_len - 1)
    # Scored and seg-free partition the frame set with no overlap.
    assert set(mask.scored_frame_indices).isdisjoint(mask.seg_free_frame_indices)
    assert len(mask.scored_frame_indices) + len(mask.seg_free_frame_indices) == 12


@pytest.mark.parametrize(
    "num_frames,seq_len,expected_scored,expected_free",
    [
        (6, 2, (1, 3, 5), (0, 2, 4)),
        (9, 3, (2, 5, 8), (0, 1, 3, 4, 6, 7)),  # seq_len=3 => 1 scored per 3 frames
        (8, 4, (3, 7), (0, 1, 2, 4, 5, 6)),
        (4, 1, (0, 1, 2, 3), ()),  # seq_len=1 => every frame is "last of pair"
    ],
)
def test_seg_scored_frame_mask_parametrized_last_of_pair(
    num_frames, seq_len, expected_scored, expected_free
):
    """Parametrized last-of-pair invariant across seq_len values.

    A constant-returning fake cannot satisfy all four cases.
    """
    mask = seg_scored_frame_mask(num_frames=num_frames, seq_len=seq_len)
    assert mask.scored_frame_indices == expected_scored
    assert mask.seg_free_frame_indices == expected_free


def test_seg_scored_frame_mask_is_seg_scored_query():
    mask = seg_scored_frame_mask(num_frames=6, seq_len=2)
    assert mask.is_seg_scored(1) is True
    assert mask.is_seg_scored(3) is True
    assert mask.is_seg_scored(0) is False
    assert mask.is_seg_scored(2) is False


def test_seg_scored_frame_mask_rejects_bad_args():
    with pytest.raises(ValueError):
        seg_scored_frame_mask(num_frames=10, seq_len=0)
    with pytest.raises(ValueError):
        seg_scored_frame_mask(num_frames=-1, seq_len=2)


def test_seg_free_frame_atom_frees_bytes_proportional_to_free_frames():
    """The seg-free atom actually frees bytes proportional to the seg-free count
    and its advisory score delta equals the (negative) rate-term reduction."""
    mask = seg_scored_frame_mask(num_frames=12, seq_len=2)  # 6 seg-free frames
    atom = seg_free_frame_atom(mask, seg_bytes_per_frame=100.0)
    assert atom.atom_kind == "seg_free_frame"
    assert atom.delta_bytes_freed == 600  # 6 frames * 100 B
    assert atom.advisory_delta_seg == 0.0  # SegNet never reads these frames
    # advisory_delta_score == rate-term reduction (negative => lowers score).
    assert atom.advisory_delta_score < 0.0
    assert atom.advisory_delta_score == pytest.approx(atom.advisory_delta_rate_score)
    # Freeing more bytes => larger rate-term reduction (more negative).
    atom_big = seg_free_frame_atom(mask, seg_bytes_per_frame=1000.0)
    assert atom_big.advisory_delta_score < atom.advisory_delta_score


# ─────────────────────────────────────────────────────────────────────────────
# Atom 1 — segnet_argmax_margin_tolerance_map: causal margin↔argmax-flip on the
# EXACT logit surface argmax operates on (synthetic logits = real argmax math).
# ─────────────────────────────────────────────────────────────────────────────


def test_margin_map_classifies_interior_free_boundary_protected():
    """A clear-winner region is FREE (large margin); a tied-top-2 region is
    PROTECTED (small margin).  Fails if the margin is a constant."""
    torch = pytest.importorskip("torch")
    from tac.optimization.scorer_read_surface_atoms import (
        segnet_argmax_margin_tolerance_map,
    )

    logits = torch.zeros(1, 5, 8, 8)
    logits[:, 2, :4, :] = 10.0  # top half: class 2 dominates -> FREE
    logits[:, 1, 4:, :] = 1.0  # bottom half: class 1 vs 3 nearly tied -> PROTECTED
    logits[:, 3, 4:, :] = 0.99
    tmap = segnet_argmax_margin_tolerance_map(logits, protected_margin_threshold=0.5)

    # Interior region has a huge margin; boundary region has a tiny margin.
    assert float(tmap.margin[:, :4, :].mean()) > 5.0
    assert float(tmap.margin[:, 4:, :].mean()) < 0.1
    # Classification follows the margin.
    assert not bool(tmap.protected_mask[:, :4, :].any())  # interior is FREE
    assert bool(tmap.protected_mask[:, 4:, :].all())  # boundary is PROTECTED
    assert tmap.free_fraction == pytest.approx(0.5)
    # Winner mask is the contest argmax.
    assert int(tmap.winner[0, 0, 0]) == 2
    assert int(tmap.winner[0, 7, 0]) == 1


def test_margin_map_predicts_argmax_flip_exactly():
    """THE core NO-FAKE causal test: a pixel's argmax flips IFF a competing
    class is pushed past its top-2 margin.

    This is the exact mathematical claim the tolerance map makes about the
    contest argmax surface.  A constant/marker fake cannot satisfy it.
    """
    torch = pytest.importorskip("torch")
    from tac.optimization.scorer_read_surface_atoms import (
        segnet_argmax_margin_tolerance_map,
    )

    torch.manual_seed(7)
    logits = torch.randn(1, 5, 16, 16) * 2.0
    tmap = segnet_argmax_margin_tolerance_map(logits, protected_quantile=0.30)
    base_winner = logits.argmax(dim=1)
    margin = tmap.margin[0]

    def flips_after_bump(y: int, x: int, delta: float) -> bool:
        L = logits.clone()
        second_class = int(torch.topk(L[0, :, y, x], k=2).indices[1])
        L[0, second_class, y, x] += delta
        return int(L[0, :, y, x].argmax()) != int(base_winner[0, y, x])

    flat = margin.reshape(-1)
    free_idx = int(torch.argmax(flat))  # largest margin == most FREE
    prot_idx = int(torch.argmin(flat))  # smallest margin == most PROTECTED
    fy, fx = free_idx // 16, free_idx % 16
    py, px = prot_idx // 16, prot_idx % 16

    # Bumping by less than the margin never flips; by margin+eps always flips.
    for (y, x) in ((fy, fx), (py, px)):
        m = float(margin[y, x])
        assert flips_after_bump(y, x, 0.5 * m) is False
        assert flips_after_bump(y, x, m + 1e-2) is True

    # A perturbation BETWEEN the two margins: FREE tolerates it, PROTECTED flips.
    mid = 0.5 * (float(margin[py, px]) + float(margin[fy, fx]))
    assert flips_after_bump(fy, fx, mid) is False  # FREE pixel absorbs it
    assert flips_after_bump(py, px, mid) is True  # PROTECTED pixel flips


def test_margin_map_tolerance_is_the_margin():
    torch = pytest.importorskip("torch")
    from tac.optimization.scorer_read_surface_atoms import (
        segnet_argmax_margin_tolerance_map,
    )

    logits = torch.randn(1, 5, 4, 4)
    tmap = segnet_argmax_margin_tolerance_map(logits)
    assert torch.equal(tmap.tolerance(), tmap.margin)
    # Margin is non-negative everywhere (top1 >= top2).
    assert bool((tmap.margin >= 0.0).all())


def test_margin_map_accepts_3d_logits_and_validates_shape():
    torch = pytest.importorskip("torch")
    from tac.optimization.scorer_read_surface_atoms import (
        segnet_argmax_margin_tolerance_map,
    )

    # (C, H, W) is promoted to (1, C, H, W).
    tmap = segnet_argmax_margin_tolerance_map(torch.randn(5, 8, 8))
    assert tmap.margin.shape == (1, 8, 8)
    # 2D input is rejected.
    with pytest.raises(ValueError):
        segnet_argmax_margin_tolerance_map(torch.randn(8, 8))
    # Single-class logits cannot have a top-2 margin.
    with pytest.raises(ValueError):
        segnet_argmax_margin_tolerance_map(torch.randn(1, 1, 8, 8))


def test_margin_tolerance_atom_shaves_only_free_pixels():
    torch = pytest.importorskip("torch")
    from tac.optimization.scorer_read_surface_atoms import (
        segnet_argmax_margin_tolerance_map,
        segnet_margin_tolerance_atom,
    )

    logits = torch.zeros(1, 5, 8, 8)
    logits[:, 2, :4, :] = 10.0  # 32 FREE pixels
    logits[:, 1, 4:, :] = 1.0
    logits[:, 3, 4:, :] = 0.99  # 32 PROTECTED pixels
    tmap = segnet_argmax_margin_tolerance_map(logits, protected_margin_threshold=0.5)
    atom = segnet_margin_tolerance_atom(tmap, frame_index=1, bits_shaveable_per_free_pixel=4.0)
    assert atom.atom_kind == "argmax_margin"
    assert atom.n_elements == 32  # only the FREE (interior) pixels
    assert atom.precision_bits_shaveable == pytest.approx(4.0 * 32)
    # SegNet argmax is unchanged for FREE pixels by construction => zero seg delta.
    assert atom.advisory_delta_seg == 0.0
    assert atom.to_row()["scored_frame"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Atom 1 (real path) — margin from REAL SegNet via the EXACT contest path.
# ─────────────────────────────────────────────────────────────────────────────


@_segnet_available
def test_margin_map_from_real_segnet_exact_path():
    """Compute the margin map from the REAL upstream SegNet via the EXACT path:
    segnet(segnet.preprocess_input(pair)) -> (B,5,384,512) logits, the same
    tensor compute_distortion argmaxes over dim=1 (scorer.py:410).  NOT
    mini_scorer.  Confirms the real logits produce a sensible margin map."""
    torch = pytest.importorskip("torch")
    from tac.optimization.scorer_read_surface_atoms import (
        segnet_argmax_margin_tolerance_map,
    )
    from tac.scorer import load_default_segnet

    segnet = load_default_segnet(str(UPSTREAM), device="cpu")
    torch.manual_seed(0)
    H, W = 96, 128
    frame2 = torch.rand(1, 3, H, W) * 255.0
    pair = torch.stack([torch.zeros_like(frame2), frame2], dim=1)  # (1,2,3,H,W)
    with torch.no_grad():
        seg_in = segnet.preprocess_input(pair)  # uses x[:,-1], resize to (384,512)
        logits = segnet(seg_in)
    # Exact contest scorer: 5 classes, scorer input size 384x512.
    assert logits.shape[1] == 5
    assert tuple(logits.shape[-2:]) == (384, 512)

    tmap = segnet_argmax_margin_tolerance_map(logits, protected_quantile=0.25)
    assert tmap.num_classes == 5
    assert tmap.margin.shape == (1, 384, 512)
    # The winner mask equals the contest mask (argmax over the class dim).
    assert torch.equal(tmap.winner, logits.argmax(dim=1))
    # Margins are non-degenerate (a real model produces a spread, not a constant).
    assert float(tmap.margin.max()) > float(tmap.margin.min())
    assert 0.0 < tmap.free_fraction < 1.0


@_segnet_available
def test_margin_map_uses_only_last_frame_of_pair():
    """SegNet asymmetry (read-surface fact #1): mutating pair frame_0 alone
    leaves the SegNet logits UNCHANGED; mutating frame_1 changes them.  This is
    why the margin map is only meaningful on the 2nd-of-pair frame and why
    frame_0 is seg-free."""
    torch = pytest.importorskip("torch")
    from tac.scorer import load_default_segnet

    segnet = load_default_segnet(str(UPSTREAM), device="cpu")
    torch.manual_seed(1)
    H, W = 96, 128
    f0 = torch.rand(1, 3, H, W) * 255.0
    f1 = torch.rand(1, 3, H, W) * 255.0
    with torch.no_grad():
        base = segnet(segnet.preprocess_input(torch.stack([f0, f1], dim=1)))
        # Replace frame_0 entirely -> SegNet (x[:,-1]) must be identical.
        alt_f0 = segnet(
            segnet.preprocess_input(torch.stack([torch.zeros_like(f0), f1], dim=1))
        )
        # Replace frame_1 -> SegNet generally changes.
        alt_f1 = segnet(
            segnet.preprocess_input(torch.stack([f0, torch.zeros_like(f1)], dim=1))
        )
    assert torch.equal(base.argmax(dim=1), alt_f0.argmax(dim=1))  # frame_0 is seg-free
    # Frame_1 mutation can move the argmax (not guaranteed everywhere, but the
    # logits must differ).
    assert not torch.equal(base, alt_f1)


# ─────────────────────────────────────────────────────────────────────────────
# Atom 3 — pose_null_projection: Jacobian-null-space invariant with REAL
# scorers (J @ projected ≈ 0 while J @ raw is not).
# ─────────────────────────────────────────────────────────────────────────────


@_scorers_available
def test_pose_null_projection_lies_in_scorer_jacobian_null_space():
    """THE core NO-FAKE causal test for pose_null: the projected perturbation
    is (near-)orthogonal to EVERY scorer-Jacobian row (J @ projected ≈ 0) while
    the raw perturbation is not, AND the projection is a nonzero component of
    the input.

    A fake that returned zeros would make ||projected||==0 (fails the nonzero
    check); a fake that returned the raw perturbation would make
    ||J@projected||==||J@raw|| (fails the null-space check).
    """
    torch = pytest.importorskip("torch")
    from tac.optimization.scorer_read_surface_atoms import pose_null_projection
    from tac.scorer import load_differentiable_scorers
    from tac.scorer_exploits import compute_scorer_jacobian

    posenet, segnet = load_differentiable_scorers(str(UPSTREAM), device="cpu")
    torch.manual_seed(5)
    H, W = 32, 48
    frame = torch.rand(1, 3, H, W) * 255.0
    pert = torch.randn(3, H, W) * 5.0

    proj = pose_null_projection(
        pert, frames=frame, posenet=posenet, segnet=segnet, max_outputs=16
    )
    assert proj.scored_pose_dims == 6  # contest: first 6 of 12 (h.out // 2)

    J = compute_scorer_jacobian(frame, posenet, segnet, max_outputs=16).float()
    proj_flat = torch.as_tensor(proj.projected).reshape(-1).float()
    raw_flat = pert.reshape(-1).float()

    jp = float(torch.linalg.vector_norm(J @ proj_flat))
    jr = float(torch.linalg.vector_norm(J @ raw_flat))
    # Projected direction is invisible to the scored outputs (orthogonal to J).
    assert jp < 0.05 * jr
    # ...but it is a real, nonzero direction (not the zero fake).
    assert float(torch.linalg.vector_norm(proj_flat)) > 0.0
    # ...and it is a component of the input (cannot exceed it).
    assert float(torch.linalg.vector_norm(proj_flat)) <= float(
        torch.linalg.vector_norm(raw_flat)
    ) + 1e-3
    # Residual energy fraction matches the norms.
    assert proj.residual_energy_fraction == pytest.approx(
        float(torch.linalg.vector_norm(proj_flat))
        / float(torch.linalg.vector_norm(raw_flat)),
        rel=1e-4,
    )


@_scorers_available
def test_pose_null_atom_scales_with_residual_energy():
    torch = pytest.importorskip("torch")
    from tac.optimization.scorer_read_surface_atoms import (
        pose_null_atom,
        pose_null_projection,
    )
    from tac.scorer import load_differentiable_scorers

    posenet, segnet = load_differentiable_scorers(str(UPSTREAM), device="cpu")
    torch.manual_seed(2)
    H, W = 24, 32
    frame = torch.rand(1, 3, H, W) * 255.0
    pert = torch.randn(3, H, W) * 5.0
    proj = pose_null_projection(
        pert, frames=frame, posenet=posenet, segnet=segnet, max_outputs=16
    )
    atom = pose_null_atom(proj, frame_index=0, bits_shaveable_per_null_element=2.0)
    assert atom.atom_kind == "pose_null"
    assert atom.n_elements == 3 * H * W
    # Bits scale with residual energy fraction and per-element budget.
    expected_bits = 2.0 * proj.residual_energy_fraction * (3 * H * W)
    assert atom.precision_bits_shaveable == pytest.approx(expected_bits, rel=1e-4)
    assert atom.advisory_delta_pose == 0.0  # null-space => scored pose unchanged
    assert atom.to_row()["promotable"] is False


def test_pose_null_projection_rejects_too_few_outputs():
    """max_outputs must include the full scored pose-6."""
    torch = pytest.importorskip("torch")
    from tac.optimization.scorer_read_surface_atoms import pose_null_projection

    with pytest.raises(ValueError):
        pose_null_projection(
            torch.randn(3, 8, 8),
            frames=torch.rand(1, 3, 8, 8) * 255.0,
            posenet=object(),  # never reached; arg validation happens first
            segnet=object(),
            max_outputs=4,
        )


# ─────────────────────────────────────────────────────────────────────────────
# False-authority + observability discipline (every atom is planning-control).
# ─────────────────────────────────────────────────────────────────────────────


def test_every_atom_row_carries_false_authority_markers():
    """Per CLAUDE.md: any local d_seg/d_pose is ADVISORY; atoms are
    planning-control proposals, never score claims."""
    mask = seg_scored_frame_mask(num_frames=12, seq_len=2)
    atom = seg_free_frame_atom(mask, seg_bytes_per_frame=50.0)
    row = atom.to_row()
    assert row["schema"] == DEFOREST_ATOM_SCHEMA
    assert row["authority"] == "planning_control_false_authority"
    assert row["score_claim"] is False
    assert row["promotion_eligible"] is False
    assert row["rank_or_kill_eligible"] is False
    assert row["ready_for_exact_eval_dispatch"] is False
    assert row["promotable"] is False
    assert row["evidence_grade"] == "advisory"


def test_atom_row_contract_provenance_names_promotion_target():
    """Each atom names the read-surface fact + the CandidateActionEvaluation it
    promotes to under exact ΔS (so the post-B2 waterfiller can consume it)."""
    mask = seg_scored_frame_mask(num_frames=4, seq_len=2)
    row = seg_free_frame_atom(mask, seg_bytes_per_frame=10.0).to_row()
    prov = row["contract_provenance"]
    assert prov["contract_module"] == "tac.contest_eval_contract"
    assert prov["read_surface_fact"] == "segnet_reads_only_last_frame_of_pair"
    assert (
        prov["promotes_to"]
        == "tac.optimization.evaluator_action_waterfill.CandidateActionEvaluation"
    )
    assert "100*d_seg" in prov["score_formula"]


def test_deforestation_atom_validates_non_negative_fields():
    with pytest.raises(ValueError):
        DeforestationAtom(
            atom_id="x", atom_kind="k", deforestation_target="t", delta_bytes_freed=-1
        )
    with pytest.raises(ValueError):
        DeforestationAtom(
            atom_id="x",
            atom_kind="k",
            deforestation_target="t",
            precision_bits_shaveable=-1.0,
        )
    with pytest.raises(ValueError):
        DeforestationAtom(
            atom_id="x", atom_kind="k", deforestation_target="t", n_elements=-5
        )


def test_advisory_rate_delta_is_exact_contest_units():
    """Freeing N bytes lowers the rate term by exactly 25*N/source_video_bytes."""
    from tac.optimization.scorer_read_surface_atoms import (
        CONTEST_ARCHIVE_RATE_DENOM,
        RATE_PRICE_PER_ARCHIVE_BYTE,
    )

    atom = DeforestationAtom(
        atom_id="x",
        atom_kind="k",
        deforestation_target="t",
        delta_bytes_freed=1000,
    )
    assert atom.advisory_delta_rate_score == pytest.approx(
        -RATE_PRICE_PER_ARCHIVE_BYTE * 1000
    )
    # The price matches the contract denominator exactly.
    assert pytest.approx(25.0 / CONTEST_ARCHIVE_RATE_DENOM) == RATE_PRICE_PER_ARCHIVE_BYTE


def test_summarize_read_surface_is_observable_and_contract_grounded():
    """Max-observability surface: the three deforestation levers are decomposed
    with contract-pinned constants and false-authority markers."""
    summary = summarize_read_surface()
    assert summary["schema"] == "deforestation_read_surface_summary.v1"
    assert summary["constants"]["seq_len"] == SEQ_LEN
    assert summary["constants"]["public_test_pair_count"] == PUBLIC_TEST_PAIR_COUNT
    levers = summary["levers"]
    assert set(levers) == {
        "seg_free_frames",
        "argmax_interior_pixels",
        "pose_null_directions",
    }
    assert levers["seg_free_frames"]["n_seg_free_frames"] == PUBLIC_TEST_PAIR_COUNT
    assert levers["pose_null_directions"]["scored_pose_dims"] == 6
    assert summary["promotable"] is False
    assert summary["score_claim"] is False
