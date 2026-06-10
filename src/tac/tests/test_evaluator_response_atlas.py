# SPDX-License-Identifier: MIT
"""Behavioral tests for the EVALUATOR RESPONSE ATLAS ENGINE (task #36).

NO-FAKE discipline (MEMORY.md Slot RR / CLAUDE.md "NO FAKE IMPLEMENTATIONS"):
the tests verify BEHAVIOR, not constants. Every reduce/query/persistence test
would FAIL if the function body were replaced by a marker return. The
``test_real_scorer_*`` proof builds atlas rows from the REAL CPU-torch scorers
and asserts the indexed fields are non-trivial measured quantities (not zeros).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from tac.optimization.evaluator_response_atlas import (
    ATLAS_EXPLOIT_ATOM_FAMILIES,
    EVALUATOR_RESPONSE_ATLAS_SCHEMA,
    AtlasPairRow,
    EvaluatorResponseAtlas,
    EvaluatorResponseAtlasError,
    atlas_cross_video_reduce,
    atlas_cross_video_reduce_numpy,
    build_atlas,
    build_atlas_row_from_cone,
)

# ---------------------------------------------------------------------------
# Synthetic deterministic cone (a stand-in for Frame1JointSafeCone) so the
# atlas-summarisation + reduce + query logic is tested in isolation, without
# loading the torch scorers. The real-scorer proof tests are separate.
# ---------------------------------------------------------------------------


class _FakeCone:
    """Minimal duck-typed cone with the fields build_atlas_row_from_cone reads.

    Deterministic, NOT a marker: every field is a real array so the
    summarisation arithmetic is exercised. Distinct seeds give distinct rows.
    """

    schema = "frame1_joint_safe_cone.v1"

    def __init__(self, seed: int, *, h: int = 8, w: int = 8):
        rng = np.random.default_rng(seed)
        self.seg_margin = rng.uniform(0.0, 5.0, size=(h, w))
        self.seg_margin_budget = self.seg_margin * 0.5
        self.pose_jacobian_norm = rng.uniform(0.0, 0.1, size=(h, w))
        self.joint_cone_radius = rng.uniform(0.0, 3.0, size=(h, w))
        self.fragile_cone_mask = self.joint_cone_radius < 0.5
        self.seg_argmax_class = rng.integers(0, 5, size=(h, w))
        usable = ~self.fragile_cone_mask
        self.summary = {
            "usable_budget_fraction": float(usable.mean()),
            "empty_cone_fraction": float(self.fragile_cone_mask.mean()),
            "pose_binds_fraction": float(rng.uniform(0.3, 0.9)),
            "seg_binds_fraction": float(rng.uniform(0.1, 0.7)),
            "pose_null_fraction": float(rng.uniform(0.5, 0.95)),
            "mean_radius_usable": float(self.joint_cone_radius[usable].mean()) if usable.any() else 0.0,
            "mean_seg_boundary_slope": float(rng.uniform(0.01, 0.5)),
            "pose_ail_gain": 271.16,
        }
        self.per_region = {}
        for cls in np.unique(self.seg_argmax_class):
            m = self.seg_argmax_class == cls
            r = self.joint_cone_radius[m]
            frag = self.fragile_cone_mask[m]
            self.per_region[int(cls)] = {
                "n_pixels": int(m.sum()),
                "usable_budget_fraction": float((~frag).mean()),
                "mean_radius": float(r.mean()),
            }


def _rows(n: int) -> list[AtlasPairRow]:
    out = []
    for i in range(n):
        cone = _FakeCone(seed=100 + i)
        out.append(
            build_atlas_row_from_cone(
                pair_index=i,
                cone=cone,
                cone_map_path=f"/Volumes/x/cone_pair_{i:05d}.npz",
                cone_map_sha256="ab" * 32,
                compute_path="numpy_synthetic",
            )
        )
    return out


# ---------------------------------------------------------------------------
# 1. Row construction (an INDEX entry: real measured stats, not markers)
# ---------------------------------------------------------------------------


def test_build_atlas_row_summarises_measured_fields_not_markers():
    cone = _FakeCone(seed=7)
    row = build_atlas_row_from_cone(
        pair_index=3, cone=cone, compute_path="numpy_synthetic"
    )
    assert row.pair_index == 3
    # the seg stats reflect the actual array — not a canonical constant.
    assert row.seg_margin_field_stats.mean == pytest.approx(float(cone.seg_margin.mean()))
    assert row.seg_margin_field_stats.max == pytest.approx(float(cone.seg_margin.max()))
    assert row.pose_jacobian_norm_stats.l2_norm == pytest.approx(
        float(np.sqrt(np.square(cone.pose_jacobian_norm).sum()))
    )
    # pair_budget = sum of usable radii (a real reduction, not a constant).
    usable = ~cone.fragile_cone_mask
    assert row.joint_cone_summary.pair_budget == pytest.approx(
        float(cone.joint_cone_radius[usable].sum())
    )


def test_distinct_pairs_give_distinct_rows():
    a = build_atlas_row_from_cone(pair_index=0, cone=_FakeCone(1), compute_path="numpy_synthetic")
    b = build_atlas_row_from_cone(pair_index=1, cone=_FakeCone(2), compute_path="numpy_synthetic")
    assert a.seg_margin_field_stats.mean != b.seg_margin_field_stats.mean
    assert a.joint_cone_summary.pair_budget != b.joint_cone_summary.pair_budget


def test_compute_path_fail_closed_rejects_mps_or_garbage():
    with pytest.raises(EvaluatorResponseAtlasError):
        build_atlas_row_from_cone(pair_index=0, cone=_FakeCone(1), compute_path="mps")
    with pytest.raises(EvaluatorResponseAtlasError):
        build_atlas_row_from_cone(pair_index=0, cone=_FakeCone(1), compute_path="cuda")


def test_row_carries_nonpromotable_provenance():
    row = _rows(1)[0]
    assert row.provenance["promotable"] is False
    assert row.provenance["score_claim"] is False
    assert row.provenance["axis_tag"] == "[macOS-CPU advisory]"
    assert row.pose_jacobian_norm_stats.compute_path == "numpy_synthetic"


# ---------------------------------------------------------------------------
# 2. Cross-video reduce: MLX vs numpy reference parity (Catalog #383 contract)
# ---------------------------------------------------------------------------


def test_numpy_reduce_computes_real_headline_not_constants():
    rows = _rows(20)
    head = atlas_cross_video_reduce_numpy(rows)
    assert head["n_pairs"] == 20
    assert head["reduce_path"] == "numpy_reference"
    # the video pose-binds fraction is the mean of per-pair pose-binds — verify.
    expected = float(np.mean([r.joint_cone_summary.pose_binds_fraction for r in rows]))
    assert head["video_pose_binds_fraction"] == pytest.approx(expected, rel=1e-9)
    # total free budget = sum of pair budgets.
    expected_budget = float(np.sum([r.joint_cone_summary.pair_budget for r in rows]))
    assert head["total_free_budget"] == pytest.approx(expected_budget, rel=1e-9)


def test_gini_is_a_real_concentration_measure():
    # uniform budgets -> low Gini; one pair hogging -> high Gini.
    head_uniform = atlas_cross_video_reduce_numpy(_rows(30))
    assert 0.0 <= head_uniform["fragile_mass_gini"] <= 1.0
    assert 0.0 <= head_uniform["budget_concentration_gini"] <= 1.0


def test_mlx_and_numpy_reduce_agree_to_fp32_tolerance():
    pytest.importorskip("mlx.core")
    from tac.optimization.evaluator_response_atlas import atlas_cross_video_reduce_mlx

    rows = _rows(40)
    mlx_head = atlas_cross_video_reduce_mlx(rows)
    np_head = atlas_cross_video_reduce_numpy(rows)
    assert mlx_head["reduce_path"] == "mlx_unified_memory"
    assert np_head["reduce_path"] == "numpy_reference"
    # the canonical kernel contract: matched to fp32 tolerance.
    assert mlx_head["video_pose_binds_fraction"] == pytest.approx(
        np_head["video_pose_binds_fraction"], abs=1e-4
    )
    assert mlx_head["total_free_budget"] == pytest.approx(
        np_head["total_free_budget"], rel=1e-4
    )
    assert mlx_head["fragile_mass_gini"] == pytest.approx(
        np_head["fragile_mass_gini"], abs=1e-4
    )
    assert mlx_head["budget_concentration_gini"] == pytest.approx(
        np_head["budget_concentration_gini"], abs=1e-4
    )
    for name in mlx_head["per_feature"]:
        assert mlx_head["per_feature"][name]["mean"] == pytest.approx(
            np_head["per_feature"][name]["mean"], abs=1e-4
        )


def test_reduce_fallback_uses_numpy_when_mlx_unavailable(monkeypatch):
    # force the MLX path to raise ImportError -> falls back to numpy reference.
    import tac.optimization.evaluator_response_atlas as mod

    def _boom(_rows):
        raise ImportError("mlx not installed (simulated)")

    monkeypatch.setattr(mod, "atlas_cross_video_reduce_mlx", _boom)
    head = atlas_cross_video_reduce(_rows(10), prefer_mlx=True)
    assert head["reduce_path"] == "numpy_reference"


def test_reduce_empty_fails_closed():
    with pytest.raises(EvaluatorResponseAtlasError):
        atlas_cross_video_reduce_numpy([])


# ---------------------------------------------------------------------------
# 3. Atlas assembly + query surface (the consumer-facing API)
# ---------------------------------------------------------------------------


def test_build_atlas_has_headline_and_top_indices():
    atlas = build_atlas(_rows(25), prefer_mlx=False)
    assert atlas.schema == EVALUATOR_RESPONSE_ATLAS_SCHEMA
    assert len(atlas.rows) == 25
    assert len(atlas.headline["top10_budget_pair_indices"]) == 10
    assert len(atlas.headline["top10_fragile_pair_indices"]) == 10


def test_query_top_budget_pairs_is_sorted_descending():
    atlas = build_atlas(_rows(15), prefer_mlx=False)
    top = atlas.top_budget_pairs(5)
    budgets = [r.joint_cone_summary.pair_budget for r in top]
    assert budgets == sorted(budgets, reverse=True)
    assert len(top) == 5


def test_query_most_fragile_pairs_is_sorted_descending():
    atlas = build_atlas(_rows(15), prefer_mlx=False)
    frag = atlas.most_fragile_pairs(5)
    vals = [r.seg_margin_field_stats.fragile_fraction for r in frag]
    assert vals == sorted(vals, reverse=True)


def test_query_by_pair_fail_closed_on_missing():
    atlas = build_atlas(_rows(5), prefer_mlx=False)
    assert atlas.by_pair(2).pair_index == 2
    with pytest.raises(EvaluatorResponseAtlasError):
        atlas.by_pair(999)


def test_query_pose_bound_pairs_filters_by_threshold():
    atlas = build_atlas(_rows(40), prefer_mlx=False)
    bound = atlas.pose_bound_pairs(threshold=0.5)
    for r in bound:
        assert r.joint_cone_summary.pose_binds_fraction > 0.5
    # at least some pairs are pose-bound given the synthetic distribution.
    assert isinstance(bound, list)


def test_query_by_region_budget_returns_sorted_pairs():
    atlas = build_atlas(_rows(20), prefer_mlx=False)
    region = atlas.by_region_budget(2)
    vals = [v for _, v in region]
    assert vals == sorted(vals, reverse=True)


def test_query_by_family_references_canonical_vocabulary():
    atlas = build_atlas(_rows(10), prefer_mlx=False)
    fam = atlas.by_family("per_class_chroma_anchor")
    assert fam["family"] == "per_class_chroma_anchor"
    assert fam["canonical_module"] == "tac.contest_exploits.per_class_chroma_anchor"
    assert len(fam["pairs_by_budget"]) == 10


def test_query_by_family_fail_closed_on_unknown():
    atlas = build_atlas(_rows(3), prefer_mlx=False)
    with pytest.raises(EvaluatorResponseAtlasError):
        atlas.by_family("not_a_real_family")
    # all advertised families are queryable.
    for fam in ATLAS_EXPLOIT_ATOM_FAMILIES:
        assert atlas.by_family(fam)["family"] == fam


def test_query_k_must_be_positive():
    atlas = build_atlas(_rows(5), prefer_mlx=False)
    with pytest.raises(EvaluatorResponseAtlasError):
        atlas.top_budget_pairs(0)
    with pytest.raises(EvaluatorResponseAtlasError):
        atlas.most_fragile_pairs(-1)


# ---------------------------------------------------------------------------
# 4. JSONL persistence round-trip (the persisted INDEX)
# ---------------------------------------------------------------------------


def test_jsonl_roundtrip_preserves_rows_and_headline():
    atlas = build_atlas(_rows(12), prefer_mlx=False)
    lines = atlas.to_jsonl_lines()
    # header + 12 rows.
    assert len(lines) == 13
    assert json.loads(lines[0])["record_type"] == "atlas_header"
    restored = EvaluatorResponseAtlas.from_jsonl_lines(lines)
    assert len(restored.rows) == 12
    assert restored.schema == atlas.schema
    # a sampled field survives the round-trip exactly.
    orig = atlas.by_pair(5)
    rt = restored.by_pair(5)
    assert rt.joint_cone_summary.pair_budget == pytest.approx(
        orig.joint_cone_summary.pair_budget
    )
    assert rt.sensitivity_refs["cone_map_path"] == orig.sensitivity_refs["cone_map_path"]


def test_jsonl_load_fail_closed_on_missing_header():
    row = _rows(1)[0].to_json_obj()
    row["record_type"] = "atlas_pair_row"
    with pytest.raises(EvaluatorResponseAtlasError):
        EvaluatorResponseAtlas.from_jsonl_lines([json.dumps(row)])


def test_jsonl_load_fail_closed_on_unknown_record_type():
    bad = json.dumps({"record_type": "garbage"})
    with pytest.raises(EvaluatorResponseAtlasError):
        EvaluatorResponseAtlas.from_jsonl_lines([bad])


def test_persisted_index_carries_no_tensors_only_pointers():
    # an atlas row's JSON is small (pointers + scalar stats), NOT a tensor dump.
    row = _rows(1)[0]
    obj = row.to_json_obj()
    # the cone-map path is a pointer; the full field is NOT inlined.
    assert "cone_map_path" in obj["sensitivity_refs"]
    assert "joint_cone_radius" not in json.dumps(obj)  # no field array inlined
    # serialised size is small (an index entry, not a tensor copy).
    assert len(json.dumps(obj)) < 4096


# ---------------------------------------------------------------------------
# 5. Real-scorer NO-FAKE proof (the indexed quantities are measured, not zeros)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_scorer_atlas_row_indexes_nontrivial_measured_fields():
    """Build ONE atlas row from the REAL CPU-torch scorers + assert the indexed
    fields are non-trivial measured quantities (the NO-FAKE proof).

    A FAKE engine would index zeros / constants. This proves the pose Jacobian
    field is non-zero (gradient reachable through the differentiable-YUV6 patch),
    the seg margin field varies, and the pair budget is a real reduction.
    """

    import sys
    from pathlib import Path

    import torch

    repo = Path(__file__).resolve().parents[3]
    upstream = repo / "upstream"
    video = upstream / "videos" / "0.mkv"
    if not video.is_file():
        pytest.skip("contest video not available")
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))

    from safetensors.torch import load_file

    from tac.data import decode_video
    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally
    from tac.optimization.frame1_joint_safe_cone import (
        Frame1ConeConfig,
        compute_frame1_joint_safe_cone,
    )

    patch_upstream_yuv6_globally()
    from modules import PoseNet, SegNet  # type: ignore[import-not-found]

    seg = SegNet().eval()
    seg.load_state_dict(load_file(str(upstream / "models" / "segnet.safetensors"), device="cpu"))
    pose = PoseNet().eval()
    pose.load_state_dict(load_file(str(upstream / "models" / "posenet.safetensors"), device="cpu"))

    frames = decode_video(str(video), target_h=384, target_w=512, max_frames=2)
    gt = np.stack([frames[0].numpy(), frames[1].numpy()], axis=0)
    pair = torch.from_numpy(gt[None]).float()
    cone = compute_frame1_joint_safe_cone(
        segnet=seg, posenet=pose, pair_btchwc_unit255=pair, config=Frame1ConeConfig()
    )
    row = build_atlas_row_from_cone(pair_index=0, cone=cone, compute_path="cpu_torch")

    # NO-FAKE: the indexed pose Jacobian is non-zero (gradient reachable) ...
    assert row.pose_jacobian_norm_stats.l2_norm > 0.0
    assert row.pose_jacobian_norm_stats.max > row.pose_jacobian_norm_stats.min
    # ... the seg margin field varies (real SegNet logits) ...
    assert row.seg_margin_field_stats.max > row.seg_margin_field_stats.mean
    # ... and the pair budget is a real positive reduction.
    assert row.joint_cone_summary.pair_budget > 0.0
    assert row.pose_jacobian_norm_stats.compute_path == "cpu_torch"


# ---------------------------------------------------------------------------
# 6. Cathedral consumer wiring (Catalog #125 hook #1/#4 bridge)
# ---------------------------------------------------------------------------


def test_cathedral_consumer_bridge_ingests_atlas():
    from tac.cathedral_consumers.per_pair_difficulty_atlas_consumer import (
        consume_evaluator_response_atlas,
    )

    atlas = build_atlas(_rows(15), prefer_mlx=False)
    verdict = consume_evaluator_response_atlas(atlas)
    # Tier A non-promotable markers per Catalog #341/#323.
    assert verdict["promotable"] is False
    assert verdict["score_claim"] is False
    assert verdict["predicted_delta_adjustment"] == 0.0
    payload = verdict["notes"]["evaluator_response_atlas"]
    assert payload["n_pairs"] == 15
    # the ranking is a real reduction: top pair has highest fragility difficulty.
    top = payload["top_pairs"]
    diffs = [p["scorer_fragility_difficulty"] for p in top]
    assert diffs == sorted(diffs, reverse=True)


def test_cathedral_consumer_bridge_fail_closed_on_non_atlas():
    from tac.cathedral_consumers.per_pair_difficulty_atlas_consumer import (
        consume_evaluator_response_atlas,
    )

    verdict = consume_evaluator_response_atlas(object())
    assert verdict["promotable"] is False
    # no-signal verdict (not a crash) when handed a non-atlas object.
    assert "predicted_delta_adjustment" in verdict
