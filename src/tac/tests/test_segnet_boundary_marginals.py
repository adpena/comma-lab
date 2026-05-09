from __future__ import annotations

import numpy as np

from tac.analysis.segnet_boundary_marginals import (
    boundary_mask_from_labels,
    logit_margin,
    merge_feature_summaries,
    summarize_boundary_features,
)


def test_boundary_mask_marks_both_sides_of_class_edge() -> None:
    labels = np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [2, 2, 2],
        ],
        dtype=np.int64,
    )

    mask = boundary_mask_from_labels(labels, dilation=1)

    assert mask.dtype == np.bool_
    assert mask[0, 1]
    assert mask[0, 2]
    assert mask[1, 0]
    assert mask[2, 0]
    assert not mask[0, 0]


def test_logit_margin_returns_top1_minus_top2() -> None:
    logits = np.array(
        [
            [[2.0, 1.0]],
            [[4.0, 3.0]],
            [[1.5, -1.0]],
        ],
        dtype=np.float64,
    )

    margins = logit_margin(logits)

    np.testing.assert_allclose(margins, np.array([[2.0, 2.0]]))


def test_summarize_boundary_features_emits_per_pair_vectors() -> None:
    labels = np.array(
        [
            [[0, 0], [0, 1]],
            [[2, 2], [2, 2]],
        ],
        dtype=np.int64,
    )
    logits = np.array(
        [
            [
                [[3.0, 3.0], [3.0, 0.0]],
                [[1.0, 1.0], [1.0, 4.0]],
            ],
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[2.0, 2.0], [2.0, 2.0]],
            ],
        ],
        dtype=np.float64,
    )

    summary = summarize_boundary_features(
        labels=labels,
        logits=logits,
        dilation=1,
        low_margin_threshold=1.5,
    )

    assert summary.boundary_mass.shape == (2,)
    assert summary.boundary_mass[0] > summary.boundary_mass[1]
    np.testing.assert_allclose(summary.mean_logit_margin, np.array([2.5, 2.0]))
    np.testing.assert_allclose(summary.low_margin_mass, np.array([0.0, 0.0]))
    payload = summary.as_jsonable()
    assert sorted(payload) == [
        "per_pair_boundary_mass",
        "per_pair_low_margin_mass",
        "per_pair_mean_logit_margin",
        "per_pair_p10_logit_margin",
    ]


def test_merge_feature_summaries_concatenates_in_order() -> None:
    labels = np.array([[[0, 1]], [[1, 1]]], dtype=np.int64)
    logits = np.ones((2, 2, 1, 2), dtype=np.float64)
    logits[:, 1, :, :] = 2.0
    first = summarize_boundary_features(labels=labels[:1], logits=logits[:1])
    second = summarize_boundary_features(labels=labels[1:], logits=logits[1:])

    merged = merge_feature_summaries([first, second])

    np.testing.assert_allclose(
        merged.boundary_mass,
        np.concatenate([first.boundary_mass, second.boundary_mass]),
    )
