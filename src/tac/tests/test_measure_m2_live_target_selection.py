# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _module():
    path = Path(__file__).resolve().parents[3] / "tools" / "measure_m2_live_target_selection.py"
    spec = importlib.util.spec_from_file_location("measure_m2_live_target_selection", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_vectorized_fill_policies_preserve_visible_values_and_uint8() -> None:
    mod = _module()
    frames = np.arange(2 * 4 * 5 * 3, dtype=np.uint8).reshape(2, 4, 5, 3)
    mask = np.zeros((4, 5), dtype=bool)
    mask[1] = True
    mask[:, 3] = True
    for strategy in mod.STRATEGIES:
        out = mod._fill_candidate(frames, mask, strategy)
        assert out.shape == frames.shape
        assert out.dtype == np.uint8
        np.testing.assert_array_equal(out[:, ~mask, :], frames[:, ~mask, :])


def test_horizontal_and_vertical_predictor_policy_semantics() -> None:
    mod = _module()
    frames = np.arange(1 * 4 * 5, dtype=np.uint8).reshape(1, 4, 5, 1)
    frames = np.repeat(frames, 3, axis=3)
    mask = np.zeros((4, 5), dtype=bool)
    mask[1] = True
    mask[:, 0] = True
    mask[:, 3] = True

    horizontal = mod._fill_candidate(frames, mask, "horizontal_predictor")
    assert np.all(horizontal[:, 1] == 0)
    np.testing.assert_array_equal(horizontal[:, 0, 0], frames[:, 0, 1])
    np.testing.assert_array_equal(horizontal[:, 0, 3], frames[:, 0, 2])

    vertical = mod._fill_candidate(frames, mask, "vertical_predictor")
    assert np.all(vertical[:, :, 0] == 0)
    assert np.all(vertical[:, :, 3] == 0)
    np.testing.assert_array_equal(vertical[:, 1, 2], frames[:, 0, 2])


def test_source_indices_uses_first_visible_for_leading_gap() -> None:
    mod = _module()
    np.testing.assert_array_equal(
        mod._source_indices(np.array([False, False, True, False, True])),
        np.array([2, 2, 2, 2, 4]),
    )
