# SPDX-License-Identifier: MIT
"""Real-input algebra checks; research_only=true, no n600 or score verdict."""
from pathlib import Path

import numpy as np
import pytest

from experiments.ddm_rbf1_boundary_probe import FIELD, RAW, ROOT, save_array
from experiments.ddm_rbf1_boundary_treatments import camera_geometry, treat


@pytest.mark.parametrize("mode", ["guided", "ssaa", "sdf", "composition"])
def test_real_input_repeat_and_support(mode):
    if not RAW.is_file() or not FIELD.is_file():
        pytest.skip("pinned move-44 files unavailable")
    rgb = np.memmap(RAW, mode="r", dtype=np.uint8, shape=(600, 2, 874, 1164, 3))[0, 1]
    tokens = np.memmap(FIELD, mode="r", dtype=np.uint8, shape=(600, 384, 512))[0]
    output = treat(rgb, tokens, mode)
    repeat = treat(rgb, tokens, mode)
    directory = ROOT / "unit_controls"
    first = save_array(directory / f"{mode}.npy", output)
    twin = save_array(directory / f"{mode}.repeat.npy", repeat)
    assert first["sha256"] == twin["sha256"]
    assert output.dtype == np.uint8 and output.shape == rgb.shape
    band = camera_geometry(tokens, rgb.shape[:2])[-1]
    assert np.array_equal(output[~band], rgb[~band])
    assert np.count_nonzero(output != rgb) > 0
    # No numeric class ordering or learned class choice may enter an operator.
    permuted = np.array([3, 4, 0, 2, 1], dtype=np.uint8)[tokens]
    renamed = treat(rgb, permuted, mode)
    save_array(directory / f"{mode}.class_permuted.npy", renamed)
    assert np.array_equal(output, renamed)


def test_retention_refuses_replacement():
    source = ROOT / "unit_controls/guided.npy"
    if not source.exists():
        pytest.skip("run treatment control first")
    data = np.load(source, allow_pickle=False)
    data = data.copy()
    data[0, 0, 0] ^= 1
    # Persist the deliberately changed control before testing refusal.
    save_array(ROOT / "unit_controls/changed_control.npy", data)
    with pytest.raises(ValueError, match="overwrite"):
        save_array(Path(source), data)
