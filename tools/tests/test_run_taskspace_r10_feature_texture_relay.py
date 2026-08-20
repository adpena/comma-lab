from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools/run_taskspace_r10_feature_texture_relay.py"
SPEC = importlib.util.spec_from_file_location("g27_r10_runner", TOOL)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _pairs() -> np.ndarray:
    height, width = 32, 48
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.uint16),
        np.arange(width, dtype=np.uint16),
        indexing="ij",
    )
    pairs = np.empty((2, 2, height, width, 3), dtype=np.uint8)
    for pair in range(2):
        for frame in range(2):
            pairs[pair, frame, ..., 0] = (xx * 3 + yy + pair + frame) % 256
            pairs[pair, frame, ..., 1] = (xx + yy * 5 + 2 * pair + frame) % 256
            pairs[pair, frame, ..., 2] = (xx * 2 + yy * 2 + pair + 3 * frame) % 256
    return pairs


def test_write_once_or_equal_is_idempotent_and_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "immutable.bin"
    runner._write_once_or_equal(path, b"first")
    runner._write_once_or_equal(path, b"first")
    with pytest.raises(runner.R10RunnerError, match="different bytes"):
        runner._write_once_or_equal(path, b"second")
    assert path.read_bytes() == b"first"


def test_encoder_derivations_are_source_bound_and_finite() -> None:
    pairs = _pairs()
    xi = runner._derive_xi(pairs)
    labels, centers = runner._derive_support_labels(pairs)
    base_records, texture_records = runner._derive_relay_records(pairs)
    assert xi.shape == (2, 6)
    assert np.isfinite(xi).all()
    assert labels.shape == (2, 32, 48)
    assert np.count_nonzero(labels) > 0
    assert len(centers) == len(base_records) == len(texture_records) == 2


def test_regular_reader_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"custody")
    link = tmp_path / "link.bin"
    link.symlink_to(target)
    with pytest.raises(runner.R10RunnerError, match="non-symlink regular file"):
        runner._read_regular(link, label="fixture")

