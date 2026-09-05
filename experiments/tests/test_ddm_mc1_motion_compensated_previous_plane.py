from __future__ import annotations

import numpy as np

# The 2026-09-03 codex screen (commit 2c32e2767) was restored verbatim under its own
# path when the 2026-09-04 arm took the charter's module path; this test binds to it.
from experiments import ddm_mc1_motion_plane_ceiling_screen_20260903 as mc1


def _semantic_scene() -> np.ndarray:
    scene = np.zeros((mc1.H, mc1.W), dtype=np.uint8)
    scene[:96] = 2
    scene[96:288] = 0
    scene[288:] = 4
    scene[180:260, 220:226] = 1
    scene[200:232, 300:340] = 3
    return scene


def test_global_translation_recovers_and_applies_real_shift() -> None:
    source = _semantic_scene()
    target = mc1.warp_translation(source, 3, -2)
    dy, dx, score, zero, radius, boundary_hit = mc1.estimate_translation(
        source, target
    )
    assert (dy, dx) == (3, -2)
    assert score > zero
    assert radius == 4
    assert boundary_hit is False
    assert np.array_equal(mc1.warp_translation(source, dy, dx), target)


def test_row_warp_uses_distinct_destination_band_motion() -> None:
    source = _semantic_scene()
    shifts = np.asarray(
        [(0, 0), (0, 0), (1, 0), (2, -1), (3, -2), (4, -3)],
        dtype=np.int16,
    )
    output = mc1.warp_row_shifts(source, shifts)
    assert np.array_equal(output[:128], source[:128])
    # An interior destination pixel is copied from the band-specific source.
    assert output[4 * mc1.PATCH + 20, 200] == source[4 * mc1.PATCH + 17, 202]
    assert not np.array_equal(output, source)


def test_integer_affine_fit_recovers_constant_translation() -> None:
    local = np.tile(np.asarray([[2, -3]], dtype=np.int16), (mc1.TILES, 1))
    weights = np.arange(1, mc1.TILES + 1, dtype=np.int64)
    coefficients = mc1.fit_affine_integer(local, weights)
    assert np.array_equal(
        coefficients,
        np.asarray(
            [[2 * mc1.AFFINE_Q, 0, 0], [-3 * mc1.AFFINE_Q, 0, 0]],
            dtype=np.int64,
        ),
    )
    source = _semantic_scene()
    assert np.array_equal(
        mc1.warp_affine(source, coefficients),
        mc1.warp_translation(source, 2, -3),
    )


def test_crossfit_offsets_change_codelength_not_only_metadata() -> None:
    logit = np.zeros(16, dtype=np.float64)
    flip = np.asarray([0, 0, 0, 1, 1, 1, 1, 1] * 2, dtype=bool)
    true_class = np.asarray([0, 0, 0, 0, 1, 1, 1, 1] * 2, dtype=np.uint8)
    cells = true_class.astype(np.int16)
    folds = np.asarray([0] * 8 + [1] * 8, dtype=np.uint8)
    bits, tables, diagnostics = mc1._crossfit_bits_by_class(
        logit, flip, true_class, cells, folds, 2
    )
    assert bits.shape == (mc1.CLASSES,)
    assert bits[0] < 8.0
    assert bits[1] < 8.0
    assert np.any(tables["beta_train_fold_0"] != 0.0)
    assert len(diagnostics) == 2
