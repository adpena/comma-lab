from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.ddm_fcd1_field_for_coder_diagonal import (
    NATIVE_CORRECTOR_BUILD,
    PYTHON_CORRECTOR_MARKER,
    bind_generation21_python_corrector,
    classify_pool,
    stratified_assign,
)
from experiments.ddm_fcd1_incompile_schur import (
    Fcd1SchurError,
    validate_pose_gate,
)


def test_classify_pool_is_exact_and_disjoint() -> None:
    tokens = np.array([[0, 1, 2, 3]], dtype=np.uint8)
    argmax = np.array([[1, 0, 3, 3]], dtype=np.uint8)
    gt = np.array([[1, 1, 4, 3]], dtype=np.uint8)

    classes = classify_pool(tokens, argmax, gt)

    assert classes["benefit"].tolist() == [[True, False, False, False]]
    assert classes["harm"].tolist() == [[False, True, False, False]]
    assert classes["wash"].tolist() == [[False, False, True, False]]
    total = sum(mask.astype(np.uint8) for mask in classes.values())
    assert int(total.max()) == 1


def test_stratified_assign_is_deterministic_disjoint_and_complete() -> None:
    coords = np.array([[frame, frame % 7, frame % 11] for frame in range(120)], dtype=np.int32)
    old = np.zeros(120, dtype=np.uint8)
    new = np.ones(120, dtype=np.uint8)

    first = stratified_assign(coords, old, new)
    second = stratified_assign(coords, old, new)

    assert np.array_equal(first, second)
    assert set(first.tolist()) == {0, 1, 2}
    assert sum(int(np.count_nonzero(first == index)) for index in range(3)) == 120
    for block in (0, 1):
        local = first[(coords[:, 0] // 60) == block]
        counts = [int(np.count_nonzero(local == index)) for index in range(3)]
        assert max(counts) - min(counts) <= 1


def test_incompile_pose_gate_is_one_sided_and_finite() -> None:
    assert validate_pose_gate(d_pose_after=0.9, d_pose_base=1.0, band=0.0)
    assert validate_pose_gate(d_pose_after=1.00000001, d_pose_base=1.0, band=1e-8)
    assert not validate_pose_gate(d_pose_after=1.00000002, d_pose_base=1.0, band=1e-8)
    with np.testing.assert_raises(Fcd1SchurError):
        validate_pose_gate(d_pose_after=np.nan, d_pose_base=1.0, band=1e-8)


def test_generation21_runtime_binding_is_explicit_and_idempotent(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    inflate = runtime / "inflate.sh"
    inflate.write_text("#!/usr/bin/env bash\n" + NATIVE_CORRECTOR_BUILD + "echo decode\n")
    inflate.chmod(0o755)

    first = bind_generation21_python_corrector(runtime)
    second = bind_generation21_python_corrector(runtime)

    text = inflate.read_text()
    assert first["changed"] is True
    assert second["changed"] is False
    assert text.count(PYTHON_CORRECTOR_MARKER) == 1
    assert NATIVE_CORRECTOR_BUILD not in text
    assert "unset F26_CORRECTOR_NATIVE_LIBRARY" in text
    assert inflate.stat().st_mode & 0o111
