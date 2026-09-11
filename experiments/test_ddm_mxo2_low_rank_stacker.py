from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np

from experiments import ddm_mxo2_low_rank_stacker as mxo2


def _library(tmp_path: Path) -> Path:
    target = tmp_path / "libmxo2.so"
    subprocess.run(
        [
            "cc", "-O3", "-std=c11", "-shared", "-fPIC", "-ffp-contract=off",
            "-fno-fast-math", str(mxo2.REPO / "experiments/ddm_mxo2_stacker.c"),
            "-o", str(target),
        ],
        check=True,
    )
    return target


def test_group_order_is_complete_and_causal() -> None:
    np.testing.assert_array_equal(np.sort(mxo2.GROUP_ORDER), np.arange(mxo2.SIZE))
    assert len(mxo2.GROUP_POSITIONS) == 190
    for group, positions in enumerate(mxo2.GROUP_POSITIONS):
        np.testing.assert_array_equal(mxo2.GROUP[positions], group)


def test_instrumentation_changes_only_copy_payload() -> None:
    source = (
        mxo2.SOURCE / "runtime/f26_corrector_native.c"
    ).read_bytes()
    changed = mxo2.instrument_source(source)
    assert source != changed
    assert changed.count(b"f26_corrector_mxo2_surface") == 1
    assert source.count(b"f26_corrector_mxo2_surface") == 0


def test_instrumented_receiver_source_compiles(tmp_path: Path) -> None:
    source = mxo2.SOURCE / "runtime/f26_corrector_native.c"
    instrumented = tmp_path / "f26_corrector_native.c"
    instrumented.write_bytes(mxo2.instrument_source(source.read_bytes()))
    subprocess.run(
        [
            "cc", "-O3", "-std=c11", "-shared", "-fPIC", "-ffp-contract=off",
            "-fno-fast-math", str(instrumented), "-o", str(tmp_path / "libcorrector.so"), "-lm",
        ],
        check=True,
    )


def test_native_stacker_cold_identity_then_group_causal_update(tmp_path: Path) -> None:
    library = _library(tmp_path)
    rng = np.random.default_rng(7)
    n = 64
    raw = rng.integers(1, 1_000_000, size=(n, mxo2.K), dtype=np.uint32)
    freq = ((raw.astype(np.uint64) * mxo2.TOTAL) // raw.sum(axis=1)[:, None]).astype(np.uint32)
    freq[:, 0] += (mxo2.TOTAL - freq.sum(axis=1, dtype=np.uint64)).astype(np.uint32)
    hit_class = freq.argmax(axis=1).astype(np.uint8)
    symbols = rng.integers(0, mxo2.K, size=n, dtype=np.uint8)
    mixed = rng.integers(1000, 32000, size=n, dtype=np.uint16)
    family = np.clip(
        mixed[:, None].astype(np.int32) + rng.integers(-3000, 3001, size=(n, 23)),
        1,
        32767,
    ).astype(np.uint16)

    with mxo2.NativeStacker(library, 4) as model:
        first = model.process(family, mixed, freq, hit_class, symbols)
        np.testing.assert_array_equal(first, freq)
        learned = model.weights()
        assert np.any(learned != 0)
        second = model.process(family, mixed, freq, hit_class, symbols)
        assert np.all(second.sum(axis=1, dtype=np.uint64) == mxo2.TOTAL)
        assert np.any(second != freq)


def test_native_rejects_non_charter_rank(tmp_path: Path) -> None:
    library = ctypes.CDLL(str(_library(tmp_path)))
    library.mxo2_create.argtypes = [ctypes.c_int, ctypes.c_uint64]
    library.mxo2_create.restype = ctypes.c_void_p
    assert not library.mxo2_create(3, mxo2.SEED)
