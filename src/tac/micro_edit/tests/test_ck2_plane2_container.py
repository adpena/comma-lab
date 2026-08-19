"""The ck2 whole-section 2-plane container transform: totality and cross-copy agreement.

The permutation exists in THREE places and they must agree exactly or an archive decodes
to garbage:

1. the compile side (``experiments/ddm_sa3_rebase_sz1.plane2``),
2. the overlay builder (``experiments/ddm_ck2_build_receiver_overlay.interleave_planes``),
3. the GENERATED receiver text that ships inside the archive runtime.

(3) cannot import (1) or (2) -- it ships in the decoder tree -- so the duplication is
structural. These tests are what keeps the copies honest between builds; parse-back is
what keeps them honest at build time.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.ddm_ck2_build_receiver_overlay import (
    _NEW_CONSTANTS,
    interleave_planes,
    uninterleave_planes,
)


def _receiver_inverse():
    """Exec the un-split EXACTLY as it appears in the generated receiver text.

    This is the only way to test the shipped copy rather than a re-typed lookalike: if
    someone edits the embedded source, this picks it up.
    """
    body = _NEW_CONSTANTS.split("def _ck2_uninterleave_planes", 1)[1]
    namespace: dict[str, object] = {"np": np}
    exec("def _ck2_uninterleave_planes" + body, namespace)  # noqa: S102
    return namespace["_ck2_uninterleave_planes"]


@pytest.mark.parametrize("length", list(range(0, 66)) + [4_141, 22_187, 36_130])
def test_roundtrip_is_total_over_every_length(length: int) -> None:
    """Odd lengths are the boundary a single 36 KB smoke would never exercise."""
    rng = np.random.default_rng(length)
    payload = rng.integers(0, 256, size=length, dtype=np.uint8).tobytes()
    assert uninterleave_planes(interleave_planes(payload)) == payload


def test_forward_is_the_documented_plane_order() -> None:
    """Even-index plane first, then odd-index plane, odd tail byte untouched."""
    assert interleave_planes(bytes([0, 1, 2, 3, 4, 5])) == bytes([0, 2, 4, 1, 3, 5])
    assert interleave_planes(bytes([0, 1, 2, 3, 4])) == bytes([0, 2, 1, 3, 4])
    assert interleave_planes(b"") == b""
    assert interleave_planes(b"\x07") == b"\x07"


def test_generated_receiver_text_matches_the_builder() -> None:
    """The shipped inverse and the build-side inverse agree byte-for-byte."""
    receiver_inverse = _receiver_inverse()
    rng = np.random.default_rng(20260819)
    for length in (0, 1, 2, 3, 17, 4_142, 22_187, 36_130):
        payload = rng.integers(0, 256, size=length, dtype=np.uint8).tobytes()
        permuted = interleave_planes(payload)
        assert receiver_inverse(permuted) == payload
        assert receiver_inverse(permuted) == uninterleave_planes(permuted)


def test_compile_side_copy_agrees_with_the_overlay_copy() -> None:
    """``ddm_sa3_rebase_sz1``'s pair is the same permutation, not a lookalike."""
    sa3 = pytest.importorskip(
        "experiments.ddm_sa3_rebase_sz1",
        reason="the compile module pulls the sa2 build stack, which needs the SSD tier",
    )
    rng = np.random.default_rng(11)
    for length in (0, 1, 5, 4_142, 36_130):
        payload = rng.integers(0, 256, size=length, dtype=np.uint8).tobytes()
        assert sa3.plane2(payload) == interleave_planes(payload)
        assert sa3.unplane2(sa3.plane2(payload)) == payload


def test_transform_is_a_permutation_not_a_compressor() -> None:
    """Length and multiset of bytes are invariant -- it only moves bytes."""
    rng = np.random.default_rng(3)
    payload = rng.integers(0, 256, size=9_999, dtype=np.uint8).tobytes()
    permuted = interleave_planes(payload)
    assert len(permuted) == len(payload)
    assert sorted(permuted) == sorted(payload)
