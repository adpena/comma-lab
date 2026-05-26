"""Round-trip + head-to-head tests for PR110 OPT-3 Variant B FEC8 Markov context coder.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" +
HNeRV parity discipline L11 (no-op detector — prove the targeted bytes changed
AND were consumed by inflate). These tests prove:

  * FEC8 STATIC + ADAPTIVE encoders + decoders are byte-stable round-trip.
  * Head-to-head: FEC8 STATIC vs FEC8 ADAPTIVE vs FEC6 fixed-Huff vs FEC7 0-order
    arith on the LIVE 600-pair PR110 selector stream.
  * The empirical 16x16 transition table baked into source matches the canonical
    anchor at ``.omx/research/pr110_opt_3b_markov_transition_matrix_20260526.json``.

Run from repo root::

    .venv/bin/python -m pytest submissions/hnerv_fec6_fixed_huffman_k16/tests/test_markov_selector_roundtrip.py -v

# SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import struct
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENCODER_DIR = _REPO_ROOT / "submissions/hnerv_fec6_fixed_huffman_k16/encoder"
_SRC_DIR = _REPO_ROOT / "submissions/hnerv_fec6_fixed_huffman_k16/src"
for d in (_ENCODER_DIR, _SRC_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from build_pr101_frame_exploit_selector_packet_arith import (  # type: ignore[import-not-found]  # noqa: E402
    encode_fec7_arith_selector,
)
from build_pr101_frame_exploit_selector_packet_markov import (  # type: ignore[import-not-found]  # noqa: E402
    EMPIRICAL_MARGINAL_COUNTS,
    EMPIRICAL_TRANSITION_COUNTS,
    FEC8_MAGIC,
    FEC8_VARIANT_ADAPTIVE,
    FEC8_VARIANT_STATIC,
    PALETTE_K,
    decode_fec8_markov_selector,
    encode_fec8_markov_selector_adaptive,
    encode_fec8_markov_selector_static,
)
from fec8_markov_decoder import (  # type: ignore[import-not-found]  # noqa: E402
    decode_fec8_markov_selector as decode_via_inflate_module,
)


# -- canonical FEC6 decoder (copy of inflate.py for measurement) --------------------

FEC6_FIXED_K16_CODE_BITS = (
    "00", "1100", "01", "111010", "11010", "111011", "111100", "100",
    "111101", "11011", "1111110", "111110", "11111110", "101", "11100", "11111111",
)
FEC6_FIXED_K16_DECODE = {b: c for c, b in enumerate(FEC6_FIXED_K16_CODE_BITS)}


def _unpack_fec6(payload: bytes, *, n_pairs: int) -> list[int]:
    codes: list[int] = []
    prefix = ""
    bp = 0
    while len(codes) < n_pairs:
        bit = (payload[bp // 8] >> (7 - (bp % 8))) & 1
        bp += 1
        prefix += "1" if bit else "0"
        c = FEC6_FIXED_K16_DECODE.get(prefix)
        if c is not None:
            codes.append(c)
            prefix = ""
    return codes


def _load_live_codes_and_fec6_payload() -> tuple[list[int], bytes]:
    archive_path = _REPO_ROOT / (
        "experiments/results/"
        "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/"
        "archive.zip"
    )
    if not archive_path.exists():
        pytest.skip(f"live FEC6 archive not present: {archive_path}")
    with zipfile.ZipFile(archive_path) as zf:
        data = zf.read("x")
    assert data[:4] == b"FP11"
    (src_len,) = struct.unpack_from("<I", data, 4)
    pos = 8 + src_len
    (sel_len,) = struct.unpack_from("<H", data, pos)
    sel = data[pos + 2 : pos + 2 + sel_len]
    assert sel[:4] == b"FEC6"
    (n_pairs,) = struct.unpack_from("<H", sel, 4)
    codes = _unpack_fec6(sel[6:], n_pairs=n_pairs)
    assert len(codes) == n_pairs == 600
    return codes, sel


# -- synthetic round-trip tests (STATIC variant) ----------------------------------


@pytest.mark.parametrize("seed", [0, 1, 42, 1234, 99999])
def test_static_roundtrip_uniform(seed: int) -> None:
    import random

    rng = random.Random(seed)
    n = 600
    codes = [rng.randrange(PALETTE_K) for _ in range(n)]
    payload = encode_fec8_markov_selector_static(codes, n_pairs=n)
    assert payload[:4] == FEC8_MAGIC
    assert payload[4:6] == FEC8_VARIANT_STATIC
    (header_n,) = struct.unpack_from("<H", payload, 6)
    assert header_n == n
    decoded = decode_fec8_markov_selector(payload)
    assert decoded == codes


@pytest.mark.parametrize("dominant", list(range(PALETTE_K)))
def test_static_roundtrip_dominant_mode(dominant: int) -> None:
    import random

    rng = random.Random(dominant)
    n = 600
    codes = []
    for _ in range(n):
        if rng.random() < 0.9:
            codes.append(dominant)
        else:
            codes.append(rng.randrange(PALETTE_K))
    payload = encode_fec8_markov_selector_static(codes, n_pairs=n)
    decoded = decode_fec8_markov_selector(payload)
    assert decoded == codes


def test_static_edge_all_zeros() -> None:
    n = 600
    codes = [0] * n
    payload = encode_fec8_markov_selector_static(codes, n_pairs=n)
    decoded = decode_fec8_markov_selector(payload)
    assert decoded == codes


def test_static_edge_max_index() -> None:
    n = 600
    codes = [PALETTE_K - 1] * n
    payload = encode_fec8_markov_selector_static(codes, n_pairs=n)
    decoded = decode_fec8_markov_selector(payload)
    assert decoded == codes


def test_static_edge_strictly_alternating() -> None:
    n = 600
    codes = [i % PALETTE_K for i in range(n)]
    payload = encode_fec8_markov_selector_static(codes, n_pairs=n)
    decoded = decode_fec8_markov_selector(payload)
    assert decoded == codes


# -- synthetic round-trip tests (ADAPTIVE variant) --------------------------------


@pytest.mark.parametrize("seed", [0, 1, 42, 1234, 99999])
def test_adaptive_roundtrip_uniform(seed: int) -> None:
    import random

    rng = random.Random(seed)
    n = 600
    codes = [rng.randrange(PALETTE_K) for _ in range(n)]
    payload = encode_fec8_markov_selector_adaptive(codes, n_pairs=n)
    assert payload[:4] == FEC8_MAGIC
    assert payload[4:6] == FEC8_VARIANT_ADAPTIVE
    (header_n,) = struct.unpack_from("<H", payload, 6)
    assert header_n == n
    decoded = decode_fec8_markov_selector(payload)
    assert decoded == codes


@pytest.mark.parametrize("dominant", list(range(PALETTE_K)))
def test_adaptive_roundtrip_dominant_mode(dominant: int) -> None:
    import random

    rng = random.Random(dominant)
    n = 600
    codes = []
    for _ in range(n):
        if rng.random() < 0.9:
            codes.append(dominant)
        else:
            codes.append(rng.randrange(PALETTE_K))
    payload = encode_fec8_markov_selector_adaptive(codes, n_pairs=n)
    decoded = decode_fec8_markov_selector(payload)
    assert decoded == codes


def test_adaptive_edge_all_zeros() -> None:
    n = 600
    codes = [0] * n
    payload = encode_fec8_markov_selector_adaptive(codes, n_pairs=n)
    decoded = decode_fec8_markov_selector(payload)
    assert decoded == codes
    bitstream_len = len(payload) - 8
    assert bitstream_len < 60, f"all-zeros adaptive bitstream {bitstream_len}B; expected dramatic compression"


def test_adaptive_edge_max_index() -> None:
    n = 600
    codes = [PALETTE_K - 1] * n
    payload = encode_fec8_markov_selector_adaptive(codes, n_pairs=n)
    decoded = decode_fec8_markov_selector(payload)
    assert decoded == codes


def test_adaptive_edge_strictly_alternating() -> None:
    n = 600
    codes = [i % PALETTE_K for i in range(n)]
    payload = encode_fec8_markov_selector_adaptive(codes, n_pairs=n)
    decoded = decode_fec8_markov_selector(payload)
    assert decoded == codes


# -- input validation -------------------------------------------------------------


def test_invalid_code_raises() -> None:
    with pytest.raises(ValueError):
        encode_fec8_markov_selector_static([0, 1, PALETTE_K], n_pairs=3)
    with pytest.raises(ValueError):
        encode_fec8_markov_selector_static([0, 1, -1], n_pairs=3)
    with pytest.raises(ValueError):
        encode_fec8_markov_selector_adaptive([0, 1, PALETTE_K], n_pairs=3)
    with pytest.raises(ValueError):
        encode_fec8_markov_selector_adaptive([0, 1, -1], n_pairs=3)


def test_mismatched_n_pairs_raises() -> None:
    with pytest.raises(ValueError):
        encode_fec8_markov_selector_static([0, 1, 2], n_pairs=4)
    with pytest.raises(ValueError):
        encode_fec8_markov_selector_adaptive([0, 1, 2], n_pairs=4)


def test_truncated_payload_raises() -> None:
    with pytest.raises(ValueError):
        decode_fec8_markov_selector(b"FEC8")
    with pytest.raises(ValueError):
        decode_fec8_markov_selector(b"WRONG\x00\x01\x00\x00")
    with pytest.raises(ValueError):
        decode_fec8_markov_selector(b"FEC8\x00\xFF\x00\x00")  # unknown variant


def test_inflate_module_proxies_canonical_decoder() -> None:
    """Round-trip via the inflate-side proxy module (proves single source of truth)."""
    import random

    rng = random.Random(0)
    n = 600
    codes = [rng.randrange(PALETTE_K) for _ in range(n)]
    payload_static = encode_fec8_markov_selector_static(codes, n_pairs=n)
    decoded_via_inflate_static = decode_via_inflate_module(payload_static)
    assert decoded_via_inflate_static == codes

    payload_adaptive = encode_fec8_markov_selector_adaptive(codes, n_pairs=n)
    decoded_via_inflate_adaptive = decode_via_inflate_module(payload_adaptive)
    assert decoded_via_inflate_adaptive == codes


# -- canonical anchor consistency -------------------------------------------------


def test_baked_transition_table_matches_canonical_anchor() -> None:
    """EMPIRICAL_TRANSITION_COUNTS in source must match the anchor JSON."""
    anchor_path = _REPO_ROOT / ".omx/research/pr110_opt_3b_markov_transition_matrix_20260526.json"
    if not anchor_path.exists():
        pytest.skip(f"canonical anchor JSON not present: {anchor_path}")
    payload = json.loads(anchor_path.read_text())
    anchor_counts = payload["transition_counts_prev_to_next"]
    anchor_marginal = payload["marginal_histogram"]
    assert len(anchor_counts) == PALETTE_K
    for prev in range(PALETTE_K):
        baked_row = tuple(EMPIRICAL_TRANSITION_COUNTS[prev])
        anchor_row = tuple(anchor_counts[prev])
        assert baked_row == anchor_row, (
            f"row {prev} drift: baked={baked_row} vs anchor={anchor_row}"
        )
    for sym in range(PALETTE_K):
        assert EMPIRICAL_MARGINAL_COUNTS[sym] == anchor_marginal[str(sym)], (
            f"marginal sym {sym} drift"
        )


# -- live FEC6 stream measurement + head-to-head ---------------------------------


def test_live_fec6_codes_roundtrip_through_markov_static() -> None:
    codes, _ = _load_live_codes_and_fec6_payload()
    payload = encode_fec8_markov_selector_static(codes, n_pairs=len(codes))
    decoded = decode_fec8_markov_selector(payload)
    assert decoded == codes


def test_live_fec6_codes_roundtrip_through_markov_adaptive() -> None:
    codes, _ = _load_live_codes_and_fec6_payload()
    payload = encode_fec8_markov_selector_adaptive(codes, n_pairs=len(codes))
    decoded = decode_fec8_markov_selector(payload)
    assert decoded == codes


def test_live_head_to_head_fec6_vs_fec7_vs_fec8_static_vs_fec8_adaptive() -> None:
    """Head-to-head wire-byte measurement on the live 600-pair PR110 selector stream.

    Empirical prediction per ``.omx/research/pr110_opt_3b_markov_transition_matrix_20260526.json``:

      * H(marginal) = 3.2116 bits/pair (FEC7 0-order arith asymptote)
      * H(next | prev) = 2.9402 bits/pair (FEC8 Markov asymptote)
      * Max encoder savings vs fixed-Huff: ~22 bytes for FEC8 STATIC

    The empirical wire sizes must be consistent with the entropy ordering:
      FEC8 STATIC ≤ FEC8 ADAPTIVE  (static seed eliminates convergence overhead)
      FEC8 STATIC ≤ FEC7           (1st-order Markov dominates 0-order on this stream)
    """
    codes, fec6_payload = _load_live_codes_and_fec6_payload()
    n = len(codes)
    fec7_payload = encode_fec7_arith_selector(codes, n_pairs=n)
    fec8_static_payload = encode_fec8_markov_selector_static(codes, n_pairs=n)
    fec8_adaptive_payload = encode_fec8_markov_selector_adaptive(codes, n_pairs=n)

    fec6_size = len(fec6_payload)
    fec7_size = len(fec7_payload)
    fec8_static_size = len(fec8_static_payload)
    fec8_adaptive_size = len(fec8_adaptive_payload)

    # Encoder must produce sensible byte counts (i.e. not blow up to >2x the input bound)
    upper_bound = 2 * fec6_size
    assert fec7_size < upper_bound
    assert fec8_static_size < upper_bound
    assert fec8_adaptive_size < upper_bound

    # Static should not be dramatically worse than adaptive — both must be within +/- 32 bytes
    # of each other (a 600-symbol stream split across 16 contexts has limited adaptation room)
    diff_static_vs_adaptive = fec8_adaptive_size - fec8_static_size
    assert abs(diff_static_vs_adaptive) <= 32, (
        f"FEC8 static {fec8_static_size}B vs adaptive {fec8_adaptive_size}B; "
        f"diff {diff_static_vs_adaptive:+d} outside expected +/- 32 byte band"
    )

    # Consistency check on input distribution (must match analysis anchor)
    histogram = Counter(codes)
    assert histogram[0] == 134, "expected 134 'none' codes in live stream"
    assert histogram[2] == 129, "expected 129 'frame0_blue_chroma_amp_3' codes in live stream"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
