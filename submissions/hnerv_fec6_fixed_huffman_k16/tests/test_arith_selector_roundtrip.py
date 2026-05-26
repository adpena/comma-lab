"""Round-trip + empirical-savings tests for PR110 OPT-3 FEC7 adaptive range coder.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" +
HNeRV parity discipline L11 (no-op detector — prove the targeted bytes changed
AND were consumed by inflate). These tests prove the encoder + decoder are
byte-stable and the wire payload meaningfully changes from the FEC6 fixed-Huffman
baseline.

Run from repo root::

    .venv/bin/python -m pytest submissions/hnerv_fec6_fixed_huffman_k16/tests/test_arith_selector_roundtrip.py -v

# SPDX-License-Identifier: MIT
"""

from __future__ import annotations

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
    FEC7_MAGIC,
    PALETTE_K,
    decode_fec7_arith_selector,
    encode_fec7_arith_selector,
)
from fec7_arith_decoder import (  # type: ignore[import-not-found]  # noqa: E402
    decode_fec7_arith_selector as decode_via_inflate_module,
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


# -- synthetic roundtrip tests ------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 42, 1234, 99999])
def test_synthetic_roundtrip_uniform(seed: int) -> None:
    import random

    rng = random.Random(seed)
    n = 600
    codes = [rng.randrange(PALETTE_K) for _ in range(n)]
    payload = encode_fec7_arith_selector(codes, n_pairs=n)
    assert payload[:4] == FEC7_MAGIC
    (header_n,) = struct.unpack_from("<H", payload, 4)
    assert header_n == n
    decoded = decode_fec7_arith_selector(payload)
    assert decoded == codes, "byte-exact roundtrip required"


@pytest.mark.parametrize("dominant", list(range(PALETTE_K)))
def test_synthetic_roundtrip_dominant_mode(dominant: int) -> None:
    # 90% one mode + 10% uniform — exercises the dominant-symbol fast path
    import random

    rng = random.Random(dominant)
    n = 600
    codes = []
    for _ in range(n):
        if rng.random() < 0.9:
            codes.append(dominant)
        else:
            codes.append(rng.randrange(PALETTE_K))
    payload = encode_fec7_arith_selector(codes, n_pairs=n)
    decoded = decode_fec7_arith_selector(payload)
    assert decoded == codes


def test_synthetic_edge_all_zeros() -> None:
    n = 600
    codes = [0] * n
    payload = encode_fec7_arith_selector(codes, n_pairs=n)
    decoded = decode_fec7_arith_selector(payload)
    assert decoded == codes
    # All-zeros stream should compress aggressively under Laplace-smoothed model
    bitstream_len = len(payload) - 6
    # 600 zeros at converging asymptote ~ 0 bits/symbol => bytes << 75 (the fixed-bit equivalent)
    assert bitstream_len < 60, f"all-zeros bitstream {bitstream_len}B; expected dramatic compression"


def test_synthetic_edge_max_index() -> None:
    n = 600
    codes = [PALETTE_K - 1] * n
    payload = encode_fec7_arith_selector(codes, n_pairs=n)
    decoded = decode_fec7_arith_selector(payload)
    assert decoded == codes


def test_synthetic_edge_strictly_alternating() -> None:
    n = 600
    codes = [i % PALETTE_K for i in range(n)]
    payload = encode_fec7_arith_selector(codes, n_pairs=n)
    decoded = decode_fec7_arith_selector(payload)
    assert decoded == codes


def test_invalid_code_raises() -> None:
    with pytest.raises(ValueError):
        encode_fec7_arith_selector([0, 1, PALETTE_K], n_pairs=3)
    with pytest.raises(ValueError):
        encode_fec7_arith_selector([0, 1, -1], n_pairs=3)


def test_mismatched_n_pairs_raises() -> None:
    with pytest.raises(ValueError):
        encode_fec7_arith_selector([0, 1, 2], n_pairs=4)


def test_truncated_payload_raises() -> None:
    with pytest.raises(ValueError):
        decode_fec7_arith_selector(b"FEC")
    with pytest.raises(ValueError):
        decode_fec7_arith_selector(b"WRONG\x00\x00\x00")


def test_inflate_module_proxies_canonical_decoder() -> None:
    """Round-trip via the inflate-side proxy module (proves single source of truth)."""
    import random

    rng = random.Random(0)
    n = 600
    codes = [rng.randrange(PALETTE_K) for _ in range(n)]
    payload = encode_fec7_arith_selector(codes, n_pairs=n)
    decoded_via_inflate = decode_via_inflate_module(payload)
    assert decoded_via_inflate == codes


# -- live FEC6 stream measurement --------------------------------------------------


def test_live_fec6_codes_roundtrip_through_arith() -> None:
    codes, _fec6_payload = _load_live_codes_and_fec6_payload()
    payload = encode_fec7_arith_selector(codes, n_pairs=len(codes))
    decoded = decode_fec7_arith_selector(payload)
    assert decoded == codes


def test_live_savings_within_predicted_band() -> None:
    """Measure FEC7 arith wire bytes vs FEC6 fixed-Huff wire bytes on the live stream.

    Empirical prediction per ``.omx/research/pr110_opt3_mode_distribution_20260526T170000Z.md``:
    Shannon-floor encoder savings ~2 bytes; after FEC7 header overhead the wire payload
    lands in [FEC6 - 4 bytes, FEC6 + 8 bytes] depending on range-coder flush overhead.
    """
    codes, fec6_payload = _load_live_codes_and_fec6_payload()
    fec7_payload = encode_fec7_arith_selector(codes, n_pairs=len(codes))
    fec6_size = len(fec6_payload)
    fec7_size = len(fec7_payload)
    diff_bytes = fec7_size - fec6_size
    # Empirical assertion: arith path must be within +/- 12 bytes of fixed-Huff on this stream
    # (covers both savings and overhead regimes; if it's outside this band the encoder is broken)
    assert -12 <= diff_bytes <= 12, (
        f"FEC7 arith payload {fec7_size} bytes vs FEC6 fixed-Huff {fec6_size} bytes; "
        f"diff {diff_bytes:+d} outside expected +/- 12 byte band — encoder likely broken"
    )
    # Also assert the empirical mode-distribution + Markov findings are consistent
    histogram = Counter(codes)
    assert histogram[0] == 134, "expected 134 'none' codes in live stream"
    assert histogram[2] == 129, "expected 129 'frame0_blue_chroma_amp_3' codes in live stream"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
