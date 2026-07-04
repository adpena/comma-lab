# SPDX-License-Identifier: MIT
"""Python side of the tac-levelset-inflate parity contract.

Re-derives the decoded-output SHA-256 from the REAL `tac` ORACLE on the committed
golden-vector input fixtures and asserts it equals each manifest's pinned digest. This
is the Python mirror of the Rust `assert_sha256_parity` gate: BOTH sides must agree on
the same byte-for-byte digest, proving the Rust decode reproduces the Python oracle
exactly.

Run:  .venv/bin/python runtime-rs/crates/tac-levelset-inflate/python_reference_equivalence_test.py

Exit code 0 = all vectors agree (Python oracle == committed manifest). Nonzero = a vector
drifted (regenerate via generate_golden_vectors.py and re-prove the Rust side).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GV = HERE / "golden_vectors"
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from tac.lossless.range_coder import decode_static_symbols  # noqa: E402


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _i64le(arr) -> bytes:
    return np.asarray(arr, dtype="<i8").tobytes(order="C")


def _manifest(name: str) -> dict:
    return json.loads((GV / f"{name}.json").read_text())


def _bin(name: str) -> bytes:
    return (GV / name).read_bytes()


def _freqs(name: str) -> list[int]:
    return np.frombuffer(_bin(name), dtype="<u4").astype(np.int64).tolist()


def check_range_decode() -> bool:
    m = _manifest("levelset_xi_range_decode_v1")
    stream = _bin("levelset_xi_range_decode_v1_stream.bin")
    freqs = _freqs("levelset_xi_range_decode_v1_frequencies.bin")
    symbols = decode_static_symbols(stream, count=int(m["count"]), frequencies=freqs)
    return _sha(_i64le(symbols)) == m["sha256"]


def check_xi_column() -> bool:
    m = _manifest("levelset_xi_column_delta_v1")
    stream = _bin("levelset_xi_column_delta_v1_stream.bin")
    freqs = _freqs("levelset_xi_column_delta_v1_frequencies.bin")
    p_count = int(m["p_count"])
    seed = int(m["seed"])
    lo = int(m["lo"])
    syms = np.asarray(decode_static_symbols(stream, count=p_count - 1, frequencies=freqs), dtype=np.int64)
    col = np.empty(p_count, dtype=np.int64)
    col[0] = seed
    col[1:] = seed + np.cumsum(syms + lo)
    return _sha(_i64le(col)) == m["sha256"]


def main() -> int:
    checks = [
        ("levelset_xi_range_decode_v1", check_range_decode()),
        ("levelset_xi_column_delta_v1", check_xi_column()),
    ]
    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print("Python oracle == committed manifest:", "ALL PASS" if ok else "DRIFT DETECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
