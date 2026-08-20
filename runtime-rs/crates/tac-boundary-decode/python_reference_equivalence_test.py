# SPDX-License-Identifier: MIT
"""Python side of the tac-boundary-decode parity contract.

Re-derives the decoded-raw SHA-256 from the REAL `tac.boundary_math` ORACLE on
the committed golden-vector input fixtures and asserts it equals each manifest's
pinned digest. This is the Python mirror of the Rust `assert_sha256_parity`
gate: BOTH sides must agree on the same byte-for-byte digest, proving the Rust
decode reproduces the Python oracle exactly.

Run:  .venv/bin/python runtime-rs/crates/tac-boundary-decode/python_reference_equivalence_test.py

Exit code 0 = all vectors agree (Python oracle == committed manifest). Nonzero =
a vector drifted (regenerate via generate_golden_vectors.py and re-prove the
Rust side).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from tac.boundary_math.bitmask_dseg import d_seg_reference, flip_count
from tac.boundary_math.dense_raster_lzma_baseline import ContourCode, decode_partition
from tac.boundary_math.partition import connected_components

HERE = Path(__file__).resolve().parent / "golden_vectors"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _manifest(name: str) -> dict:
    return json.loads((HERE / f"{name}.json").read_text())


def _bin(name: str) -> bytes:
    return (HERE / name).read_bytes()


def check_contour(name: str, payload_fixture: str) -> bool:
    m = _manifest(name)
    payload = _bin(payload_fixture)
    code = ContourCode(payload=payload, shape=(m["height"], m["width"]), n_classes=m["n_classes"])
    decoded = decode_partition(code).astype(np.uint8).tobytes(order="C")
    return _sha(decoded) == m["sha256"]


def check_dseg() -> bool:
    m = _manifest("dseg_popcount_v1")
    h, w = m["height"], m["width"]
    cand = np.frombuffer(_bin("dseg_popcount_v1_cand.bin"), dtype=np.uint8).reshape(h, w).astype(np.int64)
    gt = np.frombuffer(_bin("dseg_popcount_v1_gt.bin"), dtype=np.uint8).reshape(h, w).astype(np.int64)
    fc = flip_count(cand, gt)
    dseg = d_seg_reference(cand, gt)
    out = int(fc).to_bytes(8, "little") + np.float64(dseg).tobytes()
    return _sha(out) == m["sha256"]


def check_connected_components() -> bool:
    m = _manifest("connected_components_v1")
    h, w = m["height"], m["width"]
    argmax = np.frombuffer(_bin("connected_components_v1_argmax.bin"), dtype=np.uint8).reshape(h, w)
    region_of, _ = connected_components(argmax, m["n_classes"])
    out = region_of.astype(np.int32).tobytes(order="C")
    return _sha(out) == m["sha256"]


def main() -> int:
    checks = [
        ("contour_decode_full_v1", check_contour("contour_decode_full_v1", "contour_decode_full_v1_payload.bin")),
        ("contour_decode_small_v1", check_contour("contour_decode_small_v1", "contour_decode_small_v1_payload.bin")),
        ("dseg_popcount_v1", check_dseg()),
        ("connected_components_v1", check_connected_components()),
    ]
    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print("Python oracle == committed manifest:", "ALL PASS" if ok else "DRIFT DETECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
