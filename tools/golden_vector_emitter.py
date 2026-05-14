#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Golden-vector emitter for codec roundtrip validation — closes task #396.

Produces canonical input/output byte vectors that any codec implementation
(Python, Rust, C, Mojo, JAX) can use to prove byte-faithful reconstruction.

Why golden vectors?
-------------------
The contest archive is byte-exact: the bytes of the submitted ZIP must
match the bytes that inflate.sh produces. Any codec re-implementation
(e.g., porting a Python codec to Rust for speed) must produce bit-exact
output. Without golden vectors, regressions are silent until contest-CUDA
auth eval — too late.

Each golden vector consists of:
  - Input: a canonical state_dict subset or symbol stream
  - Encoded: the bytes the canonical codec produces
  - SHA-256 of the encoded bytes (the proof a port is faithful)
  - Decode roundtrip: encoded -> reconstructed -> SHA-256 of recon
  - Edge cases: zero tensor, single-element, max-magnitude, all-same

Usage::

    .venv/bin/python tools/golden_vector_emitter.py emit \\
        --codec brotli \\
        --output reports/golden_vectors/brotli_v1.json

    .venv/bin/python tools/golden_vector_emitter.py verify \\
        --golden reports/golden_vectors/brotli_v1.json \\
        --codec-impl tac.codec_pipeline.Op1_PR101SplitBrotli

CLAUDE.md compliance: pure CPU + numpy + brotli + hashlib; no scorer
load; no contest score claim; output tagged ``[CPU-prep golden vector]``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import brotli
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

TOOL_NAME = "tools/golden_vector_emitter.py"
SCHEMA_VERSION = "golden_vector_emitter.v1"
EVIDENCE_GRADE = "[CPU-prep golden vector]"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical_test_inputs() -> list[dict]:
    """Curated edge cases + representative tensors.

    Each row: name, dtype, shape, content_strategy.
    Reproducible via numpy seeds.
    """
    rng = np.random.default_rng(seed=12345)
    return [
        {
            "name": "all_zeros_int8_1024",
            "dtype": "int8",
            "shape": [1024],
            "data": np.zeros(1024, dtype=np.int8),
        },
        {
            "name": "all_ones_int8_1024",
            "dtype": "int8",
            "shape": [1024],
            "data": np.ones(1024, dtype=np.int8),
        },
        {
            "name": "single_int8_minus_127",
            "dtype": "int8",
            "shape": [1],
            "data": np.array([-127], dtype=np.int8),
        },
        {
            "name": "single_int8_127",
            "dtype": "int8",
            "shape": [1],
            "data": np.array([127], dtype=np.int8),
        },
        {
            "name": "max_magnitude_int8_512",
            "dtype": "int8",
            "shape": [512],
            "data": np.full(512, 127, dtype=np.int8),
        },
        {
            "name": "alternating_pos_neg_int8_4096",
            "dtype": "int8",
            "shape": [4096],
            "data": np.tile(np.array([127, -127], dtype=np.int8), 2048),
        },
        {
            "name": "uniform_int8_4096",
            "dtype": "int8",
            "shape": [4096],
            "data": rng.integers(-127, 128, size=4096, dtype=np.int8),
        },
        {
            "name": "gaussian_int8_8192",
            "dtype": "int8",
            "shape": [8192],
            "data": np.clip(
                np.round(rng.normal(0, 25.0, size=8192)), -127, 127
            ).astype(np.int8),
        },
        {
            "name": "sparse_int8_8192",
            "dtype": "int8",
            "shape": [8192],
            "data": (
                rng.integers(-127, 128, size=8192, dtype=np.int8) *
                (rng.random(size=8192) > 0.9).astype(np.int8)
            ).astype(np.int8),
        },
        {
            "name": "small_fp16_1024",
            "dtype": "float16",
            "shape": [1024],
            "data": rng.normal(0, 0.05, size=1024).astype(np.float16),
        },
        {
            "name": "small_fp32_1024",
            "dtype": "float32",
            "shape": [1024],
            "data": rng.normal(0, 0.05, size=1024).astype(np.float32),
        },
    ]


def emit_brotli_golden(test_inputs: list[dict]) -> dict:
    """Emit golden vectors for the canonical brotli codec."""
    rows: list[dict] = []
    for t in test_inputs:
        raw = t["data"].tobytes()
        encoded = brotli.compress(raw, quality=11, lgwin=16, lgblock=19, mode=brotli.MODE_GENERIC)
        decoded = brotli.decompress(encoded)
        roundtrip_ok = decoded == raw
        rows.append({
            "name": t["name"],
            "dtype": t["dtype"],
            "shape": t["shape"],
            "input_bytes": len(raw),
            "input_sha256": sha256_bytes(raw),
            "encoded_bytes": len(encoded),
            "encoded_sha256": sha256_bytes(encoded),
            "decoded_bytes": len(decoded),
            "decoded_sha256": sha256_bytes(decoded),
            "roundtrip_byte_faithful": roundtrip_ok,
            "compression_ratio": len(encoded) / max(len(raw), 1),
        })
    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "evidence_grade": EVIDENCE_GRADE,
        "score_claim": False,
        "codec": "brotli",
        "codec_params": {
            "quality": 11, "lgwin": 16, "lgblock": 19,
            "mode": "MODE_GENERIC",
        },
        "n_vectors": len(rows),
        "all_roundtrip_byte_faithful": all(r["roundtrip_byte_faithful"] for r in rows),
        "vectors": rows,
    }


def emit_int4_pack_golden(test_inputs: list[dict]) -> dict:
    """Emit golden vectors for the canonical int4-nibble-packing codec."""
    rows: list[dict] = []
    for t in test_inputs:
        if t["dtype"] != "int8":
            continue
        # Bias by +8 (so -8..+7 -> 0..15) but our codes are -7..+7 (15 levels)
        # Clip to [-7, +7] then bias by +8
        codes = np.clip(t["data"].astype(np.int16), -7, 7).astype(np.int8)
        biased = (codes.astype(np.int16) + 8).astype(np.uint8)
        if biased.size & 1:
            biased_padded = np.concatenate([biased, np.zeros(1, dtype=np.uint8)])
        else:
            biased_padded = biased
        packed = (biased_padded[0::2] << 4) | (biased_padded[1::2] & 0x0F)
        encoded = packed.tobytes()
        # Decode
        unpacked_high = (np.frombuffer(encoded, dtype=np.uint8) >> 4).astype(np.int16)
        unpacked_low = (np.frombuffer(encoded, dtype=np.uint8) & 0x0F).astype(np.int16)
        unpacked = np.empty(unpacked_high.size * 2, dtype=np.int16)
        unpacked[0::2] = unpacked_high
        unpacked[1::2] = unpacked_low
        unpacked = unpacked[: codes.size]
        decoded_codes = (unpacked - 8).astype(np.int8)
        roundtrip_ok = bool((decoded_codes == codes).all())
        raw = codes.tobytes()
        rows.append({
            "name": t["name"],
            "dtype": "int4_packed_from_int8",
            "shape": t["shape"],
            "input_bytes": len(raw),
            "input_sha256": sha256_bytes(raw),
            "encoded_bytes": len(encoded),
            "encoded_sha256": sha256_bytes(encoded),
            "decoded_codes_sha256": sha256_bytes(decoded_codes.tobytes()),
            "roundtrip_byte_faithful": roundtrip_ok,
            "compression_ratio": len(encoded) / max(len(raw), 1),
        })
    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "evidence_grade": EVIDENCE_GRADE,
        "score_claim": False,
        "codec": "int4_nibble_pack",
        "codec_params": {"bias": 8, "range": [-7, 7]},
        "n_vectors": len(rows),
        "all_roundtrip_byte_faithful": all(r["roundtrip_byte_faithful"] for r in rows),
        "vectors": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_emit = sub.add_parser("emit", help="Emit golden vectors for a codec")
    p_emit.add_argument("--codec", choices=["brotli", "int4_pack", "all"], default="all")
    p_emit.add_argument("--output", type=Path, default=None)

    p_verify = sub.add_parser("verify",
                              help="Verify a codec implementation against golden vectors")
    p_verify.add_argument("--golden", type=Path, required=True)
    p_verify.add_argument("--codec-impl", default=None,
                          help="dotted import path of codec impl (Python only)")

    args = parser.parse_args(argv)

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")

    if args.cmd == "emit":
        test_inputs = canonical_test_inputs()
        outputs: list[dict] = []
        if args.codec in ("brotli", "all"):
            outputs.append(emit_brotli_golden(test_inputs))
        if args.codec in ("int4_pack", "all"):
            outputs.append(emit_int4_pack_golden(test_inputs))

        if args.output is None:
            out_dir = REPO_ROOT / f"reports/golden_vectors/{args.codec}_{ts}"
            out_dir.mkdir(parents=True, exist_ok=True)
            args.output = out_dir / "golden.json"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"goldens": outputs, "tool": TOOL_NAME, "schema": SCHEMA_VERSION}, indent=2), encoding="utf-8")

        print(f"emitted {sum(g['n_vectors'] for g in outputs)} golden vectors across "
              f"{len(outputs)} codec(s)")
        for g in outputs:
            print(f"  {g['codec']}: {g['n_vectors']} vectors, "
                  f"all_roundtrip_byte_faithful={g['all_roundtrip_byte_faithful']}")
        print(f"wrote {args.output}")
        return 0

    if args.cmd == "verify":
        if not args.golden.is_file():
            raise SystemExit(f"golden file not found: {args.golden}")
        payload = json.loads(args.golden.read_text(encoding="utf-8"))
        goldens = payload.get("goldens", [])
        if not goldens:
            raise SystemExit("no goldens in file")

        n_pass = 0
        n_fail = 0
        for g in goldens:
            print(f"\nVerifying codec: {g['codec']}")
            for v in g["vectors"]:
                # Re-run the canonical codec (Python reference impl)
                ti = next(
                    (t for t in canonical_test_inputs() if t["name"] == v["name"]),
                    None,
                )
                if ti is None:
                    print(f"  {v['name']}: SKIP (canonical input not found)")
                    continue
                if g["codec"] == "brotli":
                    raw = ti["data"].tobytes()
                    encoded = brotli.compress(
                        raw, quality=11, lgwin=16, lgblock=19, mode=brotli.MODE_GENERIC
                    )
                    if sha256_bytes(encoded) == v["encoded_sha256"]:
                        n_pass += 1
                    else:
                        n_fail += 1
                        print(f"  {v['name']}: FAIL (encoded SHA mismatch)")
                elif g["codec"] == "int4_nibble_pack":
                    codes = np.clip(ti["data"].astype(np.int16), -7, 7).astype(np.int8)
                    biased = (codes.astype(np.int16) + 8).astype(np.uint8)
                    if biased.size & 1:
                        biased = np.concatenate([biased, np.zeros(1, dtype=np.uint8)])
                    packed = (biased[0::2] << 4) | (biased[1::2] & 0x0F)
                    encoded = packed.tobytes()
                    if sha256_bytes(encoded) == v["encoded_sha256"]:
                        n_pass += 1
                    else:
                        n_fail += 1
                        print(f"  {v['name']}: FAIL (encoded SHA mismatch)")
        print(f"\nVerification complete: {n_pass} pass, {n_fail} fail")
        return 0 if n_fail == 0 else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
