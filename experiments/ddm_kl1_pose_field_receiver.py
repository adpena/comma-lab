#!/usr/bin/env python3
"""ddm_kl1 LEG B1 — the RECEIVER decoder (rule-118 free) + encoder for the pose
field byte-plane lossless codec. This is the REAL implementation the v4c grammar
would ship: encoder produces a compact member; the decoder (pure generic code,
no video-derived data) reconstructs the EXACT 600x6 f16 field.

Round-trip is PROVEN bit-exact on the live D2/ck1 fields -> zero d_pose risk.
Pointer 0.1910828242 UNMOVED; [macOS-CPU advisory]; score_claim=false.
"""
from __future__ import annotations

import json
import struct

import brotli
import numpy as np

MAGIC = b"KL1PWF01"  # pose-warp-field byteplane codec v01


def encode_pose_field(field_f16: np.ndarray) -> bytes:
    """field_f16: (n, d) float16. Returns compact member bytes (LOSSLESS)."""
    assert field_f16.dtype == np.float16
    n, d = field_f16.shape
    cm = np.ascontiguousarray(field_f16.T).view(np.uint16).astype(np.uint16)  # (d,n)
    hi = (cm >> 8).astype(np.uint8).reshape(-1)
    lo = (cm & 0xFF).astype(np.uint8).reshape(-1)
    payload = brotli.compress(np.concatenate([hi, lo]).tobytes(), quality=11)
    return MAGIC + struct.pack("<HHI", n, d, len(payload)) + payload


def decode_pose_field(member: bytes) -> np.ndarray:
    """RECEIVER side (rule-118 free): reconstruct exact (n,d) float16 field."""
    assert member[:8] == MAGIC, "bad magic"
    n, d, plen = struct.unpack("<HHI", member[8:16])
    raw = brotli.decompress(member[16:16 + plen])
    half = n * d
    hi = np.frombuffer(raw[:half], dtype=np.uint8).astype(np.uint16)
    lo = np.frombuffer(raw[half:2 * half], dtype=np.uint8).astype(np.uint16)
    cm = ((hi << 8) | lo).reshape(d, n)  # (d,n) uint16
    return np.ascontiguousarray(cm.T).view(np.float16)  # (n,d)


def _load(path, key):
    rows = {json.loads(l)["pair"]: json.loads(l) for l in open(path) if l.strip()}
    return np.array([rows[p][key] for p in sorted(rows)], dtype=np.float16)


def _verify(path, key, label):
    F = _load(path, key)
    m = encode_pose_field(F)
    R = decode_pose_field(m)
    ok = np.array_equal(R.view(np.uint16), F.view(np.uint16))
    return {
        "label": label, "n": F.shape[0], "d": F.shape[1],
        "raw_bytes": F.size * 2, "member_bytes": len(m),
        "header_bytes": 16, "lossless_bit_exact": bool(ok),
        "saved_vs_raw": F.size * 2 - len(m),
    }


if __name__ == "__main__":
    res = [
        _verify("/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d2/d2_ep_solve.partial.jsonl",
                "p_star", "D2_pstar_P0"),
        _verify("/Volumes/VertigoDataTier/pact/ddm_ck1_20260729/ck1_solve.partial.jsonl",
                "p_best_kneeA", "ck1_pbest"),
        _verify("/Volumes/VertigoDataTier/pact/ddm_qa43_20260729/two_plane_probe_v2.partial.jsonl",
                "p_two_star", "qa43_ptwo_tail"),
    ]
    out = {"schema": "ddm_kl1_pose_field_receiver.v1",
           "pointer": "0.1910828242 [contest-CPU] UNMOVED",
           "axis": "[macOS-CPU advisory] NON-PROMOTABLE", "score_claim": False,
           "note": ("byteplane-colmajor-brotli LOSSLESS codec; decoder is rule-118 "
                    "free generic code; brotli member includes payload len header"),
           "results": res}
    print(json.dumps(out, indent=1))
    with open("/Volumes/VertigoDataTier/pact/ddm_kl1_20260730/b1_receiver_verify.json", "w") as f:
        json.dump(out, f, indent=1)
