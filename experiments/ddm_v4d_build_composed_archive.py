#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_v4d — build the COMPOSED v4d archive on the cell_drop50 base.

v4d = v4c (base-adjudicated cell_drop50 tokens + RE-SOLVED static two-plane pose
+ per-pair (a,b) exposure + kl1 lossless trio) PLUS two rungs:
  QA66  per-pair rolling-shutter beta magnitude index (NEW section 5), sign free
        from yaw.  buys the v4c-measured -0.0135 S.
  QA65  dim0 OFFSET-CODED pose column (`--dim0-offset auto`): store f16 residual
        (mean-subtracted), 19.3x finer quantum, re-solved on that lattice.  0
        extra counted bytes vs plain f16; the offset is one manifest float.

Members tokens/renderer/selector/pose_stub are the cell_drop50 bytes verbatim;
only state/pose_warp.stp (grammar v4d PFS1WPD1) + manifest change.  Compressible
config members are DEFLATE'd (receiver reads the UNZIPPED dir; ZIP method
transparent; rate = archive.zip size => free lossless container win).

Input: a "final JSONL" (per pair {pair, p:[6], a, b, selector, beta_idx}) from
ddm_v4d_resolve.py (mode qa66 = data-ready; mode refine = re-solved).  Every
stored byte is consumed by inflate_runner_v4d (parse-back bijection #417).

Axis: [macOS-CPU advisory] NON-PROMOTABLE; pointer 0.1910828242 UNMOVED.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import struct
import zipfile
from pathlib import Path

import brotli
import numpy as np

BASE = Path("/Volumes/VertigoDataTier/pact/ddm_gr1_20260730/gr1_cell_drop50_archive.zip")
OUT = Path("/Volumes/VertigoDataTier/pact/ddm_v4d_20260731")
RECEIVER_SRC = Path("experiments/inflate_runner_v4d.py")
MAGIC = b"PFS1WPD1"
KL1_MAGIC = b"KL1PWF01"
FRAME0_POLICY = "warp_two_plane_static_photo_beta_v4d"
BETA_MAGS = (0.0, 0.5, 1.0)
MEMBER_ORDER = ("manifest.json", "state/tokens.dr7t", "state/renderer.sec",
                "state/selector.sec", "state/pose_stub.sec", "state/pose_warp.stp")
DEFLATE_MEMBERS = {"manifest.json", "state/selector.sec"}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def encode_kl1_field(field_f16: np.ndarray) -> bytes:
    """Byte-plane column-major brotli lossless codec (kl1 KL1PWF01)."""
    assert field_f16.dtype == np.float16
    n, d = field_f16.shape
    cm = np.ascontiguousarray(field_f16.T).view(np.uint16).astype(np.uint16)  # (d,n)
    hi = (cm >> 8).astype(np.uint8).reshape(-1)
    lo = (cm & 0xFF).astype(np.uint8).reshape(-1)
    payload = brotli.compress(np.concatenate([hi, lo]).tobytes(), quality=11)
    return KL1_MAGIC + struct.pack("<HHI", n, d, len(payload)) + payload


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            zi = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            zi.external_attr = 0o644 << 16
            if name in DEFLATE_MEMBERS:
                zi.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(zi, data, compresslevel=9)
            else:
                zi.compress_type = zipfile.ZIP_STORED
                z.writestr(zi, data)
    return buf.getvalue()


def _read_members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as z:
        return {i.filename: z.read(i.filename) for i in z.infolist()}


def _st_coded_from_base(pw: bytes) -> tuple[bytes, int]:
    off = 8
    (n_pairs,) = struct.unpack_from("<I", pw, off)
    off += 4
    (l1,) = struct.unpack_from("<I", pw, off)
    off += 4 + l1
    (l2,) = struct.unpack_from("<I", pw, off)
    off += 4
    return pw[off:off + l2], n_pairs


def _load_final(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for ln in path.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            rows[int(r["pair"])] = r
    return rows


def build(args: argparse.Namespace) -> None:
    members = _read_members(BASE)
    st_coded, n_pairs = _st_coded_from_base(members["state/pose_warp.stp"])

    final = _load_final(Path(args.final_jsonl))
    if sorted(final) != list(range(n_pairs)):
        raise SystemExit(f"final jsonl not complete over 0..{n_pairs-1} "
                         f"({len(final)} rows)")
    poses = np.zeros((n_pairs, 6), np.float64)
    ab = np.zeros((n_pairs, 2), np.float64)
    sel = np.zeros(n_pairs, np.uint8)
    beta_idx = np.zeros(n_pairs, np.uint8)
    for i in range(n_pairs):
        poses[i] = np.asarray(final[i]["p"], np.float64)
        ab[i] = [float(final[i]["a"]), float(final[i]["b"])]
        sel[i] = int(final[i]["selector"])
        beta_idx[i] = int(final[i]["beta_idx"])

    # QA65 dim0 offset-coding: store f16 residual, receiver adds the offset.
    dim0_offset = None
    pose_store = poses.copy()
    if args.dim0_offset == "auto":
        dim0_offset = float(np.float16(poses[:, 0].mean()))
    elif args.dim0_offset != "none":
        dim0_offset = float(args.dim0_offset)
    if dim0_offset is not None:
        pose_store[:, 0] = poses[:, 0] - dim0_offset  # f16-quantized below

    tp_member = encode_kl1_field(pose_store.astype(np.float16))
    ab_member = encode_kl1_field(ab.astype(np.float16))
    sel_packed = np.packbits(sel)
    sel_coded = brotli.compress(sel_packed.tobytes(), quality=11)
    beta_coded = brotli.compress(beta_idx.tobytes(), quality=11)
    new_pw = (MAGIC + struct.pack("<I", n_pairs)
              + struct.pack("<I", len(tp_member)) + tp_member
              + struct.pack("<I", len(st_coded)) + st_coded
              + struct.pack("<I", len(sel_coded)) + sel_coded
              + struct.pack("<I", len(ab_member)) + ab_member
              + struct.pack("<I", len(beta_coded)) + beta_coded)

    manifest = json.loads(members["manifest.json"])
    manifest["frame0_policy"] = FRAME0_POLICY
    manifest["pose_carrier"] = "two_plane_static_photo_beta_v4d"
    manifest["pose_warp_sha256"] = _sha(new_pw)
    manifest["schema"] = "ddm_v4d_composed_archive.v4d_static_photo_beta"
    manifest["selector_num_two"] = int(sel.sum())
    manifest["rs_beta_mags"] = list(BETA_MAGS)
    manifest["beta_idx_counts"] = [int((beta_idx == k).sum()) for k in range(3)]
    manifest["base"] = "cell_drop50"
    if dim0_offset is not None:
        manifest["pose_dim0_offset"] = dim0_offset
    new_manifest = (json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                    .encode())

    out_members = {}
    for name in MEMBER_ORDER:
        if name == "manifest.json":
            out_members[name] = new_manifest
        elif name == "state/pose_warp.stp":
            out_members[name] = new_pw
        else:
            out_members[name] = members[name]
    archive_bytes = _zip(out_members)

    OUT.mkdir(parents=True, exist_ok=True)
    out_zip = OUT / f"v4d_composed_{args.tag}_archive.zip"
    out_zip.write_bytes(archive_bytes)
    shutil.copy(RECEIVER_SRC, OUT / "inflate_runner_v4d.py")

    base_b = len(BASE.read_bytes())
    v4c_b = 359750
    d_final = np.array([float(final[i].get("d_final", np.nan)) for i in range(n_pairs)])
    receipt = {
        "schema": "ddm_v4d_composed_build.v1",
        "axis": "[macOS-CPU advisory] NON-PROMOTABLE; pointer 0.1910828242 UNMOVED",
        "score_claim": False, "tag": args.tag, "base": "cell_drop50",
        "grammar": "v4d_static_photo_beta", "dim0_offset": dim0_offset,
        "final_jsonl": args.final_jsonl, "n_selector_two": int(sel.sum()),
        "beta_idx_counts": manifest["beta_idx_counts"],
        "archive_zip": str(out_zip), "archive_bytes": len(archive_bytes),
        "archive_sha256": _sha(archive_bytes),
        "base_archive_bytes": base_b, "v4c_archive_bytes": v4c_b,
        "delta_bytes_vs_base": len(archive_bytes) - base_b,
        "delta_bytes_vs_v4c": len(archive_bytes) - v4c_b,
        "pose_warp_bytes": len(new_pw), "tp_member_bytes": len(tp_member),
        "st_coded_bytes": len(st_coded), "sel_coded_bytes": len(sel_coded),
        "ab_member_bytes": len(ab_member), "beta_coded_bytes": len(beta_coded),
        "rate_term": 25.0 * len(archive_bytes) / 37_545_489,
        "v4c_rate_term": 25.0 * v4c_b / 37_545_489,
        "pose_warp_sha256": manifest["pose_warp_sha256"],
        "frame0_policy": FRAME0_POLICY,
        "advisory_mean_d_final": float(np.nanmean(d_final)),
        "advisory_contribution_final": float(np.sqrt(10.0 * np.nanmean(d_final))),
        "seg_note": "seg is MEASURED at the gate; cell_drop50 tokens => "
                    "d_seg ~0.004312 (v4c gate MEASURED 0.00431179).",
        "note": "tokens/renderer/selector/pose_stub = cell_drop50 verbatim; only "
                "pose_warp.stp (v4d PFS1WPD1, +beta section) + manifest change. "
                "Verify ddm_v4d_verify_decode.py; gate stage_v4d_realized_gate.sh.",
    }
    (OUT / f"v4d_composed_{args.tag}_build_receipt.json").write_text(
        json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="qa66_celldrop50")
    ap.add_argument("--final-jsonl", required=True,
                    help="the v4d final JSONL (pose+a,b+selector+beta_idx)")
    ap.add_argument("--dim0-offset", default="none",
                    help="auto (shared f16 mean) | none | <float>")
    args = ap.parse_args()
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
