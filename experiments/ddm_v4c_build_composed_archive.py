#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_v4c — build the COMPOSED v4c archive on the cell_drop50 base.

v4c = base-adjudicated (cell_drop50: seg+rate -0.066 vs Knee-A, MEASURED) token
base + a RE-SOLVED-through-the-static-mask pose field (QA59/QA60) + per-pair
PHOTOMETRIC auto-exposure (a,b) (QA62 rung B) + the kl1 lossless trio.  Members
tokens/renderer/selector/pose_stub are the cell_drop50 bytes verbatim; only
state/pose_warp.stp is re-encoded to grammar v4c (MAGIC PFS1WPC1) and the
manifest updated.  Compressible config members are DEFLATE'd (receiver reads the
UNZIPPED dir, so ZIP method is transparent; rate = archive.zip size => a free
lossless container win).

The final pose field is the photo JSONL (authoritative: pose "p", selector,
(a,b) per pair): solved pairs use the re-solved static pose, unsolved pairs use
the realizable transferred Knee-A ship pose.  Every stored byte is consumed by
inflate_runner_v4c (parse-back bijection #417).

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
OUT = Path("/Volumes/VertigoDataTier/pact/ddm_v4c_20260730")
RECEIVER_SRC = Path("experiments/inflate_runner_v4c.py")
MAGIC = b"PFS1WPC1"
KL1_MAGIC = b"KL1PWF01"
FRAME0_POLICY = "warp_two_plane_static_photo_v4c"
MEMBER_ORDER = ("manifest.json", "state/tokens.dr7t", "state/renderer.sec",
                "state/selector.sec", "state/pose_stub.sec", "state/pose_warp.stp")
# members that compress (JSON text); the rest are entropy-coded => STORE
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
    """Deterministic ZIP: DEFLATE the compressible config members, STORE the
    rest (entropy-coded / already-compressed)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            zi = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            zi.external_attr = 0o644 << 16
            if name in DEFLATE_MEMBERS:
                zi.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(zi, data, compresslevel=9)  # deterministic zlib level
            else:
                zi.compress_type = zipfile.ZIP_STORED
                z.writestr(zi, data)
    return buf.getvalue()


def _read_members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as z:
        return {i.filename: z.read(i.filename) for i in z.infolist()}


def _st_coded_from_base(pw: bytes) -> bytes:
    """Extract the s_t stream (r7 codes) from the base pose_warp (PFS1WPB1)."""
    off = 8
    (n_pairs,) = struct.unpack_from("<I", pw, off)
    off += 4
    (l1,) = struct.unpack_from("<I", pw, off)
    off += 4 + l1
    (l2,) = struct.unpack_from("<I", pw, off)
    off += 4
    return pw[off:off + l2], n_pairs


def _load_photo(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for ln in path.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            rows[int(r["pair"])] = r
    return rows


def build(args: argparse.Namespace) -> None:
    members = _read_members(BASE)
    st_coded, n_pairs = _st_coded_from_base(members["state/pose_warp.stp"])

    photo = _load_photo(Path(args.photo_jsonl))
    if sorted(photo) != list(range(n_pairs)):
        raise SystemExit(f"photo jsonl not complete over 0..{n_pairs-1} "
                         f"({len(photo)} rows)")
    poses = np.zeros((n_pairs, 6), np.float64)
    ab = np.zeros((n_pairs, 2), np.float64)
    sel = np.zeros(n_pairs, np.uint8)
    for i in range(n_pairs):
        poses[i] = np.asarray(photo[i]["p"], np.float64)
        ab[i] = [float(photo[i]["a"]), float(photo[i]["b"])]
        sel[i] = int(photo[i]["selector"])

    tp_member = encode_kl1_field(poses.astype(np.float16))
    ab_member = encode_kl1_field(ab.astype(np.float16))
    sel_packed = np.packbits(sel)
    sel_coded = brotli.compress(sel_packed.tobytes(), quality=11)
    new_pw = (MAGIC + struct.pack("<I", n_pairs)
              + struct.pack("<I", len(tp_member)) + tp_member
              + struct.pack("<I", len(st_coded)) + st_coded
              + struct.pack("<I", len(sel_coded)) + sel_coded
              + struct.pack("<I", len(ab_member)) + ab_member)

    manifest = json.loads(members["manifest.json"])
    manifest["frame0_policy"] = FRAME0_POLICY
    manifest["pose_carrier"] = "two_plane_static_photo_v4c"
    manifest["pose_warp_sha256"] = _sha(new_pw)
    manifest["schema"] = "ddm_v4c_composed_archive.v4c_static_photo"
    manifest["selector_num_two"] = int(sel.sum())
    manifest["rs_beta_global"] = float(args.rs_beta)
    manifest["base"] = "cell_drop50"
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
    out_zip = OUT / f"v4c_composed_{args.tag}_archive.zip"
    out_zip.write_bytes(archive_bytes)
    shutil.copy(RECEIVER_SRC, OUT / "inflate_runner_v4c.py")

    base_b = len(BASE.read_bytes())
    seg = 0.00553676  # NOTE: Knee-A d_seg; cell_drop50 d_seg is MEASURED at gate
    receipt = {
        "schema": "ddm_v4c_composed_build.v1",
        "axis": "[macOS-CPU advisory] NON-PROMOTABLE; pointer 0.1910828242 UNMOVED",
        "score_claim": False, "tag": args.tag, "base": "cell_drop50",
        "grammar": "v4c_static_photo", "rs_beta_global": float(args.rs_beta),
        "photo_jsonl": args.photo_jsonl, "n_selector_two": int(sel.sum()),
        "archive_zip": str(out_zip), "archive_bytes": len(archive_bytes),
        "archive_sha256": _sha(archive_bytes),
        "base_archive_bytes": base_b,
        "delta_bytes_vs_base": len(archive_bytes) - base_b,
        "pose_warp_bytes": len(new_pw),
        "tp_member_bytes": len(tp_member), "st_coded_bytes": len(st_coded),
        "sel_coded_bytes": len(sel_coded), "ab_member_bytes": len(ab_member),
        "rate_term": 25.0 * len(archive_bytes) / 37_545_489,
        "base_rate_term": 25.0 * base_b / 37_545_489,
        "pose_warp_sha256": manifest["pose_warp_sha256"],
        "frame0_policy": FRAME0_POLICY,
        "seg_note": "seg is MEASURED at the gate; cell_drop50 tokens => "
                    "d_seg ~0.004310 (gr1 n600, evaluate.py band +-2.8e-5), "
                    f"NOT the Knee-A {seg}.",
        "note": "tokens/renderer/selector/pose_stub = cell_drop50 verbatim; "
                "only pose_warp.stp + manifest change. DEFLATE on manifest + "
                "selector.sec (container win). Verify ddm_v4c_verify_decode.py; "
                "gate via stage_v4c_realized_gate.sh.",
    }
    (OUT / f"v4c_composed_{args.tag}_build_receipt.json").write_text(
        json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="static_photo_celldrop50")
    ap.add_argument("--photo-jsonl", required=True,
                    help="the v4c photo JSONL (authoritative pose+selector+a,b)")
    ap.add_argument("--rs-beta", type=float, default=0.0,
                    help="rung-A rolling-shutter global beta magnitude (0 = off)")
    args = ap.parse_args()
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
