#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_v4b — build the COMPOSED v4b archive (Knee-A tokens + two-plane pose field
+ per-pair selector).  Task #776.

The composed candidate is the Knee-A byte-closed token base (unchanged: the
-0.197 S rate win, d_seg 0.00553676 measured at the wr1 gate) with its pose
member re-encoded to grammar v4b:
  * pose field  = p_best per pair (600 x 6 f16): p_two_star where selector=1 else
                  p_single_kneeA  (both KNEE-base re-solves, ck1_solve).
  * selector    = per-pair bit (600 -> packbits -> 75 B): 1 iff the two-plane
                  STATIC compose beat the single-plane fallback (ddm_v4b transfer).
  * s_t stream  = the Knee-A s_t index stream, REUSED VERBATIM (unchanged).
The receiver (experiments/inflate_runner_v4b.py, rule-118 FREE code, NOT counted)
warps f0 by a per-pair single/two-plane compose; the two-plane far/ground split
is a STATIC horizon at v = round(cy) = 437 DERIVED at decode from the intrinsics
(0 shipped mask bytes).

Everything is byte-closed: token/renderer/selector/pose_stub members are the
Knee-A bytes verbatim; only state/pose_warp.stp changes (new MAGIC PFS1WPB2 +
selector section).  Emits archive.zip + copies inflate_runner_v4b.py to the SSD.
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

KNEEA = Path("/Volumes/VertigoDataTier/pact/ddm_wr1_20260729/wr1_kneeA_safe_274k_archive.zip")
SHIP_TABLE = Path("/Volumes/VertigoDataTier/pact/ddm_v4b_20260730/v4b_ship_table.json")
RECEIVER_SRC = Path("experiments/inflate_runner_v4b.py")
OUT = Path("/Volumes/VertigoDataTier/pact/ddm_v4b_20260730")
MAGIC = b"PFS1WPB2"
FRAME0_POLICY = "warp_two_plane_static_v4b"
MEMBER_ORDER = ("manifest.json", "state/tokens.dr7t", "state/renderer.sec",
                "state/selector.sec", "state/pose_stub.sec", "state/pose_warp.stp")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _stored_zip(members: dict[str, bytes]) -> bytes:
    """Deterministic ZIP_STORED archive (fixed date_time, insertion order) — the
    v3_warp/v4b grammar is a plain stored ZIP the receiver reads by member name;
    no tac dependency."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            zi = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_STORED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, data)
    return buf.getvalue()


def _read_members(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as z:
        return {i.filename: z.read(i.filename) for i in z.infolist()}


def _load_ship_table() -> tuple[np.ndarray, np.ndarray]:
    obj = json.loads(SHIP_TABLE.read_text())
    rows = {int(r["pair"]): r for r in obj["rows"]}
    n = 600
    if sorted(rows) != list(range(n)):
        raise SystemExit(f"ship table not complete over 0..599 ({len(rows)} rows)")
    poses = np.zeros((n, 6), np.float64)
    sel = np.zeros(n, np.uint8)
    for i in range(n):
        poses[i] = np.asarray(rows[i]["p"], np.float64)
        sel[i] = int(rows[i]["selector"])
    return poses, sel


def build(args: argparse.Namespace) -> None:
    members = _read_members(KNEEA)
    # reuse the s_t stream verbatim (s_t index unchanged by the pose re-solve)
    pw = members["state/pose_warp.stp"]
    off = 8
    (n_pairs,) = struct.unpack_from("<I", pw, off)
    off += 4
    (l1,) = struct.unpack_from("<I", pw, off)
    off += 4 + l1
    (l2,) = struct.unpack_from("<I", pw, off)
    off += 4
    st_coded = pw[off:off + l2]

    poses, sel = _load_ship_table()
    if len(poses) != n_pairs:
        raise SystemExit(f"ship table n={len(poses)} vs Knee-A n_pairs={n_pairs}")

    poses_f16 = poses.astype(np.float16)
    tp_coded = brotli.compress(np.ascontiguousarray(poses_f16).tobytes(), quality=11)
    sel_packed = np.packbits(sel)  # 600 bits -> 75 bytes
    sel_coded = brotli.compress(sel_packed.tobytes(), quality=11)
    new_pw = (MAGIC + struct.pack("<I", n_pairs)
              + struct.pack("<I", len(tp_coded)) + tp_coded
              + struct.pack("<I", len(st_coded)) + st_coded
              + struct.pack("<I", len(sel_coded)) + sel_coded)

    manifest = json.loads(members["manifest.json"])
    manifest["frame0_policy"] = FRAME0_POLICY
    manifest["pose_carrier"] = "two_plane_static_v4b"
    manifest["pose_warp_sha256"] = _sha(new_pw)
    manifest["schema"] = "ddm_v4b_composed_archive.v4b_two_plane_static"
    manifest["selector_num_two"] = int(sel.sum())
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
    archive_bytes = _stored_zip(out_members)

    out_zip = OUT / f"v4b_composed_{args.tag}_archive.zip"
    out_zip.write_bytes(archive_bytes)
    shutil.copy(RECEIVER_SRC, OUT / "inflate_runner_v4b.py")

    kb = len(KNEEA.read_bytes())
    receipt = {
        "schema": "ddm_v4b_composed_build.v1",
        "axis": "[macOS-CPU advisory] NON-PROMOTABLE; pointer 0.1910828242 UNMOVED",
        "score_claim": False, "tag": args.tag, "grammar": "v4b_two_plane_static",
        "ship_table": str(SHIP_TABLE), "n_selector_two": int(sel.sum()),
        "archive_zip": str(out_zip), "archive_bytes": len(archive_bytes),
        "archive_sha256": _sha(archive_bytes),
        "kneeA_bytes": kb, "delta_bytes_vs_kneeA": len(archive_bytes) - kb,
        "pose_warp_bytes": len(new_pw),
        "tp_coded_bytes": len(tp_coded), "st_coded_bytes": len(st_coded),
        "sel_coded_bytes": len(sel_coded), "sel_packed_bytes": len(sel_packed),
        "rate_term": 25.0 * len(archive_bytes) / 37_545_489,
        "kneeA_rate_term": 25.0 * kb / 37_545_489,
        "pose_warp_sha256": manifest["pose_warp_sha256"],
        "frame0_policy": FRAME0_POLICY,
        "note": "token/renderer/selector/pose_stub members are Knee-A bytes "
                "verbatim; only pose_warp.stp changed (v4b MAGIC + selector "
                "section). inflate_runner_v4b.py copied alongside. Verify with "
                "ddm_v4b_verify_decode.py; gate via stage_v4b_realized_gate.sh.",
    }
    (OUT / f"v4b_composed_{args.tag}_build_receipt.json").write_text(
        json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="static_kneeA")
    args = ap.parse_args()
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
