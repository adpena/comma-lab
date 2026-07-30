#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_ck1 — build the COMPOSED candidate archive (Knee-A tokens + re-solved pose).

The composed candidate is the Knee-A byte-closed token base (unchanged: the
−0.197 S rate win) with its stale pose member REPLACED by the pose field
re-solved ON the Knee-A base (ddm_ck1 --mode solve).  The re-solved single-plane
6-DOF field uses ROTATION (s_r=1.0), so the receiver must warp with s_r=1.0 —
a one-line generic-code amendment (grammar v4a, rule-118 free, 0 counted bytes).

Grammar v4a (this builder, single-plane): ship p_solved (6 f16/pair) in the SAME
pose_warp.stp member the Knee-A archive uses; the v4a inflate_runner warps
f0 = warp(f1, H(p_solved; s_t, s_r=1.0)) instead of the gate's s_r=0.
The two-plane grammar v4b (near/far multi-plane + per-pair selector bit +
partition masks) is a SEPARATE gate — its mask source (static geometric prior
vs decoded partition) is an open A/B noted in the memo, NOT built here.

Everything is byte-closed: the token/renderer/selector/pose_stub members are the
Knee-A bytes verbatim; only pose_warp.stp changes.  Emits the composed archive.zip
+ the v4a inflate_runner decode dep on the SSD.  Axis: advisory; pointer UNMOVED.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zipfile
from pathlib import Path

import brotli
import numpy as np

KNEEA = Path("/Volumes/VertigoDataTier/pact/ddm_wr1_20260729/wr1_kneeA_safe_274k_archive.zip")
CK1_SOLVE = Path("/Volumes/VertigoDataTier/pact/ddm_ck1_20260729/ck1_solve.partial.jsonl")
OUT = Path("/Volumes/VertigoDataTier/pact/ddm_ck1_20260729")
MAGIC = b"PFS1WPB1"
MEMBER_ORDER = ("manifest.json", "state/tokens.dr7t", "state/renderer.sec",
                "state/selector.sec", "state/pose_stub.sec", "state/pose_warp.stp")

# the v4a inflate_runner: identical to the gate's inflate_runner EXCEPT the f0
# warp uses s_r=1.0 (rotation active) on the re-solved pose (rule-118 generic).
V4A_INFLATE_RUNNER = r'''from __future__ import annotations

import json
import struct
import sys
from pathlib import Path, PurePosixPath

import brotli
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ddm_r7_token_coder import decode_token_codes
from ddm_tr1_runtime import (
    _encode_tokens,
    build_packet,
    parse_packet,
    render_frame1_camera_uint8,
)
from pfs1_warp_receiver import (
    ST_GRID,
    _target_grid,
    _to_uint8,
    intrinsics_native,
    pose_to_homography,
    warp_rgb,
)

MAGIC = b"PFS1WPB1"
CAMERA_H, CAMERA_W = 874, 1164


def parse_pose_warp(payload: bytes):
    if payload[:8] != MAGIC:
        raise SystemExit("pose_warp magic differs")
    off = 8
    (n_pairs,) = struct.unpack_from("<I", payload, off); off += 4
    (l1,) = struct.unpack_from("<I", payload, off); off += 4
    tp_coded = payload[off:off + l1]; off += l1
    (l2,) = struct.unpack_from("<I", payload, off); off += 4
    st_coded = payload[off:off + l2]; off += l2
    tp = np.frombuffer(brotli.decompress(tp_coded),
                       dtype=np.float16).astype(np.float64).reshape(n_pairs, 6)
    st_idx = np.asarray(decode_token_codes(st_coded), dtype=np.int64).reshape(-1)[:n_pairs]
    return n_pairs, tp, st_idx


def main() -> None:
    archive_dir, output_dir, names_file = map(Path, sys.argv[1:4])
    names = [row.strip() for row in
             names_file.read_text().splitlines() if row.strip()]
    if names != ["0.mkv"]:
        raise SystemExit("this receiver serves exactly the custodied 0.mkv")
    manifest = json.loads((archive_dir / "manifest.json").read_text())

    codes = decode_token_codes((archive_dir / "state/tokens.dr7t").read_bytes())
    token_payload = _encode_tokens(np.ascontiguousarray(codes, dtype=np.uint8))
    packet_bytes = build_packet(manifest["tr1_metadata"], {
        "tokens": token_payload,
        "lotto_renderer": (archive_dir / "state/renderer.sec").read_bytes(),
        "selector": (archive_dir / "state/selector.sec").read_bytes(),
        "pose_stub": (archive_dir / "state/pose_stub.sec").read_bytes(),
    })
    packet = parse_packet(packet_bytes)

    if manifest["frame0_policy"] != "warp_base_solved_sr1":
        raise SystemExit("v4a receiver requires frame0_policy=warp_base_solved_sr1")
    n_pairs, p_solved, st_idx = parse_pose_warp(
        (archive_dir / "state/pose_warp.stp").read_bytes())
    if int(packet.selector["num_pairs"]) != n_pairs:
        raise SystemExit("pose_warp n_pairs differs from selector")
    st_vals = np.asarray(ST_GRID, np.float64)
    K = intrinsics_native()
    Kinv = np.linalg.inv(K)
    grid = _target_grid(CAMERA_H, CAMERA_W)

    name = PurePosixPath(names[0])
    if name.is_absolute() or ".." in name.parts:
        raise SystemExit("unsafe video name")
    target = output_dir / name.with_suffix(".raw")
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as sink:
        for i in range(n_pairs):
            f1 = render_frame1_camera_uint8(packet, i)
            H = pose_to_homography(p_solved[i], K, Kinv,
                                   float(st_vals[st_idx[i]]), 1.0, 0.0)
            f0 = _to_uint8(warp_rgb(f1.astype(np.float64), H, grid))
            sink.write(f0.tobytes())
            sink.write(f1.tobytes())


if __name__ == "__main__":
    main()
'''


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _stored_zip(members: dict[str, bytes]) -> bytes:
    """Deterministic ZIP_STORED archive (fixed date_time, insertion order) — the
    v3_warp/v4a grammar is a plain stored ZIP the inflate_runner reads by member
    name; no tac dependency (avoids the codex-worktree tac hijack + the TR1
    2-member constraint of rt._deterministic_stored_zip)."""
    import io
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


def _load_poses(solve_jl: Path, key: str) -> tuple[np.ndarray, int]:
    rows: dict[int, dict] = {}
    for ln in solve_jl.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            rows[int(r["pair"])] = r
    n = 600
    have = [i for i in range(n) if i in rows]
    poses = np.zeros((n, 6), np.float64)
    for i in range(n):
        if i in rows:
            poses[i] = np.asarray(rows[i][key], np.float64)
        else:
            poses[i] = np.nan  # flagged; caller decides fallback
    return poses, len(have)


def build(args: argparse.Namespace) -> None:
    members = _read_members(KNEEA)
    # parse the existing pose member to reuse the st stream (s_t unchanged)
    pw = members["state/pose_warp.stp"]
    off = 8
    (n_pairs,) = struct.unpack_from("<I", pw, off); off += 4
    (l1,) = struct.unpack_from("<I", pw, off); off += 4
    off += l1
    (l2,) = struct.unpack_from("<I", pw, off); off += 4
    st_coded = pw[off:off + l2]   # reused verbatim (s_t unchanged by the re-solve)

    poses, n_have = _load_poses(args.solve_jsonl, args.pose_key)
    n_missing = int(np.isnan(poses[:, 0]).sum())
    if n_missing and not args.allow_partial:
        raise SystemExit(f"{n_missing} pairs unsolved in {args.solve_jsonl}; "
                         "re-run --mode solve to completion or pass --allow-partial "
                         "(unsolved pairs fall back to the shipped t_p = stale)")
    if n_missing:
        # fallback: keep the STALE shipped t_p for unsolved pairs (honest: those
        # pairs are un-improved, an upper bound on the composed d_pose)
        tp_stale = np.frombuffer(brotli.decompress(pw[16:16 + l1]),
                                 np.float16).astype(np.float64).reshape(n_pairs, 6)
        miss = np.isnan(poses[:, 0])
        poses[miss] = tp_stale[miss]
        print(f"[build] {n_missing} unsolved -> stale t_p fallback (upper bound)")

    # re-encode pose_warp.stp with the re-solved poses (f16) + unchanged st stream
    poses_f16 = poses.astype(np.float16)
    tp_coded = brotli.compress(np.ascontiguousarray(poses_f16).tobytes(), quality=11)
    new_pw = (MAGIC + struct.pack("<I", n_pairs)
              + struct.pack("<I", len(tp_coded)) + tp_coded
              + struct.pack("<I", len(st_coded)) + st_coded)

    manifest = json.loads(members["manifest.json"])
    manifest["frame0_policy"] = "warp_base_solved_sr1"
    manifest["pose_carrier"] = ("solved_6dof_warp_of_f1_sr1_on_kneeA_base "
                                f"(ck1 {args.pose_key}); grammar v4a")
    manifest["pose_warp_sha256"] = _sha(new_pw)
    manifest["schema"] = "ddm_ck1_composed_archive.v4a_solved_warp"
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

    tag = args.tag
    out_zip = OUT / f"ck1_composed_{tag}_archive.zip"
    out_zip.write_bytes(archive_bytes)
    (OUT / "inflate_runner_v4a.py").write_text(V4A_INFLATE_RUNNER)
    receipt = {
        "schema": "ddm_ck1_composed_build.v1",
        "axis": "[macOS-CPU advisory] NON-PROMOTABLE; pointer 0.1910828242 UNMOVED",
        "score_claim": False,
        "tag": tag, "pose_key": args.pose_key, "grammar": "v4a_single_plane_sr1",
        "solve_jsonl": str(args.solve_jsonl), "n_solved": n_have,
        "n_stale_fallback": n_missing,
        "archive_zip": str(out_zip), "archive_bytes": len(archive_bytes),
        "archive_sha256": _sha(archive_bytes),
        "kneeA_bytes": len(KNEEA.read_bytes()),
        "rate_term": 25.0 * len(archive_bytes) / 37_545_489,
        "pose_warp_sha256": manifest["pose_warp_sha256"],
        "note": "token/renderer/selector/pose_stub members are Knee-A bytes "
                "verbatim; only pose_warp.stp changed. v4a inflate_runner (s_r=1.0) "
                "written alongside. Decode/gate via stage_ck1_composed_gate.sh.",
    }
    (OUT / f"ck1_composed_{tag}_build_receipt.json").write_text(
        json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solve-jsonl", type=Path, default=CK1_SOLVE)
    ap.add_argument("--pose-key", default="p_single_kneeA",
                    help="p_single_kneeA (v4a) | p_star (full-base stand-in for "
                         "build-path validation) | p_best_kneeA (needs v4b, N/A here)")
    ap.add_argument("--tag", default="single_kneeA")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
