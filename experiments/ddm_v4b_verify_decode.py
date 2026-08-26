#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_v4b verify — byte-close + parse-back + decode identity (task #776, step 4).

Runs entirely in the VENDORED decode substrate (no tac, exactly the gate's
runtime).  Sets up a decode_root with the vendored coders/receiver + the v4b
receiver + the unzipped v4b archive (and a v4a single-plane archive for the
selector-0 byte-identity cross-check), then checks:

  (A) PARSE-BACK CONSUMPTION BIJECTION (#417): the v4b receiver parses
      state/pose_warp.stp, consuming every byte (offset == len), and decodes
      p_best (600,6) + s_t (600) + selector (600).  Shapes + full consumption.
  (C) SELECTOR-0 BYTE-IDENTITY vs the v4a receiver: for selector-0 pairs the v4b
      f0 must be byte-identical to an INDEPENDENT single-plane decode of the v4a
      archive (proves the sel-0 branch == v4a, and that the shipped p_best bytes
      for sel-0 pairs round-trip to p_single_kneeA).
  (D) TWO-PLANE INDEPENDENT RECOMPUTE: for selector-1 pairs recompute the static
      two-plane compose with FRESH code (vendored primitives) and require
      byte-identity vs the Decoder -> catches any Decoder wiring bug; also assert
      the two-plane f0 actually DIFFERS from the single-plane f0 (the compose is
      doing work).

f1 renders come from Knee-A tokens (identical in both archives).  Axis:
[macOS-CPU advisory]; the n600 evaluate gate (MAIN fires) is the authority.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import struct
import sys
import zipfile
from pathlib import Path

import brotli
import numpy as np

TEMPLATE_SUB = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/submissions/pfs1")
V4B_ARCHIVE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_v4b_20260730/v4b_composed_static_kneeA_archive.zip")
V4A_ARCHIVE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_ck1_20260729/ck1_composed_v4b_xcheck_single_archive.zip")
RECEIVER_SRC = Path("experiments/inflate_runner_v4b.py")
DECODE_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_v4b_20260730/decode_root")
VENDORED = ("ddm_r7_token_coder.py", "ddm_tr1_runtime.py", "pfs1_warp_receiver.py",
            "repair_entropy_coder_runtime_adapters.py")
CAMERA_H, CAMERA_W = 874, 1164


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _unzip(archive: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    from tac.submission_archive import safe_extract_zip

    safe_extract_zip(archive, dest)


def _setup_decode_root() -> tuple[Path, Path]:
    DECODE_ROOT.mkdir(parents=True, exist_ok=True)
    for f in VENDORED:
        shutil.copy(TEMPLATE_SUB / f, DECODE_ROOT / f)
    shutil.copy(RECEIVER_SRC, DECODE_ROOT / "inflate_runner_v4b.py")
    av4b = DECODE_ROOT / "archive_v4b"
    av4a = DECODE_ROOT / "archive_v4a"
    _unzip(V4B_ARCHIVE, av4b)
    _unzip(V4A_ARCHIVE, av4a)
    return av4b, av4a


def _import_receiver():
    sys.path.insert(0, str(DECODE_ROOT))
    spec = importlib.util.spec_from_file_location(
        "inflate_runner_v4b", DECODE_ROOT / "inflate_runner_v4b.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _parse_v4a_pose(archive_dir: Path):
    """Independent single-plane parse (v4a MAGIC PFS1WPB1, no selector)."""
    payload = (archive_dir / "state/pose_warp.stp").read_bytes()
    if payload[:8] != b"PFS1WPB1":
        raise SystemExit("v4a magic differs")
    off = 8
    (n_pairs,) = struct.unpack_from("<I", payload, off)
    off += 4
    (l1,) = struct.unpack_from("<I", payload, off)
    off += 4
    tp_coded = payload[off:off + l1]
    off += l1
    (l2,) = struct.unpack_from("<I", payload, off)
    off += 4
    st_coded = payload[off:off + l2]
    tp = np.frombuffer(brotli.decompress(tp_coded),
                       dtype=np.float16).astype(np.float64).reshape(n_pairs, 6)
    from ddm_r7_token_coder import decode_token_codes
    st_idx = np.asarray(decode_token_codes(st_coded),
                        dtype=np.int64).reshape(-1)[:n_pairs]
    return n_pairs, tp, st_idx


def main() -> int:
    av4b, av4a = _setup_decode_root()
    recv_mod = _import_receiver()
    from pfs1_warp_receiver import (
        ST_GRID,
        _target_grid,
        _to_uint8,
        intrinsics_native,
        pose_to_homography,
        warp_rgb,
    )

    dec = recv_mod.Decoder(av4b)  # (A) constructs -> parse-back bijection ran
    checks: list[dict] = []
    ok = True

    # (A) bijection assertions
    a_ok = (dec.n_pairs == 600 and dec.p_best.shape == (600, 6)
            and dec.st_idx.shape == (600,) and dec.sel.shape == (600,))
    n_two = int(dec.sel.sum())
    checks.append({"check": "A_parse_back_bijection", "ok": bool(a_ok),
                   "n_pairs": int(dec.n_pairs), "p_best_shape": list(dec.p_best.shape),
                   "sel_shape": list(dec.sel.shape), "n_selector_two": n_two,
                   "note": "Decoder construction consumed every pose_warp byte "
                           "(off==len enforced in parse_pose_warp_v4b) + shapes."})
    ok = ok and a_ok

    # v4a independent parse (for the sel-0 cross-check)
    n4a, tp4a, st4a = _parse_v4a_pose(av4a)
    K = intrinsics_native()
    Kinv = np.linalg.inv(K)
    grid = _target_grid(CAMERA_H, CAMERA_W)
    v_row = recv_mod.geometric_horizon_row(K)
    far = (np.arange(CAMERA_H)[:, None] < v_row) & np.ones((1, CAMERA_W), bool)
    st_vals = np.asarray(ST_GRID, np.float64)

    # sample pairs: 3 selector-0 + 3 selector-1
    sel0 = [i for i in range(600) if dec.sel[i] == 0][:3]
    sel1 = [i for i in range(600) if dec.sel[i] == 1][:3]
    sample = sel0 + sel1

    for i in sample:
        f1_v4b = dec.f1(i)
        f0_v4b = dec.f0(i, f1_v4b)
        f1_f = f1_v4b.astype(np.float64)
        s_t = float(st_vals[dec.st_idx[i]])
        pose = dec.p_best[i]

        # INDEPENDENT recompute of the single-plane f0 (fresh code)
        hg = pose_to_homography(pose, K, Kinv, s_t, 1.0, 0.0)
        f0_single = _to_uint8(warp_rgb(f1_f, hg, grid))
        # INDEPENDENT recompute of the two-plane static f0
        hf = pose_to_homography(pose, K, Kinv, 0.0, 1.0, 0.0)
        f0_two = _to_uint8(np.where(far[..., None],
                                    warp_rgb(f1_f, hf, grid),
                                    warp_rgb(f1_f, hg, grid)))
        f0_ref = f0_single if dec.sel[i] == 0 else f0_two
        d_ref = int(np.abs(f0_v4b.astype(np.int64) - f0_ref.astype(np.int64)).max())

        row = {"pair": int(i), "selector": int(dec.sel[i]),
               "f0_vs_independent_recompute_maxabs": d_ref}

        if dec.sel[i] == 0:
            # (C) v4a cross-check: independent single-plane decode of the v4a
            # archive (its OWN pose bytes) must byte-match the v4b sel-0 f0.
            s_t_a = float(st_vals[st4a[i]])
            hga = pose_to_homography(tp4a[i], K, Kinv, s_t_a, 1.0, 0.0)
            f0_v4a = _to_uint8(warp_rgb(f1_f, hga, grid))
            d_v4a = int(np.abs(f0_v4b.astype(np.int64) - f0_v4a.astype(np.int64)).max())
            row["f0_vs_v4a_single_maxabs"] = d_v4a
            row["pose_f16_matches_v4a"] = bool(np.array_equal(
                pose.astype(np.float16), tp4a[i].astype(np.float16)))
            row["ok"] = bool(d_ref == 0 and d_v4a == 0)
        else:
            # (D) two-plane must DIFFER from single (compose is doing work)
            d_two_vs_single = int(np.abs(
                f0_two.astype(np.int64) - f0_single.astype(np.int64)).max())
            row["f0_two_vs_single_maxabs"] = d_two_vs_single
            row["ok"] = bool(d_ref == 0 and d_two_vs_single > 0)
        ok = ok and row["ok"]
        checks.append(row)

    kb = os.path.getsize(V4B_ARCHIVE)
    archive_bytes = V4B_ARCHIVE.read_bytes()
    receipt = {
        "schema": "ddm_v4b_verify_decode.v1",
        "axis": "[macOS-CPU advisory] NON-PROMOTABLE; vendored gate substrate; "
                "n600 evaluate gate is authority. pointer 0.1910828242 UNMOVED",
        "score_claim": False,
        "all_checks_ok": bool(ok),
        "archive_zip": str(V4B_ARCHIVE), "archive_bytes": kb,
        "archive_sha256": _sha(archive_bytes),
        "rate_term": 25.0 * kb / 37_545_489,
        "n_selector_two": n_two,
        "horizon_row_derived": int(v_row),
        "sample_sel0": sel0, "sample_sel1": sel1,
        "checks": checks,
    }
    (DECODE_ROOT.parent / "v4b_verify_receipt.json").write_text(
        json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
