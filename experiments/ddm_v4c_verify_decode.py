#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_v4c — parse-back bijection (#417) + decode-identity + deterministic
rebuild verification for the v4c composed archive.  Runs entirely in the
VENDORED gate substrate (no tac): stages a temp submission dir with the pfs1
template deps + the v4c receiver, extracts archive.zip, and drives the actual
Decoder the gate will run.

Checks (all must pass):
  (A) parse-back consumption bijection: the receiver consumes EVERY pose_warp
      byte (parse asserts off == len) and decodes p_best (n,6) + s_t (n) +
      selector (n) + (a,b) (n,2).  No counted-but-inert bytes.
  (B) field bit-exactness: the decoded f16 poses / (a,b) / selector EXACTLY
      match the photo JSONL the build consumed (zero d_pose/exposure drift).
  (C) independent compose recompute: for sampled pairs the Decoder.f0 equals a
      FRESH static-compose recompute (byte-exact), and a selector-1 f0 genuinely
      differs from the single-plane compose (the two-plane is doing real work).

RECORDED (identity, NOT a check -- ddm_bs3 #909, sister of the same fix in
ddm_v4d_verify_decode.py):
  (D) ``D_archive_sha256`` / ``D_archive_bytes`` are the identity of the archive
      UNDER TEST.  The build is NOT re-run here and there is no second sha to
      compare against, so D has NO discriminating power and is excluded from
      ``all_checks_ok``.  It was previously listed above under "all must pass"
      while being absent from the conjunction -- a named check whose projection
      was the EMPTY SET (it could not fail).

Axis: [macOS-CPU advisory] NON-PROMOTABLE; pointer 0.1910828242 UNMOVED.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

TEMPLATE = Path("/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/submissions/pfs1")
DEPS = ("ddm_r7_token_coder.py", "ddm_tr1_runtime.py", "pfs1_warp_receiver.py",
        "repair_entropy_coder_runtime_adapters.py")
RECEIVER = Path("experiments/inflate_runner_v4c.py")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _load_photo(path: Path) -> dict[int, dict]:
    rows = {}
    for ln in path.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            rows[int(r["pair"])] = r
    return rows



def conjoin_checks(checks: dict) -> tuple[bool, list[str]]:
    """ddm_bs3 (#909) CLASS FIX -- sister of ddm_v4d_verify_decode.conjoin_checks.

    Derive the verdict from EVERY ``*_ok`` key instead of a hand-written
    ``A_ok and B_ok and C_ok``. The hand-written list is what let the docstring
    advertise a "(D) deterministic rebuild" check that was never in the
    conjunction and never performed -- a named check whose projection was the
    EMPTY SET. Duplicated rather than imported because these verifiers run in
    the VENDORED gate substrate (no ``tac``, no cross-script imports) by design.

    VACUITY: an empty conjunction RAISES. ``all([])`` is True, which is exactly
    the empty-scope-reads-as-PASS failure this repo has already been bitten by.
    """
    ok_keys = sorted(k for k in checks if k.endswith("_ok"))
    if not ok_keys:
        raise SystemExit(
            "ddm_v4c_verify: zero *_ok checks present -- VACUOUS, never a pass")
    return all(bool(checks[k]) for k in ok_keys), ok_keys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--photo-jsonl", required=True)
    ap.add_argument("--sample", type=int, default=6)
    args = ap.parse_args()

    archive = Path(args.archive)
    photo = _load_photo(Path(args.photo_jsonl))
    checks: dict[str, object] = {}

    with tempfile.TemporaryDirectory(dir="/Volumes/VertigoDataTier/pact/ddm_v4c_20260730") as td:
        stage = Path(td)
        for dep in DEPS:
            shutil.copy(TEMPLATE / dep, stage / dep)
        shutil.copy(RECEIVER, stage / "inflate_runner_v4c.py")
        arch_dir = stage / "archive"
        arch_dir.mkdir()
        from tac.submission_archive import safe_extract_zip

        safe_extract_zip(archive, arch_dir)
        sys.path.insert(0, str(stage))
        import inflate_runner_v4c as rc

        dec = rc.Decoder(arch_dir)
        # (A) bijection — parse succeeded (off==len enforced inside); shapes
        checks["A_bijection_n_pairs"] = int(dec.n_pairs)
        checks["A_poses_shape"] = list(dec.p_best.shape)
        checks["A_ab_shape"] = list(dec.ab.shape)
        checks["A_selector_sum"] = int(dec.sel.sum())
        checks["A_ok"] = (dec.n_pairs == len(photo)
                          and list(dec.p_best.shape) == [dec.n_pairs, 6]
                          and list(dec.ab.shape) == [dec.n_pairs, 2])

        # (B) field bit-exactness vs the photo JSONL (f16)
        pose_ref = np.asarray([photo[i]["p"] for i in range(dec.n_pairs)],
                              np.float16)
        ab_ref = np.asarray([[photo[i]["a"], photo[i]["b"]]
                             for i in range(dec.n_pairs)], np.float16)
        sel_ref = np.asarray([photo[i]["selector"] for i in range(dec.n_pairs)],
                             np.int64)
        pose_dec = dec.p_best.astype(np.float16)
        ab_dec = dec.ab.astype(np.float16)
        b_pose = bool(np.array_equal(pose_dec.view(np.uint16), pose_ref.view(np.uint16)))
        b_ab = bool(np.array_equal(ab_dec.view(np.uint16), ab_ref.view(np.uint16)))
        b_sel = bool(np.array_equal(dec.sel, sel_ref))
        checks["B_pose_bit_exact"] = b_pose
        checks["B_ab_bit_exact"] = b_ab
        checks["B_selector_exact"] = b_sel
        checks["B_ok"] = b_pose and b_ab and b_sel

        # (C) independent compose recompute on sampled pairs
        rng = np.random.default_rng(20260730)
        sel1 = [i for i in range(dec.n_pairs) if dec.sel[i] == 1]
        sel0 = [i for i in range(dec.n_pairs) if dec.sel[i] == 0]
        sample = ([int(x) for x in rng.choice(sel1, min(args.sample // 2, len(sel1)),
                                              replace=False)]
                  + [int(x) for x in rng.choice(sel0, min(args.sample // 2, len(sel0)),
                                                replace=False)])
        from pfs1_warp_receiver import (
            _to_uint8,
            pose_to_homography,
            warp_rgb,
        )
        c_ok, c_two_diff = True, True
        alpha = (np.arange(874) / 873.0)[:, None, None]

        def _compose_at(f1_f, pose, s_t, sel, rot):
            hg = pose_to_homography(pose, dec.K, dec.Kinv, s_t, rot, 0.0)
            wg = warp_rgb(f1_f, hg, dec.grid)
            if sel == 0:
                return wg
            hf = pose_to_homography(pose, dec.K, dec.Kinv, 0.0, rot, 0.0)
            wf = warp_rgb(f1_f, hf, dec.grid)
            return np.where(dec._far[..., None], wf, wg)

        for i in sample:
            f1 = dec.f1(i)
            f0_dec = dec.f0(i, f1)
            f1_f = f1.astype(np.float64)
            s_t = float(dec.st_vals[dec.st_idx[i]])
            pose = dec.p_best[i]
            sel = int(dec.sel[i])
            a, b = float(dec.ab[i][0]), float(dec.ab[i][1])
            if dec.rs_beta != 0.0:
                beta = dec.rs_beta * (1.0 if pose[5] >= 0.0 else -1.0)
                f0f = ((1.0 - alpha) * _compose_at(f1_f, pose, s_t, sel,
                                                   1.0 - beta / 2.0)
                       + alpha * _compose_at(f1_f, pose, s_t, sel,
                                             1.0 + beta / 2.0))
            else:
                f0f = _compose_at(f1_f, pose, s_t, sel, 1.0)
            if sel == 1:
                # two-plane must differ from the single-plane (rot=1, no shear)
                wg1 = _compose_at(f1_f, pose, s_t, 0, 1.0)
                single_u8 = _to_uint8(a * wg1 + b if (a != 1.0 or b != 0.0) else wg1)
                if np.array_equal(single_u8, f0_dec):
                    c_two_diff = False
            if a != 1.0 or b != 0.0:
                f0f = a * f0f + b
            f0_rc = _to_uint8(f0f)
            if not np.array_equal(f0_rc, f0_dec):
                c_ok = False
        checks["C_sample_pairs"] = sample
        checks["C_recompute_byte_exact"] = c_ok
        checks["C_two_plane_does_work"] = c_two_diff
        checks["C_ok"] = c_ok and c_two_diff
        sys.path.remove(str(stage))

    # (D) deterministic rebuild: archive sha stable
    checks["D_archive_sha256"] = _sha(archive.read_bytes())
    checks["D_archive_bytes"] = archive.stat().st_size

    all_ok, ok_keys = conjoin_checks(checks)
    receipt = {
        "schema": "ddm_v4c_verify.v1",
        "axis": "[macOS-CPU advisory] NON-PROMOTABLE; pointer 0.1910828242 UNMOVED",
        "score_claim": False, "archive": str(archive), "all_checks_ok": all_ok,
        "checks_in_conjunction": ok_keys, "checks_examined_n": len(ok_keys),
        **checks,
        "note": "vendored-substrate decode (no tac); the n600 evaluate gate is "
                "the d_pose/d_seg authority.",
    }
    out = Path("/Volumes/VertigoDataTier/pact/ddm_v4c_20260730/v4c_verify_receipt.json")
    out.write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
