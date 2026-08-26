#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_v4d — parse-back bijection (#417) + decode-identity + deterministic
rebuild verification for the v4d composed archive.  Runs in the VENDORED gate
substrate (no tac): stages a temp submission dir with the pfs1 template deps +
the v4d receiver, extracts archive.zip, and drives the actual Decoder.

Checks (all must pass):
  (A) parse-back consumption bijection: the receiver consumes EVERY pose_warp
      byte (parse asserts off==len) and decodes p_best (n,6) + s_t (n) +
      selector (n) + (a,b) (n,2) + beta_idx (n).  No counted-but-inert bytes.
  (B) field bit-exactness: decoded pose (with dim0-offset reconstruction) /
      (a,b) / selector / beta_idx EXACTLY match what the final JSONL + the
      build's f16/offset encoding produce (zero drift).
  (C) independent compose recompute on sampled pairs (incl one beta!=0 pair to
      exercise the rolling-shutter path), byte-exact vs a fresh recompute; and
      a selector-1 f0 genuinely differs from the single-plane compose.

RECORDED (identity, NOT a check -- ddm_bs3 #909):
  (D) ``D_archive_sha256`` / ``D_archive_bytes`` are the identity of the archive
      UNDER TEST.  Nothing is rebuilt here and there is no second sha to compare
      against, so D has NO discriminating power and is deliberately excluded
      from ``all_checks_ok``.  It was previously listed above under "all must
      pass" while being absent from the conjunction -- a named check whose
      projection was the EMPTY SET (it could not fail).  Deterministic rebuild
      IS verified, elsewhere and for real, by re-running the builder and
      diffing shas (the ``*_rebuildcheck_archive.zip`` artifacts).

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
RECEIVER = Path("experiments/inflate_runner_v4d.py")
# The receiver reads its beta table from the manifest (inflate_runner_v4d.py
# :127), so the verifier MUST read it from the decoder too -- a module-level
# copy silently drifts the moment the solve extends the menu.  Kept only as
# the legacy default for archives whose manifest predates the key.
BETA_MAGS = (0.0, 0.5, 1.0)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def conjoin_checks(checks: dict) -> tuple[bool, list[str]]:
    """ddm_bs3 (#909) CLASS FIX: derive the verdict from EVERY ``*_ok`` key.

    The verdict was a HAND-WRITTEN conjunction ``A_ok and B_ok and C_ok``. That
    is what let the module docstring advertise a "(D) deterministic rebuild"
    check under "Checks (all must pass)" while D was absent from the
    conjunction and nothing was ever rebuilt -- a named check whose projection
    was the EMPTY SET, structurally unable to fail. Any future ``E_ok`` would
    have drifted the same way. Deriving the conjunction from the keys makes the
    denominator follow the checks automatically.

    Returns ``(all_ok, ok_keys)``. VACUITY: an empty conjunction RAISES rather
    than returning True -- ``all([])`` is True, which is exactly the empty-scope
    -reads-as-PASS failure this repo has already been bitten by."""
    ok_keys = sorted(k for k in checks if k.endswith("_ok"))
    if not ok_keys:
        raise SystemExit(
            "ddm_v4d_verify: zero *_ok checks present -- VACUOUS, never a pass "
            "(all([]) is True; refusing to report that as all_checks_ok)")
    return all(bool(checks[k]) for k in ok_keys), ok_keys


def _load_final(path: Path) -> dict[int, dict]:
    rows = {}
    for ln in path.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            rows[int(r["pair"])] = r
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--final-jsonl", required=True)
    ap.add_argument("--sample", type=int, default=6)
    args = ap.parse_args()

    archive = Path(args.archive)
    final = _load_final(Path(args.final_jsonl))
    checks: dict[str, object] = {}

    with tempfile.TemporaryDirectory(dir="/Volumes/VertigoDataTier/pact/ddm_v4d_20260731") as td:
        stage = Path(td)
        for dep in DEPS:
            shutil.copy(TEMPLATE / dep, stage / dep)
        shutil.copy(RECEIVER, stage / "inflate_runner_v4d.py")
        arch_dir = stage / "archive"
        arch_dir.mkdir()
        from tac.submission_archive import safe_extract_zip

        safe_extract_zip(archive, arch_dir)
        sys.path.insert(0, str(stage))
        import inflate_runner_v4d as rc

        dec = rc.Decoder(arch_dir)
        n = dec.n_pairs
        checks["A_bijection_n_pairs"] = int(n)
        checks["A_poses_shape"] = list(dec.p_best.shape)
        checks["A_ab_shape"] = list(dec.ab.shape)
        checks["A_beta_shape"] = list(dec.beta_idx.shape)
        checks["A_selector_sum"] = int(dec.sel.sum())
        checks["A_dim0_offset"] = dec.dim0_offset
        checks["A_ok"] = (n == len(final)
                          and list(dec.p_best.shape) == [n, 6]
                          and list(dec.ab.shape) == [n, 2]
                          and list(dec.beta_idx.shape) == [n])

        # (B) field bit-exactness.  Reconstruct the EXPECTED decoded pose exactly
        # as the encoder(store)->decoder(reconstruct) chain would.
        fp = np.asarray([final[i]["p"] for i in range(n)], np.float64)
        if dec.dim0_offset is not None:
            off = float(dec.dim0_offset)
            exp_pose = np.zeros((n, 6), np.float64)
            exp_pose[:, 0] = off + np.float16(fp[:, 0] - off).astype(np.float64)
            exp_pose[:, 1:] = np.float16(fp[:, 1:]).astype(np.float64)
        else:
            exp_pose = np.float16(fp).astype(np.float64)
        # compare in the reconstructed f64 (offset add is exact for dim0)
        b_pose = bool(np.array_equal(dec.p_best.astype(np.float64), exp_pose))
        ab_ref = np.asarray([[final[i]["a"], final[i]["b"]] for i in range(n)],
                            np.float16)
        sel_ref = np.asarray([final[i]["selector"] for i in range(n)], np.int64)
        # Compare MAGNITUDES, not indices: the table is derived from the
        # magnitudes the solve chose and sorted, so an index is only meaningful
        # against its own table.  A row written before the extended menu
        # carries beta_idx into the seed table instead.
        def _row_mag(row):
            if "beta_mag" in row:
                return float(row["beta_mag"])
            idx = int(row["beta_idx"])
            if not 0 <= idx < len(BETA_MAGS):
                raise SystemExit(f"row beta_idx={idx} with no beta_mag")
            return float(BETA_MAGS[idx])

        beta_ref = np.asarray([_row_mag(final[i]) for i in range(n)],
                              np.float64)
        beta_dec = np.asarray([float(dec.beta_mags[int(dec.beta_idx[i])])
                               for i in range(n)], np.float64)
        b_ab = bool(np.array_equal(dec.ab.astype(np.float16).view(np.uint16),
                                   ab_ref.view(np.uint16)))
        b_sel = bool(np.array_equal(dec.sel, sel_ref))
        b_beta = bool(np.array_equal(beta_dec, beta_ref))
        checks["B_pose_reconstruct_exact"] = b_pose
        checks["B_ab_bit_exact"] = b_ab
        checks["B_selector_exact"] = b_sel
        checks["B_beta_exact"] = b_beta
        checks["B_ok"] = b_pose and b_ab and b_sel and b_beta

        # (C) independent compose recompute; force a beta!=0 pair in the sample
        rng = np.random.default_rng(20260731)
        sel1 = [i for i in range(n) if dec.sel[i] == 1]
        sel0 = [i for i in range(n) if dec.sel[i] == 0]
        # index 0 is the SMALLEST magnitude in the derived table, which is
        # not necessarily 0.0 -- test the magnitude, never the index.
        betanz = [i for i in range(n)
                  if float(dec.beta_mags[int(dec.beta_idx[i])]) != 0.0]
        sample = []
        if betanz:
            sample.append(int(rng.choice(betanz)))
        sample += [int(x) for x in rng.choice(sel1, min(args.sample // 2, len(sel1)),
                                              replace=False)]
        sample += [int(x) for x in rng.choice(sel0, min(args.sample // 2, len(sel0)),
                                              replace=False)]
        sample = sorted(set(sample))
        from pfs1_warp_receiver import _to_uint8, pose_to_homography, warp_rgb
        c_ok, c_two_diff, c_beta_seen = True, True, False
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
            beta_mag = float(dec.beta_mags[int(dec.beta_idx[i])])
            if beta_mag != 0.0:
                c_beta_seen = True
                beta = beta_mag * (1.0 if pose[5] >= 0.0 else -1.0)
                f0f = ((1.0 - alpha) * _compose_at(f1_f, pose, s_t, sel,
                                                   1.0 - beta / 2.0)
                       + alpha * _compose_at(f1_f, pose, s_t, sel,
                                             1.0 + beta / 2.0))
            else:
                f0f = _compose_at(f1_f, pose, s_t, sel, 1.0)
            if sel == 1 and beta_mag == 0.0:
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
        checks["C_beta_path_exercised"] = c_beta_seen
        checks["C_ok"] = c_ok and c_two_diff and c_beta_seen
        sys.path.remove(str(stage))

    checks["D_archive_sha256"] = _sha(archive.read_bytes())
    checks["D_archive_bytes"] = archive.stat().st_size

    all_ok, ok_keys = conjoin_checks(checks)
    receipt = {
        "schema": "ddm_v4d_verify.v1",
        "axis": "[macOS-CPU advisory] NON-PROMOTABLE; pointer 0.1910828242 UNMOVED",
        "score_claim": False, "archive": str(archive), "all_checks_ok": all_ok,
        "checks_in_conjunction": ok_keys, "checks_examined_n": len(ok_keys),
        **checks,
        "note": "vendored-substrate decode (no tac); the n600 evaluate gate is "
                "the d_pose/d_seg authority.",
    }
    out = Path("/Volumes/VertigoDataTier/pact/ddm_v4d_20260731/v4d_verify_receipt.json")
    out.write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
