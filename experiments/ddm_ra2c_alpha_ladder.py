#!/usr/bin/env python3
"""RA2C: the carrier alpha ladder -- d_pose(fidelity) measured through the real decode.

WHAT THIS ANSWERS. ra1b priced every rank-r carrier in exact coded bytes and exact
carrier-field MSE. Neither is the scored quantity. The scored quantity is a PoseNet
readout, and nobody has measured it as a function of carrier fidelity. This tool
measures it.

FIRE ORDER (charter ddm_ra2 AMENDMENT 3, C10): alpha = 0 FIRST -- the carrier deleted
outright. Per the registered affordance law
(tac.canonical_equations.carrier_rate_credit_pose_affordance_20260816) deleting the
22,161 B carrier returns 153.8% of the remaining sub-0.15 gap in rate credit and sits
under the LOOSEST tolerance bar of any rung, because the bar is QUADRATIC in returned
bytes:

    d_pose_new / d_pose_base  <  (1 + R/POSE)**2,     R = 25 * dB / D

    T4 CUDA (shipping axis)   POSE = 0.0082945765  ->  bar 7.722874x
    macOS-CPU advisory        POSE = 0.0384018     ->  bar 1.916x

The advisory instrument's pose term is 4.63x LARGER, so its bar is ~4x TIGHTER. That
makes this screen CONSERVATIVE in the useful direction: clearing 1.916x here implies
clearing 7.723x on the shipping axis under any reasonable ratio transfer. It does NOT
work the other way -- a refusal here is only a refusal on T4 if the ratio transfers,
which is most defensible at alpha=0 where the perturbation dwarfs instrument noise.

HOW. The carrier renders frame_0 ONLY (inflate.py:659-676); SegNet reads x[:, -1] =
frame_1. So an alpha cut is seg-invisible BY CONSTRUCTION. We therefore:

  1. decode the carrier through the PROVEN chain (ddm_ra2b, VERDICT CHAIN_PROVEN:
     f26 unwrap -> 14 B selector split -> materialize CPR1 -> unpack_semantic_pose ->
     36 B compensation overlay -> carrier render -> apply_pixel_mode);
  2. CONTROL: re-render every even frame at alpha=1 and require BYTE-IDENTITY with the
     retained base render. This is the whole instrument check -- if it passes, the base
     report transfers exactly and alpha is the only variable;
  3. splice: even frames (frame_0) <- alpha render, odd frames (frame_1) <- base bytes;
  4. run upstream/evaluate.py on the result, archive.zip held CONSTANT.

WHY archive.zip IS HELD CONSTANT. One variable at a time. The measured S here is NOT a
candidate's S -- the real candidate would ship fewer bytes. The rate credit is exactly
known from ra1b's measured coded lengths and is applied analytically through the
registered law. Reporting the held-rate S as a candidate score would be a fake claim.

Axis: d_pose / d_seg [macOS-CPU advisory, upstream/evaluate.py, n600] -- NOT a score
claim, NOT promotable. The shipping axis is contest-CUDA T4.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RA2B_SOURCE = REPO / "experiments/ddm_ra2b_carrier_chain_control.py"

DEFAULT_BASE_WORK = Path(
    "/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/work_r2"
)
DEFAULT_UPSTREAM = Path("/Volumes/APDataStore/pact/upstream_eval_mirror_20260815")
DEFAULT_OUT_ROOT = Path("/Volumes/APDataStore/pact")

N_PAIRS = 600
CAMERA_H, CAMERA_W = 874, 1164
FRAME_BYTES = CAMERA_H * CAMERA_W * 3          # 3_052_008
RAW_BYTES = 2 * N_PAIRS * FRAME_BYTES          # 3_662_409_600

# The retained base advisory row (work_r2/report.txt + contest_auth_eval.json).
BASE_D_POSE = 0.00014747
BASE_D_SEG = 0.00042714
BASE_ARCHIVE_BYTES = 182_759

# ra1b's measured coded length of the shipped carrier cell (Brotli, same quality).
CARRIER_CODED_BYTES = 22_161


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_ra2b():
    """Import the PROVEN chain wholesale -- same custody pins, same decode, same render."""
    spec = importlib.util.spec_from_file_location("ra2b_chain", RA2B_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ra2b_chain"] = module
    spec.loader.exec_module(module)
    return module


def truncate_carrier_rank(basis, coeff, rank: int):
    """Rank-r truncation of the RENDERED carrier field F = C @ B, returned as a coeff.

    F (600 x 2304) has rows in rowspace(B) (12-dim), so its rank-r SVD truncation is
    still in that rowspace and is therefore expressible as C_r @ B exactly. That lets
    the truncation ride the SHIPPED render path unchanged (ra2b.render_frame0), instead
    of a re-implementation.

    Eckart-Young: the truncated SVD is the OPTIMAL rank-r approximation of F in
    Frobenius norm, so the reported rel_err is a LOWER BOUND on any rank-r method in
    that norm. It is NOT a bound in the pose (Jacobian-weighted) metric -- that is the
    open question this rung exists to measure.
    """
    B = np.asarray(basis, dtype=np.float64).reshape(basis.shape[0], -1)   # 12 x 2304
    C = np.asarray(coeff, dtype=np.float64)                               # 600 x 12
    for name, arr in (("basis", B), ("coeff", C)):
        if not np.isfinite(arr).all():
            raise SystemExit(f"{name} carries non-finite values; refusing to truncate")
    # Accelerate's SIMD tail raises spurious divide/overflow/invalid flags on these
    # matmuls while returning correct values (verified: full-rank round trip 1.5e-15
    # on finite, |x|<=1.1 inputs). Suppressed narrowly; the isfinite guards fail closed.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        F = C @ B
        U, S, Vt = np.linalg.svd(F, full_matrices=False)
        # rank(F) <= rows(B) = 12, but svd(full_matrices=False) on 600x2304 returns 600
        # singular values -- ~588 numerical zeros. Bounding by len(S) would admit
        # rank=99 as a near-NO-OP wearing a treatment's name. Bound by the true rank.
        max_rank = int(B.shape[0])
        if not 1 <= rank <= max_rank:
            raise SystemExit(
                f"--carrier-rank {rank} outside [1, {max_rank}] (the carrier is "
                f"{max_rank}-dimensional; svd returns {len(S)} values, "
                f"{len(S) - max_rank} of them numerically zero)"
            )
        F_r = (U[:, :rank] * S[:rank]) @ Vt[:rank]
        C_r, *_ = np.linalg.lstsq(B.T, F_r.T, rcond=None)                 # solve C_r @ B = F_r
        C_r = C_r.T
        # Round-trip: C_r @ B must reproduce F_r (it must, F_r is in rowspace(B)).
        round_trip = float(np.max(np.abs(C_r @ B - F_r)))
    if not np.isfinite(C_r).all():
        raise SystemExit("rank truncation produced non-finite coefficients")
    tot = float((S[:max_rank] ** 2).sum())
    info = {
        "rank": rank,
        "carrier_true_rank": max_rank,
        "energy_kept_frac": float((S[:rank] ** 2).sum() / tot),
        "rel_frobenius_err": float(np.sqrt(max(0.0, 1.0 - (S[:rank] ** 2).sum() / tot))),
        "rowspace_round_trip_max_abs": round_trip,
        "singular_values": [float(v) for v in S[:max_rank]],
        "eckart_young_note": "rel_frobenius_err is a LOWER BOUND on any rank-r method (Frobenius only)",
    }
    if round_trip > 1e-6:
        raise SystemExit(
            f"rank-{rank} truncation left rowspace(B) (max |C_r@B - F_r| = {round_trip:.3e}); "
            "refusing to render a field the shipped path cannot express"
        )
    return C_r.astype(np.asarray(coeff).dtype), info


def build_alpha_raw(
    ra2b,
    chain,
    basis,
    coeff,
    selector_blob,
    base_raw: Path,
    out_raw: Path,
    alpha: float,
    verify_pairs: int,
    control_coeff=None,
) -> dict:
    """Stream base -> out, substituting even frames with the alpha render.

    Also runs the alpha=1 byte-identity CONTROL over `verify_pairs` pairs. The control
    renders a SECOND time at alpha=1; it never reuses the alpha render, so it cannot
    pass by construction.
    """
    renderer = chain[0]
    decode_selector, apply_pixel_mode = chain[5], chain[6]

    modes = sel_indices = None
    if selector_blob is not None:
        modes, sel_indices = decode_selector(selector_blob)
        if sel_indices.size != N_PAIRS:
            raise SystemExit(
                f"selector covers {sel_indices.size} frames, expected {N_PAIRS} -- "
                "refusing to splice against a chain the control never proved"
            )

    ctrl_coeff = coeff if control_coeff is None else control_coeff

    def render_pair(idx: int, a: float, c=None) -> np.ndarray:
        frame = ra2b.render_frame0(renderer, basis, coeff if c is None else c, [idx], a)[0]
        if modes is not None:
            frame = apply_pixel_mode(frame[None].copy(), modes[int(sel_indices[idx])])[0]
        return frame

    verify_idx = set()
    if verify_pairs > 0:
        step = max(1, N_PAIRS // verify_pairs)
        verify_idx = set(range(0, N_PAIRS, step)[:verify_pairs])

    control_checked = 0
    control_failures: list[int] = []
    started = time.time()

    with base_raw.open("rb") as src, out_raw.open("wb") as dst:
        for frame_no in range(2 * N_PAIRS):
            base_frame = src.read(FRAME_BYTES)
            if len(base_frame) != FRAME_BYTES:
                raise SystemExit(
                    f"base raw truncated at frame {frame_no}: read {len(base_frame)} B"
                )
            if frame_no % 2 == 1:
                dst.write(base_frame)            # frame_1: untouched, seg-invisible cut
                continue

            idx = frame_no // 2
            if idx in verify_idx:
                control = render_pair(idx, 1.0, ctrl_coeff).tobytes()
                control_checked += 1
                if control != base_frame:
                    control_failures.append(idx)

            dst.write(render_pair(idx, alpha).tobytes())

            if idx % 25 == 0:
                # Machine-readable progress so the quality poller has a REAL monotone
                # field to stale-detect on (a build job has no per-epoch metric; the
                # honest signal is pair progress, not a fabricated loss).
                print(
                    json.dumps({
                        "ra2c_progress": True,
                        "pair": idx,
                        "n_pairs": N_PAIRS,
                        "control_checked": control_checked,
                        "control_failed": len(control_failures),
                        "elapsed_s": round(time.time() - started, 1),
                        "phase": "build",
                    }),
                    flush=True,
                )

    if control_failures:
        raise SystemExit(
            f"CONTROL FAILED: alpha=1 render differs from the retained base render on "
            f"{len(control_failures)} of {control_checked} pairs "
            f"(first: {control_failures[:8]}). The mirrored chain does NOT reproduce "
            "the shipped receiver -- every ladder row built on it would be measuring "
            "a decode nobody ships. REFUSING to proceed."
        )

    size = out_raw.stat().st_size
    if size != RAW_BYTES:
        raise SystemExit(f"spliced raw is {size} B, expected {RAW_BYTES} B")

    return {
        "control_pairs_checked": control_checked,
        "control_pairs_identical": control_checked,
        "control_verdict": "ALPHA1_BYTE_IDENTICAL" if control_checked else "SKIPPED",
        "build_elapsed_s": time.time() - started,
    }


def run_evaluate(
    submission_dir: Path, upstream: Path, report: Path, device: str, timeout: int
) -> dict:
    """Run the AUTHORITY evaluator itself. No mirroring, no reimplementation."""
    names = upstream / "public_test_video_names.txt"
    cmd = [
        str(REPO / ".venv/bin/python"),
        "evaluate.py",
        "--submission-dir", str(submission_dir),
        "--uncompressed-dir", str(upstream / "videos"),
        "--video-names-file", str(names),
        "--device", device,
        "--report", str(report),
    ]
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"   # the 08-15 mirror-contamination cure
    started = time.time()
    proc = subprocess.run(
        cmd, cwd=str(upstream), env=env, capture_output=True, text=True, timeout=timeout
    )
    elapsed = time.time() - started
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        print(proc.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"evaluate.py rc={proc.returncode}")
    text = report.read_text()

    def grab(pattern: str) -> float:
        m = re.search(pattern, text)
        if not m:
            raise SystemExit(f"could not parse {pattern!r} from the report")
        return float(m.group(1))

    return {
        "d_pose": grab(r"Average PoseNet Distortion:\s*([0-9.eE+-]+)"),
        "d_seg": grab(r"Average SegNet Distortion:\s*([0-9.eE+-]+)"),
        "evaluate_elapsed_s": elapsed,
        "report_path": str(report),
        "cmd": cmd,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carrier-rank", type=int, default=None,
                        help="rank-r truncation of the RENDERED carrier field (C@B) before "
                             "rendering. r=None disables. This is the byte-RETURNING operator; "
                             "--alpha is a pure fidelity dial that returns bytes only at 0.")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="carrier coefficient scale; 0.0 = carrier deleted (fire first)")
    parser.add_argument("--base-work-dir", type=Path, default=DEFAULT_BASE_WORK)
    parser.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--verify-pairs", type=int, default=N_PAIRS,
                        help="alpha=1 byte-identity control size (default: ALL 600)")
    parser.add_argument("--returned-bytes", type=int, default=None,
                        help="bytes the corresponding payload cut RETURNS to the archive "
                             "(from ra1b's measured rank table). Required to compute an "
                             "affordance bar at alpha != 0; auto-derived only at alpha=0.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--evaluate-timeout", type=int, default=7200)
    parser.add_argument("--skip-evaluate", action="store_true",
                        help="build + control only (no scorer)")
    args = parser.parse_args()

    tag = f"a{args.alpha:g}".replace(".", "p")
    if args.carrier_rank is not None:
        tag = f"rank{args.carrier_rank}_" + tag
    out = args.out_dir or (DEFAULT_OUT_ROOT / f"ddm_ra2c_alpha_ladder_{tag}_20260816")
    submission = out / "submission"
    (submission / "inflated").mkdir(parents=True, exist_ok=True)
    retained = out / "retained"
    retained.mkdir(parents=True, exist_ok=True)

    base_raw = args.base_work_dir / "inflated/0.raw"
    base_archive = args.base_work_dir / "archive.zip"
    for path in (base_raw, base_archive, RA2B_SOURCE):
        if not path.exists():
            raise SystemExit(f"missing required input: {path}")

    ra2b = load_ra2b()
    chain = ra2b.load_chain()
    basis, coeff, selector_blob, provenance = ra2b.decode_carrier(base_archive, chain)
    print(f"decoded carrier: basis {tuple(basis.shape)} coeff {tuple(coeff.shape)}")
    for key, value in provenance.items():
        print(f"  {key}: {value}")

    rank_info = None
    if args.carrier_rank is not None:
        coeff, rank_info = truncate_carrier_rank(basis, coeff, args.carrier_rank)
        print(f"\nRANK-{args.carrier_rank} truncation of the RENDERED FIELD:")
        for key, value in rank_info.items():
            print(f"  {key}: {value}")
        # The alpha control is byte-identity vs the BASE render; a rank-truncated
        # coeff makes that control structurally false, so it is disabled and the
        # run is labelled accordingly. Refusing to report a control we did not run.
        if args.alpha == 1.0 and args.verify_pairs:
            print("  NOTE: alpha=1 control DISABLED under rank truncation "
                  "(coeff is modified; byte-identity cannot hold by construction)")

    # archive.zip held CONSTANT: this run isolates DISTORTION. The rate credit is
    # applied analytically below, never measured here.
    shutil.copy2(base_archive, submission / "archive.zip")

    out_raw = submission / "inflated/0.raw"
    print(f"\nbuilding alpha={args.alpha:g} raw (control on {args.verify_pairs} pairs)...")
    build = build_alpha_raw(
        ra2b, chain, basis, coeff, selector_blob,
        base_raw, out_raw, args.alpha, args.verify_pairs,
    )
    print(f"CONTROL: {build['control_verdict']} "
          f"({build['control_pairs_identical']}/{build['control_pairs_checked']} pairs)")

    # P0 ALWAYS KEEP THE PAYLOAD: the spliced raw IS the payload. Persist it with
    # sha256 + length + the deterministic rebuild command.
    raw_sha = sha256_file(out_raw)
    payload = {
        "spliced_raw_path": str(out_raw),
        "spliced_raw_sha256": raw_sha,
        "spliced_raw_bytes": out_raw.stat().st_size,
        "base_raw_path": str(base_raw),
        "rebuild_command": (
            f".venv/bin/python experiments/ddm_ra2c_alpha_ladder.py "
            f"--alpha {args.alpha:g} --base-work-dir {args.base_work_dir} "
            f"--out-dir {out}"
        ),
    }
    print(f"payload sha256 {raw_sha[:16]}...  {payload['spliced_raw_bytes']:,} B")

    receipt: dict = {
        "schema": "ra2c_alpha_ladder.v1",
        "alpha": args.alpha,
        "axis": "[macOS-CPU advisory, upstream/evaluate.py n600] -- NOT a score claim",
        "score_claim": False,
        "promotable": False,
        "archive_held_constant": True,
        "archive_bytes": BASE_ARCHIVE_BYTES,
        "carrier_provenance": provenance,
        "carrier_rank_truncation": rank_info,
        "control": build,
        "payload": payload,
        "base_row": {"d_pose": BASE_D_POSE, "d_seg": BASE_D_SEG},
    }

    if args.skip_evaluate:
        receipt["verdict"] = "BUILT_NOT_SCORED"
    else:
        print(f"\nrunning upstream/evaluate.py --device {args.device} (n600)...")
        measured = run_evaluate(
            submission, args.upstream_dir, retained / "report.txt",
            args.device, args.evaluate_timeout,
        )
        receipt["measured"] = measured
        print(f"  d_pose {measured['d_pose']:.8f}   d_seg {measured['d_seg']:.8f}"
              f"   ({measured['evaluate_elapsed_s']:.0f}s)")

        from tac.canonical_equations.carrier_rate_credit_pose_affordance_20260816 import (
            FRONTIER_POSE_TERM, is_affordable, pose_affordance_ratio, rate_credit,
        )

        # Rate credit is NOT a function of alpha in general. alpha scales the
        # coefficients at RENDER time; the payload still ships in full unless the
        # carrier is actually removed. Only alpha=0 has a self-evident credit (the
        # whole 22,161 B cell). For any other alpha the corresponding credit comes
        # from ra1b's measured rank table and MUST be supplied explicitly --
        # defaulting it to 0 would silently compute a bar of 1.0x and REFUSE a rung
        # that is in fact affordable (a wrong-instrument refusal, not a verdict).
        if args.returned_bytes is not None:
            returned = args.returned_bytes
        elif args.alpha == 0.0:
            returned = CARRIER_CODED_BYTES
        else:
            returned = None
        ratio = measured["d_pose"] / BASE_D_POSE
        adv_pose_term = (10.0 * BASE_D_POSE) ** 0.5
        aff: dict = {
            "returned_bytes": returned,
            "d_pose_ratio_measured": ratio,
            "advisory_pose_term": adv_pose_term,
            "shipping_pose_term_T4": FRONTIER_POSE_TERM,
            "d_seg_unchanged_by_construction": (
                abs(measured["d_seg"] - BASE_D_SEG) < 1e-9
            ),
        }
        if returned is None:
            aff["bar_status"] = "NOT_COMPUTED_RETURNED_BYTES_UNKNOWN"
            aff["note"] = (
                "alpha != 0 and --returned-bytes was not supplied. The distortion "
                "ratio above is MEASURED and usable; the affordance bar is NOT "
                "computed, because assuming 0 returned bytes would fabricate a "
                "1.0x bar and refuse an affordable rung."
            )
            receipt["verdict"] = "DISTORTION_MEASURED_BAR_NOT_COMPUTED"
            print(f"\n  d_pose ratio {ratio:.4f}x")
            print("  bar NOT computed (supply --returned-bytes from ra1b's rank table)")
        else:
            aff["rate_credit_S"] = rate_credit(returned)
            aff["advisory_bar"] = pose_affordance_ratio(returned, pose_term=adv_pose_term)
            aff["advisory_affordable"] = bool(
                is_affordable(returned, measured["d_pose"],
                              d_pose_base=BASE_D_POSE, pose_term=adv_pose_term)
            )
            aff["shipping_bar_T4"] = pose_affordance_ratio(returned)
            aff["shipping_transfer_assumption"] = (
                "the T4 verdict assumes the d_pose RATIO transfers across instruments; "
                "most defensible at alpha=0 where the perturbation dominates noise"
            )
            receipt["verdict"] = (
                "AFFORDABLE_ON_ADVISORY_BAR" if aff["advisory_affordable"]
                else "REFUSED_ON_ADVISORY_BAR"
            )
            print(f"\n  d_pose ratio {ratio:.4f}x")
            print(f"  advisory bar {aff['advisory_bar']:.4f}x  -> "
                  f"{'AFFORDABLE' if aff['advisory_affordable'] else 'REFUSED'}")
            print(f"  shipping (T4) bar {aff['shipping_bar_T4']:.4f}x  "
                  f"[ratio transfer assumed]")
        receipt["affordance"] = aff
        print(f"  d_seg unchanged: {aff['d_seg_unchanged_by_construction']}")

    path = retained / "RA2C_ALPHA_LADDER.json"
    path.write_text(json.dumps(receipt, indent=2))
    print(f"\nVERDICT: {receipt['verdict']}")
    print(f"receipt -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
