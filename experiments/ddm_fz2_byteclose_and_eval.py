# SPDX-License-Identifier: MIT
"""ddm_fz2 -- byte-close and score a staged candidate through the CANONICAL chain only.

Every stage is a call into ``tac.submission_chain``; this file contributes argument
plumbing and a receipt, never rebuild logic.  That is the point: the
submission-chain-must-be-canonical law exists because ddm_pu2's rebuild lived in a
probe script with four unverified vendored files and no byte ledger.  Discovery may
happen in probes; REPRODUCTION happens here.

Stages, each fail-closed (the chain raises ``SubmissionChainError``):

    build_byte_ledger      every counted section named + hashed, framing residual 0,
                           payload re-encodes identically
    audit_runtime_tree     every staged runtime file hashed against its repo source
                           (UNMAPPED is reported, never silently treated as clean)
    run_inflate            the SHIPPED inflate.sh, foreground, refusing rc!=0 AND the
                           vacuity class (rc=0 with no/zero-byte output, m50)
    run_upstream_evaluate  the REAL upstream/evaluate.py on the exact staged bytes;
                           S is RECOMPUTED from d_seg/d_pose/bytes, never read from
                           the rounded printed field

The recomputed score is the row.  Axis is derived from ``--device`` by the chain
(``cpu`` on this host is ``[macOS-CPU advisory]``, NOT contest-CPU: contest-CPU
requires Linux x86_64 per the dual-axis rule), and is carried in the receipt.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# Staged runtime basename -> repo-relative source.  The shipped name is not always
# the repo name (``inflate_runner.py`` is the repo's ``inflate_runner_v4d.py``), and
# an unmapped file must surface as UNMAPPED rather than pass silently.
RUNTIME_REPO_MAP = {
    "inflate_runner.py": "experiments/inflate_runner_v4d.py",
    "ddm_ix2_archive_container.py": "src/tac/optimization/ddm_ix2_archive_container.py",
    "ddm_tr1_runtime.py": "src/tac/optimization/ddm_tr1_runtime.py",
    "pfs1_warp_receiver.py": "src/tac/optimization/pfs1_warp_receiver.py",
    "repair_entropy_coder_runtime_adapters.py":
        "src/tac/optimization/repair_entropy_coder_runtime_adapters.py",
    "ddm_r7_token_coder.py": "experiments/ddm_r7_token_coder.py",
}

# The v4d joint group extended with the F0PR1 frame_0 pose-repair section.
JOINT_NAMES = ("config", "renderer", "selector", "pose_warp", "frame0_pose_repair")


def _asdict(obj):
    if hasattr(obj, "__dict__"):
        return {k: _asdict(v) for k, v in vars(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [_asdict(v) for v in obj]
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub-dir", type=Path, required=True,
                    help="staged submission dir (archive.zip + runtime tree)")
    ap.add_argument("--out", type=Path, required=True, help="receipt json")
    ap.add_argument("--inflate-out", type=Path, required=True)
    ap.add_argument("--upstream", type=Path, default=REPO / "upstream")
    ap.add_argument("--videos", type=Path, default=REPO / "upstream" / "videos")
    ap.add_argument("--names", type=Path,
                    default=REPO / "upstream" / "public_test_video_names.txt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-threads", type=int, default=6)
    ap.add_argument("--skip-eval", action="store_true",
                    help="ledger + custody + inflate only (no scorer slot)")
    args = ap.parse_args()

    # upstream/evaluate.py:68 reads <submission_dir>/inflated; an eval pointed at any
    # other inflate-out dies in ds_comp.prepare_data with a bare AssertionError.  Two
    # hand-staged scripts hit this on 2026-08-05 (pe2 v1, pe4) — refuse it here.
    expected_inflate = (args.sub_dir / "inflated").resolve()
    if not args.skip_eval and args.inflate_out.resolve() != expected_inflate:
        raise SystemExit(
            f"--inflate-out must be {expected_inflate} when eval runs "
            f"(got {args.inflate_out.resolve()}); pass --skip-eval for custody-only runs"
        )

    from tac.submission_chain import (
        audit_runtime_tree,
        build_byte_ledger,
        run_inflate,
        run_upstream_evaluate,
    )

    t0 = time.time()
    sub = args.sub_dir
    archive = sub / "archive.zip"
    receipt: dict = {
        "schema": "ddm_fz2_byteclose_and_eval.v1",
        "score_claim": False,
        "promotion_eligible": False,
        "submission_dir": str(sub),
        "archive_bytes": archive.stat().st_size,
    }

    ledger = build_byte_ledger(archive, joint_names=JOINT_NAMES)
    closes = ledger.closes() if callable(ledger.closes) else ledger.closes
    receipt["byte_ledger"] = {
        "closes": bool(closes),
        "residual_bytes": ledger.residual_bytes,
        "payload_reencodes_identically": ledger.payload_reencodes_identically,
        "sections": [_asdict(s) for s in getattr(ledger, "sections", ())],
        "members": [_asdict(m) for m in getattr(ledger, "members", ())],
    }
    print(f"[fz2] ledger closes={closes} residual={ledger.residual_bytes} "
          f"reencodes={ledger.payload_reencodes_identically}", flush=True)
    if not closes:
        raise SystemExit("byte ledger does not close -- REFUSING to score")

    custody = audit_runtime_tree(sub, repo_root=REPO, repo_map=RUNTIME_REPO_MAP)
    receipt["runtime_custody"] = _asdict(custody)
    print(f"[fz2] runtime custody: {json.dumps(receipt['runtime_custody'])[:600]}", flush=True)

    # The shipped inflate.sh reads an EXTRACTED archive dir.  Extract it here from
    # the staged archive.zip rather than inheriting whatever a previous script left
    # behind: a reproduction step that depends on an earlier probe's side effect is
    # not a reproduction.  Re-extracting also guarantees the decoded tree matches
    # the bytes the ledger just hashed.
    archive_dir = sub / "archive"
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    archive_dir.mkdir(parents=True)
    from tac.submission_archive import safe_extract_zip

    safe_extract_zip(archive, archive_dir)
    receipt["archive_members"] = sorted(p.name for p in archive_dir.iterdir())

    inf = run_inflate(sub, archive_dir=archive_dir, out_dir=args.inflate_out,
                      video_names_file=args.names)
    receipt["inflate"] = _asdict(inf)
    print(f"[fz2] inflate rc={inf.returncode} files={inf.raw_files} "
          f"bytes={inf.raw_bytes} {inf.seconds:.0f}s", flush=True)

    if not args.skip_eval:
        ev = run_upstream_evaluate(
            sub, upstream_dir=args.upstream, videos_dir=args.videos,
            video_names_file=args.names, archive_bytes=archive.stat().st_size,
            device=args.device, batch_size=args.batch_size,
            num_threads=args.num_threads, require_n600=True,
        )
        receipt["evaluate"] = _asdict(ev)
        print(f"[fz2] EVALUATE {json.dumps(receipt['evaluate'])[:900]}", flush=True)

    receipt["seconds"] = time.time() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=1, default=str))
    print(f"[fz2] receipt -> {args.out} t={receipt['seconds']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
