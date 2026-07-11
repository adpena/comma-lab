#!/usr/bin/env python
"""Local (macOS-CPU advisory) exact-score-gated latent click-polish runner.

Task #399 CLICK-POLISH — a DEFENSIVE BANK (NO-FAKE #7): the click-polish
*mechanism* is borrowed from PR128 (a12dongithub, rhnerv_latent_polish, MIT,
[external unverified]); the *substrate* is OURS (the PR110-lineage frontier
archive). This banks score; it is NOT an originality claim. Every reuse claim is
gated by ``tac.click_polish.borrowed_substrate_accounting()``.

This is the CONTAINMENT-SAFE local sibling of ``experiments/modal_click_polish_cpu.py``.
It delegates ALL mechanism to ``src/tac/click_polish.py`` (parse / render / score /
``ClickPolishSearch``) and adds only: greens capture, two full-n600 authority
advisory passes (incumbent + candidate, SAME harness = apples-to-apples), and a
staged MODAL-HOLD exact-eval queue entry. NO paid dispatch is ever fired here.

Scale note (MEASURED 2026-07-11, this machine, 2 threads, sharing a live training
run): one full n600 render+score pass ~= 420 s; a full 28-dim round over n600 ~=
6.5 h. Infeasible under CPU-light containment. So we polish the first ``--n-pairs``
pairs with a FULL dim sweep (pair-locality — MEASURED byte-identical for the other
600-K pairs; ``verify_pair_locality``), which STRICTLY lowers full-n600 S because
(a) locality => only touched pairs change and (b) single +-1 clicks are archive-
BYTE-INVARIANT (MEASURED), so S-decrease == distortion-decrease and a K-mean
distortion drop is a 600-sum distortion drop. The candidate is a full 600-code
byte-closed archive; its full-n600 advisory S is then MEASURED directly (authority
pass), not spliced. Resumable: accepted rounds are replayed from the ledger.

Axis: every number here is ``[macOS-CPU advisory]`` — NON-PROMOTABLE, NOT a score.
Only a byte-closed ``upstream/evaluate.py`` n600 row on Linux x86_64 CPU (or CUDA)
moves the pointer; that command is STAGED (MODAL-HOLD), never run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# keep the loop CPU-light (a live training run owns this machine)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import torch  # noqa: E402

from tac import click_polish as cp  # noqa: E402

AXIS = "[macOS-CPU advisory]"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _full_n600_advisory(renderer, scorer, packet, Q, gt_lstars, gt_poses):
    """MEASURED full-n600 advisory components for a Q candidate (chunked, OOM-safe)."""
    comp = cp.exact_components_for_Q(
        packet, renderer, scorer, Q, gt_lstars, gt_poses,
        pair_indices=list(range(gt_lstars.shape[0])),
    )
    return {
        "d_seg": comp["d_seg"], "d_pose": comp["d_pose"],
        "archive_bytes": comp["archive_bytes"], "S": comp["S"],
    }


def _stage_exact_eval_queue(out_dir: Path, candidate_path: Path, candidate_sha: str,
                            candidate_bytes: int, submission_dir: Path) -> Path:
    """Write a MODAL-HOLD exact-eval queue entry: the exact command that WOULD
    produce a contest-CPU/CUDA score. Never executed here (operator GO required)."""
    entry = {
        "status": "MODAL-HOLD",
        "note": "STAGED ONLY. Do NOT run without explicit operator GO (Modal on HOLD "
                "per operator 2026-07-11). This is the ONLY path that yields a real "
                "score; the local rows in search_result_local.json are advisory.",
        "candidate_archive_path": str(candidate_path),
        "candidate_archive_sha256": candidate_sha,
        "candidate_archive_bytes": candidate_bytes,
        "incumbent_frontier_sha256":
            "ad02b0124cbb3405c23d3480ac16f12b4e48cbf6f75878dd77a5e621bebd079c",
        "incumbent_contest_cpu_score": 0.19108282419209976,
        "submission_runtime_src": str(submission_dir),
        "build_eval_submission": [
            "mkdir -p eval_submission",
            f"cp {submission_dir}/inflate.py eval_submission/inflate.py",
            f"cp {submission_dir}/inflate.sh eval_submission/inflate.sh",
            f"cp -r {submission_dir}/src eval_submission/src",
            f"cp -r {submission_dir}/encoder eval_submission/encoder",
            f"cp {candidate_path} eval_submission/archive.zip",
        ],
        "exact_eval_command_cpu": [
            "bash eval_submission/inflate.sh <archive_dir> <inflated_dir> "
            "upstream/public_test_video_names.txt",
            "python upstream/evaluate.py --submission-dir eval_submission "
            "--uncompressed-dir upstream/videos "
            "--video-names-file upstream/public_test_video_names.txt --device cpu "
            "--report eval_submission/report_cpu.txt",
        ],
        "exact_eval_command_cuda": [
            "# same as cpu but --device cuda on a contest-compliant NVIDIA host",
        ],
        "hardware_axis_required": "linux_x86_64_cpu (contest-CPU) AND/OR NVIDIA (contest-CUDA)",
        "score_claim": False,
        "promotable": False,
        "borrowed_substrate_accounting": cp.borrowed_substrate_accounting(),
    }
    p = out_dir / "staged_exact_eval_queue_MODAL_HOLD.json"
    p.write_text(json.dumps(entry, indent=2))
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", default=str(
        REPO / "experiments/results/clickpolish_pr110_20260710/"
               "n8_validation/candidate_archive.zip"),
        help="incumbent archive to polish (default: current frontier ad02b012)")
    ap.add_argument("--submission-dir", default=str(REPO / cp.DEFAULT_SUBMISSION_DIR))
    ap.add_argument("--gt-cache", default=str(REPO / cp.DEFAULT_GT_CACHE))
    ap.add_argument("--n-pairs", type=int, default=48,
                    help="search scope K (full dim sweep over pairs 0..K-1)")
    ap.add_argument("--max-rounds", type=int, default=1)
    ap.add_argument("--wall-clock-cap-s", type=float, default=3600.0)
    ap.add_argument("--out-dir", default=str(
        REPO / f"experiments/results/click_polish_399_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"))
    ap.add_argument("--skip-authority", action="store_true",
                    help="skip the two full-n600 authority passes (search+bank only)")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"out_dir={out_dir}")
    _log(f"axis={AXIS}  n_pairs(search)={args.n_pairs}  max_rounds={args.max_rounds}  "
         f"wall_clock_cap_s={args.wall_clock_cap_s}")

    pkt = cp.FrozenPacket.parse(args.archive, args.submission_dir)
    rt = pkt.verify_roundtrip()
    _log(f"parse+roundtrip: byte_exact={rt['archive_byte_exact']} "
         f"bytes={rt['archive_bytes']} sha={rt['archive_sha256'][:16]}")
    if not rt["archive_byte_exact"]:
        _log("FATAL: incumbent archive did not roundtrip byte-exact; abort")
        return 3

    renderer = cp.Renderer(pkt, device="cpu")
    scorer = cp.Scorer(device="cpu")

    # greens (pair-locality + diagonal-batch equivalence) — MEASURED, recorded
    loc = cp.verify_pair_locality(pkt, renderer)
    gls24, gps24, _ = cp.load_gt_targets(args.gt_cache, 24)
    be = cp.verify_batch_equivalence(pkt, renderer, scorer, gls24, gps24)
    greens = {
        "locality_holds": loc["locality_holds"],
        "pair_b_unchanged_by_pair_a_click": loc["pair_b_unchanged_by_pair_a_click"],
        "diagonal_equals_sequential_seg_exact": be["diagonal_equals_sequential_seg_exact"],
        "diagonal_vs_sequential_pose_maxabsdiff": be["diagonal_vs_sequential_pose_maxabsdiff"],
    }
    (out_dir / "greens.json").write_text(json.dumps(greens, indent=2))
    _log(f"greens: {greens}")
    if not (greens["locality_holds"] and greens["diagonal_equals_sequential_seg_exact"]):
        _log("FATAL: greens failed; diagonal batching not valid on this decoder; abort")
        return 4

    # full-n600 authority: incumbent baseline (MEASURED, advisory)
    incumbent_full = None
    if not args.skip_authority:
        _log("full-n600 incumbent authority pass (advisory) ...")
        gls600, gps600, gtsrc = cp.load_gt_targets(args.gt_cache, 600)
        t0 = time.time()
        incumbent_full = _full_n600_advisory(renderer, scorer, pkt, pkt.Q0, gls600, gps600)
        incumbent_full["gt_source"] = gtsrc
        incumbent_full["pass_s"] = round(time.time() - t0, 1)
        _log(f"incumbent full-n600 advisory: S={incumbent_full['S']:.8f} "
             f"d_seg={incumbent_full['d_seg']:.8f} d_pose={incumbent_full['d_pose']:.3e} "
             f"bytes={incumbent_full['archive_bytes']} ({incumbent_full['pass_s']}s)")

    # the search (delegates to ClickPolishSearch: resumable, banks per round)
    gls, gps, _ = cp.load_gt_targets(args.gt_cache, args.n_pairs)
    search = cp.ClickPolishSearch(
        packet=pkt, renderer=renderer, scorer=scorer,
        gt_lstars=gls, gt_poses=gps, out_dir=str(out_dir), axis_tag=AXIS,
        max_rounds=args.max_rounds, wall_clock_cap_s=args.wall_clock_cap_s, log=_log,
    )
    t0 = time.time()
    result = search.run()
    result["search_s"] = round(time.time() - t0, 1)
    cand_path = Path(result["candidate_archive_path"])
    cand_bytes = cand_path.read_bytes()
    cand_sha = cp.sha256_hex(cand_bytes)
    _log(f"search done: candidate sha={cand_sha[:16]} bytes={len(cand_bytes)} "
         f"pairs_touched={result['pairs_touched']} clicks={result['net_changed_codes']} "
         f"({result['search_s']}s)")

    # full-n600 authority: candidate (MEASURED, advisory) — the real apples-to-apples row
    candidate_full = None
    if not args.skip_authority:
        _log("full-n600 candidate authority pass (advisory) ...")
        gls600, gps600, _ = cp.load_gt_targets(args.gt_cache, 600)
        t0 = time.time()
        candidate_full = _full_n600_advisory(renderer, scorer, pkt, search.Q, gls600, gps600)
        candidate_full["pass_s"] = round(time.time() - t0, 1)
        dS = None
        if incumbent_full is not None:
            dS = candidate_full["S"] - incumbent_full["S"]
        _log(f"candidate full-n600 advisory: S={candidate_full['S']:.8f} "
             f"d_seg={candidate_full['d_seg']:.8f} d_pose={candidate_full['d_pose']:.3e} "
             f"bytes={candidate_full['archive_bytes']} ({candidate_full['pass_s']}s)  "
             f"dS_advisory={dS}")

    staged = _stage_exact_eval_queue(out_dir, cand_path, cand_sha, len(cand_bytes),
                                     Path(args.submission_dir))

    summary = {
        "axis": AXIS,
        "score_claim": False,
        "promotable": False,
        "incumbent_archive": args.archive,
        "incumbent_archive_sha256": rt["archive_sha256"],
        "candidate_archive": str(cand_path),
        "candidate_archive_sha256": cand_sha,
        "candidate_archive_bytes": len(cand_bytes),
        "search_scope_n_pairs": args.n_pairs,
        "pairs_touched": result["pairs_touched"],
        "net_changed_codes": result["net_changed_codes"],
        "greens": greens,
        "incumbent_full_n600_advisory": incumbent_full,
        "candidate_full_n600_advisory": candidate_full,
        "full_n600_advisory_delta_S": (
            None if (incumbent_full is None or candidate_full is None)
            else candidate_full["S"] - incumbent_full["S"]),
        "search_result": result,
        "staged_exact_eval_queue": str(staged),
        "borrowed_substrate_accounting": cp.borrowed_substrate_accounting(),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "search_result_local.json").write_text(json.dumps(summary, indent=2, default=str))
    _log(f"WROTE {out_dir / 'search_result_local.json'}")
    _log(f"STAGED (MODAL-HOLD) {staged}")
    _log("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
