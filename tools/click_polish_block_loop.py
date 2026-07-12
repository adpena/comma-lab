#!/usr/bin/env python
"""Resumable block-loop click-polish campaign over ALL 600 pairs (task #399 cont.).

Extends the round-1 result of ``tools/click_polish_local.py``: finishes dim
coverage on pairs 0-47 (block 0), then polishes pairs 48-599 in chunk-aligned
48-pair blocks, SERIALLY and CPU-light (a live training run owns this machine).

Accept authority (coordinator 2026-07-11: "accept only on full-n600 authority
passes — never K-scope extrapolation"): every block is accepted on the FULL
n600 S value. That value is computed by CHUNK-SPLICE — re-scoring ONLY the
block's 16-pair chunks and splicing into cached per-pair arrays — which is
EXACTLY the full-pass value because (MEASURED greens, run1 + this tool):
  (a) ``Renderer.render`` reads only the rendered pairs' Q rows
      (``latents.index_select``; pair-locality green: other pairs' frames are
      byte-identical under a click),
  (b) the scorer is per-pair (argmax/MSE; eval-mode BN),
  (c) block boundaries are 16-aligned so the chunk layout of the re-score is
      identical to the canonical full-pass layout (no batch-float confound).
Defense-in-depth: a REAL full n600 pass verifies splice==full at block 0 and
every ``--verify-every`` accepted blocks + at campaign end; any mismatch trips
a confound alarm and the full-pass arrays are adopted.

Banking: each accepted block appends to ``clicks_ledger.jsonl`` (replayable),
refreshes ``candidate_archive.zip`` (byte-closed) + the MODAL-HOLD staged
exact-eval entry, and rewrites ``campaign_state.json`` — an operator GO at ANY
moment has the best-so-far candidate. Touch ``<out_dir>/STOP`` to finish the
current block and exit cleanly. Crash-resume: ledger replay + state file.

Axis: every number is ``[macOS-CPU advisory]`` NON-PROMOTABLE. NO paid dispatch.
Borrowed-substrate framing (NO-FAKE #7) identical to run1: PR128 mechanism
(MIT, external) on OUR PR110-lineage payload — a defensive bank.
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

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tac import click_polish as cp  # noqa: E402

AXIS = "[macOS-CPU advisory]"
BLOCK = 48  # 16-aligned (3 chunks) — REQUIRED for splice equivalence
N = 600


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _splice_S(packet, renderer, dseg_pp, dpose_pp, Q):
    d_seg = float(dseg_pp.mean())
    d_pose = float(dpose_pp.mean())
    bytes_ = len(packet.repack_archive_bytes(Q, drop_sidecar=renderer.drop_sidecar))
    return cp.compute_contest_score(d_seg, d_pose, bytes_), d_seg, d_pose, bytes_


def _stage(out_dir: Path, cand_path: Path, sha: str, nbytes: int, submission_dir: Path):
    entry = {
        "status": "MODAL-HOLD",
        "note": "STAGED ONLY — operator GO required (Modal HOLD, 2026-07-11). "
                "Advisory rows are NOT scores.",
        "candidate_archive_path": str(cand_path),
        "candidate_archive_sha256": sha,
        "candidate_archive_bytes": nbytes,
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
            f"cp {cand_path} eval_submission/archive.zip",
        ],
        "exact_eval_command_cpu": [
            "bash eval_submission/inflate.sh <archive_dir> <inflated_dir> "
            "upstream/public_test_video_names.txt",
            "python upstream/evaluate.py --submission-dir eval_submission "
            "--uncompressed-dir upstream/videos "
            "--video-names-file upstream/public_test_video_names.txt --device cpu "
            "--report eval_submission/report_cpu.txt",
        ],
        "hardware_axis_required": "linux_x86_64_cpu AND/OR contest-CUDA",
        "score_claim": False,
        "promotable": False,
        "borrowed_substrate_accounting": cp.borrowed_substrate_accounting(),
        "staged_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "staged_exact_eval_queue_MODAL_HOLD.json").write_text(
        json.dumps(entry, indent=2))


class Campaign:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.ledger = self.out / "clicks_ledger.jsonl"
        self.state_path = self.out / "campaign_state.json"
        self.auth_npz = self.out / "authority_perpair.npz"
        self.packet = cp.FrozenPacket.parse(args.archive, args.submission_dir)
        if args.drop_sidecar:
            # folded-table mode: base must roundtrip byte-exact under sidecar-less repack
            b = self.packet.repack_archive_bytes(self.packet.Q0, drop_sidecar=True)
            orig = Path(args.archive).read_bytes()
            if b != orig:
                raise SystemExit("FATAL: folded base does not roundtrip byte-exact")
            _log(f"folded base sha={cp.sha256_hex(orig)[:16]} bytes={len(orig)} (roundtrip OK)")
        else:
            rt = self.packet.verify_roundtrip()
            if not rt["archive_byte_exact"]:
                raise SystemExit("FATAL: incumbent roundtrip not byte-exact")
            _log(f"incumbent sha={rt['archive_sha256'][:16]} bytes={rt['archive_bytes']}")
        self.renderer = cp.Renderer(self.packet, device="cpu",
                                    drop_sidecar=bool(args.drop_sidecar))
        self.scorer = cp.Scorer(device="cpu")
        loc = cp.verify_pair_locality(self.packet, self.renderer)
        if not loc["locality_holds"]:
            raise SystemExit("FATAL: pair-locality green failed")
        _log("greens: locality_holds=True (splice precondition)")
        self.gls, self.gps, gtsrc = cp.load_gt_targets(args.gt_cache, N)
        _log(f"gt: {gtsrc}")
        self.Q = self.packet.Q0.copy()
        self.seeded = self._seed_and_replay()
        self.dseg_pp, self.dpose_pp, self.S_auth = self._init_authority()
        self.blocks = [(0, 48)] + [(lo, min(lo + BLOCK, N)) for lo in range(48, N, BLOCK)]
        self.next_block = self._load_next_block()
        self.accepted_since_verify = 0
        # warm-start clicks from a prior campaign's ledger (e.g. the unfolded line):
        # per block we TRY the prior line's NET clicks first (exact gate) before
        # sweeping. Net = sum of deltas per (pair, dim) across all ledger rows.
        self.warm_net: dict[tuple[int, int], int] = {}
        if args.warm_ledger and Path(args.warm_ledger).exists():
            for line in Path(args.warm_ledger).read_text().splitlines():
                if not line.strip():
                    continue
                for p, d, dl in json.loads(line)["clicks"]:
                    k = (int(p), int(d))
                    self.warm_net[k] = self.warm_net.get(k, 0) + int(dl)
            self.warm_net = {k: v for k, v in self.warm_net.items() if v != 0}
            _log(f"warm-start ledger: {len(self.warm_net)} net (pair,dim) clicks "
                 f"from {args.warm_ledger}")

    # ---- state ----
    def _seed_and_replay(self) -> int:
        if not self.ledger.exists() and self.args.seed_ledger:
            seed = Path(self.args.seed_ledger)
            if seed.exists():
                rows = []
                for line in seed.read_text().splitlines():
                    if line.strip():
                        r = json.loads(line)
                        rows.append(json.dumps({
                            "block": -1, "clicks": r["clicks"],
                            "note": "seed:run1_round0", "axis_tag": AXIS}))
                self.ledger.write_text("\n".join(rows) + "\n")
                _log(f"seeded ledger with {len(rows)} row(s) from {seed}")
        n = 0
        if self.ledger.exists():
            for line in self.ledger.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                for p, d, dl in row["clicks"]:
                    self.Q[p, d] = np.clip(int(self.Q[p, d]) + int(dl), 0, 255)
                n += 1
        _log(f"replayed {n} ledger row(s)")
        return n

    def _load_next_block(self) -> int:
        if self.state_path.exists():
            st = json.loads(self.state_path.read_text())
            return int(st.get("next_block", 0))
        return 0

    def _ledger_rows(self) -> int:
        if not self.ledger.exists():
            return 0
        return sum(1 for line in self.ledger.read_text().splitlines() if line.strip())

    def _save_state(self, extra=None):
        st = {
            "next_block": self.next_block,
            "S_authority": self.S_auth,
            "ledger_rows": self._ledger_rows(),
            "axis": AXIS, "score_claim": False, "promotable": False,
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if extra:
            st.update(extra)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=2))
        tmp.rename(self.state_path)

    # ---- authority ----
    def _full_pass(self):
        ds, dp = cp.render_and_score(
            self.renderer, self.scorer, self.Q, list(range(N)), self.gls, self.gps)
        return ds, dp

    def _init_authority(self):
        # cached arrays are trusted ONLY if the state file says they were saved
        # at exactly the ledger row count we just replayed (crash-window guard).
        cached_ok = False
        if self.auth_npz.exists() and self.state_path.exists():
            try:
                st = json.loads(self.state_path.read_text())
                cached_ok = int(st.get("ledger_rows", -1)) == self.seeded
            except Exception:
                cached_ok = False
        if cached_ok:
            d = np.load(self.auth_npz)
            ds, dp = d["d_seg"], d["d_pose"]
            S, dsm, dpm, b = _splice_S(self.packet, self.renderer, ds, dp, self.Q)
            _log(f"authority (cached): S={S:.8f} d_seg={dsm:.8f} d_pose={dpm:.3e} bytes={b}")
            return ds, dp, S
        _log("authority: initial FULL n600 pass ...")
        t0 = time.time()
        ds, dp = self._full_pass()
        S, dsm, dpm, b = _splice_S(self.packet, self.renderer, ds, dp, self.Q)
        np.savez(self.auth_npz, d_seg=ds, d_pose=dp)
        _log(f"authority (full): S={S:.8f} d_seg={dsm:.8f} d_pose={dpm:.3e} "
             f"bytes={b} ({time.time()-t0:.0f}s)")
        return ds, dp, S

    def _verify_full(self, tag: str) -> bool:
        _log(f"[verify:{tag}] FULL n600 pass to check splice==full ...")
        t0 = time.time()
        ds, dp = self._full_pass()
        seg_max = float(np.abs(ds - self.dseg_pp).max())
        pose_max = float(np.abs(dp - self.dpose_pp).max())
        ok = seg_max == 0.0 and pose_max < 1e-12
        _log(f"[verify:{tag}] seg_maxabs={seg_max:.3e} pose_maxabs={pose_max:.3e} "
             f"ok={ok} ({time.time()-t0:.0f}s)")
        if not ok:
            _log(f"[verify:{tag}] CONFOUND ALARM: splice!=full — adopting full arrays")
            self.dseg_pp, self.dpose_pp = ds, dp
            self.S_auth, *_ = _splice_S(self.packet, self.renderer, ds, dp, self.Q)
            np.savez(self.auth_npz, d_seg=ds, d_pose=dp)
        return ok

    # ---- block polish ----
    def _score_W(self, Q, W):
        return cp.render_and_score(self.renderer, self.scorer, Q, W, self.gls, self.gps)

    def _sweep_block(self, W, cap_s):
        """Diagonal sweep with per-pass resumable state (sweep_state.json) — any
        crash/kill loses at most ONE W-pass (the sibling ops-note insurance)."""
        t0 = time.time()
        dseg0 = self.dseg_pp[W]
        dpose0 = self.dpose_pp[W]
        wpose = 5.0 / np.sqrt(10.0 * max(float(self.dpose_pp.mean()), 1e-9))
        deltas = [(d, dl) for d in range(cp.LATENT_DIM) for dl in self.args_sweep]
        best = {p: None for p in W}
        best_gain = {p: 0.0 for p in W}
        sw_path = self.out / "sweep_state.json"
        start_i = 0
        if sw_path.exists():
            try:
                st = json.loads(sw_path.read_text())
                if (st.get("block") == self.next_block
                        and st.get("deltas") == [list(x) for x in deltas]
                        and st.get("ledger_rows") == self._ledger_rows()):
                    start_i = int(st["next_i"])
                    for k, v in st["best"].items():
                        p = int(k)
                        if p in best and v is not None:
                            best[p] = (int(v[0]), int(v[1]))
                            best_gain[p] = float(v[2])
                    _log(f"  [sweep] resumed at pass {start_i}/{len(deltas)}")
            except Exception:
                start_i = 0
        n_pass = 0
        for i in range(start_i, len(deltas)):
            d, dl = deltas[i]
            if cap_s > 0 and time.time() - t0 > cap_s:
                _log(f"  [sweep] cap {cap_s}s hit after {n_pass} passes — partial "
                     f"(resumable at pass {i})")
                return best
            Qc = self.Q.copy()
            Qc[W, d] = np.clip(Qc[W, d].astype(np.int16) + dl, 0, 255).astype(np.uint8)
            ds, dp = self._score_W(Qc, W)
            n_pass += 1
            proxy = 100.0 * (ds - dseg0) + wpose * (dp - dpose0)
            for j, p in enumerate(W):
                if Qc[p, d] == self.Q[p, d]:
                    continue
                gain = -float(proxy[j])
                if gain > best_gain[p] + 1e-12:
                    best_gain[p] = gain
                    best[p] = (d, int(dl))
            tmp = sw_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({
                "block": self.next_block, "next_i": i + 1,
                "deltas": [list(x) for x in deltas],
                "ledger_rows": self._ledger_rows(),
                "best": {str(p): (None if best[p] is None else
                                  [best[p][0], best[p][1], best_gain[p]])
                         for p in W},
            }))
            tmp.rename(sw_path)
        sw_path.unlink(missing_ok=True)
        _log(f"  [sweep] complete: {n_pass} passes ({time.time()-t0:.0f}s)")
        return best

    def _try_accept(self, W, clicks):
        """Exact accept gate on the FULL n600 authority via chunk-splice."""
        Qcand = self.Q.copy()
        for p, d, dl in clicks:
            Qcand[p, d] = np.clip(int(Qcand[p, d]) + dl, 0, 255)
        ds_W, dp_W = self._score_W(Qcand, W)  # canonical 16-chunk layout (W 16-aligned)
        ds_new = self.dseg_pp.copy(); ds_new[W] = ds_W
        dp_new = self.dpose_pp.copy(); dp_new[W] = dp_W
        S_new, dsm, dpm, b = _splice_S(self.packet, self.renderer, ds_new, dp_new, Qcand)
        if S_new < self.S_auth:
            return Qcand, ds_new, dp_new, S_new, dsm, dpm, b
        return None

    def _bank(self, blk_idx, lo, hi, clicks, S, dsm, dpm, b, authority):
        row = {
            "block": blk_idx, "pairs": [lo, hi], "clicks": clicks,
            "n_clicks": len(clicks), "S_after_n600": S, "d_seg": dsm,
            "d_pose": dpm, "archive_bytes": b, "authority": authority,
            "axis_tag": AXIS,
        }
        with open(self.ledger, "a") as f:
            f.write(json.dumps(row) + "\n"); f.flush(); os.fsync(f.fileno())
        np.savez(self.auth_npz, d_seg=self.dseg_pp, d_pose=self.dpose_pp)
        archive = self.packet.repack_archive_bytes(
            self.Q, drop_sidecar=self.renderer.drop_sidecar)
        cand = self.out / "candidate_archive.zip"
        cand.write_bytes(archive)
        sha = cp.sha256_hex(archive)
        _stage(self.out, cand, sha, len(archive), Path(self.args.submission_dir))
        self._save_state({"candidate_sha256": sha, "candidate_bytes": len(archive)})
        xover = ""
        if self.args.ref_frontier_s:
            xover += f" vsFrontierAdv={S - self.args.ref_frontier_s:+.2e}"
        if self.args.ref_unfolded_s:
            xover += f" vsUnfoldedBest={S - self.args.ref_unfolded_s:+.2e}"
        _log(f"  BANKED block {blk_idx}: sha={sha[:16]} S={S:.8f}{xover}")
        return sha

    def run(self):
        self.args_sweep = tuple(int(x) for x in self.args.sweep_deltas.split(","))
        _log(f"campaign: blocks {self.next_block}..{len(self.blocks)-1} of "
             f"{len(self.blocks)}; S_auth={self.S_auth:.8f}; axis={AXIS}")
        stop = self.out / "STOP"
        last_sha = None
        blocks_done_this_run = 0
        while self.next_block < len(self.blocks):
            if stop.exists():
                _log("STOP sentinel — exiting cleanly (state banked)")
                break
            blk = self.next_block
            lo, hi = self.blocks[blk]
            W = list(range(lo, hi))
            _log(f"[block {blk}] pairs {lo}..{hi-1}")
            # warm-start: try the prior-campaign clicks for this block first (exact gate)
            warm = [(p, d, dl) for (p, d), dl in sorted(self.warm_net.items())
                    if lo <= p < hi]
            if warm:
                got_w = self._try_accept(W, warm)
                if got_w is not None:
                    Qw, ds_w, dp_w, S_w, dsm_w, dpm_w, b_w = got_w
                    self.Q, self.dseg_pp, self.dpose_pp, self.S_auth = Qw, ds_w, dp_w, S_w
                    _log(f"[block {blk}] warm-start accepted {len(warm)} clicks -> "
                         f"full-n600 S={S_w:.8f}")
                    self._bank(blk, lo, hi, warm, S_w, dsm_w, dpm_w, b_w, "splice-warm")
                else:
                    _log(f"[block {blk}] warm-start clicks did not improve — sweeping")
            best = self._sweep_block(W, self.args.block_sweep_cap_s)
            clicks = [(p, *best[p]) for p in W if best[p] is not None]
            if not clicks:
                _log(f"[block {blk}] plateau — no improving click")
                self.next_block += 1
                blocks_done_this_run += 1
                self._save_state()
                if self.args.max_blocks and blocks_done_this_run >= self.args.max_blocks:
                    _log("max-blocks reached — exiting (resumable)")
                    break
                continue
            got = self._try_accept(W, clicks)
            if got is None:
                # greedy halving on the same full-n600 splice gate
                cur = list(clicks)
                while cur and got is None:
                    cur = cur[: max(1, len(cur) // 2)]
                    got = self._try_accept(W, cur)
                    if got is None and len(cur) == 1:
                        cur = []
                clicks = cur
            if got is None or not clicks:
                _log(f"[block {blk}] no accepted subset improved full-n600 S")
                self.next_block += 1
                blocks_done_this_run += 1
                self._save_state()
                if self.args.max_blocks and blocks_done_this_run >= self.args.max_blocks:
                    _log("max-blocks reached — exiting (resumable)")
                    break
                continue
            Qn, ds_n, dp_n, S_new, dsm, dpm, b = got
            self.Q = Qn
            self.dseg_pp, self.dpose_pp, self.S_auth = ds_n, dp_n, S_new
            _log(f"[block {blk}] accepted {len(clicks)} clicks -> "
                 f"full-n600 S={S_new:.8f} d_seg={dsm:.8f} d_pose={dpm:.3e}")
            self.next_block += 1
            self.accepted_since_verify += 1
            authority = "splice"
            if blk == 0 or self.accepted_since_verify >= self.args.verify_every:
                self._verify_full(f"block{blk}")
                self.accepted_since_verify = 0
                authority = "splice+full-verified"
            last_sha = self._bank(blk, lo, hi, clicks, self.S_auth,
                                  float(self.dseg_pp.mean()),
                                  float(self.dpose_pp.mean()), b, authority)
            blocks_done_this_run += 1
            if self.args.max_blocks and blocks_done_this_run >= self.args.max_blocks:
                _log("max-blocks for this invocation reached — exiting (resumable)")
                break
        # final verification if we did any work
        if last_sha is not None:
            self._verify_full("final")
            self._save_state({"candidate_sha256": last_sha})
        _log(f"campaign paused/done: next_block={self.next_block} "
             f"S_auth={self.S_auth:.8f} {AXIS}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", default=str(
        REPO / "experiments/results/clickpolish_pr110_20260710/"
               "n8_validation/candidate_archive.zip"))
    ap.add_argument("--submission-dir", default=str(REPO / cp.DEFAULT_SUBMISSION_DIR))
    ap.add_argument("--gt-cache", default=str(REPO / cp.DEFAULT_GT_CACHE))
    ap.add_argument("--out-dir", default=str(
        REPO / "experiments/results/click_polish_399_campaign"))
    ap.add_argument("--seed-ledger", default=str(
        REPO / "experiments/results/click_polish_399_run1/accepted_clicks_ledger.jsonl"))
    ap.add_argument("--sweep-deltas", default="1,-1")
    ap.add_argument("--drop-sidecar", action="store_true",
                    help="folded-table mode: sidecar-less renderer + repack")
    ap.add_argument("--warm-ledger", default="",
                    help="prior campaign clicks_ledger.jsonl to TRY per block before sweeping")
    ap.add_argument("--ref-frontier-s", type=float, default=0.0,
                    help="frontier advisory S for crossover logging")
    ap.add_argument("--ref-unfolded-s", type=float, default=0.0,
                    help="unfolded best advisory S for crossover logging")
    ap.add_argument("--block-sweep-cap-s", type=float, default=3300.0)
    ap.add_argument("--verify-every", type=int, default=4,
                    help="full n600 verification pass every N accepted blocks")
    ap.add_argument("--max-blocks", type=int, default=0,
                    help="stop after this many blocks this invocation (0=all)")
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    Campaign(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
