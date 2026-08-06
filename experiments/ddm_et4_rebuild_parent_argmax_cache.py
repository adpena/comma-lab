"""Rebuild the et2 parent argmax cache under the et4 runner's own forward instrument.

Root cause (measured 2026-08-06): the et2 cache builder scored the parent decode in
BATCHES OF 16 (parent_score/batch_*.json); the et4 runner forwards ONE pair at a time.
oneDNN selects different conv kernels/blocking per batch shape, so FP summation order
differs and near-tie argmax pixels flip between the two instruments. Measured: pair 17
site (289, 43) live=4(MyCar) vs cache=0(Road) at BOTH threads=4 and threads=6 batch-1,
while batch-16 @ threads=6 reproduces the cache exactly (0/196,608 on pairs 16-31).
The et4 C2 custody gate compares live batch-1 output against the batch-16 cache and
correctly refuses -- the gate measured the instrument seam, not a decode defect.

The honest repair (never weaken C2): rebuild the reference cache with the SAME
instrument the runner solves with -- batch-1 forwards via the runner's own
load_models/forward imports -- so reference and solver share one instrument by
construction. The original batch-16 cache bytes are preserved alongside
(certify-or-block; no signal loss), and a receipt records every differing site.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO / "src"), str(REPO / "upstream"), str(REPO / "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

from ddm_et2_projected_phase_field import forward, load_models  # noqa: E402

DEFAULT_RAW = Path(
    "/Volumes/VertigoDataTier/pact/ddm_et2_20260806/parent_tq1c_decode/submission/inflated/0.raw"
)
DEFAULT_CACHE = Path(
    "/Volumes/VertigoDataTier/pact/ddm_et2_20260806/parent_score/parent_tq1c_argmax_n600.npy"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--threads", type=int, default=4, help="match the et4 shard config")
    ap.add_argument("--pairs", type=int, default=600)
    ap.add_argument("--receipt", type=Path, default=None)
    args = ap.parse_args(argv)

    old_sha = sha256_file(args.cache)
    old_cache = np.load(args.cache)  # full load; the live file is replaced at the end
    raw = np.memmap(args.raw, dtype=np.uint8, mode="r").reshape(-1, 874, 1164, 3)
    assert raw.shape[0] >= 2 * args.pairs, f"raw has {raw.shape[0]} frames"
    assert old_cache.shape[0] == args.pairs

    segnet, posenet, scorer_custody = load_models(REPO / "upstream", threads=args.threads)

    new_cache = np.empty_like(old_cache)
    diff_rows: list[dict] = []
    for pair in range(args.pairs):
        dec = np.stack([raw[2 * pair], raw[2 * pair + 1]]).astype(np.uint8)
        cells, _pose = forward(segnet, posenet, dec[None])
        live = np.asarray(cells[0], dtype=old_cache.dtype)
        new_cache[pair] = live
        diff = live != old_cache[pair]
        n = int(diff.sum())
        if n:
            ys, xs = np.nonzero(diff)
            sites = [
                [int(ys[i]), int(xs[i]), int(live[ys[i], xs[i]]), int(old_cache[pair][ys[i], xs[i]])]
                for i in range(len(ys))
            ]
            diff_rows.append({"pair": pair, "n_diff": n, "sites_y_x_new_old": sites})
            print(f"pair {pair}: {n} px differ from batch-16 cache", flush=True)
        if pair % 50 == 0:
            print(f"progress {pair}/{args.pairs}", flush=True)

    # Preserve the original batch-16 cache bytes before installing the rebuild.
    preserved = args.cache.with_suffix(".batch16.npy")
    if not preserved.exists():
        shutil.copy2(args.cache, preserved)
    tmp = args.cache.with_suffix(".tmp.npy")
    np.save(tmp, new_cache)
    os.replace(tmp, args.cache)
    new_sha = sha256_file(args.cache)

    receipt = {
        "written_at_utc": datetime.now(UTC).isoformat(),
        "instrument": {
            "forward": "ddm_et2_projected_phase_field.forward",
            "batch": 1,
            "threads": args.threads,
            "deterministic_algorithms": True,
            "seed": 1234,
            **scorer_custody,
        },
        "old_cache_instrument": "batch=16 (parent_score/batch_*.json), threads=6",
        "old_cache_sha256": old_sha,
        "old_cache_preserved_at": str(preserved),
        "new_cache_sha256": new_sha,
        "pairs": args.pairs,
        "n_pairs_differing": len(diff_rows),
        "total_px_differing": int(sum(r["n_diff"] for r in diff_rows)),
        "diffs": diff_rows,
        "raw_path": str(args.raw),
        "raw_sha256": None,  # 3.6GB memmap; decode custody held by et2's C2-passing receipt
    }
    receipt_path = args.receipt or args.cache.parent / "rebuild_batch1_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"DONE: {len(diff_rows)} pairs differ; receipt {receipt_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
