#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Build the comma10k regime prior artifact for the #426 organ's ARM I ($0 compute).

Bounded, one-time fetch of REAL comma10k mask PNGs (SegNet's actual training
distribution — 0 contest frames, L80) from the public GitHub repo, decode with PIL,
reduce to the per-class regime prior via
``tac.witness_control.scorer_model_arms.build_comma10k_prior_from_labels``, and write
the durable JSON artifact the arm reads offline forever after:

    experiments/results/comma10k_regime_prior/comma10k_class_prior.json

Provenance discipline: a manifest (file names, GitHub blob shas, byte sizes, fetch UTC)
lands next to the prior; the mask PNGs themselves are kept (tiny — ~8 KB each) so the
reduction is re-runnable offline. NO scorer forward, NO GPU, the live run untouched.
Every number [macOS advisory] NON-PROMOTABLE; the arm is admitted by BACKTEST only.

Usage:
  .venv/bin/python tools/build_comma10k_regime_prior.py --n-masks 192
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

API = "https://api.github.com/repos/commaai/comma10k/contents/masks?per_page=1000"
OUT_DIR = REPO / "experiments/results/comma10k_regime_prior"


def _fetch(url: str, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "pact-organ-arm-i"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (https only)
        return r.read()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-masks", type=int, default=192,
                    help="bounded sample size (strided across the listing for diversity)")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    import numpy as np
    from PIL import Image

    from tac.witness_control.scorer_model_arms import (
        build_comma10k_prior_from_labels, comma10k_labels_from_rgb)

    out_dir = Path(args.out_dir)
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    listing = json.loads(_fetch(API).decode())
    files = [e for e in listing if e.get("type") == "file"
             and e.get("name", "").endswith(".png")]
    if not files:
        print("no mask files in the GitHub listing — aborting", file=sys.stderr)
        return 2
    stride = max(len(files) // max(args.n_masks, 1), 1)
    picked = files[::stride][:args.n_masks]
    print(f"listing has {len(files)} masks; fetching {len(picked)} (stride {stride})")

    labels, manifest_rows = [], []
    for i, e in enumerate(picked):
        dest = masks_dir / e["name"]
        if dest.exists():
            raw = dest.read_bytes()
        else:
            raw = _fetch(e["download_url"])
            dest.write_bytes(raw)
        rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        labels.append(comma10k_labels_from_rgb(rgb))
        manifest_rows.append({"name": e["name"], "github_blob_sha": e["sha"],
                              "bytes": len(raw)})
        if (i + 1) % 32 == 0:
            print(f"  {i + 1}/{len(picked)} decoded")

    prior = build_comma10k_prior_from_labels(
        labels, source=f"comma10k@master masks n={len(labels)} (bounded strided sample)")
    print("class_pixel_share:", [round(v, 5) for v in prior.class_pixel_share])
    print("class_boundary_share:", [round(v, 5) for v in prior.class_boundary_share])
    print("unmatched_frac:", round(prior.unmatched_frac, 6))
    print("rarity_reweight:", np.round(prior.rarity_reweight(), 3).tolist())
    if prior.unmatched_frac > 0.05:
        print("REFUSING to write: >5% unmatched palette pixels (mask decode suspect)",
              file=sys.stderr)
        return 3

    out = out_dir / "comma10k_class_prior.json"
    payload = prior.to_jsonable()
    payload["fetched_utc"] = datetime.now(UTC).isoformat()
    payload["score_claim"] = False
    payload["promotable"] = False
    out.write_text(json.dumps(payload, indent=1))
    (out_dir / "fetch_manifest.json").write_text(json.dumps({
        "api": API, "n_listed": len(files), "n_fetched": len(picked),
        "stride": stride, "fetched_utc": payload["fetched_utc"],
        "files": manifest_rows}, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
