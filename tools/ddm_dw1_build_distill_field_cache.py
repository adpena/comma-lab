"""ddm_dw1 — build the concatenated teacher distill-logit cache (SSD, certify-or-block).

The QA75 distill window consumes the b2b SegNet FIELD (600 per-pair NPZs, each
``distill_logits (5,384,512) fp16``) as the *precomputed scorer response* teacher.
Reading a zip-compressed NPZ per pair per step (600 x N_ep loads) is I/O-bound; this
tool concatenates the per-pair ``distill_logits`` into ONE raw fp16 ``.npy`` memmap
(600, 5, 384, 512) on the SSD tier so the trainer memmaps it (OS page cache, no
decompress).  The teacher argmax + margin are DERIVED from these logits in-loop
(top1/top2) — no separate fields cached.

Integrity: every source pair's sha256 is checked against ``field_pass_manifest.json``
BEFORE it enters the cache (fail-closed on any mismatch).  The output carries its own
manifest with the source manifest sha, per-pair source shas rolled up, output sha256,
bytes, and shape — deterministic-reproducibility custody (rebuildable from the field).

score_claim=false; advisory [macOS-CPU]; pointer 0.1910828242 [contest-CPU] UNMOVED.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

FIELD_DIR = "/Volumes/VertigoDataTier/pact/ddm_b2b_qa75_field_20260730"
OUT_DIR = "/Volumes/VertigoDataTier/pact/ddm_dw1_20260730/distill_field_cache"
SEG_H, SEG_W, N_CLS = 384, 512, 5


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--field-dir", type=Path, default=Path(FIELD_DIR))
    ap.add_argument("--out-dir", type=Path, default=Path(OUT_DIR))
    ap.add_argument("--limit", type=int, default=0, help="0 = all pairs (n600); >0 for a smoke")
    args = ap.parse_args()

    manifest_path = args.field_dir / "field_pass_manifest.json"
    if not manifest_path.is_file():
        print(f"BLOCK: source manifest missing: {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text())
    src_manifest_sha = _sha256_file(manifest_path)
    pairs = manifest["pairs"]
    n = len(pairs) if args.limit <= 0 else min(args.limit, len(pairs))
    if manifest.get("geometry", {}) != {"seg_h": SEG_H, "seg_w": SEG_W, "n_classes": N_CLS}:
        print(f"BLOCK: unexpected geometry {manifest.get('geometry')}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_npy = args.out_dir / "distill_logits.f16.npy"
    mm = np.lib.format.open_memmap(
        out_npy, mode="w+", dtype=np.float16, shape=(n, N_CLS, SEG_H, SEG_W))

    per_pair_src_sha: list[str] = []
    for i in range(n):
        row = pairs[i]
        if int(row["pair_id"]) != i:
            print(f"BLOCK: pair_id gap at index {i}: {row['pair_id']}", file=sys.stderr)
            return 2
        p = args.field_dir / row["path"]
        got = _sha256_file(p)
        if got != row["sha256"]:
            print(f"BLOCK: sha mismatch pair {i}: {got} != {row['sha256']}", file=sys.stderr)
            return 2
        per_pair_src_sha.append(got)
        z = np.load(p, allow_pickle=False)
        dl = z["distill_logits"]
        if dl.shape != (N_CLS, SEG_H, SEG_W):
            print(f"BLOCK: pair {i} distill_logits shape {dl.shape}", file=sys.stderr)
            return 2
        mm[i] = dl.astype(np.float16)
        if (i + 1) % 100 == 0:
            print(f"  cached {i + 1}/{n} pairs")
    mm.flush()
    del mm

    out_sha = _sha256_file(out_npy)
    src_roll = hashlib.sha256("".join(per_pair_src_sha).encode()).hexdigest()
    out_manifest = {
        "schema": "ddm_dw1_distill_field_cache.v1",
        "purpose": "concatenated teacher SegNet distill-logit cache for the QA75 distill window",
        "source_field_dir": str(args.field_dir),
        "source_manifest_sha256": src_manifest_sha,
        "source_per_pair_sha_rollup": src_roll,
        "field_kind": manifest.get("field_kind"),
        "n_pairs": n,
        "shape": [n, N_CLS, SEG_H, SEG_W],
        "dtype": "float16",
        "layout": "(pair, class, H=384, W=512); teacher argmax+margin DERIVED in-loop (top1/top2)",
        "out_npy": str(out_npy),
        "out_sha256": out_sha,
        "out_bytes": out_npy.stat().st_size,
        "rebuildable": "re-run tools/ddm_dw1_build_distill_field_cache.py over the b2b field",
        "authority": "advisory [macOS-CPU]; score_claim=false",
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
    }
    (args.out_dir / "cache_manifest.json").write_text(json.dumps(out_manifest, indent=2) + "\n")
    print(json.dumps({k: out_manifest[k] for k in
                      ("n_pairs", "shape", "out_bytes", "out_sha256")}, indent=2))
    print(f"OK: cache at {out_npy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
