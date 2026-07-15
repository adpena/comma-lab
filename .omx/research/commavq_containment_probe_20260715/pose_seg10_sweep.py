"""commaVQ containment probe: extract all *_10.npy pose members via ZIP range
requests and correlate speed profile against our comma2k19 GT (segment 10).

Resumable: writes per-zip result npz to ./seg10_cache/; skips done zips.
Run: .venv/bin/python pose_seg10_sweep.py <start_idx> <end_idx>
"""
import io
import json
import os
import struct
import sys
import time
import zlib

import numpy as np
import urllib.request

REV = "610fb2854c"
BASE = f"https://huggingface.co/datasets/commaai/commavq/resolve/{REV}"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seg10_cache")
os.makedirs(CACHE, exist_ok=True)

GT = np.load(
    "/Users/adpena/Projects/pact/experiments/results/pose_feasibility_probe/comma2k19_gt_pose_raw.npz"
)
gt_speed = np.linalg.norm(GT["frame_velocities"], axis=1)  # (1200,) m/s


def http(url, headers=None, method="GET", retries=8):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers or {}, method=method)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read(), dict(r.headers)
        except Exception as e:
            if i == retries - 1:
                raise
            wait = 5 * (i + 1)
            if "429" in str(e):
                wait = 30 * (i + 1)
            time.sleep(wait)
        # gentle pacing against rate limits
        time.sleep(0.05)


def central_directory(url):
    _, h = http(url, method="HEAD")
    size = int(h["Content-Length"])
    tail = 1 << 20
    start = max(0, size - tail)
    data, _ = http(url, {"Range": f"bytes={start}-{size-1}"})
    eocd = data.rfind(b"PK\x05\x06")
    cd_size, cd_offset = struct.unpack("<II", data[eocd + 12 : eocd + 20])
    if cd_offset < start:
        data, _ = http(url, {"Range": f"bytes={cd_offset}-{size-1}"})
        start = cd_offset
    cd = data[cd_offset - start : cd_offset - start + cd_size]
    out = []  # (name, comp_method, comp_size, uncomp_size, local_hdr_offset)
    off = 0
    while off + 46 <= len(cd) and cd[off : off + 4] == b"PK\x01\x02":
        comp = struct.unpack("<H", cd[off + 10 : off + 12])[0]
        csize, usize = struct.unpack("<II", cd[off + 20 : off + 28])
        nlen, elen, clen = struct.unpack("<HHH", cd[off + 28 : off + 34])
        lho = struct.unpack("<I", cd[off + 42 : off + 46])[0]
        name = cd[off + 46 : off + 46 + nlen].decode()
        out.append((name, comp, csize, usize, lho))
        off += 46 + nlen + elen + clen
    return out


def fetch_member(url, name, comp, csize, lho):
    # local file header is 30 bytes + name + extra; fetch header first (fixed 512 window)
    data, _ = http(url, {"Range": f"bytes={lho}-{lho + 29}"})
    nlen, elen = struct.unpack("<HH", data[26:30])
    dstart = lho + 30 + nlen + elen
    blob, _ = http(url, {"Range": f"bytes={dstart}-{dstart + csize - 1}"})
    if comp == 0:
        raw = blob
    elif comp == 8:
        raw = zlib.decompress(blob, -15)
    else:
        raise RuntimeError(f"comp method {comp}")
    return np.load(io.BytesIO(raw))


def zscore(x):
    m, s = np.nanmean(x), np.nanstd(x)
    return (x - m) / (s + 1e-9)


def main(a_idx, b_idx):
    for zi in range(a_idx, b_idx):
        a, b = zi * 2500, (zi + 1) * 2500
        outp = os.path.join(CACHE, f"pose_{a}_{b}.json")
        if os.path.exists(outp):
            print(f"skip {a}-{b} (cached)")
            continue
        url = f"{BASE}/pose_data_{a}_to_{b}.zip"
        cd = central_directory(url)
        cands = [c for c in cd if c[0].endswith("_10.npy")]
        rows = []
        for name, comp, csize, usize, lho in cands:
            try:
                arr = fetch_member(url, name, comp, csize, lho)
            except Exception as e:
                rows.append({"name": name, "err": str(e)})
                continue
            if arr.shape != (1200, 6):
                rows.append({"name": name, "err": f"shape {arr.shape}"})
                continue
            sp = np.linalg.norm(arr[:, :3].astype(np.float64), axis=1) * 20.0  # m/s
            mask = np.isfinite(sp) & np.isfinite(gt_speed)
            n = int(mask.sum())
            if n < 600:
                rows.append({"name": name, "err": f"only {n} finite"})
                continue
            r = float(np.corrcoef(zscore(sp[mask]), zscore(gt_speed[mask]))[0, 1])
            mad = float(np.nanmean(np.abs(sp[mask] - gt_speed[mask])))
            rows.append(
                {
                    "name": name,
                    "corr": round(r, 4),
                    "mean_abs_diff_ms": round(mad, 3),
                    "mean_speed": round(float(np.nanmean(sp)), 2),
                    "n_finite": n,
                }
            )
        with open(outp, "w") as f:
            json.dump(rows, f)
        ok = [r for r in rows if "corr" in r]
        best = max(ok, key=lambda r: r["corr"]) if ok else None
        print(f"zip {a}-{b}: {len(cands)} seg10 members, best corr = {best}")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
