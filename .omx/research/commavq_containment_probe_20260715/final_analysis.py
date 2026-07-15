"""Final containment analysis over all cached seg-10 pose sweep results."""
import glob
import json

import numpy as np

rows = []
errs = 0
for p in sorted(glob.glob("/Users/adpena/Projects/pact/.omx/tmp/commavq_probe/seg10_cache/*.json")):
    for r in json.load(open(p)):
        if "corr" in r:
            rows.append(r)
        else:
            errs += 1
print(f"zips: {len(glob.glob('/Users/adpena/Projects/pact/.omx/tmp/commavq_probe/seg10_cache/*.json'))}  scored: {len(rows)}  errors/skipped: {errs}")

corrs = np.array([r["corr"] for r in rows])
# mean_speed stored under the (wrong) x20 assumption; raw col-norm m/s = mean_speed/20
raw_speed = np.array([r["mean_speed"] / 20.0 for r in rows])
print(f"corr: max {corrs.max():.4f}  p99 {np.percentile(corrs,99):.4f}  median {np.median(corrs):.4f}")

# GT band: 31.75 +/- (30.75..33.88) m/s
band = (raw_speed >= 29.5) & (raw_speed <= 35.0)
print(f"segments with raw mean speed in GT band [29.5,35.0] m/s: {band.sum()}")
if band.any():
    top_band = max((r for r, b in zip(rows, band) if b), key=lambda r: r["corr"])
    print("best in-band candidate:", top_band, "raw_speed", round(top_band["mean_speed"] / 20.0, 2))
rows.sort(key=lambda r: -r["corr"])
print("top5 overall:")
for r in rows[:5]:
    print("  ", r["name"], "corr", r["corr"], "raw_speed_ms", round(r["mean_speed"] / 20.0, 2))
n_pass = int(((corrs >= 0.99)).sum())
print(f"candidates passing corr>=0.99: {n_pass}")
joint = int((band & (corrs >= 0.95)).sum())
print(f"candidates passing joint (in-band AND corr>=0.95): {joint}")
