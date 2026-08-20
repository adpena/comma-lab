#!/usr/bin/env python3
"""Re-bucket ddm_sd1's measured owed list under the refined rules, without a 27-minute re-scan.

The path-pattern rules are pure functions of the path, so they can be replayed offline against the
manifest the sweep already wrote. The one rule that touches the filesystem is the nested-`.git`
clone check; it is skipped here (cache pre-seeded False) because the sweep already applied it, and
re-walking 814 SSD paths costs minutes for zero new information. Skipping it can only UNDER-count
bucket A, i.e. it errs toward keeping something in the debt — the safe direction.
"""
import collections
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("a", REPO / "tools/audit_ssd_authored_signal.py")
m = importlib.util.module_from_spec(spec)
sys.modules["a"] = m
spec.loader.exec_module(m)

SD = Path(__file__).resolve().parent
owed = json.loads((SD / "SSD_AUTHORED_GAP_CURRENT.json").read_text())["owed"]

buckets = collections.Counter()
still = []


class _NoClone(dict):
    def __contains__(self, k):  # every lookup hits, always False -> no filesystem walk
        return True

    def __getitem__(self, k):
        return False


for r in owed:
    p = Path(r["representative_path"])
    root = Path("/Volumes/APDataStore/pact") if "APDataStore" in str(p) else Path("/Volumes/VertigoDataTier/pact")
    b = m.classify(p, root, _NoClone())
    r["rebucket"] = b
    buckets[b] += 1
    if b == "C":
        still.append(r)

lin = collections.Counter()
for r in still:
    mm = re.match(r"/Volumes/(\w+)/pact/([^/]+)/", r["representative_path"])
    lin[mm.group(2) if mm else "?"] += 1

out = {
    "rebucketed_from": len(owed),
    "counts": dict(buckets),
    "still_owed_blobs": len(still),
    "still_owed_bytes": sum(r["size_bytes"] for r in still),
    "by_ext": dict(collections.Counter(r["ext"] for r in still)),
    "by_lineage": dict(lin.most_common()),
    "rows": still,
}
(SD / "DRAIN_LIST.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1)[:2200])
