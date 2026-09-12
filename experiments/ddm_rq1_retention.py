"""ddm_rq1: build the arm's retention manifest -- every payload, with its sha256.

ALWAYS KEEP THE PAYLOAD.  This walks the arm's store, records ``path``/``bytes``/
``sha256`` for every payload it holds, and refuses to report a cap breach as a warning:
if the store is over the declared cap the manifest still lists everything and the
verdict says so, because the cure for "too big" is an explicit routing decision, never
a silent discard.

This arm's payloads are small and all three encode stages of every priced realization
are kept (``body.sm3r`` / ``rider.sm1s`` / ``member.rx1m``), so a successor can price,
parse back, or splice any variant without re-encoding it.  Same shape as
``ddm_ren2_retention``; forked only on the store path and the lane id, which are the
two things that must differ.

No score claim, no promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

STORE = Path("/Volumes/VertigoDataTier/pact/ddm_rq1")
CAP_GIB = 8
RESERVE_BYTES = 40 << 30

SKIP_SUFFIXES = (".tmp",)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build(args) -> int:
    payloads: list[dict[str, Any]] = []
    total = 0
    for path in sorted(STORE.rglob("*")):
        if not path.is_file() or path.name.endswith(SKIP_SUFFIXES):
            continue
        size = path.stat().st_size
        total += size
        payloads.append({"path": str(path), "bytes": size, "sha256": sha256_file(path)})
    free = shutil.disk_usage(STORE.parent).free
    manifest = {
        "arm": "ddm_rq1",
        "lane": "ddm_rq1_renderer_coarsening_then_field_reabsorption_20260912",
        "axis": "[macOS-CPU advisory, jg1 instrument, DALI GT lineage, byte-exact encoder]",
        "score_claim": False,
        "promotion_eligible": False,
        "cap_gib": CAP_GIB,
        "cleanup": (
            "KEEP ALL; every payload immutable on the SSD tier; local disk is not a "
            "storage tier and holds source only. The ~1.8 GB camera rasters are never "
            "materialised to disk by this arm: the seg leg renders in 10-pair chunks "
            "and scores them in place, so there is no raster payload to keep."
        ),
        "payload_count": len(payloads),
        "total_bytes": total,
        "total_gib": total / (1 << 30),
        "within_cap": total <= CAP_GIB * (1 << 30),
        "free_bytes_after": free,
        "reserve_bytes": RESERVE_BYTES,
        "reserve_respected": free >= RESERVE_BYTES,
        "payloads": payloads,
    }
    out = STORE / "RETENTION.json"
    payload = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode()
    tmp = out.with_suffix(".json.tmp")
    tmp.write_bytes(payload)
    tmp.replace(out)
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "payload_count",
                    "total_bytes",
                    "total_gib",
                    "within_cap",
                    "free_bytes_after",
                    "reserve_respected",
                )
            },
            indent=1,
        )
    )
    return 0 if manifest["within_cap"] and manifest["reserve_respected"] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args(argv)
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
