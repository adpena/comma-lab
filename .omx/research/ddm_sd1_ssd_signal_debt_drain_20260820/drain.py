#!/usr/bin/env python3
"""ddm_sd1 leg 1 — copy the still-owed authored SSD blobs into a repo home, VERBATIM.

Byte identity is the whole point. Every file is copied with `read_bytes`/`write_bytes` and the
blob sha is re-verified after the write, because the guard's test is byte identity: a header
comment added "for provenance" would change the blob and leave the SSD copy still counted as
owed. Provenance therefore lives in a SIDECAR, never in the file.

Destination layout mirrors the origin so an operator can trace a recovered file back:
    experiments/ssd_recovered/<tier>/<lineage>/<path under lineage>
On a name collision between two DIFFERENT blobs, the shorter sha prefix is appended — the
alternative (last write wins) would silently drop one of them.
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

SD = Path(__file__).resolve().parent
REPO = SD.parents[2]
DEST = REPO / "experiments" / "ssd_recovered"


def blob_sha1(data: bytes) -> str:
    h = hashlib.sha1()
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def main() -> int:
    rows = json.loads((SD / "DRAIN_LIST.json").read_text())["rows"]
    DEST.mkdir(parents=True, exist_ok=True)

    manifest = []
    copied = missing = mismatch = 0
    used: dict[Path, str] = {}

    for r in rows:
        src = Path(r["representative_path"])
        try:
            data = src.read_bytes()
        except OSError as exc:
            manifest.append({**r, "status": "unreadable", "error": f"{type(exc).__name__}"})
            missing += 1
            continue
        sha = blob_sha1(data)
        if sha != r["blob_sha1"]:
            # The SSD copy changed since the sweep. Record it and take the CURRENT bytes: the
            # newer content is the live signal, and silently keeping the stale sha would make the
            # manifest lie about what was landed.
            mismatch += 1
        parts = src.parts
        try:
            i = parts.index("pact")
            tier = parts[i - 1]
            rel = Path(*parts[i + 1:])
        except (ValueError, IndexError):
            tier = "unknown_tier"
            rel = Path(src.name)
        out = DEST / tier / rel
        if out in used and used[out] != sha:
            out = out.with_name(f"{out.stem}.{sha[:8]}{out.suffix}")
        used[out] = sha
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        assert blob_sha1(out.read_bytes()) == sha, f"byte identity broken writing {out}"
        copied += 1
        manifest.append({
            "blob_sha1": sha,
            "sweep_blob_sha1": r["blob_sha1"],
            "bytes_changed_since_sweep": sha != r["blob_sha1"],
            "origin_path": str(src),
            "repo_path": str(out.relative_to(REPO)),
            "size_bytes": len(data),
            "ssd_instance_count": r["instance_count"],
            "ext": r["ext"],
            "durability_at_sweep": r["durability"],
            "status": "copied",
        })

    prov = {
        "arm": "ddm_sd1",
        "date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "what": "Authored sources recovered from the SSD tier, where they had no byte-identical "
                "blob reachable from any git ref. Copied VERBATIM; provenance is here, not in the "
                "files, so the blob identity that makes them count as preserved is unchanged.",
        "measured_by": "tools/audit_ssd_authored_signal.py",
        "sweep": "SSD_AUTHORED_GAP_CURRENT.json",
        "rebucketed_by": "reclassify.py",
        "counts": {"requested": len(rows), "copied": copied, "unreadable": missing,
                   "bytes_changed_since_sweep": mismatch},
        "files": manifest,
    }
    (DEST / "PROVENANCE.json").write_text(json.dumps(prov, indent=1))
    print(json.dumps(prov["counts"], indent=1))
    print(f"-> {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
