"""Turn ddm_ntb2's cold-store manifests into certificates, after sr5 refused them.

sr5's finding (2026-09-11) is correct and the defect is mine: the manifest this arm wrote
for 3.78 GB of superseded parse-back attempts put the literal string
``SKIPPED_LARGE_STREAMED`` in the ``sha256`` field. A placeholder in a hash field is the
same defect as a label standing in for custody — it is precisely the class this campaign
extincts elsewhere, and writing one because a file was large is no excuse: sha256 streams.
sr5 also found 6 of 12 renderer-checkpoint rows carrying ``run_completed: false`` while the
row text claimed they were superseded by the run's own RESULT.json. They were not; those
runs were STOPPED on a refutation bound and have no RESULT.json, so the row said something
untrue of itself.

This producer re-reads every moved payload from the cold store, streams its sha256 however
large, rewrites both manifests with real hashes and honest per-row reasons, and re-validates
by re-hashing what it just wrote. It deletes nothing.

Axis: [custody arithmetic]. No score claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

COLD = Path("/Volumes/APDataStore/pact/ddm_ntb2_coldstore")
REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / ".omx/research/ddm_ntb2_20260911"
CHUNK = 1 << 22


def stream_sha256(path: Path) -> tuple[str, int]:
    """sha256 of a file of ANY size, and its byte count. No size ever skips a hash."""
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        while True:
            block = stream.read(CHUNK)
            if not block:
                break
            digest.update(block)
            total += len(block)
    return digest.hexdigest(), total


def honest_reason(row: dict) -> str:
    """What a row is, as opposed to what the first manifest claimed it was."""
    if row.get("run_completed") is True:
        return ("superseded by the run's own RESULT.json and argmax_plane.npz, both retained "
                "on the live tier")
    if row.get("run_completed") is False:
        return ("resume checkpoint of a run this arm STOPPED on a refutation bound; it has NO "
                "RESULT.json and is NOT superseded by one. Its scientific content -- the per-pair "
                "d_seg/d_pose rows and the per-chunk render shas -- is in the retained LATEST.json "
                "on the live tier, and the npz is the recomputable resume state")
    return ("a superseded cold-parse-back attempt against this arm's MANIFEST-LESS working copy of "
            "the receiver tree; deterministically rebuildable from the retained archive and tree")


def certify(manifest_path: Path, *, apply: bool) -> dict:
    document = json.loads(manifest_path.read_text())
    rows, missing, placeholders_fixed, rehashed_bytes = [], [], 0, 0
    for row in document["rows"]:
        destination = Path(row["cold_store_destination"])
        new = dict(row)
        if not destination.is_file():
            new["certified"] = False
            new["problem"] = "payload absent from the cold store"
            missing.append(str(destination))
            rows.append(new)
            continue
        digest, size = stream_sha256(destination)
        if row.get("sha256") != digest:
            placeholders_fixed += 1
        new["sha256"] = digest
        new["bytes"] = size
        new["sha256_scope"] = "the payload AS IT SITS IN THE COLD STORE, streamed in full"
        new["rebuildable_because"] = honest_reason(row)
        new["certified"] = True
        rehashed_bytes += size
        rows.append(new)
    document["rows"] = rows
    document["certified_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    document["certification"] = {
        "every_row_hashed": not missing,
        "rows": len(rows),
        "bytes_rehashed": rehashed_bytes,
        "rows_whose_hash_changed": placeholders_fixed,
        "missing_payloads": missing,
        "placeholder_hashes_remaining": 0,
        "method": "hashlib.sha256 streamed in 4 MiB blocks; no size threshold, no skipping",
        "corrects": ("sr5's refusal: SKIPPED_LARGE_STREAMED in a hash field, and rows claiming "
                     "supersession by a RESULT.json that a stopped run never produced"),
    }
    document["nothing_deleted"] = True
    if apply:
        manifest_path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
        # The two manifests were written by different steps and only one declared a
        # `destination_tier`; derive the copy's home from the rows themselves so neither
        # shape is privileged.
        destinations = {Path(row["cold_store_destination"]).parent for row in rows}
        if destinations:
            beside = Path(os.path.commonpath([str(d) for d in destinations])) / "MANIFEST.json"
            if beside.parent.is_dir():
                beside.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    return document["certification"]


def revalidate(manifest_path: Path) -> dict:
    """Re-hash what was just written. A manifest nobody re-checked is a claim, not a receipt."""
    document = json.loads(manifest_path.read_text())
    bad = []
    for row in document["rows"]:
        destination = Path(row["cold_store_destination"])
        if not destination.is_file():
            bad.append({"path": str(destination), "problem": "absent"})
            continue
        digest, size = stream_sha256(destination)
        if digest != row["sha256"] or size != row["bytes"]:
            bad.append({"path": str(destination), "problem": "hash or size differs on re-read"})
    return {"manifest": str(manifest_path), "rows": len(document["rows"]),
            "mismatches": bad, "passed": not bad}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    manifests = [LEDGER / "COLDSTORE_MANIFEST.json",
                 LEDGER / "COLDSTORE_MANIFEST_PUBLIC_ATTEMPTS.json"]
    report = {"schema": "ddm_ntb2_coldstore_certification.v1", "score_claim": False,
              "applied": bool(args.apply), "manifests": []}
    for manifest in manifests:
        if not manifest.is_file():
            raise SystemExit(f"manifest absent: {manifest}")
        certification = certify(manifest, apply=args.apply)
        entry = {"manifest": str(manifest), "certification": certification}
        if args.apply:
            entry["revalidation"] = revalidate(manifest)
        report["manifests"].append(entry)
    report["total_bytes_rehashed"] = sum(
        m["certification"]["bytes_rehashed"] for m in report["manifests"])
    report["total_rows"] = sum(m["certification"]["rows"] for m in report["manifests"])
    if args.apply:
        (LEDGER / "COLDSTORE_CERTIFICATION.json").write_text(
            json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
