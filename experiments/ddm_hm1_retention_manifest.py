"""ddm_hm1 -- build the workspace retention manifest.

ALWAYS KEEP THE PAYLOAD is a precondition for running, not a report written afterwards,
but the manifest is what lets the next consumer prove byte-identity instead of re-running
a 19-minute HPAC forward pass.  Every retained artifact is hashed here with its byte
count, and the source instruments this arm consumed are recorded with theirs.

Axis: ``[macOS-CPU advisory / scorer-free byte measurement]``.  ``score_claim=false``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

CHUNK = 8 << 20

SOURCE_INSTRUMENTS = {
    "frontier_archive": Path(
        "/Volumes/APDataStore/pact/ddm_wc1_advisory_decode_wallclock_20260815"
        "/prepared/hv1_base_control/archive.zip"
    ),
    "decoded_tokens": Path(
        "/Volumes/APDataStore/pact/ddm_wc1_advisory_decode_wallclock_20260815"
        "/runs/base_optimized_n600_r3/output/.f26_decode_checkpoints"
        "/tokens_cpu_stage_complete.u8"
    ),
}


def file_record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def build(workspace: Path) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    blockers: list[str] = []
    for path in sorted(workspace.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        artifacts.append(file_record(path))

    sources: dict[str, Any] = {}
    for name, path in SOURCE_INSTRUMENTS.items():
        if path.is_file():
            sources[name] = file_record(path)
        else:
            blockers.append(f"source instrument missing: {name} at {path}")

    return {
        "schema": "ddm_hm1_retention_manifest.v1",
        "axis": "[macOS-CPU advisory / scorer-free byte measurement]",
        "score_claim": False,
        "promotable": False,
        "complete": not blockers,
        "blockers": blockers,
        "workspace": str(workspace),
        "source_instruments": sources,
        "artifact_count": len(artifacts),
        "artifact_bytes": sum(record["bytes"] for record in artifacts),
        "artifacts": artifacts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "workspace",
        type=Path,
        nargs="?",
        default=Path("/Volumes/APDataStore/pact/ddm_hm1_20260816"),
    )
    args = parser.parse_args(argv)
    manifest = build(args.workspace)
    destination = args.workspace / "RETENTION_MANIFEST.json"
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "complete",
                    "blockers",
                    "artifact_count",
                    "artifact_bytes",
                )
            },
            indent=2,
        )
    )
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
