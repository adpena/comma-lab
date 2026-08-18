"""ddm_sz1 -- bounded extension: does the byte-plane split pay on the OTHER sections?

MEASURE ONLY.  Nothing here ships this arm; the arm ships the proven semantic split.
pd1's corpus-wide prior says weight SERIALIZATION is unsaturated, so the question is
whether ``carrier_blob`` and ``hpac_blob`` carry the same un-entropy-coded fp16 metadata.

THE CONTROL GATE IS THE POINT.  A section is measurable only if re-Brotli of its raw
body reproduces the SHIPPED coded length at delta +0.  Without that the instrument is
not calibrated on that section and any delta it reports is unreadable -- it would be
measuring the parameter mismatch, not the transform.  Sections that fail the control
are reported as NOT MEASURABLE rather than given a number.

Run: .venv/bin/python experiments/ddm_sz1_extension_probe.py
"""

from __future__ import annotations

import json
import struct
import sys
from datetime import UTC, datetime
from pathlib import Path

import ddm_sz1_semantic_metadata_split as sz1

BASE_RUNTIME = Path("/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/candidate_runtime")
BASE_ARCHIVE = Path("/Volumes/APDataStore/pact/ddm_rr4_cuda_prob_reencode/retained/archive.zip")
OUT = Path("/Volumes/APDataStore/pact/ddm_sz1/probe/extension_other_sections.json")

BROTLI_QUALITY = 11
BROTLI_LGWIN = 24
SCAN_LIMIT = 400
"""Offsets 0..SCAN_LIMIT.  The split only pays where it overlaps an fp16 region, so a
bounded scan brackets the optimum; a full sweep would price coder noise, not mechanism."""


def main() -> int:
    sys.path.insert(0, str(BASE_RUNTIME))
    import brotli
    import numpy as np
    from runtime.residual_archive import _decompress_brotli, read_residual_archive

    parts = read_residual_archive(BASE_ARCHIVE)
    container = bytes(parts.compressed_models)
    header = struct.unpack_from("<4sBBBBHHH", container)
    hpac_n, semantic_n, carrier_n = header[5], header[6], header[7]
    body = container[sz1.RX1_MODEL_HEADER.size :]
    sections = {
        "hpac_blob": body[:hpac_n],
        "semantic_blob": body[hpac_n : hpac_n + semantic_n],
        "carrier_blob": body[hpac_n + semantic_n : hpac_n + semantic_n + carrier_n],
    }

    def coded(payload: bytes) -> int:
        return len(brotli.compress(payload, quality=BROTLI_QUALITY, lgwin=BROTLI_LGWIN))

    def split_at(raw: bytes, offset: int, length: int) -> bytes:
        length -= length % 2
        region = np.frombuffer(raw[offset : offset + length], dtype=np.uint8)
        planes = np.empty(length, dtype=np.uint8)
        planes[: length // 2] = region[1::2]
        planes[length // 2 :] = region[0::2]
        return raw[:offset] + planes.tobytes() + raw[offset + length :]

    rows = []
    for name, stream in sections.items():
        raw = _decompress_brotli(stream)
        control = coded(raw)
        row = {
            "section": name,
            "shipped_coded_bytes": len(stream),
            "raw_bytes": len(raw),
            "control_rebrotli_bytes": control,
            "control_delta": control - len(stream),
            "control_reproduces_shipped": control == len(stream),
        }
        if control != len(stream):
            row["verdict"] = "NOT MEASURABLE"
            row["reason"] = (
                "re-Brotli does not reproduce the shipped coded length, so the "
                "instrument is not calibrated on this section; any split delta here "
                "would be measuring the parameter mismatch, not the transform"
            )
        else:
            best = min(
                (
                    (coded(split_at(raw, offset, len(raw) - offset if name != "semantic_blob"
                                    else sz1.F12_FIXED_METADATA_BYTES)) - len(stream), offset)
                    for offset in range(0, min(SCAN_LIMIT, len(raw) - 2) + 1)
                ),
                default=(0, 0),
            )
            row["best_split_delta"] = best[0]
            row["best_split_offset"] = best[1]
            row["verdict"] = "PAYS" if best[0] < 0 else "NO WIN"
        rows.append(row)
        print(json.dumps(row, indent=2))

    document = {
        "schema": "ddm_sz1_extension_probe.v1",
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "authority": "[byte-level] exact bytes through the real Brotli; no entropy estimates",
        "disposition": "MEASURE ONLY -- nothing here ships this arm",
        "brotli": {"quality": BROTLI_QUALITY, "lgwin": BROTLI_LGWIN},
        "sections": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(document, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
