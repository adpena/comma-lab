#!/usr/bin/env python
"""Parse a wr1 realized-gate report into its receipt — the harvest step that was never written.

WHY THIS EXISTS (task #870, the orphaned-cheap-follow-on class).
``experiments/stage_wr1_realized_gate.sh`` ended with four ``echo`` lines that DESCRIBED parsing
``report.txt`` into ``wr1_<cand>_realized_gate_receipt.json``. They printed the instruction and
wrote nothing. Both candidates were then actually fired -- kneeA (2026-07-29, 9m41s) and kneeB
(2026-07-30, 11m47s), roughly 22 minutes of real n600 scorer time -- and NEITHER produced a
receipt. The measurements survived only as stdout logs on the SSD.

That is the class in its most expensive form: not a follow-on nobody ran, but one that RAN and
whose answer never reached a machine-readable surface. The schema
(``wr1_realized_gate_receipt_SCHEMA.json``) had been written; only the writer was missing.

NO-FAKE: this recomputes S from the report's OWN components and refuses on mismatch. It runs zero
scorer forwards -- it reads a report that a real ``evaluate.py`` n600 run already produced -- so it
is ``$0`` and scorer-free. Axis ``[macOS-CPU advisory]``; ``score_claim=false``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = "ddm_wr1_realized_gate.v1"
UNCOMPRESSED = 37_545_489

# The pfs1 D1 reference row this gate is apples-to-apples with (ddm_wr1 memo §0, MEASURED).
REF = {"d_seg": 0.00389011, "d_pose": 0.22144216, "bytes": 569_996, "S": 2.256641}

_PATTERNS = {
    "d_pose": re.compile(r"Average PoseNet Distortion:\s*([0-9.]+)"),
    "d_seg": re.compile(r"Average SegNet Distortion:\s*([0-9.]+)"),
    "bytes": re.compile(r"Submission file size:\s*([0-9,]+)"),
}


def score(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    """The contest objective, recomputed from components — never read from a rounded field."""
    return 100.0 * d_seg + math.sqrt(10.0 * d_pose) + 25.0 * archive_bytes / UNCOMPRESSED


def parse_report(text: str) -> dict:
    """Pull the three scored components out of an evaluate.py report or stdout log.

    Takes the LAST occurrence: a stdout log may contain a failed first attempt (the kneeA log
    holds one, killed by ``python: command not found``) followed by the real run.
    """
    out: dict[str, float | int] = {}
    for key, rx in _PATTERNS.items():
        hits = rx.findall(text)
        if not hits:
            raise ValueError(f"report does not contain {key!r} — refusing to invent it")
        raw = hits[-1].replace(",", "")
        out[key] = int(raw) if key == "bytes" else float(raw)
    return out


def build_receipt(candidate: str, source: Path, text: str) -> dict:
    comp = parse_report(text)
    d_seg, d_pose, nbytes = comp["d_seg"], comp["d_pose"], int(comp["bytes"])
    s = score(d_seg, d_pose, nbytes)

    stated = re.findall(r"Final score:[^=]*=\s*([0-9.]+)", text)
    if stated and abs(float(stated[-1]) - s) > 0.005:
        raise ValueError(
            f"recomputed S={s:.6f} disagrees with the report's stated {stated[-1]} — "
            "refusing to emit a receipt whose number the source does not support"
        )

    delta_bytes = nbytes - REF["bytes"]
    delta_s = s - REF["S"]
    # Preregistered break-even (wr1 §5 / gc6 row 6): accept iff realized ΔS < 25·ΔB/37,545,489.
    break_even = 25.0 * delta_bytes / UNCOMPRESSED
    return {
        "schema": SCHEMA,
        "candidate": candidate,
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "source_log": str(source),
        "source_sha256": hashlib.sha256(text.encode("utf-8", "replace")).hexdigest(),
        "harvested_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "harvested_by": "ddm_fo1 (task #870) — retro-harvest of an already-fired gate",
        "realized": {
            "d_seg": d_seg,
            "d_pose": d_pose,
            "archive_bytes": nbytes,
            "S": s,
            "seg_term": 100.0 * d_seg,
            "pose_term": math.sqrt(10.0 * d_pose),
            "rate_term": 25.0 * nbytes / UNCOMPRESSED,
        },
        "reference_pfs1_d1": REF,
        "delta_vs_reference": {
            "d_seg": d_seg - REF["d_seg"],
            "d_pose": d_pose - REF["d_pose"],
            "archive_bytes": delta_bytes,
            "S": delta_s,
        },
        "break_even_threshold_S": break_even,
        "verdict": "ACCEPT" if delta_s < break_even else "REJECT",
        "verdict_scope": "INSTANCE",
        "verdict_note": (
            "INSTANCE-scoped: this refutes THIS candidate's predicted-zero-cost claim on THIS "
            "reference row. It does not close the reverse-waterfill family."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidate", required=True, help="kneeA | kneeB (or any tag)")
    ap.add_argument("--report", required=True, type=Path,
                    help="evaluate.py report.txt OR the gate stdout log")
    ap.add_argument("--out", required=True, type=Path, help="receipt JSON to write")
    ap.add_argument("--print-only", action="store_true", help="do not write, just show")
    a = ap.parse_args(argv)

    if not a.report.exists():
        print(f"missing report: {a.report}", file=sys.stderr)
        return 2
    receipt = build_receipt(a.candidate, a.report,
                            a.report.read_text(encoding="utf-8", errors="replace"))
    blob = json.dumps(receipt, indent=2, sort_keys=True)
    if a.print_only:
        print(blob)
        return 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(a.out.suffix + ".tmp")
    tmp.write_text(blob + "\n", encoding="utf-8")
    tmp.replace(a.out)  # atomic: tmp+rename, per the deterministic-repro discipline
    r = receipt["realized"]
    print(f"[harvest] {a.candidate}: d_seg {r['d_seg']:.8f}  d_pose {r['d_pose']:.8f}  "
          f"{r['archive_bytes']:,} B  S {r['S']:.6f}  "
          f"ΔS {receipt['delta_vs_reference']['S']:+.6f}  -> {receipt['verdict']}")
    print(f"[harvest] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
