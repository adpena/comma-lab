#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stream 1 manifest builder — 4 substrate macOS-CPU smoke verdicts.

Operator directive 2026-05-13 LOCAL HARDWARE MAXIMIZATION SWEEP Stream 1.

For each of 4 just-landed substrates (SABOR, S2SBS, A1+wavelet, A1+LAPose),
this tool reads the smoke stdout/checkpoint/metadata + recipe predicted_band
and emits a row to the macOS-CPU advisory manifest. Smoke verdicts are
WIRING-VERIFIED only — they do NOT produce a real-archive auth-eval score
(smokes use synthetic data + skip auth-eval). Score-claim is permanently
False; ranking_only=True.

Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192 macOS-CPU advisory.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

LANE_ID = "lane_local_hardware_maximization_sweep_20260513"

SUBSTRATES = [
    {
        "name": "sabor_boundary_only_renderer",
        "predicted_band": (0.165, 0.185),
        "predicted_score_target": 0.175,
        "smoke_dir": "stream1_sabor",
        "ckpt_or_meta": "smoke_checkpoint.pt",
    },
    {
        "name": "s2sbs_byte_stuffing",
        "predicted_band": (0.168, 0.188),
        "predicted_score_target": 0.178,
        "smoke_dir": "stream1_s2sbs",
        "ckpt_or_meta": "manifest.json",
    },
    {
        "name": "a1_plus_wavelet_residual",
        "predicted_band": (0.187, 0.194),
        "predicted_score_target": 0.191,
        "smoke_dir": "stream1_a1plus_wavelet",
        "ckpt_or_meta": "smoke_metadata.json",
    },
    {
        "name": "a1_plus_lapose",
        "predicted_band": (0.185, 0.195),
        "predicted_score_target": 0.189,
        "smoke_dir": "stream1_a1plus_lapose",
        "ckpt_or_meta": "smoke_metadata.json",
    },
]


def _extract_archive_bytes_from_log(log_path: Path) -> int | None:
    if not log_path.is_file():
        return None
    txt = log_path.read_text(errors="ignore")
    # Look for "archive_bytes" or "archive bytes:" or "composition: N B"
    for pattern in (
        r"archive_bytes[=:\s]+(\d+)",
        r"archive\s+bytes[:\s]+(\d+)",
        r"composition[:\s]+(\d+)\s*B",
    ):
        m = re.search(pattern, txt)
        if m:
            return int(m.group(1))
    return None


def _utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def main(out_dir: str) -> int:
    out_root = Path(out_dir).resolve()
    rows = []
    for sub in SUBSTRATES:
        d = out_root / sub["smoke_dir"]
        log = d / "smoke_stdout.log"
        archive_bytes = _extract_archive_bytes_from_log(log)
        smoke_ok = (
            log.is_file() and "wrote" in log.read_text(errors="ignore").lower()
        ) or archive_bytes is not None
        verdict = (
            "wiring_verified_substrate_smoke_only"
            if smoke_ok
            else "substrate_smoke_failed_check_log"
        )
        rows.append({
            "schema": "macos_cpu_advisory_substrate_smoke_v1",
            "lane_id": LANE_ID,
            "substrate_name": sub["name"],
            "smoke_dir": str(d.relative_to(REPO_ROOT)),
            "smoke_verdict": verdict,
            "smoke_archive_bytes": archive_bytes,
            "predicted_band": list(sub["predicted_band"]),
            "predicted_score_target": sub["predicted_score_target"],
            "macos_cpu_total_score": None,
            "reason_no_full_eval": (
                "smoke uses synthetic data and --skip-auth-eval; "
                "full macOS-CPU eval requires real-archive train (~hrs CPU) "
                "or paired GPU train + macOS-CPU advisory eval (~9.5 min)."
            ),
            "evidence_grade": "macOS-CPU-advisory",
            "evidence_tag": "[macOS-CPU advisory only]",
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "ranking_only": True,
            "platform_node": platform.node(),
            "platform_release": platform.release(),
            "platform_machine": platform.machine(),
            "stamped_at_utc": _utc_stamp(),
        })

    manifest_path = out_root / "stream1_substrate_smoke_manifest.jsonl"
    with manifest_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary_path = out_root / "stream1_substrate_smoke_summary.json"
    summary = {
        "lane_id": LANE_ID,
        "stamped_at_utc": _utc_stamp(),
        "num_substrates": len(rows),
        "all_wiring_verified": all(
            r["smoke_verdict"] == "wiring_verified_substrate_smoke_only" for r in rows
        ),
        "rows": rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {manifest_path}")
    print(f"wrote {summary_path}")
    print(f"verdict: all_wiring_verified={summary['all_wiring_verified']}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: build_stream1_substrate_smoke_manifest.py <out_dir>")
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
