#!/usr/bin/env python3
"""Forensic readiness audit for apogee_int6.

This recovered tool used to advertise a dispatch one-liner from local
distortion-proxy evidence. That is now forbidden. Apogee intN candidates are
forensic-only until a real distortion model or exact CUDA evidence exists.

Usage:
    .venv/bin/python tools/dispatch_readiness_apogee_int6.py
    .venv/bin/python tools/dispatch_readiness_apogee_int6.py --json

NO GPU SPEND. Pure CPU-side validation. Tagged [distortion-proxy:local | sanity:5/5].
This command must return nonzero for dispatch readiness. The actual contest
score still requires upstream/evaluate.py on exact archive bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "experiments/results/apogee_int6_repack_20260504_claude/apogee_int6_archive.zip"
EXPECTED_SHA256 = "0176a2691a4daf5991170404d30a304ae30389621c0fc54914628414aef39ff1"
EXPECTED_SIZE_BYTES = 170450
EXPECTED_MAGIC_BYTE = 0xA6  # int6 marker per scripts/remote_lane_apogee_intN.sh
PR106_FRONTIER_SCORE = 0.20945673  # [contest-CUDA T4]
META_LAGRANGIAN_PREDICTED_LOW = 0.1499
META_LAGRANGIAN_PREDICTED_HIGH = 0.2499
REL_ERR_PCT = 1.55  # per repack_metadata.json


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_archive_integrity() -> tuple[bool, str]:
    if not ARCHIVE.is_file():
        return False, f"missing archive: {ARCHIVE}"
    actual_sha = _sha256(ARCHIVE)
    actual_size = ARCHIVE.stat().st_size
    if actual_sha != EXPECTED_SHA256:
        return False, f"sha mismatch: got {actual_sha[:16]}.. expected {EXPECTED_SHA256[:16]}.."
    if actual_size != EXPECTED_SIZE_BYTES:
        return False, f"size mismatch: got {actual_size} expected {EXPECTED_SIZE_BYTES}"
    with zipfile.ZipFile(ARCHIVE) as z:
        members = z.namelist()
        if members != ["0.bin"]:
            return False, f"unexpected members: {members} (expected ['0.bin'])"
        with z.open("0.bin") as f:
            head = f.read(1)
            if head[0] != EXPECTED_MAGIC_BYTE:
                return False, f"magic mismatch: got 0x{head[0]:02X} expected 0x{EXPECTED_MAGIC_BYTE:02X}"
    return True, f"sha={EXPECTED_SHA256[:16]}.. size={EXPECTED_SIZE_BYTES} magic=0xA6 single-member 0.bin"


def _check_sanity_ladder() -> tuple[bool, str]:
    """Run tools/predispatch_sanity.py with the meta-Lagrangian band."""
    cmd = [
        sys.executable, str(REPO / "tools/predispatch_sanity.py"),
        "--archive", str(ARCHIVE),
        "--predicted-low", str(META_LAGRANGIAN_PREDICTED_LOW),
        "--predicted-high", str(META_LAGRANGIAN_PREDICTED_HIGH),
        "--rel-err-pct", str(REL_ERR_PCT),
        "--lane-class", "apogee_intN",
        "--distortion-proxy-ran",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    except subprocess.TimeoutExpired:
        return False, "timeout (>30s)"
    if "ALL 5 GATES PASS" in result.stdout:
        return True, "5/5 gates pass"
    return False, f"sanity blocked: {result.stdout.strip().splitlines()[-1] if result.stdout else 'no output'}"


def _check_predicted_band_below_frontier() -> tuple[bool, str]:
    """Predicted bands are forensic only and cannot unlock dispatch."""

    return False, (
        "blocked: Apogee intN prediction lacks a valid distortion model; "
        "byte/proxy band is forensic-only and cannot authorize dispatch"
    )


def _check_preflight_clean() -> tuple[bool, str]:
    cmd = [sys.executable, "-c", "from tac.preflight import preflight_all; preflight_all(verbose=False)"]
    try:
        result = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=300, check=False)
    except subprocess.TimeoutExpired:
        return False, "timeout (>5min)"
    if result.returncode == 0:
        return True, "preflight_all() rc=0"
    return False, f"preflight failed rc={result.returncode}: {result.stderr[-200:].strip()}"


def _check_lane_registry_consistent() -> tuple[bool, str]:
    cmd = [sys.executable, str(REPO / "tools/lane_maturity.py"), "validate"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if result.returncode == 0:
        return True, "lane_maturity validate clean"
    return False, f"validate failed: {result.stderr[-200:].strip()}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    mandatory_checks = [
        ("archive_integrity", _check_archive_integrity),
        ("predicted_band_below_frontier", _check_predicted_band_below_frontier),
    ]
    expensive_checks = [
        ("sanity_ladder_5of5", _check_sanity_ladder),
        ("preflight_clean", _check_preflight_clean),
        ("lane_registry_consistent", _check_lane_registry_consistent),
    ]
    results = []
    for name, fn in mandatory_checks:
        ok, detail = fn()
        results.append({"name": name, "ok": ok, "detail": detail})

    distortion_gate_blocked = any(
        r["name"] == "predicted_band_below_frontier" and not r["ok"] for r in results
    )
    if distortion_gate_blocked:
        for name, _fn in expensive_checks:
            results.append(
                {
                    "name": name,
                    "ok": False,
                    "detail": (
                        "skipped: forensic-only Apogee intN has no valid distortion model; "
                        "expensive readiness checks cannot authorize dispatch"
                    ),
                }
            )
    else:
        for name, fn in expensive_checks:
            ok, detail = fn()
            results.append({"name": name, "ok": ok, "detail": detail})

    all_ok = all(r["ok"] for r in results)

    if args.json:
        print(json.dumps({
            "schema": "dispatch_readiness_apogee_int6_v1",
            "all_ok": all_ok,
            "checks": results,
            "expected_archive_sha256": EXPECTED_SHA256,
            "expected_size_bytes": EXPECTED_SIZE_BYTES,
            "predicted_band": [META_LAGRANGIAN_PREDICTED_LOW, META_LAGRANGIAN_PREDICTED_HIGH],
            "pr106_frontier_to_beat": PR106_FRONTIER_SCORE,
        }, indent=2))
        return 2 if all_ok else 1
    else:
        print("=== apogee_int6 dispatch readiness ===")
        for r in results:
            symbol = "PASS" if r["ok"] else "FAIL"
            print(f"  [{symbol}] {r['name']}: {r['detail']}")
        if all_ok:
            print()
            print("INTERNAL ERROR: forensic-only Apogee intN gate unexpectedly passed.")
            return 2
        else:
            print()
            failed = [r["name"] for r in results if not r["ok"]]
            print(f"BLOCKED — {len(failed)} check(s) failed: {failed}")
            print("Fix each violation before dispatching.")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
