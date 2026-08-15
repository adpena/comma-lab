#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Retain the DDM-MP2 Stage-2 exact carrier byte race.

This is the scorer-free first measurement required before any lossy rank/refit
candidate.  It races every Python Brotli quality against the exact 22,219-byte
HV1 physical carrier body, retains every compressed payload, and proves exact
decode equality.  It does not claim to implement or measure rank reduction.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import brotli

DEFAULT_SOURCE = Path(
    "/Volumes/APDataStore/pact/ddm_mp2_mixed_precision_receiver_close_20260815/"
    "generations/hv1_base_control/retained/carrier.raw.bin"
)
DEFAULT_INCUMBENT = Path(
    "/Volumes/APDataStore/pact/ddm_mp2_mixed_precision_receiver_close_20260815/"
    "generations/hv1_base_control/retained/carrier.br"
)


class CarrierRaceRefusal(RuntimeError):
    """Fail-closed refusal for a non-exact or drifted carrier race."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": _sha256_bytes(payload)}


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def _retain_exact(path: Path, payload: bytes) -> str:
    if path.exists():
        if path.read_bytes() != payload:
            raise CarrierRaceRefusal(f"retained payload differs on resume: {path}")
        return "RESUMED_EXACT"
    _atomic_bytes(path, payload)
    return "MATERIALIZED"


def run_race(source: Path, incumbent: Path, output: Path) -> dict[str, Any]:
    source = source.resolve()
    incumbent = incumbent.resolve()
    output = output.resolve()
    raw = source.read_bytes()
    incumbent_payload = incumbent.read_bytes()
    if len(raw) != 22_219:
        raise CarrierRaceRefusal(f"expected the 22,219-byte HV1 carrier body, got {len(raw)}")
    if brotli.decompress(incumbent_payload) != raw:
        raise CarrierRaceRefusal("incumbent carrier stream does not decode to the source body")

    payload_root = output / "retained" / "exact_brotli_quality"
    rows: list[dict[str, Any]] = []
    for quality in range(12):
        payload = brotli.compress(raw, quality=quality)
        if brotli.decompress(payload) != raw:
            raise CarrierRaceRefusal(f"Brotli quality {quality} failed exact decode equality")
        path = payload_root / f"carrier_q{quality:02d}.br"
        repeat_path = payload_root / f"carrier_q{quality:02d}.repeat.br"
        payload_status = _retain_exact(path, payload)
        repeat_status = _retain_exact(
            repeat_path, brotli.compress(raw, quality=quality)
        )
        if path.read_bytes() != repeat_path.read_bytes():
            raise CarrierRaceRefusal(f"Brotli quality {quality} is not deterministic")
        rows.append(
            {
                "quality": quality,
                "payload": _file_record(path),
                "repeat": _file_record(repeat_path),
                "decode_exact": True,
                "repeat_byte_identical": True,
                "payload_status": payload_status,
                "repeat_status": repeat_status,
                "delta_bytes_vs_incumbent": len(payload) - len(incumbent_payload),
            }
        )

    best_bytes = min(int(row["payload"]["bytes"]) for row in rows)
    winners = [int(row["quality"]) for row in rows if row["payload"]["bytes"] == best_bytes]
    best_delta = best_bytes - len(incumbent_payload)
    if best_delta < 0:
        verdict = "BEATS_INCUMBENT"
    elif best_delta == 0:
        verdict = "TIE_INCUMBENT"
    else:
        verdict = "LOSES_INCUMBENT"
    result = {
        "schema": "ddm_mp2_carrier_exact_byte_race.v1",
        "generated_utc": dt.datetime.now(tz=dt.UTC).isoformat(),
        "tool": _file_record(Path(__file__).resolve()),
        "brotli_version": getattr(brotli, "__version__", "unknown"),
        "scope": "exact same-decoded-physical-carrier Brotli quality race",
        "source": _file_record(source),
        "incumbent": _file_record(incumbent),
        "candidate_denominator": 12,
        "candidate_complete": 12,
        "all_payloads_retained": True,
        "all_decode_exact": all(bool(row["decode_exact"]) for row in rows),
        "rows": rows,
        "best": {
            "bytes": best_bytes,
            "qualities": winners,
            "delta_bytes_vs_incumbent": best_delta,
            "verdict": verdict,
        },
        "rank_refit_status": "NOT_MEASURED_BY_THIS_EXACT_RACE",
        "score_claim": False,
        "verdict_scope": "INSTANCE: exact Brotli quality on the fixed HV1 carrier body",
    }
    _atomic_json(output / "CARRIER_EXACT_BYTE_RACE.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--incumbent", type=Path, default=DEFAULT_INCUMBENT)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_race(args.source, args.incumbent, args.output)
    except (OSError, brotli.error, CarrierRaceRefusal) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
