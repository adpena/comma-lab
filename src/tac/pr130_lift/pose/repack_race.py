"""Lossless CPR1 repack race for PR130-style pose carrier payloads.

This module deliberately distinguishes a PR130 legacy carrier from other pose
sections in our banked archives.  CPR1 is only admitted when the input has the
exact PR130 12-D carrier layout; field-adjacent pose data is reported as
incompatible instead of being silently recoded under the wrong name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tac.optimization.ddm_ix2_archive_container import (
    CODER_NAMES,
    code_block,
    decode_block,
    parse_payload,
)
from tac.pr130_lift.pose.source_loader import load_lifted_module

N = 600
CARRIER_DIM = 12
CARRIER_H = 24
CARRIER_W = 32
BASIS_BITS = 5
BASIS_COUNT = CARRIER_DIM * 3 * CARRIER_H * CARRIER_W
COEFFICIENT_COUNT = N * CARRIER_DIM
LEGACY_CARRIER_BYTES = (
    2 * CARRIER_DIM * 4
    + BASIS_COUNT * BASIS_BITS // 8
    + ((COEFFICIENT_COUNT + 1) // 2) * 3
)

DEFAULT_OUTPUT_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_mx2_20260806/repack_race")
DEFAULT_REPORT = DEFAULT_OUTPUT_ROOT / "repack_race.json"


@dataclass(frozen=True)
class Candidate:
    label: str
    source: str
    payload: bytes
    incumbent_bytes: int
    format: str
    notes: str


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def best_generic_code(payload: bytes) -> dict[str, Any]:
    coder_id, coded = code_block(payload)
    decoded = decode_block(coder_id, coded)
    if decoded != payload:
        raise AssertionError("generic coder failed exact round-trip")
    return {
        "coder": CODER_NAMES[coder_id],
        "bytes": len(coded),
        "sha256": sha256_bytes(coded),
        "roundtrip": True,
    }


def try_cpr1_legacy_carrier(payload: bytes) -> dict[str, Any]:
    """Attempt CPR1 only for the exact PR130 legacy-carrier byte layout."""

    if len(payload) != LEGACY_CARRIER_BYTES:
        return {
            "status": "NOT_COMPATIBLE",
            "reason": (
                f"legacy carrier length mismatch: {len(payload)} "
                f"!= {LEGACY_CARRIER_BYTES}"
            ),
            "roundtrip": False,
        }

    repack_carrier = load_lifted_module("repack_carrier")
    carrier_codec = load_lifted_module("carrier_codec")
    basis_scales, basis_codes, coefficient_scales, encoded_coefficients = (
        repack_carrier.decode_legacy_carrier(payload)
    )
    compact = carrier_codec.encode_compact_carrier(
        basis_scales,
        basis_codes,
        coefficient_scales,
        encoded_coefficients,
    )
    decoded = carrier_codec.decode_compact_carrier(
        compact,
        BASIS_COUNT,
        N,
        CARRIER_DIM,
    )
    expected = (
        basis_scales,
        basis_codes,
        coefficient_scales,
        encoded_coefficients,
    )
    for actual, wanted in zip(decoded, expected, strict=True):
        if not np.array_equal(actual, wanted):
            raise AssertionError("CPR1 carrier round-trip changed a symbol")
    return {
        "status": "APPLIED",
        "bytes": len(compact),
        "sha256": sha256_bytes(compact),
        "roundtrip": True,
        "basis_symbols": BASIS_COUNT,
        "coefficient_symbols": COEFFICIENT_COUNT,
    }


def _read_single_member_zip(path: Path, *, member: str | None = None) -> tuple[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if member is None:
            if len(infos) != 1:
                raise ValueError(f"{path} has {len(infos)} members, not one")
            info = infos[0]
        else:
            info = archive.getinfo(member)
        return info.filename, archive.read(info.filename)


def ix2_candidates(path: Path, *, section_names: tuple[str, ...]) -> list[Candidate]:
    member, payload = _read_single_member_zip(path)
    bulk, sections = parse_payload(payload)
    names = ("bulk", *section_names)
    if len(names) != 1 + len(sections):
        raise ValueError(
            f"{path} yielded {len(sections)} joint sections but {len(section_names)} names"
        )
    out = [
        Candidate(
            label=f"{path.name}:{member}:bulk",
            source=str(path),
            payload=bulk,
            incumbent_bytes=len(bulk),
            format="IX2 bulk",
            notes="token/frame bulk; not a PR130 legacy pose carrier",
        )
    ]
    for name, section in zip(section_names, sections, strict=True):
        out.append(
            Candidate(
                label=f"{path.name}:{member}:{name}",
                source=str(path),
                payload=section,
                incumbent_bytes=len(section),
                format="IX2 section",
                notes="joint IX2 section; CPR1 allowed only if legacy-carrier-shaped",
            )
        )
    return out


def pfs1_candidates(path: Path) -> list[Candidate]:
    out: list[Candidate] = []
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.filename.startswith("state/") and (
                "pose" in info.filename or info.filename.endswith(".sec")
            ):
                payload = archive.read(info.filename)
                out.append(
                    Candidate(
                        label=f"{path.name}:{info.filename}",
                        source=str(path),
                        payload=payload,
                        incumbent_bytes=len(payload),
                        format="PFS1 member",
                        notes="banked PFS1 pose/member field, not PR130 legacy carrier",
                    )
                )
    return out


def sc1_ep_candidate(chunks_dir: Path) -> Candidate | None:
    paths = sorted(chunks_dir.glob("chunk_*.npz"))
    if not paths:
        return None
    residuals: list[np.ndarray] = []
    for path in paths:
        data = np.load(path)
        residuals.append(
            np.asarray(data["bp"], dtype=np.float64)
            - np.asarray(data["tp_local"], dtype=np.float64)
        )
    ep = np.concatenate(residuals, axis=0)
    if ep.shape != (N, 6):
        raise ValueError(f"SC1 e_p residual shape differs: {ep.shape} != {(N, 6)}")
    payload = np.ascontiguousarray(ep.astype("<f4")).tobytes()
    return Candidate(
        label="ddm_sc1_20260728:ep_chunks:e_p_float32",
        source=str(chunks_dir),
        payload=payload,
        incumbent_bytes=len(payload),
        format="SC1 residual array",
        notes=(
            "reconstructed e_p float32 residual array; receipt proxy bytes are separate "
            "and this is not PR130 legacy carrier layout"
        ),
    )


def race_candidate(candidate: Candidate) -> dict[str, Any]:
    generic = best_generic_code(candidate.payload)
    cpr1 = try_cpr1_legacy_carrier(candidate.payload)
    cpr1_delta = None
    if cpr1.get("status") == "APPLIED":
        cpr1_delta = int(cpr1["bytes"]) - candidate.incumbent_bytes
    return {
        "label": candidate.label,
        "source": candidate.source,
        "format": candidate.format,
        "raw_bytes": len(candidate.payload),
        "raw_sha256": sha256_bytes(candidate.payload),
        "incumbent_bytes": candidate.incumbent_bytes,
        "best_generic": generic,
        "generic_delta_bytes": int(generic["bytes"]) - candidate.incumbent_bytes,
        "cpr1": cpr1,
        "cpr1_delta_bytes": cpr1_delta,
        "notes": candidate.notes,
    }


def default_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    tq1 = Path(
        "/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/"
        "phase_b_realized_tq1c/candidate_archives/"
        "move_0023_snap_r00_c12_L13.zip.receipt-bytes"
    )
    if tq1.exists():
        candidates.extend(
            ix2_candidates(
                tq1,
                section_names=(
                    "config",
                    "renderer",
                    "selector",
                    "pose_warp",
                    "frame0_pose_repair",
                ),
            )
        )

    fz4 = Path("/Volumes/VertigoDataTier/pact/ddm_fz1_20260804/rowB/sub_final/archive.zip")
    if fz4.exists():
        candidates.extend(
            ix2_candidates(
                fz4,
                section_names=(
                    "config",
                    "renderer",
                    "selector",
                    "pose_warp",
                    "frame0_pose_repair",
                ),
            )
        )

    pfs1 = Path("/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/submission/archive.zip")
    if pfs1.exists():
        candidates.extend(pfs1_candidates(pfs1))

    sc1 = sc1_ep_candidate(Path("/Volumes/VertigoDataTier/pact/ddm_sc1_20260728/ep_chunks"))
    if sc1 is not None:
        candidates.append(sc1)
    return candidates


def run_default_race() -> dict[str, Any]:
    candidates = default_candidates()
    rows = [race_candidate(candidate) for candidate in candidates]
    applied = [row for row in rows if row["cpr1"].get("status") == "APPLIED"]
    return {
        "schema": "ddm_mx2_pr130_pose_repack_race.v1",
        "score_claim": False,
        "axis": "lossless byte-only local repack; no scorer forward",
        "legacy_carrier_bytes": LEGACY_CARRIER_BYTES,
        "candidate_count": len(rows),
        "cpr1_applied_count": len(applied),
        "cpr1_applicable_to_our_banked_sections": bool(applied),
        "rows": rows,
    }


def write_report(report: dict[str, Any], path: Path = DEFAULT_REPORT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_default_race()
    write_report(report, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
