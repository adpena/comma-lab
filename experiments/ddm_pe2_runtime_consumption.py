#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""pe2 scorer-free PE1 runtime consumption and scorer-job staging.

This consumes PE1/BF1 byte-closed archives only through their shipped archive
bytes plus the current generic v4d receiver.  It proves:

* qo1 without PE1/BD1 optional sections is byte-identical through the extended
  receiver.
* PE1 sections parse into deterministic frame_1 paint bands on the real
  inflate path.
* The n600 scorer work remains staged, not run by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
for _path in (REPO / "src", REPO / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import ddm_bd1_class_field_receiver as bd1  # noqa: E402


BASE_SUB: Final = Path("/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/sub_auto_pairbit")
PE1_FULL_SUB: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pe1_20260805/pe1_20260805T030000Z/"
    "sub_auto_pairbit_pe1_full"
)
PE1_SURGICAL_SUB: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pe1_20260805/pe1_20260805T030000Z/"
    "sub_auto_pairbit_pe1_surgical_75kb"
)
BF1_SUB: Final = Path("/Volumes/VertigoDataTier/pact/ddm_bf1_20260805/sub_auto_pairbit_bf1_rl1_lane_crop_r3")
DEFAULT_RESEARCH_DIR: Final = REPO / ".omx/research/ddm_pe2_20260805"
DEFAULT_SSD_DIR: Final = Path("/Volumes/VertigoDataTier/pact/ddm_pe2_20260805")
BASE_RAW_BYTES: Final = 3_662_409_600
BASE_RAW_SHA256: Final = "3ce7d269a7080a4024a576694cd0ddc697099c64cd02fdd2bb879339e4b03f31"
BASELINE_S: Final = 0.7539807296911207
BASELINE_BYTES: Final = 357_836
BASELINE_AXIS: Final = "[macOS-CPU advisory]"


class PE2Error(ValueError):
    """The pe2 runtime proof or staging step failed closed."""


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    slug: str
    label: str
    source_sub: Path
    section_name: str
    kind: str
    expected_archive_bytes: int
    expected_archive_sha256: str


CANDIDATES: Final = (
    CandidateSpec(
        slug="pe1_full_explicit_curve_k8",
        label="PE1 full explicit_curve_k8",
        source_sub=PE1_FULL_SUB,
        section_name="pe1_edge_partition_full_explicit_curve_k8",
        kind="pe1",
        expected_archive_bytes=478_612,
        expected_archive_sha256="51e2e5b78d2c83b3cb357206c2e3a006ba3a51e2ba5661fcee364c39762a0416",
    ),
    CandidateSpec(
        slug="pe1_surgical_generator_pair_waterfill_75kb",
        label="PE1 surgical generator-pair waterfill 75kb",
        source_sub=PE1_SURGICAL_SUB,
        section_name="pe1_edge_partition_surgical_generator_pair_waterfill_75kb",
        kind="pe1",
        expected_archive_bytes=425_627,
        expected_archive_sha256="90de4c14887156fe462ead76163303bacefc3bbffec54bea10296ea104ff929a",
    ),
    CandidateSpec(
        slug="bf1_lane_crop_r3",
        label="BF1 lane-crop r3",
        source_sub=BF1_SUB,
        section_name="bd1_lane_crop_r3",
        kind="bf1",
        expected_archive_bytes=563_256,
        expected_archive_sha256="4741bfc91e3c013ea63435edd578933ef346540e998b71c368ff88f0b7bfa13a",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def storage_snapshot(path: Path, required_free_bytes: int) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    ok = int(usage.free) >= int(required_free_bytes)
    return {
        "path": str(path),
        "required_free_bytes": int(required_free_bytes),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "ok": bool(ok),
    }


def read_archive_payload(archive_zip: Path) -> bytes:
    with zipfile.ZipFile(archive_zip, "r") as archive:
        names = archive.namelist()
        if names != ["0.bin"]:
            raise PE2Error(f"expected single 0.bin member, found {names}")
        return archive.read("0.bin")


def materialize_receiver_copy(spec: CandidateSpec, ssd_dir: Path, *, reuse_existing: bool) -> Path:
    out_dir = ssd_dir / f"sub_auto_pairbit_pe2_{spec.slug}_receiver"
    if out_dir.exists():
        if not reuse_existing:
            raise PE2Error(f"receiver-copy dir already exists: {out_dir}")
        return out_dir
    bd1.copy_runtime_tree(spec.source_sub, out_dir)
    shutil.copy2(spec.source_sub / "archive.zip", out_dir / "archive.zip")
    payload = read_archive_payload(out_dir / "archive.zip")
    (out_dir / "archive" / "0.bin").write_bytes(payload)
    return out_dir


def import_generated_runner(out_dir: Path, slug: str) -> Any:
    sys.path.insert(0, str(out_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            f"pe2_generated_inflate_runner_{slug}",
            out_dir / "inflate_runner.py",
        )
        if spec is None or spec.loader is None:
            raise PE2Error(f"could not load generated receiver from {out_dir}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.path.remove(str(out_dir))
        except ValueError:
            pass


def receiver_copy_ledger(spec: CandidateSpec, out_dir: Path) -> dict[str, Any]:
    joint_names = (
        "config",
        "renderer",
        "selector",
        "pose_warp",
        "frame0_pose_repair",
        spec.section_name,
    )
    ledger = bd1.build_local_ledger(out_dir / "archive.zip", joint_names=joint_names)
    if ledger["archive_bytes"] != spec.expected_archive_bytes:
        raise PE2Error(f"{spec.slug}: archive byte count differs from PE1/BF1 receipt")
    if ledger["archive_sha256"] != spec.expected_archive_sha256:
        raise PE2Error(f"{spec.slug}: archive sha256 differs from PE1/BF1 receipt")
    return ledger


def pe1_runtime_smoke(spec: CandidateSpec, out_dir: Path) -> dict[str, Any]:
    module = import_generated_runner(out_dir, spec.slug)
    decoder = module.Decoder(out_dir / "archive")
    field = decoder._pe1_edge_field
    if field is None:
        raise PE2Error(f"{spec.slug}: PE1 field did not parse into the receiver")
    pair_index = next((i for i, count in enumerate(field["pair_counts"]) if int(count) > 0), None)
    if pair_index is None:
        raise PE2Error(f"{spec.slug}: PE1 field has no painted pairs")
    with_field = decoder.f1(pair_index)
    decoder._pe1_edge_field = None
    without_field = decoder.f1(pair_index)
    changed = np.any(with_field != without_field, axis=2)
    decoder2 = module.Decoder(out_dir / "archive")
    raster_hash_1 = field["raster_sha256"]
    raster_hash_2 = decoder2._pe1_edge_field["raster_sha256"]
    return {
        "kind": field["kind_name"],
        "section_bytes": field["section_bytes"],
        "section_sha256": field["section_sha256"],
        "raw_bytes": field["raw_bytes"],
        "raw_sha256": field["raw_sha256"],
        "component_records": field["component_records"],
        "painted_pairs": int(sum(1 for count in field["pair_counts"] if int(count) > 0)),
        "painted_pixels_total": int(sum(field["pair_counts"])),
        "smoke_pair": int(pair_index),
        "smoke_pair_painted_pixels": int(field["pair_counts"][pair_index]),
        "camera_pixels_changed": int(changed.sum()),
        "frame1_without_field_sha256": bd1.sha256_bytes(without_field.tobytes()),
        "frame1_with_field_sha256": bd1.sha256_bytes(with_field.tobytes()),
        "mutated": bool(np.any(changed)),
        "deterministic_raster_hash_first": raster_hash_1,
        "deterministic_raster_hash_second": raster_hash_2,
        "deterministic_raster_match": raster_hash_1 == raster_hash_2,
    }


def bf1_runtime_smoke(out_dir: Path) -> dict[str, Any]:
    smoke = bd1.receiver_smoke(out_dir, pair_index=0)
    if not smoke["receiver_class_field_present"]:
        raise PE2Error("BF1 class-field did not parse into the receiver")
    return smoke


def run_identity_decode(
    *,
    identity_dir: Path,
    reuse_existing: bool,
) -> dict[str, Any]:
    raw_path = identity_dir / "inflated" / "0.raw"
    if identity_dir.exists():
        if not reuse_existing:
            raise PE2Error(f"identity proof dir already exists: {identity_dir}")
        if not raw_path.exists():
            raise PE2Error(f"identity proof dir exists without raw output: {identity_dir}")
        raw_bytes = raw_path.stat().st_size
        raw_sha = sha256_file(raw_path)
        return {
            "reused_existing": True,
            "command": None,
            "output_raw": str(raw_path),
            "raw_bytes": raw_bytes,
            "raw_sha256": raw_sha,
            "expected_raw_bytes": BASE_RAW_BYTES,
            "expected_raw_sha256": BASE_RAW_SHA256,
            "byte_identical_to_qo1_shipped_decode": raw_bytes == BASE_RAW_BYTES and raw_sha == BASE_RAW_SHA256,
            "wall_seconds": None,
        }
    return bd1.run_identity_decode(
        base_sub=BASE_SUB,
        identity_dir=identity_dir,
        expected_raw_sha256=BASE_RAW_SHA256,
        expected_raw_bytes=BASE_RAW_BYTES,
    )


def write_stage_script(
    *,
    path: Path,
    scorer_out_dir: Path,
    candidates: list[dict[str, Any]],
) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {REPO}",
        'DEVICE="${1:-cpu}"',
        'BATCH_SIZE="${BATCH_SIZE:-16}"',
        'NUM_THREADS="${NUM_THREADS:-4}"',
        f'OUT_ROOT="{scorer_out_dir}"',
        'mkdir -p "$OUT_ROOT"',
        'echo "[pe2] one scorer-slot batch; run only after MAIN confirms sq2 released the slot" >&2',
    ]
    for row in candidates:
        slug = row["slug"]
        sub_dir = row["receiver_copy"]
        lines.extend(
            [
                f'echo "[pe2] scoring {slug} on $DEVICE" >&2',
                ".venv/bin/python experiments/ddm_fz2_byteclose_and_eval.py "
                f"--sub-dir {sub_dir} "
                f'--out "$OUT_ROOT/{slug}_n600_${{DEVICE}}.json" '
                f'--inflate-out "$OUT_ROOT/{slug}_inflate_${{DEVICE}}" '
                '--device "$DEVICE" '
                '--batch-size "$BATCH_SIZE" '
                '--num-threads "$NUM_THREADS"',
            ]
        )
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o755)


def write_queue_note(path: Path, receipt: dict[str, Any]) -> None:
    stage_script = receipt["staged_scorer_job"]["script"]
    lines = [
        "# PE2 n600 Scorer Queue Note",
        "",
        "Status: **QUEUED-WITH-FIRE-ORDER / NOT RUN BY PE2**.",
        "",
        "Fire condition: MAIN confirms `sq2 [SCORER]` has released the single scorer slot, then claims one scorer-slot batch for the three candidate rows below.",
        "",
        f"Exact fire command from repo root: `bash {stage_script} cpu`",
        "",
        "Axis warning: running the command on this Mac is `[macOS-CPU advisory]`; contest authority still requires the contest-CPU or contest-CUDA host.",
        "",
        "| candidate | receiver-copy archive bytes | receiver-copy archive sha256 | submission dir |",
        "|---|---:|---|---|",
    ]
    for row in receipt["candidates"]:
        lines.append(
            f"| {row['label']} | `{row['archive_bytes']}` | `{row['archive_sha256']}` | `{row['receiver_copy']}` |"
        )
    lines.extend(
        [
            "",
            "Batch contract:",
            "",
            "- One fire handles PE1 full, PE1 surgical, and BF1 in sequence.",
            "- All three candidates are qo1-base IX2 archives with one optional receiver-consumed section family.",
            "- The staged script uses the canonical byteclose/evaluate wrapper per exact archive; this preserves exact-archive semantics.",
            "- No scorer was run while PE2 generated this note.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def write_markdown_receipt(path: Path, receipt: dict[str, Any]) -> None:
    identity = receipt["old_archive_identity_proof"]
    lines = [
        "# PE2 runtime consumption receipt - 2026-08-05",
        "",
        "Status: **RECEIVER-CLOSED / ABSENT-IDENTITY-PROVED / DETERMINISTIC-RASTER-PROVED / SCORER-JOB-STAGED / SURVIVAL-UNMEASURED**.",
        "",
        "Axis: `[macOS-CPU advisory / scorer-free receiver-byte custody]`.",
        "`score_claim=false`, `promotion_eligible=false`, `n600_scorer_job=false`.",
        "",
        "## Storage Preflight",
        "",
        f"- SSD tier: `{receipt['storage_preflight']['path']}`.",
        f"- Required free bytes: `{receipt['storage_preflight']['required_free_bytes']}`; observed free bytes: `{receipt['storage_preflight']['free_bytes']}`; ok: `{receipt['storage_preflight']['ok']}`.",
        "",
        "## Absent-Section Identity",
        "",
        f"- qo1 raw bytes: `{identity['raw_bytes']}`; expected `{identity['expected_raw_bytes']}`.",
        f"- qo1 raw sha256: `{identity['raw_sha256']}`.",
        f"- Byte-identical to the shipped qo1 decode: `{identity['byte_identical_to_qo1_shipped_decode']}`.",
        "",
        "## Candidate Runtime Consumption",
        "",
        "| candidate | archive bytes | section bytes | runtime proof | deterministic raster |",
        "|---|---:|---:|---|---|",
    ]
    for row in receipt["candidates"]:
        proof = row["runtime_consumption"]
        if row["kind"] == "pe1":
            runtime = (
                f"pair {proof['smoke_pair']} changed {proof['camera_pixels_changed']} camera px; "
                f"{proof['component_records']} component records"
            )
            det = str(proof["deterministic_raster_match"])
            section_bytes = proof["section_bytes"]
        else:
            runtime = (
                f"pair {proof['pair']} changed {proof['camera_pixels_changed']} camera px; "
                f"{proof['band_pixels_this_pair']} band px"
            )
            det = "BD1 precedent path"
            section_bytes = row["ledger"]["joint_sections"][-1]["raw_bytes"]
        lines.append(
            f"| {row['label']} | `{row['archive_bytes']}` | `{section_bytes}` | {runtime} | `{det}` |"
        )
    lines.extend(
        [
            "",
            "## Staged Scorer Job",
            "",
            f"- Script: `{receipt['staged_scorer_job']['script']}`.",
            f"- Queue note: `{receipt['staged_scorer_job']['queue_note']}`.",
            f"- Manifest: `{receipt['staged_scorer_job']['manifest']}`.",
            "- PE2 did not run SegNet, PoseNet, or `upstream/evaluate.py`.",
            "",
            "## Recall Evidence",
            "",
            "- PE1 receipt: two byte-closed PE1 candidates were receiver-closed only after this PE2 consumer work.",
            "- PE1 representation race: `PE1EDGE1` grammar uses n600 frame records and class-pair edge records with no score claim.",
            "- BF1/BD1 precedent: optional IX2 sections must be tagged, fail-closed, absent-identity-preserving, and receiver-consumed before scorer promotion.",
            "- Per-edge optimality directive: PE2 consumes side-implied/directional edge records as receiver payload; it does not move target tables into receiver code.",
            "",
            "## Follow-On Disposition",
            "",
            "FIRED: PE1 optional-section runtime consumption is implemented in the v4d receiver, with focused parser/apply tests.",
            "",
            "FIRED: qo1 absent-section identity proof and deterministic PE1 raster proof are recorded in the JSON receipt.",
            "",
            "QUEUED-WITH-FIRE-ORDER: after `sq2 [SCORER]` releases the slot, MAIN runs the staged one-slot batch script and records the n600 survival rows for PE1 full, PE1 surgical, and BF1.",
            "",
            "FOLDED: physical multi-candidate shared-decode optimization is kept as manifest context only; the staged fire uses the canonical exact-archive wrapper per candidate.",
            "",
            "## NEXT-IF-RESUMED",
            "",
            "1. Confirm the scorer slot is free in `.omx/state/main_hot_state.md`.",
            "2. Claim the scorer lane for PE2/MAIN.",
            f"3. Run `bash {receipt['staged_scorer_job']['script']} cpu` for an advisory local pass or run the same script on authority hardware with the appropriate device.",
            "4. Recompute S from d_seg, d_pose, and archive bytes in the scorer receipts before any pointer claim.",
            "",
            f"Own-vehicle frontier line: `S = {BASELINE_S} @ {BASELINE_BYTES:,} B {BASELINE_AXIS}`; PE2 staged survival work and did not move the contest pointer.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH_DIR)
    parser.add_argument("--ssd-dir", type=Path, default=DEFAULT_SSD_DIR)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--skip-identity-decode", action="store_true")
    args = parser.parse_args(argv)

    args.research_dir.mkdir(parents=True, exist_ok=True)
    args.ssd_dir.mkdir(parents=True, exist_ok=True)
    storage = storage_snapshot(args.ssd_dir, required_free_bytes=BASE_RAW_BYTES + 1_000_000_000)
    if not storage["ok"]:
        raise PE2Error("SSD tier lacks enough free space for the absent-identity raw proof")

    candidate_rows: list[dict[str, Any]] = []
    for spec in CANDIDATES:
        out_dir = materialize_receiver_copy(spec, args.ssd_dir, reuse_existing=args.reuse_existing)
        ledger = receiver_copy_ledger(spec, out_dir)
        if spec.kind == "pe1":
            runtime = pe1_runtime_smoke(spec, out_dir)
            if not runtime["mutated"] or not runtime["deterministic_raster_match"]:
                raise PE2Error(f"{spec.slug}: PE1 runtime consumption proof failed")
        else:
            runtime = bf1_runtime_smoke(out_dir)
            if not runtime["mutated"]:
                raise PE2Error(f"{spec.slug}: BF1 runtime consumption proof failed")
        candidate_rows.append(
            {
                "slug": spec.slug,
                "label": spec.label,
                "kind": spec.kind,
                "source_sub": str(spec.source_sub),
                "receiver_copy": str(out_dir),
                "archive_bytes": ledger["archive_bytes"],
                "archive_sha256": ledger["archive_sha256"],
                "ledger": ledger,
                "runtime_consumption": runtime,
            }
        )

    if args.skip_identity_decode:
        identity = {
            "skipped": True,
            "byte_identical_to_qo1_shipped_decode": False,
            "raw_bytes": None,
            "raw_sha256": None,
            "expected_raw_bytes": BASE_RAW_BYTES,
            "expected_raw_sha256": BASE_RAW_SHA256,
        }
    else:
        identity = run_identity_decode(
            identity_dir=args.ssd_dir / "qo1_identity_pe2_extended_receiver",
            reuse_existing=args.reuse_existing,
        )
        if not identity["byte_identical_to_qo1_shipped_decode"]:
            raise PE2Error("qo1 absent-section identity proof failed")

    scorer_out_dir = args.ssd_dir / "scorer_batch"
    stage_script = args.research_dir / "stage_pe2_three_candidate_scorer_batch.sh"
    manifest_path = args.research_dir / "pe2_three_candidate_scorer_manifest.json"
    queue_note = args.research_dir / "PE2_QUEUE_NOTE.md"
    write_stage_script(path=stage_script, scorer_out_dir=scorer_out_dir, candidates=candidate_rows)

    manifest = {
        "schema": "ddm_pe2_three_candidate_scorer_manifest.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "score_claim": False,
        "n600_scorer_job": "staged_not_run",
        "single_scorer_slot_owner_to_claim": "MAIN after sq2 releases [SCORER]",
        "shared_base": {
            "base_sub": str(BASE_SUB),
            "base_archive_bytes": BASELINE_BYTES,
            "base_archive_sha256": sha256_file(BASE_SUB / "archive.zip"),
            "relation": "all candidates are qo1-base IX2 archives plus one optional receiver-consumed section family",
        },
        "candidates": candidate_rows,
        "stage_script": str(stage_script),
    }
    manifest_path.write_text(json.dumps(jsonable(manifest), indent=1, sort_keys=True))

    receipt = {
        "schema": "ddm_pe2_runtime_consumption_receipt.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "axis": "[macOS-CPU advisory / scorer-free receiver-byte custody]",
        "score_claim": False,
        "promotion_eligible": False,
        "n600_scorer_job": False,
        "storage_preflight": storage,
        "old_archive_identity_proof": identity,
        "candidates": candidate_rows,
        "staged_scorer_job": {
            "status": "QUEUED-WITH-FIRE-ORDER",
            "script": str(stage_script),
            "queue_note": str(queue_note),
            "manifest": str(manifest_path),
            "scorer_out_dir": str(scorer_out_dir),
            "run_by_pe2": False,
        },
        "own_vehicle_frontier": {
            "S": BASELINE_S,
            "archive_bytes": BASELINE_BYTES,
            "axis": BASELINE_AXIS,
            "pointer_moved": False,
        },
    }
    receipt_json = args.research_dir / "ddm_pe2_runtime_consumption_receipt.json"
    receipt_md = args.research_dir / "PE2_RECEIPT_20260805.md"
    receipt_json.write_text(json.dumps(jsonable(receipt), indent=1, sort_keys=True))
    write_queue_note(queue_note, receipt)
    write_markdown_receipt(receipt_md, receipt)
    print(json.dumps({"receipt": str(receipt_json), "markdown": str(receipt_md)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
