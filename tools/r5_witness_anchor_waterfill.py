#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Task #578-R5 exact ep725 LVLS1 rate-stream measurement helpers.

This adapter is deliberately narrow.  It can mutate signed-int8 coefficient
planes while preserving the shipped LVLS1+Brotli receiver grammar, measure the
already-built #557/context and block-FP coder frames without pretending they
are receiver-bound archives, and compare the R3 description-space donor sites
to receiver-measured witness misses.  It never scores or moves a pointer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import zipfile
import zlib
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tac.canonical_equations.day_consolidation_laws_20260720 import (  # noqa: E402
    RATE_PRICE_S_PER_BYTE,
)
from tac.contest_score import compute_contest_score  # noqa: E402
from tac.optimization.arith_selfcomp_rate_coders import (  # noqa: E402
    measure_block_fp,
    measure_iid_signed_array_ladder,
    measure_signed_array_ladder,
)
from tac.packet_compiler.jrd_coefficient_prefix import (  # noqa: E402
    coefficient_sections,
    quantize_prefix,
    read_section,
    replace_section,
)

SCHEMA: Final = "r5_witness_anchor_waterfill.v1"
MAGIC: Final = b"LVLS1\x00"
SEG_HEIGHT: Final = 384
SEG_WIDTH: Final = 512
COMPONENT_HEADER: Final = struct.Struct("<HBBII")
LAWREF_RATE_PRICE: Final = (
    "tac.canonical_equations.day_consolidation_laws_20260720.RATE_PRICE_S_PER_BYTE"
)
LAWREF_CHECKPOINT_DSEG: Final = (
    ".omx/research/duty_ticket_revision_ep725_fork_20260719_claude.md"
)
LAWREF_POINTER_COMPONENTS: Final = (
    ".omx/state/canonical_equations_registry.jsonl#"
    "clickpolish_exact_gated_discrete_latent_ratchet_v1"
)
POINTER_D_SEG: Final = 0.00055961
POINTER_D_POSE: Final = 2.942e-05
POINTER_ARCHIVE_BYTES: Final = 177_169


class R5Error(ValueError):
    """Fail-closed input, grammar, or custody error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise R5Error(f"output overwrite refused: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def archive_blob(path: Path) -> tuple[bytes, dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["0.bin"]:
            raise R5Error("anchor archive must contain exactly one 0.bin member")
        blob = archive.read("0.bin")
    return blob, {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def parse_lvls1(blob: bytes) -> tuple[dict[str, Any], list[bytes]]:
    if not blob.startswith(MAGIC):
        raise R5Error("not an LVLS1 blob")
    offset = len(MAGIC)
    blocks: list[bytes] = []
    for _ in range(4):
        if offset + 4 > len(blob):
            raise R5Error("truncated LVLS1 section length")
        (size,) = struct.unpack_from("<I", blob, offset)
        offset += 4
        stop = offset + size
        if stop > len(blob):
            raise R5Error("truncated LVLS1 section")
        blocks.append(blob[offset:stop])
        offset = stop
    manifest = json.loads(blocks[0])
    optional_flags = (
        "lane_render_band",
        "pose_carrier",
        "chart_payload",
        "palette_residual",
    )
    for flag in optional_flags:
        if manifest.get(flag) is not None:
            if offset + 4 > len(blob):
                raise R5Error(f"manifest declares missing optional section {flag}")
            (size,) = struct.unpack_from("<I", blob, offset)
            offset += 4
            stop = offset + size
            if stop > len(blob):
                raise R5Error(f"truncated optional section {flag}")
            blocks.append(blob[offset:stop])
            offset = stop
    if offset != len(blob):
        raise R5Error(f"LVLS1 has {len(blob) - offset} trailing bytes")
    return manifest, blocks


def pack_lvls1(blocks: list[bytes]) -> bytes:
    return MAGIC + b"".join(struct.pack("<I", len(block)) + block for block in blocks)


def deterministic_archive(path: Path, blob: bytes) -> None:
    info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(info, blob)


def build_prefix(args: argparse.Namespace) -> int:
    anchor_blob, anchor = archive_blob(args.anchor_archive.resolve(strict=True))
    manifest, blocks = parse_lvls1(anchor_blob)
    if manifest.get("xcodec") is not None:
        raise R5Error("prefix mutation is not defined over transformed xcodec storage")
    base_raw = brotli.decompress(blocks[1])
    code_raw = brotli.decompress(blocks[2])
    sections = {
        section.name: section
        for section in coefficient_sections(
            manifest, base_raw_len=len(base_raw), code_raw_len=len(code_raw)
        )
    }
    plan = json.loads(args.plan_json)
    if not isinstance(plan, list) or not plan:
        raise R5Error("--plan-json must be a nonempty JSON list")
    rows = []
    for raw_row in plan:
        name = str(raw_row["section"])
        if name not in sections:
            raise R5Error(f"unknown coefficient section: {name}")
        section = sections[name]
        family = str(raw_row["family"])
        bits_removed = int(raw_row["bits_removed"])
        before = read_section(base_raw, code_raw, section)
        after = quantize_prefix(before, bits_removed=bits_removed, family=family)
        base_raw, code_raw = replace_section(base_raw, code_raw, section, after)
        rows.append(
            {
                "section": name,
                "stream": section.stream,
                "family": family,
                "bits_removed": bits_removed,
                "elements": int(before.size),
                "changed_elements": int(np.count_nonzero(before != after)),
                "before_sha256": hashlib.sha256(before.tobytes()).hexdigest(),
                "after_sha256": hashlib.sha256(after.tobytes()).hexdigest(),
            }
        )
    candidate_blocks = list(blocks)
    candidate_blocks[1] = brotli.compress(base_raw, quality=11)
    candidate_blocks[2] = brotli.compress(code_raw, quality=11)
    candidate_blob = pack_lvls1(candidate_blocks)
    candidate_manifest, replay_blocks = parse_lvls1(candidate_blob)
    if candidate_manifest != manifest or brotli.decompress(replay_blocks[1]) != base_raw:
        raise R5Error("candidate LVLS1 parse-back mismatch")
    if brotli.decompress(replay_blocks[2]) != code_raw:
        raise R5Error("candidate code parse-back mismatch")

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise R5Error(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "0.bin").write_bytes(candidate_blob)
    deterministic_archive(output_dir / "archive.zip", candidate_blob)
    shutil.copy2(args.anchor_inflate_py.resolve(strict=True), output_dir / "inflate.py")
    shutil.copy2(args.anchor_inflate_sh.resolve(strict=True), output_dir / "inflate.sh")
    receipt = {
        "schema": f"{SCHEMA}.prefix_candidate.v1",
        "authority": "candidate bytes only; distortion owed through full decode and hard CPU scorer",
        "lawrefs": {"rate_price": LAWREF_RATE_PRICE, "lambda_star_s_per_byte": RATE_PRICE_S_PER_BYTE},
        "anchor": anchor,
        "candidate": {
            "archive": {
                "path": str(output_dir / "archive.zip"),
                "bytes": (output_dir / "archive.zip").stat().st_size,
                "sha256": sha256_file(output_dir / "archive.zip"),
            },
            "blob_bytes": len(candidate_blob),
            "blob_sha256": hashlib.sha256(candidate_blob).hexdigest(),
            "delta_bytes": (output_dir / "archive.zip").stat().st_size - int(anchor["bytes"]),
            "plan": rows,
            "parseback_exact": True,
            "receiver_grammar": "unchanged LVLS1+Brotli",
        },
    }
    atomic_json(output_dir / "build_receipt.json", receipt)
    print(json.dumps(receipt["candidate"], sort_keys=True))
    return 0


def _spatial_view(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return array.reshape(1, 1, array.shape[0], 1)
    if array.ndim == 2:
        return array.reshape(1, array.shape[0], array.shape[1], 1)
    return array.reshape(1, *array.shape, 1)


def coder_race(args: argparse.Namespace) -> int:
    blob, anchor = archive_blob(args.anchor_archive.resolve(strict=True))
    manifest, blocks = parse_lvls1(blob)
    if manifest.get("xcodec") is not None:
        raise R5Error("coder race requires raw logical coefficient order")
    base_raw = brotli.decompress(blocks[1])
    code_raw = brotli.decompress(blocks[2])
    sections = coefficient_sections(manifest, base_raw_len=len(base_raw), code_raw_len=len(code_raw))
    rows = []
    for section in sections:
        array = read_section(base_raw, code_raw, section)
        iid = measure_iid_signed_array_ladder(array)["repository_iid_arithmetic"]
        context = measure_signed_array_ladder(_spatial_view(array))[
            "repository_spatial_context_arithmetic"
        ]
        row: dict[str, Any] = {
            "section": section.name,
            "stream": section.stream,
            "shape": list(section.shape),
            "elements": int(array.size),
            "isolated_brotli_q11_bytes": len(brotli.compress(array.tobytes(), quality=11)),
            "iid_arithmetic": iid,
            "spatial_context_arithmetic": context,
        }
        if section.stream == "base":
            scale = float(manifest["base_scales"][section.name])
            row["block_fp"] = measure_block_fp(array.astype(np.float32) * scale)
        else:
            row["block_fp"] = {
                "status": "NOT_WEIGHT_SECTION",
                "reason": "pair code is not a base weight tensor",
            }
        rows.append(row)
    aggregate = {
        "isolated_brotli_q11_bytes": sum(int(row["isolated_brotli_q11_bytes"]) for row in rows),
        "iid_arithmetic_framed_bytes": sum(
            int(row["iid_arithmetic"]["framed_bytes"]) for row in rows
        ),
        "spatial_context_arithmetic_framed_bytes": sum(
            int(row["spatial_context_arithmetic"]["framed_bytes"]) for row in rows
        ),
        "block_fp_weight_brotli_plus_current_code_bytes": (
            sum(
                int(row["block_fp"]["packed_byte_coders"]["brotli_q11"]["framed_bytes"])
                for row in rows
                if row["stream"] == "base"
            )
            + len(blocks[2])
        ),
    }
    payload = {
        "schema": f"{SCHEMA}.coder_race.v1",
        "authority": {
            "axis": "[macOS-CPU byte measurement only]",
            "archive_claim": False,
            "distortion_claim": False,
            "verdict_scope": "exact fully framed coder bytes; LVLS1 receiver binding absent",
        },
        "lawrefs": {"rate_price": LAWREF_RATE_PRICE, "lambda_star_s_per_byte": RATE_PRICE_S_PER_BYTE},
        "anchor": anchor,
        "current_lvls1": {
            "base_brotli_bytes": len(blocks[1]),
            "code_brotli_bytes": len(blocks[2]),
            "archive_bytes": int(anchor["bytes"]),
        },
        "receiver_gate": {
            "status": "BLOCKED_NO_LVLS1_CONTEXT_OR_BLOCK_FP_PARSER_CONSUMER",
            "full_decode_run": False,
            "reason": "shipped inflate.py only brotli-decompresses signed-int8 base/code streams",
        },
        "aggregate_unbound_inner_stream_bytes": aggregate,
        "sections": rows,
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps({"output": str(args.output), "sections": len(rows)}, sort_keys=True))
    return 0


def _read_uvarint(payload: bytes, offset: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        if offset >= len(payload) or shift > 63:
            raise R5Error("component varint is truncated or overlong")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7


def donor_keys(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    payload = path.read_bytes()
    offset = 0
    keys: list[np.ndarray] = []
    packets = 0
    while offset < len(payload):
        if offset + 4 > len(payload):
            raise R5Error("truncated pcomp3 packet length")
        (size,) = struct.unpack_from("<I", payload, offset)
        offset += 4
        stop = offset + size
        if stop > len(payload):
            raise R5Error("truncated pcomp3 packet")
        raw = zlib.decompress(payload[offset:stop])
        offset = stop
        if len(raw) < COMPONENT_HEADER.size:
            raise R5Error("truncated pcomp3 component")
        frame, class_id, _stratum, count, first = COMPONENT_HEADER.unpack_from(raw)
        if frame >= 600 or class_id >= 5 or count == 0:
            raise R5Error("pcomp3 component header is outside the n600 SegNet grammar")
        sites = np.empty(count, dtype=np.uint32)
        sites[0] = first
        raw_offset = COMPONENT_HEADER.size
        for index in range(1, count):
            delta, raw_offset = _read_uvarint(raw, raw_offset)
            sites[index] = int(sites[index - 1]) + delta
        if raw_offset != len(raw) or np.any(sites >= SEG_HEIGHT * SEG_WIDTH):
            raise R5Error("pcomp3 component parse-back failed")
        keys.append((np.uint64(frame) * (SEG_HEIGHT * SEG_WIDTH) + sites) * 5 + class_id)
        packets += 1
    joined = np.unique(np.concatenate(keys)) if keys else np.empty(0, dtype=np.uint64)
    return joined, {
        "path": str(path),
        "bytes": len(payload),
        "sha256": sha256_file(path),
        "packets": packets,
        "unique_description_repairable_sites": int(joined.size),
    }


def r3_overlap(args: argparse.Namespace) -> int:
    receipt = json.loads(args.miss_receipt.read_text())
    chunks = receipt["measurement"]["label_cache"]["chunks"]
    miss_keys = []
    for row in chunks:
        path = Path(row["path"])
        if sha256_file(path) != row["sha256"]:
            raise R5Error(f"mismatch chunk hash drift: {path}")
        with np.load(path, allow_pickle=False) as chunk:
            key = (
                (chunk["pair_index"].astype(np.uint64) * (SEG_HEIGHT * SEG_WIDTH)
                 + chunk["site_index"].astype(np.uint64))
                * 5
                + chunk["gt_class"].astype(np.uint64)
            )
            miss_keys.append(key)
    witness = np.unique(np.concatenate(miss_keys))
    donor, donor_receipt = donor_keys(args.donor.resolve(strict=True))
    overlap = np.intersect1d(witness, donor, assume_unique=True)
    fraction = float(overlap.size / witness.size) if witness.size else 0.0
    payload = {
        "schema": f"{SCHEMA}.r3_overlap.v1",
        "authority": "receiver-measured SegNet miss coordinates intersected with description-space donor; no RGB splice claim",
        "witness_miss_sites": int(witness.size),
        "donor": donor_receipt,
        "overlap_sites": int(overlap.size),
        "overlap_fraction_of_witness_misses": fraction,
        "threshold": 0.05,
        "splice_gate": "PASS_GT_5_PERCENT" if fraction > 0.05 else "FAIL_NOT_GT_5_PERCENT",
        "splice_executed": False,
        "verdict_scope": "site/class overlap only; donor has no LVLS1 RGB inverse-R receiver binding",
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


def _score_row(score_receipt: Path, build_receipt: Path) -> dict[str, Any]:
    score_payload = json.loads(score_receipt.read_text())
    build_payload = json.loads(build_receipt.read_text())
    aggregate = score_payload["measurement"]["aggregate"]
    candidate = build_payload["candidate"]
    archive_bytes = int(candidate["archive"]["bytes"])
    d_seg = float(aggregate["d_seg_official_float32"])
    d_pose = float(aggregate["d_pose_official_float32"])
    score_terms = {
        "seg_s": 100.0 * d_seg,
        "pose_s": float(np.sqrt(10.0 * d_pose)),
        "rate_s": RATE_PRICE_S_PER_BYTE * archive_bytes,
    }
    return {
        "d_seg": d_seg,
        "d_pose": d_pose,
        "archive_bytes": archive_bytes,
        "advisory_s": compute_contest_score(d_seg, d_pose, archive_bytes),
        "score_terms": score_terms,
        "archive": candidate["archive"],
        "score_receipt": {
            "path": str(score_receipt),
            "bytes": score_receipt.stat().st_size,
            "sha256": sha256_file(score_receipt),
        },
        "build_receipt": {
            "path": str(build_receipt),
            "bytes": build_receipt.stat().st_size,
            "sha256": sha256_file(build_receipt),
        },
        "plan": candidate["plan"],
    }


def compose(args: argparse.Namespace) -> int:
    anchor_payload = json.loads(args.anchor_score.read_text())
    anchor_aggregate = anchor_payload["measurement"]["aggregate"]
    anchor_archive = {
        "path": str(args.anchor_archive),
        "bytes": args.anchor_archive.stat().st_size,
        "sha256": sha256_file(args.anchor_archive),
    }
    baseline = {
        "d_seg": float(anchor_aggregate["d_seg_official_float32"]),
        "d_pose": float(anchor_aggregate["d_pose_official_float32"]),
        "archive_bytes": int(anchor_archive["bytes"]),
    }
    baseline["advisory_s"] = compute_contest_score(
        baseline["d_seg"], baseline["d_pose"], baseline["archive_bytes"]
    )
    baseline["score_terms"] = {
        "seg_s": 100.0 * baseline["d_seg"],
        "pose_s": float(np.sqrt(10.0 * baseline["d_pose"])),
        "rate_s": RATE_PRICE_S_PER_BYTE * baseline["archive_bytes"],
    }
    candidate_inputs = {
        "jrd_prefix_453": (args.jrd_score, args.jrd_build),
        "requant_336": (args.requant_score, args.requant_build),
    }
    candidates: dict[str, dict[str, Any]] = {}
    admitted: list[str] = []
    baseline_nonrate = 100.0 * baseline["d_seg"] + np.sqrt(10.0 * baseline["d_pose"])
    for name, (score_path, build_path) in candidate_inputs.items():
        row = _score_row(score_path, build_path)
        nonrate = 100.0 * row["d_seg"] + np.sqrt(10.0 * row["d_pose"])
        row["delta_d_seg"] = row["d_seg"] - baseline["d_seg"]
        row["delta_d_pose"] = row["d_pose"] - baseline["d_pose"]
        row["delta_bytes"] = row["archive_bytes"] - baseline["archive_bytes"]
        row["delta_nonrate_s"] = float(nonrate - baseline_nonrate)
        row["delta_rate_s"] = float(RATE_PRICE_S_PER_BYTE * row["delta_bytes"])
        row["delta_total_s"] = float(row["advisory_s"] - baseline["advisory_s"])
        row["waterfill_admit"] = bool(row["delta_total_s"] < 0.0)
        row["admission_rule"] = "delta_nonrate_s + lambda_star*delta_bytes < 0"
        if row["waterfill_admit"]:
            admitted.append(name)
        candidates[name] = row

    overlap = json.loads(args.r3_overlap.read_text())
    donor_bytes = int(overlap["donor"]["bytes"])
    overlap_sites = int(overlap["overlap_sites"])
    donor_best_case = {
        "status": "REJECTED_BEST_CASE_RATE_DOMINATED_AND_RECEIVER_OPEN",
        "overlap_receipt": {
            "path": str(args.r3_overlap),
            "sha256": sha256_file(args.r3_overlap),
        },
        "delta_d_seg_best_case": -overlap_sites / (600 * SEG_HEIGHT * SEG_WIDTH),
        "delta_nonrate_s_best_case": -100.0 * overlap_sites / (600 * SEG_HEIGHT * SEG_WIDTH),
        "delta_bytes": donor_bytes,
        "delta_rate_s": RATE_PRICE_S_PER_BYTE * donor_bytes,
    }
    donor_best_case["delta_total_s_best_case"] = (
        donor_best_case["delta_nonrate_s_best_case"] + donor_best_case["delta_rate_s"]
    )

    interaction: dict[str, Any]
    if len(admitted) <= 1:
        interaction = {
            "status": "CLOSED_ZERO_OR_ONE_ADMITTED_STREAM",
            "admitted_streams": admitted,
            "pairwise_cells": [],
            "union_once": True,
            "commutator_owed": False,
        }
        selected = "anchor" if not admitted else admitted[0]
        v5 = dict(baseline if selected == "anchor" else candidates[selected])
        v5["selected_stream"] = selected
        v5["status"] = "MEASURED_RECEIVER_CLOSED_SINGLETON_COMPOSITION"
    else:
        interaction = {
            "status": "BLOCKED_JOINT_ARCHIVE_DECODE_AND_SCORE_OWED",
            "admitted_streams": admitted,
            "pairwise_cells": [],
            "union_once": False,
            "commutator_owed": True,
        }
        v5 = {
            **baseline,
            "selected_stream": "anchor",
            "status": "ANCHOR_RETAINED_UNTIL_INTERACTION_CLOSURE",
        }

    pointer = float(args.pointer)
    pointer_terms = {
        "seg_s": 100.0 * POINTER_D_SEG,
        "pose_s": float(np.sqrt(10.0 * POINTER_D_POSE)),
        "rate_s": RATE_PRICE_S_PER_BYTE * POINTER_ARCHIVE_BYTES,
    }
    pointer_from_components = sum(pointer_terms.values())
    if abs(pointer_from_components - pointer) > 1e-9:
        raise R5Error(
            "pointer/components drift: "
            f"argument={pointer:.12g}, components={pointer_from_components:.12g}"
        )
    v5_terms = v5["score_terms"]
    gap_terms = {name: float(v5_terms[name] - pointer_terms[name]) for name in pointer_terms}
    total_gap = float(sum(gap_terms.values()))
    gap_percent = {name: 100.0 * value / total_gap for name, value in gap_terms.items()}
    v5_percent = {
        name: 100.0 * value / float(v5["advisory_s"]) for name, value in v5_terms.items()
    }
    payload = {
        "schema": f"{SCHEMA}.composed_receipt.v1",
        "task": "578-R5",
        "lane_id": "r5_witness_anchor_waterfill",
        "authority": {
            "axis": "[macOS-CPU advisory]",
            "contest_score_claim": False,
            "promotion_eligible": False,
            "pointer_mutation": False,
            "main_review_required": True,
        },
        "lawrefs": {
            "rate_price": LAWREF_RATE_PRICE,
            "lambda_star_s_per_byte": RATE_PRICE_S_PER_BYTE,
            "score": "tac.contest_score.compute_contest_score",
            "checkpoint_side_d_seg": LAWREF_CHECKPOINT_DSEG,
            "pointer_components": LAWREF_POINTER_COMPONENTS,
            "pair_local_break_even": "150 bytes per 1e-6 d_seg (DERIVED: 100e-6/lambda_star)",
        },
        "D1_anchor": {
            **baseline,
            "archive": anchor_archive,
            "score_receipt": {
                "path": str(args.anchor_score),
                "sha256": sha256_file(args.anchor_score),
            },
            "checkpoint_side_d_seg": 0.003457972208658854,
            "checkpoint_to_shipped_delta_d_seg": baseline["d_seg"] - 0.003457972208658854,
            "premise_verdict": "FALSIFIED_CHECKPOINT_DSEG_IS_NOT_SHIPPED_ARCHIVE_D1",
        },
        "D2_curves": {
            **candidates,
            "low_rank_pose_140": {
                "status": "NOT_APPLICABLE_ABSENT_DXI_STREAM",
                "manifest_has_pose_sidecar": False,
                "delta_bytes": 0,
                "delta_d_seg": 0.0,
                "delta_d_pose": 0.0,
            },
            "pair_local_polish_400": {
                "status": "ACTUATION_REFUSE_JOINT_SEG_POSE_ACCEPT_ADAPTER_ABSENT",
                "self_orient": True,
                "reason": "code click can change shared h0 and both frames; Seg-only diagonal acceptance is unsafe",
                "break_even": "150 bytes per 1e-6 d_seg",
            },
            "r3_donor_transfer": donor_best_case,
        },
        "D3_interaction_matrix": interaction,
        "D3_composed_v5": v5,
        "D4_pointer_comparison": {
            "pointer_s": pointer,
            "pointer_axis": "[contest-CPU]",
            "pointer_components": {
                "d_seg": POINTER_D_SEG,
                "d_pose": POINTER_D_POSE,
                "archive_bytes": POINTER_ARCHIVE_BYTES,
                "score_terms": pointer_terms,
                "recomputed_s": pointer_from_components,
            },
            "v5_axis": "[macOS-CPU advisory]",
            "v5_score_terms": v5_terms,
            "v5_composition_percent": v5_percent,
            "gap_score_terms": gap_terms,
            "gap_component_percent": gap_percent,
            "axis_comparable_for_promotion": False,
            "advisory_numeric_gap_v5_minus_pointer": float(v5["advisory_s"] - pointer),
            "pointer_unchanged": True,
        },
        "inputs": {
            "coder_race": {"path": str(args.coder_race), "sha256": sha256_file(args.coder_race)},
            "r3_overlap": {"path": str(args.r3_overlap), "sha256": sha256_file(args.r3_overlap)},
        },
    }
    atomic_json(args.output.resolve(), payload)
    print(json.dumps({"output": str(args.output), "admitted": admitted, "v5": v5}, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser()
    sub = top.add_subparsers(dest="command", required=True)
    prefix = sub.add_parser("build-prefix")
    prefix.add_argument("--anchor-archive", type=Path, required=True)
    prefix.add_argument("--anchor-inflate-py", type=Path, required=True)
    prefix.add_argument("--anchor-inflate-sh", type=Path, required=True)
    prefix.add_argument("--output-dir", type=Path, required=True)
    prefix.add_argument("--plan-json", required=True)
    prefix.set_defaults(func=build_prefix)
    coder = sub.add_parser("coder-race")
    coder.add_argument("--anchor-archive", type=Path, required=True)
    coder.add_argument("--output", type=Path, required=True)
    coder.set_defaults(func=coder_race)
    overlap = sub.add_parser("r3-overlap")
    overlap.add_argument("--miss-receipt", type=Path, required=True)
    overlap.add_argument("--donor", type=Path, required=True)
    overlap.add_argument("--output", type=Path, required=True)
    overlap.set_defaults(func=r3_overlap)
    composition = sub.add_parser("compose")
    composition.add_argument("--anchor-score", type=Path, required=True)
    composition.add_argument("--anchor-archive", type=Path, required=True)
    composition.add_argument("--jrd-score", type=Path, required=True)
    composition.add_argument("--jrd-build", type=Path, required=True)
    composition.add_argument("--requant-score", type=Path, required=True)
    composition.add_argument("--requant-build", type=Path, required=True)
    composition.add_argument("--coder-race", type=Path, required=True)
    composition.add_argument("--r3-overlap", type=Path, required=True)
    composition.add_argument("--pointer", type=float, required=True)
    composition.add_argument("--output", type=Path, required=True)
    composition.set_defaults(func=compose)
    return top


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
