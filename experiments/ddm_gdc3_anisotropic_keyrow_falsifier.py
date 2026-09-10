#!/usr/bin/env python3
"""Run the scorer-free n600 anisotropic key-row ribbon falsifier.

The counted program stores exact label rows on one of four fixed vertical
schedules. The generic receiver repeats the most recent anchor row. All
program, coder, render, and selected exact-residual payloads are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
for _root in (REPO, REPO / "src", REPO / "experiments"):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from experiments import ddm_et1_edge_topology_container_gate as et1
from experiments import ddm_gdc3_geometry_law_probe as geometry
from experiments import ddm_hg1_heterogeneous_analytic_generator_gate as hg1

AXIS: Final = "[macOS-CPU advisory / scorer-free exact byte measurement]"
SEED: Final = 20_260_910
DOOR_BYTES: Final = 94_010
MIN_FREE_BYTES: Final = 1_500_000_000
TARGET_PATH: Final = geometry.TARGET[0]
TARGET_SHA256: Final = geometry.TARGET[1]
GEOMETRY_RESULT: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_gdc3_next_construction_against_the_geometry_law/geometry_probe_v1/RESULT.json"
)
GEOMETRY_RESULT_SHA256: Final = "9513573ec8cb2eadb072d966e4217e80e3b5aa4970bf6dfc335339c56cab9476"
DEFAULT_OUTPUT: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_gdc3_next_construction_against_the_geometry_law/keyrow_falsifier_v1"
)
VARIANTS: Final = ("uniform_2", "uniform_4", "bands_8_4_2", "bands_16_8_4_2")
VARIANT_IDS: Final = {name: index + 1 for index, name in enumerate(VARIANTS)}
ID_VARIANTS: Final = {value: key for key, value in VARIANT_IDS.items()}

RAW_MAGIC: Final = b"G3KRRW1!"
RAW_HEADER: Final = struct.Struct("<8sBBHHHH")
PACKET_MAGIC: Final = b"G3KRPK1!"
PACKET_HEADER: Final = struct.Struct("<8sBBBII32s32s")


class KeyRowError(RuntimeError):
    """A key-row source, packet, render, residual, or custody invariant failed."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return geometry.sha256_file(path)


def file_fact(path: Path) -> dict[str, Any]:
    return geometry.file_fact(path)


def anchor_rows(variant: str) -> np.ndarray:
    if variant == "uniform_2":
        rows = range(0, geometry.HEIGHT, 2)
    elif variant == "uniform_4":
        rows = range(0, geometry.HEIGHT, 4)
    elif variant == "bands_8_4_2":
        rows = (*range(0, 128, 8), *range(128, 256, 4), *range(256, 384, 2))
    elif variant == "bands_16_8_4_2":
        rows = (
            *range(0, 96, 16),
            *range(96, 192, 8),
            *range(192, 288, 4),
            *range(288, 384, 2),
        )
    else:
        raise KeyRowError(f"unknown fixed schedule: {variant}")
    output = np.asarray(tuple(rows), dtype=np.uint16)
    if output[0] != 0 or np.any(np.diff(output.astype(np.int32)) <= 0):
        raise KeyRowError(f"fixed schedule is not canonical: {variant}")
    return output


def expand_key_rows(key_rows: np.ndarray, anchors: np.ndarray, height: int) -> np.ndarray:
    values = np.asarray(key_rows, dtype=np.uint8)
    rows = np.asarray(anchors, dtype=np.int64)
    if values.ndim != 3 or values.shape[1] != len(rows):
        raise ValueError("key rows must have shape (pairs, anchors, width)")
    if not len(rows) or rows[0] != 0 or rows[-1] >= height or np.any(np.diff(rows) <= 0):
        raise ValueError("anchors must be sorted, unique, start at zero, and fit the height")
    output = np.empty((values.shape[0], height, values.shape[2]), dtype=np.uint8)
    for index, start in enumerate(rows.tolist()):
        stop = int(rows[index + 1]) if index + 1 < len(rows) else height
        output[:, start:stop, :] = values[:, index : index + 1, :]
    return output


def build_raw_program(target: np.ndarray, variant: str) -> bytes:
    anchors = anchor_rows(variant)
    key_rows = np.asarray(target[:, anchors, :], dtype=np.uint8)
    return b"".join(
        (
            RAW_HEADER.pack(
                RAW_MAGIC,
                1,
                VARIANT_IDS[variant],
                geometry.N_PAIRS,
                geometry.HEIGHT,
                geometry.WIDTH,
                len(anchors),
            ),
            key_rows.tobytes(order="C"),
        )
    )


def parse_raw_program(payload: bytes) -> dict[str, Any]:
    if len(payload) < RAW_HEADER.size:
        raise KeyRowError("raw program is truncated")
    magic, version, variant_id, pairs, height, width, count = RAW_HEADER.unpack_from(payload)
    if magic != RAW_MAGIC or version != 1 or variant_id not in ID_VARIANTS or (pairs, height, width) != geometry.SHAPE:
        raise KeyRowError("raw program header is invalid")
    variant = ID_VARIANTS[variant_id]
    anchors = anchor_rows(variant)
    if count != len(anchors):
        raise KeyRowError("raw program anchor count disagrees with its fixed schedule")
    expected = RAW_HEADER.size + pairs * count * width
    if len(payload) != expected:
        raise KeyRowError(f"raw program has {len(payload)} B, expected {expected}")
    key_rows = np.frombuffer(payload, dtype=np.uint8, offset=RAW_HEADER.size).reshape(pairs, count, width)
    if int(np.max(key_rows)) > 4:
        raise KeyRowError("raw program class escaped 0..4")
    return {"variant": variant, "anchors": anchors, "key_rows": key_rows}


def build_packet(raw: bytes, variant: str, coder: str, coded: bytes) -> bytes:
    return (
        PACKET_HEADER.pack(
            PACKET_MAGIC,
            1,
            VARIANT_IDS[variant],
            et1.CODER_IDS[coder],
            len(raw),
            len(coded),
            bytes.fromhex(sha256_bytes(raw)),
            bytes.fromhex(sha256_bytes(coded)),
        )
        + coded
    )


def parse_packet(packet: bytes) -> dict[str, Any]:
    if len(packet) < PACKET_HEADER.size:
        raise KeyRowError("program packet is truncated")
    magic, version, variant_id, coder_id, raw_size, coded_size, raw_sha, coded_sha = PACKET_HEADER.unpack_from(packet)
    coded = packet[PACKET_HEADER.size :]
    if (
        magic != PACKET_MAGIC
        or version != 1
        or variant_id not in ID_VARIANTS
        or coder_id not in et1.CODER_NAMES
        or len(coded) != coded_size
        or sha256_bytes(coded) != coded_sha.hex()
    ):
        raise KeyRowError("program packet header or coded identity is invalid")
    coder = et1.CODER_NAMES[coder_id]
    raw = et1.decompress_payload(coded, coder)
    if len(raw) != raw_size or sha256_bytes(raw) != raw_sha.hex():
        raise KeyRowError("program packet raw identity is invalid")
    program = parse_raw_program(raw)
    if program["variant"] != ID_VARIANTS[variant_id]:
        raise KeyRowError("packet and raw-program variants disagree")
    return {**program, "coder": coder, "raw": raw, "coded": coded}


def retain_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    return geometry.retain_bytes(path, payload)


def race_program(raw: bytes, variant: str, root: Path) -> tuple[dict[str, Any], str, bytes]:
    rows: dict[str, Any] = {}
    for coder in hg1.CODERS:
        directory = root / "retained" / variant / "program_coder_race" / coder
        coded = et1.compress_payload(raw, coder)
        repeated = et1.compress_payload(raw, coder)
        coded_fact = retain_bytes(directory / "payload.coded", coded)
        repeat_fact = retain_bytes(directory / "payload.repeat.coded", repeated)
        if coded != repeated or et1.decompress_payload(coded, coder) != raw:
            raise KeyRowError(f"program coder repeat/parse-back failed: {variant}/{coder}")
        rows[coder] = {
            "coded": coded_fact,
            "repeat": repeat_fact,
            "deterministic_repeat_equal": True,
            "raw_parseback_equal": True,
        }
    winner = min(
        hg1.CODERS,
        key=lambda coder: (int(rows[coder]["coded"]["bytes"]), hg1.CODERS.index(coder)),
    )
    return rows, winner, Path(rows[winner]["coded"]["path"]).read_bytes()


def render_packet(packet: bytes, output_path: Path) -> dict[str, Any]:
    program = parse_packet(packet)
    anchors = program["anchors"]
    key_rows = program["key_rows"]
    if output_path.exists():
        if not output_path.is_file() or output_path.stat().st_size != geometry.FIELD_BYTES:
            raise KeyRowError(f"retained render has wrong identity: {output_path}")
        output = np.memmap(output_path, dtype=np.uint8, mode="r", shape=geometry.SHAPE)
        for index, start in enumerate(anchors.tolist()):
            stop = int(anchors[index + 1]) if index + 1 < len(anchors) else geometry.HEIGHT
            if not np.all(output[:, start:stop, :] == key_rows[:, index : index + 1, :]):
                raise KeyRowError(f"retained render disagrees with receiver: {output_path}")
        del output
        return file_fact(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    output = np.memmap(temporary, dtype=np.uint8, mode="w+", shape=geometry.SHAPE)
    for index, start in enumerate(anchors.tolist()):
        stop = int(anchors[index + 1]) if index + 1 < len(anchors) else geometry.HEIGHT
        output[:, start:stop, :] = key_rows[:, index : index + 1, :]
    output.flush()
    del output
    os.replace(temporary, output_path)
    return file_fact(output_path)


def geometry_rates() -> dict[str, float]:
    if file_fact(GEOMETRY_RESULT)["sha256"] != GEOMETRY_RESULT_SHA256:
        raise KeyRowError("pinned geometry receipt identity changed")
    receipt = json.loads(GEOMETRY_RESULT.read_text(encoding="utf-8"))
    source = next(row for row in receipt["sources"] if row["source"] == "gdc2_stageC_lam1_0.0003_final")
    rates: dict[str, float] = {}
    for row in source["rows"]:
        if row["scope"] != "geometry_marginal" or not row["mismatches"]:
            continue
        rates[str(row["geometry"])] = min(float(order["winning_bytes_per_mismatch"]) for order in row["orders"])
    if set(rates) != set(geometry.GEOMETRY_NAMES):
        raise KeyRowError("geometry receipt lacks one or more selection rates")
    return rates


def mismatch_summary(data: dict[str, np.ndarray]) -> dict[str, Any]:
    total = len(data["address"])
    by_class = np.bincount(data["target_class"].astype(np.int64), minlength=5)
    by_geometry = np.bincount(data["geometry"].astype(np.int64), minlength=6)
    return {
        "total": total,
        "fraction": total / geometry.FIELD_BYTES,
        "by_target_class": {name: int(by_class[index]) for index, name in enumerate(geometry.CLASS_NAMES)},
        "by_geometry": {name: int(by_geometry[index]) for index, name in enumerate(geometry.GEOMETRY_NAMES)},
    }


def fit_variant(target: np.ndarray, variant: str, rates: dict[str, float], root: Path) -> dict[str, Any]:
    raw = build_raw_program(target, variant)
    raw_fact = retain_bytes(root / "retained" / variant / "program.raw", raw)
    coder_rows, winner, coded = race_program(raw, variant, root)
    packet = build_packet(raw, variant, winner, coded)
    packet_fact = retain_bytes(root / "retained" / variant / "program.packet", packet)
    packet_repeat = retain_bytes(root / "retained" / variant / "program.repeat.packet", packet)
    render_fact = render_packet(packet, root / "retained" / variant / "receiver_render.u8")
    render = np.memmap(render_fact["path"], dtype=np.uint8, mode="r", shape=geometry.SHAPE)
    data = geometry.collect_mismatches(target, render)
    summary = mismatch_summary(data)
    projected_residual = sum(summary["by_geometry"][name] * rates[name] for name in geometry.GEOMETRY_NAMES)
    result = {
        "variant": variant,
        "anchors": anchor_rows(variant).astype(int).tolist(),
        "anchor_count": len(anchor_rows(variant)),
        "raw_program": raw_fact,
        "program_coder_race": coder_rows,
        "program_winner": winner,
        "packet": packet_fact,
        "packet_repeat": packet_repeat,
        "packet_parseback_exact": parse_packet(packet)["variant"] == variant,
        "receiver_render": render_fact,
        "mismatches": summary,
        "selection_projection": {
            "label": "DERIVED selection-only; non-additive geometry-table transfer",
            "source": file_fact(GEOMETRY_RESULT),
            "geometry_bytes_per_mismatch": rates,
            "projected_residual_bytes": projected_residual,
            "packet_plus_projected_residual_bytes": int(packet_fact["bytes"]) + projected_residual,
        },
    }
    et1.atomic_json(root / f"STAGE_{variant}.json", result)
    del data
    del render
    return result


def exact_residual(target: np.ndarray, selected: dict[str, Any], root: Path) -> dict[str, Any]:
    render_path = Path(selected["receiver_render"]["path"])
    render = np.memmap(render_path, dtype=np.uint8, mode="r", shape=geometry.SHAPE)
    data = geometry.collect_mismatches(target, render)
    rows = []
    for order in geometry.REQUIRED_ORDERS:
        payload = geometry.serialize_residual(data["address"], data["target_class"], order)
        raw_path = root / "retained" / selected["variant"] / "exact_residual" / order / "residual.raw"
        raw_fact = retain_bytes(raw_path, payload)
        geometry.verify_subset_payload(payload, data["address"], target.reshape(-1), order)
        race = geometry.coder_race_strict(f"keyrow__{selected['variant']}__exact_residual__{order}", raw_path, root)
        winner = str(race["winner"])
        coded = race["coders"][winner]["coded"]
        rows.append(
            {
                "order": order,
                "raw": raw_fact,
                "coders": race["coders"],
                "winner": winner,
                "winning_coded_bytes": int(coded["bytes"]),
                "winning_bytes_per_mismatch": int(coded["bytes"]) / len(data["address"]),
                "subset_parseback_exact": True,
            }
        )
    best = min(rows, key=lambda row: (row["winning_coded_bytes"], row["order"]))
    closure = geometry.close_full_residual(
        render,
        TARGET_PATH,
        Path(best["raw"]["path"]).read_bytes(),
        root / "retained" / selected["variant"] / "exact_closure" / "field.u8",
    )
    total = int(selected["packet"]["bytes"]) + int(best["winning_coded_bytes"])
    del data
    del render
    return {
        "orders": rows,
        "best_order": best["order"],
        "best_coder": best["winner"],
        "mismatches": int(selected["mismatches"]["total"]),
        "real_residual_bytes": int(best["winning_coded_bytes"]),
        "real_bytes_per_mismatch": int(best["winning_coded_bytes"]) / int(selected["mismatches"]["total"]),
        "packet_bytes": int(selected["packet"]["bytes"]),
        "packet_plus_real_residual_bytes": total,
        "door_bytes": DOOR_BYTES,
        "margin_bytes": DOOR_BYTES - total,
        "ratio_to_door": total / DOOR_BYTES,
        "gate_pass": total <= DOOR_BYTES,
        "exact_target_closure": closure,
    }


def manifest_rows(root: Path) -> list[dict[str, Any]]:
    manifest = root / "MANIFEST.json"
    return [file_fact(path) for path in sorted(item for item in root.rglob("*") if item.is_file() and item != manifest)]


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    root: Path = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    if free < MIN_FREE_BYTES:
        raise KeyRowError(f"storage preflight refused: {free} B < {MIN_FREE_BYTES} B")
    started = time.monotonic()
    target = geometry.verified_field(TARGET_PATH, TARGET_SHA256)
    rates = geometry_rates()
    variants = [fit_variant(target, variant, rates, root) for variant in VARIANTS]
    selected = min(
        variants,
        key=lambda row: (
            float(row["selection_projection"]["packet_plus_projected_residual_bytes"]),
            VARIANTS.index(str(row["variant"])),
        ),
    )
    residual = exact_residual(target, selected, root)
    result = {
        "schema": "ddm_gdc3_anisotropic_keyrow_falsifier.v1",
        "axis": AXIS,
        "score_claim": False,
        "research_only": True,
        "selection_mode": "full n600, no subset; four fixed schedules; selection by pinned geometry table",
        "seed": SEED,
        "rng_used": False,
        "target": file_fact(TARGET_PATH),
        "geometry_receipt": file_fact(GEOMETRY_RESULT),
        "packet_accounting": {
            "counted": "variant ID, source-derived anchor rows, coder framing and hashes",
            "free": "fixed schedules, parser, zero-order-hold receiver, physical coder algorithm",
        },
        "variants": variants,
        "selected_variant": selected["variant"],
        "exact_falsifier": residual,
        "verdict": "GATE_PASS" if residual["gate_pass"] else "FORMULATION_NO_GO",
        "verdict_scope": "four registered fixed anisotropic key-row zero-order-hold schedules",
        "implementation": file_fact(Path(__file__).resolve()),
        "git_commit_at_measurement": git_commit(),
        "host": os.uname().nodename,
        "elapsed_seconds": time.monotonic() - started,
        "storage": {
            "root": str(root),
            "free_bytes_at_preflight": free,
            "minimum_required_bytes": MIN_FREE_BYTES,
            "deletion": "none",
            "atomic_writes": True,
        },
    }
    result_path = root / "RESULT.json"
    et1.atomic_json(result_path, result)
    et1.atomic_json(
        root / "LATEST.json",
        {"schema": "ddm_gdc3_anisotropic_keyrow_falsifier.latest.v1", "result": file_fact(result_path)},
    )
    rows = manifest_rows(root)
    et1.atomic_json(
        root / "MANIFEST.json",
        {
            "schema": "ddm_gdc3_anisotropic_keyrow_falsifier.manifest.v1",
            "root": str(root),
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "files": rows,
        },
    )
    print(
        json.dumps(
            {
                "selected_variant": selected["variant"],
                "packet_bytes": residual["packet_bytes"],
                "mismatches": residual["mismatches"],
                "real_residual_bytes": residual["real_residual_bytes"],
                "packet_plus_real_residual_bytes": residual["packet_plus_real_residual_bytes"],
                "door_bytes": DOOR_BYTES,
                "margin_bytes": residual["margin_bytes"],
                "gate_pass": residual["gate_pass"],
                "result": file_fact(result_path),
                "manifest": file_fact(root / "MANIFEST.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
