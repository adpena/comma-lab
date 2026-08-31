#!/usr/bin/env python3
"""Characterize and measure horizon-shape successors on GF1's retained n600 field.

This instrument is deliberately scorer-free.  Its authority surface is token-field
``count_nonzero`` against the lb1-lineage field and the retained DALI GT field.  Phase
``characterize`` must reproduce GF1's 1,325,033 mismatch control before it emits any
derived result.  Every materialized mask/curve is retained under APDataStore.

The measurement phase is added only after the characterization chooses the candidate
family; this keeps the charter's residual-first ordering executable rather than prose.
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

import numpy as np
from scipy.interpolate import PchipInterpolator

REPO = Path(__file__).resolve().parents[1]
for root in (REPO, REPO / "src", REPO / "experiments"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from experiments import ddm_hg1_heterogeneous_analytic_generator_gate as hg1

SCHEMA = "ddm_hzs1_horizon_shape.v1"
N_PAIRS, HEIGHT, WIDTH = 600, 384, 512
FIELD_BYTES = N_PAIRS * HEIGHT * WIDTH
EXPECTED_CONTROL = 1_325_033
EXPECTED_DALI_CONTROL = 1_324_976
MIN_FREE_BYTES = 512 * 1024 * 1024
MIN_MEASURE_FREE_BYTES = 2 * 1024 * 1024 * 1024
GF1_PACKET_BYTES = 47_603
REPLACEMENT_BAR_BYTES = 85_020
GF1_CORRECTION_PRICE = 0.2909

AP_ROOT = Path("/Volumes/APDataStore/pact")
GF1_ROOT = AP_ROOT / "ddm_gf1_generator_form_on_lb1_field"
DEFAULT_OUT = AP_ROOT / "ddm_hzs1_horizon_shape"
INPUTS = {
    "lb1": (
        AP_ROOT / "ddm_dc1_20260816/retained/redecoded_tokens_n600.u8",
        117_964_800,
        "9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52",
    ),
    "dali_gt": (
        AP_ROOT / "ddm_bz2_bornsmall_capacity_ceiling/retained/targets/dali_gt_full_n600.u8",
        117_964_800,
        "a98b90678ca5d4e12b385d2c8596839b368af8d52277eea3c1d3666f7a4c9b3d",
    ),
    "gf1_generated": (
        GF1_ROOT / "retained/generated_from_lb1_fit.u8",
        117_964_800,
        "4026c4e2c805beb5b79be2879bb4a84311655d0d7d80dbc766654847522a5d19",
    ),
    "gf1_horizon": (
        GF1_ROOT / "retained/road_undrivable.raw",
        20_447,
        "91160e4dea0e3f155f4bb5bfcfca214df872cb68164e10e7aa35ae4bd4b6adb9",
    ),
    "gf1_lane": (
        GF1_ROOT / "retained/lane.raw",
        159_395,
        "370ed7e1aa302cfb74bc7be085c65333bf73f74b76884f968762ae2d09c74eef",
    ),
    "gf1_movable": (
        GF1_ROOT / "retained/movable.raw",
        24_515,
        "1d1da79d361f48a21172499e91c48f26b8dab213ccd2ccd6b54d41b0567f8f2a",
    ),
    "gf1_mycar": (
        GF1_ROOT / "retained/mycar.raw",
        24_589,
        "3efa81c8f6744afb46acb09d0c7a6f382a33237541b02af3dfdb545fa86de032",
    ),
}

HZS_MAGIC = b"HZS1"
HZS_HEADER = struct.Struct("<4sBBHHHH")
HZS_VERSION = 1
HZS_KIND_LINEAR = 1
HZS_KIND_NEAREST = 2
HZS_KIND_PCHIP = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: object) -> None:
    atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def current_git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def verify_storage(out: Path, phase: str) -> dict[str, int]:
    resolved = out.resolve()
    if AP_ROOT.resolve() not in resolved.parents:
        raise SystemExit(f"REFUSE: output must remain under APDataStore, got {resolved}")
    usage = shutil.disk_usage(AP_ROOT)
    required = MIN_MEASURE_FREE_BYTES if phase == "measure" else MIN_FREE_BYTES
    if usage.free < required:
        raise SystemExit(
            f"REFUSE: APDataStore free bytes {usage.free:,} < required {required:,}"
        )
    return {"total": usage.total, "used": usage.used, "free": usage.free, "required": required}


def verify_inputs() -> dict[str, dict[str, object]]:
    facts: dict[str, dict[str, object]] = {}
    for name, (path, expected_bytes, expected_sha) in INPUTS.items():
        if not path.is_file():
            raise SystemExit(f"REFUSE: missing input {name}: {path}")
        fact = file_fact(path)
        if fact["bytes"] != expected_bytes or fact["sha256"] != expected_sha:
            raise SystemExit(
                f"REFUSE: input identity drift for {name}: {fact}; "
                f"expected bytes={expected_bytes} sha256={expected_sha}"
            )
        facts[name] = fact
    return facts


def decode_gf1_horizon_curves(payload: bytes) -> np.ndarray:
    magic, version, pairs, height, width, knots = hg1.HORIZON_HEADER.unpack_from(payload)
    expected = (
        hg1.HORIZON_MAGIC,
        1,
        N_PAIRS,
        HEIGHT,
        WIDTH,
        hg1.HORIZON_X.size,
    )
    if (magic, version, pairs, height, width, knots) != expected:
        raise SystemExit("REFUSE: retained GF1 horizon header drifted")
    cursor = hg1.HORIZON_HEADER.size
    xs = np.frombuffer(payload, dtype="<u2", count=knots, offset=cursor).astype(np.float64)
    cursor += 2 * knots
    rows = np.frombuffer(payload, dtype="<u2", count=pairs * knots, offset=cursor)
    rows = rows.reshape(pairs, knots)
    if cursor + 2 * pairs * knots != len(payload):
        raise SystemExit("REFUSE: retained GF1 horizon has trailing/truncated bytes")
    x_full = np.arange(WIDTH, dtype=np.float64)
    curves = np.empty((N_PAIRS, WIDTH), dtype=np.uint16)
    for pair in range(N_PAIRS):
        curves[pair] = np.clip(np.rint(np.interp(x_full, xs, rows[pair])), 0, HEIGHT)
    return curves


def optimal_visible_bulk_thresholds(
    target: np.ndarray, generated: np.ndarray, base_curves: np.ndarray
) -> np.ndarray:
    """Exact per-column threshold minimizer on GF1-visible Road/Undrivable pixels."""

    thresholds = np.empty((N_PAIRS, WIDTH), dtype=np.uint16)
    y = np.arange(HEIGHT + 1, dtype=np.int32)[:, None]
    for pair in range(N_PAIRS):
        frame = np.asarray(target[pair])
        rendered = np.asarray(generated[pair])
        visible = (rendered == hg1.CLASS_ROAD) | (rendered == hg1.CLASS_UNDRIVABLE)
        road = ((frame == hg1.CLASS_ROAD) & visible).astype(np.int32)
        undrivable = ((frame == hg1.CLASS_UNDRIVABLE) & visible).astype(np.int32)
        road_above = np.vstack((np.zeros((1, WIDTH), dtype=np.int32), np.cumsum(road, axis=0)))
        und_above = np.vstack(
            (np.zeros((1, WIDTH), dtype=np.int32), np.cumsum(undrivable, axis=0))
        )
        cost = road_above + (und_above[-1:] - und_above)
        # Cost is primary.  Ties go to the retained GF1 curve, minimizing needless
        # video-derived movement and therefore giving the rate leg its best shot.
        tie_distance = np.abs(y - base_curves[pair].astype(np.int32)[None, :])
        thresholds[pair] = np.argmin(cost * (HEIGHT + 1) + tie_distance, axis=0).astype(
            np.uint16
        )
    return thresholds


def packed_mask_fact(path: Path, mask: np.ndarray) -> dict[str, object]:
    payload = np.packbits(mask.reshape(-1), bitorder="little").tobytes()
    atomic_bytes(path, payload)
    fact = file_fact(path)
    fact["logical_shape"] = list(mask.shape)
    fact["bitorder"] = "little"
    fact["true_count"] = int(np.count_nonzero(mask))
    return fact


def compare_field(
    generated_path: Path, target_path: Path, mask_path: Path
) -> dict[str, object]:
    """Count and retain an exact packed mismatch field without a 118 MB temporary."""

    generated = np.memmap(
        generated_path, mode="r", dtype=np.uint8, shape=(N_PAIRS, HEIGHT, WIDTH)
    )
    target = np.memmap(target_path, mode="r", dtype=np.uint8, shape=(N_PAIRS, HEIGHT, WIDTH))
    confusion = np.zeros((5, 5), dtype=np.int64)
    temporary = mask_path.with_name(f".{mask_path.name}.{os.getpid()}.tmp")
    mismatch_count = 0
    with temporary.open("wb") as handle:
        for start in range(0, N_PAIRS, 16):
            generated_chunk = np.asarray(generated[start : start + 16])
            target_chunk = np.asarray(target[start : start + 16])
            mismatch = generated_chunk != target_chunk
            mismatch_count += int(np.count_nonzero(mismatch))
            encoded = target_chunk.astype(np.int16) * 5 + generated_chunk.astype(np.int16)
            confusion += np.bincount(encoded.reshape(-1), minlength=25).reshape(5, 5)
            handle.write(np.packbits(mismatch.reshape(-1), bitorder="little").tobytes())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, mask_path)
    per_true_class_wrong = [
        int(confusion[label].sum() - confusion[label, label]) for label in range(5)
    ]
    return {
        "mismatches": mismatch_count,
        "per_true_class_wrong": per_true_class_wrong,
        "confusion_true_rows_generated_columns": confusion.tolist(),
        "mask": {
            **file_fact(mask_path),
            "logical_shape": [N_PAIRS, HEIGHT, WIDTH],
            "bitorder": "little",
            "true_count": mismatch_count,
        },
    }


def concentration(values: np.ndarray, fraction: float) -> float:
    count = max(1, int(np.ceil(values.size * fraction)))
    total = int(values.sum())
    if total == 0:
        return 0.0
    return float(np.sort(values)[-count:].sum() / total)


def polynomial_rms(residual: np.ndarray, degree: int) -> float:
    x = np.linspace(-1.0, 1.0, WIDTH)
    squared = 0.0
    for row in residual:
        fitted = np.polynomial.polynomial.polyval(
            x, np.polynomial.polynomial.polyfit(x, row.astype(np.float64), degree)
        )
        squared += float(np.square(row - fitted).sum())
    return float(np.sqrt(squared / residual.size))


def encode_hgh1(xs: np.ndarray, rows: np.ndarray) -> bytes:
    if xs.shape != (17,) or rows.shape != (N_PAIRS, 17):
        raise ValueError("HGH1 candidates require exactly 17 knots")
    return (
        hg1.HORIZON_HEADER.pack(hg1.HORIZON_MAGIC, 1, N_PAIRS, HEIGHT, WIDTH, 17)
        + xs.astype("<u2", copy=False).tobytes()
        + rows.astype("<u2", copy=False).tobytes()
    )


def encode_hzs1(kind: int, xs: np.ndarray, rows: np.ndarray) -> bytes:
    if kind not in (HZS_KIND_LINEAR, HZS_KIND_NEAREST, HZS_KIND_PCHIP):
        raise ValueError(f"unknown HZS1 interpolation kind {kind}")
    if xs.ndim != 1 or rows.shape != (N_PAIRS, xs.size) or xs.size < 2:
        raise ValueError("HZS1 knot/row shape mismatch")
    return (
        HZS_HEADER.pack(HZS_MAGIC, HZS_VERSION, kind, N_PAIRS, HEIGHT, WIDTH, xs.size)
        + xs.astype("<u2", copy=False).tobytes()
        + rows.astype("<u2", copy=False).tobytes()
    )


def decode_hzs1(payload: bytes) -> tuple[int, np.ndarray, np.ndarray]:
    if len(payload) < HZS_HEADER.size:
        raise hg1.HG1Error("HZS1 horizon payload truncated")
    magic, version, kind, pairs, height, width, knots = HZS_HEADER.unpack_from(payload)
    if (
        magic != HZS_MAGIC
        or version != HZS_VERSION
        or kind not in (HZS_KIND_LINEAR, HZS_KIND_NEAREST, HZS_KIND_PCHIP)
        or (pairs, height, width) != (N_PAIRS, HEIGHT, WIDTH)
        or knots < 2
    ):
        raise hg1.HG1Error("HZS1 horizon header mismatch")
    cursor = HZS_HEADER.size
    xs = np.frombuffer(payload, dtype="<u2", count=knots, offset=cursor).astype(np.int32)
    cursor += 2 * knots
    rows = np.frombuffer(payload, dtype="<u2", count=pairs * knots, offset=cursor)
    rows = rows.reshape(pairs, knots).astype(np.int32)
    if (
        cursor + 2 * pairs * knots != len(payload)
        or xs[0] != 0
        or xs[-1] != WIDTH - 1
        or np.any(np.diff(xs) <= 0)
        or np.any(rows > HEIGHT)
    ):
        raise hg1.HG1Error("HZS1 horizon payload length/range/order mismatch")
    return kind, xs, rows


def render_hzs1(payload: bytes, output: np.ndarray) -> None:
    kind, xs, rows = decode_hzs1(payload)
    x_full = np.arange(WIDTH, dtype=np.float64)
    if kind == HZS_KIND_LINEAR:
        curves = np.empty((N_PAIRS, WIDTH), dtype=np.int32)
        for pair in range(N_PAIRS):
            curves[pair] = np.rint(np.interp(x_full, xs, rows[pair])).astype(np.int32)
    elif kind == HZS_KIND_NEAREST:
        edges = (xs[:-1].astype(np.float64) + xs[1:].astype(np.float64)) / 2.0
        indices = np.searchsorted(edges, x_full, side="left")
        curves = rows[:, indices]
    else:
        curves = np.rint(PchipInterpolator(xs, rows, axis=1)(x_full)).astype(np.int32)
    curves = np.clip(curves, 0, HEIGHT)
    y_grid = np.arange(HEIGHT, dtype=np.int32)[:, None]
    for pair in range(N_PAIRS):
        output[pair] = np.where(
            y_grid >= curves[pair][None, :], hg1.CLASS_ROAD, hg1.CLASS_UNDRIVABLE
        )


def adaptive_knots(curves: np.ndarray, count: int) -> np.ndarray:
    """Greedily place shared x knots at the largest aggregate curve residual."""

    chosen = [0, WIDTH - 1]
    x_full = np.arange(WIDTH, dtype=np.float64)
    while len(chosen) < count:
        xs = np.asarray(sorted(chosen), dtype=np.int32)
        fitted = np.empty_like(curves, dtype=np.float64)
        for pair in range(N_PAIRS):
            fitted[pair] = np.interp(x_full, xs, curves[pair, xs])
        score = np.abs(curves.astype(np.float64) - fitted).sum(axis=0)
        score[xs] = -1.0
        chosen.append(int(np.argmax(score)))
    return np.asarray(sorted(chosen), dtype=np.int32)


def parse_generator_accounting_packet(packet: bytes) -> dict[str, bytes]:
    """Exact parse-back for GF1's four-stream accounting-only packet."""

    if len(packet) < hg1.PACKET_HEADER.size:
        raise hg1.HG1Error("generator accounting packet truncated")
    magic, version, count, reserved = hg1.PACKET_HEADER.unpack_from(packet)
    if (
        magic != hg1.PACKET_MAGIC
        or version != hg1.PACKET_VERSION
        or count != len(hg1.GENERATOR_STREAMS)
        or reserved
    ):
        raise hg1.HG1Error("generator accounting packet header mismatch")
    cursor = hg1.PACKET_HEADER.size
    rows = []
    for _ in range(count):
        if cursor + hg1.PACKET_ROW.size > len(packet):
            raise hg1.HG1Error("generator accounting packet roster truncated")
        rows.append(hg1.PACKET_ROW.unpack_from(packet, cursor))
        cursor += hg1.PACKET_ROW.size
    streams: dict[str, bytes] = {}
    expected_ids = [hg1.STREAM_IDS[name] for name in hg1.GENERATOR_STREAMS]
    if [row[0] for row in rows] != expected_ids:
        raise hg1.HG1Error("generator accounting packet roster/order mismatch")
    for stream_id, coder_id, raw_size, coded_size, raw_sha, coded_sha in rows:
        if coder_id not in hg1.et1.CODER_NAMES:
            raise hg1.HG1Error("generator accounting packet coder enum invalid")
        coded = packet[cursor : cursor + coded_size]
        cursor += coded_size
        if len(coded) != coded_size or hg1.sha256_bytes(coded) != coded_sha.hex():
            raise hg1.HG1Error("generator accounting packet coded identity mismatch")
        raw = hg1.et1.decompress_payload(coded, hg1.et1.CODER_NAMES[coder_id])
        if len(raw) != raw_size or hg1.sha256_bytes(raw) != raw_sha.hex():
            raise hg1.HG1Error("generator accounting packet raw identity mismatch")
        streams[hg1.ID_STREAMS[stream_id]] = raw
    if cursor != len(packet) or tuple(streams) != hg1.GENERATOR_STREAMS:
        raise hg1.HG1Error("generator accounting packet roster/trailing-byte mismatch")
    return streams


def render_candidate_field(
    streams: dict[str, bytes], *, custom_horizon: bool, output_path: Path
) -> dict[str, object]:
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    output = np.memmap(
        temporary, mode="w+", dtype=np.uint8, shape=(N_PAIRS, HEIGHT, WIDTH)
    )
    if custom_horizon:
        render_hzs1(streams["road_undrivable"], output)
    else:
        hg1.render_horizon(streams["road_undrivable"], output)
    hg1.render_lane(streams["lane"], output)
    hg1.render_movable(streams["movable"], output)
    hg1.render_mycar(streams["mycar"], output)
    output.flush()
    del output
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)
    return file_fact(output_path)


def characterize(out: Path, argv: list[str], input_facts: dict[str, dict[str, object]]) -> int:
    retained = out / "retained"
    retained.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    lb1_path = INPUTS["lb1"][0]
    gt_path = INPUTS["dali_gt"][0]
    generated_path = INPUTS["gf1_generated"][0]
    lb1 = np.memmap(lb1_path, mode="r", dtype=np.uint8, shape=(N_PAIRS, HEIGHT, WIDTH))
    gt = np.memmap(gt_path, mode="r", dtype=np.uint8, shape=(N_PAIRS, HEIGHT, WIDTH))
    generated = np.memmap(
        generated_path, mode="r", dtype=np.uint8, shape=(N_PAIRS, HEIGHT, WIDTH)
    )

    # Mandatory first control.  Nothing derived is written until both counts pass.
    full_mismatch = np.asarray(generated != lb1)
    control = int(np.count_nonzero(full_mismatch))
    dali_mismatch = np.asarray(generated != gt)
    dali_control = int(np.count_nonzero(dali_mismatch))
    if control != EXPECTED_CONTROL:
        raise SystemExit(f"REFUSE: GF1 control {control:,} != {EXPECTED_CONTROL:,}")
    if dali_control != EXPECTED_DALI_CONTROL:
        raise SystemExit(
            f"REFUSE: DALI control {dali_control:,} != {EXPECTED_DALI_CONTROL:,}"
        )
    print(f"[hzs1] controls PASS: lb1={control:,} dali_gt={dali_control:,}", flush=True)

    base_curves = decode_gf1_horizon_curves(INPUTS["gf1_horizon"][0].read_bytes())
    optimal_curves = optimal_visible_bulk_thresholds(lb1, generated, base_curves)
    base_path = retained / "base_horizon_thresholds_n600x512.u16"
    optimal_path = retained / "optimal_visible_bulk_thresholds_n600x512.u16"
    atomic_bytes(base_path, base_curves.astype("<u2", copy=False).tobytes())
    atomic_bytes(optimal_path, optimal_curves.astype("<u2", copy=False).tobytes())

    bulk = (lb1 == hg1.CLASS_ROAD) | (lb1 == hg1.CLASS_UNDRIVABLE)
    visible = (generated == hg1.CLASS_ROAD) | (generated == hg1.CLASS_UNDRIVABLE)
    horizon_owned = full_mismatch & bulk
    exposed_shape = horizon_owned & visible
    overlay_occlusion = horizon_owned & ~visible
    if int(horizon_owned.sum()) != 893_436:
        raise SystemExit(
            f"REFUSE: horizon-owned control {int(horizon_owned.sum()):,} != 893,436"
        )
    if int(exposed_shape.sum() + overlay_occlusion.sum()) != 893_436:
        raise SystemExit("REFUSE: horizon decomposition does not close")

    mask_facts = {
        "full_mismatch": packed_mask_fact(retained / "full_mismatch_lb1.packbits", full_mismatch),
        "horizon_owned": packed_mask_fact(
            retained / "horizon_owned_mismatch_lb1.packbits", horizon_owned
        ),
        "exposed_shape": packed_mask_fact(
            retained / "horizon_exposed_shape_mismatch_lb1.packbits", exposed_shape
        ),
        "overlay_occlusion": packed_mask_fact(
            retained / "horizon_overlay_occlusion_lb1.packbits", overlay_occlusion
        ),
        "dali_full_mismatch": packed_mask_fact(
            retained / "full_mismatch_dali_gt.packbits", dali_mismatch
        ),
    }

    per_frame = horizon_owned.sum(axis=(1, 2)).astype(np.int64)
    per_column = horizon_owned.sum(axis=(0, 1)).astype(np.int64)
    per_row = horizon_owned.sum(axis=(0, 2)).astype(np.int64)
    exposed_per_frame = exposed_shape.sum(axis=(1, 2)).astype(np.int64)
    scene_rows = []
    for scene in range(10):
        sl = slice(scene * 60, (scene + 1) * 60)
        scene_rows.append(
            {
                "scene": scene,
                "pair_start": scene * 60,
                "pair_stop_exclusive": (scene + 1) * 60,
                "horizon_owned": int(per_frame[sl].sum()),
                "exposed_shape": int(exposed_per_frame[sl].sum()),
            }
        )

    residual = optimal_curves.astype(np.int32) - base_curves.astype(np.int32)
    scene_profile = np.empty_like(residual)
    for scene in range(10):
        sl = slice(scene * 60, (scene + 1) * 60)
        scene_profile[sl] = np.rint(np.median(residual[sl], axis=0)).astype(np.int32)
    residual_after_scene = residual - scene_profile

    row_bands = []
    for start in range(0, HEIGHT, 16):
        row_bands.append(
            {
                "row_start": start,
                "row_stop_exclusive": min(start + 16, HEIGHT),
                "horizon_owned": int(per_row[start : start + 16].sum()),
                "exposed_shape": int(exposed_shape[:, start : start + 16].sum()),
            }
        )
    column_bands = []
    for start in range(0, WIDTH, 32):
        column_bands.append(
            {
                "column_start": start,
                "column_stop_exclusive": min(start + 32, WIDTH),
                "horizon_owned": int(per_column[start : start + 32].sum()),
                "exposed_shape": int(exposed_shape[:, :, start : start + 32].sum()),
            }
        )

    result = {
        "schema": SCHEMA,
        "phase": "characterize",
        "axis": "[macOS-CPU scorer-free exact count]",
        "score_claim": False,
        "promotable": False,
        "n_pairs": N_PAIRS,
        "field_positions": FIELD_BYTES,
        "command": argv,
        "git_head": current_git_head(),
        "inputs": input_facts,
        "controls": {
            "gf1_vs_lb1_expected": EXPECTED_CONTROL,
            "gf1_vs_lb1_measured": control,
            "gf1_vs_dali_expected": EXPECTED_DALI_CONTROL,
            "gf1_vs_dali_measured": dali_control,
            "passed": True,
        },
        "residual_decomposition": {
            "horizon_owned": int(horizon_owned.sum()),
            "exposed_shape": int(exposed_shape.sum()),
            "overlay_occlusion": int(overlay_occlusion.sum()),
            "exposed_fraction_of_horizon_owned": float(exposed_shape.sum() / horizon_owned.sum()),
            "overlay_fraction_of_horizon_owned": float(
                overlay_occlusion.sum() / horizon_owned.sum()
            ),
        },
        "concentration": {
            "top_10pct_frames_share": concentration(per_frame, 0.10),
            "top_20pct_frames_share": concentration(per_frame, 0.20),
            "top_10pct_columns_share": concentration(per_column, 0.10),
            "top_20pct_columns_share": concentration(per_column, 0.20),
        },
        "per_scene_60_pairs": scene_rows,
        "row_bands_16": row_bands,
        "column_bands_32": column_bands,
        "curve_displacement_diagnostic": {
            "definition": "exact visible-bulk threshold minus retained GF1 interpolated threshold",
            "rms_px": float(np.sqrt(np.mean(np.square(residual)))),
            "mean_abs_px": float(np.mean(np.abs(residual))),
            "per_frame_shift_rms_px": polynomial_rms(residual, 0),
            "per_frame_affine_rms_px": polynomial_rms(residual, 1),
            "per_frame_quadratic_rms_px": polynomial_rms(residual, 2),
            "per_frame_cubic_rms_px": polynomial_rms(residual, 3),
            "scene60_profile_rms_px": float(np.sqrt(np.mean(np.square(residual_after_scene)))),
            "second_difference_rms_px": float(
                np.sqrt(np.mean(np.square(np.diff(residual, n=2, axis=1))))
            ),
        },
        "curve_payloads": {
            "base": file_fact(base_path),
            "optimal_visible_bulk": file_fact(optimal_path),
        },
        "mask_payloads": mask_facts,
        "elapsed_seconds": time.monotonic() - started,
    }
    result_path = out / "CHARACTERIZATION.json"
    atomic_json(result_path, result)
    checkpoint = {
        "schema": "ddm_hzs1_stage_checkpoint.v1",
        "stage": "characterize",
        "complete": True,
        "all_payloads_retained": True,
        "result": file_fact(result_path),
        "next_stage": "select parameterizations from the measured residual",
    }
    atomic_json(out / "STAGE_characterize_COMPLETE.json", checkpoint)
    print(f"[hzs1] retained characterization at {result_path}", flush=True)
    return 0


def exact_control_counts() -> tuple[int, int]:
    lb1 = np.memmap(
        INPUTS["lb1"][0], mode="r", dtype=np.uint8, shape=(N_PAIRS, HEIGHT, WIDTH)
    )
    gt = np.memmap(
        INPUTS["dali_gt"][0], mode="r", dtype=np.uint8, shape=(N_PAIRS, HEIGHT, WIDTH)
    )
    generated = np.memmap(
        INPUTS["gf1_generated"][0], mode="r", dtype=np.uint8, shape=(N_PAIRS, HEIGHT, WIDTH)
    )
    lb1_count = 0
    dali_count = 0
    for start in range(0, N_PAIRS, 16):
        chunk = np.asarray(generated[start : start + 16])
        lb1_count += int(np.count_nonzero(chunk != np.asarray(lb1[start : start + 16])))
        dali_count += int(np.count_nonzero(chunk != np.asarray(gt[start : start + 16])))
    return lb1_count, dali_count


def build_candidate_specs(
    base_curves: np.ndarray, optimal_curves: np.ndarray
) -> list[dict[str, object]]:
    if base_curves.shape != optimal_curves.shape:
        raise RuntimeError("curve shapes drifted")
    fixed17 = hg1.HORIZON_X.astype(np.int32)
    adaptive17 = adaptive_knots(optimal_curves, 17)
    specs: list[dict[str, object]] = [
        {
            "name": "retained_gf1_17_linear_control",
            "payload": INPUTS["gf1_horizon"][0].read_bytes(),
            "knots": 17,
            "interpolation": "linear",
            "knot_layout": "retained_fixed32",
            "custom_horizon": False,
            "video_derived_scalar_dof": N_PAIRS * 17,
        },
        {
            "name": "objective17_linear",
            "payload": encode_hgh1(fixed17, optimal_curves[:, fixed17]),
            "knots": 17,
            "interpolation": "linear",
            "knot_layout": "fixed32",
            "custom_horizon": False,
            "video_derived_scalar_dof": N_PAIRS * 17,
        },
        {
            "name": "adaptive17_linear",
            "payload": encode_hgh1(adaptive17, optimal_curves[:, adaptive17]),
            "knots": 17,
            "interpolation": "linear",
            "knot_layout": f"residual_greedy:{adaptive17.tolist()}",
            "custom_horizon": False,
            "video_derived_scalar_dof": N_PAIRS * 17 + 17,
        },
        {
            "name": "objective17_nearest",
            "payload": encode_hzs1(HZS_KIND_NEAREST, fixed17, optimal_curves[:, fixed17]),
            "knots": 17,
            "interpolation": "nearest",
            "knot_layout": "fixed32",
            "custom_horizon": True,
            "video_derived_scalar_dof": N_PAIRS * 17,
        },
        {
            "name": "objective17_pchip",
            "payload": encode_hzs1(HZS_KIND_PCHIP, fixed17, optimal_curves[:, fixed17]),
            "knots": 17,
            "interpolation": "pchip",
            "knot_layout": "fixed32",
            "custom_horizon": True,
            "video_derived_scalar_dof": N_PAIRS * 17,
        },
    ]
    for knots in (33, 65, 129, 257, 512):
        xs = np.rint(np.linspace(0, WIDTH - 1, knots)).astype(np.int32)
        if np.unique(xs).size != knots:
            raise RuntimeError(f"uniform {knots}-knot layout is not unique")
        specs.append(
            {
                "name": f"objective{knots}_linear",
                "payload": encode_hzs1(HZS_KIND_LINEAR, xs, optimal_curves[:, xs]),
                "knots": knots,
                "interpolation": "linear",
                "knot_layout": "uniform_endpoints",
                "custom_horizon": True,
                "video_derived_scalar_dof": N_PAIRS * knots,
            }
        )
    if len(specs) != 10:
        raise RuntimeError("candidate denominator drift")
    return specs


def fact_is_current(fact: dict[str, object]) -> bool:
    path = Path(str(fact["path"]))
    return (
        path.is_file()
        and path.stat().st_size == int(fact["bytes"])
        and sha256_file(path) == str(fact["sha256"])
    )


def load_completed_candidate(candidate_dir: Path) -> dict[str, object] | None:
    checkpoint_path = candidate_dir / "STAGE_candidate_COMPLETE.json"
    result_path = candidate_dir / "RESULT.json"
    if not checkpoint_path.is_file() or not result_path.is_file():
        return None
    result = json.loads(result_path.read_text())
    required_facts = (
        result["payloads"]["horizon_raw"],
        result["payloads"]["accounting_packet"],
        result["payloads"]["generated"],
        result["comparisons"]["lb1"]["mask"],
        result["comparisons"]["dali_gt"]["mask"],
    )
    if not all(fact_is_current(fact) for fact in required_facts):
        raise SystemExit(f"REFUSE: completed candidate custody drift at {candidate_dir}")
    return result


def seed_fixed_coder_receipts(out: Path, candidate_dir: Path) -> None:
    """Reuse the measured control's byte-identical fixed-stream coder payloads."""

    control_root = out / "candidates/01_retained_gf1_17_linear_control/retained/coder_races"
    for stream_name in ("lane", "movable", "mycar"):
        for coder in hg1.CODERS:
            for filename in ("payload.coded", "payload.repeat.coded"):
                source = control_root / stream_name / coder / filename
                if not source.is_file():
                    raise SystemExit(f"REFUSE: missing fixed-stream control receipt {source}")
                destination = (
                    candidate_dir / "retained/coder_races" / stream_name / coder / filename
                )
                if destination.is_file():
                    continue
                atomic_bytes(destination, source.read_bytes())


def measure_candidate(out: Path, spec: dict[str, object], ordinal: int) -> dict[str, object]:
    name = str(spec["name"])
    candidate_dir = out / "candidates" / f"{ordinal:02d}_{name}"
    completed = load_completed_candidate(candidate_dir)
    if completed is not None:
        print(f"[hzs1] resume candidate {ordinal:02d}/10 {name}", flush=True)
        return completed
    retained = candidate_dir / "retained"
    stream_dir = retained / "streams"
    stream_dir.mkdir(parents=True, exist_ok=True)
    horizon_path = stream_dir / "road_undrivable.raw"
    atomic_bytes(horizon_path, bytes(spec["payload"]))
    stream_paths = {"road_undrivable": horizon_path}
    for fixed_name in ("lane", "movable", "mycar"):
        fixed_path = stream_dir / f"{fixed_name}.raw"
        atomic_bytes(fixed_path, INPUTS[f"gf1_{fixed_name}"][0].read_bytes())
        stream_paths[fixed_name] = fixed_path

    if ordinal > 1:
        seed_fixed_coder_receipts(out, candidate_dir)

    races = [
        hg1.coder_race(stream_name, stream_paths[stream_name], candidate_dir)
        for stream_name in hg1.GENERATOR_STREAMS
    ]
    packet_path = retained / "generator_accounting_packet.hg1p"
    packet_fact = hg1.build_packet(races, packet_path)
    streams = parse_generator_accounting_packet(packet_path.read_bytes())
    for stream_name, stream_path in stream_paths.items():
        if streams[stream_name] != stream_path.read_bytes():
            raise hg1.HG1Error(f"{name}: accounting packet raw parse-back drift")

    generated_path = retained / "generated.u8"
    generated_fact = render_candidate_field(
        streams,
        custom_horizon=bool(spec["custom_horizon"]),
        output_path=generated_path,
    )
    lb1_comparison = compare_field(
        generated_path, INPUTS["lb1"][0], retained / "mismatch_lb1.packbits"
    )
    dali_comparison = compare_field(
        generated_path, INPUTS["dali_gt"][0], retained / "mismatch_dali_gt.packbits"
    )
    confusion = np.asarray(lb1_comparison["confusion_true_rows_generated_columns"])
    per_true = list(lb1_comparison["per_true_class_wrong"])
    horizon_true_class = int(per_true[hg1.CLASS_ROAD] + per_true[hg1.CLASS_UNDRIVABLE])
    causal_cross_swap = int(
        confusion[hg1.CLASS_ROAD, hg1.CLASS_UNDRIVABLE]
        + confusion[hg1.CLASS_UNDRIVABLE, hg1.CLASS_ROAD]
    )
    overlay_intrusion = horizon_true_class - causal_cross_swap
    mismatch = int(lb1_comparison["mismatches"])
    packet_bytes = int(packet_fact["bytes"])
    projected_bytes = packet_bytes + GF1_CORRECTION_PRICE * mismatch
    horizon_race = races[0]
    horizon_winner = str(horizon_race["winner"])
    horizon_coded = int(horizon_race["coders"][horizon_winner]["coded"]["bytes"])
    result = {
        "schema": SCHEMA,
        "phase": "measure_candidate",
        "candidate_ordinal": ordinal,
        "candidate_denominator": 10,
        "candidate": {key: value for key, value in spec.items() if key != "payload"},
        "receiver_surface": (
            "HZS1 prototype parser/renderer in this instrument"
            if spec["custom_horizon"]
            else "existing hg1 HGH1 parser/renderer"
        ),
        "accounting_packet_boundary": (
            "GF1-equivalent four-generator accounting packet; exact parse-back passed here; "
            "not receiver-valid under hg1.parse_packet, which requires the residual fifth stream"
        ),
        "races": races,
        "payloads": {
            "horizon_raw": file_fact(horizon_path),
            "accounting_packet": packet_fact,
            "generated": generated_fact,
        },
        "comparisons": {"lb1": lb1_comparison, "dali_gt": dali_comparison},
        "decomposition": {
            "horizon_true_class_wrong": horizon_true_class,
            "causal_road_undrivable_cross_swap": causal_cross_swap,
            "later_stream_intrusion_into_true_horizon_classes": overlay_intrusion,
        },
        "rate_distortion_projection": {
            "formula": "packet_B + 0.2909 * mismatches",
            "horizon_winner": horizon_winner,
            "horizon_raw_bytes": int(horizon_path.stat().st_size),
            "horizon_coded_bytes": horizon_coded,
            "packet_bytes": packet_bytes,
            "packet_delta_vs_gf1_bytes": packet_bytes - GF1_PACKET_BYTES,
            "projected_total_bytes": projected_bytes,
            "ratio_to_85020_bar": projected_bytes / REPLACEMENT_BAR_BYTES,
            "attribution_oracle_projected_bytes": (
                packet_bytes + GF1_CORRECTION_PRICE * (mismatch - horizon_true_class)
            ),
            "attribution_plus_true_lane_oracle_projected_bytes": (
                packet_bytes
                + GF1_CORRECTION_PRICE * (mismatch - horizon_true_class - int(per_true[1]))
            ),
            "causal_horizon_perfect_projected_bytes": (
                packet_bytes + GF1_CORRECTION_PRICE * (mismatch - causal_cross_swap)
            ),
        },
    }
    result_path = candidate_dir / "RESULT.json"
    atomic_json(result_path, result)
    atomic_json(
        candidate_dir / "STAGE_candidate_COMPLETE.json",
        {
            "schema": "ddm_hzs1_stage_checkpoint.v1",
            "stage": f"candidate_{ordinal:02d}_{name}",
            "complete": True,
            "all_payloads_retained": True,
            "result": file_fact(result_path),
        },
    )
    print(
        f"[hzs1] candidate {ordinal:02d}/10 {name}: mismatch={mismatch:,} "
        f"packet={packet_bytes:,} projected={projected_bytes:,.1f} B",
        flush=True,
    )
    return result


def measure(out: Path, argv: list[str], input_facts: dict[str, dict[str, object]]) -> int:
    started = time.monotonic()
    control, dali_control = exact_control_counts()
    if (control, dali_control) != (EXPECTED_CONTROL, EXPECTED_DALI_CONTROL):
        raise SystemExit(
            f"REFUSE: measure controls {(control, dali_control)} != "
            f"{(EXPECTED_CONTROL, EXPECTED_DALI_CONTROL)}"
        )
    print(
        f"[hzs1] controls PASS before candidate loop: lb1={control:,} dali={dali_control:,}",
        flush=True,
    )
    characterization_path = out / "CHARACTERIZATION.json"
    if not characterization_path.is_file():
        raise SystemExit("REFUSE: characterization must complete before measurement")
    retained = out / "retained"
    base_path = retained / "base_horizon_thresholds_n600x512.u16"
    optimal_path = retained / "optimal_visible_bulk_thresholds_n600x512.u16"
    base_curves = np.fromfile(base_path, dtype="<u2").reshape(N_PAIRS, WIDTH)
    optimal_curves = np.fromfile(optimal_path, dtype="<u2").reshape(N_PAIRS, WIDTH)
    specs = build_candidate_specs(base_curves, optimal_curves)
    results = [measure_candidate(out, spec, ordinal) for ordinal, spec in enumerate(specs, 1)]

    rows = []
    for result in results:
        projection = result["rate_distortion_projection"]
        decomposition = result["decomposition"]
        rows.append(
            {
                "ordinal": result["candidate_ordinal"],
                "name": result["candidate"]["name"],
                "knots": result["candidate"]["knots"],
                "interpolation": result["candidate"]["interpolation"],
                "video_derived_scalar_dof": result["candidate"]["video_derived_scalar_dof"],
                "horizon_raw_bytes": projection["horizon_raw_bytes"],
                "horizon_coded_bytes": projection["horizon_coded_bytes"],
                "packet_bytes": projection["packet_bytes"],
                "packet_delta_vs_gf1_bytes": projection["packet_delta_vs_gf1_bytes"],
                "lb1_mismatches": result["comparisons"]["lb1"]["mismatches"],
                "dali_gt_mismatches": result["comparisons"]["dali_gt"]["mismatches"],
                "horizon_true_class_wrong": decomposition["horizon_true_class_wrong"],
                "causal_cross_swap": decomposition["causal_road_undrivable_cross_swap"],
                "overlay_intrusion": decomposition[
                    "later_stream_intrusion_into_true_horizon_classes"
                ],
                "projected_total_bytes": projection["projected_total_bytes"],
                "ratio_to_85020_bar": projection["ratio_to_85020_bar"],
            }
        )
    baseline = rows[0]
    if (
        baseline["lb1_mismatches"] != EXPECTED_CONTROL
        or baseline["dali_gt_mismatches"] != EXPECTED_DALI_CONTROL
        or baseline["packet_bytes"] != GF1_PACKET_BYTES
        or results[0]["payloads"]["generated"]["sha256"] != INPUTS["gf1_generated"][2]
    ):
        raise SystemExit(f"REFUSE: retained GF1 candidate failed exact control: {baseline}")
    eligible = [row for row in rows if row["packet_delta_vs_gf1_bytes"] <= 2_000]
    best_eligible = min(eligible, key=lambda row: row["projected_total_bytes"])
    best_any = min(rows, key=lambda row: row["projected_total_bytes"])
    result = {
        "schema": SCHEMA,
        "phase": "measure",
        "axis": "[macOS-CPU scorer-free exact count and real generic coders]",
        "score_claim": False,
        "promotable": False,
        "command": argv,
        "git_head": current_git_head(),
        "inputs": input_facts,
        "characterization": file_fact(characterization_path),
        "controls": {
            "gf1_vs_lb1_measured_before_candidate_loop": control,
            "gf1_vs_dali_measured_before_candidate_loop": dali_control,
            "passed": True,
        },
        "candidate_denominator": {
            "enumerated_after_characterization": len(specs),
            "measured": len(rows),
            "missing": [],
        },
        "rows": rows,
        "best_with_packet_delta_le_2000": best_eligible,
        "best_enumerated": best_any,
        "scope_of_ceiling": (
            "enumerated single-threshold-per-column horizon families only; fixed downstream paint; "
            "not a ceiling for multi-interval masks or jointly refit downstream streams"
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    result_path = out / "MEASUREMENT.json"
    atomic_json(result_path, result)
    atomic_json(
        out / "STAGE_measure_COMPLETE.json",
        {
            "schema": "ddm_hzs1_stage_checkpoint.v1",
            "stage": "measure",
            "complete": True,
            "all_payloads_retained": True,
            "result": file_fact(result_path),
            "candidate_results_retained": len(rows),
        },
    )
    print(f"[hzs1] retained complete measurement at {result_path}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("characterize", "measure"), default="characterize")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--resume-from",
        type=Path,
        required=True,
        help="durable stage root; first launch and resumes use the same APDataStore path",
    )
    args = parser.parse_args(argv)
    if args.resume_from.resolve() != args.out.resolve():
        raise SystemExit("REFUSE: --resume-from must equal --out for byte-identical stage custody")
    storage = verify_storage(args.out, args.phase)
    args.out.mkdir(parents=True, exist_ok=True)
    input_facts = verify_inputs()
    control, dali_control = exact_control_counts()
    if (control, dali_control) != (EXPECTED_CONTROL, EXPECTED_DALI_CONTROL):
        raise SystemExit(
            f"REFUSE: first control {(control, dali_control)} != "
            f"{(EXPECTED_CONTROL, EXPECTED_DALI_CONTROL)}"
        )
    complete = args.out / f"STAGE_{args.phase}_COMPLETE.json"
    if complete.is_file():
        print(f"[hzs1] resume: stage already complete at {complete}; refusing duplicate materialization")
        return 0
    command = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    atomic_json(
        args.out / f"RUN_PROVENANCE_{args.phase}.json",
        {
            "schema": "ddm_hzs1_run_provenance.v1",
            "phase": args.phase,
            "command": command,
            "git_head": current_git_head(),
            "storage_preflight": storage,
            "inputs": input_facts,
            "first_control": {
                "gf1_vs_lb1": control,
                "gf1_vs_dali_gt": dali_control,
                "passed": True,
            },
        },
    )
    if args.phase == "characterize":
        return characterize(args.out, command, input_facts)
    return measure(args.out, command, input_facts)


if __name__ == "__main__":
    raise SystemExit(main())
