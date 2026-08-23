#!/usr/bin/env python3
"""Retain a denominator-complete DX2 token-bit attribution map.

The instrument replays the shipped CPU receiver/coder law over all 600 pairs.
For every one of the 117,964,800 decoded symbols it retains the exact selected
RC64 integer-frequency cost, the packed RR4 context, the FX2 mixer cell, the
MA1 within-miss cell, and the decoded symbol.  Twenty-frame stage payloads and
complete receiver checkpoints make the replay crash-resumable.

Final analysis joins those aligned fields to DALI GT class, three canonical
vertical bands, pair index, exact top-cost masks, and MST1's joinable gross
native-render manufactured-error support.  The per-symbol cost field must hash
equal BL1's independently retained field and reconcile to the physical stream
with the same explicitly bounded arithmetic termination residual.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experiments import ddm_bl1_per_position_bit_allocation as bl1

AP_ROOT = Path("/Volumes/APDataStore/pact")
DEFAULT_STORE = AP_ROOT / "ddm_tb2_token_bit_attribution" / "measurement_v1"

BL1_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_bl1_per_position_bit_allocation/measurement_v1")
BL1_COST = BL1_ROOT / "retained/fields/position_rc64_frequency_cost_bits.f64le.bin"
BL1_TOKENS = BL1_ROOT / "retained/fields/decoded_tokens_instrumented.u8"
BL1_RESULT = BL1_ROOT / "RESULT.json"
WJ1_ROOT = REPO / ".omx/tmp/arm_receipts_local/ddm_wj1_cost_error_position_join/measurement_v1"
TOP1_MASK = WJ1_ROOT / "retained/threshold_masks/top_1pct.n600.packbits"
TOP10_MASK = WJ1_ROOT / "retained/threshold_masks/top_10pct.n600.packbits"
MANUFACTURED_MASK = WJ1_ROOT / "retained/inputs/gross_manufactured_native_render_head.n600.packbits"
GT_FIELD = Path("/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/retained/inputs/gt_argmax_n600.npy")

N = bl1.N
HEIGHT = bl1.HEIGHT
WIDTH = bl1.WIDTH
PLANE = bl1.PLANE
POSITIONS = bl1.POSITIONS
CLASSES = bl1.CLASSES
STAGE_FRAMES = bl1.STAGE_FRAMES
FIELD_DTYPE = np.dtype("<f8")
CONTEXT_DTYPE = np.dtype("<u2")
CELL_DTYPE = np.dtype("<u2")
PACKED_FRAME_BYTES = PLANE // 8
STREAM_BITS = bl1.STREAM_BITS
CONTEXT_SIZE = 5 * 64 * 2 * 2 * 8 * 5
MIXER_SIZE = 4_000
MISS_SIZE = 6**4
CHECKPOINT_SCHEMA = "ddm_tb2_stage_checkpoint.v1"
ESTIMATED_ARTIFACT_BYTES = 4_200_000_000
RESERVE_BYTES = 4_000_000_000
MIN_FREE_BYTES = ESTIMATED_ARTIFACT_BYTES + RESERVE_BYTES
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
BANDS = (
    ("sky_undriv_top", 0, 144),
    ("road_lane_midband", 144, 288),
    ("mycar_hood_bottom", 288, 384),
)

EXPECTED = {
    "cost_bytes": POSITIONS * FIELD_DTYPE.itemsize,
    "cost_sha256": "99d7833d55a9aa128f67cfc125a10ba90cffaa714de0b88801aa43b8b16e2c86",
    "tokens_bytes": POSITIONS,
    "tokens_sha256": bl1.EXPECTED["tokens_sha256"],
    "gt_bytes": 117_964_928,
    "gt_sha256": "91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248",
    "top1_bytes": POSITIONS // 8,
    "top1_sha256": "f48cd9d61c4580dda23dc1ff4c7504009612863760ad962c578c190114ce0bdf",
    "top10_bytes": POSITIONS // 8,
    "top10_sha256": "827000beaee7bd13491d22330b4c5e43096a1754a0164232fb48c94f347efbff",
    "manufactured_bytes": POSITIONS // 8,
    "manufactured_sha256": "b756ca948f5db3dd085a61803e24c5a90db946d89ea1894ba552f024a74b1d5d",
    "manufactured_positions": 28_602,
    "physical_stream_bytes": bl1.EXPECTED["stream_bytes"],
    "physical_stream_sha256": bl1.EXPECTED["stream_sha256"],
}


class Tb2Error(RuntimeError):
    """Fail-closed custody, resume, instrumentation, or accounting error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_file(path: Path, expected_sha: str, expected_bytes: int) -> dict[str, object]:
    if not path.is_file():
        raise Tb2Error(f"required file is absent: {path}")
    fact = file_fact(path)
    if fact["sha256"] != expected_sha or fact["bytes"] != expected_bytes:
        raise Tb2Error(
            f"custody drift for {path}: got {fact['bytes']} B {fact['sha256']}, "
            f"expected {expected_bytes} B {expected_sha}"
        )
    return fact


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        with tmp.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        with tmp.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_raw(path: Path, arrays: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        with tmp.open("wb") as handle:
            for array in arrays:
                handle.write(np.ascontiguousarray(array).tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + f".partial.{os.getpid()}")
    try:
        with source.open("rb") as reader, tmp.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=1 << 22)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def source_binding() -> dict[str, object]:
    bl1_binding = bl1.source_binding()
    return {
        "schema": "ddm_tb2_source_binding.v1",
        "shape": [N, HEIGHT, WIDTH],
        "positions": POSITIONS,
        "axis": "[macOS-CPU advisory / scorer-free shipped-coder instrumentation]",
        "sources": {
            "bl1_source_binding": bl1_binding,
            "bl1_cost_positive_control": verify_file(BL1_COST, EXPECTED["cost_sha256"], EXPECTED["cost_bytes"]),
            "bl1_tokens_positive_control": verify_file(BL1_TOKENS, EXPECTED["tokens_sha256"], EXPECTED["tokens_bytes"]),
            "bl1_result": file_fact(BL1_RESULT),
            "dali_gt": verify_file(GT_FIELD, EXPECTED["gt_sha256"], EXPECTED["gt_bytes"]),
            "top_1pct_mask": verify_file(TOP1_MASK, EXPECTED["top1_sha256"], EXPECTED["top1_bytes"]),
            "top_10pct_mask": verify_file(TOP10_MASK, EXPECTED["top10_sha256"], EXPECTED["top10_bytes"]),
            "gross_manufactured_native_render_mask": verify_file(
                MANUFACTURED_MASK,
                EXPECTED["manufactured_sha256"],
                EXPECTED["manufactured_bytes"],
            ),
            "implementation": file_fact(Path(__file__)),
        },
    }


def stage_paths(store: Path, start: int, end: int) -> dict[str, Path]:
    root = store / "retained/stages" / f"frames_{start:04d}_{end - 1:04d}"
    return {
        "root": root,
        "cost": root / "rc64_frequency_cost_bits.f64.npy",
        "context": root / "rr4_base_context.u16.npy",
        "mixer": root / "fx2_mixer_cell.u16.npy",
        "miss": root / "ma1_within_miss_cell.u16.npy",
        "decoded": root / "decoded_tokens.u8.npy",
        "state": root / "receiver_state.npz",
        "receipt": root / "RECEIPT.json",
    }


def validate_stage(paths: dict[str, Path], binding: dict[str, object], start: int, end: int) -> dict[str, Any]:
    if not paths["receipt"].is_file():
        raise Tb2Error(f"stage receipt is absent: {paths['receipt']}")
    receipt = json.loads(paths["receipt"].read_text())
    if receipt.get("source_binding") != binding:
        raise Tb2Error(f"stage {start}:{end} is bound to different sources")
    if (receipt.get("frame_start"), receipt.get("frame_end")) != (start, end):
        raise Tb2Error(f"stage {start}:{end} bounds drifted")
    shape = (end - start, HEIGHT, WIDTH)
    expected = {
        "cost": FIELD_DTYPE,
        "context": CONTEXT_DTYPE,
        "mixer": CELL_DTYPE,
        "miss": CELL_DTYPE,
        "decoded": np.dtype("u1"),
    }
    for key, dtype in expected.items():
        fact = receipt["artifacts"][key]
        verify_file(paths[key], str(fact["sha256"]), int(fact["bytes"]))
        field = np.load(paths[key], mmap_mode="r", allow_pickle=False)
        if field.shape != shape or field.dtype != dtype:
            raise Tb2Error(f"stage field shape/dtype drift: {paths[key]}")
    state = receipt["artifacts"]["state"]
    verify_file(paths["state"], str(state["sha256"]), int(state["bytes"]))
    return receipt


def completed_stages(store: Path, binding: dict[str, object]) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for start in range(0, N, STAGE_FRAMES):
        end = min(start + STAGE_FRAMES, N)
        paths = stage_paths(store, start, end)
        if not paths["receipt"].exists():
            break
        receipts.append(validate_stage(paths, binding, start, end))
    all_receipts = list((store / "retained/stages").glob("frames_*/RECEIPT.json"))
    if len(all_receipts) != len(receipts):
        raise Tb2Error("stage receipts are not a contiguous prefix")
    return receipts


def save_stage(
    paths: dict[str, Path],
    binding: dict[str, object],
    start: int,
    end: int,
    fields: dict[str, np.ndarray],
    receiver_state: dict[str, np.ndarray],
    decoder_snapshot: np.ndarray,
    elapsed_seconds: float,
) -> dict[str, Any]:
    paths["root"].mkdir(parents=True, exist_ok=True)
    atomic_npy(paths["cost"], np.asarray(fields["cost"], dtype=FIELD_DTYPE))
    atomic_npy(paths["context"], np.asarray(fields["context"], dtype=CONTEXT_DTYPE))
    atomic_npy(paths["mixer"], np.asarray(fields["mixer"], dtype=CELL_DTYPE))
    atomic_npy(paths["miss"], np.asarray(fields["miss"], dtype=CELL_DTYPE))
    atomic_npy(paths["decoded"], np.asarray(fields["decoded"], dtype=np.uint8))
    state_arrays = {
        "schema": np.frombuffer(CHECKPOINT_SCHEMA.encode(), dtype=np.uint8),
        "frame_end": np.asarray([end], dtype=np.int64),
        "decoder": np.asarray(decoder_snapshot, dtype=np.uint64),
        "previous": np.asarray(fields["decoded"][-1], dtype=np.uint8),
        **{f"corrector__{key}": value for key, value in receiver_state.items()},
    }
    atomic_npz(paths["state"], state_arrays)
    artifacts = {key: file_fact(paths[key]) for key in ("cost", "context", "mixer", "miss", "decoded", "state")}
    receipt: dict[str, Any] = {
        "schema": "ddm_tb2_stage_receipt.v1",
        "source_binding": binding,
        "frame_start": start,
        "frame_end": end,
        "positions": (end - start) * PLANE,
        "frequency_cost_bits": float(np.asarray(fields["cost"], dtype=np.float64).sum()),
        "decoder_bit_position": int(decoder_snapshot[4]),
        "corrector_state_arrays": len(receiver_state),
        "elapsed_seconds": elapsed_seconds,
        "artifacts": artifacts,
    }
    atomic_json(paths["receipt"], receipt)
    return receipt


def resume_receiver(runtime: dict[str, Any], receipts: list[dict[str, Any]], store: Path) -> tuple[int, Any]:
    torch = runtime["torch"]
    if not receipts:
        previous = torch.zeros((1, HEIGHT, WIDTH), dtype=torch.long, device=runtime["device"])
        return 0, previous
    end = int(receipts[-1]["frame_end"])
    paths = stage_paths(store, end - STAGE_FRAMES, end)
    with np.load(paths["state"], allow_pickle=False) as payload:
        schema = bytes(payload["schema"]).decode()
        if schema != CHECKPOINT_SCHEMA or int(payload["frame_end"][0]) != end:
            raise Tb2Error("receiver checkpoint schema/frame drifted")
        corrector_payload = {
            key.removeprefix("corrector__"): payload[key].copy()
            for key in payload.files
            if key.startswith("corrector__")
        }
        runtime["jg2"].load_corrector_state(runtime["corrector"], corrector_payload)
        bl1.restore_decoder_state(runtime["decoder"], payload["decoder"])
        previous_np = np.asarray(payload["previous"], dtype=np.uint8).copy()
    previous = torch.from_numpy(previous_np.astype(np.int64)).reshape(1, HEIGHT, WIDTH)
    return end, previous.to(runtime["device"])


def run_instrumented_replay(store: Path, binding: dict[str, object], library: Path) -> list[dict[str, Any]]:
    import torch

    receipts = completed_stages(store, binding)
    runtime = bl1.load_receiver(binding, library)
    start_frame, previous = resume_receiver(runtime, receipts, store)
    truth = np.memmap(BL1_TOKENS, dtype=np.uint8, mode="r", shape=(N, HEIGHT, WIDTH))
    residual = runtime["residual"]
    renderer = runtime["renderer"]
    parts = runtime["parts"]
    model = runtime["model"]
    sparse = runtime["sparse"]
    plans = runtime["plans"]
    corrector = runtime["corrector"]
    decoder = runtime["decoder"]
    device = runtime["device"]
    jg2 = runtime["jg2"]

    with torch.inference_mode():
        for stage_start in range(start_frame, N, STAGE_FRAMES):
            stage_end = min(stage_start + STAGE_FRAMES, N)
            started = time.perf_counter()
            shape = (stage_end - stage_start, HEIGHT, WIDTH)
            fields = {
                "cost": np.empty(shape, dtype=FIELD_DTYPE),
                "context": np.empty(shape, dtype=CONTEXT_DTYPE),
                "mixer": np.empty(shape, dtype=CELL_DTYPE),
                "miss": np.empty(shape, dtype=CELL_DTYPE),
                "decoded": np.empty(shape, dtype=np.uint8),
            }
            for frame in range(stage_start, stage_end):
                offset = frame - stage_start
                index = torch.tensor([frame], dtype=torch.long, device=device)
                current = torch.zeros_like(previous)
                frame_context = model.prepare_frame_context(index, previous)
                if frame:
                    previous_cpu = previous[0].to(device="cpu", dtype=torch.uint8).numpy()
                    boundary = residual._boundary_buckets(previous_cpu).reshape(-1)
                else:
                    boundary = np.full(PLANE, 4, dtype=np.uint8)
                corrector.begin_frame(boundary)
                for group, (device_positions, flat_positions) in enumerate(plans):
                    base_logits = sparse.selected_logits(current, frame_context, group).cpu().numpy()
                    predicted = base_logits.argmax(axis=1).astype(np.int64)
                    feature = boundary[flat_positions].astype(np.int64) * CLASSES + predicted
                    corrected = base_logits + parts.table.values[feature]
                    probability = residual._probability_table(corrected, renderer.HPAC_LOGIT_PRECISION)
                    state = corrector.group_state(probability, predicted, flat_positions)
                    coding = np.asarray(corrector.coding_row(state), dtype=np.float64)
                    pending = corrector._pending
                    miss_pending = corrector._miss_pending
                    if pending is None or miss_pending is None:
                        raise Tb2Error("live corrector did not expose its shipped context cells")
                    mixer = np.asarray(pending["mixer"], dtype=np.int64)
                    miss = np.asarray(miss_pending, dtype=np.int64)
                    context = np.asarray(state.context, dtype=np.int64)
                    if (
                        np.any(context < 0)
                        or np.any(context >= CONTEXT_SIZE)
                        or np.any(mixer < 0)
                        or np.any(mixer >= MIXER_SIZE)
                        or np.any(miss < 0)
                        or np.any(miss >= MISS_SIZE)
                    ):
                        raise Tb2Error("live context coordinate exceeded the frozen declared range")
                    symbols = decoder.decode(coding).astype(np.int64)
                    expected = np.asarray(truth[frame]).reshape(-1)[flat_positions].astype(np.int64)
                    if not np.array_equal(symbols, expected):
                        raise Tb2Error(f"shipped decoder diverged at frame={frame} group={group}")
                    frequency_cost, _ = bl1.rc64_costs(coding, symbols)
                    fields["cost"][offset].reshape(-1)[flat_positions] = frequency_cost
                    fields["context"][offset].reshape(-1)[flat_positions] = context
                    fields["mixer"][offset].reshape(-1)[flat_positions] = mixer
                    fields["miss"][offset].reshape(-1)[flat_positions] = miss
                    corrector.observe(state, symbols)
                    current.reshape(-1)[device_positions] = torch.from_numpy(symbols).to(device)
                fields["decoded"][offset] = current[0].to(device="cpu", dtype=torch.uint8).numpy()
                if not np.array_equal(fields["decoded"][offset], truth[frame]):
                    raise Tb2Error(f"decoded frame {frame} differs from the pinned field")
                corrector.end_frame(fields["decoded"][offset].reshape(-1))
                previous = current
            state_payload = jg2.corrector_state(corrector)
            lost = jg2.uncaptured_divergent_state(corrector, runtime["cold_corrector"], set(state_payload))
            if lost:
                raise Tb2Error(f"checkpoint would lose corrector state: {lost[:8]}")
            decoder_snapshot = bl1.decoder_state(decoder)
            paths = stage_paths(store, stage_start, stage_end)
            receipt = save_stage(
                paths,
                binding,
                stage_start,
                stage_end,
                fields,
                state_payload,
                decoder_snapshot,
                time.perf_counter() - started,
            )
            receipts.append(receipt)
            print(
                json.dumps(
                    {
                        "stage": [stage_start, stage_end],
                        "frequency_bits": receipt["frequency_cost_bits"],
                        "decoder_bit_position": receipt["decoder_bit_position"],
                        "receipt": str(paths["receipt"]),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return receipts


def assemble_fields(store: Path, binding: dict[str, object]) -> dict[str, dict[str, object]]:
    receipts = completed_stages(store, binding)
    if len(receipts) != N // STAGE_FRAMES:
        raise Tb2Error("cannot assemble before all 30 stage receipts exist")
    target = store / "retained/fields"
    target.mkdir(parents=True, exist_ok=True)
    paths = {
        "cost": target / "position_rc64_frequency_cost_bits.f64le.bin",
        "context": target / "position_rr4_base_context.u16le.bin",
        "mixer": target / "position_fx2_mixer_cell.u16le.bin",
        "miss": target / "position_ma1_within_miss_cell.u16le.bin",
        "decoded": target / "decoded_tokens_instrumented.u8",
    }
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in paths}
    for receipt in receipts:
        start = int(receipt["frame_start"])
        end = int(receipt["frame_end"])
        stage = stage_paths(store, start, end)
        for key in arrays:
            arrays[key].append(np.load(stage[key], mmap_mode="r", allow_pickle=False))
    for key, path in paths.items():
        atomic_raw(path, arrays[key])
    facts = {key: file_fact(path) for key, path in paths.items()}
    expected_bytes = {
        "cost": POSITIONS * FIELD_DTYPE.itemsize,
        "context": POSITIONS * CONTEXT_DTYPE.itemsize,
        "mixer": POSITIONS * CELL_DTYPE.itemsize,
        "miss": POSITIONS * CELL_DTYPE.itemsize,
        "decoded": POSITIONS,
    }
    for key, size in expected_bytes.items():
        if facts[key]["bytes"] != size:
            raise Tb2Error(f"assembled {key} field has wrong byte count")
    if facts["cost"]["sha256"] != EXPECTED["cost_sha256"]:
        raise Tb2Error("live replay cost field does not hash-equal BL1")
    if facts["decoded"]["sha256"] != EXPECTED["tokens_sha256"]:
        raise Tb2Error("live replay decoded field does not hash-equal TO2/BL1")
    return facts


def copy_join_fields(store: Path) -> dict[str, dict[str, object]]:
    target = store / "retained/join_fields"
    sources = {
        "dali_gt": GT_FIELD,
        "top_1pct": TOP1_MASK,
        "top_10pct": TOP10_MASK,
        "gross_manufactured_native_render": MANUFACTURED_MASK,
    }
    names = {
        "dali_gt": "gt_argmax_n600.npy",
        "top_1pct": "top_1pct.n600.packbits",
        "top_10pct": "top_10pct.n600.packbits",
        "gross_manufactured_native_render": ("gross_manufactured_native_render_head.n600.packbits"),
    }
    facts: dict[str, dict[str, object]] = {}
    for key, source in sources.items():
        destination = target / names[key]
        atomic_copy(source, destination)
        facts[key] = file_fact(destination)
    expected = {
        "dali_gt": (EXPECTED["gt_bytes"], EXPECTED["gt_sha256"]),
        "top_1pct": (EXPECTED["top1_bytes"], EXPECTED["top1_sha256"]),
        "top_10pct": (EXPECTED["top10_bytes"], EXPECTED["top10_sha256"]),
        "gross_manufactured_native_render": (
            EXPECTED["manufactured_bytes"],
            EXPECTED["manufactured_sha256"],
        ),
    }
    for key, (expected_bytes, expected_sha) in expected.items():
        if facts[key]["bytes"] != expected_bytes or facts[key]["sha256"] != expected_sha:
            raise Tb2Error(f"copied join field drifted during retention: {key}")
    return facts


def packed_frame(path: Path, frame: int) -> np.ndarray:
    packed = np.memmap(
        path,
        dtype=np.uint8,
        mode="r",
        offset=frame * PACKED_FRAME_BYTES,
        shape=(PACKED_FRAME_BYTES,),
    )
    return np.unpackbits(packed, bitorder="little", count=PLANE).astype(bool, copy=False)


def rows_from_axis(
    names: list[str] | None,
    counts: np.ndarray,
    bits: np.ndarray,
    top1_counts: np.ndarray,
    top1_bits: np.ndarray,
    top10_counts: np.ndarray,
    top10_bits: np.ndarray,
    manufactured_counts: np.ndarray,
    manufactured_bits: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for value in range(counts.size):
        if counts[value] == 0:
            continue
        row: dict[str, object] = {
            "value": value,
            "positions": int(counts[value]),
            "position_denominator": POSITIONS,
            "bits": float(bits[value]),
            "bit_denominator": None,
            "bits_per_position": float(bits[value]) / int(counts[value]),
            "top_1pct_positions": int(top1_counts[value]),
            "top_1pct_bits": float(top1_bits[value]),
            "top_10pct_positions": int(top10_counts[value]),
            "top_10pct_bits": float(top10_bits[value]),
            "manufactured_positions": int(manufactured_counts[value]),
            "manufactured_bits": float(manufactured_bits[value]),
        }
        if names is not None:
            row["name"] = names[value]
        rows.append(row)
    return rows


def decode_rr4_context(context: np.ndarray) -> dict[str, np.ndarray]:
    packed = np.asarray(context, dtype=np.int64)
    boundary = packed % 5
    rest = packed // 5
    run = rest % 8
    rest //= 8
    agree2 = rest % 2
    rest //= 2
    agree1 = rest % 2
    rest //= 2
    ubin = rest % 64
    base_class = rest // 64
    return {
        "base_class": base_class,
        "surprise_bin": ubin,
        "agree_t_minus_1": agree1,
        "agree_t_minus_2": agree2,
        "run_level": run,
        "boundary_bucket": boundary,
    }


def concentration_and_gini(cost_path: Path, total_bits: float) -> tuple[float, list[dict[str, object]]]:
    source = np.memmap(cost_path, dtype=FIELD_DTYPE, mode="r", shape=(POSITIONS,))
    sorted_cost = np.sort(np.asarray(source))
    rows = []
    for fraction in (0.01, 0.10):
        count = math.ceil(POSITIONS * fraction)
        bits = float(sorted_cost[-count:].sum(dtype=np.float64))
        rows.append(
            {
                "top_position_fraction": fraction,
                "positions": count,
                "position_denominator": POSITIONS,
                "bits": bits,
                "bit_denominator": total_bits,
                "bit_fraction": bits / total_bits,
                "bytes_equivalent": bits / 8.0,
                "threshold_bits": float(sorted_cost[-count]),
            }
        )
    gini = bl1.weighted_gini(sorted_cost, total_bits)
    return gini, rows


def analyze(
    store: Path,
    binding: dict[str, object],
    fields: dict[str, dict[str, object]],
    joins: dict[str, dict[str, object]],
) -> dict[str, Any]:
    cost = np.memmap(fields["cost"]["path"], dtype=FIELD_DTYPE, mode="r", shape=(N, HEIGHT, WIDTH))
    context = np.memmap(fields["context"]["path"], dtype=CONTEXT_DTYPE, mode="r", shape=(N, HEIGHT, WIDTH))
    mixer = np.memmap(fields["mixer"]["path"], dtype=CELL_DTYPE, mode="r", shape=(N, HEIGHT, WIDTH))
    miss = np.memmap(fields["miss"]["path"], dtype=CELL_DTYPE, mode="r", shape=(N, HEIGHT, WIDTH))
    decoded = np.memmap(fields["decoded"]["path"], dtype=np.uint8, mode="r", shape=(N, HEIGHT, WIDTH))
    gt = np.load(joins["dali_gt"]["path"], mmap_mode="r", allow_pickle=False)
    top1_path = Path(str(joins["top_1pct"]["path"]))
    top10_path = Path(str(joins["top_10pct"]["path"]))
    manufactured_path = Path(str(joins["gross_manufactured_native_render"]["path"]))

    total_bits = float(cost.sum(dtype=np.float64))
    residual_bits = STREAM_BITS - total_bits
    if not (0.0 <= residual_bits < bl1.TERMINAL_OVERHEAD_BOUND_BITS):
        raise Tb2Error("per-symbol costs do not reconcile to the physical stream")
    gini, concentration = concentration_and_gini(Path(str(fields["cost"]["path"])), total_bits)

    axis_sizes = {
        "gt_class": CLASSES,
        "symbol": CLASSES,
        "spatial_band": len(BANDS),
        "rr4_context": CONTEXT_SIZE,
        "fx2_mixer": MIXER_SIZE,
        "ma1_miss": MISS_SIZE,
        "base_class": CLASSES,
        "surprise_bin": 64,
        "agree_t_minus_1": 2,
        "agree_t_minus_2": 2,
        "run_level": 8,
        "boundary_bucket": 5,
    }
    metrics = {
        name: {
            "count": np.zeros(size, dtype=np.int64),
            "bits": np.zeros(size, dtype=np.float64),
            "top1_count": np.zeros(size, dtype=np.int64),
            "top1_bits": np.zeros(size, dtype=np.float64),
            "top10_count": np.zeros(size, dtype=np.int64),
            "top10_bits": np.zeros(size, dtype=np.float64),
            "manufactured_count": np.zeros(size, dtype=np.int64),
            "manufactured_bits": np.zeros(size, dtype=np.float64),
        }
        for name, size in axis_sizes.items()
    }
    pair_bits = np.zeros(N, dtype=np.float64)
    pair_top1_bits = np.zeros(N, dtype=np.float64)
    pair_top10_bits = np.zeros(N, dtype=np.float64)
    pair_manufactured_bits = np.zeros(N, dtype=np.float64)
    pair_manufactured_count = np.zeros(N, dtype=np.int64)
    band_index = np.empty(PLANE, dtype=np.int64)
    for value, (_, start, end) in enumerate(BANDS):
        band_index.reshape(HEIGHT, WIDTH)[start:end, :] = value
    top_candidates: list[np.ndarray] = []
    manufactured_total_count = 0
    manufactured_total_bits = 0.0
    manufactured_top1_count = 0
    manufactured_top1_bits = 0.0
    manufactured_top10_count = 0
    manufactured_top10_bits = 0.0

    axis_names = list(axis_sizes)
    for frame in range(N):
        costs = np.asarray(cost[frame]).reshape(-1)
        gt_values = np.asarray(gt[frame]).reshape(-1).astype(np.int64)
        symbols = np.asarray(decoded[frame]).reshape(-1).astype(np.int64)
        rr4 = np.asarray(context[frame]).reshape(-1).astype(np.int64)
        mixer_values = np.asarray(mixer[frame]).reshape(-1).astype(np.int64)
        miss_values = np.asarray(miss[frame]).reshape(-1).astype(np.int64)
        unpacked = decode_rr4_context(rr4)
        top1 = packed_frame(top1_path, frame)
        top10 = packed_frame(top10_path, frame)
        manufactured = packed_frame(manufactured_path, frame)
        values = {
            "gt_class": gt_values,
            "symbol": symbols,
            "spatial_band": band_index,
            "rr4_context": rr4,
            "fx2_mixer": mixer_values,
            "ma1_miss": miss_values,
            **unpacked,
        }
        for name in axis_names:
            ids = values[name]
            metric = metrics[name]
            size = axis_sizes[name]
            metric["count"] += np.bincount(ids, minlength=size)
            metric["bits"] += np.bincount(ids, weights=costs, minlength=size)
            metric["top1_count"] += np.bincount(ids[top1], minlength=size)
            metric["top1_bits"] += np.bincount(ids[top1], weights=costs[top1], minlength=size)
            metric["top10_count"] += np.bincount(ids[top10], minlength=size)
            metric["top10_bits"] += np.bincount(ids[top10], weights=costs[top10], minlength=size)
            metric["manufactured_count"] += np.bincount(ids[manufactured], minlength=size)
            metric["manufactured_bits"] += np.bincount(ids[manufactured], weights=costs[manufactured], minlength=size)
        pair_bits[frame] = float(costs.sum())
        pair_top1_bits[frame] = float(costs[top1].sum())
        pair_top10_bits[frame] = float(costs[top10].sum())
        pair_manufactured_count[frame] = int(manufactured.sum())
        pair_manufactured_bits[frame] = float(costs[manufactured].sum())
        manufactured_total_count += int(manufactured.sum())
        manufactured_total_bits += float(costs[manufactured].sum())
        both1 = manufactured & top1
        both10 = manufactured & top10
        manufactured_top1_count += int(both1.sum())
        manufactured_top1_bits += float(costs[both1].sum())
        manufactured_top10_count += int(both10.sum())
        manufactured_top10_bits += float(costs[both10].sum())
        candidate = np.flatnonzero(top1)
        if candidate.size:
            dtype = np.dtype(
                [
                    ("flat_index", "<u8"),
                    ("cost_bits", "<f8"),
                    ("gt_class", "u1"),
                    ("symbol", "u1"),
                    ("rr4_context", "<u2"),
                    ("fx2_mixer", "<u2"),
                    ("ma1_miss", "<u2"),
                    ("spatial_band", "u1"),
                    ("manufactured", "u1"),
                ]
            )
            rows = np.empty(candidate.size, dtype=dtype)
            rows["flat_index"] = frame * PLANE + candidate
            rows["cost_bits"] = costs[candidate]
            rows["gt_class"] = gt_values[candidate]
            rows["symbol"] = symbols[candidate]
            rows["rr4_context"] = rr4[candidate]
            rows["fx2_mixer"] = mixer_values[candidate]
            rows["ma1_miss"] = miss_values[candidate]
            rows["spatial_band"] = band_index[candidate]
            rows["manufactured"] = manufactured[candidate]
            top_candidates.append(rows)

    if manufactured_total_count != EXPECTED["manufactured_positions"]:
        raise Tb2Error("manufactured support count drifted")

    aggregate_dir = store / "retained/aggregates"
    axis_rows: dict[str, list[dict[str, object]]] = {}
    names = {
        "gt_class": list(CLASS_NAMES),
        "symbol": [str(index) for index in range(CLASSES)],
        "spatial_band": [row[0] for row in BANDS],
    }
    for name, metric in metrics.items():
        for key, array in metric.items():
            atomic_npy(aggregate_dir / f"{name}__{key}.npy", array)
        rows = rows_from_axis(
            names.get(name),
            metric["count"],
            metric["bits"],
            metric["top1_count"],
            metric["top1_bits"],
            metric["top10_count"],
            metric["top10_bits"],
            metric["manufactured_count"],
            metric["manufactured_bits"],
        )
        for row in rows:
            row["bit_denominator"] = total_bits
        axis_rows[name] = rows
    atomic_npy(aggregate_dir / "pair_bits.f64.npy", pair_bits)
    atomic_npy(aggregate_dir / "pair_manufactured_bits.f64.npy", pair_manufactured_bits)
    atomic_npy(aggregate_dir / "pair_manufactured_count.i64.npy", pair_manufactured_count)

    all_top = np.concatenate(top_candidates)
    order = np.lexsort((all_top["flat_index"], -all_top["cost_bits"]))
    top_positions = all_top[order[:1000]]
    top_positions_path = store / "retained/targets/top_1000_cost_positions.npy"
    atomic_npy(top_positions_path, top_positions)

    top1_row = next(row for row in concentration if row["top_position_fraction"] == 0.01)
    top10_row = next(row for row in concentration if row["top_position_fraction"] == 0.10)
    expected_top1_count = EXPECTED["manufactured_positions"] * 0.01
    expected_top1_bits = float(top1_row["bits"]) * EXPECTED["manufactured_positions"] / POSITIONS
    expected_top10_count = EXPECTED["manufactured_positions"] * 0.10
    expected_top10_bits = float(top10_row["bits"]) * EXPECTED["manufactured_positions"] / POSITIONS
    manufactured_join = {
        "support_definition": (
            "MST1 gross native-render correct-to-wrong transition: decoded L equals DALI GT, "
            "then native render plus frozen head is wrong"
        ),
        "positions": manufactured_total_count,
        "position_denominator": POSITIONS,
        "bits": manufactured_total_bits,
        "bit_denominator": total_bits,
        "bytes_equivalent": manufactured_total_bits / 8.0,
        "top_1pct": {
            "observed_positions": manufactured_top1_count,
            "observed_bits": manufactured_top1_bits,
            "expected_positions_under_independence": expected_top1_count,
            "expected_bits_under_independence": expected_top1_bits,
            "observed_expected_position_ratio": manufactured_top1_count / expected_top1_count,
            "observed_expected_bit_ratio": manufactured_top1_bits / expected_top1_bits,
            "share_of_manufactured_positions": manufactured_top1_count / manufactured_total_count,
            "share_of_manufactured_bits": manufactured_top1_bits / manufactured_total_bits,
        },
        "top_10pct": {
            "observed_positions": manufactured_top10_count,
            "observed_bits": manufactured_top10_bits,
            "expected_positions_under_independence": expected_top10_count,
            "expected_bits_under_independence": expected_top10_bits,
            "observed_expected_position_ratio": manufactured_top10_count / expected_top10_count,
            "observed_expected_bit_ratio": manufactured_top10_bits / expected_top10_bits,
            "share_of_manufactured_positions": manufactured_top10_count / manufactured_total_count,
            "share_of_manufactured_bits": manufactured_top10_bits / manufactured_total_bits,
        },
        "causal_boundary": ("membership association only; no field intervention, re-encode, scorer, or byte saving"),
    }

    pair_rows = [
        {
            "pair_index": frame,
            "positions": PLANE,
            "bits": float(pair_bits[frame]),
            "bit_denominator": total_bits,
            "top_1pct_bits": float(pair_top1_bits[frame]),
            "top_10pct_bits": float(pair_top10_bits[frame]),
            "manufactured_positions": int(pair_manufactured_count[frame]),
            "manufactured_bits": float(pair_manufactured_bits[frame]),
        }
        for frame in range(N)
    ]
    context_top = sorted(axis_rows["rr4_context"], key=lambda row: (-float(row["bits"]), int(row["value"])))[:100]
    mixer_top = sorted(axis_rows["fx2_mixer"], key=lambda row: (-float(row["bits"]), int(row["value"])))[:100]
    miss_top = sorted(axis_rows["ma1_miss"], key=lambda row: (-float(row["bits"]), int(row["value"])))[:100]
    aggregate_facts = {
        str(path.relative_to(aggregate_dir)): file_fact(path)
        for path in sorted(aggregate_dir.glob("*"))
        if path.is_file()
    }
    return {
        "schema": "ddm_tb2_token_bit_attribution.v1",
        "status": "MEASURED_RECONCILED_MAP",
        "axis": binding["axis"],
        "score_claim": False,
        "source_binding": binding,
        "shape": [N, HEIGHT, WIDTH],
        "positions": POSITIONS,
        "physical_stream": {
            "bytes": EXPECTED["physical_stream_bytes"],
            "bits": STREAM_BITS,
            "sha256": EXPECTED["physical_stream_sha256"],
        },
        "cost_accounting": {
            "definition": "-log2(selected_integer_frequency/2**31)",
            "selected_frequency_cost_bits": total_bits,
            "selected_frequency_cost_bytes_equivalent": total_bits / 8.0,
            "physical_minus_selected_bits": residual_bits,
            "physical_minus_selected_bytes": residual_bits / 8.0,
            "residual_bound_bits": bl1.TERMINAL_OVERHEAD_BOUND_BITS,
            "residual_basis": (
                "finite arithmetic interval overhead under 2 bits plus final partial-byte padding under 7 bits"
            ),
            "bl1_hash_identity": fields["cost"]["sha256"] == EXPECTED["cost_sha256"],
            "passed": True,
        },
        "distribution": {
            "gini": gini,
            "concentration": concentration,
            "minimum_bits": float(cost.min()),
            "maximum_bits": float(cost.max()),
            "mean_bits_per_position": total_bits / POSITIONS,
        },
        "fields": fields,
        "join_fields": joins,
        "aggregate_fields": aggregate_facts,
        "attribution": {
            "gt_class": axis_rows["gt_class"],
            "decoded_symbol": axis_rows["symbol"],
            "spatial_band": axis_rows["spatial_band"],
            "pair_index": pair_rows,
            "rr4_base_context_top_100_by_bits": context_top,
            "fx2_mixer_cell_top_100_by_bits": mixer_top,
            "ma1_within_miss_cell_top_100_by_bits": miss_top,
            "rr4_context_decomposition": {
                key: axis_rows[key]
                for key in (
                    "base_class",
                    "surprise_bin",
                    "agree_t_minus_1",
                    "agree_t_minus_2",
                    "run_level",
                    "boundary_bucket",
                )
            },
            "full_context_tables": (
                "retained/aggregates/* arrays carry all 51,200 RR4, 4,000 mixer, "
                "and 1,296 within-miss cells; JSON intentionally lists only top 100"
            ),
        },
        "manufactured_seg_join": manufactured_join,
        "targets": {"top_1000_cost_positions": file_fact(top_positions_path)},
        "exchange_rate_cited_not_rederived": {
            "score_per_byte": 6.658590e-07,
            "source": "ddm_tx1_toolbox_crosswalk_20260819.md section 0",
        },
        "adjudication": {
            "gini_prior_gt_0p8": gini > 0.8,
            "top_1pct_share_gt_0p50": float(top1_row["bit_fraction"]) > 0.50,
            "manufactured_overlap_above_independence": (
                manufactured_top1_count > expected_top1_count and manufactured_top1_bits > expected_top1_bits
            ),
            "registered_independence_falsifier": (
                manufactured_top1_count <= expected_top1_count and manufactured_top1_bits <= expected_top1_bits
            ),
            "product_scope": ("attribution map only; high modeled cost is not a removable byte claim"),
        },
    }


def write_manifest(store: Path, binding: dict[str, object], result_path: Path) -> None:
    artifacts = []
    for path in sorted((store / "retained").rglob("*")):
        if path.is_file() and not path.name.startswith("._"):
            artifacts.append(file_fact(path))
    atomic_json(
        store / "MANIFEST.json",
        {
            "schema": "ddm_tb2_manifest.v1",
            "source_binding": binding,
            "result": file_fact(result_path),
            "artifacts": artifacts,
            "retention": (
                "All stage payloads, complete receiver checkpoints, aligned final fields, joins, "
                "and aggregates are preserved on APDataStore. No cleanup is authorized without "
                "a replacement machine-readable custody manifest."
            ),
        },
    )


def verify_completed(store: Path) -> None:
    manifest_path = store / "MANIFEST.json"
    result_path = store / "RESULT.json"
    if not manifest_path.is_file() or not result_path.is_file():
        raise Tb2Error("completed result or manifest is absent")
    manifest = json.loads(manifest_path.read_text())
    result_fact = manifest["result"]
    verify_file(result_path, str(result_fact["sha256"]), int(result_fact["bytes"]))
    for row in manifest["artifacts"]:
        verify_file(Path(row["path"]), str(row["sha256"]), int(row["bytes"]))
    result = json.loads(result_path.read_text())
    if result["cost_accounting"]["passed"] is not True:
        raise Tb2Error("completed result does not carry a passed reconciliation")
    if result["fields"]["cost"]["sha256"] != EXPECTED["cost_sha256"]:
        raise Tb2Error("completed cost field no longer matches BL1")
    if result["manufactured_seg_join"]["positions"] != EXPECTED["manufactured_positions"]:
        raise Tb2Error("completed manufactured join no longer closes")
    receipt = {
        "schema": "ddm_tb2_completed_verification.v1",
        "status": "VERIFIED_COMPLETE",
        "manifest": file_fact(manifest_path),
        "result": file_fact(result_path),
        "artifacts_rehashed": len(manifest["artifacts"]),
        "verified_at_unix_seconds": time.time(),
    }
    atomic_json(store / "COMPLETED_VERIFICATION.json", receipt)
    print(json.dumps(receipt, sort_keys=True))


def self_test() -> None:
    all_contexts = np.arange(CONTEXT_SIZE, dtype=np.uint16)
    parts = decode_rr4_context(all_contexts)
    repacked = (
        (
            ((parts["base_class"] * 64 + parts["surprise_bin"]) * 2 + parts["agree_t_minus_1"]) * 2
            + parts["agree_t_minus_2"]
        )
        * 8
        + parts["run_level"]
    ) * 5 + parts["boundary_bucket"]
    if not np.array_equal(repacked, all_contexts):
        raise AssertionError("RR4 context unpack/repack failed")
    with tempfile.TemporaryDirectory(prefix="ddm_tb2_selftest_") as tmp:
        root = Path(tmp)
        source = root / "source.bin"
        source.write_bytes(b"tb2-payload")
        target = root / "nested/target.bin"
        atomic_copy(source, target)
        if target.read_bytes() != source.read_bytes():
            raise AssertionError("atomic copy failed")
    bl1.self_test()
    print("ddm_tb2 self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--verify-completed", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    store = args.store.resolve()
    if not store.is_relative_to(AP_ROOT.resolve()):
        raise Tb2Error(f"output must remain on the mandated APDataStore tier: {AP_ROOT}")
    if args.verify_completed:
        verify_completed(store)
        return
    if store.exists() and not args.resume:
        raise Tb2Error("output exists; use --resume after verifying retained receipts")
    store.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(store).free
    if free < MIN_FREE_BYTES:
        raise Tb2Error(f"storage preflight failed: {free} B free < {MIN_FREE_BYTES} B")
    binding = source_binding()
    source_dir = store / "retained/source"
    source_dir.mkdir(parents=True, exist_ok=True)
    snapshot = source_dir / Path(__file__).name
    if snapshot.exists():
        existing = file_fact(snapshot)
        implementation = binding["sources"]["implementation"]
        if existing["bytes"] != implementation["bytes"] or existing["sha256"] != implementation["sha256"]:
            raise Tb2Error(
                "retained implementation snapshot differs from this source; refuse to overwrite "
                "the measured source during resume"
            )
    else:
        atomic_copy(Path(__file__), snapshot)
    atomic_json(
        store / "PREFLIGHT.json",
        {
            "schema": "ddm_tb2_preflight.v1",
            "source_binding": binding,
            "store": str(store),
            "free_bytes_before": free,
            "minimum_free_bytes": MIN_FREE_BYTES,
            "estimated_artifact_bytes": ESTIMATED_ARTIFACT_BYTES,
            "reserve_bytes": RESERVE_BYTES,
            "argv": sys.argv,
            "source_snapshot": file_fact(snapshot),
            "stage_frames": STAGE_FRAMES,
            "determinism": "no RNG; CPU threads=4; exact source and payload pins",
        },
    )
    library = bl1.build_decoder(store)
    run_instrumented_replay(store, binding, library)
    fields = assemble_fields(store, binding)
    joins = copy_join_fields(store)
    result = analyze(store, binding, fields, joins)
    result_path = store / "RESULT.json"
    atomic_json(result_path, result)
    write_manifest(store, binding, result_path)
    verify_completed(store)


if __name__ == "__main__":
    main()
