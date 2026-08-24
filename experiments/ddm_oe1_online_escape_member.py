#!/usr/bin/env python3
"""OE1: real RC64 sweep of a zero-stored causal uniform escape member.

The shipped DX2 HPAC/FreeCorrector law is replayed unchanged.  OE1 adds one
uniform expert whose per-position weight is the recent causal escape frequency
in the already-existing ``group x (boundary bucket, predicted class)`` cell.
No state is serialized: encoder and decoder start from zero and update only
after the current symbols are encoded/decoded.

Every full-field rung is RC64 re-encoded, decoded through the same receiver
trajectory, and retained.  The adaptation-zero rung must reproduce the shipped
113,777-byte stream byte-for-byte.  Encode and decode checkpoint every 20
frames, including the complete shipped corrector, escape histories, and RC64
interval state.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

bl1 = importlib.import_module("experiments.ddm_bl1_per_position_bit_allocation")
cp = importlib.import_module("experiments.ddm_cp135_rate_compose")
jg2 = importlib.import_module("experiments.ddm_jg2_tail_reencode")

OUTPUT = REPO / ".omx/tmp/arm_receipts_local/ddm_oe1_online_escape_member"
RUNTIME = Path("/Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2")
TO2 = Path(
    "/Volumes/VertigoDataTier/pact/ddm_to2_token_ordering_race/measurement_v1/retained/input"
)
AE1 = Path("/Volumes/VertigoDataTier/pact/ddm_ae1_anti_predicted_excess/measurement_v2")
EXPERIMENT_BOOK = Path("/Volumes/VertigoDataTier/pact/pr135_intake_20260810/experiment_book")

ARCHIVE = TO2 / "archive.zip"
TOKENS = TO2 / "dx2_tokens_decoded.u8"
STREAM = TO2 / "dx2_token_stream_rc64.bin"
AE1_RESULT = AE1 / "RESULT.json"
AE1_EXCESS = AE1 / "retained/fields/excess_over_uniform_bits.f64le.bin"
AE1_MASK = AE1 / "retained/fields/overshoot_gt_log2_5.raster.packbits"
ENCODER_SOURCE = EXPERIMENT_BOOK / "src/cpr1_sub4/entropy/rc64_backend.c"

EXPECTED = {
    "archive_bytes": 180_368,
    "archive_sha256": "976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674",
    "tokens_bytes": 117_964_800,
    "tokens_sha256": "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb",
    "stream_bytes": 113_777,
    "stream_sha256": "e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5",
    "excess_bytes": 943_718_400,
    "excess_sha256": "45f94cdaeeda86a7f4e467af1f182c73a2c5de76d08ed7c0a22c3b0f8af879ed",
    "mask_bytes": 14_745_600,
    "mask_sha256": "a1fadb5a966343f79649dcd4af892e373868bb93cf6ab2347fd1f3ef4a274d18",
    "overshoot_positions": 93_580,
    "gross_excess_bits": 213_162.3832609175,
    "member_bytes": 180_268,
    "member_sha256": "365f1b8d70463b250a2fe95e3599318ac90b31875cce5d66a767819404431c7a",
    "prefix_bytes": 66_491,
    "prefix_sha256": "0e2dd639e50795a00a3013f1ba66efa06495ed7b0a2ea6bbd920aa50b4ad1877",
}

N = 600
HEIGHT = 384
WIDTH = 512
PLANE = HEIGHT * WIDTH
POSITIONS = N * PLANE
GROUPS = 190
CLASSES = 5
FEATURES = 25
TOTAL = 1 << 31
UNIFORM = np.float32(0.2)
ALPHA_CAP = np.float32(0.5)
STAGE_FRAMES = 20
WINDOWS = (0, 1, 4, 16, 64)
RATE_DENOMINATOR = 37_545_489
S_PER_BYTE = 25.0 / RATE_DENOMINATOR
DEMAND_BYTES = 42_382
AXIS = "[macOS-CPU advisory / scorer-free exact RC64 byte measurement]"
ENCODE_SCHEMA = "ddm_oe1_encode_state.v1"
DECODE_SCHEMA = "ddm_oe1_decode_state.v1"
LOCAL_RESERVE_BYTES = 20 << 30
PROJECTED_BYTES = 6 << 30


class Oe1Error(RuntimeError):
    """Fail-closed source, resume, receiver, or retention error."""


def label(window: int) -> str:
    return "control_w0" if window == 0 else f"escape_w{window}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def verify_file(path: Path, digest: str, size: int | None = None) -> dict[str, object]:
    if not path.is_file():
        raise Oe1Error(f"required custody file is absent: {path}")
    fact = file_fact(path)
    if fact["sha256"] != digest or (size is not None and fact["bytes"] != size):
        raise Oe1Error(f"custody drift: {fact}; expected bytes={size}, sha256={digest}")
    return fact


def atomic_bytes(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return file_fact(path)


def atomic_json(path: Path, payload: object) -> dict[str, object]:
    return atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def atomic_npy(path: Path, value: np.ndarray) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return file_fact(path)


def atomic_npz(path: Path, **values: np.ndarray) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return file_fact(path)


def atomic_copy(source: Path, destination: Path, expected: dict[str, object]) -> dict[str, object]:
    if destination.is_file():
        return verify_file(destination, str(expected["sha256"]), int(expected["bytes"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".partial.{os.getpid()}")
    try:
        with source.open("rb") as reader, temporary.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=1 << 24)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return verify_file(destination, str(expected["sha256"]), int(expected["bytes"]))


def local_preflight(output: Path) -> dict[str, object]:
    resolved = output.resolve()
    allowed = (REPO / ".omx/tmp/arm_receipts_local").resolve()
    if allowed not in resolved.parents:
        raise Oe1Error(f"output is not in the charter-authorized local receipt tier: {resolved}")
    if str(resolved).startswith("/Volumes/"):
        raise Oe1Error("OE1 must not write either full SSD tier")
    usage = shutil.disk_usage(REPO)
    if usage.free < PROJECTED_BYTES + LOCAL_RESERVE_BYTES:
        raise Oe1Error("local tier lacks projected bytes plus the 20-GiB fail-closed reserve")
    return {
        "tier": "local explicit opt-in",
        "root": str(resolved),
        "free_bytes_before": usage.free,
        "projected_bytes": PROJECTED_BYTES,
        "reserve_bytes": LOCAL_RESERVE_BYTES,
        "ssd_writes": False,
    }


def source_binding() -> dict[str, object]:
    facts = {
        "archive": verify_file(ARCHIVE, EXPECTED["archive_sha256"], EXPECTED["archive_bytes"]),
        "tokens": verify_file(TOKENS, EXPECTED["tokens_sha256"], EXPECTED["tokens_bytes"]),
        "stream": verify_file(STREAM, EXPECTED["stream_sha256"], EXPECTED["stream_bytes"]),
        "ae1_result": file_fact(AE1_RESULT),
        "ae1_excess": verify_file(AE1_EXCESS, EXPECTED["excess_sha256"], EXPECTED["excess_bytes"]),
        "ae1_mask": verify_file(AE1_MASK, EXPECTED["mask_sha256"], EXPECTED["mask_bytes"]),
        "encoder_source": file_fact(ENCODER_SOURCE),
        "shipped_decoder_source": file_fact(RUNTIME / "runtime/entropy/rc64_backend.c"),
        "shipped_free_corrector": file_fact(RUNTIME / "runtime/free_corrector.py"),
        "shipped_residual_archive": file_fact(RUNTIME / "runtime/residual_archive.py"),
        "implementation": file_fact(Path(__file__)),
    }
    ae1 = json.loads(AE1_RESULT.read_text())
    if (
        int(ae1["overshoot_positions"]) != EXPECTED["overshoot_positions"]
        or not math.isclose(
            float(ae1["gross_excess_bits"]),
            EXPECTED["gross_excess_bits"],
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        raise Oe1Error("AE1 result no longer carries the charter-pinned excess totals")
    return {
        "schema": "ddm_oe1_source_binding.v1",
        "axis": AXIS,
        "shape": [N, HEIGHT, WIDTH],
        "positions": POSITIONS,
        "sources": facts,
    }


def retain_and_reproduce_ae1(output: Path) -> dict[str, object]:
    retained = output / "retained/fields"
    excess_path = retained / "excess_over_uniform_bits.f64le.bin"
    mask_path = retained / "overshoot_gt_log2_5.raster.packbits"
    excess_fact = atomic_copy(AE1_EXCESS, excess_path, file_fact(AE1_EXCESS))
    mask_fact = atomic_copy(AE1_MASK, mask_path, file_fact(AE1_MASK))
    excess = np.memmap(excess_path, dtype="<f8", mode="r", shape=(POSITIONS,))
    total = 0.0
    nonzero = 0
    for start in range(0, POSITIONS, 1 << 22):
        values = np.asarray(excess[start : start + (1 << 22)], dtype=np.float64)
        total += float(values.sum(dtype=np.float64))
        nonzero += int(np.count_nonzero(values > 0.0))
    packed = np.memmap(mask_path, dtype=np.uint8, mode="r", shape=(EXPECTED["mask_bytes"],))
    mask_count = 0
    for start in range(0, packed.size, 1 << 22):
        mask_count += int(np.unpackbits(np.asarray(packed[start : start + (1 << 22)]), bitorder="little").sum())
    if nonzero != EXPECTED["overshoot_positions"] or mask_count != nonzero:
        raise Oe1Error(f"AE1 excess count failed reproduction: excess={nonzero}, mask={mask_count}")
    if not math.isclose(total, EXPECTED["gross_excess_bits"], rel_tol=0.0, abs_tol=1e-6):
        raise Oe1Error(f"AE1 excess bits failed reproduction: {total}")
    result = {
        "status": "PASS_BEFORE_SWEEP",
        "positions": POSITIONS,
        "overshoot_positions": nonzero,
        "already_below_uniform_positions": POSITIONS - nonzero,
        "gross_excess_bits": total,
        "gross_excess_bytes": total / 8.0,
        "excess_field": excess_fact,
        "overshoot_mask": mask_fact,
    }
    atomic_json(output / "AE1_REPRODUCTION.json", result)
    return result


def retained_mask_frame(output: Path, frame: int) -> np.ndarray:
    path = output / "retained/fields/overshoot_gt_log2_5.raster.packbits"
    bytes_per_frame = PLANE // 8
    packed = np.memmap(path, dtype=np.uint8, mode="r", offset=frame * bytes_per_frame, shape=(bytes_per_frame,))
    return np.unpackbits(packed, bitorder="little", count=PLANE).astype(bool, copy=False)


def import_rc64() -> Any:
    path = EXPERIMENT_BOOK / "src/cpr1_sub4/entropy/rc64.py"
    spec = importlib.util.spec_from_file_location("_ddm_oe1_rc64", path)
    if spec is None or spec.loader is None:
        raise Oe1Error(f"cannot import RC64 wrapper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compile_rc64(output: Path) -> tuple[Path, Any]:
    rc64 = import_rc64()
    library = cp._compile_checkpointable_rc64(
        SimpleNamespace(experiment_book=EXPERIMENT_BOOK, output=output)
    )
    atomic_json(
        output / "work/RC64_BUILD.json",
        {
            "schema": "ddm_oe1_rc64_build.v1",
            "source": file_fact(ENCODER_SOURCE),
            "generated_source": file_fact(output / "work/rc64_checkpoint_backend.c"),
            "library": file_fact(library),
        },
    )
    return library, rc64


def load_receiver(library: Path) -> dict[str, Any]:
    import torch

    sys.path.insert(0, str(RUNTIME / "cpr1"))
    sys.path.insert(0, str(RUNTIME))
    renderer = importlib.import_module("cpr1.inflate")
    residual = importlib.import_module("runtime.residual_archive")
    free_corrector = importlib.import_module("runtime.free_corrector")
    hpac_inference = importlib.import_module("runtime.hpac_inference")
    parts = residual.read_residual_archive(ARCHIVE)
    if len(parts.token_stream) != EXPECTED["stream_bytes"] or hashlib.sha256(parts.token_stream).hexdigest() != EXPECTED["stream_sha256"]:
        raise Oe1Error("shipped parser extracted a different token stream")
    device = torch.device("cpu")
    base_hpac = residual.materialize_ihs1(parts.hpac_blob, renderer)
    model = renderer.load_hpac(base_hpac, device)
    masks = renderer.group_masks(device)
    sparse = residual._sparse_class(RUNTIME / "cpr1")(model, HEIGHT, WIDTH)
    hpac_inference.optimize_sparse_evaluator(sparse)
    plans = []
    for mask in masks:
        flat = np.flatnonzero(mask.detach().cpu().numpy().reshape(-1)).astype(np.int64)
        plans.append((torch.from_numpy(flat).to(device), flat))
    if len(plans) != GROUPS or sum(len(row[1]) for row in plans) != PLANE:
        raise Oe1Error("shipped 190-group map no longer partitions the plane")
    return {
        "torch": torch,
        "renderer": renderer,
        "residual": residual,
        "parts": parts,
        "model": model,
        "sparse": sparse,
        "plans": plans,
        "corrector": free_corrector.FreeCorrector(PLANE),
        "cold": free_corrector.FreeCorrector(PLANE),
        "device": device,
        "library": library,
    }


class EscapeState:
    """Rolling causal overshoot counts in group x existing feature cells."""

    def __init__(self, window: int) -> None:
        self.window = int(window)
        depth = max(1, self.window)
        shape = (depth, GROUPS, FEATURES)
        self.bad_ring = np.zeros(shape, dtype=np.int64)
        self.seen_ring = np.zeros(shape, dtype=np.int64)
        self.bad_sum = np.zeros((GROUPS, FEATURES), dtype=np.int64)
        self.seen_sum = np.zeros((GROUPS, FEATURES), dtype=np.int64)

    def coding(self, base: np.ndarray, group: int, feature: np.ndarray) -> np.ndarray:
        if self.window == 0:
            return base
        seen = self.seen_sum[group, feature]
        bad = self.bad_sum[group, feature]
        alpha64 = np.divide(
            bad,
            seen,
            out=np.zeros(bad.shape, dtype=np.float64),
            where=seen > 0,
        )
        alpha = np.minimum(alpha64, float(ALPHA_CAP)).astype(np.float32)
        return (base + alpha[:, None] * (UNIFORM - base)).astype(np.float32)

    def observe(self, frame: int, group: int, feature: np.ndarray, anti: np.ndarray) -> None:
        if self.window == 0:
            return
        cursor = frame % self.window
        old_bad = self.bad_ring[cursor, group]
        old_seen = self.seen_ring[cursor, group]
        self.bad_sum[group] -= old_bad
        self.seen_sum[group] -= old_seen
        seen = np.bincount(feature, minlength=FEATURES).astype(np.int64)
        bad = np.bincount(feature[anti], minlength=FEATURES).astype(np.int64)
        old_bad[:] = bad
        old_seen[:] = seen
        self.bad_sum[group] += bad
        self.seen_sum[group] += seen
        if np.any(self.bad_sum < 0) or np.any(self.seen_sum < self.bad_sum):
            raise Oe1Error("causal escape rolling-count invariant failed")

    def arrays(self, prefix: str) -> dict[str, np.ndarray]:
        return {
            f"{prefix}__bad_ring": self.bad_ring,
            f"{prefix}__seen_ring": self.seen_ring,
            f"{prefix}__bad_sum": self.bad_sum,
            f"{prefix}__seen_sum": self.seen_sum,
        }

    def load(self, payload: Any, prefix: str) -> None:
        for name in ("bad_ring", "seen_ring", "bad_sum", "seen_sum"):
            source = np.asarray(payload[f"{prefix}__{name}"], dtype=np.int64)
            destination = getattr(self, name)
            if source.shape != destination.shape:
                raise Oe1Error(f"escape resume geometry drift: {prefix}/{name}")
            destination[:] = source


def selected_costs(rc64: Any, coding: np.ndarray, symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frequencies = rc64.quantize_probabilities(coding)
    index = np.arange(symbols.size)
    selected = frequencies[index, symbols]
    costs = 31.0 - np.log2(selected.astype(np.float64))
    return frequencies, costs


def stage_root(output: Path, phase: str, start: int, end: int) -> Path:
    return output / phase / "stages" / f"frames_{start:04d}_{end - 1:04d}"


def contiguous_receipts(output: Path, phase: str) -> list[dict[str, Any]]:
    rows = []
    for start in range(0, N, STAGE_FRAMES):
        end = min(start + STAGE_FRAMES, N)
        path = stage_root(output, phase, start, end) / "RECEIPT.json"
        if not path.is_file():
            break
        row = json.loads(path.read_text())
        if row.get("frame_start") != start or row.get("frame_end") != end:
            raise Oe1Error(f"nonmatching {phase} stage receipt: {path}")
        for artifact in row["artifacts"].values():
            verify_file(Path(artifact["path"]), str(artifact["sha256"]), int(artifact["bytes"]))
        rows.append(row)
    all_rows = list((output / phase / "stages").glob("frames_*/RECEIPT.json"))
    if len(all_rows) != len(rows):
        raise Oe1Error(f"{phase} stage receipts are not a contiguous prefix")
    return rows


def restore_common_state(runtime: dict[str, Any], states: dict[int, EscapeState], path: Path, schema: str) -> tuple[int, Any]:
    torch = runtime["torch"]
    with np.load(path, allow_pickle=False) as payload:
        if bytes(payload["schema"]).decode() != schema:
            raise Oe1Error(f"checkpoint schema drift: {path}")
        frame_end = int(payload["frame_end"][0])
        corrector_state = {
            key.removeprefix("corrector__"): payload[key].copy()
            for key in payload.files
            if key.startswith("corrector__")
        }
        jg2.load_corrector_state(runtime["corrector"], corrector_state)
        for window, state in states.items():
            state.load(payload, label(window))
        previous_np = np.asarray(payload["previous"], dtype=np.uint8).copy()
    previous = torch.from_numpy(previous_np.astype(np.int64)).reshape(1, HEIGHT, WIDTH).to(runtime["device"])
    return frame_end, previous


def common_state_arrays(runtime: dict[str, Any], states: dict[int, EscapeState], schema: str, frame_end: int, previous: np.ndarray) -> dict[str, np.ndarray]:
    captured = jg2.corrector_state(runtime["corrector"])
    lost = jg2.uncaptured_divergent_state(runtime["corrector"], runtime["cold"], set(captured))
    if lost:
        raise Oe1Error(f"checkpoint would lose shipped corrector state: {lost[:8]}")
    arrays = {
        "schema": np.frombuffer(schema.encode(), dtype=np.uint8),
        "frame_end": np.asarray([frame_end], dtype=np.int64),
        "previous": np.asarray(previous, dtype=np.uint8),
        **{f"corrector__{key}": value for key, value in captured.items()},
    }
    for window, state in states.items():
        arrays.update(state.arrays(label(window)))
    return arrays


def member_prefix(output: Path) -> tuple[bytes, dict[str, object]]:
    with zipfile.ZipFile(ARCHIVE) as archive:
        infos = archive.infolist()
        if len(infos) != 1 or infos[0].filename != "p" or infos[0].compress_type != zipfile.ZIP_STORED:
            raise Oe1Error("DX2 archive framing drifted")
        member = archive.read("p")
    stream = STREAM.read_bytes()
    if not member.endswith(stream):
        raise Oe1Error("DX2 token stream is no longer the member suffix")
    if len(member) != EXPECTED["member_bytes"] or hashlib.sha256(member).hexdigest() != EXPECTED["member_sha256"]:
        raise Oe1Error("DX2 stored member drifted")
    prefix = member[: -len(stream)]
    fact = atomic_bytes(output / "retained/input/non_token_prefix.bin", prefix)
    if fact["bytes"] != EXPECTED["prefix_bytes"] or fact["sha256"] != EXPECTED["prefix_sha256"]:
        raise Oe1Error("DX2 non-token prefix drifted")
    return prefix, fact


def encode_all(output: Path, binding: dict[str, object], library: Path, rc64: Any) -> dict[str, Any]:
    import torch

    receipts = contiguous_receipts(output, "encode")
    runtime = load_receiver(library)
    states = {window: EscapeState(window) for window in WINDOWS}
    encoders: dict[int, Any] = {}
    if receipts:
        last = receipts[-1]
        root = stage_root(output, "encode", int(last["frame_start"]), int(last["frame_end"]))
        start_frame, previous = restore_common_state(runtime, states, root / "receiver_state.npz", ENCODE_SCHEMA)
        for window in WINDOWS:
            encoders[window] = cp._rc64_resume(
                rc64.NativeEncoder,
                library,
                (root / f"encoder_{label(window)}.state").read_bytes(),
            )
    else:
        start_frame = 0
        previous = torch.zeros((1, HEIGHT, WIDTH), dtype=torch.long, device=runtime["device"])
        encoders = {window: rc64.NativeEncoder(library) for window in WINDOWS}

    truth = np.memmap(TOKENS, dtype=np.uint8, mode="r", shape=(N, HEIGHT, WIDTH))
    model = runtime["model"]
    sparse = runtime["sparse"]
    plans = runtime["plans"]
    corrector = runtime["corrector"]
    residual = runtime["residual"]
    parts = runtime["parts"]
    device = runtime["device"]
    started = time.perf_counter()
    with torch.inference_mode():
        for stage_start in range(start_frame, N, STAGE_FRAMES):
            stage_end = min(stage_start + STAGE_FRAMES, N)
            shape = (stage_end - stage_start, GROUPS, 2)
            counts = {window: np.zeros(shape, dtype=np.uint64) for window in WINDOWS}
            bits = {window: np.zeros(shape, dtype=np.float64) for window in WINDOWS}
            for frame in range(stage_start, stage_end):
                offset = frame - stage_start
                current = torch.zeros_like(previous)
                context = model.prepare_frame_context(torch.tensor([frame], dtype=torch.long, device=device), previous)
                if frame:
                    previous_cpu = previous[0].to(device="cpu", dtype=torch.uint8).numpy()
                    boundary = residual._boundary_buckets(previous_cpu).reshape(-1)
                else:
                    boundary = np.full(PLANE, 4, dtype=np.uint8)
                anti_frame = retained_mask_frame(output, frame)
                corrector.begin_frame(boundary)
                plane_target = np.asarray(truth[frame]).reshape(-1)
                for group, (device_positions, flat_positions) in enumerate(plans):
                    base_logits = sparse.selected_logits(current, context, group).cpu().numpy()
                    predicted = base_logits.argmax(axis=1).astype(np.int64)
                    feature = boundary[flat_positions].astype(np.int64) * CLASSES + predicted
                    corrected = base_logits + parts.table.values[feature]
                    probability = residual._probability_table(corrected, runtime["renderer"].HPAC_LOGIT_PRECISION)
                    receiver_state = corrector.group_state(probability, predicted, flat_positions)
                    base = np.asarray(corrector.coding_row(receiver_state), dtype=np.float32)
                    symbols = plane_target[flat_positions].astype(np.int64)
                    base_freq, base_cost = selected_costs(rc64, base, symbols)
                    selected = base_freq[np.arange(symbols.size), symbols]
                    anti = selected.astype(np.uint64) * CLASSES < TOTAL
                    expected_anti = anti_frame[flat_positions]
                    if not np.array_equal(anti, expected_anti):
                        raise Oe1Error(f"AE1 mask mismatch at frame={frame}, group={group}")
                    for window in WINDOWS:
                        candidate = states[window].coding(base, group, feature)
                        encoders[window].encode(symbols.astype(np.int32), candidate)
                        _, candidate_cost = selected_costs(rc64, candidate, symbols)
                        counts[window][offset, group] = (int(anti.sum()), int((~anti).sum()))
                        bits[window][offset, group, 0] = float((base_cost[anti] - candidate_cost[anti]).sum(dtype=np.float64))
                        bits[window][offset, group, 1] = float((candidate_cost[~anti] - base_cost[~anti]).sum(dtype=np.float64))
                        states[window].observe(frame, group, feature, anti)
                    corrector.observe(receiver_state, symbols)
                    current.reshape(-1)[device_positions] = torch.from_numpy(symbols).to(device)
                frame_tokens = current[0].to(device="cpu", dtype=torch.uint8).numpy()
                if not np.array_equal(frame_tokens, truth[frame]):
                    raise Oe1Error(f"teacher-forced encoder trajectory diverged at frame {frame}")
                corrector.end_frame(frame_tokens.reshape(-1))
                previous = current

            root = stage_root(output, "encode", stage_start, stage_end)
            artifacts: dict[str, dict[str, object]] = {}
            for window in WINDOWS:
                artifacts[f"split_{label(window)}"] = atomic_npz(
                    root / f"selectivity_{label(window)}.npz",
                    counts=counts[window],
                    bits=bits[window],
                )
                artifacts[f"encoder_{label(window)}"] = atomic_bytes(
                    root / f"encoder_{label(window)}.state", cp._rc64_snapshot(encoders[window])
                )
            arrays = common_state_arrays(
                runtime,
                states,
                ENCODE_SCHEMA,
                stage_end,
                previous[0].to(device="cpu", dtype=torch.uint8).numpy(),
            )
            artifacts["receiver_state"] = atomic_npz(root / "receiver_state.npz", **arrays)
            receipt = {
                "schema": "ddm_oe1_encode_stage.v1",
                "source_binding_sha256": hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest(),
                "frame_start": stage_start,
                "frame_end": stage_end,
                "artifacts": artifacts,
                "elapsed_seconds": time.perf_counter() - started,
            }
            atomic_json(root / "RECEIPT.json", receipt)
            receipts.append(receipt)
            print(json.dumps({"phase": "encode", "frame_end": stage_end, "elapsed_s": round(time.perf_counter() - started, 3)}), flush=True)

    if len(receipts) != N // STAGE_FRAMES:
        raise Oe1Error("encode did not preserve all 30 stage checkpoints")
    prefix, prefix_fact = member_prefix(output)
    rows = []
    for window in WINDOWS:
        payload = encoders[window].finish()
        root = output / "retained/rungs" / label(window)
        stream_fact = atomic_bytes(root / "tokens.rc64", payload)
        member_fact = atomic_bytes(root / "member.bin", prefix + payload)
        if window == 0 and (
            stream_fact["bytes"] != EXPECTED["stream_bytes"]
            or stream_fact["sha256"] != EXPECTED["stream_sha256"]
        ):
            raise Oe1Error(f"degenerate control failed byte identity: {stream_fact}")
        rows.append({
            "window_frames": window,
            "rung": label(window),
            "stream": stream_fact,
            "member": member_fact,
            "non_token_prefix": prefix_fact,
            "non_token_descriptor_growth_bytes": 0,
        })
    result = {"schema": "ddm_oe1_encode_result.v1", "axis": AXIS, "rows": rows}
    atomic_json(output / "ENCODE_RESULT.json", result)
    return result


def decode_all(output: Path, library: Path, rc64: Any, encode_result: dict[str, Any]) -> dict[str, Any]:
    import torch

    rows_by_window = {int(row["window_frames"]): row for row in encode_result["rows"]}
    receipts = contiguous_receipts(output, "decode")
    runtime = load_receiver(library)
    states = {window: EscapeState(window) for window in WINDOWS}
    decoders = {
        window: rc64.NativeDecoder(library, Path(rows_by_window[window]["stream"]["path"]).read_bytes())
        for window in WINDOWS
    }
    if receipts:
        last = receipts[-1]
        root = stage_root(output, "decode", int(last["frame_start"]), int(last["frame_end"]))
        start_frame, previous = restore_common_state(runtime, states, root / "receiver_state.npz", DECODE_SCHEMA)
        with np.load(root / "decoder_states.npz", allow_pickle=False) as saved:
            for window in WINDOWS:
                bl1.restore_decoder_state(decoders[window], saved[label(window)])
    else:
        start_frame = 0
        previous = torch.zeros((1, HEIGHT, WIDTH), dtype=torch.long, device=runtime["device"])

    truth = np.memmap(TOKENS, dtype=np.uint8, mode="r", shape=(N, HEIGHT, WIDTH))
    model = runtime["model"]
    sparse = runtime["sparse"]
    plans = runtime["plans"]
    corrector = runtime["corrector"]
    residual = runtime["residual"]
    parts = runtime["parts"]
    device = runtime["device"]
    started = time.perf_counter()
    with torch.inference_mode():
        for stage_start in range(start_frame, N, STAGE_FRAMES):
            stage_end = min(stage_start + STAGE_FRAMES, N)
            decoded_stage = {
                window: np.empty((stage_end - stage_start, HEIGHT, WIDTH), dtype=np.uint8)
                for window in WINDOWS
            }
            for frame in range(stage_start, stage_end):
                offset = frame - stage_start
                current = torch.zeros_like(previous)
                context = model.prepare_frame_context(torch.tensor([frame], dtype=torch.long, device=device), previous)
                if frame:
                    previous_cpu = previous[0].to(device="cpu", dtype=torch.uint8).numpy()
                    boundary = residual._boundary_buckets(previous_cpu).reshape(-1)
                else:
                    boundary = np.full(PLANE, 4, dtype=np.uint8)
                corrector.begin_frame(boundary)
                expected_plane = np.asarray(truth[frame]).reshape(-1)
                for group, (device_positions, flat_positions) in enumerate(plans):
                    base_logits = sparse.selected_logits(current, context, group).cpu().numpy()
                    predicted = base_logits.argmax(axis=1).astype(np.int64)
                    feature = boundary[flat_positions].astype(np.int64) * CLASSES + predicted
                    corrected = base_logits + parts.table.values[feature]
                    probability = residual._probability_table(corrected, runtime["renderer"].HPAC_LOGIT_PRECISION)
                    receiver_state = corrector.group_state(probability, predicted, flat_positions)
                    base = np.asarray(corrector.coding_row(receiver_state), dtype=np.float32)
                    expected = expected_plane[flat_positions].astype(np.int64)
                    base_freq, _ = selected_costs(rc64, base, expected)
                    selected = base_freq[np.arange(expected.size), expected]
                    anti = selected.astype(np.uint64) * CLASSES < TOTAL
                    for window in WINDOWS:
                        candidate = states[window].coding(base, group, feature)
                        decoded = decoders[window].decode(candidate).astype(np.int64)
                        if not np.array_equal(decoded, expected):
                            mismatch = int(np.flatnonzero(decoded != expected)[0])
                            raise Oe1Error(
                                f"decoded token mismatch rung={label(window)}, frame={frame}, "
                                f"group={group}, within_group={mismatch}"
                            )
                        states[window].observe(frame, group, feature, anti)
                    corrector.observe(receiver_state, expected)
                    current.reshape(-1)[device_positions] = torch.from_numpy(expected).to(device)
                frame_tokens = current[0].to(device="cpu", dtype=torch.uint8).numpy()
                for window in WINDOWS:
                    decoded_stage[window][offset] = frame_tokens
                corrector.end_frame(frame_tokens.reshape(-1))
                previous = current

            root = stage_root(output, "decode", stage_start, stage_end)
            artifacts: dict[str, dict[str, object]] = {}
            for window in WINDOWS:
                artifacts[f"decoded_{label(window)}"] = atomic_npy(
                    root / f"decoded_{label(window)}.npy", decoded_stage[window]
                )
            decoder_arrays = {label(window): bl1.decoder_state(decoders[window]) for window in WINDOWS}
            artifacts["decoder_states"] = atomic_npz(root / "decoder_states.npz", **decoder_arrays)
            arrays = common_state_arrays(
                runtime,
                states,
                DECODE_SCHEMA,
                stage_end,
                previous[0].to(device="cpu", dtype=torch.uint8).numpy(),
            )
            artifacts["receiver_state"] = atomic_npz(root / "receiver_state.npz", **arrays)
            receipt = {
                "schema": "ddm_oe1_decode_stage.v1",
                "frame_start": stage_start,
                "frame_end": stage_end,
                "artifacts": artifacts,
                "elapsed_seconds": time.perf_counter() - started,
            }
            atomic_json(root / "RECEIPT.json", receipt)
            receipts.append(receipt)
            print(json.dumps({"phase": "decode", "frame_end": stage_end, "elapsed_s": round(time.perf_counter() - started, 3)}), flush=True)

    if len(receipts) != N // STAGE_FRAMES:
        raise Oe1Error("decode did not preserve all 30 stage checkpoints")
    rows = []
    for window in WINDOWS:
        destination = output / "retained/rungs" / label(window) / "decoded_tokens.u8"
        temporary = destination.with_name(destination.name + f".partial.{os.getpid()}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with temporary.open("wb") as handle:
                for start in range(0, N, STAGE_FRAMES):
                    end = min(start + STAGE_FRAMES, N)
                    value = np.load(
                        stage_root(output, "decode", start, end) / f"decoded_{label(window)}.npy",
                        allow_pickle=False,
                    )
                    handle.write(np.ascontiguousarray(value).tobytes())
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        fact = verify_file(destination, EXPECTED["tokens_sha256"], EXPECTED["tokens_bytes"])
        rows.append({
            "window_frames": window,
            "rung": label(window),
            "decoded_tokens": fact,
            "decoded_field_identity": True,
            "decoder_bit_position": decoders[window].bit_position,
        })
    result = {"schema": "ddm_oe1_decode_result.v1", "axis": AXIS, "rows": rows}
    atomic_json(output / "DECODE_RESULT.json", result)
    return result


def assemble_selectivity(output: Path, encode_result: dict[str, Any], decode_result: dict[str, Any]) -> list[dict[str, Any]]:
    decode_rows = {int(row["window_frames"]): row for row in decode_result["rows"]}
    output_rows = []
    for encode_row in encode_result["rows"]:
        window = int(encode_row["window_frames"])
        counts_parts = []
        bits_parts = []
        for start in range(0, N, STAGE_FRAMES):
            end = min(start + STAGE_FRAMES, N)
            with np.load(
                stage_root(output, "encode", start, end) / f"selectivity_{label(window)}.npz",
                allow_pickle=False,
            ) as payload:
                counts_parts.append(payload["counts"].copy())
                bits_parts.append(payload["bits"].copy())
        counts = np.concatenate(counts_parts)
        bits = np.concatenate(bits_parts)
        split_fact = atomic_npz(
            output / "retained/rungs" / label(window) / "selectivity_by_frame_group.npz",
            counts=counts,
            bits=bits,
        )
        anti_count = int(counts[:, :, 0].sum())
        complement_count = int(counts[:, :, 1].sum())
        if (anti_count, complement_count) != (
            EXPECTED["overshoot_positions"],
            POSITIONS - EXPECTED["overshoot_positions"],
        ):
            raise Oe1Error(f"selectivity denominator drift for {label(window)}")
        recovered_bits = float(bits[:, :, 0].sum(dtype=np.float64))
        spent_bits = float(bits[:, :, 1].sum(dtype=np.float64))
        model_savings_bytes = (recovered_bits - spent_bits) / 8.0
        real_bytes = int(encode_row["stream"]["bytes"])
        real_savings = EXPECTED["stream_bytes"] - real_bytes
        ratio = recovered_bits / spent_bits if spent_bits > 0.0 else math.inf
        row = {
            **encode_row,
            **decode_rows[window],
            "axis": AXIS,
            "positions": POSITIONS,
            "anti_predicted_positions": anti_count,
            "already_below_uniform_positions": complement_count,
            "recovered_on_anti_predicted_bits": recovered_bits,
            "recovered_on_anti_predicted_bytes": recovered_bits / 8.0,
            "spent_on_already_below_uniform_bits": spent_bits,
            "spent_on_already_below_uniform_bytes": spent_bits / 8.0,
            "selectivity_ratio_recovered_over_spent": ratio,
            "model_integer_frequency_savings_bytes": model_savings_bytes,
            "real_stream_bytes": real_bytes,
            "real_stream_savings_bytes": real_savings,
            "finite_coder_reconciliation_bytes": real_savings - model_savings_bytes,
            "rate_only_delta_s_at_25_over_37545489": -real_savings * S_PER_BYTE,
            "share_of_42382_byte_demand": real_savings / DEMAND_BYTES,
            "selectivity_artifact": split_fact,
            "score_claim": False,
        }
        output_rows.append(row)
    return output_rows


def write_manifest(output: Path, result: dict[str, Any]) -> dict[str, Any]:
    result_fact = file_fact(output / "RESULT.json")
    artifacts = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            artifacts.append(file_fact(path))
    manifest = {
        "schema": "ddm_oe1_retention_manifest.v1",
        "tier": "local explicit opt-in",
        "root": str(output.resolve()),
        "artifact_count": len(artifacts),
        "artifact_bytes": sum(int(row["bytes"]) for row in artifacts),
        "artifacts": artifacts,
        "result": result_fact,
        "retention": "No listed payload may be deleted or moved without a replacement custody manifest.",
    }
    atomic_json(output / "MANIFEST.json", manifest)
    return manifest


def self_test() -> None:
    base = np.asarray([[0.7, 0.1, 0.1, 0.05, 0.05], [0.1, 0.2, 0.3, 0.2, 0.2]], dtype=np.float32)
    feature = np.asarray([3, 7], dtype=np.int64)
    state = EscapeState(1)
    if not np.array_equal(state.coding(base, 0, feature), base):
        raise Oe1Error("cold escape state must nest the incumbent exactly")
    state.observe(0, 0, feature, np.asarray([True, False]))
    mixed = state.coding(base, 0, feature)
    if not np.allclose(mixed[0], 0.5 * base[0] + 0.1) or not np.array_equal(mixed[1], base[1]):
        raise Oe1Error("escape mixture response is wrong")
    state.observe(1, 0, feature, np.asarray([False, False]))
    if not np.array_equal(state.coding(base, 0, feature), base):
        raise Oe1Error("window-one state did not forget the prior frame")
    control = EscapeState(0)
    control.observe(0, 0, feature, np.asarray([True, True]))
    if control.coding(base, 0, feature) is not base:
        raise Oe1Error("control path must return the incumbent row object")
    snapshot = state.arrays("escape_w1")
    restored = EscapeState(1)
    restored.load(snapshot, "escape_w1")
    if not np.array_equal(restored.coding(base, 0, feature), state.coding(base, 0, feature)):
        raise Oe1Error("escape checkpoint round-trip changed the coding row")
    print(json.dumps({"self_test": "PASS", "windows": WINDOWS}))


def run(output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    preflight = local_preflight(output)
    output.mkdir(parents=True, exist_ok=True)
    binding = source_binding()
    binding_path = output / "SOURCE_BINDING.json"
    if binding_path.is_file() and json.loads(binding_path.read_text()) != binding:
        raise Oe1Error("existing receipt root is bound to a different source or implementation")
    atomic_json(binding_path, binding)
    atomic_json(output / "PREFLIGHT.json", preflight)
    atomic_json(
        output / "LAUNCH_CONFIG.json",
        {
            "schema": "ddm_oe1_launch_config.v1",
            "argv": sys.argv,
            "cwd": str(Path.cwd().resolve()),
            "python": sys.executable,
            "rng": "none used",
            "torch_threads": 4,
            "torch_interop_threads": 1,
            "write_tier": "local explicit opt-in only",
        },
    )
    reproduction = retain_and_reproduce_ae1(output)
    library, rc64 = compile_rc64(output)
    encode_result = encode_all(output, binding, library, rc64)
    decode_result = decode_all(output, library, rc64, encode_result)
    rows = assemble_selectivity(output, encode_result, decode_result)
    candidates = [row for row in rows if int(row["window_frames"]) > 0]
    best = max(candidates, key=lambda row: int(row["real_stream_savings_bytes"]))
    all_nonpositive = all(int(row["real_stream_savings_bytes"]) <= 0 for row in candidates)
    result = {
        "schema": "ddm_oe1_online_escape_member_result.v1",
        "status": "COMPLETE",
        "axis": AXIS,
        "score_claim": False,
        "pointer_moved": False,
        "source_binding": binding,
        "preflight": preflight,
        "ae1_reproduction": reproduction,
        "mechanism": {
            "member": "uniform prior",
            "weight": "min(0.5, recent_anti_predicted_count/recent_seen_count)",
            "cell": "shipped group190 x existing (boundary bucket, predicted class)",
            "history_windows_frames": list(WINDOWS),
            "causality": "weight reads only completed prior occurrences; current symbols update state after coding",
            "stored_parameter_bytes": 0,
            "other_shipped_members_changed": 0,
            "coder_order_addressing_changed": False,
        },
        "rows": rows,
        "best_noncontrol": best,
        "all_noncontrol_real_stream_savings_nonpositive": all_nonpositive,
        "verdict_scope": (
            "FAMILY: fixed-DX2 anti-predicted uniform-member routes, combining AE1 stored/static "
            "forms with OE1 zero-stored causal recent-escape windows {1,4,16,64}"
            if all_nonpositive
            else "FORMULATION: zero-stored causal recent-escape uniform member on fixed DX2"
        ),
        "distortion_boundary": (
            "No DALI-GT or scorer was read. Every rung losslessly reproduces the TO2 decoded-token "
            "field sha256; OE1 makes no d_seg, d_pose, exact-score, or shipping claim."
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(output / "RESULT.json", result)
    manifest = write_manifest(output, result)
    print(json.dumps({
        "status": "COMPLETE",
        "best_rung": best["rung"],
        "best_real_stream_savings_bytes": best["real_stream_savings_bytes"],
        "all_nonpositive": all_nonpositive,
        "result": file_fact(output / "RESULT.json"),
        "manifest": file_fact(output / "MANIFEST.json"),
        "artifact_bytes": manifest["artifact_bytes"],
    }, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    run(args.output)


if __name__ == "__main__":
    main()
