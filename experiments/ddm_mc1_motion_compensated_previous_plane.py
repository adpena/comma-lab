#!/usr/bin/env python3
"""ddm_mc1 -- a decoder-derivable MOTION-COMPENSATED previous-field plane, priced first.

The shipped integer HPAC context mixer conditions on the previous pair's semantic
field CO-LOCATED (``prepare_frame_context`` one-hots ``previous_raw`` into
``conv_past``).  Between pairs the ego-vehicle moves; the classes that carry the
bits move in the image.  This module asks, in order:

  1. ``rows``    -- run the SHIPPED encoder over the exact retained field on the fs2
                    fire tree and RECORD every coding row the RC64 coder was handed
                    (float32, 5 classes, all 117,964,800 positions).  The emitted
                    stream must be byte-identical to the shipped 113,411 B stream;
                    otherwise the rows are not the coder's rows and nothing below
                    is admissible.
  2. ``motion``  -- for each candidate INTEGER motion model, estimate the transform
                    aligning field[t-2] -> field[t-1] from decoded data only, apply it
                    once more (constant velocity) to field[t-1], and store the MC
                    plane.  Zero archive bytes.  Pairs 0-1 are co-located.
  3. ``ceiling`` -- price the MC plane as an UNCONSUMED conditioning axis with the
                    ``ddm_mi1`` instrument: ``q' = sigma(logit(1-pmax) + beta_cell)``
                    on the argmax indicator, one offset per context cell, Newton fit,
                    2-fold cross-fitted (pair-level and position-level splits), on
                    the live (non-saturated) positions.  Plus a 5-way log-linear
                    tilt generalisation and the bare categorical family the charter
                    names.  Held-out bits saved / 8 is a REFUSAL-ONLY ceiling.

AXIS: ``[macOS-CPU advisory / scorer-free EXACT byte measurement]`` for the rows
control; every ceiling number is a model-ledger code length, never a byte claim.
ALWAYS KEEP THE PAYLOAD: rows, MC planes, motion parameters, and the control stream
are all persisted with sha256 + bytes.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_jg2_tail_reencode as jg2  # noqa: E402

STORE_AP = Path("/Volumes/APDataStore/pact/ddm_mc1_motion_compensated_previous_plane")
STORE_V = Path("/Volumes/VertigoDataTier/pact/ddm_mc1_motion_compensated_previous_plane")
FS2_RUNTIME = Path(
    "/Volumes/VertigoDataTier/pact/ddm_fs2_carrier_resolve/fire_runtime_D_alternation"
)
FS2_ARCHIVE_BYTES = 180_023
FS2_ARCHIVE_SHA256 = "a8f3a3791499b2b62ee4d16bc67f15f819f454dc9b88e3cce04fe50a30427bb6"
FS2_STREAM_BYTES = 113_411
FS2_STREAM_SHA256 = "5601d6fd792c60c176e7cb7478e6033c4ed9a7e87404582340ed3f50ed60cfe3"
TOKENS = Path(
    "/Volumes/VertigoDataTier/pact/ddm_afr1_tile48_receiver_identity/identity_v1/"
    "out/.f26_decode_checkpoints/tokens_cpu_stage_complete.u8"
)
TOKENS_BYTES = 117_964_800
TOKENS_SHA256 = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"

N, H, W = 600, 384, 512
PLANE = H * W
NUM_CLASSES = 5
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
AXIS = "[macOS-CPU advisory / scorer-free EXACT byte measurement]"
MIN_FREE_AP = 1_500_000_000
MOTION_MODELS = ("shift", "zoom", "planar", "block", "block_gated", "block_median3")
CEILING_SEEDS = (20260824, 777, 31337)
REFUSE_BELOW_BYTES = 5_000.0


class Mc1Error(RuntimeError):
    """A custody, identity, or storage gate refused."""


# --------------------------------------------------------------------------------------
# Custody helpers.
# --------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def require_store() -> None:
    for root in (STORE_AP, STORE_V):
        root.mkdir(parents=True, exist_ok=True)
    stat = os.statvfs(STORE_AP)
    free = stat.f_bavail * stat.f_frsize
    if free < MIN_FREE_AP:
        raise Mc1Error(f"APDataStore free {free} B < {MIN_FREE_AP} B; refusing")


def load_field() -> np.ndarray:
    if TOKENS.stat().st_size != TOKENS_BYTES:
        raise Mc1Error("token field has the wrong size")
    return np.memmap(TOKENS, dtype=np.uint8, mode="r", shape=(N, H, W))


def verify_inputs() -> dict[str, Any]:
    archive = FS2_RUNTIME / "archive.zip"
    facts = {"archive": file_fact(archive), "tokens": file_fact(TOKENS)}
    if facts["archive"]["bytes"] != FS2_ARCHIVE_BYTES or facts["archive"]["sha256"] != FS2_ARCHIVE_SHA256:
        raise Mc1Error("fs2 archive custody failed")
    if facts["tokens"]["bytes"] != TOKENS_BYTES or facts["tokens"]["sha256"] != TOKENS_SHA256:
        raise Mc1Error("token field custody failed")
    return facts


def progress(record: dict[str, Any]) -> None:
    record = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **record}
    print(json.dumps(record, sort_keys=True), flush=True)


# --------------------------------------------------------------------------------------
# Stage 1: the shipped coder's rows, recorded along its own trajectory.
# --------------------------------------------------------------------------------------


def rows_paths() -> dict[str, Path]:
    return {
        "rows": STORE_V / "rows" / "coding_rows.f32.npy",
        "base_argmax": STORE_V / "rows" / "base_argmax.u8.npy",
        "boundary": STORE_V / "rows" / "boundary_bucket.u8.npy",
        "stream": STORE_AP / "rows" / "control_stream.bin",
        "ledger": STORE_AP / "rows" / "bits_per_frame.f64.npy",
        "result": STORE_AP / "rows" / "ROWS_RESULT.json",
    }


def stage_rows(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    require_store()
    facts = verify_inputs()
    paths = rows_paths()
    if paths["result"].is_file() and not args.force:
        raise Mc1Error(f"rows already recorded at {paths['result']}; pass --force to redo")
    work_args = SimpleNamespace(store=str(STORE_AP / "rows_work"), runtime_root=str(FS2_RUNTIME))
    env = jg2._prepare(work_args, "mc1_rows")
    residual, renderer, renderer_dir = env["residual"], env["renderer"], env["renderer_dir"]
    parts, route_b, library = env["parts"], env["route_b"], env["library"]
    shipped_stream = parts.token_stream
    if len(shipped_stream) != FS2_STREAM_BYTES or hashlib.sha256(shipped_stream).hexdigest() != FS2_STREAM_SHA256:
        raise Mc1Error("fs2 token stream custody failed")

    from runtime.free_corrector import FreeCorrector  # type: ignore[import-not-found]
    from runtime.hpac_inference import optimize_sparse_evaluator  # type: ignore[import-not-found]

    tokens = load_field()
    device = torch.device("cpu")
    base_hpac = residual.materialize_ihs1(parts.hpac_blob, renderer)
    model = renderer.load_hpac(base_hpac, device)
    masks = renderer.group_masks(device)
    sparse = residual._sparse_class(renderer_dir)(model, renderer.EVAL_H, renderer.EVAL_W)
    corrector = FreeCorrector(renderer.EVAL_H * renderer.EVAL_W)
    group_plans = []
    for mask in masks:
        flat_positions = np.flatnonzero(mask.detach().cpu().numpy().reshape(-1))
        group_plans.append((torch.from_numpy(flat_positions).to(device), flat_positions))
    encoder = route_b.NativeRc64Encoder(library)

    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.lib.format.open_memmap(paths["rows"], mode="w+", dtype=np.float32, shape=(N, PLANE, NUM_CLASSES))
    base_argmax = np.lib.format.open_memmap(paths["base_argmax"], mode="w+", dtype=np.uint8, shape=(N, PLANE))
    boundary_out = np.lib.format.open_memmap(paths["boundary"], mode="w+", dtype=np.uint8, shape=(N, PLANE))
    per_frame = np.zeros(N, dtype=np.float64)
    started = time.perf_counter()
    with torch.inference_mode():
        optimize_sparse_evaluator(sparse)
        previous = torch.zeros((1, H, W), dtype=torch.long, device=device)
        for frame in range(N):
            index = torch.tensor([frame], dtype=torch.long, device=device)
            current = torch.zeros_like(previous)
            context = model.prepare_frame_context(index, previous)
            if frame:
                previous_cpu = previous[0].to(device="cpu", dtype=torch.uint8).numpy()
                boundary = residual._boundary_buckets(previous_cpu).reshape(-1)
            else:
                boundary = np.full(PLANE, 4, dtype=np.uint8)
            corrector.begin_frame(boundary)
            boundary_out[frame] = boundary
            plane_target = np.asarray(tokens[frame], dtype=np.uint8).reshape(-1)
            frame_bits = 0.0
            for group, (device_positions, flat_positions) in enumerate(group_plans):
                selected = sparse.selected_logits(current, context, group)
                base_logits = selected.cpu().numpy()
                predicted = base_logits.argmax(axis=1).astype(np.int64)
                feature = boundary[flat_positions].astype(np.int64) * NUM_CLASSES + predicted
                corrected = base_logits + parts.table.values[feature]
                probability = residual._probability_table(corrected, renderer.HPAC_LOGIT_PRECISION)
                state = corrector.group_state(probability, predicted, flat_positions)
                coding = corrector.coding_row(state)
                symbols = plane_target[flat_positions].astype(np.int64)
                frame_bits += jg2._row_bits(coding, symbols)
                encoder.encode(symbols.astype(np.int32), coding)
                corrector.observe(state, symbols)
                current.reshape(-1)[device_positions] = torch.from_numpy(symbols).to(device)
                rows[frame, flat_positions, :] = coding.astype(np.float32)
                base_argmax[frame, flat_positions] = predicted.astype(np.uint8)
            per_frame[frame] = frame_bits
            frame_tokens = current[0].to(device="cpu", dtype=torch.uint8).numpy()
            if not np.array_equal(frame_tokens.reshape(-1), plane_target):
                raise Mc1Error(f"frame {frame}: encoded field diverged from the target")
            corrector.end_frame(frame_tokens.reshape(-1))
            previous = current
            if frame % 25 == 24 or frame == N - 1:
                progress(
                    {
                        "stage": "rows",
                        "frame": frame + 1,
                        "code_bytes_so_far": per_frame.sum() / 8.0,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
    payload = encoder.finish()
    if not payload.startswith(route_b.TOKEN_MAGIC):
        raise Mc1Error("RC64 payload lost its magic")
    size = int(encoder.library.rc64_encoder_size(encoder.context))
    pointer = encoder.library.rc64_encoder_data(encoder.context)
    body = ctypes.string_at(pointer, size)
    rows.flush()
    base_argmax.flush()
    boundary_out.flush()
    atomic_bytes(paths["stream"], body)
    np.save(paths["ledger"], per_frame)
    identical = body == shipped_stream
    result = {
        "schema": "ddm_mc1_rows.v1",
        "axis": AXIS,
        "score_claim": False,
        "inputs": facts,
        "rc64_build": env["build"],
        "shipped_stream": {"bytes": len(shipped_stream), "sha256": hashlib.sha256(shipped_stream).hexdigest()},
        "emitted_stream": file_fact(paths["stream"]),
        "byte_identical": identical,
        "code_bits": float(per_frame.sum()),
        "code_bytes_ideal": float(per_frame.sum() / 8.0),
        "rows": file_fact(paths["rows"]),
        "base_argmax": file_fact(paths["base_argmax"]),
        "boundary": file_fact(paths["boundary"]),
        "ledger": file_fact(paths["ledger"]),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(paths["result"], result)
    if not identical:
        raise Mc1Error("CONTROL FAILED: emitted stream is not the shipped stream; rows are not admissible")
    return result


# --------------------------------------------------------------------------------------
# Stage 2: integer motion models, estimated from decoded data only.
# --------------------------------------------------------------------------------------


def edge_band(plane: np.ndarray, radius: int) -> np.ndarray:
    """Positions within ``radius`` (4-neighbour dilation) of a class change."""
    edge = np.zeros(plane.shape, dtype=bool)
    edge[1:] |= plane[1:] != plane[:-1]
    edge[:-1] |= plane[:-1] != plane[1:]
    edge[:, 1:] |= plane[:, 1:] != plane[:, :-1]
    edge[:, :-1] |= plane[:, :-1] != plane[:, 1:]
    active = edge.copy()
    for _ in range(radius):
        grown = active.copy()
        grown[1:] |= active[:-1]
        grown[:-1] |= active[1:]
        grown[:, 1:] |= active[:, :-1]
        grown[:, :-1] |= active[:, 1:]
        active = grown
    return active


BAND_RADIUS = 3
SHIFT_RANGE = 8
BLOCK = 64
BLOCK_ROWS, BLOCK_COLS = H // BLOCK, W // BLOCK
ZOOM_DEN = 1024
ZOOM_KS = tuple(range(0, 42, 2))
ZOOM_SHIFT = 3
PLANAR_KS = tuple(range(0, 42, 2))
PLANAR_SHIFT_BITS = 16
PLANAR_Y0S = (160, 176, 192, 208)
PLANAR_X0 = 256
PLANAR_SHIFT = 2
YY, XX = np.meshgrid(np.arange(H, dtype=np.int64), np.arange(W, dtype=np.int64), indexing="ij")


def shift_candidates(limit: int) -> list[tuple[int, int]]:
    """Identity first, then outward: the first maximum wins, so ties prefer small motion."""
    cands = [(dy, dx) for dy in range(-limit, limit + 1) for dx in range(-limit, limit + 1)]
    return sorted(cands, key=lambda d: (abs(d[0]) + abs(d[1]), abs(d[0]), abs(d[1]), d[0], d[1]))


def zoom_source(y: np.ndarray, x: np.ndarray, k: int, dy: int, dx: int) -> tuple[np.ndarray, np.ndarray]:
    """Backward source for ``p' = c + s (p - c) + d`` with ``s = 1 + k/ZOOM_DEN`` (integer, exact)."""
    cy, cx = H // 2, W // 2
    den = 2 * (ZOOM_DEN + k)
    sy = cy + ((y - dy - cy) * (2 * ZOOM_DEN) + (ZOOM_DEN + k)) // den
    sx = cx + ((x - dx - cx) * (2 * ZOOM_DEN) + (ZOOM_DEN + k)) // den
    return np.clip(sy, 0, H - 1), np.clip(sx, 0, W - 1)


def planar_source(y: np.ndarray, x: np.ndarray, k: int, y0: int, dy: int, dx: int) -> tuple[np.ndarray, np.ndarray]:
    """Ground-plane flow: dy ~ k (y-y0)^2, dx ~ k (x-x0)(y-y0) below the horizon row ``y0``."""
    rel = np.maximum(y - y0, 0)
    fy = (k * rel * rel) >> PLANAR_SHIFT_BITS
    fx = (k * (x - PLANAR_X0) * rel) >> PLANAR_SHIFT_BITS
    return np.clip(y - dy - fy, 0, H - 1), np.clip(x - dx - fx, 0, W - 1)


def block_ids(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    return (y // BLOCK) * BLOCK_COLS + (x // BLOCK)


class MotionModel:
    """One integer motion family: ``estimate`` from (A -> B), ``predict`` by re-applying to B.

    ``extrapolate`` turns the history of raw transition estimates (oldest first, the last
    one being A=field[t-2] -> B=field[t-1]) into the parameters applied to field[t-1].
    Constant velocity is the default: re-use the last transition.
    """

    name = "identity"
    history_depth = 1

    def estimate(self, a: np.ndarray, b: np.ndarray, by: np.ndarray, bx: np.ndarray) -> Any:
        return None

    def extrapolate(self, history: list[Any]) -> Any:
        return history[-1]

    def predict(self, b: np.ndarray, params: Any) -> np.ndarray:
        return np.array(b, dtype=np.uint8, copy=True)

    @staticmethod
    def encode_params(params: Any) -> list[int]:
        return []


class ShiftModel(MotionModel):
    name = "shift"

    def __init__(self) -> None:
        self.cands = shift_candidates(SHIFT_RANGE)

    def estimate(self, a, b, by, bx):
        target = b[by, bx]
        best, best_score = (0, 0), -1
        for dy, dx in self.cands:
            sy = np.clip(by - dy, 0, H - 1)
            sx = np.clip(bx - dx, 0, W - 1)
            score = int((a[sy, sx] == target).sum())
            if score > best_score:
                best, best_score = (dy, dx), score
        return best

    def predict(self, b, params):
        dy, dx = params
        sy = np.clip(YY - dy, 0, H - 1)
        sx = np.clip(XX - dx, 0, W - 1)
        return b[sy, sx]

    @staticmethod
    def encode_params(params):
        return [int(params[0]), int(params[1])]


class BlockModel(MotionModel):
    name = "block"

    def __init__(self) -> None:
        self.cands = shift_candidates(SHIFT_RANGE)

    def estimate(self, a, b, by, bx):
        target = b[by, bx]
        blocks = block_ids(by, bx)
        count = BLOCK_ROWS * BLOCK_COLS
        best = np.zeros((count, 2), dtype=np.int64)
        best_score = np.full(count, -1, dtype=np.int64)
        for dy, dx in self.cands:
            sy = np.clip(by - dy, 0, H - 1)
            sx = np.clip(bx - dx, 0, W - 1)
            agree = (a[sy, sx] == target).astype(np.int64)
            score = np.bincount(blocks, weights=agree, minlength=count).astype(np.int64)
            better = score > best_score
            best_score[better] = score[better]
            best[better] = (dy, dx)
        return best

    def predict(self, b, params):
        blocks = block_ids(YY, XX)
        dy = params[blocks, 0]
        dx = params[blocks, 1]
        sy = np.clip(YY - dy, 0, H - 1)
        sx = np.clip(XX - dx, 0, W - 1)
        return b[sy, sx]

    @staticmethod
    def encode_params(params):
        return [int(v) for v in np.asarray(params).reshape(-1)]


class BlockGatedModel(BlockModel):
    """Block shift applied only where it beat identity on the estimation band by a margin."""

    name = "block_gated"
    GATE_FRACTION = 0.10
    GATE_MIN = 8

    def estimate(self, a, b, by, bx):
        target = b[by, bx]
        blocks = block_ids(by, bx)
        count = BLOCK_ROWS * BLOCK_COLS
        population = np.bincount(blocks, minlength=count).astype(np.int64)
        best = np.zeros((count, 2), dtype=np.int64)
        best_score = np.full(count, -1, dtype=np.int64)
        identity_score = np.zeros(count, dtype=np.int64)
        for dy, dx in self.cands:
            sy = np.clip(by - dy, 0, H - 1)
            sx = np.clip(bx - dx, 0, W - 1)
            agree = (a[sy, sx] == target).astype(np.int64)
            score = np.bincount(blocks, weights=agree, minlength=count).astype(np.int64)
            if dy == 0 and dx == 0:
                identity_score = score
            better = score > best_score
            best_score[better] = score[better]
            best[better] = (dy, dx)
        margin = np.maximum(self.GATE_MIN, (population * self.GATE_FRACTION).astype(np.int64))
        keep = (best_score - identity_score) >= margin
        gated = np.where(keep[:, None], best, 0)
        return gated


class BlockMedian3Model(BlockModel):
    """Per-block component-wise median of the last three transition estimates."""

    name = "block_median3"
    history_depth = 3

    def extrapolate(self, history):
        stack = np.stack(history[-self.history_depth :], axis=0)
        return np.median(stack, axis=0).astype(np.int64)


class ZoomModel(MotionModel):
    name = "zoom"

    def __init__(self) -> None:
        shifts = shift_candidates(ZOOM_SHIFT)
        self.cands = [(k, dy, dx) for k in ZOOM_KS for dy, dx in shifts]

    def estimate(self, a, b, by, bx):
        target = b[by, bx]
        best, best_score = (0, 0, 0), -1
        for k, dy, dx in self.cands:
            sy, sx = zoom_source(by, bx, k, dy, dx)
            score = int((a[sy, sx] == target).sum())
            if score > best_score:
                best, best_score = (k, dy, dx), score
        return best

    def predict(self, b, params):
        k, dy, dx = params
        sy, sx = zoom_source(YY, XX, k, dy, dx)
        return b[sy, sx]

    @staticmethod
    def encode_params(params):
        return [int(v) for v in params]


class PlanarModel(MotionModel):
    name = "planar"

    def __init__(self) -> None:
        shifts = shift_candidates(PLANAR_SHIFT)
        self.cands = [(k, y0, dy, dx) for k in PLANAR_KS for y0 in PLANAR_Y0S for dy, dx in shifts]

    def estimate(self, a, b, by, bx):
        target = b[by, bx]
        best, best_score = (0, PLANAR_Y0S[0], 0, 0), -1
        for k, y0, dy, dx in self.cands:
            sy, sx = planar_source(by, bx, k, y0, dy, dx)
            score = int((a[sy, sx] == target).sum())
            if score > best_score:
                best, best_score = (k, y0, dy, dx), score
        return best

    def predict(self, b, params):
        k, y0, dy, dx = params
        sy, sx = planar_source(YY, XX, k, y0, dy, dx)
        return b[sy, sx]

    @staticmethod
    def encode_params(params):
        return [int(v) for v in params]


def motion_model(name: str) -> MotionModel:
    return {
        "shift": ShiftModel,
        "zoom": ZoomModel,
        "planar": PlanarModel,
        "block": BlockModel,
        "block_gated": BlockGatedModel,
        "block_median3": BlockMedian3Model,
    }[name]()


def motion_paths(name: str) -> dict[str, Path]:
    return {
        "plane": STORE_V / "motion" / f"mc_plane_{name}.u8.npy",
        "params": STORE_AP / "motion" / f"params_{name}.json",
        "result": STORE_AP / "motion" / f"MOTION_{name}.json",
    }


def per_class_iou(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    out = np.zeros(NUM_CLASSES, dtype=np.float64)
    for c in range(NUM_CLASSES):
        p, t = pred == c, truth == c
        inter = int((p & t).sum())
        union = int((p | t).sum())
        out[c] = inter / union if union else 1.0
    return out


def stage_motion(args: argparse.Namespace) -> dict[str, Any]:
    require_store()
    facts = verify_inputs()
    field = load_field()
    results = {}
    for name in args.models:
        paths = motion_paths(name)
        if paths["result"].is_file() and not args.force:
            progress({"stage": "motion", "model": name, "event": "skip_existing"})
            results[name] = json.loads(paths["result"].read_text())
            continue
        model = motion_model(name)
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        plane = np.lib.format.open_memmap(paths["plane"], mode="w+", dtype=np.uint8, shape=(N, H, W))
        params_out: list[list[int]] = []
        iou_mc = np.zeros((N, NUM_CLASSES), dtype=np.float64)
        iou_co = np.zeros((N, NUM_CLASSES), dtype=np.float64)
        band_agree_mc = np.zeros(N, dtype=np.float64)
        band_agree_co = np.zeros(N, dtype=np.float64)
        band_size = np.zeros(N, dtype=np.int64)
        history: list[Any] = []
        started = time.perf_counter()
        for t in range(N):
            truth = np.asarray(field[t], dtype=np.uint8)
            if t < 2:
                prev = np.asarray(field[t - 1], dtype=np.uint8) if t else np.zeros((H, W), dtype=np.uint8)
                pred = prev.copy()
                params_out.append([])
            else:
                a = np.asarray(field[t - 2], dtype=np.uint8)
                prev = np.asarray(field[t - 1], dtype=np.uint8)
                band = edge_band(prev, BAND_RADIUS)
                by, bx = np.nonzero(band)
                history.append(model.estimate(a, prev, by, bx))
                history = history[-model.history_depth :]
                params = model.extrapolate(history)
                pred = model.predict(prev, params)
                params_out.append(model.encode_params(params))
            plane[t] = pred
            iou_mc[t] = per_class_iou(pred, truth)
            iou_co[t] = per_class_iou(prev, truth)
            band_t = edge_band(truth, BAND_RADIUS)
            band_size[t] = int(band_t.sum())
            band_agree_mc[t] = float((pred[band_t] == truth[band_t]).mean()) if band_size[t] else 1.0
            band_agree_co[t] = float((prev[band_t] == truth[band_t]).mean()) if band_size[t] else 1.0
            if t % 50 == 49 or t == N - 1:
                progress(
                    {
                        "stage": "motion",
                        "model": name,
                        "pair": t + 1,
                        "elapsed_seconds": time.perf_counter() - started,
                        "mean_iou_mc_lane": float(iou_mc[2 : t + 1, 1].mean()),
                        "mean_iou_co_lane": float(iou_co[2 : t + 1, 1].mean()),
                    }
                )
        plane.flush()
        atomic_json(paths["params"], {"schema": "ddm_mc1_motion_params.v1", "model": name, "params": params_out})
        sel = slice(2, N)
        result = {
            "schema": "ddm_mc1_motion.v1",
            "model": name,
            "inputs": facts,
            "plane": file_fact(paths["plane"]),
            "params": file_fact(paths["params"]),
            "pairs_motion_compensated": N - 2,
            "mean_iou_mc_by_class": {CLASS_NAMES[c]: float(iou_mc[sel, c].mean()) for c in range(NUM_CLASSES)},
            "mean_iou_colocated_by_class": {CLASS_NAMES[c]: float(iou_co[sel, c].mean()) for c in range(NUM_CLASSES)},
            "band_agreement_mc": float(band_agree_mc[sel].mean()),
            "band_agreement_colocated": float(band_agree_co[sel].mean()),
            "band_radius": BAND_RADIUS,
            "mean_band_positions": float(band_size[sel].mean()),
            "elapsed_seconds": time.perf_counter() - started,
            "score_claim": False,
        }
        # Record FIRST, bulk payloads after: a failed bulk save must not strand the receipt.
        atomic_json(paths["result"], result)
        np.save(STORE_AP / "motion" / f"iou_mc_{name}.f64.npy", iou_mc)
        np.save(STORE_AP / "motion" / f"iou_co_{name}.f64.npy", iou_co)
        results[name] = result
        progress({"stage": "motion", "model": name, "event": "done", **{k: v for k, v in result.items() if isinstance(v, (int, float))}})
    return results


def stage_oracle(args: argparse.Namespace) -> dict[str, Any]:
    """DIAGNOSTIC ONLY (not decoder-derivable): estimate the motion t-1 -> t on the TRUE
    field_t and align field_{t-1} with it.  This is the best any member of the family can
    do on alignment; it separates 'the extrapolation is bad' from 'the field's inter-pair
    change is not rigid motion'.  Never a candidate; never enters the ceiling verdict."""
    require_store()
    verify_inputs()
    field = load_field()
    results = {}
    for name in args.models:
        model = motion_model(name)
        oracle_paths = motion_paths(f"oracle_{name}")
        oracle_paths["plane"].parent.mkdir(parents=True, exist_ok=True)
        plane = np.lib.format.open_memmap(oracle_paths["plane"], mode="w+", dtype=np.uint8, shape=(N, H, W))
        plane[0] = 0
        iou_or = np.zeros((N, NUM_CLASSES), dtype=np.float64)
        iou_co = np.zeros((N, NUM_CLASSES), dtype=np.float64)
        band_or = np.zeros(N, dtype=np.float64)
        band_co = np.zeros(N, dtype=np.float64)
        identity_count = 0
        params_out: list[list[int]] = [[]]
        started = time.perf_counter()
        for t in range(1, N):
            truth = np.asarray(field[t], dtype=np.uint8)
            prev = np.asarray(field[t - 1], dtype=np.uint8)
            band = edge_band(truth, BAND_RADIUS)
            by, bx = np.nonzero(band)
            params = model.estimate(prev, truth, by, bx)
            encoded = model.encode_params(params)
            params_out.append(encoded)
            identity_count += int(all(v == 0 for v in encoded) if name != "planar" else (encoded[0] == 0 and encoded[2] == 0 and encoded[3] == 0))
            pred = model.predict(prev, params)
            plane[t] = pred
            iou_or[t] = per_class_iou(pred, truth)
            iou_co[t] = per_class_iou(prev, truth)
            band_or[t] = float((pred[band] == truth[band]).mean()) if by.size else 1.0
            band_co[t] = float((prev[band] == truth[band]).mean()) if by.size else 1.0
        sel = slice(1, N)
        result = {
            "schema": "ddm_mc1_motion_oracle.v1",
            "model": name,
            "authority": "DIAGNOSTIC ORACLE: uses field_t to estimate; not decoder-derivable; never a candidate",
            "pairs": N - 1,
            "identity_fraction": identity_count / (N - 1),
            "mean_iou_oracle_by_class": {CLASS_NAMES[c]: float(iou_or[sel, c].mean()) for c in range(NUM_CLASSES)},
            "mean_iou_colocated_by_class": {CLASS_NAMES[c]: float(iou_co[sel, c].mean()) for c in range(NUM_CLASSES)},
            "band_agreement_oracle": float(band_or[sel].mean()),
            "band_agreement_colocated": float(band_co[sel].mean()),
            "elapsed_seconds": time.perf_counter() - started,
            "score_claim": False,
        }
        plane.flush()
        result["plane"] = file_fact(oracle_paths["plane"])
        # Record FIRST (twice: the oracle receipt, and a motion-shaped receipt so the
        # ceiling stage can price the oracle plane under the model name ``oracle_<name>``).
        atomic_json(STORE_AP / "motion" / f"ORACLE_{name}.json", result)
        atomic_json(oracle_paths["result"], {**result, "model": f"oracle_{name}", "mean_iou_mc_by_class": result["mean_iou_oracle_by_class"], "band_agreement_mc": result["band_agreement_oracle"]})
        atomic_json(oracle_paths["params"], {"schema": "ddm_mc1_oracle_params.v1", "model": name, "params": params_out})
        atomic_json(STORE_AP / "motion" / f"oracle_params_{name}.json", {"schema": "ddm_mc1_oracle_params.v1", "model": name, "params": params_out})
        results[name] = result
        progress({"stage": "oracle", "model": name, "event": "done", "band_oracle": result["band_agreement_oracle"], "band_co": result["band_agreement_colocated"], "lane_oracle": result["mean_iou_oracle_by_class"]["Lane"], "lane_co": result["mean_iou_colocated_by_class"]["Lane"]})
    return results


# --------------------------------------------------------------------------------------
# Stage 3: the ceiling -- mi1's instrument on the exact rows, plus generalisations.
# --------------------------------------------------------------------------------------


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def fit_indicator_offsets(cell: np.ndarray, logit_q: np.ndarray, flip: np.ndarray, cells: int, iters: int = 30) -> np.ndarray:
    """Per-cell beta maximising the held-in log-likelihood of ``flip ~ sigma(logit_q + beta)``.

    Independent 1-D Newton problems, vectorised by ``bincount``; steps clipped at +-4.
    """
    beta = np.zeros(cells, dtype=np.float64)
    f = flip.astype(np.float64)
    for _ in range(iters):
        q = _sigmoid(logit_q + beta[cell])
        grad = np.bincount(cell, weights=f - q, minlength=cells)
        hess = np.bincount(cell, weights=q * (1.0 - q), minlength=cells) + 1e-9
        step = np.clip(grad / hess, -4.0, 4.0)
        beta += step
        if float(np.abs(step).max()) < 1e-7:
            break
    return beta


def indicator_bits(logit_q: np.ndarray, flip: np.ndarray, offset: np.ndarray) -> float:
    q = np.clip(_sigmoid(logit_q + offset), 1e-12, 1.0 - 1e-12)
    return float(-(np.where(flip, np.log2(q), np.log2(1.0 - q))).sum())


def fit_tilt(cell: np.ndarray, logp: np.ndarray, truth: np.ndarray, cells: int, iters: int = 15, ridge: float = 1e-3) -> np.ndarray:
    """Per-cell 5-way log-linear tilt ``q'(c) ~ p(c) exp(beta[cell, c])``; damped Newton, ridge."""
    beta = np.zeros((cells, NUM_CLASSES), dtype=np.float64)
    logp = logp.astype(np.float64)
    onehot = np.zeros((truth.size, NUM_CLASSES), dtype=np.float32)
    onehot[np.arange(truth.size), truth] = 1.0
    for _ in range(iters):
        z = logp + beta[cell]
        z -= z.max(axis=1, keepdims=True)
        q = np.exp(z, dtype=np.float64)
        q /= q.sum(axis=1, keepdims=True)
        q = q.astype(np.float32)  # bincount accumulates its weights in float64 regardless
        del z
        grad = np.zeros((cells, NUM_CLASSES), dtype=np.float64)
        hess = np.zeros((cells, NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
        diff = onehot - q
        for c in range(NUM_CLASSES):
            grad[:, c] = np.bincount(cell, weights=diff[:, c], minlength=cells)
            for d in range(c, NUM_CLASSES):
                w = (q[:, c] * (1.0 if c == d else 0.0) - q[:, c] * q[:, d])
                h = np.bincount(cell, weights=w, minlength=cells)
                hess[:, c, d] = h
                hess[:, d, c] = h
        del diff
        grad -= ridge * beta
        hess += ridge * np.eye(NUM_CLASSES)[None]
        step = np.linalg.solve(hess, grad[:, :, None])[:, :, 0]
        step = np.clip(step, -4.0, 4.0)
        beta += step
        if float(np.abs(step).max()) < 1e-7:
            break
    return beta


def tilt_bits(logp: np.ndarray, truth: np.ndarray, offsets: np.ndarray | None) -> float:
    z = logp.astype(np.float64)
    if offsets is not None:
        z += offsets
    z -= z.max(axis=1, keepdims=True)
    q = np.exp(z)
    q /= q.sum(axis=1, keepdims=True)
    picked = np.clip(q[np.arange(truth.size), truth], 1e-300, 1.0)
    return float(-np.log2(picked).sum())


def cross_fit_indicator(cell, logit_q, flip, cells, fold, base_bits) -> dict[str, float]:
    held = 0.0
    insample = 0.0
    for k in (0, 1):
        train, test = fold != k, fold == k
        beta = fit_indicator_offsets(cell[train], logit_q[train], flip[train], cells)
        held += indicator_bits(logit_q[test], flip[test], beta[cell[test]])
        insample += indicator_bits(logit_q[train], flip[train], beta[cell[train]])
    beta_all = fit_indicator_offsets(cell, logit_q, flip, cells)
    # The two training folds are complementary halves, so their in-sample bits sum to
    # the whole data coded once (with fold-local offsets); held-out likewise sums to once.
    return {
        "held_out_bytes_saved": (base_bits - held) / 8.0,
        "in_sample_bytes_saved": (base_bits - insample) / 8.0,
        "max_abs_beta": float(np.abs(beta_all).max()),
    }


def cross_fit_tilt(cell, logp, truth, cells, fold, base_bits) -> dict[str, float]:
    held = 0.0
    insample = 0.0
    for k in (0, 1):
        train, test = fold != k, fold == k
        beta = fit_tilt(cell[train], logp[train], truth[train], cells)
        held += tilt_bits(logp[test], truth[test], beta[cell[test]])
        insample += tilt_bits(logp[train], truth[train], beta[cell[train]])
    return {
        "held_out_bytes_saved": (base_bits - held) / 8.0,
        "in_sample_bytes_saved": (base_bits - insample) / 8.0,
    }


def bare_categorical_bits(ctx: np.ndarray, truth: np.ndarray, contexts: int, fold: np.ndarray, alpha: float = 0.5) -> float:
    """Held-out code length of ``truth`` under a KT-smoothed categorical per context cell."""
    total = 0.0
    for k in (0, 1):
        train, test = fold != k, fold == k
        counts = np.bincount(ctx[train] * NUM_CLASSES + truth[train], minlength=contexts * NUM_CLASSES).reshape(contexts, NUM_CLASSES).astype(np.float64)
        prob = (counts + alpha) / (counts.sum(axis=1, keepdims=True) + alpha * NUM_CLASSES)
        total += float(-np.log2(prob[ctx[test], truth[test]]).sum())
    return total


def build_cells(name: str, coloc, mc, arg, bd) -> tuple[np.ndarray, int]:
    if name == "none":
        return np.zeros(coloc.size, dtype=np.int64), 1
    if name == "coloc":
        return coloc.astype(np.int64), 5
    if name == "mc":
        return mc.astype(np.int64), 5
    if name == "agree":
        return (mc == coloc).astype(np.int64), 2
    if name == "mc_x_coloc":
        return mc.astype(np.int64) * 5 + coloc, 25
    if name == "mc_x_arg":
        return mc.astype(np.int64) * 5 + arg, 25
    if name == "coloc_x_arg":
        return coloc.astype(np.int64) * 5 + arg, 25
    if name == "mc_x_coloc_x_arg":
        return (mc.astype(np.int64) * 5 + coloc) * 5 + arg, 125
    if name == "mc_x_arg_x_bd":
        return (mc.astype(np.int64) * 5 + arg) * 5 + bd, 125
    if name == "coloc_x_arg_x_bd":
        return (coloc.astype(np.int64) * 5 + arg) * 5 + bd, 125
    if name == "mc_x_coloc_x_arg_x_bd":
        return ((mc.astype(np.int64) * 5 + coloc) * 5 + arg) * 5 + bd, 625
    raise Mc1Error(f"unknown cell family {name}")


INDICATOR_CELLS = (
    "none",
    "coloc",
    "coloc_x_arg",
    "coloc_x_arg_x_bd",
    "mc",
    "agree",
    "mc_x_coloc",
    "mc_x_arg",
    "mc_x_coloc_x_arg",
    "mc_x_arg_x_bd",
    "mc_x_coloc_x_arg_x_bd",
)
TILT_CELLS = ("none", "coloc_x_arg", "mc_x_arg", "mc_x_coloc_x_arg", "mc_x_coloc_x_arg_x_bd")


def load_live(rows_result: dict[str, Any]) -> dict[str, np.ndarray]:
    """One streaming pass -> compact per-position arrays over the live positions (pmax < 1 in float32)."""
    paths = rows_paths()
    rows = np.load(paths["rows"], mmap_mode="r")
    bd_all = np.load(paths["boundary"], mmap_mode="r")
    field = load_field()
    # Two passes so the compact arrays are preallocated once: the list-then-concatenate
    # form doubled the resident set transiently (the 23:47Z compressor ramp, three arms
    # loading at once).  Pass 1 counts live positions per pair; pass 2 fills in place.
    live_counts = np.zeros(N, dtype=np.int64)
    saturated = 0
    saturated_flips = 0
    for t in range(N):
        row = np.asarray(rows[t], dtype=np.float32)
        truth = np.asarray(field[t], dtype=np.uint8).reshape(-1)
        arg = row.argmax(axis=1)
        pmax = row[np.arange(PLANE), arg]
        sat = pmax >= np.float32(1.0)
        saturated += int(sat.sum())
        saturated_flips += int((sat & (truth != arg)).sum())
        live_counts[t] = PLANE - int(sat.sum())
    total = int(live_counts.sum())
    starts = np.concatenate([[0], np.cumsum(live_counts)[:-1]])
    out: dict[str, np.ndarray] = {
        "pair": np.empty(total, dtype=np.int16),
        "pos": np.empty(total, dtype=np.int32),
        "truth": np.empty(total, dtype=np.uint8),
        "arg": np.empty(total, dtype=np.uint8),
        "pmax": np.empty(total, dtype=np.float64),
        "logp": np.empty((total, NUM_CLASSES), dtype=np.float32),
        "bd": np.empty(total, dtype=np.uint8),
        "coloc": np.empty(total, dtype=np.uint8),
    }
    for t in range(N):
        row = np.asarray(rows[t], dtype=np.float32)
        truth = np.asarray(field[t], dtype=np.uint8).reshape(-1)
        arg = row.argmax(axis=1)
        pmax = row[np.arange(PLANE), arg]
        live = np.flatnonzero(pmax < np.float32(1.0))
        s, e = int(starts[t]), int(starts[t] + live_counts[t])
        if live.size != e - s:
            raise Mc1Error(f"pair {t}: live count drifted between passes")
        prev = np.asarray(field[t - 1], dtype=np.uint8).reshape(-1) if t else np.zeros(PLANE, dtype=np.uint8)
        out["pair"][s:e] = t
        out["pos"][s:e] = live
        out["truth"][s:e] = truth[live]
        out["arg"][s:e] = arg[live]
        out["pmax"][s:e] = pmax[live]
        out["logp"][s:e] = np.log(np.maximum(row[live].astype(np.float64), 1e-300)).astype(np.float32)
        out["bd"][s:e] = np.asarray(bd_all[t])[live]
        out["coloc"][s:e] = prev[live]
    out["saturated_positions"] = np.int64(saturated)
    out["saturated_flips"] = np.int64(saturated_flips)
    return out


def stage_ceiling(args: argparse.Namespace) -> dict[str, Any]:
    require_store()
    rows_result_path = rows_paths()["result"]
    if not rows_result_path.is_file():
        raise Mc1Error("rows stage has not produced ROWS_RESULT.json")
    rows_result = json.loads(rows_result_path.read_text())
    if not rows_result.get("byte_identical"):
        raise Mc1Error("rows control was not byte-identical; ceiling refused")
    started = time.perf_counter()
    live = load_live(rows_result)
    n_live = int(live["truth"].size)
    flip = live["truth"] != live["arg"]
    q = 1.0 - live["pmax"]
    q = np.clip(q, 1e-12, 1.0 - 1e-12)
    logit_q = np.log(q) - np.log1p(-q)
    zero = np.zeros(n_live, dtype=np.float64)
    base_indicator_bits = indicator_bits(logit_q, flip, zero)
    base_full_bits = tilt_bits(live["logp"], live["truth"], None)
    progress(
        {
            "stage": "ceiling",
            "event": "live_loaded",
            "live_positions": n_live,
            "saturated_positions": int(live["saturated_positions"]),
            "saturated_flips": int(live["saturated_flips"]),
            "base_indicator_bytes": base_indicator_bits / 8.0,
            "base_full_row_bytes": base_full_bits / 8.0,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    folds: dict[str, dict[int, np.ndarray]] = {"pair": {}, "position": {}}
    for seed in CEILING_SEEDS:
        rng = np.random.default_rng(seed)
        pair_perm = rng.permutation(N)
        pair_fold = np.zeros(N, dtype=np.int8)
        pair_fold[pair_perm[N // 2 :]] = 1
        folds["pair"][seed] = pair_fold[live["pair"].astype(np.int64)]
        folds["position"][seed] = (rng.permutation(n_live) % 2).astype(np.int8)

    report: dict[str, Any] = {
        "schema": "ddm_mc1_ceiling.v1",
        "axis": "[model-ledger code length on the exact fs2 coding rows; REFUSAL-ONLY, never a byte claim]",
        "score_claim": False,
        "rows_result": rows_result_path.as_posix(),
        "live_positions": n_live,
        "saturated_positions": int(live["saturated_positions"]),
        "saturated_flips": int(live["saturated_flips"]),
        "base_indicator_bytes": base_indicator_bits / 8.0,
        "base_full_row_bytes": base_full_bits / 8.0,
        "seeds": list(CEILING_SEEDS),
        "models": {},
    }
    bd = live["bd"].astype(np.int64)
    arg = live["arg"].astype(np.int64)
    coloc = live["coloc"]
    for name in args.models:
        mpaths = motion_paths(name)
        if not mpaths["result"].is_file():
            raise Mc1Error(f"motion stage missing for {name}")
        plane = np.load(mpaths["plane"], mmap_mode="r").reshape(N, PLANE)
        mc = np.empty(n_live, dtype=np.uint8)
        # gather the MC class per live position, pair by pair (the live arrays are pair-sorted)
        starts = np.searchsorted(live["pair"], np.arange(N), side="left")
        ends = np.searchsorted(live["pair"], np.arange(N), side="right")
        for t in range(N):
            s, e = int(starts[t]), int(ends[t])
            if e > s:
                mc[s:e] = np.asarray(plane[t])[live["pos"][s:e]]
        model_report: dict[str, Any] = {"motion": json.loads(mpaths["result"].read_text()), "indicator": {}, "tilt": {}, "bare": {}}
        for split in ("pair", "position"):
            model_report["indicator"][split] = {}
            model_report["tilt"][split] = {}
            for cells_name in INDICATOR_CELLS:
                cell, count = build_cells(cells_name, coloc, mc, arg, bd)
                per_seed = {}
                for seed in (CEILING_SEEDS if cells_name in ("mc_x_arg", "mc_x_coloc_x_arg", "mc_x_arg_x_bd") else CEILING_SEEDS[:1]):
                    per_seed[str(seed)] = cross_fit_indicator(cell, logit_q, flip, count, folds[split][seed], base_indicator_bits)
                model_report["indicator"][split][cells_name] = {"cells": count, "by_seed": per_seed}
                progress({"stage": "ceiling", "model": name, "family": "indicator", "split": split, "cells": cells_name, **per_seed[str(CEILING_SEEDS[0])], "elapsed_seconds": time.perf_counter() - started})
            if split != "pair":
                continue  # the 5-way tilt is the expensive family; the decision split is pair-level
            for cells_name in TILT_CELLS:
                cell, count = build_cells(cells_name, coloc, mc, arg, bd)
                res = cross_fit_tilt(cell, live["logp"], live["truth"], count, folds[split][CEILING_SEEDS[0]], base_full_bits)
                model_report["tilt"][split][cells_name] = {"cells": count, "by_seed": {str(CEILING_SEEDS[0]): res}}
                progress({"stage": "ceiling", "model": name, "family": "tilt", "split": split, "cells": cells_name, **res, "elapsed_seconds": time.perf_counter() - started})
        # The charter's bare categorical family, pair-level two-fold, all positions of pairs >= 2.
        bare = bare_family(name, folds["pair"][CEILING_SEEDS[0]], live)
        model_report["bare"] = bare
        model_report["elapsed_seconds"] = time.perf_counter() - started
        report["models"][name] = model_report
        atomic_json(STORE_AP / "ceiling" / f"CEILING_{name}.json", model_report)
    report["elapsed_seconds"] = time.perf_counter() - started
    atomic_json(STORE_AP / "ceiling" / f"CEILING_PARTIAL_{'_'.join(args.models)}.json", report)
    progress({"stage": "ceiling", "event": "models_done", "models": list(args.models), "elapsed_seconds": report["elapsed_seconds"]})
    return report


def stage_verdict(args: argparse.Namespace) -> dict[str, Any]:
    """Merge the per-model ceiling receipts and apply the pre-registered refusal rule."""
    rows_result = json.loads(rows_paths()["result"].read_text())
    report: dict[str, Any] = {
        "schema": "ddm_mc1_ceiling.v1",
        "axis": "[model-ledger code length on the exact fs2 coding rows; REFUSAL-ONLY, never a byte claim]",
        "score_claim": False,
        "rows_byte_identical": bool(rows_result.get("byte_identical")),
        "models": {},
    }
    for name in args.models:
        path = STORE_AP / "ceiling" / f"CEILING_{name}.json"
        if not path.is_file():
            raise Mc1Error(f"ceiling receipt missing for {name}: {path}")
        report["models"][name] = json.loads(path.read_text())
    report["verdict"] = ceiling_verdict(report)
    atomic_json(STORE_AP / "ceiling" / "CEILING_RESULT.json", report)
    progress({"stage": "verdict", "event": "done", **report["verdict"]})
    return report


def stage_report(args: argparse.Namespace) -> dict[str, Any]:
    """Render the memo's ceiling table (markdown) from the per-model receipts; no computation."""
    lines = [
        "| plane | Lane IoU | band | ind `mc` | ind `agree` | ind `mc_x_arg` (min 3 seeds) | ind `mc_x_coloc_x_arg` (min 3) | ind `mc_x_arg_x_bd` (min 3) | ind 625-cell | tilt `mc_x_arg` | tilt `mc_x_coloc_x_arg` | tilt 625-cell | best derivable cell | bare Δ (11× baseline) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    rows_out: dict[str, Any] = {}
    for name in args.models:
        path = STORE_AP / "ceiling" / f"CEILING_{name}.json"
        if not path.is_file():
            lines.append(f"| `{name}` | — | — | (receipt missing) |")
            continue
        r = json.loads(path.read_text())
        ind = r["indicator"]["pair"]
        tilt = r["tilt"]["pair"]

        def imin(cell: str) -> float:
            return min(v["held_out_bytes_saved"] for v in ind[cell]["by_seed"].values())

        def tmin(cell: str) -> float:
            return min(v["held_out_bytes_saved"] for v in tilt[cell]["by_seed"].values())

        candidates = {f"ind:{c}": imin(c) for c in ind if c.startswith(("mc", "agree"))}
        candidates.update({f"tilt:{c}": tmin(c) for c in tilt if c.startswith("mc")})
        best_cell = max(candidates, key=candidates.__getitem__)
        motion = r["motion"]
        lane = motion["mean_iou_mc_by_class"]["Lane"]
        band = motion["band_agreement_mc"]
        bare = r["bare"]["ideal_coder_bytes_saved_adding_mc_REFUSAL_ONLY"]
        lines.append(
            f"| `{name}` | {lane:.4f} | {band:.4f} | {imin('mc'):+.2f} | {imin('agree'):+.2f} | {imin('mc_x_arg'):+.2f} | "
            f"{imin('mc_x_coloc_x_arg'):+.2f} | {imin('mc_x_arg_x_bd'):+.2f} | {imin('mc_x_coloc_x_arg_x_bd'):+.2f} | "
            f"{tmin('mc_x_arg'):+.2f} | {tmin('mc_x_coloc_x_arg'):+.2f} | {tmin('mc_x_coloc_x_arg_x_bd'):+.2f} | "
            f"{best_cell} **{candidates[best_cell]:+.2f} B** | {bare:,.0f} |"
        )
        rows_out[name] = {"best_cell": best_cell, "best_held_out_bytes": candidates[best_cell], "lane_iou": lane, "band": band}
    text = "\n".join(lines)
    print(text)
    atomic_json(STORE_AP / "ceiling" / "REPORT_TABLE.json", {"markdown": text, "rows": rows_out})
    return rows_out


def bare_family(name: str, pair_fold_live: np.ndarray, live: dict[str, np.ndarray]) -> dict[str, Any]:
    """Charter (b) literal: categorical codelength of field_t with contexts {coloc} vs {coloc, mc}.

    Over ALL positions of pairs 2..599 (no coder involved), pair-level two-fold, KT alpha=0.5.
    """
    field = load_field()
    plane = np.load(motion_paths(name)["plane"], mmap_mode="r")
    rng = np.random.default_rng(CEILING_SEEDS[0])
    perm = rng.permutation(N)
    pair_fold = np.zeros(N, dtype=np.int8)
    pair_fold[perm[N // 2 :]] = 1
    truth_all, coloc_all, mc_all, fold_all = [], [], [], []
    for t in range(2, N):
        truth_all.append(np.asarray(field[t], dtype=np.uint8).reshape(-1))
        coloc_all.append(np.asarray(field[t - 1], dtype=np.uint8).reshape(-1))
        mc_all.append(np.asarray(plane[t], dtype=np.uint8).reshape(-1))
        fold_all.append(np.full(PLANE, pair_fold[t], dtype=np.int8))
    truth = np.concatenate(truth_all)
    coloc = np.concatenate(coloc_all).astype(np.int64)
    mc = np.concatenate(mc_all).astype(np.int64)
    fold = np.concatenate(fold_all)
    bits_coloc = bare_categorical_bits(coloc, truth, 5, fold)
    bits_both = bare_categorical_bits(coloc * 5 + mc, truth, 25, fold)
    bits_mc = bare_categorical_bits(mc, truth, 5, fold)
    return {
        "positions": int(truth.size),
        "held_out_bytes_ctx_coloc": bits_coloc / 8.0,
        "held_out_bytes_ctx_mc": bits_mc / 8.0,
        "held_out_bytes_ctx_coloc_mc": bits_both / 8.0,
        "screen_bits_saved_adding_mc": bits_coloc - bits_both,
        "ideal_coder_bytes_saved_adding_mc_REFUSAL_ONLY": (bits_coloc - bits_both) / 8.0,
    }


def ceiling_verdict(report: dict[str, Any]) -> dict[str, Any]:
    """Best held-out saving over DECODER-DERIVABLE planes only; ``oracle_*`` planes are
    diagnostic (they read field_t) and are reported beside the verdict, never inside it."""
    best = {"model": None, "family": None, "cells": None, "split": "pair", "held_out_bytes_saved": -math.inf}
    diagnostic: dict[str, float] = {}
    for name, model_report in report["models"].items():
        if name.startswith("oracle_"):
            diagnostic[name] = max(
                min(v["held_out_bytes_saved"] for v in entry["by_seed"].values())
                for family in ("indicator", "tilt")
                for cells_name, entry in model_report[family]["pair"].items()
                if cells_name.startswith(("mc", "agree"))
            )
            continue
        for family in ("indicator", "tilt"):
            for cells_name, entry in model_report[family]["pair"].items():
                if not cells_name.startswith(("mc", "agree")):
                    continue
                value = min(v["held_out_bytes_saved"] for v in entry["by_seed"].values())
                if value > best["held_out_bytes_saved"]:
                    best = {"model": name, "family": family, "cells": cells_name, "split": "pair", "held_out_bytes_saved": value}
    refused = best["held_out_bytes_saved"] < REFUSE_BELOW_BYTES
    return {
        **best,
        "refuse_below_bytes": REFUSE_BELOW_BYTES,
        "typed_verdict": "CEILING-REFUSED" if refused else "CEILING-PASSED",
        "diagnostic_oracle_planes_best_held_out_bytes": diagnostic,
        "note": "held-out bytes saved on the coder's own rows, pair-level two-fold, min over seeds; REFUSAL-ONLY; oracle planes excluded from the verdict",
    }


# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=("rows", "motion", "oracle", "ceiling", "verdict", "report"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MOTION_MODELS),
        choices=MOTION_MODELS + tuple(f"oracle_{name}" for name in MOTION_MODELS),
        help="motion families; ``oracle_<name>`` is admitted ONLY for the ceiling/verdict stages (diagnostic plane)",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stage = {
        "rows": stage_rows,
        "motion": stage_motion,
        "oracle": stage_oracle,
        "ceiling": stage_ceiling,
        "verdict": stage_verdict,
        "report": stage_report,
    }[args.stage]
    stage(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
