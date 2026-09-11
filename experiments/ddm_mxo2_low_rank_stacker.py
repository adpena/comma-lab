#!/usr/bin/env python3
"""MXO2 exact n600 screen over move-44's 23 pre-mix family predictions.

This is a scorer-free research instrument, not a candidate builder.  It copies
and instruments the immutable move-44 receiver, retains the Q15 surface actually
consumed by the integer stacker, and tests cold online rank-{2,4,8} quadratic
models.  Updates happen only after a complete shipped decode group.
"""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "src")]
for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_key] = "1"

import numpy as np

from experiments import ddm_jg2_tail_reencode as jg2
from experiments.ddm_ls1_shipped_surprise import ARCHIVE_SHA, FIELD, FIELD_SHA, fact
from experiments.ddm_tc1_mixer_codec import frequencies

ROOT_TIER = Path("/Volumes/VertigoDataTier/pact")
ROOT_PREFIX = "ddm_mxo2_low_rank_stacker"
ROOT = ROOT_TIER / f"{ROOT_PREFIX}_v3"
SOURCE = Path("/Volumes/VertigoDataTier/pact/ddm_rlc5_cure_on_move43/candidate_runtime")
LS1 = ROOT.parent / "ddm_ls1"
N, H, W, K, TOTAL = 600, 384, 512, 5, 1 << 31
SIZE = H * W
RANKS = (2, 4, 8)
SEED = 20260911
MIN_RESERVE = 16 << 30
SURFACE_NEED = 10 << 30
TAIL_DEMAND_B = 25_899
FIRE_SAVING_B = 3_000
FIRE_NS_PER_SYMBOL = 200.0
MOVE44_LOCAL_TAIL_S = 1168.6204084999987
MOVE44_T4_TAIL_S = 1142.9962956905365
T4_OVER_LOCAL = MOVE44_T4_TAIL_S / MOVE44_LOCAL_TAIL_S
STRICT_T4_SLACK_S = 27.581274745
AXIS = "[macOS-CPU advisory / scorer-free n600 exact integer-row screen]"

_Y, _X = np.indices((H, W))
GROUP = (_X % 64 + 2 * (_Y % 64)).reshape(-1)
GROUP_POSITIONS = tuple(np.flatnonzero(group == GROUP) for group in range(190))
GROUP_ORDER = np.concatenate(GROUP_POSITIONS).astype(np.uint32)
GROUP_SLICES = tuple(
    slice(int(sum(map(len, GROUP_POSITIONS[:group]))), int(sum(map(len, GROUP_POSITIONS[: group + 1]))))
    for group in range(190)
)
FAMILY_NAMES = (
    "shipped_joint", "temporal_spatial", "surprise_only", "spatial_surprise",
    "spatial_boundary", "run_surprise", "boundary_surprise", "temporal_surprise",
    "shipped_fast256", "shipped_fast4096", "surprise_fast256", "spatial4_surprise",
    "homog_surprise", "homog_boundary_surprise", "spatial4_boundary", "homog_spatial4",
    "spatial4_temporal", "homog_surprise_fast256", "spatial4_surprise_fast256",
    "groupbin8_surprise", "cls_groupbin8", "patch192_only", "tile48_groupbin8",
)


class Mxo2Error(RuntimeError):
    """A custody, exactness, storage, or screen invariant failed."""


SURFACE_FUNCTION = r"""
/* MXO2 research-only observability.  This is injected into a COPY, never the
 * immutable move-44 source.  Q15 is the declared integer boundary consumed by
 * the screen, so these are the exact predictor values seen by the stacker. */
int f26_corrector_mxo2_surface(void *handle, uint16_t *family_q15,
                               uint16_t *mixer_q15, uint8_t *hit_class,
                               int64_t n)
{
    Corrector *self = (Corrector *)handle;
    if (!self || !self->group_open || n != self->n || !family_q15 ||
        !mixer_q15 || !hit_class) {
        return -1;
    }
    for (int64_t i = 0; i < n; ++i) {
        double p_max = self->p_max[i];
        double one_minus = self->one_minus[i];
        for (int pos = 0; pos < N_FAMILIES; ++pos) {
            Family *family = &self->families[pos];
            int64_t index = self->fam_index[(int64_t)pos * self->capacity + i];
            double multiplier = family_multiplier(family, index);
            double shifted = p_max * multiplier;
            double q = clamp_double(shifted / (shifted + one_minus),
                                    PROB_EPS, 1.0 - PROB_EPS);
            family_q15[i * N_FAMILIES + pos] = (uint16_t)rint(q * 32768.0);
        }
        mixer_q15[i] = (uint16_t)rint(self->q[i] * 32768.0);
        hit_class[i] = (uint8_t)self->arg[i];
    }
    return 0;
}

"""
SURFACE_MARKER = "int f26_corrector_coding_row(void *handle, float *output, int64_t n)\n"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def guard(need: int = 0) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(ROOT).free < MIN_RESERVE + need:
        raise Mxo2Error("STORAGE_BLOCK: retain all bytes; no cleanup or fallback authorized")


def atomic_bytes(path: Path, payload: bytes, *, immutable: bool = False) -> None:
    guard(len(payload))
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise Mxo2Error(f"write escaped owned SSD root: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_bytes() != payload:
            raise Mxo2Error(f"immutable payload changed: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_json(path: Path, value: object, *, immutable: bool = False) -> None:
    atomic_bytes(
        path,
        (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(),
        immutable=immutable,
    )


def adopt_root(candidate: Path) -> Path:
    """Bind the owned durable root named on the command line.

    Every source change mints a new immutable release, so a screen must be able
    to name a fresh sibling root without editing a constant.  The prefix guard
    keeps the choice inside this family's own SSD tier and out of any other
    arm's directory.
    """
    global ROOT
    resolved = candidate.resolve()
    if resolved.parent != ROOT_TIER or not resolved.name.startswith(ROOT_PREFIX):
        raise Mxo2Error(
            f"--resume-from must name a {ROOT_TIER}/{ROOT_PREFIX}* root owned by this screen"
        )
    ROOT = resolved
    return ROOT


def json_normalized(value: object) -> object:
    """Round-trip through JSON so a re-run compares like with like.

    Without this, a tuple in the freshly computed pins never equals the list it
    serialized to, so every resume after the first raised binding drift on
    unchanged inputs.  Serialization is unaffected: a tuple and its list image
    emit identical bytes.
    """
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def atomic_npz(
    path: Path, values: dict[str, np.ndarray], *, immutable: bool = False, compressed: bool = False
) -> None:
    guard(sum(value.nbytes for value in values.values()))
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise Mxo2Error(f"write escaped owned SSD root: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        with np.load(path, allow_pickle=False) as prior:
            if set(prior.files) != set(values) or any(
                not np.array_equal(prior[name], value) for name, value in values.items()
            ):
                raise Mxo2Error(f"immutable array payload changed: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    with temporary.open("wb") as handle:
        writer = np.savez_compressed if compressed else np.savez
        writer(handle, **values)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def checked_npz(path: Path):
    receipt = json.loads(path.with_suffix(".json").read_text())
    if fact(path) != receipt:
        raise Mxo2Error(f"retained payload hash changed: {path}")
    return np.load(path, allow_pickle=False)


def retain_npz(path: Path, values: dict[str, np.ndarray], *, compressed: bool = False) -> None:
    atomic_npz(path, values, immutable=True, compressed=compressed)
    save_json(path.with_suffix(".json"), fact(path), immutable=True)


def source_release() -> dict[str, object]:
    release = ROOT / "source_release"
    files = (
        REPO / "experiments/ddm_mxo2_low_rank_stacker.py",
        REPO / "experiments/ddm_mxo2_stacker.c",
    )
    result = {}
    for source in files:
        destination = release / source.name
        atomic_bytes(destination, source.read_bytes(), immutable=True)
        result[source.name] = fact(destination)
    return result


def instrument_source(payload: bytes) -> bytes:
    text = payload.decode()
    if text.count(SURFACE_MARKER) != 1 or "f26_corrector_mxo2_surface" in text:
        raise Mxo2Error("unknown or already instrumented native corrector source")
    return text.replace(SURFACE_MARKER, SURFACE_FUNCTION + SURFACE_MARKER).encode()


def prepare() -> dict[str, object]:
    guard(SURFACE_NEED)
    pointer = json.loads((REPO / ".omx/state/canonical_frontier_pointer.json").read_text())
    if pointer["effective_frontier"]["archive_sha256"] != ARCHIVE_SHA:
        raise Mxo2Error("POINTER_CHANGED: explicit rebind required")
    if fact(FIELD)["sha256"] != FIELD_SHA or fact(FIELD)["bytes"] != N * SIZE:
        raise Mxo2Error("move-44 field custody changed")
    if fact(SOURCE / "archive.zip")["sha256"] != ARCHIVE_SHA:
        raise Mxo2Error("move-44 source archive changed")
    if not np.array_equal(np.sort(GROUP_ORDER), np.arange(SIZE, dtype=np.uint32)):
        raise Mxo2Error("190-group plan is not a permutation")

    releases = source_release()
    copied: dict[str, object] = {}
    for source in sorted(SOURCE.rglob("*")):
        if not source.is_file() or source.suffix not in (
            ".py", ".c", ".sh", ".zip", ".json", ".txt", ".md"
        ):
            continue
        relative = source.relative_to(SOURCE)
        destination = ROOT / "runtime_copy" / relative
        payload = source.read_bytes()
        if relative == Path("runtime/f26_corrector_native.c"):
            payload = instrument_source(payload)
        atomic_bytes(destination, payload, immutable=True)
        copied[str(relative)] = {
            "source": fact(source),
            "copy": fact(destination),
            "instrumented": relative == Path("runtime/f26_corrector_native.c"),
        }
    atomic_npz(
        ROOT / "surface/group_order.npz",
        {"positions": GROUP_ORDER, "lengths": np.array(list(map(len, GROUP_POSITIONS)))},
        immutable=True,
    )
    save_json(ROOT / "surface/group_order.json", fact(ROOT / "surface/group_order.npz"), immutable=True)
    pins = {
        "axis": AXIS,
        "score_claim": False,
        "archive": fact(SOURCE / "archive.zip"),
        "field": fact(FIELD),
        "ls1_trace": fact(LS1 / "TRACE.json"),
        "ls1_atlas": fact(LS1 / "atlas/RESULT.json"),
        "runtime_sources": copied,
        "source_release": releases,
        "family_names": FAMILY_NAMES,
        "seed": SEED,
        "surface_integer_boundary": "round-to-nearest Q15 hit probability",
        "symbols": N * SIZE,
        "group_order": fact(ROOT / "surface/group_order.npz"),
        "reserve_bytes": MIN_RESERVE,
    }
    binding = ROOT / "BINDING.json"
    pins = json_normalized(pins)
    if binding.exists() and json.loads(binding.read_text()) != pins:
        raise Mxo2Error("source/input binding drift")
    save_json(binding, pins, immutable=True)
    return pins


def bind_surface_api(library: ctypes.CDLL) -> None:
    library.f26_corrector_mxo2_surface.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int64,
    ]
    library.f26_corrector_mxo2_surface.restype = ctypes.c_int


def build_geometry(runtime_root: Path) -> dict[str, object]:
    source = runtime_root / "runtime/rlc1_geometry.c"
    library = ROOT / "native_receiver/geometry.dylib"
    command = [
        os.environ.get("CC", "cc"), "-O3", "-std=c11", "-shared", "-fPIC",
        "-ffp-contract=off", "-fno-fast-math", str(source), "-o", str(library),
    ]
    receipt_path = ROOT / "native_receiver/geometry.json"
    expected_source = fact(source)
    if library.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt != {"argv": command, "source": expected_source, "library": fact(library)}:
            raise Mxo2Error("native geometry build drift")
    else:
        guard(1 << 20)
        library.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        atomic_bytes(library.parent / "geometry.build.log", (result.stdout + result.stderr).encode())
        result.check_returncode()
        receipt = {"argv": command, "source": expected_source, "library": fact(library)}
        save_json(receipt_path, receipt, immutable=True)
    os.environ["RLC1_GEOMETRY_LIBRARY"] = str(library)
    return receipt


def surface_resume_frame() -> int:
    latest = ROOT / "receiver_checkpoints/LATEST.json"
    return 0 if not latest.exists() else int(json.loads(latest.read_text())["frame"])


def extract_surface() -> dict[str, object]:
    prepare()
    completed = ROOT / "SURFACE.json"
    if completed.exists():
        result = json.loads(completed.read_text())
        if (
            result.get("full_n600")
            and result.get("symbols") == N * SIZE
            and fact(Path(result["decoded_tokens"]["path"])) == result["decoded_tokens"]
            and len(result.get("surface_files", ())) == N
            and all(fact(Path(item["path"])) == item for item in result["surface_files"])
        ):
            return result
        raise Mxo2Error("existing surface completion receipt is stale or incomplete")
    import torch

    from experiments.ddm_tc1_public_proof import build_libraries

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    runtime_root = ROOT / "runtime_copy"
    builds = build_libraries(runtime_root, ROOT / "native_receiver")
    builds.append(build_geometry(runtime_root))
    os.environ["TC1_RECEIVER_CHECKPOINT_DIR"] = str(ROOT / "receiver_checkpoints")
    os.environ["TC1_RECEIVER_STOP_AFTER"] = str(N)
    residual, renderer, code_dir = jg2.load_runtime(runtime_root)
    from runtime.entropy.rc64 import NativeDecoder
    from runtime.native_free_corrector import EXPECTED_SHIPPED_CONFIG, NativeFreeCorrector

    if tuple(EXPECTED_SHIPPED_CONFIG["families"]) != FAMILY_NAMES:
        raise Mxo2Error("declared 23-family surface order differs from shipped native receiver")

    bind_surface_api(ctypes.CDLL(os.environ["F26_CORRECTOR_NATIVE_LIBRARY"]))
    target = np.memmap(FIELD, dtype=np.uint8, mode="r", shape=(N, H, W))
    observer: dict[str, object] = {
        "frame": surface_resume_frame(),
        "offset": 0,
        "family": np.empty((SIZE, len(FAMILY_NAMES)), dtype=np.uint16),
        "mixer": np.empty(SIZE, dtype=np.uint16),
        "hit_class": np.empty(SIZE, dtype=np.uint8),
        "frequencies": np.empty((SIZE, K), dtype=np.uint32),
        "symbols": np.empty(SIZE, dtype=np.uint8),
        "open": None,
    }
    original_coding = NativeFreeCorrector.coding_row
    original_decode = NativeDecoder.decode
    original_end = NativeFreeCorrector.end_frame

    def coding(self, state):
        result = original_coding(self, state)
        if observer["open"] is not None:
            raise Mxo2Error("surface observer saw overlapping groups")
        n = state.n
        start = int(observer["offset"])
        stop = start + n
        family = np.empty((n, len(FAMILY_NAMES)), dtype=np.uint16)
        mixed = np.empty(n, dtype=np.uint16)
        hit_class = np.empty(n, dtype=np.uint8)
        bind_surface_api(self.library)
        status = self.library.f26_corrector_mxo2_surface(
            self.handle,
            family.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            mixed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            hit_class.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int64(n),
        )
        if status:
            raise Mxo2Error(f"native surface extraction failed with status {status}")
        observer["family"][start:stop] = family
        observer["mixer"][start:stop] = mixed
        observer["hit_class"][start:stop] = hit_class
        observer["open"] = slice(start, stop)
        observer["offset"] = stop
        return result

    def decode(self, probability):
        symbols = original_decode(self, probability)
        group_slice = observer["open"]
        if group_slice is None or group_slice.stop - group_slice.start != len(symbols):
            raise Mxo2Error("surface decoder observation lost group alignment")
        observer["frequencies"][group_slice] = frequencies(probability).astype(np.uint32)
        observer["symbols"][group_slice] = symbols.astype(np.uint8)
        observer["open"] = None
        return symbols

    def end(self, tokens_flat):
        frame = int(observer["frame"])
        if observer["open"] is not None or int(observer["offset"]) != SIZE:
            raise Mxo2Error("surface frame ended before all 190 groups were observed")
        original_end(self, tokens_flat)
        np.testing.assert_array_equal(tokens_flat, target[frame].reshape(-1))
        np.testing.assert_array_equal(observer["symbols"], target[frame].reshape(-1)[GROUP_ORDER])
        with checked_npz(LS1 / "rows" / f"frame_{frame:04d}.npz") as prior:
            np.testing.assert_array_equal(observer["frequencies"], prior["frequencies"][GROUP_ORDER])
            np.testing.assert_array_equal(observer["symbols"], prior["symbols"][GROUP_ORDER])
        path = ROOT / "surface/frames" / f"frame_{frame:04d}.npz"
        retain_npz(
            path,
            {
                "family_q15": observer["family"].copy(),
                "mixer_q15": observer["mixer"].copy(),
                "hit_class": observer["hit_class"].copy(),
                "frequencies": observer["frequencies"].copy(),
                "symbols": observer["symbols"].copy(),
            },
        )
        observer["frame"] = frame + 1
        observer["offset"] = 0
        if (frame + 1) % 25 == 0:
            print(json.dumps({"stage": "surface", "frames": frame + 1}), flush=True)

    NativeFreeCorrector.coding_row = coding
    NativeDecoder.decode = decode
    NativeFreeCorrector.end_frame = end
    started = time.perf_counter()
    try:
        parts = residual.read_residual_archive(runtime_root / "archive.zip")
        tokens, report = residual.decode_production_tokens(
            parts, renderer, code_dir, torch.device("cpu")
        )
    finally:
        NativeFreeCorrector.coding_row = original_coding
        NativeDecoder.decode = original_decode
        NativeFreeCorrector.end_frame = original_end
    tokens_path = ROOT / "surface/decoded_tokens.u8"
    atomic_bytes(tokens_path, tokens.numpy().tobytes(), immutable=True)
    if fact(tokens_path)["sha256"] != FIELD_SHA:
        raise Mxo2Error("instrumented receiver decoded field differs")

    bits = 0.0
    surfaces = []
    for frame in range(N):
        path = ROOT / "surface/frames" / f"frame_{frame:04d}.npz"
        with checked_npz(path) as data:
            selected = data["frequencies"][np.arange(SIZE), data["symbols"]]
            bits += float((-np.log2(selected.astype(np.float64) / TOTAL)).sum())
        surfaces.append(fact(path))
    stream_bytes = len(parts.token_stream)
    difference = stream_bytes - bits / 8
    relative_percent = abs(difference) / stream_bytes * 100
    if relative_percent > 0.0006:
        raise Mxo2Error("INSTRUMENT_FALSIFIED: surface exceeds LS1 reconciliation standard")
    result = {
        "axis": AXIS,
        "score_claim": False,
        "full_n600": True,
        "symbols": N * SIZE,
        "family_predictions": len(FAMILY_NAMES),
        "integer_boundary": "Q15",
        "group_order": "190 shipped groups, then in-group flat-position order",
        "ideal_bits": bits,
        "ideal_bytes": bits / 8,
        "physical_stream_bytes": stream_bytes,
        "framing_bytes": difference,
        "relative_percent": relative_percent,
        "ls1_standard_percent": 0.0006,
        "field_identity": True,
        "decode_report": report,
        "elapsed_seconds": time.perf_counter() - started,
        "surface_files": surfaces,
        "decoded_tokens": fact(tokens_path),
        "binding": fact(ROOT / "BINDING.json"),
        "builds": builds,
    }
    save_json(ROOT / "SURFACE.json", result, immutable=True)
    return result


class NativeStacker:
    """Thin owner for the exact C predict+update kernel."""

    def __init__(self, library: Path, rank: int, weights: np.ndarray | None = None) -> None:
        self.rank = rank
        self.library = ctypes.CDLL(str(library))
        self.library.mxo2_create.argtypes = [ctypes.c_int, ctypes.c_uint64]
        self.library.mxo2_create.restype = ctypes.c_void_p
        self.library.mxo2_destroy.argtypes = [ctypes.c_void_p]
        self.library.mxo2_process_group.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64, ctypes.POINTER(ctypes.c_uint32),
        ]
        self.library.mxo2_process_group.restype = ctypes.c_int
        self.library.mxo2_get_weights.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int
        ]
        self.library.mxo2_set_weights.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int
        ]
        self.handle = self.library.mxo2_create(rank, SEED)
        if not self.handle:
            raise Mxo2Error(f"native stacker refused rank {rank}")
        if weights is not None:
            values = np.ascontiguousarray(weights, dtype=np.int32)
            status = self.library.mxo2_set_weights(
                self.handle,
                values.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                values.size,
            )
            if status:
                raise Mxo2Error(f"native stacker restore failed with status {status}")

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.library.mxo2_destroy(self.handle)
            self.handle = None

    def weights(self) -> np.ndarray:
        result = np.empty((K, self.rank), dtype=np.int32)
        status = self.library.mxo2_get_weights(
            self.handle,
            result.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            result.size,
        )
        if status:
            raise Mxo2Error(f"native stacker snapshot failed with status {status}")
        return result

    def process(
        self,
        family: np.ndarray,
        mixed: np.ndarray,
        freq: np.ndarray,
        hit_class: np.ndarray,
        symbols: np.ndarray,
    ) -> np.ndarray:
        family = np.ascontiguousarray(family, dtype=np.uint16)
        mixed = np.ascontiguousarray(mixed, dtype=np.uint16)
        freq = np.ascontiguousarray(freq, dtype=np.uint32)
        hit_class = np.ascontiguousarray(hit_class, dtype=np.uint8)
        symbols = np.ascontiguousarray(symbols, dtype=np.uint8)
        output = np.empty_like(freq)
        status = self.library.mxo2_process_group(
            self.handle,
            family.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            mixed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            freq.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            hit_class.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            symbols.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            len(symbols),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        )
        if status:
            raise Mxo2Error(f"native stacker process failed with status {status}")
        return output

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def build_stacker() -> Path:
    source = ROOT / "source_release/ddm_mxo2_stacker.c"
    library = ROOT / "native_stacker/libmxo2_stacker.dylib"
    command = [
        os.environ.get("CC", "cc"), "-O3", "-std=c11", "-shared", "-fPIC",
        "-ffp-contract=off", "-fno-fast-math", str(source), "-o", str(library),
    ]
    receipt_path = ROOT / "native_stacker/BUILD.json"
    if library.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt != {"argv": command, "source": fact(source), "library": fact(library)}:
            raise Mxo2Error("native stacker build drift")
    else:
        guard(1 << 20)
        library.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        atomic_bytes(library.parent / "build.log", (result.stdout + result.stderr).encode())
        result.check_returncode()
        save_json(
            receipt_path,
            {"argv": command, "source": fact(source), "library": fact(library)},
            immutable=True,
        )
    return library


def latest_screen_state() -> tuple[int, dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray]:
    paths = sorted((ROOT / "screen/checkpoints").glob("state_*.npz"))
    paths = [path for path in paths if path.with_suffix(".json").exists()]
    if not paths:
        return 0, {rank: np.zeros((K, rank), np.int32) for rank in RANKS}, {
            rank: np.zeros((K, 7), np.float64) for rank in RANKS
        }, np.zeros(len(RANKS), np.float64)
    with checked_npz(paths[-1]) as data:
        frame = int(data["frame"][0])
        weights = {rank: data[f"weights_{rank}"].copy() for rank in RANKS}
        tables = {rank: data[f"class_geometry_{rank}"].copy() for rank in RANKS}
        seconds = data["kernel_seconds"].copy()
    return frame, weights, tables, seconds


def screen() -> dict[str, object]:
    prepare()
    surface_result = json.loads((ROOT / "SURFACE.json").read_text())
    if not surface_result["full_n600"] or surface_result["symbols"] != N * SIZE:
        raise Mxo2Error("complete n600 surface is required")
    library = build_stacker()
    start, weights, tables, kernel_seconds = latest_screen_state()
    models = {
        rank: NativeStacker(library, rank, weights[rank])
        for rank in RANKS
    }
    try:
        for frame in range(start, N):
            with checked_npz(ROOT / "surface/frames" / f"frame_{frame:04d}.npz") as data:
                family = data["family_q15"].copy()
                mixed = data["mixer_q15"].copy()
                base = data["frequencies"].copy()
                hit_class = data["hit_class"].copy()
                symbols = data["symbols"].copy()
            with checked_npz(LS1 / "atlas/cells" / f"frame_{frame:04d}.npz") as atlas:
                geometry = atlas["geometry"][GROUP_ORDER].astype(np.int64)
            geometry[geometry == 255] = 6
            if np.any((geometry < 0) | (geometry > 6)):
                raise Mxo2Error("LS1 geometry escaped seven accounting cells")
            baseline_selected = base[np.arange(SIZE), symbols]
            for rank_index, rank in enumerate(RANKS):
                pieces = []
                tick = time.perf_counter_ns()
                for group_slice in GROUP_SLICES:
                    pieces.append(models[rank].process(
                        family[group_slice], mixed[group_slice], base[group_slice],
                        hit_class[group_slice], symbols[group_slice]
                    ))
                kernel_seconds[rank_index] += (time.perf_counter_ns() - tick) / 1e9
                candidate = np.concatenate(pieces)
                if np.any(candidate.sum(axis=1, dtype=np.uint64) != TOTAL):
                    raise Mxo2Error(f"rank {rank} emitted invalid integer frequency mass")
                selected = candidate[np.arange(SIZE), symbols]
                gain = np.log2(selected.astype(np.float64) / baseline_selected)
                tables[rank] += np.bincount(
                    symbols.astype(np.int64) * 7 + geometry,
                    weights=gain / 8,
                    minlength=K * 7,
                ).reshape(K, 7)
            boundary = frame + 1
            values = {
                "frame": np.array([boundary], dtype=np.int64),
                "kernel_seconds": kernel_seconds,
            }
            for rank in RANKS:
                values[f"weights_{rank}"] = models[rank].weights()
                values[f"class_geometry_{rank}"] = tables[rank]
            retain_npz(
                ROOT / "screen/checkpoints" / f"state_{boundary:04d}.npz",
                values,
                compressed=True,
            )
            if boundary % 25 == 0:
                print(json.dumps({
                    "stage": "screen", "frames": boundary,
                    "saving_bytes": {str(rank): float(tables[rank].sum()) for rank in RANKS},
                }, sort_keys=True), flush=True)
    finally:
        for model in models.values():
            model.close()

    result_models = {}
    for rank_index, rank in enumerate(RANKS):
        saving = float(tables[rank].sum())
        ns_per_symbol = float(kernel_seconds[rank_index] * 1e9 / (N * SIZE))
        projected_t4_ns = ns_per_symbol * T4_OVER_LOCAL
        projected_t4_seconds = projected_t4_ns * N * SIZE / 1e9
        result_models[str(rank)] = {
            "integer_row_saving_bytes": saving,
            "shortfall_vs_3000_bytes": FIRE_SAVING_B - saving,
            "shortfall_vs_25899_bytes": TAIL_DEMAND_B - saving,
            "class_geometry_saving_bytes": tables[rank].tolist(),
            "kernel_seconds": float(kernel_seconds[rank_index]),
            "host_ns_per_symbol": ns_per_symbol,
            "projected_t4_ns_per_symbol": projected_t4_ns,
            "projected_t4_incremental_seconds": projected_t4_seconds,
            "saving_trigger_pass": saving >= FIRE_SAVING_B,
            "timing_trigger_pass": projected_t4_ns <= FIRE_NS_PER_SYMBOL,
            "slack_trigger_pass": projected_t4_seconds <= STRICT_T4_SLACK_S,
            "final_weights": fact(ROOT / "screen/checkpoints/state_0600.npz"),
        }
    any_joint_pass = any(
        row["saving_trigger_pass"] and row["timing_trigger_pass"] and row["slack_trigger_pass"]
        for row in result_models.values()
    )
    result = {
        "axis": AXIS,
        "score_claim": False,
        "full_n600": True,
        "symbols": N * SIZE,
        "models": result_models,
        "surface": fact(ROOT / "SURFACE.json"),
        "native_build": fact(ROOT / "native_stacker/BUILD.json"),
        "t4_projection_factor": T4_OVER_LOCAL,
        "fire_saving_bytes": FIRE_SAVING_B,
        "tail_demand_bytes": TAIL_DEMAND_B,
        "fire_ns_per_symbol": FIRE_NS_PER_SYMBOL,
        "strict_t4_slack_seconds": STRICT_T4_SLACK_S,
        "verdict": "PASS_RECEIVER_INPUTS_REQUIRED" if any_joint_pass else "CLOSE_FORMULATION",
        "union_not_sum": "each rank is one joint online run over all family outputs",
    }
    save_json(ROOT / "SCREEN.json", result, immutable=True)
    return result


def benchmark() -> dict[str, object]:
    prepare()
    screen_result = json.loads((ROOT / "SCREEN.json").read_text())
    result = {
        "axis": AXIS,
        "score_claim": False,
        "scope": "tight native process_group calls on the real n600 surface, including group-causal updates",
        "host": {rank: screen_result["models"][rank]["host_ns_per_symbol"] for rank in map(str, RANKS)},
        "projected_t4": {
            rank: screen_result["models"][rank]["projected_t4_ns_per_symbol"]
            for rank in map(str, RANKS)
        },
        "factor": T4_OVER_LOCAL,
        "threshold_ns_per_symbol": FIRE_NS_PER_SYMBOL,
        "slack_seconds": STRICT_T4_SLACK_S,
        "screen": fact(ROOT / "SCREEN.json"),
    }
    save_json(ROOT / "BENCHMARK.json", result, immutable=True)
    return result


def retention() -> None:
    payloads = [
        fact(path) for path in sorted(ROOT.rglob("*"))
        if path.is_file() and path.suffix in (".npz", ".u8", ".bin", ".dylib", ".so", ".c", ".py")
    ]
    save_json(ROOT / "RETENTION.json", {
        "policy": "KEEP all surface frames, complete model states, receiver checkpoints, sources, and builds",
        "cleanup": "none authorized",
        "files": payloads,
        "bytes": sum(item["bytes"] for item in payloads),
        "free_bytes": shutil.disk_usage(ROOT).free,
        "reserve_bytes": MIN_RESERVE,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("surface", "screen", "benchmark"), required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    args = parser.parse_args()
    adopt_root(args.resume_from)
    guard()
    lock = (ROOT / ".stage.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    try:
        stages = {"surface": extract_surface, "screen": screen, "benchmark": benchmark}
        result = stages[args.stage]()
        save_json(ROOT / f"STAGE_{args.stage}.json", {
            "argv": sys.argv,
            "result": result,
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
        })
        retention()
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
