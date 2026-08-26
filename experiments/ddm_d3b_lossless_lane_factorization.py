#!/usr/bin/env python3
"""Losslessly factor the GB1 five-class token field into quotient + Lane.

The four-class quotient stream is the receiver-closed D3 artifact.  This
instrument codes the remaining Road-versus-Lane bit only at quotient-Road
sites.  The shipped HPAC/F26 model supplies the base conditional probability;
adaptive variants condition it on the already-decoded whole quotient field.

Every candidate is a real RC64 payload, is repeated deterministically, is
packed into a complete factorized archive, and is independently decoded from
that archive before any byte verdict is admitted.  No scorer is needed because
the reconstructed five-class field must match the source byte-for-byte.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

d3 = importlib.import_module("experiments.ddm_d3_alphabet_merge")
d3a = importlib.import_module("experiments.ddm_d3a_analytic_lane_carrier")
jg2 = importlib.import_module("experiments.ddm_jg2_tail_reencode")


DEFAULT_STORE = Path("/Volumes/APDataStore/pact/ddm_d3b_lossless_lane_factorization")
D3_STORE = Path("/Volumes/APDataStore/pact/ddm_d3_alphabet_merge")
SOURCE_FIELD = Path(
    "/Volumes/APDataStore/pact/ddm_tb2_token_bit_attribution/measurement_v1/"
    "retained/fields/decoded_tokens_instrumented.u8"
)
SOURCE_FIELD_SHA256 = "cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb"
MERGED_FIELD = D3_STORE / "retained/fields/tokens_lane_to_road_canonical.u8"
MERGED_FIELD_SHA256 = "deafcb2f77e0f2ab0895b4cef8e789189aeddb2d24902a84dd2d1f44ee81cb07"
FOUR_STREAM = D3_STORE / "retained/encode/token_stream_alphabet4_n600.bin"
FOUR_STREAM_SHA256 = "84fa2f499fb6c052cf6a43f8cae98c227ac32412ce1495cc715aa5af94b8692d"
D3_RATE_ARCHIVE = D3_STORE / "retained/candidates/candidate_d3_rate_only.zip"
D3_RATE_ARCHIVE_SHA256 = "34d4e598136e43d0c162c6ab21ebee65e34c8a72294d8e80b0338d2e3f24e2c6"

N, H, W = d3.N, d3.H, d3.W
PLANE = H * W
FIELD_BYTES = N * PLANE
ROAD = d3.ROAD
LANE = d3.LANE
AXIS = "[macOS-CPU advisory / scorer-free exact rate and receiver measurement]"
MODEL_BYTES = 13_515
DX2_TOKEN_SUBSYSTEM_BYTES = 127_292
DEMAND_CLOSING_BYTES = 85_064
OTHER_ARCHIVE_BYTES = 53_076
MINIMUM_FREE_BYTES = 2 << 30

FACTOR_HEADER = struct.Struct("<4sII")
FACTOR_MAGIC = b"D3BF"
LANE_HEADER = struct.Struct("<4sBBBBHHQQQ32s")
LANE_MAGIC = b"D3B1"
LANE_HEADER_BYTES = LANE_HEADER.size
REFERENCE_PREFIX = struct.Struct("<4sI")
REFERENCE_MAGIC = b"D3BR"
D3A_BUILD_RESULT = Path(
    "/Volumes/APDataStore/pact/ddm_d3a_analytic_lane_carrier/BUILD_RESULT.json"
)
ENCODE_CHECKPOINT_SCHEMA = "ddm_d3b_encode_checkpoint.v1"
QUOTIENT_CHECKPOINT_SCHEMA = "ddm_d3b_quotient_decode_checkpoint.v1"
LANE_CHECKPOINT_SCHEMA = "ddm_d3b_lane_decode_checkpoint.v1"


class D3BError(RuntimeError):
    """The retained factorization or its receiver violated a hard invariant."""


@dataclass(frozen=True)
class ContextConfig:
    name: str
    design_id: int
    design: str
    radius: int
    context_bits: int
    geom_bins: int
    prior_strength: int
    adaptive: bool = True
    mixer: bool = False
    d3a_rung: str | None = None


CONFIGS = (
    ContextConfig("hpac_conditional", 0, "hpac_conditional", 0, 0, 0, 0, False),
    ContextConfig("field_r1", 1, "field_only", 1, 18, 0, 64),
    ContextConfig("field_r2", 2, "field_only", 2, 18, 0, 64),
    ContextConfig("field_geometry_r1", 3, "field_geometry", 1, 18, 8, 64),
    ContextConfig("field_geometry_r2", 4, "field_geometry", 2, 18, 8, 64),
    ContextConfig(
        "field_geometry_temporal_r1", 5, "field_geometry_temporal", 1, 18, 8, 64
    ),
    ContextConfig(
        "field_geometry_temporal_r2", 6, "field_geometry_temporal", 2, 18, 8, 64
    ),
    ContextConfig(
        "reference_d3a_q8_mixer_r2",
        7,
        "reference_d3a_geometry_temporal_mixer",
        2,
        18,
        8,
        16,
        True,
        True,
        "q8",
    ),
    ContextConfig(
        "reference_d3a_q1_mixer_r2",
        8,
        "reference_d3a_geometry_temporal_mixer",
        2,
        18,
        8,
        16,
        True,
        True,
        "q1",
    ),
)
CONFIG_BY_ID = {config.design_id: config for config in CONFIGS}
CONFIG_BY_NAME = {config.name: config for config in CONFIGS}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))


def progress(**record: Any) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


def require_file(path: Path, size: int, sha256: str, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size != size or sha256_file(path) != sha256:
        raise D3BError(f"{label} custody pin failed: {path}")
    return file_fact(path)


def verify_inputs(store: Path) -> dict[str, Any]:
    store.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(store).free
    if free < MINIMUM_FREE_BYTES:
        raise D3BError(f"storage preflight failed: {free} B free < {MINIMUM_FREE_BYTES} B")
    return {
        "storage": {
            "path": str(store),
            "observed_free_bytes": free,
            "minimum_free_bytes": MINIMUM_FREE_BYTES,
            "status": "PASS",
        },
        "source_field": require_file(
            SOURCE_FIELD, FIELD_BYTES, SOURCE_FIELD_SHA256, "source five-class field"
        ),
        "merged_field": require_file(
            MERGED_FIELD, FIELD_BYTES, MERGED_FIELD_SHA256, "D3 quotient field"
        ),
        "four_stream": require_file(
            FOUR_STREAM, 49_696, FOUR_STREAM_SHA256, "D3 four-class stream"
        ),
        "d3_rate_archive": require_file(
            D3_RATE_ARCHIVE, 116_287, D3_RATE_ARCHIVE_SHA256, "D3 rate-only archive"
        ),
    }


def completed_result_if_current(path: Path, *, require_identity: bool) -> dict[str, Any] | None:
    """Return a fully verified completed stage so --resume is idempotent after success."""
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    expected = {config.name for config in CONFIGS}
    rows = result.get("rows", [])
    observed = {row.get("config", {}).get("name") for row in rows}
    if not result.get("complete") or observed != expected:
        return None
    for row in rows:
        for key in ("lane_packet", "lane_packet_repeat", "candidate_archive"):
            fact = row.get(key)
            artifact = Path(fact["path"]) if isinstance(fact, dict) else None
            if artifact is None or not artifact.is_file() or file_fact(artifact) != fact:
                return None
        if require_identity:
            identity = row.get("receiver_identity")
            if not isinstance(identity, dict) or not (
                identity.get("byte_identical") and identity.get("receiver_closed")
            ):
                return None
            for key in ("reconstructed_field", "decoded_lane_bitplane"):
                fact = identity.get(key)
                artifact = Path(fact["path"]) if isinstance(fact, dict) else None
                if artifact is None or not artifact.is_file() or file_fact(artifact) != fact:
                    return None
    return result


ADAPTIVE_EXTENSION = r'''

static uint32_t d3b_mixed_lane_frequency(
    uint32_t base_lane,
    uint32_t count_road,
    uint32_t count_lane,
    uint32_t prior_strength
) {
    uint64_t observed = (uint64_t)count_road + (uint64_t)count_lane;
    uint64_t weight;
    uint64_t empirical_lane;
    uint64_t denominator;
    uint64_t mixed;
    if (!observed || !prior_strength) return base_lane;
    weight = observed < 4096u ? observed : 4096u;
    empirical_lane = (((uint64_t)count_lane + 1u) * RC64_TOTAL) / (observed + 2u);
    denominator = (uint64_t)prior_strength + weight;
    mixed = ((uint64_t)base_lane * prior_strength + empirical_lane * weight + denominator / 2u)
        / denominator;
    if (mixed < 1u) mixed = 1u;
    if (mixed >= RC64_TOTAL) mixed = RC64_TOTAL - 1u;
    return (uint32_t)mixed;
}

static void d3b_observe(
    uint32_t *count_road,
    uint32_t *count_lane,
    uint32_t symbol
) {
    uint64_t total = (uint64_t)(*count_road) + (uint64_t)(*count_lane);
    if (total >= (1u << 20)) {
        *count_road = (*count_road + 1u) >> 1u;
        *count_lane = (*count_lane + 1u) >> 1u;
    }
    if (symbol) (*count_lane)++;
    else (*count_road)++;
}

int d3b_encoder_encode_adaptive(
    void *opaque,
    const int32_t *symbols,
    const uint32_t *base_frequencies,
    const uint32_t *context_keys,
    size_t count,
    uint32_t *counts_road,
    uint32_t *counts_lane,
    size_t context_count,
    uint32_t prior_strength
) {
    size_t index;
    if (!opaque || (!symbols && count) || (!base_frequencies && count) ||
        (!context_keys && count) || !counts_road || !counts_lane || !context_count) return -20;
    for (index = 0u; index < count; ++index) {
        uint32_t key = context_keys[index];
        uint32_t row[2];
        uint32_t lane_frequency;
        int status;
        if (key >= context_count || symbols[index] < 0 || symbols[index] > 1) return -21;
        lane_frequency = d3b_mixed_lane_frequency(
            base_frequencies[index * 2u + 1u], counts_road[key], counts_lane[key], prior_strength
        );
        row[0] = (uint32_t)(RC64_TOTAL - lane_frequency);
        row[1] = lane_frequency;
        status = rc64_encoder_encode(opaque, symbols + index, row, 1u);
        if (status) return status - 30;
        d3b_observe(counts_road + key, counts_lane + key, (uint32_t)symbols[index]);
    }
    return 0;
}

int d3b_decoder_decode_adaptive(
    void *opaque,
    const uint32_t *base_frequencies,
    const uint32_t *context_keys,
    size_t count,
    uint32_t *counts_road,
    uint32_t *counts_lane,
    size_t context_count,
    uint32_t prior_strength,
    int32_t *symbols
) {
    rc64_decoder *decoder = (rc64_decoder *)opaque;
    size_t index;
    if (!decoder || (!base_frequencies && count) || (!context_keys && count) ||
        !counts_road || !counts_lane || !context_count || (!symbols && count)) return -40;
    for (index = 0u; index < count; ++index) {
        uint32_t key = context_keys[index];
        uint32_t row[2];
        uint32_t lane_frequency;
        int status;
        if (key >= context_count) return -41;
        lane_frequency = d3b_mixed_lane_frequency(
            base_frequencies[index * 2u + 1u], counts_road[key], counts_lane[key], prior_strength
        );
        row[0] = (uint32_t)(RC64_TOTAL - lane_frequency);
        row[1] = lane_frequency;
        status = rc64_decoder_decode_row(decoder, row, symbols + index);
        if (status) return status - 50;
        d3b_observe(counts_road + key, counts_lane + key, (uint32_t)symbols[index]);
    }
    return 0;
}
'''


def mixer_logit_table_source() -> str:
    """Emit the generic 12-bit probability/log-odds map used by the reference mixer."""
    values = [0]
    for frequency in range(1, 4096):
        probability = frequency / 4096.0
        values.append(round(math.log(probability / (1.0 - probability)) * (1 << 20)))
    values.append(values[-1])
    rows = [", ".join(str(value) for value in values[index:index + 12])
            for index in range(0, len(values), 12)]
    return "static const int32_t D3B_LOGIT_Q20[4097] = {\n" + ",\n".join(rows) + "\n};\n"


MIXER_EXTENSION = r'''

static uint32_t d3b_to_q12(uint32_t lane_frequency) {
    uint64_t scaled = (uint64_t)lane_frequency * 4096u + RC64_TOTAL / 2u;
    uint32_t value = (uint32_t)(scaled / RC64_TOTAL);
    if (value < 1u) value = 1u;
    if (value > 4095u) value = 4095u;
    return value;
}

static uint32_t d3b_from_logit_q20(int64_t target) {
    uint32_t low = 1u;
    uint32_t high = 4095u;
    while (low < high) {
        uint32_t middle = low + (high - low) / 2u;
        if ((int64_t)D3B_LOGIT_Q20[middle] < target) low = middle + 1u;
        else high = middle;
    }
    if (low > 1u) {
        int64_t above = (int64_t)D3B_LOGIT_Q20[low] - target;
        int64_t below = target - (int64_t)D3B_LOGIT_Q20[low - 1u];
        if (below <= above) low--;
    }
    return low;
}

static int64_t d3b_round_shift_signed(int64_t value, uint32_t shift) {
    int64_t half;
    if (!shift) return value;
    half = (int64_t)1 << (shift - 1u);
    if (value >= 0) return (value + half) >> shift;
    return -(((-value) + half) >> shift);
}

static uint32_t d3b_mixer_lane_frequency(
    uint32_t base_lane,
    uint32_t count_road,
    uint32_t count_lane,
    int32_t weight_q16,
    uint32_t *base_q12_out,
    uint32_t *empirical_q12_out
) {
    uint64_t observed = (uint64_t)count_road + (uint64_t)count_lane;
    uint32_t base_q12 = d3b_to_q12(base_lane);
    uint32_t empirical_q12 = (uint32_t)((((uint64_t)count_lane + 1u) * 4096u)
        / (observed + 2u));
    int64_t base_logit;
    int64_t empirical_logit;
    int64_t mixed_logit;
    uint32_t mixed_q12;
    uint64_t mixed_lane;
    if (empirical_q12 < 1u) empirical_q12 = 1u;
    if (empirical_q12 > 4095u) empirical_q12 = 4095u;
    base_logit = D3B_LOGIT_Q20[base_q12];
    empirical_logit = D3B_LOGIT_Q20[empirical_q12];
    mixed_logit = base_logit + d3b_round_shift_signed(
        (int64_t)weight_q16 * (empirical_logit - base_logit), 16u
    );
    mixed_q12 = d3b_from_logit_q20(mixed_logit);
    mixed_lane = ((uint64_t)mixed_q12 * RC64_TOTAL + 2048u) / 4096u;
    if (mixed_lane < 1u) mixed_lane = 1u;
    if (mixed_lane >= RC64_TOTAL) mixed_lane = RC64_TOTAL - 1u;
    *base_q12_out = base_q12;
    *empirical_q12_out = empirical_q12;
    return (uint32_t)mixed_lane;
}

static void d3b_update_mixer_weight(
    int32_t *weight_q16,
    uint32_t symbol,
    uint32_t mixed_lane,
    uint32_t base_q12,
    uint32_t empirical_q12,
    uint32_t learning_shift
) {
    int64_t residual_q12 = symbol ? (4096 - (int64_t)mixed_lane) : -(int64_t)mixed_lane;
    int64_t stretch_q20 = (int64_t)D3B_LOGIT_Q20[empirical_q12]
        - (int64_t)D3B_LOGIT_Q20[base_q12];
    uint32_t shift = 16u + learning_shift;
    int64_t step = d3b_round_shift_signed(residual_q12 * stretch_q20, shift);
    int64_t updated = (int64_t)(*weight_q16) + step;
    if (updated < -(4 << 16)) updated = -(4 << 16);
    if (updated > (8 << 16)) updated = (8 << 16);
    *weight_q16 = (int32_t)updated;
}

int d3b_encoder_encode_mixer(
    void *opaque,
    const int32_t *symbols,
    const uint32_t *base_frequencies,
    const uint32_t *context_keys,
    size_t count,
    uint32_t *counts_road,
    uint32_t *counts_lane,
    size_t context_count,
    int32_t *weight_q16,
    uint32_t learning_shift
) {
    size_t index;
    if (!opaque || (!symbols && count) || (!base_frequencies && count) ||
        (!context_keys && count) || !counts_road || !counts_lane || !context_count ||
        !weight_q16) return -60;
    for (index = 0u; index < count; ++index) {
        uint32_t key = context_keys[index];
        uint32_t row[2];
        uint32_t base_q12;
        uint32_t empirical_q12;
        uint32_t lane_frequency;
        uint32_t mixed_q12;
        int status;
        if (key >= context_count || symbols[index] < 0 || symbols[index] > 1) return -61;
        lane_frequency = d3b_mixer_lane_frequency(
            base_frequencies[index * 2u + 1u], counts_road[key], counts_lane[key],
            *weight_q16, &base_q12, &empirical_q12
        );
        row[0] = (uint32_t)(RC64_TOTAL - lane_frequency);
        row[1] = lane_frequency;
        status = rc64_encoder_encode(opaque, symbols + index, row, 1u);
        if (status) return status - 70;
        mixed_q12 = d3b_to_q12(lane_frequency);
        d3b_update_mixer_weight(
            weight_q16, (uint32_t)symbols[index], mixed_q12, base_q12,
            empirical_q12, learning_shift
        );
        d3b_observe(counts_road + key, counts_lane + key, (uint32_t)symbols[index]);
    }
    return 0;
}

int d3b_decoder_decode_mixer(
    void *opaque,
    const uint32_t *base_frequencies,
    const uint32_t *context_keys,
    size_t count,
    uint32_t *counts_road,
    uint32_t *counts_lane,
    size_t context_count,
    int32_t *weight_q16,
    uint32_t learning_shift,
    int32_t *symbols
) {
    rc64_decoder *decoder = (rc64_decoder *)opaque;
    size_t index;
    if (!decoder || (!base_frequencies && count) || (!context_keys && count) ||
        !counts_road || !counts_lane || !context_count || !weight_q16 ||
        (!symbols && count)) return -80;
    for (index = 0u; index < count; ++index) {
        uint32_t key = context_keys[index];
        uint32_t row[2];
        uint32_t base_q12;
        uint32_t empirical_q12;
        uint32_t lane_frequency;
        uint32_t mixed_q12;
        int status;
        if (key >= context_count) return -81;
        lane_frequency = d3b_mixer_lane_frequency(
            base_frequencies[index * 2u + 1u], counts_road[key], counts_lane[key],
            *weight_q16, &base_q12, &empirical_q12
        );
        row[0] = (uint32_t)(RC64_TOTAL - lane_frequency);
        row[1] = lane_frequency;
        status = rc64_decoder_decode_row(decoder, row, symbols + index);
        if (status) return status - 90;
        mixed_q12 = d3b_to_q12(lane_frequency);
        d3b_update_mixer_weight(
            weight_q16, (uint32_t)symbols[index], mixed_q12, base_q12,
            empirical_q12, learning_shift
        );
        d3b_observe(counts_road + key, counts_lane + key, (uint32_t)symbols[index]);
    }
    return 0;
}
'''


def compile_rc64(store: Path, alphabet: int, tag: str):
    route_b = jg2.load_route_b()
    route_b.ALPHABET = alphabet
    build = store / "retained/build" / tag
    build.mkdir(parents=True, exist_ok=True)
    base = jg2.resolve_rc64_base(route_b, build)
    source = base.read_text()
    needle = "#define RC64_ALPHABET 5u"
    if source.count(needle) != 1:
        raise D3BError("pinned RC64 source lost its unique alphabet macro")
    source = source.replace(needle, f"#define RC64_ALPHABET {alphabet}u")
    generated = build / f"rc64_backend_alphabet{alphabet}.c"
    library = build / f"librc64_alphabet{alphabet}.dylib"
    extension = "\n" + route_b.RC64_CHECKPOINT_EXTENSION
    if alphabet == 2:
        extension += "\n" + ADAPTIVE_EXTENSION
        extension += "\n" + mixer_logit_table_source() + "\n" + MIXER_EXTENSION
    atomic_bytes(generated, (source + extension).encode("utf-8"))
    command = [
        "/usr/bin/cc", "-O3", "-std=c11", "-shared", "-fPIC",
        "-ffp-contract=off", "-fno-fast-math", "-Wall", "-Wextra",
        str(generated), "-o", str(library),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    if completed.stderr.strip():
        raise D3BError(f"RC64 build emitted warnings: {completed.stderr}")
    return route_b, library, {
        "alphabet": alphabet,
        "argv": command,
        "base_source": file_fact(base),
        "generated_source": file_fact(generated),
        "library": file_fact(library),
    }


def bind_adaptive(library: ctypes.CDLL) -> None:
    i32p = ctypes.POINTER(ctypes.c_int32)
    u32p = ctypes.POINTER(ctypes.c_uint32)
    library.d3b_encoder_encode_adaptive.argtypes = [
        ctypes.c_void_p, i32p, u32p, u32p, ctypes.c_size_t,
        u32p, u32p, ctypes.c_size_t, ctypes.c_uint32,
    ]
    library.d3b_encoder_encode_adaptive.restype = ctypes.c_int
    library.d3b_decoder_decode_adaptive.argtypes = [
        ctypes.c_void_p, u32p, u32p, ctypes.c_size_t,
        u32p, u32p, ctypes.c_size_t, ctypes.c_uint32, i32p,
    ]
    library.d3b_decoder_decode_adaptive.restype = ctypes.c_int


def bind_mixer(library: ctypes.CDLL) -> None:
    i32p = ctypes.POINTER(ctypes.c_int32)
    u32p = ctypes.POINTER(ctypes.c_uint32)
    library.d3b_encoder_encode_mixer.argtypes = [
        ctypes.c_void_p,
        i32p,
        u32p,
        u32p,
        ctypes.c_size_t,
        u32p,
        u32p,
        ctypes.c_size_t,
        i32p,
        ctypes.c_uint32,
    ]
    library.d3b_encoder_encode_mixer.restype = ctypes.c_int
    library.d3b_decoder_decode_mixer.argtypes = [
        ctypes.c_void_p,
        u32p,
        u32p,
        ctypes.c_size_t,
        u32p,
        u32p,
        ctypes.c_size_t,
        i32p,
        ctypes.c_uint32,
        i32p,
    ]
    library.d3b_decoder_decode_mixer.restype = ctypes.c_int


def load_environment() -> dict[str, Any]:
    os.environ.setdefault("CP135_BROTLI_CLI", "/opt/homebrew/bin/brotli")
    residual, renderer, renderer_dir = jg2.load_runtime(d3.RUNTIME)
    parts = residual.read_residual_archive(d3.BASE_ARCHIVE)
    sections = jg2.split_member(jg2.read_archive_member(d3.BASE_ARCHIVE))
    if sections["tail"][jg2.RESIDUAL_COMPACT_BYTES:] != parts.token_stream:
        raise D3BError("GB1 member tail disagrees with the runtime parser")
    return {
        "residual": residual,
        "renderer": renderer,
        "renderer_dir": renderer_dir,
        "parts": parts,
        "sections": sections,
    }


def group_machine(env: dict[str, Any]):
    return d3.group_machine(env)


def probability_state_five(
    env: dict[str, Any], sparse: Any, corrector: Any, current: Any, context: Any,
    boundary: np.ndarray, group: int, flat_positions: np.ndarray,
) -> tuple[Any, np.ndarray, np.ndarray]:
    selected = sparse.selected_logits(current, context, group)
    base_logits = selected.cpu().numpy()
    predicted = base_logits.argmax(axis=1).astype(np.int64)
    feature = boundary[flat_positions].astype(np.int64) * d3.CANONICAL_ALPHABET + predicted
    corrected = base_logits + env["parts"].table.values[feature]
    probability = env["residual"]._probability_table(
        corrected, env["renderer"].HPAC_LOGIT_PRECISION
    )
    state = corrector.group_state(probability, predicted, flat_positions)
    coding_five = np.ascontiguousarray(corrector.coding_row(state), dtype=np.float32)
    return state, coding_five, predicted


def conditional_binary(coding_five: np.ndarray) -> np.ndarray:
    pair = coding_five[:, [ROAD, LANE]].astype(np.float64)
    pair /= pair.sum(axis=1, keepdims=True)
    values = np.ascontiguousarray(pair, dtype=np.float32)
    values /= values.sum(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    if np.any(values <= 0.0) or np.any(~np.isfinite(values)):
        raise D3BError("Road/Lane conditional probability is invalid")
    return values


def dense_quotient_probability(coding_five: np.ndarray) -> np.ndarray:
    return d3.pool_road_lane(coding_five)


def _shift_with_sentinel(plane: np.ndarray, dy: int, dx: int, sentinel: int = 5) -> np.ndarray:
    output = np.full(plane.shape, sentinel, dtype=np.uint8)
    y0 = max(0, -dy)
    y1 = min(H, H - dy)
    x0 = max(0, -dx)
    x1 = min(W, W - dx)
    output[y0:y1, x0:x1] = plane[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    return output


def _hash_add(state: np.ndarray, value: np.ndarray | int) -> np.ndarray:
    values = np.asarray(value, dtype=np.uint64)
    return ((state ^ (values + np.uint64(0x9E3779B9))) * np.uint64(16777619)) & np.uint64(0xFFFFFFFF)


def field_hash(plane: np.ndarray, radius: int) -> np.ndarray:
    state = np.full((H, W), np.uint64(2166136261), dtype=np.uint64)
    for dy, dx in (
        (-radius, -radius), (-radius, 0), (-radius, radius),
        (0, -radius), (0, radius),
        (radius, -radius), (radius, 0), (radius, radius),
    ):
        state = _hash_add(state, _shift_with_sentinel(plane, dy, dx))
    return state


def geometry_hash(state: np.ndarray, plane: np.ndarray, bins: int, radius: int) -> np.ndarray:
    yy, xx = np.indices((H, W), dtype=np.int32)
    support = plane == ROAD
    left = np.empty((H, W), dtype=np.uint16)
    right = np.empty((H, W), dtype=np.uint16)
    run = np.zeros(H, dtype=np.uint16)
    for x in range(W):
        run = np.where(support[:, x], run + 1, 0)
        left[:, x] = run
    run.fill(0)
    for x in range(W - 1, -1, -1):
        run = np.where(support[:, x], run + 1, 0)
        right[:, x] = run
    width = np.maximum(1, left.astype(np.int32) + right.astype(np.int32) - 1)
    lateral = np.minimum(bins - 1, (left.astype(np.int32) * bins) // (width + 1))
    y_bin = np.minimum(bins - 1, (yy * bins) // H)
    perspective_angle = np.minimum(
        bins - 1,
        (np.abs(2 * xx - (W - 1)) * bins * H) // (W * np.maximum(1, H - yy)),
    )
    output = _hash_add(state, lateral)
    output = _hash_add(output, y_bin)
    output = _hash_add(output, perspective_angle)
    output = _hash_add(output, (xx >= W // 2).astype(np.uint8))
    output = _hash_add(output, np.minimum(bins - 1, width // max(1, radius * 8)))
    return output


def temporal_hash(
    state: np.ndarray, merged: np.memmap, frame: int, radius: int, previous_lane: np.ndarray,
) -> np.ndarray:
    previous = np.asarray(merged[frame - 1], dtype=np.uint8) if frame else np.full((H, W), 5, np.uint8)
    following = (
        np.asarray(merged[frame + 1], dtype=np.uint8)
        if frame + 1 < N else np.full((H, W), 5, np.uint8)
    )
    output = _hash_add(state, previous)
    output = _hash_add(output, following)
    output = _hash_add(output, _shift_with_sentinel(previous, 0, -radius))
    output = _hash_add(output, _shift_with_sentinel(previous, 0, radius))
    output = _hash_add(output, previous_lane.astype(np.uint8))
    output = _hash_add(output, _shift_with_sentinel(previous_lane.astype(np.uint8), 0, -radius, 2))
    output = _hash_add(output, _shift_with_sentinel(previous_lane.astype(np.uint8), 0, radius, 2))
    return output


def reference_geometry_hash(
    state: np.ndarray, coverage: np.ndarray, bins: int, radius: int,
) -> np.ndarray:
    """Hash D3A AA-SDF distance/angle features with integer-only quantization."""
    clipped = np.clip(np.asarray(coverage, dtype=np.float32), 0.0, 1.0)
    quantized = np.rint(clipped * 255.0).astype(np.int16)
    dx = (
        _shift_with_sentinel(quantized.astype(np.uint8), 0, radius, 0).astype(np.int16)
        - _shift_with_sentinel(quantized.astype(np.uint8), 0, -radius, 0).astype(np.int16)
    )
    dy = (
        _shift_with_sentinel(quantized.astype(np.uint8), radius, 0, 0).astype(np.int16)
        - _shift_with_sentinel(quantized.astype(np.uint8), -radius, 0, 0).astype(np.int16)
    )
    dominant_vertical = np.abs(dy) > np.abs(dx)
    angle8 = (
        (dx < 0).astype(np.uint8)
        + 2 * (dy < 0).astype(np.uint8)
        + 4 * dominant_vertical.astype(np.uint8)
    )
    coverage_bin = np.minimum(bins - 1, (quantized * bins) // 256)
    distance_bin = np.minimum(bins - 1, ((255 - quantized) * bins) // 256)
    angle_bin = (angle8.astype(np.int16) * bins) // 8
    output = _hash_add(state, coverage_bin)
    output = _hash_add(output, distance_bin)
    output = _hash_add(output, angle_bin)
    output = _hash_add(output, (quantized >= 242).astype(np.uint8))
    output = _hash_add(
        output,
        _shift_with_sentinel(coverage_bin.astype(np.uint8), 0, -radius, bins),
    )
    output = _hash_add(
        output,
        _shift_with_sentinel(coverage_bin.astype(np.uint8), -radius, 0, bins),
    )
    return output


def context_key_planes(
    merged: np.memmap,
    frame: int,
    previous_lane: np.ndarray,
    reference_coverages: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    plane = np.asarray(merged[frame], dtype=np.uint8)
    outputs: dict[str, np.ndarray] = {}
    for radius in sorted({config.radius for config in CONFIGS if config.adaptive}):
        field = field_hash(plane, radius)
        for config in CONFIGS:
            if not config.adaptive or config.radius != radius:
                continue
            state = field
            if "geometry" in config.design:
                state = geometry_hash(state, plane, config.geom_bins, radius)
            if "temporal" in config.design:
                state = temporal_hash(state, merged, frame, radius, previous_lane)
            outputs[config.name] = np.ascontiguousarray(
                (state.reshape(-1) & np.uint64((1 << config.context_bits) - 1)), dtype=np.uint32
            )
    if reference_coverages is not None:
        plane = np.asarray(merged[frame], dtype=np.uint8)
        for config in CONFIGS:
            if not config.mixer:
                continue
            if config.d3a_rung not in reference_coverages:
                raise D3BError(f"reference coverage {config.d3a_rung} is absent")
            state = field_hash(plane, config.radius)
            state = reference_geometry_hash(
                state, reference_coverages[config.d3a_rung], config.geom_bins, config.radius
            )
            state = temporal_hash(state, merged, frame, config.radius, previous_lane)
            outputs[config.name] = np.ascontiguousarray(
                state.reshape(-1) & np.uint64((1 << config.context_bits) - 1),
                dtype=np.uint32,
            )
    return outputs


def apply_causal_lane_context(
    base_keys: np.ndarray,
    flat_positions: np.ndarray,
    current_lane: np.ndarray,
    known: np.ndarray,
    config: ContextConfig,
) -> np.ndarray:
    """Add only already-decoded Lane-plane neighbors to reference context keys."""
    flat = np.asarray(flat_positions, dtype=np.int64)
    y = flat // W
    x = flat % W
    state = np.asarray(base_keys, dtype=np.uint64)
    for dy, dx in ((0, -config.radius), (-config.radius, 0), (-config.radius, -config.radius),
                   (-config.radius, config.radius)):
        ny = y + dy
        nx = x + dx
        valid = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
        neighbor = np.zeros(len(flat), dtype=np.int64)
        neighbor[valid] = ny[valid] * W + nx[valid]
        value = np.full(len(flat), 2, dtype=np.uint8)
        observed = valid & known[neighbor]
        value[observed] = current_lane[neighbor[observed]].astype(np.uint8)
        state = _hash_add(state, value)
    return np.ascontiguousarray(
        state & np.uint64((1 << config.context_bits) - 1), dtype=np.uint32
    )


def d3a_build_rows() -> dict[str, dict[str, Any]]:
    if not D3A_BUILD_RESULT.is_file():
        raise D3BError("D3A BUILD_RESULT.json is absent")
    result = json.loads(D3A_BUILD_RESULT.read_text(encoding="utf-8"))
    if not result.get("complete"):
        raise D3BError("D3A build receipt is incomplete")
    return {row["candidate_id"]: row for row in result["rows"]}


def reference_encoder_inputs() -> tuple[dict[str, np.ndarray], dict[str, bytes], dict[str, Any]]:
    rows = d3a_build_rows()
    coverages: dict[str, np.ndarray] = {}
    carriers: dict[str, bytes] = {}
    receipts: dict[str, Any] = {}
    for rung in sorted({config.d3a_rung for config in CONFIGS if config.mixer}):
        assert rung is not None
        row = rows[rung]
        coverage_path = Path(row["coverage_n600"]["path"])
        carrier_path = Path(row["counted_carrier"]["path"])
        if file_fact(coverage_path) != row["coverage_n600"]:
            raise D3BError(f"D3A {rung} coverage custody drifted")
        if file_fact(carrier_path) != row["counted_carrier"]:
            raise D3BError(f"D3A {rung} counted carrier custody drifted")
        coverage = np.load(coverage_path, mmap_mode="r", allow_pickle=False)
        if coverage.shape != (N, H, W) or coverage.dtype != np.float32:
            raise D3BError(f"D3A {rung} coverage shape/dtype drifted")
        coverages[rung] = coverage
        carriers[rung] = carrier_path.read_bytes()
        receipts[rung] = {
            "coverage": row["coverage_n600"],
            "counted_carrier": row["counted_carrier"],
            "tolerance": row["tolerance"],
            "coverage_threshold": row["coverage_threshold"],
        }
    return coverages, carriers, receipts


def counted_lane_body(
    config: ContextConfig, rc64_body: bytes, reference_carriers: dict[str, bytes],
) -> tuple[bytes, dict[str, Any] | None]:
    if not config.mixer:
        return rc64_body, None
    if config.d3a_rung is None or config.d3a_rung not in reference_carriers:
        raise D3BError(f"reference carrier for {config.name} is absent")
    carrier = reference_carriers[config.d3a_rung]
    body = REFERENCE_PREFIX.pack(REFERENCE_MAGIC, len(carrier)) + carrier + rc64_body
    return body, {
        "rung": config.d3a_rung,
        "prefix_bytes": REFERENCE_PREFIX.size,
        "carrier_bytes": len(carrier),
        "carrier_sha256": sha256_bytes(carrier),
    }


def lane_packet(config: ContextConfig, body: bytes, support_symbols: int, lane_symbols: int) -> bytes:
    header = LANE_HEADER.pack(
        LANE_MAGIC,
        config.design_id,
        config.radius,
        config.context_bits,
        config.geom_bins,
        config.prior_strength,
        0,
        support_symbols,
        lane_symbols,
        len(body),
        bytes.fromhex(sha256_bytes(body)),
    )
    return header + body


def parse_lane_packet(payload: bytes) -> tuple[ContextConfig, bytes, dict[str, Any]]:
    if len(payload) < LANE_HEADER_BYTES:
        raise D3BError("Lane packet is truncated")
    unpacked = LANE_HEADER.unpack_from(payload)
    magic, design_id, radius, context_bits, geom_bins, strength, reserved = unpacked[:7]
    support_symbols, lane_symbols, body_bytes, body_sha = unpacked[7:]
    body = payload[LANE_HEADER_BYTES:]
    if magic != LANE_MAGIC or reserved != 0 or len(body) != body_bytes:
        raise D3BError("Lane packet framing changed")
    if sha256_bytes(body) != body_sha.hex():
        raise D3BError("Lane packet body hash mismatch")
    config = CONFIG_BY_ID.get(design_id)
    if config is None:
        raise D3BError(f"unknown Lane context design {design_id}")
    observed = (radius, context_bits, geom_bins, strength)
    expected = (config.radius, config.context_bits, config.geom_bins, config.prior_strength)
    if observed != expected:
        raise D3BError(f"Lane packet parameter drift for {config.name}: {observed} != {expected}")
    reference_carrier: bytes | None = None
    reference_fact: dict[str, Any] | None = None
    rc64_body = body
    if config.mixer:
        if len(body) < REFERENCE_PREFIX.size:
            raise D3BError("reference Lane body is truncated")
        reference_magic, carrier_bytes = REFERENCE_PREFIX.unpack_from(body)
        start = REFERENCE_PREFIX.size
        stop = start + carrier_bytes
        if reference_magic != REFERENCE_MAGIC or stop > len(body):
            raise D3BError("reference Lane carrier framing changed")
        reference_carrier = body[start:stop]
        rc64_body = body[stop:]
        reference_fact = {
            "rung": config.d3a_rung,
            "prefix_bytes": REFERENCE_PREFIX.size,
            "carrier_bytes": len(reference_carrier),
            "carrier_sha256": sha256_bytes(reference_carrier),
        }
    return config, rc64_body, {
        "support_symbols": int(support_symbols),
        "lane_symbols": int(lane_symbols),
        "body_bytes": int(body_bytes),
        "body_sha256": body_sha.hex(),
        "rc64_body_bytes": len(rc64_body),
        "rc64_body_sha256": sha256_bytes(rc64_body),
        "packet_bytes": len(payload),
        "header_bytes": LANE_HEADER_BYTES,
        "reference_carrier": reference_fact,
        "reference_carrier_payload": reference_carrier,
    }


def pack_factor_archive(
    env: dict[str, Any], four_stream: bytes, packet: bytes, destination: Path,
) -> None:
    sections = dict(env["sections"])
    sections["tail"] = (
        sections["tail"][:jg2.RESIDUAL_COMPACT_BYTES]
        + FACTOR_HEADER.pack(FACTOR_MAGIC, len(four_stream), len(packet))
        + four_stream
        + packet
    )
    jg2.pack_archive(jg2.join_member(sections), destination)


def parse_factor_archive(path: Path) -> tuple[dict[str, bytes], bytes, bytes]:
    sections = jg2.split_member(jg2.read_archive_member(path))
    tail = sections["tail"]
    offset = jg2.RESIDUAL_COMPACT_BYTES
    if len(tail) < offset + FACTOR_HEADER.size:
        raise D3BError("factor archive tail is truncated")
    magic, four_bytes, lane_bytes = FACTOR_HEADER.unpack_from(tail, offset)
    if magic != FACTOR_MAGIC:
        raise D3BError("factor archive lacks D3BF header")
    offset += FACTOR_HEADER.size
    four = tail[offset:offset + four_bytes]
    packet = tail[offset + four_bytes:offset + four_bytes + lane_bytes]
    if len(four) != four_bytes or len(packet) != lane_bytes or offset + four_bytes + lane_bytes != len(tail):
        raise D3BError("factor archive lengths do not close")
    return sections, four, packet


@dataclass
class EncodeCoder:
    config: ContextConfig
    encoder: Any
    repeat_encoder: Any
    counts_road: np.ndarray | None
    counts_lane: np.ndarray | None
    repeat_counts_road: np.ndarray | None
    repeat_counts_lane: np.ndarray | None
    mixer_weight: np.ndarray | None
    repeat_mixer_weight: np.ndarray | None


@dataclass
class DecodeCoder:
    config: ContextConfig
    decoder: Any
    counts_road: np.ndarray | None
    counts_lane: np.ndarray | None
    mixer_weight: np.ndarray | None
    output_path: Path
    lane_path: Path


def empty_counts(config: ContextConfig) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not config.adaptive:
        return None, None
    count = 1 << config.context_bits
    return np.zeros(count, dtype=np.uint32), np.zeros(count, dtype=np.uint32)


def empty_mixer_weight(config: ContextConfig) -> np.ndarray | None:
    return np.zeros(1, dtype=np.int32) if config.mixer else None


def adaptive_encode(
    coder: EncodeCoder, symbols: np.ndarray, frequencies: np.ndarray, keys: np.ndarray,
    *, repeat: bool,
) -> None:
    if not coder.config.adaptive:
        probabilities = frequencies.astype(np.float64) / float(1 << 31)
        encoder = coder.repeat_encoder if repeat else coder.encoder
        encoder.encode(symbols, np.ascontiguousarray(probabilities, dtype=np.float32))
        return
    encoder = coder.repeat_encoder if repeat else coder.encoder
    counts_road = coder.repeat_counts_road if repeat else coder.counts_road
    counts_lane = coder.repeat_counts_lane if repeat else coder.counts_lane
    assert counts_road is not None and counts_lane is not None
    library = encoder.library
    source = np.ascontiguousarray(symbols, dtype=np.int32)
    rows = np.ascontiguousarray(frequencies, dtype=np.uint32)
    context = np.ascontiguousarray(keys, dtype=np.uint32)
    u32p = ctypes.POINTER(ctypes.c_uint32)
    if coder.config.mixer:
        bind_mixer(library)
        weight = coder.repeat_mixer_weight if repeat else coder.mixer_weight
        assert weight is not None
        status = library.d3b_encoder_encode_mixer(
            encoder.context,
            source.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            rows.ctypes.data_as(u32p),
            context.ctypes.data_as(u32p),
            len(source),
            counts_road.ctypes.data_as(u32p),
            counts_lane.ctypes.data_as(u32p),
            len(counts_road),
            weight.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            coder.config.prior_strength,
        )
    else:
        bind_adaptive(library)
        status = library.d3b_encoder_encode_adaptive(
            encoder.context,
            source.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            rows.ctypes.data_as(u32p),
            context.ctypes.data_as(u32p),
            len(source),
            counts_road.ctypes.data_as(u32p),
            counts_lane.ctypes.data_as(u32p),
            len(counts_road),
            coder.config.prior_strength,
        )
    if status:
        raise D3BError(f"adaptive encode {coder.config.name} failed with status {status}")


def adaptive_decode(
    coder: DecodeCoder, probabilities: np.ndarray, frequencies: np.ndarray, keys: np.ndarray,
) -> np.ndarray:
    if not coder.config.adaptive:
        return coder.decoder.decode(None, probabilities)
    assert coder.counts_road is not None and coder.counts_lane is not None
    library = coder.decoder.library
    rows = np.ascontiguousarray(frequencies, dtype=np.uint32)
    context = np.ascontiguousarray(keys, dtype=np.uint32)
    output = np.empty(len(context), dtype=np.int32)
    u32p = ctypes.POINTER(ctypes.c_uint32)
    if coder.config.mixer:
        bind_mixer(library)
        assert coder.mixer_weight is not None
        status = library.d3b_decoder_decode_mixer(
            coder.decoder.context,
            rows.ctypes.data_as(u32p),
            context.ctypes.data_as(u32p),
            len(context),
            coder.counts_road.ctypes.data_as(u32p),
            coder.counts_lane.ctypes.data_as(u32p),
            len(coder.counts_road),
            coder.mixer_weight.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            coder.config.prior_strength,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
    else:
        bind_adaptive(library)
        status = library.d3b_decoder_decode_adaptive(
            coder.decoder.context,
            rows.ctypes.data_as(u32p),
            context.ctypes.data_as(u32p),
            len(context),
            coder.counts_road.ctypes.data_as(u32p),
            coder.counts_lane.ctypes.data_as(u32p),
            len(coder.counts_road),
            coder.config.prior_strength,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        )
    if status:
        raise D3BError(f"adaptive decode {coder.config.name} failed with status {status}")
    coder.decoder.decoded_symbols += len(output)
    return output


def initialize_encode_coders(route_b: Any, library: Path, checkpoint: Any | None) -> dict[str, EncodeCoder]:
    coders: dict[str, EncodeCoder] = {}
    for config in CONFIGS:
        if checkpoint is None:
            encoder = route_b.NativeRc64Encoder(library)
            repeat_encoder = route_b.NativeRc64Encoder(library)
            counts_road, counts_lane = empty_counts(config)
            repeat_counts_road, repeat_counts_lane = empty_counts(config)
            mixer_weight = empty_mixer_weight(config)
            repeat_mixer_weight = empty_mixer_weight(config)
        else:
            encoder = route_b.NativeRc64Encoder(
                library, np.asarray(checkpoint[f"encoder__{config.name}"], dtype=np.uint8).tobytes()
            )
            repeat_encoder = route_b.NativeRc64Encoder(
                library,
                np.asarray(checkpoint[f"repeat_encoder__{config.name}"], dtype=np.uint8).tobytes(),
            )
            if config.adaptive:
                counts_road = np.asarray(checkpoint[f"counts_road__{config.name}"], dtype=np.uint32).copy()
                counts_lane = np.asarray(checkpoint[f"counts_lane__{config.name}"], dtype=np.uint32).copy()
                repeat_counts_road = np.asarray(
                    checkpoint[f"repeat_counts_road__{config.name}"], dtype=np.uint32
                ).copy()
                repeat_counts_lane = np.asarray(
                    checkpoint[f"repeat_counts_lane__{config.name}"], dtype=np.uint32
                ).copy()
            else:
                counts_road = counts_lane = repeat_counts_road = repeat_counts_lane = None
            if config.mixer:
                mixer_weight = np.asarray(
                    checkpoint[f"mixer_weight__{config.name}"], dtype=np.int32
                ).copy()
                repeat_mixer_weight = np.asarray(
                    checkpoint[f"repeat_mixer_weight__{config.name}"], dtype=np.int32
                ).copy()
            else:
                mixer_weight = repeat_mixer_weight = None
        coders[config.name] = EncodeCoder(
            config, encoder, repeat_encoder, counts_road, counts_lane,
            repeat_counts_road, repeat_counts_lane, mixer_weight, repeat_mixer_weight,
        )
    return coders


def save_encode_checkpoint(
    path: Path, coders: dict[str, EncodeCoder], corrector: Any, cold: Any,
    frame: int, previous: np.ndarray, counters: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": np.array([ENCODE_CHECKPOINT_SCHEMA]),
        "frame": np.array([frame], dtype=np.int64),
        "previous": np.asarray(previous, dtype=np.uint8),
        "counters_json": np.frombuffer(json.dumps(counters, sort_keys=True).encode(), dtype=np.uint8),
    }
    state = jg2.corrector_state(corrector)
    lost = jg2.uncaptured_divergent_state(corrector, cold, set(state))
    if lost:
        raise D3BError(f"encode checkpoint would lose corrector state: {lost[:8]}")
    for key, value in state.items():
        payload[f"corrector__{key}"] = value
    for name, coder in coders.items():
        payload[f"encoder__{name}"] = np.frombuffer(coder.encoder.snapshot(), dtype=np.uint8)
        payload[f"repeat_encoder__{name}"] = np.frombuffer(
            coder.repeat_encoder.snapshot(), dtype=np.uint8
        )
        if coder.config.adaptive:
            payload[f"counts_road__{name}"] = coder.counts_road
            payload[f"counts_lane__{name}"] = coder.counts_lane
            payload[f"repeat_counts_road__{name}"] = coder.repeat_counts_road
            payload[f"repeat_counts_lane__{name}"] = coder.repeat_counts_lane
        if coder.config.mixer:
            payload[f"mixer_weight__{name}"] = coder.mixer_weight
            payload[f"repeat_mixer_weight__{name}"] = coder.repeat_mixer_weight
    temporary = path.with_suffix(".partial.npz")
    np.savez(temporary, **payload)
    os.replace(temporary, path)


def restore_corrector(corrector: Any, checkpoint: Any, prefix: str = "corrector__") -> None:
    state = {
        key[len(prefix):]: checkpoint[key]
        for key in checkpoint.files if key.startswith(prefix)
    }
    jg2.load_corrector_state(corrector, state)


def source_lane_payload(store: Path) -> dict[str, Any]:
    destination = store / "retained/source/lane_exact.packbits"
    if not destination.is_file():
        atomic_bytes(destination, (D3_STORE / "retained/carriers/lane_mask_exact.packbits").read_bytes())
    if destination.stat().st_size != (FIELD_BYTES + 7) // 8:
        raise D3BError("retained source Lane bitplane has the wrong length")
    return file_fact(destination)


def stage_encode(args: argparse.Namespace) -> dict[str, Any]:
    store = Path(args.store)
    if args.resume:
        completed = completed_result_if_current(
            store / "ENCODE_RESULT.json", require_identity=False
        )
        if completed is not None:
            return completed
    custody = verify_inputs(store)
    lane_source_fact = source_lane_payload(store)
    env = load_environment()
    route_b, library, build = compile_rc64(store, 2, "rc64_alphabet2_adaptive")
    reference_coverages, reference_carriers, reference_receipts = reference_encoder_inputs()
    source = np.memmap(SOURCE_FIELD, dtype=np.uint8, mode="r", shape=(N, H, W))
    merged = np.memmap(MERGED_FIELD, dtype=np.uint8, mode="r", shape=(N, H, W))
    if np.any((source == LANE) & (merged != ROAD)):
        raise D3BError("Lane support is not a subset of quotient Road")

    checkpoint_path = store / "retained/checkpoints/encode_latest.npz"
    checkpoint = None
    start = 0
    previous_seed: np.ndarray | None = None
    counters = {
        "positions": 0,
        "support": 0,
        "lane": 0,
        "joint_argmax_right": 0,
        "quotient_argmax_right": 0,
        "conditional_lane_argmax_right": 0,
    }
    if args.resume and checkpoint_path.is_file():
        checkpoint = np.load(checkpoint_path, allow_pickle=False)
        schema = str(np.asarray(checkpoint["schema"]).reshape(-1)[0])
        if schema != ENCODE_CHECKPOINT_SCHEMA:
            raise D3BError(f"refusing encode checkpoint schema {schema}")
        start = int(checkpoint["frame"][0])
        previous_seed = np.asarray(checkpoint["previous"], dtype=np.uint8).copy()
        counters = json.loads(np.asarray(checkpoint["counters_json"], dtype=np.uint8).tobytes())

    torch_mod, device, model, sparse, corrector, groups = group_machine(env)
    from runtime.free_corrector import FreeCorrector  # type: ignore[import-not-found]
    cold = FreeCorrector(PLANE)
    if checkpoint is not None:
        restore_corrector(corrector, checkpoint)
    coders = initialize_encode_coders(route_b, library, checkpoint)
    started = time.perf_counter()

    with torch_mod.inference_mode():
        if previous_seed is None:
            previous = torch_mod.zeros((1, H, W), dtype=torch_mod.long, device=device)
        else:
            previous = torch_mod.from_numpy(previous_seed.astype(np.int64)).reshape(1, H, W).to(device)
        for frame in range(start, N):
            index = torch_mod.tensor([frame], dtype=torch_mod.long, device=device)
            current = torch_mod.zeros_like(previous)
            context = model.prepare_frame_context(index, previous)
            if frame:
                previous_cpu = previous[0].to(device="cpu", dtype=torch_mod.uint8).numpy()
                boundary = env["residual"]._boundary_buckets(previous_cpu).reshape(-1)
                previous_lane = previous_cpu == LANE
            else:
                boundary = np.full(PLANE, 4, dtype=np.uint8)
                previous_lane = np.zeros((H, W), dtype=bool)
            corrector.begin_frame(boundary)
            frame_coverages = {
                rung: np.asarray(coverage[frame], dtype=np.float32)
                for rung, coverage in reference_coverages.items()
            }
            key_planes = context_key_planes(merged, frame, previous_lane, frame_coverages)
            current_lane_context = np.zeros(PLANE, dtype=bool)
            known_lane_context = np.zeros(PLANE, dtype=bool)
            target_plane = np.asarray(source[frame], dtype=np.uint8).reshape(-1)
            merged_plane = np.asarray(merged[frame], dtype=np.uint8).reshape(-1)
            for group, (device_positions, flat_positions) in enumerate(groups):
                state, coding_five, _ = probability_state_five(
                    env, sparse, corrector, current, context, boundary, group, flat_positions
                )
                canonical = target_plane[flat_positions].astype(np.int64)
                quotient = merged_plane[flat_positions].astype(np.int64)
                support = quotient == ROAD
                symbols = (canonical[support] == LANE).astype(np.int32)
                if len(symbols):
                    conditional = conditional_binary(coding_five[support])
                    frequencies = route_b.quantize_probabilities(conditional)
                    for name, coder in coders.items():
                        if coder.config.adaptive:
                            support_positions = flat_positions[support]
                            keys = key_planes[name][support_positions]
                            if coder.config.mixer:
                                keys = apply_causal_lane_context(
                                    keys,
                                    support_positions,
                                    current_lane_context,
                                    known_lane_context,
                                    coder.config,
                                )
                        else:
                            keys = np.zeros(len(symbols), dtype=np.uint32)
                        adaptive_encode(coder, symbols, frequencies, keys, repeat=False)
                        adaptive_encode(coder, symbols, frequencies, keys, repeat=True)
                    conditional_right = int((conditional.argmax(axis=1) == symbols).sum())
                else:
                    conditional_right = 0

                predicted_joint = coding_five.argmax(axis=1)
                predicted_quotient = dense_quotient_probability(coding_five).argmax(axis=1)
                quotient_dense = d3.CANONICAL_TO_DENSE[quotient]
                counters["positions"] += len(canonical)
                counters["support"] += int(support.sum())
                counters["lane"] += int(symbols.sum())
                counters["joint_argmax_right"] += int((predicted_joint == canonical).sum())
                counters["quotient_argmax_right"] += int((predicted_quotient == quotient_dense).sum())
                counters["conditional_lane_argmax_right"] += conditional_right

                current_lane_context[flat_positions] = canonical == LANE
                known_lane_context[flat_positions] = True

                corrector.observe(state, canonical)
                current.reshape(-1)[device_positions] = torch_mod.from_numpy(canonical).to(device)
            frame_tokens = current[0].to(device="cpu", dtype=torch_mod.uint8).numpy()
            if not np.array_equal(frame_tokens.reshape(-1), target_plane):
                raise D3BError(f"encode causal field diverged at frame {frame}")
            corrector.end_frame(frame_tokens.reshape(-1))
            previous = current
            if args.checkpoint_every and (frame + 1) % args.checkpoint_every == 0 and frame + 1 < N:
                save_encode_checkpoint(
                    checkpoint_path, coders, corrector, cold, frame + 1, frame_tokens, counters
                )
                progress(
                    stage="encode", event="checkpoint", frame=frame + 1,
                    support_symbols=counters["support"],
                    elapsed_seconds=time.perf_counter() - started,
                )

    if counters["positions"] != FIELD_BYTES or counters["lane"] != 691_095:
        raise D3BError(f"encode denominators drifted: {counters}")
    four_stream = FOUR_STREAM.read_bytes()
    rows = []
    candidate_root = store / "retained/candidates"
    stream_root = store / "retained/lane_streams"
    for name, coder in coders.items():
        body = coder.encoder.finish()
        repeat = coder.repeat_encoder.finish()
        if body != repeat:
            raise D3BError(f"determinism repeat differs for {name}")
        counted_body, reference_fact = counted_lane_body(
            coder.config, body, reference_carriers
        )
        counted_repeat, repeat_reference_fact = counted_lane_body(
            coder.config, repeat, reference_carriers
        )
        if reference_fact != repeat_reference_fact:
            raise D3BError(f"reference carrier repeat differs for {name}")
        packet = lane_packet(
            coder.config, counted_body, counters["support"], counters["lane"]
        )
        stream_path = stream_root / f"{name}.d3b"
        repeat_path = stream_root / f"{name}.repeat.d3b"
        atomic_bytes(stream_path, packet)
        atomic_bytes(
            repeat_path,
            lane_packet(
                coder.config, counted_repeat, counters["support"], counters["lane"]
            ),
        )
        archive_path = candidate_root / f"candidate_{name}.zip"
        pack_factor_archive(env, four_stream, packet, archive_path)
        _, parsed_four, parsed_packet = parse_factor_archive(archive_path)
        if parsed_four != four_stream or parsed_packet != packet:
            raise D3BError(f"packed archive parse-back differs for {name}")
        token_subsystem = MODEL_BYTES + len(four_stream) + FACTOR_HEADER.size + len(packet)
        if archive_path.stat().st_size != token_subsystem + OTHER_ARCHIVE_BYTES:
            raise D3BError(f"archive accounting does not close for {name}")
        rows.append({
            "config": asdict(coder.config),
            "lane_rc64_body": {"bytes": len(body), "sha256": sha256_bytes(body)},
            "counted_reference_carrier": reference_fact,
            "lane_packet": file_fact(stream_path),
            "lane_packet_repeat": file_fact(repeat_path),
            "factor_framing_bytes": FACTOR_HEADER.size,
            "counted_lane_context_parameters_bytes": (
                REFERENCE_PREFIX.size if coder.config.mixer
                else 6 if coder.config.adaptive else 1
            ),
            "counted_lane_packet_header_bytes": LANE_HEADER_BYTES,
            "four_class_stream_bytes": len(four_stream),
            "four_class_model_bytes": MODEL_BYTES,
            "token_subsystem_bytes": token_subsystem,
            "delta_vs_127292_bytes": token_subsystem - DX2_TOKEN_SUBSYSTEM_BYTES,
            "delta_vs_85064_bytes": token_subsystem - DEMAND_CLOSING_BYTES,
            "candidate_archive": file_fact(archive_path),
            "archive_delta_vs_gb1_bytes": archive_path.stat().st_size - d3.BASE_ARCHIVE_BYTES,
            "receiver_identity": "PENDING independent decode stage",
            "final_mixer_weight_q16": (
                int(coder.mixer_weight[0]) if coder.mixer_weight is not None else None
            ),
        })
    rows.sort(key=lambda row: row["token_subsystem_bytes"])
    result = {
        "schema": "ddm_d3b_encode.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": False,
        "custody": custody,
        "source_lane_bitplane": lane_source_fact,
        "rc64_build": build,
        "reference_context_receipts": reference_receipts,
        "denominators": counters,
        "binary_question_decomposition": {
            "joint_hpac_argmax_right_fraction": counters["joint_argmax_right"] / FIELD_BYTES,
            "quotient_argmax_right_fraction": counters["quotient_argmax_right"] / FIELD_BYTES,
            "lane_support_fraction": counters["support"] / FIELD_BYTES,
            "lane_one_fraction_within_support": counters["lane"] / counters["support"],
            "conditional_lane_argmax_right_fraction": (
                counters["conditional_lane_argmax_right"] / counters["support"]
            ),
        },
        "rows": rows,
        "best_config": rows[0]["config"]["name"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(store / "ENCODE_RESULT.json", result)
    atomic_json(store / "ENCODE_STAGE_COMPLETE.json", {
        "schema": "ddm_d3b_stage_complete.v1",
        "stage": "encode",
        "result": file_fact(store / "ENCODE_RESULT.json"),
        "completed_configs": [row["config"]["name"] for row in rows],
    })
    return result


def save_quotient_checkpoint(
    path: Path, decoder: Any, corrector: Any, cold: Any, frame: int, previous: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = jg2.corrector_state(corrector)
    lost = jg2.uncaptured_divergent_state(corrector, cold, set(state))
    if lost:
        raise D3BError(f"quotient checkpoint would lose corrector state: {lost[:8]}")
    payload: dict[str, Any] = {
        "schema": np.array([QUOTIENT_CHECKPOINT_SCHEMA]),
        "frame": np.array([frame], dtype=np.int64),
        "previous": np.asarray(previous, dtype=np.uint8),
        "decoder": decoder.get_compressed(),
    }
    for key, value in state.items():
        payload[f"corrector__{key}"] = value
    temporary = path.with_suffix(".partial.npz")
    np.savez(temporary, **payload)
    os.replace(temporary, path)


def decode_quotient(
    store: Path, env: dict[str, Any], four_stream: bytes, checkpoint_every: int, resume: bool,
) -> tuple[Path, dict[str, Any]]:
    route_b, library, build = compile_rc64(store, 4, "rc64_alphabet4_receiver")
    torch_mod, device, model, sparse, corrector, groups = group_machine(env)
    from runtime.free_corrector import FreeCorrector  # type: ignore[import-not-found]
    cold = FreeCorrector(PLANE)
    root = store / "retained/decode"
    root.mkdir(parents=True, exist_ok=True)
    output = root / "quotient_receiver.u8"
    partial = output.with_suffix(".u8.partial")
    checkpoint_path = store / "retained/checkpoints/quotient_decode_latest.npz"
    start = 0
    previous_seed: np.ndarray | None = None
    if resume and checkpoint_path.is_file():
        checkpoint = np.load(checkpoint_path, allow_pickle=False)
        schema = str(np.asarray(checkpoint["schema"]).reshape(-1)[0])
        if schema != QUOTIENT_CHECKPOINT_SCHEMA:
            raise D3BError(f"refusing quotient checkpoint schema {schema}")
        start = int(checkpoint["frame"][0])
        previous_seed = np.asarray(checkpoint["previous"], dtype=np.uint8).copy()
        restore_corrector(corrector, checkpoint)
        decoder = route_b.NativeRc64Decoder(
            library, np.asarray(checkpoint["decoder"], dtype="<u4").tobytes()
        )
        if not partial.is_file() or partial.stat().st_size != start * PLANE:
            raise D3BError("quotient checkpoint lacks its exact partial field")
    else:
        decoder = route_b.NativeRc64Decoder(library, route_b.TOKEN_MAGIC + four_stream)
    started = time.perf_counter()
    mode = "ab" if start else "wb"
    with partial.open(mode) as handle, torch_mod.inference_mode():
        if previous_seed is None:
            previous = torch_mod.zeros((1, H, W), dtype=torch_mod.long, device=device)
        else:
            previous = torch_mod.from_numpy(previous_seed.astype(np.int64)).reshape(1, H, W).to(device)
        for frame in range(start, N):
            index = torch_mod.tensor([frame], dtype=torch_mod.long, device=device)
            current = torch_mod.zeros_like(previous)
            context = model.prepare_frame_context(index, previous)
            if frame:
                previous_cpu = previous[0].to(device="cpu", dtype=torch_mod.uint8).numpy()
                boundary = env["residual"]._boundary_buckets(previous_cpu).reshape(-1)
            else:
                boundary = np.full(PLANE, 4, dtype=np.uint8)
            corrector.begin_frame(boundary)
            for group, (device_positions, flat_positions) in enumerate(groups):
                state, coding_five, _ = probability_state_five(
                    env, sparse, corrector, current, context, boundary, group, flat_positions
                )
                dense = decoder.decode(None, dense_quotient_probability(coding_five))
                if np.any((dense < 0) | (dense >= 4)):
                    raise D3BError(f"quotient decoder left alphabet at frame {frame} group {group}")
                canonical = d3.LIVE_CANONICAL[dense]
                corrector.observe(state, canonical.astype(np.int64))
                current.reshape(-1)[device_positions] = torch_mod.from_numpy(
                    canonical.astype(np.int64)
                ).to(device)
            frame_tokens = current[0].to(device="cpu", dtype=torch_mod.uint8).numpy()
            handle.write(frame_tokens.tobytes(order="C"))
            corrector.end_frame(frame_tokens.reshape(-1))
            previous = current
            if checkpoint_every and (frame + 1) % checkpoint_every == 0 and frame + 1 < N:
                handle.flush()
                os.fsync(handle.fileno())
                save_quotient_checkpoint(
                    checkpoint_path, decoder, corrector, cold, frame + 1, frame_tokens
                )
                progress(
                    stage="decode", substage="quotient", event="checkpoint", frame=frame + 1,
                    elapsed_seconds=time.perf_counter() - started,
                )
    os.replace(partial, output)
    if output.stat().st_size != FIELD_BYTES or sha256_file(output) != MERGED_FIELD_SHA256:
        raise D3BError("independently decoded quotient field is not byte-identical to D3")
    if not decoder.is_empty():
        raise D3BError("quotient decoder did not consume exactly n600 symbols")
    return output, {
        "decoded_field": file_fact(output),
        "target_sha256": MERGED_FIELD_SHA256,
        "byte_identical": True,
        "decoded_symbols": FIELD_BYTES,
        "rc64_build": build,
        "elapsed_seconds": time.perf_counter() - started,
    }


def initialize_decode_coders(
    route_b: Any, library: Path, packet_rows: dict[str, tuple[bytes, dict[str, Any]]],
    root: Path, checkpoint: Any | None,
) -> dict[str, DecodeCoder]:
    coders: dict[str, DecodeCoder] = {}
    for config in CONFIGS:
        body, _ = packet_rows[config.name]
        if checkpoint is None:
            decoder_payload = body
            counts_road, counts_lane = empty_counts(config)
            mixer_weight = empty_mixer_weight(config)
        else:
            decoder_payload = np.asarray(
                checkpoint[f"decoder__{config.name}"], dtype="<u4"
            ).tobytes()
            if config.adaptive:
                counts_road = np.asarray(
                    checkpoint[f"counts_road__{config.name}"], dtype=np.uint32
                ).copy()
                counts_lane = np.asarray(
                    checkpoint[f"counts_lane__{config.name}"], dtype=np.uint32
                ).copy()
            else:
                counts_road = counts_lane = None
            if config.mixer:
                mixer_weight = np.asarray(
                    checkpoint[f"mixer_weight__{config.name}"], dtype=np.int32
                ).copy()
            else:
                mixer_weight = None
        decoder = route_b.NativeRc64Decoder(library, decoder_payload)
        coders[config.name] = DecodeCoder(
            config,
            decoder,
            counts_road,
            counts_lane,
            mixer_weight,
            root / f"reconstructed_{config.name}.u8",
            root / f"lane_{config.name}.packbits",
        )
    return coders


def save_lane_checkpoint(
    path: Path, coders: dict[str, DecodeCoder], corrector: Any, cold: Any,
    frame: int, previous: np.ndarray, support_symbols: int, lane_symbols: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = jg2.corrector_state(corrector)
    lost = jg2.uncaptured_divergent_state(corrector, cold, set(state))
    if lost:
        raise D3BError(f"Lane checkpoint would lose corrector state: {lost[:8]}")
    payload: dict[str, Any] = {
        "schema": np.array([LANE_CHECKPOINT_SCHEMA]),
        "frame": np.array([frame], dtype=np.int64),
        "previous": np.asarray(previous, dtype=np.uint8),
        "support_symbols": np.array([support_symbols], dtype=np.int64),
        "lane_symbols": np.array([lane_symbols], dtype=np.int64),
    }
    for key, value in state.items():
        payload[f"corrector__{key}"] = value
    for name, coder in coders.items():
        payload[f"decoder__{name}"] = coder.decoder.get_compressed()
        if coder.config.adaptive:
            payload[f"counts_road__{name}"] = coder.counts_road
            payload[f"counts_lane__{name}"] = coder.counts_lane
        if coder.config.mixer:
            payload[f"mixer_weight__{name}"] = coder.mixer_weight
    temporary = path.with_suffix(".partial.npz")
    np.savez(temporary, **payload)
    os.replace(temporary, path)


def reference_decoder_inputs(
    packet_rows: dict[str, tuple[bytes, dict[str, Any]]],
) -> tuple[dict[str, list[list[Any]]], dict[str, Any]]:
    decoded: dict[str, list[list[Any]]] = {}
    receipts: dict[str, Any] = {}
    for config in CONFIGS:
        if not config.mixer:
            continue
        facts = packet_rows[config.name][1]
        carrier = facts.get("reference_carrier_payload")
        if not isinstance(carrier, bytes):
            raise D3BError(f"{config.name} lacks its counted D3A carrier")
        rung = config.d3a_rung
        assert rung is not None
        if rung in receipts:
            if receipts[rung]["carrier_sha256"] != sha256_bytes(carrier):
                raise D3BError(f"reference configs disagree on D3A {rung} carrier")
            continue
        lines, headers, restored = d3a.decode_counted_carrier(carrier)
        if len(lines) != N:
            raise D3BError(f"D3A {rung} receiver returned {len(lines)} frames")
        decoded[rung] = lines
        receipts[rung] = {
            "carrier_bytes": len(carrier),
            "carrier_sha256": sha256_bytes(carrier),
            "restored_lbnd2_bytes": len(restored),
            "pairs": len(lines),
            "carrier_candidate_id": headers["carrier"].get("candidate_id"),
            "independent_parse_back": True,
        }
        if receipts[rung]["carrier_candidate_id"] != rung:
            raise D3BError(f"D3A packet says {receipts[rung]['carrier_candidate_id']} not {rung}")
    return decoded, receipts


def reference_coverage_frame(lines: list[list[Any]], frame: int) -> np.ndarray:
    cfg = d3a.render_config()
    return d3a.rasterize_lane_coverage_range_dependent(
        lines[frame],
        h=H,
        w=W,
        softness=cfg.softness,
        dash_gate=cfg.dash_gate,
        dash_forward_max_m=cfg.dash_forward_max_m,
        v_h=cfg.v_h,
        cx=cfg.cx,
    ).astype(np.float32, copy=False)


def decode_lane_packets(
    store: Path, env: dict[str, Any], quotient_path: Path,
    packet_rows: dict[str, tuple[bytes, dict[str, Any]]], checkpoint_every: int, resume: bool,
) -> dict[str, Any]:
    route_b, library, build = compile_rc64(store, 2, "rc64_alphabet2_adaptive")
    quotient = np.memmap(quotient_path, dtype=np.uint8, mode="r", shape=(N, H, W))
    torch_mod, device, model, sparse, corrector, groups = group_machine(env)
    from runtime.free_corrector import FreeCorrector  # type: ignore[import-not-found]
    cold = FreeCorrector(PLANE)
    root = store / "retained/decode"
    checkpoint_path = store / "retained/checkpoints/lane_decode_latest.npz"
    checkpoint = None
    start = 0
    previous_seed: np.ndarray | None = None
    support_symbols = 0
    lane_symbols = 0
    if resume and checkpoint_path.is_file():
        checkpoint = np.load(checkpoint_path, allow_pickle=False)
        schema = str(np.asarray(checkpoint["schema"]).reshape(-1)[0])
        if schema != LANE_CHECKPOINT_SCHEMA:
            raise D3BError(f"refusing Lane checkpoint schema {schema}")
        start = int(checkpoint["frame"][0])
        previous_seed = np.asarray(checkpoint["previous"], dtype=np.uint8).copy()
        support_symbols = int(checkpoint["support_symbols"][0])
        lane_symbols = int(checkpoint["lane_symbols"][0])
        restore_corrector(corrector, checkpoint)
    coders = initialize_decode_coders(route_b, library, packet_rows, root, checkpoint)
    reference_lines, reference_receipts = reference_decoder_inputs(packet_rows)
    partial_outputs = {
        name: coder.output_path.with_suffix(".u8.partial") for name, coder in coders.items()
    }
    partial_lanes = {
        name: coder.lane_path.with_suffix(".packbits.partial") for name, coder in coders.items()
    }
    if start:
        for name in coders:
            if partial_outputs[name].stat().st_size != start * PLANE:
                raise D3BError(f"Lane checkpoint partial field drifted for {name}")
            if partial_lanes[name].stat().st_size != start * (PLANE // 8):
                raise D3BError(f"Lane checkpoint partial bitplane drifted for {name}")
    output_handles = {
        name: path.open("ab" if start else "wb") for name, path in partial_outputs.items()
    }
    lane_handles = {
        name: path.open("ab" if start else "wb") for name, path in partial_lanes.items()
    }
    started = time.perf_counter()
    try:
        with torch_mod.inference_mode():
            if previous_seed is None:
                previous = torch_mod.zeros((1, H, W), dtype=torch_mod.long, device=device)
            else:
                previous = torch_mod.from_numpy(previous_seed.astype(np.int64)).reshape(1, H, W).to(device)
            for frame in range(start, N):
                index = torch_mod.tensor([frame], dtype=torch_mod.long, device=device)
                current = torch_mod.zeros_like(previous)
                context = model.prepare_frame_context(index, previous)
                if frame:
                    previous_cpu = previous[0].to(device="cpu", dtype=torch_mod.uint8).numpy()
                    boundary = env["residual"]._boundary_buckets(previous_cpu).reshape(-1)
                    previous_lane = previous_cpu == LANE
                else:
                    boundary = np.full(PLANE, 4, dtype=np.uint8)
                    previous_lane = np.zeros((H, W), dtype=bool)
                corrector.begin_frame(boundary)
                frame_coverages = {
                    rung: reference_coverage_frame(lines, frame)
                    for rung, lines in reference_lines.items()
                }
                key_planes = context_key_planes(
                    quotient, frame, previous_lane, frame_coverages
                )
                quotient_plane = np.asarray(quotient[frame], dtype=np.uint8).reshape(-1)
                frame_lane = np.zeros(PLANE, dtype=bool)
                known_lane_context = np.zeros(PLANE, dtype=bool)
                for group, (device_positions, flat_positions) in enumerate(groups):
                    state, coding_five, _ = probability_state_five(
                        env, sparse, corrector, current, context, boundary, group, flat_positions
                    )
                    quotient_symbols = quotient_plane[flat_positions].astype(np.int64)
                    support = quotient_symbols == ROAD
                    decoded_by_name: dict[str, np.ndarray] = {}
                    if int(support.sum()):
                        conditional = conditional_binary(coding_five[support])
                        frequencies = route_b.quantize_probabilities(conditional)
                        for name, coder in coders.items():
                            if coder.config.adaptive:
                                support_positions = flat_positions[support]
                                keys = key_planes[name][support_positions]
                                if coder.config.mixer:
                                    keys = apply_causal_lane_context(
                                        keys,
                                        support_positions,
                                        frame_lane,
                                        known_lane_context,
                                        coder.config,
                                    )
                            else:
                                keys = np.zeros(int(support.sum()), dtype=np.uint32)
                            decoded_by_name[name] = adaptive_decode(
                                coder, conditional, frequencies, keys
                            )
                        reference = decoded_by_name[CONFIGS[0].name]
                        for name, decoded in decoded_by_name.items():
                            if not np.array_equal(decoded, reference):
                                raise D3BError(
                                    f"Lane candidate {name} disagrees with the independent base receiver "
                                    f"at frame {frame} group {group}"
                                )
                        if np.any((reference < 0) | (reference > 1)):
                            raise D3BError("Lane decoder left the binary alphabet")
                        frame_lane[flat_positions[support]] = reference.astype(bool)
                    canonical = quotient_symbols.copy()
                    canonical[support] = np.where(
                        frame_lane[flat_positions[support]], LANE, ROAD
                    )
                    frame_lane[flat_positions] = canonical == LANE
                    known_lane_context[flat_positions] = True
                    corrector.observe(state, canonical.astype(np.int64))
                    current.reshape(-1)[device_positions] = torch_mod.from_numpy(
                        canonical.astype(np.int64)
                    ).to(device)
                frame_tokens = current[0].to(device="cpu", dtype=torch_mod.uint8).numpy()
                packed_lane = np.packbits(frame_lane, bitorder="little").tobytes()
                for name in coders:
                    output_handles[name].write(frame_tokens.tobytes(order="C"))
                    lane_handles[name].write(packed_lane)
                support_symbols += int((quotient_plane == ROAD).sum())
                lane_symbols += int(frame_lane.sum())
                corrector.end_frame(frame_tokens.reshape(-1))
                previous = current
                if checkpoint_every and (frame + 1) % checkpoint_every == 0 and frame + 1 < N:
                    for handle in (*output_handles.values(), *lane_handles.values()):
                        handle.flush()
                        os.fsync(handle.fileno())
                    save_lane_checkpoint(
                        checkpoint_path, coders, corrector, cold, frame + 1, frame_tokens,
                        support_symbols, lane_symbols,
                    )
                    progress(
                        stage="decode", substage="lane", event="checkpoint", frame=frame + 1,
                        support_symbols=support_symbols,
                        elapsed_seconds=time.perf_counter() - started,
                    )
    finally:
        for handle in (*output_handles.values(), *lane_handles.values()):
            handle.close()

    rows = []
    for name, coder in coders.items():
        os.replace(partial_outputs[name], coder.output_path)
        os.replace(partial_lanes[name], coder.lane_path)
        output_fact = file_fact(coder.output_path)
        lane_fact = file_fact(coder.lane_path)
        source_lane = store / "retained/source/lane_exact.packbits"
        byte_identical = (
            output_fact["bytes"] == FIELD_BYTES
            and output_fact["sha256"] == SOURCE_FIELD_SHA256
            and lane_fact["sha256"] == sha256_file(source_lane)
        )
        expected_support = packet_rows[name][1]["support_symbols"]
        expected_lane = packet_rows[name][1]["lane_symbols"]
        if not byte_identical or coder.decoder.decoded_symbols != expected_support:
            raise D3BError(f"independent five-class identity failed for {name}")
        if support_symbols != expected_support or lane_symbols != expected_lane:
            raise D3BError(f"Lane denominator mismatch for {name}")
        rows.append({
            "config": name,
            "reconstructed_field": output_fact,
            "decoded_lane_bitplane": lane_fact,
            "source_field_sha256": SOURCE_FIELD_SHA256,
            "source_lane_sha256": sha256_file(source_lane),
            "support_symbols": expected_support,
            "lane_symbols": expected_lane,
            "byte_identical": True,
            "receiver_closed": True,
        })
    return {
        "rows": rows,
        "rc64_build": build,
        "support_symbols": support_symbols,
        "lane_symbols": lane_symbols,
        "reference_context_receivers": reference_receipts,
        "elapsed_seconds": time.perf_counter() - started,
    }


def stage_decode(args: argparse.Namespace) -> dict[str, Any]:
    store = Path(args.store)
    if args.resume:
        completed = completed_result_if_current(store / "RESULT.json", require_identity=True)
        if completed is not None:
            return completed
    custody = verify_inputs(store)
    encode_result_path = store / "ENCODE_RESULT.json"
    if not encode_result_path.is_file():
        raise D3BError("decode requires a complete ENCODE_RESULT.json")
    encode_result = json.loads(encode_result_path.read_text())
    if not encode_result.get("complete"):
        raise D3BError("encode stage is incomplete")
    env = load_environment()
    packet_rows: dict[str, tuple[bytes, dict[str, Any]]] = {}
    parsed_four: bytes | None = None
    for row in encode_result["rows"]:
        name = row["config"]["name"]
        archive = Path(row["candidate_archive"]["path"])
        if file_fact(archive) != row["candidate_archive"]:
            raise D3BError(f"candidate archive drifted before decode: {name}")
        _, four, packet = parse_factor_archive(archive)
        config, body, facts = parse_lane_packet(packet)
        if config.name != name:
            raise D3BError(f"archive config identity drifted: {name}")
        if parsed_four is None:
            parsed_four = four
        elif four != parsed_four:
            raise D3BError("factor archives disagree on their four-class stream")
        packet_rows[name] = (body, facts)
    assert parsed_four is not None
    if sha256_bytes(parsed_four) != FOUR_STREAM_SHA256:
        raise D3BError("factor archive four-class body hash drifted")

    quotient_path, quotient_result = decode_quotient(
        store, env, parsed_four, args.checkpoint_every, args.resume
    )
    lane_result = decode_lane_packets(
        store, env, quotient_path, packet_rows, args.checkpoint_every, args.resume
    )
    identities = {row["config"]: row for row in lane_result["rows"]}
    for row in encode_result["rows"]:
        row["receiver_identity"] = identities[row["config"]["name"]]
    winner = min(encode_result["rows"], key=lambda row: row["token_subsystem_bytes"])
    result = {
        "schema": "ddm_d3b_decode.v1",
        "complete": True,
        "axis": AXIS,
        "score_claim": False,
        "custody": custody,
        "quotient_receiver": quotient_result,
        "lane_receivers": lane_result,
        "rows": encode_result["rows"],
        "winner": winner,
        "verdict": (
            "CLEARS_127292" if winner["token_subsystem_bytes"] < DX2_TOKEN_SUBSYSTEM_BYTES
            else "REFUSED_AT_FORMULATION_SCOPE"
        ),
        "decode_identity_all_candidates": True,
    }
    atomic_json(store / "DECODE_RESULT.json", result)
    atomic_json(store / "RESULT.json", result)
    atomic_json(store / "DECODE_STAGE_COMPLETE.json", {
        "schema": "ddm_d3b_stage_complete.v1",
        "stage": "decode",
        "result": file_fact(store / "RESULT.json"),
        "decode_identity_all_candidates": True,
    })
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=("encode", "decode", "all"))
    parser.add_argument("--store", default=str(DEFAULT_STORE))
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.checkpoint_every < 0:
        raise SystemExit("--checkpoint-every must be nonnegative")
    stages = ("encode", "decode") if args.stage == "all" else (args.stage,)
    result: dict[str, Any] = {}
    for stage in stages:
        result = {"encode": stage_encode, "decode": stage_decode}[stage](args)
        progress(stage=stage, event="done", complete=result.get("complete"), axis=AXIS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
