"""Decode class planes with a learned prior, then render RGB and a pose carrier."""
import argparse
import ctypes
import math
import struct
import numpy as np
import bisect
from collections import OrderedDict
import torch
import torch.nn.functional
from torch import nn
from dataclasses import dataclass
from pathlib import Path
from torch.nn import functional as F
from types import MethodType
from collections import defaultdict
import lzma
import zipfile
try:
    import brotli
except ImportError:
    raise SystemExit('Brotli is required: install brotli in the Python environment.')
import hashlib
import sys

def load_range_library():
    """Use the optional local C loop, keeping Python as the reference fallback."""
    try:
        library = ctypes.CDLL(str(Path(__file__).with_name('range_decoder.so').resolve()))
        library.decode_group.argtypes = [
            ctypes.POINTER(ctypes.c_uint64), ctypes.c_char_p, ctypes.c_size_t,
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
            ctypes.c_size_t,
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
        ]
        library.decode_group.restype = ctypes.c_int
        return library
    except (OSError, AttributeError):
        print('Native range decoder unavailable; using the identical Python decoder.', file=sys.stderr)
        return None

_RANGE_LIBRARY = load_range_library()

class ArithmeticDecoder:
    """Five-symbol, 63-bit arithmetic decoder with a 31-bit frequency total."""
    TOTAL = 1 << 31
    QUARTER = 1 << 61
    HALF = 1 << 62
    TOP = (1 << 63) - 1

    def __init__(self, payload: bytes):
        if not payload:
            raise ValueError('empty arithmetic stream')
        self.payload = bytes(payload)
        self.bit_position = 0
        self.low, self.high, self.code = (0, self.TOP, 0)
        for _ in range(63):
            self.code = self.code << 1 | self._read_bit()

    def _read_bit(self):
        byte, shift = divmod(self.bit_position, 8)
        self.bit_position += 1
        return self.payload[byte] >> 7 - shift & 1 if byte < len(self.payload) else 0

    def decode(self, probabilities):
        """Decode one causal group; preserve the original float32 conversion."""
        rows = np.ascontiguousarray(probabilities, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != 5 or (not len(rows)):
            raise ValueError('probabilities must have shape [N, 5]')
        if _RANGE_LIBRARY is not None:
            state = (ctypes.c_uint64 * 4)(self.low, self.high, self.code, self.bit_position)
            result = np.empty(len(rows), dtype=np.int32)
            status = _RANGE_LIBRARY.decode_group(state, self.payload, len(self.payload), rows, len(rows), result)
            if status:
                raise ValueError(f'native arithmetic decoding failed ({status})')
            self.low, self.high, self.code, self.bit_position = map(int, state)
            return result
        wide = rows.astype(np.float64)
        if not np.isfinite(wide).all() or np.any(wide <= 0) or np.any(wide > 1.00002):
            raise ValueError('invalid arithmetic probability')
        totals = np.zeros(len(rows), dtype=np.float64)
        for lane in range(5):
            totals += wide[:, lane]
        if np.any(totals < 0.99998) or np.any(totals > 1.00002):
            raise ValueError('probabilities do not sum to one')
        frequencies = np.maximum((wide * self.TOTAL).astype(np.int64), 1)
        winners = rows.argmax(axis=1)
        frequencies[np.arange(len(rows)), winners] += self.TOTAL - frequencies.sum(axis=1)
        if np.any(frequencies <= 0) or np.any(frequencies >= self.TOTAL):
            raise ValueError('invalid arithmetic frequencies')
        result = np.empty(len(rows), dtype=np.int32)
        low, high, code = (self.low, self.high, self.code)
        for index, row in enumerate(frequencies.tolist()):
            if not low <= code <= high:
                raise ValueError('arithmetic state outside interval')
            width = high - low + 1
            scaled = ((code - low + 1) * self.TOTAL - 1) // width
            before, after = (0, 0)
            for symbol, frequency in enumerate(row):
                after += frequency
                if scaled < after:
                    break
                before = after
            else:
                raise ValueError('arithmetic symbol outside alphabet')
            lower, upper = (width * before >> 31, width * after >> 31)
            if upper <= lower:
                raise ValueError('empty arithmetic interval')
            high, low = (low + upper - 1, low + lower)
            while True:
                if high < self.HALF:
                    offset = 0
                elif low >= self.HALF:
                    offset = self.HALF
                elif low >= self.QUARTER and high < 3 * self.QUARTER:
                    offset = self.QUARTER
                else:
                    break
                low = low - offset << 1
                high = high - offset << 1 | 1
                code = code - offset << 1 | self._read_bit()
            result[index] = symbol
        self.low, self.high, self.code = (low, high, code)
        return result
HEIGHT, WIDTH, CLASSES, SLOT_WIDTH = (384, 512, 5, 64)
SLOTS, FIXED_ONE = (WIDTH // SLOT_WIDTH, 1 << 16)
GEOMETRY_FORMAT = struct.Struct('<BHH8B6B')
ROW, COLUMN = np.indices((HEIGHT, WIDTH), dtype=np.int64)
GROUP_INDEX = COLUMN % SLOT_WIDTH + 2 * (ROW % SLOT_WIDTH)
GROUP_POSITIONS = tuple((np.flatnonzero(GROUP_INDEX.ravel() == group) for group in range(190)))
MOMENTS = (np.ones_like(ROW), ROW, ROW * ROW, COLUMN, COLUMN * ROW, COLUMN * COLUMN)

def parse_geometry_config(payload):
    """Validate the sealed receiver's exact supported counted parameter domain."""
    if len(payload) != GEOMETRY_FORMAT.size:
        raise ValueError('geometry configuration must contain 19 bytes')
    values = GEOMETRY_FORMAT.unpack(payload)
    if not (values[0] < CLASSES and 0 <= values[1] < values[2] <= HEIGHT):
        raise ValueError('invalid geometry class or row band')
    if not (1 <= values[3] <= 16 and 1 <= values[4] <= 4 and (1 <= values[5] <= 255) and (1 <= values[6] <= 255) and (1 <= values[7] <= 2) and (1 <= values[8] <= 255) and (1 <= values[9] <= 255) and (values[10] < 64)):
        raise ValueError('geometry configuration exceeds the supported domain')
    if any((a >= b for a, b in zip(values[11:-1], values[12:]))):
        raise ValueError('geometry distance edges must increase')
    return values

def preceding_window(values, window):
    """Sum the preceding rows, excluding the current row, exactly as the C loop."""
    prefix = np.concatenate((np.zeros_like(values[:, :1]), np.cumsum(values, axis=1)), axis=1)
    rows = np.arange(HEIGHT)
    return prefix[:, rows] - prefix[:, np.maximum(0, rows - window)]

class CausalGeometry:
    """Integer local line fits from completed groups and the preceding frame."""

    def __init__(self, previous, config):
        self.config = parse_geometry_config(config)
        if previous is not None:
            previous = np.asarray(previous)
            if previous.shape != (HEIGHT, WIDTH) or previous.dtype.kind not in 'iu' or np.any(previous >= CLASSES) or np.any(previous < 0):
                raise ValueError('previous must be a complete class plane')
        prior = np.zeros((HEIGHT, WIDTH), bool) if previous is None else previous == self.config[0]
        self.old = np.stack([(value * prior).reshape(HEIGHT, SLOTS, SLOT_WIDTH).sum(axis=2) for value in MOMENTS])
        self.current = np.zeros_like(self.old)
        prior_slots = prior.reshape(HEIGHT, SLOTS, SLOT_WIDTH)
        x = COLUMN.reshape(HEIGHT, SLOTS, SLOT_WIDTH)
        self.old_low = np.where(prior_slots, x, WIDTH).min(axis=2)
        self.old_high = np.where(prior_slots, x, -1).max(axis=2)
        preceding = np.concatenate((np.zeros((HEIGHT, SLOTS, 1), bool), prior_slots[:, :, :-1]), axis=2)
        self.ambiguous = (prior_slots & ~preceding).sum(axis=2) > 1
        self.low = np.full((HEIGHT, SLOTS), WIDTH, dtype=np.int64)
        self.high = np.full((HEIGHT, SLOTS), -1, dtype=np.int64)
        self.other = np.zeros((HEIGHT, SLOTS), dtype=np.uint64)
        self.plane = np.full((HEIGHT, WIDTH), CLASSES, dtype=np.uint8)
        self.group, self.pending = (0, False)

    def _positions(self, positions):
        positions = np.asarray(positions)
        if positions.dtype.kind not in 'iu' or self.group >= len(GROUP_POSITIONS) or (not np.array_equal(positions, GROUP_POSITIONS[self.group])):
            raise ValueError('expected the complete next causal group')
        return positions.astype(np.int64, copy=False)

    def contexts(self, positions):
        positions = self._positions(positions)
        if self.pending:
            raise ValueError('observe must follow the preceding contexts call')
        cfg = self.config
        moments = preceding_window(self.current * cfg[4] + self.old, cfg[3])
        one = np.uint64(1)
        upper = (one << (self.high % SLOT_WIDTH).astype(np.uint64)) - one
        lower = (one << (self.low % SLOT_WIDTH + 1).astype(np.uint64)) - one
        between = np.where(self.high > self.low + 1, upper ^ lower, np.uint64(0))
        gap = self.other & between != 0
        wide = np.maximum(self.high, self.old_high) - np.minimum(self.low, self.old_low) > cfg[10]
        bad = preceding_window((gap | wide | self.ambiguous)[None].astype(np.int64), cfg[3])[0] != 0
        count, sy, syy, sx, sxy, sxx = moments
        determinant, slope = (count * syy - sy * sy, count * sxy - sy * sx)
        valid = (count >= cfg[5] * cfg[4]) & (determinant > 0) & ~bad & (np.abs(slope) <= cfg[7] * determinant)
        center, radius = (np.zeros_like(count), np.zeros_like(count))
        for y, slot in zip(*np.nonzero(valid)):
            c, sum_y, sum_yy, sum_x, sum_xy, sum_xx = map(int, moments[:, y, slot])
            d, b = (c * sum_yy - sum_y * sum_y, c * sum_xy - sum_y * sum_x)
            residual = d * (c * sum_xx - sum_x * sum_x) - b * b
            denominator = c * c * d
            if residual > cfg[6] * denominator:
                valid[y, slot] = False
                continue
            center[y, slot] = (sum_x * d + b * (int(y) * c - sum_y)) * FIXED_ONE // (c * d)
            radius[y, slot] = max(FIXED_ONE // cfg[8], math.isqrt(max(residual, 0) * cfg[9] * FIXED_ONE * FIXED_ONE // denominator))
        y, x = (positions // WIDTH, positions % WIDTH)
        distance = np.minimum(np.abs(x[:, None] * FIXED_ONE - center[y] + radius[y]), np.abs(x[:, None] * FIXED_ONE - center[y] - radius[y]))
        quotient, remainder = np.divmod(distance, FIXED_ONE)
        rounded = quotient + ((remainder > FIXED_ONE // 2) | (remainder == FIXED_ONE // 2) & (quotient & 1 != 0))
        best = np.where(valid[y], rounded, np.iinfo(np.int64).max).min(axis=1)
        result = np.searchsorted(cfg[11:], best, side='left').astype(np.uint8)
        result[~valid[y].any(axis=1)] = 7
        result[(y < cfg[1]) | (y >= cfg[2])] = 8
        self.pending = True
        return result

    def observe(self, positions, symbols):
        positions, symbols = (self._positions(positions), np.asarray(symbols))
        if not self.pending or symbols.shape != positions.shape or symbols.dtype.kind not in 'iu' or np.any(symbols < 0) or np.any(symbols >= CLASSES):
            raise ValueError('expected the decoded symbols of the pending group')
        self.plane.ravel()[positions] = symbols
        chosen = positions[symbols == self.config[0]]
        cells = chosen // WIDTH * SLOTS + chosen % WIDTH // SLOT_WIDTH
        for index, values in enumerate(MOMENTS):
            np.add.at(self.current[index].ravel(), cells, values.ravel()[chosen])
        np.minimum.at(self.low.ravel(), cells, chosen % WIDTH)
        np.maximum.at(self.high.ravel(), cells, chosen % WIDTH)
        other = positions[symbols != self.config[0]]
        cells = other // WIDTH * SLOTS + other % WIDTH // SLOT_WIDTH
        np.bitwise_or.at(self.other.ravel(), cells, np.uint64(1) << (other % SLOT_WIDTH).astype(np.uint64))
        self.group += 1
        self.pending = False

def load_geometry_library():
    """Bind the optional portable integer geometry loop."""
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name('geometry.so').resolve()))
        i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS')
        u8 = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags='C_CONTIGUOUS')
        lib.geometry_new.argtypes = [i64, ctypes.c_void_p]
        lib.geometry_new.restype = ctypes.c_void_p
        lib.geometry_free.argtypes = [ctypes.c_void_p]
        lib.geometry_free.restype = None
        lib.geometry_contexts.argtypes = [ctypes.c_void_p, ctypes.c_int, i64, u8]
        lib.geometry_contexts.restype = None
        lib.geometry_observe.argtypes = [ctypes.c_void_p, ctypes.c_int, i64, i64]
        lib.geometry_observe.restype = None
        return lib
    except (OSError, AttributeError):
        print('Native geometry unavailable; using the identical Python geometry.', file=sys.stderr)
        return None

_GEOMETRY_LIBRARY = load_geometry_library()

class NativeGeometry(CausalGeometry):
    """Keep causal validation in Python and perform moment geometry in C11."""
    def __init__(self, previous, config):
        self.handle = None
        self.config = parse_geometry_config(config)
        if previous is not None:
            previous = np.asarray(previous)
            if previous.shape != (HEIGHT, WIDTH) or previous.dtype.kind not in 'iu' or np.any(previous >= CLASSES) or np.any(previous < 0):
                raise ValueError('previous must be a complete class plane')
            previous = np.ascontiguousarray(previous, dtype=np.uint8)
        self.handle = _GEOMETRY_LIBRARY.geometry_new(np.array(self.config, dtype=np.int64), None if previous is None else previous.ctypes.data)
        if not self.handle:
            raise MemoryError('native geometry allocation failed')
        self.plane = np.full((HEIGHT, WIDTH), CLASSES, dtype=np.uint8)
        self.group, self.pending = 0, False

    def contexts(self, positions):
        positions = np.ascontiguousarray(self._positions(positions))
        if self.pending:
            raise ValueError('observe must follow the preceding contexts call')
        result = np.empty(len(positions), dtype=np.uint8)
        _GEOMETRY_LIBRARY.geometry_contexts(self.handle, len(positions), positions, result)
        self.pending = True
        return result

    def observe(self, positions, symbols):
        positions, symbols = self._positions(positions), np.asarray(symbols)
        if not self.pending or symbols.shape != positions.shape or symbols.dtype.kind not in 'iu' or np.any(symbols < 0) or np.any(symbols >= CLASSES):
            raise ValueError('expected the decoded symbols of the pending group')
        _GEOMETRY_LIBRARY.geometry_observe(self.handle, len(positions), np.ascontiguousarray(positions), np.ascontiguousarray(symbols, dtype=np.int64))
        self.plane.ravel()[positions] = symbols
        self.group += 1
        self.pending = False

    def __del__(self):
        if self.handle:
            _GEOMETRY_LIBRARY.geometry_free(self.handle)
            self.handle = None

class model_bits_RangeDecoder:
    __slots__ = ('buffer', 'code', 'position', 'range')

    def __init__(self, payload):
        self.buffer = payload
        self.position = 0
        self.range = 4294967295
        self.code = 0
        for _ in range(4):
            self.code = (self.code << 8 | self._byte()) & 4294967295

    def _byte(self):
        if self.position < len(self.buffer):
            value = self.buffer[self.position]
            self.position += 1
            return value

    def decode_frequency(self, total):
        unit = self.range // total
        value = self.code // unit
        return total - 1 if value >= total else value

    def update(self, cumulative_low, frequency, total):
        unit = self.range // total
        self.code -= unit * cumulative_low
        self.range = unit * frequency
        while self.range < 1 << 24:
            self.range <<= 8
            self.code = (self.code << 8 | self._byte()) & 4294967295

def model_bits_pack_signed_codes(values, bits):
    unsigned = (np.asarray(values, dtype=np.int32) & (1 << bits) - 1).astype(np.int32)
    stream = np.zeros((unsigned.size, bits), dtype=np.uint8)
    for index in range(bits):
        stream[:, index] = unsigned >> index & 1
    flat = stream.reshape(-1)
    pad = -flat.size % 8
    return np.packbits(flat, bitorder='little').tobytes()
model_bits_ROW_PRUNE_NAMES = frozenset({'blocks.1.film.weight', 'blocks.2.film.weight', 'blocks.3.film.weight'})

def model_bits_walk_renderer_rows(read, template, header):
    _version, _mode, keep_percent, _reserved = header
    names = [name for name, value in template.items() if value.ndim >= 2]
    plan: list[dict] = []
    depth_bytes = (len(names) + 1) // 2
    depth_blob = read('depth_table', depth_bytes)
    packed = np.frombuffer(depth_blob, dtype=np.uint8)
    values = np.empty(depth_bytes * 2, dtype=np.uint8)
    values[0::2] = packed & 15
    values[1::2] = packed >> 4
    depths = {name: int(value) for name, value in zip(names, values[:len(names)].tolist(), strict=True)}
    plan.append({'kind': 'depth_table', 'length': depth_bytes, 'blob': depth_blob})
    group = 0
    for name, value in template.items():
        numel = int(value.numel())
        if value.ndim < 2:
            plan.append({'kind': 'meta', 'length': numel * 2, 'blob': read('fp16_tensor', numel * 2)})
            continue
        bits = depths[name]
        if name not in model_bits_ROW_PRUNE_NAMES:
            scale_count = int(value.shape[-1] if name.endswith('embed.weight') else value.shape[0])
            plan.append({'kind': 'meta', 'length': scale_count * 2, 'blob': read('fp16_scales', scale_count * 2)})
            length = (numel * bits + 7) // 8
            plan.append({'kind': 'codes', 'length': length, 'group': group, 'bits': bits, 'count': numel, 'blob': read('codes', length)})
            group += 1
            continue
        rows = int(value.shape[0])
        mask_bytes = (rows + 7) // 8
        mask_blob = read('prune_mask', mask_bytes)
        plan.append({'kind': 'meta', 'length': mask_bytes, 'blob': mask_blob})
        selected = np.unpackbits(np.frombuffer(mask_blob, dtype=np.uint8), bitorder='little')[:rows]
        keep = int(selected.sum())
        expected = max(1, round(rows * keep_percent / 100.0))
        plan.append({'kind': 'meta', 'length': keep * 2, 'blob': read('fp16_scales', keep * 2)})
        count = keep * (numel // rows)
        length = (count * bits + 7) // 8
        plan.append({'kind': 'codes', 'length': length, 'group': group, 'bits': bits, 'count': count, 'blob': read('codes', length)})
        group += 1
    return plan
adaptive_FAMILY_PREV_BITLEN = 3
adaptive_FAMILY_EXACT_PREV = 4
adaptive_PROBABILITY_ONE = 4096
adaptive_PROBABILITY_INITIAL = 2048
adaptive_EXPERT_COUNT = 8

class adaptive_ModelCodecError(ValueError):
    pass

class adaptive_RangeDecoder:
    __slots__ = ('buffer', 'code', 'position', 'range')

    def __init__(self, payload):
        self.buffer = payload
        self.position = 0
        self.range = 4294967295
        self.code = 0
        for _ in range(4):
            self.code = (self.code << 8 | self._byte()) & 4294967295

    def _byte(self):
        if self.position < len(self.buffer):
            value = self.buffer[self.position]
            self.position += 1
            return value

    def decode_bit(self, probability_zero):
        probability_zero = int(probability_zero)
        unit = self.range // adaptive_PROBABILITY_ONE
        scaled = min(adaptive_PROBABILITY_ONE - 1, self.code // unit)
        bit = int(scaled >= probability_zero)
        if bit:
            low, frequency = (probability_zero, adaptive_PROBABILITY_ONE - probability_zero)
        else:
            low, frequency = (0, probability_zero)
        self.code -= unit * low
        self.range = unit * frequency
        while self.range < 1 << 24:
            self.range <<= 8
            self.code = (self.code << 8 | self._byte()) & 4294967295
        return bit

def adaptive_updated_probability(probability_zero, bit, shift=5):
    if bit:
        return probability_zero - (probability_zero >> shift)
    return probability_zero + (adaptive_PROBABILITY_ONE - probability_zero >> shift)

def adaptive_depths(prefix, row_count):
    depth_bytes = (row_count + 1) // 2
    packed = np.frombuffer(prefix[4:], dtype=np.uint8)
    values = np.empty(depth_bytes * 2, dtype=np.uint8)
    values[0::2] = packed & 15
    values[1::2] = packed >> 4
    result = values[:row_count].astype(np.int64)
    return result

def adaptive_pack_rows(rows, depths):
    chunks: list[np.ndarray] = []
    for values, depth in zip(rows, depths.tolist(), strict=True):
        depth = int(depth)
        if depth == 0:
            continue
        unsigned = np.asarray(values, dtype=np.int32) & (1 << depth) - 1
        block = np.empty((unsigned.size, depth), dtype=np.uint8)
        for index in range(depth):
            block[:, index] = unsigned >> index & 1
        chunks.append(block.reshape(-1))
    bits = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint8)
    if -bits.size % 8:
        bits = np.concatenate([bits, np.zeros(-bits.size % 8, dtype=np.uint8)])
    return np.packbits(bits, bitorder='little').tobytes()

def adaptive_context(family, previous, depth):
    if family == adaptive_FAMILY_PREV_BITLEN:
        if previous == 0:
            return 0
        magnitude_bits = abs(int(previous)).bit_length()
        return magnitude_bits if previous > 0 else depth + magnitude_bits
    if family == adaptive_FAMILY_EXACT_PREV:
        return int(previous) & (1 << depth) - 1
    raise adaptive_ModelCodecError(f'unknown semi-static family {family}')

def adaptive_log2_ratio_fixed(numerator, denominator, fractional_bits=12):
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator < denominator << exponent:
            exponent -= 1
    elif numerator << -exponent < denominator:
        exponent -= 1
    if exponent >= 0:
        n, d = (numerator, denominator << exponent)
    else:
        n, d = (numerator << -exponent, denominator)
    fraction = 0
    for _ in range(fractional_bits):
        n *= n
        d *= d
        fraction <<= 1
        if n >= 2 * d:
            n //= 2
            fraction |= 1
    return exponent * (1 << fractional_bits) + fraction
adaptive_STRETCH = tuple((0 if probability == adaptive_PROBABILITY_INITIAL else adaptive_log2_ratio_fixed(probability, adaptive_PROBABILITY_ONE - probability) for probability in range(1, adaptive_PROBABILITY_ONE)))

def adaptive_stretch(probability_zero):
    return adaptive_STRETCH[probability_zero - 1]

def adaptive_squash(stretched):
    index = bisect.bisect_left(adaptive_STRETCH, int(stretched))
    if index >= len(adaptive_STRETCH):
        return adaptive_PROBABILITY_ONE - 1
    before, after = (adaptive_STRETCH[index - 1], adaptive_STRETCH[index])
    return index if stretched - before <= after - stretched else index + 1

def adaptive_round_div_signed(value, denominator):
    if value >= 0:
        return (value + denominator // 2) // denominator
    return -((-value + denominator // 2) // denominator)

class adaptive_AdaptiveExperts:
    __slots__ = ('banks',)

    def __init__(self):
        self.banks: list[dict[object, int]] = [{} for _ in range(adaptive_EXPERT_COUNT)]

    def keys(self, depth, node, bit_position, previous):
        unknown = previous is None
        previous_value = 0 if previous is None else int(previous)
        zero_context = 2 if unknown else int(previous_value != 0)
        sign_context = 3 if unknown else 0 if previous_value < 0 else 1 if previous_value == 0 else 2
        bitlen_context = 2 * depth + 1 if unknown else adaptive_context(adaptive_FAMILY_PREV_BITLEN, previous_value, depth)
        exact_context = 1 << depth if unknown else adaptive_context(adaptive_FAMILY_EXACT_PREV, previous_value, depth)
        return [(depth, node), (depth, zero_context, node), (depth, sign_context, node), (depth, bitlen_context, node), (depth, exact_context, node), (bit_position, node), bit_position, 0]

    def predictions(self, keys):
        return [bank.get(key, adaptive_PROBABILITY_INITIAL) for bank, key in zip(self.banks, keys, strict=True)]

    def update(self, keys, bit):
        for bank, key in zip(self.banks, keys, strict=True):
            probability = bank.get(key, adaptive_PROBABILITY_INITIAL)
            bank[key] = adaptive_updated_probability(probability, bit)
weight_mixer_HEADER = struct.Struct('<4sBBHI')

def weight_mixer_scale_buckets(blob):
    scales = np.frombuffer(blob, dtype='<u2').astype(np.int64)
    ordered = np.sort(scales)
    buckets = np.zeros(len(scales), dtype=np.int16)
    for quartile in (1, 2, 3):
        position, remainder = divmod((len(scales) - 1) * quartile, 4)
        lo, hi = (int(ordered[position]), int(ordered[min(position + 1, len(scales) - 1)]))
        buckets += 4 * scales > (4 - remainder) * lo + remainder * hi
    return buckets.tolist()

def weight_mixer_plan_metadata(metadata, template):
    cursor = 10

    def read(kind, length):
        nonlocal cursor
        if kind == 'codes':
            return bytes.fromhex('')
        value = metadata[cursor:cursor + length]
        cursor += length
        return value
    plan = model_bits_walk_renderer_rows(read, template, tuple(metadata[4:8]))
    names = [(name, value) for name, value in template.items() if value.ndim >= 2]
    descriptors = []
    for index, item in enumerate(plan):
        if item['kind'] != 'codes':
            continue
        name, tensor = names[len(descriptors)]
        shape = tuple(tensor.shape)
        kind = 0 if name == 'token_embed.weight' else 1 if name == 'frame_embed.weight' else 2 if name == 'coord_mix.weight' else 3 if '.dw.' in name else 4 if '.pw.' in name else 5 if '.film.' in name else 6
        descriptors.append(dict(name=name, shape=shape, count=item['count'], bits=item['bits'], cols=math.prod(shape[1:]), scales=weight_mixer_scale_buckets(plan[index - 1]['blob']), embedding=name.endswith('embed.weight'), kind=kind, block=int(name.split('.')[1]) if name.startswith('blocks.') else 4))
        if name in model_bits_ROW_PRUNE_NAMES:
            descriptors[-1]['selected_rows'] = np.flatnonzero(np.unpackbits(np.frombuffer(plan[index - 2]['blob'], dtype=np.uint8), bitorder='little')[:shape[0]]).tolist()
    return (plan, descriptors)

def weight_mixer_bucket(value):
    return 3 if value is None else 0 if value < 0 else 1 if value == 0 else 2

class weight_mixer_Predictors:

    def __init__(self):
        self.banks = [{} for _ in range(23)]
        self.completed = {}

    def keys(self, group, desc, pos, node, current):
        row, col = divmod(pos, desc['cols'])
        scale = desc['scales'][col if desc['embedding'] else row]
        quartile = min(3, 4 * col // desc['cols'])
        sibling = current[pos - desc['cols']] if row else None
        if 'selected_rows' in desc:
            original_row = desc['selected_rows'][row]
            sibling = current[pos - desc['cols']] if row and desc['selected_rows'][row - 1] == original_row - 1 else 0 if original_row else None
        left = current[-1] if col else None
        dw = None
        if desc['kind'] == 4:
            dw_codes = self.completed[desc['name'].replace('.pw.', '.dw.')]
            dw = dw_codes[col * 9 + 4]
        g, d = ((group, node), (desc['bits'], node))
        return [g, g, g, g, g, (*g, scale), (*g, scale), (*d, scale), (*d, scale), (*g, quartile), (*d, quartile), (*g, sibling), (*g, weight_mixer_bucket(sibling)), (*d, sibling), (*d, weight_mixer_bucket(sibling)), (*g, weight_mixer_bucket(left)), (*d, weight_mixer_bucket(left)), (*g, dw), (*d, weight_mixer_bucket(dw)), (desc['block'], desc['kind'], *d), (desc['kind'], *d), d, (*g, scale)]

    def predict(self, keys):
        output = []
        for j, key in enumerate(keys):
            if j in (4, 21, 22):
                zero, n = self.banks[j].get(key, (0, 0))
                p = min(4095, max(1, (2 * zero + 1) * 4096 // (2 * n + 2)))
            else:
                p = self.banks[j].get(key, 2048)
            output.append(adaptive_stretch(p))
        return output + [4096]

    def update(self, keys, bit):
        for j, key in enumerate(keys):
            if j in (4, 21, 22):
                zero, n = self.banks[j].get(key, (0, 0))
                self.banks[j][key] = (zero + int(bit == 0), n + 1)
            else:
                shift = {1: 4, 2: 5, 3: 7, 6: 4, 8: 4}.get(j, 6)
                p = self.banks[j].get(key, 2048)
                self.banks[j][key] = adaptive_updated_probability(p, bit, shift)

def weight_mixer_walk(descriptors, weights, *, payload):
    decoder = model_bits_RangeDecoder(payload)
    state, groups = (weight_mixer_Predictors(), [])
    for group, desc in enumerate(descriptors):
        current = []
        for pos in range(desc['count']):
            node, value = (1, 0)
            for shift in reversed(range(desc['bits'])):
                keys = state.keys(group, desc, pos, node, current)
                x = state.predict(keys)
                p = adaptive_squash(adaptive_round_div_signed(sum((int(w) * v for w, v in zip(weights, x, strict=True))), 32))
                bit = int(decoder.decode_frequency(4096) >= p)
                decoder.update(p if bit else 0, 4096 - p if bit else p, 4096)
                state.update(keys, bit)
                value = value * 2 + bit
                node = node * 2 + bit
            if value >= 1 << desc['bits'] - 1:
                value -= 1 << desc['bits']
            current.append(value)
        state.completed[desc['name']] = current
        groups.append(np.asarray(current, dtype=np.int32))
    return groups

def weight_mixer_restore_semantic(rider, template):
    magic, version, count, reserved, length = weight_mixer_HEADER.unpack_from(rider)
    end = len(rider) - length - 24
    metadata = rider[weight_mixer_HEADER.size:end]
    plan, descriptors = weight_mixer_plan_metadata(metadata, template)
    weights = np.frombuffer(rider[end:end + 24], dtype=np.int8)
    groups = weight_mixer_walk(descriptors, weights, payload=rider[end + 24:])
    parts, cursor = ([metadata[:10]], 0)
    for item in plan:
        if item['kind'] == 'codes':
            parts.append(model_bits_pack_signed_codes(groups[cursor], item['bits']))
            cursor += 1
        else:
            parts.append(item['blob'])
    return bytes.fromhex('').join(parts)
weights_RENDERER_ROWS_MAGIC = bytes.fromhex('534d3352')
weights_RENDERER_ROWS_ROW_PRUNE_MIXED_MODE = 6
weights_ROW_PRUNE_NAMES = frozenset({'blocks.1.film.weight', 'blocks.2.film.weight', 'blocks.3.film.weight'})

class weights_WeightFormatError(ValueError):
    pass

def weights_take(blob, count, label):
    return (blob[:count], blob[count:])

def weights_quantized_names(template):
    return [name for name, value in template.items() if value.ndim >= 2]

def weights_scale_count(name, value):
    return int(value.shape[-1] if name.endswith('embed.weight') else value.shape[0])

def weights_unpack_signed_bits(blob, count, bits):
    byte_count = (count * bits + 7) // 8
    packed_view, remaining = weights_take(blob, byte_count, 'signed code stream')
    packed = np.frombuffer(packed_view, dtype=np.uint8)
    bitstream = np.unpackbits(packed, bitorder='little')[:count * bits]
    bitstream = bitstream.reshape(count, bits).astype(np.int16, copy=False)
    shifts = (1 << np.arange(bits, dtype=np.int16))[None]
    unsigned = (bitstream * shifts).sum(axis=1, dtype=np.int16)
    sign = 1 << bits - 1
    values = np.where(unsigned >= sign, unsigned - (1 << bits), unsigned)
    return (torch.from_numpy(values.astype(np.int8, copy=False)), remaining)

def weights_decode_quantized(name, template, blob, bits):
    scale_count = weights_scale_count(name, template)
    scale_view, remaining = weights_take(blob, scale_count * 2, f'fp16 scales for {name}')
    scales = np.frombuffer(scale_view, dtype='<f2').copy()
    codes, remaining = weights_unpack_signed_bits(remaining, template.numel(), bits)
    scale_shape = [1] * template.ndim
    scale_shape[-1 if name.endswith('embed.weight') else 0] = scale_count
    restored = codes.reshape(template.shape).float()
    restored *= torch.from_numpy(scales).float().reshape(scale_shape)
    return (restored, remaining)

def weights_decode_fp16(name, template, blob):
    payload, remaining = weights_take(blob, template.numel() * 2, f'fp16 tensor {name}')
    array = np.frombuffer(payload, dtype='<f2').copy()
    return (torch.from_numpy(array.reshape(template.shape)).float(), remaining)

def weights_decode_depth_nibbles(blob, count, label):
    depth_bytes = (count + 1) // 2
    depth_view, remaining = weights_take(blob, depth_bytes, f'{label} depth allocation')
    packed = np.frombuffer(depth_view, dtype=np.uint8)
    depths_array = np.empty(depth_bytes * 2, dtype=np.uint8)
    depths_array[0::2] = packed & 15
    depths_array[1::2] = packed >> 4
    depths = depths_array[:count].astype(int).tolist()
    return (depths, remaining)

def weights_decode_row_prune_mixed(blob, template):
    version, mode, keep_percent, reserved = blob[4:8]
    remaining = memoryview(blob)[8:]
    mask_view, remaining = weights_take(remaining, 2, 'RENDERER_ROWS row-prune selection mask')
    mask = struct.unpack_from('<H', mask_view)[0]
    names = weights_quantized_names(template)
    depths, remaining = weights_decode_depth_nibbles(remaining, len(names), 'RENDERER_ROWS')
    allocation = dict(zip(names, depths, strict=True))
    restored: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in template.items():
        if value.ndim < 2:
            restored[name], remaining = weights_decode_fp16(name, value, remaining)
        elif name not in weights_ROW_PRUNE_NAMES:
            restored[name], remaining = weights_decode_quantized(name, value, remaining, allocation[name])
        else:
            rows = int(value.shape[0])
            mask_view, remaining = weights_take(remaining, (rows + 7) // 8, f'RENDERER_ROWS selected rows for {name}')
            selected = np.unpackbits(np.frombuffer(mask_view, dtype=np.uint8), bitorder='little')[:rows].astype(bool)
            expected_keep = max(1, round(rows * keep_percent / 100.0))
            columns = value.numel() // rows
            compact_template = torch.empty((expected_keep, columns), dtype=torch.float32)
            compact, remaining = weights_decode_quantized('pruned.rows', compact_template, remaining, allocation[name])
            dense = torch.zeros((rows, columns), dtype=torch.float32)
            dense[torch.from_numpy(selected)] = compact
            restored[name] = dense.reshape(value.shape)
    return restored

def weights_unpack_variant_semantic_or_none(blob, template):
    if blob.startswith(weights_RENDERER_ROWS_MAGIC):
        if blob[5] == weights_RENDERER_ROWS_ROW_PRUNE_MIXED_MODE:
            return weights_decode_row_prune_mixed(blob, template)
        raise weights_WeightFormatError(f'unsupported RENDERER_ROWS mode {blob[5]}')

def prior_ste_round(value):
    return value + (value.round() - value).detach()

def prior_integer_activation(value, mode):
    if mode == 'relu':
        return torch.nn.functional.relu(value)
    raise ValueError(f'unsupported integer activation: {mode}')

def prior_patch_group_mask(kernel, delta, type_):
    mask = torch.zeros(kernel, kernel, dtype=torch.float32)
    center = (kernel - 1) // 2
    for row in range(kernel):
        for column in range(kernel):
            offset = column - center + delta * (row - center)
            if offset < 0 or (type_ == 'B' and offset == 0):
                mask[row, column] = 1.0
    return mask

class prior_IntegerConv2d(nn.Module):

    def __init__(self, c_in, c_out, kernel, *, padding=0, dilation=1, groups=1, mask=None, weight_bound=127, use_weight_scales=False, exponent_min=-6):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(c_out, c_in // groups, kernel, kernel))
        self.bias = nn.Parameter(torch.zeros(c_out))
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight_bound = weight_bound
        self.exponent_min = exponent_min
        nn.init.normal_(self.weight, mean=0.0, std=1.5)
        if use_weight_scales:
            self.weight.data.mul_(8)
            self.exponent = nn.Parameter(torch.full((c_out,), -3.0))
        if mask is None:
            mask = torch.ones(kernel, kernel)
        self.register_buffer('mask', mask.view(1, 1, kernel, kernel), persistent=False)

    def codes(self):
        weight = prior_ste_round(self.weight.clamp(-self.weight_bound, self.weight_bound)) * self.mask
        bias = prior_ste_round(self.bias.clamp(-32768, 32767))
        exponent = None
        if hasattr(self, 'exponent'):
            exponent = prior_ste_round(self.exponent.clamp(self.exponent_min, 0))
        return (weight, bias, exponent)

    def forward(self, value):
        weight, bias, exponent = self.codes()
        result = torch.nn.functional.conv2d(value, weight, None if exponent is not None else bias, padding=self.padding, dilation=self.dilation, groups=self.groups)
        if exponent is not None:
            result = result * torch.pow(2.0, exponent).view(1, -1, 1, 1)
            result = result + bias.view(1, -1, 1, 1)
        return result

class prior_IntegerLinear(nn.Module):

    def __init__(self, c_in, c_out, weight_bound=127, use_weight_scales=False, exponent_min=-6):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(c_out, c_in))
        self.bias = nn.Parameter(torch.zeros(c_out))
        self.weight_bound = weight_bound
        self.exponent_min = exponent_min
        nn.init.normal_(self.weight, mean=0.0, std=1.5)
        if use_weight_scales:
            self.weight.data.mul_(8)
            self.exponent = nn.Parameter(torch.full((c_out,), -3.0))

    def codes(self):
        weight = prior_ste_round(self.weight.clamp(-self.weight_bound, self.weight_bound))
        bias = prior_ste_round(self.bias.clamp(-32768, 32767))
        exponent = None
        if hasattr(self, 'exponent'):
            exponent = prior_ste_round(self.exponent.clamp(self.exponent_min, 0))
        return (weight, bias, exponent)

    def forward(self, value):
        weight, bias, exponent = self.codes()
        result = torch.nn.functional.linear(value, weight, None if exponent is not None else bias)
        if exponent is not None:
            result = result * torch.pow(2.0, exponent).view(1, -1)
            result = result + bias.view(1, -1)
        return result

class prior_IntegerPrior(nn.Module):

    def __init__(self, num_pairs=600, num_classes=5, patch=32, delta=2, channels=64, frame_dim=8, norm_mode='none', activation='relu', use_frame_scale=False, weight_bound=127, activation_bound=127, use_weight_scales=False, weight_exponent_min=-6, use_spm=False, use_norm_gates=False):
        super().__init__()
        self.num_pairs = num_pairs
        self.num_classes = num_classes
        self.P = patch
        self.delta = delta
        self.ch = channels
        self.norm_mode = norm_mode
        self.activation = activation
        self.use_frame_scale = use_frame_scale
        self.weight_bound = weight_bound
        self.activation_bound = activation_bound
        self.use_weight_scales = use_weight_scales
        self.use_spm = use_spm
        self.use_norm_gates = use_norm_gates
        self.frame_embed = nn.Embedding(num_pairs, frame_dim)
        nn.init.normal_(self.frame_embed.weight, mean=0.0, std=2.0)
        linear_kwargs = {'weight_bound': weight_bound, 'use_weight_scales': use_weight_scales, 'exponent_min': weight_exponent_min}
        conv_kwargs = {'weight_bound': weight_bound, 'use_weight_scales': use_weight_scales, 'exponent_min': weight_exponent_min}
        self.frame_shift = prior_IntegerLinear(frame_dim, channels, **linear_kwargs)
        if use_frame_scale:
            self.frame_scale = prior_IntegerLinear(frame_dim, channels, **linear_kwargs)
            nn.init.zeros_(self.frame_scale.weight)
            nn.init.zeros_(self.frame_scale.bias)
        self.conv_a = prior_IntegerConv2d(num_classes + 2, channels, 7, padding=3, mask=prior_patch_group_mask(7, delta, 'A'), **conv_kwargs)
        self.conv_b1 = prior_IntegerConv2d(channels, channels, 5, padding=4, dilation=2, groups=channels, mask=prior_patch_group_mask(5, delta, 'B'), **conv_kwargs)
        self.conv_b2 = prior_IntegerConv2d(channels, channels, 3, padding=4, dilation=4, groups=channels, mask=prior_patch_group_mask(3, delta, 'B'), **conv_kwargs)
        self.conv_past = prior_IntegerConv2d(num_classes, channels, 3, padding=1, **conv_kwargs)
        if use_spm:
            self.spm_dw = prior_IntegerConv2d(channels, channels, 3, padding=1, groups=channels, **conv_kwargs)
            self.spm_pw = prior_IntegerConv2d(channels, channels, 1, **conv_kwargs)
            nn.init.zeros_(self.spm_pw.weight)
            nn.init.zeros_(self.spm_pw.bias)
        self.head = prior_IntegerConv2d(channels, num_classes, 1, **conv_kwargs)
        self.register_buffer('_coord_cache', torch.zeros(0), persistent=False)

    def frame_codes(self):
        return prior_ste_round(self.frame_embed.weight.clamp(-127, 127))

    def _to_patches(self, value):
        batch, channels, height, width = value.shape
        patch_rows, patch_cols = (height // self.P, width // self.P)
        value = value.view(batch, channels, patch_rows, self.P, patch_cols, self.P).permute(0, 2, 4, 1, 3, 5).contiguous()
        return value.view(batch * patch_rows * patch_cols, channels, self.P, self.P)

    def prepare_frame_context(self, idx, previous_raw):
        batch, height, width = previous_raw.shape
        patch_count = height // self.P * (width // self.P)
        embedding = self.frame_codes()[idx]
        shift = prior_requantize(self.frame_shift(embedding), 1, -self.activation_bound, self.activation_bound)
        shift = shift.view(batch, 1, self.ch, 1, 1).expand(batch, patch_count, self.ch, 1, 1).reshape(batch * patch_count, self.ch, 1, 1)
        previous_one_hot = torch.nn.functional.one_hot(previous_raw, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        past = prior_requantize(self.conv_past(previous_one_hot), 0, -self.activation_bound, self.activation_bound)
        spm = None
        if self.use_spm:
            patch_rows, patch_cols = (height // self.P, width // self.P)
            pooled = past.view(batch, self.ch, patch_rows, self.P, patch_cols, self.P).mean(dim=(3, 5))
            pooled = prior_ste_round(pooled)
            pooled = prior_integer_activation(prior_requantize(self.spm_dw(pooled), 3, -self.activation_bound, self.activation_bound), self.activation)
            pooled = prior_requantize(self.spm_pw(pooled), 4, -self.activation_bound, self.activation_bound)
            spm = pooled.unsqueeze(3).unsqueeze(5).expand(batch, self.ch, patch_rows, self.P, patch_cols, self.P).contiguous().view(batch, self.ch, height, width)
        scale = None
        if self.use_frame_scale:
            scale = prior_requantize(self.frame_scale(embedding), 4, -8, 8)
            scale = scale.view(batch, 1, self.ch, 1, 1).expand(batch, patch_count, self.ch, 1, 1).reshape(batch * patch_count, self.ch, 1, 1)
        return (shift, self._to_patches(past), scale, None if spm is None else self._to_patches(spm))

@dataclass
class sparse_GroupPlan:
    targets: torch.Tensor
    h_positions: torch.Tensor
    b1_gather: torch.Tensor
    b2_gather: torch.Tensor
    output_order: torch.Tensor

def sparse_active_offsets(module):
    kernel = module.mask.shape[-1]
    center = (kernel - 1) // 2
    offsets = []
    for row, col in module.mask[0, 0].nonzero(as_tuple=False).tolist():
        offsets.append(((row - center) * module.dilation, (col - center) * module.dilation))
    return offsets

def sparse_positions_for_group(patch, delta, group):
    return [(row, col) for row in range(patch) for col in range(patch) if col + delta * row == group]

def sparse_expanded_positions(positions, offsets, patch):
    return sorted({(row + dy, col + dx) for row, col in positions for dy, dx in offsets if 0 <= row + dy < patch and 0 <= col + dx < patch})

def sparse_gather_map(outputs, inputs, offsets, patch):
    lookup = {position: index for index, position in enumerate(inputs)}
    sentinel = len(inputs)
    return [[lookup.get((row + dy, col + dx), sentinel) if 0 <= row + dy < patch and 0 <= col + dx < patch else sentinel for dy, dx in offsets] for row, col in outputs]

class sparse_SparseIntegerPrior:

    def __init__(self, model, height=384, width=512):
        self.model = model
        self.patch = model.P
        self.patch_rows = height // model.P
        self.patch_cols = width // model.P
        self.patch_count = self.patch_rows * self.patch_cols
        self.a_offsets = sparse_active_offsets(model.conv_a)
        self.b1_offsets = sparse_active_offsets(model.conv_b1)
        self.b2_offsets = sparse_active_offsets(model.conv_b2)
        device = next(model.parameters()).device
        self.plans = [self._build_plan(group, device) for group in range((1 + model.delta) * model.P - model.delta)]

    def _build_plan(self, group, device):
        targets = sparse_positions_for_group(self.patch, self.model.delta, group)
        b1_positions = sparse_expanded_positions(targets, self.b2_offsets, self.patch)
        h_positions = sparse_expanded_positions(b1_positions, self.b1_offsets, self.patch)
        b1_gather = sparse_gather_map(b1_positions, h_positions, self.b1_offsets, self.patch)
        b2_gather = sparse_gather_map(targets, b1_positions, self.b2_offsets, self.patch)
        patch_major = []
        for patch_index in range(self.patch_count):
            patch_row, patch_col = divmod(patch_index, self.patch_cols)
            for row, col in targets:
                global_row = patch_row * self.patch + row
                global_col = patch_col * self.patch + col
                patch_major.append(global_row * self.patch_cols * self.patch + global_col)
        output_order = sorted(range(len(patch_major)), key=patch_major.__getitem__)
        return sparse_GroupPlan(targets=torch.tensor(targets, dtype=torch.long, device=device), h_positions=torch.tensor(h_positions, dtype=torch.long, device=device), b1_gather=torch.tensor(b1_gather, dtype=torch.long, device=device), b2_gather=torch.tensor(b2_gather, dtype=torch.long, device=device), output_order=torch.tensor(output_order, dtype=torch.long, device=device))
prior_io_SELF_COMPRESSED_MAGIC = bytes.fromhex('49485331')
prior_io_COMPRESSIBLE_TYPES = (prior_IntegerConv2d, prior_IntegerLinear)

def prior_io_unpack_nibbles(raw, count):
    byte_count = (count + 1) // 2
    packed = np.frombuffer(raw[:byte_count], dtype=np.uint8)
    values = np.empty(byte_count * 2, dtype=np.uint8)
    values[0::2] = packed & 15
    values[1::2] = packed >> 4
    return (values[:count].copy(), raw[byte_count:])

def prior_io_weight_rows(module, weight):
    if isinstance(module, prior_IntegerConv2d):
        mask = module.mask.to(torch.bool).expand_as(weight)
        return [weight[index][mask[index]] for index in range(weight.shape[0])]
    return [weight[index].reshape(-1) for index in range(weight.shape[0])]

def prior_io_restore_weight_row(module, parameter, index, values):
    values = torch.from_numpy(values.astype(np.float32))
    if isinstance(module, prior_IntegerConv2d):
        mask = module.mask.to(torch.bool).expand_as(parameter)[index]
        parameter[index].zero_()
        parameter[index][mask] = values
    else:
        parameter[index].copy_(values.reshape(parameter[index].shape))

def prior_io_deserialize_self_compressed(model, raw):
    view = memoryview(raw)[len(prior_io_SELF_COMPRESSED_MAGIC):]
    modules = [module for module in model.modules() if isinstance(module, prior_io_COMPRESSIBLE_TYPES)]
    channel_count = sum((module.weight.shape[0] for module in modules))
    depths, view = prior_io_unpack_nibbles(view, channel_count)
    total_weight_bits = 0
    depth_offset = 0
    for module in modules:
        module_depths = depths[depth_offset:depth_offset + module.weight.shape[0]]
        row_counts = [row.numel() for row in prior_io_weight_rows(module, module.weight)]
        total_weight_bits += sum((int(bits) * count for bits, count in zip(module_depths, row_counts)))
        depth_offset += module.weight.shape[0]
    weight_bytes = (total_weight_bits + 7) // 8
    packed = np.frombuffer(view[:weight_bytes], dtype=np.uint8)
    weight_bits = np.unpackbits(packed, bitorder='little')[:total_weight_bits]
    view = view[weight_bytes:]
    bit_offset = 0
    depth_offset = 0
    with torch.no_grad():
        for module in modules:
            parameter = module.weight
            module_depths = depths[depth_offset:depth_offset + parameter.shape[0]]
            for index, (bits, template) in enumerate(zip(module_depths, prior_io_weight_rows(module, parameter))):
                count = template.numel()
                bits = int(bits)
                if bits:
                    count_bits = count * bits
                    rows = weight_bits[bit_offset:bit_offset + count_bits].reshape(count, bits).astype(np.int16)
                    unsigned = (rows * (1 << np.arange(bits, dtype=np.int16))).sum(axis=1, dtype=np.int16)
                    sign = 1 << bits - 1
                    values = np.where(unsigned >= sign, unsigned - (1 << bits), unsigned).astype(np.int16)
                    bit_offset += count_bits
                else:
                    values = np.zeros(count, dtype=np.int16)
                prior_io_restore_weight_row(module, parameter, index, values)
            depth_offset += parameter.shape[0]
        module_by_name = dict(model.named_modules())
        for name, parameter in model.named_parameters():
            module_name, field = name.rsplit('.', 1)
            module = module_by_name[module_name]
            if field == 'weight' and isinstance(module, prior_io_COMPRESSIBLE_TYPES):
                continue
            dtype = np.dtype('<i2' if field == 'bias' else 'i1')
            byte_count = parameter.numel() * dtype.itemsize
            value = np.frombuffer(view[:byte_count], dtype=dtype, count=parameter.numel()).copy().reshape(parameter.shape)
            parameter.copy_(torch.from_numpy(value.astype(np.float32)))
            view = view[byte_count:]
render_N = 600
render_NUM_CLASSES = 5
render_EVAL_H, render_EVAL_W = (384, 512)
render_CAMERA_H, render_CAMERA_W = (874, 1164)
render_SEMANTIC_WIDTH = 96
render_SEMANTIC_FRAME_DIM = 8
render_CARRIER_DIM = 12
render_CARRIER_AMPLITUDE = 64.0
render_Prior_PATCH = 64
render_Prior_DELTA = 2
render_Prior_CHANNELS = 64
render_Prior_FILM_DIM = 8
render_Prior_LOGIT_PRECISION = 8

class render_TokenBlock(nn.Module):

    def __init__(self, width, frame_dim, dilation):
        super().__init__()
        self.dw = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation, groups=width)
        self.pw = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(max(1, width // 8), width)
        self.film = nn.Linear(frame_dim, 2 * width)

    def forward(self, value, frame):
        residual = self.norm(self.pw(self.dw(value)))
        scale, shift = self.film(frame).chunk(2, dim=1)
        residual = residual * (1.0 + scale[:, :, None, None])
        residual = residual + shift[:, :, None, None]
        return value + F.gelu(residual)

class render_SemanticTokenRenderer(nn.Module):

    def __init__(self, width=render_SEMANTIC_WIDTH):
        super().__init__()
        self.token_embed = nn.Embedding(render_NUM_CLASSES, width)
        self.frame_embed = nn.Embedding(render_N, render_SEMANTIC_FRAME_DIM)
        self.coord_mix = nn.Conv2d(width + 4, width, 1)
        self.blocks = nn.ModuleList([render_TokenBlock(width, render_SEMANTIC_FRAME_DIM, dilation) for dilation in (1, 1, 2, 4)])
        self.head = nn.Conv2d(width, 3, 3, padding=1)

    @staticmethod
    def coordinates(batch, device, dtype):
        yy, xx = torch.meshgrid(torch.linspace(-1.0, 1.0, render_EVAL_H, device=device, dtype=dtype), torch.linspace(-1.0, 1.0, render_EVAL_W, device=device, dtype=dtype), indexing='ij')
        coordinates = torch.stack([xx, yy, xx.square(), yy.square()], dim=0)
        return coordinates.unsqueeze(0).expand(batch, -1, -1, -1)

    def forward(self, tokens, pair_indices):
        value = self.token_embed(tokens).permute(0, 3, 1, 2)
        value = self.coord_mix(torch.cat([value, self.coordinates(value.shape[0], value.device, value.dtype)], dim=1))
        frame = self.frame_embed(pair_indices)
        for block in self.blocks:
            value = block(value, frame)
        return torch.sigmoid(self.head(F.gelu(value))) * 255.0

def render_group_masks(device):
    rows = torch.arange(render_Prior_PATCH, device=device).view(render_Prior_PATCH, 1)
    columns = torch.arange(render_Prior_PATCH, device=device).view(1, render_Prior_PATCH)
    grid = columns + render_Prior_DELTA * rows
    patch_rows, patch_columns = (render_EVAL_H // render_Prior_PATCH, render_EVAL_W // render_Prior_PATCH)
    masks = []
    for group in range((1 + render_Prior_DELTA) * render_Prior_PATCH - render_Prior_DELTA):
        local = grid == group
        full = local[None, None].expand(patch_rows, patch_columns, render_Prior_PATCH, render_Prior_PATCH)
        masks.append(full.permute(0, 2, 1, 3).reshape(render_EVAL_H, render_EVAL_W))
    return masks

def render_normalized_basis(raw_basis):
    basis = F.interpolate(raw_basis, size=(render_EVAL_H, render_EVAL_W), mode='bicubic', align_corners=False)
    basis = basis - basis.mean(dim=(1, 2, 3), keepdim=True)
    rms = basis.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-05)
    return basis / rms

@torch.no_grad()
def render_render_video(semantic, basis, coefficients, tokens, destination, device):
    semantic = semantic.eval().to(device)
    basis = render_normalized_basis(basis.to(device))
    coefficients = coefficients.to(device)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = np.memmap(destination, mode='w+', dtype=np.uint8, shape=(render_N * 2, render_CAMERA_H, render_CAMERA_W, 3))
    semantic_batch = 8 if device.type == 'cuda' else 1
    for start in range(0, render_N, semantic_batch):
        end = min(start + semantic_batch, render_N)
        indices = torch.arange(start, end, device=device)
        master = F.interpolate(semantic(tokens[start:end].long().to(device), indices), size=(render_CAMERA_H, render_CAMERA_W), mode='bilinear', align_corners=False).clamp(0.0, 255.0).round()
        master_np = master.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for offset in range(end - start):
            output[2 * (start + offset) + 1] = master_np[offset]
    pose_batch = 64 if device.type == 'cuda' else 1
    for start in range(0, render_N, pose_batch):
        end = min(start + pose_batch, render_N)
        carrier = torch.einsum('bk,kchw->bchw', coefficients[start:end], basis)
        carrier = carrier / math.sqrt(render_CARRIER_DIM)
        slave = F.interpolate((127.5 + render_CARRIER_AMPLITUDE * carrier).clamp(0.0, 255.0).round(), size=(render_CAMERA_H, render_CAMERA_W), mode='bicubic', align_corners=False).clamp(0.0, 255.0).round()
        slave_np = slave.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        for offset in range(end - start):
            output[2 * (start + offset)] = slave_np[offset]
    output.flush()

def bits_bounds(bits):
    return (-(1 << bits - 1), (1 << bits - 1) - 1)

def bits_packed_length(count, bits):
    bits_bounds(bits)
    return (count * bits + 7) // 8

def bits_unpack_signed(blob, count, bits):
    expected = bits_packed_length(count, bits)
    total_bits = count * bits
    if total_bits % 8:
        padding_mask = ~((1 << total_bits % 8) - 1) & 255
    unsigned_mask = (1 << bits) - 1
    sign = 1 << bits - 1
    result: list[int] = []
    acc = 0
    available = 0
    index = 0
    for _ in range(count):
        while available < bits:
            acc |= blob[index] << available
            index += 1
            available += 8
        value = acc & unsigned_mask
        acc >>= bits
        available -= bits
        result.append(value - (1 << bits) if value & sign else value)
    return tuple(result)
selector_SPARSE_MAGIC = bytes.fromhex('46304531')
selector_SPARSE_VERSION = 1
selector_HEADER = struct.Struct('<4sBH')
selector_IDENTITY = 0
selector_LUMA = 3
selector_CHANNEL = 4
selector_ROLL = 5
selector_TILE = 6

class selector_Frame0SelectorError(ValueError):
    pass

@dataclass(frozen=True, order=True)
class selector_SelectorMode:
    kind: int
    a: int = 0
    b: int = 0
    c: int = 0
selector_SPARSE_PIXEL_MODES = (selector_SelectorMode(selector_IDENTITY), selector_SelectorMode(selector_LUMA, 1), selector_SelectorMode(selector_LUMA, -1), selector_SelectorMode(selector_CHANNEL, 1, 0, -1), selector_SelectorMode(selector_ROLL, 1, 0), selector_SelectorMode(selector_ROLL, 0, 1), selector_SelectorMode(selector_TILE, 0, 1), selector_SelectorMode(selector_TILE, 3, 1))

def selector_combination_unrank(rank, count, frames):
    if not 0 <= count <= frames or not 0 <= rank < math.comb(frames, count):
        raise selector_Frame0SelectorError('selector combination rank is out of range')
    positions = np.empty(count, dtype=np.int64)
    upper = frames - 1
    remaining = rank
    for width in range(count, 0, -1):
        low, high = (width - 1, upper)
        while low < high:
            middle = (low + high + 1) // 2
            if math.comb(middle, width) <= remaining:
                low = middle
            else:
                high = middle - 1
        positions[width - 1] = low
        remaining -= math.comb(low, width)
        upper = low - 1
    if remaining:
        raise selector_Frame0SelectorError('non-canonical selector combination rank')
    return positions

def selector_unpack_labels(payload, count):
    bit_count = count * 3
    if len(payload) != (bit_count + 7) // 8:
        raise selector_Frame0SelectorError('invalid selector label length')
    if bit_count % 8 and payload[-1] & (1 << 8 - bit_count % 8) - 1:
        raise selector_Frame0SelectorError('non-zero selector label padding')
    labels = np.empty(count, dtype=np.uint8)
    cursor = 0
    for index in range(count):
        label = 0
        for _ in range(3):
            byte = payload[cursor // 8]
            shift = 7 - cursor % 8
            label = label << 1 | byte >> shift & 1
            cursor += 1
        if label >= len(selector_SPARSE_PIXEL_MODES) - 1:
            raise selector_Frame0SelectorError('selector label is out of range')
        labels[index] = label + 1
    return labels

def selector_decode_selector(payload):
    if len(payload) < selector_HEADER.size:
        raise selector_Frame0SelectorError('truncated sparse selector header')
    magic, version, count = selector_HEADER.unpack_from(payload)
    if magic != selector_SPARSE_MAGIC or version != selector_SPARSE_VERSION or (not 1 <= count <= 600):
        raise selector_Frame0SelectorError('invalid sparse selector header')
    limit = math.comb(600, count)
    rank_bytes = ((limit - 1).bit_length() + 7) // 8
    label_bytes = (count * 3 + 7) // 8
    expected = selector_HEADER.size + rank_bytes + label_bytes
    if len(payload) != expected:
        raise selector_Frame0SelectorError('truncated or trailing selector payload')
    offset = selector_HEADER.size
    rank = int.from_bytes(payload[offset:offset + rank_bytes], 'big')
    positions = selector_combination_unrank(rank, count, 600)
    labels = selector_unpack_labels(payload[offset + rank_bytes:], count)
    choices = np.zeros(600, dtype=np.uint8)
    choices[positions] = labels
    return (selector_SPARSE_PIXEL_MODES, choices)

def selector_apply_pixel_mode(frames, mode):
    values = np.asarray(frames)
    if values.ndim != 4 or values.shape[-1] != 3 or values.dtype != np.uint8:
        raise selector_Frame0SelectorError('selector requires BxHxWx3 uint8 frames')
    if mode.kind == selector_IDENTITY:
        return values.copy()
    if mode.kind == selector_ROLL:
        return np.roll(values, shift=(mode.b, mode.a), axis=(1, 2))
    if mode.kind == selector_TILE:
        height, width = values.shape[1:3]
        yy, xx = np.indices((height, width), dtype=np.int32)
        if mode.a == 0:
            signs = (yy + xx & 1) * 2 - 1
        elif mode.a == 3:
            signs = ((yy >> 2) + (xx >> 2) & 1) * 2 - 1
        delta = signs[None, :, :, None] * mode.b
    elif mode.kind == selector_LUMA:
        delta = mode.a
    elif mode.kind == selector_CHANNEL:
        delta = np.asarray((mode.a, mode.b, mode.c), dtype=np.int16).reshape(1, 1, 1, 3)
    return np.clip(values.astype(np.int16) + delta, 0, 255).astype(np.uint8)
overlay_PAIR_COUNT = 600
overlay_SELECTOR_HEADER = struct.Struct('<4sBH')

def overlay_selector_payload_bytes(payload):
    magic, version, count = overlay_SELECTOR_HEADER.unpack_from(payload)
    limit = math.comb(overlay_PAIR_COUNT, count)
    rank_bytes = ((limit - 1).bit_length() + 7) // 8
    label_bytes = (count * 3 + 7) // 8
    expected = overlay_SELECTOR_HEADER.size + rank_bytes + label_bytes
    return expected

def overlay_split_selector_compensation(payload):
    selector_bytes = overlay_selector_payload_bytes(payload)
    selector = payload[:selector_bytes]
    overlay = payload[selector_bytes:]
    if not overlay:
        return (selector, None)
basis_CARRIER_DIM = 12
basis_CARRIER_H, basis_CARRIER_W = (24, 32)
basis_BASIS_PLANES = 3
basis_BASIS_ALPHABET = 32
basis_CODE_BITS = 32
basis_TOP = (1 << basis_CODE_BITS) - 1
basis_QTR = 1 << basis_CODE_BITS - 2
basis_HALF = 2 * basis_QTR
basis_3QTR = 3 * basis_QTR

class basis_BitReader:

    def __init__(self, payload, bit_count):
        self._bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder='big')[:bit_count]
        self._cursor = 0

    def get(self):
        if self._cursor >= self._bits.size:
            return 0
        bit = int(self._bits[self._cursor])
        self._cursor += 1
        return bit

class basis_AdaptiveModel:

    def __init__(self, alphabet, n_contexts, increment=32):
        self.alphabet = alphabet
        self.increment = increment
        self._freq = np.ones((n_contexts, alphabet), dtype=np.int64)
        self._total = np.full(n_contexts, alphabet, dtype=np.int64)

    def total(self, context):
        return int(self._total[context])

    def find(self, context, target):
        row = self._freq[context]
        cumulative = np.cumsum(row)
        symbol = int(np.searchsorted(cumulative, target, side='right'))
        low = int(cumulative[symbol - 1]) if symbol else 0
        return (symbol, low, int(cumulative[symbol]))

    def update(self, context, symbol):
        self._freq[context, symbol] += self.increment
        self._total[context] += self.increment

def basis_basis_contexts():
    return np.repeat(np.arange(basis_CARRIER_DIM), basis_BASIS_PLANES * basis_CARRIER_H * basis_CARRIER_W)

def basis_arith_decode(payload, bit_count, contexts, model):
    reader = basis_BitReader(payload, bit_count)
    low, high = (0, basis_TOP)
    value = 0
    for _ in range(basis_CODE_BITS):
        value = value << 1 | reader.get()
    out = np.empty(contexts.size, dtype=np.int64)
    for index, context in enumerate(contexts.tolist()):
        total = model.total(context)
        span = high - low + 1
        target = ((value - low + 1) * total - 1) // span
        symbol, cum_low, cum_high = model.find(context, target)
        high = low + span * cum_high // total - 1
        low = low + span * cum_low // total
        while True:
            if high < basis_HALF:
                pass
            elif low >= basis_HALF:
                low -= basis_HALF
                high -= basis_HALF
                value -= basis_HALF
            elif low >= basis_QTR and high < basis_3QTR:
                low -= basis_QTR
                high -= basis_QTR
                value -= basis_QTR
            else:
                break
            low = low << 1 & basis_TOP
            high = (high << 1 | 1) & basis_TOP
            value = (value << 1 | reader.get()) & basis_TOP
        out[index] = symbol
        model.update(context, symbol)
    return out

def basis_decode_basis_arith(payload, bit_count):
    model = basis_AdaptiveModel(basis_BASIS_ALPHABET, basis_CARRIER_DIM)
    return basis_arith_decode(payload, bit_count, basis_basis_contexts(), model)
coefficients_N_FRAMES = 600
coefficients_CARRIER_DIM = 12
coefficients_SYMBOL_COUNT = coefficients_N_FRAMES * coefficients_CARRIER_DIM
coefficients_CABAC_CONTEXT_CAP = 8

class coefficients_RangeDecoder:
    __slots__ = ('buffer', 'code', 'position', 'range')

    def __init__(self, payload):
        self.buffer = payload
        self.position = 0
        self.range = 4294967295
        self.code = 0
        for _ in range(4):
            self.code = (self.code << 8 | self._byte()) & 4294967295

    def _byte(self):
        if self.position < len(self.buffer):
            value = self.buffer[self.position]
            self.position += 1
            return value

    def decode_frequency(self, total):
        unit = self.range // total
        value = self.code // unit
        return total - 1 if value >= total else value

    def update(self, cumulative_low, frequency, total):
        unit = self.range // total
        self.code -= unit * cumulative_low
        self.range = unit * frequency
        while self.range < 1 << 24:
            self.range <<= 8
            self.code = (self.code << 8 | self._byte()) & 4294967295

@dataclass
class coefficients_BinaryModel:
    probability_zero: int = 2048

def coefficients_validate_ks(ks):
    values = np.asarray(ks, dtype=np.int64).reshape(-1)
    return values

def coefficients_dimension_sequence():
    return np.tile(np.arange(coefficients_CARRIER_DIM, dtype=np.int64), coefficients_N_FRAMES)

def coefficients_cabac_decode(payload, ks):
    parameters = coefficients_validate_ks(ks)
    contexts = [[coefficients_BinaryModel() for _ in range(coefficients_CABAC_CONTEXT_CAP + 1)] for _ in range(coefficients_CARRIER_DIM)]
    decoder = coefficients_RangeDecoder(bytes(payload))
    output = np.empty(coefficients_SYMBOL_COUNT, dtype=np.int32)
    for index, dimension in enumerate(coefficients_dimension_sequence().tolist()):
        k = int(parameters[dimension])
        models = contexts[dimension]
        quotient = 0
        while True:
            model = models[min(quotient, coefficients_CABAC_CONTEXT_CAP)]
            target = decoder.decode_frequency(4096)
            if target < model.probability_zero:
                decoder.update(0, model.probability_zero, 4096)
                model.probability_zero += 4096 - model.probability_zero >> 4
                quotient += 1
            else:
                decoder.update(model.probability_zero, 4096 - model.probability_zero, 4096)
                model.probability_zero -= model.probability_zero >> 4
                break
        value = quotient << k
        for shift in range(k - 1, -1, -1):
            target = decoder.decode_frequency(4096)
            bit = int(target >= 2048)
            decoder.update(2048 * bit, 2048, 4096)
            value |= bit << shift
        output[index] = value
    return output.reshape(coefficients_N_FRAMES, coefficients_CARRIER_DIM)
counts_NUM_CLASSES = 5
counts_U_BINS = 64
counts_RUN_LEVELS = 8
counts_RUN_CAP = 255
counts_BOUNDARY_LEVELS = 5
counts_KT_ALPHA = 0.5
counts_MIN_COUNT = 32
counts_ODDS_LOW = 0.0625
counts_ODDS_HIGH = 16.0
counts_PROB_EPS = 1e-09
counts_PHAT_SCALE = 1 << 30
counts_CONTEXT_SIZE = counts_NUM_CLASSES * counts_U_BINS * 2 * 2 * counts_RUN_LEVELS * counts_BOUNDARY_LEVELS
counts_INV_SQRT2 = math.sqrt(0.5)

def counts_surprise_thresholds():
    table = np.empty(counts_U_BINS - 1, dtype=np.float64)
    for k in range(1, counts_U_BINS):
        value = math.ldexp(1.0, -(k // 2))
        if k % 2:
            value *= counts_INV_SQRT2
        table[k - 1] = value
    return table
counts_SURPRISE_ASC = counts_surprise_thresholds()[::-1].copy()

class counts_GroupState:
    __slots__ = ('arg', 'context', 'one_minus', 'p_max', 'p_max_q', 'row64')

    def __init__(self, row64, arg, p_max, one_minus, p_max_q, context):
        self.row64 = row64
        self.arg = arg
        self.p_max = p_max
        self.one_minus = one_minus
        self.p_max_q = p_max_q
        self.context = context

class counts_FreeCorrector:
    __slots__ = ('boundary', 'counts', 'have_prev', 'hits', 'phat_q', 'plane', 'prev1', 'prev2', 'run')

    def __init__(self, plane):
        self.plane = int(plane)
        self.counts = np.zeros(counts_CONTEXT_SIZE, dtype=np.int64)
        self.hits = np.zeros(counts_CONTEXT_SIZE, dtype=np.int64)
        self.phat_q = np.zeros(counts_CONTEXT_SIZE, dtype=np.int64)
        self.prev1 = np.zeros(self.plane, dtype=np.uint8)
        self.prev2 = np.zeros(self.plane, dtype=np.uint8)
        self.run = np.zeros(self.plane, dtype=np.int64)
        self.have_prev = False
        self.boundary = np.full(self.plane, counts_BOUNDARY_LEVELS - 1, dtype=np.int64)

    def begin_frame(self, boundary_flat):
        boundary = np.asarray(boundary_flat, dtype=np.int64).reshape(-1)
        if boundary.size != self.plane:
            raise ValueError('boundary bucket plane size mismatch')
        self.boundary = boundary

    def group_state(self, probability, predicted, positions):
        row64 = np.asarray(probability, dtype=np.float32).astype(np.float64)
        if row64.ndim != 2 or row64.shape[1] != counts_NUM_CLASSES:
            raise ValueError('probability rows must have shape [n, 5]')
        index = np.arange(row64.shape[0])
        arg = row64.argmax(axis=1)
        p_max = row64[index, arg]
        one_minus = np.maximum(1.0 - p_max, counts_PROB_EPS)
        base_class = np.asarray(predicted, dtype=np.int64).reshape(-1)
        flat = np.asarray(positions, dtype=np.int64).reshape(-1)
        if base_class.size != row64.shape[0] or flat.size != row64.shape[0]:
            raise ValueError('group predicted/positions length mismatch')
        below = np.searchsorted(counts_SURPRISE_ASC, one_minus, side='left')
        ubin = np.clip(counts_U_BINS - 1 - below, 0, counts_U_BINS - 1).astype(np.int64)
        if self.have_prev:
            agree1 = (self.prev1[flat].astype(np.int64) == base_class).astype(np.int64)
            agree2 = (self.prev2[flat].astype(np.int64) == base_class).astype(np.int64)
        else:
            agree1 = np.zeros(flat.size, dtype=np.int64)
            agree2 = np.zeros(flat.size, dtype=np.int64)
        run = np.minimum(self.run[flat], counts_RUN_LEVELS - 1)
        head = ((base_class * counts_U_BINS + ubin) * 2 + agree1) * 2 + agree2
        context = (head * counts_RUN_LEVELS + run) * counts_BOUNDARY_LEVELS + self.boundary[flat]
        p_max_q = np.rint(p_max * counts_PHAT_SCALE).astype(np.int64)
        return counts_GroupState(row64, arg, p_max, one_minus, p_max_q, context)

    def end_frame(self, tokens_flat):
        current = np.asarray(tokens_flat, dtype=np.uint8).reshape(-1)
        if current.size != self.plane:
            raise ValueError('token plane size mismatch')
        if self.have_prev:
            self.run = np.where(current == self.prev1, np.minimum(self.run + 1, counts_RUN_CAP), 0)
            self.prev2 = self.prev1
        self.prev1 = current.copy()
        self.have_prev = True
odds_HEIGHT = 384
odds_WIDTH = 512
odds_SPATIAL_LEVELS = 5
odds_WEIGHT_STORE_BITS = 20
odds_WEIGHT_STORE_ONE = 1 << odds_WEIGHT_STORE_BITS
odds_POWER_BITS = 6
odds_INT_POWER_BITS = 4
odds_ERR_SCALE = 1 << 20
odds_STRETCH_SCALE = 1 << 20
odds_STRETCH_CLAMP = 32 * odds_STRETCH_SCALE
odds_COUNT_BUCKET_EDGES = np.array([1, 2, 4, 8, 16, 32, 64, 128, 512, 2048, 8192, 32768, 131072, 524288, 2097152], dtype=np.int64)
odds_COUNT_HALVING_PASSES = 40
odds_BASE_MEMBER = 'base_odds'
odds_LR_BASE_SHIFT = odds_ERR_SCALE.bit_length() - 1 + odds_STRETCH_SCALE.bit_length() - 1 - odds_WEIGHT_STORE_BITS
odds_WEIGHT_LOW = -4 * odds_WEIGHT_STORE_ONE
odds_WEIGHT_HIGH = 8 * odds_WEIGHT_STORE_ONE

def odds_assert_sqrt_is_correctly_rounded():
    for root in (1.0, 1.5, 2.0, 3.0, 4.0, 1.0625, 65536.0, 100000000.0):
        squared = np.float64(root) * np.float64(root)
        if np.sqrt(squared) != np.float64(root):
            raise RuntimeError(f'platform sqrt is not correctly rounded at {root}; refusing to decode')
    value = np.float64(1.0)
    for _ in range(8):
        value = value / np.float64(4.0)
    for _ in range(16):
        if np.sqrt(value * value) != value:
            raise RuntimeError('platform sqrt is not correctly rounded on a power of four')
        value = value * np.float64(4.0)

def odds_round_shift(value, bits):
    return value + (np.int64(1) << np.int64(bits - 1)) >> np.int64(bits)

def odds_stretch_from_radical(radical, power_bits):
    return (radical - 1.0) * float(1 << power_bits)

def odds_dyadic_power(value, radicals, weight, power_bits):
    negative = weight < 0
    magnitude = np.abs(weight)
    integer_part = magnitude >> np.int64(power_bits)
    fraction = magnitude & np.int64((1 << power_bits) - 1)
    shape = np.broadcast_shapes(np.shape(value), np.shape(weight))
    accumulator = np.ones(shape, dtype=np.float64)
    base = np.broadcast_to(np.asarray(value, dtype=np.float64), shape).astype(np.float64, copy=True)
    remaining = integer_part
    for _ in range(odds_INT_POWER_BITS):
        accumulator = np.where(remaining & 1 == 1, accumulator * base, accumulator)
        base = base * base
        remaining = remaining >> np.int64(1)
    if np.any(fraction != 0):
        if radicals is None:
            raise ValueError('fractional weights require the radicals')
        for index in range(power_bits):
            bit = fraction >> np.int64(power_bits - 1 - index) & 1
            accumulator = np.where(bit == 1, accumulator * radicals[index], accumulator)
    return np.where(negative, 1.0 / accumulator, accumulator)

class odds_MixerFamily:
    __slots__ = ('count_limit', 'counts', 'hits', 'name', 'phat_q', 'rule', 'size')

    def __init__(self, name, size, rule, count_limit=0):
        self.name = name
        self.size = int(size)
        self.rule = rule
        self.count_limit = int(count_limit)
        self.counts = np.zeros(self.size, dtype=np.int64)
        self.hits = np.zeros(self.size, dtype=np.int64)
        self.phat_q = np.zeros(self.size, dtype=np.int64)

    def multiplier(self, index):
        count = self.counts[index].astype(np.float64)
        denominator = count + 2.0 * counts_KT_ALPHA
        hit_numerator = self.hits[index].astype(np.float64) + counts_KT_ALPHA
        hit_denominator = denominator - hit_numerator
        expected = self.phat_q[index].astype(np.float64) / counts_PHAT_SCALE
        exp_numerator = expected + counts_KT_ALPHA
        exp_denominator = denominator - exp_numerator
        multiplier = np.ones(index.shape, dtype=np.float64)
        usable = (self.counts[index] >= counts_MIN_COUNT) & (hit_numerator > 0.0) & (hit_denominator > 0.0) & (exp_numerator > 0.0) & (exp_denominator > 0.0)
        multiplier[usable] = hit_numerator[usable] * exp_denominator[usable] / (hit_denominator[usable] * exp_numerator[usable])
        np.clip(multiplier, counts_ODDS_LOW, counts_ODDS_HIGH, out=multiplier)
        return multiplier

    def observe(self, index, hit, p_max_q):
        np.add.at(self.counts, index, 1)
        np.add.at(self.hits, index, hit)
        np.add.at(self.phat_q, index, p_max_q)
        if self.count_limit:
            touched = np.unique(index)
            for _ in range(odds_COUNT_HALVING_PASSES):
                hot = touched[self.counts[touched] > self.count_limit]
                if not hot.size:
                    break
                self.counts[hot] >>= 1
                self.hits[hot] >>= 1
                self.phat_q[hot] >>= 1

def odds_family_specs():

    def shipped(f):
        head = ((f['cls'] * counts_U_BINS + f['ubin']) * 2 + f['agree1']) * 2 + f['agree2']
        return (head * counts_RUN_LEVELS + f['run']) * counts_BOUNDARY_LEVELS + f['boundary']

    def spatial_surprise(f):
        return (f['cls'] * odds_SPATIAL_LEVELS + f['spatial']) * counts_U_BINS + f['ubin']

    def spatial_boundary(f):
        return (f['cls'] * odds_SPATIAL_LEVELS + f['spatial']) * counts_BOUNDARY_LEVELS + f['boundary']

    def surprise_only(f):
        return f['cls'] * counts_U_BINS + f['ubin']

    def temporal_spatial(f):
        head = (f['cls'] * 2 + f['agree1']) * 2 + f['agree2']
        return head * odds_SPATIAL_LEVELS + f['spatial']

    def run_surprise(f):
        return (f['cls'] * counts_RUN_LEVELS + f['run']) * counts_U_BINS + f['ubin']

    def boundary_surprise(f):
        return (f['cls'] * counts_BOUNDARY_LEVELS + f['boundary']) * counts_U_BINS + f['ubin']

    def temporal_surprise(f):
        head = (f['cls'] * 2 + f['agree1']) * 2 + f['agree2']
        return head * counts_U_BINS + f['ubin']
    joint_size = counts_NUM_CLASSES * counts_U_BINS * 2 * 2 * counts_RUN_LEVELS * counts_BOUNDARY_LEVELS
    return {'shipped_joint': (joint_size, shipped), 'shipped_fast256': (joint_size, shipped, 256), 'shipped_fast4096': (joint_size, shipped, 4096), 'surprise_fast256': (counts_NUM_CLASSES * counts_U_BINS, surprise_only, 256), 'spatial_surprise': (counts_NUM_CLASSES * odds_SPATIAL_LEVELS * counts_U_BINS, spatial_surprise), 'spatial_boundary': (counts_NUM_CLASSES * odds_SPATIAL_LEVELS * counts_BOUNDARY_LEVELS, spatial_boundary), 'surprise_only': (counts_NUM_CLASSES * counts_U_BINS, surprise_only), 'temporal_spatial': (counts_NUM_CLASSES * 2 * 2 * odds_SPATIAL_LEVELS, temporal_spatial), 'run_surprise': (counts_NUM_CLASSES * counts_RUN_LEVELS * counts_U_BINS, run_surprise), 'boundary_surprise': (counts_NUM_CLASSES * counts_BOUNDARY_LEVELS * counts_U_BINS, boundary_surprise), 'temporal_surprise': (counts_NUM_CLASSES * 2 * 2 * counts_U_BINS, temporal_surprise)}
odds_MIXER_CONTEXTS: dict[str, tuple[int, object]] = {'none': (1, lambda f: np.zeros(f['cls'].shape, dtype=np.int64)), 'cls': (counts_NUM_CLASSES, lambda f: f['cls']), 'boundary': (counts_BOUNDARY_LEVELS, lambda f: f['boundary']), 'cls_boundary': (counts_NUM_CLASSES * counts_BOUNDARY_LEVELS, lambda f: f['cls'] * counts_BOUNDARY_LEVELS + f['boundary']), 'ubin8': (8, lambda f: np.minimum(f['ubin'] >> 3, 7)), 'cls_ubin8': (counts_NUM_CLASSES * 8, lambda f: f['cls'] * 8 + np.minimum(f['ubin'] >> 3, 7)), 'cls_boundary_ubin8': (counts_NUM_CLASSES * counts_BOUNDARY_LEVELS * 8, lambda f: (f['cls'] * counts_BOUNDARY_LEVELS + f['boundary']) * 8 + np.minimum(f['ubin'] >> 3, 7)), 'cls_agree_ubin8': (counts_NUM_CLASSES * 4 * 8, lambda f: (f['cls'] * 4 + f['agree1'] * 2 + f['agree2']) * 8 + np.minimum(f['ubin'] >> 3, 7)), 'cls_run_ubin8': (counts_NUM_CLASSES * counts_RUN_LEVELS * 8, lambda f: (f['cls'] * counts_RUN_LEVELS + f['run']) * 8 + np.minimum(f['ubin'] >> 3, 7)), 'cls_boundary_agree_ubin8': (counts_NUM_CLASSES * counts_BOUNDARY_LEVELS * 4 * 8, lambda f: ((f['cls'] * counts_BOUNDARY_LEVELS + f['boundary']) * 4 + f['agree1'] * 2 + f['agree2']) * 8 + np.minimum(f['ubin'] >> 3, 7))}

class odds_FixedPointLogisticMixer(counts_FreeCorrector):

    def __init__(self, plane, *, families=('shipped_joint',), mixer_context='none', count_buckets=1, power_bits=odds_POWER_BITS, learn=True, lr_shift=odds_LR_BASE_SHIFT + 4, normalize=True, initial_weights=None):
        super().__init__(plane)
        if plane != odds_HEIGHT * odds_WIDTH:
            raise ValueError('the mixer assumes the shipped 384x512 plane')
        if not 1 <= power_bits <= 16:
            raise ValueError('power_bits must be in [1, 16]')
        odds_assert_sqrt_is_correctly_rounded()
        specs = odds_family_specs()
        if not families:
            raise ValueError('at least one family is required')
        missing = [name for name in families if name not in specs]
        if missing:
            raise ValueError(f'unknown families: {missing}')
        self.families = [odds_MixerFamily(name, *specs[name]) for name in families]
        if mixer_context not in odds_MIXER_CONTEXTS:
            raise ValueError(f'unknown mixer context {mixer_context!r}')
        self.mixer_context_name = mixer_context
        self.n_mixer_contexts, self._mixer_rule = odds_MIXER_CONTEXTS[mixer_context]
        if not 1 <= count_buckets <= len(odds_COUNT_BUCKET_EDGES) + 1:
            raise ValueError('count_buckets out of range')
        self.count_buckets = int(count_buckets)
        self._count_edges = odds_COUNT_BUCKET_EDGES[:self.count_buckets - 1]
        self.n_weight_sets = self.n_mixer_contexts * self.count_buckets
        self.power_bits = int(power_bits)
        self.store_shift = odds_WEIGHT_STORE_BITS - self.power_bits
        self.learn = bool(learn)
        self.lr_shift = int(lr_shift)
        self.normalize = bool(normalize)
        if initial_weights is None:
            initial = [1.0 if f.name == 'shipped_joint' else 0.0 for f in self.families]
        self.weights = np.zeros((self.n_weight_sets, len(self.families)), dtype=np.int64)
        for position, value in enumerate(initial):
            self.weights[:, position] = np.int64(round(value * odds_WEIGHT_STORE_ONE))
        self.current = np.zeros(plane, dtype=np.uint8)
        self.known = np.zeros(plane, dtype=bool)
        self._pending: dict | None = None

    def begin_frame(self, boundary_flat):
        super().begin_frame(boundary_flat)
        self.known[:] = False
        self.current[:] = 0

    def _spatial_level(self, flat, base_class):
        x = flat % odds_WIDTH
        y = flat // odds_WIDTH
        left = np.maximum(flat - 1, 0)
        up = np.maximum(flat - odds_WIDTH, 0)
        has_left = (x > 0) & self.known[left]
        has_up = (y > 0) & self.known[up]
        agree_left = has_left & (self.current[left].astype(np.int64) == base_class)
        agree_up = has_up & (self.current[up].astype(np.int64) == base_class)
        available = has_left.astype(np.int64) + has_up.astype(np.int64)
        agreeing = agree_left.astype(np.int64) + agree_up.astype(np.int64)
        return np.where(available == 0, 0, agreeing + 1).astype(np.int64)

    def odds_multiplier(self, state):
        if self._pending is None:
            raise RuntimeError('odds_multiplier() called without a group_state()')
        mixer = self._pending['mixer']
        blended = np.ones(state.context.shape, dtype=np.float64)
        stretches: list[np.ndarray] = []
        weight_indices: list[np.ndarray] = []
        for position, (family, index) in enumerate(zip(self.families, self._pending['indices'], strict=True)):
            multiplier = state.p_max / state.one_minus if family.name == odds_BASE_MEMBER else family.multiplier(index)
            weight_index = mixer * self.count_buckets
            weight_indices.append(weight_index)
            grid_weight = odds_round_shift(self.weights[weight_index, position], self.store_shift)
            need = self.learn or bool(np.any(grid_weight & np.int64((1 << self.power_bits) - 1) != 0))
            radicals: list[np.ndarray] | None = None
            if need:
                radicals = []
                root = multiplier
                for _ in range(self.power_bits):
                    root = np.sqrt(root)
                    radicals.append(root)
            stretches.append(odds_stretch_from_radical(radicals[-1], self.power_bits) if radicals else None)
            blended = blended * odds_dyadic_power(multiplier, radicals, grid_weight, self.power_bits)
        np.clip(blended, counts_ODDS_LOW, counts_ODDS_HIGH, out=blended)
        self._pending['stretches'] = stretches
        self._pending['weight_indices'] = weight_indices
        return blended

    def observe(self, state, symbols):
        if self._pending is None:
            raise RuntimeError('observe() called without a group_state()')
        decoded = np.asarray(symbols, dtype=np.int64).reshape(-1)
        hit = (decoded == state.arg).astype(np.int64)
        if self.learn and 'q' in self._pending:
            self._update_weights(hit, self._pending['q'], self._pending['mixer'])
        for family, index in zip(self.families, self._pending['indices'], strict=True):
            if family.name != odds_BASE_MEMBER:
                family.observe(index, hit, state.p_max_q)
        flat = self._pending['flat']
        self.current[flat] = np.asarray(symbols, dtype=np.uint8).reshape(-1)
        self.known[flat] = True
        self._pending = None

    def _update_weights(self, hit, q, mixer):
        assert self._pending is not None
        stretches = self._pending.get('stretches')
        residual = np.rint((hit.astype(np.float64) - q) * odds_ERR_SCALE).astype(np.int64)
        weight_indices = self._pending['weight_indices']
        if len(stretches) != len(weight_indices) or len(stretches) != len(self.families):
            raise RuntimeError(f'mixer bookkeeping fell out of step: {len(stretches)} stretches, {len(weight_indices)} indices, {len(self.families)} members')
        for position, stretch in enumerate(stretches):
            index = weight_indices[position]
            counts = np.bincount(index, minlength=self.n_weight_sets).astype(np.int64)
            quantised = np.clip(np.rint(stretch * odds_STRETCH_SCALE), -odds_STRETCH_CLAMP, odds_STRETCH_CLAMP).astype(np.int64)
            gradient = np.zeros(self.n_weight_sets, dtype=np.int64)
            np.add.at(gradient, index, residual * quantised)
            if self.normalize:
                gradient = np.where(counts > 0, gradient // np.maximum(counts, 1), 0)
            step = odds_round_shift(gradient, self.lr_shift)
            self.weights[:, position] = np.clip(self.weights[:, position] + step, odds_WEIGHT_LOW, odds_WEIGHT_HIGH)
context_HEIGHT = 384
context_WIDTH = 512
context_CAUSAL_OFFSETS: tuple[tuple[int, int], ...] = ((-1, 0), (0, -1), (1, -1), (-1, -1))
context_GROUP_BINS = 8
context_SPATIAL4_LEVELS = 6
context_HOMOGENEITY_LEVELS = 5
context_SSE_BINS = counts_U_BINS
context_SSE_CONTEXTS: dict[str, tuple[int, object]] = {'off': (0, None), 'qbin': (context_SSE_BINS, lambda f: f['qbin']), 'cls_qbin': (counts_NUM_CLASSES * context_SSE_BINS, lambda f: f['cls'] * context_SSE_BINS + f['qbin']), 'bnd_qbin': (counts_BOUNDARY_LEVELS * context_SSE_BINS, lambda f: f['boundary'] * context_SSE_BINS + f['qbin']), 'cls_bnd_qbin': (counts_NUM_CLASSES * counts_BOUNDARY_LEVELS * context_SSE_BINS, lambda f: (f['cls'] * counts_BOUNDARY_LEVELS + f['boundary']) * context_SSE_BINS + f['qbin']), 'cls_homog_qbin': (counts_NUM_CLASSES * context_HOMOGENEITY_LEVELS * context_SSE_BINS, lambda f: (f['cls'] * context_HOMOGENEITY_LEVELS + f['homog']) * context_SSE_BINS + f['qbin'])}

def context_family_specs():
    specs = dict(odds_family_specs())

    def spatial4_surprise(f):
        return (f['cls'] * context_SPATIAL4_LEVELS + f['spatial4']) * counts_U_BINS + f['ubin']

    def spatial4_boundary(f):
        return (f['cls'] * context_SPATIAL4_LEVELS + f['spatial4']) * counts_BOUNDARY_LEVELS + f['boundary']

    def homog_surprise(f):
        return (f['cls'] * context_HOMOGENEITY_LEVELS + f['homog']) * counts_U_BINS + f['ubin']

    def homog_spatial4(f):
        return (f['cls'] * context_HOMOGENEITY_LEVELS + f['homog']) * context_SPATIAL4_LEVELS + f['spatial4']

    def homog_boundary_surprise(f):
        head = (f['cls'] * context_HOMOGENEITY_LEVELS + f['homog']) * counts_BOUNDARY_LEVELS + f['boundary']
        return head * counts_U_BINS + f['ubin']

    def spatial4_temporal(f):
        head = (f['cls'] * 2 + f['agree1']) * 2 + f['agree2']
        return head * context_SPATIAL4_LEVELS + f['spatial4']

    def patch192_only(f):
        return f['patch192']
    specs.update({'patch192_only': (192, patch192_only)})

    def tile48_groupbin8(f):
        return f['tile48'] * context_GROUP_BINS + f['groupbin8']
    specs.update({'tile48_groupbin8': (48 * context_GROUP_BINS, tile48_groupbin8)})

    def cls_groupbin8(f):
        return f['cls'] * context_GROUP_BINS + f['groupbin8']

    def groupbin8_surprise(f):
        return (f['cls'] * context_GROUP_BINS + f['groupbin8']) * counts_U_BINS + f['ubin']
    specs.update({'cls_groupbin8': (counts_NUM_CLASSES * context_GROUP_BINS, cls_groupbin8), 'groupbin8_surprise': (counts_NUM_CLASSES * context_GROUP_BINS * counts_U_BINS, groupbin8_surprise)})
    specs.update({'spatial4_surprise': (counts_NUM_CLASSES * context_SPATIAL4_LEVELS * counts_U_BINS, spatial4_surprise), 'spatial4_boundary': (counts_NUM_CLASSES * context_SPATIAL4_LEVELS * counts_BOUNDARY_LEVELS, spatial4_boundary), 'homog_surprise': (counts_NUM_CLASSES * context_HOMOGENEITY_LEVELS * counts_U_BINS, homog_surprise), 'homog_spatial4': (counts_NUM_CLASSES * context_HOMOGENEITY_LEVELS * context_SPATIAL4_LEVELS, homog_spatial4), 'homog_boundary_surprise': (counts_NUM_CLASSES * context_HOMOGENEITY_LEVELS * counts_BOUNDARY_LEVELS * counts_U_BINS, homog_boundary_surprise), 'spatial4_temporal': (counts_NUM_CLASSES * 2 * 2 * context_SPATIAL4_LEVELS, spatial4_temporal), 'spatial4_surprise_fast256': (counts_NUM_CLASSES * context_SPATIAL4_LEVELS * counts_U_BINS, spatial4_surprise, 256), 'homog_surprise_fast256': (counts_NUM_CLASSES * context_HOMOGENEITY_LEVELS * counts_U_BINS, homog_surprise, 256)})
    return specs

def context_mixer_contexts():
    contexts = dict(odds_MIXER_CONTEXTS)
    contexts.update({'cls_homog_ubin8': (counts_NUM_CLASSES * context_HOMOGENEITY_LEVELS * 8, lambda f: (f['cls'] * context_HOMOGENEITY_LEVELS + f['homog']) * 8 + np.minimum(f['ubin'] >> 3, 7)), 'cls_spatial4_ubin8': (counts_NUM_CLASSES * context_SPATIAL4_LEVELS * 8, lambda f: (f['cls'] * context_SPATIAL4_LEVELS + f['spatial4']) * 8 + np.minimum(f['ubin'] >> 3, 7)), 'cls_boundary_homog_ubin8': (counts_NUM_CLASSES * counts_BOUNDARY_LEVELS * context_HOMOGENEITY_LEVELS * 8, lambda f: ((f['cls'] * counts_BOUNDARY_LEVELS + f['boundary']) * context_HOMOGENEITY_LEVELS + f['homog']) * 8 + np.minimum(f['ubin'] >> 3, 7)), 'cls_boundary_agree_ubin16': (counts_NUM_CLASSES * counts_BOUNDARY_LEVELS * 4 * 16, lambda f: ((f['cls'] * counts_BOUNDARY_LEVELS + f['boundary']) * 4 + f['agree1'] * 2 + f['agree2']) * 16 + np.minimum(f['ubin'] >> 2, 15)), 'cls_boundary_agree_homog_ubin8': (counts_NUM_CLASSES * counts_BOUNDARY_LEVELS * 4 * context_HOMOGENEITY_LEVELS * 8, lambda f: (((f['cls'] * counts_BOUNDARY_LEVELS + f['boundary']) * 4 + f['agree1'] * 2 + f['agree2']) * context_HOMOGENEITY_LEVELS + f['homog']) * 8 + np.minimum(f['ubin'] >> 3, 7)), 'cls_boundary_agree_spatial4_ubin8': (counts_NUM_CLASSES * counts_BOUNDARY_LEVELS * 4 * context_SPATIAL4_LEVELS * 8, lambda f: (((f['cls'] * counts_BOUNDARY_LEVELS + f['boundary']) * 4 + f['agree1'] * 2 + f['agree2']) * context_SPATIAL4_LEVELS + f['spatial4']) * 8 + np.minimum(f['ubin'] >> 3, 7))})
    return contexts

class context_ContextOddsMixer(odds_FixedPointLogisticMixer):

    def __init__(self, plane, *, sse_context='off', sse_learn_weight=False, **kwargs):
        families = tuple(kwargs.pop('families', ('shipped_joint',)))
        mixer_context = str(kwargs.pop('mixer_context', 'none'))
        specs = context_family_specs()
        contexts = context_mixer_contexts()
        if not families:
            raise ValueError('at least one family is required')
        missing = [n for n in families if n not in specs]
        if missing:
            raise ValueError(f'unknown families: {missing}')
        if mixer_context not in contexts:
            raise ValueError(f'unknown mixer context {mixer_context!r}')
        super().__init__(plane, families=('shipped_joint',), mixer_context='none', **kwargs)
        self.families = [odds_MixerFamily(name, *specs[name]) for name in families]
        self.mixer_context_name = mixer_context
        self.n_mixer_contexts, self._mixer_rule = contexts[mixer_context]
        self.n_weight_sets = self.n_mixer_contexts * self.count_buckets
        initial = [1.0 if f.name == 'shipped_joint' else 0.0 for f in self.families]
        self.weights = np.zeros((self.n_weight_sets, len(self.families)), dtype=np.int64)
        for position, value in enumerate(initial):
            self.weights[:, position] = np.int64(round(value * odds_WEIGHT_STORE_ONE))
        if sse_context not in context_SSE_CONTEXTS:
            raise ValueError(f'unknown sse context {sse_context!r}')
        self.sse_context_name = sse_context
        sse_size, self._sse_rule = context_SSE_CONTEXTS[sse_context]
        self.sse_learn_weight = bool(sse_learn_weight)
        self.sse: odds_MixerFamily | None = odds_MixerFamily('sse', sse_size, self._sse_rule) if sse_size else None
        self.sse_weight = np.full(self.n_mixer_contexts, np.int64(odds_WEIGHT_STORE_ONE), dtype=np.int64)

    def _causal_neighbours(self, flat):
        x = flat % context_WIDTH
        y = flat // context_WIDTH
        classes = np.zeros((len(context_CAUSAL_OFFSETS), flat.size), dtype=np.int64)
        available = np.zeros((len(context_CAUSAL_OFFSETS), flat.size), dtype=bool)
        for slot, (dx, dy) in enumerate(context_CAUSAL_OFFSETS):
            nx = x + dx
            ny = y + dy
            inside = (nx >= 0) & (nx < context_WIDTH) & (ny >= 0) & (ny < context_HEIGHT)
            neighbour = np.clip(ny, 0, context_HEIGHT - 1) * context_WIDTH + np.clip(nx, 0, context_WIDTH - 1)
            available[slot] = inside & self.known[neighbour]
            classes[slot] = np.where(available[slot], self.current[neighbour].astype(np.int64), -1)
        return (classes, available)

    @staticmethod
    def _spatial4_level(classes, available, base_class):
        agreeing = ((classes == base_class[None, :]) & available).sum(axis=0)
        any_available = available.any(axis=0)
        return np.where(any_available, np.minimum(agreeing + 1, context_SPATIAL4_LEVELS - 1), 0).astype(np.int64)

    @staticmethod
    def _homogeneity_level(classes, available):
        present = np.zeros((counts_NUM_CLASSES, classes.shape[1]), dtype=bool)
        for slot in range(classes.shape[0]):
            usable = available[slot]
            for value in range(counts_NUM_CLASSES):
                present[value] |= usable & (classes[slot] == value)
        distinct = present.sum(axis=0).astype(np.int64)
        return np.minimum(distinct, context_HOMOGENEITY_LEVELS - 1)

    def group_state(self, probability, predicted, positions):
        state = counts_FreeCorrector.group_state(self, probability, predicted, positions)
        flat = np.asarray(positions, dtype=np.int64).reshape(-1)
        base_class = np.asarray(predicted, dtype=np.int64).reshape(-1)
        classes, available = self._causal_neighbours(flat)
        spatial4 = self._spatial4_level(classes, available, base_class)
        homog = self._homogeneity_level(classes, available)
        packed = state.context
        boundary = packed % counts_BOUNDARY_LEVELS
        rest = packed // counts_BOUNDARY_LEVELS
        run = rest % counts_RUN_LEVELS
        rest //= counts_RUN_LEVELS
        agree2 = rest % 2
        rest //= 2
        agree1 = rest % 2
        rest //= 2
        ubin = rest % counts_U_BINS
        cls = rest // counts_U_BINS
        features = {'cls': cls, 'ubin': ubin, 'agree1': agree1, 'agree2': agree2, 'run': run, 'boundary': boundary, 'spatial': self._spatial_level(flat, base_class), 'spatial4': spatial4, 'homog': homog, 'tile48': flat // context_WIDTH // 64 * (context_WIDTH // 64) + flat % context_WIDTH // 64, 'patch192': flat // context_WIDTH // 32 * (context_WIDTH // 32) + flat % context_WIDTH // 32, 'groupbin8': (flat % context_WIDTH % 64 + 2 * (flat // context_WIDTH % 64)) * 8 // 190}
        self._pending = {'flat': flat, 'features': features, 'indices': [family.rule(features) for family in self.families], 'mixer': np.asarray(self._mixer_rule(features), dtype=np.int64).reshape(-1)}
        return state

    def coding_row(self, state):
        multiplier = self.odds_multiplier(state)
        pending = self._pending
        assert pending is not None
        shifted = state.p_max * multiplier
        q_mix = np.clip(shifted / (shifted + state.one_minus), counts_PROB_EPS, 1.0 - counts_PROB_EPS)
        pending['q'] = q_mix
        row = state.row64.copy()
        active = multiplier != 1.0
        if np.any(active):
            scale = (1.0 - q_mix[active]) / state.one_minus[active]
            rows = row[active] * scale[:, None]
            rows[np.arange(rows.shape[0]), state.arg[active]] = q_mix[active]
            row[active] = rows
        return row.astype(np.float32)

    def observe(self, state, symbols):
        pending = self._pending
        if pending is None:
            raise RuntimeError('observe() called without a group_state()')
        decoded = np.asarray(symbols, dtype=np.int64).reshape(-1)
        hit = (decoded == state.arg).astype(np.int64)
        super().observe(state, symbols)
context_SHIPPED_CONFIG: dict = {'families': ('shipped_joint', 'temporal_spatial', 'surprise_only', 'spatial_surprise', 'spatial_boundary', 'run_surprise', 'boundary_surprise', 'temporal_surprise', 'shipped_fast256', 'shipped_fast4096', 'surprise_fast256', 'spatial4_surprise', 'homog_surprise', 'homog_boundary_surprise', 'spatial4_boundary', 'homog_spatial4', 'spatial4_temporal', 'homog_surprise_fast256', 'spatial4_surprise_fast256', 'groupbin8_surprise', 'cls_groupbin8', 'patch192_only', 'tile48_groupbin8'), 'mixer_context': 'cls_boundary_agree_homog_ubin8', 'count_buckets': 1, 'sse_context': 'off', 'sse_learn_weight': False, 'normalize': True, 'learn': True}
miss_UNKNOWN = counts_NUM_CLASSES
miss_MISS_KT_ALPHA = 0.5
miss_MISS_MIN_COUNT = 1
miss_MISS_CLAMP_HIGH = 16.0
miss_PHAT_SCALE = 1 << 30
miss_SLOT_LEFT, miss_SLOT_UP, miss_SLOT_UPRIGHT, miss_SLOT_UPLEFT = (0, 1, 2, 3)

def miss_miss_cells():
    b = counts_NUM_CLASSES + 1

    def nb3_prev1(nb, prev1):
        head = (nb[miss_SLOT_UP] * b + nb[miss_SLOT_UPRIGHT]) * b + nb[miss_SLOT_LEFT]
        return head * b + prev1
    b2 = b * b
    b3 = b2 * b
    b4 = b3 * b
    return {'nb3_prev1': (b4, nb3_prev1)}

class miss_MissCorrector(context_ContextOddsMixer):

    def __init__(self, plane, **kwargs):
        self._within_miss = bool(kwargs.pop('within_miss', True))
        cell_name = str(kwargs.pop('miss_cell', 'nb3_prev1'))
        self._miss_min_count = int(kwargs.pop('miss_min_count', miss_MISS_MIN_COUNT))
        clamp_high = float(kwargs.pop('miss_clamp', miss_MISS_CLAMP_HIGH))
        super().__init__(plane, **kwargs)
        cells = miss_miss_cells()
        if cell_name not in cells:
            raise ValueError(f'unknown miss cell {cell_name!r}; have {sorted(cells)}')
        self._miss_cell_name = cell_name
        self._n_miss_cells, self._miss_rule = cells[cell_name]
        self._miss_clamp_high = clamp_high
        self._miss_clamp_low = 1.0 / clamp_high
        self._miss_counts = np.zeros((self._n_miss_cells, counts_NUM_CLASSES), dtype=np.int64)
        self._miss_expect = np.zeros((self._n_miss_cells, counts_NUM_CLASSES), dtype=np.int64)
        self._miss_seen = np.zeros(self._n_miss_cells, dtype=np.int64)
        self._miss_pending: np.ndarray | None = None
        self._nb_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._lanes = np.arange(counts_NUM_CLASSES)[None, :]

    def _causal_neighbours(self, flat):
        result = super()._causal_neighbours(flat)
        self._nb_cache = result
        return result

    def _miss_cell(self, flat):
        cached = self._nb_cache
        self._nb_cache = None
        if cached is not None and cached[0].shape[1] == flat.size:
            classes, available = cached
            nb = np.where(available, classes, miss_UNKNOWN)
        if self.have_prev:
            prev1 = self.prev1[flat].astype(np.int64)
        else:
            prev1 = np.full(flat.size, miss_UNKNOWN, dtype=np.int64)
        return self._miss_rule(nb, prev1)

    def _miss_multiplier(self, cell):
        m = np.ones((cell.size, counts_NUM_CLASSES), dtype=np.float64)
        warm = self._miss_seen[cell] >= self._miss_min_count
        if not warm.any():
            return m
        ratio = (self._miss_counts[cell] + miss_MISS_KT_ALPHA) / (self._miss_expect[cell] / miss_PHAT_SCALE + miss_MISS_KT_ALPHA)
        np.clip(ratio, self._miss_clamp_low, self._miss_clamp_high, out=ratio)
        return np.where(warm[:, None], ratio, m)

    def group_state(self, probability, predicted, positions):
        state = super().group_state(probability, predicted, positions)
        if self._within_miss:
            self._miss_pending = self._miss_cell(np.asarray(positions, dtype=np.int64).reshape(-1))
        return state

    def coding_row(self, state):
        row = super().coding_row(state)
        cell = self._miss_pending
        m = self._miss_multiplier(cell)
        arg = state.arg
        index = np.arange(arg.size)
        m[index, arg] = 1.0
        active = np.any(m != 1.0, axis=1)
        if not np.any(active):
            return row
        row64 = row.astype(np.float64)
        weighted = row64 * m
        weighted[index, arg] = 0.0
        base = row64.copy()
        base[index, arg] = 0.0
        big_w = np.zeros(arg.size, dtype=np.float64)
        big_s = np.zeros(arg.size, dtype=np.float64)
        for lane in range(counts_NUM_CLASSES):
            big_w += weighted[:, lane]
            big_s += base[:, lane]
        usable = active & (big_w > 0.0) & (big_s > 0.0)
        scale = np.ones(arg.size, dtype=np.float64)
        scale[usable] = big_s[usable] / big_w[usable]
        updated = weighted * scale[:, None]
        updated[index, arg] = row64[index, arg]
        out = np.where(usable[:, None], updated, row64)
        return out.astype(np.float32)

    def observe(self, state, symbols):
        if self._within_miss and self._miss_pending is not None:
            decoded = np.asarray(symbols, dtype=np.int64).reshape(-1)
            miss = decoded != state.arg
            if np.any(miss):
                cell = self._miss_pending[miss]
                row64 = state.row64[miss]
                one_minus = state.one_minus[miss]
                relative = row64 / one_minus[:, None]
                relative[np.arange(cell.size), state.arg[miss]] = 0.0
                np.add.at(self._miss_counts, (cell, decoded[miss]), 1)
                np.add.at(self._miss_expect, (cell[:, None], self._lanes), np.rint(relative * miss_PHAT_SCALE).astype(np.int64))
                np.add.at(self._miss_seen, cell, 1)
            self._miss_pending = None
        super().observe(state, symbols)
miss_SHIPPED_CONFIG: dict = dict(context_SHIPPED_CONFIG)
miss_SHIPPED_CONFIG.update({'within_miss': True, 'miss_cell': 'nb3_prev1', 'miss_min_count': miss_MISS_MIN_COUNT, 'miss_clamp': miss_MISS_CLAMP_HIGH})

class miss_FreeCorrector(miss_MissCorrector):

    def __init__(self, plane):
        super().__init__(plane, **miss_SHIPPED_CONFIG)

@dataclass(frozen=True)
class inference_ConvAPlan:
    class_indices: torch.Tensor
    valid: torch.Tensor
    coordinates: torch.Tensor

@dataclass(frozen=True)
class inference_SparseCache:
    conv_a_weight: torch.Tensor
    conv_a_plans: tuple[inference_ConvAPlan, ...]
    depthwise_weights: dict[int, torch.Tensor]
    affine_values: dict[int, tuple[torch.Tensor, torch.Tensor | None]]
    depthwise_zero: torch.Tensor

def inference_constant_result(values):

    def result():
        return values
    return result

def inference_constant_tensor_result(value):

    def result():
        return value
    return result

def inference_inference_round(value):
    return value.round()

def inference_inference_requantize(value, shift, low=-127, high=127):
    if shift:
        value = value / (1 << shift)
    return value.round().clamp(low, high)

def inference_freeze_model_codes(model):
    for module in model.modules():
        codes = getattr(module, 'codes', None)
        if not callable(codes):
            continue
        values = tuple((None if value is None else value.detach() for value in codes()))
        module.codes = inference_constant_result(values)
    frame_codes = model.frame_codes().detach()
    model.frame_codes = inference_constant_tensor_result(frame_codes)

def inference_apply_codes(self, value, module):
    bias, multiplier = self._sparse_cache.affine_values[id(module)]
    if multiplier is not None:
        value = value * multiplier
    return value + bias

def inference_conv_a_features(self, current, group):
    cache = self._sparse_cache
    plan = cache.conv_a_plans[group]
    classes = current.reshape(-1)[plan.class_indices]
    one_hot = F.one_hot(classes, num_classes=self.model.num_classes)
    one_hot = one_hot.permute(0, 1, 3, 2).to(cache.conv_a_weight.dtype)
    one_hot = one_hot * plan.valid
    coordinates = plan.coordinates.expand(self.patch_count, -1, -1, -1)
    features = torch.cat([one_hot, coordinates], dim=2)
    features = features.reshape(self.patch_count, features.shape[1], -1)
    value = F.linear(features, cache.conv_a_weight)
    return self._apply_codes(value, self.model.conv_a)

def inference_depthwise(self, value, gather, module):
    cache = self._sparse_cache
    value = torch.cat([value, cache.depthwise_zero], dim=1)
    gathered = value[:, gather, :]
    weight = cache.depthwise_weights[id(module)]
    result = (gathered * weight).sum(2)
    return self._apply_codes(result, module)

def inference_selected_logits(self, current, context, group):
    plan = self.plans[group]
    hidden = inference_inference_requantize(self._conv_a_features(current, group), 1, -self.model.activation_bound, self.model.activation_bound)
    shift, past, scale, spm = context
    if scale is not None:
        hidden = inference_inference_requantize(hidden * (16 + scale.squeeze(-1).transpose(1, 2)), 4, -self.model.activation_bound, self.model.activation_bound)
    position_index = plan.h_positions[:, 0] * self.patch + plan.h_positions[:, 1]
    past = past.flatten(2)[:, :, position_index].transpose(1, 2)
    if spm is not None:
        past = inference_inference_requantize(past + spm.flatten(2)[:, :, position_index].transpose(1, 2), 0, -self.model.activation_bound, self.model.activation_bound)
    hidden = inference_inference_requantize(hidden + shift.squeeze(-1).transpose(1, 2) + past, 0, -self.model.activation_bound, self.model.activation_bound)
    hidden = F.relu(hidden)
    hidden = inference_inference_requantize(self._depthwise(hidden, plan.b1_gather, self.model.conv_b1), 3, -self.model.activation_bound, self.model.activation_bound)
    hidden = F.relu(hidden)
    hidden = inference_inference_requantize(self._depthwise(hidden, plan.b2_gather, self.model.conv_b2), 3, -self.model.activation_bound, self.model.activation_bound)
    hidden = F.relu(hidden)
    head = self.model.head
    weight, _, _ = head.codes()
    logits = F.linear(hidden, weight[:, :, 0, 0])
    logits = inference_inference_requantize(self._apply_codes(logits, head), 3, -32768, 32767)
    logits = logits / 8.0
    logits = logits.reshape(-1, self.model.num_classes)
    return logits[plan.output_order]

def inference_conv_a_plans(sparse):
    device = sparse.plans[0].targets.device
    offsets = torch.tensor(sparse.a_offsets, dtype=torch.long, device=device)
    patch_indices = torch.arange(sparse.patch_count, dtype=torch.long, device=device)
    patch_rows = patch_indices // sparse.patch_cols
    patch_columns = patch_indices % sparse.patch_cols
    plans = []
    for plan in sparse.plans:
        local_rows = plan.h_positions[:, 0, None] + offsets[None, :, 0]
        local_columns = plan.h_positions[:, 1, None] + offsets[None, :, 1]
        valid = (local_rows >= 0) & (local_rows < sparse.patch) & (local_columns >= 0) & (local_columns < sparse.patch)
        global_rows = patch_rows[:, None, None] * sparse.patch + local_rows[None]
        global_columns = patch_columns[:, None, None] * sparse.patch + local_columns[None]
        class_indices = global_rows * (sparse.patch_cols * sparse.patch) + global_columns
        class_indices = torch.where(valid[None], class_indices, torch.zeros_like(class_indices))
        rows = (local_rows - sparse.patch // 2).to(torch.float32)
        columns = (local_columns - sparse.patch // 2).to(torch.float32)
        coordinates = torch.stack([rows, columns], dim=1)
        coordinates = coordinates * valid[:, None]
        plans.append(inference_ConvAPlan(class_indices=class_indices, valid=valid[None, :, None, :], coordinates=coordinates[None]))
    return tuple(plans)

def inference_optimize_sparse_evaluator(sparse):
    model = sparse.model
    inference_freeze_model_codes(model)
    global prior_ste_round, prior_requantize
    prior_ste_round = inference_inference_round
    prior_requantize = inference_inference_requantize
    conv_a = model.conv_a
    conv_a_weight, _, _ = conv_a.codes()
    active = conv_a.mask[0, 0].to(torch.bool).flatten()
    conv_a_weight = conv_a_weight.flatten(2)[:, :, active].reshape(conv_a_weight.shape[0], -1)
    depthwise_weights = {}
    affine_values = {}
    for module in (model.conv_a, model.conv_b1, model.conv_b2, model.head):
        weight, bias, exponent = module.codes()
        multiplier = None if exponent is None else torch.pow(2.0, exponent).view(1, 1, -1)
        affine_values[id(module)] = (bias.view(1, 1, -1), multiplier)
        if module in (model.conv_b1, model.conv_b2):
            active = module.mask[0, 0].to(torch.bool).flatten()
            depthwise_weights[id(module)] = weight.flatten(2)[:, 0, active].t().view(1, 1, -1, weight.shape[0])
    sparse._sparse_cache = inference_SparseCache(conv_a_weight=conv_a_weight, conv_a_plans=inference_conv_a_plans(sparse), depthwise_weights=depthwise_weights, affine_values=affine_values, depthwise_zero=torch.zeros(sparse.patch_count, 1, model.ch, dtype=conv_a_weight.dtype, device=conv_a_weight.device))
    sparse._apply_codes = MethodType(inference_apply_codes, sparse)
    sparse._conv_a_features = MethodType(inference_conv_a_features, sparse)
    sparse._depthwise = MethodType(inference_depthwise, sparse)
    sparse.selected_logits = MethodType(inference_selected_logits, sparse)
prior_mixer_HEADER = struct.Struct('<4sBBBBHIII')
prior_mixer_WEIGHT_Q = 65536

def prior_mixer_bucket(value):
    return 3 if value is None else 0 if value < 0 else 1 if value == 0 else 2

class prior_mixer_Features:

    def __init__(self, family):
        self.family = family
        self.base = adaptive_AdaptiveExperts()
        self.banks = [{} for _ in range(15)]
        self.history = defaultdict(list)
        self.group = -1
        self.width = None
        self.sibling = None
        self.current = []
        self.depth = 0

    def start_row(self, count, depth):
        if count != self.width:
            self.group += 1
            self.width = count
            self.sibling = None
        self.depth = depth
        self.current = []

    def predict(self, pos, node, bitpos):
        d = self.depth
        h = self.history[d]
        prev, prev2 = (h[-1] if h else None, h[-2] if len(h) > 1 else None)
        sib = self.sibling[pos] if self.sibling is not None else None
        ps, p2s, ss = (prior_mixer_bucket(prev), prior_mixer_bucket(prev2), prior_mixer_bucket(sib))
        quart = min(3, 4 * pos // self.width)
        keys = self.base.keys(d, node, bitpos, prev)
        x = [adaptive_stretch(p) for p in self.base.predictions(keys)]
        common = [(d, node)] * 5
        if self.family == 1:
            extra = [(d, node, quart), (d, node, self.group), (d, node, pos % 3), (d, node, pos % 8), (d, node, ss), (d, node, prior_mixer_bucket(self.current[-1]) if pos else 3), (bitpos, node), (d, node, prev, quart), (d, node, prev, self.group), (d, node, sib)]
        for j, key in enumerate(common + extra):
            bank = self.banks[j]
            if j == 4 or (self.family == 2 and j == 10):
                z, n = bank.get(key, (0, 0))
                p = max(1, min(4095, (2 * z + 1) * 4096 // (2 * n + 2)))
            else:
                p = bank.get(key, 2048)
            x.append(adaptive_stretch(p))
        x.append(4096)
        return (x, keys, common + extra)

    def update(self, keys, extra, bit):
        self.base.update(keys, bit)
        for j, key in enumerate(extra):
            bank = self.banks[j]
            if j == 4 or (self.family == 2 and j == 10):
                z, n = bank.get(key, (0, 0))
                bank[key] = (z + int(bit == 0), n + 1)
            else:
                p = bank.get(key, 2048)
                bank[key] = adaptive_updated_probability(p, bit, (3, 4, 6, 7)[j] if j < 4 else 5)

    def symbol(self, value):
        self.current.append(value)
        self.history[self.depth].append(value)

    def finish_row(self):
        self.sibling = self.current

def prior_mixer_walk(counts, depths, family, weights, learning_shift, *, payload):
    decoder = adaptive_RangeDecoder(payload)
    state = prior_mixer_Features(family)
    w = [int(v) * 2048 for v in weights.tolist()]
    result = []
    for ri, (count, depth) in enumerate(zip(counts, depths.tolist(), strict=True)):
        state.start_row(count, int(depth))
        row = []
        for pos in range(count):
            unsigned, node = (0, 1)
            for bitpos, shift in enumerate(reversed(range(int(depth)))):
                x, keys, extra = state.predict(pos, node, bitpos)
                p = adaptive_squash(adaptive_round_div_signed(sum((a * b for a, b in zip(w, x, strict=True))), prior_mixer_WEIGHT_Q))
                bit = decoder.decode_bit(p)
                state.update(keys, extra, bit)
                unsigned = unsigned << 1 | bit
                node = 2 * node + bit
            sign = 1 << int(depth) - 1 if depth else 0
            value = unsigned - (1 << int(depth)) if depth and unsigned >= sign else unsigned
            state.symbol(value)
            row.append(value)
        state.finish_row()
        result.append(np.asarray(row, dtype=np.int16))
    return result

def prior_mixer_restore_prior(rider, counts):
    magic, version, family, nweights, flags, nrows, prefix_n, payload_n, tail_n = prior_mixer_HEADER.unpack_from(rider)
    off = prior_mixer_HEADER.size
    prefix = rider[off:off + prefix_n]
    off += prefix_n
    depths = adaptive_depths(prefix, len(counts))
    rate = rider[off]
    weights = np.frombuffer(rider[off + 1:off + 25], dtype=np.int8)
    off += 25
    rows = prior_mixer_walk(counts, depths, family, weights, rate, payload=rider[off:off + payload_n])
    return prefix + adaptive_pack_rows(rows, depths) + rider[off + payload_n:]
shared_N, shared_H, shared_W, shared_K, shared_F = (600, 384, 512, 5, 7)
shared_TOTAL, shared_Q, shared_SCALE = (1 << 31, 1024, 32)
shared_LEVELS = dict(spatial2=36, spatial3=216, previous=6, run=64, rowband=12)
shared_Y, shared_X = np.indices((shared_H, shared_W))
shared_GROUP = (shared_X % 64 + 2 * (shared_Y % 64)).reshape(-1)
shared_ALL = np.arange(shared_H * shared_W)

def shared_frequencies(rows):
    values = np.asarray(rows, dtype=np.float32).astype(np.float64)
    freq = np.maximum((values * shared_TOTAL).astype(np.int64), 1)
    winner = values.argmax(axis=1)
    freq[np.arange(len(freq)), winner] += shared_TOTAL - freq.sum(axis=1)
    return freq

def shared_log2_fixed(value):
    if _CORRECTOR_LIBRARY is not None:
        values = np.ascontiguousarray(value, dtype=np.float64)
        output = np.empty(values.shape, dtype=np.int16)
        _CORRECTOR_LIBRARY.mixer_log2(values, output, values.size)
        return output
    value = np.asarray(value, dtype=np.float64)
    mantissa, exponent = np.frexp(value)
    work = mantissa * 2.0
    code = (exponent.astype(np.int64) - 1) * (shared_Q * 2)
    fraction = np.zeros(value.shape, dtype=np.int64)
    for _ in range(11):
        work = work * work
        bit = work >= 2.0
        work = np.where(bit, work * 0.5, work)
        fraction = fraction * 2 + bit
    result = (code + fraction + 1) // 2
    return result.astype(np.int16)

def shared_neighbour_map(dy, dx):
    yy, xx = (shared_Y + dy, shared_X + dx)
    valid = (yy >= 0) & (yy < shared_H) & (xx >= 0) & (xx < shared_W)
    source = (np.clip(yy, 0, shared_H - 1) * shared_W + np.clip(xx, 0, shared_W - 1)).reshape(-1)
    valid = valid.reshape(-1) & (shared_GROUP[source] < shared_GROUP)
    return (source, valid)
shared_NEIGHBOURS = {shared_offset: shared_neighbour_map(*shared_offset) for shared_offset in [(0, -shared_i) for shared_i in range(1, 8)] + [(-1, 0), (-1, 1)]}

def shared_contexts(plane, previous, run, positions=shared_ALL):
    flat = np.asarray(plane).reshape(-1)
    positions = np.asarray(positions, dtype=np.int64)

    def neighbour(offset):
        source, valid = shared_NEIGHBOURS[offset]
        return np.where(valid[positions], flat[source[positions]], shared_K).astype(np.int64)
    left, up, upright = (neighbour((0, -1)), neighbour((-1, 0)), neighbour((-1, 1)))
    spatial_run = np.zeros(len(positions), dtype=np.int64)
    active = left != shared_K
    for distance in range(1, 8):
        value = neighbour((0, -distance))
        active &= (value != shared_K) & (value == left)
        spatial_run += active
    coloc = np.full(len(positions), shared_K, dtype=np.int64) if previous is None else np.asarray(previous).reshape(-1)[positions].astype(np.int64)
    return dict(spatial2=left * 6 + up, spatial3=(left * 6 + up) * 6 + upright, previous=coloc, run=spatial_run * 8 + np.minimum(run.reshape(-1)[positions], 7), rowband=positions // shared_W // 32)

def shared_power_table():
    codes = np.arange(shared_Q * shared_SCALE, dtype=np.int64)
    values = np.ones(len(codes), dtype=np.float64)
    radical = 2.0
    for bit in range(14, -1, -1):
        radical = float(np.sqrt(radical))
        values *= np.where(codes >> bit & 1, radical, 1.0)
    return values
shared_POW2 = shared_power_table()

def shared_mix_probabilities(freq, phi, weights, original_rows=None):
    if _CORRECTOR_LIBRARY is not None:
        output = np.empty(freq.shape, dtype=np.float32)
        _CORRECTOR_LIBRARY.mixer_probability(np.ascontiguousarray(freq), np.ascontiguousarray(phi),
            np.ascontiguousarray(weights), shared_POW2, output, len(freq), phi.shape[2])
        return output
    winner = freq.argmax(axis=1)
    exponent = np.sum(phi.astype(np.int64) * weights[winner, None, :].astype(np.int64), axis=2)
    exponent -= exponent.max(axis=1, keepdims=True)
    integer, fraction = np.divmod(exponent, shared_Q * shared_SCALE)
    factor = np.ldexp(shared_POW2[fraction], integer.astype(np.int32))
    raw = freq.astype(np.float64) / shared_TOTAL * factor
    raw /= raw.sum(axis=1, keepdims=True)
    output = np.maximum(raw, np.finfo(np.float32).tiny).astype(np.float32)
    inactive = ~np.any(weights[winner], axis=1)
    return output

class shared_SharedMixer:

    def __init__(self, weight_bytes):
        self.weights = np.frombuffer(weight_bytes, dtype=np.int8).reshape(shared_K, shared_F).copy()
        self.frame = 0
        self.counts = {name: np.zeros((shared_K * levels, shared_K), dtype=np.int64) for name, levels in shared_LEVELS.items()}
        self.expected = {name: np.zeros((shared_K * levels, shared_K), dtype=np.int64) for name, levels in shared_LEVELS.items()}
        self.run = np.zeros((shared_H, shared_W), dtype=np.uint8)
        self.base = np.empty((shared_H * shared_W, shared_K), dtype=np.int64)
        self.seen = np.zeros(shared_H * shared_W, dtype=bool)
        self.tables = None

    def begin_frame(self):
        self.seen.fill(False)
        self.tables = {}
        for name in shared_LEVELS:
            ratio = (self.counts[name].astype(np.float64) + 0.5) / (self.expected[name].astype(np.float64) / shared_TOTAL + 0.5)
            self.tables[name] = shared_log2_fixed(np.clip(ratio, 1 / 16, 16))

    def features(self, rows, positions, plane, previous):
        positions = np.asarray(positions, dtype=np.int64)
        freq = shared_frequencies(rows)
        arg = freq.argmax(axis=1)
        context = shared_contexts(plane, previous if self.frame else None, self.run, positions)
        phi = np.zeros((len(rows), shared_K, shared_F), dtype=np.int16)
        for j, (name, levels) in enumerate(shared_LEVELS.items()):
            phi[:, :, j] = self.tables[name][arg * levels + context[name]]
        phi[:, :, 5] = shared_log2_fixed(freq.astype(np.float64) / shared_TOTAL)
        phi[np.arange(len(rows)), arg, 6] = shared_Q
        self.base[positions] = freq
        self.seen[positions] = True
        return (phi, freq)

    def coding(self, rows, positions, plane, previous):
        phi, freq = self.features(rows, positions, plane, previous)
        return shared_mix_probabilities(freq, phi, self.weights, rows)

    def end_frame(self, plane, previous):
        truth = np.asarray(plane, dtype=np.uint8).reshape(-1)
        context = shared_contexts(plane, previous if self.frame else None, self.run)
        arg = self.base.argmax(axis=1)
        for name, levels in shared_LEVELS.items():
            code = arg * levels + context[name]
            self.counts[name] += np.bincount(code * shared_K + truth, minlength=levels * shared_K * shared_K).reshape(levels * shared_K, shared_K)
            for k in range(shared_K):
                self.expected[name][:, k] += np.bincount(code, weights=self.base[:, k], minlength=levels * shared_K).astype(np.int64)
        self.run = np.zeros((shared_H, shared_W), dtype=np.uint8) if not self.frame else np.where(np.asarray(plane).reshape(shared_H, shared_W) == np.asarray(previous).reshape(shared_H, shared_W), np.minimum(self.run + 1, 7), 0).astype(np.uint8)
        self.frame += 1
        self.tables = None
geometry_mixer_FORMAT = struct.Struct('<BHH8B6B')
geometry_mixer_LaneGeometry = NativeGeometry if _GEOMETRY_LIBRARY is not None else CausalGeometry
geometry_mixer_parse_config = parse_geometry_config
geometry_mixer_K, geometry_mixer_H, geometry_mixer_W, geometry_mixer_BINS = (5, 384, 512, 9)
geometry_mixer_TOTAL, geometry_mixer_Q = (shared_TOTAL, shared_Q)

def geometry_mixer_unpack_rider(payload):
    length = 1 + 40 + geometry_mixer_FORMAT.size
    config, stream = (bytes([payload[4] & 15]) + payload[5:4 + length], payload[4 + length:])
    geometry_mixer_parse_config(config[41:])
    return (config, stream)

class geometry_mixer_LaneMixer:

    def __init__(self, config):
        self.config = bytes(config)
        self.variant = config[0]
        geometry_mixer_parse_config(config[41:])
        raw = np.frombuffer(config[1:41], dtype=np.int8).copy()
        self.old = shared_SharedMixer(raw[:35].tobytes())
        self.weights = raw[35:].reshape(geometry_mixer_K, 1)
        self.counts = np.zeros((geometry_mixer_K * geometry_mixer_BINS, geometry_mixer_K), dtype=np.int64)
        self.expected = np.zeros_like(self.counts)
        self.base = np.empty((geometry_mixer_H * geometry_mixer_W, geometry_mixer_K), dtype=np.int64)
        self.bins = np.empty(geometry_mixer_H * geometry_mixer_W, dtype=np.uint8)
        self.seen = np.zeros(geometry_mixer_H * geometry_mixer_W, dtype=bool)
        self.geometry = None
        self.table = None
        self.pending = None

    @property
    def frame(self):
        return self.old.frame

    def begin_frame(self):
        self.old.begin_frame()
        ratio = (self.counts + 0.5) / (self.expected.astype(np.float64) / geometry_mixer_TOTAL + 0.5)
        self.table = shared_log2_fixed(np.clip(ratio, 1 / 16, 16))
        self.geometry = None
        self.pending = None
        self.seen.fill(False)

    def features(self, rows, positions, plane, previous):
        positions = np.asarray(positions, dtype=np.int64)
        if self.geometry is None:
            self.geometry = geometry_mixer_LaneGeometry(previous, self.config[41:])
        original = self.old.coding(rows, positions, plane, previous)
        freq = shared_frequencies(original)
        phi = np.empty((len(rows), geometry_mixer_K, 1), dtype=np.int16)
        bins = self.geometry.contexts(positions)
        arg = freq.argmax(axis=1)
        extra = self.table[arg * geometry_mixer_BINS + bins]
        extra[bins == 8] = 0
        phi[:, :, -1] = extra
        self.bins[positions] = bins
        self.base[positions] = freq
        self.seen[positions] = True
        self.pending = positions.copy()
        return (phi, freq, original)

    def coding(self, rows, positions, plane, previous):
        phi, freq, original = self.features(rows, positions, plane, previous)
        result = shared_mix_probabilities(freq, phi, self.weights, original)
        inactive = np.all(phi[:, :, 0] == 0, axis=1)
        result[inactive] = original[inactive]
        return result

    def observe(self, positions, symbols):
        self.geometry.observe(positions, symbols)
        self.pending = None

    def end_frame(self, plane, previous):
        truth = np.asarray(plane, dtype=np.uint8).reshape(-1)
        np.testing.assert_array_equal(self.geometry.plane.reshape(-1), truth)
        code = self.base.argmax(axis=1) * geometry_mixer_BINS + self.bins
        self.counts += np.bincount(code * geometry_mixer_K + truth, minlength=geometry_mixer_K * geometry_mixer_BINS * geometry_mixer_K).reshape(-1, geometry_mixer_K)
        for k in range(geometry_mixer_K):
            self.expected[:, k] += np.bincount(code, weights=self.base[:, k], minlength=geometry_mixer_K * geometry_mixer_BINS).astype(np.int64)
        self.old.end_frame(plane, previous)
        self.table = None
        self.geometry = None
archive_NUM_CLASSES = 5
archive_FIXED_SCHEMA = 'fixed_boundary_int6'
archive_FIXED_MAGIC = bytes.fromhex('52434631')
archive_FIXED_STATES = 25
archive_FIXED_BITS = 6
archive_SPARSE_SELECTOR_PREFIX = bytes.fromhex('4630453101')
archive_MODEL_HEADER = struct.Struct('<4sBBBBHHH')
archive_XZ_CODEC = 1

def archive_decompress_brotli(stream):
    if brotli is not None:
        return brotli.decompress(stream)

def archive_unpack_unsigned(raw, count, bits):
    output = np.empty(count, dtype=np.int16)
    for index in range(count):
        offset = index * bits
        byte, shift = divmod(offset, 8)
        word = raw[byte]
        if byte + 1 < len(raw):
            word |= raw[byte + 1] << 8
        output[index] = word >> shift & (1 << bits) - 1
    return output

@dataclass(frozen=True)
class archive_QuantizedTable:
    name: str
    bits: int
    codes: np.ndarray
    scale: float
    values: np.ndarray

@dataclass(frozen=True)
class archive_ResidualArchiveParts:
    semantic_blob: bytes
    carrier_blob: bytes
    prior_blob: bytes
    token_stream: bytes
    table: archive_QuantizedTable
    schema: str
    residual_payload: bytes
    compressed_models: bytes
    compensation_blob: bytes | None = None
    token_codec: str = 'rc64'
    mixer_parameters: bytes | None = None

def archive_decode_fixed_table(raw):
    expected = len(archive_FIXED_MAGIC) + 2 + bits_packed_length(archive_FIXED_STATES * archive_NUM_CLASSES, archive_FIXED_BITS)
    scale = float(np.frombuffer(raw[4:6], dtype='<f2')[0])
    codes = np.asarray(bits_unpack_signed(raw[6:], archive_FIXED_STATES * archive_NUM_CLASSES, archive_FIXED_BITS), dtype=np.int8).reshape(archive_FIXED_STATES, archive_NUM_CLASSES)
    return archive_QuantizedTable(name='boundary_predicted', bits=archive_FIXED_BITS, codes=codes, scale=scale, values=codes.astype(np.float32) * scale)

def archive_read_residual_archive(archive_path):
    with zipfile.ZipFile(archive_path) as archive:
        outer = archive.read('p')
    model_sections = archive_decode_model_sections(outer)
    if model_sections is not None:
        semantic, carrier, prior, compensation, section, compressed = model_sections
    fixed_size = len(archive_FIXED_MAGIC) + 2 + bits_packed_length(archive_FIXED_STATES * archive_NUM_CLASSES, archive_FIXED_BITS)
    compact_size = fixed_size - len(archive_FIXED_MAGIC)
    residual = archive_FIXED_MAGIC + section[:compact_size]
    table = archive_decode_fixed_table(residual)
    tokens = section[compact_size:]
    mixer_parameters = None
    if model_sections is not None and outer[7] & 128:
        mixer_parameters, tokens = geometry_mixer_unpack_rider(tokens)
    return archive_ResidualArchiveParts(semantic_blob=semantic, carrier_blob=carrier, prior_blob=prior, token_stream=tokens, mixer_parameters=mixer_parameters, table=table, schema=archive_FIXED_SCHEMA, residual_payload=residual, compressed_models=compressed, compensation_blob=compensation)

def archive_boundary_buckets(previous, max_distance=4):
    edge = np.zeros(previous.shape, dtype=bool)
    edge[1:] |= previous[1:] != previous[:-1]
    edge[:-1] |= previous[:-1] != previous[1:]
    edge[:, 1:] |= previous[:, 1:] != previous[:, :-1]
    edge[:, :-1] |= previous[:, :-1] != previous[:, 1:]
    result = np.full(previous.shape, max_distance, dtype=np.uint8)
    active = edge.copy()
    result[active] = 0
    for distance in range(1, max_distance):
        grown = active.copy()
        grown[1:] |= active[:-1]
        grown[:-1] |= active[1:]
        grown[:, 1:] |= active[:, :-1]
        grown[:, :-1] |= active[:, 1:]
        active = grown
        result[(result == max_distance) & active] = distance
    return result

def archive_probability_table(logits, precision):
    quantized = np.clip(np.rint(np.asarray(logits, dtype=np.float32) * precision), -32768, 32767).astype(np.int16)
    values = quantized.astype(np.float32) / precision
    values = values.astype(np.float64)
    values -= values.max(axis=1, keepdims=True)
    probabilities = np.exp(values)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities.astype(np.float32)
archive_RENDERER_PLANES = 2
archive_ARITHMETIC_BASIS = 8
archive_ARITHMETIC_COEFFICIENTS = 16
archive_PRIOR_ADAPTIVE = 64

def archive_interleave_bytes(body):
    span = len(body) & ~1
    half = span // 2
    restored = np.empty(span, dtype=np.uint8)
    planes = np.frombuffer(body[:span], dtype=np.uint8)
    restored[0::2] = planes[:half]
    restored[1::2] = planes[half:]
    return restored.tobytes() + body[span:]

def load_corrector_library():
    """Bind the local accelerator; missing or incompatible libraries use Python."""
    try:
        lib = ctypes.CDLL(str(Path(__file__).with_name('corrector.so').resolve()))
        array = lambda dtype: np.ctypeslib.ndpointer(dtype=dtype, flags='C_CONTIGUOUS')
        handle, size = ctypes.c_void_p, ctypes.c_int64
        signatures = {
            'corrector_create': ([size], handle),
            'corrector_destroy': ([handle], None),
            'corrector_begin_frame': ([handle, array(np.int64), size], ctypes.c_int),
            'corrector_group_state': ([handle, array(np.float32), array(np.int64), array(np.int64), size], ctypes.c_int),
            'corrector_coding_row': ([handle, array(np.float32), size], ctypes.c_int),
            'corrector_observe': ([handle, array(np.int64), size], ctypes.c_int),
            'corrector_end_frame': ([handle, array(np.uint8), size], ctypes.c_int),
            'mixer_log2': ([array(np.float64), array(np.int16), size], None),
            'mixer_probability': ([array(np.int64), array(np.int16), array(np.int8), array(np.float64), array(np.float32), size, ctypes.c_int], None),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(lib, name)
            function.argtypes, function.restype = arguments, result
        return lib
    except (OSError, AttributeError):
        print('Native corrector unavailable; using the identical Python corrector and mixer.', file=sys.stderr)
        return None

_CORRECTOR_LIBRARY = load_corrector_library()

class NativeCorrector:
    """Ordered float64 implementation of the fixed 23-family corrector."""

    def __init__(self, plane):
        if plane != 384 * 512:
            raise ValueError('corrector requires a 384 by 512 plane')
        expected = {'families': ('shipped_joint', 'temporal_spatial', 'surprise_only', 'spatial_surprise', 'spatial_boundary', 'run_surprise', 'boundary_surprise', 'temporal_surprise', 'shipped_fast256', 'shipped_fast4096', 'surprise_fast256', 'spatial4_surprise', 'homog_surprise', 'homog_boundary_surprise', 'spatial4_boundary', 'homog_spatial4', 'spatial4_temporal', 'homog_surprise_fast256', 'spatial4_surprise_fast256', 'groupbin8_surprise', 'cls_groupbin8', 'patch192_only', 'tile48_groupbin8'), 'mixer_context': 'cls_boundary_agree_homog_ubin8', 'count_buckets': 1, 'sse_context': 'off', 'sse_learn_weight': False, 'normalize': True, 'learn': True, 'within_miss': True, 'miss_cell': 'nb3_prev1', 'miss_min_count': 1, 'miss_clamp': 16.0}
        if dict(miss_SHIPPED_CONFIG) != expected:
            raise ValueError('Python corrector configuration differs from the native definition')
        constants = (counts_NUM_CLASSES, counts_U_BINS, counts_RUN_LEVELS, counts_BOUNDARY_LEVELS,
            counts_MIN_COUNT, odds_POWER_BITS, odds_INT_POWER_BITS, odds_WEIGHT_STORE_BITS,
            odds_SPATIAL_LEVELS, context_SPATIAL4_LEVELS, context_HOMOGENEITY_LEVELS,
            odds_LR_BASE_SHIFT + 4, context_GROUP_BINS)
        if constants != (5, 64, 8, 5, 32, 6, 4, 20, 5, 6, 5, 24, 8):
            raise ValueError('Python corrector constants differ from the native definition')
        self.library = _CORRECTOR_LIBRARY
        self.handle = self.library.corrector_create(plane)
        if not self.handle:
            raise MemoryError('native corrector allocation or arithmetic check failed')
        self.serial = 0
        self.pending = None

    def close(self):
        if getattr(self, 'handle', None):
            self.library.corrector_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()

    def begin_frame(self, boundary):
        values = np.ascontiguousarray(boundary, dtype=np.int64).reshape(-1)
        self._call('begin_frame', values, len(values))
        self.pending = None

    def _call(self, name, *args):
        if not self.handle or getattr(self.library, 'corrector_' + name)(self.handle, *args):
            raise RuntimeError('native corrector failed at ' + name)

    def group_state(self, probability, predicted, positions):
        rows = np.ascontiguousarray(probability, dtype=np.float32)
        predicted = np.ascontiguousarray(predicted, dtype=np.int64).reshape(-1)
        positions = np.ascontiguousarray(positions, dtype=np.int64).reshape(-1)
        if rows.ndim != 2 or rows.shape[1] != 5 or len(predicted) != len(rows) or len(positions) != len(rows):
            raise ValueError('invalid corrector group shape')
        self._call('group_state', rows, predicted, positions, len(rows))
        self.serial += 1
        self.pending = (self, self.serial, len(rows))
        return self.pending

    def coding_row(self, state):
        if state is not self.pending:
            raise ValueError('stale corrector group')
        output = np.empty((state[2], 5), dtype=np.float32)
        self._call('coding_row', output, len(output))
        return output

    def observe(self, state, symbols):
        if state is not self.pending:
            raise ValueError('stale corrector group')
        decoded = np.ascontiguousarray(symbols, dtype=np.int64).reshape(-1)
        if len(decoded) != state[2] or np.any(decoded < 0) or np.any(decoded >= 5):
            raise ValueError('invalid decoded symbols')
        self._call('observe', decoded, len(decoded))
        self.pending = None

    def end_frame(self, tokens):
        values = np.ascontiguousarray(tokens, dtype=np.uint8).reshape(-1)
        self._call('end_frame', values, len(values))


def make_corrector(plane):
    if _CORRECTOR_LIBRARY is not None:
        try:
            return NativeCorrector(plane)
        except (MemoryError, ValueError) as error:
            print('Native corrector unavailable; using the identical Python corrector: ' + str(error), file=sys.stderr)
    return miss_FreeCorrector(plane)

class TokenDecoder:
    """Decode the next class plane using only already decoded symbols."""

    def __init__(self, parts, device):
        self.device = device
        self.parts = parts
        self.model = render_load_prior(parts.prior_blob, device)
        self.sparse = sparse_SparseIntegerPrior(self.model, render_EVAL_H, render_EVAL_W)
        inference_optimize_sparse_evaluator(self.sparse)
        self.corrector = make_corrector(render_EVAL_H * render_EVAL_W)
        self.range_decoder = ArithmeticDecoder(parts.token_stream)
        self.mixer = geometry_mixer_LaneMixer(parts.mixer_parameters)
        self.plans = []
        for mask in render_group_masks(device):
            positions = np.flatnonzero(mask.cpu().numpy().reshape(-1))
            self.plans.append((torch.from_numpy(positions).to(device), positions))
        self.previous = torch.zeros((1, render_EVAL_H, render_EVAL_W), dtype=torch.long, device=device)
        self.frame = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.frame == render_N:
            raise StopIteration
        index = torch.tensor([self.frame], dtype=torch.long, device=self.device)
        current = torch.zeros_like(self.previous)
        context = self.model.prepare_frame_context(index, self.previous)
        previous_field = None if self.frame == 0 else self.previous[0].cpu().to(torch.uint8).numpy()
        boundary = np.full(render_EVAL_H * render_EVAL_W, 4, dtype=np.uint8) if previous_field is None else archive_boundary_buckets(previous_field).reshape(-1)
        self.corrector.begin_frame(boundary)
        self.mixer.begin_frame()
        for group, (device_positions, positions) in enumerate(self.plans):
            logits = self.sparse.selected_logits(current, context, group).cpu().numpy()
            predicted = logits.argmax(axis=1).astype(np.int64)
            feature = boundary[positions].astype(np.int64) * 5 + predicted
            probability = archive_probability_table(logits + self.parts.table.values[feature], render_Prior_LOGIT_PRECISION)
            state = self.corrector.group_state(probability, predicted, positions)
            coding = self.corrector.coding_row(state)
            partial = current[0].cpu().to(torch.uint8).numpy()
            coding = self.mixer.coding(coding, positions, partial, previous_field)
            symbols = self.range_decoder.decode(coding).astype(np.int64)
            self.mixer.observe(positions, symbols)
            self.corrector.observe(state, symbols)
            current.reshape(-1)[device_positions] = torch.from_numpy(symbols).to(self.device)
        tokens = current[0].cpu().to(torch.uint8)
        self.corrector.end_frame(tokens.numpy().reshape(-1))
        self.mixer.end_frame(tokens.numpy(), previous_field)
        self.previous = current
        self.frame += 1
        return tokens

def write_video(model, basis, coefficients, tokens, selector_blob, destination, device):
    render_render_video(model, basis, coefficients, tokens, destination, device)
    modes, indices = selector_decode_selector(selector_blob)
    raw = np.memmap(destination, mode='r+', dtype=np.uint8, shape=(1200, 874, 1164, 3))
    for mode_index, mode in enumerate(modes):
        frame_ids = np.flatnonzero(indices == mode_index)
        if frame_ids.size:
            raw[2 * frame_ids] = selector_apply_pixel_mode(np.asarray(raw[2 * frame_ids]).copy(), mode)
    raw.flush()

def direct_carrier_layout(body):
    if len(body) < 142:
        raise ValueError('truncated packed carrier metadata')
    basis_bits = int.from_bytes(body[:3], 'little')
    coefficient_bits = int.from_bytes(body[3:6], 'little')
    basis_end = 142 + (basis_bits + 7) // 8
    coefficient_end = basis_end + (coefficient_bits + 7) // 8
    if coefficient_end > len(body):
        raise ValueError('truncated packed carrier streams')
    return (basis_bits, basis_end, coefficient_end)

def decode_carrier(body):
    """Decode counted spatial bases and their signed 12-bit AR(1) trajectory."""
    basis_bits, basis_end, coefficient_end = direct_carrier_layout(body)
    symbols = basis_decode_basis_arith(body[142:basis_end], basis_bits)
    basis_codes = symbols >> 1 ^ -(symbols & 1)
    factors = body[102] + archive_unpack_unsigned(body[103:114], 12, 7)
    biases = archive_unpack_unsigned(body[114:123], 12, 6)
    biases = np.where(biases >= 32, biases - 64, biases).astype(np.int16)
    parameters = body[139] + archive_unpack_unsigned(body[140:142], 12, 1)
    symbols = coefficients_cabac_decode(body[basis_end:coefficient_end], parameters)
    residuals = symbols >> 1 ^ -(symbols & 1)
    codes = np.empty((600, 12), dtype=np.int32)
    codes[0] = residuals[0]
    for frame in range(1, 600):
        products = codes[frame - 1].astype(np.int64) * factors.astype(np.int64)
        rounded = np.where(products >= 0, (products + 128) // 256, -((-products + 128) // 256))
        prediction = (rounded + biases + 2048 & 4095) - 2048
        codes[frame] = (prediction + residuals[frame] + 2048 & 4095) - 2048
    scales = np.frombuffer(body[6:54], dtype='<f4').copy()
    coefficient_scales = np.frombuffer(body[54:102], dtype='<f4').copy()
    basis = torch.from_numpy(basis_codes.astype(np.int8)).reshape(12, 3, 24, 32).float()
    basis = basis * torch.from_numpy(scales)[:, None, None, None]
    coefficients = torch.from_numpy(codes).float() * torch.from_numpy(coefficient_scales)[None]
    selector, _ = overlay_split_selector_compensation(archive_SPARSE_SELECTOR_PREFIX + body[coefficient_end:])
    selector_decode_selector(selector)
    return (basis, coefficients, selector)

def archive_decode_model_sections(outer):
    """Keep the packed carrier bytes; decode them directly when loading models."""
    magic, version, codec, table_mode, reserved, prior_bytes, semantic_bytes, carrier_bytes = archive_MODEL_HEADER.unpack_from(outer)
    model_end = archive_MODEL_HEADER.size + prior_bytes + semantic_bytes + carrier_bytes
    offset = archive_MODEL_HEADER.size
    prior_stream = outer[offset:offset + prior_bytes]
    offset += prior_bytes
    semantic_stream = outer[offset:offset + semantic_bytes]
    offset += semantic_bytes
    carrier_stream = outer[offset:offset + carrier_bytes]
    prior = lzma.decompress(prior_stream, format=lzma.FORMAT_XZ) if codec == archive_XZ_CODEC else archive_decompress_brotli(prior_stream)
    if reserved & archive_PRIOR_ADAPTIVE:
        prior = archive_interleave_bytes(prior)
    semantic = archive_decompress_brotli(semantic_stream)
    if reserved & archive_RENDERER_PLANES:
        semantic = archive_interleave_bytes(semantic)
    required = archive_ARITHMETIC_BASIS | archive_ARITHMETIC_COEFFICIENTS
    if reserved & required != required:
        raise ValueError('expected arithmetic basis and binary-coded coefficients')
    carrier = archive_decompress_brotli(carrier_stream)
    _, _, coefficient_end = direct_carrier_layout(carrier)
    _, compensation = overlay_split_selector_compensation(archive_SPARSE_SELECTOR_PREFIX + carrier[coefficient_end:])
    return (semantic, carrier, prior, compensation, outer[model_end:], outer[:model_end])

def read_models(archive_path):
    parts = archive_read_residual_archive(archive_path)
    model = render_SemanticTokenRenderer(96)
    semantic_blob = weight_mixer_restore_semantic(parts.semantic_blob, model.state_dict())
    model.load_state_dict(weights_unpack_variant_semantic_or_none(semantic_blob, model.state_dict()), strict=True)
    basis, coefficients, selector_blob = decode_carrier(parts.carrier_blob)
    return (parts, model, basis, coefficients, selector_blob)

def prior_weight_counts(model):
    """Count the coded weights in the architecture's existing masked rows."""
    rows = []
    for module in model.modules():
        if isinstance(module, prior_IntegerConv2d):
            mask = module.mask.to(bool).expand_as(module.weight)
            rows.extend((int(mask[index].sum().item()) for index in range(module.weight.shape[0])))
        elif isinstance(module, prior_IntegerLinear):
            rows.extend((int(row.numel()) for row in module.weight))
    return rows

def render_load_prior(raw, device):
    model = prior_IntegerPrior(num_pairs=render_N, num_classes=render_NUM_CLASSES, patch=render_Prior_PATCH, delta=render_Prior_DELTA, channels=render_Prior_CHANNELS, frame_dim=render_Prior_FILM_DIM, norm_mode='none', activation='relu', use_frame_scale=True, weight_bound=127, activation_bound=127, use_weight_scales=True, weight_exponent_min=-6, use_spm=True, use_norm_gates=False).eval()
    prior_io_deserialize_self_compressed(model, prior_mixer_restore_prior(raw, prior_weight_counts(model)))
    return model.to(device)
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('file_list', type=Path)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    if args.file_list.read_text().split() != ['0.mkv']:
        raise SystemExit('this archive contains only 0.mkv')
    archive = Path(__file__).with_name('archive.zip')
    content = archive.read_bytes()
    if len(content) != 179286 or hashlib.sha256(content).hexdigest() != 'aab908d3b32c65582b7de1a3855b67a4f6fdb9373b0dc81e3049ee8164ec3957':
        raise SystemExit('archive.zip has an unexpected size or SHA-256')
    with zipfile.ZipFile(archive) as compressed:
        if compressed.namelist() != ['p'] or (args.archive_dir / 'p').read_bytes() != compressed.read('p'):
            raise SystemExit('extracted payload does not match archive.zip')
    if args.device == 'cuda' and (not torch.cuda.is_available()):
        raise SystemExit('CUDA is required by default; --device cpu explicitly selects CPU output, which may differ.')
    if args.device == 'cpu':
        print('CPU selected explicitly; output may differ from CUDA.', file=sys.stderr)
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260916)
    np.random.seed(20260916)
    if device.type == 'cuda':
        try:
            torch.backends.cuda.matmul.fp32_precision = 'ieee'
            torch.backends.cudnn.conv.fp32_precision = 'ieee'
        except (AttributeError, RuntimeError):
            torch.set_float32_matmul_precision('highest')
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        parts, model, basis, coefficients, selector = read_models(archive)
        tokens = torch.stack(list(TokenDecoder(parts, device)))
        write_video(model, basis, coefficients, tokens, selector, args.output_dir / '0.raw', device)
if __name__ == '__main__':
    main()
