"""SM1S: 24 counted shared int8 weights over causal SM3R bit predictors.

Integer-only receiver decisions reuse the shipped RC1 range coder and RC2
stretch/squash. All video-derived scales, codes, and mixer weights are counted.
"""
from __future__ import annotations

import math
import struct

import numpy as np

try:
    from . import rc1_adaptive_model_sections as rc1
    from . import rc2_hpac_semistatic_mixing as fixed
except ImportError:
    from experiments import ddm_rc1_adaptive_section_codec as rc1
    from experiments import ddm_rc2_hpac_semistatic_mixing_codec as fixed

MAGIC = b'SM1S'
HEADER = struct.Struct('<4sBBHI')
WEIGHT_COUNT = 24


def scale_buckets(blob):
    """Exact quartile partition using ordered positive fp16 representations."""
    scales = np.frombuffer(blob, dtype='<u2').astype(np.int64)
    if not len(scales) or np.any(scales >= 0x7c00):
        raise ValueError('SM1 requires finite nonnegative scales')
    ordered = np.sort(scales)
    buckets = np.zeros(len(scales), dtype=np.int16)
    for quartile in (1, 2, 3):
        position, remainder = divmod((len(scales) - 1) * quartile, 4)
        lo, hi = int(ordered[position]), int(ordered[min(position + 1, len(scales) - 1)])
        buckets += 4 * scales > (4 - remainder) * lo + remainder * hi
    return buckets.tolist()


def plan_metadata(metadata, template):
    if len(metadata) < 18 or metadata[:8] != b'SM3R\x01\x06\x01\x00':
        raise ValueError('SM1 covers SM3R version1 mode6 keep1 only')
    cursor = 10

    def read(kind, length):
        nonlocal cursor
        if kind == 'codes':
            return b''
        if cursor + length > len(metadata):
            raise ValueError('truncated SM1 metadata')
        value = metadata[cursor:cursor + length]
        cursor += length
        return value

    plan = rc1.walk_sm3r(read, template, tuple(metadata[4:8]))
    if cursor != len(metadata):
        raise ValueError('trailing SM1 metadata')
    names = [(name, value) for name, value in template.items() if value.ndim >= 2]
    descriptors = []
    for index, item in enumerate(plan):
        if item['kind'] != 'codes':
            continue
        name, tensor = names[len(descriptors)]
        shape = tuple(tensor.shape)
        kind = (0 if name == 'token_embed.weight' else 1 if name == 'frame_embed.weight'
                else 2 if name == 'coord_mix.weight' else 3 if '.dw.' in name
                else 4 if '.pw.' in name else 5 if '.film.' in name else 6)
        descriptors.append(dict(name=name, shape=shape, count=item['count'], bits=item['bits'],
                                cols=math.prod(shape[1:]), scales=scale_buckets(plan[index - 1]['blob']),
                                embedding=name.endswith('embed.weight'), kind=kind,
                                block=int(name.split('.')[1]) if name.startswith('blocks.') else 4))
        if name in rc1.ROW_PRUNE_NAMES:
            descriptors[-1]['selected_rows'] = np.flatnonzero(np.unpackbits(
                np.frombuffer(plan[index - 2]['blob'], dtype=np.uint8), bitorder='little')[:shape[0]]).tolist()
    return plan, descriptors


def split_body(body, template):
    if len(body) < 18:
        raise ValueError('truncated SM3R body')
    cursor = 10

    def read(_kind, length):
        nonlocal cursor
        if cursor + length > len(body):
            raise ValueError('truncated SM3R field')
        result = body[cursor:cursor + length]
        cursor += length
        return result

    plan = rc1.walk_sm3r(read, template, tuple(body[4:8]))
    if cursor != len(body):
        raise ValueError('SM3R body has trailing bytes')
    metadata = body[:10] + b''.join(i['blob'] for i in plan if i['kind'] != 'codes')
    groups = [rc1.unpack_signed_codes(i['blob'], i['count'], i['bits']) for i in plan if i['kind'] == 'codes']
    if any(rc1.pack_signed_codes(g, i['bits']) != i['blob']
           for g, i in zip(groups, [p for p in plan if p['kind'] == 'codes'], strict=True)):
        raise ValueError('noncanonical code padding')
    return metadata, groups


def bucket(value):
    return 3 if value is None else 0 if value < 0 else 1 if value == 0 else 2


class Predictors:
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
            sibling = (current[pos - desc['cols']] if row and desc['selected_rows'][row - 1] == original_row - 1
                       else 0 if original_row else None)
        left = current[-1] if col else None
        dw = None
        if desc['kind'] == 4:
            dw_codes = self.completed[desc['name'].replace('.pw.', '.dw.')]
            dw = dw_codes[col * 9 + 4]
        g, d = (group, node), (desc['bits'], node)
        return [g, g, g, g, g,
                (*g, scale), (*g, scale), (*d, scale), (*d, scale),
                (*g, quartile), (*d, quartile),
                (*g, sibling), (*g, bucket(sibling)), (*d, sibling), (*d, bucket(sibling)),
                (*g, bucket(left)), (*d, bucket(left)),
                (*g, dw), (*d, bucket(dw)),
                (desc['block'], desc['kind'], *d), (desc['kind'], *d),
                d, (*g, scale)]

    def predict(self, keys):
        output = []
        for j, key in enumerate(keys):
            if j in (4, 21, 22):
                zero, n = self.banks[j].get(key, (0, 0))
                p = min(4095, max(1, (2 * zero + 1) * 4096 // (2 * n + 2)))
            else:
                p = self.banks[j].get(key, 2048)
            output.append(fixed.stretch(p))
        return output + [4096]

    def update(self, keys, bit):
        for j, key in enumerate(keys):
            if j in (4, 21, 22):
                zero, n = self.banks[j].get(key, (0, 0))
                self.banks[j][key] = (zero + int(bit == 0), n + 1)
            else:
                shift = {1: 4, 2: 5, 3: 7, 6: 4, 8: 4}.get(j, 6)
                p = self.banks[j].get(key, 2048)
                self.banks[j][key] = fixed._updated_probability(p, bit, shift)


def walk(descriptors, weights, *, source=None, payload=None, observe=False):
    if (source is None) == (payload is None):
        raise ValueError('exactly one source or payload required')
    if weights.shape != (24,) or weights.dtype != np.int8:
        raise ValueError('24 counted int8 weights required')
    encoder = rc1._RangeEncoder() if source is not None and not observe else None
    decoder = rc1._RangeDecoder(payload) if payload is not None else None
    state, groups, events, truth = Predictors(), [], [], []
    for group, desc in enumerate(descriptors):
        current = []
        for pos in range(desc['count']):
            node, value = 1, 0
            for shift in reversed(range(desc['bits'])):
                keys = state.keys(group, desc, pos, node, current)
                x = state.predict(keys)
                p = fixed.squash(fixed._round_div_signed(sum(int(w) * v for w, v in zip(weights, x, strict=True)), 32))
                if source is not None:
                    bit = (int(source[group][pos]) >> shift) & 1
                    if encoder is not None:
                        encoder.encode(p if bit else 0, 4096 - p if bit else p, 4096)
                else:
                    bit = int(decoder.decode_frequency(4096) >= p)
                    decoder.update(p if bit else 0, 4096 - p if bit else p, 4096)
                if observe:
                    events.append(x)
                    truth.append(bit)
                state.update(keys, bit)
                value = value * 2 + bit
                node = node * 2 + bit
            if value >= 1 << (desc['bits'] - 1):
                value -= 1 << desc['bits']
            current.append(value)
        state.completed[desc['name']] = current
        groups.append(np.asarray(current, dtype=np.int32))
    return (encoder.finish() if encoder is not None else None), groups, events, truth


def encode(body, template, weights):
    metadata, source = split_body(body, template)
    _, descriptors = plan_metadata(metadata, template)
    payload, decoded, _, _ = walk(descriptors, weights, source=source)
    if any(not np.array_equal(a, b) for a, b in zip(source, decoded, strict=True)):
        raise ValueError('encoder changed source codes')
    rider = HEADER.pack(MAGIC, 1, 24, 0, len(payload)) + metadata + weights.tobytes() + payload
    return rider, payload, metadata


def restore_semantic(rider, template):
    if len(rider) < HEADER.size:
        raise ValueError('truncated SM1 header')
    magic, version, count, reserved, length = HEADER.unpack_from(rider)
    if (magic, version, count, reserved) != (MAGIC, 1, 24, 0) or length < 4:
        raise ValueError('unsupported SM1 header')
    end = len(rider) - length - 24
    if end < HEADER.size + 18:
        raise ValueError('SM1 length overrun')
    metadata = rider[HEADER.size:end]
    plan, descriptors = plan_metadata(metadata, template)
    weights = np.frombuffer(rider[end:end + 24], dtype=np.int8)
    _, groups, _, _ = walk(descriptors, weights, payload=rider[end + 24:])
    parts, cursor = [metadata[:10]], 0
    for item in plan:
        if item['kind'] == 'codes':
            parts.append(rc1.pack_signed_codes(groups[cursor], item['bits']))
            cursor += 1
        else:
            parts.append(item['blob'])
    return b''.join(parts)
