"""Scorer-free SM3R context diagnostics on the complete pinned live body.

No encoder is invoked. The shipped decoder's probability calls are observed;
Miller--Madow values are asymptotic diagnostics, never universal coding bounds.
Every stage is immutable, retained on SSD, and restartable from its input pins.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import brotli
import numpy as np

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments import ddm_rc1_model_section_adaptive_recode as layout_tools

ROOT = Path('/Volumes/VertigoDataTier/pact/ddm_sm1_semantic_shared_mixer')
LIVE = Path('/Volumes/VertigoDataTier/pact/ddm_sj1_multipass_token_predistortion/candidate_pass4/candidate_runtime')
POINTER = REPO / '.omx/state/canonical_frontier_pointer.json'
AXIS = '[macOS-CPU advisory / scorer-free EXACT byte measurement]'
SEED = 20260909
CONTEXTS = ('row_position', 'scale_bucket', 'previous_row', 'left_sign', 'dw_pw', 'block_depth')


def fact(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def retain(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f'immutable artifact mismatch: {path}')
    else:
        tmp = path.with_name(path.name + '.partial')
        tmp.write_bytes(payload)
        os.replace(tmp, path)
    return fact(path)


def save(path, value):
    return retain(path, (json.dumps(value, indent=2, sort_keys=True) + '\n').encode())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def load_source(root):
    runtime = root / 'source_runtime'
    sys.path.insert(0, str(runtime))
    sys.path.insert(0, str(runtime / 'cpr1'))
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    renderer = module(runtime / 'cpr1/inflate.py', 'sm1_source_renderer')
    codec = module(runtime / 'runtime/rc1_adaptive_model_sections.py', 'sm1_source_rc1')
    template = renderer.SemanticTokenRenderer(renderer.SEMANTIC_WIDTH).state_dict()
    return renderer, codec, template


def sections(archive):
    with zipfile.ZipFile(archive) as z:
        if z.namelist() != ['p']:
            raise RuntimeError('unexpected archive members')
        member = z.read('p')
    h = struct.Struct('<4sBBBBHHH')
    header = h.unpack_from(member)
    if header[:2] != (b'RX1M', 1):
        raise RuntimeError('unexpected RX1 header')
    out, offset = {'header': member[:h.size]}, h.size
    for name, length in zip(('hpac', 'semantic', 'carrier'), header[5:], strict=True):
        out[name] = member[offset:offset + length]
        offset += length
    out['tail'] = member[offset:]
    return out, header


def initialize(root, live):
    # Small source copy + traces; refuse a full SSD before materialization.
    free = shutil.disk_usage(root.parent).free
    if free < 128 * 1024**2:
        raise RuntimeError('storage preflight: 128 MiB required on assigned SSD')
    root.mkdir(parents=True, exist_ok=True)
    pointer = json.loads(POINTER.read_text())
    if fact(live / 'archive.zip')['sha256'] != pointer['effective_frontier']['archive_sha256']:
        raise RuntimeError('source is not current pointer; resolve new runtime explicitly')
    target = root / 'INPUTS.json'
    if target.exists():
        old = json.loads(target.read_text())
        if old['source_archive'] != fact(live / 'archive.zip'):
            raise RuntimeError('source moved; use a new store generation')
        for item in old['source_files']:
            if fact(item['path']) != item:
                raise RuntimeError('source copy drift')
        return old
    runtime = root / 'source_runtime'
    for source in sorted(live.rglob('*')):
        if source.is_file() and '__pycache__' not in source.parts and source.suffix not in ('.pyc', '.so', '.dylib'):
            retain(runtime / source.relative_to(live), source.read_bytes())
    out, header = sections(runtime / 'archive.zip')
    census = {name: retain(root / 'retained/source' / (name + '.bin'), value)
              for name, value in out.items()}
    if not header[4] & 0x20:
        raise RuntimeError('semantic adaptive flag is not set')
    transformed = brotli.decompress(out['semantic'])
    retain(root / 'retained/source/semantic_ck2.bin', transformed)
    rider = layout_tools.ck2_uninterleave(transformed) if header[4] & 2 else transformed
    retain(root / 'retained/source/rider.rc1s', rider)
    result = dict(schema='ddm_sm1_inputs.v1', axis=AXIS, score_claim=False,
                  seed=SEED, source_archive=fact(live / 'archive.zip'),
                  pointer=pointer, census=census,
                  source_files=[fact(p) for p in sorted(runtime.rglob('*')) if p.is_file()],
                  git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                  host=platform.platform(), command=sys.argv, free_bytes=free,
                  retention_policy='retain all stages on SSD; block deletion without certificate; no bulk scratch generated')
    save(target, result)
    return result


def sign(v):
    return 0 if v < 0 else 1 if v == 0 else 2


def context_arrays(fields, body, template):
    """All contexts are available before a symbol is decoded, including scales."""
    by_name = {f.name: f for f in fields}
    groups, previous = [], {}
    for f in fields:
        if f.kind != 'codes':
            continue
        shape = tuple(template[f.name].shape)
        cols = math.prod(shape[1:])
        values = layout_tools.unpack_signed_codes(body[f.start:f.stop], f.count, f.bits)
        rows = f.count // cols
        matrix = values.reshape(rows, cols)
        selected_rows = np.arange(rows)
        if f.name in layout_tools.ROW_PRUNE_NAMES:
            mask = by_name[f.name + ':mask']
            selected_rows = np.flatnonzero(np.unpackbits(np.frombuffer(body[mask.start:mask.stop], dtype=np.uint8), bitorder='little')[:shape[0]])
        sf = by_name[f.name + ':scales']
        scales = np.frombuffer(body[sf.start:sf.stop], dtype='<f2').astype(np.float64)
        # Quartiles defined by the counted scales themselves; ties stay together.
        thresholds = np.quantile(scales, [.25, .5, .75])
        buckets = np.searchsorted(thresholds, scales, side='left')
        contexts = {key: [] for key in CONTEXTS}
        block = int(f.name.split('.')[1]) if f.name.startswith('blocks.') else 4
        coloc = None
        if '.pw.weight' in f.name:
            # Depthwise is earlier in public state_dict order; input-channel center.
            dw = previous[f.name.replace('.pw.', '.dw.')]
            coloc = dw.reshape(shape[1], -1)[:, 4]
        for row in range(rows):
            for col in range(cols):
                contexts['row_position'].append(min(3, 4 * col // cols))
                contexts['scale_bucket'].append(int(buckets[col if f.name.endswith('embed.weight') else row]))
                original_row = int(selected_rows[row])
                prior = (int(matrix[row - 1, col]) if row and selected_rows[row - 1] == original_row - 1
                         else 0 if original_row else 99)
                contexts['previous_row'].append(prior)
                contexts['left_sign'].append(sign(matrix[row, col - 1]) if col else 3)
                contexts['dw_pw'].append(int(coloc[col]) if coloc is not None else 99)
                contexts['block_depth'].append(block)
        groups.append(dict(name=f.name, shape=shape, bits=f.bits, count=f.count,
                           packed_bytes=f.length, values=values,
                           contexts={k: np.asarray(v, dtype=np.int16) for k, v in contexts.items()}))
        previous[f.name] = matrix
    return groups


def conditional(values, contexts):
    tables = defaultdict(Counter)
    for value, context in zip(values.tolist(), contexts.tolist(), strict=True):
        tables[context][value] += 1
    plugin = correction = 0.0
    supported, supported_contexts = 0, []
    for context, counts in tables.items():
        n = sum(counts.values())
        plugin += n * math.log2(n) - sum(c * math.log2(c) for c in counts.values())
        correction += (len(counts) - 1) / (2 * math.log(2))
        if n >= 5 * len(counts) and min(counts.values()) >= 2:
            supported += n
            supported_contexts.append(context)
    return dict(plugin_bits=plugin, mm_bits=plugin + correction,
                correction_bits=correction, cells=len(tables), supported_symbols=supported,
                supported_contexts=supported_contexts)


def bound(root):
    record = json.loads((root / 'INPUTS.json').read_text())
    for item in record['source_files']:
        if fact(item['path']) != item:
            raise RuntimeError('pinned source drift')
    renderer, codec, template = load_source(root)
    rider = (root / 'retained/source/rider.rc1s').read_bytes()
    source_sections, header = sections(root / 'source_runtime/archive.zip')
    derived = brotli.decompress(source_sections['semantic'])
    derived = layout_tools.ck2_uninterleave(derived) if header[4] & 2 else derived
    if derived != rider:
        raise RuntimeError('retained rider differs from pinned archive')
    events = []
    original_model, original_decoder = codec._BinaryTreeModel, codec._RangeDecoder

    class ObservedDecoder(original_decoder):
        def update(self, low, frequency, total):
            events.append((self.group, self.symbol, int(low != 0), frequency, total))
            super().update(low, frequency, total)

    class ObservedModel(original_model):
        def decode(self, decoder, group, bits):
            decoder.group = group
            decoder.symbol = len(events) // bits if not events else self.positions.get(group, 0)
            self.positions[group] = decoder.symbol + 1
            return super().decode(decoder, group, bits)

        def __init__(self, shift):
            super().__init__(shift)
            self.positions = {}

    codec._RangeDecoder, codec._BinaryTreeModel = ObservedDecoder, ObservedModel
    try:
        body = codec.restore_semantic(rider, template)
    finally:
        codec._RangeDecoder, codec._BinaryTreeModel = original_decoder, original_model
    retain(root / 'retained/source/body.sm3r', body)
    plain = codec.restore_semantic(rider, template)
    if plain != body:
        raise RuntimeError('observation changed decoded body')
    walk = layout_tools.parse_sm3r_mixed(body, template)
    if b''.join(body[f.start:f.stop] for f in walk.fields) != body:
        raise RuntimeError('layout does not tile body')
    groups = context_arrays(walk.fields, body, template)
    array = np.asarray(events, dtype=np.int32)
    trace_path = root / 'retained/decoder_events.npy'
    # NPY storage via memory buffer is kept atomically, never scalar-only.
    import io
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    retain(trace_path, buffer.getvalue())
    table, geometry = [], []
    for gi, group in enumerate(groups):
        ev = array[array[:, 0] == gi]
        if ev.shape[0] != group['count'] * group['bits']:
            raise RuntimeError('incomplete tensor trace')
        traced_unsigned = (ev[:, 2].reshape(-1, group['bits']) * (1 << np.arange(group['bits'] - 1, -1, -1))).sum(axis=1)
        if not np.array_equal(traced_unsigned, group['values'] & ((1 << group['bits']) - 1)):
            raise RuntimeError('observer symbols disagree with independent packed-body parser')
        per_symbol = -np.log2(ev[:, 3] / ev[:, 4]).reshape(-1, group['bits']).sum(axis=1)
        actual = float(per_symbol.sum())
        marginal = conditional(group['values'], np.zeros(group['count'], dtype=np.int16))
        geometry.append(dict(name=group['name'], shape=group['shape'], codes=group['count'],
                             bits=group['bits'], packed_bytes=group['packed_bytes'],
                             realized_rc1_bits=actual, marginal_mm_bits=marginal['mm_bits']))
        for context, values in group['contexts'].items():
            stat = conditional(group['values'], values)
            supported_mask = np.isin(values, stat['supported_contexts'])
            supported_stat = conditional(group['values'][supported_mask], values[supported_mask])
            supported_gain = float(per_symbol[supported_mask].sum()) - supported_stat['mm_bits']
            table.append(dict(tensor=group['name'], context=context, n=group['count'],
                              realized_rc1_bits=actual, **stat,
                              bound_bits_saved=actual - stat['mm_bits'],
                              incremental_vs_marginal_mm_bits=marginal['mm_bits'] - stat['mm_bits'],
                              supported_gain_bits=supported_gain,
                              shared_parameter_bytes=24,
                              net_bytes_standalone=(actual - stat['mm_bits']) / 8 - 24))
    totals = []
    for context in CONTEXTS:
        rows = [r for r in table if r['context'] == context]
        gain = sum(r['bound_bits_saved'] for r in rows) / 8
        supported_gain = sum(r['supported_gain_bits'] for r in rows) / 8
        totals.append(dict(context=context, mm_gain_bytes=gain, counted_weights_bytes=24,
                           header_increment_bytes=2, net_estimate_bytes=gain - 26,
                           supported_gain_bytes=supported_gain,
                           supported_oracle_net_bytes=supported_gain - 26,
                           supported_symbols=sum(r['supported_symbols'] for r in rows)))
    result = dict(schema='ddm_sm1_context_bound.v1', axis=AXIS, score_claim=False,
                  input_binding=fact(root / 'INPUTS.json'), source_code=fact(__file__),
                  decoder_trace=fact(trace_path), body=fact(root / 'retained/source/body.sm3r'),
                  context_scope=list(CONTEXTS), table=table, geometry=geometry, totals=totals,
                  coded_symbols=sum(g['count'] for g in groups), coded_bits=len(events),
                  rc1_logloss_bytes=sum(g['realized_rc1_bits'] for g in geometry) / 8,
                  rc1_payload_bytes=codec.RC1_HEADER.unpack_from(rider)[-1],
                  noncode_bytes=sum(f.length for f in walk.fields if f.kind != 'codes'),
                  fixed_regions=[dict(name=f.name, kind=f.kind, bytes=f.length, claimed_gain_bytes=0)
                                 for f in walk.fields if f.kind != 'codes'],
                  raw_body_bytes=len(body), container_bytes=record['census']['semantic']['bytes'],
                  limitation='MM is an asymptotic estimate, not a finite-sample or universal compressor ceiling; shared-model attainability and container delta are unmeasured',
                  block_depth_note='redundant given per-tensor group; incremental information is exactly zero',
                  gate_threshold_bytes=150, gate_pass=max(r['supported_oracle_net_bytes'] for r in totals if r['context'] != 'block_depth') >= 150,
                  gate_scope='supported empirical conditional estimates minus assumed shared-model budget; attainability must be priced separately')
    save(root / 'BOUND_SUPPORTED.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('initialize', 'bound'), required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--live', type=Path, default=LIVE)
    parser.add_argument('--resume-from', type=Path)
    args = parser.parse_args()
    root = args.resume_from or args.root
    result = initialize(root, args.live) if args.stage == 'initialize' else bound(root)
    print(json.dumps({k: result[k] for k in ('totals', 'gate_pass', 'rc1_logloss_bytes') if k in result}, indent=2))
