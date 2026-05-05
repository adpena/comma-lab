"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``25:16: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``profile_pr95_hnerv_muon_packing.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'experiments/profile_pr95_hnerv_muon_packing.py'
__recovery_spec__ = 'profile_pr95_hnerv_muon_packing.recovery_spec.json'
__recovery_ast_error__ = '25:16: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: profile_pr95_hnerv_muon_packing.cpython-312.pyc (Python 3.12)

'''Profile and safely repack PR95 HNeRV Muon archives.

This tool only performs byte-preserving transformations: brotli parameter
search, compact equivalent JSON metadata, and raw decoder-record reordering.
It does not dequantize/requantize tensors or alter latent values. Any emitted
candidate therefore has the same decoded model/latent contract as the source
blob, but it still requires exact CUDA eval before making a score claim.
'''
from __future__ import annotations
import argparse
import collections
import dataclasses
import hashlib
import io
import json
import math
import struct
import zipfile
from pathlib import Path
from typing import Iterable, Sequence
import brotli
BrotliChoice = <NODE:12>()
DecoderRecord = <NODE:12>()
LatentPayload = <NODE:12>()

def sha256_bytes(data = dataclasses.dataclass(frozen = True)):
    return hashlib.sha256(data).hexdigest()


def shannon_entropy_bits(data = None):
    pass
# WARNING: Decompyle incomplete


def read_single_member_zip(path = None):
    zf = zipfile.ZipFile(path, 'r')
    infos = zf.infolist()
    if len(infos) != 1:
        raise ValueError(f'''expected exactly one archive member, got {len(infos)}''')
    info = infos[0]
    if info.filename != '0.bin':
        raise ValueError(f'''expected member 0.bin, got {info.filename!r}''')
    data = zf.read(info)
    None(None, None)
    return 
    with None:
        if not None, (info.filename, data, {
            'compress_type': int(info.compress_type),
            'file_size': int(info.file_size),
            'compress_size': int(info.compress_size),
            'crc': int(info.CRC),
            'date_time': list(info.date_time) }):
            pass


def write_stored_zip(path = None, member_name = None, payload = None):
    path.parent.mkdir(parents = True, exist_ok = True)
    info = zipfile.ZipInfo(member_name, date_time = (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 27525120
    zf = zipfile.ZipFile(path, 'w', compression = zipfile.ZIP_STORED, allowZip64 = False)
    zf.writestr(info, payload)
    None(None, None)
    return None
    with None:
        if not None:
            pass


def parse_top_blob(blob = None):
    buf = io.BytesIO(blob)
    meta_len = struct.unpack('<I', buf.read(4))[0]
    meta_brotli = buf.read(meta_len)
    dec_len = struct.unpack('<I', buf.read(4))[0]
    decoder_brotli = buf.read(dec_len)
    lat_len = struct.unpack('<I', buf.read(4))[0]
    latents_brotli = buf.read(lat_len)
    rest = buf.read()
    if rest:
        raise ValueError(f'''trailing bytes after PR95 blob: {len(rest)}''')
    return {
        'meta_brotli': meta_brotli,
        'meta_raw': brotli.decompress(meta_brotli),
        'decoder_brotli': decoder_brotli,
        'decoder_raw': brotli.decompress(decoder_brotli),
        'latents_brotli': latents_brotli,
        'latents_raw': brotli.decompress(latents_brotli) }


def encode_top_blob(meta_brotli = None, decoder_brotli = None, latents_brotli = None):
    out = io.BytesIO()
    out.write(struct.pack('<I', len(meta_brotli)))
    out.write(meta_brotli)
    out.write(struct.pack('<I', len(decoder_brotli)))
    out.write(decoder_brotli)
    out.write(struct.pack('<I', len(latents_brotli)))
    out.write(latents_brotli)
    return out.getvalue()


def brotli_search(label = None, raw = None, *, qualities, lgwins):
    best = None
# WARNING: Decompyle incomplete


def compact_meta_raw(meta_raw = None):
    payload = json.loads(meta_raw)
    return json.dumps(payload, separators = (',', ':'), sort_keys = True).encode('utf-8')


def parse_decoder_records(decoder_raw = None):
    pass
# WARNING: Decompyle incomplete


def parse_decoder_records_structured(decoder_raw = None):
    pass
# WARNING: Decompyle incomplete


def decoder_record_name(record = None):
    buf = io.BytesIO(record)
    name_len = struct.unpack('<I', buf.read(4))[0]
    return buf.read(name_len).decode('utf-8')


def rebuild_decoder_raw(records = None):
    out = io.BytesIO()
    out.write(struct.pack('<I', len(records)))
    for record in records:
        out.write(record)
    return out.getvalue()


def rebuild_structured_decoder_raw(records = None):
    pass
# WARNING: Decompyle incomplete


def decoder_raw_variants(decoder_raw = None):
    records = parse_decoder_records(decoder_raw)
    variants = {
        'original': decoder_raw,
        'name_asc': rebuild_decoder_raw(sorted(records, key = decoder_record_name)),
        'name_desc': rebuild_decoder_raw(sorted(records, key = decoder_record_name, reverse = True)),
        'size_desc': rebuild_decoder_raw(sorted(records, key = len, reverse = True)),
        'size_asc': rebuild_decoder_raw(sorted(records, key = len)) }
    return variants


def parse_latents_raw(latents_raw = None):
    buf = io.BytesIO(latents_raw)
    (n_pairs, latent_dim) = struct.unpack('<II', buf.read(8))
    mins_f16 = buf.read(latent_dim * 2)
    scales_f16 = buf.read(latent_dim * 2)
    total = n_pairs * latent_dim
    lo = buf.read(total)
    hi = buf.read(total)
    rest = buf.read()
    if len(mins_f16) != latent_dim * 2 or len(scales_f16) != latent_dim * 2:
        raise ValueError('latent raw header is truncated')
    if len(lo) != total or len(hi) != total:
        raise ValueError('latent delta streams are truncated')
    if rest:
        raise ValueError(f'''latent raw has trailing bytes: {len(rest)}''')
    rows = []
    prev = [
        0] * latent_dim
    for pair_index in range(n_pairs):
        row = []
        for dim_index in range(latent_dim):
            offset = pair_index * latent_dim + dim_index
            zz = lo[offset] | hi[offset] << 8
            delta = zz // 2 if zz % 2 == 0 else -(zz // 2) - 1
            value = delta if pair_index == 0 else prev[dim_index] + delta
            raise ValueError(f'''latent quantized value out of uint8 range at pair {pair_index}, dim {dim_index}: {value}''')
            row.append(value)
        rows.append(tuple(row))
    return LatentPayload(n_pairs = n_pairs, latent_dim = latent_dim, mins_f16 = mins_f16, scales_f16 = scales_f16, quantized = tuple(rows))


def reorder_latents_raw(latents_raw = None, permutation = None):
    pass
# WARNING: Decompyle incomplete


def validate_permutation(permutation = None, width = None):
    if sorted(permutation) != list(range(width)):
        raise ValueError(f'''not a width-{width} permutation: {list(permutation)}''')


def reorder_stem_weight_record(record = None, permutation = None):
    pass
# WARNING: Decompyle incomplete


def decoder_raw_with_latent_permutation(decoder_raw = None, permutation = None):
    records = parse_decoder_records_structured(decoder_raw)
# WARNING: Decompyle incomplete


def latent_dimension_profiles(decoder_raw = None, latents_raw = None):
    pass
# WARNING: Decompyle incomplete


def permutation_variants(decoder_raw = None, latents_raw = None):
    profiles = latent_dimension_profiles(decoder_raw, latents_raw)
    latent_dim = len(profiles)
# WARNING: Decompyle incomplete


def decoder_record_accounting(decoder_raw = None):
    records = parse_decoder_records_structured(decoder_raw)
    out = []
    for index, record in enumerate(records):
        record_bytes = record.to_bytes()
        out.append({
            'index': index,
            'name': record.name,
            'shape': list(record.shape),
            'numel': record.numel,
            'raw_record_bytes': len(record_bytes),
            'q_bytes': len(record.q_zz),
            'q_unique': len(set(record.q_zz)),
            'q_zero_zigzag_fraction': record.q_zz.count(0) / len(record.q_zz) if record.q_zz else 0,
            'q_entropy_bits': shannon_entropy_bits(record.q_zz),
            'standalone_brotli_q11_bytes': len(brotli.compress(record_bytes, quality = 11)),
            'sha256': sha256_bytes(record_bytes) })
    return out

CoupledCandidate = <NODE:12>()

def run(args = None):
    source_zip = Path(args.archive)
    output_dir = Path(args.output_dir)
    (member_name, blob, zip_meta) = read_single_member_zip(source_zip)
    parts = parse_top_blob(blob)
    qualities = range(args.min_quality, args.max_quality + 1)
    lgwins = range(args.min_lgwin, args.max_lgwin + 1)
    meta_raws = {
        'original': parts['meta_raw'],
        'compact_sorted': compact_meta_raw(parts['meta_raw']) }
# WARNING: Decompyle incomplete


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--archive', default = 'experiments/results/public_pr95_intake_20260504_codex/archive.zip')
    parser.add_argument('--output-dir', default = 'experiments/results/pr95_hnerv_muon_packing_profile_20260504_codex')
    parser.add_argument('--min-quality', type = int, default = 8)
    parser.add_argument('--max-quality', type = int, default = 11)
    parser.add_argument('--min-lgwin', type = int, default = 18)
    parser.add_argument('--max-lgwin', type = int, default = 24)
    return run(parser.parse_args())

if __name__ == '__main__':
    raise SystemExit(main())

"""
