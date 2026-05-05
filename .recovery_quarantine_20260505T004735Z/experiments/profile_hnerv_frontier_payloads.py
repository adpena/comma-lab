"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``29:18: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``profile_hnerv_frontier_payloads.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'experiments/profile_hnerv_frontier_payloads.py'
__recovery_spec__ = 'profile_hnerv_frontier_payloads.recovery_spec.json'
__recovery_ast_error__ = '29:18: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: profile_hnerv_frontier_payloads.cpython-312.pyc (Python 3.12)

'''Profile byte layout for public HNeRV frontier archives.

This is a deconstruction tool, not a scorer. It reads a contest archive, extracts
the single charged payload member, and emits deterministic section bytes,
SHA-256, and entropy so late-public-submission ideas can feed repacking and
Lagrangian allocation work without relying on prose notes.
'''
from __future__ import annotations
import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from zipfile import ZipFile
PR101_DECODER_BLOB_LEN = 162164
PR101_LATENT_BLOB_LEN = 15387
PR103_SCA_LEN = 56
PR103_BR_LEN = 7097
PR103_HIST_LEN = 895
PR103_MERGED_AC_LEN = 153856
PR103_LATENT_META_LEN = 112
PR103_LO_LEN = 15537
PR103_HI_HIST_LEN = 15
SectionProfile = <NODE:12>()

def sha256_bytes(data = None):
    return hashlib.sha256(data).hexdigest()


def entropy_bits_per_byte(data = None):
    pass
# WARNING: Decompyle incomplete


def section(name = None, blob = None, start = None, end = ('name', 'str', 'blob', 'bytes', 'start', 'int', 'end', 'int', 'return', 'SectionProfile')):
    if start < 0 and end < start or end > len(blob):
        raise ValueError(f'''bad section {name}: start={start} end={end} len={len(blob)}''')
    data = blob[start:end]
    return SectionProfile(name = name, start = start, end = end, bytes = len(data), sha256 = sha256_bytes(data), entropy_bits_per_byte = round(entropy_bits_per_byte(data), 6))


def extract_single_member(archive = None):
    archive_bytes = archive.read_bytes()
    zf = ZipFile(archive)
# WARNING: Decompyle incomplete


def infer_profile_kind(kind = None, archive = None, member_name = None, payload = ('kind', 'str', 'archive', 'Path', 'member_name', 'str', 'payload', 'bytes', 'return', 'str')):
    if kind != 'auto':
        return kind
    text = f'''{None} {member_name}'''.lower()
    if 'pr101' in text or 'hnerv_ft_microcodec' in text:
        return 'pr101_microcodec'
    if 'pr103' in text or 'hnerv_lc_ac' in text:
        return 'pr103_lc_ac'
    if payload[:1] == b'\xff' and len(payload) >= 4:
        dec_len = int.from_bytes(payload[1:4], 'little')
        if  < 0, dec_len or 0, dec_len < len(payload) - 4:
            return 'ff_packed_brotli_hnerv'
    if len(payload) == PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN + 607:
        return 'pr101_microcodec'
    if len(payload) >= PR103_SCA_LEN + PR103_BR_LEN + PR103_HIST_LEN + PR103_MERGED_AC_LEN + PR103_LATENT_META_LEN + PR103_LO_LEN + PR103_HI_HIST_LEN:
        return 'pr103_lc_ac'
    return 'unknown_single_payload'


def profile_payload(kind = None, payload = None):
    sections = []
    if kind == 'pr101_microcodec':
        a = PR101_DECODER_BLOB_LEN
        b = a + PR101_LATENT_BLOB_LEN
        sections.append(section('decoder_compact_brotli_streams', payload, 0, a))
        sections.append(section('latents_raw_lzma_delta_u8', payload, a, b))
        sections.append(section('sidecar_dim_delta_huffman_enum', payload, b, len(payload)))
        return sections
    if None == 'pr103_lc_ac':
        cursor = 0
        for name, size in (('scales_fp16', PR103_SCA_LEN), ('non_ac_weights_brotli', PR103_BR_LEN), ('ac_histograms_brotli', PR103_HIST_LEN), ('merged_range_coded_weights_and_hi_latents', PR103_MERGED_AC_LEN), ('latent_min_scale_fp16', PR103_LATENT_META_LEN), ('latent_low_bytes_brotli', PR103_LO_LEN), ('latent_hi_histogram_brotli', PR103_HI_HIST_LEN)):
            sections.append(section(name, payload, cursor, cursor + size))
            cursor += size
        sections.append(section('sidecar_corrections_brotli', payload, cursor, len(payload)))
        return sections
    if None == 'ff_packed_brotli_hnerv':
        dec_len = int.from_bytes(payload[1:4], 'little')
        sections.append(section('packed_header_ff_len24', payload, 0, 4))
        sections.append(section('decoder_packed_brotli', payload, 4, 4 + dec_len))
        sections.append(section('latents_and_sidecar_brotli', payload, 4 + dec_len, len(payload)))
        return sections
    None.append(section('opaque_single_payload', payload, 0, len(payload)))
    return sections


def render_markdown(record = None):
    lines = [
        f'''# HNeRV Frontier Payload Profile: {record['archive']}''',
        '',
        f'''- archive_bytes: `{record['archive_bytes']}`''',
        f'''- archive_sha256: `{record['archive_sha256']}`''',
        f'''- zip_member: `{record['member_name']}`''',
        f'''- member_bytes: `{record['member_bytes']}`''',
        f'''- member_sha256: `{record['member_sha256']}`''',
        f'''- inferred_kind: `{record['kind']}`''',
        f'''- zip_overhead_bytes: `{record['zip_overhead_bytes']}`''',
        '',
        '| section | start | end | bytes | entropy b/B | sha256 |',
        '|---|---:|---:|---:|---:|---|']
# WARNING: Decompyle incomplete


def build_record(archive = None, kind = None):
    (member_name, payload, archive_bytes) = extract_single_member(archive)
    resolved_kind = infer_profile_kind(kind, archive, member_name, payload)
    sections = profile_payload(resolved_kind, payload)
# WARNING: Decompyle incomplete


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('archives', nargs = '+', type = Path)
    parser.add_argument('--kind', default = 'auto')
    parser.add_argument('--json-out', type = Path)
    parser.add_argument('--md-out', type = Path)
    args = parser.parse_args()
# WARNING: Decompyle incomplete

if __name__ == '__main__':
    raise SystemExit(main())

"""
