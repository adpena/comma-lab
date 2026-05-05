# Source Generated with Decompyle++
# File: test_profile_pr95_hnerv_muon_packing.cpython-312.pyc (Python 3.12)

from __future__ import annotations
import io
import json
import struct
import zipfile
from pathlib import Path
import brotli
from experiments.profile_pr95_hnerv_muon_packing import LatentPayload, decoder_raw_variants, parse_decoder_records_structured, parse_latents_raw, parse_top_blob, reorder_latents_raw, reorder_stem_weight_record, run

def _decoder_record(name = None, q = None, shape = None):
    if not shape:
        shape
    shape = (len(q),)
    total = 1
    for dim in shape:
        total *= dim
# WARNING: Decompyle incomplete


def _decoder_raw(records = None):
    out = io.BytesIO()
    out.write(struct.pack('<I', len(records)))
    for record in records:
        out.write(record)
    return out.getvalue()


def _top_blob(meta_raw = None, decoder_raw = None, latents_raw = None):
    meta = brotli.compress(meta_raw, quality = 5)
    dec = brotli.compress(decoder_raw, quality = 5)
    lat = brotli.compress(latents_raw, quality = 5)
    out = io.BytesIO()
    for payload in (meta, dec, lat):
        out.write(struct.pack('<I', len(payload)))
        out.write(payload)
    return out.getvalue()


def test_decoder_raw_variants_preserve_record_set():
    records = [
        _decoder_record('b.weight', b'\x01\x02'),
        _decoder_record('a.weight', b'\x03')]
    raw = _decoder_raw(records)
    variants = decoder_raw_variants(raw)
# WARNING: Decompyle incomplete


def test_parse_top_blob_roundtrips_raw_streams():
    meta_raw = b'{"latent_dim": 1, "n_pairs": 1}'
    decoder_raw = _decoder_raw([
        _decoder_record('x', b'\x01')])
    latents_raw = b'latent-bytes'
    blob = _top_blob(meta_raw, decoder_raw, latents_raw)
    parsed = parse_top_blob(blob)
# WARNING: Decompyle incomplete


def test_latent_dimension_permutation_compensates_stem_weight():
    stem = _decoder_record('stem.weight', bytes([
        1,
        2,
        3,
        4,
        5,
        6]), shape = (2, 3))
    decoder_raw = _decoder_raw([
        stem])
    latents_raw = LatentPayload(n_pairs = 3, latent_dim = 3, mins_f16 = b'aabbcc', scales_f16 = b'ddeeff', quantized = ((1, 2, 3), (2, 4, 6), (3, 6, 9))).to_bytes()
    permutation = [
        2,
        0,
        1]
    reordered_latents = parse_latents_raw(reorder_latents_raw(latents_raw, permutation))
    reordered_stem = reorder_stem_weight_record(parse_decoder_records_structured(decoder_raw)[0], permutation)
# WARNING: Decompyle incomplete


def test_profiler_emits_no_score_claim_candidate(tmp_path = None):
    pass
# WARNING: Decompyle incomplete

