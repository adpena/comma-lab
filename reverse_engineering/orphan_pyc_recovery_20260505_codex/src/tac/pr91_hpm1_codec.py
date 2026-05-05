"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``61:19: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``pr91_hpm1_codec.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'src/tac/pr91_hpm1_codec.py'
__recovery_spec__ = 'pr91_hpm1_codec.recovery_spec.json'
__recovery_ast_error__ = '61:19: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: pr91_hpm1_codec.cpython-312.pyc (Python 3.12)

'''Local fail-closed PR91 HPM1 mask replay and re-encode helpers.

The functions in this module are forensic/preflight tooling only. They never
load contest scorers, run exact eval, dispatch GPU work, or make score claims.
'''
from __future__ import annotations
import hashlib
import json
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
import numpy as np
import torch

functional
from tac.pr85_bundle import HPM1_HEADER_BYTES, HPM1_MAGIC, Pr85BundleError, Pr85SegmentContract, SEGMENT_ORDER, pack_pr85_bundle, parse_hpm1_mask_segment, parse_pr85_bundle
HPM1_MAGIC = HPM1_MAGIC
Pr85BundleError = Pr85BundleError
Pr85SegmentContract = Pr85SegmentContract
SEGMENT_ORDER = SEGMENT_ORDER
pack_pr85_bundle = pack_pr85_bundle
parse_hpm1_mask_segment = parse_hpm1_mask_segment
parse_pr85_bundle = parse_pr85_bundle
import torch.nn.functional, nn
from tac.pr86_hpac_codec import DEFAULT_PR86_ARCHIVE, DEFAULT_HPAC_PROBABILITY_VARIANT, EXPECTED_PR86_TOKENS_SHA256, HPACMini, Pr86HpacReplayError, _categorical_from_probs, _group_masks, _normalize_probability_row, collect_dependency_report, decode_tokens_hpac, encode_tokens_hpac, encode_symbols_hpac_with_prev_context, load_hpac_model_from_ppmd, read_pr86_archive, resolve_hpac_probability_variant, sha256_bytes, supported_hpac_probability_variant_names
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PR91_INTAKE_DIR = REPO_ROOT / 'experiments/results/public_pr91_intake_20260504_codex'
DEFAULT_PR91_RUNTIME_SOURCE_DIR = DEFAULT_PR91_INTAKE_DIR / 'replay_submission/hpac_coder_hybrid'
DEFAULT_PR91_ARCHIVE = REPO_ROOT / 'experiments/results/public_pr91_intake_20260504_codex/archive.zip'
DEFAULT_PR85_STBM_EXACT_DIR = REPO_ROOT / 'experiments/results/lightning_batch/exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z'
DEFAULT_PR85_STBM_ARCHIVE = DEFAULT_PR85_STBM_EXACT_DIR / 'archive.zip'
DEFAULT_PR85_STBM_ADJUDICATED_JSON = DEFAULT_PR85_STBM_EXACT_DIR / 'contest_auth_eval.adjudicated.json'
DEFAULT_PR85_QMA9_TOKEN_SOURCE = REPO_ROOT / 'experiments/results/public_pr85_intake_20260503_codex/qma9_token_source/pr85_qma9_tokens_u8_storage_order.bin'
CONTEST_ARCHIVE_BYTE_DENOMINATOR = 37545489
EXPECTED_PR91_ARCHIVE_BYTES = 222404
EXPECTED_PR91_ARCHIVE_SHA256 = '4c16d04c746c981feb902e4dd508ffadaf3615e532d351993c3d2f6eccda1b4f'
EXPECTED_PR91_MEMBER_X_BYTES = 222304
EXPECTED_PR91_MEMBER_X_SHA256 = '5c213c61cc4d29b62286063bfdcb97e812af6b06c0021aeaecc8bc46644e17bf'
EXPECTED_PR91_HPM1_MASK_BYTES = 145087
EXPECTED_PR91_HPM1_MASK_SHA256 = 'a4ed57ff0af1d8c914f004de165aeead50ec8dd61e99b0afdfbfa2d1e7fd9fcc'
EXPECTED_PR91_HPM1_TOKENS_SHA256 = '541016d83852a5bb3e0738caa3b44d7b2b0f7372f1841085cf9554f039c6cf6b'
EXPECTED_PR91_HPM1_HPAC_SHA256 = 'de7638c531c9dafa06148602cf784bf3ae9997f326f85cc25b9f3646b536abdd'
EXPECTED_PR85_QMA9_TOKEN_SOURCE_SHA256 = 'c1c47434fd1e6c876cb3e44910f5ab2e124285d9dba2f300bcf322d03fb8bb5a'
EXPECTED_PR85_STBM_ARCHIVE_BYTES = 229756
EXPECTED_PR85_STBM_ARCHIVE_SHA256 = 'c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6'
EXPECTED_PR85_STBM_MEMBER_X_SHA256 = 'c7586795bb29fb0ef611ad44715aec77e0e815370e19674d4c89ef2a54b417b5'
EXPECTED_PR85_STBM_HPM1_PROJECTION_SCORE = 0.248795
DEFAULT_PR91_HPM1_CONTEXT_WINDOWS = ((33, 8), (5948, 8))
PR91_HPM1_CONTEXT_MODES = ('decoded_context', 'reference_context')

class Pr91Hpm1Error(RuntimeError):
    pass
# WARNING: Decompyle incomplete

Hpm1MaskPayload = <NODE:12>()

def repo_rel(path = None):
    
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return 



def sha256_path(path = None):
    pass
# WARNING: Decompyle incomplete


def _jsonable(value = None):
    if isinstance(value, Path):
        return repo_rel(value)
    if None(value, np.generic):
        return value.item()
# WARNING: Decompyle incomplete


def _validate_safe_single_x_archive(archive = None):
    if not archive.is_file():
        raise Pr91Hpm1Error('archive_contract', 'archive_missing', archive = archive)
    archive_size = archive.stat().st_size
    archive_sha = sha256_path(archive)
# WARNING: Decompyle incomplete


def split_hpm1_mask_segment(segment = None):
    '''Parse HPM1 bytes and return the typed contract plus token/model slices.'''
    contract = parse_hpm1_mask_segment(segment)
    meta = contract.metadata
    token_start = HPM1_HEADER_BYTES
    token_end = token_start + int(meta['tokens_len'])
    hpac_end = token_end + int(meta['hpac_len'])
    tokens_blob = segment[token_start:token_end]
    hpac_ppmd_blob = segment[token_end:hpac_end]
    if len(tokens_blob) % 4 != 0:
        raise Pr91Hpm1Error('hpm1_token_stream_contract', 'tokens_blob_not_uint32_aligned', tokens_bytes = len(tokens_blob))
    if sha256_bytes(tokens_blob) != meta['tokens_sha256']:
        raise Pr91Hpm1Error('hpm1_token_stream_contract', 'tokens_sha256_metadata_mismatch')
    if sha256_bytes(hpac_ppmd_blob) != meta['hpac_ppmd_sha256']:
        raise Pr91Hpm1Error('hpm1_hpac_model_contract', 'hpac_sha256_metadata_mismatch')
    return (contract, tokens_blob, hpac_ppmd_blob)


def extract_pr91_hpm1_payload(archive = None):
    \"\"\"Extract the HPM1 mask payload from PR91's single-member archive.\"\"\"
    (raw, archive_report) = _validate_safe_single_x_archive(Path(archive))
    
    try:
        bundle = parse_pr85_bundle(raw)
        segment = bytes(bundle.segments['mask'])
        if not segment.startswith(HPM1_MAGIC):
            raise Pr91Hpm1Error('hpm1_mask_contract', 'mask_segment_is_not_hpm1', magic = segment[:4].hex())
        (contract, tokens_blob, hpac_ppmd_blob) = split_hpm1_mask_segment(segment)
        if contract.bytes == EXPECTED_PR91_HPM1_MASK_BYTES:
            contract.bytes == EXPECTED_PR91_HPM1_MASK_BYTES
        bundle_report = {
            'format': bundle.format,
            'header_bytes': bundle.header_bytes,
            'segment_lengths': bundle.segment_lengths,
            'mask_sha256': contract.sha256,
            'mask_expected_bytes': EXPECTED_PR91_HPM1_MASK_BYTES,
            'mask_expected_sha256': EXPECTED_PR91_HPM1_MASK_SHA256,
            'mask_matches_expected_pr91_hpm1': contract.sha256 == EXPECTED_PR91_HPM1_MASK_SHA256,
            'fixed_length_segments': dict(bundle.fixed_length_segments) }
        return Hpm1MaskPayload(archive_path = Path(archive), segment = segment, contract = contract, tokens_blob = tokens_blob, hpac_ppmd_blob = hpac_ppmd_blob, archive_report = archive_report, bundle_report = bundle_report)
    except Pr85BundleError:
        exc = None
        raise Pr91Hpm1Error('bundle_contract', 'pr85_family_bundle_parse_failed', error = str(exc)), exc
        exc = None
        del exc



def _validate_dependency_report(report = None):
    if str(report.get('status', '')).startswith('failed_closed'):
        raise Pr91Hpm1Error('dependency_contract', str(report.get('status')), dependency_report = report)


def _common_prefix_bytes(left = None, right = None):
    count = 0
    for left_byte, right_byte in zip(left, right, strict = False):
        if left_byte != right_byte:
            zip(left, right, strict = False)
            return count
        zip(left, right, strict = False) += 1
    return count


def _extract_call_argument(call_text = None, argument_index = None):
    '''Best-effort static extraction for simple source-contract reporting.'''
    depth = 0
    current = []
    args = []
    for char in call_text:
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        if char == ',' and depth == 0:
            args.append(''.join(current).strip())
            current = []
            continue
        current.append(char)
    if current:
        args.append(''.join(current).strip())
    if  <= 0, argument_index or 0, argument_index < len(args):
        return args[argument_index]
    return None


def _extract_first_call_body(source_text = None, call_name = None):
    marker = f'''{call_name}('''
    start = source_text.find(marker)
    if start < 0:
        return ''
    pos = start + len(marker)
    depth = 0
    body = []
    for char in source_text[pos:]:
        if char == '(':
            depth += 1
        elif char == ')':
            if depth == 0:
                
                return source_text[pos:], ''.join(body)
        body.append(char)
    return ''


def analyze_pr91_hpm1_runtime_sources(source_dir = None):
    \"\"\"Summarize PR91's submitted HPM1 decode contract from downloaded sources.\"\"\"
    source_dir = Path(source_dir)
    report = {
        'status': 'missing',
        'source_dir': repo_rel(source_dir),
        'score_claim': False,
        'dispatch_performed': False,
        'local_only': True,
        'files': { },
        'hpm1_runtime_contract': { },
        'probability_model_contract': { } }
    inflate_path = source_dir / 'inflate.py'
    hpac_path = source_dir / 'pr86_hpac.py'
# WARNING: Decompyle incomplete


def compare_hpm1_to_pr86_hpac_contract(payload = None, *, pr86_archive):
    \"\"\"Compare PR91's embedded HPM1 HPAC blobs against the PR86 source archive.

    This is a static custody comparison. It does not decode tokens, load
    scorers, dispatch, or imply that either HPAC stream is locally replayable.
    \"\"\"
    report = {
        'status': 'unknown',
        'score_claim': False,
        'dispatch_performed': False,
        'local_only': True,
        'pr86_archive': repo_rel(Path(pr86_archive)),
        'relationship': 'unknown',
        'tokens': { },
        'hpac_model': { } }
# WARNING: Decompyle incomplete


def validate_hpm1_static_contract(payload = None):
    '''Validate header/token/model facts without loading scorers or decoding frames.'''
    meta = dict(payload.contract.metadata)
    failures = []
    if payload.contract.bytes != EXPECTED_PR91_HPM1_MASK_BYTES:
        failures.append('mask_bytes_mismatch')
    if payload.contract.sha256 != EXPECTED_PR91_HPM1_MASK_SHA256:
        failures.append('mask_sha256_mismatch')
    if meta.get('tokens_sha256') != EXPECTED_PR91_HPM1_TOKENS_SHA256:
        failures.append('tokens_sha256_mismatch')
    if meta.get('hpac_ppmd_sha256') != EXPECTED_PR91_HPM1_HPAC_SHA256:
        failures.append('hpac_ppmd_sha256_mismatch')
    if not meta.get('tokens_uint32_aligned'):
        failures.append('tokens_not_uint32_aligned')
    if (meta.get('N'), meta.get('H'), meta.get('W')) != (600, 384, 512):
        failures.append('unexpected_hpm1_geometry')
    return {
        'status': 'passed' if not failures else 'failed_closed',
        'failures': failures,
        'mask': {
            'bytes': payload.contract.bytes,
            'sha256': payload.contract.sha256,
            'magic': payload.contract.magic,
            'metadata': meta },
        'tokens': {
            'bytes': len(payload.tokens_blob),
            'sha256': sha256_bytes(payload.tokens_blob),
            'uint32_words': len(payload.tokens_blob) // 4 },
        'hpac_ppmd': {
            'bytes': len(payload.hpac_ppmd_blob),
            'sha256': sha256_bytes(payload.hpac_ppmd_blob) } }


def load_hpm1_hpac_model(payload = None, *, device):
    '''Load the HPM1 HPAC model blob with the PR86-compatible model contract.'''
    return load_hpac_model_from_ppmd(payload.hpac_ppmd_blob, config = payload.config, device = device)


def run_pr91_hpm1_preflight(archive = None, *, max_frames, attempt_reencode, probability_variant, device):
    '''Run local HPM1 preflight and return a JSON-safe fail-closed report.'''
    started_at = time.time()
# WARNING: Decompyle incomplete


def run_pr91_hpm1_probability_variant_matrix(archive = None, *, variants, max_frames, attempt_reencode, require_expected_pr91_identity, device):
    '''Probe PR91 HPM1 decode under explicit HPAC probability contracts.

    The matrix is fail-closed by construction: it is local-only, never unlocks
    dispatch, and treats prefix decode as diagnostic unless a full-stream
    decode plus byte-exact re-encode is proven.
    '''
    started_at = time.time()
    if not variants:
        variants
    requested_variants = tuple(supported_hpac_probability_variant_names())
# WARNING: Decompyle incomplete


def _hpm1_token_stream_transform_candidates(token_blob = None):
    if len(token_blob) % 4 != 0:
        raise Pr91Hpm1Error('hpm1_token_stream_contract', 'tokens_blob_not_uint32_aligned', tokens_bytes = len(token_blob))
# WARNING: Decompyle incomplete


def run_pr91_hpm1_stream_transform_probe(archive = None, *, max_frames, probability_variant, device):
    '''Probe low-level token stream byte/word contracts without score claims.

    This is a local-only forensic probe. It helps distinguish a genuine HPAC
    probability/model mismatch from simpler range-coder byte-order, word-order,
    or queue-orientation mistakes.
    '''
    started_at = time.time()
# WARNING: Decompyle incomplete


def _load_reference_tokens(path = None, *, N, H, W, layout):
    path = Path(path)
    if not path.is_file():
        raise Pr91Hpm1Error('reference_token_contract', 'reference_tokens_missing', path = path)
    raw = path.read_bytes()
    expected_bytes = int(N) * int(H) * int(W)
    if len(raw) != expected_bytes:
        raise Pr91Hpm1Error('reference_token_contract', 'reference_token_size_mismatch', path = path, expected_bytes = expected_bytes, actual_bytes = len(raw), actual_sha256 = sha256_bytes(raw))
    if layout == 'qma9_storage_wh_to_render_hw':
        arr = np.frombuffer(raw, dtype = np.uint8).reshape(int(N), int(W), int(H)).transpose(0, 2, 1)
        returned_shape = [
            int(N),
            int(H),
            int(W)]
        storage_shape = [
            int(N),
            int(W),
            int(H)]
        storage_order = 'frame_major_header_width_by_header_height'
        render_transform = 'reshape_NWH_transpose_to_NHW'
    elif layout == 'legacy_assume_nhw':
        arr = np.frombuffer(raw, dtype = np.uint8).reshape(int(N), int(H), int(W))
        returned_shape = [
            int(N),
            int(H),
            int(W)]
        storage_shape = [
            int(N),
            int(H),
            int(W)]
        storage_order = 'legacy_frame_major_header_height_by_header_width_assumption'
        render_transform = 'none'
    else:
        raise Pr91Hpm1Error('reference_token_contract', 'unsupported_reference_token_layout', requested_layout = layout, supported_layouts = [
            'qma9_storage_wh_to_render_hw',
            'legacy_assume_nhw'])
    observed_min = int(arr.min()) if arr.size else None
    observed_max = int(arr.max()) if arr.size else None
# WARNING: Decompyle incomplete


def _symbol_position(mask = None, symbol_in_group = None):
    coords = torch.nonzero(mask, as_tuple = False)
    if symbol_in_group < 0 or symbol_in_group >= int(coords.shape[0]):
        return None
    (row, col) = coords[int(symbol_in_group)].detach().cpu().tolist()
    return {
        'y': int(row),
        'x': int(col) }


def _probability_row_profile(row = None, *, reference_symbol, variant_name, prob_eps):
    variant = resolve_hpac_probability_variant(variant_name)
    dtype = np.float32 if variant.probability_dtype == 'float32' else np.float64
    clipped = np.clip(row.astype(dtype, copy = False), dtype(prob_eps), dtype(1))
    clipped = clipped / clipped.sum()
# WARNING: Decompyle incomplete


def _torch_scalar_at(tensor = None, y = None, x = None):
    if y < 0 and x < 0 and y >= int(tensor.shape[-2]) or x >= int(tensor.shape[-1]):
        return None
    return int(tensor[(0, y, x)].detach().cpu().item())


def _symbol_context_profile(cur = None, prev = None, pixel_yx = None):
    pass
# WARNING: Decompyle incomplete


def _normalize_symbol_windows(windows = None):
    normalized = []
    for item in windows:
        if len(item) != 2:
            raise Pr91Hpm1Error('context_window_probe_contract', 'window_must_be_start_and_count', window = list(item))
        count = int(item[1])
        start = int(item[0])
        if start < 0 or count <= 0:
            raise Pr91Hpm1Error('context_window_probe_contract', 'window_start_must_be_nonnegative_and_count_positive', window = {
                'start_global_symbol': start,
                'count': count })
        normalized.append((start, count))
    if not normalized:
        raise Pr91Hpm1Error('context_window_probe_contract', 'at_least_one_symbol_window_required')
    return tuple(sorted(dict.fromkeys(normalized)))


def _window_for_symbol(global_symbol = None, windows = None):
    for start, count in windows:
        if  <= start, global_symbol:
            if not start, global_symbol < start + count:
                continue
            else:
                windows
            
            return windows, (start, count)
        return None


def _window_trace_report(windows = None, traces = None):
    rows = []
    for start, count in windows:
        trace = traces[(start, count)]
        rows.append({
            'start_global_symbol': start,
            'requested_count': count,
            'end_global_symbol_exclusive': start + count,
            'recorded_count': len(trace),
            'trace': trace,
            'trace_sha256': sha256_bytes(json.dumps(_jsonable(trace), sort_keys = True, separators = (',', ':')).encode('utf-8')) })
    return rows


def _context_mode_description(context_mode = None):
    if context_mode == 'decoded_context':
        return 'Submitted stream is replayed normally: decoded symbols update the current-frame and previous-frame HPAC contexts.'
    if context_mode == 'reference_context':
        return 'Submitted stream is still consumed by RangeDecoder, but after each group the HPAC model context is teacher-forced from the reference token tensor. This separates accumulated decoded-context drift from range/probability numeric mismatch.'
    raise Pr91Hpm1Error('context_window_probe_contract', 'unsupported_context_mode', context_mode = context_mode, supported_context_modes = list(PR91_HPM1_CONTEXT_MODES))


def _teacher_forced_reference_probability_windows(model = None, reference_tokens = None, masks = None, *, config, windows, variant_names, prob_eps_values, device):
    '''Record reference-context probability rows without consuming RangeDecoder.'''
    max_requested_symbol = (lambda .0: pass# WARNING: Decompyle incomplete
)(windows())
    results = []
# WARNING: Decompyle incomplete


def _probability_state_profile(probability_row = None, logits_row = None, *, reference_symbol, decoded_symbol, variant_names, prob_eps):
    raw_probs = probability_row.astype(np.float64, copy = False)
    logits = logits_row.astype(np.float64, copy = False)
    variants = { }
# WARNING: Decompyle incomplete

run_pr91_hpm1_reference_prefix_probe = (lambda archive = None, *, reference_tokens_path: started_at = time.time()# WARNING: Decompyle incomplete
)()
run_pr91_hpm1_first_symbol_state_probe = (lambda archive = None, *, reference_tokens_path: started_at = time.time()# WARNING: Decompyle incomplete
)()
run_pr91_hpm1_context_window_probe = (lambda archive = None, *, reference_tokens_path: started_at = time.time()# WARNING: Decompyle incomplete
)()

def _read_single_x_archive_for_fusion(archive = None, *, label):
    archive = Path(archive)
    if not archive.is_file():
        raise Pr91Hpm1Error('fusion_archive_contract', 'archive_missing', label = label, archive = archive)
    archive_bytes = archive.stat().st_size
    archive_sha = sha256_path(archive)
# WARNING: Decompyle incomplete


def _segment_codec_label(name = None, segment = None):
    if name == 'mask' and segment.startswith(b'STBM1BR\x00'):
        return 'STBM1BR'
    if name == 'mask' and segment.startswith(HPM1_MAGIC):
        return 'HPM1'
    if name == 'mask' and segment.startswith(b'QMA'):
        return segment[:4].decode('ascii', errors = 'replace')


def _segment_digest_rows(left = None, right = None):
    rows = []
    for name in SEGMENT_ORDER:
        left_segment = bytes(left[name])
        right_segment = bytes(right[name])
        rows.append({
            'segment': name,
            'same_bytes': left_segment == right_segment,
            'left_bytes': len(left_segment),
            'right_bytes': len(right_segment),
            'byte_delta_right_minus_left': len(right_segment) - len(left_segment),
            'left_sha256': sha256_bytes(left_segment),
            'right_sha256': sha256_bytes(right_segment),
            'left_codec': _segment_codec_label(name, left_segment),
            'right_codec': _segment_codec_label(name, right_segment) })
    return rows


def _load_pr85_stbm_score_anchor(path = None):
    pass
# WARNING: Decompyle incomplete


def plan_pr91_hpm1_pr85_stbm_fusion(*, pr85_stbm_archive, pr91_archive, pr85_stbm_adjudicated_json, include_hpm1_prefix_probe, hpm1_prefix_probe_max_frames):
    '''Prove the byte-level PR85+STBM -> PR91/HPM1 fusion relation.

    This planner intentionally never unlocks dispatch. A smaller HPM1 mask can
    beat STBM only after the HPM1 stream proves exact decode parity under the
    submitted runtime contract. Falling back to STBM/QMA9 after HPM1 failure is
    not a valid rate win unless that fallback payload is charged in the archive.
    '''
    started_at = time.time()
# WARNING: Decompyle incomplete


def build_hpm1_mask_segment(tokens_blob = None, hpac_ppmd_blob = None, *, N, H, W, P, delta, ch, use_spm, hpac_d_film, ppmd_order):
    '''Build an HPM1 segment from explicit charged token/model bytes.'''
    if len(tokens_blob) <= 0 or len(tokens_blob) % 4 != 0:
        raise Pr91Hpm1Error('hpm1_encoder_contract', 'tokens_blob_must_be_nonempty_uint32_words', tokens_bytes = len(tokens_blob))
    if len(hpac_ppmd_blob) <= 0:
        raise Pr91Hpm1Error('hpm1_encoder_contract', 'hpac_ppmd_blob_must_be_nonempty')
    header = b''.join + (lambda .0: pass# WARNING: Decompyle incomplete
)((N, H, W, P, delta, ch, 1 if use_spm else 0, hpac_d_film, len(tokens_blob), len(hpac_ppmd_blob), ppmd_order)())
    segment = header + tokens_blob + hpac_ppmd_blob
    parse_hpm1_mask_segment(segment)
    return segment


def prototype_reencode_hpm1_from_raw_tokens(raw_tokens = None, source_payload = None, *, max_frames, probability_variant, device):
    '''Local-only HPM1 re-encode prototype from decoded mask tokens.

    This is a byte-construction prototype, not a dispatchable candidate. It is
    useful only after decoded PR85/PR91 mask-token custody is already proven.
    '''
    started_at = time.time()
    if raw_tokens.ndim != 3:
        raise Pr91Hpm1Error('hpm1_encoder_contract', 'raw_tokens_must_be_nhw', shape = list(raw_tokens.shape))
    config = source_payload.config
# WARNING: Decompyle incomplete


def raw_tokens_to_mod5_residual_symbols(raw_tokens = None):
    '''Return mod-5 residual symbols plus raw previous-frame context tokens.'''
    if raw_tokens.ndim != 3:
        raise Pr91Hpm1Error('hpm1_residual_contract', 'raw_tokens_must_be_nhw', shape = list(raw_tokens.shape))
    raw = raw_tokens.astype(np.uint8, copy = False)
    if raw.size:
        if int(raw.min()) < 0 or int(raw.max()) > 4:
            raise Pr91Hpm1Error('hpm1_residual_contract', 'raw_token_class_value_out_of_range', min_value = int(raw.min()), max_value = int(raw.max()))
    prev = np.zeros_like(raw, dtype = np.uint8)
    if raw.shape[0] > 1:
        prev[1:] = raw[:-1]
    symbols = ((raw.astype(np.int16) - prev.astype(np.int16)) % 5).astype(np.uint8)
    return (symbols, prev)


def reconstruct_raw_tokens_from_mod5_residual_symbols(symbols = None, prev_context_tokens = None):
    '''Reconstruct raw token maps from mod-5 residual symbols and previous context.'''
    if symbols.shape != prev_context_tokens.shape:
        raise Pr91Hpm1Error('hpm1_residual_contract', 'residual_prev_shape_mismatch', symbols_shape = list(symbols.shape), prev_context_shape = list(prev_context_tokens.shape))
    return ((symbols.astype(np.int16) + prev_context_tokens.astype(np.int16)) % 5).astype(np.uint8)


def prototype_reencode_hpm1_residual_from_raw_tokens(raw_tokens = None, source_payload = None, *, max_frames, probability_variant, device):
    '''Local-only HPM1 residual-symbol prototype from decoded mask tokens.'''
    started_at = time.time()
    if raw_tokens.ndim != 3:
        raise Pr91Hpm1Error('hpm1_residual_contract', 'raw_tokens_must_be_nhw', shape = list(raw_tokens.shape))
    config = source_payload.config
# WARNING: Decompyle incomplete


def write_json_report(report = None, path = None):
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(json.dumps(_jsonable(report), indent = 2, sort_keys = True) + '\n', encoding = 'utf-8')


"""
