"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``31:39: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``pr86_hpac_codec.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'src/tac/pr86_hpac_codec.py'
__recovery_spec__ = 'pr86_hpac_codec.recovery_spec.json'
__recovery_ast_error__ = '31:39: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: pr86_hpac_codec.cpython-312.pyc (Python 3.12)

'''Fail-closed local PR86 HPAC replay and byte-parity helpers.

This module is deliberately local-only: it never runs contest eval, never
dispatches remote work, and never makes a score claim. It validates the public
PR86 archive contract, decodes its torch/PPMd payloads, and attempts to prove
that submitted ``tokens.bin`` can be decoded and re-encoded byte-for-byte with
``constriction.stream.queue.RangeEncoder.get_compressed().tobytes()``.
'''
from __future__ import annotations
import gzip
import hashlib
import importlib.metadata as importlib
import io
import json
import sys
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
import numpy as np
import torch
from torch.nn import nn

functional
Path(__file__).resolve().parents[2] = import torch.nn.functional, nn
DEFAULT_PR86_DIR = REPO_ROOT / 'experiments/results/public_pr86_intake_20260504_codex'
DEFAULT_PR86_ARCHIVE = DEFAULT_PR86_DIR / 'archive.zip'
DEFAULT_PR86_PROBABILITY_CONTRACT_DIR = REPO_ROOT / 'experiments/results/pr86_hpac_probability_contract_20260504_worker'
DEFAULT_PR86_PROBABILITY_CONTRACT_REPORT = DEFAULT_PR86_PROBABILITY_CONTRACT_DIR / 'pr86_hpac_probability_contract_variants.json'
DEFAULT_PR86_MERGED_SOURCE_DIR = REPO_ROOT / 'experiments/results/public_pr86_intake_20260504_merged_refresh'
DEFAULT_MERGED_INTAKE_SUMMARY = DEFAULT_PR86_MERGED_SOURCE_DIR / 'intake_summary.json'
DEFAULT_MERGED_SOURCE_MANIFEST = DEFAULT_PR86_MERGED_SOURCE_DIR / 'source_manifest.json'
DEFAULT_MERGED_PR_API = DEFAULT_PR86_MERGED_SOURCE_DIR / 'pr86_api.json'
DEFAULT_FULL_REENCODE = DEFAULT_PR86_DIR / 'pr86_hpac_full_decode_reencode_gate_20260504_codex.json'
DEFAULT_TOKEN_ANATOMY = DEFAULT_PR86_DIR / 'pr86_hpac_token_anatomy_forensics.json'
DEFAULT_PR85_PROBE = DEFAULT_PR86_DIR / 'pr86_hpac_pr85_qma9_parity_probe.json'
DEFAULT_PR_VIEW = DEFAULT_PR86_DIR / 'pr86_view.json'
EXPECTED_PR86_ARCHIVE_BYTES = 207579
EXPECTED_PR86_ARCHIVE_SHA256 = 'e67b7c22240dbe33853c19d049b0044a5df16ce5f751ba8f1021cab8ceb03cef'
EXPECTED_PR86_TOKENS_SHA256 = '14144bde496631f89a02646496bc2e66306bba6da149ddca37e21d85d175f225'
EXPECTED_PR86_MEMBERS = ('master.pt.gz', 'slave.pt.gz', 'hpac.pt.ppmd', 'tokens.bin', 'meta.pt')
EXPECTED_PR86_MEMBER_BYTES = {
    'master.pt.gz': 31144,
    'slave.pt.gz': 32287,
    'hpac.pt.ppmd': 28243,
    'tokens.bin': 113900,
    'meta.pt': 1499 }
RECORDED_PR86_DEPENDENCIES = {
    'python': '3.12.13',
    'torch': '2.11.0',
    'numpy': '2.4.4',
    'constriction': '0.4.2',
    'pyppmd': '1.3.1' }
NUM_CLASSES = 5
SEGNET_IN_H = 384
SEGNET_IN_W = 512
PPMD_MAX_ORDER = 4
PPMD_MEM_SIZE = 16777216
PROB_EPS = 1e-07
DEFAULT_HPAC_PROBABILITY_VARIANT = 'source_float64_perfect_false'

class Pr86HpacReplayError(RuntimeError):
    pass
# WARNING: Decompyle incomplete

Pr86ArchiveContract = <NODE:12>()
Pr86ArchiveBundle = <NODE:12>()
HpacProbabilityVariant = <NODE:12>()
HPAC_PROBABILITY_VARIANTS: 'Mapping[str, HpacProbabilityVariant]' = {
    'source_float64_perfect_false': HpacProbabilityVariant(name = 'source_float64_perfect_false', probability_dtype = 'float64', categorical_perfect = False, source_contract = True, description = 'Merged PR86 source contract: clipped/renormalized numpy float64 probabilities with Categorical(..., perfect=False).'),
    'source_float32_perfect_false': HpacProbabilityVariant(name = 'source_float32_perfect_false', probability_dtype = 'float32', categorical_perfect = False, source_contract = False, description = 'Off-contract probe: pass clipped/renormalized numpy float32 probabilities to Categorical(..., perfect=False).'),
    'source_float64_perfect_true': HpacProbabilityVariant(name = 'source_float64_perfect_true', probability_dtype = 'float64', categorical_perfect = True, source_contract = False, description = 'Off-contract probe: keep float64 probabilities but construct Categorical(..., perfect=True).'),
    'source_float32_perfect_true': HpacProbabilityVariant(name = 'source_float32_perfect_true', probability_dtype = 'float32', categorical_perfect = True, source_contract = False, description = 'Off-contract combined probe: pass float32 probabilities to Categorical(..., perfect=True).') }

def repo_rel(path = dataclass(frozen = True)):
    
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return 



def sha256_bytes(data = None):
    return hashlib.sha256(data).hexdigest()


def sha256_path(path = None):
    pass
# WARNING: Decompyle incomplete


def supported_hpac_probability_variant_names():
    '''Return stable CLI choices for HPAC probability-contract probes.'''
    return tuple(HPAC_PROBABILITY_VARIANTS.keys())


def resolve_hpac_probability_variant(variant = None):
    '''Resolve a named HPAC probability variant or fail closed on unknown input.'''
    if isinstance(variant, HpacProbabilityVariant):
        return variant
    
    try:
        return HPAC_PROBABILITY_VARIANTS[str(variant)]
    except KeyError:
        exc = None
        raise Pr86HpacReplayError('probability_variant_contract', 'unknown_hpac_probability_variant', requested_variant = str(variant), supported_variants = list(supported_hpac_probability_variant_names())), exc
        exc = None
        del exc



def default_source_artifact_paths():
    return (DEFAULT_MERGED_INTAKE_SUMMARY, DEFAULT_MERGED_SOURCE_MANIFEST, DEFAULT_MERGED_PR_API, DEFAULT_FULL_REENCODE, DEFAULT_TOKEN_ANATOMY, DEFAULT_PR85_PROBE, DEFAULT_PR_VIEW)


def _jsonable(value = None):
    if isinstance(value, Path):
        return repo_rel(value)
    if None(value, np.generic):
        return value.item()
    if None(value, torch.Tensor):
        if value.device.type == 'cpu' and value.numel() <= 16384:
            return {
                'shape': list(value.shape),
                'dtype': str(value.dtype),
                'sha256': sha256_bytes(value.detach().cpu().numpy().tobytes()) }
        return {
            'shape': None,
            'dtype': list(value.shape),
            'sha256': str(value.dtype) }
# WARNING: Decompyle incomplete


def _package_version(name = None):
    
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None



def _load_json_file(path = None):
    raw = path.read_bytes()
    
    try:
        payload = json.loads(raw.decode('utf-8'))
        return (payload, {
            'path': repo_rel(path),
            'bytes': len(raw),
            'sha256': sha256_bytes(raw) })
    except json.JSONDecodeError:
        exc = None
        raise Pr86HpacReplayError('source_artifacts', 'source_artifact_json_decode_failed', artifact = repo_rel(path), error = str(exc)), exc
        exc = None
        del exc



def _validate_safe_member_name(name = None):
    path = PurePosixPath(name)
    if name and name.endswith('/') and path.is_absolute() or '..' in path.parts:
        raise Pr86HpacReplayError('archive_member_contract', 'unsafe_zip_member_name', member_name = name)


def read_pr86_archive(archive = None, *, contract):
    '''Read a PR86 archive after strict member, identity, and zip-slip checks.'''
    archive = Path(archive)
    if not archive.is_file():
        raise Pr86HpacReplayError('archive_custody', 'archive_missing', archive = repo_rel(archive))
    archive_bytes = archive.stat().st_size
    archive_sha = sha256_path(archive)
# WARNING: Decompyle incomplete


def load_source_artifact_summaries(paths = None):
    '''Load default PR86 intake JSONs and keep only replay-relevant fields.'''
    summaries = { }
# WARNING: Decompyle incomplete


def analyze_pr86_current_source_context(source_dir = None):
    '''Summarize the current merged PR86 source contract used for HPAC replay.'''
    pass
# WARNING: Decompyle incomplete


def collect_dependency_report(*, expected_versions, require_behavior_self_test):
    '''Record installed replay dependency versions and constriction behavior.'''
    installed = {
        'python': sys.version.split()[0],
        'torch': _package_version('torch'),
        'numpy': _package_version('numpy'),
        'constriction': _package_version('constriction'),
        'pyppmd': _package_version('pyppmd') }
# WARNING: Decompyle incomplete


def _torch_load_bytes(raw = None, *, map_location):
    return torch.load(io.BytesIO(raw), map_location = map_location, weights_only = False)


def decode_gzip_torch_member(data = None, *, member_name):
    '''Gzip-decompress and torch-load a PR86 state-dict member.'''
    
    try:
        raw = gzip.decompress(data)
        
        try:
            payload = _torch_load_bytes(raw, map_location = 'cpu')
            report = _torch_payload_report(payload)
            report.update({
                'member_name': member_name,
                'compressed_bytes': len(data),
                'compressed_sha256': sha256_bytes(data),
                'decompressed_bytes': len(raw),
                'decompressed_sha256': sha256_bytes(raw),
                'status': 'passed' })
            return (payload, report)
            except OSError:
                exc = None
                raise Pr86HpacReplayError(f'''decode_{member_name}''', 'gzip_decode_failed', member_name = member_name, error = str(exc)), exc
                exc = None
                del exc
        except Exception:
            exc = None
            raise Pr86HpacReplayError(f'''decode_{member_name}''', 'torch_load_failed', member_name = member_name, error_type = type(exc).__name__, error = str(exc)), exc
            exc = None
            del exc




def decode_meta_member(data = None):
    pass
# WARNING: Decompyle incomplete


def _torch_payload_report(payload = None):
    report = {
        'payload_type': type(payload).__name__ }
    if isinstance(payload, Mapping):
        keys = (lambda .0: pass# WARNING: Decompyle incomplete
)(payload.keys()())
        report['key_count'] = len(keys)
        report['key_prefix'] = keys[:20]
    return report


def _patch_group_mask(k = None, delta = None, type_ = None):
    mask = torch.zeros(k, k, dtype = torch.float32)
    center = (k - 1) // 2
    for dr_idx in range(k):
        for dc_idx in range(k):
            dr = dr_idx - center
            dc = dc_idx - center
            val = dc + delta * dr
            if type_ == 'A':
                if not val < 0:
                    continue
                mask[(dr_idx, dc_idx)] = 1
                continue
            if not val <= 0:
                continue
            mask[(dr_idx, dc_idx)] = 1
    return mask


class _MaskedConv2dPG(nn.Module):
    pass
# WARNING: Decompyle incomplete


class _ChannelNorm2d(nn.Module):
    pass
# WARNING: Decompyle incomplete


class _CausalSPM(nn.Module):
    pass
# WARNING: Decompyle incomplete


class HPACMini(nn.Module):
    pass
# WARNING: Decompyle incomplete


def reconstruct_hpac_state_dict(packed_sd = None, *, device):
    '''Rehydrate PR86 INT8-packed HPAC weights to FP32 state-dict entries.'''
    out = { }
    bases = (lambda .0: pass# WARNING: Decompyle incomplete
)(packed_sd())
# WARNING: Decompyle incomplete


def load_hpac_model_from_ppmd(data = None, *, config, device):
    '''Decode ``hpac.pt.ppmd`` and load a PR86 HPACMini entropy model.'''
    if device != 'cpu':
        raise Pr86HpacReplayError('device_contract', 'pr86_hpac_replay_is_cpu_only', requested_device = device)
    
    try:
        import pyppmd
        
        try:
            raw = pyppmd.decompress(data, max_order = PPMD_MAX_ORDER, mem_size = PPMD_MEM_SIZE)
            
            try:
                packed_sd = _torch_load_bytes(raw, map_location = 'cpu')
                if not isinstance(packed_sd, Mapping):
                    raise Pr86HpacReplayError('decode_hpac_pt_ppmd', 'hpac_payload_not_state_dict', payload_type = type(packed_sd).__name__)
                n_frames = int(config.get('N', 600))
                patch = int(config.get('P', 32))
                delta = int(config.get('delta', 2))
                ch = int(config.get('ch', 64))
                d_film = int(config.get('hpac_d_film', config.get('d_film', 32)))
                use_spm = bool(config.get('use_spm', False))
                sd = reconstruct_hpac_state_dict(packed_sd, device = device)
                gen = HPACMini(num_pairs = n_frames, num_classes = NUM_CLASSES, P = patch, delta = delta, ch = ch, d_film = d_film, use_spm = use_spm).to(device).eval()
                incompatible = gen.load_state_dict(sd, strict = False)
                missing = list(incompatible.missing_keys)
                unexpected = list(incompatible.unexpected_keys)
                if missing or unexpected:
                    raise Pr86HpacReplayError('decode_hpac_pt_ppmd', 'hpac_state_dict_key_mismatch', missing_keys = missing, unexpected_keys = unexpected)
                return (gen, {
                    'member_name': 'hpac.pt.ppmd',
                    'compressed_bytes': len(data),
                    'compressed_sha256': sha256_bytes(data),
                    'decompressed_bytes': len(raw),
                    'decompressed_sha256': sha256_bytes(raw),
                    'packed_state_key_count': len(packed_sd),
                    'reconstructed_state_key_count': len(sd),
                    'config': {
                        'N': n_frames,
                        'P': patch,
                        'delta': delta,
                        'ch': ch,
                        'hpac_d_film': d_film,
                        'use_spm': use_spm },
                    'ppmd_max_order': PPMD_MAX_ORDER,
                    'ppmd_mem_size': PPMD_MEM_SIZE,
                    'load_state_dict_strict': False,
                    'missing_keys': missing,
                    'unexpected_keys': unexpected,
                    'status': 'passed' })
                except ImportError:
                    exc = None
                    raise Pr86HpacReplayError('dependency_contract', 'missing_pyppmd'), exc
                    exc = None
                    del exc
                except Exception:
                    exc = None
                    raise Pr86HpacReplayError('decode_hpac_pt_ppmd', 'ppmd_decompress_failed', error_type = type(exc).__name__, error = str(exc)), exc
                    exc = None
                    del exc
            except Exception:
                exc = None
                raise Pr86HpacReplayError('decode_hpac_pt_ppmd', 'torch_load_failed', error_type = type(exc).__name__, error = str(exc)), exc
                exc = None
                del exc





def _patch_group_grid(P = None, delta = None, device = None):
    rows = torch.arange(P, device = device).view(P, 1).expand(P, P)
    cols = torch.arange(P, device = device).view(1, P).expand(P, P)
    return cols + delta * rows


def _full_mask_for_group(s_grid = None, group = None, n_row_patches = None, n_col_patches = ('s_grid', 'torch.Tensor', 'group', 'int', 'n_row_patches', 'int', 'n_col_patches', 'int', 'return', 'torch.Tensor')):
    patch = s_grid.shape[0]
    mask_p = s_grid == group
    full = mask_p.unsqueeze(0).unsqueeze(0).expand(n_row_patches, n_col_patches, patch, patch)
    return full.permute(0, 2, 1, 3).reshape(n_row_patches * patch, n_col_patches * patch)


def _group_masks(H, W = None, P = None, delta = None, device = ('H', 'int', 'W', 'int', 'P', 'int', 'delta', 'int', 'device', 'torch.device', 'return', 'list[torch.Tensor | None]')):
    if H % P != 0 or W % P != 0:
        raise Pr86HpacReplayError('hpac_geometry_contract', 'frame_dimensions_not_divisible_by_patch', height = H, width = W, P = P)
    n_col_patches = W // P
    n_row_patches = H // P
    s_grid = _patch_group_grid(P, delta, device)
    masks = []
    for group in range(int((1 + delta) * P - delta)):
        mask = _full_mask_for_group(s_grid, group, n_row_patches, n_col_patches)
        masks.append(mask if bool(mask.any().item()) else None)
    return masks


def _normalize_probability_row(probs = None, *, prob_eps, variant):
    dtype = np.float32 if variant.probability_dtype == 'float32' else np.float64
    clipped = np.clip(probs.astype(dtype, copy = False), dtype(prob_eps), dtype(1))
    clipped = clipped / clipped.sum()
    return clipped.astype(dtype, copy = False)


def _categorical_from_probs(probs = None, *, prob_eps, variant):
    import constriction
    clipped = _normalize_probability_row(probs, prob_eps = prob_eps, variant = variant)
    return constriction.stream.model.Categorical(probabilities = clipped, perfect = variant.categorical_perfect)

decode_tokens_hpac = (lambda gen = None, token_blob = None, *, N, H: if device != 'cpu':
raise Pr86HpacReplayError('device_contract', 'pr86_hpac_replay_is_cpu_only', requested_device = device)if len(token_blob) % 4 != 0:
raise Pr86HpacReplayError('tokens_bin_contract', 'tokens_bin_not_uint32_aligned', tokens_bytes = len(token_blob))# WARNING: Decompyle incomplete
)()
encode_tokens_hpac = (lambda gen = None, tokens = None, *, P, delta: if device != 'cpu':
raise Pr86HpacReplayError('device_contract', 'pr86_hpac_replay_is_cpu_only', requested_device = device)# WARNING: Decompyle incomplete
)()
decode_symbols_hpac_with_prev_context = (lambda gen = None, token_blob = None, prev_context_tokens = None, *, P, delta, device: if device != 'cpu':
raise Pr86HpacReplayError('device_contract', 'pr86_hpac_replay_is_cpu_only', requested_device = device)if prev_context_tokens.ndim != 3:
raise Pr86HpacReplayError('decode_symbols', 'prev_context_must_be_nhw', shape = list(prev_context_tokens.shape))if len(token_blob) % 4 != 0:
raise Pr86HpacReplayError('tokens_bin_contract', 'tokens_bin_not_uint32_aligned', tokens_bytes = len(token_blob))# WARNING: Decompyle incomplete
)()
encode_symbols_hpac_with_prev_context = (lambda gen = None, symbols = None, prev_context_tokens = None, *, P, delta, device: if device != 'cpu':
raise Pr86HpacReplayError('device_contract', 'pr86_hpac_replay_is_cpu_only', requested_device = device)# WARNING: Decompyle incomplete
)()

def _first_mismatch(left = None, right = None):
    for left_byte, right_byte in enumerate(zip(left, right, strict = False)):
        if not left_byte != right_byte:
            continue
        
        return None, index
    if len(left) != len(right):
        return min(len(left), len(right))


def _validate_dependency_report(report = None):
    if str(report.get('status', '')).startswith('failed_closed'):
        raise Pr86HpacReplayError('dependency_contract', str(report.get('status')), dependency_contract = _jsonable(report))


def _decode_required_members(bundle = None, *, device):
    members = bundle.members
    (_master_payload, master_report) = decode_gzip_torch_member(members['master.pt.gz'], member_name = 'master.pt.gz')
    (_slave_payload, slave_report) = decode_gzip_torch_member(members['slave.pt.gz'], member_name = 'slave.pt.gz')
    (meta_payload, meta_report) = decode_meta_member(members['meta.pt'])
    (hpac_model, hpac_report) = load_hpac_model_from_ppmd(members['hpac.pt.ppmd'], config = meta_payload, device = device)
    reports = {
        'master.pt.gz': master_report,
        'slave.pt.gz': slave_report,
        'meta.pt': meta_report,
        'hpac.pt.ppmd': hpac_report }
    return (reports, hpac_model, meta_payload)


def run_pr86_hpac_replay(archive = None, *, contract, source_dir, source_artifacts, device, max_frames, attempt_reencode, probability_variant):
    '''Run the fail-closed local replay gate and return a JSON-safe report.'''
    started_at = time.time()
# WARNING: Decompyle incomplete


def run_pr86_hpac_probability_variant_matrix(archive = None, *, variants, contract, source_dir, source_artifacts, device, max_frames, attempt_reencode):
    '''Run named HPAC probability variants and summarize the fail-closed gate.'''
    started_at = time.time()
    variant_names = dict.fromkeys((lambda .0: pass# WARNING: Decompyle incomplete
)(variants()))
    results = []
    for name in variant_names:
        results.append(run_pr86_hpac_replay(archive = archive, contract = contract, source_dir = source_dir, source_artifacts = source_artifacts, device = device, max_frames = max_frames, attempt_reencode = attempt_reencode, probability_variant = name))
    dispatch_unlocked = (lambda .0: pass# WARNING: Decompyle incomplete
)(results())
# WARNING: Decompyle incomplete


"""
