"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``73:81: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``profile_pr95_hnerv_muon_intake.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'experiments/profile_pr95_hnerv_muon_intake.py'
__recovery_spec__ = 'profile_pr95_hnerv_muon_intake.recovery_spec.json'
__recovery_ast_error__ = '73:81: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: profile_pr95_hnerv_muon_intake.cpython-312.pyc (Python 3.12)

'''Static PR95 HNeRV/Muon intake profiler.

This tool performs local byte/source accounting only. It does not run the
contest scorer, does not load scorer models, and does not dispatch GPU work.
'''
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
import re
import struct
import zipfile
from pathlib import Path
from typing import Any
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INTAKE_DIR = REPO_ROOT / 'experiments/results/public_pr95_intake_20260504_codex'
DEFAULT_ARCHIVE = DEFAULT_INTAKE_DIR / 'archive.zip'
DEFAULT_SOURCE_DIR = DEFAULT_INTAKE_DIR / 'pr95_src/submissions/hnerv_muon'
DEFAULT_STATIC_INTAKE = DEFAULT_INTAKE_DIR / 'pr95_static_intake.json'
DEFAULT_JSON_OUT = DEFAULT_INTAKE_DIR / 'profile_pr95_hnerv_muon_intake.json'
DEFAULT_MARKDOWN_OUT = DEFAULT_INTAKE_DIR / 'profile_pr95_hnerv_muon_intake.md'
CONTEST_ORIGINAL_BYTES = 37545489
SCHEMA = 'pr95_hnerv_muon_static_intake_profile_v1'
TOOL = 'experiments/profile_pr95_hnerv_muon_intake.py'

def _sha256_bytes(data = None):
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path = None):
    pass
# WARNING: Decompyle incomplete


def _json_text(payload = None):
    return json.dumps(payload, indent = 2, sort_keys = True, allow_nan = False) + '\n'


def _rel(path = None):
    
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return 



def _read_u32(raw = None, cursor = None, label = None):
    if cursor + 4 > len(raw):
        raise ValueError(f'''truncated u32 while reading {label}''')
    return (struct.unpack_from('<I', raw, cursor)[0], cursor + 4)


def _product(values = None):
    out = 1
    for value in values:
        out *= value
    return out


def _require_brotli():
    
    try:
        import brotli
        return brotli
    except ImportError:
        exc = None
        raise RuntimeError('brotli is required for PR95 HNeRV/Muon blob intake'), exc
        exc = None
        del exc



def read_pr95_archive(path = None):
    '''Read and validate the public PR95 single-member archive.'''
    zf = zipfile.ZipFile(path, 'r')
# WARNING: Decompyle incomplete


def parse_decoder_blob(blob = None):
    '''Parse ``codec.encode_decoder`` output after brotli decompression.'''
    cursor = 0
    (tensor_count, cursor) = _read_u32(blob, cursor, 'decoder tensor count')
    tensors = []
    total_params = 0
    muon_params = 0
    adamw_decoder_params = 0
    groups = { }
    for index in range(tensor_count):
        (name_len, cursor) = _read_u32(blob, cursor, f'''tensor {index} name length''')
        if cursor + name_len > len(blob):
            raise ValueError(f'''truncated tensor name at index {index}''')
        name = blob[cursor:cursor + name_len].decode('utf-8')
        cursor += name_len
        (ndim, cursor) = _read_u32(blob, cursor, f'''{name} ndim''')
        shape_values = []
        for _ in range(ndim):
            (value, cursor) = _read_u32(blob, cursor, f'''{name} shape''')
            shape_values.append(value)
        if cursor + 4 > len(blob):
            raise ValueError(f'''truncated scale for tensor {name}''')
        scale = struct.unpack_from('<f', blob, cursor)[0]
        cursor += 4
        (size, cursor) = _read_u32(blob, cursor, f'''{name} quantized size''')
        if cursor + size > len(blob):
            raise ValueError(f'''truncated quantized payload for tensor {name}''')
        qbytes = blob[cursor:cursor + size]
        cursor += size
        shape = tuple(shape_values)
        params = _product(shape)
        if params != size:
            raise ValueError(f'''tensor {name} shape product {params} != stored size {size}''')
        lower = name.lower()
        if ndim >= 2:
            ndim >= 2
            if 'stem' not in lower:
                'stem' not in lower
                if not lower.startswith('rgb'):
                    not lower.startswith('rgb')
        is_muon = '.rgb_' not in lower
        if is_muon:
            muon_params += params
            optimizer_partition = 'muon_hidden_2d_plus_weight'
        else:
            adamw_decoder_params += params
            optimizer_partition = 'adamw_decoder_or_bias'
        group = name.split('.', 1)[0]
        groups[group] = groups.get(group, 0) + params
        total_params += params
        tensors.append({
            'name': name,
            'shape': list(shape),
            'params': params,
            'quantized_bytes': size,
            'scale': scale,
            'zigzag_byte_min': min(qbytes) if qbytes else None,
            'zigzag_byte_max': max(qbytes) if qbytes else None,
            'optimizer_partition': optimizer_partition })
    if cursor != len(blob):
        raise ValueError(f'''decoder blob has trailing bytes: {len(blob) - cursor}''')
    return {
        'tensor_count': tensor_count,
        'total_params': total_params,
        'muon_partition_params': muon_params,
        'adamw_decoder_partition_params': adamw_decoder_params,
        'parameter_groups': dict(sorted(groups.items())),
        'top_tensors_by_params': sorted(tensors, key = (lambda item: item['params']), reverse = True)[:12],
        'tensors': tensors,
        'raw_decoder_table_bytes': len(blob) }


def parse_latents_payload(raw = None):
    '''Parse ``codec.encode_latents`` output after brotli decompression.'''
    cursor = 0
    (n_pairs, cursor) = _read_u32(raw, cursor, 'latent pair count')
    (latent_dim, cursor) = _read_u32(raw, cursor, 'latent dim')
    fp16_table_bytes = latent_dim * 2
    mins_start = cursor
    cursor += fp16_table_bytes
    scales_start = cursor
    cursor += fp16_table_bytes
    if cursor > len(raw):
        raise ValueError('truncated latent min/scale tables')
    total = n_pairs * latent_dim
    lo_start = cursor
    cursor += total
    hi_start = cursor
    cursor += total
    if cursor != len(raw):
        raise ValueError(f'''latent payload accounting mismatch: cursor={cursor}, len={len(raw)}''')
    mins = struct.unpack('<' + 'e' * latent_dim, raw[mins_start:mins_start + fp16_table_bytes])
    scales = struct.unpack('<' + 'e' * latent_dim, raw[scales_start:scales_start + fp16_table_bytes])
    hi = raw[hi_start:hi_start + total]
    if scales:
        return {
            'n_frame_pairs': 8,
            'latent_dim': fp16_table_bytes,
            'latent_values': fp16_table_bytes,
            'payload_bytes': total,
            'header_bytes': total,
            'mins_fp16_bytes': None,
            'scales_fp16_bytes': sum,
            'lo_delta_bytes': (lambda .0: pass# WARNING: Decompyle incomplete
)(hi()),
            'hi_delta_bytes': None,
            'hi_nonzero_count': sum,
            'hi_nonzero_fraction': (lambda .0: pass# WARNING: Decompyle incomplete
)(hi()) / total if total else 0,
            'mins_range': [
                float(min(mins)),
                float(max(mins))] if mins else [
                None,
                None],
            'scales_range': [
                float(min(scales)),
                float(max(scales))] }
    return {
        'n_frame_pairs': len(raw),
        'latent_dim': 8,
        'latent_values': fp16_table_bytes,
        'payload_bytes': fp16_table_bytes,
        'header_bytes': total,
        'mins_fp16_bytes': total,
        'scales_fp16_bytes': None,
        'lo_delta_bytes': sum,
        'hi_delta_bytes': (lambda .0: pass# WARNING: Decompyle incomplete
)(hi()),
        'hi_nonzero_count': None,
        'hi_nonzero_fraction': sum,
        'mins_range': (lambda .0: pass# WARNING: Decompyle incomplete
)(hi()) / total if total else 0,
        'scales_range': [
            [
                float(min(mins)),
                float(max(mins))] if mins else [
                None,
                None],
            None] }


def parse_hnerv_muon_member(payload = None):
    '''Parse PR95 ``0.bin`` into meta, decoder, and latent sections.'''
    brotli = _require_brotli()
    cursor = 0
    sections = []
    (meta_len, cursor) = _read_u32(payload, cursor, 'meta brotli length')
    meta_brotli_start = cursor
    meta_brotli = payload[cursor:cursor + meta_len]
    cursor += meta_len
    if cursor > len(payload):
        raise ValueError('truncated meta section')
    meta = json.loads(brotli.decompress(meta_brotli))
    sections.append({
        'name': 'meta_json_brotli',
        'length_prefix_offset': 0,
        'compressed_bytes': meta_len,
        'uncompressed_bytes': len(json.dumps(meta, sort_keys = True).encode('utf-8')),
        'sha256': _sha256_bytes(meta_brotli) })
    decoder_len_offset = cursor
    (decoder_len, cursor) = _read_u32(payload, cursor, 'decoder blob length')
    decoder_blob = payload[cursor:cursor + decoder_len]
    cursor += decoder_len
    if cursor > len(payload):
        raise ValueError('truncated decoder section')
    decoder_raw = brotli.decompress(decoder_blob)
    decoder = parse_decoder_blob(decoder_raw)
    sections.append({
        'name': 'decoder_state_int8_brotli',
        'length_prefix_offset': decoder_len_offset,
        'compressed_bytes': decoder_len,
        'uncompressed_bytes': len(decoder_raw),
        'sha256': _sha256_bytes(decoder_blob) })
    latents_len_offset = cursor
    (latents_len, cursor) = _read_u32(payload, cursor, 'latents brotli length')
    latents_blob = payload[cursor:cursor + latents_len]
    cursor += latents_len
    if cursor != len(payload):
        raise ValueError(f'''0.bin has trailing bytes or truncation: cursor={cursor}, len={len(payload)}''')
    latents_raw = brotli.decompress(latents_blob)
    latents = parse_latents_payload(latents_raw)
    sections.append({
        'name': 'latents_delta_uint8_brotli',
        'length_prefix_offset': latents_len_offset,
        'compressed_bytes': latents_len,
        'uncompressed_bytes': len(latents_raw),
        'sha256': _sha256_bytes(latents_blob) })
    return {
        'member_format': 12,
        'member_bytes': meta_brotli_start,
        'section_length_prefix_bytes': sections,
        'meta_brotli_offset': meta,
        'sections': decoder,
        'meta': latents,
        'decoder': None,
        'latents': sum,
        'compressed_payload_bytes': (lambda .0: pass# WARNING: Decompyle incomplete
)(sections()) }


def _docstring(path = None):
    
    try:
        module = ast.parse(path.read_text(encoding = 'utf-8'))
        if not ast.get_docstring(module):
            ast.get_docstring(module)
        return ''
    except SyntaxError:
        return ''



def _first_match(pattern = None, text = None):
    match = re.search(pattern, text)
    if match:
        return match.group(1)


def _literal_number(pattern = None, text = None):
    value = _first_match(pattern, text)
# WARNING: Decompyle incomplete


def parse_source_summary(source_dir = None):
    '''Extract deterministic PR95 HNeRV/Muon source and curriculum facts.'''
    readme_path = source_dir / 'README.md'
    model_path = source_dir / 'src/model.py'
    optim_path = source_dir / 'src/optim.py'
    codec_path = source_dir / 'src/codec.py'
    score_path = source_dir / 'src/score.py'
    train_path = source_dir / 'src/train.py'
    stage_dir = source_dir / 'src/stages'
    source_files = (lambda .0: pass# WARNING: Decompyle incomplete
)(source_dir.rglob('*')())
    readme = readme_path.read_text(encoding = 'utf-8') if readme_path.exists() else ''
    model_text = model_path.read_text(encoding = 'utf-8') if model_path.exists() else ''
    optim_text = optim_path.read_text(encoding = 'utf-8') if optim_path.exists() else ''
    score_text = score_path.read_text(encoding = 'utf-8') if score_path.exists() else ''
    stage_records = []
    for stage_path in sorted(stage_dir.glob('stage*.py')):
        text = stage_path.read_text(encoding = 'utf-8')
        if not _first_match('name="([^"]+)"', text):
            _first_match('name="([^"]+)"', text)
        if 'ce_seg_loss' in text:
            pass
        elif 'tau_softplus_seg_loss' in text:
            pass
        elif 'smooth_disagreement_seg_loss' in text:
            pass
        elif 'l7_softplus_seg_loss' in text:
            pass
        
        _literal_number('epochs:\\s*int\\s*=\\s*(\\d+)', text)({
            'file': _literal_number('adamw_lr=([0-9.eE+-]+)', text),
            'name': _literal_number('muon_lr=([0-9.eE+-]+)', text),
            'docstring_head': _literal_number('muon_weight_decay:\\s*float\\s*=\\s*([0-9.eE+-]+)', text),
            'epochs_default': _literal_number('cat_lambda=([0-9.eE+-]+)', text),
            'adamw_lr': _literal_number('cat_sigma=([0-9.eE+-]+)', text),
            'muon_lr': 'use_qat=True' in text,
            'muon_weight_decay_default': 'use_muon=True' in text,
            'cat_lambda': 'ce',
            'cat_sigma': 'tau_softplus',
            'uses_qat': 'smooth_disagreement',
            'uses_muon': 'l7_softplus',
            'loss_family': 'unknown' })
    score_formula = _docstring(score_path).splitlines()[2].strip() if score_path.exists() else ''
    default_latent_dim = _literal_number('latent_dim=(\\d+)', model_text)
    default_base_channels = _literal_number('base_channels=(\\d+)', model_text)
    eval_size = _first_match('eval_size=\\((\\d+,\\s*\\d+)\\)', model_text)
    if not score_formula:
        score_formula
    return {
        'source_dir': _rel(source_dir),
        'source_file_count': len(source_files),
        'source_tree_sha256': _source_tree_sha256(source_files),
        'key_files': {
            'README': _rel(readme_path),
            'model': _rel(model_path),
            'optimizer': _rel(optim_path),
            'codec': _rel(codec_path),
            'score': _rel(score_path),
            'train': _rel(train_path) },
        'readme_summary': {
            'title': readme.splitlines()[0] if readme else '',
            'archive_claim': _first_match('A\\s+([^\\n]+archive[^\\n]+)', readme),
            'curriculum_claim': _first_match('The pipeline is an ([^\\n]+)', readme),
            'reproduce_command': _first_match('```bash\\n([^`]+)\\n```', readme),
            'claimed_training_wallclock': _first_match('(~\\d+ hours[^\\n]+)', readme),
            'external_writeup': _first_match('Full writeup:\\s*(\\S+)', readme) },
        'model_defaults': {
            'latent_dim': default_latent_dim,
            'base_channels': default_base_channels,
            'eval_size': eval_size,
            'base_grid': [
                6,
                8] if 'self.base_h, self.base_w = 6, 8' in model_text else None,
            'architecture': 'linear latent stem + six PixelShuffle upsample blocks + dilated refine + two RGB heads' },
        'optimizer_summary': {
            'muon_newton_schulz_steps_default': _literal_number('ns_steps=([0-9]+)', optim_text),
            'muon_default_lr': _literal_number('lr=([0-9.eE+-]+)', optim_text),
            'muon_default_momentum': _literal_number('momentum=([0-9.eE+-]+)', optim_text),
            'muon_default_weight_decay': _literal_number('weight_decay=([0-9.eE+-]+)', optim_text),
            'partition_rule': 'Muon receives 2D+ weights outside stem/RGB heads; AdamW receives stem, RGB heads, biases, 1D params, and latents.' },
        'codec_summary': {
            'decoder_path': 'per-tensor symmetric INT8 -> zigzag -> state table -> brotli q11',
            'latent_path': 'per-dim uint8 min/max -> first-order temporal delta -> zigzag uint16 -> lo/hi streams -> brotli q11',
            'source_notes_hybrid_ac_delta_bytes': _literal_number('was ~(\\d+) bytes smaller', codec_path.read_text(encoding = 'utf-8') if codec_path.exists() else '') },
        'score_formula': 'score = 100 * seg_distortion + sqrt(10 * pose_distortion) + 25 * archive_bytes / total_video_bytes',
        'training_stages': stage_records }


def _source_tree_sha256(files = None):
    digest = hashlib.sha256()
    for path in files:
        digest.update(_rel(path).encode('utf-8'))
        digest.update(b'\x00')
        digest.update(_sha256_file(path).encode('ascii'))
        digest.update(b'\x00')
    return digest.hexdigest()


def load_static_intake(path = None):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding = 'utf-8'))


def compute_score_terms(seg = None, pose = None, archive_bytes = None, denominator = (CONTEST_ORIGINAL_BYTES,)):
    seg_component = 100 * seg
    pose_component = math.sqrt(10 * pose)
    rate_component = 25 * archive_bytes / denominator
    return {
        'seg_component': seg_component,
        'pose_component': pose_component,
        'rate_component': rate_component,
        'recomputed_score': seg_component + pose_component + rate_component }


def build_profile(archive_path = None, source_dir = None, static_intake_path = None):
    (archive, payload) = read_pr95_archive(archive_path)
    member = parse_hnerv_muon_member(payload)
    source = parse_source_summary(source_dir)
    static_intake = load_static_intake(static_intake_path) if static_intake_path else None
    score_claims = {
        'score_claim': False,
        'score_terms_from_static_intake': None,
        'formula': 'score = 100 * seg + sqrt(10 * pose) + 25 * archive_bytes / 37,545,489',
        'denominator_bytes': CONTEST_ORIGINAL_BYTES }
    if static_intake and static_intake.get('claimed_body_score_inputs'):
        inputs = static_intake['claimed_body_score_inputs']
        recomputed = compute_score_terms(float(inputs['seg']), float(inputs['pose']), int(inputs['archive_bytes']))
        score_claims.update({
            'score_claim': bool(static_intake.get('score_claim', False)),
            'score_terms_from_static_intake': {
                'inputs': inputs,
                'recomputed': recomputed,
                'matches_recorded_recomputed_score': abs(recomputed['recomputed_score'] - float(inputs.get('recomputed_score', recomputed['recomputed_score']))) < 1e-12 } })
    return {
        'schema': SCHEMA,
        'tool': TOOL,
        'inputs': {
            'archive': _rel(archive_path),
            'source_dir': _rel(source_dir),
            'static_intake': _rel(static_intake_path) if static_intake_path and static_intake_path.exists() else None },
        'evidence_grade': 'external_static_intake_only',
        'archive_anatomy': archive,
        'hnerv_muon_blob': member,
        'source_intake': source,
        'score_term_math': score_claims,
        'dispatch_readiness': dispatch_readiness(),
        'immediate_improvement_hypotheses': improvement_hypotheses(member) }


def dispatch_readiness():
    return {
        'ready_for_dispatch': False,
        'fail_closed': True,
        'remote_dispatch_requested': False,
        'required_before_score_claims': [
            'Replay exact eval through archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA before any PR95 score claim.',
            'Record contest_auth_eval.json, archive SHA-256, archive bytes, runtime tree hash, hardware, sample count, and recomputed score.',
            'Owned retraining needs explicit manifest/checkpoint custody for every stage: source SHA, seed, stage config, checkpoint path, checkpoint SHA-256, optimizer state policy, and final archive builder provenance.'],
        'blocked_claims': [
            'Static PR95 public intake is not promotable score evidence.',
            'README/body score inputs remain external until replayed under our exact CUDA auth eval custody.',
            'Any HNeRV retrain without checkpoint and manifest custody is non-promotable replay work.'] }


def improvement_hypotheses(member_profile = None):
    decoder = member_profile['decoder']
    latents = member_profile['latents']
# WARNING: Decompyle incomplete


def render_markdown(profile = None):
    archive = profile['archive_anatomy']
    member = profile['hnerv_muon_blob']
    decoder = member['decoder']
    latents = member['latents']
    readiness = profile['dispatch_readiness']
    score_terms = profile['score_term_math']['score_terms_from_static_intake']
    lines = [
        '# PR95 HNeRV/Muon Static Intake',
        '',
        '## Archive Anatomy',
        '',
        f'''- archive: `{archive['path']}`''',
        f'''- bytes: `{archive['archive_bytes']}`''',
        f'''- sha256: `{archive['archive_sha256']}`''',
        f'''- member: `{archive['members'][0]['name']}` stored bytes `{archive['members'][0]['file_size']}`''',
        f'''- rate component at contest denominator: `{archive['rate_score_component']:.12f}`''',
        '',
        '## Blob Sections',
        '']
    for section in member['sections']:
        lines.append(f'''- `{section['name']}`: compressed `{section['compressed_bytes']}`, uncompressed `{section['uncompressed_bytes']}`, sha256 `{section['sha256']}`''')
    lines.extend([
        '',
        '## Parameter And Latent Counts',
        '',
        f'''- decoder tensors: `{decoder['tensor_count']}`''',
        f'''- decoder params: `{decoder['total_params']}`''',
        f'''- Muon partition params: `{decoder['muon_partition_params']}`''',
        f'''- AdamW decoder partition params: `{decoder['adamw_decoder_partition_params']}`''',
        f'''- latent matrix: `{latents['n_frame_pairs']} x {latents['latent_dim']}`''',
        f'''- latent hi-byte nonzero fraction: `{latents['hi_nonzero_fraction']:.12f}`''',
        '',
        '## Score-Term Math',
        ''])
# WARNING: Decompyle incomplete


def write_outputs(profile = None, json_out = None, markdown_out = None):
    json_out.parent.mkdir(parents = True, exist_ok = True)
    markdown_out.parent.mkdir(parents = True, exist_ok = True)
    json_out.write_text(_json_text(profile), encoding = 'utf-8')
    markdown_out.write_text(render_markdown(profile), encoding = 'utf-8')


def parse_args(argv = None):
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--archive', type = Path, default = DEFAULT_ARCHIVE)
    parser.add_argument('--source-dir', type = Path, default = DEFAULT_SOURCE_DIR)
    parser.add_argument('--static-intake', type = Path, default = DEFAULT_STATIC_INTAKE)
    parser.add_argument('--json-out', type = Path, default = DEFAULT_JSON_OUT)
    parser.add_argument('--markdown-out', type = Path, default = DEFAULT_MARKDOWN_OUT)
    parser.add_argument('--no-write', action = 'store_true', help = 'Build the profile and print JSON without writing artifacts.')
    return parser.parse_args(argv)


def main(argv = None):
    args = parse_args(argv)
    profile = build_profile(args.archive, args.source_dir, args.static_intake)
    if args.no_write:
        print(_json_text(profile), end = '')
        return 0
    write_outputs(profile, args.json_out, args.markdown_out)
    print(f'''wrote {_rel(args.json_out)}''')
    print(f'''wrote {_rel(args.markdown_out)}''')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

"""
