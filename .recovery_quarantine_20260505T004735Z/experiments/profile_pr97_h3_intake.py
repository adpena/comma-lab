"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``41:14: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``profile_pr97_h3_intake.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'experiments/profile_pr97_h3_intake.py'
__recovery_spec__ = 'profile_pr97_h3_intake.recovery_spec.json'
__recovery_ast_error__ = '41:14: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: profile_pr97_h3_intake.cpython-312.pyc (Python 3.12)

'''Offline byte/profile intake for PR97 H3 public archive.

This tool is deliberately static. It parses the single-member ``p`` payload,
checks the PR97-specific H3 subformats, and builds deterministic byte-only
repack candidates. It never runs inflate, loads scorers, uses CUDA, or submits
remote work.
'''
from __future__ import annotations
import argparse
import ast
import dataclasses
import hashlib
import io
import json
import lzma
import math
import struct
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence

try:
    import brotli
    SCHEMA = 'pr97_h3_static_intake_profile_v1'
    TOOL = 'experiments/profile_pr97_h3_intake.py'
    EVIDENCE_GRADE = 'external_archive_byte_intake_only_until_exact_cuda_replay'
    CONTEST_ORIGINAL_BYTES = 37545489
    DEFAULT_ARCHIVE = 'experiments/results/leaderboard_intel_20260504_codex/pr97_archive.zip'
    DEFAULT_RUNTIME = 'experiments/results/leaderboard_intel_20260504_codex/pr97_runtime'
    DEFAULT_OUTPUT_DIR = 'experiments/results/pr97_h3_intake_20260504_codex'
    PAYLOAD_PARTS = ('mask', 'pose', 'model', 'sidecar')
    
    class PR97ProfileError(ValueError):
        '''Raised when the PR97 archive cannot be safely parsed.'''
        pass

    Member = <NODE:12>()
    
    def sha256_bytes(data = None):
        return hashlib.sha256(data).hexdigest()

    
    def sha256_file(path = None):
        pass
    # WARNING: Decompyle incomplete

    
    def json_text(payload = None):
        return json.dumps(payload, indent = 2, sort_keys = True, allow_nan = False) + '\n'

    
    def contest_rate_term(byte_count = None):
        return 25 * int(byte_count) / CONTEST_ORIGINAL_BYTES

    
    def safe_member_blockers(name = None):
        blockers = []
        if not name:
            blockers.append('empty_member_name')
        if '\x00' in name:
            blockers.append('nul_in_member_name')
        if '\\' in name:
            blockers.append('backslash_in_member_name')
        posix = PurePosixPath(name)
        windows = PureWindowsPath(name)
        if posix.is_absolute() and windows.is_absolute() or windows.drive:
            blockers.append('absolute_or_drive_member_path')
        if (lambda .0: pass# WARNING: Decompyle incomplete
)(posix.parts()):
            blockers.append('zip_slip_member_path')
        if (lambda .0: pass# WARNING: Decompyle incomplete
)(posix.parts()):
            blockers.append('hidden_or_resource_fork_member')
        return blockers

    
    def read_members(path = None):
        if not path.is_file():
            raise FileNotFoundError(f'''archive not found: {path}''')
        zf = zipfile.ZipFile(path, 'r')
    # WARNING: Decompyle incomplete

    
    def split_payload(blob = None):
        offset = 0
        parts = { }
        for name in PAYLOAD_PARTS:
            if offset + 4 > len(blob):
                raise PR97ProfileError(f'''truncated payload length before {name}''')
            size = struct.unpack_from('<I', blob, offset)[0]
            offset += 4
            end = offset + size
            if end > len(blob):
                raise PR97ProfileError(f'''truncated payload part {name}: need {size} bytes''')
            parts[name] = blob[offset:end]
            offset = end
        if offset != len(blob):
            raise PR97ProfileError(f'''payload trailing bytes: {offset} vs {len(blob)}''')
        return parts

    
    def parse_mask(mask = None):
        if len(mask) < 4:
            raise PR97ProfileError('mask payload is too short')
        offset = 0
        n_chunks = struct.unpack_from('<I', mask, offset)[0]
        offset += 4
        chunks = []
        for index in range(n_chunks):
            if offset + 4 > len(mask):
                raise PR97ProfileError(f'''truncated mask chunk length at {index}''')
            size = struct.unpack_from('<I', mask, offset)[0]
            offset += 4
            end = offset + size
            if end > len(mask):
                raise PR97ProfileError(f'''truncated mask chunk {index}: need {size} bytes''')
            data = mask[offset:end]
            chunks.append({
                'index': index,
                'bytes': size,
                'sha256': sha256_bytes(data),
                'magic_ascii': data[:4].decode('ascii', errors = 'replace'),
                'magic_hex': data[:8].hex() })
            offset = end
        if offset != len(mask):
            raise PR97ProfileError(f'''mask trailing bytes: {offset} vs {len(mask)}''')
    # WARNING: Decompyle incomplete

    
    def parse_pose(blob = None):
        
        try:
            raw = brotli.decompress(blob)
            if len(raw) < 8:
                raise PR97ProfileError('pose raw payload is too short')
            offset = 0
            (n_pairs, n_dim) = struct.unpack_from('<II', raw, offset)
            offset += 8
            bits_per_dim = list(raw[offset:offset + n_dim])
            if len(bits_per_dim) != n_dim:
                raise PR97ProfileError('truncated pose bits_per_dim')
            offset += n_dim
            ranges = []
            for dim, bits in enumerate(bits_per_dim):
                if offset + 8 > len(raw):
                    raise PR97ProfileError(f'''truncated pose lo/scale for dim {dim}''')
                (lo, scale) = struct.unpack_from('<ff', raw, offset)
                offset += 8
                ranges.append({
                    'dim': dim,
                    'bits': int(bits),
                    'lo': lo,
                    'scale': scale,
                    'covered_range': scale * ((1 << int(bits)) - 1) })
            bitstream_bytes = len(raw) - offset
            needed_bytes = (sum(bits_per_dim) * n_pairs + 7) // 8
            if bitstream_bytes < needed_bytes:
                raise PR97ProfileError(f'''pose bitstream too short: {bitstream_bytes} < {needed_bytes}''')
            return {
                'format': 'pr97_per_dim_packed_pose_brotli',
                'compressed_bytes': len(blob),
                'compressed_sha256': sha256_bytes(blob),
                'raw_bytes': len(raw),
                'raw_sha256': sha256_bytes(raw),
                'n_pairs': n_pairs,
                'n_dim': n_dim,
                'bits_per_dim': bits_per_dim,
                'header_bytes': offset,
                'bitstream_bytes': bitstream_bytes,
                'needed_bitstream_bytes': needed_bytes,
                'ranges': ranges,
                'brotli_ratio': len(blob) / len(raw) }
        except brotli.error:
            exc = None
            raise PR97ProfileError(f'''pose brotli decode failed: {exc}'''), exc
            exc = None
            del exc


    
    def load_schema(schema_py = None):
        tree = ast.parse(schema_py.read_text(encoding = 'utf-8'))
    # WARNING: Decompyle incomplete

    
    def parse_model(blob = None, schema_py = None):
        
        try:
            raw = brotli.decompress(blob)
            schema = load_schema(schema_py)
            offset = 0
            kind_counts = Counter()
            fp4_params = 0
            fp16_params = 0
            fp4_nibble_bytes = 0
            fp4_scale_bytes = 0
            fp16_bytes = 0
            rows = []
            for name, kind, shape in schema:
                math.prod(shape) = schema
                start = offset
                if kind == 'fp4_w':
                    n_blocks = (n + 31) // 32
                    packed_bytes = (n_blocks * 32 + 1) // 2
                    scale_bytes = n_blocks * 2
                    offset += packed_bytes + scale_bytes
                    fp4_params += n
                    fp4_nibble_bytes += packed_bytes
                    fp4_scale_bytes += scale_bytes
                elif kind in frozenset({'fp16_b', 'fp16_w'}):
                    byte_count = n * 2
                    offset += byte_count
                    fp16_params += n
                    fp16_bytes += byte_count
                else:
                    raise PR97ProfileError(f'''unknown model schema kind {kind!r}''')
                rows.append({
                    'name': name,
                    'kind': kind,
                    'shape': list(shape),
                    'params': n,
                    'raw_bytes': offset - start,
                    'offset': start })
            if offset != len(raw):
                raise PR97ProfileError(f'''model schema consumed {offset}, raw has {len(raw)}''')
            return {
                'format': 'pr97_h3_flat_fp4_model_brotli',
                'compressed_bytes': len(blob),
                'compressed_sha256': sha256_bytes(blob),
                'raw_bytes': len(raw),
                'raw_sha256': sha256_bytes(raw),
                'schema_entries': len(schema),
                'schema_sha256': sha256_file(schema_py),
                'kind_counts': dict(kind_counts),
                'fp4_params': fp4_params,
                'fp16_params': fp16_params,
                'fp4_nibble_bytes': fp4_nibble_bytes,
                'fp4_scale_bytes': fp4_scale_bytes,
                'fp16_bytes': fp16_bytes,
                'brotli_ratio': len(blob) / len(raw),
                'largest_raw_entries': sorted(rows, key = (lambda row: int(row['raw_bytes'])), reverse = True)[:10] }
        except brotli.error:
            exc = None
            raise PR97ProfileError(f'''model brotli decode failed: {exc}'''), exc
            exc = None
            del exc


    
    def parse_sidecar(blob = None):
        pass
    # WARNING: Decompyle incomplete

    
    def _consume_patch_list(raw = None, pos = None, label = None):
        if pos >= len(raw):
            raise PR97ProfileError(f'''truncated sidecar {label} count''')
        n = raw[pos]
        pos += 1
        end = pos + 3 * n
        if end > len(raw):
            raise PR97ProfileError(f'''truncated sidecar {label} patches''')
        return (n, end)

    
    def deterministic_zip_bytes(payload = None, *, compression):
        handle = io.BytesIO()
        compresslevel = 9 if compression == zipfile.ZIP_DEFLATED else None
        zf = zipfile.ZipFile(handle, 'w', compression = compression, compresslevel = compresslevel)
        info = zipfile.ZipInfo('p', (1980, 1, 1, 0, 0, 0))
        info.compress_type = compression
        info.external_attr = 27525120
        zf.writestr(info, payload)
        None(None, None)
        return handle.getvalue()
        with None:
            if not None:
                pass
        return handle.getvalue()

    
    def write_candidate(path = None, payload = None, *, compression):
        raw = deterministic_zip_bytes(payload, compression = compression)
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_bytes(raw)
        zf = zipfile.ZipFile(path, 'r')
        info = zf.getinfo('p')
        member_sha = sha256_bytes(zf.read('p'))
        None(None, None)
    # WARNING: Decompyle incomplete

    
    def build_payload(parts = None):
        return (lambda .0: pass# WARNING: Decompyle incomplete
)(parts())

    
    def build_candidates(*, output_dir, source_archive_bytes, source_payload, parts, pose_raw_sha256, model_raw_sha256, sidecar_raw_sha256):
        pass
    # WARNING: Decompyle incomplete

    
    def runtime_static_report(runtime_dir = None):
        files = []
        imports = Counter()
        for path in sorted(runtime_dir.iterdir()):
            if not path.is_file():
                continue
            row = {
                'name': path.name,
                'bytes': path.stat().st_size,
                'sha256': sha256_file(path) }
            if path.suffix == '.py':
                tree = ast.parse(path.read_text(encoding = 'utf-8'))
                mods = sorted(_top_level_imports(tree))
                row['imports'] = mods
                imports.update(mods)
            files.append(row)
        return {
            'runtime_dir': len(files),
            'files': None,
            'runtime_file_count': sha256_bytes,
            'runtime_tree_sha256': b''.join((lambda .0: pass# WARNING: Decompyle incomplete
)(sorted(runtime_dir.iterdir())())),
            'top_level_imports': sorted(imports),
            'static_risks': [
                'inflate.sh attempts pip install brotli if missing; exact replay should pin dependency closure',
                'inflate.py compiles range_mask_codec.cpp at inflate time and requires c++/g++/clang++'] }
        except SyntaxError:
            exc = runtime_dir.as_posix()
            row['parse_error'] = str(exc)
            exc = None
            del exc
            continue
            exc = None
            del exc

    
    def _top_level_imports(tree = None):
        pass
    # WARNING: Decompyle incomplete

    
    def build_profile(archive = None, runtime_dir = None, output_dir = None):
        pass
    # WARNING: Decompyle incomplete

    
    def render_markdown(profile = None):
        archive = profile['archive']
        payload = profile['payload']
        lines = [
            '# PR97 H3 Static Intake',
            '',
            f'''- Evidence grade: `{profile['evidence_grade']}`''',
            f'''- Archive: `{archive['path']}`''',
            f'''- Archive bytes/SHA: `{archive['bytes']}` / `{archive['sha256']}`''',
            f'''- ZIP overhead: `{archive['zip_overhead_bytes']}` bytes''',
            f'''- Payload `p`: `{payload['bytes']}` bytes / `{payload['sha256']}`''',
            '',
            '## Payload Split',
            '',
            '| part | bytes | sha256 |',
            '|---|---:|---|']
        for name in PAYLOAD_PARTS:
            row = payload['parts'][name]
            lines.append(f'''| {name} | {row['bytes']} | `{row['sha256']}` |''')
        lines.extend([
            '',
            '## Parsed Subformats',
            '',
            f'''- Mask: `{profile['mask']['chunk_count']}` chunks, `{profile['mask']['bytes']}` bytes.''',
            f'''- Pose: bits per dim `{profile['pose']['bits_per_dim']}`, raw `{profile['pose']['raw_bytes']}` bytes.''',
            f'''- Model: `{profile['model']['schema_entries']}` schema entries, raw `{profile['model']['raw_bytes']}` bytes.''',
            f'''- Sidecar: `{profile['sidecar']['pair_record_count']}` pair records, raw `{profile['sidecar']['raw_bytes']}` bytes.''',
            '',
            '## Byte Candidates',
            '',
            '| label | archive bytes | delta | runtime change |',
            '|---|---:|---:|---|'])
        for row in profile['byte_opportunities']['safe_repack_candidates']:
            bytes_value = row.get('archive_bytes', row.get('archive_bytes_estimate'))
            delta = row.get('archive_byte_delta', row.get('archive_byte_delta_estimate'))
            lines.append(f'''| {row['label']} | {bytes_value} | {delta} | {row.get('requires_runtime_change')} |''')
        lines.extend([
            '',
            '## Risks',
            ''])
        for risk in profile['runtime_static']['static_risks']:
            lines.append(f'''- {risk}''')
        lines.append('')
        return '\n'.join(lines)

    
    def run(args = None):
        archive = Path(args.archive)
        runtime_dir = Path(args.runtime_dir)
        output_dir = Path(args.output_dir) if args.output_dir else None
        profile = build_profile(archive, runtime_dir, output_dir)
    # WARNING: Decompyle incomplete

    
    def parse_args(argv = None):
        parser = argparse.ArgumentParser(description = __doc__)
        parser.add_argument('--archive', default = DEFAULT_ARCHIVE)
        parser.add_argument('--runtime-dir', default = DEFAULT_RUNTIME)
        parser.add_argument('--output-dir', default = DEFAULT_OUTPUT_DIR)
        return parser.parse_args(argv)

    
    def main(argv = None):
        return run(parse_args(argv))

    if __name__ == '__main__':
        raise SystemExit(main())
    return None
except ImportError:
    exc = None
    raise SystemExit('brotli is required for PR97 H3 intake profiling'), exc
    exc = None
    del exc


"""
