"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``75:86: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``build_pr85_qh0_serializer_candidates.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'experiments/build_pr85_qh0_serializer_candidates.py'
__recovery_spec__ = 'build_pr85_qh0_serializer_candidates.recovery_spec.json'
__recovery_ast_error__ = '75:86: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: build_pr85_qh0_serializer_candidates.cpython-312.pyc (Python 3.12)

'''Build local-only PR85 QH0/QM0 serializer candidates.

The tool only rewrites the PR85 model segment when the decoded model payload is
still accepted by an existing runtime path. It never edits runtime files, runs
CUDA, dispatches jobs, or claims score evidence.
'''
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from tac.pr85_bundle import Pr85BundleError, pack_pr85_bundle, parse_pr85_bundle, validate_pr85_member_name
from tac.qh0_record_serializer import build_serialized_variants, choose_byte_win_candidates, prove_decoded_tensor_parity, record_set_summary, sha256_bytes
TOOL = 'experiments/build_pr85_qh0_serializer_candidates.py'
SCHEMA = 'pr85_qh0_serializer_candidates_v1'
MANIFEST_SCHEMA = 'pr85_qh0_serializer_candidate_v1'
DEFAULT_ARCHIVE = REPO_ROOT / 'experiments/results/public_pr85_intake_20260503_codex/archive.zip'
DEFAULT_OUT_DIR = REPO_ROOT / 'experiments/results/pr85_qh0_serializer_candidates_20260504_codex'
DEFAULT_REPLAY_INFLATE = REPO_ROOT / 'experiments/results/public_pr85_intake_20260503_codex/replay_submission/inflate.py'
DEFAULT_ROBUST_CURRENT = REPO_ROOT / 'submissions/robust_current'
ORIGINAL_VIDEO_BYTES = 37545489
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
KNOWN_PUBLIC_PR85 = {
    'archive_bytes': 236328,
    'archive_sha256': 'eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e' }

class QH0SerializerCandidateError(RuntimeError):
    '''Raised when a PR85 serializer candidate cannot be built safely.'''
    pass


def _sha256_file(path = None):
    pass
# WARNING: Decompyle incomplete


def _repo_rel(path = None):
    resolved = path.resolve()
    
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return 



def _json_bytes(payload = None):
    return (json.dumps(payload, indent = 2, sort_keys = True, allow_nan = False) + '\n').encode('utf-8')


def _read_single_x_archive(path = None):
    if not path.is_file():
        raise QH0SerializerCandidateError(f'''source archive not found: {_repo_rel(path)}''')
    zf = zipfile.ZipFile(path, 'r')
# WARNING: Decompyle incomplete


def _brotli_compress(data = None, *, quality, lgwin):
    
    try:
        import brotli
        return brotli.compress(data, quality = int(quality), lgwin = int(lgwin))
    except ImportError:
        exc = None
        raise QH0SerializerCandidateError('brotli is required for PR85 model recode'), exc
        exc = None
        del exc



def _compression_grid(data = None, *, qualities, lgwins):
    rows = []
    seen = set()
    for quality in qualities:
        for lgwin in lgwins:
            encoded = _brotli_compress(data, quality = quality, lgwin = lgwin)
            key = (sha256_bytes(encoded), len(encoded))
            rows.append({
                'codec': 'brotli',
                'quality': int(quality),
                'lgwin': int(lgwin),
                'bytes': len(encoded),
                'sha256': sha256_bytes(encoded),
                'payload': encoded,
                'duplicate_stream': key in seen })
            seen.add(key)
    return rows


def _source_header_mode(bundle_format = None):
    if bundle_format == 'pr85_explicit_30byte_lengths':
        return 'explicit_30'


def _zip_info_x():
    info = zipfile.ZipInfo('x', FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 27525120
    info.create_system = 3
    return info


def _write_single_x_archive(path = None, x_payload = None):
    path.parent.mkdir(parents = True, exist_ok = True)
    zf = zipfile.ZipFile(path, 'w', compression = zipfile.ZIP_STORED)
    zf.writestr(_zip_info_x(), x_payload, compress_type = zipfile.ZIP_STORED)
    None(None, None)
    zf = zipfile.ZipFile(path, 'r')
# WARNING: Decompyle incomplete


def _text(path = None):
    if path.is_file():
        return path.read_text(encoding = 'utf-8', errors = 'replace')


def runtime_compatibility(magic = None, *, replay_inflate_py, robust_current_dir):
    '''Static no-edit runtime support check for a serialized model magic.'''
    replay_text = _text(replay_inflate_py)
    robust_inflate_renderer = _text(robust_current_dir / 'inflate_renderer.py')
    robust_unpacker = _text(robust_current_dir / 'unpack_renderer_payload.py')
    magic_token = f'''b"{magic}"'''
    if 'get_decoded_state_dict_custom' in replay_text:
        'get_decoded_state_dict_custom' in replay_text
        if magic_token in replay_text:
            magic_token in replay_text
            if 'load_compact_archive_bundle' in replay_text:
                'load_compact_archive_bundle' in replay_text
    replay_model_loader = 'path = data_dir / "x"' in replay_text
    robust_renderer_member_loader = magic_token in robust_inflate_renderer
    if not 'path = data_dir / "x"' in robust_unpacker:
        'path = data_dir / "x"' in robust_unpacker
        if not 'ARCHIVE_DIR/x' in robust_unpacker:
            'ARCHIVE_DIR/x' in robust_unpacker
    robust_single_x_unpacker = '"/x"' in robust_unpacker
    blockers = []
    if not replay_model_loader:
        blockers.append(f'''public_pr85_replay_missing_{magic}_model_loader''')
    if not robust_renderer_member_loader:
        blockers.append(f'''robust_current_missing_{magic}_renderer_member_loader''')
    if not robust_single_x_unpacker:
        blockers.append('robust_current_missing_pr85_single_x_unpacker')
    runtime_can_decode_without_edits = replay_model_loader
    if runtime_can_decode_without_edits:
        return {
            'magic': magic,
            'runtime_can_decode_without_edits': runtime_can_decode_without_edits,
            'dispatch_unlocked': runtime_can_decode_without_edits,
            'public_pr85_replay_single_x_can_decode': replay_model_loader,
            'public_pr85_replay_inflate_py': _repo_rel(replay_inflate_py),
            'robust_current_renderer_member_can_decode': robust_renderer_member_loader,
            'robust_current_single_x_can_unpack': robust_single_x_unpacker,
            'robust_current_dir': _repo_rel(robust_current_dir),
            'blockers': blockers,
            'blocker_class': None if runtime_can_decode_without_edits else 'runtime_incompatibility',
            'minimal_runtime_implementation_needed': None }
    return {
        'magic': None,
        'runtime_can_decode_without_edits': magic,
        'dispatch_unlocked': runtime_can_decode_without_edits,
        'public_pr85_replay_single_x_can_decode': runtime_can_decode_without_edits,
        'public_pr85_replay_inflate_py': replay_model_loader,
        'robust_current_renderer_member_can_decode': _repo_rel(replay_inflate_py),
        'robust_current_single_x_can_unpack': robust_renderer_member_loader,
        'robust_current_dir': robust_single_x_unpacker,
        'blockers': _repo_rel(robust_current_dir),
        'blocker_class': blockers,
        'minimal_runtime_implementation_needed': None if runtime_can_decode_without_edits else 'runtime_incompatibility' }


def _best_screened_row(rows = None):
    if not rows:
        return None
    return min(rows, key = (lambda row: (int(row.get('candidate_model_delta_bytes_vs_source', 0)), str(row.get('candidate_id', '')))))


def build_candidates(archive = None, out_dir = None, *, qualities, lgwins, replay_inflate_py, robust_current_dir):
    (source_archive, x_raw) = _read_single_x_archive(archive)
    bundle = parse_pr85_bundle(x_raw)
# WARNING: Decompyle incomplete


def _strip_private(row = None):
    pass
# WARNING: Decompyle incomplete


def _candidate_summary(candidate = None):
    pass
# WARNING: Decompyle incomplete


def _parse_int_csv(text = None, *, label):
    values = []
    for part in text.split(','):
        stripped = part.strip()
        if not stripped:
            continue
        values.append(int(stripped))
    if not values:
        raise argparse.ArgumentTypeError(f'''{label} must not be empty''')
    return tuple(values)
    except ValueError:
        exc = None
        raise argparse.ArgumentTypeError(f'''{label} must be comma-separated ints'''), exc
        exc = None
        del exc


def parse_args(argv = None):
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--archive', type = Path, default = DEFAULT_ARCHIVE)
    parser.add_argument('--out-dir', type = Path, default = DEFAULT_OUT_DIR)
    parser.add_argument('--replay-inflate-py', type = Path, default = DEFAULT_REPLAY_INFLATE)
    parser.add_argument('--robust-current-dir', type = Path, default = DEFAULT_ROBUST_CURRENT)
    parser.add_argument('--qualities', default = '0,1,2,3,4,5,6,7,8,9,10,11')
    parser.add_argument('--lgwins', default = '18,20,22,24')
    return parser.parse_args(argv)


def main(argv = None):
    args = parse_args(argv)
    summary = build_candidates(args.archive, args.out_dir, qualities = _parse_int_csv(args.qualities, label = 'qualities'), lgwins = _parse_int_csv(args.lgwins, label = 'lgwins'), replay_inflate_py = args.replay_inflate_py, robust_current_dir = args.robust_current_dir)
    print(json.dumps({
        'summary_path': _repo_rel(args.out_dir / 'candidate_summary.json'),
        'built_candidate_count': summary['built_candidate_count'],
        'dispatch_unlocked': summary['dispatch_unlocked'],
        'blocker_class': summary['blocker_class'],
        'best_screened_candidate': summary['best_screened_candidate'],
        'best_built_candidate': summary['best_built_candidate'] }, indent = 2, sort_keys = True))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

"""
