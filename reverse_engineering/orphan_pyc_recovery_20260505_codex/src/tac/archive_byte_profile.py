"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``56:25: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``archive_byte_profile.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'src/tac/archive_byte_profile.py'
__recovery_spec__ = 'archive_byte_profile.recovery_spec.json'
__recovery_ast_error__ = '56:25: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: archive_byte_profile.cpython-312.pyc (Python 3.12)

'''Deterministic ZIP byte attribution for contest archive research.

The profiler is intentionally byte-only: it does not extract archive payloads,
inflate contest outputs, load scorer models, or make score claims.
'''
from __future__ import annotations
import argparse
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Sequence
SCHEMA = 'archive_byte_profile_collection_v1'
ARCHIVE_SCHEMA = 'archive_byte_profile_v1'
TOOL = 'experiments/profile_archive_bytes.py'
EVIDENCE_GRADE = 'byte_profile_only'
CONTEST_ORIGINAL_BYTES = 37545489
RATE_TERM_COEFFICIENT = 25 / CONTEST_ORIGINAL_BYTES

class ArchiveByteProfileError(ValueError):
    '''Raised when an archive cannot be safely profiled.'''
    pass


def contest_rate_term(byte_count = None):
    '''Return the contest formula rate contribution for ``byte_count``.'''
    return 25 * int(byte_count) / CONTEST_ORIGINAL_BYTES


def _json_bytes(payload = None):
    return (json.dumps(payload, indent = 2, sort_keys = True, allow_nan = False) + '\n').encode('utf-8')


def _sha256_file(path = None):
    pass
# WARNING: Decompyle incomplete


def _validate_zip_member_name(name = None):
    if not name:
        raise ArchiveByteProfileError('archive member name is empty')
    if '\x00' in name:
        raise ArchiveByteProfileError(f'''archive member contains NUL byte: {name!r}''')
    if '\\' in name:
        raise ArchiveByteProfileError(f'''archive member uses backslashes: {name!r}''')
    posix_path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    if posix_path.is_absolute() and windows_path.is_absolute() or windows_path.drive:
        raise ArchiveByteProfileError(f'''zip-slip archive member path: {name!r}''')
    parts = posix_path.parts
    if parts or (lambda .0: pass# WARNING: Decompyle incomplete
)(parts()):
        raise ArchiveByteProfileError(f'''zip-slip archive member path: {name!r}''')
    return name


def _zip_method_name(compress_type = None):
    names = {
        zipfile.ZIP_DEFLATED: 'deflated',
        zipfile.ZIP_STORED: 'stored' }
    if hasattr(zipfile, 'ZIP_BZIP2'):
        names[zipfile.ZIP_BZIP2] = 'bzip2'
    if hasattr(zipfile, 'ZIP_LZMA'):
        names[zipfile.ZIP_LZMA] = 'lzma'
    return names.get(compress_type, f'''unknown_{compress_type}''')


def _extension_group(name = None, is_dir = None):
    if is_dir:
        return '(directory)'
    suffixes = PurePosixPath(name).suffixes
    if not suffixes:
        return '(no_extension)'
    if len(suffixes) >= 2 and suffixes[-1].lower() in frozenset({'.br', '.gz', '.xz', '.bz2', '.zst'}):
        return (lambda .0: pass# WARNING: Decompyle incomplete
)(suffixes[-2:]())
    return None[-1].lower()


def _path_group(name = None):
    parts = PurePosixPath(name).parts
    if len(parts) <= 1:
        return '(root)'
    return None[0]


def _top_bytes(histogram = None, total = None, *, limit):
    rows = []
    for value, count in sorted(enumerate(histogram), key = (lambda item: (-item[1], item[0])))[:limit]:
        if count == 0:
            sorted(enumerate(histogram), key = (lambda item: (-item[1], item[0])))[:limit]
            return rows
        sorted(enumerate(histogram), key = (lambda item: (-item[1], item[0])))[:limit].append({
            'byte': value,
            'hex': f'''0x{value:02x}''',
            'count': int(count),
            'fraction': 0 if total == 0 else round(count / total, 12) })
    return rows


def _histogram_stats(histogram = None, total = None):
    pass
# WARNING: Decompyle incomplete


def _read_member_stats(zf = None, info = None):
    pass
# WARNING: Decompyle incomplete


def _duplicate_rows(counter = None):
    pass
# WARNING: Decompyle incomplete


def _totals_record(name = None, rows = None):
    compressed = (lambda .0: pass# WARNING: Decompyle incomplete
)(rows())
    uncompressed = (lambda .0: pass# WARNING: Decompyle incomplete
)(rows())
    return {
        'name': name,
        'member_count': len(rows),
        'compressed_size': compressed,
        'uncompressed_size': uncompressed,
        'rate_term': round(contest_rate_term(compressed), 12) }


def profile_archive(path = None):
    '''Profile one ZIP archive without extracting it.'''
    archive_path = Path(path)
    if not archive_path.is_file():
        raise FileNotFoundError(f'''archive not found: {archive_path}''')
    total_bytes = archive_path.stat().st_size
    archive_sha256 = _sha256_file(archive_path)
# WARNING: Decompyle incomplete


def invalid_archive_record(path = None, error = None):
    '''Return a structured byte-only record for an archive that failed profiling.'''
    archive_path = Path(path)
    total_bytes = archive_path.stat().st_size if archive_path.is_file() else None
    sha256 = _sha256_file(archive_path) if archive_path.is_file() else None
# WARNING: Decompyle incomplete


def build_profile_collection(paths = None, *, continue_on_error):
    archives = []
    invalid_archives = []
    for path in paths:
        archives.append(profile_archive(path))
    if not archives:
        raise ArchiveByteProfileError('at least one archive path is required')
    cross_hashes = defaultdict(list)
# WARNING: Decompyle incomplete


def _md_escape(value = None):
    return str(value).replace('|', '\\|')


def _md_optional_int(value = None):
    pass
# WARNING: Decompyle incomplete


def _md_optional_float(value = None, *, digits):
    pass
# WARNING: Decompyle incomplete


def render_markdown(profile = None):
    lines = [
        '# Archive Byte Profile',
        '',
        f'''- schema: `{profile['schema']}`''',
        f'''- evidence_grade: `{profile['evidence_grade']}`''',
        f'''- score_claim: `{profile['score_claim']}`''',
        f'''- rate formula: `{profile['rate_formula']}`''',
        f'''- archives: `{profile['archive_count']}`''',
        f'''- invalid archives: `{profile.get('invalid_archive_count', 0)}`''',
        '',
        'This is byte attribution only. It does not inflate payloads, run scorers, dispatch jobs, promote methods, or claim contest score.',
        '',
        '## Archives',
        '',
        '| archive | total bytes | rate term | members | ZIP overhead est. |',
        '|---|---:|---:|---:|---:|']
    for archive in profile['archives']:
        lines.append('| ' + ' | '.join([
            _md_escape(archive['archive_path']),
            _md_optional_int(archive.get('total_bytes')),
            _md_optional_float(archive.get('rate_term')),
            _md_optional_int(archive.get('member_count', 0)),
            _md_optional_int(archive.get('zip_overhead_estimate', { }).get('archive_non_payload_bytes'))]) + ' |')
# WARNING: Decompyle incomplete


def write_outputs(profile = None, *, json_out, markdown_out):
    pass
# WARNING: Decompyle incomplete


def _build_arg_parser():
    parser = argparse.ArgumentParser(description = 'Profile ZIP archive byte attribution without extracting payloads.')
    parser.add_argument('archives', nargs = '+', type = Path, help = 'archive.zip paths to profile')
    parser.add_argument('--json-out', type = Path, help = 'write deterministic JSON profile')
    parser.add_argument('--markdown-out', type = Path, help = 'write markdown summary')
    parser.add_argument('--continue-on-error', action = 'store_true', help = 'record invalid/nonstandard archives instead of aborting the whole collection')
    return parser


def main(argv = None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    profile = build_profile_collection(args.archives, continue_on_error = args.continue_on_error)
    write_outputs(profile, json_out = args.json_out, markdown_out = args.markdown_out)
# WARNING: Decompyle incomplete

if __name__ == '__main__':
    raise SystemExit(main())

"""
