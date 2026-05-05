# pyc-recovery: STUB unreconstructible -- see .recovery_spec.json for dis() ground-truth
# pycdc could not produce parseable output; raw decompiled text preserved in _PYCDC_PARTIAL_OUTPUT below.
"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``60:71: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``build_contest_submission_packet.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'scripts/build_contest_submission_packet.py'
__recovery_spec__ = 'build_contest_submission_packet.recovery_spec.json'
__recovery_ast_error__ = '60:71: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: build_contest_submission_packet.cpython-312.pyc (Python 3.12)

'''Build a deterministic contest-faithful submission packet manifest.

By default the packet is metadata-only: it records custody facts for an exact
eval artifact directory and writes a JSON manifest plus markdown checklist. When
``--runtime-dir`` is provided it also builds the concrete submission directory
from the auth-eval runtime manifest plus ``archive.zip`` and ``report.txt``.
Optional planner, visualization, and next-action files are recorded as non-score
supporting artifacts.
'''
from __future__ import annotations
import argparse
import hashlib
import json
import math
import shutil
import struct
import zipfile
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any
REPO_ROOT = Path(__file__).resolve().parent.parent
SCORE_DENOMINATOR = 37545489
DEFAULT_MANIFEST_NAME = 'submission_packet_manifest.json'
DEFAULT_CHECKLIST_NAME = 'submission_packet_checklist.md'
DEFAULT_SUBMISSION_DIR_NAME = 'submission'
KNOWN_OPTIONAL_ARTIFACTS = ('component_trace.json', 'report.txt', 'eval_provenance.json', 'auth_eval.log', 'contest_auth_eval.adjudicated.json', 'adjudication_provenance.json')
CDO1_ARCHIVE_MEMBERS = ('masks.cdo1', 'masks.cdo1.zlib', 'masks.cdo1.xz', 'masks.cdo1.br')
AMR1_ARCHIVE_MEMBERS = ('alpha4_residual_repair.amr1', 'alpha4_residual_repair.amr1.xz', 'alpha4_residual_repair.amr1.zlib', 'alpha4_residual_repair.amr1.br')
PACKED_PAYLOAD_MEMBER_NAMES = ('renderer_payload.bin', 'renderer_payload.bin.br', 'p')
RPK1_MAGIC = b'RPK1'
RPK1_HEADER_STRUCT = struct.Struct('<I')
ZIP_LOCAL_HEADER_STRUCT = struct.Struct('<4sHHHHHIIIHH')
SUPPORTING_ARTIFACT_SECTIONS = {
    'planner_ledgers': 'planning_or_proxy_only',
    'visualizations': 'visual_audit_only',
    'next_action_tranches': 'roadmap_only' }

class PacketError(RuntimeError):
    '''Raised when the source artifact directory is not packet-ready.'''
    pass


def _sha256(path = None):
    pass
# WARNING: Decompyle incomplete


def _read_json(path = None):
    
    try:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise PacketError(f'''{path.name} must contain a JSON object''')
        return payload
    except json.JSONDecodeError:
        exc = None
        raise PacketError(f'''{path.name} is not valid JSON: {exc}'''), exc
        exc = None
        del exc



def _rel(path = None, base = None):
    
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return 



def _file_record(path = None, repo_root = None, artifact_dir = None):
    return {
        'path': _rel(path, repo_root),
        'artifact_relative_path': _rel(path, artifact_dir),
        'size_bytes': path.stat().st_size,
        'sha256': _sha256(path) }


def _copied_file_record(path = None, repo_root = None, submission_dir = None):
    return {
        'path': _rel(path, repo_root),
        'submission_relative_path': _rel(path, submission_dir),
        'size_bytes': path.stat().st_size,
        'sha256': _sha256(path),
        'mode_octal': oct(path.stat().st_mode & 511) }


def _artifact_file(artifact_dir = None, artifact_relative_path = None):
    path = (artifact_dir / artifact_relative_path).resolve()
    
    try:
        path.relative_to(artifact_dir)
        return path
    except ValueError:
        exc = None
        raise PacketError(f'''artifact file must stay inside artifact directory: {artifact_relative_path}'''), exc
        exc = None
        del exc



def _safe_runtime_relative_path(value = None):
    if not isinstance(value, str) or value:
        raise PacketError(f'''runtime manifest relative_path must be a nonempty string: {value!r}''')
    if '\\' in value and '\x00' in value or (lambda .0: pass# WARNING: Decompyle incomplete
)(value()):
        raise PacketError(f'''unsafe runtime manifest path: {value!r}''')
    rel = PurePosixPath(value)
    if rel.is_absolute() or '..' in rel.parts:
        raise PacketError(f'''runtime manifest path must stay inside runtime directory: {value!r}''')
    if rel.name in frozenset({'.DS_Store', 'Thumbs.db'}) or (lambda .0: pass# WARNING: Decompyle incomplete
)(rel.parts()):
        raise PacketError(f'''runtime manifest path is a resource-fork sidecar: {value!r}''')
    return rel.as_posix()


def _unsafe_archive_member_name(name = None):
    if not name:
        return 'empty_member_name'
    if '\\' in name and '\x00' in name or (lambda .0: pass# WARNING: Decompyle incomplete
)(name()):
        return 'unsafe_member_name'
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts:
        return 'zip_slip_member_name'
    if '__MACOSX' in rel.parts:
        return 'macosx_resource_directory'
    if rel.name in frozenset({'.DS_Store', 'Thumbs.db'}) or (lambda .0: pass# WARNING: Decompyle incomplete
)(rel.parts()):
        return 'resource_fork_or_hidden_sidecar'
    if (lambda .0: pass# WARNING: Decompyle incomplete
)(rel.parts()):
        return 'hidden_sidecar_member_name'


def _decode_zip_name(raw = None, flag_bits = None):
    encoding = 'utf-8' if flag_bits & 2048 else 'cp437'
    
    try:
        return raw.decode(encoding, errors = 'strict')
    except UnicodeDecodeError:
        return None



def _local_header_name(path = None, info = None):
    handle = path.open('rb')
    handle.seek(info.header_offset)
    header = handle.read(ZIP_LOCAL_HEADER_STRUCT.size)
    if len(header) != ZIP_LOCAL_HEADER_STRUCT.size:
        None(None, None)
        return (None, None)
# WARNING: Decompyle incomplete


def _inspect_archive_integrity(archive_path = None, *, checks):
    record = {
        'members': [],
        'duplicate_members': [],
        'packed_payload_members': [],
        'bad_crc_member': None }
    zf = zipfile.ZipFile(archive_path)
    infos = zf.infolist()
# WARNING: Decompyle incomplete


def _runtime_manifest_from_payload(payload = None):

"""
