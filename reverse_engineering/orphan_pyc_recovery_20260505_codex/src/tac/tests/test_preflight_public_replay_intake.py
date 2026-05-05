# pyc-recovery: STUB unreconstructible -- see .recovery_spec.json for dis() ground-truth
# pycdc could not produce parseable output; raw decompiled text preserved in _PYCDC_PARTIAL_OUTPUT below.
"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``72:30: invalid decimal literal``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``test_preflight_public_replay_intake.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'src/tac/tests/test_preflight_public_replay_intake.py'
__recovery_spec__ = 'test_preflight_public_replay_intake.recovery_spec.json'
__recovery_ast_error__ = '72:30: invalid decimal literal'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: test_preflight_public_replay_intake.cpython-312-pytest-9.0.3.pyc (Python 3.12)

from __future__ import annotations
import builtins as @py_builtins

rewrite
import hashlib = import _pytest.assertion.rewrite, assertion
import importlib.util as importlib
import sys
import zipfile
from pathlib import Path
import pytest
from tac.pr85_bundle import SEGMENT_ORDER, pack_pr85_bundle
brotli = pytest.importorskip('brotli')
REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / 'experiments' / 'preflight_public_replay_intake.py'

def _load_script():
    spec = importlib.util.spec_from_file_location('preflight_public_replay_intake_test', SCRIPT)
    @py_assert1 = []
    @py_assert4 = None
    @py_assert3 = spec is not @py_assert4
    @py_assert0 = @py_assert3
    if @py_assert3:
        @py_assert10 = spec.loader
        @py_assert13 = None
        @py_assert12 = @py_assert10 is not @py_assert13
        @py_assert0 = @py_assert12
# WARNING: Decompyle incomplete

module = _load_script()

def _det_bytes(label = None, n = None):
    out = bytearray()
    counter = 0
    if len(out) < n:
        out.extend(hashlib.sha256(f'''{label}:{counter}'''.encode('ascii')).digest())
        counter += 1
        if len(out) < n:
            continue
    return bytes(out[:n])


def _br(data = None):
    return brotli.compress(data, quality = 5)


def _write_runtime(root = None, *, embedded_payload):
    root.mkdir(parents = True, exist_ok = True)
    inflate_sh = root / 'inflate.sh'
    inflate_sh.write_text('#!/usr/bin/env bash\nset -euo pipefail\npython "$(dirname "$0")/inflate.py" "$@"\n', encoding = 'utf-8')
    if embedded_payload:
        payload = 'A' * 70000
        inflate_py = f'''import base64\nPAYLOAD = base64.b85decode({payload!r})\nprint(len(PAYLOAD))\n'''
    else:
        inflate_py = 'from pathlib import Path\nprint(Path(__file__).name)\n'
    (root / 'inflate.py').write_text(inflate_py, encoding = 'utf-8')
    return inflate_sh


def _write_upstream(root = None):
    upstream = root / 'upstream'
    upstream.mkdir(parents = True)
    (upstream / 'evaluate.py').write_text("print('fixture evaluate')\n", encoding = 'utf-8')
    return upstream


def _write_x_archive(path = None):
    mask_bitstream = _det_bytes('qma9-bitstream', 64)
    segments = {
        'mask': b'QMA9' + 600.to_bytes(4, 'little') + 512.to_bytes(4, 'little') + 384.to_bytes(4, 'little') + len(mask_bitstream).to_bytes(4, 'little') + mask_bitstream,
        'model': _br(b'QH0' + _det_bytes('model', 96)),
        'pose': _br(b'P1D1' + _det_bytes('pose', 64)),
        'post': _br(_det_bytes('post', 128)),
        'shift': _br(b'SD4' + _det_bytes('shift', 32)),
        'frac': _br(b'FV1' + _det_bytes('frac', 32)),
        'frac2': _br(b'FH2' + _det_bytes('frac2', 32)),
        'frac3': _br(b'FD3' + _det_bytes('frac3', 32)),
        'bias': _br(b'BD1' + _det_bytes('bias', 32)),
        'region': _br(b'RH1' + _det_bytes('region', 32)),
        'randmulti': _br(_det_bytes('randmulti', 32)) }
    raw = pack_pr85_bundle(segments, header_mode = 'explicit_30')
    info = zipfile.ZipInfo('x', (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 27525120
    path.parent.mkdir(parents = True, exist_ok = True)
    zf = zipfile.ZipFile(path, 'w')
    zf.writestr(info, raw)
    None(None, None)
    return None
    with None:
        if not None:
            pass


def _blocker_codes(payload = None):
    pass
# WARNING: Decompyle incomplete


def test_public_replay_preflight_accepts_byte_closed_x_archive_and_runtime(tmp_path = None):
    archive = tmp_path / 'archive.zip'
    inflate_sh = _write_runtime(tmp_path / 'replay_submission')
    upstream = _write_upstream(tmp_path)
    _write_x_archive(archive)
    payload = module.build_preflight(archive, inflate_sh, upstream_dir = upstream)
    @py_assert0 = payload['ready_for_exact_eval_dispatch']
    @py_assert3 = True
    @py_assert2 = @py_assert0 is @py_assert3
    if not @py_assert2:
        @py_format5 = @pytest_ar._call_reprcompare(('is',), (@py_assert2,), ('%(py1)s is %(py4)s',), (@py_assert0, @py_assert3)) % {
            'py1': @pytest_ar._saferepr(@py_assert0),
            'py4': @pytest_ar._saferepr(@py_assert3) }
        @py_format7 = 'assert %(py6)s' % {
            'py6': @py_format5 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format7))
    @py_assert0 = None
    @py_assert2 = None
    @py_assert3 = None
    @py_assert0 = payload['evidence_grade']
    @py_assert4 = module.EVIDENCE_GRADE
    @py_assert2 = @py_assert0 == @py_assert4
    if not @py_assert2:
        @py_format6 = @pytest_ar._call_reprcompare(('==',), (@py_assert2,), ('%(py1)s == %(py5)s\n{%(py5)s = %(py3)s.EVIDENCE_GRADE\n}',), (@py_assert0, @py_assert4)) % {
            'py1': @pytest_ar._saferepr(@py_assert0),
            'py3': @pytest_ar._saferepr(module) if 'module' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(module) else 'module',
            'py5': @pytest_ar._saferepr(@py_assert4) }
        @py_format8 = 'assert %(py7)s' % {
            'py7': @py_format6 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format8))
    @py_assert0 = None
    @py_assert2 = None
    @py_assert4 = None
    @py_assert0 = payload['score_claim']
    @py_assert3 = False
    @py_assert2 = @py_assert0 is @py_assert3
    if not @py_assert2:
        @py_format5 = @pytest_ar._call_reprcompare(('is',), (@py_assert2,), ('%(py1)s is %(py4)s',), (@py_assert0, @py_assert3)) % {
            'py1': @pytest_ar._saferepr(@py_assert0),
            'py4': @pytest_ar._saferepr(@py_assert3) }
        @py_format7 = 'assert %(py6)s' % {
            'py6': @py_format5 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format7))
    @py_assert0 = None
    @py_assert2 = None
    @py_assert3 = None
    @py_assert0 = payload['promotion_eligible']
    @py_assert3 = False
    @py_assert2 = @py_assert0 is @py_assert3
    if not @py_assert2:
        @py_format5 = @pytest_ar._call_reprcompare(('is',), (@py_assert2,), ('%(py1)s is %(py4)s',), (@py_assert0, @py_assert3)) % {
            'py1': @pytest_ar._saferepr(@py_assert0),
            'py4': @pytest_ar._saferepr(@py_assert3) }
        @py_format7 = 'assert %(py6)s' % {
            'py6': @py_format5 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format7))
    @py_assert0 = None
    @py_assert2 = None
    @py_assert3 = None
    @py_assert0 = payload['dispatch_performed']
    @py_assert3 = False
    @py_assert2 = @py_assert0 is @py_assert3
    if not @py_assert2:
        @py_format5 = @pytest_ar._call_reprcompare(('is',), (@py_assert2,), ('%(py1)s is %(py4)s',), (@py_assert0, @py_assert3)) % {
            'py1': @pytest_ar._saferepr(@py_assert0),
            'py4': @pytest_ar._saferepr(@py_assert3) }
        @py_format7 = 'assert %(py6)s' % {
            'py6': @py_format5 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format7))
    @py_assert0 = None
    @py_assert2 = None
    @py_assert3 = None
    @py_assert0 = payload['archive']['charged_member_allowlist']
    @py_assert3 = 'passed'
    @py_assert2 = @py_assert0 == @py_assert3
    if not @py_assert2:
        @py_format5 = @pytest_ar._call_reprcompare(('==',), (@py_assert2,), ('%(py1)s == %(py4)s',), (@py_assert0, @py_assert3)) % {
            'py1': @pytest_ar._saferepr(@py_assert0),
            'py4': @pytest_ar._saferepr(@py_assert3) }
        @py_format7 = 'assert %(py6)s' % {
            'py6': @py_format5 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format7))
    @py_assert0 = None
    @py_assert2 = None
    @py_assert3 = None
    smoke = payload['archive']['members'][0]['decode_smoke']['format']
    @py_assert0 = smoke['format']
    @py_assert3 = 'pr85_explicit_30byte_lengths'
    @py_assert2 = @py_assert0 == @py_assert3
    if not @py_assert2:
        @py_format5 = @pytest_ar._call_reprcompare(('==',), (@py_assert2,), ('%(py1)s == %(py4)s',), (@py_assert0, @py_assert3)) % {
            'py1': @pytest_ar._saferepr(@py_assert0),
            'py4': @pytest_ar._saferepr(@py_assert3) }
        @py_format7 = 'assert %(py6)s' % {
            'py6': @py_format5 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format7))
    @py_assert0 = None
    @py_assert2 = None
    @py_assert3 = None
# WARNING: Decompyle incomplete


def test_public_replay_preflight_blocks_duplicate_and_sidecar_members(tmp_path = None):
    archive = tmp_path / 'archive.zip'
    inflate_sh = _write_runtime(tmp_path / 'replay_submission')
    upstream = _write_upstream(tmp_path)
    zf = zipfile.ZipFile(archive, 'w')
    zf.writestr('x', b'not-a-pr85-bundle')
    zf.writestr('x', b'duplicate')
    zf.writestr('notes.debug', b'sidecar')
    None(None, None)
    payload = module.build_preflight(archive, inflate_sh, upstream_dir = upstream)
    @py_assert0 = payload['ready_for_exact_eval_dispatch']
    @py_assert3 = False
    @py_assert2 = @py_assert0 is @py_assert3
    if not @py_assert2:
        @py_format5 = @pytest_ar._call_reprcompare(('is',), (@py_assert2,), ('%(py1)s is %(py4)s',), (@py_assert0, @py_assert3)) % {
            'py1': @pytest_ar._saferepr(@py_assert0),
            'py4': @pytest_ar._saferepr(@py_assert3) }
        @py_format7 = 'assert %(py6)s' % {
            'py6': @py_format5 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format7))
    @py_assert0 = None
    @py_assert2 = None
    @py_assert3 = None
    @py_assert0 = {
        'member_decode_smoke',
        'duplicate_member_names',
        'charged_member_allowlist'}
    @py_assert5 = _blocker_codes(payload)
    @py_assert2 = @py_assert0 <= @py_assert5
    if not @py_assert2:
        @py_format7 = @pytest_ar._call_reprcompare(('<=',), (@py_assert2,), ('%(py1)s <= %(py6)s\n{%(py6)s = %(py3)s(%(py4)s)\n}',), (@py_assert0, @py_assert5)) % {
            'py1': @pytest_ar._saferepr(@py_assert0),
            'py3': @pytest_ar._saferepr(_blocker_codes) if '_blocker_codes' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(_blocker_codes) else '_blocker_codes',
            'py4': @pytest_ar._saferepr(payload) if 'payload' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(payload) else 'payload',
            'py6': @pytest_ar._saferepr(@py_assert5) }
        @py_format9 = 'assert %(py8)s' % {
            'py8': @py_format7 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format9))
    @py_assert0 = None
    @py_assert2 = None
    @py_assert5 = None
    return None
    with None:
        if not None:
            pass
    continue


def test_public_replay_preflight_blocks_zip_central_local_name_mismatch(tmp_path = None):
    archive = tmp_path / 'archive.zip'
    inflate_sh = _write_runtime(tmp_path / 'replay_submission')
    upstream = _write_upstream(tmp_path)
    _write_x_archive(archive)
    raw = bytearray(archive.read_bytes())
    zf = zipfile.ZipFile(archive, 'r')
    offset = zf.getinfo('x').header_offset
    None(None, None)
# WARNING: Decompyle incomplete


def test_public_replay_preflight_blocks_source_embedded_payload_runtime(tmp_path = None):
    archive = tmp_path / 'archive.zip'
    inflate_sh = _write_runtime(tmp_path / 'replay_submission', embedded_payload = True)
    upstream = _write_upstream(tmp_path)
    _write_x_archive(archive)
    payload = module.build_preflight(archive, inflate_sh, upstream_dir = upstream)
    @py_assert0 = payload['ready_for_exact_eval_dispatch']
    @py_assert3 = False
    @py_assert2 = @py_assert0 is @py_assert3
    if not @py_assert2:
        @py_format5 = @pytest_ar._call_reprcompare(('is',), (@py_assert2,), ('%(py1)s is %(py4)s',), (@py_assert0, @py_assert3)) % {
            'py1': @pytest_ar._saferepr(@py_assert0),
            'py4': @pytest_ar._saferepr(@py_assert3) }
        @py_format7 = 'assert %(py6)s' % {
            'py6': @py_format5 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format7))
    @py_assert0 = None
    @py_assert2 = None
    @py_assert3 = None
    @py_assert0 = 'runtime_source_or_sidecar_payload'
    @py_assert5 = _blocker_codes(payload)
    @py_assert2 = @py_assert0 in @py_assert5
    if not @py_assert2:
        @py_format7 = @pytest_ar._call_reprcompare(('in',), (@py_assert2,), ('%(py1)s in %(py6)s\n{%(py6)s = %(py3)s(%(py4)s)\n}',), (@py_assert0, @py_assert5)) % {
            'py1': @pytest_ar._saferepr(@py_assert0),
            'py3': @pytest_ar._saferepr(_blocker_codes) if '_blocker_codes' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(_blocker_codes) else '_blocker_codes',
            'py4': @pytest_ar._saferepr(payload) if 'payload' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(payload) else 'payload',
            'py6': @pytest_ar._saferepr(@py_assert5) }
        @py_format9 = 'assert %(py8)s' % {
            'py8': @py_format7 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format9))
    @py_assert0 = None
    @py_assert2 = None
    @py_assert5 = None
    @py_assert0 = payload['runtime']['source_payload_scan']['status']
    @py_assert3 = 'failed'
    @py_assert2 = @py_assert0 == @py_assert3
    if not @py_assert2:
        @py_format5 = @pytest_ar._call_reprcompare(('==',), (@py_assert2,), ('%(py1)s == %(py4)s',), (@py_assert0, @py_assert3)) % {
            'py1': @pytest_ar._saferepr(@py_assert0),
            'py4': @pytest_ar._saferepr(@py_assert3) }
        @py_format7 = 'assert %(py6)s' % {
            'py6': @py_format5 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format7))
    @py_assert0 = None
    @py_assert2 = None
    @py_assert3 = None


def test_public_replay_preflight_expected_runtime_tree_mismatch_fails_closed(tmp_path = None):
    archive = tmp_path / 'archive.zip'
    inflate_sh = _write_runtime(tmp_path / 'replay_submission')
    upstream = _write_upstream(tmp_path)
    _write_x_archive(archive)
    payload = module.build_preflight(archive, inflate_sh, upstream_dir = upstream, expected_runtime_tree_sha256 = '0000000000000000000000000000000000000000000000000000000000000000')
    @py_assert0 = payload['ready_for_exact_eval_dispatch']
    @py_assert3 = False
    @py_assert2 = @py_assert0 is @py_assert3
    if not @py_assert2:
        @py_format5 = @pytest_ar._call_reprcompare(('is',), (@py_assert2,), ('%(py1)s is %(py4)s',), (@py_assert0, @py_assert3)) % {
            'py1': @pytest_ar._saferepr(@py_assert0),
            'py4': @pytest_ar._saferepr(@py_assert3) }
        @py_format7 = 'assert %(py6)s' % {
            'py6': @py_format5 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format7))
    @py_assert0 = None
    @py_assert2 = None
    @py_assert3 = None
    @py_assert0 = 'expected_runtime_tree_sha256_matches'
    @py_assert5 = _blocker_codes(payload)
    @py_assert2 = @py_assert0 in @py_assert5
    if not @py_assert2:
        @py_format7 = @pytest_ar._call_reprcompare(('in',), (@py_assert2,), ('%(py1)s in %(py6)s\n{%(py6)s = %(py3)s(%(py4)s)\n}',), (@py_assert0, @py_assert5)) % {
            'py1': @pytest_ar._saferepr(@py_assert0),
            'py3': @pytest_ar._saferepr(_blocker_codes) if '_blocker_codes' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(_blocker_codes) else '_blocker_codes',
            'py4': @pytest_ar._saferepr(payload) if 'payload' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(payload) else 'payload',
            'py6': @pytest_ar._saferepr(@py_assert5) }
        @py_format9 = 'assert %(py8)s' % {
            'py8': @py_format7 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format9))
    @py_assert0 = None
    @py_assert2 = None
    @py_assert5 = None


"""
