# pyc-recovery: STUB unreconstructible -- see .recovery_spec.json for dis() ground-truth
# pycdc could not produce parseable output; raw decompiled text preserved in _PYCDC_PARTIAL_OUTPUT below.
"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``5:20: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``test_profile_pr97_h3_intake.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'src/tac/tests/test_profile_pr97_h3_intake.py'
__recovery_spec__ = 'test_profile_pr97_h3_intake.recovery_spec.json'
__recovery_ast_error__ = '5:20: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: test_profile_pr97_h3_intake.cpython-312-pytest-9.0.3.pyc (Python 3.12)

from __future__ import annotations
import builtins as @py_builtins

rewrite
import importlib.util = import _pytest.assertion.rewrite, assertion
import io
import lzma
import struct
import sys
import zipfile
from pathlib import Path
import pytest
brotli = pytest.importorskip('brotli')
REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / 'experiments' / 'profile_pr97_h3_intake.py'

def load_module():
    spec = importlib.util.spec_from_file_location('profile_pr97_h3_intake', MODULE_PATH)
    @py_assert2 = None
    @py_assert1 = spec is not @py_assert2
    if not @py_assert1:
        @py_format4 = @pytest_ar._call_reprcompare(('is not',), (@py_assert1,), ('%(py0)s is not %(py3)s',), (spec, @py_assert2)) % {
            'py0': @pytest_ar._saferepr(spec) if 'spec' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(spec) else 'spec',
            'py3': @pytest_ar._saferepr(@py_assert2) }
        @py_format6 = 'assert %(py5)s' % {
            'py5': @py_format4 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format6))
    @py_assert1 = None
    @py_assert2 = None
    @py_assert1 = spec.loader
    @py_assert4 = None
    @py_assert3 = @py_assert1 is not @py_assert4
    if not @py_assert3:
        @py_format6 = @pytest_ar._call_reprcompare(('is not',), (@py_assert3,), ('%(py2)s\n{%(py2)s = %(py0)s.loader\n} is not %(py5)s',), (@py_assert1, @py_assert4)) % {
            'py0': @pytest_ar._saferepr(spec) if 'spec' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(spec) else 'spec',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py5': @pytest_ar._saferepr(@py_assert4) }
        @py_format8 = 'assert %(py7)s' % {
            'py7': @py_format6 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format8))
    @py_assert1 = None
    @py_assert3 = None
    @py_assert4 = None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_runtime(root = None):
    root.mkdir()
    (root / 'schema_h3.py').write_text("SCHEMA = [\n  ('conv.weight', 'fp4_w', (2, 1, 3, 3)),\n  ('conv.bias', 'fp16_b', (2,)),\n  ('linear.weight', 'fp16_w', (2, 2)),\n]\n", encoding = 'utf-8')
    (root / 'inflate.py').write_text('import brotli, lzma\nfrom pathlib import Path\n', encoding = 'utf-8')
    (root / 'inflate.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n', encoding = 'utf-8')
    (root / 'sidecar.py').write_text('import lzma\n', encoding = 'utf-8')
    return root


def _model_raw():
    fp4_blocks = 1
    conv = b'\x12\x12\x12\x12\x12\x12\x12\x12\x12\x12\x12\x12\x12\x12\x12\x12' + b'\x00<' * fp4_blocks
    bias = b'\x00\x00\x00\x00'
    linear = b'\x00\x00\x00\x00\x00\x00\x00\x00'
    return conv + bias + linear


def _pose_blob():
    raw = io.BytesIO()
    raw.write(struct.pack('<II', 3, 2))
    raw.write(bytes([
        3,
        2]))
    raw.write(struct.pack('<ff', 1, 0.25))
    raw.write(struct.pack('<ff', -1, 0.5))
    raw.write(b'\xaa\xbb')
    return brotli.compress(raw.getvalue(), quality = 5)


def _sidecar_blob():
    raw = io.BytesIO()
    raw.write(b'BPGD')
    raw.write(struct.pack('<H', 2))
    raw.write(struct.pack('<H', 1))
    raw.write(bytes([
        1]))
    raw.write(bytes([
        1]))
    raw.write(b'\x01\x02\x03')
    raw.write(bytes([
        2]))
    raw.write(bytes([
        16]))
    raw.write(struct.pack('<bb', 2, -1))
    return lzma.compress(raw.getvalue(), format = lzma.FORMAT_XZ, preset = 6)


def _payload():
    mask = struct.pack('<I', 2) + struct.pack('<I', 64) + b'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA' + struct.pack('<I', 32) + b'BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB'
    parts = [
        mask,
        _pose_blob(),
        brotli.compress(_model_raw(), quality = 5),
        _sidecar_blob()]
    return (lambda .0: pass# WARNING: Decompyle incomplete
)(parts())


def _write_archive(path = None, payload = None):
    info = zipfile.ZipInfo('p', (2026, 5, 3, 23, 13, 44))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 27525120
    zf = zipfile.ZipFile(path, 'w')
    zf.writestr(info, payload)
    None(None, None)
    return None
    with None:
        if not None:
            pass


def test_profile_pr97_h3_parses_subformats_and_candidates(tmp_path = None):
    module = load_module()
    runtime = _write_runtime(tmp_path / 'runtime')
    archive = tmp_path / 'archive.zip'
    payload = _payload()
    _write_archive(archive, payload)
    profile = module.build_profile(archive, runtime, tmp_path / 'out')
    @py_assert0 = profile['schema']
    @py_assert3 = 'pr97_h3_static_intake_profile_v1'
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
    @py_assert0 = profile['score_claim']
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
    @py_assert0 = profile['dispatch_performed']
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
    @py_assert0 = profile['archive']['members'][0]['name']
    @py_assert3 = 'p'
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
    @py_assert0 = profile['payload']['parts']['mask']['bytes']
    @py_assert3 = 108
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
    @py_assert0 = profile['mask']['chunk_count']
    @py_assert3 = 2
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
    @py_assert0 = profile['pose']['bits_per_dim']
    @py_assert3 = [
        3,
        2]
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
    @py_assert0 = profile['pose']['needed_bitstream_bytes']
    @py_assert3 = 2
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
    @py_assert0 = profile['model']['schema_entries']
    @py_assert3 = 3
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
    @py_assert0 = profile['model']['fp4_params']
    @py_assert3 = 18
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
    @py_assert0 = profile['sidecar']['pair_record_count']
    @py_assert3 = 2
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
    @py_assert0 = profile['sidecar']['counts']['x2_pairs']
    @py_assert3 = 1
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
    @py_assert0 = profile['sidecar']['counts']['warp_pairs']
    @py_assert3 = 1
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


def test_profile_pr97_h3_rejects_bad_model_schema_length(tmp_path = None):
    module = load_module()
    runtime = _write_runtime(tmp_path / 'runtime')
    archive = tmp_path / 'archive.zip'
    (runtime / 'schema_h3.py').write_text("SCHEMA = [\n  ('conv.weight', 'fp4_w', (2, 1, 3, 3)),\n  ('conv.bias', 'fp16_b', (2,)),\n  ('linear.weight', 'fp16_w', (2, 2)),\n  ('unexpected.extra', 'fp16_w', (8,)),\n]\n", encoding = 'utf-8')
    _write_archive(archive, _payload())
    pytest.raises(module.PR97ProfileError, match = 'model schema consumed')
    module.build_profile(archive, runtime, None)
    None(None, None)
    return None
    with None:
        if not None:
            pass


"""
