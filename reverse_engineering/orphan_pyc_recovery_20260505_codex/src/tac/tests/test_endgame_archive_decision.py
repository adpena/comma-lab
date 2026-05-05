# pyc-recovery: STUB unreconstructible -- see .recovery_spec.json for dis() ground-truth
# pycdc could not produce parseable output; raw decompiled text preserved in _PYCDC_PARTIAL_OUTPUT below.
"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``5:20: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``test_endgame_archive_decision.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'src/tac/tests/test_endgame_archive_decision.py'
__recovery_spec__ = 'test_endgame_archive_decision.recovery_spec.json'
__recovery_ast_error__ = '5:20: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: test_endgame_archive_decision.cpython-312-pytest-9.0.3.pyc (Python 3.12)

from __future__ import annotations
import builtins as @py_builtins

rewrite
import struct = import _pytest.assertion.rewrite, assertion
import zipfile
from pathlib import Path
import pytest
from tac.endgame_archive_decision import build_endgame_decision_profile, render_markdown
from tac.pr85_bundle import PR85_HEADERLESS_RANDMULTI_SPECS, SEGMENT_ORDER, pack_pr85_bundle

def _br(data = None):
    brotli = pytest.importorskip('brotli')
    return brotli.compress(data, quality = 5)


def _qma9(bitstream = None):
    return b'QMA9' + struct.pack('<IIII', 600, 512, 384, len(bitstream)) + bitstream


def _legacy_randmulti():
    return _br(b'legacy-randmulti-payload')


def _rmb1_randmulti():
    raw_mask = None * sum * (lambda .0: pass# WARNING: Decompyle incomplete
)(PR85_HEADERLESS_RANDMULTI_SPECS())
    mask_br = _br(raw_mask)
    vals_br = _br(b'')
    return b'RMB1' + len(mask_br).to_bytes(2, 'little') + mask_br + vals_br


def _rsb1_actions(count = None):
    pass
# WARNING: Decompyle incomplete


def _segments(*, randmulti):
    pass
# WARNING: Decompyle incomplete


def _write_archive(path = None, members = None):
    zf = zipfile.ZipFile(path, 'w')
    for name, data in members:
        info = zipfile.ZipInfo(name)
        info.date_time = (1980, 1, 1, 0, 0, 0)
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 27525120
        zf.writestr(info, data)
    None(None, None)
    return None
    with None:
        if not None:
            pass


def test_endgame_profile_estimates_rmb1_side_info_transplant_delta(tmp_path = None):
    baseline = tmp_path / 'baseline.zip'
    candidate = tmp_path / 'candidate.zip'
    baseline_x = pack_pr85_bundle(_segments(), header_mode = 'explicit_30')
    candidate_x = pack_pr85_bundle(_segments(randmulti = _rmb1_randmulti()), header_mode = 'explicit_30')
    _write_archive(baseline, [
        ('x', baseline_x)])
    _write_archive(candidate, [
        ('x', candidate_x),
        ('a', _rsb1_actions())])
    profile = build_endgame_decision_profile({
        'frontier': baseline,
        'candidate': candidate }, frontier_label = 'frontier')
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
# WARNING: Decompyle incomplete


def test_endgame_profile_fails_closed_on_bad_rsb1_side_info(tmp_path = None):
    archive = tmp_path / 'bad_side.zip'
    x_payload = pack_pr85_bundle(_segments(), header_mode = 'explicit_30')
    _write_archive(archive, [
        ('x', x_payload),
        ('a', b'RSB1X\x02\x01\x00not-brotli')])
    profile = build_endgame_decision_profile({
        'bad': archive }, frontier_label = 'bad')
    report = profile['archives'][0]
    @py_assert0 = report['decision_support']['valid_for_byte_decision']
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
    @py_assert0 = 'side_member_validation_failed:a#1'
    @py_assert3 = report['decision_support']['blockers']
    @py_assert2 = @py_assert0 in @py_assert3
    if not @py_assert2:
        @py_format5 = @pytest_ar._call_reprcompare(('in',), (@py_assert2,), ('%(py1)s in %(py4)s',), (@py_assert0, @py_assert3)) % {
            'py1': @pytest_ar._saferepr(@py_assert0),
            'py4': @pytest_ar._saferepr(@py_assert3) }
        @py_format7 = 'assert %(py6)s' % {
            'py6': @py_format5 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format7))
    @py_assert0 = None
    @py_assert2 = None
    @py_assert3 = None
    @py_assert0 = report['side_info']['members'][0]['validation']['status']
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


def test_endgame_profile_rejects_central_local_name_mismatch(tmp_path = None):
    archive = tmp_path / 'mismatch.zip'
    x_payload = pack_pr85_bundle(_segments(), header_mode = 'explicit_30')
    _write_archive(archive, [
        ('x', x_payload)])
    raw = bytearray(archive.read_bytes())
# WARNING: Decompyle incomplete


"""
