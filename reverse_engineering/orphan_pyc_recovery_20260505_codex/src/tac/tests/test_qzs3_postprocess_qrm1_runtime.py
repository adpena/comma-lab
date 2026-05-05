# pyc-recovery: STUB unreconstructible -- see .recovery_spec.json for dis() ground-truth
# pycdc could not produce parseable output; raw decompiled text preserved in _PYCDC_PARTIAL_OUTPUT below.
"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``360:22: invalid decimal literal``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``test_qzs3_postprocess_qrm1_runtime.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'src/tac/tests/test_qzs3_postprocess_qrm1_runtime.py'
__recovery_spec__ = 'test_qzs3_postprocess_qrm1_runtime.recovery_spec.json'
__recovery_ast_error__ = '360:22: invalid decimal literal'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: test_qzs3_postprocess_qrm1_runtime.cpython-312-pytest-9.0.3.pyc (Python 3.12)

from __future__ import annotations
import builtins as @py_builtins

rewrite
import importlib.util = import _pytest.assertion.rewrite, assertion
import struct
import sys
import zipfile
from pathlib import Path
import brotli
import numpy as np
import pytest
import torch
from tac.henosis_pr82_transfer import Pr82RandmultiGroup, encode_randmulti_qrm1
REPO = Path(__file__).resolve().parents[3]
APPLY_PATH = REPO / 'submissions' / 'robust_current' / 'apply_qzs3_postprocess.py'
INFLATE_RENDERER_PATH = REPO / 'submissions' / 'robust_current' / 'inflate_renderer.py'
PR81_PR82_STACK_DIR = REPO / 'experiments/results/pr81_pr82_henosis_stack_20260503_codex'

def _load_apply():
    spec = importlib.util.spec_from_file_location('apply_qzs3_postprocess_qrm1_test', APPLY_PATH)
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


def _load_inflate_renderer():
    spec = importlib.util.spec_from_file_location('inflate_renderer_qpost_no_router_test', INFLATE_RENDERER_PATH)
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


def _qpost_with_randmulti(randmulti = None):
    lengths = [
        0] * 8
    lengths[-1] = len(randmulti)
# WARNING: Decompyle incomplete


def test_joint_generator_without_router_actions_does_not_raise_nameerror(tmp_path = None, monkeypatch = None):
    pass
# WARNING: Decompyle incomplete


def _group_for_spec(apply = None, spec = None, *, choice):
    group_id = apply.PR82_QRM1_RANDMULTI_SPECS.index(spec)
    rows = np.zeros((spec[3], 600), dtype = np.uint8)
    rows[(0, 0)] = choice
    return Pr82RandmultiGroup(group_index = group_id, height = spec[0], width = spec[1], amplitude = spec[2], scount = spec[3], rows = rows, payload_bytes = 0)


def test_qrm1_generic_randmulti_decodes_encoder_output(tmp_path = None):
    apply = _load_apply()
    group = _group_for_spec(apply, (4, 4, 1, 1), choice = 7)
    qpost_path = tmp_path / 'qpost.bin'
    qpost_path.write_bytes(_qpost_with_randmulti(encode_randmulti_qrm1([
        group])))
    state = apply.read_qpost(qpost_path, torch.device('cpu'))
    @py_assert1 = state.f1_randmulti
    @py_assert4 = None
    @py_assert3 = @py_assert1 is not @py_assert4
    if not @py_assert3:
        @py_format6 = @pytest_ar._call_reprcompare(('is not',), (@py_assert3,), ('%(py2)s\n{%(py2)s = %(py0)s.f1_randmulti\n} is not %(py5)s',), (@py_assert1, @py_assert4)) % {
            'py0': @pytest_ar._saferepr(state) if 'state' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(state) else 'state',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py5': @pytest_ar._saferepr(@py_assert4) }
        @py_format8 = 'assert %(py7)s' % {
            'py7': @py_format6 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format8))
    @py_assert1 = None
    @py_assert3 = None
    @py_assert4 = None
    @py_assert2 = state.f1_randmulti
    @py_assert4 = len(@py_assert2)
    @py_assert7 = 1
    @py_assert6 = @py_assert4 == @py_assert7
    if not @py_assert6:
        @py_format9 = @pytest_ar._call_reprcompare(('==',), (@py_assert6,), ('%(py5)s\n{%(py5)s = %(py0)s(%(py3)s\n{%(py3)s = %(py1)s.f1_randmulti\n})\n} == %(py8)s',), (@py_assert4, @py_assert7)) % {
            'py0': @pytest_ar._saferepr(len) if 'len' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(len) else 'len',
            'py1': @pytest_ar._saferepr(state) if 'state' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(state) else 'state',
            'py3': @pytest_ar._saferepr(@py_assert2),
            'py5': @pytest_ar._saferepr(@py_assert4),
            'py8': @pytest_ar._saferepr(@py_assert7) }
        @py_format11 = 'assert %(py10)s' % {
            'py10': @py_format9 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format11))
    @py_assert2 = None
    @py_assert4 = None
    @py_assert6 = None
    @py_assert7 = None
    (choices, lh, lw, amp) = state.f1_randmulti[0]
    @py_assert0 = (lh, lw, amp)
    @py_assert3 = (4, 4, 1)
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
    @py_assert2 = choices.shape
    @py_assert4 = tuple(@py_assert2)
    @py_assert7 = (1, 600)
    @py_assert6 = @py_assert4 == @py_assert7
    if not @py_assert6:
        @py_format9 = @pytest_ar._call_reprcompare(('==',), (@py_assert6,), ('%(py5)s\n{%(py5)s = %(py0)s(%(py3)s\n{%(py3)s = %(py1)s.shape\n})\n} == %(py8)s',), (@py_assert4, @py_assert7)) % {
            'py0': @pytest_ar._saferepr(tuple) if 'tuple' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(tuple) else 'tuple',
            'py1': @pytest_ar._saferepr(choices) if 'choices' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(choices) else 'choices',
            'py3': @pytest_ar._saferepr(@py_assert2),
            'py5': @pytest_ar._saferepr(@py_assert4),
            'py8': @pytest_ar._saferepr(@py_assert7) }
        @py_format11 = 'assert %(py10)s' % {
            'py10': @py_format9 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format11))
    @py_assert2 = None
    @py_assert4 = None
    @py_assert6 = None
    @py_assert7 = None
    @py_assert1 = choices[(0, 0)]
    @py_assert3 = @py_assert1.item
    @py_assert5 = @py_assert3()
    @py_assert7 = int(@py_assert5)
    @py_assert10 = 7
    @py_assert9 = @py_assert7 == @py_assert10
    if not @py_assert9:
        @py_format12 = @pytest_ar._call_reprcompare(('==',), (@py_assert9,), ('%(py8)s\n{%(py8)s = %(py0)s(%(py6)s\n{%(py6)s = %(py4)s\n{%(py4)s = %(py2)s.item\n}()\n})\n} == %(py11)s',), (@py_assert7, @py_assert10)) % {
            'py0': @pytest_ar._saferepr(int) if 'int' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(int) else 'int',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py4': @pytest_ar._saferepr(@py_assert3),
            'py6': @pytest_ar._saferepr(@py_assert5),
            'py8': @pytest_ar._saferepr(@py_assert7),
            'py11': @pytest_ar._saferepr(@py_assert10) }
        @py_format14 = 'assert %(py13)s' % {
            'py13': @py_format12 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format14))
    @py_assert1 = None
    @py_assert3 = None
    @py_assert5 = None
    @py_assert7 = None
    @py_assert9 = None
    @py_assert10 = None


def _encode_rmb1_from_headerless_rows(raw = None):
    cursor = 0
    mask = bytearray()
    values = bytearray()
    if cursor < len(raw):
        count = raw[cursor]
        cursor += 1
        if count == 255:
            count = int.from_bytes(raw[cursor:cursor + 2], 'little')
            cursor += 2
        row_mask = bytearray(75)
        idx = -1
        for _ in range(count):
            delta = 0
            shift = 0
            byte = raw[cursor]
            cursor += 1
            delta |= (byte & 127) << shift
            if byte < 128:
                pass
            else:
                shift += 7
            idx += delta + 1
        values.extend(raw[cursor:cursor + count])
        cursor += count = None
        mask.extend(row_mask)
        if cursor < len(raw):
            continue
    mask_br = brotli.compress(bytes(mask), quality = 11, lgwin = 24)
    vals_br = brotli.compress(bytes(values), quality = 11, lgwin = 24)
    return b'RMB1' + len(mask_br).to_bytes(2, 'little') + mask_br + vals_br


def test_rmb1_randmulti_decodes_to_headerless_sparse_rows():
    apply = _load_apply()
    first_group = b'\x02\x00\x02\x05\x07\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    empty_groups = b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    encoded = _encode_rmb1_from_headerless_rows(first_group + empty_groups)
    decoded = apply._decode_randmulti(encoded, torch.device('cpu'))
    @py_assert2 = None
    @py_assert1 = decoded is not @py_assert2
    if not @py_assert1:
        @py_format4 = @pytest_ar._call_reprcompare(('is not',), (@py_assert1,), ('%(py0)s is not %(py3)s',), (decoded, @py_assert2)) % {
            'py0': @pytest_ar._saferepr(decoded) if 'decoded' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(decoded) else 'decoded',
            'py3': @pytest_ar._saferepr(@py_assert2) }
        @py_format6 = 'assert %(py5)s' % {
            'py5': @py_format4 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format6))
    @py_assert1 = None
    @py_assert2 = None
    (choices, lh, lw, amp) = decoded[0]
    @py_assert0 = (lh, lw, amp)
    @py_assert3 = (24, 32, 1)
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
    @py_assert2 = choices.shape
    @py_assert4 = tuple(@py_assert2)
    @py_assert7 = (12, 600)
    @py_assert6 = @py_assert4 == @py_assert7
    if not @py_assert6:
        @py_format9 = @pytest_ar._call_reprcompare(('==',), (@py_assert6,), ('%(py5)s\n{%(py5)s = %(py0)s(%(py3)s\n{%(py3)s = %(py1)s.shape\n})\n} == %(py8)s',), (@py_assert4, @py_assert7)) % {
            'py0': @pytest_ar._saferepr(tuple) if 'tuple' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(tuple) else 'tuple',
            'py1': @pytest_ar._saferepr(choices) if 'choices' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(choices) else 'choices',
            'py3': @pytest_ar._saferepr(@py_assert2),
            'py5': @pytest_ar._saferepr(@py_assert4),
            'py8': @pytest_ar._saferepr(@py_assert7) }
        @py_format11 = 'assert %(py10)s' % {
            'py10': @py_format9 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format11))
    @py_assert2 = None
    @py_assert4 = None
    @py_assert6 = None
    @py_assert7 = None
    @py_assert1 = choices[(0, 0)]
    @py_assert3 = @py_assert1.item
    @py_assert5 = @py_assert3()
    @py_assert7 = int(@py_assert5)
    @py_assert10 = 5
    @py_assert9 = @py_assert7 == @py_assert10
    if not @py_assert9:
        @py_format12 = @pytest_ar._call_reprcompare(('==',), (@py_assert9,), ('%(py8)s\n{%(py8)s = %(py0)s(%(py6)s\n{%(py6)s = %(py4)s\n{%(py4)s = %(py2)s.item\n}()\n})\n} == %(py11)s',), (@py_assert7, @py_assert10)) % {
            'py0': @pytest_ar._saferepr(int) if 'int' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(int) else 'int',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py4': @pytest_ar._saferepr(@py_assert3),
            'py6': @pytest_ar._saferepr(@py_assert5),
            'py8': @pytest_ar._saferepr(@py_assert7),
            'py11': @pytest_ar._saferepr(@py_assert10) }
        @py_format14 = 'assert %(py13)s' % {
            'py13': @py_format12 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format14))
    @py_assert1 = None
    @py_assert3 = None
    @py_assert5 = None
    @py_assert7 = None
    @py_assert9 = None
    @py_assert10 = None
    @py_assert1 = choices[(0, 3)]
    @py_assert3 = @py_assert1.item
    @py_assert5 = @py_assert3()
    @py_assert7 = int(@py_assert5)
    @py_assert10 = 7
    @py_assert9 = @py_assert7 == @py_assert10
    if not @py_assert9:
        @py_format12 = @pytest_ar._call_reprcompare(('==',), (@py_assert9,), ('%(py8)s\n{%(py8)s = %(py0)s(%(py6)s\n{%(py6)s = %(py4)s\n{%(py4)s = %(py2)s.item\n}()\n})\n} == %(py11)s',), (@py_assert7, @py_assert10)) % {
            'py0': @pytest_ar._saferepr(int) if 'int' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(int) else 'int',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py4': @pytest_ar._saferepr(@py_assert3),
            'py6': @pytest_ar._saferepr(@py_assert5),
            'py8': @pytest_ar._saferepr(@py_assert7),
            'py11': @pytest_ar._saferepr(@py_assert10) }
        @py_format14 = 'assert %(py13)s' % {
            'py13': @py_format12 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format14))
    @py_assert1 = None
    @py_assert3 = None
    @py_assert5 = None
    @py_assert7 = None
    @py_assert9 = None
    @py_assert10 = None


def test_qrm1_supported_global_special_branch_applies_to_second_frame(tmp_path = None):
    apply = _load_apply()
    group = _group_for_spec(apply, (222, 222, 4, 1), choice = 1)
    qpost_path = tmp_path / 'qpost.bin'
    qpost_path.write_bytes(_qpost_with_randmulti(encode_randmulti_qrm1([
        group])))
    state = apply.read_qpost(qpost_path, torch.device('cpu'))
    raw_path = tmp_path / '0.raw'
    frame0 = np.full((2, 2, 3), 10, dtype = np.uint8)
    frame1 = np.full((2, 2, 3), 10, dtype = np.uint8)
    raw_path.write_bytes(np.stack([
        frame0,
        frame1], axis = 0).tobytes())
    apply.apply_qpost_to_raw(raw_path, state, height = 2, width = 2, batch_pairs = 1, device = torch.device('cpu'))
    decoded = np.frombuffer(raw_path.read_bytes(), dtype = np.uint8).reshape(2, 2, 2, 3)
    @py_assert1 = np.array_equal
    @py_assert3 = decoded[0]
    @py_assert6 = @py_assert1(@py_assert3, frame0)
    if not @py_assert6:
        @py_format8 = 'assert %(py7)s\n{%(py7)s = %(py2)s\n{%(py2)s = %(py0)s.array_equal\n}(%(py4)s, %(py5)s)\n}' % {
            'py0': @pytest_ar._saferepr(np) if 'np' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(np) else 'np',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py4': @pytest_ar._saferepr(@py_assert3),
            'py5': @pytest_ar._saferepr(frame0) if 'frame0' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(frame0) else 'frame0',
            'py7': @pytest_ar._saferepr(@py_assert6) }
        raise AssertionError(@pytest_ar._format_explanation(@py_format8))
    @py_assert1 = None
    @py_assert3 = None
    @py_assert6 = None
    @py_assert1 = np.array_equal
    @py_assert3 = decoded[1]
    @py_assert6 = np.full
    @py_assert8 = (2, 2, 3)
    @py_assert10 = 6
    @py_assert13 = np.uint8
    @py_assert15 = @py_assert6(@py_assert8, @py_assert10, dtype = @py_assert13)
    @py_assert17 = @py_assert1(@py_assert3, @py_assert15)
    if not @py_assert17:
        @py_format19 = 'assert %(py18)s\n{%(py18)s = %(py2)s\n{%(py2)s = %(py0)s.array_equal\n}(%(py4)s, %(py16)s\n{%(py16)s = %(py7)s\n{%(py7)s = %(py5)s.full\n}(%(py9)s, %(py11)s, dtype=%(py14)s\n{%(py14)s = %(py12)s.uint8\n})\n})\n}' % {
            'py0': @pytest_ar._saferepr(np) if 'np' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(np) else 'np',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py4': @pytest_ar._saferepr(@py_assert3),
            'py5': @pytest_ar._saferepr(np) if 'np' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(np) else 'np',
            'py7': @pytest_ar._saferepr(@py_assert6),
            'py9': @pytest_ar._saferepr(@py_assert8),
            'py11': @pytest_ar._saferepr(@py_assert10),
            'py12': @pytest_ar._saferepr(np) if 'np' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(np) else 'np',
            'py14': @pytest_ar._saferepr(@py_assert13),
            'py16': @pytest_ar._saferepr(@py_assert15),
            'py18': @pytest_ar._saferepr(@py_assert17) }
        raise AssertionError(@pytest_ar._format_explanation(@py_format19))
    @py_assert1 = None
    @py_assert3 = None
    @py_assert6 = None
    @py_assert8 = None
    @py_assert10 = None
    @py_assert13 = None
    @py_assert15 = None
    @py_assert17 = None


def test_qrm1_duplicate_group_id_fails_closed(tmp_path = None):
    apply = _load_apply()
    group_id = apply.PR82_QRM1_RANDMULTI_SPECS.index((4, 4, 1, 1))
    raw = b'QRM1' + 2.to_bytes(2, 'little')
    raw += int(group_id).to_bytes(2, 'little') + b'\x00'
    raw += int(group_id).to_bytes(2, 'little') + b'\x00'
    qpost_path = tmp_path / 'qpost.bin'
    qpost_path.write_bytes(_qpost_with_randmulti(brotli.compress(raw, quality = 11)))
    pytest.raises(ValueError, match = 'duplicate randmulti group id')
    apply.read_qpost(qpost_path, torch.device('cpu'))
    None(None, None)
    return None
    with None:
        if not None:
            pass


def test_qrm1_mask_dependent_special_branch_requires_source_masks(tmp_path = None):
    apply = _load_apply()
    group = _group_for_spec(apply, (223, 222, 2, 1), choice = 1)
    qpost_path = tmp_path / 'qpost.bin'
    qpost_path.write_bytes(_qpost_with_randmulti(encode_randmulti_qrm1([
        group])))
    state = apply.read_qpost(qpost_path, torch.device('cpu'))
    @py_assert1 = apply.qpost_requires_source_masks
    @py_assert4 = @py_assert1(state)
    @py_assert7 = True
    @py_assert6 = @py_assert4 is @py_assert7
    if not @py_assert6:
        @py_format9 = @pytest_ar._call_reprcompare(('is',), (@py_assert6,), ('%(py5)s\n{%(py5)s = %(py2)s\n{%(py2)s = %(py0)s.qpost_requires_source_masks\n}(%(py3)s)\n} is %(py8)s',), (@py_assert4, @py_assert7)) % {
            'py0': @pytest_ar._saferepr(apply) if 'apply' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(apply) else 'apply',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py3': @pytest_ar._saferepr(state) if 'state' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(state) else 'state',
            'py5': @pytest_ar._saferepr(@py_assert4),
            'py8': @pytest_ar._saferepr(@py_assert7) }
        @py_format11 = 'assert %(py10)s' % {
            'py10': @py_format9 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format11))
    @py_assert1 = None
    @py_assert4 = None
    @py_assert6 = None
    @py_assert7 = None
    raw_path = tmp_path / '0.raw'
    frame0 = np.full((2, 2, 3), 10, dtype = np.uint8)
    frame1 = np.full((2, 2, 3), 10, dtype = np.uint8)
    raw_path.write_bytes(np.stack([
        frame0,
        frame1], axis = 0).tobytes())
    pytest.raises(ValueError, match = 'requires source masks')
    apply.apply_qpost_to_raw(raw_path, state, height = 2, width = 2, batch_pairs = 1, device = torch.device('cpu'))
    None(None, None)
    return None
    with None:
        if not None:
            pass


def test_qrm1_mask_dependent_class_branch_applies_with_source_masks(tmp_path = None):
    apply = _load_apply()
    group = _group_for_spec(apply, (223, 222, 2, 1), choice = 1)
    qpost_path = tmp_path / 'qpost.bin'
    qpost_path.write_bytes(_qpost_with_randmulti(encode_randmulti_qrm1([
        group])))
    state = apply.read_qpost(qpost_path, torch.device('cpu'))
    raw_path = tmp_path / '0.raw'
    frame0 = np.full((2, 2, 3), 10, dtype = np.uint8)
    frame1 = np.full((2, 2, 3), 10, dtype = np.uint8)
    raw_path.write_bytes(np.stack([
        frame0,
        frame1], axis = 0).tobytes())
    source_masks = torch.tensor([
        [
            [
                0,
                1],
            [
                1,
                0]]], dtype = torch.long)
    apply.apply_qpost_to_raw(raw_path, state, height = 2, width = 2, batch_pairs = 1, device = torch.device('cpu'), source_masks = source_masks)
    decoded = np.frombuffer(raw_path.read_bytes(), dtype = np.uint8).reshape(2, 2, 2, 3)
    expected_f1 = frame1.copy()
    expected_f1[(source_masks[0].numpy() == 0, 0)] = 8
    @py_assert1 = np.array_equal
    @py_assert3 = decoded[0]
    @py_assert6 = @py_assert1(@py_assert3, frame0)
    if not @py_assert6:
        @py_format8 = 'assert %(py7)s\n{%(py7)s = %(py2)s\n{%(py2)s = %(py0)s.array_equal\n}(%(py4)s, %(py5)s)\n}' % {
            'py0': @pytest_ar._saferepr(np) if 'np' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(np) else 'np',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py4': @pytest_ar._saferepr(@py_assert3),
            'py5': @pytest_ar._saferepr(frame0) if 'frame0' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(frame0) else 'frame0',
            'py7': @pytest_ar._saferepr(@py_assert6) }
        raise AssertionError(@pytest_ar._format_explanation(@py_format8))
    @py_assert1 = None
    @py_assert3 = None
    @py_assert6 = None
    @py_assert1 = np.array_equal
    @py_assert3 = decoded[1]
    @py_assert6 = @py_assert1(@py_assert3, expected_f1)
    if not @py_assert6:
        @py_format8 = 'assert %(py7)s\n{%(py7)s = %(py2)s\n{%(py2)s = %(py0)s.array_equal\n}(%(py4)s, %(py5)s)\n}' % {
            'py0': @pytest_ar._saferepr(np) if 'np' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(np) else 'np',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py4': @pytest_ar._saferepr(@py_assert3),
            'py5': @pytest_ar._saferepr(expected_f1) if 'expected_f1' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(expected_f1) else 'expected_f1',
            'py7': @pytest_ar._saferepr(@py_assert6) }
        raise AssertionError(@pytest_ar._format_explanation(@py_format8))
    @py_assert1 = None
    @py_assert3 = None
    @py_assert6 = None


def test_qrm1_support_classifier_reports_active_unsupported_groups(tmp_path = None):
    pass
# WARNING: Decompyle incomplete


def test_qrm1_support_classifier_marks_inactive_source_mask_group_dispatchable_without_requirement(tmp_path = None):
    apply = _load_apply()
    inactive_source_mask = _group_for_spec(apply, (223, 222, 2, 1), choice = 0)
    qpost_path = tmp_path / 'qpost.bin'
    qpost_path.write_bytes(_qpost_with_randmulti(encode_randmulti_qrm1([
        inactive_source_mask])))
    report = apply.classify_qpost_qrm1_support(qpost_path)
    @py_assert0 = report['dispatchable_qrm1']
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
    @py_assert0 = report['active_unsupported_group_ids']
    @py_assert3 = []
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
    @py_assert0 = report['source_mask_required_group_ids']
    @py_assert3 = []
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
    @py_assert1 = inactive_source_mask.group_index
    @py_assert4 = report['supported_group_ids']
    @py_assert3 = @py_assert1 in @py_assert4
    if not @py_assert3:
        @py_format6 = @pytest_ar._call_reprcompare(('in',), (@py_assert3,), ('%(py2)s\n{%(py2)s = %(py0)s.group_index\n} in %(py5)s',), (@py_assert1, @py_assert4)) % {
            'py0': @pytest_ar._saferepr(inactive_source_mask) if 'inactive_source_mask' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(inactive_source_mask) else 'inactive_source_mask',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py5': @pytest_ar._saferepr(@py_assert4) }
        @py_format8 = 'assert %(py7)s' % {
            'py7': @py_format6 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format8))
    @py_assert1 = None
    @py_assert3 = None
    @py_assert4 = None
    state = apply.read_qpost(qpost_path, torch.device('cpu'))
    @py_assert1 = state.f1_randmulti
    @py_assert4 = None
    @py_assert3 = @py_assert1 is not @py_assert4
    if not @py_assert3:
        @py_format6 = @pytest_ar._call_reprcompare(('is not',), (@py_assert3,), ('%(py2)s\n{%(py2)s = %(py0)s.f1_randmulti\n} is not %(py5)s',), (@py_assert1, @py_assert4)) % {
            'py0': @pytest_ar._saferepr(state) if 'state' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(state) else 'state',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py5': @pytest_ar._saferepr(@py_assert4) }
        @py_format8 = 'assert %(py7)s' % {
            'py7': @py_format6 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format8))
    @py_assert1 = None
    @py_assert3 = None
    @py_assert4 = None
    @py_assert1 = apply.qpost_requires_source_masks
    @py_assert4 = @py_assert1(state)
    @py_assert7 = False
    @py_assert6 = @py_assert4 is @py_assert7
    if not @py_assert6:
        @py_format9 = @pytest_ar._call_reprcompare(('is',), (@py_assert6,), ('%(py5)s\n{%(py5)s = %(py2)s\n{%(py2)s = %(py0)s.qpost_requires_source_masks\n}(%(py3)s)\n} is %(py8)s',), (@py_assert4, @py_assert7)) % {
            'py0': @pytest_ar._saferepr(apply) if 'apply' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(apply) else 'apply',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py3': @pytest_ar._saferepr(state) if 'state' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(state) else 'state',
            'py5': @pytest_ar._saferepr(@py_assert4),
            'py8': @pytest_ar._saferepr(@py_assert7) }
        @py_format11 = 'assert %(py10)s' % {
            'py10': @py_format9 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format11))
    @py_assert1 = None
    @py_assert4 = None
    @py_assert6 = None
    @py_assert7 = None


def test_qrm1_archive_classifier_reads_candidate_qpost_member(tmp_path = None):
    apply = _load_apply()
    group = _group_for_spec(apply, (223, 224, 4, 1), choice = 3)
    archive = tmp_path / 'candidate.zip'
    zf = zipfile.ZipFile(archive, 'w')
    zf.writestr('p', b'payload')
    zf.writestr('qpost.bin', _qpost_with_randmulti(encode_randmulti_qrm1([
        group])))
    None(None, None)
    report = apply.classify_archive_qrm1_support(archive)
    @py_assert0 = report['archive_members']
    @py_assert3 = [
        'p',
        'qpost.bin']
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
    @py_assert0 = report['active_unsupported_group_ids']
    @py_assert3 = []
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
    @py_assert0 = report['source_mask_required_group_ids']
    @py_assert3 = [
        group.group_index]
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
    return None
    with None:
        if not None:
            pass
    continue


def test_qrm1_group_id_outside_replay_specs_fails_closed(tmp_path = None):
    apply = _load_apply()
    raw = b'QRM1' + 1.to_bytes(2, 'little') + 999.to_bytes(2, 'little') + b'\x00'
    qpost_path = tmp_path / 'qpost.bin'
    qpost_path.write_bytes(_qpost_with_randmulti(brotli.compress(raw, quality = 11)))
    pytest.raises(ValueError, match = 'outside PR82 replay specs')
    apply.read_qpost(qpost_path, torch.device('cpu'))
    None(None, None)
    return None
    with None:
        if not None:
            pass


def test_qrm1_archive_classifier_fails_closed_on_duplicate_qpost_member(tmp_path = None):
    apply = _load_apply()
    archive = tmp_path / 'candidate.zip'
    zf = zipfile.ZipFile(archive, 'w')
    zf.writestr('qpost.bin', b'first')
    zf.writestr('qpost.bin', b'second')
    None(None, None)
    pytest.raises(ValueError, match = 'duplicate qpost.bin')
    apply.classify_archive_qrm1_support(archive)
    None(None, None)
    return None
    with None:
        if not None:
            pass
    continue
    with None:
        if not None:
            pass

test_generated_pr81_pr82_qrm1_candidates_have_precise_unsupported_group_ids = (lambda : apply = _load_apply()expected_source_mask_required = [
62,
63,
64,
65,
66,
67,
68,
70]for candidate_id in ('pr81_qma9_pr82_qps1_qrm1_all072_randmulti', 'pr81_qma9_pr82_qps1_controls_qrm1_all072'):
report = apply.classify_archive_qrm1_support(PR81_PR82_STACK_DIR / candidate_id / 'archive.zip')@py_assert0 = report['active_unsupported_group_ids']@py_assert3 = []@py_assert2 = @py_assert0 == @py_assert3if not @py_assert2:
@py_format5 = @pytest_ar._call_reprcompare(('==',), (@py_assert2,), ('%(py1)s == %(py4)s',), (@py_assert0, @py_assert3)) % {
'py1': @pytest_ar._saferepr(@py_assert0),
'py4': @pytest_ar._saferepr(@py_assert3) }@py_format7 = 'assert %(py6)s' % {
'py6': @py_format5 }raise AssertionError(@pytest_ar._format_explanation(@py_format7))@py_assert0 = None@py_assert2 = None@py_assert3 = None@py_assert0 = report['source_mask_required_group_ids']@py_assert2 = @py_assert0 == expected_source_mask_requiredif not @py_assert2:
@py_format4 = @pytest_ar._call_reprcompare(('==',), (@py_assert2,), ('%(py1)s == %(py3)s',), (@py_assert0, expected_source_mask_required)) % {
'py1': @pytest_ar._saferepr(@py_assert0),
'py3': @pytest_ar._saferepr(expected_source_mask_required) if 'expected_source_mask_required' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(expected_source_mask_required) else 'expected_source_mask_required' }@py_format6 = 'assert %(py5)s' % {
'py5': @py_format4 }raise AssertionError(@pytest_ar._format_explanation(@py_format6))@py_assert0 = None@py_assert2 = None@py_assert0 = report['dispatchable_qrm1']@py_assert3 = True@py_assert2 = @py_assert0 is @py_assert3if not @py_assert2:
@py_format5 = @pytest_ar._call_reprcompare(('is',), (@py_assert2,), ('%(py1)s is %(py4)s',), (@py_assert0, @py_assert3)) % {
'py1': @pytest_ar._saferepr(@py_assert0),
'py4': @pytest_ar._saferepr(@py_assert3) }@py_format7 = 'assert %(py6)s' % {
'py6': @py_format5 }raise AssertionError(@pytest_ar._format_explanation(@py_format7))@py_assert0 = None@py_assert2 = None@py_assert3 = Nonefor candidate_id in ('pr81_qma9_pr82_qps1_qrm1_supported_subset_randmulti', 'pr81_qma9_pr82_qps1_controls_qrm1_supported_subset'):
report = apply.classify_archive_qrm1_support(PR81_PR82_STACK_DIR / candidate_id / 'archive.zip')@py_assert0 = report['active_unsupported_group_ids']@py_assert3 = []@py_assert2 = @py_assert0 == @py_assert3if not @py_assert2:
@py_format5 = @pytest_ar._call_reprcompare(('==',), (@py_assert2,), ('%(py1)s == %(py4)s',), (@py_assert0, @py_assert3)) % {
'py1': @pytest_ar._saferepr(@py_assert0),
'py4': @pytest_ar._saferepr(@py_assert3) }@py_format7 = 'assert %(py6)s' % {
'py6': @py_format5 }raise AssertionError(@pytest_ar._format_explanation(@py_format7))@py_assert0 = None@py_assert2 = None@py_assert3 = None@py_assert0 = report['source_mask_required_group_ids']@py_assert2 = @py_assert0 == expected_source_mask_requiredif not @py_assert2:
@py_format4 = @pytest_ar._call_reprcompare(('==',), (@py_assert2,), ('%(py1)s == %(py3)s',), (@py_assert0, expected_source_mask_required)) % {
'py1': @pytest_ar._saferepr(@py_assert0),
'py3': @pytest_ar._saferepr(expected_source_mask_required) if 'expected_source_mask_required' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(expected_source_mask_required) else 'expected_source_mask_required' }@py_format6 = 'assert %(py5)s' % {
'py5': @py_format4 }raise AssertionError(@pytest_ar._format_explanation(@py_format6))@py_assert0 = None@py_assert2 = None@py_assert0 = report['dispatchable_qrm1']@py_assert3 = True@py_assert2 = @py_assert0 is @py_assert3if not @py_assert2:
@py_format5 = @pytest_ar._call_reprcompare(('is',), (@py_assert2,), ('%(py1)s is %(py4)s',), (@py_assert0, @py_assert3)) % {
'py1': @pytest_ar._saferepr(@py_assert0),
'py4': @pytest_ar._saferepr(@py_assert3) }@py_format7 = 'assert %(py6)s' % {
'py6': @py_format5 }raise AssertionError(@pytest_ar._format_explanation(@py_format7))@py_assert0 = None@py_assert2 = None@py_assert3 = None@py_assert0 = 61@py_assert3 = report['supported_group_ids']@py_assert2 = @py_assert0 in @py_assert3if not @py_assert2:
@py_format5 = @pytest_ar._call_reprcompare(('in',), (@py_assert2,), ('%(py1)s in %(py4)s',), (@py_assert0, @py_assert3)) % {
'py1': @pytest_ar._saferepr(@py_assert0),
'py4': @pytest_ar._saferepr(@py_assert3) }@py_format7 = 'assert %(py6)s' % {
'py6': @py_format5 }raise AssertionError(@pytest_ar._format_explanation(@py_format7))@py_assert0 = None@py_assert2 = None@py_assert3 = None@py_assert0 = 69@py_assert3 = report['supported_group_ids']@py_assert2 = @py_assert0 in @py_assert3if not @py_assert2:
@py_format5 = @pytest_ar._call_reprcompare(('in',), (@py_assert2,), ('%(py1)s in %(py4)s',), (@py_assert0, @py_assert3)) % {
'py1': @pytest_ar._saferepr(@py_assert0),
'py4': @pytest_ar._saferepr(@py_assert3) }@py_format7 = 'assert %(py6)s' % {
'py6': @py_format5 }raise AssertionError(@pytest_ar._format_explanation(@py_format7))@py_assert0 = None@py_assert2 = None@py_assert3 = None@py_assert0 = 71@py_assert3 = report['supported_group_ids']@py_assert2 = @py_assert0 in @py_assert3if not @py_assert2:
@py_format5 = @pytest_ar._call_reprcompare(('in',), (@py_assert2,), ('%(py1)s in %(py4)s',), (@py_assert0, @py_assert3)) % {
'py1': @pytest_ar._saferepr(@py_assert0),
'py4': @pytest_ar._saferepr(@py_assert3) }@py_format7 = 'assert %(py6)s' % {
'py6': @py_format5 }raise AssertionError(@pytest_ar._format_explanation(@py_format7))@py_assert0 = None@py_assert2 = None@py_assert3 = Nonenm2_report = apply.classify_archive_qrm1_support(PR81_PR82_STACK_DIR / 'pr81_qma9_pr82_qps1_nm2_generic_randmulti' / 'archive.zip')@py_assert0 = nm2_report['contract']@py_assert3 = 'not_qrm1'@py_assert2 = @py_assert0 == @py_assert3if not @py_assert2:
@py_format5 = @pytest_ar._call_reprcompare(('==',), (@py_assert2,), ('%(py1)s == %(py4)s',), (@py_assert0, @py_assert3)) % {
'py1': @pytest_ar._saferepr(@py_assert0),
'py4': @pytest_ar._saferepr(@py_assert3) }@py_format7 = 'assert %(py6)s' % {
'py6': @py_format5 }raise AssertionError(@pytest_ar._format_explanation(@py_format7))@py_assert0 = None@py_assert2 = None@py_assert3 = None@py_assert0 = nm2_report['unsupported_group_ids']@py_assert3 = []@py_assert2 = @py_assert0 == @py_assert3if not @py_assert2:
@py_format5 = @pytest_ar._call_reprcompare(('==',), (@py_assert2,), ('%(py1)s == %(py4)s',), (@py_assert0, @py_assert3)) % {
'py1': @pytest_ar._saferepr(@py_assert0),
'py4': @pytest_ar._saferepr(@py_assert3) }@py_format7 = 'assert %(py6)s' % {
'py6': @py_format5 }raise AssertionError(@pytest_ar._format_explanation(@py_format7))@py_assert0 = None@py_assert2 = None@py_assert3 = None)()

"""
