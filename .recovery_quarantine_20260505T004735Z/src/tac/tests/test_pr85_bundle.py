"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``5:20: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``test_pr85_bundle.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'src/tac/tests/test_pr85_bundle.py'
__recovery_spec__ = 'test_pr85_bundle.recovery_spec.json'
__recovery_ast_error__ = '5:20: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: test_pr85_bundle.cpython-312-pytest-9.0.3.pyc (Python 3.12)

from __future__ import annotations
import builtins as @py_builtins

rewrite
import struct = import _pytest.assertion.rewrite, assertion
import zipfile
from pathlib import Path
import pytest
from tac.pr85_bundle import FIXED_V5_LENGTHS, PR85_HEADERLESS_RANDMULTI_SPECS, Pr85BundleError, SEGMENT_ORDER, build_pr85_qpost_bin, compare_pr85_randmulti_decoded_rows, decode_pr85_p1d1_pose_to_fp16, decode_pr85_randmulti_to_headerless_rows, decode_rmb1_randmulti_payload, expand_pr85_bundle_to_runtime_members, pack_pr85_bundle, parse_hpm1_mask_segment, parse_pr85_bundle, transcode_pr85_randmulti_to_qrm1, validate_pr85_member_name
REPO = Path(__file__).resolve().parents[3]

def _segments():
    return {
        'mask': b'QMA9mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm',
        'model': b'QH0rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr',
        'pose': b'P1D1pppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppppp',
        'post': b'post',
        'shift': b'shift',
        'frac': b'frac',
        'frac2': b'frac2',
        'frac3': b'frac3',
        'bias': b'B' * FIXED_V5_LENGTHS['bias'],
        'region': b'R' * FIXED_V5_LENGTHS['region'],
        'randmulti': b'randmulti' }


def _vlq(value = None):
    out = bytearray()
    byte = value & 127
    value >>= 7
    if value:
        out.append(byte | 128)
    else:
        out.append(byte)
        return bytes(out)


def _zigzag(value = None):
    return value << 1 ^ value >> 31


def _p1d1_pose_raw(*, dims):
    streams = []
    for dim in dims:
        values = [
            0] * 600
        previous = 0
        stream = bytearray()
        for value in values:
            stream.extend(_vlq(_zigzag(value - previous)))
            previous = value
        streams.append(bytes(stream))
    header = b'P1D1' + bytes([
        len(dims)])
    for dim, stream in zip(dims, streams):
        header += bytes([
            dim]) + len(stream).to_bytes(2, 'little')
    return header + b''.join(streams)


def _randmulti_zero_payload():
    return sum * (lambda .0: pass# WARNING: Decompyle incomplete
)(PR85_HEADERLESS_RANDMULTI_SPECS())


def _rmb1_from_headerless_rows(raw = None):
    brotli = pytest.importorskip('brotli')
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
    mask_br = brotli.compress(bytes(mask), quality = 5)
    vals_br = brotli.compress(bytes(values), quality = 5)
    return b'RMB1' + len(mask_br).to_bytes(2, 'little') + mask_br + vals_br


def _archive_member(path = None, member_name = None):
    if not path.is_file():
        pytest.skip(f'''local archive artifact is missing: {path}''')
    zf = zipfile.ZipFile(path, 'r')
    None(None, None)
    return 
    with None:
        if not None, zf.read(member_name):
            pass


def _hpm1_mask_segment():
    tokens = b'TOKNTOKNTOKN'
    hpac = b'HPACMODEL'
    header = b'HPM1' + struct.pack('<IIIIIIIIIII', 600, 384, 512, 32, 2, 64, 1, 8, len(tokens), len(hpac), 4)
    return header + tokens + hpac


def test_v5_pack_parse_roundtrip():
    segments = _segments()
    raw = pack_pr85_bundle(segments, header_mode = 'v5')
    parsed = parse_pr85_bundle(raw)
    @py_assert1 = parsed.format
    @py_assert4 = 'pr85_v5_micro_24bit_lengths_fixed_bias_region'
    @py_assert3 = @py_assert1 == @py_assert4
    if not @py_assert3:
        @py_format6 = @pytest_ar._call_reprcompare(('==',), (@py_assert3,), ('%(py2)s\n{%(py2)s = %(py0)s.format\n} == %(py5)s',), (@py_assert1, @py_assert4)) % {
            'py0': @pytest_ar._saferepr(parsed) if 'parsed' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(parsed) else 'parsed',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py5': @pytest_ar._saferepr(@py_assert4) }
        @py_format8 = 'assert %(py7)s' % {
            'py7': @py_format6 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format8))
    @py_assert1 = None
    @py_assert3 = None
    @py_assert4 = None
    @py_assert1 = parsed.header_bytes
    @py_assert4 = 24
    @py_assert3 = @py_assert1 == @py_assert4
    if not @py_assert3:
        @py_format6 = @pytest_ar._call_reprcompare(('==',), (@py_assert3,), ('%(py2)s\n{%(py2)s = %(py0)s.header_bytes\n} == %(py5)s',), (@py_assert1, @py_assert4)) % {
            'py0': @pytest_ar._saferepr(parsed) if 'parsed' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(parsed) else 'parsed',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py5': @pytest_ar._saferepr(@py_assert4) }
        @py_format8 = 'assert %(py7)s' % {
            'py7': @py_format6 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format8))
    @py_assert1 = None
    @py_assert3 = None
    @py_assert4 = None
    @py_assert1 = parsed.fixed_length_segments
    @py_assert3 = @py_assert1 == FIXED_V5_LENGTHS
    if not @py_assert3:
        @py_format5 = @pytest_ar._call_reprcompare(('==',), (@py_assert3,), ('%(py2)s\n{%(py2)s = %(py0)s.fixed_length_segments\n} == %(py4)s',), (@py_assert1, FIXED_V5_LENGTHS)) % {
            'py0': @pytest_ar._saferepr(parsed) if 'parsed' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(parsed) else 'parsed',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py4': @pytest_ar._saferepr(FIXED_V5_LENGTHS) if 'FIXED_V5_LENGTHS' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(FIXED_V5_LENGTHS) else 'FIXED_V5_LENGTHS' }
        @py_format7 = 'assert %(py6)s' % {
            'py6': @py_format5 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format7))
    @py_assert1 = None
    @py_assert3 = None
    @py_assert1 = parsed.segment_lengths
# WARNING: Decompyle incomplete


def test_explicit_30_pack_parse_roundtrip_with_changed_bias_region():
    segments = _segments()
    segments['bias'] = b'bias-short'
    segments['region'] = b'region-short'
    raw = pack_pr85_bundle(segments, header_mode = 'explicit_30')
    parsed = parse_pr85_bundle(raw)
    @py_assert1 = parsed.format
    @py_assert4 = 'pr85_explicit_30byte_lengths'
    @py_assert3 = @py_assert1 == @py_assert4
    if not @py_assert3:
        @py_format6 = @pytest_ar._call_reprcompare(('==',), (@py_assert3,), ('%(py2)s\n{%(py2)s = %(py0)s.format\n} == %(py5)s',), (@py_assert1, @py_assert4)) % {
            'py0': @pytest_ar._saferepr(parsed) if 'parsed' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(parsed) else 'parsed',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py5': @pytest_ar._saferepr(@py_assert4) }
        @py_format8 = 'assert %(py7)s' % {
            'py7': @py_format6 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format8))
    @py_assert1 = None
    @py_assert3 = None
    @py_assert4 = None
    @py_assert1 = parsed.header_bytes
    @py_assert4 = 30
    @py_assert3 = @py_assert1 == @py_assert4
    if not @py_assert3:
        @py_format6 = @pytest_ar._call_reprcompare(('==',), (@py_assert3,), ('%(py2)s\n{%(py2)s = %(py0)s.header_bytes\n} == %(py5)s',), (@py_assert1, @py_assert4)) % {
            'py0': @pytest_ar._saferepr(parsed) if 'parsed' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(parsed) else 'parsed',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py5': @pytest_ar._saferepr(@py_assert4) }
        @py_format8 = 'assert %(py7)s' % {
            'py7': @py_format6 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format8))
    @py_assert1 = None
    @py_assert3 = None
    @py_assert4 = None
    @py_assert1 = parsed.fixed_length_segments
    @py_assert4 = { }
    @py_assert3 = @py_assert1 == @py_assert4
    if not @py_assert3:
        @py_format6 = @pytest_ar._call_reprcompare(('==',), (@py_assert3,), ('%(py2)s\n{%(py2)s = %(py0)s.fixed_length_segments\n} == %(py5)s',), (@py_assert1, @py_assert4)) % {
            'py0': @pytest_ar._saferepr(parsed) if 'parsed' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(parsed) else 'parsed',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py5': @pytest_ar._saferepr(@py_assert4) }
        @py_format8 = 'assert %(py7)s' % {
            'py7': @py_format6 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format8))
    @py_assert1 = None
    @py_assert3 = None
    @py_assert4 = None
    @py_assert2 = parsed.segments
    @py_assert4 = dict(@py_assert2)
    @py_assert6 = @py_assert4 == segments
    if not @py_assert6:
        @py_format8 = @pytest_ar._call_reprcompare(('==',), (@py_assert6,), ('%(py5)s\n{%(py5)s = %(py0)s(%(py3)s\n{%(py3)s = %(py1)s.segments\n})\n} == %(py7)s',), (@py_assert4, segments)) % {
            'py0': @pytest_ar._saferepr(dict) if 'dict' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(dict) else 'dict',
            'py1': @pytest_ar._saferepr(parsed) if 'parsed' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(parsed) else 'parsed',
            'py3': @pytest_ar._saferepr(@py_assert2),
            'py5': @pytest_ar._saferepr(@py_assert4),
            'py7': @pytest_ar._saferepr(segments) if 'segments' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(segments) else 'segments' }
        @py_format10 = 'assert %(py9)s' % {
            'py9': @py_format8 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format10))
    @py_assert2 = None
    @py_assert4 = None
    @py_assert6 = None


def test_v5_rejects_changed_fixed_length_segments():
    segments = _segments()
    segments['bias'] = b'too-short'
    pytest.raises(Pr85BundleError, match = "fixed-length segment 'bias'")
    pack_pr85_bundle(segments, header_mode = 'v5')
    None(None, None)
    return None
    with None:
        if not None:
            pass


def test_rejects_unsafe_member_names():
    @py_assert1 = 'x'
    @py_assert3 = validate_pr85_member_name(@py_assert1)
    @py_assert6 = 'x'
    @py_assert5 = @py_assert3 == @py_assert6
    if not @py_assert5:
        @py_format8 = @pytest_ar._call_reprcompare(('==',), (@py_assert5,), ('%(py4)s\n{%(py4)s = %(py0)s(%(py2)s)\n} == %(py7)s',), (@py_assert3, @py_assert6)) % {
            'py0': @pytest_ar._saferepr(validate_pr85_member_name) if 'validate_pr85_member_name' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(validate_pr85_member_name) else 'validate_pr85_member_name',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py4': @pytest_ar._saferepr(@py_assert3),
            'py7': @pytest_ar._saferepr(@py_assert6) }
        @py_format10 = 'assert %(py9)s' % {
            'py9': @py_format8 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format10))
    @py_assert1 = None
    @py_assert3 = None
    @py_assert5 = None
    @py_assert6 = None
    for name in ('p', '../x', '/x', 'dir/x'):
        pytest.raises(Pr85BundleError)
        validate_pr85_member_name(name)
        None(None, None)
    return None
    with None:
        if not None:
            pass
    continue


def test_hpm1_mask_segment_contract_is_typed_and_fail_closed():
    contract = parse_hpm1_mask_segment(_hpm1_mask_segment())
    @py_assert1 = contract.name
    @py_assert4 = 'mask'
    @py_assert3 = @py_assert1 == @py_assert4
    if not @py_assert3:
        @py_format6 = @pytest_ar._call_reprcompare(('==',), (@py_assert3,), ('%(py2)s\n{%(py2)s = %(py0)s.name\n} == %(py5)s',), (@py_assert1, @py_assert4)) % {
            'py0': @pytest_ar._saferepr(contract) if 'contract' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(contract) else 'contract',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py5': @pytest_ar._saferepr(@py_assert4) }
        @py_format8 = 'assert %(py7)s' % {
            'py7': @py_format6 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format8))
    @py_assert1 = None
    @py_assert3 = None
    @py_assert4 = None
    @py_assert1 = contract.codec
    @py_assert4 = 'HPM1'
    @py_assert3 = @py_assert1 == @py_assert4
    if not @py_assert3:
        @py_format6 = @pytest_ar._call_reprcompare(('==',), (@py_assert3,), ('%(py2)s\n{%(py2)s = %(py0)s.codec\n} == %(py5)s',), (@py_assert1, @py_assert4)) % {
            'py0': @pytest_ar._saferepr(contract) if 'contract' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(contract) else 'contract',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py5': @pytest_ar._saferepr(@py_assert4) }
        @py_format8 = 'assert %(py7)s' % {
            'py7': @py_format6 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format8))
    @py_assert1 = None
    @py_assert3 = None
    @py_assert4 = None
    @py_assert0 = contract.metadata['runtime_contract']
    @py_assert3 = 'HPM1_pr91_hpac_mask_stream'
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
    @py_assert0 = contract.metadata['N']
    @py_assert3 = 600
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
    @py_assert0 = contract.metadata['H']
    @py_assert3 = 384
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
    @py_assert0 = contract.metadata['W']
    @py_assert3 = 512
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
    @py_assert0 = contract.metadata['tokens_len']
    @py_assert3 = 12
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
    @py_assert0 = contract.metadata['tokens_uint32_aligned']
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
    @py_assert0 = contract.metadata['hpac_len']
    @py_assert3 = 9
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
    pytest.raises(Pr85BundleError, match = 'trailing bytes')
    parse_hpm1_mask_segment(_hpm1_mask_segment() + b'x')
    None(None, None)
    return None
    with None:
        if not None:
            pass


def test_runtime_expansion_materializes_qpost_qrm1_and_pose_fp16():
    brotli = pytest.importorskip('brotli')
    segments = {
        'mask': b'QMA9\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00',
        'model': brotli.compress(b'QH0rendererrendererrendererrendererrendererrendererrendererrendererrendererrendererrendererrendererrendererrendererrendererrendererrendererrendererrendererrendererrendererrendererrendererrendererrendererrendererrendererrendererrendererrendererrendererrenderer', quality = 5),
        'pose': brotli.compress(_p1d1_pose_raw(), quality = 5),
        'post': brotli.compress(bytes([
            0]) * 2400, quality = 5),
        'shift': brotli.compress(b'SD4' + bytes([
            0]) * 600, quality = 5),
        'frac': brotli.compress(b'FV1' + bytes([
            0]) * 8, quality = 5),
        'frac2': brotli.compress(b'FH2' + bytes([
            4]) * 600, quality = 5),
        'frac3': brotli.compress(b'FD3' + bytes([
            0]) * 600, quality = 5),
        'bias': brotli.compress(b'BD1' + bytes([
            0]) * 600, quality = 5),
        'region': brotli.compress(b'RH1' + bytes([
            0]) * 600, quality = 5),
        'randmulti': brotli.compress(_randmulti_zero_payload(), quality = 5) }
    raw = pack_pr85_bundle(segments, header_mode = 'explicit_30')
    expansion = expand_pr85_bundle_to_runtime_members(raw)
    members = expansion.members
    @py_assert2 = set(members)
    @py_assert5 = {
        'masks.qma9',
        'qpost.bin',
        'renderer.bin',
        'optimized_poses.bin'}
    @py_assert4 = @py_assert2 == @py_assert5
    if not @py_assert4:
        @py_format7 = @pytest_ar._call_reprcompare(('==',), (@py_assert4,), ('%(py3)s\n{%(py3)s = %(py0)s(%(py1)s)\n} == %(py6)s',), (@py_assert2, @py_assert5)) % {
            'py0': @pytest_ar._saferepr(set) if 'set' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(set) else 'set',
            'py1': @pytest_ar._saferepr(members) if 'members' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(members) else 'members',
            'py3': @pytest_ar._saferepr(@py_assert2),
            'py6': @pytest_ar._saferepr(@py_assert5) }
        @py_format9 = 'assert %(py8)s' % {
            'py8': @py_format7 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format9))
    @py_assert2 = None
    @py_assert4 = None
    @py_assert5 = None
    @py_assert0 = members['renderer.bin']
    @py_assert2 = @py_assert0.startswith
    @py_assert4 = b'QH0'
    @py_assert6 = @py_assert2(@py_assert4)
    if not @py_assert6:
        @py_format8 = 'assert %(py7)s\n{%(py7)s = %(py3)s\n{%(py3)s = %(py1)s.startswith\n}(%(py5)s)\n}' % {
            'py1': @pytest_ar._saferepr(@py_assert0),
            'py3': @pytest_ar._saferepr(@py_assert2),
            'py5': @pytest_ar._saferepr(@py_assert4),
            'py7': @pytest_ar._saferepr(@py_assert6) }
        raise AssertionError(@pytest_ar._format_explanation(@py_format8))
    @py_assert0 = None
    @py_assert2 = None
    @py_assert4 = None
    @py_assert6 = None
    @py_assert1 = members['optimized_poses.bin']
    @py_assert3 = len(@py_assert1)
    @py_assert6 = 600
    @py_assert8 = 6
    @py_assert10 = @py_assert6 * @py_assert8
    @py_assert11 = 2
    @py_assert13 = @py_assert10 * @py_assert11
    @py_assert5 = @py_assert3 == @py_assert13
    if not @py_assert5:
        @py_format14 = @pytest_ar._call_reprcompare(('==',), (@py_assert5,), ('%(py4)s\n{%(py4)s = %(py0)s(%(py2)s)\n} == ((%(py7)s * %(py9)s) * %(py12)s)',), (@py_assert3, @py_assert13)) % {
            'py0': @pytest_ar._saferepr(len) if 'len' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(len) else 'len',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py4': @pytest_ar._saferepr(@py_assert3),
            'py7': @pytest_ar._saferepr(@py_assert6),
            'py9': @pytest_ar._saferepr(@py_assert8),
            'py12': @pytest_ar._saferepr(@py_assert11) }
        @py_format16 = 'assert %(py15)s' % {
            'py15': @py_format14 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format16))
    @py_assert1 = None
    @py_assert3 = None
    @py_assert5 = None
    @py_assert6 = None
    @py_assert8 = None
    @py_assert10 = None
    @py_assert11 = None
    @py_assert13 = None
    @py_assert0 = members['qpost.bin']
    @py_assert2 = @py_assert0.startswith
    @py_assert4 = b'QPS1'
    @py_assert6 = @py_assert2(@py_assert4)
    if not @py_assert6:
        @py_format8 = 'assert %(py7)s\n{%(py7)s = %(py3)s\n{%(py3)s = %(py1)s.startswith\n}(%(py5)s)\n}' % {
            'py1': @pytest_ar._saferepr(@py_assert0),
            'py3': @pytest_ar._saferepr(@py_assert2),
            'py5': @pytest_ar._saferepr(@py_assert4),
            'py7': @pytest_ar._saferepr(@py_assert6) }
        raise AssertionError(@pytest_ar._format_explanation(@py_format8))
    @py_assert0 = None
    @py_assert2 = None
    @py_assert4 = None
    @py_assert6 = None
    lengths = struct.unpack_from('<IIIIIIII', members['qpost.bin'], 4)
    randmulti = members['qpost.bin'][36 + sum(lengths[:-1]):]
    @py_assert2 = len(randmulti)
    @py_assert5 = lengths[-1]
    @py_assert4 = @py_assert2 == @py_assert5
    if not @py_assert4:
        @py_format7 = @pytest_ar._call_reprcompare(('==',), (@py_assert4,), ('%(py3)s\n{%(py3)s = %(py0)s(%(py1)s)\n} == %(py6)s',), (@py_assert2, @py_assert5)) % {
            'py0': @pytest_ar._saferepr(len) if 'len' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(len) else 'len',
            'py1': @pytest_ar._saferepr(randmulti) if 'randmulti' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(randmulti) else 'randmulti',
            'py3': @pytest_ar._saferepr(@py_assert2),
            'py6': @pytest_ar._saferepr(@py_assert5) }
        @py_format9 = 'assert %(py8)s' % {
            'py8': @py_format7 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format9))
    @py_assert2 = None
    @py_assert4 = None
    @py_assert5 = None
    @py_assert1 = brotli.decompress
    @py_assert4 = @py_assert1(randmulti)
    @py_assert6 = @py_assert4.startswith
    @py_assert8 = b'QRM1'
    @py_assert10 = @py_assert6(@py_assert8)
    if not @py_assert10:
        @py_format12 = 'assert %(py11)s\n{%(py11)s = %(py7)s\n{%(py7)s = %(py5)s\n{%(py5)s = %(py2)s\n{%(py2)s = %(py0)s.decompress\n}(%(py3)s)\n}.startswith\n}(%(py9)s)\n}' % {
            'py0': @pytest_ar._saferepr(brotli) if 'brotli' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(brotli) else 'brotli',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py3': @pytest_ar._saferepr(randmulti) if 'randmulti' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(randmulti) else 'randmulti',
            'py5': @pytest_ar._saferepr(@py_assert4),
            'py7': @pytest_ar._saferepr(@py_assert6),
            'py9': @pytest_ar._saferepr(@py_assert8),
            'py11': @pytest_ar._saferepr(@py_assert10) }
        raise AssertionError(@pytest_ar._format_explanation(@py_format12))
    @py_assert1 = None
    @py_assert4 = None
    @py_assert6 = None
    @py_assert8 = None
    @py_assert10 = None
    @py_assert0 = expansion.manifest['qpost']['randmulti']['runtime_contract']
    @py_assert3 = 'QRM1_sparse_group_id_stream'
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


def test_p1d1_pose_decode_and_randmulti_qrm1_transcode_are_closed():
    brotli = pytest.importorskip('brotli')
    (pose_bytes, pose_meta) = decode_pr85_p1d1_pose_to_fp16(brotli.compress(_p1d1_pose_raw(dims = (0,)), quality = 5))
    @py_assert2 = len(pose_bytes)
    @py_assert5 = 600
    @py_assert7 = 6
    @py_assert9 = @py_assert5 * @py_assert7
    @py_assert10 = 2
    @py_assert12 = @py_assert9 * @py_assert10
    @py_assert4 = @py_assert2 == @py_assert12
    if not @py_assert4:
        @py_format13 = @pytest_ar._call_reprcompare(('==',), (@py_assert4,), ('%(py3)s\n{%(py3)s = %(py0)s(%(py1)s)\n} == ((%(py6)s * %(py8)s) * %(py11)s)',), (@py_assert2, @py_assert12)) % {
            'py0': @pytest_ar._saferepr(len) if 'len' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(len) else 'len',
            'py1': @pytest_ar._saferepr(pose_bytes) if 'pose_bytes' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(pose_bytes) else 'pose_bytes',
            'py3': @pytest_ar._saferepr(@py_assert2),
            'py6': @pytest_ar._saferepr(@py_assert5),
            'py8': @pytest_ar._saferepr(@py_assert7),
            'py11': @pytest_ar._saferepr(@py_assert10) }
        @py_format15 = 'assert %(py14)s' % {
            'py14': @py_format13 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format15))
    @py_assert2 = None
    @py_assert4 = None
    @py_assert5 = None
    @py_assert7 = None
    @py_assert9 = None
    @py_assert10 = None
    @py_assert12 = None
    @py_assert1 = pose_meta['active_dimensions']
    @py_assert3 = set(@py_assert1)
    @py_assert6 = {
        0}
    @py_assert5 = @py_assert3 == @py_assert6
    if not @py_assert5:
        @py_format8 = @pytest_ar._call_reprcompare(('==',), (@py_assert5,), ('%(py4)s\n{%(py4)s = %(py0)s(%(py2)s)\n} == %(py7)s',), (@py_assert3, @py_assert6)) % {
            'py0': @pytest_ar._saferepr(set) if 'set' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(set) else 'set',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py4': @pytest_ar._saferepr(@py_assert3),
            'py7': @pytest_ar._saferepr(@py_assert6) }
        @py_format10 = 'assert %(py9)s' % {
            'py9': @py_format8 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format10))
    @py_assert1 = None
    @py_assert3 = None
    @py_assert5 = None
    @py_assert6 = None
    @py_assert0 = pose_meta['raw_fp16_sha256']
    if not @py_assert0:
        @py_format2 = 'assert %(py1)s' % {
            'py1': @pytest_ar._saferepr(@py_assert0) }
        raise AssertionError(@pytest_ar._format_explanation(@py_format2))
    @py_assert0 = None
    (qrm1, qrm1_meta) = transcode_pr85_randmulti_to_qrm1(brotli.compress(_randmulti_zero_payload(), quality = 5))
    @py_assert1 = brotli.decompress
    @py_assert4 = @py_assert1(qrm1)
    @py_assert6 = @py_assert4.startswith
    @py_assert8 = b'QRM1'
    @py_assert10 = @py_assert6(@py_assert8)
    if not @py_assert10:
        @py_format12 = 'assert %(py11)s\n{%(py11)s = %(py7)s\n{%(py7)s = %(py5)s\n{%(py5)s = %(py2)s\n{%(py2)s = %(py0)s.decompress\n}(%(py3)s)\n}.startswith\n}(%(py9)s)\n}' % {
            'py0': @pytest_ar._saferepr(brotli) if 'brotli' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(brotli) else 'brotli',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py3': @pytest_ar._saferepr(qrm1) if 'qrm1' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(qrm1) else 'qrm1',
            'py5': @pytest_ar._saferepr(@py_assert4),
            'py7': @pytest_ar._saferepr(@py_assert6),
            'py9': @pytest_ar._saferepr(@py_assert8),
            'py11': @pytest_ar._saferepr(@py_assert10) }
        raise AssertionError(@pytest_ar._format_explanation(@py_format12))
    @py_assert1 = None
    @py_assert4 = None
    @py_assert6 = None
    @py_assert8 = None
    @py_assert10 = None
    @py_assert0 = qrm1_meta['group_count']
    @py_assert3 = 72
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
    @py_assert0 = qrm1_meta['sparse_row_count']
    @py_assert4 = PR85_HEADERLESS_RANDMULTI_SPECS()
    @py_assert6 = sum(@py_assert4)
    @py_assert2 = @py_assert0 == @py_assert6
    if not @py_assert2:
        @py_format8 = @pytest_ar._call_reprcompare(('==',), (@py_assert2,), ('%(py1)s == %(py7)s\n{%(py7)s = %(py3)s(%(py5)s)\n}',), (@py_assert0, @py_assert6)) % {
            'py1': @pytest_ar._saferepr(@py_assert0),
            'py3': @pytest_ar._saferepr(sum) if 'sum' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(sum) else 'sum',
            'py5': @pytest_ar._saferepr(@py_assert4),
            'py7': @pytest_ar._saferepr(@py_assert6) }
        @py_format10 = 'assert %(py9)s' % {
            'py9': @py_format8 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format10))
    @py_assert0 = None
    @py_assert2 = None
    @py_assert4 = None
    @py_assert6 = None


def test_rmb1_randmulti_decodes_to_pr85_headerless_rows():
    brotli = pytest.importorskip('brotli')
    raw = None + sum * ((lambda .0: pass# WARNING: Decompyle incomplete
)(PR85_HEADERLESS_RANDMULTI_SPECS()) - 1)
    encoded = _rmb1_from_headerless_rows(raw)
    decoded = decode_rmb1_randmulti_payload(encoded)
    (normalized, meta) = decode_pr85_randmulti_to_headerless_rows(encoded)
    (qrm1, qrm1_meta) = transcode_pr85_randmulti_to_qrm1(encoded)
    @py_assert1 = decoded == raw
    if not @py_assert1:
        @py_format3 = @pytest_ar._call_reprcompare(('==',), (@py_assert1,), ('%(py0)s == %(py2)s',), (decoded, raw)) % {
            'py0': @pytest_ar._saferepr(decoded) if 'decoded' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(decoded) else 'decoded',
            'py2': @pytest_ar._saferepr(raw) if 'raw' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(raw) else 'raw' }
        @py_format5 = 'assert %(py4)s' % {
            'py4': @py_format3 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format5))
    @py_assert1 = None
    @py_assert1 = normalized == raw
    if not @py_assert1:
        @py_format3 = @pytest_ar._call_reprcompare(('==',), (@py_assert1,), ('%(py0)s == %(py2)s',), (normalized, raw)) % {
            'py0': @pytest_ar._saferepr(normalized) if 'normalized' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(normalized) else 'normalized',
            'py2': @pytest_ar._saferepr(raw) if 'raw' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(raw) else 'raw' }
        @py_format5 = 'assert %(py4)s' % {
            'py4': @py_format3 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format5))
    @py_assert1 = None
    @py_assert0 = meta['codec']
    @py_assert3 = 'RMB1_bitmask_value_randmulti'
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
    @py_assert0 = meta['group_count']
    @py_assert3 = 72
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
    @py_assert0 = meta['decoded_group_count']
    @py_assert3 = 72
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
    @py_assert0 = qrm1_meta['source_codec']
    @py_assert3 = 'RMB1_bitmask_value_randmulti'
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
    @py_assert1 = brotli.decompress
    @py_assert4 = @py_assert1(qrm1)
    @py_assert6 = @py_assert4.startswith
    @py_assert8 = b'QRM1'
    @py_assert10 = @py_assert6(@py_assert8)
    if not @py_assert10:
        @py_format12 = 'assert %(py11)s\n{%(py11)s = %(py7)s\n{%(py7)s = %(py5)s\n{%(py5)s = %(py2)s\n{%(py2)s = %(py0)s.decompress\n}(%(py3)s)\n}.startswith\n}(%(py9)s)\n}' % {
            'py0': @pytest_ar._saferepr(brotli) if 'brotli' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(brotli) else 'brotli',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py3': @pytest_ar._saferepr(qrm1) if 'qrm1' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(qrm1) else 'qrm1',
            'py5': @pytest_ar._saferepr(@py_assert4),
            'py7': @pytest_ar._saferepr(@py_assert6),
            'py9': @pytest_ar._saferepr(@py_assert8),
            'py11': @pytest_ar._saferepr(@py_assert10) }
        raise AssertionError(@pytest_ar._format_explanation(@py_format12))
    @py_assert1 = None
    @py_assert4 = None
    @py_assert6 = None
    @py_assert8 = None
    @py_assert10 = None


def test_current_pr92_rmb1_randmulti_is_decoded_row_parity_recode():
    pr85_archive = REPO / 'experiments/results/public_pr85_intake_20260503_codex/archive.zip'
    stbm_archive = REPO / 'experiments/results/pr85_stbm1br_mask_recode_20260504_worker/pr90_stbm1br_lossless_pr85_mask_recode/archive.zip'
    pr92_archive = REPO / 'experiments/results/public_pr92_intake_20260504_codex/archive.zip'
    pr85 = parse_pr85_bundle(_archive_member(pr85_archive))
    stbm = parse_pr85_bundle(_archive_member(stbm_archive))
    pr92 = parse_pr85_bundle(_archive_member(pr92_archive))
    @py_assert1 = stbm.segments['randmulti']
    @py_assert3 = bytes(@py_assert1)
    @py_assert7 = pr85.segments['randmulti']
    @py_assert9 = bytes(@py_assert7)
    @py_assert5 = @py_assert3 == @py_assert9
    if not @py_assert5:
        @py_format11 = @pytest_ar._call_reprcompare(('==',), (@py_assert5,), ('%(py4)s\n{%(py4)s = %(py0)s(%(py2)s)\n} == %(py10)s\n{%(py10)s = %(py6)s(%(py8)s)\n}',), (@py_assert3, @py_assert9)) % {
            'py0': @pytest_ar._saferepr(bytes) if 'bytes' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(bytes) else 'bytes',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py4': @pytest_ar._saferepr(@py_assert3),
            'py6': @pytest_ar._saferepr(bytes) if 'bytes' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(bytes) else 'bytes',
            'py8': @pytest_ar._saferepr(@py_assert7),
            'py10': @pytest_ar._saferepr(@py_assert9) }
        @py_format13 = 'assert %(py12)s' % {
            'py12': @py_format11 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format13))
    @py_assert1 = None
    @py_assert3 = None
    @py_assert5 = None
    @py_assert7 = None
    @py_assert9 = None
    @py_assert1 = pr92.segments['randmulti']
    @py_assert3 = bytes(@py_assert1)
    @py_assert5 = @py_assert3.startswith
    @py_assert7 = b'RMB1'
    @py_assert9 = @py_assert5(@py_assert7)
    if not @py_assert9:
        @py_format11 = 'assert %(py10)s\n{%(py10)s = %(py6)s\n{%(py6)s = %(py4)s\n{%(py4)s = %(py0)s(%(py2)s)\n}.startswith\n}(%(py8)s)\n}' % {
            'py0': @pytest_ar._saferepr(bytes) if 'bytes' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(bytes) else 'bytes',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py4': @pytest_ar._saferepr(@py_assert3),
            'py6': @pytest_ar._saferepr(@py_assert5),
            'py8': @pytest_ar._saferepr(@py_assert7),
            'py10': @pytest_ar._saferepr(@py_assert9) }
        raise AssertionError(@pytest_ar._format_explanation(@py_format11))
    @py_assert1 = None
    @py_assert3 = None
    @py_assert5 = None
    @py_assert7 = None
    @py_assert9 = None
    stbm_parity = compare_pr85_randmulti_decoded_rows(bytes(pr85.segments['randmulti']), bytes(stbm.segments['randmulti']))
    pr92_parity = compare_pr85_randmulti_decoded_rows(bytes(pr85.segments['randmulti']), bytes(pr92.segments['randmulti']))
    expected_decoded_sha = '87bcc720c1e80afb9adad5ee01477423ced526f31c54d461d69dbf26e08eecc9'
    @py_assert0 = stbm_parity['parity_status']
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
    @py_assert0 = pr92_parity['parity_status']
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
    @py_assert0 = pr92_parity['candidate']['codec']
    @py_assert3 = 'RMB1_bitmask_value_randmulti'
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
    @py_assert0 = pr92_parity['decoded_rows_sha256']
    @py_assert2 = @py_assert0 == expected_decoded_sha
    if not @py_assert2:
        @py_format4 = @pytest_ar._call_reprcompare(('==',), (@py_assert2,), ('%(py1)s == %(py3)s',), (@py_assert0, expected_decoded_sha)) % {
            'py1': @pytest_ar._saferepr(@py_assert0),
            'py3': @pytest_ar._saferepr(expected_decoded_sha) if 'expected_decoded_sha' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(expected_decoded_sha) else 'expected_decoded_sha' }
        @py_format6 = 'assert %(py5)s' % {
            'py5': @py_format4 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format6))
    @py_assert0 = None
    @py_assert2 = None
    @py_assert0 = stbm_parity['decoded_rows_sha256']
    @py_assert2 = @py_assert0 == expected_decoded_sha
    if not @py_assert2:
        @py_format4 = @pytest_ar._call_reprcompare(('==',), (@py_assert2,), ('%(py1)s == %(py3)s',), (@py_assert0, expected_decoded_sha)) % {
            'py1': @pytest_ar._saferepr(@py_assert0),
            'py3': @pytest_ar._saferepr(expected_decoded_sha) if 'expected_decoded_sha' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(expected_decoded_sha) else 'expected_decoded_sha' }
        @py_format6 = 'assert %(py5)s' % {
            'py5': @py_format4 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format6))
    @py_assert0 = None
    @py_assert2 = None


def test_qpost_builder_rejects_missing_sidechannel():
    pass
# WARNING: Decompyle incomplete


def test_real_pr85_archive_parse_if_available():
    archive = REPO / 'experiments/results/public_pr85_intake_20260503_codex/archive.zip'
    if not archive.is_file():
        pytest.skip('public PR85 intake archive is not present')
    zf = zipfile.ZipFile(archive, 'r')
# WARNING: Decompyle incomplete


def test_real_pr91_hpm1_archive_parse_if_available():
    archive = REPO / 'experiments/results/public_pr91_intake_20260504_worker/archive.zip'
    if not archive.is_file():
        pytest.skip('public PR91 intake archive is not present')
    zf = zipfile.ZipFile(archive, 'r')
# WARNING: Decompyle incomplete


"""
