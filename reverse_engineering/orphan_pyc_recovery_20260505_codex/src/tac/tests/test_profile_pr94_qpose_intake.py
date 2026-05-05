"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``146:28: invalid decimal literal``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``test_profile_pr94_qpose_intake.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'src/tac/tests/test_profile_pr94_qpose_intake.py'
__recovery_spec__ = 'test_profile_pr94_qpose_intake.recovery_spec.json'
__recovery_ast_error__ = '146:28: invalid decimal literal'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: test_profile_pr94_qpose_intake.cpython-312-pytest-9.0.3.pyc (Python 3.12)

from __future__ import annotations
import builtins as @py_builtins

rewrite
import importlib.util = import _pytest.assertion.rewrite, assertion
import sys
from pathlib import Path
import brotli
REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / 'experiments/profile_pr94_qpose_intake.py'

def load_module():
    spec = importlib.util.spec_from_file_location('profile_pr94_qpose_intake', MODULE_PATH)
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


def uvarint(value = None):
    out = bytearray()
    byte = value & 127
    value >>= 7
    if value:
        out.append(byte | 128)
    else:
        out.append(byte)
        return bytes(out)


def zigzag(value = None):
    return value << 1 ^ value >> 31


def test_infer_pr94_fixed_range_layout_for_observed_payload_size():
    module = load_module()
    layout = module.infer_pr94_layout(b'\x00' * 276987)
    @py_assert1 = layout.payload_format
    @py_assert4 = 'pr94_fixed_range_qpose_tile_actions'
    @py_assert3 = @py_assert1 == @py_assert4
    if not @py_assert3:
        @py_format6 = @pytest_ar._call_reprcompare(('==',), (@py_assert3,), ('%(py2)s\n{%(py2)s = %(py0)s.payload_format\n} == %(py5)s',), (@py_assert1, @py_assert4)) % {
            'py0': @pytest_ar._saferepr(layout) if 'layout' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(layout) else 'layout',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py5': @pytest_ar._saferepr(@py_assert4) }
        @py_format8 = 'assert %(py7)s' % {
            'py7': @py_format6 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format8))
    @py_assert1 = None
    @py_assert3 = None
    @py_assert4 = None
    @py_assert1 = layout.mask_len
    @py_assert4 = 219472
    @py_assert3 = @py_assert1 == @py_assert4
    if not @py_assert3:
        @py_format6 = @pytest_ar._call_reprcompare(('==',), (@py_assert3,), ('%(py2)s\n{%(py2)s = %(py0)s.mask_len\n} == %(py5)s',), (@py_assert1, @py_assert4)) % {
            'py0': @pytest_ar._saferepr(layout) if 'layout' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(layout) else 'layout',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py5': @pytest_ar._saferepr(@py_assert4) }
        @py_format8 = 'assert %(py7)s' % {
            'py7': @py_format6 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format8))
    @py_assert1 = None
    @py_assert3 = None
    @py_assert4 = None
    @py_assert1 = layout.model_len
    @py_assert4 = 55756
    @py_assert3 = @py_assert1 == @py_assert4
    if not @py_assert3:
        @py_format6 = @pytest_ar._call_reprcompare(('==',), (@py_assert3,), ('%(py2)s\n{%(py2)s = %(py0)s.model_len\n} == %(py5)s',), (@py_assert1, @py_assert4)) % {
            'py0': @pytest_ar._saferepr(layout) if 'layout' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(layout) else 'layout',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py5': @pytest_ar._saferepr(@py_assert4) }
        @py_format8 = 'assert %(py7)s' % {
            'py7': @py_format6 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format8))
    @py_assert1 = None
    @py_assert3 = None
    @py_assert4 = None
    @py_assert1 = layout.actions_len
    @py_assert4 = 861
    @py_assert3 = @py_assert1 == @py_assert4
    if not @py_assert3:
        @py_format6 = @pytest_ar._call_reprcompare(('==',), (@py_assert3,), ('%(py2)s\n{%(py2)s = %(py0)s.actions_len\n} == %(py5)s',), (@py_assert1, @py_assert4)) % {
            'py0': @pytest_ar._saferepr(layout) if 'layout' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(layout) else 'layout',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py5': @pytest_ar._saferepr(@py_assert4) }
        @py_format8 = 'assert %(py7)s' % {
            'py7': @py_format6 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format8))
    @py_assert1 = None
    @py_assert3 = None
    @py_assert4 = None
    @py_assert1 = layout.pose_len
    @py_assert4 = 898
    @py_assert3 = @py_assert1 == @py_assert4
    if not @py_assert3:
        @py_format6 = @pytest_ar._call_reprcompare(('==',), (@py_assert3,), ('%(py2)s\n{%(py2)s = %(py0)s.pose_len\n} == %(py5)s',), (@py_assert1, @py_assert4)) % {
            'py0': @pytest_ar._saferepr(layout) if 'layout' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(layout) else 'layout',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py5': @pytest_ar._saferepr(@py_assert4) }
        @py_format8 = 'assert %(py7)s' % {
            'py7': @py_format6 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format8))
    @py_assert1 = None
    @py_assert3 = None
    @py_assert4 = None


def test_profile_payload_decodes_qp1_velocity_and_sg2_actions():
    module = load_module()
    mask = brotli.compress(b'mask-obu')
    model = brotli.compress(b'QZS3renderer')
    actions_raw = b'SG2' + uvarint(5) + uvarint(2) + uvarint(10) + bytes([
        1]) + uvarint(2) + bytes([
        3])
    actions = brotli.compress(actions_raw)
    pose_raw = b'QP1' + 100.to_bytes(2, 'little') + uvarint(zigzag(1)) + uvarint(zigzag(-2))
    pose = brotli.compress(pose_raw)
    payload = mask + model + actions + pose
    layout = module.SegmentLayout(payload_format = 'test', boundary_authority = 'unit_test', header_bytes = 0, mask_len = len(mask), model_len = len(model), actions_len = len(actions), pose_len = len(pose))
    profile = module.profile_payload(payload, layout = layout)
    @py_assert0 = profile['classification']['renderer_magic']
    @py_assert3 = 'QZS3'
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
    @py_assert0 = profile['classification']['qpose']['format']
    @py_assert3 = 'QP1_velocity_delta_varint'
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
    @py_assert0 = profile['classification']['qpose']['pose_rows']
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
    @py_assert0 = profile['classification']['qpose']['non_velocity_columns_fixed_zero']
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
    @py_assert0 = profile['classification']['tile_actions']['format']
    @py_assert3 = 'sg2_tile_group_varint'
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
    @py_assert0 = profile['classification']['tile_actions']['record_count']
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
    @py_assert0 = profile['classification']['tile_actions']['first_records']
    @py_assert3 = [
        {
            'frame': 10,
            'tile': 5,
            'action': 1 },
        {
            'frame': 12,
            'tile': 5,
            'action': 3 }]
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


"""
