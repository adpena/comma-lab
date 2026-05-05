"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``5:20: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``test_qh0_record_serializer.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'src/tac/tests/test_qh0_record_serializer.py'
__recovery_spec__ = 'test_qh0_record_serializer.recovery_spec.json'
__recovery_ast_error__ = '5:20: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: test_qh0_record_serializer.cpython-312-pytest-9.0.3.pyc (Python 3.12)

from __future__ import annotations
import builtins as @py_builtins

rewrite
import zipfile = import _pytest.assertion.rewrite, assertion
from pathlib import Path
import pytest
from experiments.build_pr85_qh0_serializer_candidates import build_candidates, runtime_compatibility
from tac.pr85_bundle import parse_pr85_bundle
from tac.qh0_record_serializer import QH0Record, build_serialized_variants, choose_byte_win_candidates, pack_hilo_fp4_bytes, prove_decoded_tensor_parity, serialize_records, split_even_odd_bytes, unpack_hilo_fp4_bytes, unsplit_even_odd_bytes
from tac.qh0_renderer_codec import QH0_MAGIC, QM0_MAGIC
brotli = pytest.importorskip('brotli')
REPO = Path(__file__).resolve().parents[3]

def test_low_level_qh0_splits_and_synthetic_record_serializer_are_deterministic():
    direct = bytes(range(17))
    split = split_even_odd_bytes(direct)
    @py_assert2 = unsplit_even_odd_bytes(split)
    @py_assert4 = @py_assert2 == direct
    if not @py_assert4:
        @py_format6 = @pytest_ar._call_reprcompare(('==',), (@py_assert4,), ('%(py3)s\n{%(py3)s = %(py0)s(%(py1)s)\n} == %(py5)s',), (@py_assert2, direct)) % {
            'py0': @pytest_ar._saferepr(unsplit_even_odd_bytes) if 'unsplit_even_odd_bytes' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(unsplit_even_odd_bytes) else 'unsplit_even_odd_bytes',
            'py1': @pytest_ar._saferepr(split) if 'split' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(split) else 'split',
            'py3': @pytest_ar._saferepr(@py_assert2),
            'py5': @pytest_ar._saferepr(direct) if 'direct' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(direct) else 'direct' }
        @py_format8 = 'assert %(py7)s' % {
            'py7': @py_format6 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format8))
    @py_assert2 = None
    @py_assert4 = None
    @py_assert2 = split_even_odd_bytes(direct)
    @py_assert4 = @py_assert2 == split
    if not @py_assert4:
        @py_format6 = @pytest_ar._call_reprcompare(('==',), (@py_assert4,), ('%(py3)s\n{%(py3)s = %(py0)s(%(py1)s)\n} == %(py5)s',), (@py_assert2, split)) % {
            'py0': @pytest_ar._saferepr(split_even_odd_bytes) if 'split_even_odd_bytes' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(split_even_odd_bytes) else 'split_even_odd_bytes',
            'py1': @pytest_ar._saferepr(direct) if 'direct' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(direct) else 'direct',
            'py3': @pytest_ar._saferepr(@py_assert2),
            'py5': @pytest_ar._saferepr(split) if 'split' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(split) else 'split' }
        @py_format8 = 'assert %(py7)s' % {
            'py7': @py_format6 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format8))
    @py_assert2 = None
    @py_assert4 = None
    packed = bytes([
        18,
        52,
        171,
        205])
    hilo = pack_hilo_fp4_bytes(packed)
    @py_assert4 = len(packed)
    @py_assert6 = unpack_hilo_fp4_bytes(hilo, @py_assert4)
    @py_assert8 = @py_assert6 == packed
    if not @py_assert8:
        @py_format10 = @pytest_ar._call_reprcompare(('==',), (@py_assert8,), ('%(py7)s\n{%(py7)s = %(py0)s(%(py1)s, %(py5)s\n{%(py5)s = %(py2)s(%(py3)s)\n})\n} == %(py9)s',), (@py_assert6, packed)) % {
            'py0': @pytest_ar._saferepr(unpack_hilo_fp4_bytes) if 'unpack_hilo_fp4_bytes' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(unpack_hilo_fp4_bytes) else 'unpack_hilo_fp4_bytes',
            'py1': @pytest_ar._saferepr(hilo) if 'hilo' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(hilo) else 'hilo',
            'py2': @pytest_ar._saferepr(len) if 'len' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(len) else 'len',
            'py3': @pytest_ar._saferepr(packed) if 'packed' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(packed) else 'packed',
            'py5': @pytest_ar._saferepr(@py_assert4),
            'py7': @pytest_ar._saferepr(@py_assert6),
            'py9': @pytest_ar._saferepr(packed) if 'packed' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(packed) else 'packed' }
        @py_format12 = 'assert %(py11)s' % {
            'py11': @py_format10 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format12))
    @py_assert4 = None
    @py_assert6 = None
    @py_assert8 = None
    @py_assert2 = pack_hilo_fp4_bytes(packed)
    @py_assert4 = @py_assert2 == hilo
    if not @py_assert4:
        @py_format6 = @pytest_ar._call_reprcompare(('==',), (@py_assert4,), ('%(py3)s\n{%(py3)s = %(py0)s(%(py1)s)\n} == %(py5)s',), (@py_assert2, hilo)) % {
            'py0': @pytest_ar._saferepr(pack_hilo_fp4_bytes) if 'pack_hilo_fp4_bytes' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(pack_hilo_fp4_bytes) else 'pack_hilo_fp4_bytes',
            'py1': @pytest_ar._saferepr(packed) if 'packed' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(packed) else 'packed',
            'py3': @pytest_ar._saferepr(@py_assert2),
            'py5': @pytest_ar._saferepr(hilo) if 'hilo' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(hilo) else 'hilo' }
        @py_format8 = 'assert %(py7)s' % {
            'py7': @py_format6 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format8))
    @py_assert2 = None
    @py_assert4 = None
    record = QH0Record(name = 'synthetic.weight', category = 'module_weight', record_kind = 'fp16', offset = 3, source_nbytes = 1 + len(direct), direct_record = b'\x00' + direct, qh0_record = b'\x00' + split, tensor_shape = (len(direct) // 2,), element_count = len(direct) // 2, kind_byte = 0)
    @py_assert1 = [
        record]
    @py_assert4 = serialize_records(@py_assert1, magic = QH0_MAGIC)
    @py_assert8 = b'\x00'
    @py_assert10 = QH0_MAGIC + @py_assert8
    @py_assert12 = @py_assert10 + split
    @py_assert6 = @py_assert4 == @py_assert12
    if not @py_assert6:
        @py_format13 = @pytest_ar._call_reprcompare(('==',), (@py_assert6,), ('%(py5)s\n{%(py5)s = %(py0)s(%(py2)s, magic=%(py3)s)\n} == ((%(py7)s + %(py9)s) + %(py11)s)',), (@py_assert4, @py_assert12)) % {
            'py0': @pytest_ar._saferepr(serialize_records) if 'serialize_records' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(serialize_records) else 'serialize_records',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py3': @pytest_ar._saferepr(QH0_MAGIC) if 'QH0_MAGIC' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(QH0_MAGIC) else 'QH0_MAGIC',
            'py5': @pytest_ar._saferepr(@py_assert4),
            'py7': @pytest_ar._saferepr(QH0_MAGIC) if 'QH0_MAGIC' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(QH0_MAGIC) else 'QH0_MAGIC',
            'py9': @pytest_ar._saferepr(@py_assert8),
            'py11': @pytest_ar._saferepr(split) if 'split' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(split) else 'split' }
        @py_format15 = 'assert %(py14)s' % {
            'py14': @py_format13 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format15))
    @py_assert1 = None
    @py_assert4 = None
    @py_assert6 = None
    @py_assert8 = None
    @py_assert10 = None
    @py_assert12 = None
    @py_assert1 = [
        record]
    @py_assert4 = serialize_records(@py_assert1, magic = QM0_MAGIC)
    @py_assert8 = b'\x00'
    @py_assert10 = QM0_MAGIC + @py_assert8
    @py_assert12 = @py_assert10 + direct
    @py_assert6 = @py_assert4 == @py_assert12
    if not @py_assert6:
        @py_format13 = @pytest_ar._call_reprcompare(('==',), (@py_assert6,), ('%(py5)s\n{%(py5)s = %(py0)s(%(py2)s, magic=%(py3)s)\n} == ((%(py7)s + %(py9)s) + %(py11)s)',), (@py_assert4, @py_assert12)) % {
            'py0': @pytest_ar._saferepr(serialize_records) if 'serialize_records' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(serialize_records) else 'serialize_records',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py3': @pytest_ar._saferepr(QM0_MAGIC) if 'QM0_MAGIC' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(QM0_MAGIC) else 'QM0_MAGIC',
            'py5': @pytest_ar._saferepr(@py_assert4),
            'py7': @pytest_ar._saferepr(QM0_MAGIC) if 'QM0_MAGIC' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(QM0_MAGIC) else 'QM0_MAGIC',
            'py9': @pytest_ar._saferepr(@py_assert8),
            'py11': @pytest_ar._saferepr(direct) if 'direct' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(direct) else 'direct' }
        @py_format15 = 'assert %(py14)s' % {
            'py14': @py_format13 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format15))
    @py_assert1 = None
    @py_assert4 = None
    @py_assert6 = None
    @py_assert8 = None
    @py_assert10 = None
    @py_assert12 = None
    @py_assert1 = [
        record]
    @py_assert4 = serialize_records(@py_assert1, magic = QM0_MAGIC)
    @py_assert8 = [
        record]
    @py_assert11 = serialize_records(@py_assert8, magic = QM0_MAGIC)
    @py_assert6 = @py_assert4 == @py_assert11
    if not @py_assert6:
        @py_format13 = @pytest_ar._call_reprcompare(('==',), (@py_assert6,), ('%(py5)s\n{%(py5)s = %(py0)s(%(py2)s, magic=%(py3)s)\n} == %(py12)s\n{%(py12)s = %(py7)s(%(py9)s, magic=%(py10)s)\n}',), (@py_assert4, @py_assert11)) % {
            'py0': @pytest_ar._saferepr(serialize_records) if 'serialize_records' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(serialize_records) else 'serialize_records',
            'py2': @pytest_ar._saferepr(@py_assert1),
            'py3': @pytest_ar._saferepr(QM0_MAGIC) if 'QM0_MAGIC' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(QM0_MAGIC) else 'QM0_MAGIC',
            'py5': @pytest_ar._saferepr(@py_assert4),
            'py7': @pytest_ar._saferepr(serialize_records) if 'serialize_records' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(serialize_records) else 'serialize_records',
            'py9': @pytest_ar._saferepr(@py_assert8),
            'py10': @pytest_ar._saferepr(QM0_MAGIC) if 'QM0_MAGIC' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(QM0_MAGIC) else 'QM0_MAGIC',
            'py12': @pytest_ar._saferepr(@py_assert11) }
        @py_format15 = 'assert %(py14)s' % {
            'py14': @py_format13 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format15))
    @py_assert1 = None
    @py_assert4 = None
    @py_assert6 = None
    @py_assert8 = None
    @py_assert11 = None


def test_byte_win_candidate_filter_keeps_only_runtime_compatible_wins():
    rows = [
        {
            'candidate_id': 'win_ok',
            'candidate_model_delta_bytes_vs_source': -3,
            'runtime_compatibility': {
                'runtime_can_decode_without_edits': True } },
        {
            'candidate_id': 'byte_negative',
            'candidate_model_delta_bytes_vs_source': 4,
            'runtime_compatibility': {
                'runtime_can_decode_without_edits': True } },
        {
            'candidate_id': 'win_runtime_blocked',
            'candidate_model_delta_bytes_vs_source': -8,
            'runtime_compatibility': {
                'runtime_can_decode_without_edits': False } }]
    selected = choose_byte_win_candidates(rows)
# WARNING: Decompyle incomplete


def test_runtime_compatibility_fails_closed_when_replay_loader_lacks_magic(tmp_path = None):
    replay = tmp_path / 'inflate.py'
    replay.write_text('def load_compact_archive_bundle(data_dir):\n    path = data_dir / "x"\ndef get_decoded_state_dict_custom(payload_data, device):\n    if payload_data[:3] == b"QH0":\n        return {}\n', encoding = 'utf-8')
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    (runtime / 'inflate_renderer.py').write_text('# QH0 only\n', encoding = 'utf-8')
    (runtime / 'unpack_renderer_payload.py').write_text('# no single x support\n', encoding = 'utf-8')
    compat = runtime_compatibility('QM0', replay_inflate_py = replay, robust_current_dir = runtime)
    @py_assert0 = compat['runtime_can_decode_without_edits']
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
    @py_assert0 = compat['dispatch_unlocked']
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
    @py_assert0 = compat['blocker_class']
    @py_assert3 = 'runtime_incompatibility'
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
    @py_assert0 = 'public_pr85_replay_missing_QM0_model_loader'
    @py_assert3 = compat['blockers']
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


def test_real_pr85_qh0_to_qm0_serializer_parity_and_local_smoke(tmp_path = None):
    archive = REPO / 'experiments/results/public_pr85_intake_20260503_codex/archive.zip'
    if not archive.is_file():
        pytest.skip('public PR85 intake archive is not present')
    zf = zipfile.ZipFile(archive, 'r')
    source_x = zf.read('x')
    None(None, None)
# WARNING: Decompyle incomplete


"""
