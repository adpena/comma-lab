# pyc-recovery: STUB unreconstructible -- see .recovery_spec.json for dis() ground-truth
# pycdc could not produce parseable output; raw decompiled text preserved in _PYCDC_PARTIAL_OUTPUT below.
"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``5:20: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``test_quantizr_torch_fp4_codec.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'src/tac/tests/test_quantizr_torch_fp4_codec.py'
__recovery_spec__ = 'test_quantizr_torch_fp4_codec.recovery_spec.json'
__recovery_ast_error__ = '5:20: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: test_quantizr_torch_fp4_codec.cpython-312-pytest-9.0.3.pyc (Python 3.12)

from __future__ import annotations
import builtins as @py_builtins

rewrite
import importlib.util = import _pytest.assertion.rewrite, assertion
import io
import zipfile
from pathlib import Path
import brotli
import pytest
import torch
from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer
from tac.quantizr_qzs3_codec import decode_qzs3_state_dict
from tac.quantizr_torch_fp4_codec import decode_torch_fp4_payload, encode_torch_fp4_state_dict, is_torch_fp4_payload, load_torch_fp4_bytes
REPO = Path(__file__).resolve().parents[3]
PR63_ARCHIVE = REPO / 'experiments/results/top_submission_current_floor_20260501/external_archives/pr63_qpose14_archive.zip'
PR63_UNPACKED_RUNTIME = REPO / 'experiments/results/top_submission_reverse_roundtrip_20260501/' / 'public_pr63_unpacked_runtime_20260501T2158Z'

def _load_builder():
    path = REPO / 'experiments/repack_quantizr_faithful_qzs3_archive.py'
    spec = importlib.util.spec_from_file_location('_qfaithful_repack_builder_test', path)
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


def test_torch_fp4_codec_round_trips_into_jointframegenerator():
    torch.manual_seed(0)
    source = build_quantizr_faithful_renderer().eval()
    payload = encode_torch_fp4_state_dict(source)
    decoded = load_torch_fp4_bytes(payload, device = 'cpu')
    @py_assert1 = decoded.parameters()()
    @py_assert3 = sum(@py_assert1)
    @py_assert6 = 87836
    @py_assert5 = @py_assert3 == @py_assert6
    if not @py_assert5:
        @py_format8 = @pytest_ar._call_reprcompare(('==',), (@py_assert5,), ('%(py4)s\n{%(py4)s = %(py0)s(%(py2)s)\n} == %(py7)s',), (@py_assert3, @py_assert6)) % {
            'py0': @pytest_ar._saferepr(sum) if 'sum' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(sum) else 'sum',
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
    mask = torch.zeros(1, 384, 512, dtype = torch.long)
    pose = torch.zeros(1, 6)
    torch.no_grad()
    (f1, f2) = decoded(mask, pose)
    None(None, None)
# WARNING: Decompyle incomplete


def test_torch_fp4_codec_is_deterministic():
    torch.manual_seed(123)
    source = build_quantizr_faithful_renderer().eval()
    first = encode_torch_fp4_state_dict(source)
    second = encode_torch_fp4_state_dict(source)
    @py_assert1 = first == second
    if not @py_assert1:
        @py_format3 = @pytest_ar._call_reprcompare(('==',), (@py_assert1,), ('%(py0)s == %(py2)s',), (first, second)) % {
            'py0': @pytest_ar._saferepr(first) if 'first' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(first) else 'first',
            'py2': @pytest_ar._saferepr(second) if 'second' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(second) else 'second' }
        @py_format5 = 'assert %(py4)s' % {
            'py4': @py_format3 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format5))
    @py_assert1 = None


def test_torch_fp4_payload_shape_detector_and_state_decode():
    source = build_quantizr_faithful_renderer().eval()
    payload = torch.load(io.BytesIO(encode_torch_fp4_state_dict(source)), map_location = 'cpu', weights_only = False)
    @py_assert2 = is_torch_fp4_payload(payload)
    if not @py_assert2:
        @py_format4 = 'assert %(py3)s\n{%(py3)s = %(py0)s(%(py1)s)\n}' % {
            'py0': @pytest_ar._saferepr(is_torch_fp4_payload) if 'is_torch_fp4_payload' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(is_torch_fp4_payload) else 'is_torch_fp4_payload',
            'py1': @pytest_ar._saferepr(payload) if 'payload' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(payload) else 'payload',
            'py3': @pytest_ar._saferepr(@py_assert2) }
        raise AssertionError(@pytest_ar._format_explanation(@py_format4))
    @py_assert2 = None
    state = decode_torch_fp4_payload(payload, device = 'cpu')
    fresh = build_quantizr_faithful_renderer()
    fresh.load_state_dict(state, strict = True)


def test_loader_decodes_public_pr63_model_payload_when_available():
    if not PR63_ARCHIVE.exists():
        pytest.skip(f'''public PR63 archive fixture missing: {PR63_ARCHIVE}''')
    zf = zipfile.ZipFile(PR63_ARCHIVE)
    packed = zf.read('p')
    None(None, None)
# WARNING: Decompyle incomplete


def test_repack_builder_emits_torch_fp4_archive_from_qfai(tmp_path = None):
    save_qfai = save_qfai
    import tac.quantizr_faithful_export
    RENDERER_COMPACT_MANIFEST = RENDERER_COMPACT_MANIFEST
    build_submission_archive = build_submission_archive
    import tac.submission_archive
    builder = _load_builder()
    model = build_quantizr_faithful_renderer().eval()
    renderer = tmp_path / 'renderer.bin'
    save_qfai(model, renderer)
    masks = tmp_path / 'masks.mkv'
    masks.write_bytes(b'maskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmaskmask')
    poses = tmp_path / 'optimized_poses.bin'
    poses.write_bytes(torch.zeros(600, 6, dtype = torch.float16).numpy().tobytes())
    source = tmp_path / 'source.zip'
    build_submission_archive(source, renderer_bin = renderer, masks_mkv = masks, optimized_poses_bin = poses, manifest = RENDERER_COMPACT_MANIFEST, validate = False)
    out = tmp_path / 'torchfp4.zip'
    provenance = builder.build_archive(source, out, renderer_codec = builder.RENDERER_CODEC_TORCH_FP4)
    @py_assert0 = provenance['renderer']['renderer_codec']
    @py_assert4 = builder.RENDERER_CODEC_TORCH_FP4
    @py_assert2 = @py_assert0 == @py_assert4
    if not @py_assert2:
        @py_format6 = @pytest_ar._call_reprcompare(('==',), (@py_assert2,), ('%(py1)s == %(py5)s\n{%(py5)s = %(py3)s.RENDERER_CODEC_TORCH_FP4\n}',), (@py_assert0, @py_assert4)) % {
            'py1': @pytest_ar._saferepr(@py_assert0),
            'py3': @pytest_ar._saferepr(builder) if 'builder' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(builder) else 'builder',
            'py5': @pytest_ar._saferepr(@py_assert4) }
        @py_format8 = 'assert %(py7)s' % {
            'py7': @py_format6 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format8))
    @py_assert0 = None
    @py_assert2 = None
    @py_assert4 = None
    @py_assert0 = provenance['renderer']['source_renderer_format']
    @py_assert3 = 'QFAI'
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
    zf = zipfile.ZipFile(out)
    raw = zf.read('renderer.bin')
    None(None, None)
# WARNING: Decompyle incomplete


def test_repack_builder_emits_qzs3_archive_from_public_pr63_torch_fp4(tmp_path = None):
    if not PR63_UNPACKED_RUNTIME.exists():
        pytest.skip(f'''public PR63 unpacked runtime fixture missing: {PR63_UNPACKED_RUNTIME}''')
    builder = _load_builder()
    source = tmp_path / 'source.zip'
    zf = zipfile.ZipFile(source, 'w', compression = zipfile.ZIP_DEFLATED, compresslevel = 9)
    for name in ('renderer.bin', 'masks.mkv', 'optimized_poses.bin'):
        info = zipfile.ZipInfo(name, date_time = (1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 27525120
        zf.writestr(info, (PR63_UNPACKED_RUNTIME / name).read_bytes())
    None(None, None)
    out = tmp_path / 'qzs3.zip'
    provenance = builder.build_archive(source, out, renderer_codec = builder.RENDERER_CODEC_QZS3)
    @py_assert0 = provenance['renderer']['renderer_codec']
    @py_assert4 = builder.RENDERER_CODEC_QZS3
    @py_assert2 = @py_assert0 == @py_assert4
    if not @py_assert2:
        @py_format6 = @pytest_ar._call_reprcompare(('==',), (@py_assert2,), ('%(py1)s == %(py5)s\n{%(py5)s = %(py3)s.RENDERER_CODEC_QZS3\n}',), (@py_assert0, @py_assert4)) % {
            'py1': @pytest_ar._saferepr(@py_assert0),
            'py3': @pytest_ar._saferepr(builder) if 'builder' in @py_builtins.locals() or @pytest_ar._should_repr_global_name(builder) else 'builder',
            'py5': @pytest_ar._saferepr(@py_assert4) }
        @py_format8 = 'assert %(py7)s' % {
            'py7': @py_format6 }
        raise AssertionError(@pytest_ar._format_explanation(@py_format8))
    @py_assert0 = None
    @py_assert2 = None
    @py_assert4 = None
    @py_assert0 = provenance['renderer']['source_renderer_format']
    @py_assert3 = 'Torch-FP4'
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
    zf = zipfile.ZipFile(out)
    raw = zf.read('renderer.bin')
    None(None, None)
# WARNING: Decompyle incomplete


"""
