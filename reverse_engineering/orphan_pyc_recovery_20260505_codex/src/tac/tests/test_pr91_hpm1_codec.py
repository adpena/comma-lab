# Source Generated with Decompyle++
# File: test_pr91_hpm1_codec.cpython-312.pyc (Python 3.12)

from __future__ import annotations
import io
import importlib.util as importlib
import json
import sys
import zipfile
from pathlib import Path
import numpy as np
import pytest
import torch
from tac.pr85_bundle import SEGMENT_ORDER, pack_pr85_bundle
from tac.pr86_hpac_codec import HPACMini, encode_tokens_hpac, sha256_bytes
from tac.pr91_hpm1_codec import DEFAULT_PR85_QMA9_TOKEN_SOURCE, DEFAULT_PR85_STBM_ARCHIVE, DEFAULT_PR91_ARCHIVE, Pr91Hpm1Error, analyze_pr91_hpm1_runtime_sources, build_hpm1_mask_segment, compare_hpm1_to_pr86_hpac_contract, extract_pr91_hpm1_payload, plan_pr91_hpm1_pr85_stbm_fusion, prototype_reencode_hpm1_from_raw_tokens, prototype_reencode_hpm1_residual_from_raw_tokens, raw_tokens_to_mod5_residual_symbols, reconstruct_raw_tokens_from_mod5_residual_symbols, run_pr91_hpm1_context_window_probe, run_pr91_hpm1_first_symbol_state_probe, run_pr91_hpm1_probability_variant_matrix, run_pr91_hpm1_reference_prefix_probe, run_pr91_hpm1_preflight, run_pr91_hpm1_stream_transform_probe, split_hpm1_mask_segment, validate_hpm1_static_contract
REPO = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO / 'experiments' / 'replay_pr91_hpm1_mask.py'

def _load_cli_script():
    spec = importlib.util.spec_from_file_location('replay_pr91_hpm1_mask_test', SCRIPT_PATH)
# WARNING: Decompyle incomplete


def _ppmd_torch(payload = None):
    pyppmd = pytest.importorskip('pyppmd')
    buf = io.BytesIO()
    torch.save(payload, buf)
    return pyppmd.compress(buf.getvalue(), max_order = 4, mem_size = 16777216)


def _synthetic_hpm1_segment():
    pytest.importorskip('constriction')
    model = HPACMini(num_pairs = 2, P = 2, delta = 1, ch = 4, d_film = 2, use_spm = False).eval()
    tokens = np.array([
        [
            [
                0,
                1,
                2,
                3],
            [
                4,
                0,
                1,
                2],
            [
                3,
                4,
                0,
                1],
            [
                2,
                3,
                4,
                0]],
        [
            [
                1,
                1,
                2,
                2],
            [
                3,
                3,
                4,
                4],
            [
                0,
                0,
                1,
                1],
            [
                2,
                2,
                3,
                3]]], dtype = np.uint8)
    (token_blob, _report) = encode_tokens_hpac(model, tokens, P = 2, delta = 1)
    hpac_ppmd = _ppmd_torch(model.state_dict())
    segment = build_hpm1_mask_segment(token_blob, hpac_ppmd, N = 2, H = 4, W = 4, P = 2, delta = 1, ch = 4, use_spm = False, hpac_d_film = 2)
    return (segment, tokens)


def _synthetic_archive(tmp_path = None):
    (segment, tokens) = _synthetic_hpm1_segment()
    archive = _synthetic_bundle_archive(tmp_path / 'archive.zip', segment)
    return (archive, tokens)


def _synthetic_bundle_archive(path = None, mask_segment = None, *, header_mode):
    pass
# WARNING: Decompyle incomplete


def test_hpm1_segment_parser_validates_token_and_hpac_blobs():
    (segment, _tokens) = _synthetic_hpm1_segment()
    (contract, token_blob, hpac_blob) = split_hpm1_mask_segment(segment)
# WARNING: Decompyle incomplete


def test_residual_symbol_helpers_roundtrip_raw_tokens():
    raw = np.array([
        [
            [
                0,
                1,
                2],
            [
                3,
                4,
                0]],
        [
            [
                1,
                1,
                3],
            [
                3,
                0,
                4]]], dtype = np.uint8)
    (residual, prev) = raw_tokens_to_mod5_residual_symbols(raw)
    reconstructed = reconstruct_raw_tokens_from_mod5_residual_symbols(residual, prev)
# WARNING: Decompyle incomplete


def test_cli_raw_token_loader_normalizes_qma9_storage_layout(tmp_path = None):
    module = _load_cli_script()
    storage_nwh = np.arange(24, dtype = np.uint8).reshape(2, 4, 3) % 5
    token_file = tmp_path / 'tokens.bin'
    token_file.write_bytes(storage_nwh.tobytes(order = 'C'))
    render_nhw = module._load_raw_tokens(token_file, '2,4,3', 'qma9_storage_wh_to_render_hw')
# WARNING: Decompyle incomplete


def test_residual_hpm1_prototype_is_local_only_and_roundtrip_grounded(tmp_path = None):
    (archive, raw_tokens) = _synthetic_archive(tmp_path)
    payload = extract_pr91_hpm1_payload(archive)
    report = prototype_reencode_hpm1_residual_from_raw_tokens(raw_tokens, payload)
# WARNING: Decompyle incomplete


def test_hpm1_segment_builder_fails_closed_on_misaligned_tokens():
    excinfo = pytest.raises(Pr91Hpm1Error)
    build_hpm1_mask_segment(b'abc', b'model', N = 1, H = 1, W = 1, P = 1, delta = 0, ch = 1, use_spm = False, hpac_d_film = 1)
    None(None, None)
# WARNING: Decompyle incomplete


def test_pr91_hpm1_pr85_stbm_fusion_planner_proves_byte_swap_on_synthetic(tmp_path = None):
    (hpm1_segment, _tokens) = _synthetic_hpm1_segment()
    stbm_segment = b'STBM1BR\x00' + b's' * (len(hpm1_segment) + 16)
    stbm_archive = _synthetic_bundle_archive(tmp_path / 'stbm.zip', stbm_segment, header_mode = 'v5')
    hpm1_archive = _synthetic_bundle_archive(tmp_path / 'hpm1.zip', hpm1_segment, header_mode = 'v5')
    report = plan_pr91_hpm1_pr85_stbm_fusion(pr85_stbm_archive = stbm_archive, pr91_archive = hpm1_archive, pr85_stbm_adjudicated_json = None, include_hpm1_prefix_probe = False)
# WARNING: Decompyle incomplete


def test_real_pr91_static_contract_if_available():
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip('PR91 archive is not present')
    payload = extract_pr91_hpm1_payload(DEFAULT_PR91_ARCHIVE)
    report = validate_hpm1_static_contract(payload)
# WARNING: Decompyle incomplete


def test_real_pr91_hpm1_pr85_stbm_fusion_is_byte_faithful_but_blocked_if_available():
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip('PR91 archive is not present')
    if not DEFAULT_PR85_STBM_ARCHIVE.is_file():
        pytest.skip('PR85+STBM archive is not present')
    report = plan_pr91_hpm1_pr85_stbm_fusion(include_hpm1_prefix_probe = False)
# WARNING: Decompyle incomplete


def test_real_pr91_prefix_decode_reproduces_entropy_failure_if_available():
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip('PR91 archive is not present')
    report = run_pr91_hpm1_preflight(DEFAULT_PR91_ARCHIVE, max_frames = 1)
# WARNING: Decompyle incomplete


def test_real_pr91_reuses_pr86_hpac_model_but_not_tokens_if_available():
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip('PR91 archive is not present')
    payload = extract_pr91_hpm1_payload(DEFAULT_PR91_ARCHIVE)
    report = compare_hpm1_to_pr86_hpac_contract(payload)
    if report['status'] == 'failed_closed_pr86_archive_unavailable':
        pytest.skip('PR86 archive is not present')
# WARNING: Decompyle incomplete


def test_pr91_runtime_source_contract_is_hpm1_cuda_sensitive_if_available():
    report = analyze_pr91_hpm1_runtime_sources()
    if report['status'] == 'failed_closed_missing_sources':
        pytest.skip('downloaded PR91 runtime sources are not present')
# WARNING: Decompyle incomplete


def test_hpm1_probability_variant_matrix_passes_on_synthetic_archive(tmp_path = None):
    (archive, _tokens) = _synthetic_archive(tmp_path)
    report = run_pr91_hpm1_probability_variant_matrix(archive, variants = ('source_float64_perfect_false',), max_frames = None, attempt_reencode = True, require_expected_pr91_identity = False)
# WARNING: Decompyle incomplete


def test_hpm1_context_window_probe_covers_context_and_eps_variants_on_synthetic(tmp_path = None):
    (archive, tokens) = _synthetic_archive(tmp_path)
    reference = tmp_path / 'reference_tokens.bin'
    reference.write_bytes(tokens.tobytes(order = 'C'))
    report = run_pr91_hpm1_context_window_probe(archive, reference_tokens_path = reference, reference_layout = 'legacy_assume_nhw', windows = ((0, 4), (10, 4)), variants = ('source_float64_perfect_false',), context_modes = ('decoded_context', 'reference_context'), prob_eps_values = (1e-07, 1e-09), require_expected_pr91_identity = False)
# WARNING: Decompyle incomplete


def test_real_pr91_probability_matrix_fails_closed_without_local_decode_if_available():
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip('PR91 archive is not present')
    report = run_pr91_hpm1_probability_variant_matrix(DEFAULT_PR91_ARCHIVE, variants = None, max_frames = 1)
# WARNING: Decompyle incomplete


def test_real_pr91_reference_prefix_probe_shrinks_pr85_identity_claim_if_available():
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip('PR91 archive is not present')
    if not DEFAULT_PR85_QMA9_TOKEN_SOURCE.is_file():
        pytest.skip('PR85 QMA9 reference token source is not present')
    report = run_pr91_hpm1_reference_prefix_probe(DEFAULT_PR91_ARCHIVE, variants = ('source_float64_perfect_false',), max_frames = 1)
# WARNING: Decompyle incomplete


def test_real_pr91_stream_transform_probe_rules_out_byte_word_order_if_available():
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip('PR91 archive is not present')
    report = run_pr91_hpm1_stream_transform_probe(DEFAULT_PR91_ARCHIVE, max_frames = 1)
# WARNING: Decompyle incomplete


def test_real_pr91_first_symbol_state_probe_exposes_source_contract_prefix_if_available():
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip('PR91 archive is not present')
    if not DEFAULT_PR85_QMA9_TOKEN_SOURCE.is_file():
        pytest.skip('PR85 QMA9 reference token source is not present')
    report = run_pr91_hpm1_first_symbol_state_probe(DEFAULT_PR91_ARCHIVE, variants = ('source_float64_perfect_false',), symbol_count = 16)
# WARNING: Decompyle incomplete


def test_real_pr91_symbol_window_probe_shrinks_entropy_failure_if_available():
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip('PR91 archive is not present')
    if not DEFAULT_PR85_QMA9_TOKEN_SOURCE.is_file():
        pytest.skip('PR85 QMA9 reference token source is not present')
    report = run_pr91_hpm1_first_symbol_state_probe(DEFAULT_PR91_ARCHIVE, variants = ('source_float64_perfect_false',), symbol_offset = 5948, symbol_count = 8)
# WARNING: Decompyle incomplete


def test_prototype_reencode_hpm1_from_synthetic_tokens(tmp_path = None):
    (archive, tokens) = _synthetic_archive(tmp_path)
    payload = extract_pr91_hpm1_payload(archive)
    report = prototype_reencode_hpm1_from_raw_tokens(tokens, payload)
# WARNING: Decompyle incomplete


def test_cli_writes_json_report(tmp_path = None):
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip('PR91 archive is not present')
    script = _load_cli_script()
    out = tmp_path / 'report.json'
# WARNING: Decompyle incomplete


def test_cli_probability_variant_matrix_writes_fail_closed_blocker(tmp_path = None):
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip('PR91 archive is not present')
    script = _load_cli_script()
    out = tmp_path / 'probability_matrix.json'
# WARNING: Decompyle incomplete


def test_cli_context_window_probe_writes_structured_failure(tmp_path = None):
    if not DEFAULT_PR91_ARCHIVE.is_file():
        pytest.skip('PR91 archive is not present')
    if not DEFAULT_PR85_QMA9_TOKEN_SOURCE.is_file():
        pytest.skip('PR85 QMA9 reference token source is not present')
    script = _load_cli_script()
    out = tmp_path / 'context_windows.json'
# WARNING: Decompyle incomplete

