# Source Generated with Decompyle++
# File: test_preflight_pr91_pr92_replay_contracts.cpython-312.pyc (Python 3.12)

from __future__ import annotations
import importlib.util as importlib
import json
import sys
from pathlib import Path
import pytest
REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / 'experiments' / 'preflight_pr91_pr92_replay_contracts.py'

def _load_script():
    spec = importlib.util.spec_from_file_location('preflight_pr91_pr92_replay_contracts_test', SCRIPT)
# WARNING: Decompyle incomplete

module = _load_script()

def _write_json(path = None, payload = None):
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(json.dumps(payload, indent = 2, sort_keys = True), encoding = 'utf-8')
    return path


def _synthetic_pr91_probability_report(path = None):
    return _write_json(path, {
        'status': 'failed_closed',
        'failure_reason': 'no_probability_variant_decodes_pr91_hpm1_prefix',
        'blocker_class': 'real_invalid_entropy_or_probability_model_contract_mismatch',
        'dispatch_unlocked': False,
        'pr91_ready_for_exact_eval': False,
        'local_decode_variants': [],
        'variant_results': [
            {
                'variant': 'source_float64_perfect_false',
                'status': 'failed_closed',
                'failure_context': {
                    'failed_at': {
                        'frame': 0,
                        'group': 10,
                        'symbol_in_group': 191 },
                    'decoded_symbol_count_before_failure': 5951 } },
            {
                'variant': 'source_float32_perfect_false',
                'status': 'failed_closed',
                'failure_context': {
                    'decoded_symbol_count_before_failure': 30513 } }] })


def _synthetic_pr92_manifest_and_exact(tmp_path = None):
    archive = tmp_path / 'archive.zip'
    inflate = tmp_path / 'runtime' / 'inflate.sh'
    archive.write_bytes(b'archive')
    inflate.parent.mkdir(parents = True, exist_ok = True)
    inflate.write_text('#!/bin/sh\n', encoding = 'utf-8')
    score = 0.253506
    manifest = _write_json(tmp_path / 'manifest.json', {
        'score_claim': False,
        'dispatch_performed': False,
        'candidate_archive': {
            'path': str(archive.relative_to(REPO)) if archive.is_relative_to(REPO) else str(archive),
            'archive_bytes': 229480,
            'archive_sha256': 'abc' },
        'strict_zip': {
            'valid': True },
        'non_noop_byte_change': {
            'changed_segments_vs_stbm': [
                'randmulti'] },
        'randmulti_decoded_row_parity': {
            'decoded_rows_match': True },
        'exact_eval_runtime_contract': {
            'ready_for_exact_eval_runtime': True,
            'remaining_blockers': [],
            'required_inflate_sh': str(inflate.relative_to(REPO)) if inflate.is_relative_to(REPO) else str(inflate) },
        'dispatch_readiness': { } })
    exact = _write_json(tmp_path / 'exact.json', {
        'canonical_score': score,
        'archive_size_bytes': 229480,
        'avg_segnet_dist': 0.00057185,
        'avg_posenet_dist': 0.0001894,
        'n_samples': 600,
        'provenance': {
            'archive_sha256': 'abc',
            'device': 'cuda',
            'gpu_t4_match': True,
            'inflate_runtime_manifest': {
                'runtime_tree_sha256': 'runtime-tree',
                'runtime_root': '/remote/replay_submission_stbm_rmb1' } } })
    return (manifest, exact)


def test_pr91_hpm1_contract_fails_closed_from_probability_report(tmp_path = None):
    report_path = _synthetic_pr91_probability_report(tmp_path / 'pr91.json')
    report = module.validate_pr91_hpm1_contract(archive = tmp_path / 'archive.zip', probability_report = report_path, rerun = False)
# WARNING: Decompyle incomplete


def test_pr92_rmb1_contract_validates_exact_t4_runtime_path(tmp_path = None):
    (manifest, exact) = _synthetic_pr92_manifest_and_exact(tmp_path)
    report = module.validate_pr92_rmb1_contract(manifest_path = manifest, exact_json_path = exact)
# WARNING: Decompyle incomplete


def test_real_pr91_pr92_preflight_if_artifacts_available(tmp_path = None):
    required = [
        module.DEFAULT_PR91_PROBABILITY_REPORT,
        module.DEFAULT_PR92_MANIFEST,
        module.DEFAULT_PR92_EXACT_JSON]
# WARNING: Decompyle incomplete

