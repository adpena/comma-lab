"""Retain cmp2's full carrier and evaluate a finite covering-count certificate.

Scorer-free input audit, not a generator, lattice solver, or pose predictor.
No family is admitted by the singleton counting floor returned here.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/Users/adpena/Projects/pact')
WORK = Path('/Volumes/VertigoDataTier/pact/ddm_gb2_generated_basis_bound')


def covering_count_floor_bytes(target_count: int, max_targets_per_message: int) -> int:
    """Worst-case fixed-length bound for a certified finite target cover.

This does NOT lower-bound an individual target's variable-length description.
The caller must prove its target set and maximum coverage per decoded message.
"""
    if type(target_count) is not int or type(max_targets_per_message) is not int:
        raise TypeError('counts must be integers')
    if target_count < 1 or not 1 <= max_targets_per_message <= target_count:
        raise ValueError('require 1 <= max_targets_per_message <= target_count')
    messages = (target_count + max_targets_per_message - 1) // max_targets_per_message
    return ((messages - 1).bit_length() + 7) // 8


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume-from', type=Path)
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(WORK.resolve()):
        raise ValueError('output outside charter-owned SSD tree')
    args.out.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.out).free < 64 * 1024**2:
        raise RuntimeError('less than 64 MiB available for retained input audit')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / 'experiments'))
    import numpy as np
    import ddm_gb1_pricing_input_audit as custody
    import ddm_up3_carrier_splice as up3

    pointer_path = ROOT / '.omx/state/canonical_frontier_pointer.json'
    pointer_data = pointer_path.read_bytes()
    frontier = json.loads(pointer_data)['effective_frontier']
    if args.resume_from:
        receipt = json.loads(args.resume_from.read_text())
        if receipt['archive']['sha256'] != frontier['archive_sha256']:
            raise RuntimeError('retained generation no longer matches live pointer')
        for record in receipt['retained']:
            raw = Path(record['path']).read_bytes()
            if len(raw) != record['bytes'] or custody.digest(raw) != record['sha256']:
                raise RuntimeError(f'custody mismatch: {record["path"]}')
        print(json.dumps({'status': 'RESUME_VERIFIED', 'retained_files': len(receipt['retained'])}))
        return 0

    archive_data = (args.runtime / 'archive.zip').read_bytes()
    if custody.digest(archive_data) != frontier['archive_sha256']:
        raise RuntimeError('runtime archive differs from live pointer')
    retained = [custody.retain(args.out / 'pointer_start.json', pointer_data)]
    archive = custody.retain(args.out / 'retained/archive.zip', archive_data)
    retained.append(archive)
    body = up3.parse_shipped_body(args.runtime, verify_sha=False)
    if body.archive_sha256 != archive['sha256'] or body.codes.shape != (600, 12):
        raise RuntimeError('parse changed source or population')
    for key, value in vars(body).items():
        if isinstance(value, bytes):
            retained.append(custody.retain(args.out / 'retained' / (key + '.bin'), value))
        elif isinstance(value, np.ndarray):
            buffer = io.BytesIO()
            np.save(buffer, value, allow_pickle=False)
            retained.append(custody.retain(args.out / 'retained' / (key + '.npy'), buffer.getvalue()))

    # Import only byte readers; no torch, rendering, scorer, or fitted model.
    from runtime import residual_archive as ra
    from runtime.compensation_overlay import apply_compensation_overlay, split_selector_compensation
    from runtime.frame0_selector import decode_selector
    selector, overlay = split_selector_compensation(ra.SPARSE_SELECTOR_PREFIX + body.body_tail)
    selected = decode_selector(selector)
    effective = body.codes.copy() if overlay is None else apply_compensation_overlay(body.codes, overlay)
    scales = np.frombuffer(body.scales, dtype='<f4').copy()
    if scales.shape != (24,) or not np.all(np.isfinite(scales)) or not np.all(scales > 0):
        raise RuntimeError('unexpected basis/coefficient scales')
    coefficients = effective.astype(np.float32) * scales[None, 12:]
    spec = importlib.util.spec_from_file_location('gb2_carrier_codec', args.runtime / 'cpr1/carrier_codec.py')
    codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)
    symbols = codec._decode_huffman(body.lengths, body.basis_blob, body.basis_bits, 12 * 3 * 24 * 32)
    basis_codes = ((symbols.astype(np.int64) >> 1) ^ -(symbols.astype(np.int64) & 1)).astype(np.int8).reshape(12, 3, 24, 32)
    for name, value in [('effective_codes', effective), ('coefficients_fp32', coefficients),
                        ('basis_codes', basis_codes), ('scales_fp32', scales)]:
        buffer = io.BytesIO()
        np.save(buffer, value, allow_pickle=False)
        retained.append(custody.retain(args.out / 'retained' / (name + '.npy'), buffer.getvalue()))
    retained.append(custody.retain(args.out / 'retained/selector.bin', selector))
    if overlay is not None:
        retained.append(custody.retain(args.out / 'retained/overlay.bin', overlay))
    sections = {'zip_overhead': len(archive_data) - len(body.outer),
                'rx1_header': ra.RX1_MODEL_HEADER.size,
                'hpac': len(body.hpac_stream), 'semantic': len(body.semantic_stream),
                'carrier': len(body.carrier_stream), 'tail': len(body.section_tail)}
    if sum(sections.values()) != len(archive_data):
        raise RuntimeError('archive section accounting differs')
    families = [{'family': family, 'K': k, 'step_multipliers': [1, 0.5, 0.125],
                 'singleton_counting_floor_bytes': covering_count_floor_bytes(1, 1),
                 'mandatory_seed_bytes': 0, 'allowed_seed_bytes_max': 8,
                 'achieved_total_bytes': None, 'pose_budget_certified': False,
                 'build_admitted': False, 'verdict': 'VACUOUS_FORMULATION_BOUND'}
                for family in ('Zernike', 'steerable_pyramid', 'generated_Gabor') for k in (6, 8, 12)]
    retained.append(custody.retain_json(args.out / 'FAMILY_TABLE.json', families))
    source_paths = [Path(__file__), ROOT / 'experiments/ddm_gb1_pricing_input_audit.py',
                    ROOT / 'experiments/ddm_up3_carrier_splice.py', ROOT / 'upstream/evaluate.py',
                    ROOT / 'upstream/README.md', args.runtime / 'cpr1/inflate.py',
                    args.runtime / 'cpr1/carrier_codec.py',
                    args.runtime / 'runtime/residual_archive.py',
                    args.runtime / 'runtime/compensation_overlay.py',
                    args.runtime / 'runtime/frame0_selector.py',
                    args.runtime / 'runtime/entropy/coefficient_predictor.py',
                    args.runtime / 'runtime/entropy/coefficient_ar1_codec.py',
                    args.runtime / 'runtime/carrier_repack.py',
                    args.runtime.parent / 'SEAL_ddm_cmp2_sm1_fe1_composed.json',
                    ROOT / '.omx/research/charters/ddm_gb2_generated_basis_unconditional_lattice_bound_20260909.md',
                    ROOT / '.omx/research/ddm_gb1_generated_carrier_basis_lattice_priced_closed_form_20260909.md']
    retained.extend(custody.source_receipt(p, args.out) for p in source_paths)
    if json.loads(pointer_path.read_bytes())['effective_frontier'] != frontier:
        raise RuntimeError('pointer moved during audit')
    receipt = {'schema': 'ddm_gb2_lattice_bound_audit_v1', 'archive': archive,
               'axis': '[closed-form, constants sourced; exact local bytes, scorer-free]',
               'research_only': True, 'score_claim': False, 'promotable': False,
               'archive_sections': sections, 'frontier': frontier, 'runtime': str(args.runtime),
               'base_codes_shape': list(body.codes.shape), 'effective_codes_shape': list(effective.shape),
               'effective_unique_pair_vectors': int(np.unique(effective, axis=0).shape[0]),
               'base_code_range': [int(body.codes.min()), int(body.codes.max())],
               'effective_code_range': [int(effective.min()), int(effective.max())],
               'overlay_changed_coordinates': int(np.count_nonzero(effective != body.codes)),
               'selector_repr': repr(selected), 'overlay_bytes': len(overlay or b''),
               'basis_bits': body.basis_bits, 'basis_blob_bytes': len(body.basis_blob),
               'basis_symbols': int(basis_codes.size), 'residual_bits': body.residual_bits,
               'rice_payload_bytes': len(body.rice_payload), 'restored_carrier_bytes': len(body.carrier_body),
               'metadata_bytes': len(body.packed_metadata), 'body_tail_bytes': len(body.body_tail),
               'basis_scales': scales[:12].tolist(), 'coefficient_scales': scales[12:].tolist(),
               'coefficients_min': coefficients.min(axis=0).tolist(),
               'coefficients_max': coefficients.max(axis=0).tolist(),
               'nonzero_code_count': int(np.count_nonzero(effective)),
               'whole_carrier_300B_target': sections['carrier'] - 300,
               'restored_basis_300B_reference_not_final_cost': len(body.basis_blob) - 300,
               'minimum_finite_code_step': float(scales[12:].min()),
               'seed': None, 'rng_used': False, 'processes': 1, 'numeric_threads': 1,
               'new_scored_pairs': 0, 'candidate_builds': 0, 'searches': 0,
               'argv': sys.argv, 'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
               'retained': retained, 'cleanup': 'Small retained inputs; no deletion, inflation, or scratch bulk.'}
    custody.retain_json(args.out / 'AUDIT.json', receipt)
    print(json.dumps({key: receipt[key] for key in ('archive_sections', 'overlay_changed_coordinates',
                        'effective_unique_pair_vectors', 'basis_bits', 'residual_bits', 'coefficient_scales')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
