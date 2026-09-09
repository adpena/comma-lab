"""Retain the real carrier and audit inputs to gb1's blocked pricing gate.

Scorer-free, research_only=True. This does not price generated families, compute
Jacobians, solve coefficients, or claim a distortion bound. It deliberately
leaves those fields null when the required evidence is unavailable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WORK = Path('/Volumes/VertigoDataTier/pact/ddm_gb1_generated_carrier_basis')
POINTER = ROOT / '.omx/state/canonical_frontier_pointer.json'
PC1 = Path('/Volumes/VertigoDataTier/pact/ddm_pc1_pose_carrier_efficiency')
PC2 = Path('/Volumes/VertigoDataTier/pact/ddm_pc2_carrier_kwidth_rankcut')


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def retain(path: Path, data: bytes) -> dict:
    """Atomic, idempotent retention; refuse to overwrite different evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f'different evidence already exists: {path}')
    else:
        scratch = path.with_name(path.name + '.partial')
        with scratch.open('wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        scratch.replace(path)
    return {'path': str(path), 'bytes': len(data), 'sha256': digest(data)}


def retain_json(path: Path, value) -> dict:
    return retain(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())


def source_receipt(path: Path, out: Path) -> dict:
    data = path.read_bytes()
    record = retain(out / 'sources' / (digest(data)[:16] + '_' + path.name), data)
    record['source_path'] = str(path)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=WORK / 'input_audit')
    parser.add_argument('--resume-from', type=Path)
    args = parser.parse_args()
    # Small archive/metadata audit only. No renders, caches, or scratch inflation.
    if not args.out.resolve().is_relative_to(WORK.resolve()):
        raise RuntimeError('output must be in the charter-owned tree')
    args.out.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.out).free < 64 * 1024**2:
        raise RuntimeError('insufficient space for retained archive/metadata')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    pointer_bytes = POINTER.read_bytes()
    pointer = json.loads(pointer_bytes)
    frontier = pointer['effective_frontier']
    archive_path = args.runtime / 'archive.zip'
    archive_bytes = archive_path.read_bytes()
    if digest(archive_bytes) != frontier['archive_sha256']:
        raise RuntimeError('runtime archive is not the live pointer; rebase required')
    if args.resume_from:
        receipt = json.loads(args.resume_from.read_text())
        if receipt['archive']['sha256'] != frontier['archive_sha256']:
            raise RuntimeError('resume input is stale')
        for item in receipt['retained']:
            data = Path(item['path']).read_bytes()
            if len(data) != item['bytes'] or digest(data) != item['sha256']:
                raise RuntimeError(f'retention verification failed: {item["path"]}')
        print(json.dumps({'status': 'RESUME_VERIFIED', 'receipt': str(args.resume_from)}))
        return 0

    sys.path.insert(0, str(ROOT / 'experiments'))
    import numpy as np
    import ddm_up3_carrier_splice as up3

    retained = [retain(args.out / 'pointer_start.json', pointer_bytes)]
    archive_record = retain(args.out / 'retained/archive.zip', archive_bytes)
    retained.append(archive_record)
    body = up3.parse_shipped_body(args.runtime, verify_sha=False)
    if body.archive_sha256 != frontier['archive_sha256']:
        raise RuntimeError('archive changed while parsing')
    for key, value in vars(body).items():
        if isinstance(value, bytes):
            retained.append(retain(args.out / 'retained' / (key + '.bin'), value))
        elif isinstance(value, np.ndarray):
            import io
            buf = io.BytesIO()
            np.save(buf, value, allow_pickle=False)
            retained.append(retain(args.out / 'retained' / (key + '.npy'), buf.getvalue()))
    if body.codes.shape != (600, 12):
        raise RuntimeError('carrier code population is not 600 x 12')
    fields = {
        'zip_overhead': len(archive_bytes) - len(body.outer),
        'rx1_header': len(body.outer) - sum(map(len, [body.hpac_stream, body.semantic_stream,
                                                    body.carrier_stream, body.section_tail])),
        'hpac_stream': len(body.hpac_stream), 'semantic_stream': len(body.semantic_stream),
        'carrier_stream': len(body.carrier_stream), 'section_tail': len(body.section_tail),
    }
    if sum(fields.values()) != len(archive_bytes):
        raise RuntimeError('section byte accounting failed')
    originals = [
        Path(__file__),
        ROOT / 'experiments/ddm_up3_carrier_splice.py',
        PC1 / 'rate_v4/RATE.json', PC1 / 'coverage/COVERAGE.json',
        PC2 / 'jacobian/jacobian_leverage.json', PC2 / 'base/base_d_pose.json',
        PC2 / 'base/base_per_pair.npz',
        ROOT / 'experiments/ddm_up2_shipping_pose_solve.py',
        ROOT / 'experiments/ddm_pc2_carrier_kwidth_rankcut.py',
        ROOT / 'experiments/ddm_jg5_pose_resolve_on_edited_renders.py',
        ROOT / '.omx/research/ddm_jc1_carrier_jacobian_posemetric_refit_20260816.md',
        ROOT / '.omx/research/ddm_gb1_prereg_20260909.md',
    ]
    sources = [source_receipt(p, args.out) for p in originals]
    retained.extend(sources)
    jac = json.loads((PC2 / 'jacobian/jacobian_leverage.json').read_text())
    rate = json.loads((PC1 / 'rate_v4/RATE.json').read_text())
    coverage = json.loads((PC1 / 'coverage/COVERAGE.json').read_text())
    # This explicitly enumerates unfinished work. Null is not a numerical result.
    rows = []
    for family in ('zernike', 'steerable_pyramid', 'generated_gabor', 'jacobian_fitted_svd'):
        for k in (6, 8, 12):
            for multiplier in (1.0, 0.5, 0.125):
                rows.append({
                    'family': family, 'K': k, 'step_multiplier': multiplier,
                    'delta_absolute': None, 'counted_bytes': None,
                    'd_pose_lower_bound': None, 'net_delta_s': None,
                    'per_pair_receipt': None, 'matched_lattice_penalty': None,
                    'status': 'BLOCKED_INPUT_AND_CERTIFICATE',
                    'blockers': ['NO_LIVE_N600_SPATIAL_JACOBIAN',
                                 'NO_REALIZED_REMAINDER_CERTIFICATE',
                                 'NO_PROJECTED_COEFFICIENT_RATE',
                                 'NO_MATCHED_ABSOLUTE_LATTICE_PENALTY'] +
                                (['FITTED_ATOMS_ARE_COUNTED'] if family == 'jacobian_fitted_svd' else []),
                    'score_claim': False, 'build_admitted': False,
                })
    retained.append(retain_json(args.out / 'pricing_rows.json', rows))
    # All nonnegative losses have this bound; it is not a lattice certificate.
    retained.append(retain_json(args.out / 'universal_nonnegative_bound.json', {
        'status': 'DERIVED_TAUTOLOGY_NOT_A_MEASUREMENT', 'n_pairs': 600,
        'per_pair_lower_bound': [0.0] * 600, 'n600_lower_bound': 0.0,
        'usable_for_generated_family_closure_or_admission': False,
    }))
    if json.loads(POINTER.read_bytes())['effective_frontier'] != frontier:
        raise RuntimeError('pointer moved during audit; retain evidence and rebase')
    receipt = {
        'schema': 'ddm_gb1_pricing_input_audit_v1', 'research_only': True,
        'score_claim': False, 'promotable': False, 'status': 'PRICING_BLOCKED',
        'frontier': frontier, 'archive': archive_record, 'runtime': str(args.runtime),
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'producer_sha256': digest(Path(__file__).read_bytes()),
        'argv': sys.argv, 'rng': 'not used; deterministic parse only', 'threads': 1,
        'axis': '[exact local byte arithmetic; scorer-free]',
        'archive_sections': fields, 'restored_carrier_body_bytes': len(body.carrier_body),
        'restored_basis_huffman_bytes': len(body.basis_blob),
        'restored_rice_bytes': len(body.rice_payload), 'scales_bytes': len(body.scales),
        'coefficient_scales': np.frombuffer(body.scales, dtype='<f4')[12:].tolist(),
        'codes_shape': list(body.codes.shape),
        'codes_scope': 'CAP1 base codes; no claim of compensation-overlay application or rendered-pixel identity',
        'code_range': [int(body.codes.min()), int(body.codes.max())],
        'jacobian_receipt_pairs': jac['pairs'], 'retained_jacobian_tensor_in_pc2_receipt': False,
        'historical_v4_rate': rate['rows'], 'historical_v4_base_sha256': rate['body_archive_sha256'],
        'historical_field_coverage': coverage['rows'],
        'completed_generated_prices': 0, 'planned_generated_prices': len(rows),
        'new_pose_pairs_scored': 0, 'solver_launches': 0, 'candidate_archives_built': 0,
        'sources': sources, 'retained': retained,
        'cleanup': 'No scratch inflation or deletion. Partial outputs are retained and idempotently reusable.',
    }
    retain_json(args.out / 'AUDIT.json', receipt)
    print(json.dumps({k: receipt[k] for k in ['status', 'archive_sections', 'code_range',
                                              'completed_generated_prices', 'planned_generated_prices']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
