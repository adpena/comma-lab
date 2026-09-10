"""Assemble the charter's measured inventories into data-pinned VR3 inputs.

Requires the completed rebuild certificates and hash ledger. It never applies
a reclaim plan or mutates source receipts, runtimes, or inventory payloads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_vr3_certified_raw_reclaim as vr3
from experiments.ddm_vr7_rebuild import SCRATCH, atomic, fact
from experiments.ddm_vr7_reproducer_descriptor import SCHEMA
from tac.candidate_seal import measure_runtime_digest


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def find_seal(root: Path, archive: dict) -> Path:
    for p in sorted(root.glob('SEAL*.json')):
        seal = read(p)
        if (seal.get('archive', {}).get('sha256') == archive['sha256']
                and seal['archive']['path'] == archive['path']
                and measure_runtime_digest(Path(seal['runtime']['path'])).sha256 == seal['runtime']['sha256']):
            return p
    raise ValueError('no current retained seal for ' + archive['path'])


def assemble(evidence: Path) -> None:
    inventory = read(evidence / 'selected_inventory.json')['rows']
    hashes = vr3.unique_rows(evidence / 'rehash.jsonl')
    if set(hashes) != {r['path'] for r in inventory}:
        raise ValueError('fresh hash coverage incomplete')
    runtime_cache = {}

    def runtime_record(root: Path) -> dict:
        if str(root) not in runtime_cache:
            runtime_cache[str(root)] = {
                'root': str(root), 'files': [fact(p) for p in sorted(root.rglob('*'))
                                            if p.is_file() and '__pycache__' not in p.parts]}
        return runtime_cache[str(root)]

    tc3 = vr3.VERTIGO_ROOT / 'ddm_tc3_lane_predictor_seal'
    tc3_inputs = read(tc3 / 'move39/INPUTS.json')
    tc3_config = read(evidence / 'tc3_replay_current_config.json')
    descriptors, sources = {}, []
    for rank, row in enumerate(inventory):
        target, original = Path(row['path']), Path(row['original_path'])
        kind = row['kind']
        descriptor = {'schema': SCHEMA, 'kind': kind, 'target_path': str(target),
                      'original_path': str(original), 'expected_sha256': row['expected_sha256'],
                      'expected_bytes': row['bytes'], 'cwd': str(REPO),
                      'environment': {'PYTHONDONTWRITEBYTECODE': '1', 'F26_TOKEN_DECODER': 'python'},
                      'score_claim': False}
        if kind.startswith('sj1_'):
            base = original
            while base.parent != vr3.VERTIGO_ROOT:
                base = base.parent
                if base == base.parent:
                    raise ValueError('SJ1 original root missing')
            tag = {'ddm_sj1_pass5_price': 'pass5', 'ddm_sj1_compose39_price': 'compose39',
                   'ddm_sj1_multipass_token_predistortion': 'multipass'}[base.name]
            descriptor['moved_manifest'] = vr3.pinned_file(Path(str(original) + '.MOVED.json'))
            if kind == 'sj1_parseback':
                output_receipt = original.parent / 'PARSEBACK_RESULT.json'
                prior = read(output_receipt)
                runtime = Path(prior['runtime'])
                descriptor['proof'] = vr3.pinned_file(evidence / f'rebuild_{tag}_certificate.json')
                descriptor['output_receipt_keys'] = ['rendered_raw']
                descriptor['code_pins'] = [fact(REPO / 'experiments/ddm_sj1_joint_admission.py')]
                descriptor['rebuild_argv'] = [sys.executable, '-B', 'experiments/ddm_sj1_joint_admission.py',
                                              'parseback', '--runtime', str(runtime), '--out-dir',
                                              str(SCRATCH / f'restored_raw_{rank:04d}'), '--threads', '4']
            else:
                output_receipt = original.parent / 'OVERLAY.json'
                prior = read(output_receipt)
                config = read(evidence / f'overlay_{tag}.json')
                runtime = Path(config['runtime'])
                config_path = evidence / f'overlay_row_{rank:04d}.json'
                config = {**config, 'field': prior['field_npz'], 'original': str(target),
                          'expected_bytes': row['bytes'], 'expected_sha256': row['expected_sha256'],
                          'scratch': str(SCRATCH / f'restored_overlay_{rank:04d}'),
                          'argv': [sys.executable, '-B', 'experiments/ddm_vr7_overlay_rebuild.py',
                                   '--resume-from', str(config_path)]}
                # Replace instance-specific input pins, retaining the same numerical code.
                config['pins'] = [fact(Path(p['path'])) for p in config['pins'] if p['path'].endswith('.py')]
                config['pins'] += [fact(output_receipt), fact(Path(prior['field_npz'])),
                                   fact(Path(str(original) + '.MOVED.json'))]
                atomic(config_path, config)
                descriptor['rebuild_config'] = vr3.pinned_file(config_path)
                descriptor['rebuild_argv'] = [*config['argv'], '--retain-output']
                descriptor['proof'] = vr3.pinned_file(evidence / f'overlay_{tag}_certificate.json')
                descriptor['output_receipt_keys'] = []
                descriptor['code_pins'] = [fact(REPO / 'experiments' / name) for name in (
                    'ddm_jg1_seg_solve.py', 'ddm_sj1_joint_admission.py', 'ddm_vr7_overlay_rebuild.py')]
                descriptor['supplemental_inputs'] = [fact(Path(prior['field_npz']))]
            descriptor['archive'] = fact(runtime / 'archive.zip')
            seal_path = find_seal(base, descriptor['archive'])
            descriptor['seal'] = vr3.pinned_file(seal_path)
            descriptor['seal_archive'] = descriptor['archive']
            descriptor['runtime'] = runtime_record(runtime)
            descriptor['output_receipt'] = vr3.pinned_file(output_receipt)
            descriptor['receipt_output_path'] = str(original)
            proof = read(Path(descriptor['proof']['path']))
            descriptor['inheritance'] = {
                'kind': kind, 'representative_original': proof['original']['path'],
                'directly_rebuilt_this_instance': proof['original']['path'] == str(target),
                'rationale': (f'{tag}: same {kind} interface, n600 geometry, four CPU threads, '
                              'pinned SJ1/JG1 numerical implementation and retained sealed runtime. '
                              'Other archives/fields are instance data; their current bytes and '
                              'original output receipts are checked individually. Only the named '
                              'representative was rerun; this is declared kind-level inheritance.')}
        else:
            descriptor['base_move'] = 39
            runtime = tc3 / 'move39/source_runtime'
            descriptor['archive'] = fact(runtime / 'archive.zip')
            descriptor['runtime'] = runtime_record(runtime)
            if kind == 'tc3_public_raw':
                descriptor['output_receipt'] = vr3.pinned_file(tc3 / 'move39/source_raw_custody/PARSEBACK_RESULT.json')
                descriptor['output_receipt_keys'] = ['rendered_raw']
                descriptor['receipt_output_path'] = read(Path(descriptor['output_receipt']['path']))['rendered_raw']['path']
                descriptor['proof'] = vr3.pinned_file(evidence / 'tc3_raw_identity.json')
                descriptor['code_pins'] = [fact(REPO / 'experiments/ddm_sj1_joint_admission.py')]
                descriptor['rebuild_argv'] = [sys.executable, '-B', 'experiments/ddm_sj1_joint_admission.py',
                                              'parseback', '--runtime', str(runtime), '--out-dir',
                                              str(SCRATCH / 'restored_tc3_raw'), '--threads', '4']
            else:
                stage = 'trace' if kind == 'tc3_trace' else 'prepare_v2'
                descriptor['output_receipt'] = vr3.pinned_file(target.with_suffix('.json'))
                descriptor['output_receipt_keys'] = ['payload'] if kind == 'tc3_trace' else []
                descriptor['receipt_output_path'] = str(target)
                descriptor['proof'] = vr3.pinned_file(tc3 / 'move39' / stage / 'RESULT.json')
                descriptor['inputs_receipt'] = vr3.pinned_file(tc3 / 'move39/INPUTS.json')
                descriptor['rebuild_config'] = vr3.pinned_file(evidence / 'tc3_replay_current_config.json')
                descriptor['rebuild_argv'] = [sys.executable, '-B', 'experiments/ddm_vr7_tc3_replay.py',
                                              '--resume-from', str(evidence / 'tc3_replay_current_config.json'),
                                              '--stop-after', '600', '--retain-output']
                descriptor['replay_proof'] = vr3.pinned_file(evidence / 'tc3_replay_current_certificate.json')
                descriptor['code_pins'] = [p for p in tc3_config['pins'] if p['path'].endswith('.py')]
                descriptor['supplemental_inputs'] = [tc3_inputs['field']]
        descriptors[str(target)] = descriptor
        current = hashes[str(target)]
        sources.append({'path': str(target), 'owner': 'MAIN / ' + ('ddm_sj1' if kind.startswith('sj1') else 'ddm_tc3'),
                        'certificate_status': 'DESCRIPTOR_VALIDATION_OWED', 'reproducer': None,
                        'bytes': current['bytes'], 'mtime_ns': current['mtime_ns'],
                        'device': current['device'], 'inode': current['inode'],
                        'historical_sha256': row['expected_sha256'],
                        'reference_aliases': list(dict.fromkeys([str(target), str(original)]))})
    bundle = evidence / 'descriptors.json'
    families = {}
    instance_keys = {'target_path', 'original_path', 'expected_sha256', 'expected_bytes',
                     'output_receipt', 'receipt_output_path'}
    for path, descriptor in list(descriptors.items()):
        if descriptor['kind'] not in {'tc3_trace', 'tc3_prepare'}:
            continue
        key = descriptor['kind']
        shared = {k: v for k, v in descriptor.items() if k not in instance_keys}
        if key in families and families[key] != shared:
            raise ValueError('TC3 family descriptor kind drift')
        families[key] = shared
        descriptors[path] = {'family_key': key, **{k: v for k, v in descriptor.items() if k in instance_keys}}
    atomic(bundle, {'schema': 'ddm_vr7.descriptor_bundle.v1', 'families': families,
                   'descriptors': descriptors})
    bundle_pin = vr3.pinned_file(bundle)
    for source in sources:
        source['reproducer_descriptor'] = {**bundle_pin, 'entry': source['path']}
    vr3.atomic_jsonl(evidence / 'sources.jsonl', sources)
    print(json.dumps({'sources': len(sources), 'descriptors': str(bundle),
                      'descriptor_bytes': bundle.stat().st_size, 'score_claim': False}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path, required=True)
    args = parser.parse_args()
    assemble(args.evidence_root.resolve())


if __name__ == '__main__':
    main()
