"""Data-pinned SJ1/cold-store admission for the VR3 plan/apply split.

This module never deletes. The caller must still enforce current stat, hash,
process, descriptor, and consumer gates immediately before applying a plan.
"""

from __future__ import annotations

import copy
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

SCHEMA = 'ddm_vr7.reproducer_descriptor.v1'
KINDS = {'sj1_parseback', 'sj1_overlay', 'tc3_trace', 'tc3_prepare', 'tc3_public_raw'}
SJ1_PATHS = {
    'ddm_sj1_pass5_price': {'parseback/0.raw', 'pose/base_overlay/odd_frames.u8',
                          'pose/cand_overlay/odd_frames.u8'},
    'ddm_sj1_compose39_price': {'parseback/0.raw', 'pose/base_overlay/odd_frames.u8',
                              'pose/cand_overlay/odd_frames.u8'},
    'ddm_sj1_multipass_token_predistortion': {
        'candidate/parseback/0.raw', 'candidate_pass3/parseback/0.raw',
        'pose/overlay/odd_frames.u8', 'pose_pass3/overlay/odd_frames.u8',
        'pose_pass4/overlay/odd_frames.u8'},
}


@lru_cache(maxsize=8)
def _descriptor_document(path: str, sha256: str) -> dict:
    import hashlib

    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError('descriptor changed while loading')
    return json.loads(payload)


def read_descriptor(pin: dict, vr3: Any) -> dict:
    """Bundles avoid thousands of duplicate runtime manifests; rehash at every use."""
    base = {'path': pin['path'], 'sha256': pin['sha256']}
    vr3.verify_pin(base)
    document = _descriptor_document(base['path'], base['sha256'])
    if 'entry' in pin:
        if document.get('schema') != 'ddm_vr7.descriptor_bundle.v1':
            raise vr3.CertifyError('DESCRIPTOR_BUNDLE_SCHEMA_MISMATCH')
        entry = document['descriptors'][pin['entry']]
        if 'family_key' in entry:
            shared = document['families'][entry['family_key']]
            return copy.deepcopy({**shared, **{k: v for k, v in entry.items() if k != 'family_key'}})
        return copy.deepcopy(entry)
    return copy.deepcopy(document)


def target_reason(path: Path, descriptor: dict, vr3: Any) -> str | None:
    """Narrow structural exception; never bypass the rest of certificate validation."""
    if descriptor.get('schema') != SCHEMA or descriptor.get('target_path') != str(path):
        return 'DESCRIPTOR_TARGET_MISMATCH'
    if descriptor.get('kind') not in KINDS:
        return 'DESCRIPTOR_KIND_UNKNOWN'
    try:
        root = vr3.storage_root(path)
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError, vr3.CertifyError):
        return 'DESCRIPTOR_TARGET_OUTSIDE_SSD'
    if not path.is_file() or path.is_symlink():
        return 'DESCRIPTOR_TARGET_NOT_REGULAR'
    current = path.parent
    while current != root:
        if current.is_symlink():
            return 'SYMLINK_ANCESTOR_BELOW_SSD_ROOT'
        if root not in current.parents:
            return 'ANCESTOR_ESCAPES_SSD_ROOT'
        current = current.parent
    rel = path.relative_to(root)
    if descriptor['kind'].startswith('sj1_'):
        if (root != vr3.AP_ROOT or len(rel.parts) < 4
                or rel.parts[0] != 'cold_store_sj1_bulk_20260910'
                or rel.parts[1] not in SJ1_PATHS):
            return 'SJ1_TARGET_NOT_CHARTERED_COLD_STORE'
        tail = rel.parts[2:]
        if '/'.join(tail) not in SJ1_PATHS[rel.parts[1]]:
            return 'SJ1_TARGET_NOT_IN_CHARTER_INVENTORY'
        if descriptor['kind'] == 'sj1_parseback':
            if tail not in {('parseback', '0.raw'), ('candidate', 'parseback', '0.raw'),
                            ('candidate_pass3', 'parseback', '0.raw')}:
                return 'SJ1_TARGET_NOT_PARSEBACK_BULK'
        elif tail not in {('pose', 'base_overlay', 'odd_frames.u8'),
                          ('pose', 'cand_overlay', 'odd_frames.u8'),
                          ('pose', 'overlay', 'odd_frames.u8'),
                          ('pose_pass3', 'overlay', 'odd_frames.u8'),
                          ('pose_pass4', 'overlay', 'odd_frames.u8')}:
            return 'SJ1_TARGET_NOT_OVERLAY_BULK'
        expected_original = vr3.VERTIGO_ROOT.joinpath(*rel.parts[1:])
        if descriptor.get('original_path') != str(expected_original):
            return 'SJ1_ORIGINAL_PATH_MISMATCH'
    else:
        if root != vr3.VERTIGO_ROOT or rel.parts[:2] != ('ddm_tc3_lane_predictor_seal', 'move39'):
            return 'TC3_LIVE_OR_UNCHARTERED_TREE'
        tail = rel.parts[2:]
        if descriptor['kind'] == 'tc3_public_raw':
            if tail != ('public_identity', 'output', '0.raw'):
                return 'TC3_NOT_SUPERSEDED_PUBLIC_RAW'
        else:
            stage = 'trace' if descriptor['kind'] == 'tc3_trace' else 'prepare_v2'
            if (len(tail) != 3 or tail[:2] != (stage, 'frames')
                    or not re.fullmatch(r'frame_0[0-5][0-9][0-9]\.npz', tail[2])):
                return 'TC3_NOT_SUPERSEDED_FRAME_TRACE'
    return None


def _fact(expected: dict, vr3: Any) -> Path:
    path = vr3.verify_pin({'path': expected['path'], 'sha256': expected['sha256']})
    if path.stat().st_size != expected['bytes']:
        raise vr3.CertifyError(f'PIN_SIZE_DRIFT:{path}')
    return path


def _record(descriptor: dict, vr3: Any) -> dict:
    record = vr3.load_json(vr3.verify_pin(descriptor['output_receipt']))
    for key in descriptor['output_receipt_keys']:
        record = record[key]
    if not isinstance(record, dict):
        raise vr3.CertifyError('OUTPUT_RECEIPT_NOT_OBJECT')
    # Overlay receipts use named scalars; parse-back and TC3 use file-fact objects.
    if descriptor['kind'] == 'sj1_overlay':
        record = {'path': record['odd_frames_path'], 'sha256': record['odd_frames_sha256'],
                  'bytes': record['odd_frames_bytes']}
    if (record.get('sha256') != descriptor['expected_sha256']
            or record.get('bytes') != descriptor['expected_bytes']
            or record.get('path') != descriptor['receipt_output_path']):
        raise vr3.CertifyError('OUTPUT_RECEIPT_IDENTITY_MISMATCH')
    return record


def _command_binding(descriptor: dict, vr3: Any) -> None:
    """An argv is a claim: bind its effective inputs, not merely its presence."""
    argv = descriptor['rebuild_argv']
    kind = descriptor['kind']
    if kind in {'sj1_parseback', 'tc3_public_raw'}:
        if (len(argv) != 10 or argv[1:4] != ['-B', 'experiments/ddm_sj1_joint_admission.py', 'parseback']
                or argv[4:6] != ['--runtime', descriptor['runtime']['root']]
                or argv[6] != '--out-dir' or argv[8:] != ['--threads', '4']):
            raise vr3.CertifyError('PARSEBACK_ARGV_INPUT_DRIFT')
        if not Path(argv[7]).is_relative_to(vr3.VERTIGO_ROOT / 'ddm_vr7_certify'):
            raise vr3.CertifyError('REBUILD_OUTPUT_NOT_OWNED_SCRATCH')
    else:
        config_path = vr3.verify_pin(descriptor['rebuild_config'])
        config = vr3.load_json(config_path)
        if not config.get('pins'):
            raise vr3.CertifyError('REBUILD_CONFIG_INPUT_PINS_MISSING')
        for pin in config['pins']:
            _fact(pin, vr3)
        script = ('experiments/ddm_vr7_overlay_rebuild.py' if kind == 'sj1_overlay'
                  else 'experiments/ddm_vr7_tc3_replay.py')
        expected = [argv[0], '-B', script, '--resume-from', str(config_path)]
        if kind != 'sj1_overlay':
            expected += ['--stop-after', '600']
        expected += ['--retain-output']
        if argv != expected:
            raise vr3.CertifyError('REBUILD_CONFIG_ARGV_DRIFT')
        if kind == 'sj1_overlay':
            output_receipt = vr3.load_json(vr3.verify_pin(descriptor['output_receipt']))
            if (config['field'] != output_receipt['field_npz']
                    or config['runtime'] != descriptor['runtime']['root']
                    or config['archive'] != descriptor['archive']['path']
                    or config['expected_sha256'] != descriptor['expected_sha256']
                    or config['expected_bytes'] != descriptor['expected_bytes']):
                raise vr3.CertifyError('OVERLAY_FIELD_OR_ARCHIVE_BINDING_DRIFT')
            if vr3.sha256_file(Path(config['field'])) != output_receipt['field_npz_sha256']:
                raise vr3.CertifyError('OVERLAY_FIELD_PIN_DRIFT')
        elif config['archive_sha256'] != descriptor['archive']['sha256']:
            raise vr3.CertifyError('TC3_REPLAY_ARCHIVE_DRIFT')


def certify(descriptor_pin: dict, path: Path, raw_sha: str, vr3: Any) -> dict:
    descriptor = read_descriptor(descriptor_pin, vr3)
    reason = target_reason(path, descriptor, vr3)
    if reason:
        raise vr3.CertifyError(reason)
    if (not vr3.valid_sha256(raw_sha) or raw_sha != descriptor['expected_sha256']
            or path.stat().st_size != descriptor['expected_bytes']):
        raise vr3.CertifyError('DESCRIPTOR_OUTPUT_IDENTITY_MISMATCH')
    argv = descriptor.get('rebuild_argv')
    if (not isinstance(argv, list) or not argv or
            not all(isinstance(x, str) and x for x in argv)):
        raise vr3.CertifyError('EXACT_REBUILD_ARGV_REQUIRED')
    if not Path(descriptor['cwd']).is_dir():
        raise vr3.CertifyError('REBUILD_CWD_MISSING')
    _command_binding(descriptor, vr3)
    _record(descriptor, vr3)
    if descriptor['kind'].startswith('sj1_'):
        moved_path = vr3.verify_pin(descriptor['moved_manifest'])
        if str(moved_path) != descriptor['original_path'] + '.MOVED.json':
            raise vr3.CertifyError('MOVED_MANIFEST_SOURCE_MISMATCH')
        moved = vr3.load_json(moved_path)
        if (moved.get('moved_to') != str(path) or moved.get('sha256') != raw_sha
                or moved.get('bytes') != descriptor['expected_bytes']):
            raise vr3.CertifyError('MOVED_DESTINATION_IDENTITY_MISMATCH')
        if Path(descriptor['original_path']).exists():
            raise vr3.CertifyError('MOVED_SOURCE_REAPPEARED')
        seal = vr3.load_json(vr3.verify_pin(descriptor['seal']))
        if (seal['archive']['sha256'] != descriptor['seal_archive']['sha256']
                or seal['archive']['path'] != descriptor['seal_archive']['path']):
            raise vr3.CertifyError('SEAL_ARCHIVE_MISMATCH')
        _fact(descriptor['seal_archive'], vr3)
        from tac.candidate_seal import measure_runtime_digest

        measured = measure_runtime_digest(Path(seal['runtime']['path']))
        if (measured.sha256 != seal['runtime']['sha256']
                or descriptor['runtime']['root'] != seal['runtime']['path']
                or descriptor['archive'] != descriptor['seal_archive']):
            raise vr3.CertifyError('SEAL_RUNTIME_DRIFT')
    _fact(descriptor['archive'], vr3)
    runtime = descriptor['runtime']
    root = Path(runtime['root'])
    entries = list(root.rglob('*'))
    if not root.is_dir() or root.is_symlink() or any(p.is_symlink() for p in entries):
        raise vr3.CertifyError('RUNTIME_MISSING_OR_SYMLINKED')
    expected_set = {p['path'] for p in runtime['files']}
    actual_set = {str(p) for p in entries if p.is_file() and '__pycache__' not in p.parts}
    if not expected_set or len(expected_set) != len(runtime['files']) or actual_set != expected_set:
        raise vr3.CertifyError('RUNTIME_FILE_SET_DRIFT')
    for pin in runtime['files']:
        _fact(pin, vr3)
    if not descriptor.get('code_pins'):
        raise vr3.CertifyError('REBUILD_CODE_PINS_MISSING')
    for pin in descriptor['code_pins'] + descriptor.get('supplemental_inputs', []):
        _fact(pin, vr3)
    proof = vr3.load_json(vr3.verify_pin(descriptor['proof']))
    if descriptor['kind'].startswith('sj1_'):
        if (proof.get('kind') != descriptor['kind'] or proof.get('complete') is not True
                or proof.get('twin_hash_equal') is not True):
            raise vr3.CertifyError('REAL_REPRESENTATIVE_REBUILD_REQUIRED')
        original, rebuilt = proof['original'], proof['rebuilt']
        if ((original['sha256'], original['bytes']) != (rebuilt['sha256'], rebuilt['bytes'])
                or not vr3.valid_sha256(original['sha256'])):
            raise vr3.CertifyError('REBUILD_TWIN_HASH_MISMATCH')
        config = vr3.load_json(_fact(proof['config'], vr3))
        if proof['argv'] != config['argv']:
            raise vr3.CertifyError('REBUILD_ARGV_BINDING_MISMATCH')
        if (config['expected_sha256'] != original['sha256']
                or config['expected_bytes'] != original['bytes']
                or config['original'] != original['path']):
            raise vr3.CertifyError('REBUILD_ORIGINAL_BINDING_MISMATCH')
        inheritance = descriptor['inheritance']
        if (inheritance.get('representative_original') != original['path']
                or inheritance.get('kind') != descriptor['kind']
                or not inheritance.get('rationale')):
            raise vr3.CertifyError('REBUILD_INHERITANCE_NOT_DECLARED')
        cold = vr3.AP_ROOT / 'cold_store_sj1_bulk_20260910'
        try:
            same_family = (Path(original['path']).relative_to(cold).parts[0]
                           == path.relative_to(cold).parts[0])
        except (ValueError, IndexError):
            same_family = False
        if not same_family:
            raise vr3.CertifyError('REBUILD_FAMILY_INHERITANCE_MISMATCH')
        representative_code = {p['path']: p['sha256'] for p in config['pins']}
        if any(representative_code.get(p['path']) != p['sha256'] for p in descriptor['code_pins']):
            raise vr3.CertifyError('REBUILD_INHERITED_CODE_DRIFT')
    else:
        if descriptor.get('base_move') not in (37, 39):
            raise vr3.CertifyError('TC3_NOT_SUPERSEDED_BASE')
        if descriptor['kind'] == 'tc3_trace':
            if proof.get('frames') != 600 or proof.get('twin_byte_identical') is not True:
                raise vr3.CertifyError('TC3_COMPLETED_TRACE_RECEIPT_REQUIRED')
            inputs = vr3.load_json(vr3.verify_pin(descriptor['inputs_receipt']))
            if (proof['binding']['inputs_sha'] != descriptor['inputs_receipt']['sha256']
                    or proof['field_sha256'] != inputs['field']['sha256']
                    or inputs['archive_sha256'] != descriptor['archive']['sha256']):
                raise vr3.CertifyError('TC3_TRACE_INPUT_BINDING_DRIFT')
            frame = vr3.load_json(vr3.verify_pin(descriptor['output_receipt']))
            if frame['binding'] != proof['binding']:
                raise vr3.CertifyError('TC3_FRAME_BINDING_DRIFT')
        elif descriptor['kind'] == 'tc3_prepare':
            if proof.get('frames') != 600 or proof.get('full_n600') is not True:
                raise vr3.CertifyError('TC3_COMPLETED_PREPARE_RECEIPT_REQUIRED')
            if proof['binding']['archive'] != descriptor['archive']['sha256']:
                raise vr3.CertifyError('TC3_PREPARE_ARCHIVE_DRIFT')
        elif (proof.get('schema') != 'ddm_vr7.tc3_raw_identity.v1'
              or proof.get('same_host_full_raw_equal') is not True
              or proof.get('sha256') != raw_sha
              or proof.get('reproducer_archive_sha256') != descriptor['archive']['sha256']):
            raise vr3.CertifyError('TC3_FULL_RAW_IDENTITY_REQUIRED')
        if descriptor['kind'] in {'tc3_trace', 'tc3_prepare'}:
            replay = vr3.load_json(vr3.verify_pin(descriptor['replay_proof']))
            replay_config = _fact(replay['config'], vr3)
            if str(replay_config) != descriptor['rebuild_config']['path']:
                raise vr3.CertifyError('TC3_REPLAY_CONFIG_BINDING_DRIFT')
            if (replay.get('complete') is not True
                    or replay.get('same_bytes_as_retained_receipts') is not True
                    or replay.get('originals_twin_hashed') is not True
                    or replay.get('frames_per_stage', 0) < 1):
                raise vr3.CertifyError('TC3_REPLAY_COMMAND_NOT_VERIFIED')
    return {'descriptor': descriptor_pin, 'archive_path': descriptor['archive']['path'],
            'archive_sha256': descriptor['archive']['sha256'], 'kind': descriptor['kind'],
            'reproducer_argv': argv, 'proof': descriptor['proof'],
            'reference_aliases': list(dict.fromkeys([str(path), descriptor['original_path']]))}


def descriptor_for(source: dict, vr3: Any) -> dict | None:
    pin = source.get('reproducer_descriptor')
    return None if pin is None else read_descriptor(pin, vr3)
