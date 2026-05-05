"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``72:20: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``q_faithful_snapshot_loop.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'scripts/q_faithful_snapshot_loop.py'
__recovery_spec__ = 'q_faithful_snapshot_loop.recovery_spec.json'
__recovery_ast_error__ = '72:20: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: q_faithful_snapshot_loop.cpython-312.pyc (Python 3.12)

'''Bounded Q-FAITHFUL checkpoint snapshot export and H100 screen scaffold.

This script is intentionally separate from the long Q-FAITHFUL launcher. It can
be run beside an active training job to turn stable checkpoints into deterministic
archive snapshots, then optionally screen those exact bytes through the existing
archive-only CUDA eval wrapper.
'''
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any
EXPECTED_MASK_FRAMES = 1200
HALF_FRAME_MASK_FRAMES = EXPECTED_MASK_FRAMES // 2
SNAPSHOT_RUNTIME_CONTRACT_VERSION = 'qfaithful_snapshot_runtime_contract_v2'
ZOOM_WARP_ARCHIVE_MEMBER = 'zoom_scalars.bin'
QFAITHFUL_RENDERER_ARCHITECTURE = 'quantizr_faithful_joint_frame_generator'
QFAITHFUL_EXPORT_CONTRACT_REQUIRED_KEYS = ('runtime_contract_version', 'mask_frame_contract', 'renderer_zoom_contract', 'eval_roundtrip_required', 'profile', 'promotable_exact_screen', 'pose_tensor_contract', 'training_pose_contract', 'packed_from_ema_shadow')
REPO_RUNTIME_SHA_PATHS = ('scripts/q_faithful_snapshot_loop.py', 'experiments/repack_quantizr_faithful_qzs3_archive.py', 'scripts/remote_archive_only_eval.sh', 'submissions/robust_current/inflate.sh', 'submissions/robust_current/inflate_renderer.py', 'submissions/robust_current/unpack_renderer_payload.py', 'submissions/robust_current/apply_qzs3_postprocess.py', 'src/tac/quantizr_qzs3_codec.py', 'src/tac/quantizr_faithful_export.py', 'src/tac/quantizr_faithful_renderer.py', 'src/tac/profiles.py')

class SnapshotError(RuntimeError):
    pass
# WARNING: Decompyle incomplete


def utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond = 0).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha256_file(path = None):
    pass
# WARNING: Decompyle incomplete


def _stat_size(path = None):
    return path.stat().st_size


def _json_dump(path = None, payload = None):
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(json.dumps(payload, indent = 2, sort_keys = True) + '\n')


def _repo_path(workspace = None, rel = None):
    if rel.startswith('/') or '..' in Path(rel).parts:
        raise SnapshotError('unsafe_runtime_path', f'''unsafe runtime path: {rel}''')
    return workspace / rel


def source_runtime_shas(workspace = None):
    shas = { }
    for rel in REPO_RUNTIME_SHA_PATHS:
        path = _repo_path(workspace, rel)
        if not path.is_file():
            raise SnapshotError('missing_runtime_source', f'''missing runtime source: {rel}''')
        shas[rel] = sha256_file(path)
    return shas


def required_source_sha_env(shas = None):
    return (lambda .0: pass# WARNING: Decompyle incomplete
)(sorted(shas.items())())


def verify_eval_roundtrip_profile(workspace = None, profile = None):
    sys.path.insert(0, str(workspace / 'src'))
    
    try:
        get_profile = get_profile
        import tac.profiles
        
        try:
            sys.path.remove(str(workspace / 'src'))
            cfg = dict(get_profile(profile))
            if cfg.get('eval_roundtrip') is not True:
                raise SnapshotError('eval_roundtrip_not_proven', f'''profile {profile!r} does not declare eval_roundtrip=True''')
            return {
                'profile': profile,
                'eval_roundtrip': True,
                'source': 'tac.profiles.get_profile' }
        except ValueError:
            continue
            sys.path.remove(str(workspace / 'src'))




def checkpoint_profile_hint(checkpoint = None):
    '''Best-effort profile lookup without requiring CUDA.

    Returns None for legacy/raw state_dict checkpoints. The profile argument is
    still used as the authoritative roundtrip proof in that case.
    '''
    
    try:
        import torch
        state = torch.load(checkpoint, map_location = 'cpu', weights_only = False)
        if not isinstance(state, dict):
            return None
        if not state.get('__meta__'):
            state.get('__meta__')
            if not state.get('arch_meta'):
                state.get('arch_meta')
        meta = { }
        if isinstance(meta, dict) and isinstance(meta.get('profile'), str):
            return meta['profile']
    except Exception:
        return None



def _checkpoint_meta_candidates(payload = None):
    if not isinstance(payload, dict):
        return []
    candidates = None
    for key in ('qfaithful_training_pose_contract', 'training_pose_contract', 'snapshot_training_pose_contract', 'arch_meta', '__meta__', 'meta'):
        value = payload.get(key)
        if not isinstance(value, dict):
            continue
        candidates.append(value)
    for container_key in ('arch_meta', '__meta__', 'meta'):
        container = payload.get(container_key)
        if not isinstance(container, dict):
            continue
        for key in ('qfaithful_training_pose_contract', 'training_pose_contract', 'snapshot_training_pose_contract'):
            value = container.get(key)
            if not isinstance(value, dict):
                continue
            candidates.append(value)
    return candidates


def _contract_bool(contract = None, *keys):
    pass
# WARNING: Decompyle incomplete


def inspect_checkpoint_training_pose_contract(checkpoint = None, *, deployed_pose_contract, profile):
    '''Require proof that Q-FAITHFUL training used the deployed pose stream.'''
    base = {
        'checkpoint_path': str(checkpoint),
        'profile': profile,
        'required_for_qfaithful_successor': True,
        'required_pose_dim': 6,
        'zero_pose_fallback_allowed': False,
        'training_pose_contract_promotable': False }
# WARNING: Decompyle incomplete


def select_state_dict(payload = None, state_source = None):
    if not isinstance(payload, dict):
        if state_source not in frozenset({'raw', 'auto'}):
            raise SnapshotError('checkpoint_state_missing', 'checkpoint is not a dict')
        return (payload, 'raw')
    priority = None
    if state_source != 'auto':
        priority = ((state_source, state_source),)
    for key, label in priority:
        value = payload.get(key)
        if not isinstance(value, dict):
            continue
        
        return priority, (value, label)
    if state_source == 'auto' and payload and (lambda .0: pass# WARNING: Decompyle incomplete
)(payload.values()()):
        return (payload, 'raw')
    raise all('checkpoint_state_missing', f'''checkpoint lacks requested state source {state_source!r}''')


def export_qfai_renderer(checkpoint = None, renderer_bin = None, *, state_source, brotli_quality, extra_meta):
    import brotli
    import torch
    save_qfai = save_qfai
    import tac.quantizr_faithful_export
    build_quantizr_faithful_renderer = build_quantizr_faithful_renderer
    import tac.quantizr_faithful_renderer
    payload = torch.load(checkpoint, map_location = 'cpu', weights_only = False)
    (state_dict, selected_source) = select_state_dict(payload, state_source)
# WARNING: Decompyle incomplete


def enforce_ema_export_contract(qfai_meta = None, *, allow_live_weight_export):
    if qfai_meta.get('packed_from_ema_shadow') is True:
        return None
    if allow_live_weight_export:
        return None
    raise SnapshotError('ema_shadow_export_missing', f'''refusing Q-FAITHFUL eval because export did not pack ema_shadow; selected={qfai_meta.get('checkpoint_state_source')!r}''')


def build_raw_archive(*, renderer_bin, masks_mkv, poses_pt, output_archive, zoom_warp_path):
    for path, failure in ((renderer_bin, 'missing_renderer_bin'), (masks_mkv, 'missing_masks_mkv'), (poses_pt, 'missing_poses_pt')):
        if path.is_file():
            continue
        raise SnapshotError(failure, f'''missing required archive input: {path}''')
    output_archive.parent.mkdir(parents = True, exist_ok = True)
    sources = {
        'renderer.bin': renderer_bin,
        'masks.mkv': masks_mkv,
        'optimized_poses.bin': poses_pt }
# WARNING: Decompyle incomplete


def archive_metadata(path = None):
    meta = {
        'path': str(path),
        'bytes': _stat_size(path),
        'sha256': sha256_file(path) }
# WARNING: Decompyle incomplete


def validate_repacked_geometry_contract(*, screen_contract, repacked_archive_meta):
    if not screen_contract.get('zoom_warp_geometry'):
        screen_contract.get('zoom_warp_geometry')
    zoom_contract = { }
    if not zoom_contract.get('required_for_half_frame'):
        return None
    if not zoom_contract.get('source'):
        zoom_contract.get('source')
    source = { }
    if not source.get('present'):
        return None
    if not zoom_contract.get('archive_member_name'):
        zoom_contract.get('archive_member_name')
    member_name = ZOOM_WARP_ARCHIVE_MEMBER
    if not repacked_archive_meta.get('members'):
        repacked_archive_meta.get('members')
    members = { }
    member = members.get(member_name)
    if not member:
        raise SnapshotError('zoom_warp_geometry_not_preserved_by_repack_contract', f'''repacked archive missing required zoom/warp geometry member {member_name!r}''')
    if member.get('sha256') != source.get('sha256'):
        raise SnapshotError('zoom_warp_geometry_sha_mismatch', f'''repacked archive member {member_name!r} does not match supplied zoom/warp geometry''')


def _optional_path_metadata(path = None):
    pass
# WARNING: Decompyle incomplete


def _env_truthy(name = None):
    return os.environ.get(name, '').strip().lower() in frozenset({'1', 'y', 'on', 'yes', 'true'})


def _tensor_summary(value = None):
    pass
# WARNING: Decompyle incomplete


def _select_pose_tensor(payload = None):
    
    try:
        import torch
        if torch.is_tensor(payload):
            return (payload, 'raw_tensor')
        if None(payload, dict):
            for key in ('poses', 'optimized_poses', 'pose_tensor', 'pose', 'optimized_poses_bin'):
                value = payload.get(key)
                if not torch.is_tensor(value):
                    continue
                
                return ('poses', 'optimized_poses', 'pose_tensor', 'pose', 'optimized_poses_bin'), (value, key)
            for key, value in payload.items():
                if not torch.is_tensor(value):
                    continue
                
                return payload.items(), (value, str(key))
        raise SnapshotError('pose_tensor_missing', 'pose file did not contain a tensor')
    except Exception:
        raise SnapshotError('torch_unavailable_for_pose_contract', str(exc)), exc
        None = None
        del exc



def inspect_pose_tensor_contract(path = None):
    '''Inspect deployed pose tensors and reject zero-pose/silent-fallback inputs.'''
    meta = _optional_path_metadata(path)
# WARNING: Decompyle incomplete


def enforce_pose_tensor_contract(contract = None, *, allow_unproven):
    if contract.get('promotable_pose_contract') is True:
        return None
    if allow_unproven and contract.get('failure_class') not in frozenset({'pose_tensor_empty', 'missing_pose_tensor', 'pose_tensor_all_zero'}):
        return None
    if not contract.get('failure_class'):
        contract.get('failure_class')
    raise SnapshotError('pose_tensor_contract_missing', f'''refusing Q-FAITHFUL eval with unproven pose tensor contract: {contract}''')


def enforce_checkpoint_training_pose_contract(contract = None):
    if contract.get('training_pose_contract_promotable') is True:
        return None
    if not contract.get('failure_class'):
        contract.get('failure_class')
    raise SnapshotError('qfaithful_training_pose_contract_missing', f'''refusing Q-FAITHFUL eval with unproven training pose contract: {contract}''')


def _ffprobe_frame_count(path = None):
    ffprobe = shutil.which('ffprobe')
# WARNING: Decompyle incomplete


def inspect_mask_frame_contract(masks_mkv = None, *, declared_contract):
    if declared_contract not in frozenset({'auto', 'full', 'half'}):
        raise SnapshotError('invalid_mask_frame_contract', f'''unsupported mask frame contract: {declared_contract!r}''')
    if declared_contract != 'auto':
        frames = EXPECTED_MASK_FRAMES if declared_contract == 'full' else HALF_FRAME_MASK_FRAMES
        return {
            'contract': declared_contract,
            'frame_count': frames,
            'source': 'operator_declared',
            'masks_mkv': _optional_path_metadata(masks_mkv),
            'expected_full_frame_count': EXPECTED_MASK_FRAMES,
            'expected_half_frame_count': HALF_FRAME_MASK_FRAMES }
    (frames, evidence) = None(masks_mkv)
    if frames == EXPECTED_MASK_FRAMES:
        contract = 'full'
    elif frames == HALF_FRAME_MASK_FRAMES:
        contract = 'half'
    else:
        contract = 'unknown'
    return {
        'contract': contract,
        'frame_count': frames,
        'source': 'ffprobe',
        'ffprobe': evidence,
        'masks_mkv': _optional_path_metadata(masks_mkv),
        'expected_full_frame_count': EXPECTED_MASK_FRAMES,
        'expected_half_frame_count': HALF_FRAME_MASK_FRAMES }


def qfaithful_renderer_zoom_contract():
    '''Return the static zoom-consumption contract for this snapshot exporter.

    The Q-FAITHFUL snapshot path currently serializes
    ``tac.quantizr_faithful_renderer.JointFrameGenerator`` through QFAI/QZS3.
    That public-floor architecture has no ``use_zoom_flow``/``ego_flow`` input,
    but the contest inflate runtime can still consume charged
    ``zoom_scalars.bin`` for half-frame mask expansion before invoking the
    renderer.  This distinction matters: zoom geometry need not be a renderer
    input to be score-affecting and contest-compliant.
    '''
    return {
        'architecture': QFAITHFUL_RENDERER_ARCHITECTURE,
        'detection': 'static_export_builder',
        'consumes_zoom_warp': True,
        'renderer_consumes_ego_flow': False,
        'runtime_consumes_zoom_warp_for_mask_expansion': True,
        'consumption_proof': 'submissions/robust_current/inflate_renderer.py loads charged zoom_scalars.bin whenever half-frame masks are present, even if renderer.use_zoom_flow is false',
        'failure_class_if_required': 'zoom_warp_geometry_not_consumed_by_runtime',
        'unblock_requirement': 'half-frame masks with charged zoom geometry require an archive runtime that preserves zoom_scalars.bin and uses it for half-frame mask expansion before renderer invocation' }


def build_snapshot_screen_contract(args = None):
    mask_contract = inspect_mask_frame_contract(args.masks_mkv, declared_contract = args.mask_frame_contract)
    zoom_warp_path = args.zoom_warp_path.resolve() if args.zoom_warp_path else None
    zoom_meta = _optional_path_metadata(zoom_warp_path)
    renderer_zoom_contract = qfaithful_renderer_zoom_contract()
    poses_path = getattr(args, 'poses_pt', None)
# WARNING: Decompyle incomplete


def enforce_exact_screen_contract(contract = None, *, eval_mode):
    if eval_mode != 'run':
        return None
    if contract.get('promotable_exact_screen') is True:
        return None
    if not contract.get('non_promotable_reasons'):
        contract.get('non_promotable_reasons')
    reasons = ', '.join([
        'unknown_contract_failure'])
    raise SnapshotError('non_promotable_runtime_contract', f'''refusing exact screen run for non-promotable Q-FAITHFUL snapshot contract: {reasons}''')


def qfai_export_contract_metadata(*, checkpoint, checkpoint_sha, profile, screen_contract, training_pose_contract, packed_from_ema_shadow):
    return {
        'checkpoint_path': str(checkpoint),
        'checkpoint_sha256': checkpoint_sha,
        'runtime_contract_version': SNAPSHOT_RUNTIME_CONTRACT_VERSION,
        'mask_frame_contract': screen_contract['mask_frame_contract']['contract'],
        'renderer_zoom_contract': screen_contract['renderer_zoom_contract'],
        'eval_roundtrip_required': True,
        'profile': profile,
        'promotable_exact_screen': bool(screen_contract['promotable_exact_screen']),
        'non_promotable_reasons': list(screen_contract['non_promotable_reasons']),
        'pose_tensor_contract': screen_contract.get('pose_tensor_contract'),
        'training_pose_contract': training_pose_contract,
        'packed_from_ema_shadow': bool(packed_from_ema_shadow) }


def build_repack_command(*, python_bin, workspace, source_archive, output_dir, output_archive, renderer_codec, qzs3_block_size, submission_layout, pose_codec, pose_residual_topk, brotli_quality):
    return [
        python_bin,
        str(workspace / 'experiments' / 'repack_quantizr_faithful_qzs3_archive.py'),
        '--source-archive',
        str(source_archive),
        '--output-dir',
        str(output_dir),
        '--output-archive',
        str(output_archive),
        '--renderer-codec',
        renderer_codec,
        '--qzs3-block-size',
        str(qzs3_block_size),
        '--submission-layout',
        submission_layout,
        '--pose-codec',
        pose_codec,
        '--pose-residual-topk',
        str(pose_residual_topk),
        '--brotli-quality',
        str(brotli_quality)]


def build_eval_invocation(*, workspace, archive_path, archive_label, log_dir, predicted_low, predicted_high, controlled_baseline, source_shas, eval_script):
    env = {
        'WORKSPACE': str(workspace),
        'ARCHIVE_PATH': str(archive_path),
        'ARCHIVE_LABEL': archive_label,
        'LOG_DIR': str(log_dir),
        'PREDICTED_LOW': str(predicted_low),
        'PREDICTED_HIGH': str(predicted_high),
        'CONTROLLED_BASELINE': controlled_baseline,
        'REQUIRED_SOURCE_SHA256S': required_source_sha_env(source_shas) }
    return ([
        'bash',
        str(eval_script)], env)


def build_claim_command(*, workspace, lane_id, platform, instance_job_id, agent, predicted_eta_utc, child_of, parallel_reason):
    cmd = [
        sys.executable,
        str(workspace / 'tools' / 'claim_lane_dispatch.py'),
        'claim',
        '--lane-id',
        lane_id,
        '--platform',
        platform,
        '--instance-job-id',
        instance_job_id,
        '--agent',
        agent,
        '--predicted-eta-utc',
        predicted_eta_utc,
        '--status',
        'eval',
        '--notes',
        'bounded_q_faithful_snapshot_h100_screen']
    if child_of or parallel_reason:
        if not child_of or parallel_reason:
            raise SnapshotError('dispatch_claim_incomplete', '--dispatch-child-of and --dispatch-parallel-reason must be paired')
        cmd.extend([
            '--allow-parallel',
            '--child-of',
            child_of,
            '--parallel-reason',
            parallel_reason])
    return cmd


def write_manifest(path = None, payload = None):
    base = {
        'schema_version': 1,
        'tool': 'scripts/q_faithful_snapshot_loop.py',
        'recorded_at_utc': utc_now(),
        'score_claim': False,
        'score_claim_reason': 'snapshot screen is non-claiming; exact CUDA JSON is required before any claim',
        'exact_cuda_json_required': True,
        'all_score_affecting_bits_inside_archive': True,
        'sidecars_score_affecting': False }
    base.update(payload)
    _json_dump(path, base)


def stable_checkpoints(checkpoint_dir = None, *, glob_pattern, min_age_seconds, processed_shas):
    now = time.time()
    candidates = []
    for path in sorted(checkpoint_dir.glob(glob_pattern), key = (lambda p: p.stat().st_mtime)):
        if path.is_file() or path.suffix == '.tmp':
            continue
        if now - path.stat().st_mtime < min_age_seconds:
            continue
        sha = sha256_file(path)
        if sha in processed_shas:
            continue
        candidates.append(path)
    return candidates


def validate_static_inputs(args = None):
    for attr, failure in (('checkpoint_dir', 'missing_checkpoint_dir'), ('masks_mkv', 'missing_masks_mkv'), ('poses_pt', 'missing_poses_pt')):
        path = Path(getattr(args, attr))
        if ok:
            continue
        raise SnapshotError(failure, f'''{attr.replace('_', '-')} not found: {path}''')
    if args.eval_mode == 'run' and args.dispatch_claim_mode == 'none':
        raise SnapshotError('dispatch_claim_required', '--eval-mode run requires a dispatch claim; use claim or already-claimed')
    if args.dispatch_claim_mode == 'already-claimed':
        if not args.existing_dispatch_claim_id:
            raise SnapshotError('dispatch_claim_required', '--dispatch-claim-mode already-claimed requires --existing-dispatch-claim-id')
        return None


def _run(cmd = None, *, env, cwd):
    merged = os.environ.copy()
    if env:
        merged.update(env)
    subprocess.run(cmd, cwd = str(cwd) if cwd else None, env = merged, check = True)


def process_checkpoint(args = None, checkpoint = None, source_shas = None):
    checkpoint_sha = sha256_file(checkpoint)
    snapshot_id = f'''{checkpoint.stem}-{checkpoint_sha[:12]}'''
    snapshot_dir = args.output_root / snapshot_id
    export_dir = snapshot_dir / 'export'
    raw_archive = snapshot_dir / 'raw_qfai' / 'archive.zip'
    repack_dir = snapshot_dir / 'qzs3'
    qzs3_archive = repack_dir / 'archive.zip'
    manifest_path = snapshot_dir / 'snapshot_manifest.json'
    eval_log_dir = snapshot_dir / 'h100_exact_screen'
    eval_label = f'''{args.archive_label_prefix}_{snapshot_id}'''
    export_command = [
        args.python_bin,
        str(Path('scripts') / 'q_faithful_snapshot_loop.py'),
        '--checkpoint-dir',
        str(args.checkpoint_dir),
        '--masks-mkv',
        str(args.masks_mkv),
        '--poses-pt',
        str(args.poses_pt),
        '--output-root',
        str(args.output_root),
        '--profile',
        args.profile,
        '--max-snapshots',
        '1']
    repack_command = build_repack_command(python_bin = args.python_bin, workspace = args.workspace, source_archive = raw_archive, output_dir = repack_dir, output_archive = qzs3_archive, renderer_codec = args.renderer_codec, qzs3_block_size = args.qzs3_block_size, submission_layout = args.submission_layout, pose_codec = args.pose_codec, pose_residual_topk = args.pose_residual_topk, brotli_quality = args.brotli_quality)
    (eval_command, eval_env) = build_eval_invocation(workspace = args.workspace, archive_path = qzs3_archive, archive_label = eval_label, log_dir = eval_log_dir, predicted_low = args.predicted_low, predicted_high = args.predicted_high, controlled_baseline = args.controlled_baseline, source_shas = source_shas, eval_script = args.eval_script)
    screen_contract = build_snapshot_screen_contract(args)
    training_pose_contract = inspect_checkpoint_training_pose_contract(checkpoint, deployed_pose_contract = screen_contract.get('pose_tensor_contract'), profile = args.profile)
# WARNING: Decompyle incomplete


def parse_args(argv = None):
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--workspace', type = Path, default = Path.cwd())
    parser.add_argument('--python-bin', default = sys.executable)
    parser.add_argument('--checkpoint-dir', type = Path, required = True)
    parser.add_argument('--checkpoint-glob', default = '*.pt')
    parser.add_argument('--min-checkpoint-age-seconds', type = float, default = 60)
    parser.add_argument('--masks-mkv', type = Path, required = True)
    parser.add_argument('--mask-frame-contract', choices = ('auto', 'full', 'half'), default = 'auto', help = 'Expected masks.mkv frame contract. auto uses ffprobe; exact screens fail closed if the contract is unknown or half-frame without charged zoom/warp geometry.')
    parser.add_argument('--zoom-warp-path', type = Path, default = None, help = 'Optional zoom/warp geometry source for half-frame masks. Half-frame snapshots remain non-promotable unless the runtime contract proves the renderer consumes this charged geometry member.')
    parser.add_argument('--poses-pt', type = Path, required = True)
    parser.add_argument('--output-root', type = Path, required = True)
    parser.add_argument('--profile', default = 'q_faithful_dilated_88k')
    parser.add_argument('--state-source', choices = ('auto', 'model_state_dict', 'ema_shadow', 'model', 'state_dict', 'raw'), default = 'auto')
    parser.add_argument('--allow-live-weight-export', action = 'store_true', default = _env_truthy('QFAITHFUL_ALLOW_LIVE_WEIGHT_EXPORT'), help = 'Explicit diagnostic escape hatch for non-EMA exports. Exact eval runs fail closed unless the checkpoint export packed ema_shadow.')
    parser.add_argument('--allow-unproven-pose-custody', action = 'store_true', default = _env_truthy('QFAITHFUL_ALLOW_UNPROVEN_POSE_CUSTODY'), help = 'Explicit diagnostic escape hatch for unreadable/legacy pose custody. Missing, empty, and all-zero pose tensors still fail.')
    parser.add_argument('--renderer-codec', choices = ('qzs3', 'qzs4', 'torch_fp4'), default = 'qzs3')
    parser.add_argument('--qzs3-block-size', type = int, default = 32)
    parser.add_argument('--submission-layout', choices = ('multi_member', 'rpk1_single_blob', 'pr64_single_blob', 'pr64_mask_first_single_blob'), default = 'multi_member')
    parser.add_argument('--pose-codec', choices = ('raw', 'pose_fp16_col_delta_v1', 'pose_qpose14_col_delta_v1', 'pose_qp1_v1', 'pose_fp16_velocity_only_v1', 'pose_fp16_velocity_residual_topk_v1'), default = 'raw')
    parser.add_argument('--pose-residual-topk', type = int, default = 0)
    parser.add_argument('--brotli-quality', type = int, default = 11)
    parser.add_argument('--max-snapshots', type = int, default = 1)
    parser.add_argument('--max-idle-polls', type = int, default = 0)
    parser.add_argument('--poll-seconds', type = float, default = 900)
    parser.add_argument('--dry-run', action = 'store_true')
    parser.add_argument('--eval-mode', choices = ('none', 'command', 'run'), default = 'none')
    parser.add_argument('--eval-script', type = Path, default = Path('scripts/remote_archive_only_eval.sh'))
    parser.add_argument('--archive-label-prefix', default = 'qfaithful_snapshot')
    parser.add_argument('--predicted-low', type = float, default = 0)
    parser.add_argument('--predicted-high', type = float, default = 9.99)
    parser.add_argument('--controlled-baseline', default = 'Q-FAITHFUL long-run checkpoint snapshot; non-claiming until exact CUDA JSON')
    parser.add_argument('--dispatch-claim-mode', choices = ('claim', 'already-claimed', 'none'), default = 'claim')
    parser.add_argument('--existing-dispatch-claim-id')
    parser.add_argument('--dispatch-lane-id', default = 'q_faithful_snapshot')
    parser.add_argument('--dispatch-platform', default = 'h100')
    parser.add_argument('--dispatch-agent', default = 'codex:q_faithful_snapshot_loop')
    parser.add_argument('--dispatch-predicted-eta-utc', default = utc_now())
    parser.add_argument('--dispatch-child-of')
    parser.add_argument('--dispatch-parallel-reason')
    return parser.parse_args(argv)


def main(argv = None):
    args = parse_args(argv)
    args.workspace = args.workspace.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.masks_mkv = args.masks_mkv.resolve()
# WARNING: Decompyle incomplete

if __name__ == '__main__':
    
    try:
        raise SystemExit(main())
        return None
    except SnapshotError:
        exc = None
        print(f'''FATAL[{exc.failure_class}]: {exc}''', file = sys.stderr)
        raise SystemExit(2)
        exc = None
        del exc


"""
