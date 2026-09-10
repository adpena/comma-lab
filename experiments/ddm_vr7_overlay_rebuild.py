"""Scorer-free, per-frame resumable SJ1 overlay rebuild from a seal and retained field.

Uses the same SJ1 field loader and JG1 batch-one renderer as render-edits.
Avoids SJ1 load_body, which unnecessarily loads SegNet and the GT table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.ddm_vr7_rebuild import SCRATCH, atomic, fact, verify


def run(config_path: Path, *, retain_output: bool = False) -> int:
    config = json.loads(config_path.read_text())
    verify(config)
    root = Path(config['scratch'])
    if root.parent != SCRATCH or root.is_symlink() or SCRATCH.is_symlink():
        raise ValueError('overlay scratch must be a private child of VR7 root')
    root.mkdir(exist_ok=True)
    binding = fact(config_path)
    checkpoint = root / 'BINDING.json'
    if checkpoint.exists() and json.loads(checkpoint.read_text()) != binding:
        raise ValueError('overlay resume binding drift')
    output = root / 'odd_frames.u8'
    if output.is_symlink() or (output.exists() and not checkpoint.exists()):
        raise ValueError('unowned overlay output')
    atomic(checkpoint, binding)
    certificate = root / 'REBUILD_CERTIFICATE.json'
    if certificate.exists():
        prior = json.loads(certificate.read_text())
        if prior.get('complete') and prior.get('cleanup_complete') and not retain_output:
            print(json.dumps({'already_complete': str(certificate)}), flush=True)
            return 0
    if os.statvfs(root).f_bavail * os.statvfs(root).f_frsize < 4 * (1 << 30):
        raise ValueError('overlay reserve below 4 GiB')
    import numpy as np
    import torch

    from experiments import ddm_jg1_seg_solve as jg1
    from experiments import ddm_sj1_joint_admission as sj1

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260910)
    np.random.seed(20260910)
    torch.use_deterministic_algorithms(True)
    planes = sj1.load_edit_planes(Path(config['field']))
    semantic = jg1.load_semantic_renderer(
        archive_path=Path(config['archive']), runtime_dir=Path(config['runtime']) / 'runtime')
    pairs = sorted(planes)
    frame_bytes = jg1.CAMERA_H * jg1.CAMERA_W * 3
    if len(pairs) * frame_bytes != config['expected_bytes']:
        raise ValueError('overlay frame geometry differs from receipt')
    progress_path = root / 'PROGRESS.json'
    saved = json.loads(progress_path.read_text()) if progress_path.exists() else {'frames': []}
    if not output.exists() and certificate.exists():
        prior = json.loads(certificate.read_text())
        if prior.get('complete') and prior.get('cleanup_complete'):
            saved = {'frames': []}
    if saved.get('config', binding) != binding:
        raise ValueError('overlay progress binding drift')
    mode = 'r+b' if output.exists() else 'w+b'
    with output.open(mode) as handle:
        # Verify the durable prefix before discarding only a crash-interrupted suffix.
        for slot, record in enumerate(saved['frames']):
            chunk = handle.read(frame_bytes)
            if (len(chunk) != frame_bytes or record['pair'] != pairs[slot]
                    or hashlib.sha256(chunk).hexdigest() != record['sha256']):
                raise ValueError('overlay resume prefix drift')
        if output.stat().st_size > handle.tell():
            interrupted = fact(output)
            atomic(root / 'INTERRUPTED_SUFFIX.json', {
                'original': interrupted, 'retained_prefix_bytes': handle.tell(),
                'reason': 'own uncommitted suffix deterministically regenerated from pinned config',
                'config': binding, 'argv': config['argv'], 'score_claim': False})
            handle.truncate()
        for slot in range(len(saved['frames']), len(pairs)):
            pair = pairs[slot]
            frame = jg1.render_frame1(semantic, planes[pair][None], np.array([pair]))[0]
            payload = frame.tobytes()
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            saved['frames'].append({'pair': pair, 'sha256': hashlib.sha256(payload).hexdigest()})
            saved['config'] = binding
            atomic(progress_path, saved)
            if (slot + 1) % 25 == 0:
                print(f'rendered {slot + 1}/{len(pairs)}', flush=True)
    verify(config)
    original_exists = Path(config['original']).exists()
    original = fact(Path(config['original'])) if original_exists else {
        'path': config['original'], 'bytes': config['expected_bytes'],
        'sha256': config['expected_sha256']}
    rebuilt = fact(output)
    matches = (original['bytes'], original['sha256']) == (
        config['expected_bytes'], config['expected_sha256']) == (rebuilt['bytes'], rebuilt['sha256'])
    result = {'schema': 'ddm_vr7.rebuild.v1', 'kind': 'sj1_overlay',
              'config': binding, 'argv': config['argv'], 'cwd': config['cwd'],
              'original': original, 'rebuilt': rebuilt,
              'twin_hash_equal': matches and original_exists,
              'original_hash_measured_current': original_exists,
              'complete': matches, 'cleanup_complete': False, 'score_claim': False,
              'created_files': [rebuilt], 'supplemental_field': fact(Path(config['field'])),
              'cleanup_reason': 'own successful rebuild; original, seal, field and renderer retained'}
    atomic(certificate, result)
    if not matches:
        return 3
    if retain_output:
        print(json.dumps({'certificate': str(certificate), 'retained_output': str(output)}), flush=True)
        return 0
    if fact(output) != rebuilt:
        raise ValueError('overlay scratch cleanup drift')
    output.unlink()
    result['cleanup_complete'] = True
    atomic(certificate, result)
    print(json.dumps({'certificate': str(certificate), 'twin_hash_equal': True}), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-from', type=Path, required=True)
    parser.add_argument('--retain-output', action='store_true',
                        help='keep regenerated output for a future consumer')
    args = parser.parse_args()
    config_path = args.resume_from.resolve()
    config = json.loads(config_path.read_text())
    if config['argv'] != [sys.executable, '-B', 'experiments/ddm_vr7_overlay_rebuild.py',
                          '--resume-from', str(config_path)]:
        raise ValueError('overlay argv binding drift')
    return run(config_path, retain_output=args.retain_output)


if __name__ == '__main__':
    raise SystemExit(main())
