"""Retain a pinned SJ1 overlay with durable, append-only frame checkpoints.

The historical parse-back cannot meet the per-frame resume/quiet-window contract;
that kind refuses before importing or launching its receiver. No scorer is loaded.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import ddm_vr7_rebuild as vr7

GO = Path('/Volumes/VertigoDataTier/pact/MAIN_GO_CUSTODY_AUDIT')
RETAINED = Path('/Volumes/VertigoDataTier/pact/ddm_vr8_audit/retained')
RESERVE = 8 * 2**30


def safe_path(path: Path) -> None:
    """Refuse symlink routing, including existing ancestor symlinks."""
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('absolute canonical path required')
    for node in (path, *path.parents):
        if node.is_symlink():
            raise ValueError(f'symlink refused: {node}')


def go() -> None:
    if not GO.is_file() or GO.is_symlink():
        raise ValueError('MAIN_GO_CUSTODY_AUDIT required')


def persist(path: Path, value: dict) -> None:
    """Write a new immutable receipt; interrupted receipts remain and block."""
    safe_path(path)
    with path.open('x') as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def reserve(output: Path, remaining: int) -> None:
    stat = os.statvfs(output.parent)
    if stat.f_bavail * stat.f_frsize < remaining + RESERVE:
        raise ValueError('output tier lacks remaining payload bytes plus 8 GiB reserve')


def prefix(output: Path, root: Path, binding: dict, pairs: list[int], frame_bytes: int) -> int:
    """Verify only checkpointed full frames; never discard an uncommitted suffix."""
    paths = sorted(root.glob('frame_*.json'))
    if len(paths) > len(pairs):
        raise ValueError('too many frame checkpoints')
    for slot, path in enumerate(paths):
        if path.name != f'frame_{slot:04d}.json':
            raise ValueError('frame checkpoint gap')
    expected = len(paths) * frame_bytes
    if not output.exists():
        if paths:
            raise ValueError('checkpointed payload missing')
        return 0
    safe_path(output)
    if not output.is_file() or output.stat().st_size != expected:
        raise ValueError('uncommitted or incomplete payload suffix retained; refusing truncation')
    with output.open('rb') as handle:
        for slot, path in enumerate(paths):
            go()
            safe_path(path)
            record = json.loads(path.read_text())
            chunk = handle.read(frame_bytes)
            if record != {'binding': binding, 'pair': pairs[slot], 'slot': slot,
                          'bytes': frame_bytes, 'sha256': hashlib.sha256(chunk).hexdigest()}:
                raise ValueError('frame checkpoint or payload drift')
    return len(paths)


def render_overlay(source: dict):
    """Load the original SJ1 field and actual JG1 renderer without load_body/scorers."""
    import numpy as np
    import torch

    from experiments import ddm_jg1_seg_solve as jg1
    from experiments import ddm_sj1_joint_admission as sj1

    random.seed(20260910)
    np.random.seed(20260910)
    torch.manual_seed(20260910)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    planes = sj1.load_edit_planes(Path(source['field']))
    semantic = jg1.load_semantic_renderer(
        archive_path=Path(source['archive']), runtime_dir=Path(source['runtime']) / 'runtime')

    def frame(pair):
        return jg1.render_frame1(semantic, planes[pair][None], np.array([pair]))[0].tobytes()

    return sorted(planes), jg1.CAMERA_H * jg1.CAMERA_W * 3, frame


def restore_frames(source: dict, output: Path, root: Path, binding: dict) -> int:
    go()
    pairs, frame_bytes, render = render_overlay(source)
    if pairs != list(range(600)) or len(pairs) * frame_bytes != source['expected_bytes']:
        raise ValueError('original n600 overlay geometry required')
    start = prefix(output, root, binding, pairs, frame_bytes)
    reserve(output, source['expected_bytes'] - start * frame_bytes)
    # A completed result is verified again rather than trusting its scalar receipt.
    if not output.exists():
        with output.open('xb') as handle:
            handle.flush()
            os.fsync(handle.fileno())
    with output.open('ab') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for slot in range(start, len(pairs)):
            go()
            reserve(output, source['expected_bytes'] - slot * frame_bytes)
            payload = render(pairs[slot])
            if len(payload) != frame_bytes:
                with (root / f'malformed_frame_{slot:04d}.bin').open('xb') as evidence:
                    evidence.write(payload)
                    evidence.flush()
                    os.fsync(evidence.fileno())
                raise ValueError('rendered frame size drift; malformed payload retained')
            if output.stat().st_size != slot * frame_bytes:
                with (root / f'blocked_frame_{slot:04d}.bin').open('xb') as evidence:
                    evidence.write(payload)
                    evidence.flush()
                    os.fsync(evidence.fileno())
                raise ValueError('payload size changed during render; rendered bytes retained')
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            persist(root / f'frame_{slot:04d}.json', {
                'binding': binding, 'pair': pairs[slot], 'slot': slot,
                'bytes': frame_bytes, 'sha256': hashlib.sha256(payload).hexdigest()})
    go()
    vr7.verify(source)
    rebuilt = vr7.fact(output)
    matches = (rebuilt['bytes'], rebuilt['sha256']) == (
        source['expected_bytes'], source['expected_sha256'])
    certificate = {'schema': 'ddm_vr8.restore_certificate.v1', 'complete': matches,
                   'kind': 'sj1_overlay', 'config': binding, 'rebuilt': rebuilt,
                   'expected_sha256': source['expected_sha256'], 'score_claim': False,
                   'hygiene': 'retain every output and frame receipt; no cleanup permitted',
                   'consumer_store': binding['consumer_store'],
                   'seed': 20260910, 'threads': 4, 'original_inputs_verified': True}
    path = root / 'RESTORE_CERTIFICATE.json'
    safe_path(path)
    if path.exists():
        if json.loads(path.read_text()) != certificate:
            raise ValueError('existing certificate drift; retained')
    else:
        persist(path, certificate)
    return 0 if matches else 3


def run(config_path: Path) -> int:
    safe_path(config_path)
    config = json.loads(config_path.read_text())
    if config.get('schema') != 'ddm_vr8.restore.v1':
        raise ValueError('restore schema required')
    if config.get('kind') == 'sj1_parseback':
        raise ValueError('RAW_PARSEBACK_NOT_FRAME_RESUMABLE: SJ1 rewrites build/data and F26 '
                         'restarts interrupted render; no per-frame GO checks; no launch')
    if config.get('kind') != 'sj1_overlay' or config.get('seed') != 20260910 or config.get('threads') != 4:
        raise ValueError('only pinned deterministic SJ1 overlay configuration admitted')
    output, root = Path(config['output']), Path(config['checkpoint_dir'])
    safe_path(output)
    safe_path(root)
    if root.parent != RETAINED or root.name.startswith('.'):
        raise ValueError('checkpoint directory must be private VR8 retained child')
    go()
    expected = config['source_config']
    source_path = Path(expected['path'])
    safe_path(source_path)
    if vr7.fact(source_path) != expected:
        raise ValueError('source config pin drift')
    source = json.loads(source_path.read_text())
    if str(output) != source['original'] or not str(output).startswith(
            '/Volumes/APDataStore/pact/cold_store_sj1_bulk_20260910/'):
        raise ValueError('output must be exact historical SJ1 destination')
    if not isinstance(config.get('consumer_store'), str) or not config['consumer_store']:
        raise ValueError('named consumer store required')
    reserve(output, max(0, source['expected_bytes'] - (output.stat().st_size if output.exists() else 0)))
    vr7.verify(source)
    root.mkdir(parents=True, exist_ok=True)
    binding = {'config': vr7.fact(config_path), 'source_config': expected,
               'runner': vr7.fact(Path(__file__).absolute()),
               'consumer_store': config['consumer_store']}
    lock_path = root / 'LOCK'
    safe_path(lock_path)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = root / 'BINDING.json'
        safe_path(receipt)
        if receipt.exists():
            if json.loads(receipt.read_text()) != binding:
                raise ValueError('resume binding drift')
        elif output.exists():
            raise ValueError('unowned historical destination exists')
        else:
            persist(receipt, binding)
        return restore_frames(source, output, root, binding)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-from', type=Path, required=True)
    args = parser.parse_args()
    return run(args.resume_from.absolute())


if __name__ == '__main__':
    raise SystemExit(main())
