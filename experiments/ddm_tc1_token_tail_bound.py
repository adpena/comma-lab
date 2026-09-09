#!/usr/bin/env python3
"""Trace the live HPAC/RC64 rows without modifying its probability mechanism.

Research-only, scorer-free. The control must reproduce the pinned stream before
any bound can be admitted. All rows, streams and restart checkpoints are kept on
the SSD. A trace can stop at any pair boundary and resume through jg2's complete
corrector/encoder checkpoint. No runtime or upstream source is edited.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiments import ddm_jg2_tail_reencode as jg2

ROOT = Path('/Volumes/VertigoDataTier/pact/ddm_tc1_tail_shared_mixer')
POINTER = REPO / '.omx/state/canonical_frontier_pointer.json'
AXIS = '[macOS-CPU advisory / scorer-free EXACT byte measurement]'
N, H, W, K = 600, 384, 512, 5
SEED = 20260909


def fact(path: Path) -> dict:
    return jg2.file_fact(path)


def pinned(root: Path) -> dict:
    """Refuse stale pointer before every encode; do not silently rebase a trace."""
    record = json.loads((root / 'INPUTS.json').read_text())
    pointer = json.loads(POINTER.read_text())['our_local_frontier_contest_cuda']
    if pointer['archive_sha256'] != record['archive']['sha256']:
        raise RuntimeError('POINTER_MOVED: rebase into a distinct store generation')
    if fact(root / 'source_runtime/archive.zip') != record['archive']:
        raise RuntimeError('pinned archive custody changed')
    if fact(Path(record['field']['path'])) != record['field']:
        raise RuntimeError('source field custody changed')
    return record


def initialize(args) -> dict:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'INPUTS.json').exists():
        return pinned(root)
    source = Path(args.runtime)
    field = Path(args.field)
    if shutil.disk_usage(root).free < 6 * 1024**3:
        raise RuntimeError('storage preflight: need 6 GiB on assigned SSD tier')
    source_archive = fact(source / 'archive.zip')
    pointer = json.loads(POINTER.read_text())
    if source_archive['sha256'] != pointer['our_local_frontier_contest_cuda']['archive_sha256']:
        raise RuntimeError('source runtime archive is not the live pointer')
    destination = root / 'source_runtime'
    # Runtime source is small; copy once so imports/native builds cannot touch MAIN.
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.so', '*.dylib'))
    sys.dont_write_bytecode = True
    sections = jg2.split_member(jg2.read_archive_member(destination / 'archive.zip'))
    census = {}
    for name, payload in sections.items():
        path = root / 'retained/source' / (name + '.bin')
        jg2.atomic_write(path, payload)
        census[name] = fact(path)
    field_fact = fact(field)
    if field_fact['bytes'] != N * H * W:
        raise RuntimeError('source field must contain all 600 planes')
    sources = {str(p.relative_to(destination)): fact(p) for p in sorted(destination.rglob('*')) if p.is_file()}
    result = dict(schema='ddm_tc1_inputs.v1', axis=AXIS, score_claim=False,
                  seed=SEED, archive=fact(destination / 'archive.zip'),
                  source_archive=source_archive, field=field_fact, census=census,
                  pointer=pointer, sources=sources,
                  storage=dict(free_bytes=shutil.disk_usage(root).free,
                               artifact_budget_bytes=6 * 1024**3,
                               cleanup_policy='retain all payloads; no deletion without certification'))
    jg2.atomic_json(root / 'INPUTS.json', result)
    return result


def adopt_identical_tail(args) -> dict:
    """Reuse immutable trace bytes only after a strict tail/runtime identity census."""
    root, previous = Path(args.root), Path(args.reuse_from)
    current = pinned(root)
    jg2.atomic_json(root / 'ADOPTION_INTENT.json', dict(source=str(previous), destination=str(root)))
    old = json.loads((previous / 'INPUTS.json').read_text())
    if current['field'] != old['field']:
        raise RuntimeError('changed field requires a fresh trace')
    for name in ('hpac', 'tail'):
        if (root / 'retained/source' / (name + '.bin')).read_bytes() != (previous / 'retained/source' / (name + '.bin')).read_bytes():
            raise RuntimeError(f'changed {name} requires a fresh trace')
    a, b = previous / 'source_runtime', root / 'source_runtime'
    source_a = {str(p.relative_to(a)): p for p in a.rglob('*') if p.is_file() and p.suffix in ('.py', '.c', '.sh')}
    source_b = {str(p.relative_to(b)): p for p in b.rglob('*') if p.is_file() and p.suffix in ('.py', '.c', '.sh')}
    if set(source_a) != set(source_b):
        raise RuntimeError('source inventory changed; cannot reuse the trace')
    for relative in source_a:
        content = source_a[relative].read_bytes()
        if relative == 'inflate.py':
            content = content.replace(old['archive']['sha256'].encode(), current['archive']['sha256'].encode())
            content = content.replace(f"ARCHIVE_BYTES = {old['archive']['bytes']}".encode(),
                                      f"ARCHIVE_BYTES = {current['archive']['bytes']}".encode())
        if content != source_b[relative].read_bytes():
            raise RuntimeError(f'non-pin runtime change requires a fresh trace: {relative}')
    traces = sorted(previous.glob('TRACE_*.json'))
    if not traces:
        raise RuntimeError('no completed source stage to adopt')
    terminal = json.loads(traces[-1].read_text())
    through = terminal['frames']
    facts = []
    for folder in ('rows', 'predictor_bound_by_winner/frames', 'statistics', 'predictor_bound_by_winner'):
        for source in sorted((previous / folder).glob('*.npz')):
            if source.stem.startswith('frame_') and int(source.stem.split('_')[1]) >= through:
                continue
            destination = root / folder / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                os.link(source, destination)
            old_receipt = source.with_suffix('.json')
            source_fact = json.loads(old_receipt.read_text()) if source.stem.startswith('frame_') and old_receipt.exists() else fact(source)
            shared_inode = os.path.samefile(source, destination)
            if shared_inode:
                # Both paths reference one immutable inode. Re-reading the same
                # multi-GiB trace four times adds no byte-identity evidence.
                if source.stat().st_size != source_fact['bytes']:
                    raise RuntimeError('source size differs from retained receipt')
                payload = dict(source_fact, path=str(destination.resolve()))
            else:
                payload = fact(destination)
                if (payload['sha256'], payload['bytes']) != (source_fact['sha256'], source_fact['bytes']):
                    raise RuntimeError('adopted immutable payload differs')
            if source.stem.startswith('frame_'):
                jg2.atomic_json(destination.with_suffix('.json'), payload)
            facts.append(dict(source=source_fact, destination=payload,
                              method='same immutable inode; SHA from retained source receipt' if shared_inode else 'existing copy hash compared'))
    copied_trace = dict(terminal, adopted_from=fact(traces[-1]),
                        rebase_admission='identical tail, HPAC, field and runtime except archive pins')
    jg2.atomic_json(root / traces[-1].name, copied_trace)
    result = dict(schema='ddm_tc1_identical_tail_rebase.v1', axis=AXIS, score_claim=False,
                  old_inputs=fact(previous / 'INPUTS.json'), new_inputs=fact(root / 'INPUTS.json'),
                  source_stage=through, source_inventory_files=len(source_a),
                  change_scope='carrier and archive pins only',
                  full_stream_identity_still_required=True, immutable_payloads=facts)
    jg2.atomic_json(root / 'REBASE.json', result)
    return result


class RowObserver:
    """Side-effect-only observer; the original corrector return values pass through."""

    def __init__(self, root: Path, frame: int):
        self.root = root
        self.frame = frame
        self.rows = np.empty((H * W, K), dtype=np.float32)
        self.base = np.empty(H * W, dtype=np.uint8)
        self.context = np.empty(H * W, dtype=np.uint32)
        self.seen = np.zeros(H * W, dtype=bool)
        self.positions = None

    def group(self, state, predicted, flat):
        self.positions = flat
        self.base[flat] = predicted
        self.context[flat] = state.context

    def coding(self, row):
        if row.dtype != np.float32:
            raise RuntimeError('RC64 probability input is expected to be float32')
        self.rows[self.positions] = row
        self.seen[self.positions] = True

    def finish_frame(self):
        if not self.seen.all():
            raise RuntimeError('incomplete coding-row plane')
        path = self.root / 'rows' / f'frame_{self.frame:04d}.npz'
        values = dict(rows=self.rows, base_argmax=self.base, hpac_context=self.context)
        if path.exists():
            with np.load(path, allow_pickle=False) as old:
                if set(old.files) != set(values) or any(not np.array_equal(old[k], v) for k, v in values.items()):
                    raise RuntimeError('resumed trace changed retained rows')
        else:
            jg2.atomic_npz(path, values)
        jg2.atomic_json(path.with_suffix('.json'), fact(path))
        self.frame += 1
        self.seen.fill(False)


def trace(args) -> dict:
    import torch
    root = Path(args.root)
    record = pinned(root)
    if (root / 'ADOPTION_INTENT.json').exists() and not (root / 'REBASE.json').exists():
        raise RuntimeError('trace requires the completed rebase receipt before continuation')
    source_at_launch = dict(instrument=fact(Path(__file__)), dependency=fact(Path(jg2.__file__)),
                            git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                            platform=platform.platform(), machine=platform.machine(), seed=SEED,
                            upstream_evaluator=fact(REPO / 'upstream/evaluate.py'))
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    sys.dont_write_bytecode = True
    stage_root = root / 'trace' / f'through_{args.frames:04d}'
    env = jg2._prepare(SimpleNamespace(store=str(stage_root), runtime_root=str(root / 'source_runtime')), 'tc1_trace')
    work = env['work']
    checkpoint = work / 'encode_control.checkpoint.npz'
    start = 0
    resume_checkpoint = None
    resume_encoder = None
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as d:
            start = int(d['frame'][0])
    else:
        earlier = sorted(p for p in root.glob('TRACE_*.json') if int(p.stem.split('_')[1]) < args.frames)
        if earlier:
            terminal = json.loads(earlier[-1].read_text())['terminal_checkpoint']
            start = terminal['frame']
            resume_checkpoint = Path(terminal['checkpoint']['path'])
            resume_encoder = Path(terminal['encoder_state']['path'])
    if not start < args.frames <= N:
        raise RuntimeError(f'frames must be in {start + 1}..600')
    observer = RowObserver(root, start)
    target = jg2.load_tokens(Path(record['field']['path']))
    from runtime.free_corrector import FreeCorrector
    original_group = FreeCorrector.group_state
    original_coding = FreeCorrector.coding_row
    original_end = FreeCorrector.end_frame

    def group_state(self, probability, predicted, flat):
        state = original_group(self, probability, predicted, flat)
        observer.group(state, predicted, flat)
        return state

    def coding_row(self, state):
        row = original_coding(self, state)
        observer.coding(row)
        return row

    def end_frame(self, tokens):
        original_end(self, tokens)
        observer.finish_frame()

    FreeCorrector.group_state = group_state
    FreeCorrector.coding_row = coding_row
    FreeCorrector.end_frame = end_frame
    # Stages have distinct output paths; the next stage restores the prior
    # terminal state, including all already-emitted bytes and learned tables.
    result = jg2.encode_tail(residual=env['residual'], renderer=env['renderer'],
                            renderer_dir=env['renderer_dir'], parts=env['parts'],
                            target=target,
                            library=env['library'], route_b=env['route_b'], work=work,
                            tag='control', frames=args.frames, checkpoint_every=10,
                            resume=True, retain_terminal_checkpoint=True,
                            resume_checkpoint=resume_checkpoint,
                            resume_encoder_state=resume_encoder)
    stream = Path(result['stream']['path'])
    saved = root / 'retained' / f'control_through_{args.frames:04d}.bin'
    jg2.persist_immutable_bytes(saved, stream.read_bytes(), label='control prefix')
    result['full_control_byte_identical'] = (stream.read_bytes() == env['parts'].token_stream) if args.frames == N else None
    result.update(axis=AXIS, score_claim=False, inputs=record['archive'], rc64_build=env['build'],
                  source_at_launch=source_at_launch,
                  progress_bits_scope='jg2 pre-quantization float probabilities only; ENTROPY_0600.json supplies actual RC64-frequency costs',
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    result_path = root / f'TRACE_{args.frames:04d}.json'
    jg2.atomic_json(result_path, result)
    if args.frames == N and not result['full_control_byte_identical']:
        raise RuntimeError('CONTROL FAILED: trace did not reproduce the shipped RC64 stream')
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['init', 'trace', 'adopt'])
    p.add_argument('--root', default=str(ROOT))
    p.add_argument('--runtime')
    p.add_argument('--field')
    p.add_argument('--reuse-from')
    p.add_argument('--frames', type=int, default=N)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    result = {'init': initialize, 'trace': trace, 'adopt': adopt_identical_tail}[args.stage](args)
    print(json.dumps(result, sort_keys=True))
