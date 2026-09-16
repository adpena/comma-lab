"""External resumable, scorer-free proof runner; never shipped to a reviewer."""
from __future__ import annotations

import argparse
import hashlib
import gzip
import importlib.util
import json
import os
from pathlib import Path
import pickle
import platform
import random
import sys
import time

import numpy as np
import torch

ROOT = Path('/Volumes/APDataStore/pact/ddm_mrs1')
ARCHIVE = Path('/Volumes/APDataStore/pact/ddm_pd6/candidate2/candidate_runtime/archive.zip')
EXPECTED_TOKENS = '4cb0b147bdae8ce0618938453bb6ccc5b7b5a4927d6ad4e1b0bd9dd9ff46119c'
EXPECTED_RAW = '8a14f55a6a8b141501836f511f4222dde75dca21714d923ad865b5f4757ef66b'


def fact(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def capture(value):
    if callable(value):
        return ('callable', None)
    if isinstance(value, np.ndarray):
        return ('array', value.copy())
    if isinstance(value, (str, bytes, int, float, bool, type(None), np.generic)):
        return ('scalar', value)
    if isinstance(value, dict):
        return ('dict', {key: capture(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, [capture(item) for item in value])
    if hasattr(value, '__dict__') or hasattr(value, '__slots__'):
        members = dict(vars(value)) if hasattr(value, '__dict__') else {}
        for base in type(value).__mro__:
            slots = base.__dict__.get('__slots__', ())
            for key in (slots,) if isinstance(slots, str) else slots:
                if key not in {'__dict__', '__weakref__'} and hasattr(value, key):
                    members[key] = getattr(value, key)
        return ('object', (type(value).__name__, {key: capture(item) for key, item in members.items()}))
    raise TypeError(f'unsupported persistent state: {type(value)}')


def restore(current, state):
    kind, value = state
    if kind == 'callable':
        if not callable(current):
            raise ValueError('persistent function structure changed')
        return current
    if kind in {'array', 'scalar'}:
        return value
    if kind == 'dict':
        return {key: restore(current.get(key), item) for key, item in value.items()}
    if kind in {'list', 'tuple'}:
        values = [restore(current[index] if current is not None else None, item) for index, item in enumerate(value)]
        return tuple(values) if kind == 'tuple' else values
    if kind == 'object':
        name, members = value
        if type(current).__name__ != name:
            raise ValueError(f'persistent object changed: {name}')
        for key, item in members.items():
            setattr(current, key, restore(getattr(current, key, None), item))
        return current
    raise ValueError('unknown persistent state kind')


def cleanup_interrupted_renders(run, binding, replacement):
    """Called only after full raw identity is certified, including on reuse."""
    for interrupted in sorted(run.glob('render_interrupted_*.raw')):
        atomic_json(interrupted.with_suffix('.cleanup.json'), dict(binding=binding, raw=fact(interrupted),
            replacement=replacement, reason='Certified interrupted scratch; complete replacement has exact reference identity',
            promotable=False, score_claim=False, argv=sys.argv))
        interrupted.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receiver', type=Path, required=True)
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    parser.add_argument('--resume-from', type=Path, required=True)
    parser.add_argument('--trace', action='store_true')
    parser.add_argument('--stop-after', type=int, default=600)
    args = parser.parse_args()
    if not 1 <= args.stop_after <= 600:
        raise ValueError('stop-after must be in [1, 600]')
    receiver, run, archive = args.receiver.resolve(), args.resume_from.resolve(), args.archive.resolve()
    public_source = Path(__file__).resolve().parents[1] / 'submissions/mrs1/inflate.py'
    if not (receiver.is_relative_to(ROOT) or receiver == public_source) or not run.is_relative_to(ROOT):
        raise ValueError('proof must remain in the charter store')
    run.mkdir(parents=True, exist_ok=True)
    binding = dict(receiver=fact(receiver), archive=fact(archive), seed=20260916,
                   runner=fact(Path(__file__).resolve()), axis='[macOS-CPU advisory]',
                   trace=args.trace, score_claim=False,
                   environment=dict(python=sys.version, executable=sys.executable, numpy=np.__version__,
                                    torch=torch.__version__, platform=platform.platform(), threads=4))
    if binding['archive']['bytes'] != 179286 or binding['archive']['sha256'] != 'aab908d3b32c65582b7de1a3855b67a4f6fdb9373b0dc81e3049ee8164ec3957':
        raise ValueError('archive must be byte-identical to the charter input')
    inputs = run / 'INPUTS.json'
    if inputs.exists() and json.loads(inputs.read_text()) != binding:
        raise ValueError('proof source or archive changed; use a new proof directory')
    atomic_json(inputs, binding)
    identity_path = run / 'IDENTITY.json'
    if identity_path.exists():
        certificate = json.loads(identity_path.read_text())
        if certificate['binding'] != binding or certificate['raw']['sha256'] != EXPECTED_RAW:
            raise ValueError('completed certificate binding mismatch')
        if fact(Path(certificate['tokens']['path'])) != certificate['tokens']:
            raise ValueError('completed token payload changed')
        existing_raw = Path(certificate['raw']['path'])
        if existing_raw.exists():
            if fact(existing_raw) != certificate['raw']:
                raise ValueError('certified raw changed')
            existing_raw.unlink()
        cleanup_interrupted_renders(run, binding, certificate['raw'])
        print(json.dumps(dict(status='complete_reused', receipt=str(identity_path))), flush=True)
        return
    sys.dont_write_bytecode = True
    random.seed(binding['seed'])
    np.random.seed(binding['seed'])
    torch.manual_seed(binding['seed'])
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    coverage_file = run / 'COVERAGE.json'
    coverage = set(json.loads(coverage_file.read_text())['lines']) if coverage_file.exists() else set()
    chosen = sorted(np.random.default_rng(binding['seed']).choice(600, 32, replace=False).tolist())
    traced_pairs = set()

    def trace(frame, event, argument):
        if frame.f_code.co_filename != str(receiver):
            return None
        if event == 'line':
            coverage.add(frame.f_lineno)
        return trace

    def save_coverage():
        atomic_json(coverage_file, dict(receiver=binding['receiver'], lines=sorted(coverage),
                    seed=binding['seed'], planned_pairs=chosen, completed_traced_pairs=sorted(traced_pairs),
                    trace_enabled=args.trace))

    if args.trace:
        sys.settrace(trace)
    spec = importlib.util.spec_from_file_location('minimal_receiver', receiver)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    start = time.monotonic()
    with torch.inference_mode():
        parts, model, basis, coefficients, selector = module.read_models(archive)
        decoder = module.TokenDecoder(parts, torch.device('cpu'))
        latest = run / 'LATEST.json'
        resumed = 0
        prior_seconds = 0.0
        token_path = run / 'tokens.u8'
        if latest.exists():
            receipt = json.loads(latest.read_text())
            checkpoint = Path(receipt['state']['path'])
            if fact(checkpoint) != receipt['state'] or receipt['binding'] != binding:
                raise ValueError('checkpoint custody mismatch')
            # This pickle is our own source-bound state; it never accepts external input.
            with gzip.open(checkpoint, 'rb') as stream:
                state = pickle.load(stream)
            decoder.frame = resumed = state['frame']
            if resumed != receipt['frame'] or not 0 < resumed <= 600:
                raise ValueError('invalid checkpoint frame')
            decoder.previous = torch.from_numpy(state['previous'])
            restore(decoder.corrector, state['corrector'])
            restore(decoder.mixer, state['mixer'])
            restore(decoder.range_decoder, state['range'])
            prior_seconds = receipt['seconds']
            traced_pairs.update(receipt['completed_traced_pairs'])
            if not token_path.exists():
                raise ValueError('retained token output is absent')
            with token_path.open('rb') as stream:
                prefix_sha = hashlib.sha256(stream.read(resumed * 384 * 512)).hexdigest()
            if prefix_sha != receipt['token_prefix_sha256']:
                raise ValueError('retained token prefix changed')
            if decoder.mixer.frame != resumed or decoder.previous.shape != (1, 384, 512) or decoder.previous.dtype != torch.int64:
                raise ValueError('checkpoint phase or previous-plane geometry mismatch')
        output = np.memmap(token_path, mode='r+' if token_path.exists() else 'w+', dtype=np.uint8, shape=(600, 384, 512))
        if resumed and not np.array_equal(decoder.previous.numpy()[0], output[resumed - 1]):
            raise ValueError('checkpoint previous plane differs from retained token output')
        original_latest = ROOT / 'trace/token_stages/LATEST.json'
        reference = None
        if original_latest.exists():
            original = json.loads(original_latest.read_text())
            original_path = Path(original['path'])
            if fact(original_path)['sha256'] != original['sha256']:
                raise ValueError('reference checkpoint hash mismatch')
            if original['binding']['stream'] != hashlib.sha256(parts.token_stream).hexdigest():
                raise ValueError('reference checkpoint belongs to another stream')
            with np.load(original['path'], allow_pickle=False) as bank:
                reference = bank['tokens'].copy()
        while decoder.frame < args.stop_after:
            sys.settrace(trace if args.trace and (decoder.frame == 0 or decoder.frame in chosen) else None)
            frame = decoder.frame
            output[frame] = next(decoder).numpy()
            output.flush()
            if reference is not None and frame < len(reference) and not np.array_equal(output[frame], reference[frame]):
                atomic_json(run / 'TOKEN_MISMATCH.json', dict(frame=frame,
                    unequal=int(np.count_nonzero(output[frame] != reference[frame])),
                    output=fact(token_path), original_checkpoint=original, binding=binding))
                raise ValueError(f'token mismatch at frame {frame}; payload retained')
            if args.trace and frame in chosen:
                traced_pairs.add(frame)
            if decoder.frame % 25 == 0 or decoder.frame == args.stop_after:
                sys.settrace(None)
                state = dict(frame=decoder.frame, previous=decoder.previous.numpy().copy(),
                    corrector=capture(decoder.corrector), mixer=capture(decoder.mixer), range=capture(decoder.range_decoder))
                checkpoint = run / f'stage_{decoder.frame:04d}.pickle.gz'
                temporary = checkpoint.with_suffix('.pending')
                with gzip.GzipFile(filename=str(temporary), mode='wb', mtime=0, compresslevel=3) as stream:
                    pickle.dump(state, stream, protocol=5)
                    stream.flush()
                with temporary.open('rb') as stream:
                    os.fsync(stream.fileno())
                temporary.replace(checkpoint)
                prefix_sha = hashlib.sha256(output[:decoder.frame].tobytes()).hexdigest()
                receipt = dict(binding=binding, frame=decoder.frame, state=fact(checkpoint),
                    completed_traced_pairs=sorted(traced_pairs),
                    token_prefix_sha256=prefix_sha, seconds=prior_seconds + time.monotonic() - start)
                atomic_json(checkpoint.with_suffix('.json'), receipt)
                atomic_json(latest, receipt)
                save_coverage()
                print(json.dumps(dict(frame=decoder.frame, elapsed=receipt['seconds'])), flush=True)
        sys.settrace(None)
        save_coverage()
        if args.stop_after < 600:
            return
        token_fact = fact(token_path)
        if token_fact['sha256'] != EXPECTED_TOKENS:
            raise ValueError('full token hash mismatch; retain bytes')
        if args.trace and len(traced_pairs) < 24:
            raise ValueError('fewer than 24 seeded pairs actually traced')
        destination = run / '0.raw'
        if not destination.exists():
            in_progress = run / 'render_in_progress.raw'
            if in_progress.exists():
                interrupted = in_progress.with_name(f'render_interrupted_{time.time_ns()}.raw')
                interrupted_fact = fact(in_progress)
                atomic_json(interrupted.with_suffix('.json'), dict(binding=binding, raw=interrupted_fact,
                    retained_path=str(interrupted), reason='Interrupted non-authority render scratch; preserve until final raw identity'))
                in_progress.replace(interrupted)
            if args.trace:
                sys.settrace(trace)
            try:
                module.write_video(model, basis, coefficients, torch.from_numpy(np.asarray(output).copy()),
                                   selector, in_progress, torch.device('cpu'))
            finally:
                sys.settrace(None)
                save_coverage()
            in_progress.replace(destination)
        raw = fact(destination)
        if raw['sha256'] != EXPECTED_RAW or raw['bytes'] != 3662409600:
            atomic_json(run / 'RAW_MISMATCH.json', dict(binding=binding, actual=raw, expected=EXPECTED_RAW))
            raise ValueError('raw mismatch; retain bytes')
        with destination.open('rb') as stream:
            pair_sha256 = [hashlib.sha256(stream.read(6104016)).hexdigest() for _ in range(600)]
        certificate = dict(binding=binding, raw=raw, tokens=token_fact, pair_sha256=pair_sha256,
            matched_pairs=600, resumed_from=resumed, traced=args.trace,
            completed_traced_pairs=sorted(traced_pairs),
            seconds=prior_seconds + time.monotonic() - start, argv=sys.argv,
            reason='Raw is rebuildable from source-bound archive and receiver; full SHA equality proven before removal',
            promotable=False, score_claim=False)
        atomic_json(run / 'IDENTITY.json', certificate)
        destination.unlink()
        cleanup_interrupted_renders(run, binding, raw)
        print(json.dumps(dict(status='complete', receipt=str(run / 'IDENTITY.json'))), flush=True)


if __name__ == '__main__':
    main()
