"""Scorer-free, resumable timings and retained vectors for the one-C receiver.

Baseline samples two seeded uniform pairs in each 25-pair stratum. Causal
replay starts from the source-bound predecessor checkpoint for that stratum.
The native comparison consumes the exact pre-pair states retained here.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import pickle
import platform
import random
import shutil
import sys
import time

import numpy as np
import torch

from ddm_mrs1_validate import capture, restore, fact, atomic_json

ROOT = Path('/Volumes/APDataStore/pact/ddm_mrs2')
PRIOR = Path('/Volumes/APDataStore/pact/ddm_mrs1/final_cold')
REPO = Path(__file__).resolve().parents[1]
SEED = 20260916


def save_state(path, decoder):
    state = dict(frame=decoder.frame, previous=decoder.previous.numpy().copy(),
                 corrector=capture(decoder.corrector), mixer=capture(decoder.mixer),
                 range=capture(decoder.range_decoder))
    temporary = path.with_suffix('.pending')
    with gzip.GzipFile(filename=str(temporary), mode='wb', mtime=0, compresslevel=3) as stream:
        pickle.dump(state, stream, protocol=5)
    temporary.replace(path)
    return fact(path)


def load_state(path, decoder):
    # Only our own SHA-bound checkpoints are accepted by the caller.
    with gzip.open(path, 'rb') as stream:
        state = pickle.load(stream)
    decoder.frame = state['frame']
    decoder.previous = torch.from_numpy(state['previous'])
    restore(decoder.corrector, state['corrector'])
    restore(decoder.mixer, state['mixer'])
    restore(decoder.range_decoder, state['range'])


def install_timers(module):
    totals = defaultdict(float)
    enabled = [False]

    def wrap(cls, name, category):
        original = getattr(cls, name)

        def measured(self, *args, **kwargs):
            if not enabled[0]:
                return original(self, *args, **kwargs)
            start = time.perf_counter()
            try:
                return original(self, *args, **kwargs)
            finally:
                totals[category] += time.perf_counter() - start

        setattr(cls, name, measured)

    wrap(module.ArithmeticDecoder, 'decode', 'arithmetic_and_frequency_table')
    for name in ('begin_frame', 'group_state', 'coding_row', 'observe', 'end_frame'):
        wrap(module.miss_FreeCorrector, name, 'prior_context_statistics')
    for name in ('begin_frame', 'coding', 'observe', 'end_frame'):
        wrap(module.geometry_mixer_LaneMixer, name, 'prior_context_statistics')
    # selected_logits is bound onto each instance by the original optimizer.
    original = module.inference_selected_logits

    def logits(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            if enabled[0]:
                totals['prior_network'] += time.perf_counter() - start

    module.inference_selected_logits = logits
    wrap(module.prior_IntegerPrior, 'prepare_frame_context', 'prior_network')
    return totals, enabled


def render_pair(module, model, normalized_basis, coefficients, modes, labels, token, index):
    """Same CPU batch-one expressions as public rendering and selector application."""
    start = time.perf_counter()
    indices = torch.tensor([index])
    master = module.F.interpolate(model(token.long()[None], indices),
        size=(module.render_CAMERA_H, module.render_CAMERA_W), mode='bilinear',
        align_corners=False).clamp(0.0, 255.0).round()
    master_np = master.to(torch.uint8).permute(0, 2, 3, 1).numpy()[0]
    renderer = time.perf_counter() - start
    start = time.perf_counter()
    carrier = torch.einsum('bk,kchw->bchw', coefficients[index:index + 1], normalized_basis)
    carrier = carrier / math.sqrt(module.render_CARRIER_DIM)
    slave = module.F.interpolate((127.5 + module.render_CARRIER_AMPLITUDE * carrier)
        .clamp(0.0, 255.0).round(), size=(module.render_CAMERA_H, module.render_CAMERA_W),
        mode='bicubic', align_corners=False).clamp(0.0, 255.0).round()
    slave_np = slave.to(torch.uint8).permute(0, 2, 3, 1).numpy()
    slave_np = module.selector_apply_pixel_mode(slave_np.copy(), modes[int(labels[index])])
    pose = time.perf_counter() - start
    return np.stack((slave_np[0], master_np)), renderer, pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-from', type=Path, required=True)
    parser.add_argument('--native', action='store_true')
    args = parser.parse_args()
    run = args.resume_from.resolve()
    if not run.is_relative_to(ROOT) or run == ROOT:
        raise ValueError('proof must be inside the charter store')
    run.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(ROOT).free < 2 * 1024**3:
        raise ValueError('need 2 GiB free for retained profile vectors')
    sys.dont_write_bytecode = True
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    receiver = REPO / ('submissions/mrs2/inflate.py' if args.native else 'submissions/mrs1/inflate.py')
    archive = receiver.with_name('archive.zip')
    baseline = ROOT / 'profile_python'
    binding = dict(receiver=fact(receiver), archive=fact(archive), runner=fact(Path(__file__)),
        helper=fact(REPO / 'experiments/ddm_mrs1_validate.py'), seed=SEED,
        axis='[macOS-CPU advisory]', python=sys.version, numpy=np.__version__,
        torch=torch.__version__, platform=platform.platform(), threads=4, native=args.native,
        score_claim=False, promotable=False)
    if args.native:
        binding['c_source'] = fact(receiver.with_name('range_decoder.c'))
        binding['library'] = fact(receiver.with_name('range_decoder.so'))
    inputs = run / 'INPUTS.json'
    if inputs.exists() and json.loads(inputs.read_text()) != binding:
        raise ValueError('source binding changed; use a new directory')
    atomic_json(inputs, binding)
    rng = np.random.default_rng(SEED)
    strata = [sorted((rng.choice(25, 2, replace=False) + start).tolist()) for start in range(0, 600, 25)]
    reference = json.loads((PRIOR / 'IDENTITY.json').read_text())
    reference_tokens = np.memmap(PRIOR / 'tokens.u8', dtype=np.uint8, mode='r', shape=(600, 384, 512))
    if fact(PRIOR / 'tokens.u8') != reference['tokens']:
        raise ValueError('predecessor token bank changed')
    spec = importlib.util.spec_from_file_location('profile_receiver', receiver)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if args.native and module._RANGE_LIBRARY is None:
        raise ValueError('native path was not loaded')
    totals, enabled = install_timers(module)
    with torch.inference_mode():
        parts, model, basis, coefficients, selector = module.read_models(archive)
        basis = module.render_normalized_basis(basis)
        modes, labels = module.selector_decode_selector(selector)
        rows = []
        for block, selected in enumerate(strata):
            completed = run / f'block_{block:02d}.json'
            if completed.exists():
                saved = json.loads(completed.read_text())
                if saved['binding'] != binding:
                    raise ValueError('completed block binding changed')
                for item in saved['artifacts']:
                    if fact(Path(item['path'])) != item:
                        raise ValueError('retained block payload changed')
                rows.extend(saved['rows'])
                continue
            decoder = module.TokenDecoder(parts, torch.device('cpu'))
            source = None
            if not args.native and block:
                source = PRIOR / f'stage_{block * 25:04d}.pickle.gz'
                receipt = json.loads(source.with_suffix('.json').read_text())
                if fact(source) != receipt['state'] or receipt['binding'] != reference['binding']:
                    raise ValueError('predecessor checkpoint binding changed')
                load_state(source, decoder)
            artifacts, block_rows = [], []
            targets = selected if args.native else range(block * 25, max(selected) + 1)
            for frame in targets:
                if args.native:
                    prior_row = next(row for row in json.loads((baseline / f'block_{block:02d}.json').read_text())['rows'] if row['frame'] == frame)
                    source = Path(prior_row['state']['path'])
                    if fact(source) != prior_row['state']:
                        raise ValueError('pre-pair state changed')
                    load_state(source, decoder)
                state_fact = None
                if frame in selected and not args.native:
                    state_fact = save_state(run / f'pair_{frame:04d}.state.gz', decoder)
                    artifacts.append(state_fact)
                totals.clear()
                enabled[0] = frame in selected
                start = time.perf_counter()
                token = next(decoder)
                token_seconds = time.perf_counter() - start
                enabled[0] = False
                token_path = run / f'pair_{frame:04d}.tokens.u8'
                token_path.write_bytes(token.numpy().tobytes())
                artifacts.append(fact(token_path))
                if not np.array_equal(token.numpy(), reference_tokens[frame]):
                    raise ValueError(f'token mismatch at {frame}; payload retained')
                if frame not in selected:
                    continue
                pair, render_seconds, pose_seconds = render_pair(module, model, basis, coefficients, modes, labels, token, frame)
                raw = run / f'pair_{frame:04d}.raw'
                raw.write_bytes(pair.tobytes())
                raw_fact = fact(raw)
                artifacts.append(raw_fact)
                if raw_fact['sha256'] != reference['pair_sha256'][frame]:
                    raise ValueError(f'raw mismatch at {frame}; payload retained')
                row = dict(frame=frame, token_seconds=token_seconds, timings=dict(totals),
                    renderer_seconds=render_seconds, pose_seconds=pose_seconds,
                    state=state_fact if state_fact else prior_row['state'], token=fact(token_path), raw=raw_fact,
                    matched_reference=True)
                block_rows.append(row)
            atomic_json(completed, dict(binding=binding, rows=block_rows, artifacts=artifacts,
                checkpoint_source=fact(source) if source else None, selection=selected))
            rows.extend(block_rows)
            print(json.dumps(dict(completed_pairs=len(rows), block=block)), flush=True)
        summary = defaultdict(float)
        for row in rows:
            for name, seconds in row['timings'].items():
                summary[name] += seconds
            summary['token_total'] += row['token_seconds']
            summary['renderer'] += row['renderer_seconds']
            summary['pose_carrier_and_selector'] += row['pose_seconds']
        summary['other_token_work'] = summary['token_total'] - sum(summary[key] for key in ('arithmetic_and_frequency_table', 'prior_context_statistics', 'prior_network'))
        summary['total_measured_pair_work'] = summary['token_total'] + summary['renderer'] + summary['pose_carrier_and_selector']
        atomic_json(run / 'RESULT.json', dict(binding=binding, rows=rows, seconds=dict(summary),
            matched_pairs=len(rows), sample=strata, sampling='seeded uniform 2-of-25 within each of 24 strata',
            timing_scope='48 selected pair calculations, excludes replay, setup, checkpoints and raw writes',
            reference=fact(PRIOR / 'IDENTITY.json'), tokens_reference=fact(PRIOR / 'tokens.u8')))
        print(json.dumps(dict(status='complete', seconds=dict(summary))), flush=True)


if __name__ == '__main__':
    main()
