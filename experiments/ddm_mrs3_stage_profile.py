"""Scorer-free finer profile of the exact stage named by the mrs3 charter.

Uses the predecessor's SHA-bound stratified states; retains every token/raw
payload. Each completed pair is a resumable stage. Nested timers are reported
separately and must never be summed into an end-to-end timing estimate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import importlib.util
import json
from pathlib import Path
import platform
import random
import shutil
import sys
import time

import numpy as np
import torch

from ddm_mrs1_validate import atomic_json, fact
from ddm_mrs2_profile import install_timers, load_state, render_pair

ROOT = Path('/Volumes/APDataStore/pact/ddm_mrs3')
REPO = Path(__file__).resolve().parents[1]
BANK = Path('/Volumes/APDataStore/pact/ddm_mrs2/profile_python/RESULT.json')


def detail_timers(module, enabled):
    totals, calls = defaultdict(float), defaultdict(int)

    def wrap(cls, name):
        original = getattr(cls, name)
        key = cls.__name__ + '.' + name

        def measured(self, *args, **kwargs):
            if not enabled[0]:
                return original(self, *args, **kwargs)
            start = time.perf_counter()
            try:
                return original(self, *args, **kwargs)
            finally:
                totals[key] += time.perf_counter() - start
                calls[key] += 1

        setattr(cls, name, measured)

    for name in ('begin_frame', 'group_state', 'coding_row', 'observe', 'end_frame'):
        wrap(module.miss_FreeCorrector, name)
    for name in ('begin_frame', 'coding', 'observe', 'end_frame'):
        wrap(module.geometry_mixer_LaneMixer, name)
    for name in ('__init__', 'contexts', 'observe'):
        wrap(module.CausalGeometry, name)
    for name in ('begin_frame', 'features', 'end_frame'):
        wrap(module.shared_SharedMixer, name)
    for name in ('odds_multiplier', '_update_weights'):
        wrap(module.odds_FixedPointLogisticMixer, name)
    return totals, calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-from', type=Path, required=True)
    args = parser.parse_args()
    run = args.resume_from.resolve()
    if not run.is_relative_to(ROOT) or run == ROOT:
        raise ValueError('profile must remain inside the charter store')
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / 'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if shutil.disk_usage(ROOT).free < 1024**3:
        raise ValueError('need 1 GiB free for retained proof payloads')
    sys.dont_write_bytecode = True
    random.seed(20260916)
    np.random.seed(20260916)
    torch.manual_seed(20260916)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    public = ROOT / 'diagnostic_public'
    source = public / 'inflate.py'
    bank = json.loads(BANK.read_text())
    if len(bank['rows']) != 48 or bank['binding']['seed'] != 20260916:
        raise ValueError('the pre-registered 48-pair bank is required')
    binding = dict(public=[fact(p) for p in sorted(public.iterdir()) if p.is_file()],
        bank=fact(BANK), runner=fact(Path(__file__).resolve()),
        helpers=[fact(REPO / 'experiments' / name) for name in
                 ('ddm_mrs1_validate.py', 'ddm_mrs2_profile.py')],
        axis='[macOS-CPU advisory]', python=sys.version, numpy=np.__version__,
        torch=torch.__version__, platform=platform.platform(), seed=20260916,
        threads=4, score_claim=False, promotable=False)
    inputs = run / 'INPUTS.json'
    if inputs.exists() and json.loads(inputs.read_text()) != binding:
        raise ValueError('binding changed; a new proof directory is required')
    atomic_json(inputs, binding)
    spec = importlib.util.spec_from_file_location('mrs3_diagnostic_receiver', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if module._RANGE_LIBRARY is None:
        raise ValueError('diagnostic must use the native range decoder')
    totals, enabled = install_timers(module)
    details, calls = detail_timers(module, enabled)
    rows = []
    with torch.inference_mode():
        parts, model, basis, coefficients, selector = module.read_models(public / 'archive.zip')
        basis = module.render_normalized_basis(basis)
        modes, labels = module.selector_decode_selector(selector)
        for original in bank['rows']:
            index = original['frame']
            receipt = run / f'pair_{index:04d}.json'
            if receipt.exists():
                row = json.loads(receipt.read_text())
                if row['binding'] != binding:
                    raise ValueError('completed pair binding changed')
                for item in (row['token'], row['raw']):
                    if fact(Path(item['path'])) != item:
                        raise ValueError('retained payload changed')
                rows.append(row)
                continue
            state = Path(original['state']['path'])
            if fact(state) != original['state']:
                raise ValueError('pre-pair state changed')
            decoder = module.TokenDecoder(parts, torch.device('cpu'))
            load_state(state, decoder)
            totals.clear()
            details.clear()
            calls.clear()
            enabled[0] = True
            start = time.perf_counter()
            token = next(decoder)
            token_seconds = time.perf_counter() - start
            enabled[0] = False
            token_path = run / f'pair_{index:04d}.tokens.u8'
            token_path.write_bytes(token.numpy().tobytes())
            pair, render_seconds, pose_seconds = render_pair(
                module, model, basis, coefficients, modes, labels, token, index)
            raw = run / f'pair_{index:04d}.raw'
            raw.write_bytes(pair.tobytes())
            token_fact, raw_fact = fact(token_path), fact(raw)
            if (token_fact['sha256'] != original['token']['sha256'] or
                    raw_fact['sha256'] != original['raw']['sha256']):
                raise ValueError(f'pair {index} differs; both payloads retained')
            row = dict(binding=binding, frame=index, state=original['state'],
                token=token_fact, raw=raw_fact, matched=True, token_seconds=token_seconds,
                seconds=dict(totals), nested_seconds=dict(details), calls=dict(calls),
                renderer_seconds=render_seconds, pose_seconds=pose_seconds)
            atomic_json(receipt, row)
            rows.append(row)
            print(json.dumps(dict(completed_pairs=len(rows), frame=index)), flush=True)
    combined, nested, ncalls = defaultdict(float), defaultdict(float), defaultdict(int)
    for row in rows:
        for name, value in row['seconds'].items():
            combined[name] += value
        for name, value in row['nested_seconds'].items():
            nested[name] += value
        for name, value in row['calls'].items():
            ncalls[name] += value
        combined['token_total'] += row['token_seconds']
        combined['renderer'] += row['renderer_seconds']
        combined['pose_carrier_and_selector'] += row['pose_seconds']
    combined['other_token_work'] = combined['token_total'] - sum(combined[k] for k in
        ('arithmetic_and_frequency_table', 'prior_context_statistics', 'prior_network'))
    combined['total_measured_pair_work'] = combined['token_total'] + combined['renderer'] + combined['pose_carrier_and_selector']
    atomic_json(run / 'RESULT.json', dict(binding=binding, rows=rows, matched_pairs=len(rows),
        seconds=dict(combined), nested_seconds=dict(nested), calls=dict(ncalls),
        timing_scope='48 selected pairs, setup/replay/writes excluded; nested timers overlap and cannot be summed',
        sampling=bank['sampling'], sample=bank['sample']))
    print(json.dumps(dict(status='complete', seconds=dict(combined), nested_seconds=dict(nested))), flush=True)


if __name__ == '__main__':
    main()
