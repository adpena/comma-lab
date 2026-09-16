"""Bare-venv public shell proof, with cold native output and 48 fallback pairs.

All large outputs stay on the charter SSD. Public cold decode is a restartable
stage; interrupted raws are retained until a complete replacement is certified.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np
import torch

import ddm_mrs1_public_smoke as stages
from ddm_mrs2_profile import load_state, render_pair

ROOT = Path('/Volumes/APDataStore/pact/ddm_mrs2')
REPO = Path(__file__).resolve().parents[1]
PUBLIC = REPO / 'submissions/mrs2'
NAMES = ('inflate.sh', 'inflate.py', 'range_decoder.c', 'README.md', 'archive.zip')
fact, save = stages.fact, stages.save


def pair_worker(run, receiver):
    """Decode the 48 real pre-pair states in the actual compiler-absent venv."""
    sys.dont_write_bytecode = True
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260916)
    np.random.seed(20260916)
    profile = ROOT / 'profile_python/RESULT.json'
    bank = json.loads(profile.read_text())
    spec = importlib.util.spec_from_file_location('smoke_receiver', receiver)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if module._RANGE_LIBRARY is not None:
        raise ValueError('compiler-absent proof loaded native code')
    binding = dict(receiver=fact(receiver), profile=fact(profile),
                   runner=fact(Path(__file__)), backend='python', axis='[macOS-CPU advisory]',
                   score_claim=False, promotable=False)
    rows = []
    with torch.inference_mode():
        parts, model, basis, coefficients, selector = module.read_models(receiver.with_name('archive.zip'))
        basis = module.render_normalized_basis(basis)
        modes, labels = module.selector_decode_selector(selector)
        for original in bank['rows']:
            index = original['frame']
            completed = run / f'fallback_pair_{index:04d}.json'
            if completed.exists():
                row = json.loads(completed.read_text())
                if row['binding'] != binding:
                    raise ValueError('fallback binding changed')
                for item in (row['tokens'], row['raw']):
                    if fact(Path(item['path'])) != item:
                        raise ValueError('fallback payload changed')
                rows.append(row)
                continue
            source = Path(original['state']['path'])
            if fact(source) != original['state']:
                raise ValueError('pre-pair state changed')
            decoder = module.TokenDecoder(parts, torch.device('cpu'))
            load_state(source, decoder)
            token = next(decoder)
            tokens = run / f'fallback_pair_{index:04d}.tokens.u8'
            tokens.write_bytes(token.numpy().tobytes())
            pair, _, _ = render_pair(module, model, basis, coefficients, modes, labels, token, index)
            raw = run / f'fallback_pair_{index:04d}.raw'
            raw.write_bytes(pair.tobytes())
            token_fact, raw_fact = fact(tokens), fact(raw)
            if (token_fact['sha256'] != original['token']['sha256'] or
                    raw_fact['sha256'] != original['raw']['sha256']):
                raise ValueError(f'fallback differs at {index}; outputs retained')
            row = dict(binding=binding, frame=index, tokens=token_fact, raw=raw_fact, matched=True)
            save(completed, row)
            rows.append(row)
    save(run / 'FALLBACK_PAIRS.json', dict(binding=binding, rows=rows,
         matched_pairs=len(rows), source_bound_resume=True))
    print(json.dumps(dict(status='fallback_complete', pairs=len(rows))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-from', type=Path, required=True)
    parser.add_argument('--without-compiler', action='store_true')
    parser.add_argument('--pair-worker', type=Path)
    args = parser.parse_args()
    run = args.resume_from.resolve()
    if not run.is_relative_to(ROOT) or run == ROOT:
        raise ValueError('smoke store escaped charter root')
    run.mkdir(parents=True, exist_ok=True)
    if args.pair_worker:
        if args.pair_worker.resolve() != run / 'public/inflate.py':
            raise ValueError('pair worker receiver escaped public copy')
        pair_worker(run, args.pair_worker)
        return
    lock = (run / 'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ready_path = ROOT / 'bootstrap/READY.json'
    ready = json.loads(ready_path.read_text())
    venv = Path(ready['environment']['prefix'])
    if not venv.is_relative_to(REPO / '.omx/tmp') or not ready['bare']:
        raise ValueError('APFS bare environment is required')
    if 'include-system-site-packages = false' not in (venv / 'pyvenv.cfg').read_text():
        raise ValueError('system site packages are enabled')
    runtime = run / 'public'
    runtime.mkdir(exist_ok=True)
    originals = [fact(PUBLIC / name) for name in NAMES]
    for item in originals:
        source = Path(item['path'])
        destination = runtime / source.name
        if not destination.exists():
            shutil.copyfile(source, destination)
        if destination.read_bytes() != source.read_bytes():
            raise ValueError('public copy changed')
    if originals[-1]['sha256'] != stages.ARCHIVE_SHA:
        raise ValueError('archive changed')
    stages.PUBLIC = runtime
    reference = Path('/Volumes/APDataStore/pact/ddm_mrs1/public_smoke/IDENTITY.json')
    prior = json.loads(reference.read_text())
    binding = dict(public_files=[fact(runtime / name) for name in NAMES], originals=originals,
        runner=fact(Path(__file__)), helper=fact(REPO / 'experiments/ddm_mrs1_public_smoke.py'),
        profile_helper=fact(REPO / 'experiments/ddm_mrs2_profile.py'),
        bootstrap_ready=fact(ready_path), reference=fact(reference),
        without_compiler=args.without_compiler, axis='[macOS-CPU advisory]', seed=20260916,
        score_claim=False, promotable=False)
    inputs = run / 'INPUTS.json'
    if inputs.exists() and json.loads(inputs.read_text()) != binding:
        raise ValueError('smoke binding changed')
    save(inputs, binding)
    environment = {key: value for key, value in os.environ.items() if not key.startswith('PYTHON')}
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment['PYTHONNOUSERSITE'] = '1'
    if args.without_compiler:
        bin_dir = REPO / '.omx/tmp/ddm_mrs2_compiler_absent_bin'
        bin_dir.mkdir(exist_ok=True)
        for name, target in [('dirname', Path('/usr/bin/dirname')),
                             ('rm', Path('/bin/rm'))]:
            path = bin_dir / name
            if not path.is_symlink():
                path.symlink_to(target)
        environment['PATH'] = str(venv / 'bin') + os.pathsep + str(bin_dir)
        if shutil.which('cc', path=environment['PATH']) is not None:
            raise ValueError('compiler unexpectedly available')
        # Prove failed compilation cannot keep using an old native library.
        native_library = ROOT / 'public_native/public/range_decoder.so'
        if not (run / 'help.DONE.json').exists():
            shutil.copyfile(native_library, runtime / 'range_decoder.so')
            save(run / 'STALE_LIBRARY_CONTROL.json', dict(library=fact(runtime / 'range_decoder.so'),
                 source=fact(native_library), reason='Rebuildable compiler output must be removed before failed rebuild'))
    else:
        environment['PATH'] = str(venv / 'bin') + ':/usr/bin:/bin'
        if shutil.which('cc', path=environment['PATH']) is None:
            raise ValueError('compiler-present smoke has no compiler')
    def stage(name, command, timeout=None):
        return stages.command_stage(run, name, command, binding, environment, lock.fileno(), timeout)
    probe = ('import sys,site,json,numpy,torch,brotli; print(json.dumps(dict(prefix=sys.prefix,'
             'base=sys.base_prefix,user_site=site.ENABLE_USER_SITE,packages={m.__name__:'
             'dict(version=m.__version__,path=m.__file__) for m in (numpy,torch,brotli)})))')
    observed = stage('environment', [str(venv / 'bin/python3'), '-c', probe], 60)
    if observed['returncode']:
        raise ValueError('bare import failed')
    isolation = json.loads(Path(observed['log']['path']).read_text())
    if isolation['prefix'] != str(venv) or isolation['user_site'] or isolation['packages'] != ready['environment']['packages']:
        raise ValueError('bare imports differ')
    shell = ['/bin/sh', str(runtime / 'inflate.sh')]
    help_result = stage('help', shell + ['--help'], 60)
    if help_result['returncode'] != 0:
        raise ValueError('public shell help failed')
    log_text = Path(help_result['log']['path']).read_text()
    if args.without_compiler:
        if (runtime / 'range_decoder.so').exists() or 'using the identical Python decoder' not in log_text:
            raise ValueError('compile failure did not select clear Python fallback')
        result = stage('fallback_pairs', [str(venv / 'bin/python3'), str(Path(__file__).resolve()),
            '--resume-from', str(run), '--pair-worker', str(runtime / 'inflate.py')])
        if result['returncode'] != 0:
            raise ValueError('fallback pair proof failed')
        proof = json.loads((run / 'FALLBACK_PAIRS.json').read_text())
        if proof['matched_pairs'] != 48:
            raise ValueError('fallback proof has fewer than 48 pairs')
        save(run / 'RESULT.json', dict(binding=binding, help=help_result, pairs=fact(run / 'FALLBACK_PAIRS.json'),
             matched_pairs=48, bare_environment=isolation, compiler_available=False,
             stale_library_removed=True, full_cold_n600=False))
        return
    if not (runtime / 'range_decoder.so').exists() or 'using the identical Python decoder' in log_text:
        raise ValueError('compiler-present help did not load C')
    extracted, output = run / 'extracted', run / 'output'
    extracted.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)
    with zipfile.ZipFile(runtime / 'archive.zip') as compressed:
        if compressed.namelist() != ['p']:
            raise ValueError('unexpected archive contents')
        (extracted / 'p').write_bytes(compressed.read('p'))
    file_list = run / 'file-list.txt'
    file_list.write_text('0.mkv\n')
    arguments = [str(extracted), str(output), str(file_list)]
    guard = stage('default_guard', shell + arguments, 60)
    if guard['returncode'] != 1 or 'CUDA is required by default' not in Path(guard['log']['path']).read_text():
        raise ValueError('default CUDA guard failed')
    if shutil.disk_usage(ROOT).free < 5 * 1024**3:
        raise ValueError('less than 5 GiB free for cold raw and verification')
    save(run / 'STORAGE_PREFLIGHT.json', dict(free_bytes=shutil.disk_usage(ROOT).free,
         transient_raw_bytes=stages.RAW_BYTES, retained_limit_bytes=2 * 1024**3))
    raw = output / '0.raw'
    completed = run / 'IDENTITY.json'
    if completed.exists():
        stages.cleanup_certified(run, json.loads(completed.read_text()), binding)
        return
    if not (run / 'public_decode.DONE.json').exists():
        stages.preserve_interrupted(run, raw, binding)
    decode = stage('public_decode', shell + arguments + ['--device', 'cpu'])
    if decode['returncode'] or 'using the identical Python decoder' in Path(decode['log']['path']).read_text():
        stages.preserve_interrupted(run, raw, binding)
        raise ValueError('public native decode failed or fell back; outputs retained')
    before = raw.stat()
    full, hashes = hashlib.sha256(), []
    with raw.open('rb') as stream:
        for _ in range(600):
            block = stream.read(stages.PAIR_BYTES)
            full.update(block)
            hashes.append(hashlib.sha256(block).hexdigest())
        if stream.read(1):
            raise ValueError('unexpected trailing raw bytes')
    actual = dict(path=str(raw), bytes=raw.stat().st_size, sha256=full.hexdigest())
    if (actual['sha256'] != stages.RAW_SHA or actual['bytes'] != stages.RAW_BYTES or
            hashes != prior['pair_sha256'] or before.st_mtime_ns != raw.stat().st_mtime_ns):
        save(run / 'RAW_MISMATCH.json', dict(binding=binding, raw=actual, pair_sha256=hashes))
        raise ValueError('cold public raw differs; keep bytes')
    for item in binding['public_files'] + binding['originals']:
        if fact(Path(item['path'])) != item:
            raise ValueError('public source changed during proof')
    certificate = dict(binding=binding, raw=actual, pair_sha256=hashes, matched_pairs=600,
         public_decode=decode, seconds=decode['seconds'], native_library=fact(runtime / 'range_decoder.so'),
         bare_environment=isolation, cold_public_subprocess=True, compiler_available=True,
         argv=sys.argv, reason='Exact full and 600 per-pair hashes match; raw rebuilds from bound archive/runtime',
         score_claim=False, promotable=False)
    save(completed, certificate)
    stages.cleanup_certified(run, certificate, binding)
    print(json.dumps(dict(status='complete', matched_pairs=600, seconds=decode['seconds'])), flush=True)


if __name__ == '__main__':
    main()
