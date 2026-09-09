#!/usr/bin/env python3
"""Retained public entrypoint probes and resumable full-field identity, no scorer."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'src'))
sys.dont_write_bytecode = True
from experiments import ddm_jg2_tail_reencode as jg2
from experiments.ddm_tc1_token_tail_bound import AXIS, ROOT, SEED, pinned
from tac.candidate_seal import _public_smoke_problems, measure_runtime_digest


class ProbeReached(Exception):
    pass


class FieldComplete(Exception):
    pass


def identity(runtime):
    return dict(runtime_path=str(runtime.resolve()),
                tree_sha256=measure_runtime_digest(runtime).sha256,
                archive_path=str((runtime / 'archive.zip').resolve()),
                archive_sha256=jg2.file_fact(runtime / 'archive.zip')['sha256'])


def build_libraries(runtime, work):
    work.mkdir(parents=True, exist_ok=True)
    facts = []
    for key, source, name, extra in [
        ('CPR1_RC64_LIBRARY', 'runtime/entropy/rc64_backend.c', 'librc64.so', []),
        ('F26_CORRECTOR_NATIVE_LIBRARY', 'runtime/f26_corrector_native.c', 'libcorrector.so',
         ['-ffp-contract=off', '-fno-fast-math', '-lm'])]:
        target = work / name
        command = [os.environ.get('CC', 'cc'), '-O3', '-std=c11', '-shared', '-fPIC', str(runtime / source), '-o', str(target)] + extra
        source_fact = jg2.file_fact(runtime / source)
        receipt_path = target.with_suffix('.json')
        if target.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt['source'] != source_fact or receipt['library'] != jg2.file_fact(target) or receipt['argv'] != command:
                raise RuntimeError('native build custody changed')
        else:
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            (work / (name + '.build.log')).write_text(result.stdout + result.stderr)
            result.check_returncode()
            receipt = dict(source=source_fact, library=jg2.file_fact(target), argv=command)
            jg2.atomic_json(receipt_path, receipt)
        os.environ[key] = str(target)
        facts.append(receipt)
    return facts


def public_worker(root, role, stop, probe, proof_name='public_identity'):
    inputs = pinned(root)
    runtime = root / ('candidate_runtime' if role == 'candidate' else 'source_runtime')
    store = root / ('public_smoke_work/' + role if probe else proof_name)
    store.mkdir(parents=True, exist_ok=True)
    if not probe and shutil.disk_usage(store).free < 3 * 1024**3:
        raise RuntimeError('public receiver storage preflight requires 3 GiB free SSD')
    build = build_libraries(runtime, store / 'native')
    os.environ['F26_TOKEN_DECODER'] = 'python'
    os.environ.pop('F26_ADVISORY_DECODE_CACHE_ROOT', None)
    os.environ.pop('TC1_RECEIVER_CHECKPOINT_DIR', None)
    os.environ.pop('TC1_RECEIVER_STOP_AFTER', None)
    if not probe:
        os.environ['TC1_RECEIVER_CHECKPOINT_DIR'] = str(store / 'frame_checkpoints')
        os.environ['TC1_RECEIVER_STOP_AFTER'] = str(stop)
    sys.path.insert(0, str(runtime))
    import numpy as np
    import runtime.f26_inflate as public
    import torch
    torch.manual_seed(SEED); np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    runtime_identity = identity(runtime)
    if probe:
        def reached(*args, **kwargs):
            result = dict(**runtime_identity, outcome='REACHED_TOKEN_DECODE',
                          seconds=time.monotonic() - started, exception_class=None,
                          exception_message='', evidence='observer entered the actual public token-stage call after semantic/carrier setup',
                          build=build)
            jg2.atomic_json(store / 'DIRECT.json', result)
            print('TC1_OBSERVED_PUBLIC_TOKEN_ENTRY', flush=True)
            raise ProbeReached()
        public.decode_production_tokens = reached
    else:
        original_write = public._write_token_checkpoint
        def retained_full_field(checkpoint_dir, tokens, *, binding, token_report):
            original_write(checkpoint_dir, tokens, binding=binding, token_report=token_report)
            decoded = jg2.file_fact(checkpoint_dir / 'tokens_cpu_stage_complete.u8')
            identical = all(decoded[key] == inputs['field'][key] for key in ('bytes', 'sha256'))
            result = dict(**runtime_identity, axis=AXIS, score_claim=False, pairs=600,
                          symbols=117964800, public_function='runtime.f26_inflate.inflate_archive',
                          actual_public_token_decoder='runtime.residual_archive.decode_production_tokens',
                          retained_field=decoded, expected_field=inputs['field'],
                          exact_field_identity=identical, token_report=token_report,
                          scope='actual public CPU token phase; stopped at its durable stage checkpoint before rendering; no scorer',
                          seed=SEED, platform=platform.platform(), build=build,
                          stage_seconds=time.monotonic() - started)
            jg2.atomic_json(store / 'PUBLIC_FIELD_IDENTITY.json', result)
            if not identical:
                raise RuntimeError('public receiver field differs from the exact source field')
            raise FieldComplete()
        public._write_token_checkpoint = retained_full_field
        complete = store / 'PUBLIC_FIELD_IDENTITY.json'
        if complete.exists():
            prior = json.loads(complete.read_text())
            field = prior['retained_field']
            if (prior['exact_field_identity'] and all(prior[k] == v for k, v in runtime_identity.items())
                    and jg2.file_fact(Path(field['path'])) == field):
                return prior
            raise RuntimeError('existing public identity is not a current passing receipt')
    try:
        public.inflate_archive(runtime / 'archive.zip', store / 'render_not_started.raw',
                               renderer_dir=runtime / 'cpr1', device_name='cpu', num_threads=4,
                               checkpoint_dir=store / 'public_stage_checkpoint')
    except ProbeReached:
        return json.loads((store / 'DIRECT.json').read_text())
    except FieldComplete:
        return json.loads((store / 'PUBLIC_FIELD_IDENTITY.json').read_text())
    except RuntimeError as error:
        if not probe and str(error) == 'TC1_RECEIVER_STAGE_COMPLETE':
            result = dict(stage_complete=stop, public_n600_identity='PENDING', seconds=time.monotonic() - started)
            jg2.atomic_json(store / f'STAGE_{stop:04d}.json', result)
            return result
        raise
    raise RuntimeError('public worker unexpectedly reached rendering')


def retained_child(command, work, env, timeout):
    log = work / 'run.log'
    started = time.monotonic()
    with log.open('wb') as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                   env=env, cwd=REPO, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise RuntimeError('public smoke timeout; process group killed')
    receipt = dict(argv=command, seconds=time.monotonic() - started, returncode=code,
                   log=jg2.file_fact(log))
    jg2.atomic_json(work / 'PROCESS.json', receipt)
    return receipt


def smoke(root):
    inputs = pinned(root)
    block = dict(schema='candidate_public_entrypoint_smoke.v1', public_path_probe_seconds=180,
                 public_path_probes={}, inflate_sh_smokes={})
    for role in ('candidate', 'frontier'):
        runtime = root / ('candidate_runtime' if role == 'candidate' else 'source_runtime')
        work = root / 'public_smoke_work' / role
        work.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        direct_process = retained_child([sys.executable, str(Path(__file__).resolve()), 'probe', '--root', str(root), '--role', role], work, env, 180)
        if direct_process['returncode'] != 0:
            raise RuntimeError(f'{role} public entry failed; retained log {work / "run.log"}')
        direct = json.loads((work / 'DIRECT.json').read_text())
        direct['process'] = direct_process
        block['public_path_probes'][role] = direct
        shell_work = work / 'shell'; shell_work.mkdir(exist_ok=True)
        data, out, scratch = [shell_work / name for name in ('data', 'output', 'scratch')]
        for path in (data, out, scratch):
            path.mkdir(exist_ok=True)
        with zipfile.ZipFile(runtime / 'archive.zip') as archive:
            member = archive.read('p')
        jg2.persist_immutable_bytes(data / 'p', member, label='actual inflate.sh input member')
        files = shell_work / 'file_list.txt'; files.write_text('0.hevc\n')
        env.update(PATH=str(REPO / '.venv/bin') + os.pathsep + env.get('PATH', ''), TMPDIR=str(scratch))
        shell = retained_child(['bash', str(runtime / 'inflate.sh'), str(data), str(out), str(files)], shell_work, env, 180)
        lines = (shell_work / 'run.log').read_text().splitlines()
        errors = [line[len('RuntimeError: '):] for line in lines if line.startswith('RuntimeError: ')]
        if shell['returncode'] == 0 or not errors:
            raise RuntimeError(f'{role} shell did not reach its measured CUDA gate')
        block['inflate_sh_smokes'][role] = dict(**identity(runtime), **shell, outcome='REACHED_CUDA_GATE',
                                                exception_class='RuntimeError', exception_message=errors[-1])
    problems, observed = _public_smoke_problems(block, candidate_runtime_dir=root / 'candidate_runtime',
                                               candidate_archive_path=root / 'candidate_runtime/archive.zip',
                                               pointer_archive_sha256=inputs['archive']['sha256'])
    jg2.atomic_json(root / 'PUBLIC_SMOKE.json', block)
    jg2.atomic_json(root / 'PUBLIC_SMOKE_VALIDATOR.json', dict(problems=problems, observed=observed))
    if problems:
        raise RuntimeError('validator rejected public smoke: ' + repr(problems))
    return block


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('decode', 'probe', 'smoke'))
    parser.add_argument('--root', type=Path, default=ROOT / 'rebase_pc2')
    parser.add_argument('--role', choices=('candidate', 'frontier'), default='candidate')
    parser.add_argument('--stop', type=int, default=600)
    parser.add_argument('--proof-name', choices=('public_identity', 'resume_control'), default='public_identity')
    args = parser.parse_args()
    result = smoke(args.root) if args.stage == 'smoke' else public_worker(args.root, args.role, args.stop, args.stage == 'probe', args.proof_name)
    print(json.dumps(result, sort_keys=True))
