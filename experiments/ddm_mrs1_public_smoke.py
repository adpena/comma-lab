"""External, stage-resumable proof of the actual public shell in a bare venv.

An interrupted public decode restarts cold: the public receiver has no checkpoint
API. Its partial raw stays on disk until a complete replacement is certified.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

ROOT = Path('/Volumes/APDataStore/pact/ddm_mrs1')
PUBLIC = Path(__file__).resolve().parents[1] / 'submissions/mrs1'
ARCHIVE_SHA = 'aab908d3b32c65582b7de1a3855b67a4f6fdb9373b0dc81e3049ee8164ec3957'
RAW_SHA = '8a14f55a6a8b141501836f511f4222dde75dca21714d923ad865b5f4757ef66b'
RAW_BYTES = 3662409600
PAIR_BYTES = 6104016


def fact(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)


def save(path, value):
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def check_fact(value):
    if fact(Path(value['path'])) != value:
        raise ValueError(f'changed input or retained artifact: {value["path"]}')


def check_binding(binding):
    for value in binding['public_files'] + [binding['runner'], binding['bootstrap_ready'],
                                           binding['prior_identity'], binding['prior_inputs']]:
        check_fact(value)


def command_stage(run, name, command, binding, environment, lock_fd, timeout=None):
    """Keep every attempt; reuse only a successful source-bound stage receipt."""
    completed = run / f'{name}.DONE.json'
    if completed.exists() and name != 'environment':
        result = json.loads(completed.read_text())
        if result['binding'] != binding or result['command'] != command:
            raise ValueError('stage binding changed')
        check_fact(result['log'])
        return result
    active = run / 'ACTIVE.json'
    if active.exists():
        previous = json.loads(active.read_text())
        try:
            os.kill(previous['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError(f'prior child PID {previous["pid"]} is still live; do not overlap')
    attempt = run / 'attempts' / f'{name}_{time.time_ns()}'
    attempt.mkdir(parents=True)
    started = time.monotonic()
    record = dict(binding=binding, command=command, cwd=str(PUBLIC),
                  started_utc=datetime.datetime.now(datetime.UTC).isoformat(),
                  restart_policy='Replay this public subprocess cold after interruption',
                  environment=dict(PATH=environment['PATH'],
                                   PYTHONDONTWRITEBYTECODE=environment['PYTHONDONTWRITEBYTECODE'],
                                   pythonpath_absent='PYTHONPATH' not in environment,
                                   pythonhome_absent='PYTHONHOME' not in environment,
                                   home_unchanged=environment.get('HOME') == os.environ.get('HOME')))
    save(attempt / 'START.json', record)
    with (attempt / 'output.log').open('wb') as log:
        child = subprocess.Popen(command, cwd=PUBLIC, env=environment, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock_fd,))
        save(active, dict(pid=child.pid, attempt=str(attempt), **record))

        def stop(signum, frame):
            raise InterruptedError(f'interrupted by signal {signum}')

        previous_handlers = {number: signal.signal(number, stop)
                             for number in (signal.SIGINT, signal.SIGTERM)}
        failure = None
        try:
            returncode = child.wait(timeout=timeout)
        except BaseException as error:
            failure = error
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            returncode = child.returncode
        finally:
            for number, handler in previous_handlers.items():
                signal.signal(number, handler)
    result = dict(**record, returncode=returncode, seconds=time.monotonic() - started,
                  log=fact(attempt / 'output.log'), exception=repr(failure) if failure else None)
    save(attempt / 'DONE.json', result)
    active.unlink()
    if failure is not None:
        raise failure
    # Guard is deliberately nonzero; its exact reason is checked by the caller.
    if returncode == 0 or name == 'default_guard':
        save(completed, result)
    return result


def preserve_interrupted(run, raw, binding):
    if not raw.exists():
        return
    retained = run / 'interrupted' / f'0_{time.time_ns()}.raw'
    retained.parent.mkdir(exist_ok=True)
    save(retained.with_suffix('.json'), dict(binding=binding, raw=fact(raw),
         retained_path=str(retained), reason='Interrupted public decode; keep until full raw identity',
         score_claim=False, promotable=False))
    raw.replace(retained)


def cleanup_certified(run, certificate, binding):
    """Write cleanup custody before each unlink, including on completion reuse."""
    if (certificate['binding'] != binding or certificate['raw']['sha256'] != RAW_SHA or
            certificate['raw']['bytes'] != RAW_BYTES or certificate['matched_pairs'] != 600 or
            len(certificate['pair_sha256']) != 600):
        raise ValueError('completion binding mismatch')
    for metadata in sorted((run / 'interrupted').glob('*.json')):
        if metadata.name.startswith('._') or metadata.name.endswith('.cleanup.json'):
            continue
        item = json.loads(metadata.read_text())
        if item['binding'] != binding:
            raise ValueError('interrupted scratch binding differs')
        retained = Path(item['retained_path'])
        if retained.parent != run / 'interrupted':
            raise ValueError('interrupted scratch escaped its store')
        if retained.exists():
            actual = fact(retained)
            if (actual['bytes'], actual['sha256']) != (item['raw']['bytes'], item['raw']['sha256']):
                raise ValueError('interrupted scratch changed; retain it')
            save(metadata.with_suffix('.cleanup.json'), dict(binding=binding, raw=actual,
                 replacement=certificate['raw'], reason='Rebuildable partial raw replaced by certified complete raw',
                 score_claim=False, promotable=False))
            retained.unlink()
    raw = Path(certificate['raw']['path'])
    if raw != run / 'output/0.raw':
        raise ValueError('completed raw escaped its output store')
    if raw.exists():
        check_fact(certificate['raw'])
        raw.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-from', type=Path, required=True)
    parser.add_argument('--prior-identity', type=Path, required=True)
    parser.add_argument('--bootstrap-ready', type=Path, default=ROOT / 'bootstrap/READY.json')
    args = parser.parse_args()
    run, prior_path, ready_path = (p.resolve() for p in
                                  (args.resume_from, args.prior_identity, args.bootstrap_ready))
    if any(not p.is_relative_to(ROOT) for p in (run, prior_path, ready_path)) or run == ROOT:
        raise ValueError('proof paths must remain within the charter store')
    run.mkdir(parents=True, exist_ok=True)
    # The child inherits this lock, so even SIGKILL of this runner cannot admit
    # a second launch while the original public process still owns its output.
    lock = (run / 'RUN.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prior = json.loads(prior_path.read_text())
    prior_inputs = prior_path.with_name('INPUTS.json')
    if prior['binding'] != json.loads(prior_inputs.read_text()):
        raise ValueError('prerequisite certificate does not match its original input binding')
    if (prior['matched_pairs'] != 600 or prior['resumed_from'] != 0 or
            prior['raw']['sha256'] != RAW_SHA or prior['raw']['bytes'] != RAW_BYTES or
            len(prior['pair_sha256']) != 600 or prior['score_claim'] is not False or
            prior['binding']['archive']['sha256'] != ARCHIVE_SHA):
        raise ValueError('a completed cold n600 raw-identity prerequisite is required')
    ready = json.loads(ready_path.read_text())
    venv = Path(ready['isolation']['prefix']).resolve()
    if not venv.is_relative_to(ROOT) or ready['bare'] is not True or ready['system_site_packages'] is not False:
        raise ValueError('bootstrap is not the isolated charter environment')
    for value in ready['wheels'] + [ready['base_interpreter']['executable']] + ready['base_interpreter']['shared_libraries']:
        check_fact(value)
    if 'include-system-site-packages = false' not in (venv / 'pyvenv.cfg').read_text():
        raise ValueError('venv permits system site packages')
    public_files = [fact(PUBLIC / name) for name in ('inflate.sh', 'inflate.py', 'archive.zip', 'README.md')]
    archive = public_files[2]
    if archive['sha256'] != ARCHIVE_SHA or archive['bytes'] != 179286:
        raise ValueError('public archive differs from the sealed bytes')
    binding = dict(public_files=public_files, runner=fact(Path(__file__).resolve()),
                   bootstrap_ready=fact(ready_path), prior_identity=fact(prior_path), prior_inputs=fact(prior_inputs),
                   prior_receiver=prior['binding']['receiver'],
                   prior_scope='Precursor cold proof and per-pair reference, not current public receiver proof',
                   seed=20260916, axis='[macOS-CPU advisory]', score_claim=False, promotable=False)
    inputs = run / 'INPUTS.json'
    if inputs.exists() and json.loads(inputs.read_text()) != binding:
        raise ValueError('inputs changed; use a new proof directory')
    save(inputs, binding)
    completed = run / 'IDENTITY.json'
    if completed.exists():
        cleanup_certified(run, json.loads(completed.read_text()), binding)
        print(json.dumps(dict(status='complete_reused', receipt=str(completed))), flush=True)
        return
    if (run / 'RAW_MISMATCH.json').exists():
        raise ValueError('previous mismatch is retained; investigate before any replay')
    environment = {key: value for key, value in os.environ.items() if not key.startswith('PYTHON')}
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment['PATH'] = str(venv / 'bin') + os.pathsep + environment.get('PATH', '')
    executable = shutil.which('python3', path=environment['PATH'])
    if executable != str(venv / 'bin/python3'):
        raise ValueError('public shell would use the wrong interpreter')
    probe = ('import sys,site,json,numpy,torch,brotli; '
             'print(json.dumps(dict(executable=sys.executable,prefix=sys.prefix,base=sys.base_prefix,'
             'path=sys.path,user_site=site.ENABLE_USER_SITE,cuda=torch.cuda.is_available(),'
             'packages={m.__name__:dict(version=m.__version__,path=m.__file__) for m in (numpy,torch,brotli)})))')
    result = command_stage(run, 'environment', [executable, '-c', probe], binding, environment, lock.fileno(), 60)
    if result['returncode'] != 0:
        raise ValueError('bare environment import probe failed')
    observed = json.loads(Path(result['log']['path']).read_text())
    if (observed['prefix'] != str(venv) or observed['prefix'] == observed['base'] or
            observed['user_site'] or observed['cuda'] or observed['packages'] != ready['packages'] or
            observed['executable'] != executable):
        raise ValueError('actual public interpreter/dependencies differ or this is not a CPU host')
    extracted, output = run / 'extracted', run / 'output'
    extracted.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)
    with zipfile.ZipFile(PUBLIC / 'archive.zip') as compressed:
        if compressed.namelist() != ['p']:
            raise ValueError('unexpected archive members')
        payload = compressed.read('p')
    payload_path, file_list = extracted / 'p', run / 'file-list.txt'
    for path, value in ((payload_path, payload), (file_list, b'0.mkv\n')):
        if path.exists() and path.read_bytes() != value:
            raise ValueError('retained public input changed')
        if not path.exists():
            temporary = path.with_name(path.name + '.pending')
            temporary.write_bytes(value)
            temporary.replace(path)
    save(run / 'SETUP.json', dict(binding=binding, extracted=fact(payload_path), file_list=fact(file_list),
         environment=observed, environment_stage=result))
    shell = ['/bin/sh', str(PUBLIC / 'inflate.sh')]
    help_result = command_stage(run, 'help', shell + ['--help'], binding, environment, lock.fileno(), 60)
    if help_result['returncode'] != 0 or '--device' not in Path(help_result['log']['path']).read_text():
        raise ValueError('public help smoke failed')
    arguments = [str(extracted), str(output), str(file_list)]
    guard = command_stage(run, 'default_guard', shell + arguments, binding, environment, lock.fileno(), 60)
    guard_text = Path(guard['log']['path']).read_text()
    if guard['returncode'] != 1 or 'CUDA is required by default; --device cpu' not in guard_text:
        raise ValueError('exact three-argument default-CUDA guard did not fire clearly')
    raw = output / '0.raw'
    decode_done = run / 'public_decode.DONE.json'
    if not decode_done.exists():
        preserve_interrupted(run, raw, binding)
        free = shutil.disk_usage(ROOT).free
        save(run / 'STORAGE_PREFLIGHT.json', dict(free_bytes=free, minimum_bytes=5 * 1024 ** 3,
             transient_raw_bytes=RAW_BYTES, retained_interrupted_bytes=sum(p.stat().st_size for p in
                 (run / 'interrupted').glob('*.raw')), reason='Cold public decode and verification headroom'))
        if free < 5 * 1024 ** 3:
            raise ValueError('less than 5 GiB free on the charter SSD; keep retained bytes')
    check_binding(binding)
    try:
        decode = command_stage(run, 'public_decode', shell + arguments + ['--device', 'cpu'],
                               binding, environment, lock.fileno())
    except BaseException:
        preserve_interrupted(run, raw, binding)
        raise
    if decode['returncode'] != 0:
        preserve_interrupted(run, raw, binding)
        raise ValueError('public CPU decode failed; logs and partial raw retained')
    before = raw.stat()
    digest, pair_hashes = hashlib.sha256(), []
    with raw.open('rb') as stream:
        for _ in range(600):
            block = stream.read(PAIR_BYTES)
            digest.update(block)
            pair_hashes.append(hashlib.sha256(block).hexdigest())
        trailing = stream.read(1024 * 1024)
        extra_bytes = len(trailing)
        while trailing:
            digest.update(trailing)
            trailing = stream.read(1024 * 1024)
            extra_bytes += len(trailing)
    after = raw.stat()
    actual = dict(path=str(raw), bytes=after.st_size, sha256=digest.hexdigest())
    if (before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns or extra_bytes or
            actual['bytes'] != RAW_BYTES or actual['sha256'] != RAW_SHA or pair_hashes != prior['pair_sha256']):
        save(run / 'RAW_MISMATCH.json', dict(binding=binding, raw=actual, pair_sha256=pair_hashes,
             reason='Public raw does not match the full reference certificate; retain it'))
        raise ValueError('public raw mismatch; retained output is not a successful proof')
    check_binding(binding)
    certificate = dict(binding=binding, raw=actual, pair_sha256=pair_hashes, matched_pairs=600,
         public_decode=decode, help=help_result, default_guard=guard, environment=observed,
         seconds=decode['seconds'], cold_public_subprocess=True, resumed_stage_only=True,
         predecessor_receiver_sha256=prior['binding']['receiver']['sha256'],
         public_receiver_sha256=public_files[1]['sha256'],
         proof_scope='The precursor certificate is a prerequisite and reference; this cold public shell run proves the bound final receiver bytes',
         argv=sys.argv, score_claim=False, promotable=False,
         reason='Full and 600 per-pair SHA equality proved; raw is reproducible from bound public archive and receiver')
    save(completed, certificate)
    cleanup_certified(run, certificate, binding)
    print(json.dumps(dict(status='complete', seconds=decode['seconds'], receipt=str(completed))), flush=True)


if __name__ == '__main__':
    main()
