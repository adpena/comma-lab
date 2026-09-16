"""Cold public proof for submissions/mrs7: one implementation, byte-identical output.

Stages, in order: build a bare APFS venv from the retained offline wheels; copy the
eight public files; prove the shell refuses to run without a C compiler; prove the
compiler-present shell loads all three libraries; prove the default CUDA guard; decode
all 600 pairs cold; compare the full raw and every pair against move 53. The raw is
deleted only after both comparisons pass, and its facts are recorded first.

Axis: [macOS-CPU advisory]. score_claim=false, promotable=false. No score is claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import venv
import zipfile

from ddm_mrs1_public_smoke import ARCHIVE_SHA, PAIR_BYTES, RAW_BYTES, RAW_SHA, fact, save

ROOT = Path('/Volumes/APDataStore/pact/ddm_mrs7')
REPO = Path(__file__).resolve().parents[1]
PUBLIC = REPO / 'submissions/mrs7'
VENV = REPO / '.omx/tmp/ddm_mrs7_bare_venv'
# The 3.42 GiB raw is transient scratch, deleted as soon as both comparisons pass. The SSD tier is
# shared and has less headroom than the charter's 8 GiB floor allows, so the raw goes to local APFS
# scratch while every receipt, log and certificate stays on the SSD. Both tiers are recorded.
RAW_ROOT = REPO / '.omx/tmp/ddm_mrs7_raw'
FLOOR_BYTES = 8 * 1024 ** 3
WHEELS = Path("/Volumes/APDataStore/pact/ddm_mrs5/bootstrap/wheels")
PAIR_REFERENCE = Path('/Volumes/APDataStore/pact/ddm_mrs1/public_smoke/IDENTITY.json')
FULL_REFERENCE = Path('/Volumes/APDataStore/pact/ddm_pd6/parseback2/PARSEBACK_RESULT.json')
NAMES = ('inflate.sh', 'inflate.py', 'range_decoder.c', 'corrector.c', 'geometry.c',
         'README.md', 'FORMAT.md', 'archive.zip')
LIBRARIES = ('range_decoder.so', 'corrector.so', 'geometry.so')
FALLBACK_WORDS = ('fallback', 'identical Python')


def build_environment(run):
    """Create the bare venv offline from wheels this arm only reads."""
    receipt = run / 'ENVIRONMENT.json'
    if receipt.exists() and (VENV / 'bin/python3').exists():
        return json.loads(receipt.read_text())
    if VENV.exists():
        shutil.rmtree(VENV)
    venv.EnvBuilder(with_pip=True, symlinks=True, clear=True).create(VENV)
    wheels = sorted(path for path in WHEELS.glob('*.whl') if not path.name.startswith('._'))
    if not wheels:
        raise ValueError('no retained wheels to install from')
    log = subprocess.run([str(VENV / 'bin/python3'), '-m', 'pip', 'install', '--no-index',
                          '--no-deps', '--disable-pip-version-check', *map(str, wheels)],
                         capture_output=True, text=True, check=True)
    (run / 'install.log').write_text(log.stdout + log.stderr)
    configuration = (VENV / 'pyvenv.cfg').read_text()
    if 'include-system-site-packages = false' not in configuration:
        raise ValueError('system site packages are enabled')
    probe = ('import sys, site, json, numpy, torch, brotli; print(json.dumps(dict('
             'prefix=sys.prefix, base=sys.base_prefix, user_site=site.ENABLE_USER_SITE, '
             'packages={m.__name__: dict(version=m.__version__, path=m.__file__) '
             'for m in (numpy, torch, brotli)})))')
    observed = subprocess.run([str(VENV / 'bin/python3'), '-c', probe],
                              capture_output=True, text=True, check=True)
    inventory = json.loads(observed.stdout)
    if inventory['prefix'] != str(VENV) or inventory['user_site']:
        raise ValueError('the venv is not isolated')
    value = dict(environment=inventory, wheels=[fact(path) for path in wheels],
                 bare=True, score_claim=False, promotable=False)
    save(receipt, value)
    return value


def run_stage(run, name, command, environment, timeout=None):
    """Run one command, keep its log, and record it as a resumable stage."""
    receipt = run / f'{name}.DONE.json'
    if receipt.exists():
        return json.loads(receipt.read_text())
    log = run / f'{name}.log'
    started = time.time()
    with log.open('wb') as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                   env=environment, timeout=timeout)
    value = dict(argv=[str(item) for item in command], returncode=completed.returncode,
                 seconds=time.time() - started, log=fact(log),
                 score_claim=False, promotable=False)
    save(receipt, value)
    return value


def pair_digests(raw):
    """Hash the whole file once and each of the 600 pairs, in one pass."""
    whole = hashlib.sha256()
    pairs = []
    with raw.open('rb') as stream:
        for _ in range(RAW_BYTES // PAIR_BYTES):
            block = stream.read(PAIR_BYTES)
            if len(block) != PAIR_BYTES:
                raise ValueError('short read while hashing pairs')
            whole.update(block)
            pairs.append(hashlib.sha256(block).hexdigest())
        if stream.read(1):
            raise ValueError('raw is longer than 600 pairs')
    return whole.hexdigest(), pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-from', type=Path, required=True)
    arguments = parser.parse_args()
    run = arguments.resume_from.resolve()
    if not run.is_relative_to(ROOT) or run == ROOT:
        raise ValueError('proof store escaped the charter root')
    run.mkdir(parents=True, exist_ok=True)

    public = run / 'public'
    public.mkdir(exist_ok=True)
    originals = [fact(PUBLIC / name) for name in NAMES]
    for item in originals:
        source = Path(item['path'])
        destination = public / source.name
        if not destination.exists() or destination.read_bytes() != source.read_bytes():
            shutil.copyfile(source, destination)
            shutil.copymode(source, destination)
    if fact(public / 'archive.zip')['sha256'] != ARCHIVE_SHA:
        raise ValueError('archive.zip is not move 53')

    environment_receipt = build_environment(run)
    binding = dict(public_files=[fact(public / name) for name in NAMES], originals=originals,
                   runner=fact(Path(__file__)), environment=environment_receipt['environment'],
                   full_reference=fact(FULL_REFERENCE), pair_reference=fact(PAIR_REFERENCE),
                   axis='[macOS-CPU advisory]', seed=20260916,
                   score_claim=False, promotable=False)
    inputs = run / 'INPUTS.json'
    if inputs.exists() and json.loads(inputs.read_text()) != binding:
        raise ValueError('proof binding changed')
    save(inputs, binding)

    base = {key: value for key, value in os.environ.items() if not key.startswith('PYTHON')}
    base['PYTHONDONTWRITEBYTECODE'] = '1'
    base['PYTHONNOUSERSITE'] = '1'
    with_compiler = dict(base, PATH=f'{VENV / "bin"}:/usr/bin:/bin')
    if shutil.which('cc', path=with_compiler['PATH']) is None:
        raise ValueError('the compiler-present stages need a compiler')

    # 1. Without a compiler the shell must stop, loudly, before decoding anything.
    sandbox = REPO / '.omx/tmp/ddm_mrs7_no_compiler_bin'
    sandbox.mkdir(parents=True, exist_ok=True)
    for name in ('dirname', 'rm', 'echo'):
        link = sandbox / name
        if not link.is_symlink():
            link.symlink_to(Path('/usr/bin') / name if name != 'rm' else Path('/bin/rm'))
    without_compiler = dict(base, PATH=f'{VENV / "bin"}:{sandbox}')
    if shutil.which('cc', path=without_compiler['PATH']) is not None:
        raise ValueError('the no-compiler stage found a compiler')
    for name in LIBRARIES:
        (public / name).unlink(missing_ok=True)
    shell = ['/bin/sh', str(public / 'inflate.sh')]
    refusal = run_stage(run, 'no_compiler', shell + ['--help'], without_compiler, 120)
    refusal_text = Path(refusal['log']['path']).read_text()
    if refusal['returncode'] == 0 or 'no C compiler on PATH' not in refusal_text:
        raise ValueError('the shell did not refuse a missing compiler')
    if any((public / name).exists() for name in LIBRARIES):
        raise ValueError('a library survived the refused build')

    # 2. With a compiler the shell must build all three libraries and take no other path.
    help_stage = run_stage(run, 'help', shell + ['--help'], with_compiler, 300)
    help_text = Path(help_stage['log']['path']).read_text()
    if help_stage['returncode'] or not all((public / name).exists() for name in LIBRARIES):
        raise ValueError('the compiler-present shell did not build the three libraries')
    if any(word in help_text for word in FALLBACK_WORDS):
        raise ValueError('the receiver reported a fallback path')

    extracted, output = run / 'extracted', RAW_ROOT
    extracted.mkdir(exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(public / 'archive.zip') as archive:
        if archive.namelist() != ['p']:
            raise ValueError('unexpected archive contents')
        (extracted / 'p').write_bytes(archive.read('p'))
    file_list = run / 'file-list.txt'
    file_list.write_text('0.mkv\n')
    decode_argv = [str(extracted), str(output), str(file_list)]

    # 3. The default device is CUDA, and saying nothing must not silently give CPU output.
    guard = run_stage(run, 'default_guard', shell + decode_argv, with_compiler, 300)
    if guard['returncode'] != 1 or 'CUDA is required by default' not in Path(guard['log']['path']).read_text():
        raise ValueError('the default CUDA guard did not fire')

    raw = output / '0.raw'
    certificate = run / 'IDENTITY.json'
    if not certificate.exists():
        ssd_free = shutil.disk_usage(ROOT).free
        raw_free = shutil.disk_usage(RAW_ROOT).free
        save(run / 'STORAGE_PREFLIGHT.json', dict(
            receipt_tier=dict(path=str(ROOT), free_bytes=ssd_free, floor_bytes=FLOOR_BYTES),
            raw_tier=dict(path=str(RAW_ROOT), free_bytes=raw_free, transient_bytes=RAW_BYTES,
                          remaining_after_bytes=raw_free - RAW_BYTES),
            reason='The raw is transient and deleted once both comparisons pass; the shared SSD tier '
                   'cannot hold it above the 8 GiB floor, so it goes to local APFS scratch',
            retained_limit_bytes=2 * 1024 ** 3, score_claim=False, promotable=False))
        if ssd_free < FLOOR_BYTES:
            raise ValueError('the receipt tier is below its floor')
        if raw_free - RAW_BYTES < 8 * 1024 ** 3:
            raise ValueError('not enough local scratch for the cold raw')
        decode = run_stage(run, 'public_decode', shell + decode_argv + ['--device', 'cpu'],
                           with_compiler, 4 * 3600)
        decode_text = Path(decode['log']['path']).read_text()
        if decode['returncode'] or any(word in decode_text for word in FALLBACK_WORDS):
            raise ValueError('the cold decode failed or reported a fallback')
        # The frozen contract must accept the decoder's own report line.
        from tac.decode_wall_clock import _cold_public_report
        report = _cold_public_report({'artifacts': {'contest_auth_eval.stdout.log': decode_text}})
        reference_report = json.loads(FULL_REFERENCE.read_text())['inflate_report']
        if report['archive_sha256'] != ARCHIVE_SHA or report['archive_bytes'] != 179286:
            raise ValueError('the report names a different archive')
        if report['token_decoder']['decoded_token_sha256'] != reference_report['token_decoder']['decoded_token_sha256']:
            raise ValueError('the decoded token plane differs from move 53')
        if report['raw_sha256'] != RAW_SHA:
            raise ValueError('the report raw sha differs from move 53')
        observed = fact(raw)
        if observed['bytes'] != RAW_BYTES:
            raise ValueError('the raw is the wrong size')
        whole, pairs = pair_digests(raw)
        reference_pairs = json.loads(PAIR_REFERENCE.read_text())['pair_sha256']
        reference_full = json.loads(FULL_REFERENCE.read_text())['rendered_raw']['sha256']
        if reference_full != RAW_SHA or len(reference_pairs) != 600:
            raise ValueError('the move 53 references disagree with each other')
        differing = [index for index, value in enumerate(pairs) if value != reference_pairs[index]]
        save(certificate, dict(binding=binding, raw=observed, full_sha256=whole,
             reference_full_sha256=reference_full, matched_pairs=600 - len(differing),
             differing_pairs=differing, pair_sha256=pairs, decode=decode,
             cold_report=report, help=help_stage, no_compiler=refusal, default_guard=guard,
             reason='Full-file and 600 per-pair SHA equality with move 53',
             axis='[macOS-CPU advisory]', score_claim=False, promotable=False))
        if whole != reference_full or differing:
            raise ValueError(f'output differs from move 53 at {len(differing)} pairs')

    value = json.loads(certificate.read_text())
    if raw.exists():
        if fact(raw)['sha256'] != value['full_sha256']:
            raise ValueError('the raw changed after certification')
        save(run / 'RAW_CLEANUP.json', dict(raw=value['raw'], reason='Certified and reproducible '
             'from the eight public files; deleted to stay inside the retention cap',
             score_claim=False, promotable=False))
        raw.unlink()
    print(json.dumps(dict(matched_pairs=value['matched_pairs'], full_sha256=value['full_sha256'],
                          differing_pairs=value['differing_pairs'],
                          decode_seconds=value['decode']['seconds']), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
