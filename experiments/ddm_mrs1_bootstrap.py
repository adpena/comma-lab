"""Prepare an isolated offline dependency environment and certify its cleanup."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import venv
import zipfile

ROOT = Path('/Volumes/APDataStore/pact/ddm_mrs1')
CACHE = Path('/Users/adpena/.cache/uv')
PACKAGES = {
    'torch': '2.12.1-cp313-cp313-macosx_14_0_arm64',
    'brotli': '1.2.0-cp313-cp313-macosx_10_13_universal2',
    'filelock': '3.29.6-py3-none-any',
    'typing-extensions': '4.16.0-py3-none-any',
    'setuptools': '81.0.0-py3-none-any',
    'sympy': '1.14.0-py3-none-any',
    'networkx': '3.6.1-py3-none-any',
    'jinja2': '3.1.6-py3-none-any',
    'fsspec': '2026.6.0-py3-none-any',
    'mpmath': '1.3.0-py3-none-any',
    'markupsafe': '3.0.3-cp313-cp313-macosx_11_0_arm64',
}


def fact(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)


def save(path, value):
    temporary = path.with_suffix('.pending')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def preserve_history(path):
    """Keep the old receipt or runner bytes before replacing their live name."""
    if not path.exists():
        return None
    digest = fact(path)['sha256']
    history = path.parent / 'history'
    history.mkdir(exist_ok=True)
    retained = history / f'{path.stem}.{digest}{path.suffix}'
    if not retained.exists():
        shutil.copyfile(path, retained)
    if fact(retained)['sha256'] != digest:
        raise ValueError('historical artifact changed')
    return fact(retained)


def base_interpreter():
    executable = Path(sys._base_executable).resolve(strict=True)
    prefix = Path(sys.base_prefix).resolve(strict=True)
    return dict(executable=fact(executable), prefix=str(prefix), version=sys.version,
                shared_libraries=[fact(p) for p in sorted((prefix / 'lib').glob('libpython*.dylib'))])


def environment_inventory(environment):
    files, links = [], []
    for path in sorted(environment.rglob('*')):
        if path.name.startswith('._') or path.name.endswith('.pending'):
            continue
        if path.is_symlink():
            links.append(dict(path=str(path), target=os.readlink(path)))
        elif path.is_file():
            files.append(fact(path))
    return files, links


def certified_cleanup(root, environment, rebuild, label, reason):
    """Keep the initial full inventory when resuming a partially finished removal."""
    certificate_path = root / f'{label}.json'
    done_path = root / f'{label}_DONE.json'
    new_generation = certificate_path.exists() and done_path.exists() and environment.exists()
    if new_generation:
        preserve_history(certificate_path)
        preserve_history(done_path)
        # A rebuilt environment is a new cleanup generation.
        create_certificate = True
    else:
        create_certificate = not certificate_path.exists()
    files, links = environment_inventory(environment)
    if create_certificate:
        if not environment.exists():
            raise ValueError('absent environment has no prior cleanup certificate')
        certificate = dict(files=files, symlinks=links, rebuild=rebuild, argv=sys.argv,
            reason=reason, environment=str(environment), score_claim=False, promotable=False)
        save(certificate_path, certificate)
        if new_generation:
            done_path.unlink()
    else:
        certificate = json.loads(certificate_path.read_text())
        if certificate['rebuild'] != rebuild or certificate.get('environment') != str(environment):
            raise ValueError('cleanup generation binding changed; keep remaining files')
        expected = {item['path']: item for item in certificate['files']}
        expected_links = {item['path']: item for item in certificate['symlinks']}
        if any(expected.get(item['path']) != item for item in files):
            raise ValueError('remaining files differ from initial cleanup certificate')
        if any(expected_links.get(item['path']) != item for item in links):
            raise ValueError('remaining symlinks differ from initial cleanup certificate')
    if environment.exists():
        shutil.rmtree(environment)
    save(done_path, dict(certificate=fact(certificate_path), environment_absent=True,
                        score_claim=False, promotable=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-from', type=Path, required=True)
    parser.add_argument('--cleanup-venv', action='store_true')
    args = parser.parse_args()
    root = args.resume_from.resolve()
    if not root.is_relative_to(ROOT):
        raise ValueError('bootstrap must remain in the charter store')
    root.mkdir(parents=True, exist_ok=True)
    wheels = root / 'wheels'
    wheels.mkdir(exist_ok=True)
    environment = root / 'venv'
    if args.cleanup_venv:
        ready = json.loads((root / 'READY.json').read_text())
        for item in ready['wheels']:
            if Path(item['path']).name.startswith('._'):
                continue
            if fact(Path(item['path'])) != item:
                raise ValueError('rebuild wheel changed')
        if ready.get('base_interpreter') != base_interpreter():
            raise ValueError('base Python prerequisite differs; keep environment')
        certified_cleanup(root, environment, ready, 'VENV_CLEANUP',
            'Rebuildable third-party installation from retained wheels and the bound base Python prerequisite')
        return
    if shutil.disk_usage(root).free < 2 * 1024**3:
        raise RuntimeError('less than 2 GiB available for isolated bootstrap')
    for name, version in PACKAGES.items():
        destination = wheels / f'{name.replace("-", "_")}-{version}.whl'
        receipt = destination.with_suffix('.json')
        if receipt.exists():
            if fact(destination) != json.loads(receipt.read_text())['wheel']:
                raise ValueError('retained wheel changed')
            continue
        source = (CACHE / 'wheels-v6/pypi' / name / version).resolve(strict=True)
        members = [p for p in sorted(source.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
                   and not p.name.startswith('._') and p.suffix != '.pyc']
        inputs = [dict(relative_path=str(p.relative_to(source)), **fact(p)) for p in members]
        pending = destination.with_suffix('.pending')
        with zipfile.ZipFile(pending, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for path in members:
                info = zipfile.ZipInfo(str(path.relative_to(source)), date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, path.read_bytes(), compresslevel=1)
        pending.replace(destination)
        save(receipt, dict(source=str(source), inputs=inputs, wheel=fact(destination),
             note='Repacked local wheel cache, not an original upstream wheel hash', argv=sys.argv))
        print(json.dumps(dict(wheel=name, bytes=destination.stat().st_size)), flush=True)
    numpy_source = CACHE / 'sdists-v9/pypi/numpy/1.26.4/15jxwJ9_nXOwnrh_/numpy-1.26.4-cp313-cp313-macosx_26_0_arm64.whl'
    numpy_target = wheels / numpy_source.name
    numpy_receipt = root / 'NUMPY_SOURCE.json'
    if numpy_receipt.exists():
        if fact(numpy_target) != json.loads(numpy_receipt.read_text())['retained']:
            raise ValueError('retained NumPy wheel changed')
    else:
        shutil.copyfile(numpy_source, numpy_target)
        if fact(numpy_target)['sha256'] != fact(numpy_source)['sha256']:
            raise ValueError('NumPy wheel copy differs')
        save(numpy_receipt, dict(source=fact(numpy_source), retained=fact(numpy_target)))
    interpreter = base_interpreter()
    expected_wheels = [wheels / f'{name.replace("-", "_")}-{version}.whl'
                       for name, version in PACKAGES.items()] + [numpy_target]
    wheel_facts = [fact(p) for p in sorted(expected_wheels)]
    if environment.exists() and not (environment / 'bin/python').is_symlink():
        certified_cleanup(root, environment,
            dict(wheels=wheel_facts, base_interpreter=interpreter), 'INCOMPLETE_VENV_CLEANUP',
            'Rebuildable incomplete venv; recreate with standard interpreter symlinks')
    if not (environment / 'bin/python').exists() or not (environment / 'pyvenv.cfg').exists():
        venv.EnvBuilder(with_pip=True, symlinks=True, system_site_packages=False).create(environment)
    elif subprocess.run([str(environment / 'bin/python'), '-I', '-m', 'pip', '--version'], capture_output=True).returncode:
        subprocess.run([str(environment / 'bin/python'), '-I', '-m', 'ensurepip', '--upgrade'], check=True)
    isolated = json.loads(subprocess.check_output([str(environment / 'bin/python'), '-I', '-c',
        'import sys,json; print(json.dumps(dict(prefix=sys.prefix,base=sys.base_prefix,path=sys.path)))'], text=True))
    if isolated['prefix'] != str(environment) or isolated['prefix'] == isolated['base']:
        raise ValueError('interpreter is not using the isolated environment')
    if 'include-system-site-packages = false' not in (environment / 'pyvenv.cfg').read_text():
        raise ValueError('virtual environment exposes system packages')
    command = [str(environment / 'bin/python'), '-I', '-m', 'pip', 'install', '--no-cache-dir',
               '--no-index', '--no-deps', '--no-compile', '--force-reinstall',
               *[item['path'] for item in wheel_facts]]
    preserve_history(root / 'install.log')
    with (root / 'install.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    freeze = subprocess.check_output([str(environment / 'bin/python'), '-I', '-m', 'pip', 'freeze'], text=True)
    packages = json.loads(subprocess.check_output([str(environment / 'bin/python'), '-I', '-c',
        'import numpy,torch,brotli,json; print(json.dumps({m.__name__:dict(version=m.__version__,path=m.__file__) for m in (numpy,torch,brotli)}))'], text=True))
    for name, version in [('numpy', '1.26.4'), ('torch', '2.12.1'), ('brotli', '1.2.0')]:
        if packages[name]['version'] != version or not Path(packages[name]['path']).resolve().is_relative_to(environment):
            raise ValueError('dependency version or isolated import path differs')
    previous_ready = preserve_history(root / 'READY.json')
    runner = Path(__file__).resolve()
    saved_runner = root / f'runner_{fact(runner)["sha256"]}.py'
    if not saved_runner.exists():
        shutil.copyfile(runner, saved_runner)
    if fact(saved_runner)['sha256'] != fact(runner)['sha256']:
        raise ValueError('retained bootstrap runner differs')
    save(root / 'READY.json', dict(argv=sys.argv, install_command=command, freeze=freeze,
         wheels=wheel_facts, runner=fact(runner), retained_runner=fact(saved_runner),
         previous_ready=previous_ready, base_interpreter=interpreter, packages=packages,
         interpreter_symlinks=[dict(path=str(p), target=os.readlink(p))
            for p in sorted((environment / 'bin').glob('python*')) if p.is_symlink()],
         isolation=isolated, bare=True, system_site_packages=False, score_claim=False, promotable=False))
    print(json.dumps(dict(status='ready', environment=str(environment))), flush=True)


if __name__ == '__main__':
    main()
