"""Rebuild the bare test venv on APFS from retained offline predecessor wheels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import venv

from ddm_mrs1_bootstrap import base_interpreter, certified_cleanup, fact, save

ROOT = Path('/Volumes/APDataStore/pact/ddm_mrs4')
ENVIRONMENT = Path(__file__).resolve().parents[1] / '.omx/tmp/ddm_mrs4_bare_venv'
PREDECESSOR = Path('/Volumes/APDataStore/pact/ddm_mrs1/bootstrap/READY.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-from', type=Path, required=True)
    parser.add_argument('--cleanup', action='store_true')
    args = parser.parse_args()
    root = args.resume_from.resolve()
    if not root.is_relative_to(ROOT) or root == ROOT:
        raise ValueError('bootstrap store must be inside the charter root')
    root.mkdir(parents=True, exist_ok=True)
    wheels = root / 'wheels'
    wheels.mkdir(exist_ok=True)
    source = json.loads(PREDECESSOR.read_text())
    retained = []
    for item in source['wheels']:
        path = Path(item['path'])
        if fact(path) != item:
            raise ValueError('predecessor rebuild wheel changed')
        destination = wheels / path.name
        if not destination.exists():
            shutil.copyfile(path, destination)
        actual = fact(destination)
        if (actual['bytes'], actual['sha256']) != (item['bytes'], item['sha256']):
            raise ValueError('retained wheel copy differs')
        retained.append(actual)
    binding = dict(wheels=retained, base_interpreter=base_interpreter(), predecessor=fact(PREDECESSOR),
        environment=str(ENVIRONMENT), runner=fact(Path(__file__)),
        helper=fact(Path(__file__).with_name('ddm_mrs1_bootstrap.py')), score_claim=False, promotable=False)
    inputs = root / 'INPUTS.json'
    if inputs.exists() and json.loads(inputs.read_text()) != binding:
        raise ValueError('bootstrap inputs changed')
    save(inputs, binding)
    if args.cleanup:
        ready = json.loads((root / 'READY.json').read_text())
        if ready['binding'] != binding:
            raise ValueError('cleanup generation mismatch')
        certified_cleanup(root, ENVIRONMENT, ready, 'VENV_CLEANUP',
            'APFS test venv rebuildable from retained offline wheels and bound base interpreter')
        return
    if shutil.disk_usage(ENVIRONMENT.parent).free < 2 * 1024**3:
        raise ValueError('less than 2 GiB available for the APFS test venv')
    if not (ENVIRONMENT / 'pyvenv.cfg').exists():
        venv.EnvBuilder(with_pip=True, symlinks=True, system_site_packages=False).create(ENVIRONMENT)
    if 'include-system-site-packages = false' not in (ENVIRONMENT / 'pyvenv.cfg').read_text():
        raise ValueError('venv exposes system site packages')
    command = [str(ENVIRONMENT / 'bin/python3'), '-I', '-m', 'pip', 'install', '--no-cache-dir',
               '--no-index', '--no-deps', '--no-compile', *[item['path'] for item in retained]]
    with (root / 'install.log').open('w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    probe = ('import sys,site,numpy,torch,brotli,json; '
        'print(json.dumps(dict(prefix=sys.prefix,base=sys.base_prefix,path=sys.path,'
        'user_site=site.ENABLE_USER_SITE,executable=sys.executable,packages={m.__name__:'
        'dict(version=m.__version__,path=m.__file__) for m in (numpy,torch,brotli)})))')
    observed = json.loads(subprocess.check_output([str(ENVIRONMENT / 'bin/python3'), '-I', '-c', probe], text=True))
    if observed['prefix'] != str(ENVIRONMENT) or observed['user_site']:
        raise ValueError('bare interpreter isolation failed')
    for info in observed['packages'].values():
        if not Path(info['path']).is_relative_to(ENVIRONMENT):
            raise ValueError('dependency escaped isolated venv')
    save(root / 'READY.json', dict(binding=binding, environment=observed,
        install_command=command, bare=True, system_site_packages=False, argv=sys.argv))
    print(json.dumps(dict(status='ready', environment=str(ENVIRONMENT))), flush=True)


if __name__ == '__main__':
    main()
