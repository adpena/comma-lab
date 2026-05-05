"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``28:19: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``build_public_site_bundle.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'reports/graphs/build_public_site_bundle.py'
__recovery_spec__ = 'build_public_site_bundle.recovery_spec.json'
__recovery_ast_error__ = '28:19: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: build_public_site_bundle.cpython-312.pyc (Python 3.12)

'''Build a sanitized Cloudflare Pages bundle from reports/graphs/site.

The historical site directory contains useful generated artifacts, but some of
its JSON timelines preserve local operator paths for private custody. This
builder copies the site into a separate public bundle and redacts private ops
surfaces before the strict publish hygiene guard runs.
'''
from __future__ import annotations
import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tac.preflight import check_public_release_hygiene
DEFAULT_SOURCE = ROOT / 'site'
DEFAULT_OUTPUT = ROOT / 'public_site'
DEFAULT_MAX_ASSET_BYTES = 26214400
REDACTIONS: 'tuple[tuple[str, re.Pattern[str], str], ...]' = (('local_absolute_operator_path', re.compile('(?<![A-Za-z0-9_])/(?:Users|home|private|tmp|teamspace|var/folders)(?:/[^\\s)\\"\'<>`]*)?'), '${LOCAL_PATH_REDACTED}'), ('private_lightning_studio_url', re.compile('https://lightning\\.ai/[^/\\s)]+/[^/\\s)]+/studios/[^\\s)\\"\'<>]*'), '${LIGHTNING_PRIVATE_URL_REDACTED}'), ('vast_ssh_endpoint', re.compile('\\bssh\\d+\\.vast\\.ai(?::\\d+)?\\b'), '${VAST_SSH_REDACTED}'), ('api_token', re.compile('\\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,})\\b'), '${TOKEN_REDACTED}'), ('secret_env_assignment', re.compile('\\b(VAST_API_KEY|LIGHTNING_API_KEY|CLOUDFLARE_API_TOKEN|OPENAI_API_KEY)\\s*=\\s*[^\\s\\"\'`]+'), '\\1=${SECRET_REDACTED}'), ('modal_call_id', re.compile('\\bfc-[A-Z0-9]{20,}\\b'), '${MODAL_ID_REDACTED}'), ('modal_app_id', re.compile('\\bap-[A-Za-z0-9]{10,}\\b'), '${MODAL_ID_REDACTED}'))
RedactionRecord = <NODE:12>()

def _is_relative_to(path = None, parent = None):
    
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False



def _display_path(path = None):
    
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return 



def sanitize_text(text = None):
    records = []
    out = text
    for label, pattern, replacement in REDACTIONS:
        (out, count) = pattern.subn(replacement, out)
        if not count:
            continue
        records.append((label, count))
    return (out, records)


def _sanitize_file(path = None, root = None):
    pass
# WARNING: Decompyle incomplete


def _asset_size_violations(root = None, max_asset_bytes = None):
    violations = []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        if not path.stat().st_size > max_asset_bytes:
            continue
        violations.append(f'''{path.relative_to(root).as_posix()}:{path.stat().st_size}''')
    return violations


def _remove_oversized_assets(root = None, max_asset_bytes = None):
    omitted = []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size <= max_asset_bytes:
            continue
        omitted.append({
            'path': path.relative_to(root).as_posix(),
            'bytes': size })
        path.unlink()
    return omitted


def build_public_site_bundle(source = None, output = None, *, max_asset_bytes, oversized_policy, strict_hygiene):
    if oversized_policy not in frozenset({'fail', 'omit'}):
        raise ValueError("oversized_policy must be 'omit' or 'fail'")
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f'''missing source site directory: {source}''')
    if source == output or _is_relative_to(output, source):
        raise ValueError('output must not be inside source')
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(source, output)
    redactions = []
    for path in sorted(output.rglob('*')):
        if not path.is_file():
            continue
        redactions.extend(_sanitize_file(path, output))
    omitted_oversized_assets = []
    if oversized_policy == 'omit':
        omitted_oversized_assets = _remove_oversized_assets(output, max_asset_bytes)
    size_violations = _asset_size_violations(output, max_asset_bytes)
    if size_violations:
        raise None('\n'.join + (lambda .0: pass# WARNING: Decompyle incomplete
)(size_violations()))
    hygiene_violations = check_public_release_hygiene(repo_root = REPO_ROOT, strict = strict_hygiene, verbose = False, scan_paths = [
        output])
# WARNING: Decompyle incomplete


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--source', type = Path, default = DEFAULT_SOURCE)
    parser.add_argument('--output', type = Path, default = DEFAULT_OUTPUT)
    parser.add_argument('--max-asset-bytes', type = int, default = DEFAULT_MAX_ASSET_BYTES)
    parser.add_argument('--oversized-policy', choices = ('omit', 'fail'), default = 'omit')
    parser.add_argument('--no-strict-hygiene', action = 'store_true')
    args = parser.parse_args()
    manifest = build_public_site_bundle(args.source, args.output, max_asset_bytes = args.max_asset_bytes, oversized_policy = args.oversized_policy, strict_hygiene = not (args.no_strict_hygiene))
    print(json.dumps(manifest, indent = 2, sort_keys = True))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

"""
