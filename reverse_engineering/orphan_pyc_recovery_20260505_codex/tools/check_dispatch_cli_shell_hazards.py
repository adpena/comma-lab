"""RECOVERY STUB - pycdc output preserved inside r-string for hand-rehydration.

This file was decompiled from a .pyc orphan whose .py source was never
committed. pycdc 3.12 produces substantially-complete output but trips on
@dataclass/@property decorators, complex lambdas, and walrus operators,
so the raw output does not parse: ``37:10: invalid syntax``.

The raw pycdc output is preserved verbatim in ``_PYCDC_PARTIAL_OUTPUT``
below. The companion ``check_dispatch_cli_shell_hazards.recovery_spec.json`` contains co_names, co_consts,
co_varnames, and dis() output for every code object - the structural
ground-truth a hand-rehydrator should consult.

This stub itself is a no-op; importing it just exposes the partial
output as a string. Replace the stub with hand-rewritten Python once
rehydration is done.
"""
from __future__ import annotations

__recovery_status__ = "partial"
__recovery_orphan__ = 'tools/check_dispatch_cli_shell_hazards.py'
__recovery_spec__ = 'check_dispatch_cli_shell_hazards.recovery_spec.json'
__recovery_ast_error__ = '37:10: invalid syntax'

_PYCDC_PARTIAL_OUTPUT = r"""
# Source Generated with Decompyle++
# File: check_dispatch_cli_shell_hazards.cpython-312.pyc (Python 3.12)

\"\"\"Scan for dispatch CLI and shell portability hazards.

This is a lightweight guard for bug classes that cost wall-clock during the
May 2026 sprint:

* passing adjudicator-only flags to ``launch_lightning_batch_job.py``;
* using zsh's special ``path`` variable name in shell snippets;
* using GNU ``find -printf`` in local/macOS-facing snippets.

The scanner is intentionally conservative. Historical ledgers and result
artifacts are excluded by default, and Python heredocs inside shell scripts are
ignored so legitimate Python variables named ``path`` are not flagged.
\"\"\"
from __future__ import annotations
import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
DEFAULT_SCAN_PATHS = ('scripts', 'docs', 'reports')
DEFAULT_EXCLUDES = ('.git', '.omx', '__pycache__', 'experiments/results', 'reports/raw', 'runtime-rs/target')
TEXT_SUFFIXES = {
    '.py',
    '.sh',
    '.bash',
    '.md',
    '.rst',
    '.txt',
    '.zsh'}
LAUNCHER_STALE_FLAGS = {
    '--required-device': 'adjudicator-only; use launch_lightning_batch_job.py exact-eval --adjudicate instead',
    '--required-samples': 'adjudicator-only; use launch_lightning_batch_job.py exact-eval --adjudicate instead' }
Hazard = <NODE:12>()

def _repo_rel(path = None, root = None):
    
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return 



def _is_excluded(path = None, root = None, excludes = None):
    rel = _repo_rel(path, root)
    parts = set(Path(rel).parts)
    for item in excludes:
        if '/' in item:
            if not rel == item and rel.startswith(item.rstrip('/') + '/'):
                continue
            excludes
            return True
        if not item in parts:
            continue
        return True
    return False


def iter_scan_files(root = None, scan_paths = None, excludes = None):
    files = []
# WARNING: Decompyle incomplete

_HEREDOC_START_RE = re.compile('<<-?\\s*[\'\\"]?(?P<tag>[A-Za-z_][A-Za-z0-9_]*)[\'\\"]?')

def strip_heredoc_bodies(text = None):
    '''Return ``(original_lineno, line)`` pairs excluding heredoc bodies.'''
    out = []
    stop_tag = None
# WARNING: Decompyle incomplete


def _logical_commands(numbered_lines = None):
    commands = []
    current_lineno = None
    current = []
# WARNING: Decompyle incomplete


def _find_printf_is_remote_linux_context(line = None):
    if not '/workspace/' in line:
        '/workspace/' in line
        if not 'root@' in line:
            'root@' in line
            if not 'ssh ' in line:
                'ssh ' in line
                if not 'SSH_BASE' in line:
                    'SSH_BASE' in line
                    if not 'SSH_OPTS' in line:
                        'SSH_OPTS' in line
    return 'remote' in line.lower()


def scan_text(path = None, text = None, *, root):
    rel = _repo_rel(path, root)
    hazards = []
    numbered = strip_heredoc_bodies(text)
    for lineno, command in _logical_commands(numbered):
        if 'launch_lightning_batch_job.py' in command:
            for flag, reason in LAUNCHER_STALE_FLAGS.items():
                if not re.search(f'''(?<!\\S){re.escape(flag)}(?![A-Za-z0-9_-])''', command):
                    continue
                hazards.append(Hazard(rel, lineno, 'stale_lightning_launcher_flag', f'''{flag} passed to launch_lightning_batch_job.py; {reason}'''))
        if not 'find ' in command:
            continue
        if not '-printf' in command:
            continue
        if _find_printf_is_remote_linux_context(command):
            continue
        hazards.append(Hazard(rel, lineno, 'macos_find_printf', 'GNU find -printf is not available on macOS; use Python pathlib/stat or a POSIX/BSD-safe form for local commands'))
    if path.suffix.lower() in frozenset({'.md', '.rst', '.txt', '.zsh'}):
        for lineno, line in numbered:
            if not re.search('(^|[;&|]\\s*)(local\\s+)?path=', line) and re.search('\\bfor\\s+path\\s+in\\b', line) and re.search('\\bread\\b[^\\n#]*\\bpath\\b', line):
                continue
            hazards.append(Hazard(rel, lineno, 'zsh_path_special_variable', "avoid shell variable name 'path' in zsh-facing snippets; it mutates command lookup"))
    return hazards


def scan_paths(root = None, scan_paths = None, excludes = None):
    hazards = []
    for file_path in iter_scan_files(root, scan_paths = scan_paths, excludes = excludes):
        text = file_path.read_text(encoding = 'utf-8', errors = 'ignore')
        hazards.extend(scan_text(file_path, text, root = root))
    return hazards
    except OSError:
        continue


def build_parser():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--repo-root', type = Path, default = Path.cwd())
    parser.add_argument('--scan-path', action = 'append', default = [])
    parser.add_argument('--json-out', type = Path)
    parser.add_argument('--strict', action = 'store_true')
    return parser


def main(argv = None):
    args = build_parser().parse_args(argv)
    root = args.repo_root.resolve()
    scan_roots = tuple(args.scan_path) if args.scan_path else DEFAULT_SCAN_PATHS
    hazards = scan_paths(root, scan_paths = scan_roots)
# WARNING: Decompyle incomplete

if __name__ == '__main__':
    raise SystemExit(main())

"""
