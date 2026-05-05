# Source Generated with Decompyle++
# File: test_dispatch_cli_shell_hazards.cpython-312.pyc (Python 3.12)

from __future__ import annotations
import importlib.util as importlib
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
HELPER = REPO / 'tools' / 'check_dispatch_cli_shell_hazards.py'

def _load_helper():
    spec = importlib.util.spec_from_file_location('dispatch_cli_shell_hazards_test', HELPER)
# WARNING: Decompyle incomplete


def test_scanner_catches_adjudicator_flags_passed_to_lightning_launcher(tmp_path = None):
    helper = _load_helper()
    script = tmp_path / 'scripts' / 'bad.sh'
    script.parent.mkdir()
    script.write_text('\n#!/usr/bin/env bash\npython scripts/launch_lightning_batch_job.py exact-eval \\\n  --job-name j \\\n  --archive a.zip \\\n  --required-device cuda \\\n  --required-samples 600\n'.lstrip())
    hazards = helper.scan_paths(tmp_path, scan_paths = ('scripts',))
# WARNING: Decompyle incomplete


def test_scanner_does_not_flag_adjudicator_flags_on_adjudicator_cli(tmp_path = None):
    helper = _load_helper()
    script = tmp_path / 'scripts' / 'good.sh'
    script.parent.mkdir()
    script.write_text('\n#!/usr/bin/env bash\npython scripts/adjudicate_contest_auth_eval.py \\\n  --contest-json contest_auth_eval.json \\\n  --required-device cuda \\\n  --required-samples 600\n'.lstrip())
# WARNING: Decompyle incomplete


def test_scanner_catches_zsh_path_variable_without_flagging_python_heredoc(tmp_path = None):
    helper = _load_helper()
    snippet = tmp_path / 'docs' / 'ops.md'
    snippet.parent.mkdir()
    snippet.write_text('\n```zsh\nfor path in artifacts/*.json; do\n  dirname "$path"\ndone\n```\n\n```bash\npython - <<\'PY\'\nfor path in sorted(root.rglob("*")):\n    print(path)\nPY\n```\n'.lstrip())
    hazards = helper.scan_paths(tmp_path, scan_paths = ('docs',))
# WARNING: Decompyle incomplete


def test_scanner_catches_local_find_printf_but_allows_remote_workspace_find(tmp_path = None):
    helper = _load_helper()
    script = tmp_path / 'scripts' / 'finds.sh'
    script.parent.mkdir()
    script.write_text('\n#!/usr/bin/env bash\nfind . -name \'*.json\' -printf \'%p\\n\'\nssh root@host "find /workspace/pact -name heartbeat.log -printf \'%T@\\n\'"\n'.lstrip())
    hazards = helper.scan_paths(tmp_path, scan_paths = ('scripts',))
# WARNING: Decompyle incomplete

