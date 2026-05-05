# Source Generated with Decompyle++
# File: test_build_pr85_stbm1br_pr92_rmb1_randmulti_candidate.cpython-312.pyc (Python 3.12)

from __future__ import annotations
import importlib.util as importlib
import json
import sys
import zipfile
from pathlib import Path
import pytest
REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / 'experiments' / 'build_pr85_stbm1br_pr92_rmb1_randmulti_candidate.py'

def _load_script():
    spec = importlib.util.spec_from_file_location('build_pr85_stbm1br_pr92_rmb1_test', SCRIPT)
# WARNING: Decompyle incomplete

module = _load_script()

def _require_real_inputs():
    required = [
        module.DEFAULT_PR85_ARCHIVE,
        module.DEFAULT_STBM_ARCHIVE,
        module.DEFAULT_STBM_MANIFEST,
        module.DEFAULT_PR92_ARCHIVE,
        module.DEFAULT_PR92_PROFILE,
        module.DEFAULT_STBM_REPLAY_RUNTIME / 'inflate.py',
        module.DEFAULT_STBM_EXACT_T4]
# WARNING: Decompyle incomplete


def test_real_pr85_stbm1br_pr92_rmb1_candidate_builds_with_dispatch_guards(tmp_path = None):
    _require_real_inputs()
    summary = module.build_candidate(out_dir = tmp_path / 'candidate')
    manifest_path = REPO / summary['candidate_manifest'] if not Path(summary['candidate_manifest']).is_absolute() else Path(summary['candidate_manifest'])
    if not manifest_path.is_file():
        manifest_path = tmp_path / 'candidate' / module.CANDIDATE_ID / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding = 'utf-8'))
# WARNING: Decompyle incomplete

