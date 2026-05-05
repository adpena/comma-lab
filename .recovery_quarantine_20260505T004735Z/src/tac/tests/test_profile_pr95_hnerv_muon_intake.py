# Source Generated with Decompyle++
# File: test_profile_pr95_hnerv_muon_intake.cpython-312.pyc (Python 3.12)

from __future__ import annotations
import importlib.util as importlib
import sys
from pathlib import Path
import pytest
REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / 'experiments/profile_pr95_hnerv_muon_intake.py'
INTAKE_DIR = REPO_ROOT / 'experiments/results/public_pr95_intake_20260504_codex'
ARCHIVE = INTAKE_DIR / 'archive.zip'
SOURCE_DIR = INTAKE_DIR / 'pr95_src/submissions/hnerv_muon'
STATIC_INTAKE = INTAKE_DIR / 'pr95_static_intake.json'

def load_module():
    spec = importlib.util.spec_from_file_location('profile_pr95_hnerv_muon_intake', MODULE_PATH)
# WARNING: Decompyle incomplete


def require_pr95_intake():
    pass
# WARNING: Decompyle incomplete


def test_pr95_hnerv_muon_profile_parses_archive_source_and_readiness():
    require_pr95_intake()
    module = load_module()
    profile = module.build_profile(ARCHIVE, SOURCE_DIR, STATIC_INTAKE)
# WARNING: Decompyle incomplete


def test_pr95_hnerv_muon_hooks_include_prior_repo_gems():
    require_pr95_intake()
    module = load_module()
    profile = module.build_profile(ARCHIVE, SOURCE_DIR, STATIC_INTAKE)
# WARNING: Decompyle incomplete


def test_pr95_hnerv_muon_score_terms_recompute_static_inputs():
    require_pr95_intake()
    if not STATIC_INTAKE.exists():
        pytest.skip('PR95 static intake JSON missing; cannot test score-term recomputation')
    module = load_module()
    profile = module.build_profile(ARCHIVE, SOURCE_DIR, STATIC_INTAKE)
    score_terms = profile['score_term_math']['score_terms_from_static_intake']
# WARNING: Decompyle incomplete

