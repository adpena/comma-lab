# SPDX-License-Identifier: MIT
"""Tests for B8 — pr101 cross-paradigm torch.load(weights_only=False) allowlist.

Cross-paradigm tools (tools/pr101_*.py, tools/build_admm_*.py,
tools/build_cross_paradigm_*.py) calling torch.load(..., weights_only=False)
must EITHER carry a # WEIGHTS_ONLY_FALSE_OK: waiver in a 5-line window OR
have a sha256/magic-byte preceding-30-line validation.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SCANNER = REPO_ROOT / "tools" / "check_pr101_tools_torch_load_allowlist.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_b8_test", SCANNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make(repo: Path, rel: str, body: str) -> Path:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)
    return p


def test_b8_flags_unguarded_weights_only_false(tmp_path: Path) -> None:
    mod = _load_module()
    _make(
        tmp_path,
        "tools/pr101_x_empirical.py",
        '''
import torch
def main(p):
    sd = torch.load(p, map_location="cpu", weights_only=False)
    return sd
''',
    )
    findings = mod.scan(tmp_path)
    assert len(findings) == 1


def test_b8_accepts_inline_waiver(tmp_path: Path) -> None:
    mod = _load_module()
    _make(
        tmp_path,
        "tools/pr101_x_empirical.py",
        '''
import torch
def main(p):
    # WEIGHTS_ONLY_FALSE_OK:trusted-internal-pr101-state-dict
    sd = torch.load(p, map_location="cpu", weights_only=False)
    return sd
''',
    )
    findings = mod.scan(tmp_path)
    assert findings == []


def test_b8_accepts_preceding_sha_validation(tmp_path: Path) -> None:
    mod = _load_module()
    _make(
        tmp_path,
        "tools/pr101_x_empirical.py",
        '''
import hashlib, torch
def main(p):
    expected_sha = "abc"
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if h != expected_sha:
        raise SystemExit(2)
    sd = torch.load(p, map_location="cpu", weights_only=False)
    return sd
''',
    )
    findings = mod.scan(tmp_path)
    assert findings == []


def test_b8_flags_build_admm_too(tmp_path: Path) -> None:
    mod = _load_module()
    _make(
        tmp_path,
        "tools/build_admm_thing.py",
        '''
import torch
def main(p):
    sd = torch.load(p, weights_only=False)
    return sd
''',
    )
    findings = mod.scan(tmp_path)
    assert len(findings) == 1


def test_b8_strict_exits_nonzero(tmp_path: Path) -> None:
    mod = _load_module()
    _make(
        tmp_path,
        "tools/pr101_x_empirical.py",
        '''
import torch
def main(p):
    sd = torch.load(p, weights_only=False)
    return sd
''',
    )
    rc = mod.main(["--repo-root", str(tmp_path), "--strict"])
    assert rc == 1
