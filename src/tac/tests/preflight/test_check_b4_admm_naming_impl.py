"""Tests for B4 — ADMM naming-vs-implementation drift scanner.

Files/classes/functions named ``admm`` or ``primal_dual`` must contain real
iterative consensus updates (rho/z/u) OR be renamed OR carry a
``# ADMM_WAIVED:`` annotation.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SCANNER = REPO_ROOT / "tools" / "check_admm_naming_matches_iterative_consensus_implementation.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_b4_test", SCANNER)
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


def test_b4_flags_admm_named_function_without_consensus(tmp_path: Path) -> None:
    mod = _load_module()
    _make(
        tmp_path,
        "tools/build_admm_thing.py",
        '''
def run_admm_alloc(state, target):
    lo, hi = 0, 1e9
    for _ in range(50):
        mid = (lo + hi) / 2
        if probe(mid, state) < target:
            lo = mid
        else:
            hi = mid
    return mid
def probe(x, s):
    return x
''',
    )
    findings = mod.scan(tmp_path)
    assert any(f.kind == "file" for f in findings) or any(
        f.kind == "function" and "admm" in f.name.lower() for f in findings
    )


def test_b4_accepts_real_admm_with_rho_z_u(tmp_path: Path) -> None:
    mod = _load_module()
    _make(
        tmp_path,
        "tools/build_admm_real.py",
        '''
def run_admm(x_init, A, b):
    rho = 1.0
    z = x_init.copy()
    u = 0.0 * x_init
    for _ in range(50):
        x = solve_x(A, b, z, u, rho)
        z_new = soft_threshold(x + u, 1 / rho)
        u = u + x - z_new
        z = z_new
        rho = rho * 1.05
    return x
def solve_x(*a, **k): return None
def soft_threshold(*a, **k): return None
''',
    )
    findings = mod.scan(tmp_path)
    # File-level should not trigger because body has rho/z/u + for loop.
    assert not any(f.kind == "file" for f in findings)


def test_b4_accepts_admm_waived_marker(tmp_path: Path) -> None:
    mod = _load_module()
    _make(
        tmp_path,
        "tools/build_admm_named_for_legacy.py",
        '''
# ADMM_WAIVED:legacy-naming-from-prior-art
def run_admm_alloc(state, target):
    lo, hi = 0, 1e9
    for _ in range(50):
        mid = (lo + hi) / 2
    return mid
''',
    )
    findings = mod.scan(tmp_path)
    assert findings == []


def test_b4_skips_files_without_admm_string(tmp_path: Path) -> None:
    mod = _load_module()
    _make(
        tmp_path,
        "tools/build_lagrangian_alloc.py",
        '''
def run_alloc(state, target):
    return state
''',
    )
    findings = mod.scan(tmp_path)
    assert findings == []


def test_b4_strict_exits_nonzero(tmp_path: Path) -> None:
    mod = _load_module()
    _make(
        tmp_path,
        "tools/build_admm_thing.py",
        '''
def run_admm_alloc(x):
    return x
''',
    )
    rc = mod.main(["--repo-root", str(tmp_path), "--strict"])
    assert rc == 1
