"""Tests for B1 — encoder/decoder dequantization roundtrip scanner.

Bug class: encoder uses ``(rounded / N) * scale``, decoder uses
``rounded * scale``. The scanner detects tools that quantize + emit an
archive without a paired roundtrip test.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SCANNER = REPO_ROOT / "tools" / "check_encoder_decoder_dequantization_roundtrip_tested.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_b1_test", SCANNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_tool(repo: Path, rel: str, body: str) -> Path:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)
    return p


def test_b1_flags_quantize_and_archive_without_roundtrip(tmp_path: Path) -> None:
    mod = _load_module()
    repo = tmp_path
    _make_tool(
        repo,
        "tools/build_admm_x_lossy_coarsening_path_b_step6.py",
        '''
import zipfile
def encode(x, scale):
    rounded = (x / scale).round()
    return rounded * scale  # paired with archive emit
def emit(blob, out):
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("archive.zip", blob)
''',
    )
    findings = mod.scan(repo)
    assert len(findings) == 1
    assert "build_admm_x_lossy_coarsening_path_b_step6" in findings[0].rel_path


def test_b1_accepts_inline_marker(tmp_path: Path) -> None:
    mod = _load_module()
    repo = tmp_path
    _make_tool(
        repo,
        "tools/build_admm_x_lossy_coarsening_path_b_step6.py",
        '''
# ROUNDTRIP_TESTED: src/tac/tests/test_step6_roundtrip.py
import zipfile
def encode(x, scale):
    rounded = (x / scale).round()
    return rounded * scale
def emit(blob, out):
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("archive.zip", blob)
''',
    )
    findings = mod.scan(repo)
    assert findings == []


def test_b1_accepts_sibling_pytest_with_token(tmp_path: Path) -> None:
    mod = _load_module()
    repo = tmp_path
    _make_tool(
        repo,
        "tools/build_admm_x_lossy_coarsening_path_b_step6.py",
        '''
import zipfile
def encode(x, scale):
    rounded = (x / scale).round()
    return rounded * scale
def emit(blob, out):
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("archive.zip", blob)
''',
    )
    sib = repo / "src" / "tac" / "tests" / "test_build_admm_x_lossy_coarsening_path_b_step6_roundtrip.py"
    sib.parent.mkdir(parents=True, exist_ok=True)
    sib.write_text(
        '"""ENCODE_INFLATE_ROUNDTRIP smoke for step6 archive."""\n'
        "def test_encode_inflate_roundtrip():\n"
        "    pass\n"
    )
    findings = mod.scan(repo)
    assert findings == []


def test_b1_skips_tools_without_quant_arithmetic(tmp_path: Path) -> None:
    mod = _load_module()
    repo = tmp_path
    _make_tool(
        repo,
        "tools/pr101_orchestrator_empirical.py",
        '''
import zipfile
def emit(blob, out):
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("archive.zip", blob)
''',
    )
    findings = mod.scan(repo)
    assert findings == []


def test_b1_strict_exits_nonzero(tmp_path: Path) -> None:
    mod = _load_module()
    repo = tmp_path
    _make_tool(
        repo,
        "tools/build_admm_x_lossy_coarsening_path_b_step6.py",
        '''
import zipfile
def encode(x, scale):
    rounded = (x / scale).round()
    return rounded * scale
def emit(blob, out):
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("archive.zip", blob)
''',
    )
    rc = mod.main(["--repo-root", str(repo), "--strict"])
    assert rc == 1
