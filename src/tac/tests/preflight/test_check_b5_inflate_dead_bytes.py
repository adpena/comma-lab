# SPDX-License-Identifier: MIT
"""Tests for B5 — inflate wire-format dead-bytes scanner.

Variables read via struct.unpack/read in inflate.py must be loaded in
state-dict assembly OR carry a ``# DEAD_BYTES_AUDIT_OK:`` annotation.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SCANNER = REPO_ROOT / "tools" / "check_inflate_wire_format_no_dead_bytes.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_b5_test", SCANNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_inflate(repo: Path, rel: str, body: str) -> None:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)


def test_b5_flags_unused_unpack_target(tmp_path: Path) -> None:
    mod = _load_module()
    _make_inflate(
        tmp_path,
        "experiments/results/lane_x/submission_dir/inflate.py",
        '''
import struct
def main(buf):
    n_tensors = struct.unpack("<I", buf[:4])[0]  # used
    audit_token = struct.unpack("<I", buf[4:8])[0]  # NOT used
    return [n_tensors]
''',
    )
    findings = mod.scan(tmp_path)
    assert any("audit_token" in f.var_name for f in findings)
    # n_tensors is loaded later (return [n_tensors]) → not flagged
    assert not any(f.var_name == "n_tensors" for f in findings)


def test_b5_accepts_dead_bytes_audit_ok_waiver(tmp_path: Path) -> None:
    mod = _load_module()
    _make_inflate(
        tmp_path,
        "experiments/results/lane_x/submission_dir/inflate.py",
        '''
import struct
def main(buf):
    audit_token = struct.unpack("<I", buf[4:8])[0]  # DEAD_BYTES_AUDIT_OK:reproducibility-checksum
    return []
''',
    )
    findings = mod.scan(tmp_path)
    assert findings == []


def test_b5_skips_vendored_public_pr_intakes(tmp_path: Path) -> None:
    mod = _load_module()
    # Use canonical vendored marker
    rel = "experiments/results/public_pr_archive_release_view/lane_x/submission_dir/inflate.py"
    _make_inflate(
        tmp_path,
        rel,
        '''
import struct
def main(buf):
    audit_token = struct.unpack("<I", buf[4:8])[0]
    return []
''',
    )
    findings = mod.scan(tmp_path)
    assert findings == []


def test_b5_skips_intake_marker_paths(tmp_path: Path) -> None:
    mod = _load_module()
    rel = "experiments/results/public_pr91_intake_20260504_worker/x/inflate.py"
    _make_inflate(
        tmp_path,
        rel,
        '''
import struct
def main(buf):
    audit_token = struct.unpack("<I", buf[4:8])[0]
    return []
''',
    )
    findings = mod.scan(tmp_path)
    assert findings == []


_VIOLATION_BODY = '''
import struct
def main(buf):
    audit_token = struct.unpack("<I", buf[4:8])[0]
    return []
'''


def test_b5_skips_frozen_d1_20260515_sweep_artifacts(tmp_path: Path) -> None:
    # r94 adjudication 2026-08-27: the 2026-05-15 d1 sweep population is
    # frozen custody — its `_lambda` template defect is cured at the
    # GENERATOR, and the stamped artifacts are excluded by (d1_ prefix AND
    # 20260515 date) path components.
    mod = _load_module()
    rel = (
        "experiments/results/d1_payload_budget_sweep_real_a1_20260515_codex/"
        "arm_x/submission_dir/inflate.py"
    )
    _make_inflate(tmp_path, rel, _VIOLATION_BODY)
    findings = mod.scan(tmp_path)
    assert findings == []


def test_b5_exclusion_is_precision_scoped_not_blanket(tmp_path: Path) -> None:
    # Negative direction: the frozen-population rule must NOT swallow live
    # coverage — a d1 dir from another date, and a same-date dir without the
    # d1_ prefix, are both still scanned and flagged.
    mod = _load_module()
    _make_inflate(
        tmp_path,
        "experiments/results/d1_new_sweep_20260901_codex/a/submission_dir/inflate.py",
        _VIOLATION_BODY,
    )
    _make_inflate(
        tmp_path,
        "experiments/results/other_lane_20260515_codex/b/submission_dir/inflate.py",
        _VIOLATION_BODY,
    )
    findings = mod.scan(tmp_path)
    flagged_dirs = {f.rel_path.split("/")[2] for f in findings}
    assert "d1_new_sweep_20260901_codex" in flagged_dirs
    assert "other_lane_20260515_codex" in flagged_dirs


def test_b5_strict_exits_nonzero(tmp_path: Path) -> None:
    mod = _load_module()
    _make_inflate(
        tmp_path,
        "experiments/results/lane_x/submission_dir/inflate.py",
        '''
import struct
def main(buf):
    audit_token = struct.unpack("<I", buf[4:8])[0]
    return []
''',
    )
    rc = mod.main(["--repo-root", str(tmp_path), "--strict"])
    assert rc == 1
