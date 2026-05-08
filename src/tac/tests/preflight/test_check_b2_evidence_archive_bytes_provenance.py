"""Tests for B2 — phantom byte-proxy claim scanner.

Every evidence row that sets ``empirical_archive_bytes`` must satisfy ONE of:
  1. archive_sha256 set,
  2. byte_proxy_only=True with all guards,
  3. proxy status tag (measured_config_status),
  4. textual provenance tag in source field.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SCANNER = REPO_ROOT / "tools" / "check_evidence_row_archive_bytes_has_provenance.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_b2_test", SCANNER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_b2_flags_row_with_no_provenance(tmp_path: Path) -> None:
    mod = _load_module()
    _write_jsonl(
        tmp_path / "reports" / "cathedral_autopilot_evidence.jsonl",
        [{"technique": "x", "empirical_archive_bytes": 137531, "source": "internal"}],
    )
    findings = mod.scan(tmp_path)
    assert len(findings) == 1


def test_b2_accepts_archive_sha256(tmp_path: Path) -> None:
    mod = _load_module()
    _write_jsonl(
        tmp_path / "reports" / "cathedral_autopilot_evidence.jsonl",
        [
            {
                "technique": "x",
                "empirical_archive_bytes": 137469,
                "archive_sha256": "c33243a1e367fc64466ff65dc11e267aa14",
                "source": "internal",
            }
        ],
    )
    findings = mod.scan(tmp_path)
    assert findings == []


def test_b2_accepts_byte_proxy_only_with_guards(tmp_path: Path) -> None:
    mod = _load_module()
    _write_jsonl(
        tmp_path / "reports" / "cathedral_autopilot_evidence.jsonl",
        [
            {
                "technique": "x",
                "empirical_archive_bytes": 137469,
                "byte_proxy_only": True,
                "cuda_eval_worth_testing": False,
                "ready_for_exact_eval_dispatch": False,
                "source": "internal",
            }
        ],
    )
    findings = mod.scan(tmp_path)
    assert findings == []


def test_b2_rejects_byte_proxy_only_without_all_guards(tmp_path: Path) -> None:
    mod = _load_module()
    _write_jsonl(
        tmp_path / "reports" / "cathedral_autopilot_evidence.jsonl",
        [
            {
                "technique": "x",
                "empirical_archive_bytes": 137469,
                "byte_proxy_only": True,
                "cuda_eval_worth_testing": True,  # missing guard
                "ready_for_exact_eval_dispatch": False,
                "source": "internal",
            }
        ],
    )
    findings = mod.scan(tmp_path)
    assert len(findings) == 1


def test_b2_accepts_textual_provenance_tag(tmp_path: Path) -> None:
    mod = _load_module()
    _write_jsonl(
        tmp_path / "reports" / "cathedral_autopilot_evidence.jsonl",
        [
            {
                "technique": "x",
                "empirical_archive_bytes": 100799,
                "source": "[CPU-prep empirical] reports/raw/pr101_int4_sweep/manifest.json",
            }
        ],
    )
    findings = mod.scan(tmp_path)
    assert findings == []


def test_b2_strict_exits_nonzero(tmp_path: Path) -> None:
    mod = _load_module()
    _write_jsonl(
        tmp_path / "reports" / "cathedral_autopilot_evidence.jsonl",
        [{"technique": "x", "empirical_archive_bytes": 137531, "source": "internal"}],
    )
    rc = mod.main(["--repo-root", str(tmp_path), "--strict"])
    assert rc == 1
