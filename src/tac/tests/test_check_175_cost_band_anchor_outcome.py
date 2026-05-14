# SPDX-License-Identifier: MIT
"""Tests for Catalog #175 — check_cost_band_anchor_writers_declare_outcome.

FIX-WAVE-1 R1 Medium #6 (2026-05-13). Defense-in-depth gate over the
``cost_band_calibration.py:333`` back-compat trap that allows untagged
anchors via ambient-default fallback. Refuses direct writes to
``cost_band_posterior.jsonl`` that don't go through the canonical
helper OR declare ``outcome=`` explicitly.

Memory: feedback_fix_wave_1_r1_findings_LANDED_20260513.md.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_cost_band_anchor_writers_declare_outcome,
)


def _mk(root: Path, rel: str, body: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def test_check_175_live_count_zero():
    """The check MUST have 0 live violations at landing (strict-flip atom)."""
    repo_root = Path(__file__).resolve().parents[3]
    violations = check_cost_band_anchor_writers_declare_outcome(
        repo_root=repo_root, strict=False, verbose=False
    )
    assert violations == [], (
        f"Live violations should be 0, got: {violations}"
    )


def test_check_175_detects_bare_write_text_without_outcome(tmp_path):
    _mk(
        tmp_path,
        "tools/bad_writer.py",
        "from pathlib import Path\n"
        "import json\n"
        "def go():\n"
        "    p = Path('.omx/state/cost_band_posterior.jsonl')\n"
        "    p.write_text(json.dumps({'platform': 'modal', 'sec': 60}))\n",
    )
    violations = check_cost_band_anchor_writers_declare_outcome(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert any("bad_writer.py" in v for v in violations)


def test_check_175_strict_raises_preflight_error(tmp_path):
    _mk(
        tmp_path,
        "tools/bad_writer.py",
        "from pathlib import Path\n"
        "p = Path('.omx/state/cost_band_posterior.jsonl')\n"
        "p.write_text('{}')\n",
    )
    with pytest.raises(PreflightError):
        check_cost_band_anchor_writers_declare_outcome(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_check_175_accepts_canonical_helper_in_window(tmp_path):
    _mk(
        tmp_path,
        "tools/good_writer.py",
        "from pathlib import Path\n"
        "from tac.cost_band_calibration import append_anchor\n"
        "def go():\n"
        "    # Note path mentioned for clarity:\n"
        "    # cost_band_posterior.jsonl\n"
        "    append_anchor(platform='modal', gpu='T4', sec=60,\n"
        "                  outcome='success', usd=0.30)\n"
        "    # Then a confirmation read\n"
        "    p = Path('.omx/state/cost_band_posterior.jsonl')\n"
        "    p.write_text('{}')\n",
    )
    violations = check_cost_band_anchor_writers_declare_outcome(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_175_accepts_outcome_kwarg_at_call_site(tmp_path):
    _mk(
        tmp_path,
        "tools/explicit_outcome.py",
        "from pathlib import Path\n"
        "import json\n"
        "outcome = 'success'\n"
        "def go(record_outcome):\n"
        "    record = {'platform': 'modal', 'outcome=' + record_outcome: 1}\n"
        "    p = Path('.omx/state/cost_band_posterior.jsonl')\n"
        "    p.write_text(json.dumps(record))\n",
    )
    violations = check_cost_band_anchor_writers_declare_outcome(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_175_accepts_same_line_waiver(tmp_path):
    _mk(
        tmp_path,
        "tools/waived.py",
        "from pathlib import Path\n"
        "p = Path('.omx/state/cost_band_posterior.jsonl')\n"
        "p.write_text('{}')  # COST_BAND_ANCHOR_OUTCOME_OK: migration shim\n",
    )
    violations = check_cost_band_anchor_writers_declare_outcome(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_175_accepts_append_platform_training_anchor(tmp_path):
    _mk(
        tmp_path,
        "tools/platform_writer.py",
        "from pathlib import Path\n"
        "from tac.cost_band_calibration import append_platform_training_anchor\n"
        "def go():\n"
        "    append_platform_training_anchor('modal', gpu='T4', sec=60,\n"
        "                                    outcome='success')\n"
        "    p = Path('.omx/state/cost_band_posterior.jsonl')\n"
        "    p.write_text('{}')\n",
    )
    violations = check_cost_band_anchor_writers_declare_outcome(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_175_exempts_canonical_helper_module(tmp_path):
    _mk(
        tmp_path,
        "src/tac/cost_band_calibration.py",
        "from pathlib import Path\n"
        "p = Path('.omx/state/cost_band_posterior.jsonl')\n"
        "p.write_text('{}')  # canonical helper - it's allowed\n",
    )
    violations = check_cost_band_anchor_writers_declare_outcome(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_175_ignores_test_files(tmp_path):
    _mk(
        tmp_path,
        "src/tac/tests/test_bad.py",
        "from pathlib import Path\n"
        "p = Path('.omx/state/cost_band_posterior.jsonl')\n"
        "p.write_text('{}')\n",
    )
    violations = check_cost_band_anchor_writers_declare_outcome(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_175_ignores_intake_clones(tmp_path):
    _mk(
        tmp_path,
        "experiments/results/public_pr_99_intake_codex/src/bad.py",
        "from pathlib import Path\n"
        "p = Path('.omx/state/cost_band_posterior.jsonl')\n"
        "p.write_text('{}')\n",
    )
    violations = check_cost_band_anchor_writers_declare_outcome(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_175_json_dump_form(tmp_path):
    _mk(
        tmp_path,
        "tools/dump_form.py",
        "import json\n"
        "def go():\n"
        "    with open('.omx/state/cost_band_posterior.jsonl', 'w') as f:\n"
        "        json.dump({'platform': 'modal'}, f)\n",
    )
    # This file doesn't have the write_text / json.dumps / json.dump triggers
    # on the SAME line as the posterior token; the check requires both on the
    # same line. (Conservative scoping by design - reduces false positives.)
    violations = check_cost_band_anchor_writers_declare_outcome(
        repo_root=tmp_path, strict=False, verbose=False
    )
    # json.dump appears on a different line from the path token; conservative
    # gate skips. Sister behavior to Catalog #131 / #138 lookback windows.
    # Adding a same-line trigger would be a stronger surface.
    assert violations == []


def test_check_175_v1_filename_variant_detected(tmp_path):
    _mk(
        tmp_path,
        "tools/v1_writer.py",
        "from pathlib import Path\n"
        "p = Path('.omx/state/cost_band_posterior_v1.jsonl')\n"
        "p.write_text('{}')\n",
    )
    violations = check_cost_band_anchor_writers_declare_outcome(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert any("v1_writer.py" in v for v in violations)
