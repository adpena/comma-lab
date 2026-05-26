# SPDX-License-Identifier: MIT
"""Cascade C' trainer wrapper authority-label guards."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINER = (
    REPO_ROOT
    / "experiments"
    / "train_substrate_cascade_c_prime_frame_1_segnet_waterfill.py"
)


def test_smoke_completion_marker_uses_local_research_axis() -> None:
    """Smoke runs must not print contest-axis completion markers."""

    text = TRAINER.read_text(encoding="utf-8")
    assert 'completion_axis_tag = str(stats.get("axis_tag") or "[advisory only]")' in text
    assert 'if not stats.get("auth_eval_skipped_reason"):' in text
    assert 'f"axis={completion_axis_tag} elapsed={elapsed_total:.1f}s"' in text
    assert 'axis=[contest-{axis_label}]' not in text


def test_generated_inflate_runtime_defaults_to_python3() -> None:
    """Generated runtime should execute on hosts without a `python` shim."""

    text = TRAINER.read_text(encoding="utf-8")
    assert '#!/usr/bin/env python3' in text
    assert (
        'PYBIN="${PYBIN:-${PYTHON:-${PYTHON_BIN:-${PACT_PYTHON_BIN:-${UV_PYTHON:-python3}}}}}"'
        in text
    )
    assert 'PYBIN="${PYBIN:-python}"' not in text


def test_generated_inflate_shim_calls_nullary_main_cli() -> None:
    """Top-level shim must match the vendored main_cli signature."""

    text = TRAINER.read_text(encoding="utf-8")
    assert "from tac.substrates.cascade_c_prime_frame_1_segnet_waterfill.inflate import main_cli as main" in text
    assert "sys.exit(main())" in text
    assert "sys.exit(main(sys.argv[1:]))" not in text
