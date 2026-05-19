# SPDX-License-Identifier: MIT
"""Tests for substrate trainer optimization-helper audit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_substrate_trainers_use_canonical_optimization_helpers,
)
from tac.trainer_optimization_helper_audit import (
    audit_trainer_file,
    audit_trainer_optimization_helpers,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _trainer(root: Path, name: str, text: str) -> Path:
    path = root / "experiments" / f"train_substrate_{name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_canonical_import_plus_assigned_call_is_accepted(tmp_path: Path) -> None:
    path = _trainer(
        tmp_path,
        "canonical",
        "from tac.substrates._shared.trainer_skeleton import "
        "build_optimized_training_context as build_ctx\n\n"
        "def train(args, scorers, pairs, model, device):\n"
        "    ctx = build_ctx(args, scorers=scorers, gt_pairs=pairs, "
        "substrate_model=model, device=device)\n"
        "    return ctx\n",
    )

    row = audit_trainer_file(path, tmp_path)

    assert row.accepted is True
    assert row.status == "canonical_helper"
    assert row.canonical_helper_assigned_call is True


def test_import_without_call_is_rejected(tmp_path: Path) -> None:
    path = _trainer(
        tmp_path,
        "import_only",
        "from tac.substrates._shared.trainer_skeleton import "
        "build_optimized_training_context\n\n"
        "def train():\n"
        "    return None\n",
    )

    row = audit_trainer_file(path, tmp_path)

    assert row.accepted is False
    assert row.canonical_helper_imported is True
    assert row.canonical_helper_assigned_call is False


@pytest.mark.parametrize(
    "helper_call",
    [
        "cache = build_gt_scorer_cache(target_pixels=x, posenet=p, segnet=s, device=d)",
        "model = compile_with_fallback(model, enabled=True)",
        "y = autocast_aware_forward(model, x, autocast_cfg=cfg)",
        "cache = GTScorerCache(pose_batches=[], seg_batches=[])",
    ],
)
def test_direct_training_optimization_helper_call_is_accepted(
    tmp_path: Path, helper_call: str
) -> None:
    path = _trainer(
        tmp_path,
        "direct",
        "from tac.training_optimization import build_gt_scorer_cache, "
        "compile_with_fallback, autocast_aware_forward, GTScorerCache\n\n"
        "def train(model, x, p, s, d, cfg):\n"
        f"    {helper_call}\n"
        "    return model\n",
    )

    row = audit_trainer_file(path, tmp_path)

    assert row.accepted is True
    assert row.status == "direct_helper"
    assert row.direct_helper_called is True


def test_docstring_and_comment_tokens_do_not_satisfy_contract(tmp_path: Path) -> None:
    path = _trainer(
        tmp_path,
        "comments_only",
        '"""build_optimized_training_context compile_with_fallback."""\n'
        "# build_gt_scorer_cache appears in a comment only\n"
        "def train():\n"
        "    return 'build_optimized_training_context'\n",
    )

    row = audit_trainer_file(path, tmp_path)

    assert row.accepted is False
    assert row.reason == "no_ast_import_plus_call_contract"


def test_explicit_comment_waiver_is_accepted(tmp_path: Path) -> None:
    path = _trainer(
        tmp_path,
        "sidecar",
        "# OPTIMIZATION_HELPERS_WAIVED:sidecar-no-train-hot-loop\n"
        "def main():\n"
        "    return 0\n",
    )

    row = audit_trainer_file(path, tmp_path)

    assert row.accepted is True
    assert row.status == "waived"


def test_placeholder_waiver_is_rejected(tmp_path: Path) -> None:
    path = _trainer(
        tmp_path,
        "placeholder",
        "# OPTIMIZATION_HELPERS_WAIVED:<reason>\n"
        "def main():\n"
        "    return 0\n",
    )

    row = audit_trainer_file(path, tmp_path)

    assert row.accepted is False
    assert row.waiver_present is False


def test_preflight_check_warn_only_reports_violations(tmp_path: Path) -> None:
    _trainer(tmp_path, "bad", "def train():\n    return None\n")

    violations = check_substrate_trainers_use_canonical_optimization_helpers(
        repo_root=tmp_path, strict=False, verbose=False
    )

    assert len(violations) == 1
    assert "train_substrate_bad.py" in violations[0]


def test_preflight_check_strict_raises(tmp_path: Path) -> None:
    _trainer(tmp_path, "bad", "def train():\n    return None\n")

    with pytest.raises(PreflightError):
        check_substrate_trainers_use_canonical_optimization_helpers(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_no_experiments_dir_is_empty(tmp_path: Path) -> None:
    audit = audit_trainer_optimization_helpers(tmp_path)

    assert audit.scanned_trainers == 0
    assert audit.violations == ()


def test_live_audit_has_known_nonzero_backfill_surface() -> None:
    audit = audit_trainer_optimization_helpers(REPO_ROOT)

    assert audit.scanned_trainers >= 40
    assert audit.missing_trainers >= 1
    assert any("train_substrate_s2sbs_byte_stuffing.py" in v for v in audit.violations)


def test_tool_json_output_matches_audit_schema() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "tools/check_trainer_optimization_helpers.py",
            "--json",
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(proc.stdout)

    assert payload["schema"] == "trainer_optimization_helper_audit_v1"
    assert payload["scanned_trainers"] >= 40
    assert "violations" in payload


def test_preflight_all_wires_trainer_optimization_helper_check() -> None:
    text = (REPO_ROOT / "src/tac/preflight.py").read_text(encoding="utf-8")

    assert "check_substrate_trainers_use_canonical_optimization_helpers" in text
    assert '"[trainer-optimization-helpers]"' in text
