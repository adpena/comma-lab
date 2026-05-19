# SPDX-License-Identifier: MIT
"""Tests for Catalog #249 — phantom-score directory class permanent fix.

Anchor: Z3 v2 FULL Modal A100 dispatch 2026-05-15. Trainer hardcoded
``contest_auth_eval_cuda.json`` regardless of ``auth_eval_device``; Modal
dispatcher injected ``AUTH_EVAL_DEVICE=cpu``; CPU-evaluated bytes landed in
a file named ``*_cuda.json`` and a directory named ``*_cuda_work/``. Parent
agent quoted "0.19869 [contest-CUDA T4]" from the FILENAME despite metadata
saying CPU. Paired re-eval revealed true CUDA = 0.2317.

This test file exercises:
1. The runtime auto-redirect helper at
   ``tac.substrates._shared.smoke_auth_eval_gate._redirect_output_json_to_match_device``
2. The STRICT preflight gate
   ``tac.preflight.check_no_misleading_device_named_output_directories``
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_no_misleading_device_named_output_directories,
)
from tac.source_index import source_index_context
from tac.substrates._shared.smoke_auth_eval_gate import (
    _redirect_output_json_to_match_device,
)


# ----------------------------------------------------------------------------
# Runtime auto-redirect helper tests (the actual lie-extinction surface).
# ----------------------------------------------------------------------------


def test_redirect_preserves_device_agnostic_filename(tmp_path: Path) -> None:
    """Device-agnostic `contest_auth_eval.json` is honest by construction."""

    p = tmp_path / "contest_auth_eval.json"
    out = _redirect_output_json_to_match_device(p, "cpu", substrate_tag="t")
    assert out == p
    out2 = _redirect_output_json_to_match_device(p, "cuda", substrate_tag="t")
    assert out2 == p


def test_redirect_preserves_matched_cuda_filename(tmp_path: Path) -> None:
    """`*_cuda.json` with cuda device is the truth — preserve."""

    p = tmp_path / "contest_auth_eval_cuda.json"
    out = _redirect_output_json_to_match_device(p, "cuda", substrate_tag="t")
    assert out == p


def test_redirect_preserves_matched_cpu_filename(tmp_path: Path) -> None:
    """`*_cpu.json` with cpu device is the truth — preserve."""

    p = tmp_path / "contest_auth_eval_cpu.json"
    out = _redirect_output_json_to_match_device(p, "cpu", substrate_tag="t")
    assert out == p


def test_redirect_rewrites_cuda_filename_when_device_is_cpu_z3_anchor(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The Z3 v2 anchor: caller passes `*_cuda.json` but device is cpu."""

    p = tmp_path / "contest_auth_eval_cuda.json"
    out = _redirect_output_json_to_match_device(p, "cpu", substrate_tag="z3")
    assert out.name == "contest_auth_eval_cpu.json"
    assert out.parent == p.parent
    captured = capsys.readouterr()
    assert "[phantom-score-fix]" in captured.out
    assert "z3" in captured.out
    assert "Catalog #249" in captured.out


def test_redirect_rewrites_cpu_filename_when_device_is_cuda(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reverse case: caller passes `*_cpu.json` but device is cuda."""

    p = tmp_path / "contest_auth_eval_cpu.json"
    out = _redirect_output_json_to_match_device(p, "cuda", substrate_tag="t")
    assert out.name == "contest_auth_eval_cuda.json"
    captured = capsys.readouterr()
    assert "[phantom-score-fix]" in captured.out


def test_redirect_handles_mps_token(tmp_path: Path) -> None:
    """MPS device tokens also recognized."""

    # mps in filename, cuda actual — rewrite
    p = tmp_path / "auth_eval_result_mps.json"
    out = _redirect_output_json_to_match_device(p, "cuda", substrate_tag="t")
    assert out.name == "auth_eval_result_cuda.json"


def test_redirect_handles_unrelated_filenames(tmp_path: Path) -> None:
    """A path with no recognized device suffix is preserved."""

    p = tmp_path / "manifest.json"
    out = _redirect_output_json_to_match_device(p, "cpu", substrate_tag="t")
    assert out == p


# ----------------------------------------------------------------------------
# STRICT preflight gate tests.
# ----------------------------------------------------------------------------


def test_check_249_clean_repo_no_violations_in_self_exempt_files(
    tmp_path: Path,
) -> None:
    """A repo with no in-scope files returns 0 violations."""

    violations = check_no_misleading_device_named_output_directories(
        repo_root=tmp_path
    )
    assert violations == []


def test_check_249_flags_hardcoded_cuda_json_literal_in_substrate(
    tmp_path: Path,
) -> None:
    """The Z3 v2 bug class: hardcoded `contest_auth_eval_cuda.json`."""

    src = tmp_path / "experiments"
    src.mkdir()
    (src / "train_substrate_demo.py").write_text(
        "from pathlib import Path\n"
        "out_dir = Path('/tmp/x')\n"
        "result_json_path = out_dir / 'contest_auth_eval_cuda.json'\n"
    )
    violations = check_no_misleading_device_named_output_directories(
        repo_root=tmp_path
    )
    assert len(violations) == 1
    assert "_cuda" in violations[0]
    assert "contest_auth_eval_cuda.json" in violations[0]


def test_check_249_source_index_prefilter_matches_plain_scan(
    tmp_path: Path,
) -> None:
    """Indexed prefiltering narrows the candidate set without changing truth."""

    src = tmp_path / "experiments"
    src.mkdir()
    violation = src / "train_substrate_demo.py"
    violation.write_text(
        "from pathlib import Path\n"
        "result_json_path = Path('/tmp/contest_auth_eval_cuda.json')\n"
    )
    (src / "device_only_decoy.py").write_text("p = '/tmp/not_auth_cuda.json'\n")
    (src / "context_only_decoy.py").write_text("p = '/tmp/contest_auth_eval.json'\n")

    plain = check_no_misleading_device_named_output_directories(
        repo_root=tmp_path
    )
    with source_index_context(tmp_path) as index:
        indexed = check_no_misleading_device_named_output_directories(
            repo_root=tmp_path
        )
        stats = index.stats()

    assert indexed == plain
    assert len(indexed) == 1
    assert "train_substrate_demo.py" in indexed[0]
    assert "device_only_decoy.py" not in "\n".join(indexed)
    assert "context_only_decoy.py" not in "\n".join(indexed)
    assert stats["substring_index_entries"] >= 2


def test_check_249_accepts_same_line_waiver_with_rationale(
    tmp_path: Path,
) -> None:
    """Same-line `# DEVICE_NAMED_DIR_OK:<rationale>` opt-out is honored."""

    src = tmp_path / "experiments"
    src.mkdir()
    (src / "train_substrate_demo.py").write_text(
        "from pathlib import Path\n"
        "out_dir = Path('/tmp/x')\n"
        "p = out_dir / 'contest_auth_eval_cuda.json'  "
        "# DEVICE_NAMED_DIR_OK:cuda-only benchmark sweep fixture\n"
    )
    violations = check_no_misleading_device_named_output_directories(
        repo_root=tmp_path
    )
    assert violations == []


def test_check_249_rejects_placeholder_waiver(tmp_path: Path) -> None:
    """Placeholder `<rationale>` literal must NOT self-waive."""

    src = tmp_path / "experiments"
    src.mkdir()
    (src / "train_substrate_demo.py").write_text(
        "from pathlib import Path\n"
        "p = Path('/tmp/contest_auth_eval_cuda.json')  "
        "# DEVICE_NAMED_DIR_OK:<rationale>\n"
    )
    violations = check_no_misleading_device_named_output_directories(
        repo_root=tmp_path
    )
    assert len(violations) == 1


def test_check_249_excludes_test_files(tmp_path: Path) -> None:
    """Test files (test_*.py / *_test.py / /tests/) excluded by scope."""

    src = tmp_path / "src" / "tac" / "tests"
    src.mkdir(parents=True)
    (src / "test_something.py").write_text(
        "from pathlib import Path\n"
        "p = Path('/tmp/contest_auth_eval_cuda.json')\n"
    )
    violations = check_no_misleading_device_named_output_directories(
        repo_root=tmp_path
    )
    assert violations == []


def test_check_249_excludes_results_artifacts(tmp_path: Path) -> None:
    """`experiments/results/` is DERIVED_OUTPUT (Catalog #113) — excluded."""

    src = tmp_path / "experiments" / "results" / "lane_x"
    src.mkdir(parents=True)
    (src / "build.py").write_text(
        "from pathlib import Path\n"
        "p = Path('/tmp/contest_auth_eval_cuda.json')\n"
    )
    violations = check_no_misleading_device_named_output_directories(
        repo_root=tmp_path
    )
    assert violations == []


def test_check_249_excludes_intake_clones(tmp_path: Path) -> None:
    """Vendored `_intake_` clones (Catalog #109) excluded."""

    src = tmp_path / "experiments" / "results" / "public_pr95_intake_codex"
    src.mkdir(parents=True)
    (src / "trainer.py").write_text(
        "p = '/tmp/contest_auth_eval_cuda.json'\n"
    )
    violations = check_no_misleading_device_named_output_directories(
        repo_root=tmp_path
    )
    assert violations == []


def test_check_249_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    """STRICT mode raises PreflightError when violations present."""

    src = tmp_path / "experiments"
    src.mkdir()
    (src / "train_substrate_demo.py").write_text(
        "p = '/tmp/contest_auth_eval_cuda.json'\n"
    )
    with pytest.raises(PreflightError) as exc_info:
        check_no_misleading_device_named_output_directories(
            repo_root=tmp_path, strict=True
        )
    assert "Catalog #249" in str(exc_info.value)
    assert "phantom-score" in str(exc_info.value)


def test_check_249_strict_mode_silent_on_clean_repo(tmp_path: Path) -> None:
    """STRICT mode does not raise when there are no violations."""

    check_no_misleading_device_named_output_directories(
        repo_root=tmp_path, strict=True
    )


def test_check_249_aggregates_multi_file_violations(tmp_path: Path) -> None:
    """Multiple files with violations are aggregated."""

    src = tmp_path / "experiments"
    src.mkdir()
    for i in range(3):
        (src / f"train_substrate_t{i}.py").write_text(
            f"# trainer {i}\n"
            "p = '/tmp/contest_auth_eval_cuda.json'\n"
        )
    violations = check_no_misleading_device_named_output_directories(
        repo_root=tmp_path
    )
    assert len(violations) == 3


def test_check_249_recognizes_cpu_token_in_lie(tmp_path: Path) -> None:
    """`*_cpu.json` is also a misleading filename pattern (the reverse lie)."""

    src = tmp_path / "experiments"
    src.mkdir()
    (src / "train_substrate_demo.py").write_text(
        "p = '/tmp/contest_auth_eval_cpu.json'\n"
    )
    violations = check_no_misleading_device_named_output_directories(
        repo_root=tmp_path
    )
    assert len(violations) == 1
    assert "_cpu" in violations[0]


def test_check_249_recognizes_work_dir_suffix(tmp_path: Path) -> None:
    """`*_cuda_work` directory-name pattern is also flagged."""

    src = tmp_path / "experiments"
    src.mkdir()
    (src / "train_substrate_demo.py").write_text(
        "p = '/tmp/contest_auth_eval_cuda_work'\n"
    )
    violations = check_no_misleading_device_named_output_directories(
        repo_root=tmp_path
    )
    assert len(violations) == 1


def test_check_249_self_exempts_canonical_helper(tmp_path: Path) -> None:
    """Canonical helper file is self-exempt (carries the redirect logic)."""

    helper = tmp_path / "src" / "tac" / "substrates" / "_shared"
    helper.mkdir(parents=True)
    (helper / "smoke_auth_eval_gate.py").write_text(
        "p = '/tmp/contest_auth_eval_cuda.json'\n"
    )
    violations = check_no_misleading_device_named_output_directories(
        repo_root=tmp_path
    )
    assert violations == []


def test_check_249_live_repo_baseline_is_warn_only_bounded() -> None:
    """Live repo violation count is bounded (warn-only at landing).

    Initial landing has ~32 violations across substrate trainers + deploy
    helpers. Strict-flip planned alongside sister wave that drives count
    to 0. This test pins the upper bound so a regression past landing
    baseline surfaces immediately.
    """

    violations = check_no_misleading_device_named_output_directories()
    assert len(violations) <= 60, (
        f"Live count {len(violations)} exceeds warn-only baseline of 60. "
        "If this is a real new violation, fix it. If this is a planned "
        "increase, update this bound + the CLAUDE.md catalog #249 row."
    )


def test_check_249_orchestrator_callsite_is_warn_only_at_landing() -> None:
    """preflight_all() wires Catalog #249 with strict=False at landing."""

    src = Path(__file__).resolve().parents[3] / "src" / "tac" / "preflight.py"
    text = src.read_text(encoding="utf-8")
    # The wire-in carries strict=False per landing plan.
    assert (
        "check_no_misleading_device_named_output_directories(\n"
        "            strict=False, verbose=verbose,\n"
        "        )"
    ) in text
