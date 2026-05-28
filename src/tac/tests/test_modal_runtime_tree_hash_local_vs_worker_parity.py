# SPDX-License-Identifier: MIT
"""Tests for LOCAL projector vs Modal worker runtime_tree_sha256 parity.

Per Wave N+9 Slot 4 PR111-candidate paired-CUDA RATIFICATION DEFER
(commit ``6bc74e074``; landing memo
``.omx/research/pr111_composite_paired_cuda_ratification_infrastructure_deferred_landed_20260528.md``):
4 consecutive dispatches at ``2026-05-28T17:52`` and ``17:55`` failed rc=1 at
pre-validation with ``inflate runtime tree hash mismatch``. Root cause was
``_module_exists`` and ``_module_paths`` in ``experiments/contest_auth_eval.py``
using ``Path.exists()`` which case-folds on macOS HFS+/APFS / Windows NTFS;
the LOCAL projector picked up a phantom ``tac.dykstra_pareto_solver.Polytope``
(capital P) module via ``polytope.py`` case-fold match, but Linux Modal worker
(case-sensitive) did not.

Canonical fix lands ``_path_exists_case_sensitive`` and routes both helpers
through it. These tests pin the invariant.

Sister of:

- Catalog #146 (contest-compliant inflate runtime template)
- Catalog #205 (canonical select_inflate_device)
- Catalog #229 (Premise Verification before edit)
- Catalog #270 (canonical dispatch optimization protocol)
- Catalog #307 (paradigm-vs-implementation falsification)

Apples-to-apples evidence discipline per CLAUDE.md non-negotiable: the LOCAL
projector MUST produce the same ``runtime_tree_sha256`` that the Modal worker
computes on the extracted submission_dir.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.contest_auth_eval import (  # noqa: E402
    _module_exists,
    _module_paths,
    _path_exists_case_sensitive,
    _runtime_dependency_manifest,
)
from tac.deploy.modal.auth_eval import (  # noqa: E402
    modal_uploaded_submission_dir_runtime_manifest,
)
from tac.deploy.modal.paired_dispatch import (  # noqa: E402
    MODAL_AUTH_EVAL_CPU_REMOTE_SUBMISSION_DIR,
    MODAL_AUTH_EVAL_CUDA_REMOTE_SUBMISSION_DIR,
)

# =============================================================================
# _path_exists_case_sensitive unit tests
# =============================================================================


def test_path_exists_case_sensitive_matches_existing_lowercase(tmp_path: Path) -> None:
    (tmp_path / "alpha.py").write_text("# alpha\n")
    assert _path_exists_case_sensitive(tmp_path / "alpha.py") is True


def test_path_exists_case_sensitive_rejects_case_fold_basename(tmp_path: Path) -> None:
    # On case-insensitive filesystems Path.exists() would return True for
    # "ALPHA.py" when "alpha.py" exists; our helper must return False.
    (tmp_path / "alpha.py").write_text("# alpha\n")
    assert _path_exists_case_sensitive(tmp_path / "ALPHA.py") is False
    assert _path_exists_case_sensitive(tmp_path / "Alpha.py") is False


def test_path_exists_case_sensitive_rejects_case_fold_directory(tmp_path: Path) -> None:
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "child.py").write_text("# child\n")
    assert _path_exists_case_sensitive(tmp_path / "subdir" / "child.py") is True
    assert _path_exists_case_sensitive(tmp_path / "SUBDIR" / "child.py") is False
    assert _path_exists_case_sensitive(tmp_path / "subdir" / "CHILD.py") is False


def test_path_exists_case_sensitive_handles_missing_path(tmp_path: Path) -> None:
    assert _path_exists_case_sensitive(tmp_path / "does_not_exist.py") is False


def test_path_exists_case_sensitive_handles_empty_path() -> None:
    assert _path_exists_case_sensitive(Path("")) is False


def test_path_exists_case_sensitive_handles_non_path_input() -> None:
    assert _path_exists_case_sensitive(None) is False  # type: ignore[arg-type]


# =============================================================================
# _module_exists tests (case-sensitive on macOS / Linux / Windows)
# =============================================================================


@pytest.fixture
def fake_tac_repo(tmp_path: Path) -> Path:
    """Build a minimal ``src/tac`` tree with a known case-sensitive collision."""

    tac = tmp_path / "src" / "tac"
    tac.mkdir(parents=True)
    (tac / "__init__.py").write_text("")
    sub = tac / "dykstra_pareto_solver"
    sub.mkdir()
    (sub / "__init__.py").write_text("")
    # Canonical lowercase module
    (sub / "polytope.py").write_text("# polytope\n")
    return tmp_path


def test_module_exists_accepts_lowercase_canonical(fake_tac_repo: Path) -> None:
    assert _module_exists("tac.dykstra_pareto_solver.polytope", fake_tac_repo) is True


def test_module_exists_rejects_capitalized_phantom(fake_tac_repo: Path) -> None:
    # PR111 anchor: macOS case-fold would falsely return True for "Polytope".
    # The canonical helper MUST return False so LOCAL projector matches Linux
    # Modal worker.
    assert _module_exists("tac.dykstra_pareto_solver.Polytope", fake_tac_repo) is False


def test_module_exists_accepts_package_init(fake_tac_repo: Path) -> None:
    assert _module_exists("tac.dykstra_pareto_solver", fake_tac_repo) is True


def test_module_exists_rejects_capitalized_package(fake_tac_repo: Path) -> None:
    assert _module_exists("tac.DYKSTRA_PARETO_SOLVER", fake_tac_repo) is False


def test_module_exists_handles_non_tac_module(fake_tac_repo: Path) -> None:
    assert _module_exists("numpy", fake_tac_repo) is False
    assert _module_exists("torch.nn", fake_tac_repo) is False


# =============================================================================
# _module_paths tests
# =============================================================================


def test_module_paths_returns_only_case_sensitive_match(fake_tac_repo: Path) -> None:
    paths = _module_paths("tac.dykstra_pareto_solver.polytope", fake_tac_repo)
    names = {p.name for p in paths}
    assert "polytope.py" in names
    # No phantom Polytope.py via case-fold:
    assert "Polytope.py" not in names


def test_module_paths_rejects_capitalized_module(fake_tac_repo: Path) -> None:
    paths = _module_paths("tac.dykstra_pareto_solver.Polytope", fake_tac_repo)
    # The capital-P module returns only __init__.py walks UP TO the parent
    # package (which IS case-correct); the leaf .py / pkg dir does NOT match.
    leaf_paths = [p for p in paths if p.name == "Polytope.py"]
    assert leaf_paths == []
    # The parent __init__.py walks should still resolve (case-correct ancestors):
    init_paths = {str(p.relative_to(fake_tac_repo)) for p in paths if p.name == "__init__.py"}
    assert "src/tac/__init__.py" in init_paths
    assert "src/tac/dykstra_pareto_solver/__init__.py" in init_paths


# =============================================================================
# PR111 EMPIRICAL REGRESSION GUARD — apples-to-apples projector vs worker
# =============================================================================


PR111_SUBMISSION_DIR = REPO_ROOT / (
    "experiments/results/"
    "composite_nscs06_v8_plus_compound_c_pr111_candidate_20260528/submission"
)
PR111_CUDA_WORKER_HASH = (
    "1e9bf123e8eac353591c2fa57af96d3eb330855d34d375faa886dc9c32026afb"
)
PR111_CPU_WORKER_HASH = (
    "60256159c7d65405fca5139d8ffd4a81a8444c13ed0dd804fd0aecb6097b55e6"
)


@pytest.mark.skipif(
    not PR111_SUBMISSION_DIR.is_dir(),
    reason="PR111 composite submission archive required for empirical regression guard",
)
def test_pr111_local_projector_matches_modal_worker_cuda() -> None:
    """LOCAL CUDA projector MUST produce the exact Modal worker hash.

    PR111 paired-CUDA RATIFICATION 4× DEFER 2026-05-28 anchor.
    """

    inflate_sh = (PR111_SUBMISSION_DIR / "inflate.sh").resolve()
    local_manifest = _runtime_dependency_manifest(inflate_sh, Path("upstream"))
    projected = modal_uploaded_submission_dir_runtime_manifest(
        local_manifest, remote_submission_dir=MODAL_AUTH_EVAL_CUDA_REMOTE_SUBMISSION_DIR
    )
    assert projected["runtime_tree_sha256"] == PR111_CUDA_WORKER_HASH, (
        "PR111 paired-CUDA RATIFICATION anchor regression: "
        f"projected {projected['runtime_tree_sha256']!r} "
        f"does NOT match Modal worker actual {PR111_CUDA_WORKER_HASH!r}. "
        "This re-introduces the bug class that DEFERRED PR111 RATIFICATION 4× "
        "on 2026-05-28 with cumulative $0.06 paid Modal spend producing zero "
        "score evidence. Restore _path_exists_case_sensitive routing in "
        "experiments/contest_auth_eval.py::_module_exists and ::_module_paths."
    )


@pytest.mark.skipif(
    not PR111_SUBMISSION_DIR.is_dir(),
    reason="PR111 composite submission archive required for empirical regression guard",
)
def test_pr111_local_projector_matches_modal_worker_cpu() -> None:
    """LOCAL CPU projector MUST produce the exact Modal worker hash."""

    inflate_sh = (PR111_SUBMISSION_DIR / "inflate.sh").resolve()
    local_manifest = _runtime_dependency_manifest(inflate_sh, Path("upstream"))
    projected = modal_uploaded_submission_dir_runtime_manifest(
        local_manifest, remote_submission_dir=MODAL_AUTH_EVAL_CPU_REMOTE_SUBMISSION_DIR
    )
    assert projected["runtime_tree_sha256"] == PR111_CPU_WORKER_HASH, (
        "PR111 paired-CPU RATIFICATION anchor regression: "
        f"projected {projected['runtime_tree_sha256']!r} "
        f"does NOT match Modal worker actual {PR111_CPU_WORKER_HASH!r}."
    )


@pytest.mark.skipif(
    not PR111_SUBMISSION_DIR.is_dir(),
    reason="PR111 composite submission archive required for empirical regression guard",
)
def test_pr111_repo_local_tac_no_phantom_polytope_module() -> None:
    """The PR111 LOCAL repo_local_tac_import_manifest must NOT carry phantom Polytope."""

    inflate_sh = (PR111_SUBMISSION_DIR / "inflate.sh").resolve()
    local_manifest = _runtime_dependency_manifest(inflate_sh, Path("upstream"))
    rl = local_manifest["repo_local_tac_import_manifest"]
    assert "tac.dykstra_pareto_solver.Polytope" not in rl["root_import_modules"], (
        "Phantom Polytope module (capital P, case-fold artifact on macOS) "
        "re-introduced into repo_local_tac_import_manifest. "
        "This is the empirical bug class that DEFERRED PR111 paired-CUDA "
        "RATIFICATION 2026-05-28."
    )
    # Module count should be 36 (was 37 before fix).
    assert rl["module_count"] == 36, (
        f"Expected module_count=36 post-fix; got {rl['module_count']}. "
        "Phantom Polytope (or new sister case-fold collision) may have "
        "re-emerged."
    )
    assert rl["file_count"] == 39, (
        f"Expected file_count=39 post-fix; got {rl['file_count']}. "
        "Phantom Polytope (or new sister case-fold collision) may have "
        "re-emerged."
    )
