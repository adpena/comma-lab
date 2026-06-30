# SPDX-License-Identifier: MIT
"""Tests for the residual-INR hybrid pipeline self-protection gates (Catalog #393/#394/#395).

Two-landing pattern (CLAUDE.md "Bugs must be permanently fixed AND self-protected against"): the fix
(B1/B2/B3) PLUS a STRICT gate per bug class so the class cannot re-emerge at a different surface.

  * #393 check_orchestrator_emits_valid_trainer_contract — B1 + META-bug (broken seam).
  * #394 check_residual_override_has_coverage_proof — B2 (geometry-ceiling override mask).
  * #395 check_axis_solved_claim_has_pipeline_validation — B3 / NO-FAKE #8 (borrowed 'solved' claim).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_axis_solved_claim_has_pipeline_validation,
    check_orchestrator_emits_valid_trainer_contract,
    check_residual_override_has_coverage_proof,
)

REAL_REPO = Path(__file__).resolve().parents[3]


# ===========================================================================
# real-repo: all three are clean (the fix landed; live count must be 0).
# ===========================================================================
def test_all_three_gates_clean_on_real_repo():
    assert check_orchestrator_emits_valid_trainer_contract(strict=False) == []
    assert check_residual_override_has_coverage_proof(strict=False) == []
    assert check_axis_solved_claim_has_pipeline_validation(strict=False) == []


def test_all_three_gates_pass_strict_on_real_repo():
    # must not raise (strict-flip safety in the same batch)
    check_orchestrator_emits_valid_trainer_contract(strict=True)
    check_residual_override_has_coverage_proof(strict=True)
    check_axis_solved_claim_has_pipeline_validation(strict=True)


# ===========================================================================
# #393 check_orchestrator_emits_valid_trainer_contract
# ===========================================================================
_GOOD_ORCH = (
    "from tac.v2_compose.launch_command import build_residual_only_command\n"
    "def residual_blob_from_weights_npz(p):\n"
    "    return ag.build_residual_blob(params, cfg)\n"
    "def phase_a():\n"
    "    build_residual_only_command(out_dir='x')\n"
)
_GOOD_E2E_TEST = (
    "import subprocess\n"
    "from tac.v2_compose.archive_grammar import residual_inflate_reference\n"
    "def test_e2e():\n"
    "    subprocess.run(['inflate.py'])\n"
    "    assert array_equal(raw, residual_inflate_reference(...))\n"
)


def _fake_repo_393(tmp_path: Path, *, orch: str, e2e: str | None) -> Path:
    (tmp_path / "tools").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tools" / "compose_witness_archive.py").write_text(orch)
    tdir = tmp_path / "src" / "tac" / "tests"
    tdir.mkdir(parents=True, exist_ok=True)
    if e2e is not None:
        (tdir / "test_compose_witness_archive_pipeline.py").write_text(e2e)
    return tmp_path


def test_393_clean_orchestrator(tmp_path):
    root = _fake_repo_393(tmp_path, orch=_GOOD_ORCH, e2e=_GOOD_E2E_TEST)
    assert check_orchestrator_emits_valid_trainer_contract(repo_root=root, strict=False) == []


def test_393_missing_e2e_test_flags(tmp_path):
    root = _fake_repo_393(tmp_path, orch=_GOOD_ORCH, e2e=None)
    v = check_orchestrator_emits_valid_trainer_contract(repo_root=root, strict=False)
    assert any("no end-to-end handoff-contract test" in x for x in v)


def test_393_weak_e2e_test_flags(tmp_path):
    weak = "def test_x():\n    assert 1\n"  # no subprocess/inflate/array_equal/oracle
    root = _fake_repo_393(tmp_path, orch=_GOOD_ORCH, e2e=weak)
    v = check_orchestrator_emits_valid_trainer_contract(repo_root=root, strict=False)
    assert any("must run inflate.py as a subprocess" in x for x in v)


def test_393_phase_b_stub_flags(tmp_path):
    stub_orch = (
        "from tac.v2_compose.launch_command import build_residual_only_command\n"
        "def phase_b():\n"
        "    raise SystemExit('NEEDS-WIRING')\n"
        "build_residual_only_command(out_dir='x')\n"
    )
    root = _fake_repo_393(tmp_path, orch=stub_orch, e2e=_GOOD_E2E_TEST)
    v = check_orchestrator_emits_valid_trainer_contract(repo_root=root, strict=False)
    assert any("must ASSEMBLE the residual archive" in x for x in v)


def test_393_superseded_emitter_flags(tmp_path):
    bad_orch = (
        "from tac.v2_compose.launch_command import build_residual_inr_command\n"
        "def residual_blob_from_weights_npz(p): return build_residual_blob(p)\n"
        "build_residual_inr_command(out_dir='x')\n"
    )
    root = _fake_repo_393(tmp_path, orch=bad_orch, e2e=_GOOD_E2E_TEST)
    v = check_orchestrator_emits_valid_trainer_contract(repo_root=root, strict=False)
    assert any("SUPERSEDED build_residual_inr_command" in x for x in v)


def test_393_waiver_respected(tmp_path):
    bad_orch = _GOOD_ORCH.replace("build_residual_blob", "nope") + "# ORCHESTRATOR_CONTRACT_OK: smoke\n"
    root = _fake_repo_393(tmp_path, orch=bad_orch, e2e=None)
    assert check_orchestrator_emits_valid_trainer_contract(repo_root=root, strict=False) == []


def test_393_strict_raises(tmp_path):
    root = _fake_repo_393(tmp_path, orch=_GOOD_ORCH, e2e=None)
    with pytest.raises(PreflightError, match="Catalog #393"):
        check_orchestrator_emits_valid_trainer_contract(repo_root=root, strict=True)


def test_393_no_orchestrator_is_clean(tmp_path):
    assert check_orchestrator_emits_valid_trainer_contract(repo_root=tmp_path, strict=False) == []


# ===========================================================================
# #394 check_residual_override_has_coverage_proof
# ===========================================================================
def _fake_tool_394(tmp_path: Path, body: str, name: str = "mytool.py") -> Path:
    (tmp_path / "tools").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tools" / name).write_text(body)
    return tmp_path


def test_394_override_with_coverage_and_gate_clean(tmp_path):
    body = (
        "bundle = build_residual_training_bundle(rgb, labels)\n"
        "rep = measure_composition_coverage(residual_mask, bundle.composition_mask)\n"
        "if not rep.passes_gate:\n    raise SystemExit('NO-GO')\n"
    )
    root = _fake_tool_394(tmp_path, body)
    assert check_residual_override_has_coverage_proof(repo_root=root, strict=False) == []


def test_394_override_without_coverage_flags(tmp_path):
    body = "bundle = build_residual_training_bundle(rgb, labels)\n"
    root = _fake_tool_394(tmp_path, body)
    v = check_residual_override_has_coverage_proof(repo_root=root, strict=False)
    assert any("NO measure_composition_coverage proof" in x for x in v)


def test_394_coverage_without_gate_flags(tmp_path):
    body = (
        "mask = derive_composition_mask(lbl)\n"
        "rep = measure_composition_coverage(res, mask)\n"
        "print(rep.coverage)\n"  # measured but not gated
    )
    root = _fake_tool_394(tmp_path, body)
    v = check_residual_override_has_coverage_proof(repo_root=root, strict=False)
    assert any("does NOT GATE on it" in x for x in v)


def test_394_waiver_respected(tmp_path):
    body = "bundle = build_residual_training_bundle(rgb, labels)  # RESIDUAL_OVERRIDE_COVERAGE_OK: smoke\n"
    root = _fake_tool_394(tmp_path, body)
    assert check_residual_override_has_coverage_proof(repo_root=root, strict=False) == []


def test_394_non_override_tool_clean(tmp_path):
    root = _fake_tool_394(tmp_path, "x = 1\nprint('hello')\n")
    assert check_residual_override_has_coverage_proof(repo_root=root, strict=False) == []


def test_394_test_files_excluded(tmp_path):
    (tmp_path / "tools").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tools" / "test_mytool.py").write_text("build_residual_training_bundle(a, b)\n")
    assert check_residual_override_has_coverage_proof(repo_root=root if (root := tmp_path) else tmp_path, strict=False) == []


def test_394_strict_raises(tmp_path):
    root = _fake_tool_394(tmp_path, "build_residual_training_bundle(rgb, labels)\n")
    with pytest.raises(PreflightError, match="Catalog #394"):
        check_residual_override_has_coverage_proof(repo_root=root, strict=True)


def test_394_default_mode_is_boundary_annulus():
    # the real module's default mode must be boundary_annulus (this is what the gate's part-1 asserts)
    import inspect

    from tac.v2_compose.residual_compose import MASK_MODE_BOUNDARY_ANNULUS, derive_composition_mask

    sig = inspect.signature(derive_composition_mask)
    assert sig.parameters["mode"].default == MASK_MODE_BOUNDARY_ANNULUS


# ===========================================================================
# #395 check_axis_solved_claim_has_pipeline_validation
# ===========================================================================
def _fake_v2_395(tmp_path: Path, body: str, name: str = "foo.py") -> Path:
    d = tmp_path / "src" / "tac" / "v2_compose"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(body)
    return tmp_path


def test_395_bare_pose_solved_flags(tmp_path):
    root = _fake_v2_395(tmp_path, "# Pose is solved with the sidecar.\n")
    v = check_axis_solved_claim_has_pipeline_validation(repo_root=root, strict=False)
    assert any("claims a scorer axis is 'solved'" in x for x in v)


def test_395_solved_with_open_qualifier_clean(tmp_path):
    root = _fake_v2_395(tmp_path, "# pose was thought solved but is OPEN until measured.\n")
    assert check_axis_solved_claim_has_pipeline_validation(repo_root=root, strict=False) == []


def test_395_solved_with_advisory_qualifier_clean(tmp_path):
    root = _fake_v2_395(tmp_path, "# d_pose solved only on the advisory CPU path; not authority.\n")
    assert check_axis_solved_claim_has_pipeline_validation(repo_root=root, strict=False) == []


def test_395_reversed_order_solved_pose_flags(tmp_path):
    root = _fake_v2_395(tmp_path, "note = 'solved the d_pose with stored scalars'\n")
    v = check_axis_solved_claim_has_pipeline_validation(repo_root=root, strict=False)
    assert any("'solved'" in x for x in v)


def test_395_rate_solved_flags_and_qualifier_clears(tmp_path):
    bad = _fake_v2_395(tmp_path, "msg = 'rate is solved'\n", name="a.py")
    assert any("solved" in x for x in check_axis_solved_claim_has_pipeline_validation(repo_root=bad, strict=False))
    good = _fake_v2_395(tmp_path, "msg = 'rate is solved [advisory budget]'\n", name="b.py")
    # both files exist now; b.py has a qualifier but a.py still flags -> filter by file
    vs = check_axis_solved_claim_has_pipeline_validation(repo_root=good, strict=False)
    assert any("a.py" in x for x in vs) and not any("b.py" in x for x in vs)


def test_395_waiver_respected(tmp_path):
    root = _fake_v2_395(tmp_path, "# pose solved  # AXIS_SOLVED_CLAIM_OK: validated elsewhere\n")
    assert check_axis_solved_claim_has_pipeline_validation(repo_root=root, strict=False) == []


def test_395_benign_line_clean(tmp_path):
    root = _fake_v2_395(tmp_path, "# the pose sidecar stores 6 scalars per pair.\n")
    assert check_axis_solved_claim_has_pipeline_validation(repo_root=root, strict=False) == []


def test_395_test_files_excluded(tmp_path):
    d = tmp_path / "src" / "tac" / "v2_compose"
    d.mkdir(parents=True, exist_ok=True)
    (d / "test_foo.py").write_text("# pose is solved\n")
    assert check_axis_solved_claim_has_pipeline_validation(repo_root=tmp_path, strict=False) == []


def test_395_strict_raises(tmp_path):
    root = _fake_v2_395(tmp_path, "# d_seg solved.\n")
    with pytest.raises(PreflightError, match="Catalog #395"):
        check_axis_solved_claim_has_pipeline_validation(repo_root=root, strict=True)


def test_395_compose_tool_surface_scanned(tmp_path):
    (tmp_path / "tools").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tools" / "compose_witness_archive.py").write_text("# pose solved\n")
    v = check_axis_solved_claim_has_pipeline_validation(repo_root=tmp_path, strict=False)
    assert any("compose_witness_archive.py" in x for x in v)
