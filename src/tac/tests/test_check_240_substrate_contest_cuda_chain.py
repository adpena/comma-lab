"""Tests for Catalog #240 - check_substrate_contest_cuda_chain_complete_or_research_only_tagged.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against": this
gate refuses substrate recipes that implicitly claim contest-CUDA dispatch
capability while the matching trainer's ``_full_main`` raises NotImplementedError.

Anchors:
- Z3 v2 smoke ``fc-01KRNHWD6AEMWGDJXBEY8GM0P3`` (2026-05-15): substrate looks
  "built" but recipe is ``smoke_only: true`` so v2 path never activates.
- Z4 + Z5 (pre-fix 2026-05-15): contest recipes existed without research_only
  tag, but trainers' ``_full_main`` raised NotImplementedError.

Sister of Catalog #220 (`check_substrate_l1_scaffold_no_byte_addition_without_
operational_score_improvement_mechanism`).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    _check_240_recipe_id_from_path,
    _check_240_recipe_is_research_only,
    _check_240_recipe_has_waiver,
    _check_240_trainer_full_main_raises_notimplementederror,
    check_substrate_contest_cuda_chain_complete_or_research_only_tagged,
)


# ---------- helper unit tests ----------


def test_recipe_id_from_path_modal_t4():
    p = Path(".omx/operator_authorize_recipes/substrate_z4_cooperative_receiver_loss_modal_t4_dispatch.yaml")
    assert _check_240_recipe_id_from_path(p) == "z4_cooperative_receiver_loss"


def test_recipe_id_from_path_modal_a100():
    p = Path("substrate_balle_renderer_modal_a100_dispatch.yaml")
    assert _check_240_recipe_id_from_path(p) == "balle_renderer"


def test_recipe_id_from_path_smoke_dispatch():
    p = Path("substrate_c1_world_model_foveation_modal_t4_smoke_dispatch.yaml")
    assert _check_240_recipe_id_from_path(p) == "c1_world_model_foveation"


def test_recipe_id_from_path_local_apple_silicon():
    p = Path("substrate_mlx_mask_renderer_local_apple_silicon_dispatch.yaml")
    assert _check_240_recipe_id_from_path(p) == "mlx_mask_renderer"


def test_recipe_research_only_via_smoke_only():
    text = "name: foo\nsmoke_only: true\n"
    assert _check_240_recipe_is_research_only(text) is True


def test_recipe_research_only_via_research_only():
    text = "name: foo\nresearch_only: true\n"
    assert _check_240_recipe_is_research_only(text) is True


def test_recipe_research_only_via_dispatch_enabled_false():
    text = "name: foo\ndispatch_enabled: false\n"
    assert _check_240_recipe_is_research_only(text) is True


def test_recipe_research_only_negative_when_no_flag():
    text = "name: foo\nplatform: modal\n"
    assert _check_240_recipe_is_research_only(text) is False


def test_recipe_research_only_negative_when_smoke_only_false():
    text = "name: foo\nsmoke_only: false\n"
    assert _check_240_recipe_is_research_only(text) is False


def test_recipe_waiver_with_rationale():
    text = "name: foo  # CONTEST_CUDA_CHAIN_PARTIAL_OK:operator approved partial state for special case\n"
    waived, rationale = _check_240_recipe_has_waiver(text)
    assert waived is True
    assert "operator approved" in rationale


def test_recipe_waiver_placeholder_rejected():
    text = "name: foo  # CONTEST_CUDA_CHAIN_PARTIAL_OK:<reason>\n"
    waived, rationale = _check_240_recipe_has_waiver(text)
    assert waived is False
    assert rationale == ""


def test_recipe_waiver_placeholder_rationale_rejected():
    text = "name: foo  # CONTEST_CUDA_CHAIN_PARTIAL_OK:<rationale>\n"
    waived, rationale = _check_240_recipe_has_waiver(text)
    assert waived is False


def test_recipe_waiver_absent():
    text = "name: foo\n"
    waived, rationale = _check_240_recipe_has_waiver(text)
    assert waived is False


def test_trainer_full_main_raises_not_implemented(tmp_path):
    trainer = tmp_path / "train_substrate_x.py"
    trainer.write_text(textwrap.dedent("""\
        def _smoke_main(args):
            return 0

        def _full_main(args):
            '''Phase 2 council approval required.'''
            raise NotImplementedError("Phase 2 council approval required.")
    """))
    raises, msg = _check_240_trainer_full_main_raises_notimplementederror(trainer)
    assert raises is True
    assert "NotImplementedError" in msg


def test_trainer_full_main_implemented(tmp_path):
    trainer = tmp_path / "train_substrate_x.py"
    trainer.write_text(textwrap.dedent("""\
        def _full_main(args):
            do_real_training()
            return 0
    """))
    raises, msg = _check_240_trainer_full_main_raises_notimplementederror(trainer)
    assert raises is False


def test_trainer_full_main_missing(tmp_path):
    trainer = tmp_path / "train_substrate_x.py"
    trainer.write_text("def _smoke_main(args): return 0\n")
    raises, msg = _check_240_trainer_full_main_raises_notimplementederror(trainer)
    assert raises is False
    assert "no _full_main" in msg


def test_trainer_missing_file(tmp_path):
    raises, msg = _check_240_trainer_full_main_raises_notimplementederror(
        tmp_path / "nonexistent.py"
    )
    assert raises is False
    assert "not found" in msg


# ---------- end-to-end gate tests ----------


def _make_repo(tmp_path: Path) -> Path:
    """Build a synthetic repo skeleton for the gate to scan."""
    (tmp_path / ".omx" / "operator_authorize_recipes").mkdir(parents=True)
    (tmp_path / "experiments").mkdir(parents=True)
    return tmp_path


def test_gate_passes_no_recipes_dir(tmp_path):
    out = check_substrate_contest_cuda_chain_complete_or_research_only_tagged(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert out == []


def test_gate_passes_complete_substrate(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / ".omx/operator_authorize_recipes/substrate_x_modal_t4_dispatch.yaml").write_text(
        "name: substrate_x_modal_t4_dispatch\nplatform: modal\n"
    )
    (repo / "experiments/train_substrate_x.py").write_text(
        "def _full_main(args):\n    do_training()\n    return 0\n"
    )
    out = check_substrate_contest_cuda_chain_complete_or_research_only_tagged(
        repo_root=repo, strict=False, verbose=False
    )
    assert out == []


def test_gate_passes_research_only_substrate(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / ".omx/operator_authorize_recipes/substrate_x_modal_t4_dispatch.yaml").write_text(
        "name: substrate_x_modal_t4_dispatch\nresearch_only: true\nplatform: modal\n"
    )
    (repo / "experiments/train_substrate_x.py").write_text(
        "def _full_main(args):\n    raise NotImplementedError('council pending')\n"
    )
    out = check_substrate_contest_cuda_chain_complete_or_research_only_tagged(
        repo_root=repo, strict=False, verbose=False
    )
    assert out == []


def test_gate_passes_smoke_only_substrate(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / ".omx/operator_authorize_recipes/substrate_x_modal_t4_dispatch.yaml").write_text(
        "name: substrate_x_modal_t4_dispatch\nsmoke_only: true\nplatform: modal\n"
    )
    (repo / "experiments/train_substrate_x.py").write_text(
        "def _full_main(args):\n    raise NotImplementedError('council pending')\n"
    )
    out = check_substrate_contest_cuda_chain_complete_or_research_only_tagged(
        repo_root=repo, strict=False, verbose=False
    )
    assert out == []


def test_gate_passes_dispatch_disabled_substrate(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / ".omx/operator_authorize_recipes/substrate_x_modal_t4_dispatch.yaml").write_text(
        "name: substrate_x_modal_t4_dispatch\ndispatch_enabled: false\nplatform: modal\n"
    )
    (repo / "experiments/train_substrate_x.py").write_text(
        "def _full_main(args):\n    raise NotImplementedError('council pending')\n"
    )
    out = check_substrate_contest_cuda_chain_complete_or_research_only_tagged(
        repo_root=repo, strict=False, verbose=False
    )
    assert out == []


def test_gate_passes_orphan_recipe_no_trainer(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / ".omx/operator_authorize_recipes/substrate_orphan_modal_a100_dispatch.yaml").write_text(
        "name: substrate_orphan_modal_a100_dispatch\nplatform: modal\n"
    )
    out = check_substrate_contest_cuda_chain_complete_or_research_only_tagged(
        repo_root=repo, strict=False, verbose=False
    )
    assert out == []


def test_gate_flags_z3_v2_bug_class(tmp_path):
    """The Z4 / Z5 anchor: contest recipe + NotImplementedError trainer = bug."""
    repo = _make_repo(tmp_path)
    (repo / ".omx/operator_authorize_recipes/substrate_z4_cooperative_modal_t4_dispatch.yaml").write_text(
        "name: substrate_z4_cooperative_modal_t4_dispatch\nplatform: modal\n"
    )
    (repo / "experiments/train_substrate_z4_cooperative.py").write_text(
        textwrap.dedent("""\
        def _full_main(args):
            '''Phase 2 council approval required.'''
            raise NotImplementedError("Phase 2 council approval required.")
        """)
    )
    out = check_substrate_contest_cuda_chain_complete_or_research_only_tagged(
        repo_root=repo, strict=False, verbose=False
    )
    assert len(out) == 1
    assert "z4_cooperative" in out[0]
    assert "NotImplementedError" in out[0]


def test_gate_strict_raises_on_violation(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / ".omx/operator_authorize_recipes/substrate_x_modal_t4_dispatch.yaml").write_text(
        "name: substrate_x_modal_t4_dispatch\nplatform: modal\n"
    )
    (repo / "experiments/train_substrate_x.py").write_text(
        "def _full_main(args):\n    raise NotImplementedError('council pending')\n"
    )
    with pytest.raises(PreflightError) as excinfo:
        check_substrate_contest_cuda_chain_complete_or_research_only_tagged(
            repo_root=repo, strict=True, verbose=False
        )
    assert "Catalog #240" in str(excinfo.value)
    assert "Z3 v2 / Z4 / Z5" in str(excinfo.value)


def test_gate_strict_silent_on_clean(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / ".omx/operator_authorize_recipes/substrate_x_modal_t4_dispatch.yaml").write_text(
        "name: substrate_x_modal_t4_dispatch\nresearch_only: true\nplatform: modal\n"
    )
    out = check_substrate_contest_cuda_chain_complete_or_research_only_tagged(
        repo_root=repo, strict=True, verbose=False
    )
    assert out == []


def test_gate_aggregates_multiple_violations(tmp_path):
    repo = _make_repo(tmp_path)
    for i in range(3):
        (repo / f".omx/operator_authorize_recipes/substrate_x{i}_modal_t4_dispatch.yaml").write_text(
            f"name: substrate_x{i}_modal_t4_dispatch\nplatform: modal\n"
        )
        (repo / f"experiments/train_substrate_x{i}.py").write_text(
            "def _full_main(args):\n    raise NotImplementedError('pending')\n"
        )
    out = check_substrate_contest_cuda_chain_complete_or_research_only_tagged(
        repo_root=repo, strict=False, verbose=False
    )
    assert len(out) == 3


def test_gate_waiver_accepted(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / ".omx/operator_authorize_recipes/substrate_x_modal_t4_dispatch.yaml").write_text(
        "name: substrate_x_modal_t4_dispatch  # CONTEST_CUDA_CHAIN_PARTIAL_OK:operator approved special case\nplatform: modal\n"
    )
    (repo / "experiments/train_substrate_x.py").write_text(
        "def _full_main(args):\n    raise NotImplementedError('pending')\n"
    )
    out = check_substrate_contest_cuda_chain_complete_or_research_only_tagged(
        repo_root=repo, strict=False, verbose=False
    )
    assert out == []


def test_gate_waiver_placeholder_rejected(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / ".omx/operator_authorize_recipes/substrate_x_modal_t4_dispatch.yaml").write_text(
        "name: substrate_x_modal_t4_dispatch  # CONTEST_CUDA_CHAIN_PARTIAL_OK:<reason>\nplatform: modal\n"
    )
    (repo / "experiments/train_substrate_x.py").write_text(
        "def _full_main(args):\n    raise NotImplementedError('pending')\n"
    )
    out = check_substrate_contest_cuda_chain_complete_or_research_only_tagged(
        repo_root=repo, strict=False, verbose=False
    )
    assert len(out) == 1


# ---------- live-repo regression guard ----------


def test_live_repo_regression_zero_violations():
    """The live repo MUST be at 0 violations (Z4 + Z5 fix lands in same commit batch)."""
    out = check_substrate_contest_cuda_chain_complete_or_research_only_tagged(
        strict=False, verbose=False
    )
    assert out == [], f"Catalog #240 regressed; violations: {out}"


def test_orchestrator_callsite_strict_true_regression_guard():
    """preflight.py must wire #240 with strict=True per the strict-flip atomicity rule."""
    import inspect
    from tac import preflight
    src = inspect.getsource(preflight.preflight_all)
    assert "check_substrate_contest_cuda_chain_complete_or_research_only_tagged" in src, (
        "preflight_all() must invoke check #240"
    )
    # Check it's wired strict=True
    # Find the line with the check name and ensure strict=True is in the next 5 lines
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if "check_substrate_contest_cuda_chain_complete_or_research_only_tagged(" in line:
            window = "\n".join(lines[i : i + 5])
            assert "strict=True" in window, (
                f"Catalog #240 must be wired strict=True per CLAUDE.md "
                f"'Strict-flip atomicity rule'; found:\n{window}"
            )
            return
    pytest.fail("could not locate #240 callsite in preflight_all()")
