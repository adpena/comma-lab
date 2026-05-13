"""Catalog #167: smoke-before-full pattern tests.

PHASE-B1-PIVOT bug-class anchor (2026-05-12). Two consecutive 2000-epoch
sane_hnerv Modal A100 dispatches crashed rc=1 within 15s and 72s
respectively - burned $0.30 + a harvest slot each. A 100-epoch ~$0.30
smoke would have caught the integration failure for the same cost.

Catalog #167 refuses operator-authorize wrappers that fire a "full" canary
(``cost_band.epochs >= 1000``) without routing through
``tools/run_modal_smoke_before_full.py``.

Coverage targets:

- compliant wrapper (routes through run_modal_smoke_before_full) -> no violation
- non-compliant wrapper (full canary, no smoke routing, no waiver) -> violation
- compliant via SMOKE_BEFORE_FULL_OK same-line waiver
- partial compliance: full canary + waiver elsewhere -> still violation if waiver text not present
- recipe with epochs < 1000 -> not a full canary, no violation
- recipe with no cost_band block -> no violation (out of scope)
- recipe missing entirely (orphan wrapper) -> no violation (out of scope)
- malformed recipe (epochs not parseable) -> no violation (no ambient claim)
- multiple wrappers, mixed compliance -> only non-compliant ones flagged
- non-existent scripts/ dir -> empty violations
- strict mode raises PreflightError when violations present
- non-strict returns the violation list
- verbose mode prints diagnostic banner
- _check_167_extract_recipe_epochs handles inline / quoted / unquoted forms
- _check_167_extract_recipe_epochs returns None for missing block
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    _check_167_extract_recipe_epochs,
    check_substrate_dispatch_uses_smoke_before_full_pattern,
)


def _make_recipe(epochs: int | None) -> str:
    if epochs is None:
        return "schema_version: 1\nname: foo\n"
    return (
        "schema_version: 1\n"
        "name: foo\n"
        "platform: modal\n"
        "cost_band:\n"
        f"  epochs: {epochs}\n"
        "  all_flags_on: true\n"
    )


def _make_compliant_wrapper() -> str:
    return (
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        ".venv/bin/python tools/run_modal_smoke_before_full.py "
        "--recipe substrate_foo_modal_a100_dispatch \"$@\"\n"
    )


def _make_non_compliant_wrapper(recipe_name: str) -> str:
    return (
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        f".venv/bin/python tools/operator_authorize.py "
        f"--recipe {recipe_name} \"$@\"\n"
    )


def _make_waiver_wrapper(recipe_name: str) -> str:
    return (
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        f".venv/bin/python tools/operator_authorize.py "
        f"--recipe {recipe_name} \"$@\"  "
        "# SMOKE_BEFORE_FULL_OK:established-trainer-3-anchors\n"
    )


@pytest.fixture()
def fake_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / ".omx" / "operator_authorize_recipes").mkdir(parents=True)
    return repo


def test_extract_epochs_quoted() -> None:
    text = "cost_band:\n  epochs: \"2000\"\n"
    assert _check_167_extract_recipe_epochs(text) == 2000


def test_extract_epochs_unquoted() -> None:
    text = "cost_band:\n  epochs: 1500\n"
    assert _check_167_extract_recipe_epochs(text) == 1500


def test_extract_epochs_missing_block() -> None:
    text = "schema_version: 1\nname: foo\n"
    assert _check_167_extract_recipe_epochs(text) is None


def test_extract_epochs_other_top_key_closes_block() -> None:
    text = (
        "cost_band:\n"
        "  epochs: 100\n"
        "another_block:\n"
        "  epochs: 9999\n"
    )
    assert _check_167_extract_recipe_epochs(text) == 100


def test_compliant_wrapper_no_violation(fake_repo: Path) -> None:
    rname = "substrate_foo_modal_a100_dispatch"
    (fake_repo / ".omx" / "operator_authorize_recipes" / f"{rname}.yaml").write_text(
        _make_recipe(2000)
    )
    (fake_repo / "scripts" / f"operator_authorize_{rname}.sh").write_text(
        _make_compliant_wrapper()
    )
    out = check_substrate_dispatch_uses_smoke_before_full_pattern(
        repo_root=fake_repo, strict=False
    )
    assert out == []


def test_non_compliant_full_canary_is_violation(fake_repo: Path) -> None:
    rname = "substrate_foo_modal_a100_dispatch"
    (fake_repo / ".omx" / "operator_authorize_recipes" / f"{rname}.yaml").write_text(
        _make_recipe(2000)
    )
    (fake_repo / "scripts" / f"operator_authorize_{rname}.sh").write_text(
        _make_non_compliant_wrapper(rname)
    )
    out = check_substrate_dispatch_uses_smoke_before_full_pattern(
        repo_root=fake_repo, strict=False
    )
    assert any("fires a full canary" in v for v in out)


def test_waiver_clears_violation(fake_repo: Path) -> None:
    rname = "substrate_foo_modal_a100_dispatch"
    (fake_repo / ".omx" / "operator_authorize_recipes" / f"{rname}.yaml").write_text(
        _make_recipe(2000)
    )
    (fake_repo / "scripts" / f"operator_authorize_{rname}.sh").write_text(
        _make_waiver_wrapper(rname)
    )
    out = check_substrate_dispatch_uses_smoke_before_full_pattern(
        repo_root=fake_repo, strict=False
    )
    assert out == []


def test_smoke_canary_below_threshold_not_required(fake_repo: Path) -> None:
    rname = "substrate_foo_modal_a100_dispatch"
    (fake_repo / ".omx" / "operator_authorize_recipes" / f"{rname}.yaml").write_text(
        _make_recipe(100)
    )
    (fake_repo / "scripts" / f"operator_authorize_{rname}.sh").write_text(
        _make_non_compliant_wrapper(rname)
    )
    out = check_substrate_dispatch_uses_smoke_before_full_pattern(
        repo_root=fake_repo, strict=False
    )
    assert out == []


def test_recipe_with_no_cost_band_block_skipped(fake_repo: Path) -> None:
    rname = "substrate_foo_modal_a100_dispatch"
    (fake_repo / ".omx" / "operator_authorize_recipes" / f"{rname}.yaml").write_text(
        _make_recipe(None)
    )
    (fake_repo / "scripts" / f"operator_authorize_{rname}.sh").write_text(
        _make_non_compliant_wrapper(rname)
    )
    out = check_substrate_dispatch_uses_smoke_before_full_pattern(
        repo_root=fake_repo, strict=False
    )
    assert out == []


def test_orphan_wrapper_with_no_recipe_skipped(fake_repo: Path) -> None:
    (fake_repo / "scripts" / "operator_authorize_substrate_orphan_modal_a100_dispatch.sh").write_text(
        _make_non_compliant_wrapper("does_not_exist")
    )
    out = check_substrate_dispatch_uses_smoke_before_full_pattern(
        repo_root=fake_repo, strict=False
    )
    assert out == []


def test_mixed_wrappers_only_non_compliant_flagged(fake_repo: Path) -> None:
    rname1 = "substrate_alpha_modal_a100_dispatch"
    rname2 = "substrate_beta_modal_a100_dispatch"
    (fake_repo / ".omx" / "operator_authorize_recipes" / f"{rname1}.yaml").write_text(
        _make_recipe(2000)
    )
    (fake_repo / ".omx" / "operator_authorize_recipes" / f"{rname2}.yaml").write_text(
        _make_recipe(2000)
    )
    (fake_repo / "scripts" / f"operator_authorize_{rname1}.sh").write_text(
        _make_compliant_wrapper()
    )
    (fake_repo / "scripts" / f"operator_authorize_{rname2}.sh").write_text(
        _make_non_compliant_wrapper(rname2)
    )
    out = check_substrate_dispatch_uses_smoke_before_full_pattern(
        repo_root=fake_repo, strict=False
    )
    assert len(out) == 1
    assert rname2 in out[0]
    assert rname1 not in out[0]


def test_no_scripts_dir_returns_empty(tmp_path: Path) -> None:
    repo = tmp_path / "no_scripts_repo"
    repo.mkdir()
    out = check_substrate_dispatch_uses_smoke_before_full_pattern(
        repo_root=repo, strict=False
    )
    assert out == []


def test_strict_mode_raises_preflight_error(fake_repo: Path) -> None:
    rname = "substrate_foo_modal_a100_dispatch"
    (fake_repo / ".omx" / "operator_authorize_recipes" / f"{rname}.yaml").write_text(
        _make_recipe(2000)
    )
    (fake_repo / "scripts" / f"operator_authorize_{rname}.sh").write_text(
        _make_non_compliant_wrapper(rname)
    )
    with pytest.raises(PreflightError) as excinfo:
        check_substrate_dispatch_uses_smoke_before_full_pattern(
            repo_root=fake_repo, strict=True
        )
    assert "Catalog #167" in str(excinfo.value)
    assert "PHASE-B1-PIVOT" in str(excinfo.value)


def test_non_strict_returns_violation_list(fake_repo: Path) -> None:
    rname = "substrate_foo_modal_a100_dispatch"
    (fake_repo / ".omx" / "operator_authorize_recipes" / f"{rname}.yaml").write_text(
        _make_recipe(2000)
    )
    (fake_repo / "scripts" / f"operator_authorize_{rname}.sh").write_text(
        _make_non_compliant_wrapper(rname)
    )
    out = check_substrate_dispatch_uses_smoke_before_full_pattern(
        repo_root=fake_repo, strict=False
    )
    assert isinstance(out, list)
    assert len(out) == 1


def test_verbose_prints_status_banner(fake_repo: Path, capsys) -> None:
    check_substrate_dispatch_uses_smoke_before_full_pattern(
        repo_root=fake_repo, strict=False, verbose=True
    )
    captured = capsys.readouterr()
    assert "smoke-before-full" in captured.out


def test_verbose_with_violations_prints_count(fake_repo: Path, capsys) -> None:
    rname = "substrate_foo_modal_a100_dispatch"
    (fake_repo / ".omx" / "operator_authorize_recipes" / f"{rname}.yaml").write_text(
        _make_recipe(2000)
    )
    (fake_repo / "scripts" / f"operator_authorize_{rname}.sh").write_text(
        _make_non_compliant_wrapper(rname)
    )
    check_substrate_dispatch_uses_smoke_before_full_pattern(
        repo_root=fake_repo, strict=False, verbose=True
    )
    captured = capsys.readouterr()
    assert "smoke-before-full" in captured.out
    assert "violation" in captured.out


def test_threshold_exactly_1000_is_full(fake_repo: Path) -> None:
    rname = "substrate_foo_modal_a100_dispatch"
    (fake_repo / ".omx" / "operator_authorize_recipes" / f"{rname}.yaml").write_text(
        _make_recipe(1000)
    )
    (fake_repo / "scripts" / f"operator_authorize_{rname}.sh").write_text(
        _make_non_compliant_wrapper(rname)
    )
    out = check_substrate_dispatch_uses_smoke_before_full_pattern(
        repo_root=fake_repo, strict=False
    )
    assert any("fires a full canary" in v for v in out)


def test_threshold_999_is_smoke(fake_repo: Path) -> None:
    rname = "substrate_foo_modal_a100_dispatch"
    (fake_repo / ".omx" / "operator_authorize_recipes" / f"{rname}.yaml").write_text(
        _make_recipe(999)
    )
    (fake_repo / "scripts" / f"operator_authorize_{rname}.sh").write_text(
        _make_non_compliant_wrapper(rname)
    )
    out = check_substrate_dispatch_uses_smoke_before_full_pattern(
        repo_root=fake_repo, strict=False
    )
    assert out == []
