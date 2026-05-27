# SPDX-License-Identifier: MIT
"""Tests for Catalogs #365 / #366 / #367 — Cascade C' bug-class extinction wave.

FIX-WAVE-CASCADE-C-PRIME-ALL-BUGS-PERMANENT-FIX-AND-SELF-PROTECT 2026-05-26.

Catalog #365 (canonical helper signature drift): per Cascade C' subagent C
empirical anchor (commit a885ea2e5) — Modal T4 dispatch
fc-01KSK7GTPEF27FX0AAH2319GVR stage 7 auth_eval TypeError because
gate_auth_eval_call() was called with deprecated kwargs (archive=, json_out=,
lane_id=, substrate_id=) instead of canonical (archive_zip=, output_json=,
substrate_tag=, args=, contest_auth_eval_script=).

Catalog #366 (inflate shim import drift): per Cascade C' Wave 2 empirical
anchor (commit 3c2ce7fc2) — ImportError 'cannot import name main from
cascade_c_prime_frame_1_segnet_waterfill.inflate' because the trainer
wrapper's emitted inflate.py imported 'main' but the canonical module
exports 'main_cli'.

Catalog #367 (inflate raw bytes fail-open): per Cascade C' WAVE-3 empirical
anchor (commit 39e1db080 + fix commits 5bcb53070 + d0c4517ea) — Modal T4
dispatch fc-01KSKB4B30DCYTCP883XYV5BNV emitted 708MB instead of 3.66GB
(1200 frames at 384x512x3 instead of 1164x874x3); contest_auth_eval crashed
WRONG-SIZE because inflate.py emitted raw bytes without a fail-closed check
against the contest output contract.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac import preflight
from tac.preflight import PreflightError


REPO_ROOT = Path(__file__).resolve().parents[3]


# ============================================================================
# Catalog #365: canonical helper signature drift
# ============================================================================


def _write_trainer_with_kwargs(tmp_path: Path, kwargs_block: str) -> Path:
    """Write a synthetic substrate trainer file with the given kwargs block."""
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    fp = exp_dir / "train_substrate_synthetic_for_365.py"
    fp.write_text(
        f"""from tac.substrates._shared.smoke_auth_eval_gate import gate_auth_eval_call

def main():
    result = gate_auth_eval_call(
{kwargs_block}
    )
"""
    )
    return fp


def test_check_365_clean_repo_live_count_under_threshold():
    """Live-repo regression guard: count must be 0 at landing (STRICT @ 0)."""
    violations = preflight.check_substrate_trainer_routes_through_canonical_gate_auth_eval_call_with_correct_kwargs(
        strict=False, verbose=False
    )
    assert len(violations) == 0, (
        f"Catalog #365 live count regressed from 0 to {len(violations)}: {violations[:3]}"
    )


def test_check_365_deprecated_archive_kwarg_flagged(tmp_path):
    """archive=path detected as deprecated (canonical = archive_zip=)."""
    _write_trainer_with_kwargs(
        tmp_path,
        '''        args=args,
        archive=archive_path,
        inflate_sh=inflate_sh,
        upstream_dir=upstream_dir,
        output_json=output_json,
        contest_auth_eval_script=script,
        substrate_tag="x",''',
    )
    v = preflight.check_substrate_trainer_routes_through_canonical_gate_auth_eval_call_with_correct_kwargs(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) >= 1
    assert any("archive" in vi and "Catalog #365" in vi for vi in v)


def test_check_365_deprecated_json_out_kwarg_flagged(tmp_path):
    """json_out=path detected as deprecated (canonical = output_json=)."""
    _write_trainer_with_kwargs(
        tmp_path,
        '''        args=args,
        archive_zip=archive_path,
        inflate_sh=inflate_sh,
        upstream_dir=upstream_dir,
        json_out=output_json,
        contest_auth_eval_script=script,
        substrate_tag="x",''',
    )
    v = preflight.check_substrate_trainer_routes_through_canonical_gate_auth_eval_call_with_correct_kwargs(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) >= 1
    assert any("json_out" in vi for vi in v)


def test_check_365_deprecated_lane_id_kwarg_flagged(tmp_path):
    """lane_id= is not in canonical signature; flagged."""
    _write_trainer_with_kwargs(
        tmp_path,
        '''        args=args,
        archive_zip=archive_path,
        inflate_sh=inflate_sh,
        upstream_dir=upstream_dir,
        output_json=output_json,
        contest_auth_eval_script=script,
        substrate_tag="x",
        lane_id="lane_foo",''',
    )
    v = preflight.check_substrate_trainer_routes_through_canonical_gate_auth_eval_call_with_correct_kwargs(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert any("lane_id" in vi for vi in v)


def test_check_365_substrate_id_replaced_by_substrate_tag(tmp_path):
    """substrate_id= deprecated; canonical = substrate_tag=."""
    _write_trainer_with_kwargs(
        tmp_path,
        '''        args=args,
        archive_zip=archive_path,
        inflate_sh=inflate_sh,
        upstream_dir=upstream_dir,
        output_json=output_json,
        contest_auth_eval_script=script,
        substrate_id="x",''',
    )
    v = preflight.check_substrate_trainer_routes_through_canonical_gate_auth_eval_call_with_correct_kwargs(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert any("substrate_id" in vi for vi in v)
    # substrate_tag is missing in this fixture; also flagged
    assert any("substrate_tag" in vi for vi in v)


def test_check_365_missing_required_args_flagged(tmp_path):
    """Missing args= kwarg is a violation."""
    _write_trainer_with_kwargs(
        tmp_path,
        '''        archive_zip=archive_path,
        inflate_sh=inflate_sh,
        upstream_dir=upstream_dir,
        output_json=output_json,
        contest_auth_eval_script=script,
        substrate_tag="x",''',
    )
    v = preflight.check_substrate_trainer_routes_through_canonical_gate_auth_eval_call_with_correct_kwargs(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert any("args" in vi for vi in v)


def test_check_365_canonical_kwargs_clean(tmp_path):
    """All canonical kwargs present + no deprecated kwargs = clean."""
    _write_trainer_with_kwargs(
        tmp_path,
        '''        args=args,
        archive_zip=archive_path,
        inflate_sh=inflate_sh,
        upstream_dir=upstream_dir,
        output_json=output_json,
        contest_auth_eval_script=script,
        substrate_tag="x",''',
    )
    v = preflight.check_substrate_trainer_routes_through_canonical_gate_auth_eval_call_with_correct_kwargs(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_check_365_strict_raises_on_violation(tmp_path):
    """strict=True raises PreflightError on violation."""
    _write_trainer_with_kwargs(
        tmp_path,
        '''        archive=archive_path,
        inflate_sh=inflate_sh,''',
    )
    with pytest.raises(PreflightError, match="Catalog #365"):
        preflight.check_substrate_trainer_routes_through_canonical_gate_auth_eval_call_with_correct_kwargs(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_check_365_strict_silent_on_clean(tmp_path):
    """strict=True with no violations does not raise."""
    _write_trainer_with_kwargs(
        tmp_path,
        '''        args=args,
        archive_zip=archive_path,
        inflate_sh=inflate_sh,
        upstream_dir=upstream_dir,
        output_json=output_json,
        contest_auth_eval_script=script,
        substrate_tag="x",''',
    )
    # Should not raise
    v = preflight.check_substrate_trainer_routes_through_canonical_gate_auth_eval_call_with_correct_kwargs(
        repo_root=tmp_path, strict=True, verbose=False
    )
    assert v == []


def test_check_365_same_line_waiver_accepted(tmp_path):
    """Same-line waiver with rationale >=4 chars accepted."""
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    fp = exp_dir / "train_substrate_synthetic_for_365_waiver.py"
    fp.write_text(
        '''from tac.substrates._shared.smoke_auth_eval_gate import gate_auth_eval_call

def main():
    result = gate_auth_eval_call(  # CANONICAL_GATE_KWARG_DRIFT_OK:test_fixture_with_substantive_rationale
        archive=archive_path,
    )
'''
    )
    v = preflight.check_substrate_trainer_routes_through_canonical_gate_auth_eval_call_with_correct_kwargs(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_check_365_placeholder_waiver_rejected(tmp_path):
    """Placeholder <rationale> literal rejected per Catalog #287."""
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    fp = exp_dir / "train_substrate_synthetic_for_365_placeholder.py"
    fp.write_text(
        '''from tac.substrates._shared.smoke_auth_eval_gate import gate_auth_eval_call

def main():
    result = gate_auth_eval_call(  # CANONICAL_GATE_KWARG_DRIFT_OK:<rationale>
        archive=archive_path,
    )
'''
    )
    v = preflight.check_substrate_trainer_routes_through_canonical_gate_auth_eval_call_with_correct_kwargs(
        repo_root=tmp_path, strict=False, verbose=False
    )
    # placeholder rejected; violation persists
    assert len(v) >= 1


def test_check_365_short_rationale_rejected(tmp_path):
    """Rationale <4 chars rejected."""
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    fp = exp_dir / "train_substrate_synthetic_for_365_short.py"
    fp.write_text(
        '''from tac.substrates._shared.smoke_auth_eval_gate import gate_auth_eval_call

def main():
    result = gate_auth_eval_call(  # CANONICAL_GATE_KWARG_DRIFT_OK:abc
        archive=archive_path,
    )
'''
    )
    v = preflight.check_substrate_trainer_routes_through_canonical_gate_auth_eval_call_with_correct_kwargs(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) >= 1


def test_check_365_string_repo_root_accepted(tmp_path):
    """repo_root parameter accepts str (not only Path)."""
    v = preflight.check_substrate_trainer_routes_through_canonical_gate_auth_eval_call_with_correct_kwargs(
        repo_root=str(tmp_path), strict=False, verbose=False
    )
    assert isinstance(v, list)


def test_check_365_orchestrator_strict_true():
    """Catalog #365 callsite in preflight_all uses strict=True."""
    src = (REPO_ROOT / "src/tac/preflight.py").read_text()
    # find the callsite in preflight_all body
    needle = "check_substrate_trainer_routes_through_canonical_gate_auth_eval_call_with_correct_kwargs(\n            strict=True"
    assert needle in src, "Catalog #365 must be wired strict=True in preflight_all"


def test_check_365_catalog_185_callable_via_globals():
    """Per Catalog #185, the gate function must be callable via module globals."""
    fn = getattr(preflight, "check_substrate_trainer_routes_through_canonical_gate_auth_eval_call_with_correct_kwargs", None)
    assert fn is not None and callable(fn)


def test_check_365_message_includes_catalog_id(tmp_path):
    """Violation message must cite 'Catalog #365'."""
    _write_trainer_with_kwargs(
        tmp_path,
        '''        archive=archive_path,''',
    )
    v = preflight.check_substrate_trainer_routes_through_canonical_gate_auth_eval_call_with_correct_kwargs(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert any("Catalog #365" in vi for vi in v)


def test_check_365_renderer_files_in_scope(tmp_path):
    """train_renderer*.py files also scanned (not only substrate trainers)."""
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    fp = exp_dir / "train_renderer_synthetic.py"
    fp.write_text(
        '''from tac.substrates._shared.smoke_auth_eval_gate import gate_auth_eval_call
def main():
    result = gate_auth_eval_call(archive=foo)
'''
    )
    v = preflight.check_substrate_trainer_routes_through_canonical_gate_auth_eval_call_with_correct_kwargs(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) >= 1
    assert any("train_renderer" in vi for vi in v)


def test_check_365_aliased_helper_call_detected(tmp_path):
    """`gate as _canon_gate_auth_eval_call` aliasing must still be caught."""
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    fp = exp_dir / "train_substrate_alias_synthetic.py"
    fp.write_text(
        '''from tac.substrates._shared.smoke_auth_eval_gate import (
    gate_auth_eval_call as _canon_gate_auth_eval_call,
)
def main():
    result = _canon_gate_auth_eval_call(archive=foo)
'''
    )
    v = preflight.check_substrate_trainer_routes_through_canonical_gate_auth_eval_call_with_correct_kwargs(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert any("Catalog #365" in vi for vi in v)


# ============================================================================
# Catalog #366: inflate shim import drift
# ============================================================================


def test_check_366_clean_repo_live_count_zero():
    """Live-repo regression guard: count must be 0 at landing (STRICT @ 0)."""
    v = preflight.check_substrate_trainer_emitted_inflate_shim_imports_match_canonical_module_exports(
        strict=False, verbose=False
    )
    assert len(v) == 0, (
        f"Catalog #366 live count regressed from 0 to {len(v)}: {v[:3]}"
    )


def test_check_366_main_vs_main_cli_drift_flagged(tmp_path):
    """Synthetic Cascade C' Wave 2 bug class: shim imports 'main' but module exports 'main_cli'."""
    # Build the substrate module exporting main_cli
    sub_dir = tmp_path / "src" / "tac" / "substrates" / "synthetic_366"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "__init__.py").write_text("")
    (sub_dir / "inflate.py").write_text(
        '''def main_cli():
    return 0
'''
    )
    # Trainer emits a shim importing the wrong name
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "train_substrate_synthetic_366.py").write_text(
        '''shim = """from tac.substrates.synthetic_366.inflate import main
if __name__ == "__main__":
    sys.exit(main())
"""
'''
    )
    v = preflight.check_substrate_trainer_emitted_inflate_shim_imports_match_canonical_module_exports(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) >= 1
    assert any("Catalog #366" in vi for vi in v)
    assert any("main" in vi for vi in v)


def test_check_366_main_cli_as_main_alias_clean(tmp_path):
    """The Cascade C' fix `import main_cli as main` is the canonical pattern."""
    sub_dir = tmp_path / "src" / "tac" / "substrates" / "synthetic_366_clean"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "__init__.py").write_text("")
    (sub_dir / "inflate.py").write_text(
        '''def main_cli():
    return 0
'''
    )
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "train_substrate_synthetic_366_clean.py").write_text(
        '''shim = """from tac.substrates.synthetic_366_clean.inflate import main_cli as main
if __name__ == "__main__":
    sys.exit(main())
"""
'''
    )
    v = preflight.check_substrate_trainer_emitted_inflate_shim_imports_match_canonical_module_exports(
        repo_root=tmp_path, strict=False, verbose=False
    )
    # 'main_cli' is exported; gate accepts even though shim's local alias is 'main'
    assert v == []


def test_check_366_missing_module_flagged(tmp_path):
    """Shim referencing non-existent module is flagged."""
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "train_substrate_synthetic_366_missing.py").write_text(
        '''shim = """from tac.substrates.nonexistent_module.inflate import main_cli
"""
'''
    )
    v = preflight.check_substrate_trainer_emitted_inflate_shim_imports_match_canonical_module_exports(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) >= 1


def test_check_366_strict_raises_on_violation(tmp_path):
    """strict=True raises PreflightError on shim drift."""
    sub_dir = tmp_path / "src" / "tac" / "substrates" / "synthetic_366_strict"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "__init__.py").write_text("")
    (sub_dir / "inflate.py").write_text("def main_cli():\n    pass\n")
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "train_substrate_synthetic_366_strict.py").write_text(
        '''shim = """from tac.substrates.synthetic_366_strict.inflate import main
"""
'''
    )
    with pytest.raises(PreflightError, match="Catalog #366"):
        preflight.check_substrate_trainer_emitted_inflate_shim_imports_match_canonical_module_exports(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_check_366_strict_silent_on_clean(tmp_path):
    """strict=True with no violations does not raise."""
    sub_dir = tmp_path / "src" / "tac" / "substrates" / "synthetic_366_silent"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "__init__.py").write_text("")
    (sub_dir / "inflate.py").write_text("def inflate_one_video():\n    pass\n")
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "train_substrate_synthetic_366_silent.py").write_text(
        '''shim = """from tac.substrates.synthetic_366_silent.inflate import inflate_one_video
"""
'''
    )
    v = preflight.check_substrate_trainer_emitted_inflate_shim_imports_match_canonical_module_exports(
        repo_root=tmp_path, strict=True, verbose=False
    )
    assert v == []


def test_check_366_waiver_with_rationale_accepted(tmp_path):
    """Same-line waiver with substantive rationale accepted."""
    sub_dir = tmp_path / "src" / "tac" / "substrates" / "synthetic_366_waiver"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "__init__.py").write_text("")
    (sub_dir / "inflate.py").write_text("def main_cli():\n    pass\n")
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "train_substrate_synthetic_366_waiver.py").write_text(
        '''shim = """from tac.substrates.synthetic_366_waiver.inflate import main"""  # INFLATE_SHIM_IMPORT_DRIFT_OK:test_fixture_intentional_legacy_alias_pattern_documented
'''
    )
    v = preflight.check_substrate_trainer_emitted_inflate_shim_imports_match_canonical_module_exports(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_check_366_placeholder_waiver_rejected(tmp_path):
    """Placeholder rationale rejected per Catalog #287."""
    sub_dir = tmp_path / "src" / "tac" / "substrates" / "synthetic_366_placeholder"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "__init__.py").write_text("")
    (sub_dir / "inflate.py").write_text("def main_cli():\n    pass\n")
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "train_substrate_synthetic_366_placeholder.py").write_text(
        '''shim = """from tac.substrates.synthetic_366_placeholder.inflate import main"""  # INFLATE_SHIM_IMPORT_DRIFT_OK:<rationale>
'''
    )
    v = preflight.check_substrate_trainer_emitted_inflate_shim_imports_match_canonical_module_exports(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) >= 1


def test_check_366_short_rationale_rejected(tmp_path):
    """Rationale <4 chars rejected."""
    sub_dir = tmp_path / "src" / "tac" / "substrates" / "synthetic_366_short"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "__init__.py").write_text("")
    (sub_dir / "inflate.py").write_text("def main_cli():\n    pass\n")
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "train_substrate_synthetic_366_short.py").write_text(
        '''shim = """from tac.substrates.synthetic_366_short.inflate import main"""  # INFLATE_SHIM_IMPORT_DRIFT_OK:ab
'''
    )
    v = preflight.check_substrate_trainer_emitted_inflate_shim_imports_match_canonical_module_exports(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) >= 1


def test_check_366_no_trainer_dir_silent(tmp_path):
    """Missing experiments/ dir handled gracefully."""
    v = preflight.check_substrate_trainer_emitted_inflate_shim_imports_match_canonical_module_exports(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert v == []


def test_check_366_string_repo_root_accepted(tmp_path):
    """repo_root accepts str."""
    v = preflight.check_substrate_trainer_emitted_inflate_shim_imports_match_canonical_module_exports(
        repo_root=str(tmp_path), strict=False, verbose=False
    )
    assert isinstance(v, list)


def test_check_366_orchestrator_strict_true():
    """Catalog #366 callsite in preflight_all uses strict=True."""
    src = (REPO_ROOT / "src/tac/preflight.py").read_text()
    needle = "check_substrate_trainer_emitted_inflate_shim_imports_match_canonical_module_exports(\n            strict=True"
    assert needle in src


def test_check_366_catalog_185_callable_via_globals():
    """Per Catalog #185, the gate function must be callable via module globals."""
    fn = getattr(preflight, "check_substrate_trainer_emitted_inflate_shim_imports_match_canonical_module_exports", None)
    assert fn is not None and callable(fn)


def test_check_366_message_includes_catalog_id(tmp_path):
    """Violation message must cite 'Catalog #366'."""
    sub_dir = tmp_path / "src" / "tac" / "substrates" / "synthetic_366_msg"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "__init__.py").write_text("")
    (sub_dir / "inflate.py").write_text("def main_cli():\n    pass\n")
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "train_substrate_synthetic_366_msg.py").write_text(
        '''shim = """from tac.substrates.synthetic_366_msg.inflate import main"""
'''
    )
    v = preflight.check_substrate_trainer_emitted_inflate_shim_imports_match_canonical_module_exports(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert any("Catalog #366" in vi for vi in v)


def test_check_366_renderer_files_in_scope(tmp_path):
    """train_renderer*.py files also scanned."""
    sub_dir = tmp_path / "src" / "tac" / "substrates" / "synthetic_366_renderer"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "__init__.py").write_text("")
    (sub_dir / "inflate.py").write_text("def main_cli():\n    pass\n")
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "train_renderer_synthetic_366.py").write_text(
        '''shim = """from tac.substrates.synthetic_366_renderer.inflate import main"""
'''
    )
    v = preflight.check_substrate_trainer_emitted_inflate_shim_imports_match_canonical_module_exports(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert any("train_renderer" in vi for vi in v)


# ============================================================================
# Catalog #367: inflate raw bytes fail-open
# ============================================================================


def test_check_367_live_count_bounded():
    """Live-repo regression guard: count must be <=10 at landing.

    WARN-ONLY initial wire-in per CLAUDE.md "Strict-flip atomicity rule"
    because pre-existing inflate.py files need backfill / waiver.
    """
    v = preflight.check_substrate_inflate_emits_expected_frame_count_or_fail_closed(
        strict=False, verbose=False
    )
    assert len(v) <= 10, (
        f"Catalog #367 live count regressed beyond ceiling 10: {len(v)} violations"
    )


def test_check_367_synthetic_fail_open_flagged(tmp_path):
    """Synthetic inflate.py that emits raw + cites contract but no fail-closed = flagged."""
    sub_dir = tmp_path / "src" / "tac" / "substrates" / "synthetic_367_open"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "__init__.py").write_text("")
    (sub_dir / "inflate.py").write_text(
        '''import numpy as np
CONTEST_RAW_BYTES = 3662409600
def inflate_one_video(archive_bytes, output_stem):
    out_path = output_stem.with_suffix(".raw")
    out_path.write_bytes(b"\\x00" * 700_000_000)
    return out_path
'''
    )
    v = preflight.check_substrate_inflate_emits_expected_frame_count_or_fail_closed(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(v) >= 1
    assert any("Catalog #367" in vi and "synthetic_367_open" in vi for vi in v)


def test_check_367_fail_closed_check_clean(tmp_path):
    """Inflate with fail-closed check is clean."""
    sub_dir = tmp_path / "src" / "tac" / "substrates" / "synthetic_367_closed"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "__init__.py").write_text("")
    (sub_dir / "inflate.py").write_text(
        '''import numpy as np
CONTEST_RAW_BYTES = 3662409600
def inflate_one_video(archive_bytes, output_stem):
    out_path = output_stem.with_suffix(".raw")
    raw_bytes = 1200 * 1164 * 874 * 3
    if raw_bytes != CONTEST_RAW_BYTES:
        raise AssertionError("contest raw byte contract drifted")
    out_path.write_bytes(b"\\x00" * raw_bytes)
    return out_path
'''
    )
    v = preflight.check_substrate_inflate_emits_expected_frame_count_or_fail_closed(
        repo_root=tmp_path, strict=False, verbose=False
    )
    # Check that the synthetic file is NOT in violations
    assert not any("synthetic_367_closed" in vi for vi in v)


def test_check_367_scaffold_without_raw_write_out_of_scope(tmp_path):
    """Inflate scaffold that doesn't write raw bytes is out of scope."""
    sub_dir = tmp_path / "src" / "tac" / "substrates" / "synthetic_367_scaffold"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "__init__.py").write_text("")
    (sub_dir / "inflate.py").write_text(
        '''def main_cli():
    pass
'''
    )
    v = preflight.check_substrate_inflate_emits_expected_frame_count_or_fail_closed(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert not any("synthetic_367_scaffold" in vi for vi in v)


def test_check_367_file_level_waiver_accepted(tmp_path):
    """File-level waiver with substantive rationale accepted."""
    sub_dir = tmp_path / "src" / "tac" / "substrates" / "synthetic_367_waiver"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "__init__.py").write_text("")
    (sub_dir / "inflate.py").write_text(
        '''# INFLATE_FRAME_COUNT_FAIL_OPEN_OK:test_fixture_legitimate_streaming_emission_pattern_delegates_to_sister_module
import numpy as np
CONTEST_RAW_BYTES = 3662409600
def inflate_one_video(archive_bytes, output_stem):
    out_path = output_stem.with_suffix(".raw")
    out_path.write_bytes(b"\\x00" * 700_000_000)
    return out_path
'''
    )
    v = preflight.check_substrate_inflate_emits_expected_frame_count_or_fail_closed(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert not any("synthetic_367_waiver" in vi for vi in v)


def test_check_367_placeholder_waiver_rejected(tmp_path):
    """Placeholder <rationale> rejected per Catalog #287."""
    sub_dir = tmp_path / "src" / "tac" / "substrates" / "synthetic_367_placeholder"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "__init__.py").write_text("")
    (sub_dir / "inflate.py").write_text(
        '''# INFLATE_FRAME_COUNT_FAIL_OPEN_OK:<rationale>
import numpy as np
CONTEST_RAW_BYTES = 3662409600
def inflate_one_video(archive_bytes, output_stem):
    out_path = output_stem.with_suffix(".raw")
    out_path.write_bytes(b"\\x00" * 700_000_000)
    return out_path
'''
    )
    v = preflight.check_substrate_inflate_emits_expected_frame_count_or_fail_closed(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert any("synthetic_367_placeholder" in vi for vi in v)


def test_check_367_short_rationale_rejected(tmp_path):
    """Rationale <4 chars rejected."""
    sub_dir = tmp_path / "src" / "tac" / "substrates" / "synthetic_367_short"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "__init__.py").write_text("")
    (sub_dir / "inflate.py").write_text(
        '''# INFLATE_FRAME_COUNT_FAIL_OPEN_OK:ab
import numpy as np
CONTEST_RAW_BYTES = 3662409600
def inflate_one_video(archive_bytes, output_stem):
    out_path = output_stem.with_suffix(".raw")
    out_path.write_bytes(b"\\x00" * 700_000_000)
    return out_path
'''
    )
    v = preflight.check_substrate_inflate_emits_expected_frame_count_or_fail_closed(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert any("synthetic_367_short" in vi for vi in v)


def test_check_367_exact_current_exempt(tmp_path):
    """submissions/exact_current/ is exempt per CLAUDE.md mutation frontier."""
    sub_dir = tmp_path / "submissions" / "exact_current"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "inflate.py").write_text(
        '''import numpy as np
CONTEST_RAW_BYTES = 3662409600
def inflate_one_video(archive_bytes, output_stem):
    out_path = output_stem.with_suffix(".raw")
    out_path.write_bytes(b"\\x00" * 700_000_000)
    return out_path
'''
    )
    v = preflight.check_substrate_inflate_emits_expected_frame_count_or_fail_closed(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert not any("exact_current" in vi for vi in v)


def test_check_367_warn_only_orchestrator():
    """Catalog #367 callsite in preflight_all uses strict=False (WARN-ONLY initial)."""
    src = (REPO_ROOT / "src/tac/preflight.py").read_text()
    needle = "check_substrate_inflate_emits_expected_frame_count_or_fail_closed(\n            strict=False"
    assert needle in src, "Catalog #367 must be wired strict=False initially per Strict-flip atomicity rule"


def test_check_367_strict_raises_on_violation(tmp_path):
    """strict=True raises PreflightError on violation."""
    sub_dir = tmp_path / "src" / "tac" / "substrates" / "synthetic_367_strict_test"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "__init__.py").write_text("")
    (sub_dir / "inflate.py").write_text(
        '''import numpy as np
CONTEST_RAW_BYTES = 3662409600
def inflate_one_video(archive_bytes, output_stem):
    out_path = output_stem.with_suffix(".raw")
    out_path.write_bytes(b"\\x00" * 700_000_000)
    return out_path
'''
    )
    with pytest.raises(PreflightError, match="Catalog #367"):
        preflight.check_substrate_inflate_emits_expected_frame_count_or_fail_closed(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_check_367_string_repo_root_accepted(tmp_path):
    """repo_root accepts str."""
    v = preflight.check_substrate_inflate_emits_expected_frame_count_or_fail_closed(
        repo_root=str(tmp_path), strict=False, verbose=False
    )
    assert isinstance(v, list)


def test_check_367_catalog_185_callable_via_globals():
    """Per Catalog #185, the gate function must be callable via module globals."""
    fn = getattr(preflight, "check_substrate_inflate_emits_expected_frame_count_or_fail_closed", None)
    assert fn is not None and callable(fn)


def test_check_367_message_cites_wave_3_anchor(tmp_path):
    """Violation message must cite the WAVE-3 empirical anchor + canonical fix."""
    sub_dir = tmp_path / "src" / "tac" / "substrates" / "synthetic_367_anchor"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "__init__.py").write_text("")
    (sub_dir / "inflate.py").write_text(
        '''import numpy as np
CONTEST_RAW_BYTES = 3662409600
def inflate_one_video(archive_bytes, output_stem):
    out_path = output_stem.with_suffix(".raw")
    out_path.write_bytes(b"\\x00" * 700_000_000)
    return out_path
'''
    )
    v = preflight.check_substrate_inflate_emits_expected_frame_count_or_fail_closed(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert any("fc-01KSKB4B30DCYTCP883XYV5BNV" in vi for vi in v)
    assert any("Cascade C'" in vi for vi in v)


def test_check_367_cascade_c_prime_canonical_clean():
    """The actual Cascade C' inflate.py at HEAD passes (fix already landed)."""
    v = preflight.check_substrate_inflate_emits_expected_frame_count_or_fail_closed(
        strict=False, verbose=False
    )
    assert not any(
        "cascade_c_prime_frame_1_segnet_waterfill" in vi for vi in v
    ), "Cascade C' inflate.py MUST pass after WAVE-4 fixes (5bcb53070 + d0c4517ea)"


def test_check_367_strict_silent_on_clean(tmp_path):
    """strict=True with no violations does not raise."""
    sub_dir = tmp_path / "src" / "tac" / "substrates" / "synthetic_367_silent"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "__init__.py").write_text("")
    (sub_dir / "inflate.py").write_text("def main_cli(): pass\n")
    v = preflight.check_substrate_inflate_emits_expected_frame_count_or_fail_closed(
        repo_root=tmp_path, strict=True, verbose=False
    )
    assert v == []
