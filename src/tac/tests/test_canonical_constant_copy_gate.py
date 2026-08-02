"""ddm_hl1 (#847 family) — the DISGUISED built-elsewhere-unwired form.

A measured quantity copied out of its canonical producer into a live consumer as
bare float literals LOOKS wired: it passes review, carries a plausible comment
citing the memo it came from, and cannot track its source. A stub sweep marks it
GREEN because a mechanism does exist -- in the module it was copied from.

Two surfaces under test:
  1. ``tac.run_constant_gates`` P5 -- the detector (copied-TABLE signature).
  2. ``tac.optimization.lane_guard`` -- the named instance, now RESOLVED through
     ``tac.canonical_equations.segnet_head_rank4_flipdist_20260715``, byte-identically.

The gate tests are hermetic (synthetic repo trees), so they assert BEHAVIOUR, not
the live tree's current contents -- they would still fire if the live count changed.
The mutation guards assert what a marker-returning body could not do.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.optimization import lane_guard as lg
from tac.run_constant_gates import (
    CanonicalConstantCopyViolation,
    check_no_canonical_equation_constant_copied_as_literal,
    scan_repo_for_canonical_constant_copies,
)

# 4 distinct 4-significant-digit values: above both thresholds.
_TABLE = "(3.9531, 2.6018, 2.9415, 2.7053)"


def _make_repo(tmp_path: Path, consumer_rel: str, consumer_body: str) -> Path:
    """Synthetic repo: one canonical producer + one consumer."""
    canon = tmp_path / "src" / "tac" / "canonical_equations"
    canon.mkdir(parents=True)
    (canon / "demo_law_20260801.py").write_text(
        f'"""demo."""\nDEMO_PAIR_NORMS = {_TABLE}\n', encoding="utf-8"
    )
    consumer = tmp_path / consumer_rel
    consumer.parent.mkdir(parents=True, exist_ok=True)
    consumer.write_text(consumer_body, encoding="utf-8")
    return tmp_path


# ------------------------------------------------------------------ POSITIVE control
def test_copied_table_in_consumer_is_detected(tmp_path):
    """The canary: a consumer that copies the table and never names the producer."""
    root = _make_repo(
        tmp_path, "src/tac/optimization/consumer.py", f"NORMS = {_TABLE}\n"
    )
    findings = scan_repo_for_canonical_constant_copies(root)
    assert len(findings) == 1
    only = findings[0]
    assert only.path == "src/tac/optimization/consumer.py"
    assert only.owner_module == "demo_law_20260801"
    assert only.owner_constant == "DEMO_PAIR_NORMS"
    assert set(only.values) == {3.9531, 2.6018, 2.9415, 2.7053}


def test_detected_in_tools_subtree_too(tmp_path):
    root = _make_repo(tmp_path, "tools/deep/nested/consumer.py", f"N = {_TABLE}\n")
    assert len(scan_repo_for_canonical_constant_copies(root)) == 1


def test_scientific_notation_spelling_is_detected(tmp_path):
    """A value's repr is NOT its source text: 0.0004203 is written 4.203e-4.

    A repr-substring prefilter would silently miss this whole spelling class.
    """
    canon = tmp_path / "src" / "tac" / "canonical_equations"
    canon.mkdir(parents=True)
    (canon / "demo_sci_20260801.py").write_text(
        "MARGINALS = {'a': 4.203e-4, 'b': 3.298e-4, 'c': 3.059e-4}\n", encoding="utf-8"
    )
    consumer = tmp_path / "tools" / "c.py"
    consumer.parent.mkdir(parents=True)
    consumer.write_text("M = (4.203e-4, 3.298e-4, 3.059e-4)\n", encoding="utf-8")
    findings = scan_repo_for_canonical_constant_copies(tmp_path)
    assert len(findings) == 1
    assert set(findings[0].values) == {0.0004203, 0.0003298, 0.0003059}


# ------------------------------------------------------------------ NEGATIVE controls
def test_consumer_that_names_its_producer_is_silent(tmp_path):
    root = _make_repo(
        tmp_path,
        "src/tac/optimization/consumer.py",
        "from tac.canonical_equations.demo_law_20260801 import DEMO_PAIR_NORMS\n"
        "NORMS = tuple(sorted(DEMO_PAIR_NORMS, reverse=True))\n",
    )
    assert scan_repo_for_canonical_constant_copies(root) == []


def test_below_table_threshold_is_silent(tmp_path):
    """Two coincidental collisions are not a copied table (0.999 is every Adam beta2)."""
    root = _make_repo(tmp_path, "src/tac/optimization/c.py", "N = (3.9531, 2.6018)\n")
    assert scan_repo_for_canonical_constant_copies(root) == []


def test_three_significant_digit_values_are_not_indexed(tmp_path):
    """A 3-sig-digit value is usually a rounded convention, not a measurement."""
    canon = tmp_path / "src" / "tac" / "canonical_equations"
    canon.mkdir(parents=True)
    (canon / "demo_round_20260801.py").write_text(
        "ROUNDED = (1.25, 3.14, 2.71, 1.41)\n", encoding="utf-8"
    )
    consumer = tmp_path / "tools" / "c.py"
    consumer.parent.mkdir(parents=True)
    consumer.write_text("R = (1.25, 3.14, 2.71, 1.41)\n", encoding="utf-8")
    assert scan_repo_for_canonical_constant_copies(tmp_path) == []


def test_multi_owner_values_are_not_indexed(tmp_path):
    """A value claimed by two canonical modules is a shared tolerance, not a copy."""
    canon = tmp_path / "src" / "tac" / "canonical_equations"
    canon.mkdir(parents=True)
    for name in ("demo_a_20260801.py", "demo_b_20260801.py"):
        (canon / name).write_text(f"SHARED = {_TABLE}\n", encoding="utf-8")
    consumer = tmp_path / "tools" / "c.py"
    consumer.parent.mkdir(parents=True)
    consumer.write_text(f"N = {_TABLE}\n", encoding="utf-8")
    assert scan_repo_for_canonical_constant_copies(tmp_path) == []


@pytest.mark.parametrize(
    "rel",
    [
        "src/tac/tests/consumer.py",
        "src/tac/optimization/test_consumer.py",
        "tools/.venv/lib/consumer.py",
        "tools/vendor/site-packages/consumer.py",
    ],
)
def test_excluded_paths_are_not_scanned(tmp_path, rel):
    root = _make_repo(tmp_path, rel, f"N = {_TABLE}\n")
    assert scan_repo_for_canonical_constant_copies(root) == []


def test_exclusions_match_the_repo_relative_path_not_the_absolute_one(tmp_path):
    """Regression: an absolute path containing an exclusion marker must not empty scope.

    The first cut matched markers against ``str(path)``, so a checkout under any
    directory named ``tests``/``test_*`` (a pytest tmpdir, ``~/test_runs/pact``)
    skipped EVERY file and reported the same clean symbol as a genuinely clean tree.
    Vacuity is indistinguishable from PASS, so it is asserted against here.
    """
    poisoned = tmp_path / "test_runs" / "tests" / ".venv_backup" / "pact"
    root = _make_repo(
        poisoned, "src/tac/optimization/consumer.py", f"N = {_TABLE}\n"
    )
    assert len(scan_repo_for_canonical_constant_copies(root)) == 1


def test_experiments_subtree_is_out_of_scope(tmp_path):
    """Documented scope boundary: a trainer is the DSL's compile target, not a consumer."""
    root = _make_repo(tmp_path, "experiments/train_demo.py", f"N = {_TABLE}\n")
    assert scan_repo_for_canonical_constant_copies(root) == []


# ------------------------------------------------------------------------- waiver
def test_same_line_waiver_with_real_rationale_is_respected(tmp_path):
    root = _make_repo(
        tmp_path,
        "src/tac/optimization/c.py",
        f"N = {_TABLE}  # CANONICAL_CONSTANT_COPY_OK:frozen replay fixture, "
        "re-derive when the head weights sha changes\n",
    )
    assert scan_repo_for_canonical_constant_copies(root) == []


@pytest.mark.parametrize("rationale", ["<rationale>", "<reason>", "TBD", "todo", ""])
def test_placeholder_waiver_rationales_are_rejected(tmp_path, rationale):
    root = _make_repo(
        tmp_path,
        "src/tac/optimization/c.py",
        f"N = {_TABLE}  # CANONICAL_CONSTANT_COPY_OK:{rationale}\n",
    )
    assert len(scan_repo_for_canonical_constant_copies(root)) == 1


# -------------------------------------------------------------------- strict / API
def test_strict_raises_and_quotes_the_real_owner(tmp_path):
    root = _make_repo(tmp_path, "src/tac/optimization/c.py", f"N = {_TABLE}\n")
    with pytest.raises(RuntimeError) as excinfo:
        check_no_canonical_equation_constant_copied_as_literal(strict=True, repo_root=root)
    message = str(excinfo.value)
    assert "demo_law_20260801" in message and "DEMO_PAIR_NORMS" in message


def test_non_strict_returns_findings_without_raising(tmp_path):
    root = _make_repo(tmp_path, "src/tac/optimization/c.py", f"N = {_TABLE}\n")
    findings = check_no_canonical_equation_constant_copied_as_literal(
        strict=False, repo_root=root)
    assert len(findings) == 1
    assert isinstance(findings[0], CanonicalConstantCopyViolation)


def test_mutation_guard_describe_varies_with_input(tmp_path):
    """A canned marker string cannot pass: describe() must quote THIS finding."""
    root_a = _make_repo(tmp_path / "a", "src/tac/optimization/c.py", f"N = {_TABLE}\n")
    canon_b = tmp_path / "b" / "src" / "tac" / "canonical_equations"
    canon_b.mkdir(parents=True)
    (canon_b / "other_law_20260801.py").write_text(
        "OTHER = (7.1234, 8.5678, 9.8765)\n", encoding="utf-8")
    other = tmp_path / "b" / "tools" / "z.py"
    other.parent.mkdir(parents=True)
    other.write_text("Z = (7.1234, 8.5678, 9.8765)\n", encoding="utf-8")

    a = scan_repo_for_canonical_constant_copies(root_a)[0].describe()
    b = scan_repo_for_canonical_constant_copies(tmp_path / "b")[0].describe()
    assert a != b
    assert "DEMO_PAIR_NORMS" in a and "DEMO_PAIR_NORMS" not in b
    assert "7.1234" in b and "7.1234" not in a
    # The message must carry the actionable fix, not just a label.
    assert "import the constant from" in a


def test_gate_catches_the_real_pre_fix_lane_guard_instance(tmp_path):
    """THE positive control: the gate must catch the exact instance that motivated it.

    Rebuilds the pre-fix ``lane_guard`` shape -- the ten REAL head-pair normals as bare
    literals in a consumer that never names its producer -- against the REAL canonical
    constant. A detector that cannot reproduce its own founding case is not a detector.
    """
    from tac.canonical_equations.segnet_head_rank4_flipdist_20260715 import (
        HEAD_PAIR_NORMS,
    )

    canon = tmp_path / "src" / "tac" / "canonical_equations"
    canon.mkdir(parents=True)
    (canon / "segnet_head_rank4_flipdist_20260715.py").write_text(
        f"HEAD_PAIR_NORMS = {dict(HEAD_PAIR_NORMS)!r}\n", encoding="utf-8"
    )
    lane = sorted(
        (v for k, v in HEAD_PAIR_NORMS.items() if "Lane" in k.split("-")), reverse=True
    )
    every = sorted(HEAD_PAIR_NORMS.values(), reverse=True)
    consumer = tmp_path / "src" / "tac" / "optimization" / "lane_guard.py"
    consumer.parent.mkdir(parents=True)
    consumer.write_text(
        f"_LANE_PAIR_NORMS = {tuple(lane)!r}\n_ALL_PAIR_NORMS = {tuple(every)!r}\n",
        encoding="utf-8",
    )

    findings = scan_repo_for_canonical_constant_copies(tmp_path)
    assert len(findings) == 1
    assert findings[0].owner_constant == "HEAD_PAIR_NORMS"
    assert findings[0].path.endswith("lane_guard.py")
    # Every distinctive normal is recovered, so the finding is the whole table.
    assert len(findings[0].values) >= _COPIED_TABLE_MIN_DISTINCT_FOR_TEST


# The gate's own threshold, restated here so the control is independent of it.
_COPIED_TABLE_MIN_DISTINCT_FOR_TEST = 3


# ------------------------------------- the named instance: lane_guard, byte-identical
def test_lane_guard_resolves_pair_norms_from_the_canonical_producer():
    from tac.canonical_equations.segnet_head_rank4_flipdist_20260715 import (
        HEAD_PAIR_NORMS,
    )

    assert set(lg._ALL_PAIR_NORMS) == set(HEAD_PAIR_NORMS.values())
    assert set(lg._LANE_PAIR_NORMS) == {
        v for k, v in HEAD_PAIR_NORMS.items() if "Lane" in k.split("-")
    }


def test_lane_guard_pair_norms_are_byte_identical_to_the_pre_fix_literals():
    """The wiring must not move a single bit of the shipped behaviour."""
    assert lg._LANE_PAIR_NORMS == (4.007, 3.953, 3.862, 3.748)
    assert lg._ALL_PAIR_NORMS == (
        4.007, 3.953, 3.862, 3.748, 2.946, 2.942, 2.910, 2.869, 2.705, 2.602)
    assert lg.LANE_HEAD_SENSITIVITY_RATIO == 1.1960730088495577


def test_lane_guard_flip_distance_default_dw_norm_unchanged():
    import inspect

    default = inspect.signature(lg.per_component_min_flip_distance).parameters[
        "dw_norm"].default
    assert default == 4.007  # max Lane-pair normal (Lane-Movable), the conservative bound


def test_lane_guard_source_no_longer_carries_the_copied_table():
    """Regression: the ten normals must not reappear as bare literals."""
    source = Path(lg.__file__).read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines()
        if not line.lstrip().startswith("#")
    )
    # The docstring may still quote them prosaically; executable code must not.
    body = code.split('"""', 2)[-1]
    for literal in ("2.946", "2.942", "2.869", "2.705", "2.602"):
        assert literal not in body, f"{literal} reappeared as a bare literal"
