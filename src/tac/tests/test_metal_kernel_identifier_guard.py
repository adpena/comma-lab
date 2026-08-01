"""Controls for the Metal reserved-identifier guard.

Every assertion here exists because a REAL failure had this shape. The
positive control reproduces the exact 2026-07-12 defect; the negative controls
exist because a guard that cannot stay silent on correct code gets overridden,
and an overridden guard is the silence it was built to end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.metal_kernel_identifier_guard import (
    RESERVED_METAL_IDENTIFIERS,
    check_metal_kernel_identifiers_not_reserved,
    scan_source,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------
# POSITIVE CONTROL — the guard must FIRE on the historical defect.
# --------------------------------------------------------------------------


def test_positive_control_reproduces_the_20_day_defect() -> None:
    """The exact 2026-07-12 shape: a buffer named `signed`.

    MEASURED (ddm_tr6, b02b99cecb): this failed to compile on every dispatch
    for 20 days and no surface noticed.
    """
    src = (
        "kernel = mx.fast.metal_kernel(\n"
        '    name="phase_vjp",\n'
        '    input_names=["signed", "direction", "reference", "eps"],\n'
        '    output_names=["out"],\n'
        ")\n"
    )
    violations = scan_source(src, "positive_control.py")
    assert len(violations) == 1, "the reserved word must be caught"
    v = violations[0]
    assert v.name == "signed"
    assert v.kwarg == "input_names"
    assert v.line == 3
    assert "FAIL TO COMPILE" in v.render()


def test_positive_control_covers_output_names_too() -> None:
    """output_names is spliced into the signature identically."""
    src = 'k = mx.fast.metal_kernel(input_names=["x"], output_names=["template"])\n'
    violations = scan_source(src)
    assert [v.name for v in violations] == ["template"]


def test_signed_is_in_the_derived_keyword_set() -> None:
    """Provenance check: `signed` must come from the C++14 list, not a patch.

    If someone rebuilds the set from a narrower source, this fails rather than
    silently re-opening the exact hole.
    """
    assert "signed" in RESERVED_METAL_IDENTIFIERS
    for metal_specific in ("kernel", "device", "constant", "threadgroup", "sampler"):
        assert metal_specific in RESERVED_METAL_IDENTIFIERS


# --------------------------------------------------------------------------
# NEGATIVE CONTROLS — the guard must stay SILENT on correct code.
# --------------------------------------------------------------------------


def test_negative_control_the_actual_fix_is_silent() -> None:
    """`sdf` — tr6's rename — must not fire."""
    src = 'k = mx.fast.metal_kernel(input_names=["sdf", "direction", "reference", "eps"])\n'
    assert scan_source(src) == []


def test_negative_control_real_buffer_names_are_silent() -> None:
    """Names measured live in the repo's kernels must not be flagged."""
    live = ["x", "op", "cot", "wt", "dims", "conv", "stride", "pad", "dil",
            "groups", "inp", "scale", "frame", "target", "g1", "g0w",
            "class_mask", "starts", "counts", "source", "weight", "hidx"]
    src = f"k = mx.fast.metal_kernel(input_names={live!r})\n"
    assert scan_source(src) == [], "a false positive here trains people to override"


def test_negative_control_unrelated_calls_are_ignored() -> None:
    """Only the spliced kwargs matter; a dict key named `signed` is fine."""
    src = 'cfg = dict(other_names=["signed"])\nd = {"signed": 1}\n'
    assert scan_source(src) == []


def test_dynamic_name_lists_are_not_guessed_at() -> None:
    """A computed list is NOT flagged — a firing guess is a false positive."""
    src = "k = mx.fast.metal_kernel(input_names=[n for n in names])\n"
    assert scan_source(src) == []


# --------------------------------------------------------------------------
# WAIVER — and the trailing-comment trap that bit tr6 on this same family.
# --------------------------------------------------------------------------


def test_same_line_waiver_is_respected() -> None:
    src = (
        "k = mx.fast.metal_kernel(input_names=[\n"
        '    "signed",  # METAL_RESERVED_IDENTIFIER_OK: vendored upstream fixture\n'
        "])\n"
    )
    assert scan_source(src) == [], (
        "waiver invisible — this is the ast.get_source_segment trailing-comment "
        "bug ddm_tr6 hit while fixing this same family"
    )


def test_waiver_on_a_different_line_does_not_leak() -> None:
    """The waiver is same-LINE; it must not silence a neighbouring entry."""
    src = (
        "k = mx.fast.metal_kernel(input_names=[\n"
        '    "signed",  # METAL_RESERVED_IDENTIFIER_OK: intentional\n'
        '    "template",\n'
        "])\n"
    )
    assert [v.name for v in scan_source(src)] == ["template"]


# --------------------------------------------------------------------------
# LIVE STATE — count 0, and the DENOMINATOR is reported.
# --------------------------------------------------------------------------


def test_live_repo_is_clean_and_reports_its_denominator() -> None:
    violations, examined = check_metal_kernel_identifiers_not_reserved(
        REPO_ROOT, strict=False
    )
    assert violations == [], "\n".join(v.render() for v in violations)
    assert examined > 0, (
        "VACUOUS: examined 0 files. An empty scope is not a clean pass — this is "
        "the exact genus that let the defect live 20 days."
    )


def test_strict_mode_raises_with_the_count_in_the_message(tmp_path: Path) -> None:
    """Strict must refuse loudly AND state how many files it looked at.

    Uses a PLANTED fixture. The first version of this test assumed
    ``src/tac/tests`` already contained a violation — it does not, because the
    reserved words in this file live inside string literals, not calls. The
    code was right and the assertion was false; asserting against a fixture I
    control removes the guess.
    """
    pkg = tmp_path / "src" / "tac"
    pkg.mkdir(parents=True)
    (pkg / "planted.py").write_text(
        'k = mx.fast.metal_kernel(input_names=["signed"], output_names=["out"])\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as excinfo:
        check_metal_kernel_identifiers_not_reserved(
            tmp_path, strict=True, roots=("src/tac",)
        )
    msg = str(excinfo.value)
    assert "file(s) examined" in msg
    assert "signed" in msg


def test_scope_excludes_vendored_intake_clones(tmp_path: Path) -> None:
    """Vendored public-PR clones are forensic inputs — never flagged.

    MEASURED: the generic first cut swept 62,799 files (8,724 in intake clones)
    and timed out. CLAUDE.md forbids editing those trees at all, so a violation
    there is unactionable noise.
    """
    ours = tmp_path / "src" / "tac"
    ours.mkdir(parents=True)
    (ours / "mine.py").write_text('k = f(input_names=["sdf"])\n', encoding="utf-8")
    vendored = tmp_path / "src" / "tac" / "public_pr95_intake_20260504" / "source"
    vendored.mkdir(parents=True)
    (vendored / "theirs.py").write_text(
        'k = mx.fast.metal_kernel(input_names=["signed"])\n', encoding="utf-8"
    )
    violations, examined = check_metal_kernel_identifiers_not_reserved(
        tmp_path, strict=False, roots=("src/tac",)
    )
    assert violations == [], "vendored intake clone must not be flagged"
    assert examined == 1, f"expected only our 1 file in scope, examined {examined}"
