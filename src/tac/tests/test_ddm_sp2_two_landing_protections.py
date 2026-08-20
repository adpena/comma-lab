# SPDX-License-Identifier: MIT
"""Tests for the two ``ddm_sp2`` self-protection scope extensions (2026-08-19).

CLASS 1 -- environment-sensitive ZIP member metadata (``ddm_jg2`` S1i).
CLASS 2 -- GT decode-lineage objective custody (``ddm_pi2`` / ``qs1.GT_POSE``).

Both extensions are WARN-ONLY at landing. These tests pin the DETECTOR's
behaviour (what it catches, what it clears, what it refuses to clear) and the
FIX's byte-level claim -- not merely the presence of a marker, which is the
"tests verify constants not behavior" fake this repo forbids.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

from tac.preflight import (
    PreflightError,
    _check_351_gt_lineage_objective_custody,
    _gt_artifact_hits_outside_comment,
    _gt_lineage_waiver_rationale,
    _python_imports_gt_lineage_registry,
    _python_docstring_line_numbers,
    _scan_library_for_env_sensitive_zip_metadata,
    _scan_python_for_env_sensitive_zip_metadata,
    _write_mode_zipfile_call_lines,
    _zip_metadata_waiver_rationale,
    check_archive_builders_use_deterministic_zip,
    check_evidence_authority_claims_are_custodied,
    check_gt_lineage_objective_custody,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DETERMINISTIC_DATE = (1980, 1, 1, 0, 0, 0)


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


# ═══════════════════════════════════════════════════════════════════════════
# CLASS 1 -- the DEFECT itself, reproduced from first principles
# ═══════════════════════════════════════════════════════════════════════════


def _pack(*, create_system: int | None, external_attr: int | None) -> bytes:
    buf = io.BytesIO()
    info = zipfile.ZipInfo("0.bin", date_time=DETERMINISTIC_DATE)
    info.compress_type = zipfile.ZIP_STORED
    if create_system is not None:
        info.create_system = create_system
    if external_attr is not None:
        info.external_attr = external_attr
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as zf:
        zf.writestr(info, b"payload" * 64)
    return buf.getvalue()


def test_unset_external_attr_is_silently_substituted_by_cpython():
    """The mechanism: an unset external_attr does NOT emit 0."""
    packed = _pack(create_system=3, external_attr=None)
    with zipfile.ZipFile(io.BytesIO(packed)) as zf:
        info = zf.infolist()[0]
    assert info.external_attr == 0o600 << 16, (
        "CPython's ZipFile._open_to_write substitutes 0o600 << 16 for an unset "
        "external_attr; if this ever changes, the pin is what protects us."
    )


def test_metadata_defect_costs_zero_bytes_and_breaks_a_byte_seal():
    """The jg2 signature: EQUAL LENGTH, differing central-directory bytes."""
    a = _pack(create_system=3, external_attr=0o600 << 16)
    b = _pack(create_system=3, external_attr=0o100644 << 16)
    assert len(a) == len(b), "the defect must be invisible to any size check"
    differing = sum(x != y for x, y in zip(a, b))
    assert differing == 2, f"expected the 2 external_attr bytes, got {differing}"
    assert hashlib.sha256(a).hexdigest() != hashlib.sha256(b).hexdigest()


def test_pinning_the_already_emitted_value_is_byte_identical():
    """The FIX's core claim: pinning what the builder already emits changes nothing."""
    assert _pack(create_system=3, external_attr=None) == _pack(
        create_system=3, external_attr=0o600 << 16,
    )
    assert _pack(create_system=None, external_attr=0o100644 << 16) == _pack(
        create_system=3, external_attr=0o100644 << 16,
    )


# ═══════════════════════════════════════════════════════════════════════════
# CLASS 1 -- write-mode detection
# ═══════════════════════════════════════════════════════════════════════════


def test_write_mode_detected_positionally():
    import ast

    tree = ast.parse('import zipfile\nzipfile.ZipFile(p, "w")\n')
    assert _write_mode_zipfile_call_lines(tree) == [2]


def test_write_mode_detected_as_keyword():
    import ast

    tree = ast.parse('import zipfile\nzipfile.ZipFile(p, mode="w")\n')
    assert _write_mode_zipfile_call_lines(tree) == [2]


def test_append_and_exclusive_modes_are_write_modes():
    import ast

    tree = ast.parse(
        'import zipfile\nzipfile.ZipFile(p, "a")\nzipfile.ZipFile(q, mode="x")\n'
    )
    assert _write_mode_zipfile_call_lines(tree) == [2, 3]


def test_read_mode_is_not_a_write_mode():
    import ast

    tree = ast.parse('import zipfile\nzipfile.ZipFile(p, "r")\nzipfile.ZipFile(q)\n')
    assert _write_mode_zipfile_call_lines(tree) == []


# ═══════════════════════════════════════════════════════════════════════════
# CLASS 1 -- scanner behaviour
# ═══════════════════════════════════════════════════════════════════════════


def test_bare_writestr_builder_is_flagged(tmp_path):
    path = _write(
        tmp_path,
        "builder.py",
        'import zipfile\n\n\ndef build(p):\n    with zipfile.ZipFile(p, "w") as zf:\n'
        '        zf.writestr("a.bin", b"x")\n',
    )
    found = _scan_python_for_env_sensitive_zip_metadata(path, tmp_path)
    assert len(found) == 1
    assert "builder.py:5" in found[0]


def test_file_pinning_both_fields_is_clear(tmp_path):
    path = _write(
        tmp_path,
        "pinned.py",
        'import zipfile\n\n\ndef build(p):\n    i = zipfile.ZipInfo("a.bin")\n'
        "    i.create_system = 3\n    i.external_attr = 0o600 << 16\n"
        '    with zipfile.ZipFile(p, "w") as zf:\n        zf.writestr(i, b"x")\n',
    )
    assert _scan_python_for_env_sensitive_zip_metadata(path, tmp_path) == []


def test_half_pin_create_system_only_is_flagged_and_names_the_gap(tmp_path):
    path = _write(
        tmp_path,
        "half_a.py",
        'import zipfile\n\n\ndef build(p):\n    i = zipfile.ZipInfo("a.bin")\n'
        "    i.create_system = 3\n"
        '    with zipfile.ZipFile(p, "w") as zf:\n        zf.writestr(i, b"x")\n',
    )
    found = _scan_python_for_env_sensitive_zip_metadata(path, tmp_path)
    assert len(found) == 1
    assert "external_attr" in found[0] and "create_system +" not in found[0]


def test_half_pin_external_attr_only_is_flagged_and_names_the_gap(tmp_path):
    path = _write(
        tmp_path,
        "half_b.py",
        'import zipfile\n\n\ndef build(p):\n    i = zipfile.ZipInfo("a.bin")\n'
        "    i.external_attr = 0o600 << 16\n"
        '    with zipfile.ZipFile(p, "w") as zf:\n        zf.writestr(i, b"x")\n',
    )
    found = _scan_python_for_env_sensitive_zip_metadata(path, tmp_path)
    assert len(found) == 1
    assert "create_system" in found[0] and "external_attr" not in found[0]


def test_canonical_helper_route_is_clear(tmp_path):
    """archive_optimizer.py / pr79 are correct THIS way; a token grep misses it."""
    path = _write(
        tmp_path,
        "helper_route.py",
        "import zipfile\nfrom tac.submission_archive import write_deterministic_zip_member\n\n\n"
        'def build(p):\n    with zipfile.ZipFile(p, "w") as zf:\n'
        '        write_deterministic_zip_member(zf, "a.bin", b"x")\n',
    )
    assert _scan_python_for_env_sensitive_zip_metadata(path, tmp_path) == []


def test_read_only_consumer_is_clear(tmp_path):
    path = _write(
        tmp_path,
        "reader.py",
        'import zipfile\n\n\ndef read(p):\n    with zipfile.ZipFile(p, "r") as zf:\n'
        '        return zf.read("a.bin")\n',
    )
    assert _scan_python_for_env_sensitive_zip_metadata(path, tmp_path) == []


def test_file_without_zipfile_is_clear(tmp_path):
    path = _write(tmp_path, "plain.py", "x = 1\n")
    assert _scan_python_for_env_sensitive_zip_metadata(path, tmp_path) == []


def test_syntax_error_does_not_crash_the_scanner(tmp_path):
    path = _write(tmp_path, "broken.py", 'import zipfile\ndef (:\n  ZipFile("w"\n')
    assert _scan_python_for_env_sensitive_zip_metadata(path, tmp_path) == []


def test_missing_file_does_not_crash_the_scanner(tmp_path):
    assert _scan_python_for_env_sensitive_zip_metadata(tmp_path / "gone.py", tmp_path) == []


# ═══════════════════════════════════════════════════════════════════════════
# CLASS 1 -- waiver discipline
# ═══════════════════════════════════════════════════════════════════════════


def test_zip_waiver_with_substantive_rationale_clears_the_site(tmp_path):
    path = _write(
        tmp_path,
        "waived.py",
        'import zipfile\n\n\ndef build(p):\n    with zipfile.ZipFile(p, "w") as zf:  '
        "# ZIP_METADATA_ENV_OK: metadata cloned verbatim from the source archive\n"
        '        zf.writestr("a.bin", b"x")\n',
    )
    assert _scan_python_for_env_sensitive_zip_metadata(path, tmp_path) == []


def test_zip_waiver_placeholder_rationale_is_rejected():
    assert _zip_metadata_waiver_rationale("x  # ZIP_METADATA_ENV_OK:<rationale>") is None
    assert _zip_metadata_waiver_rationale("x  # ZIP_METADATA_ENV_OK: <reason>") is None
    assert _zip_metadata_waiver_rationale("x  # ZIP_METADATA_ENV_OK: TBD") is None
    assert _zip_metadata_waiver_rationale("x  # ZIP_METADATA_ENV_OK:") is None


def test_zip_waiver_too_short_rationale_is_rejected():
    assert _zip_metadata_waiver_rationale("x  # ZIP_METADATA_ENV_OK: nope") is None


def test_zip_waiver_substantive_rationale_is_accepted():
    got = _zip_metadata_waiver_rationale(
        "x  # ZIP_METADATA_ENV_OK: cloned from the source member's own ZipInfo"
    )
    assert got is not None and "cloned" in got


def test_zip_waiver_on_a_different_line_does_not_clear_the_call(tmp_path):
    path = _write(
        tmp_path,
        "misplaced.py",
        "import zipfile\n# ZIP_METADATA_ENV_OK: a rationale parked on the wrong line\n"
        'def build(p):\n    with zipfile.ZipFile(p, "w") as zf:\n'
        '        zf.writestr("a.bin", b"x")\n',
    )
    assert len(_scan_python_for_env_sensitive_zip_metadata(path, tmp_path)) == 1


# ═══════════════════════════════════════════════════════════════════════════
# CLASS 1 -- live-population regression over the shipping library
# ═══════════════════════════════════════════════════════════════════════════

_SP2_FIXED_FILES = (
    "src/tac/archive_codec.py",
    "src/tac/archive_diet.py",
    "src/tac/entropy_archive.py",
    "src/tac/hnerv_lowlevel_packer.py",
    "src/tac/mask_prior.py",
    "src/tac/optimal_stack_orchestrator.py",
    "src/tac/pr95_hnerv.py",
    "src/tac/stack_compositions.py",
)


def test_the_eight_fixed_library_builders_stay_pinned():
    """Regression: the ddm_sp2 fix must not silently rot back out."""
    findings = _scan_library_for_env_sensitive_zip_metadata(REPO_ROOT, verbose=False)
    for rel in _SP2_FIXED_FILES:
        assert not [f for f in findings if f.startswith(f"{rel}:")], (
            f"{rel} lost its ZIP metadata pin"
        )


def test_positive_control_the_jg2_cure_satisfies_the_detector():
    """POSITIVE CONTROL: the detector must agree with the known cure.

    ``ddm_jg2`` fixed the defect by pinning ``SHIPPED_ZIP_CREATE_SYSTEM`` /
    ``SHIPPED_ZIP_EXTERNAL_ATTR`` / ``SHIPPED_ZIP_DATE_TIME``. If this scanner
    flagged the CURED file, the scanner would be wrong, not the file. (jg2 lives
    in ``experiments/``, outside the library scan scope, so this exercises the
    per-file scanner directly.)
    """
    jg2 = REPO_ROOT / "experiments/ddm_jg2_tail_reencode.py"
    if not jg2.exists():  # pragma: no cover - file is expected to exist
        return
    assert _scan_python_for_env_sensitive_zip_metadata(jg2, REPO_ROOT) == []


def test_positive_control_a_jg2_shaped_prefix_builder_is_caught(tmp_path):
    """The mirror control: jg2's PRE-fix shape must be caught.

    A fixed timestamp alone was never enough -- that is precisely the state that
    produced an archive of exactly the right length with the wrong sha.
    """
    path = _write(
        tmp_path,
        "prefix_reencode.py",
        "import zipfile\n\nSHIPPED_ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)\n\n\n"
        "def repack(p, payload):\n"
        '    info = zipfile.ZipInfo("p", date_time=SHIPPED_ZIP_DATE_TIME)\n'
        "    info.compress_type = zipfile.ZIP_STORED\n"
        '    with zipfile.ZipFile(p, "w") as zf:\n'
        "        zf.writestr(info, payload)\n",
    )
    found = _scan_python_for_env_sensitive_zip_metadata(path, tmp_path)
    assert len(found) == 1
    assert "create_system + external_attr" in found[0]


def test_zip_metadata_extension_is_warn_only_not_raising():
    """The extension must never raise, even though the host gate runs strict."""
    findings = _scan_library_for_env_sensitive_zip_metadata(REPO_ROOT, verbose=False)
    returned = check_archive_builders_use_deterministic_zip(strict=True, verbose=False)
    assert len(findings) >= 1, "warn-only population is expected to be non-empty"
    for item in findings:
        assert item in returned, "extension findings must reach the caller"


# ═══════════════════════════════════════════════════════════════════════════
# CLASS 2 -- GT decode-lineage objective custody
# ═══════════════════════════════════════════════════════════════════════════


def test_pyav_pose_table_load_is_flagged(tmp_path):
    _write(
        tmp_path,
        "tools/solve.py",
        'import numpy as np\nGT = np.load("pose/gt_first6_n600.npy")\n',
    )
    found = _check_351_gt_lineage_objective_custody(tmp_path)
    assert len(found) == 1 and "tools/solve.py:2" in found[0]


def test_dali_named_artifact_declares_itself(tmp_path):
    _write(
        tmp_path,
        "tools/solve.py",
        'import numpy as np\nGT = np.load("pose/gt_first6_dali_n600.npy")\n',
    )
    assert _check_351_gt_lineage_objective_custody(tmp_path) == []


def test_canonical_gt_lineage_route_clears_the_file(tmp_path):
    _write(
        tmp_path,
        "tools/solve.py",
        "import numpy as np\nfrom tac.gt_lineage import assert_gt_lineage\n"
        'P = "pose/gt_first6_n600.npy"\nassert_gt_lineage(P, expected="PYAV")\n'
        "GT = np.load(P)\n",
    )
    assert _check_351_gt_lineage_objective_custody(tmp_path) == []


def test_a_local_verify_gt_lineage_does_NOT_clear_the_file(tmp_path):
    """The precision that matters: up2's filename-substring resolver is not the registry."""
    _write(
        tmp_path,
        "tools/solve.py",
        "import numpy as np\n\n\ndef verify_gt_lineage(path):\n"
        '    return "dali" in str(path)\n\n\n'
        'GT = np.load("caches/gt_cache_av.pt")\n',
    )
    found = _check_351_gt_lineage_objective_custody(tmp_path)
    assert len(found) == 1, "a local same-named helper must not launder lineage custody"


def test_gt_cache_pt_artifacts_are_in_scope(tmp_path):
    _write(tmp_path, "tools/a.py", 'P = "x/gt_cache_600_official_ada.pt"\n')
    assert len(_check_351_gt_lineage_objective_custody(tmp_path)) == 1


def test_gt_argmax_is_deliberately_out_of_scope(tmp_path):
    """Measured: including it takes the population 18 -> 93 files for one settled lineage."""
    _write(tmp_path, "tools/a.py", 'P = "x/gt_argmax_n600.npy"\n')
    assert _check_351_gt_lineage_objective_custody(tmp_path) == []


def test_comment_only_mention_is_not_a_consumption(tmp_path):
    """The mt1 false positive: a comment naming PyAV above a DALI load."""
    _write(
        tmp_path,
        "tools/a.py",
        "# gt_first6_n600.npy is the PyAV/advisory lineage (gap C = 1.406151e-04).\n"
        'P = "x/gt_first6_dali_n600.npy"\n',
    )
    assert _check_351_gt_lineage_objective_custody(tmp_path) == []


def test_docstring_artifact_mention_is_not_a_consumption(tmp_path):
    """The ddm_cpu1 false positive: prose describes the basename collision."""
    path = _write(
        tmp_path,
        "experiments/a.py",
        '"""The name gt_first6_n600.npy exists under opposite lineages."""\nVALUE = 1\n',
    )
    assert _python_docstring_line_numbers(path.read_text()) == frozenset({1})
    assert _check_351_gt_lineage_objective_custody(tmp_path) == []


def test_executable_read_after_same_line_docstring_is_still_flagged(tmp_path):
    """rv16 red control: docstring masking must not discard later code."""
    _write(
        tmp_path,
        "experiments/a.py",
        'def load(): """gt_first6_n600.npy is prose"""; P = "gt_first6_n600.npy"; return P\n',
    )
    found = _check_351_gt_lineage_objective_custody(tmp_path)
    assert len(found) == 1 and "experiments/a.py:1" in found[0]


def test_canonical_route_named_only_in_docstring_does_not_clear_read(tmp_path):
    _write(
        tmp_path,
        "tools/a.py",
        '"""Future work may route through tac.gt_lineage."""\nP = "gt_first6_n600.npy"\n',
    )
    found = _check_351_gt_lineage_objective_custody(tmp_path)
    assert len(found) == 1 and "tools/a.py:2" in found[0]


def test_canonical_route_named_only_in_comment_does_not_clear_read(tmp_path):
    _write(
        tmp_path,
        "tools/a.py",
        '# future: from tac import gt_lineage\nP = "gt_first6_n600.npy"\n',
    )
    found = _check_351_gt_lineage_objective_custody(tmp_path)
    assert len(found) == 1 and "tools/a.py:2" in found[0]


def test_registry_import_detection_is_ast_owned_not_substring_owned():
    assert _python_imports_gt_lineage_registry("from tac import gt_lineage\n")
    assert _python_imports_gt_lineage_registry("from tac.gt_lineage import assert_gt_lineage\n")
    assert not _python_imports_gt_lineage_registry("# from tac import gt_lineage\n")
    assert not _python_imports_gt_lineage_registry('NOTE = "tac.gt_lineage"\n')


def test_artifact_before_a_trailing_comment_is_still_a_consumption():
    hits = _gt_artifact_hits_outside_comment('P = "gt_first6_n600.npy"  # note')
    assert hits == ["gt_first6_n600.npy"]


def test_artifact_only_inside_a_trailing_comment_is_ignored():
    assert _gt_artifact_hits_outside_comment("P = Q  # see gt_first6_n600.npy") == []


def test_gt_waiver_with_substantive_rationale_clears_the_site(tmp_path):
    _write(
        tmp_path,
        "tools/a.py",
        'P = "x/gt_first6_n600.npy"  # GT_LINEAGE_OK: round-local custody, this '
        "round emitted its own GT\n",
    )
    assert _check_351_gt_lineage_objective_custody(tmp_path) == []


def test_gt_waiver_placeholder_rationale_is_rejected():
    assert _gt_lineage_waiver_rationale("x  # GT_LINEAGE_OK:<rationale>") is None
    assert _gt_lineage_waiver_rationale("x  # GT_LINEAGE_OK: pending") is None
    assert _gt_lineage_waiver_rationale("x  # GT_LINEAGE_OK:") is None


def test_gt_waiver_too_short_rationale_is_rejected():
    assert _gt_lineage_waiver_rationale("x  # GT_LINEAGE_OK: yes") is None


def test_gt_waiver_substantive_rationale_is_accepted():
    got = _gt_lineage_waiver_rationale(
        "x  # GT_LINEAGE_OK: this is the producer that builds both caches"
    )
    assert got is not None and "producer" in got


def test_tests_and_results_trees_are_out_of_scope(tmp_path):
    _write(tmp_path, "src/tac/tests/test_x.py", 'P = "gt_first6_n600.npy"\n')
    _write(tmp_path, "experiments/results/run/x.py", 'P = "gt_first6_n600.npy"\n')
    _write(tmp_path, "tools/test_helper.py", 'P = "gt_first6_n600.npy"\n')
    assert _check_351_gt_lineage_objective_custody(tmp_path) == []


def test_the_registry_module_itself_is_out_of_scope(tmp_path):
    _write(tmp_path, "src/tac/gt_lineage.py", 'P = "gt_first6_n600.npy"\n')
    assert _check_351_gt_lineage_objective_custody(tmp_path) == []


_SP2_GT_WAIVED_FILES = (
    "experiments/ddm_po1_t4_error_feedback_pose_compensation.py",
    "experiments/ddm_qs1_frame0_schur_coupled_solve.py",
    "experiments/ddm_pi2_pose_axis_attribution.py",
    "experiments/ddm_sg2_pr130_seg_axis_source_audit.py",
    "experiments/modal_dali_av_gt_cache_diff.py",
)


def test_adjudicated_lineage_correct_sites_stay_cleared():
    """Regression over the five sites ddm_sp2 adjudicated and waived."""
    findings = _check_351_gt_lineage_objective_custody(REPO_ROOT)
    for rel in _SP2_GT_WAIVED_FILES:
        assert not [f for f in findings if f.startswith(f"{rel}:")], (
            f"{rel} lost its adjudicated GT-lineage waiver"
        )


def test_qs1_pose_objective_is_the_dali_table():
    """The FIXED defect: qs1's solve objective must be the DALI lineage."""
    text = (REPO_ROOT / "experiments/ddm_qs1_frame0_schur_coupled_solve.py").read_text()
    assert 'GT_POSE: Final = CP135_BASE_POSE.with_name("gt_first6_dali_n600.npy")' in text


def test_gt_lineage_extension_host_is_warn_only_but_standalone_refuses(tmp_path):
    """The aggregate host warns while the explicit strict surface refuses.

    The host gate carries OTHER strict surfaces (#344 anchor roundtrip, #351
    producer identity) that may legitimately be red for unrelated reasons. The
    two-landing contract keeps GT findings out of that raise path, while exposing
    a strict standalone path for an executed red positive control.
    """
    _write(tmp_path, "tools/undeclared.py", 'P = "gt_first6_n600.npy"\n')
    findings = _check_351_gt_lineage_objective_custody(tmp_path)
    assert len(findings) == 1

    returned = check_evidence_authority_claims_are_custodied(
        repo_root=tmp_path, strict=True, verbose=False
    )
    for item in findings:
        assert item in returned, "warn-only extension findings must reach the host caller"

    try:
        check_gt_lineage_objective_custody(tmp_path, strict=True, verbose=False)
    except PreflightError as exc:
        assert "tools/undeclared.py:1" in str(exc)
    else:
        raise AssertionError("standalone strict GT-lineage gate accepted its red control")


# ═══════════════════════════════════════════════════════════════════════════
# CLASS 3 -- fp16 cast destroys its own floor (ddm_fx4's owed gate, folded
# here as a Catalog #161 scope extension while ddm_sp2 owns preflight.py)
# ═══════════════════════════════════════════════════════════════════════════

from tac.fp16_floor_guard import (  # noqa: E402
    FP16_MIN_POSITIVE,
    floor_is_fp16_safe,
    scan_repo_for_fp16_destroyed_floors,
    scan_text_for_fp16_destroyed_floors,
)
from tac.preflight import (  # noqa: E402
    check_quantize_degenerate_range_clamped_correctly,
)


def test_the_canonical_pre_fix_expression_is_caught():
    src = "x = ((maxs - mins).float() / 255.0).clamp(min=1e-8).to(torch.float16)\n"
    assert len(scan_text_for_fp16_destroyed_floors(src, "m.py")) == 1


def test_the_cured_form_floor_after_the_cast_is_clear():
    src = (
        "x = ((maxs - mins).float() / 255.0).to(torch.float16)"
        ".clamp(min=_FP16_MIN_POSITIVE)\n"
    )
    assert scan_text_for_fp16_destroyed_floors(src, "m.py") == []


def test_the_cross_statement_form_is_caught():
    """A same-statement-only detector would see half the class."""
    src = (
        "scale = max(max_abs, 1e-8) / 127.0\n"
        "scale_fp16 = torch.tensor([scale], torch.float16)\n"
    )
    assert len(scan_text_for_fp16_destroyed_floors(src, "m.py")) == 1


def test_a_floor_at_or_above_the_fp16_subnormal_is_safe():
    src = "x = y.clamp(min=5.960464477539063e-08).to(torch.float16)\n"
    assert scan_text_for_fp16_destroyed_floors(src, "m.py") == []


def test_half_call_form_is_detected():
    src = "x = y.clamp_min(1e-12).half()\n"
    assert len(scan_text_for_fp16_destroyed_floors(src, "m.py")) == 1


def test_a_site_with_no_floor_is_not_flagged():
    """No guard was intended there; the gate refuses unfalsifiable rows."""
    src = "x = y.to(torch.float16)\n"
    assert scan_text_for_fp16_destroyed_floors(src, "m.py") == []


def test_a_non_literal_floor_is_treated_as_safe():
    """Report only what can be PROVEN statically."""
    assert floor_is_fp16_safe("some_variable") is True


def test_floor_safety_boundary_is_pinned_one_ulp_either_side():
    assert floor_is_fp16_safe(repr(FP16_MIN_POSITIVE)) is True
    assert floor_is_fp16_safe("1e-08") is False
    assert floor_is_fp16_safe("_FP16_MIN_POSITIVE") is True


def test_a_docstring_quoting_the_pre_fix_expression_is_not_a_violation():
    """The ddm_fx3 blindness, cured: prose is documentation, not code."""
    src = (
        '"""Example of the bug:\n\n'
        "    x = y.clamp(min=1e-8).to(torch.float16)\n"
        '"""\n'
        "z = 1\n"
    )
    assert scan_text_for_fp16_destroyed_floors(src, "m.py") == []


def test_a_trailing_comment_quoting_the_pattern_is_not_a_violation():
    src = "z = 1  # x = y.clamp(min=1e-8).to(torch.float16)\n"
    assert scan_text_for_fp16_destroyed_floors(src, "m.py") == []


def test_the_live_class_is_clean_and_the_sweep_is_not_vacuous():
    violations, scanned = scan_repo_for_fp16_destroyed_floors(REPO_ROOT)
    assert scanned > 0, "a sweep that measured nothing must not read as green"
    assert violations == [], violations


def test_negative_control_reintroducing_the_bug_is_caught(tmp_path):
    """The detector must FAIL on re-introduction, or it proves nothing."""
    _write(
        tmp_path,
        "src/tac/offender.py",
        "import torch\n\n\ndef pack(maxs, mins):\n"
        "    return ((maxs - mins).float() / 255.0).clamp(min=1e-8).to(torch.float16)\n",
    )
    violations, scanned = scan_repo_for_fp16_destroyed_floors(tmp_path)
    assert scanned == 1
    assert len(violations) == 1 and "src/tac/offender.py" in violations[0]


def test_the_gate_raises_strict_when_the_class_is_reintroduced(tmp_path):
    (tmp_path / "src" / "tac" / "substrates").mkdir(parents=True)
    _write(
        tmp_path,
        "tools/offender.py",
        "import torch\n\n\ndef pack(x):\n"
        "    return x.clamp_min(1e-12).to(torch.float16)\n",
    )
    try:
        check_quantize_degenerate_range_clamped_correctly(
            repo_root=tmp_path, strict=True, verbose=False,
        )
    except PreflightError as exc:
        assert "fp16" in str(exc) and "2**-24" in str(exc)
    else:  # pragma: no cover - the gate must refuse
        raise AssertionError("strict gate did not refuse a reintroduced fp16 defect")


def test_the_gate_is_clean_on_the_live_repo_under_strict():
    out = check_quantize_degenerate_range_clamped_correctly(
        repo_root=REPO_ROOT, strict=True, verbose=False,
    )
    assert out == [], out
