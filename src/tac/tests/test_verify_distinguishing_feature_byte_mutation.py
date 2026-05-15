# SPDX-License-Identifier: MIT
"""Tests for tools/verify_distinguishing_feature_byte_mutation.py

Per Catalog #272 + CLAUDE.md "Substrate scaffolds MUST be COMPLETE or
RESEARCH-ONLY" non-negotiable. This canonical helper provides the
runtime PROOF for the `byte_mutation_smoke_passes` contract field.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest


# Load the canonical helper as a module (it lives under tools/, not src/).
def _load_helper():
    repo_root = Path(__file__).resolve().parents[3]
    helper_path = repo_root / "tools" / "verify_distinguishing_feature_byte_mutation.py"
    spec = importlib.util.spec_from_file_location("dfic_helper", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load helper from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["dfic_helper"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def helper():
    return _load_helper()


def _make_archive(path: Path, sections: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in sections.items():
            zf.writestr(name, data)


def _make_simple_inflate_sh(path: Path, payload_section: str, output_filename: str = "0.mkv") -> None:
    """Write a fake inflate.sh that copies the named ZIP section to output.

    The output BYTES depend on the section contents — so mutating the
    section will produce different output bytes (smoke passes).
    """
    body = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'ARCHIVE_DIR="$1"\n'
        'OUTPUT_DIR="$2"\n'
        'FILE_LIST="$3"\n'
        'mkdir -p "$OUTPUT_DIR"\n'
        f'ZIP_FILE=$(ls "$ARCHIVE_DIR"/*.zip | head -n1)\n'
        f'unzip -p "$ZIP_FILE" {payload_section} > "$OUTPUT_DIR/{output_filename}"\n'
    )
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _make_dead_section_inflate_sh(path: Path, output_filename: str = "0.mkv") -> None:
    """Inflate.sh that ALWAYS writes the same fixed bytes (ignoring archive).

    Any section mutation produces the SAME output — the section is dead
    in the inflate path. This is the Z3-G1 anchor pattern.
    """
    body = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'ARCHIVE_DIR="$1"\n'
        'OUTPUT_DIR="$2"\n'
        'mkdir -p "$OUTPUT_DIR"\n'
        f'echo -n "FIXED_DEAD_OUTPUT_BYTES" > "$OUTPUT_DIR/{output_filename}"\n'
    )
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _make_nondeterministic_inflate_sh(
    path: Path,
    output_filename: str = "0.mkv",
) -> None:
    """Inflate.sh whose output changes across identical baseline invocations."""
    body = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'ARCHIVE_DIR="$1"\n'
        'OUTPUT_DIR="$2"\n'
        'COUNTER_FILE="$(dirname "$ARCHIVE_DIR")/nondeterministic_counter.txt"\n'
        "COUNT=0\n"
        'if [[ -f "$COUNTER_FILE" ]]; then COUNT="$(cat "$COUNTER_FILE")"; fi\n'
        "COUNT=$((COUNT + 1))\n"
        'echo "$COUNT" > "$COUNTER_FILE"\n'
        'mkdir -p "$OUTPUT_DIR"\n'
        f'printf "NONDET_%s" "$COUNT" > "$OUTPUT_DIR/{output_filename}"\n'
    )
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


# ---------------------------------------------------------------------------
# Unit tests on internal helpers
# ---------------------------------------------------------------------------


def test_sha256_file_deterministic(helper, tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello world")
    assert helper._sha256_file(p) == helper._sha256_file(p)


def test_sha256_bytes_known_value(helper):
    h = helper._sha256_bytes(b"")
    assert h == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_list_archive_sections(helper, tmp_path):
    archive = tmp_path / "a.zip"
    _make_archive(archive, {"sec_a": b"AAAA", "sec_b": b"BBBBBBBB"})
    sections = helper._list_archive_sections(archive)
    section_names = {s[0] for s in sections}
    assert section_names == {"sec_a", "sec_b"}


def test_read_archive_section(helper, tmp_path):
    archive = tmp_path / "a.zip"
    _make_archive(archive, {"x": b"hello"})
    assert helper._read_archive_section(archive, "x") == b"hello"


def test_write_archive_with_mutated_section_replaces_target_only(helper, tmp_path):
    src = tmp_path / "src.zip"
    dst = tmp_path / "dst.zip"
    _make_archive(src, {"a": b"AAAA", "b": b"BBBB"})
    helper._write_archive_with_mutated_section(src, dst, "a", b"XXXX")
    assert helper._read_archive_section(dst, "a") == b"XXXX"
    assert helper._read_archive_section(dst, "b") == b"BBBB"


def test_generate_mutations_n_distinct(helper):
    section = b"hello world this is a test string with some bytes"
    muts = helper._generate_mutations(section, 4, seed_offset=0)
    assert len(muts) == 4
    # All mutations should be distinct from the original.
    for m in muts:
        assert m != section
        assert len(m) == len(section)


def test_generate_mutations_empty_section_returns_empty(helper):
    assert helper._generate_mutations(b"", 4) == []


def test_generate_mutations_n_capped_to_size(helper):
    section = b"abc"
    muts = helper._generate_mutations(section, 10)
    assert len(muts) == 3


def test_parse_byte_range_target_with_label(helper):
    target = helper._parse_byte_range_target("decoder=x@4:16")
    assert target.label == "decoder"
    assert target.member == "x"
    assert target.offset == 4
    assert target.length == 16
    assert target.target_basis == "member_byte_range"


def test_hash_directory_contents_deterministic(helper, tmp_path):
    d = tmp_path / "dir"
    d.mkdir()
    (d / "x.txt").write_text("hello", encoding="utf-8")
    (d / "y.txt").write_text("world", encoding="utf-8")
    h1 = helper._hash_directory_contents(d)
    h2 = helper._hash_directory_contents(d)
    assert h1 == h2 and len(h1) == 64


def test_hash_directory_contents_changes_with_file_change(helper, tmp_path):
    d = tmp_path / "dir"
    d.mkdir()
    (d / "x.txt").write_text("hello", encoding="utf-8")
    h1 = helper._hash_directory_contents(d)
    (d / "x.txt").write_text("HELLO", encoding="utf-8")
    h2 = helper._hash_directory_contents(d)
    assert h1 != h2


def test_hash_directory_contents_empty_directory(helper, tmp_path):
    assert helper._hash_directory_contents(tmp_path) == ""


# ---------------------------------------------------------------------------
# End-to-end: PASSED case
# ---------------------------------------------------------------------------


def test_verify_passed_when_section_is_consumed(helper, tmp_path):
    """Inflate.sh that copies the section into the output → mutations
    produce different outputs → smoke passes."""
    archive = tmp_path / "archive.zip"
    _make_archive(archive, {"distinguishing": b"DISTINGUISHING_BYTES_HERE"})
    inflate_sh = tmp_path / "inflate.sh"
    _make_simple_inflate_sh(inflate_sh, "distinguishing")
    output_json = tmp_path / "out.json"

    result = helper.verify_distinguishing_feature_byte_mutation(
        archive=archive,
        inflate_sh=inflate_sh,
        distinguishing_bytes_paths=["distinguishing"],
        output_json=output_json,
    )
    assert result["verdict"] == "PASSED"
    assert result["overall_passed"] is True
    assert len(result["section_results"]) == 1
    sr = result["section_results"][0]
    assert sr["section"] == "distinguishing"
    assert sr["target_basis"] == "zip_member"
    assert sr["member"] == "distinguishing"
    assert sr["mutations_changed_output"] > 0
    assert sr["passed"] is True
    assert result["baseline_repeat_deterministic"] is True
    assert result["baseline_inflated_output_sha256"] == result["baseline_repeat_inflated_output_sha256"]


def test_verify_passed_when_member_byte_range_is_consumed(helper, tmp_path):
    """Parser-section/member-offset proof: mutate only a byte range inside
    the single ZIP member, not the whole member."""
    archive = tmp_path / "archive.zip"
    _make_archive(archive, {"x": b"HEADER_DISTINGUISHING_PAYLOAD_TAIL"})
    inflate_sh = tmp_path / "inflate.sh"
    _make_simple_inflate_sh(inflate_sh, "x")
    output_json = tmp_path / "out.json"

    result = helper.verify_distinguishing_feature_byte_mutation(
        archive=archive,
        inflate_sh=inflate_sh,
        distinguishing_bytes_paths=[],
        distinguishing_byte_ranges=["distinguishing_section=x@7:22"],
        output_json=output_json,
    )
    assert result["verdict"] == "PASSED"
    sr = result["section_results"][0]
    assert sr["section"] == "distinguishing_section"
    assert sr["target_basis"] == "member_byte_range"
    assert sr["member"] == "x"
    assert sr["offset"] == 7
    assert sr["length"] == 22
    assert sr["mutations_changed_output"] > 0


# ---------------------------------------------------------------------------
# End-to-end: FAILED case (Z3-G1 anchor)
# ---------------------------------------------------------------------------


def test_verify_failed_when_section_is_dead_z3_g1_anchor(helper, tmp_path):
    """Z3-G1 anchor pattern: archive ships a 'distinguishing' section but
    inflate.sh ignores it (always outputs the same fixed bytes). Smoke
    MUST FAIL — the bytes are not operationally consumed."""
    archive = tmp_path / "archive.zip"
    _make_archive(archive, {"distinguishing": b"NEVER_READ_BY_INFLATE_BYTES"})
    inflate_sh = tmp_path / "inflate.sh"
    _make_dead_section_inflate_sh(inflate_sh)
    output_json = tmp_path / "out.json"

    result = helper.verify_distinguishing_feature_byte_mutation(
        archive=archive,
        inflate_sh=inflate_sh,
        distinguishing_bytes_paths=["distinguishing"],
        output_json=output_json,
    )
    assert result["verdict"] == "FAILED"
    assert result["overall_passed"] is False
    sr = result["section_results"][0]
    assert sr["mutations_changed_output"] == 0
    assert sr["passed"] is False


def test_verify_infra_error_when_baseline_inflate_is_nondeterministic(
    helper,
    tmp_path,
):
    """Mutation differences are not evidence until identical baselines match."""
    archive = tmp_path / "archive.zip"
    _make_archive(archive, {"distinguishing": b"DISTINGUISHING_BYTES_HERE"})
    inflate_sh = tmp_path / "inflate.sh"
    _make_nondeterministic_inflate_sh(inflate_sh)
    output_json = tmp_path / "out.json"

    result = helper.verify_distinguishing_feature_byte_mutation(
        archive=archive,
        inflate_sh=inflate_sh,
        distinguishing_bytes_paths=["distinguishing"],
        output_json=output_json,
    )

    assert result["verdict"] == "INFRASTRUCTURE_ERROR"
    assert result["baseline_repeat_deterministic"] is False
    assert "nondeterministic" in result["infrastructure_error_reason"]
    assert result["baseline_inflated_output_sha256"] != result["baseline_repeat_inflated_output_sha256"]


# ---------------------------------------------------------------------------
# Empty-section anchor: Z3-G1 SPECIFIC pattern (b"")
# ---------------------------------------------------------------------------


def test_verify_failed_when_distinguishing_section_is_empty(helper, tmp_path):
    """The CANONICAL Z3-G1 anchor: hyperprior_weights_int8 = b"". An
    empty section IS a violation — the smart thing has zero bytes."""
    archive = tmp_path / "archive.zip"
    _make_archive(archive, {"distinguishing": b"", "renderer": b"FILLER"})
    inflate_sh = tmp_path / "inflate.sh"
    _make_simple_inflate_sh(inflate_sh, "renderer")  # ignore distinguishing
    output_json = tmp_path / "out.json"

    result = helper.verify_distinguishing_feature_byte_mutation(
        archive=archive,
        inflate_sh=inflate_sh,
        distinguishing_bytes_paths=["distinguishing"],
        output_json=output_json,
    )
    assert result["verdict"] == "FAILED"
    sr = result["section_results"][0]
    assert sr["section_size_bytes"] == 0
    assert sr["mutations_attempted"] == 0
    assert sr["passed"] is False


# ---------------------------------------------------------------------------
# Multi-section: per-section verdicts
# ---------------------------------------------------------------------------


def test_verify_mixed_passed_and_failed_sections(helper, tmp_path):
    """One section consumed, one ignored — overall verdict FAILED."""
    archive = tmp_path / "archive.zip"
    _make_archive(
        archive,
        {
            "consumed": b"REAL_CONSUMED_BYTES",
            "ignored": b"NEVER_READ_PADDING",
        },
    )
    inflate_sh = tmp_path / "inflate.sh"
    _make_simple_inflate_sh(inflate_sh, "consumed")
    output_json = tmp_path / "out.json"

    result = helper.verify_distinguishing_feature_byte_mutation(
        archive=archive,
        inflate_sh=inflate_sh,
        distinguishing_bytes_paths=["consumed", "ignored"],
        output_json=output_json,
    )
    assert result["verdict"] == "FAILED"  # any-failed → overall fail
    sr_by_name = {s["section"]: s for s in result["section_results"]}
    assert sr_by_name["consumed"]["passed"] is True
    assert sr_by_name["ignored"]["passed"] is False


# ---------------------------------------------------------------------------
# Infrastructure errors (rc=2)
# ---------------------------------------------------------------------------


def test_verify_infra_error_when_archive_missing(helper, tmp_path):
    inflate_sh = tmp_path / "inflate.sh"
    _make_simple_inflate_sh(inflate_sh, "x")
    output_json = tmp_path / "out.json"

    result = helper.verify_distinguishing_feature_byte_mutation(
        archive=tmp_path / "nonexistent.zip",
        inflate_sh=inflate_sh,
        distinguishing_bytes_paths=["x"],
        output_json=output_json,
    )
    assert result["verdict"] == "INFRASTRUCTURE_ERROR"
    assert "does not exist" in result["infrastructure_error_reason"]


def test_verify_infra_error_when_inflate_sh_missing(helper, tmp_path):
    archive = tmp_path / "archive.zip"
    _make_archive(archive, {"x": b"data"})
    output_json = tmp_path / "out.json"

    result = helper.verify_distinguishing_feature_byte_mutation(
        archive=archive,
        inflate_sh=tmp_path / "nonexistent.sh",
        distinguishing_bytes_paths=["x"],
        output_json=output_json,
    )
    assert result["verdict"] == "INFRASTRUCTURE_ERROR"


def test_verify_infra_error_when_section_not_in_archive(helper, tmp_path):
    archive = tmp_path / "archive.zip"
    _make_archive(archive, {"present": b"data"})
    inflate_sh = tmp_path / "inflate.sh"
    _make_simple_inflate_sh(inflate_sh, "present")
    output_json = tmp_path / "out.json"

    result = helper.verify_distinguishing_feature_byte_mutation(
        archive=archive,
        inflate_sh=inflate_sh,
        distinguishing_bytes_paths=["missing_section"],
        output_json=output_json,
    )
    assert result["verdict"] == "INFRASTRUCTURE_ERROR"
    assert "missing_section" in result["infrastructure_error_reason"]


def test_verify_infra_error_when_byte_range_out_of_bounds(helper, tmp_path):
    archive = tmp_path / "archive.zip"
    _make_archive(archive, {"x": b"short"})
    inflate_sh = tmp_path / "inflate.sh"
    _make_simple_inflate_sh(inflate_sh, "x")
    output_json = tmp_path / "out.json"

    result = helper.verify_distinguishing_feature_byte_mutation(
        archive=archive,
        inflate_sh=inflate_sh,
        distinguishing_bytes_paths=[],
        distinguishing_byte_ranges=["smart=x@2:99"],
        output_json=output_json,
    )
    assert result["verdict"] == "INFRASTRUCTURE_ERROR"
    assert "out of bounds" in result["infrastructure_error_reason"]


# ---------------------------------------------------------------------------
# Output JSON schema
# ---------------------------------------------------------------------------


def test_output_json_written_to_disk(helper, tmp_path):
    archive = tmp_path / "archive.zip"
    _make_archive(archive, {"x": b"data"})
    inflate_sh = tmp_path / "inflate.sh"
    _make_simple_inflate_sh(inflate_sh, "x")
    output_json = tmp_path / "subdir" / "out.json"

    result = helper.verify_distinguishing_feature_byte_mutation(
        archive=archive,
        inflate_sh=inflate_sh,
        distinguishing_bytes_paths=["x"],
        output_json=output_json,
    )
    assert output_json.is_file()
    on_disk = json.loads(output_json.read_text(encoding="utf-8"))
    assert on_disk == result


def test_output_json_schema_fields_present(helper, tmp_path):
    archive = tmp_path / "archive.zip"
    _make_archive(archive, {"x": b"data"})
    inflate_sh = tmp_path / "inflate.sh"
    _make_simple_inflate_sh(inflate_sh, "x")
    output_json = tmp_path / "out.json"

    result = helper.verify_distinguishing_feature_byte_mutation(
        archive=archive,
        inflate_sh=inflate_sh,
        distinguishing_bytes_paths=["x"],
        output_json=output_json,
    )
    expected_keys = {
        "schema_version",
        "archive_sha256",
        "archive_size_bytes",
        "inflate_sh_sha256",
        "distinguishing_bytes_paths",
        "section_results",
        "baseline_inflated_output_sha256",
        "baseline_repeat_inflated_output_sha256",
        "baseline_repeat_deterministic",
        "overall_passed",
        "verdict",
        "elapsed_seconds",
    }
    assert expected_keys.issubset(result.keys())
    assert result["schema_version"] == helper.SCHEMA_VERSION


# ---------------------------------------------------------------------------
# CLI behavior
# ---------------------------------------------------------------------------


def test_cli_main_returns_0_on_passed(helper, tmp_path):
    archive = tmp_path / "archive.zip"
    _make_archive(archive, {"x": b"data"})
    inflate_sh = tmp_path / "inflate.sh"
    _make_simple_inflate_sh(inflate_sh, "x")
    output_json = tmp_path / "out.json"

    rc = helper.main(
        [
            "--archive",
            str(archive),
            "--inflate-sh",
            str(inflate_sh),
            "--distinguishing-bytes-path",
            "x",
            "--output-json",
            str(output_json),
        ]
    )
    assert rc == 0


def test_cli_main_returns_0_on_member_byte_range_passed(helper, tmp_path):
    archive = tmp_path / "archive.zip"
    _make_archive(archive, {"x": b"HEADER_DISTINGUISHING_PAYLOAD_TAIL"})
    inflate_sh = tmp_path / "inflate.sh"
    _make_simple_inflate_sh(inflate_sh, "x")
    output_json = tmp_path / "out.json"

    rc = helper.main(
        [
            "--archive",
            str(archive),
            "--inflate-sh",
            str(inflate_sh),
            "--distinguishing-byte-range",
            "distinguishing_section=x@7:22",
            "--output-json",
            str(output_json),
        ]
    )
    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["section_results"][0]["target_basis"] == "member_byte_range"


def test_cli_main_returns_1_on_failed(helper, tmp_path):
    archive = tmp_path / "archive.zip"
    _make_archive(archive, {"x": b"data"})
    inflate_sh = tmp_path / "inflate.sh"
    _make_dead_section_inflate_sh(inflate_sh)
    output_json = tmp_path / "out.json"

    rc = helper.main(
        [
            "--archive",
            str(archive),
            "--inflate-sh",
            str(inflate_sh),
            "--distinguishing-bytes-path",
            "x",
            "--output-json",
            str(output_json),
        ]
    )
    assert rc == 1


def test_cli_main_returns_2_on_infra_error(helper, tmp_path):
    output_json = tmp_path / "out.json"
    rc = helper.main(
        [
            "--archive",
            str(tmp_path / "missing.zip"),
            "--inflate-sh",
            str(tmp_path / "missing.sh"),
            "--distinguishing-bytes-path",
            "x",
            "--output-json",
            str(output_json),
        ]
    )
    assert rc == 2


# ---------------------------------------------------------------------------
# OP-2 regression: codex chunk 7 finding (TypeError on main mutation path)
# ---------------------------------------------------------------------------
#
# Anchor: `.omx/research/codex_chunked_full_codebase_review_20260515.md`
# chunk 7 finding #1 (HIGH, confidence 0.93). Pre-commit-8a91995c5 the
# verifier's main mutation loop instantiated SectionResult without the
# new required fields (target_basis / member / offset / length) raising
# TypeError BEFORE emitting a valid proof artifact. Catalog #272 could
# not accumulate real byte-mutation evidence as a result.
#
# Commit 8a91995c5 ("frontier: harden score-table and selector evidence
# gates") fixed the issue by (a) threading MutationTarget through every
# SectionResult instantiation site and (b) wiring the
# `--distinguishing-byte-range` CLI flag so the new MutationTarget
# plumbing is reachable from the public CLI.
#
# These regression tests pin the fix at THREE surfaces:
#   1. SectionResult dataclass schema invariant (required fields).
#   2. End-to-end run does NOT raise TypeError; emits valid proof.
#   3. CLI byte-range flag is reachable and produces member_byte_range
#      target_basis (not just declared dead-code).


def test_regression_section_result_requires_target_basis_member_offset_length(helper):
    """REGRESSION (codex chunk 7 finding #1): SectionResult MUST require
    target_basis / member / offset / length. Pre-fix the dataclass DID
    declare these fields but the main loop instantiated SectionResult
    WITHOUT them — TypeError at runtime. This regression proves the
    schema invariant holds: any caller missing the new fields gets a
    deterministic TypeError naming the missing parameters."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(helper.SectionResult)}
    required_new_fields = {"target_basis", "member", "offset", "length"}
    assert required_new_fields <= fields, (
        f"SectionResult missing required new fields: {required_new_fields - fields}"
    )

    # Pre-fix construction (without the 4 new fields) MUST raise TypeError
    # naming the missing args. This is the EXACT shape the broken main
    # loop was producing.
    with pytest.raises(TypeError) as excinfo:
        helper.SectionResult(
            section="x",
            section_size_bytes=1,
            mutations_attempted=0,
            mutations_changed_output=0,
            passed=False,
            first_changed_inflated_output_sha256=None,
            baseline_inflated_output_sha256="abc",
        )
    msg = str(excinfo.value)
    assert "missing" in msg
    for fld in required_new_fields:
        assert fld in msg, f"TypeError should name missing field {fld!r}; got: {msg!r}"


def test_regression_main_path_does_not_raise_typeerror_on_zip_member_target(
    helper, tmp_path
):
    """REGRESSION (codex chunk 7 finding #1): the verifier's main
    mutation loop MUST NOT raise TypeError when instantiating
    SectionResult for a ZIP-member target. Pre-fix this was the broken
    code path that blocked Catalog #272 evidence accumulation."""
    archive = tmp_path / "archive.zip"
    _make_archive(archive, {"distinguishing": b"BYTES_THAT_GET_CONSUMED"})
    inflate_sh = tmp_path / "inflate.sh"
    _make_simple_inflate_sh(inflate_sh, "distinguishing")
    output_json = tmp_path / "proof.json"

    # Pre-fix this raised TypeError before producing any output JSON.
    # Post-fix it returns a valid result dict with section_results
    # carrying all 4 new SectionResult fields.
    result = helper.verify_distinguishing_feature_byte_mutation(
        archive=archive,
        inflate_sh=inflate_sh,
        distinguishing_bytes_paths=["distinguishing"],
        output_json=output_json,
    )

    assert result["verdict"] == "PASSED"
    assert output_json.is_file(), (
        "post-fix: proof JSON MUST be written to disk on the main mutation path"
    )
    sr = result["section_results"][0]
    # All 4 new fields must be present and populated on the main path.
    assert sr["target_basis"] == "zip_member"
    assert sr["member"] == "distinguishing"
    assert sr["offset"] is None
    assert sr["length"] is None


def test_regression_main_path_does_not_raise_typeerror_on_byte_range_target(
    helper, tmp_path
):
    """REGRESSION (codex chunk 7 finding #1): the verifier's main
    mutation loop MUST NOT raise TypeError when instantiating
    SectionResult for a member-byte-range target. Pre-fix the new
    MutationTarget plumbing was unreachable from the public CLI and the
    SectionResult fields were not populated for byte-range targets."""
    archive = tmp_path / "archive.zip"
    _make_archive(archive, {"x": b"HEADER_DISTINGUISHING_PAYLOAD_TAIL"})
    inflate_sh = tmp_path / "inflate.sh"
    _make_simple_inflate_sh(inflate_sh, "x")
    output_json = tmp_path / "proof.json"

    result = helper.verify_distinguishing_feature_byte_mutation(
        archive=archive,
        inflate_sh=inflate_sh,
        distinguishing_bytes_paths=[],
        distinguishing_byte_ranges=["distinguishing_section=x@7:22"],
        output_json=output_json,
    )

    assert result["verdict"] == "PASSED"
    assert output_json.is_file()
    sr = result["section_results"][0]
    # All 4 new fields must be present AND populated for byte-range path.
    assert sr["target_basis"] == "member_byte_range"
    assert sr["member"] == "x"
    assert sr["offset"] == 7
    assert sr["length"] == 22


def test_regression_main_path_does_not_raise_typeerror_on_empty_section(
    helper, tmp_path
):
    """REGRESSION (codex chunk 7 finding #1): the empty-section branch
    of the main loop MUST also populate the 4 new SectionResult fields.
    Pre-fix this branch was a separate SectionResult instantiation that
    independently broke when the new required fields were added to the
    dataclass."""
    archive = tmp_path / "archive.zip"
    _make_archive(archive, {"distinguishing": b"", "renderer": b"FILLER"})
    inflate_sh = tmp_path / "inflate.sh"
    _make_simple_inflate_sh(inflate_sh, "renderer")
    output_json = tmp_path / "proof.json"

    result = helper.verify_distinguishing_feature_byte_mutation(
        archive=archive,
        inflate_sh=inflate_sh,
        distinguishing_bytes_paths=["distinguishing"],
        output_json=output_json,
    )

    assert result["verdict"] == "FAILED"
    assert output_json.is_file()
    sr = result["section_results"][0]
    assert sr["target_basis"] == "zip_member"
    assert sr["member"] == "distinguishing"
    assert sr["offset"] is None
    assert sr["length"] is None
    assert sr["section_size_bytes"] == 0
    assert sr["passed"] is False


def test_regression_cli_byte_range_flag_is_reachable(helper, tmp_path):
    """REGRESSION (codex chunk 7 finding #1): the codex finding noted
    the new --distinguishing-byte-range plumbing was 'unreachable from
    the public CLI', i.e. effectively dead code. Pin that the CLI flag
    is wired and produces a member_byte_range proof artifact."""
    archive = tmp_path / "archive.zip"
    _make_archive(archive, {"x": b"HEADER_DISTINGUISHING_PAYLOAD_TAIL"})
    inflate_sh = tmp_path / "inflate.sh"
    _make_simple_inflate_sh(inflate_sh, "x")
    output_json = tmp_path / "proof.json"

    rc = helper.main(
        [
            "--archive",
            str(archive),
            "--inflate-sh",
            str(inflate_sh),
            "--distinguishing-byte-range",
            "distinguishing_section=x@7:22",
            "--output-json",
            str(output_json),
        ]
    )
    assert rc == 0, "CLI byte-range flag MUST exit 0 on a passing run"
    assert output_json.is_file()
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["verdict"] == "PASSED"
    sr = payload["section_results"][0]
    assert sr["target_basis"] == "member_byte_range"
    assert sr["offset"] == 7
    assert sr["length"] == 22


def test_regression_cli_requires_at_least_one_target(helper, tmp_path):
    """REGRESSION: CLI must require at least one of the two flags
    (--distinguishing-bytes-path or --distinguishing-byte-range). Pre-fix
    --distinguishing-bytes-path was `required=True`; post-fix BOTH are
    optional but the CLI parser MUST refuse the all-empty call."""
    output_json = tmp_path / "out.json"
    archive = tmp_path / "archive.zip"
    _make_archive(archive, {"x": b"data"})
    inflate_sh = tmp_path / "inflate.sh"
    _make_simple_inflate_sh(inflate_sh, "x")

    with pytest.raises(SystemExit):
        helper.main(
            [
                "--archive",
                str(archive),
                "--inflate-sh",
                str(inflate_sh),
                "--output-json",
                str(output_json),
            ]
        )
