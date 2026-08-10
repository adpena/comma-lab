"""Tests for the ALWAYS-KEEP-THE-PAYLOAD gate (operator P0, 2026-08-09).

The anchor is real: ``ans_real_n600.py`` measured both coder payloads over n600 and
persisted only their lengths. These tests pin the four properties that matter — it
catches the anchor shape, it goes QUIET when the cure is applied (the
detector-does-not-zero-on-the-cure law), it honours a substantive waiver, and it
rejects a placeholder one.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tac.payload_retention_gate import (
    PAYLOAD_PRODUCERS,
    WAIVER_TOKEN,
    audit_measure_and_discard_payload,
    check_no_measure_and_discard_payload,
    scan_source,
)

# The verbatim shape of the incident, reduced to its defect.
ANCHOR = """
import json
rng = len(enc.get_compressed().tobytes()); del enc
an  = len(ans.get_compressed().tobytes())
res = {"range_B": rng, "ans_B": an}
open(D + '/ans_vs_range_n600_result.json', 'w').write(json.dumps(res, indent=2))
"""

CURED = """
import json
from pathlib import Path
payload = enc.get_compressed().tobytes()
Path(out / "range_n600.bin").write_bytes(payload)
rng = len(payload)
open(D + '/result.json', 'w').write(json.dumps({"range_B": rng}))
"""


def test_anchor_flags_both_discarded_payloads() -> None:
    findings = scan_source(ANCHOR, "ans_real_n600.py")
    assert len(findings) == 2, "both the range AND the ANS payload are discarded"
    assert {f.root for f in findings} == {"enc", "ans"}
    assert all(f.producer == "tobytes" for f in findings)


def test_del_is_not_the_signature() -> None:
    """Line 4 of the anchor has no ``del`` and is equally defective."""
    findings = scan_source(ANCHOR, "a.py")
    ans_finding = next(f for f in findings if f.root == "ans")
    assert "del" not in ans_finding.snippet


def test_detector_goes_quiet_when_the_cure_is_applied() -> None:
    """The standing law: ask what the gauge reads if you apply the cure."""
    assert scan_source(CURED, "cured.py") == []


def test_unrelated_write_does_not_clear_the_finding() -> None:
    """Persisting some OTHER object must not launder the discarded payload."""
    src = (
        "from pathlib import Path\n"
        "Path('notes.bin').write_bytes(unrelated)\n"
        "n = len(enc.get_compressed().tobytes())\n"
    )
    assert len(scan_source(src, "x.py")) == 1


def test_binary_open_write_counts_as_persistence() -> None:
    src = (
        "payload = enc.get_compressed().tobytes()\n"
        "with open(p, 'wb') as fh:\n"
        "    fh.write(payload)\n"
        "n = len(payload)\n"
    )
    assert scan_source(src, "x.py") == []


def test_text_mode_open_does_not_count_as_persistence() -> None:
    src = (
        "payload = enc.get_compressed().tobytes()\n"
        "open(p, 'w').write(str(len(payload)))\n"
        "n = len(payload)\n"
    )
    assert len(scan_source(src, "x.py")) == 1


def test_substantive_waiver_is_honoured() -> None:
    src = f"n = len(enc.get_compressed().tobytes())  # {WAIVER_TOKEN}: entropy-only probe, no archive built\n"
    assert scan_source(src, "x.py") == []


@pytest.mark.parametrize("bad", ["<rationale>", "<reason>", "TBD", "reason", "", "n/a"])
def test_placeholder_waivers_are_rejected(bad: str) -> None:
    src = f"n = len(enc.get_compressed().tobytes())  # {WAIVER_TOKEN}: {bad}\n"
    assert len(scan_source(src, "x.py")) == 1


def test_waiver_without_space_after_hash_is_honoured() -> None:
    src = f"n = len(enc.get_compressed().tobytes())  #{WAIVER_TOKEN}: deliberate scalar-only entropy bound\n"
    assert scan_source(src, "x.py") == []


def test_brotli_compress_argument_is_the_tracked_root() -> None:
    """For ``brotli.compress(payload)`` the payload, not the module, is the root."""
    findings = scan_source("n = len(brotli.compress(payload, quality=11))\n", "x.py")
    assert len(findings) == 1
    assert findings[0].root == "payload"


def test_brotli_compress_cleared_by_persisting_its_result() -> None:
    src = (
        "blob = brotli.compress(payload, quality=11)\n"
        "Path(o).write_bytes(blob)\n"
        "n = len(blob)\n"
    )
    assert scan_source(src, "x.py") == []


def test_nested_producers_are_one_outer_payload() -> None:
    """The charter's inflated-count anchor is one compressed payload, not three roots."""
    src = "n = len(zlib.compress(cq.tobytes() + np.float32(s).tobytes(), 9))\n"
    findings = scan_source(src, "x.py")
    assert len(findings) == 1
    assert findings[0].producer == "compress"


def test_repeated_measurement_of_one_bound_payload_counts_once() -> None:
    src = (
        "blob = zlib.compress(raw, 9)\n"
        "first = len(blob)\n"
        "second = len(blob)\n"
    )
    findings = scan_source(src, "x.py")
    assert len(findings) == 1
    assert findings[0].payload_line == 1


def test_rebinding_one_name_to_two_payloads_counts_both() -> None:
    src = (
        "blob = zlib.compress(first, 9)\n"
        "n1 = len(blob)\n"
        "blob = zlib.compress(second, 9)\n"
        "n2 = len(blob)\n"
    )
    findings = scan_source(src, "x.py")
    assert {(finding.payload_line, finding.producer) for finding in findings} == {
        (1, "compress"),
        (3, "compress"),
    }


def test_only_latest_rebinding_is_measured_when_earlier_value_was_overwritten() -> None:
    src = (
        "blob = zlib.compress(first, 9)\n"
        "blob = zlib.compress(second, 9)\n"
        "n = len(blob)\n"
    )
    findings = scan_source(src, "x.py")
    assert [(finding.payload_line, finding.producer) for finding in findings] == [
        (2, "compress")
    ]


def test_same_local_name_in_unrelated_functions_does_not_cross_bind() -> None:
    src = (
        "def build():\n"
        "    blob = zlib.compress(first, 9)\n"
        "    return blob\n"
        "\n"
        "def inspect():\n"
        "    blob = unrelated\n"
        "    return len(blob)\n"
    )
    assert scan_source(src, "x.py") == []


def test_persistence_clears_only_the_binding_it_received() -> None:
    src = (
        "blob = zlib.compress(first, 9)\n"
        "Path(out).write_bytes(blob)\n"
        "n1 = len(blob)\n"
        "blob = zlib.compress(second, 9)\n"
        "n2 = len(blob)\n"
    )
    findings = scan_source(src, "x.py")
    assert [(finding.payload_line, finding.producer) for finding in findings] == [
        (4, "compress")
    ]


def test_persisting_raw_input_does_not_clear_inline_compressed_output() -> None:
    src = (
        "Path(raw_path).write_bytes(raw)\n"
        "n = len(brotli.compress(raw, quality=11))\n"
    )
    assert len(scan_source(src, "x.py")) == 1


def test_plain_len_on_a_list_is_not_a_finding() -> None:
    """Precision guard: the gate must never fire on ordinary length checks."""
    src = "n = len(pair_ids)\nm = len(frames)\nk = len('abc')\n"
    assert scan_source(src, "x.py") == []


def test_numpy_save_counts_as_persistence() -> None:
    src = (
        "payload = arr.tobytes()\n"
        "np.save(path, payload)\n"
        "n = len(payload)\n"
    )
    assert scan_source(src, "x.py") == []


def test_checked_retention_helper_counts_as_persistence() -> None:
    src = (
        "blob = brotli.compress(raw, quality=11)\n"
        "retained = retain_payload(path, blob)\n"
        "n = len(blob)\n"
    )
    assert scan_source(src, "x.py") == []


def test_syntax_error_is_not_our_bug_class() -> None:
    assert scan_source("def (:\n", "broken.py") == []


def test_render_names_the_rule_and_the_fix() -> None:
    finding = scan_source(ANCHOR, "a.py")[0]
    text = finding.render()
    assert "ALWAYS KEEP THE PAYLOAD" in text
    assert "write_bytes" in text
    assert WAIVER_TOKEN in text


def test_producers_are_narrow_enough_to_exclude_generic_encode() -> None:
    """Precision: ``encode``/``dumps`` would sweep in ordinary string work."""
    assert "encode" not in PAYLOAD_PRODUCERS
    assert "dumps" not in PAYLOAD_PRODUCERS
    assert "get_compressed" in PAYLOAD_PRODUCERS


def test_strict_mode_raises_with_the_rule_named(tmp_path: Path) -> None:
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "bad.py").write_text(ANCHOR, encoding="utf-8")
    with pytest.raises(RuntimeError, match="ALWAYS KEEP THE PAYLOAD"):
        check_no_measure_and_discard_payload(
            repo_root=tmp_path, strict=True, roots=("experiments",)
        )


def test_non_strict_returns_findings_without_raising(tmp_path: Path) -> None:
    (tmp_path / "experiments").mkdir()
    (tmp_path / "experiments" / "bad.py").write_text(ANCHOR, encoding="utf-8")
    out = check_no_measure_and_discard_payload(
        repo_root=tmp_path, strict=False, roots=("experiments",)
    )
    assert len(out) == 2


def test_missing_root_is_skipped_not_an_error(tmp_path: Path) -> None:
    assert check_no_measure_and_discard_payload(
        repo_root=tmp_path, strict=True, roots=("does_not_exist",)
    ) == []


def test_population_reports_explicit_denominator(tmp_path: Path) -> None:
    root = tmp_path / "experiments"
    root.mkdir()
    (root / "bad.py").write_text(ANCHOR, encoding="utf-8")
    (root / "clean.py").write_text("n = len(items)\n", encoding="utf-8")
    report = audit_measure_and_discard_payload(
        repo_root=tmp_path,
        roots=("experiments",),
    )
    assert report.python_files_discovered == 2
    assert report.python_files_examined == 2
    assert report.candidate_files_parsed == 1
    assert report.unreadable_files == 0
    assert len(report.findings) == 2
    assert report.files_with_findings == 1


def test_preflight_all_calls_population_gate_warn_only() -> None:
    source = Path("src/tac/preflight.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    preflight_all = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "preflight_all"
    )
    calls = [
        node for node in ast.walk(preflight_all)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "audit_measure_and_discard_payload"
    ]
    assert len(calls) == 1
    assert not any(
        keyword.arg == "strict" and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in calls[0].keywords
    )
