"""Tests for tools/packet_census_guard.py (round-11 F3(b)).

The round-10 class cure landed with "Tests owed" disclosed in its own commit
message and none written. A guard nobody calls and nobody tests is an artifact,
not a gate -- the finding that produced this file.

Covered: the undeclared-file positive case, the clean negative case, the
double-declaration arithmetic that printed "39 declared (34 + 7)", the prep-dir
structural census (round-11 F6's surface), and the usage/IO refusals.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_GUARD = Path(__file__).resolve().parents[1] / "packet_census_guard.py"
_spec = importlib.util.spec_from_file_location("packet_census_guard", _GUARD)
assert _spec is not None and _spec.loader is not None
guard = importlib.util.module_from_spec(_spec)
sys.modules["packet_census_guard"] = guard
_spec.loader.exec_module(guard)


def _auth_eval(tmp_path: Path, relative_paths: list[str]) -> Path:
    payload = {
        "provenance": {
            "inflate_runtime_manifest": {
                "files": [
                    {"relative_path": rel, "bytes": 1, "sha256": "0" * 64}
                    for rel in relative_paths
                ]
            }
        }
    }
    path = tmp_path / "contest_auth_eval.json"
    path.write_text(json.dumps(payload))
    return path


def _packet(tmp_path: Path, names: list[str]) -> Path:
    packet = tmp_path / "packet"
    packet.mkdir()
    for name in names:
        target = packet / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x")
    return packet


# --------------------------------------------------------------------------
# packet-dir census
# --------------------------------------------------------------------------


def test_undeclared_file_is_detected(tmp_path, capsys):
    """POSITIVE: a file nobody declared refuses with rc=1 and is named."""
    auth = _auth_eval(tmp_path, ["inflate.py"])
    packet = _packet(tmp_path, ["inflate.py", "README.md", "runtime/__pycache__.pyc"])

    rc = guard.main(["--packet-dir", str(packet), "--auth-eval-json", str(auth)])

    assert rc == 1
    out = capsys.readouterr().out
    assert "UNDECLARED_FILES_PRESENT" in out
    assert "runtime/__pycache__.pyc" in out


def test_clean_packet_passes(tmp_path, capsys):
    """NEGATIVE: exactly the declared set returns rc=0 and prints the denominator."""
    auth = _auth_eval(tmp_path, ["inflate.py", "inflate.sh"])
    packet = _packet(tmp_path, ["inflate.py", "inflate.sh", "README.md", "report.txt"])

    rc = guard.main(["--packet-dir", str(packet), "--auth-eval-json", str(auth)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "CENSUS_CLEAN" in out
    assert "undeclared 0" in out


def test_missing_declared_file_is_reported_but_not_fatal(tmp_path, capsys):
    """A declared file absent from disk is REPORTED; only undeclared files refuse."""
    auth = _auth_eval(tmp_path, ["inflate.py", "inflate.sh"])
    packet = _packet(tmp_path, ["inflate.py"])

    rc = guard.main(["--packet-dir", str(packet), "--auth-eval-json", str(auth)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "MISSING:    inflate.sh" in out


def test_double_declaration_arithmetic_reconciles(tmp_path, capsys):
    """F3(c): declared total is the UNION, and the printed breakdown must equal it.

    The pre-fix header read "39 declared (34 runtime + 7 non-runtime)" against a
    real packet -- 34 + 7 = 41, not 39. The two absent names were exactly the
    files in both lists. The header now subtracts the overlap and names it.
    """
    auth = _auth_eval(tmp_path, ["inflate.py", "GENERATION_RECEIPT.json"])
    packet = _packet(tmp_path, ["inflate.py", "GENERATION_RECEIPT.json"])

    rc = guard.main(
        [
            "--packet-dir",
            str(packet),
            "--auth-eval-json",
            str(auth),
            "--json-out",
            str(tmp_path / "out.json"),
        ]
    )

    assert rc == 0
    report = json.loads((tmp_path / "out.json").read_text())
    assert report["declared_overlap"] == ["GENERATION_RECEIPT.json"]
    assert report["declared_overlap_count"] == 1
    # The identity the pre-fix printout violated.
    assert (
        report["declared_total_count"]
        == report["declared_runtime_count"]
        + report["declared_non_runtime_count"]
        - report["declared_overlap_count"]
    )
    out = capsys.readouterr().out
    assert "DOUBLE-DECLARED: GENERATION_RECEIPT.json" in out
    # Derive the expected breakdown from the constant rather than hardcoding it:
    # a literal here drifts silently the next time the declared set grows, which
    # is exactly the hardcoded-count class this test exists to catch.
    non_runtime = len(guard.DECLARED_NON_RUNTIME)
    assert f"2 runtime + {non_runtime} non-runtime - 1 in both" in out


# --------------------------------------------------------------------------
# prep-dir census (round-11 F6 surface)
# --------------------------------------------------------------------------


def test_prep_census_flags_nested_hook_written_state(tmp_path, capsys):
    """POSITIVE: the exact F6 shape -- a nested .omx/state written by a hook."""
    prep = tmp_path / "prep"
    (prep / ".omx" / "state").mkdir(parents=True)
    (prep / "SWAP_PROCEDURE.md").write_text("doc")
    (prep / ".omx" / "state" / "triality_drift_marker.json").write_text("{}")

    rc = guard.main(["--prep-dir", str(prep)])

    assert rc == 1
    out = capsys.readouterr().out
    assert "PREP_STRAYS_PRESENT" in out
    assert "STRAY: .omx/state/triality_drift_marker.json" in out


def test_prep_census_clean_flat_directory(tmp_path, capsys):
    """NEGATIVE: a flat document set passes and reports its denominator."""
    prep = tmp_path / "prep"
    prep.mkdir()
    for name in ("SWAP_PROCEDURE.md", "GAP_REPORT.json", "README_PUBLIC.md"):
        (prep / name).write_text("doc")

    rc = guard.main(["--prep-dir", str(prep)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "PREP_CLEAN" in out
    assert "3 flat document(s)" in out


def test_prep_census_flags_top_level_dotfile(tmp_path):
    """A dot-entry at depth 1 is a stray too -- it is not a prep document."""
    prep = tmp_path / "prep"
    prep.mkdir()
    (prep / "GAP_REPORT.json").write_text("{}")
    (prep / ".DS_Store").write_text("junk")

    nested, dotted = guard.prep_census(prep)

    assert nested == []
    assert dotted == [".DS_Store"]


def test_both_modes_in_one_invocation_returns_worst_rc(tmp_path):
    """A clean packet plus a dirty prep tree must still refuse."""
    auth = _auth_eval(tmp_path, ["inflate.py"])
    packet = _packet(tmp_path, ["inflate.py"])
    prep = tmp_path / "prep"
    (prep / "sub").mkdir(parents=True)
    (prep / "sub" / "stray.json").write_text("{}")

    rc = guard.main(
        [
            "--packet-dir",
            str(packet),
            "--auth-eval-json",
            str(auth),
            "--prep-dir",
            str(prep),
            "--json-out",
            str(tmp_path / "both.json"),
        ]
    )

    assert rc == 1
    payload = json.loads((tmp_path / "both.json").read_text())
    modes = [r["mode"] for r in payload["reports"]]
    assert modes == ["packet", "prep"]


# --------------------------------------------------------------------------
# usage / IO refusals
# --------------------------------------------------------------------------


def test_no_mode_refuses(tmp_path):
    assert guard.main([]) == 2


def test_packet_dir_without_auth_eval_refuses(tmp_path):
    packet = _packet(tmp_path, ["inflate.py"])
    assert guard.main(["--packet-dir", str(packet)]) == 2


def test_missing_packet_dir_refuses(tmp_path):
    auth = _auth_eval(tmp_path, ["inflate.py"])
    rc = guard.main(
        ["--packet-dir", str(tmp_path / "nope"), "--auth-eval-json", str(auth)]
    )
    assert rc == 2


def test_missing_prep_dir_refuses(tmp_path):
    assert guard.main(["--prep-dir", str(tmp_path / "nope")]) == 2


def test_manifest_without_file_list_refuses(tmp_path):
    bad = tmp_path / "auth.json"
    bad.write_text(json.dumps({"provenance": {"inflate_runtime_manifest": {}}}))
    packet = _packet(tmp_path, ["inflate.py"])
    rc = guard.main(["--packet-dir", str(packet), "--auth-eval-json", str(bad)])
    assert rc == 2


def test_manifest_absent_refuses(tmp_path):
    bad = tmp_path / "auth.json"
    bad.write_text(json.dumps({"provenance": {}}))
    with pytest.raises(KeyError):
        guard.load_manifest_files(bad)


# --------------------------------------------------------------------------
# receipts-dir census (round-12 F5)
#
# The finding: the AppleDouble cure covered the staged tree -- the directory the
# guard walks -- and stopped there, leaving one ._ sidecar per real receipt in
# gen4_receipts/ and five more one level up in generations/. These tests are the
# positive controls for both surfaces.
# --------------------------------------------------------------------------


def _receipts(tmp_path: Path, names: list[str], *, parent_names: list[str] | None = None) -> Path:
    parent = tmp_path / "generations"
    parent.mkdir(parents=True, exist_ok=True)
    receipts = parent / "gen4_receipts"
    receipts.mkdir()
    for name in names:
        target = receipts / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x")
    for name in parent_names or []:
        (parent / name).write_text("x")
    return receipts


def test_receipts_census_clean_custody_store_passes(tmp_path, capsys):
    receipts = _receipts(tmp_path, ["contest_auth_eval.json", "report.txt", "provenance.json"])
    rc = guard.main(["--receipts-dir", str(receipts)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "RECEIPTS_CLEAN" in out
    # denominator is always printed, not just the hits
    assert "3 flat custody file(s)" in out


def test_receipts_census_catches_planted_appledouble_sidecar(tmp_path, capsys):
    """Positive control: the exact round-12 F5 artefact."""
    receipts = _receipts(tmp_path, ["contest_auth_eval.json"])
    (receipts / "._contest_auth_eval.json").write_bytes(bytes([0x00, 0x05, 0x16, 0x07]))
    rc = guard.main(["--receipts-dir", str(receipts)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "RECEIPTS_STRAYS_PRESENT" in out
    assert "STRAY: ._contest_auth_eval.json" in out


def test_receipts_census_catches_sidecar_one_level_up(tmp_path, capsys):
    """The 'generations/ holds five ._* entries' surface the review measured."""
    receipts = _receipts(
        tmp_path, ["contest_auth_eval.json"], parent_names=["._gen4_receipts"]
    )
    rc = guard.main(["--receipts-dir", str(receipts)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "STRAY: ../._gen4_receipts" in out
    assert "parent dot-entries 1" in out


def test_receipts_census_catches_nested_pycache(tmp_path, capsys):
    receipts = _receipts(tmp_path, ["report.txt", "__pycache__/thing.cpython-313.pyc"])
    rc = guard.main(["--receipts-dir", str(receipts)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "STRAY: __pycache__/thing.cpython-313.pyc" in out


def test_receipts_census_catches_dot_entry_at_depth(tmp_path, capsys):
    receipts = _receipts(tmp_path, ["report.txt", "sub/._buried"])
    rc = guard.main(["--receipts-dir", str(receipts)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "STRAY: sub/._buried" in out


def test_receipts_dir_is_repeatable_and_returns_worst_rc(tmp_path, capsys):
    clean = _receipts(tmp_path, ["report.txt"])
    dirty_parent = tmp_path / "gen3"
    dirty_parent.mkdir()
    dirty = dirty_parent / "gen3_receipts"
    dirty.mkdir()
    (dirty / "report.txt").write_text("x")
    (dirty / "._report.txt").write_text("x")
    rc = guard.main(["--receipts-dir", str(clean), "--receipts-dir", str(dirty)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "RECEIPTS_CLEAN" in out
    assert "RECEIPTS_STRAYS_PRESENT" in out


def test_receipts_census_alone_is_a_valid_invocation(tmp_path):
    receipts = _receipts(tmp_path, ["report.txt"])
    assert guard.main(["--receipts-dir", str(receipts)]) == 0


def test_missing_receipts_dir_refuses(tmp_path):
    assert guard.main(["--receipts-dir", str(tmp_path / "nope")]) == 2


def test_receipts_census_json_out_carries_both_surfaces(tmp_path):
    receipts = _receipts(
        tmp_path, ["report.txt", "._report.txt"], parent_names=["._gen4_receipts"]
    )
    out_json = tmp_path / "census.json"
    rc = guard.main(["--receipts-dir", str(receipts), "--json-out", str(out_json)])
    assert rc == 1
    payload = json.loads(out_json.read_text())
    assert payload["mode"] == "receipts"
    assert payload["dot_entries"] == ["._report.txt"]
    assert payload["parent_dot_entries"] == ["../._gen4_receipts"]
    assert payload["stray_count"] == 2
    # a sidecar is not a receipt: the denominator counts real custody files only
    assert payload["flat_custody_file_count"] == 1


def test_receipts_census_does_not_delete_what_it_finds(tmp_path):
    """A guard that repairs what it measures cannot be trusted to report it."""
    receipts = _receipts(tmp_path, ["report.txt"])
    sidecar = receipts / "._report.txt"
    sidecar.write_text("x")
    guard.main(["--receipts-dir", str(receipts)])
    assert sidecar.exists()


def test_no_mode_message_names_receipts_dir(tmp_path, capsys):
    assert guard.main([]) == 2
    assert "--receipts-dir" in capsys.readouterr().err


def test_receipts_census_ignores_dot_directory_beside_the_store(tmp_path, capsys):
    """A .git beside a receipts dir must not manufacture a stray."""
    receipts = _receipts(tmp_path, ["report.txt"])
    (receipts.parent / ".git").mkdir()
    rc = guard.main(["--receipts-dir", str(receipts)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "RECEIPTS_CLEAN" in out
    assert ".git" not in out


def test_receipts_census_still_catches_dot_file_beside_the_store(tmp_path, capsys):
    """...but a dot-FILE one level up is still a stray (.DS_Store, ._sidecar)."""
    receipts = _receipts(tmp_path, ["report.txt"], parent_names=[".DS_Store"])
    rc = guard.main(["--receipts-dir", str(receipts)])
    assert rc == 1
    assert "STRAY: ../.DS_Store" in capsys.readouterr().out
