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
    assert "2 runtime + 7 non-runtime - 1 in both" in out


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
