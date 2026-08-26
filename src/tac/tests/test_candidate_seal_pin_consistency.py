"""Controls for the candidate-seal pin-consistency brick.

Both directions are executed, per the gate law: the check must go RED on the disease (a tree
whose receiver pins a different candidate than the archive staged beside it) and GREEN on the
cure (a correctly re-pinned NON-rr4 tree — the exact case the latched literal refused by
construction).  The real retained ``ddm_fx1`` candidate runtime is used as the green control
whenever the custody volume is mounted; synthetic trees carry the same controls unconditionally
so the gate never silently skips into a pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.candidate_seal import (
    ARCHIVE_MISSING,
    CONSISTENT,
    MISMATCH,
    PIN_ABSENT,
    RECEIVER_MISSING,
    SealContractError,
    check_pin_consistency,
    measure_archive_identity,
    read_frontier_archive_identity,
    read_receiver_pin,
    repin_receiver,
)

# The real fixture: fx1's byte-closed 180,601 B candidate, staged with its own re-pinned
# receiver.  It is NOT rr4, which is the point — the defect this brick replaces refused it.
FX1_RUNTIME = Path("/Volumes/APDataStore/pact/ddm_fx1/candidate_runtime")
RR4_SHA = "35ac2b9beb7e6fa81075c7d84b5247d8d24c056fe49ce1cbd22a334bc9618956"
RR4_BYTES = 181_161

RECEIVER_TEMPLATE = '''#!/usr/bin/env python3
"""A stand-in receiver shaped like the shipped one."""

from __future__ import annotations

import argparse

ARCHIVE_SHA256 = "{sha}"
ARCHIVE_BYTES = {size}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()


if __name__ == "__main__":
    main()
'''


def _stage(tmp_path: Path, payload: bytes, pin_sha: str | None = None, pin_bytes: int | None = None) -> Path:
    """Stage a runtime tree; by default the receiver is pinned correctly to the payload."""
    runtime = tmp_path / "candidate_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    archive = runtime / "archive.zip"
    archive.write_bytes(payload)
    identity = measure_archive_identity(archive)
    receiver = runtime / "inflate.py"
    receiver.write_text(
        RECEIVER_TEMPLATE.format(
            sha=pin_sha if pin_sha is not None else identity.sha256,
            size=pin_bytes if pin_bytes is not None else identity.bytes,
        )
    )
    return runtime


# --------------------------------------------------------------------------------------
# RED — the disease
# --------------------------------------------------------------------------------------


def test_red_pin_naming_a_different_candidate_is_refused(tmp_path: Path) -> None:
    """The rr4-pinned receiver staged over a different archive must REFUSE, loudly and by reason."""
    runtime = _stage(tmp_path, b"fx1-candidate-bytes", pin_sha=RR4_SHA, pin_bytes=RR4_BYTES)

    verdict = check_pin_consistency(runtime)

    assert verdict.verdict == MISMATCH
    assert verdict.ok is False
    # Both sides of the comparison survive into the verdict; a bare boolean could not say why.
    assert verdict.pinned_sha256 == RR4_SHA
    assert verdict.measured_sha256 != RR4_SHA
    assert any("ARCHIVE_SHA256 pins" in problem for problem in verdict.problems)
    assert any("ARCHIVE_BYTES pins" in problem for problem in verdict.problems)
    assert "ONE sealed object" in " ".join(verdict.problems)


def test_red_byte_count_alone_disagreeing_is_a_mismatch(tmp_path: Path) -> None:
    """A pin whose sha matches but whose byte count does not is still a broken seal."""
    payload = b"payload-under-test"
    runtime = _stage(tmp_path, payload, pin_bytes=len(payload) + 1)

    verdict = check_pin_consistency(runtime)

    assert verdict.verdict == MISMATCH
    assert verdict.pinned_sha256 == verdict.measured_sha256
    assert verdict.pinned_bytes != verdict.measured_bytes


# --------------------------------------------------------------------------------------
# GREEN — the cure
# --------------------------------------------------------------------------------------


def test_green_correctly_repinned_non_rr4_tree_passes(tmp_path: Path) -> None:
    """A NON-rr4 candidate, correctly re-pinned, must PASS. The latched literal could not."""
    runtime = _stage(tmp_path, b"a-candidate-that-is-not-rr4")

    verdict = check_pin_consistency(runtime)

    assert verdict.verdict == CONSISTENT
    assert verdict.ok is True
    assert verdict.problems == ()
    assert verdict.measured_sha256 != RR4_SHA  # the control is only meaningful if it is non-rr4
    assert "CONSISTENT" in verdict.summary()


def test_green_real_fx1_candidate_runtime_passes() -> None:
    """The real retained fx1 tree — 180,601 B, non-rr4 — passes against its own receiver."""
    if not (FX1_RUNTIME / "inflate.py").is_file():
        pytest.skip(f"retained fx1 custody volume not mounted at {FX1_RUNTIME}")

    verdict = check_pin_consistency(FX1_RUNTIME)

    assert verdict.verdict == CONSISTENT, verdict.summary()
    assert verdict.measured_bytes == 180_601
    assert verdict.measured_sha256.startswith("65c75d7f")
    # The whole point: this tree is NOT the pinned-by-literal candidate.
    assert verdict.measured_sha256 != RR4_SHA
    assert verdict.measured_bytes != RR4_BYTES


# --------------------------------------------------------------------------------------
# Vacuity is reported, never passed
# --------------------------------------------------------------------------------------


def test_unpinned_receiver_is_pin_absent_not_consistent(tmp_path: Path) -> None:
    """A receiver with no pin must NOT read as a pass; the vacuous case gets its own verdict."""
    runtime = tmp_path / "candidate_runtime"
    runtime.mkdir()
    (runtime / "archive.zip").write_bytes(b"unpinned-candidate")
    (runtime / "inflate.py").write_text("def main() -> None:\n    pass\n")

    verdict = check_pin_consistency(runtime)

    assert verdict.verdict == PIN_ABSENT
    assert verdict.ok is False
    assert "UNPINNED" in " ".join(verdict.problems)
    # The measurement still happened, so the caller can see what it would have pinned.
    assert verdict.measured_bytes == len(b"unpinned-candidate")


def test_missing_receiver_and_missing_archive_get_distinct_verdicts(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert check_pin_consistency(empty).verdict == RECEIVER_MISSING

    receiver_only = tmp_path / "receiver_only"
    receiver_only.mkdir()
    (receiver_only / "inflate.py").write_text(RECEIVER_TEMPLATE.format(sha=RR4_SHA, size=RR4_BYTES))
    assert check_pin_consistency(receiver_only).verdict == ARCHIVE_MISSING


# --------------------------------------------------------------------------------------
# The re-pin: the compose-time cure, and it does not zero on itself
# --------------------------------------------------------------------------------------


def test_repin_converts_mismatch_to_consistent_and_touches_only_two_lines(tmp_path: Path) -> None:
    runtime = _stage(tmp_path, b"fx1-candidate-bytes", pin_sha=RR4_SHA, pin_bytes=RR4_BYTES)
    receiver = runtime / "inflate.py"
    before_lines = receiver.read_text().splitlines()

    result = repin_receiver(runtime)

    assert result.changed is True
    assert result.verdict_before == MISMATCH
    assert result.verdict_after == CONSISTENT
    assert check_pin_consistency(runtime).verdict == CONSISTENT

    after_lines = receiver.read_text().splitlines()
    assert len(after_lines) == len(before_lines)
    differing = [i for i, (a, b) in enumerate(zip(before_lines, after_lines, strict=True)) if a != b]
    assert len(differing) == 2, f"re-pin changed {len(differing)} lines; only the two constants may move"
    assert all(after_lines[i].startswith(("ARCHIVE_SHA256", "ARCHIVE_BYTES")) for i in differing)
    # House style: the byte count keeps underscore grouping.
    assert any(line == f"ARCHIVE_BYTES = {len(b'fx1-candidate-bytes'):_d}" for line in after_lines)


def test_repin_is_idempotent_and_reports_no_change(tmp_path: Path) -> None:
    runtime = _stage(tmp_path, b"already-correct")
    result = repin_receiver(runtime)
    assert result.changed is False
    assert result.verdict_before == CONSISTENT
    assert result.verdict_after == CONSISTENT


def test_detector_does_not_zero_on_its_own_cure(tmp_path: Path) -> None:
    """After a re-pin, staging different bytes must go RED again — the gauge stays live."""
    runtime = _stage(tmp_path, b"first-candidate", pin_sha=RR4_SHA, pin_bytes=RR4_BYTES)
    repin_receiver(runtime)
    assert check_pin_consistency(runtime).verdict == CONSISTENT

    (runtime / "archive.zip").write_bytes(b"a-different-candidate-entirely")

    assert check_pin_consistency(runtime).verdict == MISMATCH


def test_repin_dry_run_leaves_the_receiver_untouched(tmp_path: Path) -> None:
    runtime = _stage(tmp_path, b"fx1-candidate-bytes", pin_sha=RR4_SHA, pin_bytes=RR4_BYTES)
    receiver = runtime / "inflate.py"
    original = receiver.read_bytes()

    result = repin_receiver(runtime, dry_run=True)

    assert result.dry_run is True
    assert result.changed is True
    assert receiver.read_bytes() == original
    assert check_pin_consistency(runtime).verdict == MISMATCH


def test_repin_refuses_an_unpinned_receiver(tmp_path: Path) -> None:
    """Adding a pin where none exists is a receiver design change, not a staging step."""
    runtime = tmp_path / "candidate_runtime"
    runtime.mkdir()
    (runtime / "archive.zip").write_bytes(b"unpinned")
    (runtime / "inflate.py").write_text("def main() -> None:\n    pass\n")

    with pytest.raises(SealContractError, match="declares no ARCHIVE_SHA256"):
        repin_receiver(runtime)


# --------------------------------------------------------------------------------------
# Reading the pin must never execute the receiver
# --------------------------------------------------------------------------------------


def test_pin_is_read_without_importing_the_receiver(tmp_path: Path) -> None:
    """A shipped receiver imports torch at module scope; the reader must never run that."""
    runtime = tmp_path / "candidate_runtime"
    runtime.mkdir()
    (runtime / "archive.zip").write_bytes(b"payload")
    receiver = runtime / "inflate.py"
    receiver.write_text(
        "import a_module_that_does_not_exist_anywhere\n"
        'raise SystemExit("importing this receiver must never happen")\n'
        'ARCHIVE_SHA256 = "deadbeef"\n'
        "ARCHIVE_BYTES = 7\n"
    )

    pin = read_receiver_pin(receiver)

    assert pin.archive_sha256 == "deadbeef"
    assert pin.archive_bytes == 7


def test_multiline_pin_assignment_is_not_treated_as_pinnable(tmp_path: Path) -> None:
    """A pin we cannot rewrite by single-line substitution must not be reported as rewritable."""
    receiver = tmp_path / "inflate.py"
    receiver.write_text('ARCHIVE_SHA256 = (\n    "abc"\n)\nARCHIVE_BYTES = 3\n')

    pin = read_receiver_pin(receiver)

    assert pin.archive_sha256 is None
    assert pin.archive_bytes == 3
    assert pin.is_present is False


# --------------------------------------------------------------------------------------
# The dynamic default: derive the bar at call time, never latch it
# --------------------------------------------------------------------------------------


def test_frontier_identity_is_read_from_the_pointer_at_call_time(tmp_path: Path) -> None:
    pointer = tmp_path / "canonical_frontier_pointer.json"
    pointer.write_text(
        json.dumps(
            {
                "our_local_frontier_contest_cuda": {
                    "archive_sha256": "a" * 64,
                    "extra": {"archive_bytes": 123_456},
                }
            }
        )
    )

    identity = read_frontier_archive_identity(pointer, axis="contest_cuda")
    assert identity.sha256 == "a" * 64
    assert identity.bytes == 123_456

    # Move the pointer; the derived bar moves with it. That is the whole property.
    pointer.write_text(
        json.dumps(
            {
                "our_local_frontier_contest_cuda": {
                    "archive_sha256": "b" * 64,
                    "extra": {"archive_bytes": 99},
                }
            }
        )
    )
    assert read_frontier_archive_identity(pointer, axis="contest_cuda").bytes == 99


def test_frontier_identity_refuses_a_missing_bar_rather_than_guessing(tmp_path: Path) -> None:
    pointer = tmp_path / "canonical_frontier_pointer.json"
    pointer.write_text(json.dumps({"our_local_frontier_contest_cuda": {"archive_sha256": "c" * 64}}))

    with pytest.raises(SealContractError, match="refusing to guess an admission bar"):
        read_frontier_archive_identity(pointer, axis="contest_cuda")

    with pytest.raises(SealContractError, match="unknown pointer axis"):
        read_frontier_archive_identity(pointer, axis="mps")


def test_live_pointer_supplies_a_usable_bar() -> None:
    """The repo's real pointer must answer the question the fire path will ask it."""
    identity = read_frontier_archive_identity()
    assert len(identity.sha256) == 64
    assert identity.bytes > 0
    assert identity.source.startswith("frontier_pointer:contest_cuda:")


# --------------------------------------------------------------------------------------
# The re-introduction gate: the byte-close entry point must not latch a candidate again
# --------------------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[3]
_PQ2 = _REPO / "experiments" / "ddm_pq2_compress_e2e.py"
_FIRE = _REPO / "tools" / "fire_modal_auth_eval.py"


def _module_level_constants(path: Path) -> set[str]:
    import ast

    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return names


def test_pq2_does_not_latch_an_expected_archive_identity_again() -> None:
    """The defect: a module-level expected sha/bytes refuses every other candidate by construction.

    The recipe may name rr4 — that describes the pipeline. The *expected archive identity*
    may not be a module constant, because that is the thing a caller must be able to change.
    """
    constants = _module_level_constants(_PQ2)

    assert "EXPECTED_ARCHIVE_SHA256" not in constants
    assert "EXPECTED_ARCHIVE_BYTES" not in constants

    source = _PQ2.read_text()
    for flag in ("--expected-archive-sha256", "--expected-archive-bytes", "--candidate-runtime", "--recipe-json"):
        assert flag in source, f"{_PQ2.name} must expose {flag} so the candidate is caller-supplied"
    assert "read_frontier_archive_identity" in source, "the default bar must be derived at run time"


def test_the_fire_path_consumes_the_seal_check() -> None:
    """Both seal bricks are only a cure if the one Modal fire path actually runs them.

    The refusal assertions are regexes, not substrings: the original ``"return 6" in source``
    matched a single formatting of the refusal call and went red the moment the line wrapped,
    which is a gate that fails on style instead of on the property it guards.
    """
    import re

    source = _FIRE.read_text()

    # Brick 1 — receiver-pin consistency, and it must REFUSE rather than merely report.
    assert "from tac.candidate_seal import" in source
    assert "check_pin_consistency" in source
    assert "repin_receiver" in source
    assert re.search(r"refuse\(\s*out_dir,\s*6,", source), "the pin MISMATCH branch must refuse with rc=6"

    # Brick 2 — the seal DOCUMENT, validated before any other stage and refused on its own rc.
    assert "validate_seal" in source
    assert re.search(r"refuse_seal\(\s*seal_path,\s*out_dir,\s*7,", source), "a seal refusal owns rc=7"
