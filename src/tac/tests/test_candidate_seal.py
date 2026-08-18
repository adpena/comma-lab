"""Controls for the candidate-seal DOCUMENT (brick 2, task #1115).

Every negative here is EXECUTED, not described: each measured failure class that motivated a
field gets a test that reproduces the drift on disk and asserts the validator refuses with the
named verdict. A gate whose red direction is never run is a gate that has only been imagined.

The consumer control is the load-bearing one. It drives the real ``tools/fire_modal_auth_eval.py``
``main()`` with a drifted seal and asserts the refusal happens with NO subprocess reached — the
paid call is not merely undesirable in tests, it must be structurally out of reach before the
seal is trusted.
"""

from __future__ import annotations

import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

from tac.candidate_seal import (
    SEAL_BAR_DRIFT,
    SEAL_BYTE_DRIFT,
    SEAL_FILE_MISSING,
    SEAL_PLACEHOLDER_PIN,
    SEAL_RUNTIME_DRIFT,
    SEAL_SCHEMA,
    SEAL_SCHEMA_VIOLATION,
    SEAL_SHA_DRIFT,
    SEAL_TAMPERED,
    SEAL_VALID,
    AdmitBar,
    SealContractError,
    build_seal,
    compute_seal_sha256,
    load_seal,
    measure_runtime_digest,
    validate_seal,
    write_seal,
)

REPO = Path(__file__).resolve().parents[3]
FIRE_TOOL = REPO / "tools" / "fire_modal_auth_eval.py"
MAKE_TOOL = REPO / "tools" / "make_candidate_seal.py"

POINTER_SCORE = 0.15771357797660338
POINTER_SHA = "debb025f45bb42e3b8131714cf462a9963e449bc65ff5eade9484fde094b037a"

RECEIVER_PY = '''#!/usr/bin/env python3
"""A stand-in receiver shaped like the shipped one."""

ARCHIVE_SHA256 = "{sha}"
ARCHIVE_BYTES = {size}


def main() -> None:
    return None
'''


def _write_pointer(tmp_path: Path, score: float = POINTER_SCORE, sha: str = POINTER_SHA) -> Path:
    """A synthetic frontier pointer. The bar's baseline is an INPUT to these tests, never the
    live repo state — a control that moves under you is not a control."""
    path = tmp_path / "canonical_frontier_pointer.json"
    path.write_text(
        json.dumps(
            {
                "our_local_frontier_contest_cuda": {
                    "score": score,
                    "archive_sha256": sha,
                    "axis": "contest_cuda",
                    "extra": {"archive_bytes": 179930},
                }
            }
        )
    )
    return path


def _stage_candidate(tmp_path: Path, payload: bytes = b"token-stream-bytes" * 64) -> tuple[Path, Path]:
    """Stage a runtime tree shaped like a real candidate_runtime: archive + receiver + shell."""
    runtime = tmp_path / "candidate_runtime"
    (runtime / "cpr1").mkdir(parents=True)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("0.bin", payload)
    archive = runtime / "archive.zip"
    archive.write_bytes(buffer.getvalue())

    import hashlib

    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    (runtime / "inflate.py").write_text(RECEIVER_PY.format(sha=sha, size=archive.stat().st_size))
    (runtime / "inflate.sh").write_text("#!/bin/sh\nexec python inflate.py \"$@\"\n")
    (runtime / "cpr1" / "semantic_receiver.py").write_text("DECODER = 'SM3R'\n")
    return runtime, archive


def _seal(tmp_path: Path, runtime: Path, **overrides) -> Path:
    retained = tmp_path / "retained" / "cand"
    retained.mkdir(parents=True, exist_ok=True)
    (retained / "tokens.ans").write_bytes(b"the payload we kept")

    bar = AdmitBar(
        rule="net dS < -3.5e-6",
        net_dS_threshold=-3.5e-6,
        pointer_axis="contest_cuda",
        pointer_score_at_seal=POINTER_SCORE,
        pointer_archive_sha256_at_seal=POINTER_SHA,
        pointer_tolerance_abs=overrides.pop("tolerance", 0.0),
    )
    document = build_seal(
        candidate_id=overrides.pop("candidate_id", "sm3r_keep01"),
        runtime_dir=runtime,
        axis=overrides.pop("axis", "contest_cuda"),
        admit_bar=bar,
        archive_member_name="0.bin",
        retained_payload_paths=(str(retained),),
        falsifiers=("net dS >= -3.5e-6 at n600 refutes the rate credit",),
        sealed_by="test",
        **overrides,
    )
    return write_seal(document, tmp_path / "SEAL.json")


# ----------------------------------------------------------------------------------
# POSITIVE
# ----------------------------------------------------------------------------------


def test_a_freshly_sealed_candidate_validates(tmp_path: Path) -> None:
    runtime, _ = _stage_candidate(tmp_path)
    seal_path = _seal(tmp_path, runtime)
    pointer = _write_pointer(tmp_path)

    verdict = validate_seal(seal_path, pointer_path=pointer)

    assert verdict.verdict == SEAL_VALID, verdict.summary()
    assert verdict.ok
    assert verdict.candidate_id == "sm3r_keep01"
    assert verdict.axis == "contest_cuda"
    document = load_seal(seal_path)
    assert document["schema"] == SEAL_SCHEMA
    assert document["seal_sha256"] == compute_seal_sha256(document)
    assert {pin["relative_path"] for pin in document["receiver_pins"]} == {"inflate.py", "inflate.sh"}
    assert document["archive_member"]["name"] == "0.bin"


def test_the_runtime_digest_is_invariant_under_the_fire_paths_sanitize_stage(tmp_path: Path) -> None:
    """The seal must survive its own consumer's first mutation, or it is a seal in name only.

    Stage 1 of the fire path deletes macOS ``._`` litter and Python bytecode caches. If those
    entered the digest, every sealed fire on an ExFAT custody volume would refuse itself.
    """
    runtime, _ = _stage_candidate(tmp_path)
    before = measure_runtime_digest(runtime)

    (runtime / "._inflate.py").write_bytes(b"\x00\x05\x16\x07AppleDouble")
    (runtime / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")
    (runtime / "cpr1" / "__pycache__").mkdir()
    (runtime / "cpr1" / "__pycache__" / "semantic_receiver.cpython-311.pyc").write_bytes(b"\x00cached")

    after_litter = measure_runtime_digest(runtime)
    assert after_litter.sha256 == before.sha256
    assert after_litter.file_count == before.file_count

    seal_path = _seal(tmp_path, runtime)
    pointer = _write_pointer(tmp_path)
    assert validate_seal(seal_path, pointer_path=pointer).verdict == SEAL_VALID

    for litter in ("._inflate.py", ".DS_Store"):
        (runtime / litter).unlink()
    assert validate_seal(seal_path, pointer_path=pointer).verdict == SEAL_VALID


# ----------------------------------------------------------------------------------
# EXECUTED NEGATIVE CONTROLS — one per measured failure class
# ----------------------------------------------------------------------------------


def test_a_drifted_archive_byte_refuses(tmp_path: Path) -> None:
    """rr2: the fired bytes were never the proved bytes. Same size, one bit different."""
    runtime, archive = _stage_candidate(tmp_path)
    seal_path = _seal(tmp_path, runtime)
    pointer = _write_pointer(tmp_path)
    assert validate_seal(seal_path, pointer_path=pointer).verdict == SEAL_VALID

    raw = bytearray(archive.read_bytes())
    raw[-20] ^= 0xFF
    archive.write_bytes(bytes(raw))

    verdict = validate_seal(seal_path, pointer_path=pointer)
    assert verdict.verdict == SEAL_SHA_DRIFT
    assert not verdict.ok
    assert "same size, different bytes" in verdict.summary()


def test_a_resized_archive_refuses_on_bytes(tmp_path: Path) -> None:
    runtime, archive = _stage_candidate(tmp_path)
    seal_path = _seal(tmp_path, runtime)
    pointer = _write_pointer(tmp_path)

    archive.write_bytes(archive.read_bytes() + b"trailing")

    assert validate_seal(seal_path, pointer_path=pointer).verdict == SEAL_BYTE_DRIFT


def test_an_edited_receiver_refuses_and_names_the_file(tmp_path: Path) -> None:
    """ps1u r1: a receiver sha pin drifted between seal and fire."""
    runtime, _ = _stage_candidate(tmp_path)
    seal_path = _seal(tmp_path, runtime)
    pointer = _write_pointer(tmp_path)
    assert validate_seal(seal_path, pointer_path=pointer).verdict == SEAL_VALID

    receiver = runtime / "inflate.py"
    receiver.write_text(receiver.read_text() + "\n# a one-line 'harmless' edit\n")

    verdict = validate_seal(seal_path, pointer_path=pointer)
    assert verdict.verdict == SEAL_RUNTIME_DRIFT
    assert "inflate.py" in verdict.summary(), "a drift report that cannot name the file is unactionable"


def test_an_edited_unpinned_shipped_file_still_refuses_at_tree_level(tmp_path: Path) -> None:
    """The per-file pins cover the receivers; the tree digest covers everything else that ships."""
    runtime, _ = _stage_candidate(tmp_path)
    seal_path = _seal(tmp_path, runtime)
    pointer = _write_pointer(tmp_path)

    (runtime / "cpr1" / "semantic_receiver.py").write_text("DECODER = 'SOMETHING_ELSE'\n")

    verdict = validate_seal(seal_path, pointer_path=pointer)
    assert verdict.verdict == SEAL_RUNTIME_DRIFT
    assert "unpinned shipped file" in verdict.summary()


def test_a_moved_pointer_beyond_tolerance_refuses(tmp_path: Path) -> None:
    """qs4: a bar carried onto a different regime. The baseline moved; the bar is stale."""
    runtime, _ = _stage_candidate(tmp_path)
    seal_path = _seal(tmp_path, runtime, tolerance=1e-6)

    assert validate_seal(seal_path, pointer_path=_write_pointer(tmp_path)).verdict == SEAL_VALID

    within = _write_pointer(tmp_path, score=POINTER_SCORE + 5e-7)
    assert validate_seal(seal_path, pointer_path=within).verdict == SEAL_VALID, "tolerance must be usable"

    beyond = _write_pointer(tmp_path, score=POINTER_SCORE - 1e-3)
    verdict = validate_seal(seal_path, pointer_path=beyond)
    assert verdict.verdict == SEAL_BAR_DRIFT
    assert "baseline moved" in verdict.summary()


def test_a_pointer_that_names_a_different_candidate_refuses(tmp_path: Path) -> None:
    runtime, _ = _stage_candidate(tmp_path)
    seal_path = _seal(tmp_path, runtime, tolerance=1.0)

    swapped = _write_pointer(tmp_path, sha="a" * 64)
    verdict = validate_seal(seal_path, pointer_path=swapped)

    assert verdict.verdict == SEAL_BAR_DRIFT
    assert "DIFFERENT candidate" in verdict.summary()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("candidate_id",), "TBD"),
        (("candidate_id",), ""),
        (("archive", "sha256"), "pending_ratification"),
        (("archive", "sha256"), "0" * 64),
        (("archive", "sha256"), "not-a-sha"),
        (("runtime", "sha256"), "<value>"),
        (("admit_bar", "derivation", "pointer_archive_sha256_at_seal"), "TBD"),
    ],
)
def test_a_placeholder_pin_refuses(tmp_path: Path, path: tuple[str, ...], value: str) -> None:
    """Catalog #287 lifted from waiver rationales to data pins: a stand-in is not a value."""
    runtime, _ = _stage_candidate(tmp_path)
    seal_path = _seal(tmp_path, runtime)
    document = load_seal(seal_path)

    node = document
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    # Re-sign, so the refusal is provably about the placeholder and not about tampering.
    document["seal_sha256"] = compute_seal_sha256(document)
    seal_path.write_text(json.dumps(document, indent=2))

    verdict = validate_seal(seal_path, pointer_path=_write_pointer(tmp_path))
    assert verdict.verdict == SEAL_PLACEHOLDER_PIN, verdict.summary()
    assert ".".join(path) in verdict.summary() or path[-1] in verdict.summary()


def test_an_empty_receiver_pin_list_refuses(tmp_path: Path) -> None:
    runtime, _ = _stage_candidate(tmp_path)
    seal_path = _seal(tmp_path, runtime)
    document = load_seal(seal_path)
    document["receiver_pins"] = []
    document["seal_sha256"] = compute_seal_sha256(document)
    seal_path.write_text(json.dumps(document))

    verdict = validate_seal(seal_path, pointer_path=_write_pointer(tmp_path))
    assert verdict.verdict == SEAL_PLACEHOLDER_PIN
    assert "unpinned receiver" in verdict.summary()


def test_an_edited_seal_refuses_as_tampered(tmp_path: Path) -> None:
    """The signature is what makes every other field trustworthy."""
    runtime, _ = _stage_candidate(tmp_path)
    seal_path = _seal(tmp_path, runtime)
    document = load_seal(seal_path)
    document["admit_bar"]["net_dS_threshold"] = 1.0  # "just relax the bar a little"
    seal_path.write_text(json.dumps(document, indent=2))

    verdict = validate_seal(seal_path, pointer_path=_write_pointer(tmp_path))
    assert verdict.verdict == SEAL_TAMPERED
    assert "edited after it was signed" in verdict.summary()


def test_a_missing_archive_or_runtime_refuses(tmp_path: Path) -> None:
    runtime, archive = _stage_candidate(tmp_path)
    seal_path = _seal(tmp_path, runtime)
    pointer = _write_pointer(tmp_path)
    archive.unlink()

    verdict = validate_seal(seal_path, pointer_path=pointer)
    assert verdict.verdict == SEAL_FILE_MISSING
    assert "sealed archive is gone" in verdict.summary()


def test_a_vanished_retained_payload_refuses(tmp_path: Path) -> None:
    """ALWAYS KEEP THE PAYLOAD, checked rather than asserted."""
    runtime, _ = _stage_candidate(tmp_path)
    seal_path = _seal(tmp_path, runtime)
    pointer = _write_pointer(tmp_path)
    assert validate_seal(seal_path, pointer_path=pointer).verdict == SEAL_VALID

    document = load_seal(seal_path)
    custody = Path(document["retained_payload_paths"][0])
    for child in custody.iterdir():
        child.unlink()
    custody.rmdir()

    verdict = validate_seal(seal_path, pointer_path=pointer)
    assert verdict.verdict == SEAL_FILE_MISSING
    assert "retained payload custody" in verdict.summary()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("archive", "35ac2b9beb7e6fa81075c7d84b5247d8d24c056fe49ce1cbd22a334bc9618956"),
        ("runtime", ["not", "an", "object"]),
        ("admit_bar", -3.5e-6),
        ("receiver_pins", {"inflate.py": "abc"}),
        ("retained_payload_paths", "/a/single/path/not/a/list"),
        ("schema", None),
        ("schema", "candidate_seal.v2"),
    ],
)
def test_a_wrongly_typed_field_refuses_instead_of_crashing(tmp_path: Path, field: str, value: object) -> None:
    """A malformed seal must REFUSE, never raise into the fire path.

    Found on the second review pass, not by a failing run: the placeholder scan skips
    non-objects, so a ``archive`` that was a bare sha string would have reached the drift
    comparison and raised AttributeError — a crash where a refusal belongs, and a traceback a
    reader can mistake for a tooling bug rather than a bad seal.
    """
    runtime, _ = _stage_candidate(tmp_path)
    seal_path = _seal(tmp_path, runtime)
    document = load_seal(seal_path)
    document[field] = value
    document["seal_sha256"] = compute_seal_sha256(document)
    seal_path.write_text(json.dumps(document))

    verdict = validate_seal(seal_path, pointer_path=_write_pointer(tmp_path))
    assert verdict.verdict == SEAL_SCHEMA_VIOLATION, verdict.summary()
    assert field in verdict.summary()


def test_a_missing_required_field_refuses_as_schema_violation(tmp_path: Path) -> None:
    runtime, _ = _stage_candidate(tmp_path)
    seal_path = _seal(tmp_path, runtime)
    document = load_seal(seal_path)
    del document["admit_bar"]
    seal_path.write_text(json.dumps(document))

    verdict = validate_seal(seal_path, pointer_path=_write_pointer(tmp_path))
    assert verdict.verdict == SEAL_SCHEMA_VIOLATION
    assert "admit_bar" in verdict.summary()


def test_sealing_refuses_a_receiver_that_does_not_ship(tmp_path: Path) -> None:
    """The producer must not pin a file the transport zip will never carry."""
    runtime, _ = _stage_candidate(tmp_path)
    with pytest.raises(SealContractError, match="not in the shipped file set"):
        _seal(tmp_path, runtime, receiver_relative_paths=("inflate.py", "does_not_exist.py"))


def test_sealing_refuses_a_placeholder_candidate_id(tmp_path: Path) -> None:
    runtime, _ = _stage_candidate(tmp_path)
    with pytest.raises(SealContractError, match="placeholder"):
        _seal(tmp_path, runtime, candidate_id="TBD")


# ----------------------------------------------------------------------------------
# CONSUMER INTEGRATION — the fire path must refuse BEFORE any dispatch
# ----------------------------------------------------------------------------------


def _load_fire_tool():
    spec = importlib.util.spec_from_file_location("fire_modal_auth_eval_under_test", FIRE_TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _forbid_subprocess(monkeypatch, module) -> None:
    """Make a paid call structurally unreachable, so 'no dispatch' is proved, not hoped."""

    def _explode(*args, **kwargs):  # pragma: no cover - firing here is the failure
        raise AssertionError(f"the fire path reached a subprocess on a refused seal: {args!r}")

    monkeypatch.setattr(module.subprocess, "run", _explode)


def test_the_fire_path_refuses_a_drifted_seal_without_dispatching(tmp_path: Path, monkeypatch) -> None:
    runtime, archive = _stage_candidate(tmp_path)
    seal_path = _seal(tmp_path, runtime, tolerance=1.0)
    module = _load_fire_tool()
    _forbid_subprocess(monkeypatch, module)

    raw = bytearray(archive.read_bytes())
    raw[-20] ^= 0xFF
    archive.write_bytes(bytes(raw))

    out_dir = tmp_path / "out"
    rc = module.main(
        [
            "--seal", str(seal_path),
            "--output-dir", str(out_dir),
            "--lane-id", "lane_test",
            "--instance-job-id", "job_test",
        ]
    )

    assert rc == 7, "a seal refusal needs its own exit code, distinguishable from every other stage"
    receipt = seal_path.with_name(seal_path.name + ".REFUSED.json")
    assert receipt.is_file(), "the refusal must survive whatever the shell does with the exit status"
    payload = json.loads(receipt.read_text())
    assert payload["refusal_rc"] == 7
    assert payload["seal_validation"]["verdict"] == SEAL_SHA_DRIFT
    assert (out_dir / "FIRE_REFUSED.json").is_file()
    assert not (out_dir / "FIRE_MANIFEST.json").exists()


def test_the_fire_path_refuses_hand_typed_duplicates_of_sealed_values(tmp_path: Path, monkeypatch) -> None:
    """Two sources for one truth is the hand-assembly hazard the seal exists to remove."""
    runtime, _ = _stage_candidate(tmp_path)
    seal_path = _seal(tmp_path, runtime)
    module = _load_fire_tool()
    _forbid_subprocess(monkeypatch, module)

    rc = module.main(
        [
            "--seal", str(seal_path),
            "--runtime-dir", str(runtime),
            "--output-dir", str(tmp_path / "out"),
            "--lane-id", "lane_test",
            "--instance-job-id", "job_test",
        ]
    )

    assert rc == 7
    payload = json.loads(seal_path.with_name(seal_path.name + ".REFUSED.json").read_text())
    assert "--runtime-dir" in payload["refusal_reason"]


def test_the_fire_path_refuses_a_repin_against_a_seal(tmp_path: Path, monkeypatch) -> None:
    """--repin-receiver mutates the receiver, invalidating the seal it was handed."""
    runtime, _ = _stage_candidate(tmp_path)
    seal_path = _seal(tmp_path, runtime)
    module = _load_fire_tool()
    _forbid_subprocess(monkeypatch, module)

    rc = module.main(
        [
            "--seal", str(seal_path),
            "--repin-receiver",
            "--output-dir", str(tmp_path / "out"),
            "--lane-id", "lane_test",
            "--instance-job-id", "job_test",
        ]
    )

    assert rc == 7
    payload = json.loads(seal_path.with_name(seal_path.name + ".REFUSED.json").read_text())
    assert "invalidate" in payload["refusal_reason"]


def test_the_fire_path_refuses_an_advisory_seal(tmp_path: Path, monkeypatch) -> None:
    """A paid Modal row is contest evidence; an advisory seal would mislabel its axis."""
    runtime, _ = _stage_candidate(tmp_path)
    seal_path = _seal(tmp_path, runtime, axis="advisory")
    module = _load_fire_tool()
    _forbid_subprocess(monkeypatch, module)

    rc = module.main(
        [
            "--seal", str(seal_path),
            "--output-dir", str(tmp_path / "out"),
            "--lane-id", "lane_test",
            "--instance-job-id", "job_test",
        ]
    )

    assert rc == 7
    payload = json.loads(seal_path.with_name(seal_path.name + ".REFUSED.json").read_text())
    assert "advisory" in payload["refusal_reason"]


def test_the_seal_axis_selects_the_worker_entrypoint(tmp_path: Path) -> None:
    """The axis is DERIVED from the seal, never hand-supplied at fire time."""
    module = _load_fire_tool()

    assert set(module.SEAL_AXIS_TO_FIRE_AXIS) == {"contest_cuda", "contest_cpu"}
    assert "advisory" not in module.SEAL_AXIS_TO_FIRE_AXIS
    assert module.axis_spec(module.SEAL_AXIS_TO_FIRE_AXIS["contest_cpu"])["evidence_axis_tag"] == "[contest-CPU]"
    assert module.axis_spec(module.SEAL_AXIS_TO_FIRE_AXIS["contest_cuda"])["evidence_axis_tag"] == "[contest-CUDA]"


def test_the_no_seal_path_is_unchanged(tmp_path: Path, monkeypatch) -> None:
    """Backward compatibility: existing invocations must keep working, on the cuda default."""
    runtime, _ = _stage_candidate(tmp_path)
    module = _load_fire_tool()
    monkeypatch.setattr(module, "reconcile_claims", lambda *a, **k: {"closed": []})

    out_dir = tmp_path / "out"
    rc = module.main(
        [
            "--runtime-dir", str(runtime),
            "--output-dir", str(out_dir),
            "--lane-id", "lane_test",
            "--instance-job-id", "job_test",
            "--dry-run",
        ]
    )

    assert rc == 0
    manifest = json.loads((out_dir / "FIRE_MANIFEST.json").read_text())
    assert manifest["axis"] == "cuda"
    assert manifest["evidence_axis_tag"] == "[contest-CUDA]"
    assert "seal_path" not in manifest


def test_the_producer_cli_seals_and_validates_its_own_output(tmp_path: Path, monkeypatch) -> None:
    """The producer runs the CONSUMER's gate on what it just wrote, or deletes it."""
    spec = importlib.util.spec_from_file_location("make_candidate_seal_under_test", MAKE_TOOL)
    make = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(make)

    runtime, archive = _stage_candidate(tmp_path)
    pointer = _write_pointer(tmp_path)
    monkeypatch.setattr(
        make, "read_pointer_state", lambda axis="contest_cuda": make_pointer_state(pointer, axis)
    )

    out = tmp_path / "SEAL_produced.json"
    rc = make.main(
        [
            "--candidate-id", "sm3r_keep01",
            "--runtime-dir", str(runtime),
            "--axis", "contest_cuda",
            "--admit-bar-net-ds", "-3.5e-6",
            "--archive-member", "0.bin",
            "--out", str(out),
        ]
    )

    assert rc == 0
    assert validate_seal(out, pointer_path=pointer).verdict == SEAL_VALID

    import hashlib

    measured = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert load_seal(out)["archive"]["sha256"] == measured, "the producer must hash disk, not accept a claim"

    # A hand-held expectation is CHECKED, never stored.
    assert (
        make.main(
            [
                "--candidate-id", "sm3r_keep01",
                "--runtime-dir", str(runtime),
                "--axis", "contest_cuda",
                "--admit-bar-net-ds", "-3.5e-6",
                "--verify-archive-sha", "b" * 64,
                "--out", str(tmp_path / "SEAL_rejected.json"),
            ]
        )
        == 4
    )
    assert not (tmp_path / "SEAL_rejected.json").exists()


def make_pointer_state(pointer_path: Path, axis: str) -> dict:
    from tac.candidate_seal import read_pointer_state

    return read_pointer_state(pointer_path=pointer_path, axis=axis)
