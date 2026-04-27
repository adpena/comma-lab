"""Codex R5-2 #4 regression: Lane C δ approval requires external attestation.

Trust model the gate enforces:
  1. The optimizer (experiments/optimize_uniward_delta.py) can only issue
     ``compliance_status`` ∈ {pending_ruling, rejected}. ``approved`` is
     refused at argparse.
  2. To bundle a δ as approved, an attestation file must exist at
     ``.omx/state/lane_c_compliance_attestations/<sha256>.json`` matching
     the actual δ.bin sha256.
  3. The signing tool (tools/sign_lane_c_compliance.py) refuses empty
     ruling text and empty approver.
  4. The archive builder (experiments/build_baseline_archive.py) rejects
     bundling an approved δ that has no matching attestation, even with
     ``--allow-pending-compliance`` (which only handles pending).

These tests pin all four layers. The "subprocess test the actual archive
builder" cases work without CUDA / GT video by hitting the script with
``--with-uniward-delta`` pointing at a fake-but-validly-packed δ and
expecting the gate to fail FAST, before it tries to load SegNet.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
OPTIMIZER_SCRIPT = REPO / "experiments" / "optimize_uniward_delta.py"
ARCHIVE_SCRIPT = REPO / "experiments" / "build_baseline_archive.py"
SIGN_SCRIPT = REPO / "tools" / "sign_lane_c_compliance.py"


# ---------------------------------------------------------------------------
# Helpers — build a real-shape, in-memory δ.bin for gate testing.
# ---------------------------------------------------------------------------


def _build_delta_blob(
    *, compliance_status: str = "pending_ruling",
    n_frames: int = 4, h: int = 8, w: int = 8,
) -> bytes:
    """Build a tiny but well-formed UWD1 δ.bin blob for gate testing.

    Avoids a full Lane C run (which requires CUDA + a 5GB scorer + GT
    video). All we need is a blob whose header parses, so the archive
    builder can read its compliance_status and hit the gate.
    """
    import torch
    from tac.uniward_delta import pack_sparse_delta
    delta = torch.randn(n_frames, 3, h, w) * 2.0  # within budget
    cost = torch.rand(n_frames, h, w) * 10.0
    return pack_sparse_delta(
        delta, cost,
        l_inf_budget=4.0,
        target_bytes=1024,
        seed=1234,
        compliance_status=compliance_status,
    )


def _replace_pyproject_path() -> Path:
    """Path to the project pyproject.toml (used for env var setup in
    subprocess tests, so the script can import tac.* from src/)."""
    return REPO / "pyproject.toml"


# ---------------------------------------------------------------------------
# Layer A — optimizer cannot issue 'approved'
# ---------------------------------------------------------------------------


def test_optimizer_cannot_issue_approved_via_argparse() -> None:
    """The optimizer's --compliance-status MUST refuse 'approved'. This is
    the first line of defense against operator self-asserted approval."""
    # We use --help-style introspection (running --help would also work
    # but is slower). Instead, run with --compliance-status approved and
    # an otherwise-incomplete invocation; argparse rejects the choice
    # BEFORE checking required args.
    proc = subprocess.run(
        [sys.executable, str(OPTIMIZER_SCRIPT),
         "--compliance-status", "approved"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode != 0, (
        "optimize_uniward_delta.py accepted --compliance-status approved. "
        "It MUST refuse — only pending_ruling and rejected are issuable. "
        f"stderr: {proc.stderr}"
    )
    # argparse error message format: "invalid choice: 'approved'"
    combined = proc.stdout + proc.stderr
    assert "approved" in combined and "invalid choice" in combined.lower(), (
        "Expected argparse to reject 'approved' with an 'invalid choice' "
        f"message. Got:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )


def test_optimizer_help_documents_attestation_path() -> None:
    """--help must point at the attestation tool so an operator who tries
    'approved' immediately sees the supported workflow."""
    proc = subprocess.run(
        [sys.executable, str(OPTIMIZER_SCRIPT), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, f"--help exited {proc.returncode}: {proc.stderr}"
    assert "sign_lane_c_compliance" in proc.stdout, (
        "Optimizer --help must reference tools/sign_lane_c_compliance.py "
        "so an operator who hits the 'approved' refusal knows the next "
        "step. Otherwise the gate is just a wall."
    )


def test_optimizer_argparse_choices_only_pending_and_rejected() -> None:
    """Source-level pin: the --compliance-status choices list must contain
    exactly pending_ruling and rejected — adding 'approved' to the list
    re-opens the gate."""
    src = OPTIMIZER_SCRIPT.read_text()
    # Find the --compliance-status add_argument call.
    import re
    match = re.search(
        r'add_argument\(\s*"--compliance-status".*?\)',
        src, flags=re.DOTALL,
    )
    assert match is not None, "Could not locate --compliance-status argparse"
    block = match.group(0)
    assert '"pending_ruling"' in block, "pending_ruling missing from choices"
    assert '"rejected"' in block, "rejected missing from choices"
    # Critical: 'approved' must not appear in the choices list. We allow
    # it to appear in the help text (as a forbidden value).
    choices_match = re.search(r'choices=\[(.*?)\]', block)
    assert choices_match is not None, "Could not parse choices list"
    choices_block = choices_match.group(1)
    assert '"approved"' not in choices_block, (
        "'approved' is in the optimizer's --compliance-status choices. "
        "This re-opens the operator-self-asserted approval bypass that "
        "Codex R5-2 #4 closed."
    )


# ---------------------------------------------------------------------------
# Layer B — attestation primitives
# ---------------------------------------------------------------------------


def test_attestation_path_is_sha_keyed(tmp_path: Path) -> None:
    """Two different δ.bin blobs MUST resolve to two different attestation
    paths. Same blob ⇒ same path."""
    from tac.lane_c_compliance import attestation_path_for, compute_blob_sha256
    a = b"alpha-delta-bytes" * 32
    b = b"beta-delta-bytes" * 32
    pa = attestation_path_for(compute_blob_sha256(a), root=tmp_path)
    pb = attestation_path_for(compute_blob_sha256(b), root=tmp_path)
    assert pa != pb, "Two distinct blobs hit the same attestation path"
    pa2 = attestation_path_for(compute_blob_sha256(a), root=tmp_path)
    assert pa == pa2, "Same blob produced different attestation paths"
    assert pa.name.endswith(".json"), "Attestation file must be .json"
    assert ".omx/state/lane_c_compliance_attestations" in str(pa), (
        f"Attestation must live under canonical .omx path; got {pa}"
    )


def test_attestation_path_rejects_malformed_sha() -> None:
    from tac.lane_c_compliance import attestation_path_for
    with pytest.raises(ValueError):
        attestation_path_for("not-a-hex-sha")
    with pytest.raises(ValueError):
        attestation_path_for("abc" * 21)  # wrong length


def test_write_attestation_refuses_empty_ruling(tmp_path: Path) -> None:
    """Even at the lowest layer, empty ruling_text is refused. If the
    layer-A signing tool ever forgets, the library still refuses."""
    from tac.lane_c_compliance import write_attestation
    blob = b"some-delta-bytes" * 32
    with pytest.raises(ValueError, match="ruling_text"):
        write_attestation(
            blob=blob, approver="yousfi", ruling_text="",
            signed_by_user="adpena", git_head="abc123",
            root=tmp_path,
        )
    with pytest.raises(ValueError, match="ruling_text"):
        write_attestation(
            blob=blob, approver="yousfi", ruling_text="    ",
            signed_by_user="adpena", git_head="abc123",
            root=tmp_path,
        )


def test_write_attestation_refuses_empty_approver(tmp_path: Path) -> None:
    from tac.lane_c_compliance import write_attestation
    blob = b"some-delta-bytes" * 32
    with pytest.raises(ValueError, match="approver"):
        write_attestation(
            blob=blob, approver="", ruling_text="real ruling",
            signed_by_user="adpena", git_head="abc123",
            root=tmp_path,
        )


def test_write_then_verify_round_trip(tmp_path: Path) -> None:
    """Happy path: write attestation, then verify_attestation_for_blob
    returns the parsed Attestation."""
    from tac.lane_c_compliance import write_attestation, verify_attestation_for_blob
    blob = b"some-delta-bytes" * 32
    path = write_attestation(
        blob=blob, approver="yousfi",
        ruling_text="PR #35 approved 2026-04-27",
        signed_by_user="adpena", git_head="abc123def",
        root=tmp_path,
    )
    assert path.exists()
    att = verify_attestation_for_blob(blob, root=tmp_path)
    assert att.approver == "yousfi"
    assert att.ruling_text == "PR #35 approved 2026-04-27"
    assert att.signed_by_user == "adpena"
    assert att.git_head == "abc123def"


def test_verify_raises_missing_when_no_file(tmp_path: Path) -> None:
    from tac.lane_c_compliance import verify_attestation_for_blob, AttestationMissing
    blob = b"unsigned-delta" * 32
    with pytest.raises(AttestationMissing):
        verify_attestation_for_blob(blob, root=tmp_path)


def test_verify_raises_mismatch_on_sha_drift(tmp_path: Path) -> None:
    """The SHA cross-check is the load-bearing belt: even if someone
    placed an attestation file at the canonical path for blob A,
    verifying blob B against the same path must fail."""
    from tac.lane_c_compliance import (
        write_attestation, verify_attestation_for_blob, AttestationMismatch,
        attestation_path_for, compute_blob_sha256,
    )
    blob_a = b"approved-delta-A" * 32
    blob_b = b"unauthorized-delta-B" * 32
    # Sign A.
    write_attestation(
        blob=blob_a, approver="yousfi",
        ruling_text="approved", signed_by_user="adpena",
        git_head="abc", root=tmp_path,
    )
    # Move A's attestation to B's expected path. (Simulates someone
    # copy-pasting an attestation between sha-keyed slots.)
    sha_a = compute_blob_sha256(blob_a)
    sha_b = compute_blob_sha256(blob_b)
    src = attestation_path_for(sha_a, root=tmp_path)
    dst = attestation_path_for(sha_b, root=tmp_path)
    src.rename(dst)
    # Now: dst.name is sha_b but the file's delta_sha256 is still sha_a.
    with pytest.raises(AttestationMismatch):
        verify_attestation_for_blob(blob_b, root=tmp_path)


def test_verify_raises_malformed_on_missing_fields(tmp_path: Path) -> None:
    from tac.lane_c_compliance import (
        verify_attestation_for_blob, AttestationMalformed,
        attestation_path_for, compute_blob_sha256,
    )
    blob = b"delta-with-broken-attestation" * 32
    sha = compute_blob_sha256(blob)
    path = attestation_path_for(sha, root=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Missing ruling_text + approver.
    path.write_text(json.dumps({
        "schema_version": 1,
        "delta_sha256": sha,
        "signed_at_utc": "2026-04-27T00:00:00Z",
        "signed_by_user": "adpena",
        "git_head": "abc",
    }))
    with pytest.raises(AttestationMalformed):
        verify_attestation_for_blob(blob, root=tmp_path)


def test_verify_raises_malformed_on_wrong_schema(tmp_path: Path) -> None:
    from tac.lane_c_compliance import (
        verify_attestation_for_blob, AttestationMalformed,
        attestation_path_for, compute_blob_sha256,
    )
    blob = b"delta-with-future-schema" * 32
    sha = compute_blob_sha256(blob)
    path = attestation_path_for(sha, root=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": 999,
        "delta_sha256": sha,
        "approver": "yousfi",
        "ruling_text": "ok",
        "signed_at_utc": "2026-04-27T00:00:00Z",
        "signed_by_user": "adpena",
        "git_head": "abc",
    }))
    with pytest.raises(AttestationMalformed):
        verify_attestation_for_blob(blob, root=tmp_path)


def test_write_refuses_overwrite_by_default(tmp_path: Path) -> None:
    from tac.lane_c_compliance import write_attestation
    blob = b"delta-being-overwritten" * 32
    write_attestation(
        blob=blob, approver="yousfi", ruling_text="first",
        signed_by_user="adpena", git_head="abc", root=tmp_path,
    )
    with pytest.raises(FileExistsError):
        write_attestation(
            blob=blob, approver="fridrich", ruling_text="second",
            signed_by_user="adpena", git_head="def", root=tmp_path,
        )
    # overwrite=True works.
    write_attestation(
        blob=blob, approver="fridrich", ruling_text="second",
        signed_by_user="adpena", git_head="def", root=tmp_path,
        overwrite=True,
    )


# ---------------------------------------------------------------------------
# Layer A — signing tool CLI
# ---------------------------------------------------------------------------


def test_sign_tool_refuses_empty_ruling(tmp_path: Path) -> None:
    """The signing tool MUST refuse empty --ruling-text. Without ruling
    text, the attestation has no forensic value."""
    delta = tmp_path / "delta.bin"
    delta.write_bytes(b"x" * 256)
    proc = subprocess.run(
        [sys.executable, str(SIGN_SCRIPT),
         "--delta-bin", str(delta),
         "--approver", "yousfi",
         "--ruling-text", "",
         "--root", str(tmp_path)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode != 0, (
        "Signing tool accepted empty --ruling-text. The trust gate "
        f"requires non-empty ruling. stderr: {proc.stderr}"
    )
    assert "ruling-text is empty" in (proc.stdout + proc.stderr).lower() or \
           "ruling_text" in (proc.stdout + proc.stderr).lower(), (
        "Expected an actionable error mentioning the empty ruling. "
        f"Got:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )


def test_sign_tool_refuses_empty_approver(tmp_path: Path) -> None:
    delta = tmp_path / "delta.bin"
    delta.write_bytes(b"x" * 256)
    proc = subprocess.run(
        [sys.executable, str(SIGN_SCRIPT),
         "--delta-bin", str(delta),
         "--approver", "",
         "--ruling-text", "ruling",
         "--root", str(tmp_path)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode != 0
    assert "approver" in (proc.stdout + proc.stderr).lower()


def test_sign_tool_refuses_empty_delta(tmp_path: Path) -> None:
    delta = tmp_path / "delta.bin"
    delta.write_bytes(b"")
    proc = subprocess.run(
        [sys.executable, str(SIGN_SCRIPT),
         "--delta-bin", str(delta),
         "--approver", "yousfi",
         "--ruling-text", "ruling",
         "--root", str(tmp_path)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode != 0
    assert "empty" in (proc.stdout + proc.stderr).lower()


def test_sign_tool_refuses_missing_delta(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(SIGN_SCRIPT),
         "--delta-bin", str(tmp_path / "nope.bin"),
         "--approver", "yousfi",
         "--ruling-text", "ruling",
         "--root", str(tmp_path)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode != 0
    assert "does not exist" in (proc.stdout + proc.stderr).lower()


def test_sign_tool_happy_path_writes_attestation(tmp_path: Path) -> None:
    delta = tmp_path / "delta.bin"
    delta.write_bytes(b"y" * 1024)
    proc = subprocess.run(
        [sys.executable, str(SIGN_SCRIPT),
         "--delta-bin", str(delta),
         "--approver", "yousfi",
         "--ruling-text", "PR #35: Lane C δ approved 2026-04-27",
         "--root", str(tmp_path),
         "--signed-by-user", "test-user"],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, (
        f"Signing tool failed unexpectedly. "
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    sha = hashlib.sha256(b"y" * 1024).hexdigest()
    att_path = tmp_path / ".omx" / "state" / "lane_c_compliance_attestations" / f"{sha}.json"
    assert att_path.exists(), f"Attestation not written at {att_path}"
    record = json.loads(att_path.read_text())
    assert record["delta_sha256"] == sha
    assert record["approver"] == "yousfi"
    assert "PR #35" in record["ruling_text"]
    assert record["signed_by_user"] == "test-user"


def test_sign_tool_refuses_overwrite_without_flag(tmp_path: Path) -> None:
    delta = tmp_path / "delta.bin"
    delta.write_bytes(b"z" * 512)
    base_args = [
        sys.executable, str(SIGN_SCRIPT),
        "--delta-bin", str(delta),
        "--approver", "yousfi",
        "--ruling-text", "first",
        "--root", str(tmp_path),
        "--signed-by-user", "test-user",
    ]
    p1 = subprocess.run(base_args, capture_output=True, text=True, timeout=30)
    assert p1.returncode == 0
    # Second run without --overwrite must fail.
    p2 = subprocess.run(base_args, capture_output=True, text=True, timeout=30)
    assert p2.returncode != 0, "Second sign without --overwrite should fail"
    assert "already exists" in (p2.stdout + p2.stderr).lower() or \
           "overwrite" in (p2.stdout + p2.stderr).lower()


# ---------------------------------------------------------------------------
# Layer B — archive builder gate (subprocess)
# ---------------------------------------------------------------------------


def _archive_subprocess(
    tmp_path: Path, delta_path: Path, *, allow_pending: bool = False,
) -> subprocess.CompletedProcess:
    """Invoke build_baseline_archive.py with a δ but otherwise-impossible
    inputs. The compliance gate fires BEFORE the script tries to load
    SegNet, so we use a fake renderer/poses/gt-video pointing at small
    placeholder files. The gate either fails fast (the case we want to
    test) or proceeds to the next stage and crashes on something else
    (which we'd see as a non-zero exit but a different error message).
    """
    fake_renderer = tmp_path / "renderer.bin"
    fake_renderer.write_bytes(b"fake")
    fake_poses = tmp_path / "poses.pt"
    fake_poses.write_bytes(b"fake")
    fake_gt = tmp_path / "gt.mkv"
    fake_gt.write_bytes(b"fake")
    fake_out = tmp_path / "archive.zip"
    cmd = [
        sys.executable, str(ARCHIVE_SCRIPT),
        "--renderer", str(fake_renderer),
        "--poses", str(fake_poses),
        "--gt-video", str(fake_gt),
        "--output", str(fake_out),
        "--with-uniward-delta", str(delta_path),
    ]
    if allow_pending:
        cmd.append("--allow-pending-compliance")
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60)


def test_archive_builder_refuses_pending_without_flag(tmp_path: Path) -> None:
    """Sanity check the existing R5 HIGH gate still fires: a pending-ruling
    δ without --allow-pending-compliance is refused."""
    delta_path = tmp_path / "delta.bin"
    delta_path.write_bytes(_build_delta_blob(compliance_status="pending_ruling"))
    proc = _archive_subprocess(tmp_path, delta_path)
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "pending_ruling" in combined or "PENDING" in combined.upper()


def test_archive_builder_refuses_approved_without_attestation(tmp_path: Path) -> None:
    """CODEX R5-2 #4 core regression: a δ.bin with header
    compliance_status='approved' but NO matching attestation file MUST
    be refused. Belt-and-braces: even passing --allow-pending-compliance
    must not save it (that flag handles pending only)."""
    # Forge a δ.bin with approved status (the optimizer can't issue
    # this, but we can via the library directly to test the gate). This
    # simulates an attacker / mistaken operator producing a
    # self-asserted-approved δ.bin.
    delta_path = tmp_path / "delta.bin"
    delta_path.write_bytes(_build_delta_blob(compliance_status="approved"))

    # No attestation in the canonical path (REPO/.omx/...). The gate
    # checks REPO, not tmp_path, so we know there's no real attestation
    # for these random bytes.
    proc = _archive_subprocess(tmp_path, delta_path)
    assert proc.returncode != 0, (
        "Archive builder accepted approved δ without external attestation. "
        f"This is the Codex R5-2 #4 bypass.\nstdout: {proc.stdout}\n"
        f"stderr: {proc.stderr}"
    )
    combined = proc.stdout + proc.stderr
    assert "attestation" in combined.lower(), (
        "Refusal must mention 'attestation' so the operator knows what "
        f"to fix.\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "sign_lane_c_compliance" in combined or "tools/sign" in combined, (
        "Refusal must point at the signing tool. "
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )

    # --allow-pending-compliance must NOT bypass the approved-without-
    # attestation gate (that flag is for pending only).
    proc2 = _archive_subprocess(tmp_path, delta_path, allow_pending=True)
    assert proc2.returncode != 0, (
        "--allow-pending-compliance bypassed the approved-without-"
        "attestation gate. The flag must be scoped to pending only."
    )


def test_archive_builder_refuses_approved_with_sha_mismatch(tmp_path: Path) -> None:
    """If an attestation exists for δ blob A but the bundled δ is B, the
    SHA cross-check must fire. We do this by signing one blob, then
    swapping in another at the same delta.bin path."""
    delta_path = tmp_path / "delta.bin"
    blob_a = _build_delta_blob(compliance_status="approved", n_frames=4)
    blob_b = _build_delta_blob(compliance_status="approved", n_frames=8)
    assert blob_a != blob_b, "test setup error: blobs must differ"
    delta_path.write_bytes(blob_a)

    # Sign blob_a IN THE REPO (not tmp_path) because the archive
    # builder checks attestations relative to REPO. This means we ARE
    # writing into the real .omx tree for the duration of the test.
    # We therefore use a uniquely-shaped blob that won't collide with
    # any real attestations, and we clean up afterwards.
    from tac.lane_c_compliance import (
        write_attestation, attestation_path_for, compute_blob_sha256,
    )
    sha_a = compute_blob_sha256(blob_a)
    att_path_a = attestation_path_for(sha_a, root=REPO)
    try:
        write_attestation(
            blob=blob_a, approver="test-suite",
            ruling_text="test_archive_builder_refuses_approved_with_sha_mismatch",
            signed_by_user="pytest", git_head="test",
            root=REPO,
        )
        # Now swap in blob B at the delta.bin path — sha doesn't match
        # the attestation any more.
        delta_path.write_bytes(blob_b)
        proc = _archive_subprocess(tmp_path, delta_path)
        assert proc.returncode != 0, (
            "Archive builder accepted approved δ whose sha doesn't match "
            f"the attestation.\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
        combined = (proc.stdout + proc.stderr).lower()
        assert "different" in combined or "mismatch" in combined or \
               "sha" in combined, (
            "Refusal must indicate the sha mismatch so the operator can "
            f"diagnose.\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
    finally:
        if att_path_a.exists():
            att_path_a.unlink()


def test_archive_builder_accepts_approved_with_valid_attestation(
    tmp_path: Path,
) -> None:
    """Positive path: approved δ + matching attestation passes the gate.
    The build will still fail downstream (fake renderer / GT video) but
    the gate output must indicate VERIFIED before that happens."""
    delta_path = tmp_path / "delta.bin"
    blob = _build_delta_blob(compliance_status="approved", n_frames=4)
    delta_path.write_bytes(blob)

    from tac.lane_c_compliance import (
        write_attestation, attestation_path_for, compute_blob_sha256,
    )
    sha = compute_blob_sha256(blob)
    att_path = attestation_path_for(sha, root=REPO)
    try:
        write_attestation(
            blob=blob, approver="test-suite",
            ruling_text="test_archive_builder_accepts_approved_with_valid_attestation",
            signed_by_user="pytest", git_head="test",
            root=REPO,
        )
        proc = _archive_subprocess(tmp_path, delta_path)
        # Build will fail for a downstream reason (fake renderer can't
        # be loaded, fake GT can't be decoded) but the COMPLIANCE GATE
        # must have printed VERIFIED and proceeded past it.
        combined = proc.stdout + proc.stderr
        assert "VERIFIED" in combined, (
            "Archive builder did not log VERIFIED for an approved δ "
            f"with a matching attestation.\nstdout: {proc.stdout}\n"
            f"stderr: {proc.stderr}"
        )
    finally:
        if att_path.exists():
            att_path.unlink()


# ---------------------------------------------------------------------------
# Misc — provenance recording for approved builds.
# ---------------------------------------------------------------------------


def test_archive_provenance_records_attestation_fields() -> None:
    """Source-level pin: the provenance JSON for an approved δ build
    must include the attestation chain so an auditor can reconstruct
    who approved what without re-reading the attestation file."""
    src = ARCHIVE_SCRIPT.read_text()
    assert "external_attestation" in src, (
        "Provenance JSON must record external_attestation for approved δ"
    )
    assert "attestation_approver" in src, (
        "Provenance must record the approver identity"
    )
    assert "attestation_signed_at_utc" in src, (
        "Provenance must record the signing timestamp"
    )
    assert "attestation_git_head" in src, (
        "Provenance must record the git HEAD at signing"
    )
    assert "attestation_ruling_text" in src, (
        "Provenance must record the ruling text"
    )


def test_archive_builder_imports_lane_c_compliance() -> None:
    """The archive builder MUST import from tac.lane_c_compliance for the
    verifier — pin the dependency so a refactor doesn't accidentally
    duplicate the verification logic and let the two diverge."""
    src = ARCHIVE_SCRIPT.read_text()
    assert "from tac.lane_c_compliance import" in src, (
        "Archive builder must import tac.lane_c_compliance helpers; "
        "duplicating the SHA / file-path logic risks divergence."
    )
    assert "verify_attestation_for_blob" in src
    assert "AttestationMissing" in src
    assert "AttestationMismatch" in src
    assert "AttestationMalformed" in src
