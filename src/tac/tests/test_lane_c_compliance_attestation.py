# SPDX-License-Identifier: MIT
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

    Codex R5-3 #2: when compliance_status='approved', this passes the
    INTERNAL_PROMOTION_TOKEN to pack_sparse_delta. Negative-path tests
    that simulate "operator tries to forge an approved blob" use this
    helper precisely BECAUSE it exercises the same codepath the
    promotion tool uses — the test is the one place we deliberately
    bypass the library refusal in order to verify the downstream gate.
    """
    import torch
    from tac.uniward_delta import pack_sparse_delta
    from tac.lane_c_compliance import INTERNAL_PROMOTION_TOKEN
    delta = torch.randn(n_frames, 3, h, w) * 2.0  # within budget
    cost = torch.rand(n_frames, h, w) * 10.0
    kwargs = {
        "l_inf_budget": 4.0,
        "target_bytes": 1024,
        "seed": 1234,
        "compliance_status": compliance_status,
    }
    if compliance_status == "approved":
        kwargs["_internal_promotion_token"] = INTERNAL_PROMOTION_TOKEN
    return pack_sparse_delta(delta, cost, **kwargs)


def _gen_test_signing_key():
    """Codex R5-3 #1 helper: generate an in-memory Ed25519 signing key
    for tests. Returns (private_key, pubkey_hex).

    Tests that need to write a verifiable attestation must:
      1. Call this to get (priv, pubkey_hex).
      2. Write a trust root JSON via _write_test_trust_root.
      3. Pass priv as ``private_key`` to write_attestation.
    """
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )
    priv = Ed25519PrivateKey.generate()
    pub_bytes = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return priv, pub_bytes.hex()


def _write_test_trust_root(root: Path, entries: dict) -> Path:
    """Codex R5-3 #1 helper: write a trust-root pubkey registry under
    the given root for tests. ``entries`` is a dict mapping
    approver_id → pubkey_hex.
    """
    import json as _json
    from tac.lane_c_compliance import trust_root_path
    path = trust_root_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        approver: {
            "pubkey_hex": pubkey_hex,
            "comment": f"test trust root for {approver}",
            "registered_at": "2026-04-27T00:00:00Z",
        }
        for approver, pubkey_hex in entries.items()
    }
    path.write_text(_json.dumps(record, indent=2, sort_keys=True))
    return path


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
    priv, _pub = _gen_test_signing_key()
    blob = b"some-delta-bytes" * 32
    with pytest.raises(ValueError, match="ruling_text"):
        write_attestation(
            blob=blob, approver="yousfi", ruling_text="",
            signed_by_user="adpena", git_head="abc123",
            private_key=priv, root=tmp_path,
        )
    with pytest.raises(ValueError, match="ruling_text"):
        write_attestation(
            blob=blob, approver="yousfi", ruling_text="    ",
            signed_by_user="adpena", git_head="abc123",
            private_key=priv, root=tmp_path,
        )


def test_write_attestation_refuses_empty_approver(tmp_path: Path) -> None:
    from tac.lane_c_compliance import write_attestation
    priv, _pub = _gen_test_signing_key()
    blob = b"some-delta-bytes" * 32
    with pytest.raises(ValueError, match="approver"):
        write_attestation(
            blob=blob, approver="", ruling_text="real ruling",
            signed_by_user="adpena", git_head="abc123",
            private_key=priv, root=tmp_path,
        )


def test_write_then_verify_round_trip(tmp_path: Path) -> None:
    """Happy path: write attestation, then verify_attestation_for_blob
    returns the parsed Attestation."""
    from tac.lane_c_compliance import write_attestation, verify_attestation_for_blob
    priv, pubkey_hex = _gen_test_signing_key()
    _write_test_trust_root(tmp_path, {"yousfi": pubkey_hex})
    blob = b"some-delta-bytes" * 32
    path = write_attestation(
        blob=blob, approver="yousfi",
        ruling_text="PR #35 approved 2026-04-27",
        signed_by_user="adpena", git_head="abc123def",
        private_key=priv, root=tmp_path,
    )
    assert path.exists()
    att = verify_attestation_for_blob(blob, root=tmp_path)
    assert att.approver == "yousfi"
    assert att.ruling_text == "PR #35 approved 2026-04-27"
    assert att.signed_by_user == "adpena"
    assert att.git_head == "abc123def"
    # Codex R5-3 #1: verify the signature_hex round-trips and is non-empty.
    assert att.signature_hex
    assert len(att.signature_hex) == 128


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
    priv, pubkey_hex = _gen_test_signing_key()
    _write_test_trust_root(tmp_path, {"yousfi": pubkey_hex})
    blob_a = b"approved-delta-A" * 32
    blob_b = b"unauthorized-delta-B" * 32
    # Sign A.
    write_attestation(
        blob=blob_a, approver="yousfi",
        ruling_text="approved", signed_by_user="adpena",
        git_head="abc", private_key=priv, root=tmp_path,
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
    # Missing ruling_text + approver + signature_hex (Codex R5-3 #1).
    path.write_text(json.dumps({
        "schema_version": 2,
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
        "signature_hex": "00" * 64,
    }))
    with pytest.raises(AttestationMalformed):
        verify_attestation_for_blob(blob, root=tmp_path)


def test_write_refuses_overwrite_by_default(tmp_path: Path) -> None:
    from tac.lane_c_compliance import write_attestation
    priv, _pub = _gen_test_signing_key()
    blob = b"delta-being-overwritten" * 32
    write_attestation(
        blob=blob, approver="yousfi", ruling_text="first",
        signed_by_user="adpena", git_head="abc",
        private_key=priv, root=tmp_path,
    )
    with pytest.raises(FileExistsError):
        write_attestation(
            blob=blob, approver="fridrich", ruling_text="second",
            signed_by_user="adpena", git_head="def",
            private_key=priv, root=tmp_path,
        )
    # overwrite=True works.
    write_attestation(
        blob=blob, approver="fridrich", ruling_text="second",
        signed_by_user="adpena", git_head="def",
        private_key=priv, root=tmp_path,
        overwrite=True,
    )


# ---------------------------------------------------------------------------
# Layer A — signing tool CLI
# ---------------------------------------------------------------------------


def _make_test_pem(tmp_path: Path) -> Path:
    """Codex R5-3 #1: write an Ed25519 PEM file under tmp_path so the
    sign-tool subprocess tests can use --private-key without touching
    the operator's real key."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )
    pem_path = tmp_path / "signing_key.pem"
    priv = Ed25519PrivateKey.generate()
    pem_path.write_bytes(
        priv.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return pem_path


def test_sign_tool_refuses_empty_ruling(tmp_path: Path) -> None:
    """The signing tool MUST refuse empty --ruling-text. Without ruling
    text, the attestation has no forensic value."""
    delta = tmp_path / "delta.bin"
    delta.write_bytes(b"x" * 256)
    pem = _make_test_pem(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(SIGN_SCRIPT),
         "--delta-bin", str(delta),
         "--approver", "yousfi",
         "--ruling-text", "",
         "--root", str(tmp_path),
         "--private-key", str(pem)],
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
    pem = _make_test_pem(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(SIGN_SCRIPT),
         "--delta-bin", str(delta),
         "--approver", "",
         "--ruling-text", "ruling",
         "--root", str(tmp_path),
         "--private-key", str(pem)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode != 0
    assert "approver" in (proc.stdout + proc.stderr).lower()


def test_sign_tool_refuses_empty_delta(tmp_path: Path) -> None:
    delta = tmp_path / "delta.bin"
    delta.write_bytes(b"")
    pem = _make_test_pem(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(SIGN_SCRIPT),
         "--delta-bin", str(delta),
         "--approver", "yousfi",
         "--ruling-text", "ruling",
         "--root", str(tmp_path),
         "--private-key", str(pem)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode != 0
    assert "empty" in (proc.stdout + proc.stderr).lower()


def test_sign_tool_refuses_missing_delta(tmp_path: Path) -> None:
    pem = _make_test_pem(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(SIGN_SCRIPT),
         "--delta-bin", str(tmp_path / "nope.bin"),
         "--approver", "yousfi",
         "--ruling-text", "ruling",
         "--root", str(tmp_path),
         "--private-key", str(pem)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode != 0
    assert "does not exist" in (proc.stdout + proc.stderr).lower()


def test_sign_tool_happy_path_writes_attestation(tmp_path: Path) -> None:
    delta = tmp_path / "delta.bin"
    delta.write_bytes(b"y" * 1024)
    pem = _make_test_pem(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(SIGN_SCRIPT),
         "--delta-bin", str(delta),
         "--approver", "yousfi",
         "--ruling-text", "PR #35: Lane C δ approved 2026-04-27",
         "--root", str(tmp_path),
         "--signed-by-user", "test-user",
         "--private-key", str(pem)],
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
    # Codex R5-3 #1: attestation must carry an Ed25519 signature.
    assert "signature_hex" in record
    assert len(record["signature_hex"]) == 128


def test_sign_tool_refuses_overwrite_without_flag(tmp_path: Path) -> None:
    delta = tmp_path / "delta.bin"
    delta.write_bytes(b"z" * 512)
    pem = _make_test_pem(tmp_path)
    base_args = [
        sys.executable, str(SIGN_SCRIPT),
        "--delta-bin", str(delta),
        "--approver", "yousfi",
        "--ruling-text", "first",
        "--root", str(tmp_path),
        "--signed-by-user", "test-user",
        "--private-key", str(pem),
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


@pytest.fixture
def _repo_test_trust_root():
    """Codex R5-3 #1: register a 'test-suite' approver in the REAL repo's
    trust root for the duration of the test, then clean up.

    The archive-builder gate tests must write into the REPO's .omx tree
    (because the builder uses REPO root for attestation lookup), so the
    trust root must also live there during the test. We yield the
    Ed25519 private key the test should use to sign with.

    Cleanup is best-effort but careful: if a real trust root file
    already exists in the repo (e.g. an operator added council pubkeys),
    we MERGE rather than overwrite, then on cleanup restore the prior
    contents. This avoids destroying real trust-root entries.
    """
    import json as _json
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )
    from tac.lane_c_compliance import trust_root_path

    priv = Ed25519PrivateKey.generate()
    pub_bytes = priv.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    pub_hex = pub_bytes.hex()

    tr_path = trust_root_path(REPO)
    tr_path.parent.mkdir(parents=True, exist_ok=True)
    prior_contents: bytes | None = None
    if tr_path.exists():
        prior_contents = tr_path.read_bytes()
        try:
            existing = _json.loads(prior_contents)
            if not isinstance(existing, dict):
                existing = {}
        except _json.JSONDecodeError:
            existing = {}
    else:
        existing = {}
    existing["test-suite"] = {
        "pubkey_hex": pub_hex,
        "comment": "ephemeral test suite key — automatically removed",
        "registered_at": "2026-04-27T00:00:00Z",
    }
    tr_path.write_text(_json.dumps(existing, indent=2, sort_keys=True))
    try:
        yield priv
    finally:
        if prior_contents is not None:
            tr_path.write_bytes(prior_contents)
        else:
            try:
                tr_path.unlink()
            except OSError:
                pass


def test_archive_builder_refuses_approved_with_sha_mismatch(
    tmp_path: Path, _repo_test_trust_root,
) -> None:
    """If an attestation exists for δ blob A but the bundled δ is B, the
    SHA cross-check must fire. We do this by signing one blob, then
    swapping in another at the same delta.bin path."""
    priv = _repo_test_trust_root
    delta_path = tmp_path / "delta.bin"
    blob_a = _build_delta_blob(compliance_status="approved", n_frames=4)
    blob_b = _build_delta_blob(compliance_status="approved", n_frames=8)
    assert blob_a != blob_b, "test setup error: blobs must differ"
    delta_path.write_bytes(blob_a)

    # Sign blob_a IN THE REPO (not tmp_path) because the archive
    # builder checks attestations relative to REPO.
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
            private_key=priv, root=REPO,
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
    tmp_path: Path, _repo_test_trust_root,
) -> None:
    """Positive path: approved δ + matching attestation passes the gate.
    The build will still fail downstream (fake renderer / GT video) but
    the gate output must indicate VERIFIED before that happens."""
    priv = _repo_test_trust_root
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
            private_key=priv, root=REPO,
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
    # Codex R5-3 #5 fix: provenance enrichment.
    assert "attestation_file_sha256" in src, (
        "Codex R5-3 #5: provenance must record the on-disk attestation "
        "JSON SHA so auditors can verify file-level integrity."
    )
    assert "attestation_record" in src, (
        "Codex R5-3 #5: provenance must include the FULL attestation "
        "record so offline auditors can re-verify the signature without "
        "needing the .omx file."
    )
    assert "attestation_signature_hex" in src, (
        "Codex R5-3 #5: provenance must record the Ed25519 signature_hex"
    )
    assert "attestation_schema_version" in src, (
        "Codex R5-3 #5: provenance must record the attestation schema "
        "version (so a future schema bump doesn't silently invalidate "
        "audit reads)."
    )
    assert '"delta_sha256":' in src or '"delta_sha256"' in src, (
        "Codex R5-3 #5: provenance must use a clearly-named delta_sha256 "
        "field (the previous attestation_sha256 was misleading)."
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
    # Codex R5-3 #1: builder must also handle the new exception types.
    assert "AttestationSignatureInvalid" in src
    assert "AttestationApproverNotInTrustRoot" in src
    assert "TrustRootMissing" in src
    assert "TrustRootMalformed" in src


# ---------------------------------------------------------------------------
# Codex R5-3 #1 — Ed25519 trust-root regression suite
# ---------------------------------------------------------------------------


def test_r5_3_1_unsigned_attestation_refused(tmp_path: Path) -> None:
    """REGRESSION (Codex R5-3 #1): an attestation lacking signature_hex
    is refused. Old "structure-only" v1 attestations are no longer valid.

    Without this, the same operator who builds the δ can also write
    the attestation file, and the verifier had no cryptographic check.
    """
    from tac.lane_c_compliance import (
        verify_attestation_for_blob, AttestationMalformed,
        attestation_path_for, compute_blob_sha256,
    )
    blob = b"some-delta-bytes" * 32
    sha = compute_blob_sha256(blob)
    path = attestation_path_for(sha, root=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Forge a structure-only v1 attestation (no signature_hex).
    path.write_text(json.dumps({
        "schema_version": 1,
        "delta_sha256": sha,
        "approver": "yousfi",
        "ruling_text": "self-asserted",
        "signed_at_utc": "2026-04-27T00:00:00Z",
        "signed_by_user": "attacker",
        "git_head": "abc",
        # Note: no signature_hex.
    }))
    with pytest.raises(AttestationMalformed):
        verify_attestation_for_blob(blob, root=tmp_path)


def test_r5_3_1_attestation_with_wrong_signature_refused(tmp_path: Path) -> None:
    """REGRESSION (Codex R5-3 #1): an attestation signed by a key whose
    pubkey does NOT match what's registered for the approver in the
    trust root must be refused with AttestationSignatureInvalid.
    """
    from tac.lane_c_compliance import (
        write_attestation, verify_attestation_for_blob,
        AttestationSignatureInvalid,
    )
    real_priv, real_pub_hex = _gen_test_signing_key()
    wrong_priv, _wrong_pub_hex = _gen_test_signing_key()
    # Trust root knows ONLY the real pubkey for "yousfi".
    _write_test_trust_root(tmp_path, {"yousfi": real_pub_hex})
    blob = b"forged-delta-bytes" * 32
    # Sign with the WRONG key (the attacker's key).
    write_attestation(
        blob=blob, approver="yousfi", ruling_text="forged",
        signed_by_user="attacker", git_head="abc",
        private_key=wrong_priv, root=tmp_path,
    )
    with pytest.raises(AttestationSignatureInvalid):
        verify_attestation_for_blob(blob, root=tmp_path)


def test_r5_3_1_approver_not_in_trust_root_refused(tmp_path: Path) -> None:
    """REGRESSION (Codex R5-3 #1): an attestation whose approver isn't
    in the trust root is refused REGARDLESS of whether the signature
    would have verified. Only allowlisted approvers can issue."""
    from tac.lane_c_compliance import (
        write_attestation, verify_attestation_for_blob,
        AttestationApproverNotInTrustRoot,
    )
    priv, pub_hex = _gen_test_signing_key()
    # Trust root has fridrich, NOT yousfi.
    _write_test_trust_root(tmp_path, {"fridrich": pub_hex})
    blob = b"impostor-delta" * 32
    write_attestation(
        blob=blob, approver="yousfi", ruling_text="impostor",
        signed_by_user="attacker", git_head="abc",
        private_key=priv, root=tmp_path,
    )
    with pytest.raises(AttestationApproverNotInTrustRoot):
        verify_attestation_for_blob(blob, root=tmp_path)


def test_r5_3_1_trust_root_missing_refused(tmp_path: Path) -> None:
    """REGRESSION (Codex R5-3 #1): if the trust root file doesn't exist,
    no attestation can verify. Fail-closed default for any repo that
    hasn't bootstrapped its registry yet.
    """
    from tac.lane_c_compliance import (
        write_attestation, verify_attestation_for_blob,
        TrustRootMissing,
    )
    priv, _pub_hex = _gen_test_signing_key()
    # Do NOT call _write_test_trust_root — file is absent.
    blob = b"orphan-delta" * 32
    write_attestation(
        blob=blob, approver="yousfi", ruling_text="orphan",
        signed_by_user="x", git_head="abc",
        private_key=priv, root=tmp_path,
    )
    with pytest.raises(TrustRootMissing):
        verify_attestation_for_blob(blob, root=tmp_path)


def test_r5_3_1_trust_root_malformed_refused(tmp_path: Path) -> None:
    """REGRESSION (Codex R5-3 #1): a trust-root file with non-hex pubkey
    is refused with TrustRootMalformed."""
    from tac.lane_c_compliance import (
        write_attestation, verify_attestation_for_blob,
        TrustRootMalformed, trust_root_path,
    )
    priv, _pub_hex = _gen_test_signing_key()
    tr_path = trust_root_path(tmp_path)
    tr_path.parent.mkdir(parents=True, exist_ok=True)
    # Pubkey is not hex-decodable.
    tr_path.write_text(json.dumps({
        "yousfi": {"pubkey_hex": "ZZ" * 32, "comment": "broken"},
    }))
    blob = b"some-delta" * 32
    write_attestation(
        blob=blob, approver="yousfi", ruling_text="x",
        signed_by_user="x", git_head="abc",
        private_key=priv, root=tmp_path,
    )
    with pytest.raises(TrustRootMalformed):
        verify_attestation_for_blob(blob, root=tmp_path)


def test_r5_3_1_signed_payload_does_not_include_signature_field(
    tmp_path: Path,
) -> None:
    """REGRESSION (Codex R5-3 #1): canonical_signed_payload must NOT
    include signature_hex — otherwise the signing operation would have
    a chicken-and-egg dependency.

    Also pin the field set explicitly so any future reordering or
    addition is a deliberate, reviewed change (a wire break).
    """
    from tac.lane_c_compliance import canonical_signed_payload
    payload = canonical_signed_payload({
        "schema_version": 2,
        "delta_sha256": "a" * 64,
        "approver": "yousfi",
        "ruling_text": "ok",
        "signed_at_utc": "2026-04-27T00:00:00Z",
        "signed_by_user": "test",
        "git_head": "abc",
        "signature_hex": "deadbeef" * 16,  # MUST be ignored.
        "delta_size_bytes": 1024,  # MUST be ignored.
    })
    text = payload.decode("utf-8")
    assert "signature_hex" not in text, (
        "canonical_signed_payload included signature_hex — circular dep"
    )
    assert "delta_size_bytes" not in text, (
        "canonical_signed_payload included a non-signed informational field"
    )
    # Sorted-keys compact form: alphabetical key order.
    expected_keys_in_order = (
        "approver", "delta_sha256", "git_head", "ruling_text",
        "schema_version", "signed_at_utc", "signed_by_user",
    )
    last_idx = -1
    for k in expected_keys_in_order:
        idx = text.find(f'"{k}":')
        assert idx > last_idx, (
            f"canonical_signed_payload key order broken at {k!r}; "
            f"sorted-keys compact form is the wire contract."
        )
        last_idx = idx


def test_r5_3_1_canonical_payload_missing_field_raises(tmp_path: Path) -> None:
    """REGRESSION (Codex R5-3 #1): canonical_signed_payload must refuse
    a record missing any of the trust-bearing fields. Otherwise a
    rotated record could partially commit."""
    from tac.lane_c_compliance import (
        canonical_signed_payload, AttestationMalformed,
    )
    with pytest.raises(AttestationMalformed):
        canonical_signed_payload({
            "schema_version": 2,
            "delta_sha256": "a" * 64,
            # Missing approver.
            "ruling_text": "ok",
            "signed_at_utc": "2026-04-27T00:00:00Z",
            "signed_by_user": "test",
            "git_head": "abc",
        })


def test_r5_3_1_tampered_ruling_text_invalidates_signature(
    tmp_path: Path,
) -> None:
    """REGRESSION (Codex R5-3 #1): if someone edits the ruling_text on
    a signed attestation file (e.g. softens "rejected for safety" into
    "approved"), the Ed25519 signature must no longer verify."""
    from tac.lane_c_compliance import (
        write_attestation, verify_attestation_for_blob,
        AttestationSignatureInvalid, attestation_path_for, compute_blob_sha256,
    )
    priv, pub_hex = _gen_test_signing_key()
    _write_test_trust_root(tmp_path, {"yousfi": pub_hex})
    blob = b"sensitive-delta" * 32
    write_attestation(
        blob=blob, approver="yousfi",
        ruling_text="REJECTED — non-compliant per PR #35",
        signed_by_user="yousfi", git_head="abc",
        private_key=priv, root=tmp_path,
    )
    sha = compute_blob_sha256(blob)
    path = attestation_path_for(sha, root=tmp_path)
    record = json.loads(path.read_text())
    # Attacker tampers with the ruling text.
    record["ruling_text"] = "Approved"
    path.write_text(json.dumps(record, indent=2, sort_keys=True))
    with pytest.raises(AttestationSignatureInvalid):
        verify_attestation_for_blob(blob, root=tmp_path)


# ---------------------------------------------------------------------------
# Codex R5-3 #2 — pack_sparse_delta refuses approved without internal token
# ---------------------------------------------------------------------------


def test_r5_3_2_pack_refuses_approved_without_token(tmp_path: Path) -> None:
    """REGRESSION (Codex R5-3 #2): pack_sparse_delta MUST refuse to write
    compliance_status='approved' when no internal promotion token is
    supplied. This is the library-level closure of the bypass.
    """
    import torch
    from tac.uniward_delta import pack_sparse_delta, COMPLIANCE_APPROVED
    delta = torch.randn(2, 3, 8, 8) * 1.5
    cost = torch.rand(2, 8, 8) * 5.0
    with pytest.raises(ValueError, match="promotion tool"):
        pack_sparse_delta(
            delta, cost, l_inf_budget=4.0, target_bytes=1024,
            compliance_status=COMPLIANCE_APPROVED,
        )


def test_r5_3_2_pack_refuses_approved_with_wrong_token(tmp_path: Path) -> None:
    """REGRESSION (Codex R5-3 #2): even passing _internal_promotion_token
    with a bogus value must be refused (constant-time comparison so
    prefix attacks don't help)."""
    import torch
    from tac.uniward_delta import pack_sparse_delta, COMPLIANCE_APPROVED
    delta = torch.randn(2, 3, 8, 8) * 1.5
    cost = torch.rand(2, 8, 8) * 5.0
    with pytest.raises(ValueError, match="promotion tool"):
        pack_sparse_delta(
            delta, cost, l_inf_budget=4.0, target_bytes=1024,
            compliance_status=COMPLIANCE_APPROVED,
            _internal_promotion_token="not-the-real-token",
        )


def test_r5_3_2_pack_accepts_approved_with_correct_token(
    tmp_path: Path,
) -> None:
    """REGRESSION (Codex R5-3 #2, positive direction): with the correct
    token, pack succeeds. The promotion tool relies on this contract."""
    import torch
    from tac.uniward_delta import (
        pack_sparse_delta, unpack_sparse_delta, COMPLIANCE_APPROVED,
    )
    from tac.lane_c_compliance import INTERNAL_PROMOTION_TOKEN
    delta = torch.randn(2, 3, 8, 8) * 1.5
    cost = torch.rand(2, 8, 8) * 5.0
    blob = pack_sparse_delta(
        delta, cost, l_inf_budget=4.0, target_bytes=1024,
        compliance_status=COMPLIANCE_APPROVED,
        _internal_promotion_token=INTERNAL_PROMOTION_TOKEN,
    )
    spec = unpack_sparse_delta(blob)
    assert spec.compliance_status == COMPLIANCE_APPROVED


def test_r5_3_2_pending_and_rejected_do_not_need_token(tmp_path: Path) -> None:
    """SANITY (Codex R5-3 #2): pending_ruling and rejected statuses
    do NOT require the token (they're operator-issuable — the gate is
    only on approved)."""
    import torch
    from tac.uniward_delta import (
        pack_sparse_delta, COMPLIANCE_PENDING, COMPLIANCE_REJECTED,
    )
    delta = torch.randn(2, 3, 8, 8) * 1.5
    cost = torch.rand(2, 8, 8) * 5.0
    for status in (COMPLIANCE_PENDING, COMPLIANCE_REJECTED):
        blob = pack_sparse_delta(
            delta, cost, l_inf_budget=4.0, target_bytes=1024,
            compliance_status=status,
        )
        assert isinstance(blob, bytes)


# ---------------------------------------------------------------------------
# Codex R5-3 #3 — atomic write
# ---------------------------------------------------------------------------


def test_r5_3_3_concurrent_write_attestation_only_one_succeeds(
    tmp_path: Path,
) -> None:
    """REGRESSION (Codex R5-3 #3): two threads racing on the same SHA
    must produce exactly ONE successful write and ONE clean
    FileExistsError. The previous path.exists() + path.write_text()
    pattern allowed BOTH to write, with the second silently winning."""
    import threading
    from tac.lane_c_compliance import write_attestation
    priv, _pub = _gen_test_signing_key()
    blob = b"race-condition-delta" * 32
    results: list[Exception | str] = []
    barrier = threading.Barrier(8)

    def attempt_write() -> None:
        try:
            barrier.wait(timeout=5)
            write_attestation(
                blob=blob, approver="yousfi",
                ruling_text="concurrent",
                signed_by_user="x", git_head="abc",
                private_key=priv, root=tmp_path,
            )
            results.append("ok")
        except FileExistsError:
            results.append("file-exists")
        except Exception as e:
            results.append(e)

    threads = [threading.Thread(target=attempt_write) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    successes = [r for r in results if r == "ok"]
    file_exists = [r for r in results if r == "file-exists"]
    other = [r for r in results if r not in ("ok", "file-exists")]
    assert len(successes) == 1, (
        f"Codex R5-3 #3: expected exactly 1 successful write, got "
        f"{len(successes)}. Results: {results!r}"
    )
    assert len(file_exists) == 7, (
        f"Codex R5-3 #3: expected 7 FileExistsError losers, got "
        f"{len(file_exists)}. Results: {results!r}"
    )
    assert not other, f"Unexpected exceptions: {other!r}"


def test_r5_3_3_overwrite_uses_atomic_replace(tmp_path: Path) -> None:
    """REGRESSION (Codex R5-3 #3): with overwrite=True, the write goes
    via temp file + atomic rename. We can't easily prove atomicity in a
    unit test, but we CAN prove (a) overwrite succeeds and (b) the
    final file is well-formed JSON (i.e. not a partial truncated write
    from a concurrent crash)."""
    from tac.lane_c_compliance import write_attestation
    priv, _pub = _gen_test_signing_key()
    blob = b"overwrite-target" * 32
    write_attestation(
        blob=blob, approver="yousfi", ruling_text="first",
        signed_by_user="x", git_head="abc",
        private_key=priv, root=tmp_path,
    )
    path = write_attestation(
        blob=blob, approver="yousfi", ruling_text="SECOND-overwriting",
        signed_by_user="x", git_head="abc",
        private_key=priv, root=tmp_path, overwrite=True,
    )
    record = json.loads(path.read_text())
    assert record["ruling_text"] == "SECOND-overwriting"
    # Final file is a real, complete JSON object (not a partial write).
    # All required fields present.
    for k in ("schema_version", "delta_sha256", "approver",
              "ruling_text", "signed_at_utc", "signed_by_user",
              "git_head", "signature_hex"):
        assert k in record, f"overwrite produced incomplete record: {k!r} missing"


# ---------------------------------------------------------------------------
# Codex R5-3 #4 — path-traversal defense
# ---------------------------------------------------------------------------


def test_r5_3_4_attestation_path_rejects_traversal(tmp_path: Path) -> None:
    """REGRESSION (Codex R5-3 #4): a sha256 input containing path
    separators must be refused — old length-only check would have let
    "../etc/passwd_padded_to_64_chars..." escape the attestation dir.
    """
    from tac.lane_c_compliance import attestation_path_for
    # Various traversal attempts, all 64 chars long:
    attempts = [
        "../etc/passwd_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",  # parent-dir
        "/" * 64,  # all slashes
        "a/b/c/d/e/f/g/h/i/j/k/l/m/n/o/p/q/r/s/t/u/v/w/x/y/z/0/1/2/3/4",  # ~64 with /
        "..\\..\\windows\\system32\\config\\sam_padded_to_64_aaaaaaaaa",  # windows
        "deadbeef" * 7 + "GG",  # not hex
        "Z" * 64,  # not hex
    ]
    for bad in attempts:
        if len(bad) != 64:
            bad = (bad * 2)[:64]
        with pytest.raises(ValueError):
            attestation_path_for(bad, root=tmp_path)


def test_r5_3_4_attestation_path_resolves_under_attestation_dir(
    tmp_path: Path,
) -> None:
    """REGRESSION (Codex R5-3 #4): a valid hex SHA must resolve to a
    path STRICTLY under the attestation directory."""
    from tac.lane_c_compliance import (
        attestation_path_for, ATTESTATION_DIR,
    )
    valid_sha = "a" * 64
    path = attestation_path_for(valid_sha, root=tmp_path)
    expected_dir = (tmp_path / ATTESTATION_DIR).resolve()
    assert path.parent == expected_dir
    # Resolving the path must not escape the attestation dir.
    path.resolve().relative_to(expected_dir)  # raises if outside

