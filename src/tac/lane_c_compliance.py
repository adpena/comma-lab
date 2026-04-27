"""Lane C δ compliance attestation utilities (Codex R5-2 #4 fix, 2026-04-27).

The Lane C δ pipeline is scorer-derived; Yousfi PR #35 strict-scorer-rule
may class it as non-compliant. To prevent operators from self-asserting
contest approval at δ-build time, this module enforces a TWO-PARTY trust
model:

  1. The δ producer (``experiments/optimize_uniward_delta.py``) can only
     issue ``pending_ruling`` or ``rejected``. ``approved`` is REJECTED
     by argparse choices.

  2. To bundle an ``approved`` δ into a submission archive,
     ``experiments/build_baseline_archive.py`` requires a separate
     attestation JSON file at the canonical path
     ``.omx/state/lane_c_compliance_attestations/<sha256>.json``,
     where ``<sha256>`` is the SHA256 of the actual ``delta.bin`` bytes.

  3. The attestation is produced by a SEPARATE tool
     (``tools/sign_lane_c_compliance.py``) which records the approver's
     identity, the ruling text, the timestamp, the git HEAD, and the
     ``whoami`` value. The tool refuses to run with empty ruling text.

The attestation file's ``delta_sha256`` MUST match the bundled δ.bin's
actual sha256 — this catches typos, copy-paste between attestations,
and any drift between the δ that was approved and the δ that ships.

Why a separate file rather than a signature embedded in the blob:
  - The blob's compliance_status is operator-controlled at build time;
    the attestation is operator-controlled at SIGN time and lives in
    a different place. You need BOTH to ship.
  - The attestation is git-committable: a future auditor can inspect
    the commit log to see who approved what and when.
  - SHA-keyed file naming makes "wrong δ but right attestation"
    impossible: the gate cross-checks SHA before accepting.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


__all__ = [
    "ATTESTATION_DIR",
    "ATTESTATION_SCHEMA_VERSION",
    "AttestationMissing",
    "AttestationMismatch",
    "AttestationMalformed",
    "Attestation",
    "compute_blob_sha256",
    "attestation_path_for",
    "load_attestation",
    "verify_attestation_for_blob",
    "write_attestation",
]


ATTESTATION_DIR = Path(".omx") / "state" / "lane_c_compliance_attestations"
ATTESTATION_SCHEMA_VERSION = 1

# Required keys in any attestation JSON. Missing keys ⇒ AttestationMalformed.
_REQUIRED_KEYS = frozenset({
    "schema_version", "delta_sha256", "approver", "ruling_text",
    "signed_at_utc", "signed_by_user", "git_head",
})


class AttestationMissing(Exception):
    """No attestation file found at the canonical path for this δ.bin sha."""


class AttestationMismatch(Exception):
    """Attestation file exists but its delta_sha256 doesn't match the
    actual δ.bin bytes. Most common cause: someone copy-pasted an
    attestation from a different δ, or the δ was re-built after signing."""


class AttestationMalformed(Exception):
    """Attestation file exists and parses as JSON but is missing required
    fields, has the wrong schema version, or has empty ruling text. The
    gate fails CLOSED on malformed attestations."""


@dataclass(frozen=True)
class Attestation:
    """A loaded, validated attestation record.

    Attributes mirror the fields ``write_attestation`` produces. Use
    ``verify_attestation_for_blob`` rather than constructing this
    directly — the verifier handles the SHA cross-check.
    """

    schema_version: int
    delta_sha256: str
    approver: str
    ruling_text: str
    signed_at_utc: str
    signed_by_user: str
    git_head: str
    # Optional / informational — not part of the trust check.
    delta_size_bytes: int | None = None
    delta_path_at_signing: str | None = None


def compute_blob_sha256(blob: bytes) -> str:
    """Return the hex digest of ``blob``. The single source of truth for
    "the SHA of a δ.bin" — used by both the signer and the verifier."""
    return hashlib.sha256(blob).hexdigest()


def attestation_path_for(
    sha256: str, *, root: Path | str | None = None,
) -> Path:
    """Compute the canonical attestation file path for a given δ.bin sha.

    Args:
        sha256: hex digest from ``compute_blob_sha256``.
        root: optional override for the repo root (defaults to CWD).
            Tests use this to point at a tmp dir without chdir.

    Returns:
        Absolute path to the would-be attestation JSON. The file may
        not yet exist; the caller decides whether that's OK.
    """
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError(
            f"sha256 must be a 64-char hex digest, got {sha256!r}"
        )
    base = Path(root) if root is not None else Path.cwd()
    return base / ATTESTATION_DIR / f"{sha256}.json"


def load_attestation(path: Path) -> Attestation:
    """Read and parse an attestation JSON. Raises AttestationMalformed on
    any structural problem (no JSON, missing keys, empty ruling, wrong
    schema). Does NOT verify against a δ blob — see
    ``verify_attestation_for_blob`` for the full check."""
    if not path.exists():
        raise AttestationMissing(f"no attestation at {path}")
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise AttestationMalformed(
            f"attestation at {path} is not valid JSON: {e!r}"
        ) from e
    if not isinstance(data, dict):
        raise AttestationMalformed(
            f"attestation at {path} root is not a JSON object: "
            f"{type(data).__name__}"
        )
    missing = _REQUIRED_KEYS - data.keys()
    if missing:
        raise AttestationMalformed(
            f"attestation at {path} is missing required fields: "
            f"{sorted(missing)}"
        )
    schema_version = data["schema_version"]
    if not isinstance(schema_version, int) or schema_version != ATTESTATION_SCHEMA_VERSION:
        raise AttestationMalformed(
            f"attestation at {path} has schema_version="
            f"{schema_version!r}; this build expects "
            f"{ATTESTATION_SCHEMA_VERSION}"
        )
    delta_sha256 = data["delta_sha256"]
    if not isinstance(delta_sha256, str) or len(delta_sha256) != 64:
        raise AttestationMalformed(
            f"attestation at {path} has malformed delta_sha256="
            f"{delta_sha256!r}; must be a 64-char hex digest"
        )
    ruling_text = data["ruling_text"]
    if not isinstance(ruling_text, str) or not ruling_text.strip():
        raise AttestationMalformed(
            f"attestation at {path} has empty ruling_text — the signer "
            "tool requires non-empty ruling justification."
        )
    approver = data["approver"]
    if not isinstance(approver, str) or not approver.strip():
        raise AttestationMalformed(
            f"attestation at {path} has empty approver — must record the "
            "identity (council member, yousfi, fridrich, …) that signed."
        )
    return Attestation(
        schema_version=schema_version,
        delta_sha256=delta_sha256.lower(),
        approver=approver,
        ruling_text=ruling_text,
        signed_at_utc=data["signed_at_utc"],
        signed_by_user=data["signed_by_user"],
        git_head=data["git_head"],
        delta_size_bytes=data.get("delta_size_bytes"),
        delta_path_at_signing=data.get("delta_path_at_signing"),
    )


def verify_attestation_for_blob(
    blob: bytes, *, root: Path | str | None = None,
) -> Attestation:
    """Look up the canonical attestation for a δ blob and SHA-verify it.

    The full trust check the archive builder performs:

      1. Compute sha256 of the actual δ.bin bytes.
      2. Read ``.omx/state/lane_c_compliance_attestations/<sha>.json``.
         Missing ⇒ AttestationMissing.
      3. Parse + structural-validate. Bad ⇒ AttestationMalformed.
      4. Cross-check that the attestation's ``delta_sha256`` field
         equals the computed sha (case-insensitive). Bad ⇒
         AttestationMismatch. (Belt-and-braces vs the file-name path:
         a future loader that switches to a different naming scheme
         would still be caught.)

    Returns the parsed Attestation on full success; raises one of the
    three exception types on any failure. The caller decides how loud
    to be (the build script raises SystemExit).
    """
    sha = compute_blob_sha256(blob)
    path = attestation_path_for(sha, root=root)
    attestation = load_attestation(path)  # may raise Missing / Malformed
    if attestation.delta_sha256.lower() != sha.lower():
        raise AttestationMismatch(
            f"attestation at {path} claims delta_sha256="
            f"{attestation.delta_sha256!r} but the actual δ blob hashes "
            f"to {sha!r}. The attestation is for a DIFFERENT δ.bin — "
            f"either the δ was re-built after signing, or the wrong "
            f"attestation was placed at the canonical path."
        )
    return attestation


def write_attestation(
    *,
    blob: bytes,
    approver: str,
    ruling_text: str,
    signed_by_user: str,
    git_head: str,
    delta_path_at_signing: str | None = None,
    root: Path | str | None = None,
    signed_at_utc: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Write an attestation JSON for a δ blob at the canonical path.

    Args:
        blob: the δ.bin bytes being approved. The file path is derived
            from sha256(blob) — same as the verifier expects.
        approver: free-form identity (e.g. "yousfi", "council").
            Must be non-empty.
        ruling_text: free-form justification (e.g. URL to council
            ruling thread, paragraph text). Must be non-empty.
        signed_by_user: typically ``getpass.getuser()`` or whoami.
        git_head: current git commit sha for forensic context.
        delta_path_at_signing: optional; recorded so a future auditor
            can find the build artifact.
        root: optional override for the repo root (tests use this).
        signed_at_utc: ISO8601 timestamp; if None, computed now.
        overwrite: if False (default) and a file already exists at the
            target path, refuse — pre-existing attestation may have
            different ruling text and overwriting is a forensic loss.

    Returns:
        Path to the written attestation file.

    Raises:
        ValueError: empty approver / ruling_text / git_head /
            signed_by_user.
        FileExistsError: an attestation already exists at the path
            and ``overwrite=False``.
    """
    import time
    if not isinstance(approver, str) or not approver.strip():
        raise ValueError(
            "approver must be a non-empty string (e.g. 'yousfi', "
            "'council', 'fridrich')"
        )
    if not isinstance(ruling_text, str) or not ruling_text.strip():
        raise ValueError(
            "ruling_text must be a non-empty string — record the council "
            "decision URL or full ruling paragraph for forensic auditing."
        )
    if not isinstance(signed_by_user, str) or not signed_by_user.strip():
        raise ValueError("signed_by_user must be a non-empty string (e.g. whoami)")
    if not isinstance(git_head, str) or not git_head.strip():
        raise ValueError(
            "git_head must be a non-empty string — record the commit "
            "sha so the attestation can be reproduced."
        )
    sha = compute_blob_sha256(blob)
    path = attestation_path_for(sha, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"attestation already exists at {path}; pass overwrite=True "
            "to replace it (and explain why in the new ruling_text — "
            "overwriting is a forensic loss)."
        )
    record = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "delta_sha256": sha,
        "delta_size_bytes": len(blob),
        "delta_path_at_signing": delta_path_at_signing,
        "approver": approver,
        "ruling_text": ruling_text,
        "signed_at_utc": signed_at_utc or time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "signed_by_user": signed_by_user,
        "git_head": git_head,
    }
    path.write_text(json.dumps(record, indent=2, sort_keys=True))
    return path
