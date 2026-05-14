# SPDX-License-Identifier: MIT
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
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Codex R5-3 #1: type-hint references for the Ed25519 trust-root.
    # Imported under TYPE_CHECKING so static analyzers (ruff F821) resolve
    # the names without requiring cryptography at import-time for callers
    # that don't actually use the trust-root path.
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )


__all__ = [
    "ATTESTATION_DIR",
    "ATTESTATION_SCHEMA_VERSION",
    "TRUST_ROOT_FILENAME",
    "AttestationMissing",
    "AttestationMismatch",
    "AttestationMalformed",
    "AttestationSignatureInvalid",
    "AttestationApproverNotInTrustRoot",
    "TrustRootMissing",
    "TrustRootMalformed",
    "Attestation",
    "compute_blob_sha256",
    "attestation_path_for",
    "load_attestation",
    "verify_attestation_for_blob",
    "write_attestation",
    "canonical_signed_payload",
    "load_trust_root",
    "trust_root_path",
    "INTERNAL_PROMOTION_TOKEN",
    "verify_internal_promotion_token",
]


ATTESTATION_DIR = Path(".omx") / "state" / "lane_c_compliance_attestations"
# Codex R5-3 #1 fix (2026-04-27): bumped from 1 → 2 to mark the introduction of
# the mandatory ``signature_hex`` Ed25519 trust-root field. v1 attestations
# (structure-only) are now refused by ``verify_attestation_for_blob`` because
# the gate they backed is forgeable by the same operator who's asking for
# approval. There is intentionally NO backward-compat fallback — operators
# must re-sign any v1 attestation against the new trust root.
ATTESTATION_SCHEMA_VERSION = 2

# Codex R5-3 #4 fix: strict hex-only sha256 regex used by attestation_path_for.
# Without this, ``attestation_path_for("../etc/passwd")`` would happily compose
# a path traversal. Length-only check was insufficient — the original check
# accepted ANY 64-char string including "/", "..", control chars.
_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")

# Codex R5-3 #1 fix: trust-root pubkey registry filename. Lives next to the
# attestations themselves so audit-trail diff includes BOTH the registry and
# the attestations that depend on it.
TRUST_ROOT_FILENAME = "trust_root_pubkeys.json"

# Required keys in any attestation JSON. Missing keys ⇒ AttestationMalformed.
# Codex R5-3 #1 fix: ``signature_hex`` is now MANDATORY. Old v1 structure-only
# attestations are rejected (no backward compatibility — that compatibility was
# the bypass).
_REQUIRED_KEYS = frozenset({
    "schema_version", "delta_sha256", "approver", "ruling_text",
    "signed_at_utc", "signed_by_user", "git_head", "signature_hex",
})

# The exact set of fields that go into the Ed25519 canonical-JSON payload.
# DO NOT CHANGE THE ORDER OR THE SET — any change is a wire break that
# invalidates every existing attestation. ``signature_hex`` is intentionally
# NOT in this set: it's the SIGNATURE OVER these fields, not part of them.
# ``delta_size_bytes`` and ``delta_path_at_signing`` are informational and
# also not signed (an operator legitimately moves files post-sign).
_SIGNED_FIELDS = (
    "schema_version", "delta_sha256", "approver", "ruling_text",
    "signed_at_utc", "signed_by_user", "git_head",
)

# Codex R5-3 #2 fix: only the dedicated promotion tool may set
# ``compliance_status='approved'`` in a δ.bin. Library callers (tests,
# the optimizer, ad-hoc scripts) are refused. The tool passes this exact
# string as ``_internal_promotion_token`` to ``pack_sparse_delta``. Tests
# that need to forge approved blobs (for negative-path testing of the
# verifier) also pass this token explicitly, which is the entire point —
# the test exercises the SAME codepath the promotion tool uses, so a
# library refactor that breaks the gate is caught.
INTERNAL_PROMOTION_TOKEN = (
    "lane_c_internal_promotion_token_DO_NOT_PASS_FROM_CALLER_CODE_v1"
)


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


class AttestationSignatureInvalid(Exception):
    """Codex R5-3 #1: attestation file is structurally valid and the
    approver is in the trust root, but the Ed25519 signature does not
    verify against the canonical-JSON serialization of the signed
    fields. Treated as forged — the gate fails CLOSED."""


class AttestationApproverNotInTrustRoot(Exception):
    """Codex R5-3 #1: the attestation's ``approver`` field does not appear
    in the trust-root pubkey registry. Refused regardless of whether
    the signature would have verified — only allowlisted approvers can
    issue Lane C compliance attestations."""


class TrustRootMissing(Exception):
    """Codex R5-3 #1: ``.omx/state/lane_c_compliance_attestations/
    trust_root_pubkeys.json`` does not exist. Without the registry no
    approver can be allowlisted and no attestation can pass; verifier
    fails CLOSED."""


class TrustRootMalformed(Exception):
    """Codex R5-3 #1: the trust-root JSON is present but malformed —
    not a JSON object, missing pubkey_hex fields, pubkey_hex not parseable
    as 32-byte Ed25519 public key, etc."""


@dataclass(frozen=True)
class Attestation:
    """A loaded, validated attestation record.

    Attributes mirror the fields ``write_attestation`` produces. Use
    ``verify_attestation_for_blob`` rather than constructing this
    directly — the verifier handles the SHA cross-check AND the
    Ed25519 signature verification.
    """

    schema_version: int
    delta_sha256: str
    approver: str
    ruling_text: str
    signed_at_utc: str
    signed_by_user: str
    git_head: str
    # Codex R5-3 #1: Ed25519 signature over ``canonical_signed_payload(record)``.
    # The trust root for ``approver`` is in
    # ``.omx/state/lane_c_compliance_attestations/trust_root_pubkeys.json``.
    signature_hex: str = ""
    # Optional / informational — not part of the trust check, not signed.
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
        sha256: hex digest from ``compute_blob_sha256``. MUST match the
            strict regex ``^[0-9a-fA-F]{64}$`` — anything else is a
            ``ValueError`` (Codex R5-3 #4: prevents path traversal via a
            crafted "sha" containing slashes).
        root: optional override for the repo root (defaults to CWD).
            Tests use this to point at a tmp dir without chdir.

    Returns:
        Absolute path to the would-be attestation JSON. The file may
        not yet exist; the caller decides whether that's OK.

    Raises:
        ValueError: ``sha256`` is not a 64-char hex string, OR the
            resolved path escapes the canonical attestation directory
            (defense-in-depth: even if the regex were ever weakened,
            the path-relative-check still refuses traversal).
    """
    if not isinstance(sha256, str) or not _SHA256_HEX_RE.match(sha256):
        raise ValueError(
            f"sha256 must match ^[0-9a-fA-F]{{64}}$ (Codex R5-3 #4 — strict "
            f"hex-only check); got {sha256!r}. The hex constraint is what "
            "prevents path traversal in the canonical attestation path."
        )
    base = Path(root) if root is not None else Path.cwd()
    attestation_dir = (base / ATTESTATION_DIR).resolve()
    candidate = (attestation_dir / f"{sha256}.json").resolve()
    # Defense-in-depth (Codex R5-3 #4): even if the regex above were ever
    # weakened, the candidate must STILL resolve to a path under the
    # attestation directory. ``Path.is_relative_to`` is the safe primitive
    # introduced in Python 3.9.
    try:
        candidate.relative_to(attestation_dir)
    except ValueError as e:
        raise ValueError(
            f"Codex R5-3 #4: refusing path-traversal attestation path. "
            f"Computed candidate={candidate!r} escapes attestation_dir="
            f"{attestation_dir!r}. sha256 input={sha256!r}."
        ) from e
    return candidate


def trust_root_path(root: Path | str | None = None) -> Path:
    """Return the canonical path to the trust-root pubkey registry.

    Codex R5-3 #1: this file maps approver IDs to Ed25519 pubkey hex.
    The verifier refuses any attestation whose ``approver`` is not a key
    in this file, and refuses any signature that doesn't verify against
    the registered pubkey.
    """
    base = Path(root) if root is not None else Path.cwd()
    return base / ATTESTATION_DIR / TRUST_ROOT_FILENAME


def canonical_signed_payload(record: dict) -> bytes:
    """Codex R5-3 #1: produce the canonical-JSON serialization of the
    attestation fields that get signed.

    The serialization rules are deterministic and minimal:
      - Only the keys in ``_SIGNED_FIELDS`` are included.
      - ``json.dumps(..., sort_keys=True, separators=(',', ':'),
        ensure_ascii=False)`` — no whitespace, no Unicode escapes,
        sorted keys.
      - UTF-8 encoded.

    A signature over this payload binds the approver to EVERY one of the
    signed fields. Changing any of them (e.g. swapping ruling text after
    signing) breaks the signature.

    Args:
        record: dict containing at least the keys in ``_SIGNED_FIELDS``.

    Returns:
        UTF-8 bytes ready for ``Ed25519PrivateKey.sign(...)``.

    Raises:
        AttestationMalformed: a required signed field is missing.
    """
    missing = [k for k in _SIGNED_FIELDS if k not in record]
    if missing:
        raise AttestationMalformed(
            f"canonical_signed_payload: record missing signed fields "
            f"{missing!r}. The signature would not commit to all required "
            "trust-bearing fields."
        )
    payload = {k: record[k] for k in _SIGNED_FIELDS}
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def verify_internal_promotion_token(token: str | None) -> bool:
    """Codex R5-3 #2: constant-time-ish comparison for the internal
    promotion token. ``hmac.compare_digest`` is the standard primitive
    for short fixed-length secret-equality checks; we reuse it here to
    avoid leaking timing information about prefix matches even though
    the token is not strictly secret (it's hard-coded in this module).
    """
    if not isinstance(token, str):
        return False
    import hmac
    return hmac.compare_digest(token, INTERNAL_PROMOTION_TOKEN)


def load_trust_root(
    root: Path | str | None = None,
) -> dict[str, "Ed25519PublicKey"]:  # type: ignore[name-defined]
    """Codex R5-3 #1: load and validate the trust-root pubkey registry.

    File format::

        {
          "<approver_id>": {
              "pubkey_hex": "<64-char hex of 32-byte Ed25519 pub key>",
              "comment": "<free-form>",
              "registered_at": "<ISO8601 timestamp>"
          },
          ...
        }

    Returns a dict mapping approver_id → Ed25519PublicKey. Raises
    ``TrustRootMissing`` or ``TrustRootMalformed`` on any problem.

    Empty registry is permitted (returns ``{}``) — but in that state NO
    attestation can verify, which is the safe default before any council
    member registers their pubkey.
    """
    # Lazy-import cryptography so module import doesn't pull a heavy dep
    # for callers that never touch the verifier (e.g. unpack_sparse_delta).
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
    except ImportError as e:
        raise TrustRootMalformed(
            "Codex R5-3 #1: 'cryptography' library is required for "
            "Lane C attestation verification. Install via 'uv pip install "
            "cryptography' (already in pyproject.toml dependencies)."
        ) from e

    path = trust_root_path(root)
    if not path.exists():
        raise TrustRootMissing(
            f"Codex R5-3 #1: trust-root pubkey registry not found at "
            f"{path}. Without the registry no approver can be allowlisted "
            f"and no Lane C compliance attestation can pass.\n"
            f"To bootstrap: have each council member generate an Ed25519 "
            f"keypair via 'python tools/lane_c_keygen.py' (private key "
            f"OUTSIDE the repo), then write the pubkey hex into the file."
        )
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise TrustRootMalformed(
            f"trust root at {path} is not valid JSON: {e!r}"
        ) from e
    if not isinstance(data, dict):
        raise TrustRootMalformed(
            f"trust root at {path} root is not a JSON object: "
            f"{type(data).__name__}"
        )
    keys: dict[str, "Ed25519PublicKey"] = {}
    for approver_id, entry in data.items():
        if not isinstance(approver_id, str) or not approver_id.strip():
            raise TrustRootMalformed(
                f"trust root at {path} contains an empty/non-string "
                f"approver_id: {approver_id!r}"
            )
        if not isinstance(entry, dict):
            raise TrustRootMalformed(
                f"trust root at {path} has non-object entry for "
                f"approver={approver_id!r}: {entry!r}"
            )
        pubkey_hex = entry.get("pubkey_hex")
        if not isinstance(pubkey_hex, str) or len(pubkey_hex) != 64:
            raise TrustRootMalformed(
                f"trust root at {path} has malformed pubkey_hex for "
                f"approver={approver_id!r}: must be a 64-char hex string "
                f"(32 bytes Ed25519 public key); got {pubkey_hex!r}"
            )
        try:
            pubkey_bytes = bytes.fromhex(pubkey_hex)
        except ValueError as e:
            raise TrustRootMalformed(
                f"trust root at {path}: pubkey_hex for approver="
                f"{approver_id!r} is not valid hex: {e!r}"
            ) from e
        try:
            pubkey = Ed25519PublicKey.from_public_bytes(pubkey_bytes)
        except Exception as e:
            raise TrustRootMalformed(
                f"trust root at {path}: pubkey for approver={approver_id!r} "
                f"failed Ed25519 parse: {e!r}"
            ) from e
        keys[approver_id] = pubkey
    return keys


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
    # Codex R5-3 #1: signature_hex is mandatory; validate format here so
    # the verifier can rely on the bytes parsing cleanly.
    signature_hex = data["signature_hex"]
    if not isinstance(signature_hex, str) or len(signature_hex) != 128:
        raise AttestationMalformed(
            f"attestation at {path} has malformed signature_hex="
            f"{signature_hex!r}; must be a 128-char hex string (64 bytes "
            "Ed25519 signature). v1 structure-only attestations are no "
            "longer accepted (Codex R5-3 #1) — re-sign with "
            "tools/sign_lane_c_compliance.py."
        )
    try:
        bytes.fromhex(signature_hex)
    except ValueError as e:
        raise AttestationMalformed(
            f"attestation at {path}: signature_hex is not valid hex: {e!r}"
        ) from e
    return Attestation(
        schema_version=schema_version,
        delta_sha256=delta_sha256.lower(),
        approver=approver,
        ruling_text=ruling_text,
        signed_at_utc=data["signed_at_utc"],
        signed_by_user=data["signed_by_user"],
        git_head=data["git_head"],
        signature_hex=signature_hex.lower(),
        delta_size_bytes=data.get("delta_size_bytes"),
        delta_path_at_signing=data.get("delta_path_at_signing"),
    )


def verify_attestation_for_blob(
    blob: bytes, *, root: Path | str | None = None,
) -> Attestation:
    """Look up the canonical attestation for a δ blob and FULLY verify it.

    Codex R5-3 #1 fix (2026-04-27): the verifier no longer trusts a
    well-formed JSON file — it now requires:

      1. Compute sha256 of the actual δ.bin bytes.
      2. Read ``.omx/state/lane_c_compliance_attestations/<sha>.json``.
         Missing ⇒ AttestationMissing.
      3. Parse + structural-validate (now requires signature_hex). Bad
         ⇒ AttestationMalformed.
      4. Cross-check that the attestation's ``delta_sha256`` field
         equals the computed sha (case-insensitive). Bad ⇒
         AttestationMismatch.
      5. Look up ``approver`` in the trust-root pubkey registry. Not
         found ⇒ AttestationApproverNotInTrustRoot.
      6. Recompute ``canonical_signed_payload`` over the loaded record
         and verify the Ed25519 signature against the registered
         pubkey. Bad ⇒ AttestationSignatureInvalid.

    The trust root file at
    ``.omx/state/lane_c_compliance_attestations/trust_root_pubkeys.json``
    lists the council members whose signatures count as approval. Only
    the pubkeys are committed to the repo; the matching private keys
    live OUTSIDE the repo on each council member's machine.

    Returns the parsed Attestation on full success; raises a specific
    exception type on each failure mode so the caller (the archive
    builder) can produce actionable error messages.
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
    # ── Codex R5-3 #1: cryptographic verification ───────────────────────
    trust_root = load_trust_root(root)  # may raise TrustRootMissing/Malformed
    pubkey = trust_root.get(attestation.approver)
    if pubkey is None:
        raise AttestationApproverNotInTrustRoot(
            f"Codex R5-3 #1: attestation at {path} names approver="
            f"{attestation.approver!r} which is NOT registered in the "
            f"trust-root pubkey file ({trust_root_path(root)}). Only "
            f"approvers whose pubkeys are committed to the trust root "
            f"can issue Lane C compliance attestations.\n"
            f"Registered approvers: {sorted(trust_root.keys())!r}\n"
            f"To add: have {attestation.approver!r} generate a key with "
            f"'python tools/lane_c_keygen.py' and PR the pubkey hex "
            f"into the trust root."
        )
    # Re-serialize the SIGNED fields to canonical JSON. Anything outside
    # _SIGNED_FIELDS (including delta_size_bytes, delta_path_at_signing)
    # is intentionally NOT covered by the signature.
    record_for_sign = {
        "schema_version": attestation.schema_version,
        "delta_sha256": attestation.delta_sha256,
        "approver": attestation.approver,
        "ruling_text": attestation.ruling_text,
        "signed_at_utc": attestation.signed_at_utc,
        "signed_by_user": attestation.signed_by_user,
        "git_head": attestation.git_head,
    }
    payload = canonical_signed_payload(record_for_sign)
    try:
        signature_bytes = bytes.fromhex(attestation.signature_hex)
    except ValueError as e:  # pragma: no cover — load_attestation already validated
        raise AttestationSignatureInvalid(
            f"signature_hex for {path} is not valid hex despite passing "
            f"load_attestation: {e!r}"
        ) from e
    try:
        pubkey.verify(signature_bytes, payload)
    except Exception as e:
        # cryptography raises InvalidSignature, plus a few subclasses for
        # malformed sig length etc. We collapse all into one exception
        # type with maximum diagnostic context.
        raise AttestationSignatureInvalid(
            f"Codex R5-3 #1: Ed25519 signature on attestation at {path} "
            f"FAILED to verify against approver={attestation.approver!r}'s "
            f"registered pubkey. The attestation has been tampered with "
            f"(ruling_text changed, approver swapped, etc.) OR was not "
            f"signed by the registered key holder. Underlying error: {e!r}"
        ) from e
    return attestation


def write_attestation(
    *,
    blob: bytes,
    approver: str,
    ruling_text: str,
    signed_by_user: str,
    git_head: str,
    private_key: "Ed25519PrivateKey | bytes",  # type: ignore[name-defined]
    delta_path_at_signing: str | None = None,
    root: Path | str | None = None,
    signed_at_utc: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a SIGNED attestation JSON for a δ blob at the canonical path.

    Codex R5-3 #1 fix (2026-04-27): a ``private_key`` parameter is now
    REQUIRED (no default). The function signs the canonical-JSON
    serialization of the trust-bearing fields with Ed25519 and writes
    the signature into the attestation file. Without the matching
    pubkey in the trust root, the resulting attestation will not verify.

    Codex R5-3 #3 fix (2026-04-27): the file write is now ATOMIC. We
    use ``open(path, 'x')`` (O_CREAT|O_EXCL) when overwrite=False, which
    is race-safe across concurrent signers. For overwrite=True we write
    to a temp file in the same dir and ``os.replace`` over the target,
    which is also atomic on POSIX.

    Args:
        blob: the δ.bin bytes being approved. The file path is derived
            from sha256(blob) — same as the verifier expects.
        approver: free-form identity (e.g. "yousfi", "council"). Must be
            non-empty AND must match a key in the trust-root pubkey
            registry, otherwise ``verify_attestation_for_blob`` will
            refuse the resulting attestation.
        ruling_text: free-form justification (e.g. URL to council
            ruling thread, paragraph text). Must be non-empty.
        signed_by_user: typically ``getpass.getuser()`` or whoami.
        git_head: current git commit sha for forensic context.
        private_key: Ed25519 private key. Either an already-loaded
            ``cryptography.hazmat.primitives.asymmetric.ed25519.
            Ed25519PrivateKey`` instance or 32 raw bytes. Tests pass
            in-memory keys; the signing tool loads from a PEM file
            OUTSIDE the repo.
        delta_path_at_signing: optional; recorded so a future auditor
            can find the build artifact. NOT signed.
        root: optional override for the repo root (tests use this).
        signed_at_utc: ISO8601 timestamp; if None, computed now.
        overwrite: if False (default) and a file already exists at the
            target path, refuse atomically — pre-existing attestation
            may have different ruling text and overwriting is a
            forensic loss. (Codex R5-3 #3: atomic exclusive-create.)

    Returns:
        Path to the written attestation file.

    Raises:
        ValueError: empty approver / ruling_text / git_head /
            signed_by_user, or invalid private_key argument.
        FileExistsError: an attestation already exists at the path
            and ``overwrite=False`` — RACE-SAFE via O_EXCL.
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

    # Codex R5-3 #1: normalize private_key argument and sign.
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
    except ImportError as e:
        raise ValueError(
            "Codex R5-3 #1: 'cryptography' library is required to write "
            "signed attestations. Install via 'uv pip install cryptography'."
        ) from e
    if isinstance(private_key, (bytes, bytearray, memoryview)):
        pk_bytes = bytes(private_key)
        if len(pk_bytes) != 32:
            raise ValueError(
                f"Codex R5-3 #1: private_key bytes must be 32 (Ed25519 "
                f"seed), got {len(pk_bytes)}"
            )
        try:
            ed_key = Ed25519PrivateKey.from_private_bytes(pk_bytes)
        except Exception as e:
            raise ValueError(
                f"Codex R5-3 #1: failed to parse private_key bytes as "
                f"Ed25519: {e!r}"
            ) from e
    elif isinstance(private_key, Ed25519PrivateKey):
        ed_key = private_key
    else:
        raise ValueError(
            f"Codex R5-3 #1: private_key must be Ed25519PrivateKey or "
            f"32-byte seed; got {type(private_key).__name__}"
        )

    sha = compute_blob_sha256(blob)
    path = attestation_path_for(sha, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build the trust-bearing record FIRST and sign it.
    signed_record = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "delta_sha256": sha,
        "approver": approver,
        "ruling_text": ruling_text,
        "signed_at_utc": signed_at_utc or time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "signed_by_user": signed_by_user,
        "git_head": git_head,
    }
    payload = canonical_signed_payload(signed_record)
    signature = ed_key.sign(payload)
    full_record = dict(signed_record)
    full_record["signature_hex"] = signature.hex()
    # Informational, NOT signed.
    full_record["delta_size_bytes"] = len(blob)
    full_record["delta_path_at_signing"] = delta_path_at_signing

    serialized = json.dumps(full_record, indent=2, sort_keys=True)

    # ── Codex R5-3 #3: atomic write ─────────────────────────────────────
    if not overwrite:
        # O_CREAT | O_EXCL is atomic against concurrent signers — at
        # most one process gets the create. Others see FileExistsError
        # cleanly even if they raced past a path.exists() probe.
        try:
            fd = os.open(
                path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644,
            )
        except FileExistsError:
            raise FileExistsError(
                f"attestation already exists at {path}; pass overwrite=True "
                "to replace it (and explain why in the new ruling_text — "
                "overwriting is a forensic loss)."
            )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(serialized)
                f.flush()
                os.fsync(f.fileno())
        except BaseException:
            # If we crashed mid-write, the partial file is misleading;
            # remove it so the next signer can retry cleanly.
            try:
                path.unlink()
            except OSError:
                pass
            raise
    else:
        # Overwrite path: write to a temp file in the same dir, fsync,
        # then atomic rename. ``tempfile.NamedTemporaryFile`` keeps the
        # tmp file in the SAME directory so ``os.replace`` is a true
        # cross-FS-safe atomic rename (POSIX guarantees rename within
        # one directory is atomic).
        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(tmp_fd, "w") as f:
                f.write(serialized)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    return path
