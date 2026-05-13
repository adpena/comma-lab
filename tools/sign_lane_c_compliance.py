#!/usr/bin/env python3
"""sign_lane_c_compliance.py — write a SIGNED Lane C δ.bin compliance attestation.

Codex R5-3 #1 fix (2026-04-27): attestations now carry a mandatory Ed25519
detached signature over the canonical-JSON serialization of the trust-bearing
fields. The signing key is loaded from a PEM file OUTSIDE the repo (default
``~/.config/pact/lane_c_signing_key.pem``); only the matching pubkey is
committed (in
``.omx/state/lane_c_compliance_attestations/trust_root_pubkeys.json``).

Without ``--private-key`` pointing at a valid Ed25519 PEM, this tool refuses
to write — there is intentionally no fallback to "structure only" attestations
because that was the bypass.

Usage
-----
    python tools/sign_lane_c_compliance.py \\
        --delta-bin path/to/delta.bin \\
        --approver yousfi \\
        --ruling-text "PR #35 ruling: Lane C δ approved as in-archive bundle" \\
        --private-key ~/.config/pact/lane_c_signing_key.pem

The attestation is written to:
    .omx/state/lane_c_compliance_attestations/<sha256>.json

where ``<sha256>`` is the SHA256 of the δ.bin bytes. The archive builder
``experiments/build_baseline_archive.py`` cross-checks this file (Ed25519
sig + approver allowlist + δ-sha cross-check) before allowing a δ marked
``compliance_status="approved"`` into the archive.

Hard refusals
-------------
- Empty ``--ruling-text``: the attestation must record WHY this δ was
  approved (PR URL, council decision, etc). A bare timestamp is not
  enough forensic evidence.
- Empty ``--approver``: must record the identity (council, yousfi, …).
- Missing or unreadable ``--private-key`` PEM file.
- Pre-existing attestation at the target path: pass ``--overwrite`` to
  replace; doing so loses forensic history of the previous ruling.
- Missing or empty ``--delta-bin`` path.

What this tool does NOT do
--------------------------
- It does not modify the δ.bin's wire-format header. To produce an
  approved δ.bin (status='approved' in the wire header), the workflow is:

      1. Build a pending δ.bin via experiments/optimize_uniward_delta.py.
      2. Sign attestation against THAT pending δ's sha (this tool).
      3. Run tools/promote_lane_c_to_approved.py with the pending δ + the
         attestation. The promoter verifies the attestation against the
         trust root, then re-emits a NEW δ.bin with header
         compliance_status='approved'. The new δ has a DIFFERENT sha.
      4. Sign a fresh attestation against the promoted δ.bin's sha.
      5. Bundle the promoted δ.bin + matching attestation into the
         archive via experiments/build_baseline_archive.py.

  The promotion tool is the ONLY caller that may write
  compliance_status='approved' into a δ.bin wire header. This is the
  Codex R5-3 #2 fix.
"""
from __future__ import annotations

import argparse
import getpass
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.lane_c_compliance import (  # noqa: E402
    write_attestation, compute_blob_sha256, attestation_path_for,
)


def _git_head(repo: Path) -> str:
    """Return the current git HEAD sha. Returns 'unknown' if the repo
    isn't a git checkout (e.g. running from a tarball). The attestation
    schema requires this field to be non-empty so we substitute a
    sentinel rather than failing the whole sign — but we do tag it
    obviously."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return result.stdout.strip() or "unknown-empty-output"
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired) as e:
        print(
            f"[sign] WARNING: git rev-parse HEAD failed ({e!r}); "
            "recording 'unknown-not-a-git-repo' in attestation. The "
            "attestation will still be honored by the archive builder, "
            "but the forensic context will be incomplete.",
            file=sys.stderr,
        )
        return "unknown-not-a-git-repo"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--delta-bin", type=Path, required=True,
                   help="Path to the δ.bin being attested. The attestation "
                        "is keyed on this file's SHA256.")
    p.add_argument("--approver", type=str, required=True,
                   help="Identity of the approving party (e.g. 'yousfi', "
                        "'council', 'fridrich'). Must be non-empty. "
                        "Recorded in the attestation JSON.")
    p.add_argument("--ruling-text", type=str, required=True,
                   help="Free-form ruling text justifying approval. Should "
                        "include a council decision URL or paragraph. "
                        "MUST be non-empty — empty rulings are refused.")
    p.add_argument("--root", type=Path, default=REPO,
                   help="Repository root (defaults to the script's parent). "
                        "The attestation lands at "
                        "<root>/.omx/state/lane_c_compliance_attestations/<sha>.json")
    p.add_argument("--overwrite", action="store_true", default=False,
                   help="If an attestation already exists at the canonical "
                        "path, replace it. The previous ruling text is "
                        "LOST — only use when correcting an erroneous "
                        "attestation, and explain why in the new ruling.")
    p.add_argument("--signed-by-user", type=str, default=None,
                   help="Override the auto-detected user (defaults to "
                        "getpass.getuser()). Useful for paper-trail "
                        "rebuilds; production use should leave it auto.")
    p.add_argument("--private-key", type=Path,
                   default=Path.home() / ".config" / "pact"
                   / "lane_c_signing_key.pem",
                   help="Path to the PEM-encoded Ed25519 private key used "
                        "to sign the attestation. Default: "
                        "~/.config/pact/lane_c_signing_key.pem. The matching "
                        "public key MUST be registered in the trust root "
                        "(.omx/state/lane_c_compliance_attestations/"
                        "trust_root_pubkeys.json) under the same approver_id "
                        "as --approver, otherwise the attestation will not "
                        "verify. Generate keys via tools/lane_c_keygen.py.")
    args = p.parse_args()

    # Pre-flight: refuse empty strings BEFORE doing any I/O. (write_attestation
    # also validates, but we want a clean error message at the CLI layer.)
    if not args.ruling_text.strip():
        print(
            "FATAL: --ruling-text is empty. The attestation MUST include "
            "a non-empty justification (council decision URL or paragraph). "
            "An attestation without a ruling is not auditable.",
            file=sys.stderr,
        )
        return 2
    if not args.approver.strip():
        print(
            "FATAL: --approver is empty. The attestation MUST record the "
            "identity of the party signing it (e.g. 'yousfi', 'council').",
            file=sys.stderr,
        )
        return 2
    if not args.delta_bin.exists():
        print(
            f"FATAL: --delta-bin does not exist: {args.delta_bin}",
            file=sys.stderr,
        )
        return 2

    blob = args.delta_bin.read_bytes()
    if not blob:
        print(
            f"FATAL: --delta-bin is empty: {args.delta_bin}. Refusing to "
            "attest a zero-byte δ.",
            file=sys.stderr,
        )
        return 2

    # Codex R5-3 #1: load the Ed25519 private key from PEM.
    if not args.private_key.exists():
        print(
            f"FATAL: --private-key file does not exist: {args.private_key}. "
            f"Generate one via 'python tools/lane_c_keygen.py "
            f"--approver-id {args.approver}' (the private key MUST live "
            "OUTSIDE the repo).",
            file=sys.stderr,
        )
        return 2
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
    except ImportError:
        print(
            "FATAL: 'cryptography' library required. Install via "
            "'uv pip install cryptography'.",
            file=sys.stderr,
        )
        return 2
    try:
        pem_bytes = args.private_key.read_bytes()
    except OSError as e:
        print(
            f"FATAL: failed to read --private-key {args.private_key}: {e!r}",
            file=sys.stderr,
        )
        return 2
    try:
        private_key = serialization.load_pem_private_key(
            pem_bytes, password=None,
        )
    except Exception as e:
        print(
            f"FATAL: failed to parse --private-key {args.private_key} as "
            f"PEM: {e!r}. Did you generate it with tools/lane_c_keygen.py?",
            file=sys.stderr,
        )
        return 2
    if not isinstance(private_key, Ed25519PrivateKey):
        print(
            f"FATAL: --private-key {args.private_key} is not an Ed25519 "
            f"key (got {type(private_key).__name__}). Lane C attestations "
            "MUST use Ed25519. Re-generate via tools/lane_c_keygen.py.",
            file=sys.stderr,
        )
        return 2

    sha = compute_blob_sha256(blob)
    target_path = attestation_path_for(sha, root=args.root)

    signed_by_user = args.signed_by_user or getpass.getuser()
    git_head = _git_head(args.root)
    signed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    try:
        path = write_attestation(
            blob=blob,
            approver=args.approver,
            ruling_text=args.ruling_text,
            signed_by_user=signed_by_user,
            git_head=git_head,
            private_key=private_key,
            delta_path_at_signing=str(args.delta_bin.resolve()),
            root=args.root,
            signed_at_utc=signed_at,
            overwrite=args.overwrite,
        )
    except FileExistsError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        print(
            "  Pass --overwrite if you intend to replace the existing "
            "attestation — record WHY in the new --ruling-text. Note "
            "that overwriting is a forensic loss; the previous ruling "
            "is irrecoverable from this directory after overwrite.",
            file=sys.stderr,
        )
        return 3
    except ValueError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2

    # Codex R5-3 #1: surface trust-root verification proactively so the
    # operator catches an "I signed but the pubkey is not registered"
    # mistake at sign time rather than at archive-build time.
    from tac.lane_c_compliance import (  # noqa: E402
        load_trust_root, TrustRootMissing, TrustRootMalformed,
    )
    trust_status = "verified"
    trust_detail = ""
    try:
        keys = load_trust_root(args.root)
        if args.approver not in keys:
            trust_status = "WARNING: approver NOT in trust root"
            trust_detail = (
                f"  → registered approvers: {sorted(keys.keys())!r}\n"
                f"  → add {args.approver}'s pubkey to "
                f"{args.root}/.omx/state/lane_c_compliance_attestations/"
                f"trust_root_pubkeys.json before bundling, otherwise "
                "verify_attestation_for_blob will refuse this attestation."
            )
        else:
            # Verify the signature we just wrote actually validates against
            # the registered pubkey. Catches "wrong key for this approver".
            from tac.lane_c_compliance import (
                verify_attestation_for_blob, AttestationSignatureInvalid,
            )
            try:
                verify_attestation_for_blob(blob, root=args.root)
                trust_status = "VERIFIED via trust root"
            except AttestationSignatureInvalid as e:
                trust_status = "ERROR: signature does not verify against trust root"
                trust_detail = (
                    f"  → signed with key whose pubkey doesn't match the "
                    f"one registered for approver={args.approver!r}.\n"
                    f"  → {e}"
                )
    except TrustRootMissing:
        trust_status = "WARNING: trust root file missing"
        trust_detail = (
            "  → bootstrap via 'python tools/lane_c_keygen.py "
            f"--approver-id {args.approver}' and paste the printed "
            "snippet into the trust root file."
        )
    except TrustRootMalformed as e:
        trust_status = f"ERROR: trust root malformed: {e}"

    print("=== Lane C compliance attestation written ===")
    print(f"  delta.bin:    {args.delta_bin}")
    print(f"  delta size:   {len(blob):,} bytes")
    print(f"  delta sha256: {sha}")
    print(f"  approver:     {args.approver}")
    print(f"  signed by:    {signed_by_user}")
    print(f"  signed at:    {signed_at}")
    print(f"  git HEAD:     {git_head}")
    print(f"  ruling:       {args.ruling_text[:120]}"
          + ("…" if len(args.ruling_text) > 120 else ""))
    print(f"  attestation:  {path}")
    print(f"  trust root:   {trust_status}")
    if trust_detail:
        print(trust_detail)
    print()
    print("Next steps:")
    print("  - Commit the attestation file so it lands in the audit trail:")
    # ``path`` may be a resolved (canonical) absolute path while ``args.root``
    # may not; normalize both before computing the display-only relative.
    try:
        rel = path.resolve().relative_to(args.root.resolve())
    except ValueError:
        rel = path  # fall back to absolute display
    print(f"      git add {rel}")
    print("  - When bundling this δ into a submission archive, the")
    print("    archive builder will cross-check delta.bin sha against this")
    print("    attestation AND verify the Ed25519 signature against the")
    print("    trust root. If both match, an approved-status δ may ship.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
