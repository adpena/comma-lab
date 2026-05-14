#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""lane_c_keygen.py — generate an Ed25519 keypair for Lane C trust root.

Codex R5-3 #1 (2026-04-27): the Lane C compliance gate uses Ed25519
detached signatures over a canonical-JSON serialization of the
attestation's trust-bearing fields. Each council member who is allowed
to issue Lane C compliance attestations must:

  1. Generate an Ed25519 keypair OUTSIDE the repo (this tool).
  2. Send ONLY the public key (hex) to be added to
     .omx/state/lane_c_compliance_attestations/trust_root_pubkeys.json
     in a normal commit/PR.
  3. Keep the PRIVATE key on their own machine (default ~/.config/pact/
     lane_c_signing_key.pem). When signing an attestation via
     tools/sign_lane_c_compliance.py the tool reads the PEM file.

Threat model
------------
- Anyone with write access to the repo can edit attestation JSON files
  but NOT forge a signature without the corresponding private key.
- Anyone with write access to the trust root file CAN add their own
  pubkey, so changes to the trust root MUST be reviewed (the file is
  committed to git like any other source file).
- The signing tool refuses to sign with a missing/invalid PEM file.

Usage
-----
    python tools/lane_c_keygen.py \\
        --out ~/.config/pact/lane_c_signing_key.pem \\
        --approver-id yousfi

Outputs
-------
- Writes the private key as PEM (PKCS#8) to --out (mode 0600).
- Prints the pubkey hex + a JSON snippet to paste into the trust root file.

This tool MUST NOT write to the repo. The pubkey hex is printed for
the operator to manually paste into trust_root_pubkeys.json (the
purpose of that ceremony is exactly to make adding a new approver a
human-reviewed action).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--out", type=Path,
        default=Path.home() / ".config" / "pact" / "lane_c_signing_key.pem",
        help="Where to write the PEM-encoded Ed25519 private key. "
             "MUST be outside the repo. Default: "
             "~/.config/pact/lane_c_signing_key.pem",
    )
    p.add_argument(
        "--approver-id", type=str, required=True,
        help="The approver_id this key will sign for (e.g. 'yousfi', "
             "'fridrich', 'adpena'). Recorded in the printed registry "
             "snippet only — the operator pastes it into the trust root.",
    )
    p.add_argument(
        "--comment", type=str, default="",
        help="Optional free-form comment recorded in the trust-root "
             "snippet (e.g. 'M5 Max keypair, generated 2026-04-27').",
    )
    p.add_argument(
        "--force", action="store_true", default=False,
        help="Overwrite an existing PEM file at --out. Default: refuse "
             "(losing a key destroys all attestations signed by it).",
    )
    args = p.parse_args()

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

    out_path = args.out.expanduser().resolve()
    # Refuse to write into the repo (best-effort heuristic — the canonical
    # repo root contains a .git directory near this script).
    repo_marker = Path(__file__).resolve().parents[1] / ".git"
    if repo_marker.exists():
        repo_root = repo_marker.parent
        try:
            out_path.relative_to(repo_root)
            print(
                f"FATAL: refusing to write the private key inside the "
                f"repo ({out_path} is under {repo_root}). Pass --out to a "
                f"path outside the repo (default ~/.config/pact/...).",
                file=sys.stderr,
            )
            return 2
        except ValueError:
            pass  # outside repo — good

    if out_path.exists() and not args.force:
        print(
            f"FATAL: {out_path} already exists. Pass --force to overwrite. "
            "Note: overwriting destroys the previous key — every "
            "attestation it signed becomes unverifiable.",
            file=sys.stderr,
        )
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)

    private_key = Ed25519PrivateKey.generate()
    pem_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    # Write atomically with mode 0600 BEFORE the bytes hit the FS.
    fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(pem_bytes)
        f.flush()
        os.fsync(f.fileno())
    # Ensure mode is 0600 even if the umask was unusual.
    os.chmod(out_path, 0o600)

    pubkey_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    pubkey_hex = pubkey_bytes.hex()

    snippet = {
        args.approver_id: {
            "pubkey_hex": pubkey_hex,
            "comment": args.comment or "",
            "registered_at": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(),
            ),
        }
    }

    print("=== Ed25519 keypair generated ===")
    print(f"  Private key (PEM): {out_path}  (mode 0600)")
    print(f"  Pubkey hex:        {pubkey_hex}")
    print()
    print("Trust-root snippet (paste into")
    print("  .omx/state/lane_c_compliance_attestations/trust_root_pubkeys.json")
    print("merging with any existing entries):")
    print()
    print(json.dumps(snippet, indent=2, sort_keys=True))
    print()
    print("After merging, commit the trust-root file. The matching")
    print("private key NEVER leaves your machine — never commit it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
