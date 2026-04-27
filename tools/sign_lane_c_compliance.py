#!/usr/bin/env python3
"""sign_lane_c_compliance.py — write a Lane C δ.bin compliance attestation.

CODEX R5-2 #4 fix (2026-04-27): the Lane C compliance gate trusts the δ.bin
header's ``compliance_status`` field for ``rejected``/``pending_ruling`` (the
optimizer can write either) but REQUIRES an external attestation file for
``approved``. This tool is the only way to produce that attestation.

Usage
-----
    python tools/sign_lane_c_compliance.py \\
        --delta-bin path/to/delta.bin \\
        --approver yousfi \\
        --ruling-text "PR #35 ruling: Lane C δ approved as in-archive bundle"

The attestation is written to:
    .omx/state/lane_c_compliance_attestations/<sha256>.json

where ``<sha256>`` is the SHA256 of the δ.bin bytes. The archive builder
``experiments/build_baseline_archive.py`` cross-checks this file before
allowing a δ marked ``compliance_status="approved"`` into the archive.

Hard refusals
-------------
- Empty ``--ruling-text``: the attestation must record WHY this δ was
  approved (PR URL, council decision, etc). A bare timestamp is not
  enough forensic evidence.
- Empty ``--approver``: must record the identity (council, yousfi, …).
- Pre-existing attestation at the target path: pass ``--overwrite`` to
  replace; doing so loses forensic history of the previous ruling.
- Missing or empty ``--delta-bin`` path.

What this tool does NOT do
--------------------------
- It does not modify the δ.bin's wire-format header. The optimizer is
  the only writer for that field and can only set ``pending_ruling`` or
  ``rejected``. To MOVE a δ from pending to approved, you sign here AND
  re-build the archive with a δ whose header still says
  ``pending_ruling`` plus the matching attestation; the archive
  builder reads BOTH.

  Wait — that's not quite right. The archive builder reads the δ.bin
  header's ``compliance_status``. If you want the archive to ship as
  approved, the header must say ``approved``. Since the optimizer
  refuses to write ``approved``, the operator MUST use a small post-
  processing step to flip the header on a pending δ AFTER signing.
  Right now the simplest way is to re-pack the δ via
  ``tac.uniward_delta.pack_sparse_delta(..., compliance_status="approved")``
  using the in-memory tensors — but that recomputes the bytes and
  invalidates the SHA. The cleanest workflow today is:

      1. Build δ → δ.bin with header pending_ruling.
      2. Sign attestation against THAT δ's sha (this tool).
      3. To bundle: pass ``--with-uniward-delta delta.bin
         --allow-pending-compliance`` to the archive builder, which
         records both the pending status AND the attestation in
         provenance.

  In other words: signing alone does not promote a δ to approved at
  the wire level. Promotion requires either a header-rewrite tool
  (intentionally not provided — see Codex R5-2 #4 reasoning) or
  re-packing. The attestation IS however the trust anchor for any
  build that ships an approved δ.
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

    print(f"=== Lane C compliance attestation written ===")
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
    print()
    print("Next steps:")
    print("  - Commit the attestation file so it lands in the audit trail:")
    print(f"      git add {path.relative_to(args.root)}")
    print("  - When bundling this δ into a submission archive, the")
    print("    archive builder will cross-check delta.bin sha against this")
    print("    attestation. If they match, an approved-status δ may ship.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
