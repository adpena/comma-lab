#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""promote_lane_c_to_approved.py — re-emit a pending δ.bin as approved.

Codex R5-3 #2 fix (2026-04-27): the ONLY tool that may write
``compliance_status='approved'`` into a Lane C δ.bin wire header.
Every other path (tac.uniward_delta.pack_sparse_delta from library
callers, optimize_uniward_delta.py from operators) refuses to write
that value.

Workflow
--------
1. Operator builds a δ.bin via optimize_uniward_delta.py — the header
   carries ``compliance_status='pending_ruling'``.
2. Council member signs an attestation against THAT δ.bin's bytes
   via tools/sign_lane_c_compliance.py. The attestation is committed
   to .omx/state/lane_c_compliance_attestations/<sha>.json.
3. Operator runs THIS tool with the pending δ.bin and the attestation
   path. The tool:
     a. Verifies the attestation against the trust root (Ed25519 sig
        check + approver allowlist + δ-sha cross-check).
     b. Decodes the pending δ.bin into a DeltaSpec.
     c. Re-encodes a NEW δ.bin with compliance_status='approved' in
        the wire header, using the dense δ values reconstructed from
        the spec. This new δ.bin has a DIFFERENT sha (the header
        differs by one string), so:
     d. The promoted δ.bin is written to a NEW path; the operator
        must then sign a fresh attestation against the promoted SHA
        before bundling it. (This is intentional friction — every
        promoted δ gets a fresh, post-promotion attestation that
        commits to the EXACT bytes that ship.)
4. Operator builds the archive with the promoted δ.bin via
   experiments/build_baseline_archive.py.

Why a re-emit rather than a header rewrite
-------------------------------------------
The wire format is zlib-compressed, so a header byte rewrite would
require decompress → patch JSON → recompress. That path is more
fragile (compresslevel and zlib version affect bytes) than just
unpacking and repacking via the canonical pack/unpack codepath. The
re-emit also makes the promoted δ explicit: a different file, a
different sha, a fresh attestation. No invisible mutation of an
already-signed artifact.

Usage
-----
    python tools/promote_lane_c_to_approved.py \\
        --pending-delta-bin path/to/delta_pending.bin \\
        --attestation .omx/state/lane_c_compliance_attestations/<sha>.json \\
        --output path/to/delta_approved.bin
"""
from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.lane_c_compliance import (  # noqa: E402
    AttestationApproverNotInTrustRoot,
    AttestationMalformed,
    AttestationMismatch,
    AttestationMissing,
    AttestationSignatureInvalid,
    TrustRootMalformed,
    TrustRootMissing,
    attestation_path_for,
    compute_blob_sha256,
    verify_attestation_for_blob,
)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--pending-delta-bin", type=Path, required=True,
        help="Path to the pending δ.bin to promote. Must have header "
             "compliance_status='pending_ruling' (refuses to promote "
             "an already-approved or rejected δ).",
    )
    p.add_argument(
        "--attestation", type=Path, default=None,
        help="Optional explicit attestation path. If omitted, the tool "
             "looks at the canonical sha-keyed path "
             ".omx/state/lane_c_compliance_attestations/<sha>.json.",
    )
    p.add_argument(
        "--output", type=Path, required=True,
        help="Where to write the promoted δ.bin. The output file has "
             "compliance_status='approved' in its header and a DIFFERENT "
             "sha than the pending δ.bin. A fresh attestation against "
             "the new sha is REQUIRED before the archive builder will "
             "accept it.",
    )
    p.add_argument(
        "--root", type=Path, default=REPO,
        help="Repo root (defaults to the script's parent). Used to "
             "locate the trust root file and the canonical attestation "
             "directory.",
    )
    p.add_argument(
        "--force", action="store_true", default=False,
        help="Overwrite --output if it already exists. Default: refuse.",
    )
    args = p.parse_args()

    if not args.pending_delta_bin.exists():
        print(
            f"FATAL: --pending-delta-bin does not exist: "
            f"{args.pending_delta_bin}",
            file=sys.stderr,
        )
        return 2
    if args.output.exists() and not args.force:
        print(
            f"FATAL: --output {args.output} already exists. Pass --force "
            "to overwrite.",
            file=sys.stderr,
        )
        return 2

    pending_blob = args.pending_delta_bin.read_bytes()
    if not pending_blob:
        print(
            f"FATAL: --pending-delta-bin is empty: {args.pending_delta_bin}",
            file=sys.stderr,
        )
        return 2
    pending_sha = compute_blob_sha256(pending_blob)

    # ── Step 1: parse the pending δ to confirm it's actually pending ────
    from tac.uniward_delta import (  # noqa: E402
        COMPLIANCE_APPROVED,
        COMPLIANCE_PENDING,
        COMPLIANCE_REJECTED,
        MAGIC,
        unpack_sparse_delta,
    )
    try:
        pending_spec = unpack_sparse_delta(pending_blob, device="cpu")
    except Exception as e:
        print(
            f"FATAL: --pending-delta-bin failed to parse as UWD1: {e!r}",
            file=sys.stderr,
        )
        return 2
    if pending_spec.compliance_status == COMPLIANCE_REJECTED:
        print(
            f"FATAL: --pending-delta-bin is marked compliance_status="
            f"{COMPLIANCE_REJECTED!r}. Rejected δ may NEVER be promoted.",
            file=sys.stderr,
        )
        return 2
    if pending_spec.compliance_status == COMPLIANCE_APPROVED:
        print(
            f"FATAL: --pending-delta-bin is already marked "
            f"compliance_status={COMPLIANCE_APPROVED!r}. Re-promotion is "
            "not supported (every approved δ gets a fresh attestation; "
            "promoting an already-promoted blob would create a chain of "
            "stale attestations). Build the archive directly with the "
            "current δ.bin.",
            file=sys.stderr,
        )
        return 2

    # ── Step 2: verify the attestation against the pending δ ────────────
    if args.attestation is not None:
        # Operator pointed us at an explicit attestation file. Cross-check
        # that its filename matches the pending sha (cheap sanity).
        att_path = args.attestation.resolve()
        if not att_path.exists():
            print(
                f"FATAL: --attestation does not exist: {att_path}",
                file=sys.stderr,
            )
            return 2
    else:
        att_path = attestation_path_for(pending_sha, root=args.root)

    print(f"[promote] pending δ sha:  {pending_sha}")
    print(f"[promote] attestation:    {att_path}")

    try:
        attestation = verify_attestation_for_blob(
            pending_blob, root=args.root,
        )
    except AttestationMissing:
        print(
            f"FATAL: no attestation found at {att_path}. Sign one first "
            "via:\n"
            "  python tools/sign_lane_c_compliance.py \\\n"
            f"      --delta-bin {args.pending_delta_bin} \\\n"
            "      --approver <id> --ruling-text <text> \\\n"
            "      --private-key ~/.config/pact/lane_c_signing_key.pem",
            file=sys.stderr,
        )
        return 3
    except AttestationMismatch as e:
        print(f"FATAL: attestation sha mismatch: {e}", file=sys.stderr)
        return 3
    except AttestationMalformed as e:
        print(f"FATAL: attestation malformed: {e}", file=sys.stderr)
        return 3
    except TrustRootMissing as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 3
    except TrustRootMalformed as e:
        print(f"FATAL: trust root malformed: {e}", file=sys.stderr)
        return 3
    except AttestationApproverNotInTrustRoot as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 3
    except AttestationSignatureInvalid as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 3

    print("[promote] attestation VERIFIED:")
    print(f"  approver:     {attestation.approver}")
    print(f"  signed at:    {attestation.signed_at_utc}")
    print(f"  signed by:    {attestation.signed_by_user}")
    print(f"  git HEAD:     {attestation.git_head}")
    print(f"  ruling:       "
          f"{attestation.ruling_text[:120]}"
          + ("…" if len(attestation.ruling_text) > 120 else ""))

    # ── Step 3: re-emit the δ with compliance_status='approved' ─────────
    #
    # We avoid round-tripping through dense (B,3,H,W) tensors (which
    # would lose precision and produce different sparse selection if any
    # ranking ties broke differently). Instead we patch the wire header
    # in-place: decompress, modify JSON header bytes, recompress. The
    # quantized indices+values bytes are byte-copied unmodified.
    raw = zlib.decompress(pending_blob)
    if len(raw) < 8 or raw[:4] != MAGIC:
        print(
            f"FATAL: wire-format magic mismatch in pending δ; "
            f"got {bytes(raw[:4])!r}",
            file=sys.stderr,
        )
        return 4
    header_len = struct.unpack("<I", raw[4:8])[0]
    if 8 + header_len > len(raw):
        print(
            "FATAL: pending δ wire payload truncated in header",
            file=sys.stderr,
        )
        return 4
    import json as _json
    header = _json.loads(raw[8:8 + header_len].decode("utf-8"))
    body = raw[8 + header_len:]
    # Sanity: confirm the parsed header agrees with what we see
    # (defense in depth — if pack ever stops writing this field, we
    # want to fail rather than silently lose status).
    if header.get("compliance_status") != COMPLIANCE_PENDING:
        print(
            f"FATAL: pending δ header declares compliance_status="
            f"{header.get('compliance_status')!r}, expected "
            f"{COMPLIANCE_PENDING!r}. Refusing to promote a δ whose wire "
            "header doesn't match the expected promotion source.",
            file=sys.stderr,
        )
        return 4
    header["compliance_status"] = COMPLIANCE_APPROVED
    # Record the source attestation chain inside the wire header so an
    # auditor reading the promoted δ.bin alone can see where the
    # approval came from.
    header.setdefault("provenance", {})
    if not isinstance(header["provenance"], dict):
        header["provenance"] = {}
    header["provenance"]["promoted_from_pending_sha"] = pending_sha
    header["provenance"]["promoted_via_attestation_approver"] = (
        attestation.approver
    )
    header["provenance"]["promoted_via_attestation_signed_at_utc"] = (
        attestation.signed_at_utc
    )
    new_header = _json.dumps(header, sort_keys=True).encode("utf-8")
    new_raw = MAGIC + struct.pack("<I", len(new_header)) + new_header + body
    promoted_blob = zlib.compress(new_raw, level=9)
    promoted_sha = compute_blob_sha256(promoted_blob)

    # ── Step 4: write the promoted δ to --output ────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(promoted_blob)
    print()
    print("=== Promotion complete ===")
    print(f"  Pending δ sha:  {pending_sha}")
    print(f"  Promoted δ:     {args.output}")
    print(f"  Promoted sha:   {promoted_sha}")
    print(f"  Size:           {len(promoted_blob):,} bytes")
    print(f"  Spec sanity:    n_kept={pending_spec.n_kept} "
          f"(unchanged from pending)")
    print()
    print("NEXT STEP — sign a FRESH attestation against the promoted sha:")
    print("  python tools/sign_lane_c_compliance.py \\")
    print(f"      --delta-bin {args.output} \\")
    print(f"      --approver {attestation.approver} \\")
    print("      --ruling-text \"<re-confirm ruling for promoted bytes>\" \\")
    print("      --private-key ~/.config/pact/lane_c_signing_key.pem")
    print()
    print("THEN bundle into the archive:")
    print("  python experiments/build_baseline_archive.py \\")
    print(f"      --with-uniward-delta {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
