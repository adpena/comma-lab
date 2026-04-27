# Lane C δ Compliance Attestations

This directory holds external attestation records for Lane C δ.bin
artifacts that have been explicitly approved as contest-compliant.

**Trust model (Codex R5-2 #4 fix, 2026-04-27).** Lane C δ.bin is a
scorer-derived artifact; Yousfi PR #35 strict-scorer-rule may class it
as non-compliant. To prevent operator self-assertion of approval, the
gate works as follows:

1. `experiments/optimize_uniward_delta.py` can ONLY issue
   `pending_ruling` (default, safe) or `rejected` (terminal). It cannot
   write `compliance_status="approved"` into the δ.bin header.
2. To approve a specific δ.bin, run:

   ```
   python tools/sign_lane_c_compliance.py \
       --delta-bin path/to/delta.bin \
       --approver "yousfi" \
       --ruling-text "PR #35 ruling: ..."
   ```

   This writes `<sha256>.json` here, where `<sha256>` is the SHA256 of
   the actual delta.bin bytes.
3. `experiments/build_baseline_archive.py` reads the δ.bin's header. If
   `compliance_status="approved"`, it ALSO checks for the matching
   attestation file. Without a SHA-matched attestation, the build
   refuses (even with `--allow-pending-compliance`, which only handles
   the pending case).

**Files committed here are part of the audit trail.** Do not delete
them after eval; future auditors must be able to reconstruct who
approved which artifact and when.

**Format.** Each attestation is a JSON object with these fields:

```json
{
  "schema_version": 1,
  "delta_sha256": "<hex digest of delta.bin>",
  "delta_path_at_signing": "<path>",
  "delta_size_bytes": <int>,
  "approver": "<identity>",
  "ruling_text": "<free-form ruling>",
  "signed_at_utc": "<iso8601>",
  "signed_by_user": "<whoami>",
  "git_head": "<sha>"
}
```
