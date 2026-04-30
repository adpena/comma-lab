# Lane 19 / Lane 20 Forensic Hold Repair Design - 2026-04-30

## Scope

Ownership is limited to Lane 19 logit-margin and Lane 20 Ballé forensic hold
repair design/preflight. This note does not promote either lane and does not
record a score claim.

## Lane 19 Clearance Gate

Lane 19 may not be relaunched or marked hold-cleared unless all of these are
present:

1. Deterministic archive build: fixed member order, fixed `ZipInfo` timestamp
   `(1980, 1, 1, 0, 0, 0)`, fixed permissions, `writestr`, and an adjacent
   `archive_manifest.json` with member sizes and SHA-256 hashes.
2. JSON adjudication: `scripts/adjudicate_contest_auth_eval.py` reads
   `eval_work/contest_auth_eval.json`; no human score-log parsing.
3. Current frontier gates: baseline score `1.043987524793892`, archive bytes
   `686635`, PoseNet `0.00346442`, SegNet `0.00400656`, CUDA device, 600
   samples, and relative component gates.
4. Corrected provenance/comments: Lane 19 uses `logit_margin_weight=10.0` and
   `kl_distill_weight=0.0`; older "KL kept" and `0.1` comments are stale.

## Lane 20 Clearance Gate

Lane 20 may not be relaunched or marked hold-cleared until there is a real
non-static codec and inflate path:

1. `non_static_byte_precheck.json` proves `BALLE_BEATS_STATIC` with
   `best_full_balle_bytes < static_baseline_bytes` on the real qint stream.
2. The archive builder writes an actual BHv1 archive member from
   `encode_qints_balle_auto` output, not a baseline/static archive copy.
3. `submissions/robust_current/inflate_renderer.py` decodes that BHv1 member
   during inflate.
4. Only after review should the script add a `BHV1_ARCHIVE_INTEGRATION_READY`
   marker and remove the fail-closed `FATAL_BHV1_ARCHIVE_INFLATE_INTEGRATION_MISSING`
   gate.

## Enforcement

`scripts/launch_lane_with_retry.py` treats Lane 19/20 hold clearance as
conditional. If a hold entry is deleted or marked `cleared: true` while the
lane-specific requirements are still absent, dispatch still fails before a Vast
instance is created.
