Verdict: `GENERATOR-READY` (`FORMULATION` scope).

SFP1 produced:

- Three fresh dense-field proposals with 1,084, 2,831, and 9,723 changed sites.
- A persisted null control byte-identical to the 117,964,800-byte base field.
- A control-only BHW2/JF2 row with a known 108,108-byte RC64 receipt.
- Typed schema, source custody, fold table, verification, resumable checkpoints, and ordered gate-2 handoff.

Key artifacts:

- [Research memo](/Users/adpena/Projects/pact/.omx/research/ddm_sfp1_scmdl_field_proposal_prep_20260901.md)
- [Generator](/Users/adpena/Projects/pact/experiments/ddm_sfp1_scmdl_field_proposal_prep.py)
- [Tests](/Users/adpena/Projects/pact/tests/test_ddm_sfp1_scmdl_field_proposal_prep.py)
- [HANDOFF.json](/Volumes/APDataStore/pact/ddm_jc1/scmdl_projection/HANDOFF.json)
- [CANDIDATE_SET.json](/Volumes/APDataStore/pact/ddm_jc1/scmdl_projection/CANDIDATE_SET.json)

Validation completed:

- 7 behavioral tests passed.
- Payload-retention audit: 0 findings.
- Materialized verification passed, including p01⊂p02 assignment identity.
- Both generator review passes recorded; the tracker excludes `tests/`, so tests received two manual/behavioral passes.
- Exact Ruff F821 check passed using a lossless executable copy because `.venv/bin/ruff` is incorrectly mode `0644`.
- Repository dev preflight’s only failure was seven unrelated pre-existing lane-registration references in `ddm_lc3`/`ddm_ltg1`.

No scorer, RC64 proposal encode/refit, Modal launch, archive build, or exact evaluation ran. AFR1 remains `S=0.14797617125559104 @ 180,002 B`, `[contest-CUDA T4 n600]`; the pointer did not move.

Git object writes were denied by the managed sandbox. The mandated serializer returned `rc=17`; live HEAD remains `f309d68df9a6ef3129d49679cd4e94970b1a0865`, and the three files remain uncommitted. The verified fallback commit is `ad1b94b410213a820fb2f7e7c8051bea918315de`:

- [intended-commit.bundle](/Volumes/VertigoDataTier/pact/ddm_sfp1_scmdl_field_proposal_prep/commit_fallback/20260901T062545.673220Z-94119/intended-commit.bundle), SHA-256 `2f7f3b8f58594c1a58ea007e4cfe00e96857834f6875fa2bd4deb4552bc3641f`
- [intended-commit.format-patch](/Volumes/VertigoDataTier/pact/ddm_sfp1_scmdl_field_proposal_prep/commit_fallback/20260901T062545.673220Z-94119/intended-commit.format-patch), SHA-256 `5b95881b2dda1cde913443c388bfdd4784f52275c357dcefac1ede2f9db34ab6`

## NEXT_IF_RESUMED

- **Disposition: LAND. Owner: MAIN. Consumer store:** the verified bundle above. **Fire trigger:** a Git-write-capable repository context. Land fallback commit `ad1b94b410213a820fb2f7e7c8051bea918315de`; do not rerun SFP1.
- **Disposition: QUEUE. Owner:** MAIN-selected gate-2 scorer arm. **Consumer store:** `/Volumes/APDataStore/pact/ddm_jc1/scmdl_projection/HANDOFF.json`. **Fire trigger:** RXC1 publishes `GATE-1-PASSED`. Reproduce the controls, then refit and byte-close p01 before any scorer measurement.

## LIVE-HYPOTHESES

- p01 is the strongest first physical row because it concentrates G3 top-24 scorer debt into only 1,084 boundary edits, plausibly limiting distortion spill and refit disruption.
- Cross-group causal refitting may realize the RR9 cell that fixed within-group permutation could not test, because changing `X` changes group membership and therefore causal probabilities.
- p03 may compose MI1’s missing position context with realized scorer disagreements, although its 9,723 edits make it the highest-spill candidate.

## DEAD-ENDS

- Reusing retained JF or FCD fields is closed because it reproduces prior candidate cells rather than original SCMDL proposals.
- WJ1 target masks and BHW2 B/H/W masks are closed as proposal sources because their membership consumes GT; BHW2 remains control-only.
- WWC1 token-GT assignments are closed by both the charter and its realized broken-cone evidence.
- Fixed within-group RR9 permutation is closed as a byte lever on the measured object because it was exactly byte-neutral.