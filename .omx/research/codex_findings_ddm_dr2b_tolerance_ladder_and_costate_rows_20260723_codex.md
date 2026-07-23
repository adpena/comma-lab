---
schema: codex_findings.v1
created_at_utc: 2026-07-23T19:27:00Z
topic: ddm_dr2b_tolerance_ladder_and_costate_rows
research_only: true
execution_allowed: false
score_claim: false
pointer_moved: false
main_landing_review_required: true
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
---

# DDM DR2b tolerance ladder and costate rows

## Verdict

`MEASURED_E2_SPARSE_COSTATE_ROWS; BLOCKED_SDWL1_LADDER_MODE_RERACE_DR1_RECORD_PRUNING_AND_XI_DIRECTION`

The active evidence is the SHA-bound
`.omx/research/ddm_dr2b_tolerance_ladder_and_costate_rows_n600_20260723_v4/receipt.json`
(53,101 bytes; SHA-256
`9dedcc03d38942c4291a51505b754890a79ab48bfa53c1ac8daef61fe5a8c8c8`).
All measurements are `[macOS-CPU frozen-scorer advisory]`; none is a contest
score, candidate, promotion, or pointer move.

## Measured receiver-closed rows

Each edit changes exactly one E2 counted coordinate, serializes and parses back
the real member, changes exactly pair 447 in canonical scorer batch 432:448,
and rebases that batch delta exactly onto the settled n600 baseline.

| Costate rank | FIRST-RUNG | E2 coordinate | Δbytes | Δd_seg n600 | Δd_pose n600 | ΔS joint | Disposition |
|---:|:---:|---|---:|---:|---:|---:|---|
| 1 | yes | semantic `(260,511)`, role 4→2 | -36 | -9.324815538e-8 | +2.770796452e-8 | -3.329230196e-5 | locally admissible |
| 2 | yes | semantic `(189,364)`, role 0→2 | -19 | 0 | -9.576011886e-8 | -1.266319474e-5 | locally admissible |
| 3 | yes | chart anchor, `20→19` | +8 | 0 | -8.803922146e-6 | +4.235151382e-6 | stop |
| 4 | yes | chart residual, `45→46` | +9 | 0 | -7.922709581e-7 | +5.894485930e-6 | stop |
| 5 | yes | chart gradient, `-5→-1` | +19 | 0 | +3.748948922e-5 | +1.730016079e-5 | stop |

The semantic flip-distance annotations are respectively
`0.4477157433` and `0.5192216620` in the frozen SegNet head metric. These
are sparse instance rows selected to exercise the API, not a complete
tolerance field or a prospective candidate ranking.

The wider anchor rung `20→16` is a measured
`BLOCKED_RECEIVER_INVALID` instance: `ReceiverError: chart reconstruction
escaped uint8`. Smaller steps and other coordinates remain open.

## Authority deliverables

1. **U1 lossy ladder — partially measured, SDWL1 transfer blocked.** The exact
   settled row is 68,464 bytes and the pose-held cap is 154,524 bytes, leaving
   86,060 description-only bytes. Five E2 coordinate rungs and one
   receiver-invalid boundary were measured. No SHA-bound invertible mapping
   connects SDWL1 fact coordinates to E2 runtime coordinates, so E2
   tolerances cannot be applied to SDWL1 and no honest first fitting rung can
   be named. Scope: **FORMULATION bridge**; lossy description families remain
   open.
2. **Mode-at-tolerance re-race — blocked at the same bridge.** The settled exact
   challenger remains 70,700 bytes, +2,236 bytes versus exact SDWL1. There is
   no common lossy coordinate system on which to rerace modes. Scope:
   **FORMULATION-at-lossy-layer**; the mode family remains open.
3. **DR1 margin pruning — ordered redundancy blocker closed; record pruning
   still blocked.** The post-int8 positive input remains FIRST-RUNG
   `ΔS=-0.1208315524`, `Δbytes=+235`. The decoded Brotli-Q11 ordered
   diagnostic measured 127,662 bytes for the base/track/template-bank prefix,
   then 148 bytes for placements conditioned on it, then 185 bytes for sparse
   compensations conditioned on both. Pairwise conditioning saves 39 bytes
   for sparse given placements and 37 bytes for placements given sparse, but
   increases other orderings. These diagnostics are not production byte
   credits. The emitted 48 placement and 23 sparse records lack individual
   realized n600 Seg/Pose marginals, so non-additive record pruning remains
   blocked for this **FORMULATION instance**.
4. **g2 costates — measured.** Five receiver-valid rows are ranked by realized
   n600 reduced cost and stop at the exact rate dual `25/37545489`. The two
   locally favorable semantic rows retain FIRST-RUNG labels.
5. **ξ advection — pose visibility found, ξ direction not identified.** All
   three receiver-valid chart edits moved PoseNet, proving the chart stream is
   pose-visible. E2 exports neither a counted ξ member nor a ξ→chart Jacobian,
   so generic chart motion cannot be relabeled ξ-advection. Scope:
   **E2 FORMULATION**; explicit ξ inverse families remain open.

## MAIN provisional 4.6× claim: falsified premise

E2 `semantic/composed.dds` carries a `(600,384,512)` uint8 role plane:
117,964,800 per-cell categorical assignments, including exact interiors,
boundaries, and paint-role selection. Its member is 315,102 bytes and its ZIP
home is 315,153 bytes.

SDWL1's 68,464-byte description carries 6,600 aggregate records / 45,600
declared scalar facts in a `(600,11,8)` int64 tensor: moments, boxes,
separatrix summaries, topology deltas, and pair screw. It does not carry the
per-cell role plane. Therefore the 4.603× byte ratio compares different
content and yields no free-rate claim.

At identical realized E2 argmax, the extant measured semantic cost remains
315,153 ZIP-home bytes unless an invertible replacement is built. As a coder
diagnostic, Brotli-Q11 requires 315,639 conditional bytes for the raw role
plane after the decoded SDWL1 aggregate tensor, 606 bytes more than the
315,033-byte standalone raw-plane encoding; SDWL1 plus that completion totals
384,103 bytes. This is not an information-theoretic irreducibility proof, so
`replacement_bytes` remains `null`.

## Prosody and frequency-band extensions

The redundancy schema reserves continuous `prosody_amplitudes` coordinates
for per-stratum amplitude, per-boundary contrast, and global per-channel
statistics. They are attributes of grammar tokens, not new symbols, and are
priced separately from geometry.

It also reserves `prosody_spectrum_by_stratum`. Exact-R-null bands are admitted
only by the executable `frequency_band_admission` guard and must emit zero
description bytes. A scorer-visible band must instead carry its own
realized-through-R flip-distance, Pose delta, and marginal-rate test. No E2 or
DR1 bandwise R-transfer certificate exists, so no band is currently called
free.

## Triality, resumption, and stores

- DSL/config leg:
  `.omx/research/configs/ddm_dr2b_tolerance_ladder_and_costate_rows_20260723.json`
- Equations/guard leg:
  `src/tac/optimization/ddm_dr2b_tolerance_costate.py`
- DAG/feed leg:
  `.omx/research/ddm_dr2b_tolerance_ladder_and_costate_rows_DAG_FEED_20260723.md`
- Resumption: six immutable per-probe stage checkpoints, each bound to the
  typed config hash.
- STORES CONSULTED: the SHA-bound E2 archive and E2/DR1/DR2/DV2/v19b receipts;
  the DV2 fact inventory and selected payload; the frozen n600 target cache;
  the pinned SegNet/PoseNet weights; the per-arm and fleet inboxes.

## MAIN landing review required

Before merge, MAIN must review:

1. the exact one-pair locality premise used by `exact_n600_rebase`;
2. that no E2 tolerance or byte delta is transferred into SDWL1/DR1;
3. the E2-versus-SDWL1 content comparison and its non-irreducibility wording;
4. the FIRST-RUNG and verdict-scope labels on every positive/negative row;
5. the exact-R-null frequency guard and reserved prosody stream ownership;
6. serializer content hashes, the focused test receipt, and the unchanged
   frontier pointer.
