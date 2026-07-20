# R1b7 uint8-survival carrier — sealed n16 result

Date: 2026-07-20
Lane: `r1b7_uint8_survival_carrier`
Authority: `[macOS-CPU advisory]`
Pointer: `0.1910828242 [contest-CPU] UNMOVED`

## Verdict

`MEASURED_N16_FIXED_NONPOSITIVE_INTEGER_PREFIX_NO_NEW_CROSSING`

`verdict_scope`: exact sealed R1b4 n16 receiver; the 498 exact-feasible
Fisher-ordered Road-Lane sites for the fixed-magnitude arm; a post-run-
adjudicated bounded top-8 EV integer-lattice prefix with zero wrong-to-target
crossings; seed 1234, batch 16, hard CPU Torch. This is not an n600 result, not
contest-CPU/CUDA authority, not a marginal-prefix waterfill, and not a negative
on curvelet, shearlet, boundary, full-kernel, or any other carrier family.

The durable measurement receipt is
`.omx/research/r1b7_uint8_survival_carrier_20260720T224624Z.json`, SHA-256
`61f3d03930ac765b3ad5a287cbff29a3073c800eb5a5f2b98b8a701bc086d03c`.

## Premise result: fixed magnitude is not a new arm

The requested R2b fixed-magnitude constructor reproduces the sealed R1b6
payload byte for byte:

- requested sites: 512;
- exact-feasible sites: 498 (all Road-Lane);
- replay writes: 5,976;
- replay bytes: 60,008;
- reconstructed and sealed replay SHA-256:
  `063986d459618ebfac3cec08ea16fc51d2a809843d8e86fdd3c04569e457765c`.

Therefore the fixed arm is the sealed R1b6 arm, not a distinct positive-R2b
realization. Transfer of the old positive R2b magnitude without a fresh hard
receiver measurement is invalid.

## The 498-site stage autopsy

Every site has one first-death/survival bucket; the total is exactly 498.

| First death or survival stage | Count |
|---|---:|
| killed at emitted uint8 | 0 |
| killed by exact four-tap resize dilution | 0 |
| killed at SegNet stem | 0 |
| killed at head, same rival | 204 |
| killed at head, wrong rival | 0 |
| survived but caused local collateral | 5 |
| survived cleanly at the scheduled site | 289 |

The prime-suspect diagnosis is therefore falsified for this formulation:
integer camera writes survive decode, exact R, and the stem. The loss occurs at
the winner/rival decision and through collateral coupling, not at uint8.

Additional checks:

- scheduled-site survival: 294/498;
- new local collateral pixels around those sites: 7;
- fixed full-n16 flip count: 10,009 versus baseline 10,002;
- wrong-rival changes: 0;
- exact-R versus CPU Torch resize maximum delta error: 0.003426234, inside the
  measured 1/256 float32-kernel envelope;
- centered frozen head shape 5×144, measured rank exactly 4;
- reconstructed head-margin maximum absolute error: 0.00000441447.

The fixed arm recovers `-0.000660674492` combined nonrate score, costs 22,891
archive bytes over the sealed baseline, and measures 45.96586345 bytes/site.
Its conditional break-even budget under equation ID
`realization_breakeven_bytes_v1` is 0 bytes.

## Bounded integer-aware counterarm

The top-8 EV prefix generated 63 exact-R, directionally Fisher-aligned lattice
proposals. Fresh review found the original predicate only checked whether the
proposal margin was positive. Its four selected sites already had positive
baseline target/rival margins (`0.4798074`, `0.4778380`, `0.3192959`, and
`0.6401210`), so they were not hard crossings. Correct wrong-to-target
adjudication yields zero new crossings in the tested prefix. The implementation
now gates on both the baseline decision/margin and the proposal decision/margin,
with regression tests for the already-correct case.

The retained four-site diagnostic composition used 38 changed camera bytes,
624 replay bytes, and sealed to:

- archive:
  `/Volumes/VertigoDataTier/pact/evidence/r1b7_uint8_survival_carrier_20260720T224624Z/integer_aware_sealed.zip`;
- bytes: 97,835 (184 over the sealed baseline);
- SHA-256:
  `2f18fa52c1025283b9f0ed8df2d1e93a1fabee245a9e4b9d631a4fa84f9d9b3f`;
- double-decode raw SHA-256:
  `1fca2060b162d6fac1d8af78c6769a7e47abbc397c51b3f09b21bc08a0ef0d9c`.

The diagnostic composition changes Seg flips by exactly 0 and slightly worsens
Pose: combined recovery `-0.0000004290280771`. Its break-even budget is 0 bytes,
so the exact 184-byte delta does not pay. This does not constitute a successful
integer-aware counterarm: no newly crossed site was found. It is also not a
marginal waterfill. The measured gates reject the full fixed set and this
diagnostic composition only; individual prefixes remain unmeasured. There is no
integer-lattice infeasibility claim; only 8 sites and 8 multipliers were searched.

## Mechanism consequence

R1b7 changes the next mechanism from “make a rounding-bin write survive” to
“first require a genuine wrong-to-target receiver crossing, then admit only
receiver-composed sites whose head win exceeds all new collateral and pose
debt.” A future reopen needs a composed per-site secant or joint trust-region
admission rule plus measured marginal-prefix rate, and a byte delta below its
realized break-even. It must not re-dispatch this exact fixed or original
margin-only top-8 recipe. No n600 run is authorized by this result.

## Disk hygiene

The successful run atomically recorded hashes and rebuild commands for six raw
scratch files before deleting them. Retained sealed archives and decode receipts
remain on the SSD. The first fail-closed parity attempt produced no cleanup
receipt, so its four raw outputs were retained without deletion; custody and the
exact blocker are in
`.omx/research/r1b7_uint8_survival_carrier_attempt1_blocker_20260720T223801Z.json`.

## STORES CONSULTED

- `CLAUDE.md` and `AGENTS.md`;
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`;
- `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`;
- `reports/latest.md`;
- `.omx/state/lane_registry.json`;
- `.omx/state/subagent_progress.jsonl`;
- `.omx/state/master_gradient_anchors.jsonl`;
- `.omx/state/modal_call_id_ledger.jsonl`;
- `.omx/state/cost_band_posterior.jsonl` and
  `.omx/state/continual_learning_posterior.jsonl`;
- `.omx/state/probe_outcomes.jsonl`;
- latest Codex findings/session summary, council T3 memo, and design memo;
- R1b6 build spec, receipt, result memo, and sealed receiver artifacts;
- R2b source at commit `98515407bd` and its SSD receipt;
- Fisher ordering, target raw, frozen SegNet/PoseNet weights, base decoder, and
  base archive listed with full custody in the JSON receipt;
- task-specific and broadcast inboxes before checkpoints.

## Triality and pointer delta

- DSL: N/A; this is an encoder-side measurement tool and invents no trainer flag.
- DAG: `.omx/research/r1b7_uint8_survival_carrier_DAG_FEED_20260720T224624Z.md`.
- Equations: `realization_breakeven_bytes_v1` is consumed by ID and receives an
  append-only scoped domain refinement; its positive n600 numeric anchor is not
  overwritten.
- Pointer delta: none. `0.1910828242 [contest-CPU] UNMOVED`.

MAIN must review the branch diff, equation/probe ledger appends, retained failed
scratch blocker, and sealed archive custody before merge.
