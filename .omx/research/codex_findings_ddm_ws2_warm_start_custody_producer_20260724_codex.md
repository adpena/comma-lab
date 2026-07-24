# Codex findings — DDM WS2 warm-start custody producer (2026-07-24)

## Disposition

**NOT READY TO FIRE.** The durable verdict is
`REFUSE_INCOMPLETE_FOUR_STEP_WINDOW`; it is scoped to this bounded W_seg versus
W_joint warm-start arbitration instance. It is not a formulation, family,
paradigm, promotion, or score verdict.

- Evidence axis: `[macOS-CPU frozen-scorer advisory]`.
- `score_claim=false`; `promotion_eligible=false`.
- Pointer: `0.1910828242 [contest-CPU]`, unchanged.
- Lane: `lane_ddm_ws2_warm_start_custody_producer_20260724`.
- Delegation checkpoint:
  `codex_delegate:ddm_ws2_warm_start_custody_producer:20260724T053455Z`.
- `main_review_required=true`. Campaign firing remains exclusively with MAIN.

## Receiver-closed producer custody

Both WS1 endpoint rows are now real, immutable J5-consumable archives rather
than metric-only rows:

| start | exact bytes | archive SHA-256 | fresh batch32 n600 d_seg | fresh batch32 n600 d_pose |
|---|---:|---|---:|---:|
| W_seg | 138,031 | `264a09abb8f614eca104eb4ab1d0a12005ba65ec6a4fbc6620ff92f1c73281a9` | 0.024124510023328993 | 146.36493245487773 |
| W_joint | 138,801 | `5aa45850ab05d47f411583fd7582e27644c5bf289cd6d5bc32c05a52706c433e` | 0.07051923116048177 | 36.618184751411334 |

For both starts:

- receiver parse/re-emit is byte-identical;
- J5 stage-00 lift/recompile is byte-identical with 368 receiver-effective
  parameters;
- scorer replay is deterministic on the first batch;
- ground-truth argmax and scorer weights are absent from the archive;
- fresh batch32 d_seg equals the sealed batch16 endpoint exactly;
- Pose deltas are only deterministic reduction regrouping:
  W_seg `-4.101775630260818e-8`, W_joint
  `-2.6646063133739517e-8`, both within the preregistered `1e-7` tolerance.

Producer authority:
`.omx/research/ddm_ws2_warm_start_custody_producer_receipt_20260724.json`
(SHA-256
`05581b02cc6ce789b6219302ebd888f1665ab4c3882038ce29e9be18f6174ea1`).
Bulk checkpoints remain on `/Volumes/VertigoDataTier/pact`; no uncertified
scratch was deleted or moved.

## J7 reseal and governed preflight

The landed resealer generated both candidate tickets through its typed path,
with current source custody:

- W_seg typed hash
  `d6b8aecc791f97598427508bbcc953d94e3725382967c07971613dc125500804`;
- W_joint typed hash
  `346975b25fce972766ef89ebce437ba87cc2e11e37f709683682246116dbcf93`;
- launcher SHA-256
  `1239bd359f472605f0ebe1a0d1969d61ff984f086b97b2d3951803521195892a`.

Both dry-runs were governor-admitted. Fresh worst-geometry memory receipts were
source-bound after the launcher repair:

- W_seg: measured peak 16.95045 GiB, projection 21.34055 GiB, receipt SHA
  `cdae0ce43f7962934f44e8fbf458f00da10d847ff2df9f1292309f10c35eb14b`;
- W_joint: measured peak 15.98949 GiB, projection 20.18738 GiB, receipt SHA
  `67c40a04257d7f2ac2f8ac70ce0c19a426877d9057860348087c2688414fedfd`.

The governor temporarily refused during unrelated sister memory pressure. No
bypass or process termination was used; both candidates were admitted only
after the canonical memory gate became green.

## Live bounded windows

### W_seg

Receipt:
`/Volumes/VertigoDataTier/pact/experiments/results/ddm_ws2_w_seg_j7_four_step_v2_20260724T085500Z/full_run_receipt.json`
(SHA-256
`0a7cfbc940c1abb08da8aaba84b5c0eafef3fedb0eaef069b75bc3fa45bacacb`).

The first live proposal improved priced joint action by
`-0.0004297730820253919` but regressed d_seg:

- baseline: d_seg `0.024124510023328993`, d_pose `146.36493245487776`;
- step 1: d_seg `0.02414915296766493`, d_pose `146.3428045197368`;
- global error delta: `+2,907`;
- verdict: `BLOCKED_REALIZED_DSEG_REGRESSION`.

`--stage-exit-on-stop` correctly terminated before steps 2–4. This is an
INSTANCE blocker on that opening proposal, not a dead warm-start family.

### W_joint

Receipt:
`/Volumes/VertigoDataTier/pact/experiments/results/ddm_ws2_w_joint_j7_four_step_v2_20260724T092100Z/full_run_receipt.json`
(SHA-256
`02ffc33ab07ae0d77ce120d3ad0ecd7ba37612736463fd5b1f74adcd57ca3c94`).

The first two exact proposals were rejected. The third (`y_-1`) was
component-safe and admitted with joint delta `-0.05689051019463004`,
residual-trunk delta `-51,429` errors, and a green cumulative component/residual
gate. The four-step endpoint was:

- baseline: d_seg `0.07051923116048177`, d_pose `36.618184751411334`;
- step 4 live receiver: d_seg `0.0702156745062934`, d_pose
  `36.37587755493872`;
- archive: 138,804 bytes, SHA-256
  `9601e777010b1dc45ed0841e118fcf34c58452324f8730fe9958a3440502e3a4`;
- latest stage decision: `REALIZED_STAGE_DESCENT_CONTINUE`;
- terminal verdict:
  `BLOCKED_POSE_FINISH_CONDITIONING_HISTORY_INSUFFICIENT`.

The pose detector has exact points only at steps `[0,1,4]`; it therefore cannot
claim the required conditioning/plateau history.

## Preregistered arbitration

The registered callable re-derived
`R*=4.1215446777965665`. No measured slope ratio or winning warm start is
reported: W_seg has only a one-step terminal receipt, while the preregistration
requires two exact four-step windows.

The durable refusal is
`.omx/research/ddm_ws2_warm_start_slope_arbitration_receipt_20260724.json`.
Its SHA-256 is
`63d00f473c140ae456abfdfb3233c20bd04936c387d2b875f78f8a9d39693ecc`;
it binds the producer and both full-run receipt hashes and states
`REFUSE_INCOMPLETE_FOUR_STEP_WINDOW`.

Consequently, the contract's final arbitrated-start reseal and bounded re-smoke
were not run. Selecting W_joint merely because it completed would violate the
preregistered comparison; selecting W_seg would ignore its component stop.

## Round-1 adversarial findings

1. **Fixed — archive-bound baseline custody.** The launcher hard-coded the
   inherited V15 d_seg baseline, so the first W_seg attempt refused before step
   one. The launcher now selects the exact baseline by source archive SHA and
   fails closed on unknown custody. A regression test protects both WS1 starts.
2. **Preserved — incomplete arbitration must be durable.** The initial
   arbitrator printed a transient error when a run stopped early. It now writes
   an immutable non-promoting refusal receipt with input hashes, R*, scope, and
   MAIN-review requirement.
3. **Disclosed — `--max-steps` bounds admitted optimizer steps, not exact
   proposal attempts.** Rejected shrink/direction proposals do not advance
   `global_step`; the current ladder is nevertheless finite (four sources by
   eight multipliers) and each output is immutable. MAIN should decide whether
   a future launcher contract also needs an explicit exact-proposal-attempt cap.
4. **Blocking — E4 lineage is not generalized to WS1.** The materialized
   archives use the actual J5 receiver/export grammar, but they did not traverse
   `ddm_runtime_exporter`'s E4 Brotli-Q11/ImportError-only-LZMA1 packet path.
   The current E4 typed config and compiler are literal-bound to the sealed V15
   source/state byte identities. Treating the J5 archive as E4-equivalent would
   be fake custody. A typed E4 adapter/generalization for receiver-closed WS1
   states, followed by parse-back and exact remeasurement, remains owed.

## Exact blocker list

1. `W_SEG_FOUR_STEP_WINDOW_INCOMPLETE_AFTER_COMPONENT_DSEG_STOP`;
2. `PREREGISTERED_RSTAR_ARBITRATION_UNIDENTIFIABLE`;
3. `W_JOINT_POSE_FINISH_CONDITIONING_HISTORY_INSUFFICIENT`;
4. `E4_EXPORTER_WS1_SOURCE_STATE_CONTRACT_UNIMPLEMENTED`;
5. therefore `NO_ARBITRATED_START_TO_RESEAL_OR_RESMOKE`.

## Triality and directive consumption

- **DSL/control:** producer config
  `.omx/research/configs/ddm_ws2_warm_start_custody_producer_20260724.json`
  plus the two resealer-generated J7 tickets.
- **DAG/trajectory:** `FEED-603-ws2-custody` records materialization, live
  window outcomes, and the exact next edges.
- **Equation/law:** existing registered
  `ddm_ws1_warm_start_slope_falsifier_v1`; callable
  `tac.optimization.ddm_warm_start_slope_falsifier:critical_pose_to_seg_slope_ratio`.

| Directive | Consumption |
|---|---|
| authority prompt + lane/checkpoint | read and SHA-verified before action; exact checkpoint key used |
| actual receiver + exact batch32 n600 | both starts materialized, parsed, replayed, and measured |
| J5 lift roundtrip | 368-parameter lift/recompile byte identity proved for both |
| preregistered R* four-step windows | executed fail-closed; W_seg stopped at step 1, W_joint completed step 4 |
| reseal via landed tool | both candidate tickets resealed; final arbitrated reseal withheld because arbitration refused |
| governed memory + bounded runs | both source-bound memory receipts green; no campaign launch |
| E4 Brotli/LZMA path | not satisfied; exact typed-contract blocker recorded |
| MENU1 awareness | unreviewed third candidate was not consumed as authority and did not alter the preregistered two-start comparison |

## Verification seal

Round-1 review produced the four findings above. Three post-fix clean passes
were then run over lint and the targeted WS1 custody, J5 consumer, resealer,
slope-law, and arbitration suites:

1. ruff clean; pytest `45 passed in 270.75s`;
2. ruff clean; pytest `45 passed in 270.62s`;
3. ruff clean; pytest `45 passed in 269.92s`.

This memo and its JSON receipts are advisory research artifacts. MAIN must
review the entire branch diff and the blocker scoping before landing anything.
