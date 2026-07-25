---
schema: codex_findings_v1
lane_id: lane_ddm_j10_366_ema_verdict_shadow_cure_20260725
authority_sha256: dc0d7bf469056f66249abe58579c4cd09219051d91096a1384d2062adc4bc9a1
research_only: true
score_claim: false
pointer_moved: false
main_review_required: true
verdict: BLOCKED_REALIZED_NO_PURE_PRICED_DESCENT_AFTER_SHRINK_LADDER
verdict_scope: INSTANCE_MATERIALIZED_STEP50_SOURCE_X_SEALED_OPENING_PROPOSAL_SET
---

# DDM J10 #366 EMA-verdict-shadow cure — findings

## Disposition

The attempt-5 cross-shadow death mechanism is cured in code, but the required attempt-6
bounded gate is **not green**. The canonical one-step smoke stopped at global step 0 because
none of the four receiver-distinct opening proposals realized exact joint
`delta_S < 0`. Therefore:

- **NO `READY_TO_FIRE_UNDER_STANDING_GO` claim**
- **NO attempt-6 campaign launch**
- pointer `0.1910828242 [contest-CPU]` **UNMOVED**
- `score_claim=false`

`verdict_scope: INSTANCE` — the materialized attempt-5 step-50 live source under the sealed J10
opening proposal set and exact n600 acceptance rule. This is **not** a formulation, family,
scorer, receiver, or EMA-policy negative. The proposal-quality family remains open.

## Verified attempt-5 mechanism and landed cures

1. **C1 — derived EMA:** the typed J10 ticket resolves
   `ema_decay_run_geometry_v1` through `EmaDecayCalibrated` with no fallback:
   `d = 46/49 = 0.9387755102040817`, `3/(1-d)=49<50`, and the first-verdict
   blend exceeds 0.5. The 400 remaining updates are sealed as `100 + 150 + 150`.
2. **C2 — same-shadow decisions:** scheduled stage control is live-vs-live; EMA is emitted
   separately for export/promotion only. Cross-shadow comparisons refuse.
3. **C3 — informativeness:** identical receiver bytes or a realized-count delta below one
   classifies `VERDICT_NOT_YET_INFORMATIVE`; only a second consecutive **scheduled** degenerate
   verdict escalates. A bounded-smoke row cannot consume that grace.
4. **Custody hardening:** the consumer re-hashes the materialization receipt and measured
   baseline, then cross-checks archive path/SHA/bytes, receiver parse-back, fresh-zero re-emit,
   live realized count, source checkpoint, n600/batch32, and exact d_seg/d_pose.
5. **Reseal idempotence:** J10 telemetry labels are canonicalized, not appended repeatedly.
   Two consecutive final reseals produced byte-identical ticket SHA-256
   `19d130e2d767383876ef2483faa1e51fb5d1153ba61ab45068e8b3c5638bc40e`.

Regression evidence:

- `105 passed` across the focused consumer, launcher, resealer, EMA-LawRef suites.
- After the idempotence repair, `56 passed` across the affected focused suites.
- Ruff clean; two fresh `review_tracker` passes on every changed Python surface.
- The attempt-5 phantom comparison reproduces
  `BLOCKED_REALIZED_DSEG_REGRESSION`; J10 refuses it as
  `REFUSE_REALIZED_STAGE_VERDICT_SHADOW_MISMATCH`.

## Preserved step-50 live state

Attempt-5 accepted state remains preserved; old optimizer/EMA bytes were not deleted or
rewritten.

| Artifact | Measurement / custody |
|---|---|
| Original accepted checkpoint | SHA-256 `043c2a8b3c89688510cc0ff002f37a375a974205a5f8760d93133c47b7cec7c1`, global step 50 |
| Materialized live archive | 138,813 bytes, SHA-256 `2a2c0367150f8c8c0953dfb5c1485e238bbc9995c37385e149e52ae22f506241` |
| Materialization receipt | SHA-256 `774eb2aa49924fe27935bcf71b6f7921b71917b2549e87ffa717ae950fce3c86`; receiver parse-back and fresh-zero re-emit both exact |
| Fresh unchanged n600 baseline | d_seg `0.06974277072482639`, d_pose `35.49982080959101`, SHA-256 `206ad67430b013784e7cd1853a2eceb336a0990a5625d838c0958a5895a12c71` |

The materialized live d_seg is below both the attempt-5 EMA verdict
`0.07051923116048177` and its live step-1 reference `0.07030889723036024`. The old step-50
regression was therefore a shadow-inconsistent apparatus failure, not a truthful live-state
regression.

## Final seal, memory, and restore proof

- Final ticket:
  `.omx/research/configs/ddm_j10_366_ema_verdict_shadow_cure_20260725.json`
- semantic SHA-256:
  `7faa57594ef63b6e5b61f522b1bc50249f953b87eecfed254059ff1234e2ec18`
- typed config SHA-256:
  `478aaf0db82463104e0e848f95f69361a72950c70dd76cc2d2574ec8e3267a64`
- ticket SHA-256:
  `19d130e2d767383876ef2483faa1e51fb5d1153ba61ab45068e8b3c5638bc40e`
- fresh canonical-config worst-geometry receipt:
  `/Volumes/VertigoDataTier/pact/experiments/results/ddm_j10_366_fresh_memory_bootstrap_20260725T1237Z/worst_geometry_memory_preflight.json`,
  SHA-256 `50f3f541c6d0c30bc7168064a21f526d357b2004a8d0cb1166c004c49677a312`.
- memory result: **MEASURED** peak `16.939682006835938 GiB`; **DERIVED** projected peak
  `21.327618408203126 GiB`; SAFE below `116 GiB`; custom grouped backward active; fused-R
  forward and gradient bit-identical to NumPy-fp32; `campaign_launched=false`.
- fresh-process checkpoint restore:
  `/Volumes/VertigoDataTier/pact/experiments/results/ddm_j10_366_resume_proof_20260725T1352Z/process_boundary_resume_proof.json`,
  SHA-256 `2538fa90ec670121be8dec610b0fd07a59c8b46fabedcb1fcc9adf7cf17c0a91`;
  `FRESH_PROCESS_RESUME_PROOF_GREEN`, optimizer arrays loadable, checkpoint SHA
  `44312661d3b11f7389add8f90bdbb86b26b3dfb78189d51eb11ba1398280e36d`.

## Exact bounded-smoke blocker

Receipt:
`/Volumes/VertigoDataTier/pact/experiments/results/ddm_j10_366_ema_shadow_resmoke_20260725T1308Z/full_run_receipt.json`,
SHA-256 `6ad4d1653eee81cd56d8fb3a2e4f7155d35f648ff4b39d988ee861387cdabd9d`.

Baseline exact advisory action was `25.908103080501977`. Every receiver-distinct opening move
was rejected by the unchanged pure-priced exact n600 rule:

| Proposal | d_seg | d_pose | joint delta_S | Disposition |
|---|---:|---:|---:|---|
| `worldsheet_joint_active_x_+1` | 0.06969220479329427 | 35.55635026903992 | **+0.00994017010407013** | closest; d_seg descends but pose price dominates |
| `worldsheet_joint_active_y_-1` | 0.06986434936523438 | 35.61925856781073 | +0.04382082650052955 | reject |
| `local_exact_gradient` | 0.07025318569607204 | 35.55364773997821 | +0.06531830924357995 | reject |
| `worldsheet_joint_active_x_-1` | 0.07033764309353299 | 36.000787594894206 | +0.1919619536286387 | reject |

For the closest candidate, the exact decomposition is:

`seg_term=-0.005056593153211775`,
`pose_term=+0.014995431539375659`,
`rate_term=+0.0000013317179062443428`.

The receipt therefore reports
`BLOCKED_REALIZED_NO_PURE_PRICED_DESCENT_AFTER_SHRINK_LADDER`,
`global_step=0`, `telemetry_rows=0`, `ema_export_verdict_count=0`, and
`campaign_launched=false`. Because no step was admitted, the requested scheduled-style live/EMA
dual verdict could not lawfully run. The task falsifier
`BLOCKED_REALIZED_DSEG_REGRESSION_CONFIRMED_LIVE` did **not** fire: the closest live candidate
does descend d_seg; it fails the unchanged joint objective on pose.

## MAIN landing review and reopener

MAIN must review branch range `8c197dc072..HEAD`, especially:

1. same-shadow decision authority and EMA export separation;
2. scheduled-only degeneracy grace and hard second-scheduled-verdict escalation;
3. materialization/baseline fail-closed joins;
4. LawRef geometry and no-fallback provenance;
5. reseal idempotence;
6. the scoped step-0 proposal blocker.

MAIN has advanced independently beyond this branch. After merge, MAIN must reseal against the
merged-main consumer/launcher SHAs and regenerate a matching worst-geometry receipt before any
future FIRE. The reopener is a typed opening-proposal-quality cure that realizes exact
`delta_S<0` from this materialized source without weakening or changing the pure-priced
acceptance rule, followed by a new bounded live/EMA dual-verdict smoke. Until then, FIRE remains
blocked.

## Triality and stores consulted

- **DSL:** final typed J10 ticket and `VerdictShadowPolicyV1`.
- **DAG:** `FEED-603-j10` same-landing row in the canonical DDM DAG.
- **Equations:** `ema_decay_run_geometry_v1` LawRef, derived through
  `EmaDecayCalibrated`.
- **Stores consulted:** authority prompt; CLAUDE.md; AGENTS.md; craft handoff manual; attempt-5
  full receipt, baseline, step-50 verdict, telemetry, and keep-all checkpoint; J9 memo/ticket;
  canonical DAG; lane registry; subagent progress; canonical equation registry; final J10
  materialization, memory, bounded-smoke, and resume receipts.
- **Quarantine waiver consumed:** `HARVEST-SIGNAL-ONLY`. Banked R1 `0.127` was carried only as
  a comparator/fallback harvest signal. No R1 bytes or weights were loaded; it is not a binding
  target or promotion claim.
- **Inbox/broadcast directives:** no task-window directive rows were present at checkpoints or
  finalization.

