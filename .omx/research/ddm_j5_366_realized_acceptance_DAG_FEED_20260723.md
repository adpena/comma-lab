# DDM J5 #366 realized-acceptance warm start — DSL / DAG / equations / FEED

Date: 2026-07-23  
Lane: `ddm_j5_366_realized_acceptance_warmstart`  
Evidence: `[macOS-CPU frozen-scorer advisory]`; `score_claim=false`; pointer
`0.1910828242 [contest-CPU]` unmoved.

## Outcome first

The corrected bounded one-step n600 opening returned
`READY_TO_FIRE_UNDER_STANDING_GO`.

- exact candidate: `d4eb1450f461437e714d08a9349cc735fe79b53a1739a2de92ef4850287dfd0d`,
  133,936 bytes;
- 18 lifecycle-feasible G1 `center_x` coordinates realized at +1;
- `delta d_seg=-2.8203328450521203e-05`;
- `delta d_pose=-1.6296301816964842e-04`;
- `delta bytes=-5`;
- exact joint `delta S=-0.002843840398518996`;
- residual-trunk errors `-2,013`; role-correction errors `-1,314`;
- complete checkpoint parse-back and 116 GiB memory admission passed.

This branch launched no campaign. The token becomes launch authority only after
independent MAIN landing review.

## DSL

The append-only executable ticket is
`.omx/research/configs/ddm_j5_366_realized_acceptance_warmstart_20260723.json`.

- semantic SHA:
  `13e194a8a354d53489f0ff68a5042237e69b4b6841a6b7959a15873fffa7b6e8`;
- typed-config hash:
  `d43608af799b2f2d04e248413ceb944c093701441eafb222f2b3cdf3d32b8d80`;
- ticket file SHA:
  `2ae7da9058f8c5a421ffc494ab0407947391ea28ede1b1384248f298906fdf43`.

Historical J3/J4 semantic and typed hashes remain unchanged. The new fields are
identity-bearing only for `pure_priced_exact_n600`.

## Executable DAG

```text
sealed V15 receiver + n600 cache
             |
             v
storage waterfall + 116 GiB governor
             |
             v
restore full Adam/EMA/cursor or initialize fresh moments
             |
             v
J5 opening policy
  - preserve beta2=0.999 / T_rewarm=2000
  - hold Lane program
  - expose worldsheet + shared-template proposal sources
  - bank theta on camera-Q8
             |
             v
typed candidate source
  - exact v19 eight-pair active set
  - whole-lifecycle geometry feasibility
  - coherent x/y or local gradient
             |
             v
compile exact archive -> parse-back -> paint -> uint8 -> R
             |
             v
chunked n600 frozen SegNet/PoseNet + exact archive bytes
             |
       +-----+--------------------+
       | delta S < 0              | delta S >= 0
       v                          v
accept receiver state       shrink proposal
persist checkpoint          exhaust ladder -> exact rollback/block
       |
       v
fire-readiness gate
  d_seg <= 0, d_pose <= 0, one strict component descent,
  and C1 residual-trunk errors descend
       |
       v
READY_TO_FIRE_UNDER_STANDING_GO
       |
       v
MAIN adversarial landing review -> only MAIN may launch campaign
```

The first preliminary attempt correctly refused when an over-broad all-track
proposal escaped scorer geometry. Source re-derivation showed v19 selected only
tracks touching its preregistered eight-pair screen and feasible for the whole
lifecycle. The regression now requires the corrected proposal to compile
byte-identically to v19 archive `d4eb1450...`.

## Equation edge

The canonical note is
`.omx/research/ddm_j5_366_realized_acceptance_canonical_equations_20260723.md`.
J5 reuses:

- `adam_v_variance_warmup_length_v1`;
- `tac.optimization.pure_priced_realized_objective.pure_priced_realized_delta`;
- the contest joint action exactly as evaluated on realized bytes.

No duplicate canonical law was registered.

## Unified solver wire-in

1. **Sensitivity map:** proposal sources use the measured v19 active-set
   screen; exact per-class deltas expose Road/Lane/Undrivable/Movable/MyCar
   effects after every realized move.
2. **Pareto constraint:** strict joint-score admission prices Seg, Pose, and
   bytes; fire readiness additionally requires component non-regression and
   residual-trunk descent.
3. **Bit allocator:** every candidate records exact archive byte delta and rate
   term. The accepted move saved five bytes.
4. **Cathedral/autopilot:** the durable token routes to MAIN review; this branch
   has `execution_allowed_by_this_receipt=false`.
5. **Continual learning:** preliminary mask failure, exact correction, per-class
   deltas, and the ready verdict are preserved in typed receipts and Codex
   findings.
6. **Probe disambiguator:** coherent x/y and local-gradient candidates plus the
   shrink ladder are callable alternatives; exact n600 realization arbitrates.

## FEED

MAIN should review, in order:

1. lifecycle-feasible active-track mask and exact v19 archive-identity test;
2. strict pure-price accept/shrink/rollback control flow;
3. C1 residual/role bucket definition and legacy-resume fallback;
4. Q8 staging and stage checkpoint parse-back;
5. ticket reseal and source hashes;
6. the exact smoke and governed-preflight receipts.

Only after that review may MAIN use standing authority to start the sealed
13.3–13.8 hour resumable, per-stage-checkpointed campaign. This branch did not
start it.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`;
- `docs/operating_manual_craft_handoff.md`,
  `docs/vehicle_operating_system.md`;
- v7.5 and v8 canonical SPECs;
- J3/J4 tickets, diagnosis, smoke, equations, and DAG receipts;
- v19 tool, stage receipts, exact grammar archive, and pure-price helper;
- C1 composed-candidate spec, ledger, DAG, and equations;
- lane, dispatch, subagent, task, and inbox state;
- exact SSD smoke and preflight artifacts named in the committed receipts.

## Authority boundary

This is advisory local evidence. It is not a contest score, promotion, or
campaign execution receipt. MAIN review and merge are required.
