# OD1 recall evidence - 2026-08-05

Status: `RECALL_COMPLETE_FOR_SPEC / SCORER-FREE`.

Axis: `[macOS-CPU advisory / document-and-ledger recall]`.
`score_claim=false`, `promotion_eligible=false`, `n600_scorer_job=false`.

## Governing Reads

Required charter and contract:

| source | sha256 | use |
|---|---|---|
| `.omx/tmp/codex_runs/od1_prompt.md` | `acb258f0515d220fa5687b6074092d606377d6bc2b14da965a91e7b730a6cce9` | OD1 charter, phase ordering, deliverables |
| `.omx/tmp/codex_runs/_common_contract.md` | `eeae9e0035582e6bdd65fd837e4aa35a65e064fd09900b9c212d41ac02086771` | serializer, scorer slot, recall, protected-file, and denominator contract |
| `.omx/research/operator_directive_per_edge_optimality_criteria_20260805.md` | `76f83e8a7364267329bf84b0767858536d020d45cd32c60590cdcedbd1c55190` | Addendum 4 ordering law: seg first, joint pose recovery after |

Repo operating reads:

| source | sha256 | use |
|---|---|---|
| `CLAUDE.md` | `65da6dd8dcf6b11c0ecdd352938570fd5589c5e5e014d97acd63297f82a8c47c` | no-fake, serializer, exact-score discipline |
| `AGENTS.md` | `65da6dd8dcf6b11c0ecdd352938570fd5589c5e5e014d97acd63297f82a8c47c` | local agent contract |
| `PROGRAM.md` | `a6d5f79f3241ca1ae17b2587afd9940e1a4ea598804fd9efa152f2330e15db82` | protected mutation boundaries |
| `docs/operating_manual_craft_handoff.md` | `40d157a039d4dd242bfb189d53e6b82abcc5d037adceb0a52c9bb2956903f212` | answer-first, artifact-derived, skeptical handoff style |
| `.omx/state/main_hot_state.md` | `f77728ed1dbe05257dd7a7d555eecfc38865dfc639070fa1a91c5f0fec592db0` | current pointer, live fleet, pe2/OD1 coordination |
| `.omx/state/canonical_frontier_pointer.json` | `a9821ddbf60a52487fb40ab190e4d5b09943e51d5dbcb2f475d621ea5419cd49` | contest pointer calibration and official bar |
| `upstream/evaluate.py` | `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b` | score formula and archive byte denominator |
| `.omx/research/CANONICAL_RESEARCH_INDEX_20260629.md` | `b253a4f51879e2a1a63aba72a39464b10d213cec8e9647c6771627f22cf72c66` | historical calibrated recall surface |

## Sources Searched

Queries used:

- `rg --files .omx/research | rg 'ddm_(sq2|tj1|q31|bo1|js1|et1|cw1|lg1|gc15|gc12|fh1|vh1|pe1|pe2|pe3|bf1|kt1|ws|j2|b4s|tp1|sh1|fd1)'`
- `rg -n '#383|#888|#920|#366|pose-finish|hinge|sched1|event-continuation' .omx/state .omx/research`
- `.venv/bin/python tools/list_canonical_equations.py --json`
- `rg -n 'pe2|OD1|scorer slot|READY_TO_FIRE|runtime consumption' .omx/state/main_hot_state.md .omx/state/active_lane_dispatch_claims.md .omx/research/ddm_pe*`
- Memory quick pass over `/Users/adpena/.codex/memories/MEMORY.md` for Pact #899/#904/current-hot-state rules, to avoid repeating stale apparatus mistakes.

Search correction: the charter shorthand `ws2` and `ws3` resolved to the durable repo paths `ddm_ws2_warm_start_custody_producer_receipt_20260724.json`, `ddm_ws2_warm_start_slope_arbitration_receipt_20260724.json`, `ddm_ws3_window_completion_and_arbitration_DAG_FEED_20260724.md`, and `ddm_ws3_warm_start_slope_arbitration_receipt_20260724.json`.

## Seed Evidence Recalled

| item | source | denominator | recalled fact | OD1 effect |
|---|---|---:|---|---|
| SQ2 | `.omx/research/ddm_sq2_20260804/SQ2_RUN_MEMO.md` sha `4a0bf7f7e7a104068f29790be840318a1b9412335ca85e51cd093f9b46f1da6c` | n32 | eta `0.7895095948827292` at 25, `0.8620042643923241` at 50, `0.9112579957356077` at 100; 0/32 converged; raw solved paint pose term `0.8813937215422536` | seg base is live, but raw direct ship is blocked by pose erosion and cap-bound stopping |
| TJ1 | `.omx/research/ddm_tj1_20260805/tj1_summary.md` sha `5c41bed2b8b9305cde023571993939f4904fbcab23fc9d8bb2e6a7bca155eefe` | n32 replay law | step25 predicted step50 eta interval `[0.796753479934, 0.864347607988]`, measured step50 inside | OD1 launch tickets use event/trajectory stopping, not fixed cap-as-convergence |
| CW1 | `.omx/research/ddm_cw1_cap_artifact_uncap_and_guard_20260804.md` sha `60960a16e62fc041f3ba5dabaf1bd091c0bd082a6bb041f46f105db07ac4ce0b` | n32 + 21,674 Python files scanned | #935 cap artifact real; 32/32 improved at 50; 0/32 converged | any seg production ticket must expose cap and terminal status |
| ET1 | `.omx/research/ddm_et1_eta_on_the_priced_band_20260803.md` sha `141e21797d27ec0dbe60dc863b80e5c83dfe4386750d7702df9c949290791a90` | n8 | block16 reach `41.84%`, bytes `46,247`, break-even eta `0.1707`, measured eta `0.5267`; pose coupled upward | phase field is first useful regional seg actuator, but n8 banks nothing |
| Q31 | `.omx/research/ddm_q31_20260804/Q31_Q3_CONSTRAINED_SOLVE_RECEIPT_20260804.md` sha `6f410c9a6ab5b445e1ab4df5ae0ab10235602c277499d5740757f51bac3df6f7` | n32 | Q3-first survival `0.2303538325`, only 33.1% of ED1 bar; d_pose ratio `1.042`; 32/32 cap-best | Q3-first is not the campaign base; Q3 remains a recovery/corrector subspace |
| JS1 | `.omx/research/ddm_js1_staging_discriminator_20260804.md` sha `e71476863f4f7259194b2cdc10094f12e6d8045d3e6f55f3872dfb130de8f5b9` | n4 pooled / block16 | frame_0 C-PRIME preserved seg exactly on sampled rows and reduced pose; k=4 DCT carriage at 96 B/pair made one subset net positive | OD1 pose recovery starts from staging/frame_0 carriage as the first proof, not from post-hoc frame_1 DC paint |
| PE1 | `.omx/research/ddm_pe1_20260805/PE1_RECEIPT_20260805.md` sha `6980eee9cda136989b00e969b1573d68f1d4bac4d2535e842960490b68796e93` | n600 payload records, no scorer | candidates `478,612` B and `425,627` B were byte-closed but runtime-survival-unmeasured at receipt time | use only after PE2 receiver consumption and final survival measurement |
| PE2 | `.omx/research/ddm_pe2_20260805/PE2_RECEIPT_20260805.md` sha `06cbf9dc8492210a4683bf81d8872afcddfaa492ceadf186edca59c08df4fef7` | qo1 full raster `3,662,409,600` bytes, no scorer | absent identity proved; PE1 full, PE1 surgical, and BF1 receiver-consumed; scorer batch staged but PE2 did not run scorer | carrier/rate rung is receiver-closed, survival still belongs to MAIN's queued batch |
| PE3 | `.omx/research/ddm_pe3_20260805/PE3_RECEIPT_20260805.md` sha `73b9c66ad2332605fe425dcac0aabad445475ef7e4745849239bbc159da47a3a` | n600 payload records, no scorer | hybrid 75KB archive `432,428` B, flip recall `0.831522`; survival unmeasured | PE3 can be folded as a candidate after PE2 consumption or PE4 control, not as a scored row |
| BF1 | `.omx/research/ddm_bf1_20260805/BF1_RECEIPT_20260805.md` sha `bd9234e201b916e314fb9822a536f539e1ff2839d20aa994cabbf54023421c64` | band pixels, no scorer | lane-crop candidate `563,256` B, recall `1.0`, pair0 runtime smoke changed 12,408 camera px | useful full-band reference, but rate-heavy and survival-unmeasured until pe2/main batch |
| KT1 | `.omx/research/ddm_kt1_20260805/KT1_TRANSFER_MATRIX.md` sha `0578168b266f737d2969f02b731261e9f357d302315901adf133838f3ed05496` | 17 laws x 8 domains = 136 cells | top transfers include event-driven exits/waterfill, fleet scheduler by marginal S, positive-control registry, terminal resumability | OD1 tickets include event exits, positive controls, and per-stage checkpoint/resume artifacts |

## Found Beyond Charter Seeds

1. PE2 had landed after the charter snapshot. That changes PE1/PE3/BF1 from "receiver consumption pending" to "receiver consumption proved, scorer survival still pending." Plan change: OD1 no longer blocks on PE2 implementation; it blocks on MAIN's queued survival rows and any OD1-specific composition row.

2. Canonical equation recall found `pose_null_subspace_is_ac_only_v1`: frame_1 pose-null paint must be AC-only; constants live in PoseNet rowspace. Plan change: do not route OD1 through frame_1 DC/Q3 constant paint as a pose repair; prefer frame_0 carriage or AC-only inside-cell corrections.

3. Canonical equation recall found `trajectory_derived_stopping_law_v1` and `ddm_os1_termination_census_from_cost_proxy_v1`. Plan change: every seg-base ticket must record terminal census and a continuation certificate; cap-best is not convergence.

4. BO1 recalled that the correction-solver hinge at `0.05` was too quiet relative to CE boundary pressure. Plan change: any DirectDescription correction-solver use in OD1 carries a #888 hinge-weight A/B or explicit adoption of a known-good hinge before final n600.

5. The P0 ledger contains an older #366 interpretation that terminal pose should be "never trained." Plan change: this is marked superseded-in-this-scope by the 2026-08-05 Addendum 4 law and JS1/SQ2 evidence; OD1 must not mix the old mechanism label with the new ordering law.

6. Sched1 refused a fixed-stage schedule without reseal. Plan change: tickets are event-driven and fail closed if the DSL/compiler cannot express the continuation law.

7. The canonical research index is stale on several numeric frontiers but still useful for historical calibration and exact-score discipline. Plan change: all live numbers in OD1 come from 2026-08-03 through 2026-08-05 receipts or current `main_hot_state.md`, not from older index rows unless explicitly marked historical.

## Scoped Negatives

- Did not find, in the searched OD1/PE/SQ/TJ/JS receipts, a final n600 scored row for a seg-first plus pose-recovered composition. Absence scope: listed receipts and current hot state only.
- Did not find, in PE2 receipt/queue note, completed PE scorer outputs. PE2 staged the batch and proved receiver consumption; score survival remains MAIN-owned.
- Did not find, in SQ2/TJ1/CW1, convergence of the solved-field seg base. The measured n32 trajectory is cap-bound and positive, not terminal.

Own-vehicle frontier line: `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`; contest pointer remains borrowed/unmoved.
