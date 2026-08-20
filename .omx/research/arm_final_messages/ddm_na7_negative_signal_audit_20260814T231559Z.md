Completed and committed the full NA7 audit.

- Memo: [ddm_na7_negative_signal_audit_20260814.md](/Users/adpena/Projects/pact/.omx/research/ddm_na7_negative_signal_audit_20260814.md)
- Commit: `d4daa828ffc5c87b1d33d58f8fa7725364be34b4`
- Typed consumer correction: [NA7_JS8_GATE_AWARE_CONSUMER_ADJUDICATION.json](/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/NA7_JS8_GATE_AWARE_CONSUMER_ADJUDICATION.json)
- Clean-round result: zero new eligible rows after resolving 32 seed dispositions and 15 additional dry-loop findings.
- No scorer, training, paid dispatch, or payload-producing run was launched.

Conclusion: no unconsumed same-object candidate composes to `|ΔS| ≥ 1e-5` on MC36. Apparent candidates either double-count MC36 ancestry, mix instruments, or remain unmaterialized.

The audit also corrected a real routing error: current RX2 is a fixed-token rate learner and does not test JS8’s gate-aware distortion-training hypothesis. That hypothesis now routes to a separately registered JS1/#982 joint treatment. MT1 already encoded the multi-token hypothesis correctly.

Exact frontier remains MC36 Variant C: `S=0.1619344578804448 @ 186,269 B [contest-CUDA T4, n600]`. Own-vehicle frontier remains LC2: `S=0.16959899569230852 @ 187,226 B`.

## NEXT_IF_RESUMED

- **Disposition: FIRED_EXISTING_OWNER_NO_DUPLICATE; owner: MAIN F26R CPU closer; consumer store: `/Volumes/VertigoDataTier/pact/ddm_f26r_hpac_final_rung_20260814/`; fire trigger: the existing CPU call becomes terminal with retained hashes and wall clock.** Harvest and close it idempotently.
- **Disposition: FIRED_EXISTING_OWNER_NO_DUPLICATE; owner: RX2 harvester; consumer store: `/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac`; fire trigger: terminal EMA/stage receipt appears.** Run fixed-token identity, deterministic repeat, and whole-container RC64 pricing.
- **Disposition: QUEUED_WITH_A_FIRE_ORDER; owner: JS1/#982 joint receiver owner; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge`; fire trigger: RX2 is harvested and a separate resumable treatment proves the Road-hub gate is consumed during joint training.** Run scorer-free identity, price, resume, and retention preflight first.
- **Disposition: QUEUED_WITH_A_FIRE_ORDER; owner: MAIN #978 router; consumer store: MT1 `t4_sign_gate_r1`; fire trigger: higher-priority exact rows are terminal and sealed hashes match.** Run only the T4 sign gate; train only if `positive_t4_sign=true`.
- **Disposition: OPTIMIZE; owner: dispatcher/AC1 maintainer; consumer store: dispatcher tests and call/claim ledgers; fire trigger: before the next remote paired fire.** Fix the spawn-metadata signature and prove one launch produces one registered call and idempotent closure.
- **Disposition: HELD; owner: current-terminal receiver owner; consumer store: `.../free_receiver_treatments/l28/`; fire trigger: repeat-identical MC36/L28 children prove unchanged bytes and actual receiver consumption.** Then queue one governed A/B.
- **Disposition: HELD; owner: current-vehicle rate-in-loss owner; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/rate_in_loss/`; fire trigger: a matched resumable MC36 checkpoint pair exists.** Retain and recount both complete archives.
- **Disposition: HELD; owner: PZ4-QAT successor; consumer store: #984 rate branch; fire trigger: a parser-equal map-free object saves at least `2,000 B`.** Only then consider rate-aware QAT.

## LIVE-HYPOTHESES

- Joint gate-aware training may work because current JS8 applied its gate only after uniform training.
- RX2 may produce a material rate win if its epoch-one surrogate survives complete serialization and identity closure.
- MT1 may change sign on T4 because its local random `n32` screen is not population authority.
- Isolated L28 may provide a zero-counted-byte terminal correction because the old negative used another receiver and co-applied transform.
- Current-object rate-in-loss may outperform the historical row by jointly changing learned model and probability state.
- Map-free grouped QAT may survive PZ4A’s metadata failure if existing structure supplies precision classes without an allocation stream.

## DEAD-ENDS

- Re-adding QS2, RE1, HP4, or MICRO35 to MC36 is closed because MC36 already contains that ancestry.
- QS5 cannot rescue current JS8: perfect zero-byte pose repair still leaves Seg plus rate about `+0.00115865 S` worse.
- QS3’s `57.1%` beneficial rate cannot transfer from its 189-pixel instrument to JS8.
- The audited linear overlays, singleton event proposals, post-hoc gates, direct C1 representative, dense transport, per-cell precision map, and exact failed sparse compositions remain closed at their stated scopes.
- Same-state coder hunting remains closed; only RX2’s genuinely new trained probability state can reopen rate.
- Treating F26P’s runtime miss as a model failure is closed by the F26Q/F26R decode-engineering cure.
- Treating MT1 queue starvation as scientific evidence is closed: it measured no score and invalidated no seal.
- Claiming RX2 tests JS8’s distortion hypothesis is closed by the typed consumer adjudication.
- Claiming NA7 as goal progress is closed: neither exact frontier moved.