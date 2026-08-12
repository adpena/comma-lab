# ddm_js6 event-proposal realized acceptance receipt — 2026-08-12

## VERDICT

`FIRST_USEFUL_NONZERO_BARE_ADMISSION_AT_2_OF_200; F1_NOT_ELIGIBLE; BARE_EVENT_POINT_BEATS_1.28_B_PER_PROJECTED_ROBUST_FLIP`

Axis: `[macOS-CPU advisory, instrument floor 0.0131 S]` on the sealed seeded-stratified n32 relative gauge. The projected n600 flip count is a weighted projection from n32, not an n600 scorer run. No archive was built and no exact contest score was measured. The exact pointer and own-vehicle frontier did not move.

The mandatory first-admission stop fired on proposal `ec1_0001_d89c5435d9a4`, the second measured proposal. It is a one-site `boundary_offset` event on pair 7. Its realized stratified pose delta is `+1.6918629075844697e-7`, which passes the strict `<2e-6` JS5 gate. It produces projected robust Δflips `−18` and ordinary projected Δflips `−18`, with one robust beneficial flip and zero harmful flips on the sampled pair. Its retained bare Brotli-q11 event payload is 21 B, or `1.1666666666666667 B/projected robust flip`, below both the charter's rounded 1.28 bar and TF1's exact CP135 marginal bar `1.2731082153`.

This is a useful bare proposal-level admission, not a complete archive claim. Complete container growth, composition interaction, full-n600 realized components, and exact evaluator authority remain unmeasured.

## ACCEPTANCE TABLE

| Ordinal | Proposal | Family | Pair | Sites | Realized pose Δ | Pose pass | Projected robust Δflips | Bare Brotli q11 B | B/robust flip | Bare admission |
|---:|---|---|---:|---:|---:|:---:|---:|---:|---:|:---:|
| 0 | `ec1_0000_756fafa85dee` | boundary | 7 | 1 | `+2.763859626711128e-5` | no | −18 | 21 | 1.1666666667 | no |
| 1 | `ec1_0001_d89c5435d9a4` | boundary | 7 | 1 | `+1.6918629075844697e-7` | yes | −18 | 21 | 1.1666666667 | yes |

Measured acceptance is 1/2 = 50% pose-pass and 1/2 = 50% useful bare yield before the mandatory stop.

Per-family measured yield:

- Boundary: 2 measured / 151 available; 1 pose-pass; 1 useful bare admission; 50% yield in the measured prefix.
- Lane: 0 measured / 48 available because the earlier stop fired; yield is unmeasured, not zero.
- Island: 0 measured / 1 available because the earlier stop fired; yield is unmeasured, not zero.

The full retained table is `/Volumes/APDataStore/pact/ddm_js6_20260812/ACCEPTANCE_TABLE.json`, 3,146 B, SHA-256 `6e2fa0fa5e72e05c9e4ca45cfdc11bdab38e5c196ca4a4a19b79570ea362cfa1`.

## F1 AND ECONOMICS ADJUDICATION

- **F1 not eligible and not fired.** F1 requires all 200 proposals, no useful bare admission, and pose acceptance below 5%. The useful admission at ordinal 1 triggered the mandatory early stop. No family closure is authorized.
- **F2 not eligible and not fired.** The stop rule yields only one pose-passing finite B/robust-flip point, so an across-proposal trend cannot honestly be decided. The single admitted point is individually below 1.28.
- The first proposal also had robust movement and the same proposal-level economics, but failed pose by 13.8× (`2.7638596e-5 / 2e-6`) and was rejected.

## INPUT AND OUTPUT CUSTODY

The consume preflight reverified instead of trusting EC1's receipt:

- 200/200 proposal index rows, 200 unique proposal IDs, and 200 unique event-payload SHA-256 values.
- 1,200 proposal-owned payload files totaling 885,752,490 B.
- Family population: 151 boundary / 48 lane / 1 island.
- Proposal index: 798,532 B, SHA-256 `599a3ac0a9c7d7e62c162fcee595194d6d3cd79685d0ceabab92e0231bd9d47e`.
- Source archive: 186,252 B, SHA-256 `6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6`.
- JS5 measured acceptance source: SHA-256 `981240bef78e195595978241b383ea4b5ad4ac23ab321ffd4609f6c645dc5d80`.
- Measured JS6 adapter source custody: 34,004 B, SHA-256 `d578569f8bb80c2337de0be1fc99507c6e36164c48e27bfffc79a9dd7d19db18`.
- Final result: `/Volumes/APDataStore/pact/ddm_js6_20260812/FINAL_RESULT.json`, 8,398 B, SHA-256 `44ee603c93564ed29c3d71238693f6ff9228d1b9dcd925f3d1ea3aed7d7039c7`.

No EC1 producer entry point ran. The adapter consumed the retained event, camera uint8, and scorer-input payloads. It re-derived the scorer lattice from the retained uint8 camera and required exact float16 equality before scoring. Every newly materialized Seg logits, argmax, PoseNet output6, and pose error payload was retained with SHA-256 and bytes under `/Volumes/APDataStore/pact/ddm_js6_20260812/proposals/`.

## DETERMINISM REPEAT

The independent retained repeat at `/Volumes/APDataStore/pact/ddm_js6_20260812/determinism_repeat/` stopped on the same proposal with byte-identical acceptance-table SHA-256 `6e2fa0fa5e72e05c9e4ca45cfdc11bdab38e5c196ca4a4a19b79570ea362cfa1`.

For both measured proposals, all decision metrics were exactly equal and all four retained data-fork payloads were byte-identical across repeats:

- Proposal 0: logits `762cbe0a…`, argmax `ff1e55d1…`, pose output `6a8c5d26…`, pose error `3fc0e1c3…`.
- Proposal 1: logits `f2a9a2e5…`, argmax `a8480f19…`, pose output `5316f98a…`, pose error `b60af0e8…`.

## RECALL EVIDENCE

Sources searched before building and adjudicating:

- Full `.omx/research/` corpus with `realized_acceptance_200|ddm_js5|event-coordinate proposals|pose-gated robust|B/robust-flip|acceptance < 5%`.
- Canonical equations via `.venv/bin/python tools/list_canonical_equations.py --json`, filtered for event, boundary, partition, pose-null, robust, transport, and lane.
- `.omx/research/CANONICAL_RESEARCH_INDEX*`, `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`, and `.omx/state/main_hot_state.md`.
- JS2B/JS3/JS5 source call sites for the sealed n32 sample, δ-robust metric, custody-plane transport, strict pose gate, and exact argparse surface.

Beyond the charter seeds, recall found:

- `.omx/research/ddm_tf1_theoretical_floor_and_beyond_20260812.md`: the exact CP135 marginal bar is `1.2731082153 B/robust flip`, and the surviving path is representation-level event conditioning rather than full-raster XOR. This changed the table from a rounded-only comparison to both rounded and exact bars, with the complete-container caveat kept attached.
- `.omx/research/ddm_js1_reseal_skeleton_20260811.md`: JS4 had already proven robust Seg movement survives projection while nonlinear pose leakage and decode-time projector bytes remain the walls. This prevented receiver effectiveness or robust movement alone from being called admission.
- The canonical partition-transport and correspondence-first lane laws preserve local event coordinates as a plausible representation while forbidding a transfer from the failed global raster grammar. This kept the verdict scoped to the measured EC1 endpoint.
- The live board supersedes the common contract's stale frontier line: effective CP135 remains `0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`; own-vehicle LC2 remains `0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]`.

## IMPLEMENTATION AND VERIFICATION

- Adapter: `experiments/ddm_js6_event_proposal_acceptance.py`.
- Tests: `experiments/tests/test_ddm_js6_event_proposal_acceptance.py`.
- Focused tests: 7 passed.
- `py_compile`, Ruff, `git diff --check`, and two `review_tracker.py mark-file --status reviewed` passes per Python file passed before landing.
- The run was CPU-only. MPS and Metal were not touched.
- The prior ps135b CPU claim was checked before fire: PID 26406 was absent, its governed log ended at timeout, and its retained terminal-policy receipt records `solve_self_exited_before_boundary=true`.

## BOUNDARIES

- This is a seeded stratified n32 local-CPU relative gauge. The `−18` value is a projected n600 robust-flip estimate, not an n600 scorer measurement.
- The 21 B denominator is the standalone proposal's retained Brotli-q11 event stream. It is not complete archive growth.
- No complete HY1/CP135 composition, carrier interaction, archive build, independent archive decode, `upstream/evaluate.py`, contest-CPU, or contest-CUDA row ran.
- No lane or island proposal was measured because the mandatory earlier stop fired. Their yield remains unknown.
- The exact frontier and own-vehicle frontier did not move.

## QUEUED-WITH-A-FIRE-ORDER

- **Action:** compose the admitted EC1 event into the HY1 whole-container campaign. **Disposition:** QUEUED-WITH-A-FIRE-ORDER. **Owner:** HY1/js1 whole-container builder. **Consumer store:** `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/hy1_solved_carriage/`. **Fire trigger:** the JS6 admitted proposal receipt is consumed on the exact CP135 source archive; count complete container growth, prove independent decode, and retain the composed archive before any n600 scorer request.
- **Action:** seed the JS5 projector-distilled MAIN burn from a representation-level admission. **Disposition:** QUEUED-WITH-A-FIRE-ORDER. **Owner:** MAIN training-leg router. **Consumer store:** `/Volumes/VertigoDataTier/pact/ddm_js5_20260812/authoritative_seeded/main_burn`. **Fire trigger:** a receiver-known adapter converts the admitted EC1 event into a nonzero bare JS5 checkpoint and proves byte-identical event replay; MAIN then owns the training leg and the sole n600 scorer slot.

## LIVE-HYPOTHESES

1. The admitted one-site boundary event will remain net-negative after complete-container composition because its proposal-level 1.1667 B/projected robust flip is below the exact 1.273108 CP135 marginal bar and its pose delta has about 11.8× gate headroom. Plausibility is the retained receiver/scorer proof; container overhead and composition interaction are still untested.
2. Representation-level event coordinates can seed a nonzero JS5 bare checkpoint without the 453 MB decode-time projector. Plausibility is the real pose-passing robust event; the receiver-known adapter and module distillation are not built or measured.
3. Additional boundary, lane, or island events may amortize framing overhead and improve the complete-package exchange rate. Plausibility is the 200-event retained alphabet and two robust-improving first rows; yield beyond ordinal 1 is intentionally unmeasured.

## DEAD-ENDS

1. Treating receiver-effective proposals as accepted: closed by the first row, which changed the receiver and robust flips but failed the pose gate by 13.8×.
2. Continuing through all 200 after a useful bare admission: forbidden by the preregistered stop rule; lane/island absence is unmeasured, not negative evidence.
3. Claiming a score or archive win from `1.1667 B/flip`: forbidden because the denominator is a projected n32 flip count and the numerator omits complete-container growth.
4. Regenerating EC1 payloads: unnecessary and forbidden; all 200 proposal-owned payload sets reverified and the acceptance run consumed them directly.
