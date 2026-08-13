# DDM RE1 — realization-engineered candidate (2026-08-13)

Status: **PARTIAL — two real byte-closed proposals, zero accepted optimization rounds, public
receiver and complete-S absent.** The common contract does not grant this arm the scorer slot.
No Modal call was made.

## Outcome

Axis for the measurements below:
`[macOS-CPU scorer-free direct-token HP3/RC64 n600 reclose]`.

| checkpoint | exact semantic changes | archive | measured rate vs CP135 | deterministic entropy closure | alternate-surface triage only | complete-S acceptance |
|---|---:|---:|---:|---|---:|---|
| round 1, `ec1_0164_3a4e239de5b9` | 1 cell, pair 96 | 186,252 B, `7be3eb94…7dfa` | 0 B | pass; repeat identical; all 117,964,800 tokens decoded | `-1.227961429e-6 S` | **NO** |
| round 2, add `ec1_0004_3bc2b69c706c` and `ec1_0120_463b0cb756b2` | 3 cells, pairs 7/73/96 | 186,253 B, `1288cd11…3d4` | +1 B | pass; repeat identical; all 117,964,800 tokens decoded | `-1.016215831e-6 S` after rate | **NO** |

Measured here: exact archive bytes and hashes, independent repeat identity, exact requested-cell
diffs and nowhere else, candidate-conditioned HP3 export, fresh full-n600 RC64 streams, 25 retained
24-frame checkpoints per encode, physical ZIP parseback, and shipped RC64 entropy decode.

Not measured here: `inflate.sh`, decoded `0.raw`, the real public R/rendered frames, SegNet fields,
PoseNet vectors, a local complete-S row, contest-CPU, or contest-CUDA. Therefore
`entropy_receiver_closed=true`, while `receiver_closed=false` and
`full_public_runtime_receiver_closed=false`. Neither proposal is the charter's receiver-closed
deliverable. The measured share of the `0.01195513827824176` gap to 0.15 is unavailable because no
complete-S row exists.

## RECALL EVIDENCE

The arm searched beyond the charter seeds before selecting a mechanism:

- `.venv/bin/python tools/corpus_query.py --top 30 --json` with
  `realization engineered representative preimage probability object HP3 complete score`,
  `instrument floor quantum floor realized acceptance anti additive whole candidate`,
  `receiver uint8 preimage semantic support pose geometry survival cure`, and
  `2026-08-13 PZ4R PO1 HV1 JO1 CP5V SA1 exact receipt`;
- content search over `.omx/research` for realization engineering, representative/preimage, HP3,
  instrument/quantum floors, anti-additivity, complete S, PO1, JS5, JS7, and PZ4R;
- `.venv/bin/python tools/list_canonical_equations.py --json` plus the canonical research index,
  `sub015_DAG_*` FEED blocks, `.omx/research/harness_tasklist_bridge_20260803.jsonl`, live lane
  registry/dispatch state, and `.omx/state/main_hot_state.md`;
- the retained JO1, CP5V, VD1, PO1, HC1, T1R1, HR2, and CP135 receipts and payload stores.

Material changes from recall:

- PZ4R superseded the old HV1 queue and showed that parseback/rate success does not preserve Pose;
  retained real Pose vectors are mandatory before any promotion claim.
- PO1 and JS5 closed continuous sub-LSB shrinking. RE1 uses categorical one-cell lattice moves and
  never admits a model-predicted step as a realized score win.
- JO1 supplied the smallest real candidate-conditioned HP3/RC64 closure engine. T1R1 supplied the
  complete retained container/entropy-receiver pattern.
- HC1 closed direct C1 substitution: carriage was cheap, but its contest-CUDA point score was about
  0.4045 with severe Pose damage.
- CP5V and the precise PO1/VD1 component surface exposed a precision/axis mismatch. Their numbers
  can triage proposals but cannot be combined into a pointer-comparable score.
- HR1/HR2 remained prestage-only: all four typed programs are non-executable, with no argv or
  consumer. Renderer-weight optimization therefore stayed fail-closed.

### Consumed-corpus pins and dispositions

Every charter-seeded finding was reopened at its current bytes. A pin below means the finding
informed the design; it does not promote that finding's authority or claim that every prescribed cure
was executable in this scorer-free arm.

| finding | exact content pin(s) | RE1 use or disposition |
|---|---|---|
| RVS1 survival playbook | `.omx/research/ddm_rvs1_realization_survival_harvest_20260811.md` — `6282f773bc3e1acb543f4cc7de8ffb9376521650d779dae5adccfdbea613b111` | Kept categorical-lattice survival and fail-closed realized admission; renderer/scorer-dependent cures were reviewed but not mechanized. |
| RVS2 geometry class | `.omx/research/ddm_rvs2_geometry_survival_crosswalk_20260811.md` — `d6b1ecf65a7e806ee510a9efd6936fbd78d3b80cdeb71e3c1ec4260a00ce7e39` | Used one-best-event-per-pair separation; Bregman/Fisher and ξ acceptance were not claimed without whole-candidate Pose vectors. |
| HR1 design and HR2 prestage | `.omx/research/ddm_hr1_realization_engineering_20260811.md` — `d15c1e3cad98a27ea1fe4919db0e6cd942d6c82c2ff51d00324f1978f292db76`; `src/tac/witness_dsl/hr1_prestage.py` — `781cc98292f978b88716c1e1c3f8f747cd751a49e6cd9bdb2e496e975d86dc86`; `/Volumes/VertigoDataTier/pact/ddm_hr2_prestage_build_20260811/retained_v2/40_TYPED_PROGRAMS.json` — `509484ba4cf35d261eed7f32b12c4317c5c64ace790c14b7e5c518b033540822` | Audited the current configurations; all remain non-executable with empty argv/no consumer, so renderer training stayed blocked rather than proxied. |
| RHO1 survival prior | `.omx/research/ddm_rho1_survival_prior_20260811.md` — `ce87832f60869f936bdf5fdb66cd9f12e8690600124d4148e614c67653b52378` | Informed sparse, separated composition; no quantitative survival fraction became an acceptance threshold because the candidate axis is mismatched and unscored. |
| #897 solver-axis cure | `.omx/research/ddm_cv1_seven_surface_convocation_20260802.md` — `41a6787bf29a50da683d34e145bc531263c4bd3a5080c73a0e31c69652147203`; `src/tac/optimization/ddm_tr1_runtime.py` — `99e2a6408826f4f2d2125520904deefc9e8a506728078ad81a7866e74806a05c`; `.omx/research/harness_tasklist_bridge_20260803.jsonl` — `a44b15a7eb062d9bdc7cffe0fb1bea54d75a047b375b59d540fd7fb3139cf9ed` | Preserved the law that realization gaps require the real operators. The 88→3 / 96.6% result belongs to LL1/TR1 task #897, not DK1, and was not transferred numerically. |
| DK1 lattice realizer | `.omx/research/ddm_dk1_20260806/RECEIPT.md` — `8473b0d0ad960e97633263374cd638ef6d169fe69476cb6c2118fabdda97e48e`; `.omx/research/ddm_dk1_20260806/lattice_realizer_measurement.json` — `3b3ee88e3f7aee99b30b1da133f215ec3d84d2e1b471aa574c3bfca8ac1e72c4` | Kept DK1's separate lattice-realization evidence; did not conflate its four-orders recovery with #897's flip counts. |
| V14 stage diagnosis | `.omx/research/codex_findings_ddm_v14_realization_fidelity_20260722_codex.md` — `f724da020faf0fcce637297005a6050ff6c2fe1c677998676b961a1540427de7`; `.omx/research/ddm_v14_realization_fidelity_n600_20260722T215500Z/ddm_v14_realization_fidelity_n600_receipt.json` — `82d3249908d42a86575c407ab3d7acdf9b3706b31225f2e46862b2472966e5a9` | Required distinct round checkpoints and residual decomposition; no public-renderer residual exists yet to diagnose. |
| DM2/DM4 race and cures | `.omx/research/codex_findings_ddm_dm2_l3_realization_race_25_rows_20260724_codex.md` — `3127207303c6c748d32d7cd75b1e712fe41dda1ae9a3d9aa020107e5894de96a`; `.omx/research/ddm_dm2_l3_realization_race_25_rows_20260724T133300Z/ddm_dm2_l3_realization_race_receipt.json` — `8897241b7fc0ded7d4d6d1100c4d23ea162111050754e36c8ad8b3e57e294229`; `.omx/research/codex_findings_ddm_dm4_targeted_realization_cures_20260724_codex.md` — `b85d22f1080eba06524cc4fa0e6ecb8bf4229b9245980d5c4d89fcf278f8cc14`; `.omx/research/ddm_dm4_targeted_realization_cures_20260724T142722Z/ddm_dm4_targeted_realization_cures_receipt.json` — `9644ef24c8037485a6350193d9368f65f463ae102db70c3d1412a550362bf5bb` | Used their race/cure discipline to avoid inheriting constants; no cure was transferred across the fixed-CP135 and absent-public-renderer regime. |
| JS5 quantum floor | `.omx/research/ddm_js5_projector_distilled_conditioning_20260812.md` — `31b0082e4df300a8e6605672bcd53a7d2a5ad91a69bc6ffbcc4f1bef9760c6aa` | Replaced continuous amplitude shrinking with full categorical token-cell moves. |
| PO1 instrument floor | `.omx/research/ddm_po1_t4_error_feedback_pose_compensation_20260813.md` — `e31ad59d8d5a3b73ded971ffd7aaff58a36af802a87b5cb454b03c5846715358` | Withheld modeled acceptance below measured forward mismatch; its precise component surface is explicitly alternate-axis triage only. |
| JS7 anti-additive law | `.omx/research/ddm_js7_acceptance_sweep_and_compose_20260812.md` — `c07bb24f3f880b680d4fabb53e9981938cb3edbf46435db239f77997583c4b00`; `.omx/research/ddm_js7_exact_row_verdict_20260812.md` — `e887b217c8f75f43e5e950976b905778e5873d8a391b13ebdd90e1b65393053c` | Forbade singleton-additive acceptance and left both rounds unaccepted pending a whole-candidate complete-S row. |
| ET1 realized-η method | `.omx/research/ddm_et1_eta_on_the_priced_band_20260803.md` — `141e21797d27ec0dbe60dc863b80e5c83dfe4386750d7702df9c949290791a90` | Compared projected distortion against measured byte cost, but downgraded the result to cross-surface triage because realized complete S is absent. |
| HV1 composition/rank-4 target | `.omx/research/ddm_hv1_fresh_eyes_hybrid_review_20260813.md` — `f896753583378d1af2f2658392d9589f97d233ded2aa1856f1b0059bd6c7a6b3` | Selected the semantic-preimage × HP3 target and enforced whole-candidate complete-S as the only acceptance authority. |
| HC1 direct-C1 control | `.omx/research/ddm_hc1_hy1_container_push_20260812.md` — `a73d58b6fe0ebe8cb9e08ca145299a07d8f30dda93d06e5c7c550f0e7007c130` | Closed literal C1 representative substitution for this instance; retained only sparse CP135-input preimage editing. |
| PZ4R public-wire control | `.omx/research/ddm_pz4r_full_n600_eval_20260813.md` — `cd5a4b3b8b4dd026433fab55599bd382f8204c70cc6cf990876ba8c7439c2a8a` | Made retained actual Pose vectors a mandatory promotion prerequisite; parseback/rate alone is not Pose survival. |
| T1R1 container pattern | `.omx/research/ddm_t1r1_container_build_rehearsal_20260812.md` — `a9aafcf4039bee4cd48e4a55c031c7aa72058a301c6c4564388092d51d770e57` | Reused its resumable retained-payload, deterministic archive-repeat, and independent entropy-decode custody pattern. |
| JO1/VD1 proposal machinery | `experiments/ddm_jo1_joint_probability_object.py` — `77990654e90f4c4cd0d2b068e0f039da171889d0ecc6ec418ac5900868d688a9`; `.omx/research/ddm_vd1_census_verdict_20260812.md` — `72e37a18a874ac7bf6931730772f79392137a5857ae823b794d33a160252d30f` | Supplied the real candidate-conditioned HP3/RC64 builder and exact singleton inputs; their different component surface is triage, not score authority. |
| CP5V whole-stack control | `/Volumes/VertigoDataTier/pact/ddm_cp5v_20260812/20_COMPOSE_RESULT.json` — `b3b540ebe7ecfde0e8cc49a0f8b07d09c75318cca13516a3d669332e3ba1b717`; `30_TOKEN_DIFF_RESULT.json` — `e8607f1b8e1856b572c237ec2d5d8c9ba777acf0e9a72ee8ea8cff318fca51e7`; `FINAL_RESULT.json` — `62c280f83d47d7516975183db79d6bfe09950c453c57a80fb6231a572aa4f862`; `main_t4/contest_auth_eval.json` — `492686694d32f297c3149152e43fb8cbdb7397124a253c42a47a8ffaa9b50019` | Bound the five events to the evaluated archive; its report-component rounding interval prevents precise interaction calibration. |
| CP135/PO1/JS1B authority chain | `experiments/results/modal_auth_eval/ddm_cp135_composed_paired_modal_auth_20260810T193605Z_cuda/contest_auth_eval.json` — `1a62d07db424f176b64985c50f587387a5989384f75175c372eff40524683bd2`; `/Volumes/VertigoDataTier/pact/ddm_po1_20260813/round1_cp135/FINAL_RESULT.json` — `6829fec80af848426cda23dba92d8d2f9f28fccdae1aa046c6fc9bb00e64b4a2`; `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/stage0_per_edge/contest_cuda/ddm_js1b_20260813b/FINAL_RESULT.json` — `5fd65b946e2e1a5683e123554761c4216f8245a4d1cec46da2ee95b925c93a0c` | Kept canonical score authority separate from the precise alternate component surface and its explicit axis-mismatch refusal. |

## Evidence and precision boundary

The canonical CP135 receipt is pinned at
`experiments/results/modal_auth_eval/ddm_cp135_composed_paired_modal_auth_20260810T193605Z_cuda/contest_auth_eval.json`,
SHA `1a62d07d…bd2`: 186,252 B, archive `6eb1a3b7…edb6`, canonical report-component
score `0.16195513827824176`, n600 contest-CUDA T4. The pinned pointer snapshot is SHA
`001728e1…e7a`.

The precise proposal arithmetic uses PO1's component-only receipt, SHA `6829fec8…b4a2`:
`d_pose=6.885642960696714e-6` and 34,970 Seg errors. That receipt's raw SHA is
`2a8426a1…d701`; the canonical CP135 evaluator raw is `604459e2…fe1a`. JS1B receipt
`5fd65b94…3a0c` explicitly labels the 34,970-error surface `BLOCKED_AXIS_MISMATCH` and refuses it
for production admission. Its reconstructed base is `0.16195997548407048`, not the canonical
pointer.

Accordingly, round 1 reconstructs `0.1619587475226413` and round 2
`0.16195895926823925` only on that alternate surface. Round 1 is the better rate-adjusted triage
candidate. The round-1 delta's arithmetic magnitude is about 0.01027% of the gap to 0.15, but that
percentage is not measured and is not pointer-comparable.

CP5V binds its five event IDs to archive `1c66e434…a986` through compose receipt
`b3b540eb…b717`, final receipt `62c280f8…f862`, and exact evaluator receipt
`49268669…019`. Its canonical report-component point score is `0.16195716412468952`, above the
pointer, but its recorded score error bound is `±3.512376215e-6`; CP135's is
`±3.514565443e-6`. Mixing those rounded scores with the alternate precise singleton projections
gives a composition-residual point estimate `+3.172013979e-6` with interval
`[-3.854927679e-6, +1.019895564e-5]`. The sign is unresolved and this residual is not load-bearing.

## Engine and applied cures

Runner: `experiments/ddm_re1_realization_engineered_candidate.py`, receipt-refresh source SHA
`3b47fff58bb6dcabf34ac88fe283b804848d0d53ba6ad83ad70129c13916f83e`.
Tests: `experiments/tests/test_ddm_re1_realization_engineered_candidate.py`, SHA
`9d18e60d114fc4470667a09dcc07bc9dea80a0fecd93a883b0d34b1c9ea3dc83`.

| failure class | applied mechanism | measured closure | remaining boundary |
|---|---|---|---|
| uint8/lattice quantum | exact categorical token-cell moves, at least one full lattice step | requested cells survive byte-for-byte | rendered pixels and scorer cells unmeasured |
| HP3 state dependence | recompute affected frame and successor probabilities; reuse only hash-checked unaffected codes | fresh n600 RC64 streams retained | score effect unmeasured |
| entropy receiver | primary/repeat archives, physical parseback, shipped backend compile/decode | all 117,964,800 symbols/tokens exact twice per round | public F26 renderer not run |
| pose geometry | one-best-event-per-pair structural deduplication using exact-input singleton data | deterministic proposal set | axis mismatch; no whole-candidate Pose vectors |
| anti-additive composition | whole-candidate acceptance withheld; one scorer fire order | no false acceptance | complete S still required |
| rendered representative | fixed CP135 semantic-input plane only | entropy closure executable | no HR1/HR2 renderer/adapter consumer exists |

Borrowed-substrate accounting:

- `experiments/ddm_jo1_joint_probability_object.py`, SHA `77990654…88a9`, performs the real token
  materialization, candidate-conditioned HP3 export, RC64 encode, archive construction, parseback,
  and shipped entropy decode.
- `experiments/ddm_cp135_rate_compose.py`, SHA `c97bd5b4…9a5b`, supplies the exporter/coder and
  deterministic runtime-tree hash.
- RE1 adds the pinned selection law, exact two-round proposal sets, exclusive/resumable orchestration,
  content-addressed source/plan custody, runtime-tree pinning, and the retained new archive instances.

## Improvement ledger

| consumed result | banked performance | RE1 beat-target | RE1 change | measured delta / disposition |
|---|---|---|---|---|
| RVS1 survival playbook | 22 receipted mechanisms; six fully encoded, five partial, nine missing, two exclusions `[prior-receipt synthesis]` | Make at least one current-base cure survive the public R and improve whole-candidate S. | Used hard categorical steps and realized-only admission; did not inherit renderer constants. | Not attempted: the public renderer/scorer consumer is absent; no v1-optimum claim. |
| RVS2 geometry class | Geometry is the fifth survival class `[prior-receipt synthesis]`. | Hold or improve same-base Pose while Seg/rate improve on the complete candidate. | Selected at most one event per pair and withheld Pose claims without vectors. | Structural separation only; no Pose delta and no v1-optimum claim. |
| RHO1 prior | Current-object survival is unidentified; its optimal-form planning band is `[0.55, 1.00]` `[analog-conditioned prior]`. | Replace the prior with candidate-specific realized survival above the exact byte break-even. | Used the prior only to prefer sparse separated composition. | No survival fraction measured; public render/scorer absent. |
| #897 LL1/TR1 | 88→3 flips and `ΔS=-0.0144` on n3 `[macOS-CPU advisory]`. | Leave fewer than 3/88 residual flips on this base and improve complete S through the current real chain. | Preserved its solve-through-real-operators law without transferring its number. | Not attempted: current renderer/scorer consumer unavailable; no optimum claim. |
| DK1 lattice realizer | CVP/Babai reduced Pose leakage about 7,634× and scorer-delta discrepancy `0.7107→0.0653` on four blocks `[macOS-CPU frozen-PoseNet advisory, small-n]`. | Beat both leakage and discrepancy on current same-base supports, then improve complete S. | Kept lattice-native motion; fixed CP135 semantic cells expose no DK1 private-support actuator. | Not attempted on this vehicle; required support/consumer absent, no optimum claim. |
| V14 stage diagnosis | Ordered semantic paint improved `d_seg 0.02959276→0.02747030` `[macOS-CPU frozen-scorer advisory, n600]` but remained far from the frontier. | Reduce the binding residual after each retained round through the current public chain. | Kept separate round checkpoints and required residual decomposition. | No rendered residual fields exist yet; no optimum claim. |
| DM2/DM4 race and cures | DM2 preserved 25/25 obligations at 2,524.25× realized/semantic bytes; DM4 cut that ratio 18.16% `[macOS-CPU frozen-scorer advisory]`. | Preserve obligations at a frontier-admissible byte price and lower complete S. | Moved the intervention to fixed-renderer semantic preimages priced by real HP3/RC64. | One and three semantic cells closed at 0 B and +1 B; obligations are not comparable and score is unmeasured. |
| JS5 quantum floor | α=1/16 produced zero robust flips and 0/15 proposals were accepted `[macOS-CPU advisory, stratified n32]`. | Produce at least one robust, scorer-beneficial move after the hard receiver. | Replaced amplitude shrinkage with full categorical cell moves. | Exact token cells survived entropy decode; rendered/scorer benefit unmeasured. |
| PO1 instrument floor | Precise alternate-axis components expose a forward-mismatch floor and do not reproduce the canonical raw `[component-only T4 axis mismatch]`. | Admit only a step above mismatch that lowers same-object complete S. | Treated its singleton arithmetic as triage only and retained full-quantum moves. | Projection favorable only on the mismatched surface; zero accepted rounds. |
| JS7 anti-additive stack | 44-event stack scored `0.16342603740620176` at 186,575 B `[contest-CUDA T4, n600]`, worse than CP135. | Make the whole composition beat the best singleton and CP135 under complete S. | Deduplicated pairs and refused additive acceptance. | Round 2 cost +1 B and has weaker cross-surface triage than round 1; unaccepted. |
| ET1 realized-η method | Phase-field η was promising only on non-bankable/analog scopes and Pose remained adverse `[macOS-CPU advisory]`. | Measure realized η above the exact byte break-even under named caps on this candidate. | Priced every retained proposal with the real RC64 stream. | Rate is 0 B/+1 B; realized score gain and therefore η are unavailable. |
| HV1 composition/rank-4 target | Named semantic-support × representative/preimage × HP3 and complete-S-only admission. | Beat CP135 `0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`. | Built the preimage × HP3 object while keeping acceptance fail-closed. | Entropy closure passed; complete S absent, target not met. |
| HC1/C1 | C1 carriage was cheap but scored `0.4044688071472634 @ 187,046 B [contest-CUDA T4, n600]`. | Retain cheap carriage while beating CP135 complete S. | Kept CP135 renderer/carrier and edited sparse semantic preimages. | Direct substitution folded; sparse objects are unscored. |
| HR1/HR2 programs | Camera-uint8 apparatus is bound, but all four typed arms have `execution_allowed=false`. | Produce an executable current-base exporter/runtime consumer and a better complete-S row. | Audited every program and refused a proxy optimizer. | No executable arm exists; no v1-optimum claim. |
| JO1/VD1 singleton machinery | Exact singleton component rows plus real n600 HP3/RC64 closure `[mixed exact-input component surface; not canonical score]`. | Convert the strongest singleton into a whole candidate at no rate cost and lower canonical complete S. | Round 1 selected the strongest distinct-pair singleton and reclosed it. | 0 B, one exact cell, entropy receiver exact; canonical score unmeasured, queued. |
| CP5V five-event composition | Report-component point score lies above the pointer; its interaction-residual sign is unresolved within bounds `[contest-CUDA T4, n600 report-component authority]`. | Produce a composition whose lower uncertainty bound beats the pointer. | Removed the rounded residual from admission and used only structural pair separation. | Calibration folded; no positive interaction credit claimed. |
| PZ4R pose-survival control | 183,137 B direct-v6 worsened complete advisory S by `+2.471539548` through Pose collapse `[macOS-CPU advisory, n600]`. | Retain actual Pose vectors and prevent any pose collapse before score handoff. | Made full Pose fields a mandatory return artifact. | Not testable locally without the scorer slot; no optimum claim. |
| T1R1 container pattern | Real resumable C1 container/entropy receiver closed every token `[macOS-CPU scorer-free container proof]`. | Preserve that byte closure for a new sparse representative, then close the public receiver. | Reused its retained-payload and independent-decode pattern. | Entropy target met twice per round; public-receiver target remains unmet. |
| round 1 iteration seed | One exact event, 0 B, no within-stack interaction `[macOS-CPU scorer-free]`. | Round 2 must improve the same-object complete S over round 1. | Added the best projected event from each of two new pairs. | Round 2 retained three exact events at +1 B, but has weaker cross-surface triage; not accepted and no wall verdict. |

This satisfies the amendment's request for two materialized proposal iterations, but not its
requirement for two realized complete-S acceptances. A wall verdict was not emitted.

## Retained custody and resumability

Target store:
`/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/probability_object_race/ddm_re1_20260813`
(`3.8G`). All token planes, event orders, proposal payloads, probability files, RC64 checkpoints,
members, archives, repeats, decoded symbol/token payloads, runtime copies, and receipts remain kept.

The exact 25,155-byte runner that produced the candidate payloads was recovered from the tool
transcript and retained append-only at SHA
`a1a33911e3918044e35f6f0fc4e0500307cdc5e83c80126fa9e85025d3e6b646`.
`retained/source/recovery/PRODUCING_SOURCE_RECOVERY.json`, SHA `39a5adb2…e43`, binds the matching
patch call, the immediately following successful preflight, and a durable transcript snapshot.
Current runner and plans are stored under content-addressed paths; the legacy mutable source path is
not producer authority.

Candidate payload subtrees are resume-reused only after file-record and proposal-ID validation.
Top-level lock/preflight/plan/round/final/state receipts and per-round audit receipts are refreshable
control records, not immutable payload checkpoints.

Round-1 runtime tree: 25 files, SHA `63b93187…dc75`; manifest SHA `000544b3…37dc`.
Round-2 runtime tree: 25 files, SHA `feb04035…51e8`; manifest SHA `a5595f4e…62a8`.
The final safe-run seal passed; `FINAL_RESULT.json` SHA is `db5fa5f7…380b` and the current fire-order
record SHA is `068fd465…1a8e`.

## QUEUE ANNEX

Disposition: **`QUEUED-WITH-A-FIRE-ORDER` (store-local, pending MAIN ingestion)**

Owner: **`MAIN sole scorer-lane router (pending acceptance)`**

Candidate: round 1 archive `7be3eb94b229306278a6ed204e2c716d7aafa98f6f93c82a5d2be18822467dfa`,
186,252 B; runtime tree `63b93187e83cb310d68031a2b08b65b1a5e2103e830cede4941a7d3df604dc75`

Consumer store:
`/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/probability_object_race/ddm_re1_20260813/full_n600_exact/round_01_singleton_best`

Fire trigger: MAIN accepts and ingests this exact store-local order, explicitly transfers the sole
full-n600 scorer slot, confirms no other scorer job is active, verifies the archive/runtime pins,
and limits every scorer chunk to at most 120.

Required return: retained public-runtime `0.raw`, full Seg logits and argmax fields, full PoseNet
vectors, and `upstream/evaluate.py` complete S over the exact archive/runtime. `main_hot_state.md`
contains no RE1 row at seal time, so this is not an accepted live queue job. It is an exact-eval fire
order, not a T4 promotion order; the promotion condition was not met locally.

## Conclusions

- RE1 built two new deterministic, byte-closed semantic-preimage archives through real
  candidate-conditioned HP3/RC64 entropy closure.
- Round 1 is the current scorer handoff because its measured rate-adjusted triage projection is
  better than round 2's. That projection is cross-surface and makes no score claim.
- Public rendering, real R, scorer fields, and complete S remain absent. Zero optimization rounds
  were accepted, the charter is incomplete, and no wall verdict is justified.
- Renderer-weight realization engineering remains blocked by a missing executable exporter/runtime
  consumer, not by the ability to carry sparse semantic changes.

RE1 measured no complete-S row and moved neither pointer. Own exact frontier remains LC2
`0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]`; effective pointer remains CP135
`0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`.

## NEXT_IF_RESUMED

- **Disposition: `QUEUED-WITH-A-FIRE-ORDER` (pending MAIN ingestion). Owner: `MAIN sole scorer-lane router (pending acceptance)`. Consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/probability_object_race/ddm_re1_20260813/full_n600_exact/round_01_singleton_best`. Fire trigger: MAIN accepts this exact store-local order, transfers the sole full-n600 scorer slot with no other scorer active, verifies archive SHA `7be3eb94b229306278a6ed204e2c716d7aafa98f6f93c82a5d2be18822467dfa` at 186,252 B and runtime-tree SHA `63b93187e83cb310d68031a2b08b65b1a5e2103e830cede4941a7d3df604dc75`, and enforces chunks of at most 120; then retain the public raw, full Seg/Pose fields, and complete-S receipt.**

## LIVE-HYPOTHESES

- The one-event round may improve the canonical complete S because its exact-input singleton effect
  is favorable on a closely related T4 component surface, and it avoids composition interaction;
  only the public-renderer whole-candidate row can test transfer across the observed raw mismatch.
- Candidate-conditioned HP3 is a useful sparse semantic carrier because one and three categorical
  events closed at 0 B and +1 B respectively, with independent byte-identical encodes.
- A real renderer adapter could turn C1's semantic-support signal into a shippable representative;
  this is plausible because carriage is cheap, but it remains untestable until an exporter/runtime
  consumer exists.

## DEAD-ENDS

- `verdict_scope: INSTANCE` — Direct full C1 substitution on this instance: HC1's contest-CUDA point score was about 0.4045 with
  severe Pose regression.
- `verdict_scope: INSTANCE` — Whole-frame CP135/C1 selection: all 600 retained C1-rendered semantic frames were worse than CP135.
- `verdict_scope: FORMULATION` — Treating CP5V's rounded report point as a precise interaction calibration: its residual interval
  spans zero and the component surfaces do not match.
- `verdict_scope: INSTANCE` — Selecting round 2 over round 1 for the current scorer handoff: its +1 B makes its rate-adjusted
  triage projection weaker, although the payload remains retained as the second iteration.
- `verdict_scope: FORMULATION` — Continuous sub-LSB amplitude shrinking: JS5 produced no robust flips and PO1 measured a larger
  forward-mismatch floor.
- `verdict_scope: FORMULATION` — Treating HR1/HR2 bindings as an optimizer: every typed program is non-executable and has no argv or
  consumer.
- `verdict_scope: INSTANCE` — Calling either proposal an accepted optimization round or receiver-closed candidate: no public
  raw, scorer fields, or whole-candidate complete S exists.
