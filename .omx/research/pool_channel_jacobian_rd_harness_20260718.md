# Pool x channel Jacobian rate-distortion harness — BUILD receipt

**Date:** 2026-07-18 UTC
**Status:** `BUILD_ONLY`; every scientific cell is `UNMEASURED_AWAITING_GOVERNED_N600`.
**Authority boundary:** no training, no provider call, no scorer evaluation, no n600 measurement, no candidate archive, and no scientific verdict were produced. `score_claim=false`; `promotable=false`; `$0`; no launch.
**Landing boundary:** `MAIN_LANDING_REVIEW_REQUIRED`; no co-author attribution.

## Outcome first

**DERIVED (this build):** `tools/pool_channel_jacobian_rd_harness.py` now defines the production-shaped, resumable contract for the 48-cell pool x head-direction x SegNet-path x resize-component experiment. Its dependency-light `plan` validates exact input custody and emits 48 null/empty scientific rows. Its `self-test` checks only structure. Its governed `measure` mode refuses without a fresh canonical local-CPU lane claim, exact-harness operator-GO artifact, and fresh terminal-c2 receipt. Once authorized, it can execute and checkpoint the real full-path parsed-bank receiver -> canonical rounded render/R -> frozen CPU Torch SegNet geometry. The receiver inputs are the parsed int8-dequantized base parameters **and** parsed code, GT rows are seek-streamed from the ZIP_STORED n600 cache, and exact lattice probes rebuild/parse the full canonical transformed+Brotli code section. It then refuses at the honest cell-admission boundary until the exact skip/deep and full range/kernel interventions exist; it never substitutes a synthetic scorer, local head Gram, zero-skip ablation, or asserted byte counts.

**DERIVED (this build):** `admit-rd` is a separate create-only component-point gate. It independently rederives the named saddle-first pool mask from custodied n600 baseline/GT argmax, requires byte-exact equality with the supplied target mask, recounts baseline/candidate transitions, and derives every rate value. Admission additionally requires a content-hashed n600 finding-mode run contract, completed finding-eligible geometry stage, exact row-specific head/path/resize geometry, and a separate same-point intervention receipt, all bound into the receiver/scorer receipt and exact baseline/candidate archive members. A grammar-only ZIP, arbitrary mask labelled as a row, or self-asserted `d_seg` cannot enter the table. This remains `d_seg`-component authority only: `score_claim=false`; `promotable=false`.

**UNMEASURED:** no new `G_act`, intrinsic floor, extrinsic ceiling, collateral cost, or R-D point exists. The live bank, sacred c2 run, GT caches, and frontier pointer were read-only/untouched.

## Canonical pointer, re-derived instead of copied from the prompt

**MEASURED (build-time read-only snapshot, not compiled into the harness):** `/Users/adpena/Projects/pact/.omx/state/canonical_frontier_pointer.json` was 12,349 bytes, SHA-256 `700e106ce63f4030f3fa05310b7a3561412ae40049cf4ad3150aa8bde803bcd3`, refreshed `2026-07-17T18:00:11.118462+00:00`, and recorded contest-CPU score `0.1880443979880752`. Every future `plan` dynamically re-reads and hashes the canonical pointer before and after validation, refusing concurrent drift; no current score, pointer size, or pointer digest is hard-coded.

**STALE TRANSIENT SNAPSHOT:** the authority prompt's `0.19108` is not presented as current. **Pointer delta for this build: exactly `0`; pointer mutated: false.**

## Exact input custody

The plan hashes files by streaming bytes and refuses size or SHA drift. It also proves the n96 cache is the exact first-96-pair identity surface of n600 by matching the raw `lstars.npy` prefix bytes, dtype, and trailing shape without importing NumPy or materializing either cache. Governed measurement requires `lstars.npy` to remain C-order `ZIP_STORED` and seek-reads only each pending batch's rows; it does not load the 944 MiB n600 label member into memory.

| Input | Bytes | SHA-256 | Status |
|---|---:|---|---|
| `experiments/results/banks/v9c2_defensive_bank_20260718/levelset_witness_ema_BEST.npz` | 460,448 | `b0a431e9259cd3c54ae53b677076823f36e096b27eb0d9ba74ed7c54c9113cef` | **MEASURED custody; read-only** |
| `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` | 5,078,017,610 | `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6` | **MEASURED custody; read-only** |
| `experiments/results/mlx_fleet_gt_cache/gt_n96.npz` | 812,484,058 | `6aad6600d93a5c25e94207ee411d3b4daf93136b8ea4235b6f7b9d96f04ab104` | **MEASURED custody; read-only** |

The sacred live surface `experiments/results/levelset_n600_witness_20260717T113932Z/` was not read, written, resumed, signaled, or otherwise perturbed.

## Settled evidence consumed, not re-derived

- **MEASURED (settled prior artifact):** the centered frozen SegNet head singular values are `(3.128, 2.154, 2.025, 1.796, 0)`; the fifth direction is exactly zero and is excluded. A governed real run must materialize the four vectors from pinned head weights, fix their signs deterministically, and persist weights/vector hashes plus reconstruction residual. This build does not call the settled singular values a new measurement.
- **DERIVED/CERTIFIED (settled resize geometry):** exactly `22.6969%` of camera pixels have axis-aligned zero weight under the scorer resize. **DERIVED (dimension count):** the full orthogonal nullspace dimension fraction is `80.674%`. These are different objects.
- **MEASURED (settled, sample-specific prior):** approximately `52%` rendered/marginal energy in `ker(A)` was observed on a prior sample. It is deliberately absent as a harness constant and cannot replace either geometric fact.
- **DERIVED (authority taxonomy):** `intrinsic` means RATE-DOMINATED within `{ker(A), sub-uint8-LSB, GT-flicker-band}`. `extrinsic` means coherent, uint8-realizable rank-4 control in `range(A)`. Nothing is labeled unreachable.
- **DERIVED (exclusive pool law):** for flip mask `F`, saddle mask `S`, Road-Lane edge/near mask `E_RL`, pool B is `F & S`; pool A is `F & ~S & E_RL`; pool C is `F & ~(A | B)`. The helper asserts pairwise disjointness and exact exhaustion of `F`.

## Governing equations

Let `u` be the parsed bank-code coordinate in int8-bin units, not camera pixels and not a local head feature. Let `T(u) = N_seg(R(render(u)))` denote the actual canonical render, camera uint8 roundtrip/resize `R`, and frozen CPU Torch SegNet logits. The local full-path stacked spatial Jacobian uses a continuous bank-code surface:

\[
J_u[:,j] = \frac{T(u+h_j e_j)-T(u-h_j e_j)}{2h_j},
\]

where `h_j` is the smallest pre-registered in-range dyadic step that changes at least one uint8 output and keeps each plus/minus render within one LSB of baseline. The separate deployability surface applies an exact `+/-1` int8-bin mutation, losslessly round-trips the code section, canonically renders the parsed value, and records its uint8 change. A continuous derivative is never presented as proof that a quantized bin is realizable, and a realizability secant is never silently substituted for the local derivative. **UNMEASURED:** neither surface was executed in this BUILD-only landing.

The four head channels use the persisted, sign-fixed **left** singular basis `U4` and raw projection `U4.T @ centered_logits`; there is no singular-value division. Each pair owns an independent 32-coordinate code block. Therefore global geometry is the direct sum over pairs: `sum_i trace(K_i^2)`, never `trace((sum_i K_i)^2)`. Cross-pool terms are computed only within the same pair before summing over pairs, so the harness cannot invent cross-pair coupling.

For a custodied rate/coordinate metric `H_u`, the actuator-space coupling object is

\[
G_{act}=J_u H_u^{-1}J_u^T.
\]

The local centered-head Gram is not `G_act`: it omits renderer, rounding, resize, nonlinear trunk, and cross-pixel collateral. Camera-pixel VJPs are diagnostic observers only; full deployable-coordinate rounded finite differences are authority.

For streaming spatial rows `j_i` of `J_u`, the harness right-whitens by the Cholesky transpose inverse `L^-T` when `H_u=L L^T`, then accumulates

\[
\widetilde J=J_uL^{-T},\qquad
K=\widetilde J^T\widetilde J=\sum_i \widetilde j_i^T\widetilde j_i,\qquad
E_{cross}=\operatorname{tr}(K^2)-\sum_i\lVert \widetilde j_i\rVert_2^4.
\]

This is the actual stacked off-diagonal spatial energy only when every row carries real rounded render/R/frozen-scorer provenance. The self-test checks both identity and a non-diagonal SPD metric against explicit `J H_u^-1 J^T`; its constructed matrices are explicitly non-scientific.

The eventual archive waterfill is non-additive:

\[
\min_{b_c\ge 0}\sum_c D_c(b_c)\quad\text{s.t.}\quad\sum_c b_c\le B,
\]

with admitted active cells sharing marginal R-D value at the optimum. Isolated pool ceilings must never be summed as independent opportunity.

## The 48-cell admission table

Legend: `A` = `A_road_lane_edge_near`; `B` = `B_saddle`; `C` = `C_remainder`; `U` = `UNMEASURED_AWAITING_GOVERNED_N600`. Every numeric scientific field is literally `null`; every R-D list is literally `[]` in the machine receipt.

| Pool | Head direction / settled sigma | Path | Resize | Status | intrinsic floor | extrinsic ceiling | collateral | R-D points |
|---|---|---|---|---|---|---|---|---|
| A | sv1 / 3.128 | skip | range(A) | U | null | null | null | [] |
| A | sv1 / 3.128 | skip | ker(A) | U | null | null | null | [] |
| A | sv1 / 3.128 | deep | range(A) | U | null | null | null | [] |
| A | sv1 / 3.128 | deep | ker(A) | U | null | null | null | [] |
| A | sv2 / 2.154 | skip | range(A) | U | null | null | null | [] |
| A | sv2 / 2.154 | skip | ker(A) | U | null | null | null | [] |
| A | sv2 / 2.154 | deep | range(A) | U | null | null | null | [] |
| A | sv2 / 2.154 | deep | ker(A) | U | null | null | null | [] |
| A | sv3 / 2.025 | skip | range(A) | U | null | null | null | [] |
| A | sv3 / 2.025 | skip | ker(A) | U | null | null | null | [] |
| A | sv3 / 2.025 | deep | range(A) | U | null | null | null | [] |
| A | sv3 / 2.025 | deep | ker(A) | U | null | null | null | [] |
| A | sv4 / 1.796 | skip | range(A) | U | null | null | null | [] |
| A | sv4 / 1.796 | skip | ker(A) | U | null | null | null | [] |
| A | sv4 / 1.796 | deep | range(A) | U | null | null | null | [] |
| A | sv4 / 1.796 | deep | ker(A) | U | null | null | null | [] |
| B | sv1 / 3.128 | skip | range(A) | U | null | null | null | [] |
| B | sv1 / 3.128 | skip | ker(A) | U | null | null | null | [] |
| B | sv1 / 3.128 | deep | range(A) | U | null | null | null | [] |
| B | sv1 / 3.128 | deep | ker(A) | U | null | null | null | [] |
| B | sv2 / 2.154 | skip | range(A) | U | null | null | null | [] |
| B | sv2 / 2.154 | skip | ker(A) | U | null | null | null | [] |
| B | sv2 / 2.154 | deep | range(A) | U | null | null | null | [] |
| B | sv2 / 2.154 | deep | ker(A) | U | null | null | null | [] |
| B | sv3 / 2.025 | skip | range(A) | U | null | null | null | [] |
| B | sv3 / 2.025 | skip | ker(A) | U | null | null | null | [] |
| B | sv3 / 2.025 | deep | range(A) | U | null | null | null | [] |
| B | sv3 / 2.025 | deep | ker(A) | U | null | null | null | [] |
| B | sv4 / 1.796 | skip | range(A) | U | null | null | null | [] |
| B | sv4 / 1.796 | skip | ker(A) | U | null | null | null | [] |
| B | sv4 / 1.796 | deep | range(A) | U | null | null | null | [] |
| B | sv4 / 1.796 | deep | ker(A) | U | null | null | null | [] |
| C | sv1 / 3.128 | skip | range(A) | U | null | null | null | [] |
| C | sv1 / 3.128 | skip | ker(A) | U | null | null | null | [] |
| C | sv1 / 3.128 | deep | range(A) | U | null | null | null | [] |
| C | sv1 / 3.128 | deep | ker(A) | U | null | null | null | [] |
| C | sv2 / 2.154 | skip | range(A) | U | null | null | null | [] |
| C | sv2 / 2.154 | skip | ker(A) | U | null | null | null | [] |
| C | sv2 / 2.154 | deep | range(A) | U | null | null | null | [] |
| C | sv2 / 2.154 | deep | ker(A) | U | null | null | null | [] |
| C | sv3 / 2.025 | skip | range(A) | U | null | null | null | [] |
| C | sv3 / 2.025 | skip | ker(A) | U | null | null | null | [] |
| C | sv3 / 2.025 | deep | range(A) | U | null | null | null | [] |
| C | sv3 / 2.025 | deep | ker(A) | U | null | null | null | [] |
| C | sv4 / 1.796 | skip | range(A) | U | null | null | null | [] |
| C | sv4 / 1.796 | skip | ker(A) | U | null | null | null | [] |
| C | sv4 / 1.796 | deep | range(A) | U | null | null | null | [] |
| C | sv4 / 1.796 | deep | ker(A) | U | null | null | null | [] |

## Exact n600 execution and resume gate

`measure` requires all of the following before it creates its output/checkpoint directory:

1. explicit `--operator-go`;
2. `--claim-receipt` plus exact `--claim-receipt-sha256`, where the wrapper is fixed to this lane, `$0`, local CPU, and the named c2 predecessor;
3. a fresh `tac_active_lane_claim_json_v1` export whose row hash is re-found in the current canonical claims ledger, whose ledger SHA still matches, and whose TTL is at most 24 hours;
4. a JSON operator authorization under `.omx/research/operator_authorizations/`, valid for at most 24 hours and bound to the exact harness SHA, job ID, bank SHA, n600 GT SHA, batch 32, and verbatim operator quote;
5. a <=1-hour terminal predecessor receipt under the sacred c2 run or governed-run receipt directory, with `c2_complete=true` and `pid_alive=false`;
6. for findings, exactly `--pair-count 600 --segnet-batch-size 32` without `--liveness-only`; for any smaller smoke, explicit `--liveness-only`, always nonfinding;
7. the same pinned bank/cache/pointer custody and exact n96-prefix identity checks used by `plan`.

The preserved stage order is `custody -> baseline_byte_close -> geometry -> coherent_corrections -> archive_rd -> complete`. Each completed stage is a distinct atomic `stage_NN_<name>.json`; `run_contract.json` is content-bound; each stage and pair record independently binds `payload_sha256`; an existing run requires explicit `--resume`; completed records are never overwritten. Pair-level shards use atomic, preserved `pairs/<stage>/pair_NNNN.json` records under the same contract hash. Compatible resume verifies the entire baseline custody payload and every pair payload/hash; envelope-only or payload-only tampering refuses. No scientific pair shard was created in this build.

An authorized future invocation first byte-closes and parses the baseline LVLS1 manifest, base parameters, and code, hashes runtime/head/U4/H custody, and preserves pair-local geometry checkpoints from the real full-path continuous and exact-code-section lattice surfaces. The finite differences perturb frame 1 (`2*pair+1`), never frame 0. After the full-path geometry stage it fails closed with:

`BLOCKED_EXACT_SKIP_DEEP_HOOK_AND_RANGE_KERNEL_INTERVENTION:real full-path rounded deployable-code Jacobian completed and preserved;48 path/component cells remain null until same-point interventions are exact`

The two component blockers are `BLOCKED_EXACT_SKIP_DEEP_HOOK` and `BLOCKED_FULL_ORTHOGONAL_RANGE_KERNEL_INTERVENTION`. They are implementation-custody blockers, not negative verdicts on coherent correction or any representation family.

## R-D admission law

`admit-rd` refuses unless both baseline and candidate are existing exact ZIP archives with exactly one `0.bin`, exact byte counts, archive SHA-256 values, and member SHA-256 values. It hashes and memory-maps separate `(600,384,512)` uint8 baseline/candidate/GT argmax arrays and a bool target mask. For every frame it independently derives the canonical nearest-boundary, saddle-first A/B/C partition from baseline versus GT; the supplied mask must equal the named row's pool bit-for-bit. A pool-labelled, shape-correct arbitrary mask therefore refuses. In 32-pair chunks it then recounts baseline flips, candidate flips, target fixes, newly bad pixels, and all class transitions outside that canonical mask. `baseline_d_seg`, `d_seg`, rate, byte delta, d_seg delta, and Seg score-units bought are derived from those arrays and exact archive bytes—not accepted from prose.

Before that scorer receipt is considered, the gate verifies the content hash of `run_contract.json`, requires the exact current harness, pinned bank/GT, pair indices `0..599`, batch 32, and finding mode, then verifies a completed geometry checkpoint under that contract. The checkpoint must be `finding_eligible=true` and contain the exact row with pinned/sign-fixed head-direction custody, measured content-hashed `G_act`, nonnegative cross-location energy, exact skip/deep intervention plus recomposition, full orthogonal range/kernel intervention plus recomposition, and the independently derived pool-mask hash/count. A separate content-hashed cell-intervention receipt must bind that same geometry cell and mask to both archives, all argmax arrays, exact byte close, receiver closure, and the same operating point.

The separately hashed receiver/scorer receipt must bind all derived values plus those run/stage/cell/intervention hashes, the exact row ID, archives, argmax/mask hashes, pinned SegNet weights, axis, n600/batch32 geometry, and `receiver_closed=true`. A grammar-only fake ZIP is a self-test refusal. Admission is deliberately scoped to the `d_seg` component and forces `score_claim=false`, `promotable=false`; it is not an overall contest score or a Pose verdict. The currently built geometry stage is explicitly `finding_eligible=false` and has no exact 48-cell records, so it cannot admit a point.

For each future admitted candidate the owed chain is: mutate custodied deployable codes -> canonical int8/packet build -> parse back -> deterministic complete archive assembly -> hash exact archive bytes -> inflate/rerender through canonical receiver/R -> frozen CPU Torch rescore -> record transition counts and bytes. Optimizers may waterfill across cells; admission attribution remains saddle-first exactly-one pool.

## Structural validation performed

**MEASURED (local structural execution, no scorer dependency):** all acceptance commands passed.

```text
python3 tools/pool_channel_jacobian_rd_harness.py --help
  PASS
python3 tools/pool_channel_jacobian_rd_harness.py self-test
  PASS; row_count=48; scientific_measurement=false; synthetic_scorer_instantiated=false
python3 tools/pool_channel_jacobian_rd_harness.py plan --output .omx/tmp/pool_channel_rd_harness_plan_post_attribution_20260718.json
  PASS; receipt sha256=1d1c33a131059a7f59c690e41f7502ce7ac1a1676393a9712ea422bb24979d92;
  48 unique null rows; pinned custody and n600/n96 lstars-prefix identity validated
python3 -m py_compile tools/pool_channel_jacobian_rd_harness.py
  PASS
ruff check + ruff format
  PASS
measure without operator-GO
  REFUSED before writes with BLOCKED_OPERATOR_GO_REQUIRED
real ZIP_STORED GT row read [0,95,599]
  PASS; shape=(3,384,512), dtype=int64, labels=[0,4]; no full-cache materialization
real bank LVLS1 build + receiver parse-back (no SegNet forward)
  PASS; n_pairs=600; parsed base parameter count=17; parsed code shape=(1200,32);
  accounting_matches_canonical=true
independent plan inspection
  PASS; every intrinsic_floor/extrinsic_ceiling/collateral_cost is null;
  every rd_curve_points is []; score_claim=false; promotable=false; pointer_delta=0
```

The self-test exercises row cardinality/schema, hand-constructed nearest-boundary saddle-first pool derivation and labelled content hashing, deterministic SVD sign fixing, small-matrix orthogonal projectors, identity and non-diagonal-SPD `J H^-1 J^T`, strict row-attributed content-derived R-D custody including a grammar-only fake, and envelope/payload-tamper-resistant stage/per-pair resume. These are **STRUCTURAL_ONLY**, not scientific measurements and not a scorer surrogate.

## Triality and explicit debts

- **CLI artifact gate — LANDED:** explicit subcommands `plan`, `self-test`, `admit-rd`, and governed `measure`; no trainer flag or launch path was invented. **DSL DEBT:** argparse is not the typed witness DSL, so no DSL leg is claimed; an actuating coherent-correction lever must register there only after the exact interventions exist.
- **DAG / stages — LANDED LOCALLY:** the six-stage content-bound checkpoint machine above. **DEBT:** it is not appended to the shared DAG/registry consumer surfaces in this two-file isolated landing.
- **Equations — LANDED IN THIS RECEIPT:** deployable central difference, `G_act`, streamed cross-location energy, and waterfill equations. **DEBT:** no canonical-equations registry row was added because this arm owns exactly two files.
- **Consumer hook — DEBT:** waterfill task #536 must consume a completed n600 receipt only after exact path attribution and byte-closed archive R-D rows exist. No integration hook was faked.

## Round-1 adversarial self-review

1. **Is the Gram really cross-pixel collateral?** The code right-whitens real stacked spatial rows, accumulates pair-local `K_i=J_i^T J_i` (identity H in the current bank-code run), and derives the off-diagonal identity, then sums `trace(K_i^2)` without ever summing independent pair blocks before squaring. It records H, row-location, runtime, U4, and provenance custody. It does not compute or label a local head Gram as `G_act`. **PASS structurally; n600 value remains UNMEASURED.**
2. **Is attribution exclusive and cell-specific?** B is assigned first, A explicitly excludes saddle, and C is the complement within flips; the helper asserts exactly one assigned pool for each flip and none for non-flips. Admission independently rederives that pool for every baseline/GT frame, rejects any target-mask mismatch, and requires the same row's content-hashed head/path/resize geometry and intervention receipts. **PASS structurally; zero points admitted.**
3. **Are R-D points byte-closed?** Admission independently validates both archives, all argmax/mask arrays, exact counts, exact-byte rate, completed n600 row geometry, and a receipt binding the same receiver/scorer/intervention evidence. The fake-ZIP regression proves archive grammar alone cannot admit. **PASS structurally; zero points admitted.**
4. **Is skip/deep real?** No. The exact intervention is absent; the harness returns `BLOCKED_EXACT_SKIP_DEEP_HOOK`. It never treats zero-skip as an observation. **Honest blocker retained.**
5. **Could a small smoke become a finding?** No. Pair counts below 600 require explicit liveness mode and are nonfinding; n600 requires batch 32. **PASS structurally.**
6. **Did this move the goal pointer?** No. Current pointer remains `0.1880443979880752`; delta is exactly zero. This BUILD is a means, not goal completion.

## Final grounded status

**MEASURED at n600 in this landing: nothing.** **BUILT and locally structurally verified:** dynamic custody validation; 48-cell null receipt; saddle-first/nearest-boundary pool assignment; parsed LVLS1 base-parameter+code baseline; sign-fixed left-U4 projection custody; frame-1 full-path continuous bank-code finite differences; full canonical code-section int8 parseback/render realizability; bounded GT/cache and render batches; pair-independent streamed geometry; canonical-pool and exact-cell-attributed byte-closed R-D admission; and payload-hashed resumable stages. **AWAITING GOVERNED n600:** every scientific output cell. **BLOCKED FOR CELL FINDINGS:** exact skip/deep hook with recomposition proof, full orthogonal range/kernel intervention at the same operating point, a completed finding-eligible 48-cell geometry stage, and admitted receiver-closed archive R-D points. Bank/live run/pointer untouched; `$0`; no launch; no score.
