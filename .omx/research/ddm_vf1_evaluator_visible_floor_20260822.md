# DDM VF1 evaluator-visible floor census — 2026-08-22

**Lane:** `ddm_vf1_evaluator_visible_floor`

**Disposition:** `INCONCLUSIVE — MEASUREMENT OWED, DO NOT FIRE HERE`

**Authority:** scorer-free retained-receipt audit; no scorer, Metal, Modal, training, archive mutation, or JO-r9 read occurred

**Vehicle:** current DX2 archive lineage only

## Result first

The current DX2 archive needs a strict cut of **42,382 B** to cross `S < 0.12` at fixed distortion. The retained corpus does **not** contain a qualifying current-DX2 per-token or per-cell evaluator-equivalence census from which any portion of that cut can be credited.

The complete classification-status census is therefore (`[scorer-free exact dimensional/accounting]`):

| Classification | Token positions | Position denominator | Current coded mass | Byte denominator |
|---|---:|---:|---:|---:|
| `LOAD_BEARING` | 0 | 117,964,800 | 0 B | 113,777 B |
| `INERT` | 0 | 117,964,800 | 0 B | 113,777 B |
| `UNMEASURED` | **117,964,800** | **117,964,800** | **113,777 B** | **113,777 B** |

This says **all positions are unclassified**, not that none is load-bearing or inert. The measured-subset denominator for the required complete tuple—`Δbytes`, `Δd_seg`, `Δd_pose`, and registered repeat noise on the current DX2 field—is **0 / 117,964,800 positions** and **0 / 113,777 coded bytes**.

Consequently, the measured evaluator-visible MDL credit is **0 B / 42,382 B required = 0%**. The prior “minority load-bearing, inert mass on the order of the required cut” prediction is **neither confirmed nor refuted**. It receives zero frontier credit.

## Provenance and strict-byte boundary

All charter-pinned inputs matched before the audit:

| Receipt | SHA-256 |
|---|---|
| RB1 base measurement | `fa26a44444a57428910565956011e0bb26c6680174a71bfbb914002f9f564f09` |
| XT1 score algebra | `6437bc53d96e527049c3fd6cd60b91af220305881a7bcc68195fece15a728867` |
| TK1 roofline | `5519cce5a986ffd1536233c2f0865a1ce2f95996293f230cb8a0da0f30e09861` |
| FP1 carrier floor | `b594de4b53d58a1535466f8dc94f14b6fbb87c4d16d8be53b01089996aeef42d` |
| NL1 nullspace ledger | `a11e56b228513c066b803cb6c03e7ce31d2af40d7271b812abaff5e16b5ced3a` |

Current DX2 facts (`[contest-CUDA T4, n600]` for score components; `[scorer-free exact archive anatomy]` for bytes):

- `B = 180,368 B`
- `d_seg = 0.00020139`
- `d_pose = 0.00000637`
- `D = 100*d_seg + sqrt(10*d_pose) = 0.028120227975693966`
- `S = 0.14821987563243377`
- current token stream = `113,777 B / 180,368 B = 63.0805%` of the archive

At fixed distortion, the continuous boundary is `137,986.8387944436 B`; the largest integer archive satisfying strict `S < 0.12` is **137,986 B**. Thus the required cut is:

`180,368 - 137,986 = 42,382 B`.

Boundary check:

- `B = 137,986 B` gives `S = 0.11999944148120990` — passes.
- `B = 137,987 B` gives `S = 0.12000010734016302` — fails.
- `0.001` score unit corresponds to `1,501.81956 B` at the contest denominator.

## What a qualifying row requires

A token or cell is `INERT` only when a concrete alternative survives the actual receiver and has all of the following on the current DX2 decoded field and current FX5 token coder:

1. a real re-encode `Δbytes`, including arithmetic-context effects;
2. realized-through-receiver `Δd_seg`;
3. realized-through-receiver `Δd_pose`;
4. a registered repeat-noise comparison for both scorer terms; and
5. a joint-candidate check when multiple alternatives are composed.

A position is `LOAD_BEARING` only when the same evidence shows every tested cheaper realization exceeds the registered scorer-equivalence tolerance. A grouped perturbation may prove that the **group configuration** affects a scorer term, but it does not prove that every member token is individually load-bearing. Arithmetic-coded byte deltas also do not partition additively by position: changing a token changes later contexts.

## Typed retained-receipt census

| Receipt family | Axis | Scope and denominator | `Δbytes` | `Δd_seg` | `Δd_pose` | Repeat noise | Admissible conclusion |
|---|---|---|---|---|---|---|---|
| JG3 S2 joint solve | `[macOS-CPU advisory]` | 573 / 573 retained pair rows; 11,654 separation configs; 10,900 accepted edits; 15,155 repaired cells | absent in 573 / 573 rows | group repair fields present | absent in 573 / 573 rows | absent in 573 / 573 rows | Pair-configuration Seg sensitivity only. The required DROP arm was not implemented. **0 individual tokens classified.** |
| JG5 keep/drop waterfill | rate `[macOS-CPU scorer-free exact, pre-FX5]`; Pose `[macOS-CPU advisory, DALI GT, batch 8]`; Seg JG1-DALI projected to T4 | 573 edited pairs, 455 kept; one 8,654-token grouped field | real pre-FX5 grouped encode: `+4,151 B` vs zero-edit base | grouped T4 projection present | grouped fixed-batch advisory present | no per-token scorer repeat receipt | The chosen 455-pair configuration is scorer-sensitive and costs bytes on its coder. It does not classify the 8,654 members, and its byte delta is not current-FX5 attribution. |
| FS2 threshold substitution | rate `[macOS-CPU scorer-free exact, pre-FX5]`; distortion `[macOS-CPU advisory]` | `u=7.75`: 9,106 substitutions; `u=12`: 1,440 substitutions | real pre-FX5 grouped deltas: `-1,022 B`, `+37 B` | transferred from RC4 n120 amplification, not measured on the FS2 field | absent | absent | Real grouped prices, incomplete evaluator tuple. **0 individual tokens classified.** |
| FS3 reopen | rate `[macOS-CPU scorer-free exact, pre-FX5]`; Seg `[macOS-CPU advisory, jg3 DALI instrument]` | 300 marginal tokens | `+223 B` | `-116` Seg cells | unresolved | absent | Incomplete tuple; no individual classification. |
| FS3 drop137 | rate `[macOS-CPU scorer-free exact, pre-FX5]`; distortion `[cpu_env_mismatch_advisory]` | 997 dropped edit tokens; one grouped field | `-664 B` vs the shipped 455-edit stream on the pre-FX5 coder | same-instrument `Δd_seg = +0.00000333` | same-instrument `Δd_pose = +0.00040424` | no receipt-owned repeat floor | The grouped drop is evaluator-visible and dominated: rate `-0.00044213 S`, distortion `+0.03623733 S`, net `+0.03579520 S`. It does not identify which of 997 tokens carry the loss. |
| RC4 rung 4 | distortion `[macOS-CPU advisory]`; rate `[scorer-free exact first-order model, old HV1]` | old HV1; n120 Seg and n48 Pose; thresholds 5, 7, 8.5 | first-order model, not a real current re-encode | measured only on old vehicle/sample | measured at u=7 on n48 | no current-field repeat floor | Useful negative transfer warning, not a current census. |
| TD1 Schur arithmetic | `[scorer-free derived, old HV1]` | old HV1 full field, 117,964,800 positions | exact first-order model only | scorer amplification unmeasured | unmeasured | absent | No current evaluator-equivalence classification. |
| task 869 / HV2 “768×4” | `[scorer-free exact-key prep, old IX2/HV2]` | 384 live cells used to prepare four exact-keyed orders | old IX2/HV2 preparation only | scorer A/B explicitly pending | scorer A/B explicitly pending | absent | This is exact-key preparation, **not** a completed 768-cell × 4-rung scorer experiment. |
| DX2/FX5 | score `[contest-CUDA T4, n600]`; bytes/decode `[scorer-free exact]` | full current archive | DX2 carrier fold `-18 B`; FX5 global token-coder/corrector `-70 B` | decode-identical to rc2/JG5 field | decode-identical to rc2/JG5 field | exact byte/decode identities recorded | Scorer fields transfer through decode identity; old coder byte attributions do not transfer to the current FX5 stream. |

The FS3 same-instrument row is the strongest complete **group-level** evidence. It closes “drop those 997 edits as a cheap win,” but it cannot be expanded into 997 per-token LOAD_BEARING rows. Conversely, the absence of a per-token complete row cannot be expanded into inertness.

Durable retained sources used for the row audit include:

- JG3: `/Volumes/APDataStore/pact/ddm_jg3/checkpoints/seg_solve_n600_complete.jsonl` and `/Volumes/APDataStore/pact/ddm_jg3/retained/wc2_merged_n600_complete.json`;
- JG5: `/Volumes/APDataStore/pact/ddm_jg5/retained/final/WATERFILL.json` and `/Volumes/APDataStore/pact/ddm_jg5/retained/final/S1_encode_jg5_subset455.json`;
- FS2: `/Volumes/APDataStore/pact/ddm_fs2/reencode/retained/S1_encode_fs2u7p75.json` and `S1_encode_fs2u12.json` in the same retained directory;
- FS3: `/Volumes/APDataStore/pact/ddm_fs3/FS3_SAME_INSTRUMENT_LEGS.json` and `/Volumes/APDataStore/pact/ddm_fs3/reencode/retained/S1_encode_fs3_drop137.json`; and
- HV2/task 869: `/Volumes/VertigoDataTier/pact/ddm_hv2_20260803/ddm_hv2_exact_keyed_orders_receipt.json`.

## Evaluator-visible MDL

Let a current token position’s evaluator-equivalence class contain concrete token alternatives whose fully received render changes both scorer terms by no more than the registered repeat floor. Let its realization cost include the actual FX5 arithmetic context and every counted side section needed to select the alternative.

The retained evidence yields:

- observed, qualified equivalence-class alternatives: **0**;
- qualified `INERT` positions: **0 / 117,964,800**;
- qualified inert coded mass: **0 / 113,777 B**;
- receiver-realized overpayment lower bound: **0 B**;
- epistemically unclassified span: **113,777 B / 113,777 B**.

The unclassified span is not an achievable upper bound. It is merely the mass whose scorer-visible quotient is unknown. No source in the retained corpus proves a receiver-closed alternative token field with lower current-FX5 codelength and unchanged Seg and Pose.

The canonical equations agree with this stop:

- `argmax_cell_identity_ideal_bytes_v1` is an ideal known-site bound; it excludes site transport, receiver realization, and other counted structure.
- `ddm_score_quotient_functional_v1` remains incomplete and has no receiver-closed empirical anchor.
- `token_rate_model_direction_dependence_v1` expressly forbids substituting its local/first-order price for real re-encoding and records arithmetic-context leakage.

## Realization category audit

No qualified inert class exists yet, so no category can receive byte credit.

| Candidate realization | Category if it worked | Current disposition |
|---|---|---|
| Derive an evaluator-equivalent token choice entirely from generic receiver logic | (a) zero-rate derivable | **UNMEASURED.** No current receiver-closed equivalence rule is demonstrated. Video-derived choices may not be hidden in free code. |
| Revert selected JG5 edits to a cheaper counted base or quotient representation | (b) cheaper counted stand-in | **UNREACHABLE on DX2 as presently specified.** FS3 shows one large revert is scorer-visible and dominated; no current stand-in is shipped. |
| Substitute model argmax at FS2 thresholds | (a) logic may be free; token field remains counted | **NOT EQUIVALENT on retained evidence.** The real price at `u=7.75` is promising, but the scorer tuple is incomplete and its Seg transfer is adverse. |
| task 869 exact-key map | (b) cheaper counted stand-in | **UNMEASURED and ancestor-bound.** The prepared orders are not current scorer-equivalent cells. |
| NR1-style task-cell quotient | (b) cheaper counted stand-in | **UNMEASURED.** NR1 contains no measured quotient bytes and has its own current-field falsifier; VF1 does not reopen or build it. |
| Delete tokens without a receiver replacement | (c) unreachable | **DEAD.** Tokens are required by the current decoder, and grouped drop evidence incurs scorer loss. |

## Prior verdict

The prior prediction was that measured load-bearing mass would be a minority and realizable inert mass would be on the order of `42,382 B`. Its pre-registered falsifiers—load-bearing mass at least 90%, or realizable inert mass below roughly 10 kB—presume a qualifying census.

Here the qualifying measured denominator is zero. Therefore:

- the prediction is **INCONCLUSIVE**, not supported;
- the prediction is **not falsified** by the `0 B` credited lower bound;
- no “inert mass” number may be inferred from unmeasured positions; and
- the canonical frontier remains unchanged.

## The one owed scorer measurement — do not fire from VF1

**Name:** `DX2_TOKEN_PPS_EQUIV_V1`

**Disposition:** `OWED — MAIN-ONLY FIRE AFTER SCORER-LANE CLAIM`

**Owner:** `MAIN`

**Consumer store:** `/Volumes/APDataStore/pact/ddm_vf1_evaluator_visible_floor/pps_equiv_v1/`

**Fire trigger:** MAIN has claimed a non-duplicated scorer lane; pinned the exact DX2 archive, decoded token field, FX5 coder, receiver, and scorer hashes; and approved a deterministic, crash-resumable, per-chunk-checkpointed harness after the storage-waterfall preflight.

Exact configuration:

1. On the current DX2 token field, compute exact sequential FX5 codelengths without scoring.
2. Draw a deterministic PPS sample of 768 token positions with seed `20260822`, stratified by pair, current class, and codelength decile.
3. At each sampled position, test the four non-current class values one at a time through the exact receiver, frozen SegNet, and frozen PoseNet at a fixed batch shape: **768 × 4 = 3,072 singleton variants**.
4. Repeat the unchanged baseline twice for every pair to register the scorer noise surface: **2 × 600 = 1,200 pair evaluations**.
5. Choose only alternatives whose two scorer deltas fall within their registered repeat floors, compose one joint alternative field, then evaluate all 600 pairs once to detect collateral: **600 pair evaluations**.
6. Checkpoint atomically after each at-most-120-pair chunk so the experiment resumes from disk without repeating completed scorer work.
7. Real-encode the control and joint field with the current FX5 coder, parse them back, and retain every materialized token payload, rendered payload, manifest, SHA-256, and byte count.

Total planned scorer cost: **4,872 pair evaluations**, chunked at most 120 pairs at a time, plus **two** real FX5 re-encodes. Planned cash cost: **$0 local**. Wall time is unmeasured and is not guessed here.

Use PPS design weights to report classified token mass and coded mass with confidence intervals. The quotient thesis is falsified for this sampled current-DX2 formulation if the design-weighted LOAD_BEARING share has a 95% confidence lower bound of at least 90%, or if the receiver-realized inert-credit 95% upper bound is below 10,000 B. It is supported only if the load-bearing point estimate is below 50%, the realizable inert lower bound approaches the 42,382 B demand, and the jointly composed candidate survives both scorers and real re-encoding.

## Recall evidence and scope

The bounded recall covered the research corpus, current DAG/task surfaces, task-status store, harness bridge, retained stores, and canonical-equation registry using `evaluator-equivalence`, `score quotient`, `token-by-token`, `768 cells`, `4 rungs`, `three-way edit drop keep`, `jg3`, `jg5`, `fs2`, `fs3`, `rc4`, `rung-4`, `repeat-noise`, and `sensitivity`.

It found material evidence beyond the charter seeds:

- task 869 is still pending; HV2 prepared four exact-keyed orders from 384 cells but did not run the scorer A/B;
- JG3 implemented EDIT and KEEP but not DROP;
- JG5 is a pair-level waterfill and grouped field, not per-token marginal evidence;
- FS3 closes one grouped 997-token drop while leaving individual attribution unresolved;
- DX2/FX5 preserve the decoded token field but change global coding, so ancestor byte deltas are not current token prices; and
- the registered ideal-byte and quotient equations explicitly omit receiver-closed realization or remain incomplete.

These findings changed the plan from aggregating an assumed three-way/768×4 census to this typed measured-subset audit. Within the stated scope, I did not find a current-DX2 qualifying per-token complete row.

## Constraints and custody

- No scorer or model forward was launched.
- No Metal, MPS, CUDA, Modal, or paid dispatch was launched.
- No training, archive mutation, or payload materialization was launched.
- JO r9 was not read or used.
- No unrelated working-tree or staged-index content was changed.
- The current canonical frontier is unchanged: DX2 remains `S = 0.14821987563243377`, above the sub-0.12 goal.

## NEXT_IF_RESUMED

- **Disposition:** `OWED — DO NOT FIRE WITHOUT MAIN`; **owner:** `MAIN`; **consumer store:** `/Volumes/APDataStore/pact/ddm_vf1_evaluator_visible_floor/pps_equiv_v1/`; **fire trigger:** a non-duplicated scorer lane is claimed, all exact DX2/FX5/receiver/scorer hashes are pinned, and a deterministic resumable harness passes the storage preflight; **action:** execute `DX2_TOKEN_PPS_EQUIV_V1` exactly as specified above, retaining every payload and the two real re-encodes.

## LIVE-HYPOTHESES

- A useful evaluator quotient may still exist because the 113,777-byte token field represents much more spatial state than the two frozen scorer outputs observe; the retained corpus simply has no current per-position experiment capable of measuring it.
- Scorer-equivalent alternatives may be concentrated in high-predictability, non-boundary token contexts. That is plausible because JG3’s successful edits were sparse and scorer-directed, but only codelength-weighted current-DX2 sampling can determine whether those alternatives carry meaningful bytes.
- Realization cost may be the binding wall even if many positions are evaluator-inert. FX5 arithmetic contexts couple later prices to each substitution, and FS2 already shows that changing many tokens can either save or add bytes.

## DEAD-ENDS

- Treating task 869/HV2 preparation as a completed 768×4 scorer census is closed: the receipts explicitly say scorer A/B is pending.
- Treating JG3 as a completed EDIT/DROP/KEEP experiment is closed: DROP was not implemented, and its rows contain no Pose or byte delta.
- Assigning a grouped JG5, FS2, or FS3 result to every member token is closed: the retained receipts measure whole configurations, while individual necessity and arithmetic-context price are non-additive.
- Summing JG5, FS2, and FS3 byte deltas into “inert mass” is closed: their changed sets overlap, their baselines differ, and their scorer tuples are incomplete or adverse.
- Transferring old IX2/HV1 or pre-FX5 byte prices to current DX2 is closed: DX2/FX5 preserve decoded content but changed the global coder representation.
- Re-encoding the fixed current token field as a path to the missing 42,382 B is closed by RB1: exact recoding has already been exhausted by the 70-byte FX5 improvement and cannot supply the remaining order of magnitude.

**VF1 frontier line:** exact strict-byte demand **42,382 B**; measured evaluator-inert credit **0 B**; qualifying current-token census **0 / 117,964,800 positions**; disposition **INCONCLUSIVE — one MAIN-owned scorer measurement owed**; own-vehicle frontier remains **S = 0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]`**, pointer unmoved.
