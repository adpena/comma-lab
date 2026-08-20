# ddm_pq6 — EVIDENCE: every number and law in SECTIONS.md, with its receipt

`date_utc: 2026-08-20` · `owner: ddm_pq6` · `score_claim: false` · `frontier_moved: false`

**Rule applied.** A sentence in `SECTIONS.md` that carries a number or asserts a law has a row
here. Rows are marked **SOURCE** (I read the measuring artifact), **RELAYED** (I read an artifact
that quotes the measuring artifact, and did not open the measurer), or **DERIVED** (arithmetic I
did here on receipted inputs). Anything I could not receipt is in §X and is **cut from
SECTIONS.md** or flagged for MAIN.

**Axis vocabulary.** `[contest-CUDA T4, n600]` is the only score. `[macOS-CPU advisory]` is a
frozen-scorer local measurement and is never a score. `[exact byte]` is a `stat` on real bytes.

---

## §1 — The shipped row (every section is written against this)

| Claim in SECTIONS | Value | Axis | Receipt | Grade |
|---|---|---|---|---|
| Score | 0.14839100138338618 | `[contest-CUDA T4, n600]` | `.omx/research/ddm_pq1_submission_packet_prep_20260815/REPORT_PUBLIC.txt` "Recomputed score" | SOURCE |
| Archive bytes / sha | 180,625 B / `f3bce5d259a081839c48d8089c2b43a57cc7cc96cf5b8f787ff85089be8acb7e` | `[exact byte]` | same, "Exact result identity" | SOURCE |
| seg leg | 0.020139 (`d_seg` 0.00020139) | `[contest-CUDA T4, n600]` | same, "Evaluation results over 600 samples" | SOURCE |
| pose leg | 0.007981227975693965 (`d_pose` 0.00000637) | same | same | SOURCE |
| rate leg | 0.1202707734076922 | same | same | SOURCE |
| Leg shares 13.57 / 5.38 / 81.05 % | — | — | leg ÷ score, this arm | DERIVED |
| Inflation wall time 1,419.9042126240001 s | — | `[contest-CUDA T4]` | same, budget line | SOURCE |

---

## §2 — §A "Where this approach has headroom"

| Claim | Value | Receipt | Grade |
|---|---|---|---|
| Per-byte exchange rate | `25 / 37,545,489 = 6.658590e-07` S/B | `.omx/research/ddm_tx1_toolbox_crosswalk_20260819.md` §0; formula `upstream/evaluate.py:92`; helper `src/tac/contest_score.py` | SOURCE |
| Section budget of the prior body | tail 109,696 B (62.2%) · models 66,528 B (37.7%) · table 96 B · ZIP 100 B; total 176,420 B | tx1 §0, parsed **this turn** through the candidate's own receiver, not quoted from a memo | SOURCE |
| Bits per token | 0.0074392 over 117,964,800 tokens | tx1 §0 | SOURCE |
| **Scope caveat carried in the text** | budget is the **176,420 B** body, not the shipped 180,625 B | tx1 base table | SOURCE |
| Model split (uncompressed) | semantic 36,130 · carrier 22,246 · hpac 17,952 | tx1 §0 | SOURCE |
| Mixer win | −560.07 B, ΔS −3.72881e-4, byte-closed 180,601 B | `.omx/research/ddm_fx1_fixed_point_logistic_mixer_20260817.md` §9.1 | SOURCE |
| Context widening win | −710.84 B → 180,450 B | pq4 `CONTRIBUTIONS_INVENTORY.md` A9 citing `.omx/research/ddm_fx2_model_axis_all_sections_20260818.md` | RELAYED |
| Within-miss corrector win | −104.584 B (109,800.4393 → 109,695.8553 B), ΔS −6.9915e-05 | `.omx/research/ddm_ma1_model_axis_miss_cost_20260819.md` §Conclusion | SOURCE |
| Reservoir framing **withdrawn** | 77,241 B is an entropy, not a reservoir; honest remainder ≈ 400 B (hit event) + ≈ 180 B (within-miss, 105 taken) | ma1 §Conclusion table | SOURCE |
| "a few hundred bytes ≈ −4e-04 S" | 600 B × 6.658590e-07 = 3.995e-04 | this arm | DERIVED |
| Round-trip loss is one pixel wide | 99.22% of flips on the transmitted boundary; interior 88.4% of field carries **7 flips in 104 M px**; boundary rate 203,000× interior | `.omx/research/ddm_rt1_seg_roundtrip_decomposition_20260816.md` ANSWER FIRST items 1–2 | SOURCE |
| Correct class one pixel away | 93.9% | rt1 item 3 | SOURCE |
| The residual is a tie | 98.3% of flips have the wanted class as runner-up; median logit deficit 0.105; 84.5% under 0.3; none over 3; correct pixels sit at 3–10 | rt1 item 6 | SOURCE |
| Edge concentration | Road↔Lane 43.4% (0.0129 S); three edges 80.4% | rt1 item 4 | SOURCE |
| Round trip is 96.6% of that vehicle's seg axis | 33,743 flips = 0.028604 S | rt1 item 1 | SOURCE |
| **Vehicle caveat carried in the text** | rt1 measured on `hv1 ep0634`, S 0.15959729295498598 @ 182,759 B — a different body from the shipped one; SECTIONS says the shape transfers, not the magnitude | rt1 frontmatter `own_vehicle_frontier` | SOURCE |
| 95.9% of the seg leg is render/re-segment loss | — | `.omx/research/ddm_na10_negative_audit_fresh_laws_20260819.md` §0 law **L4**, sourced to `ddm_jg1` §S1a whole-field array comparison, both lineages | SOURCE (na10) / RELAYED (jg1) |
| Labels 99.9985% correct | 1,714 wrong of 117,964,800 | na10 L4 | SOURCE |
| *Sister figure, reconciled* | an earlier arm put the same quantity at "~95%" with 1,717 label errors and 33,213.6 flips | `.omx/research/ddm_td1_token_drop_schur_arithmetic_20260816.md` headline 1 | SOURCE |
| Carrier is at a fixed point | 600/600 stopped `no_improving_step`; 0 `gn_iteration_budget`; 0 `outer_round_budget`; 0 `converged_below_materiality_floor` | `.omx/research/ddm_jg5_pose_resolve_on_edited_renders_20260819.md` §6.1 | SOURCE |
| Admission left 118 pairs | 455 of 573 admitted | jg5 §5 | SOURCE |
| Floor is an ESTIMATE band 0.07–0.13 | no nontrivial proven floor; 0.11797 is the old carrier's own iid coding floor | memory `theoretical_floor_is_below_the_goal_20260817`, re-derived from `.omx/research/information_theoretic_floor_T_floor_20260610.md` | SOURCE (memory) |
| 27,620 B to reach 0.13; 42,638 B to reach 0.12 | on rate alone from the shipped row | this arm | DERIVED |

---

## §3 — §B "Directions"

| Claim | Value | Receipt | Grade |
|---|---|---|---|
| `drop` branch priced and **not shipped** | −0.002929 S = 44.9% of the gap it was measured against | pq4 `CONTRIBUTIONS_INVENTORY.md` A15, citing `.omx/research/ddm_jg3_s2_joint_solve_20260819.md` OPTIMAL FORM table row 1 | RELAYED |
| Reason it is blocked | drop is a **receiver** change; the pointer body's receiver has no such path; implementing it invalidates the byte-identity control chain the seal rests on | pq4 A15 + `MAIN_HANDOFF.md` §5 premise 4 | SOURCE |
| Token-drop rung rate+seg leg | −3.243e-3 S, rate leg **exact** | na10 §5.1, citing `.omx/research/ddm_rc4_rung4_token_drop_verdict_20260816.md` | SOURCE (na10) |
| Its stated door | compensation must cancel **99.807%** | na10 §5.1 verbatim block quote of rc4 | SOURCE |
| Measured cancellation | 99.9874% aggregate; per pair 99.9988% / 99.6705% / 100.0562% | na10 §5.1 table, sourced to `ddm_jg1` §S2b | SOURCE (na10) |
| Amplitude match | rc4 damage 3.3279e-03; jg1 mean per-pair damage 3.3366e-03; ratio 1.00× | na10 §5.1 | SOURCE |
| Honest limits on it | n=3; pair 468 alone below bar; two different bodies (182,759 B vs 176,420 B); re-coding cost unmeasured; 39–64 s/pair ⇒ 6.5–10.7 h at n600 | na10 §5.1 items 1–4 | SOURCE |
| Settling measurement + cost | ≥60 seeded-random pairs, ratio-of-sums; $0, ~40–65 min at n=60 | na10 §5.1 "Resolving measurement" | SOURCE |
| Renderer fidelity | disagrees with its own token plane at `d_seg = 0.00029639352578669786` = 99.9703606474% | `.omx/research/ddm_hr1_realization_engineering_20260811.md` Outcome | SOURCE |
| The four-arm race is specified and unrun | frozen decode / full fine-tune / counted low-rank adapters / joint token+renderer descent, one common receiver, all learned weights counted | hr1 "The full four-arm race" | SOURCE |
| Correction channel coder gate | passes at **32,270** real verified bytes vs a 35,117 B bar | rt1 frontmatter + §5 | SOURCE |
| Correction channel realization bar | η = 0.6235 at n=9 seeded-random pairs, **0 of 9** above the required 0.753; seg+rate +0.00381 S; pose −0.00129 S; total **+0.00252 S** | rt1 §6 / ANSWER FIRST | SOURCE |
| Flat-anchor repaint | +1.3808 S, 47.6× worse; worse than repainting the whole frame flat | rt1 item 7 | SOURCE |
| Wavefront group map | `group(x,y) = (x & 63) + 2·(y & 63)`; UP-RIGHT causality **98.6945%** | pq4 A9 citing `ddm_fx2` | RELAYED |
| SSE/APM stage measured dead | loses in **6 of 6** formulations | pq4 A9 | RELAYED |
| Ranker cannot accept | ranker **orders and truncates**, never accepts; adversarial ranker that puts the worst candidate first still reaches the exact optimum (executed negative control) | `.omx/research/ddm_cw1_win_family_canonicalization_20260819.md` §5 | SOURCE |
| Truncation weakens the proof | with `top_k` set, "no improving neighbour" becomes "no improving neighbour among the k kept"; `RankerConfig.convergence_proof_weakened` carries that downstream | cw1 §5 | SOURCE |
| Corpus emitter built, model unfitted | every accept **and reject** is an `AcceptanceEvent`; `LearnedRanker` refuses to construct without a model | cw1 §5 + §9 limit 3 | SOURCE |
| Decode cost | token decode **1,341.5 s of the 1,419.9 s inflation = 94.5%** | `REPORT_PUBLIC.txt` (total) + `PR_BODY_DRAFT.md` as corrected by pq5 commit `4226017206`. **CORRECTION TAKEN:** my first draft carried 95.72%, which is the share against the **1,401.58 s instrumented-stage sum**, not against inflation. pq5 fixed this in the packet on 2026-08-20 and I have matched it. | SOURCE |
| Split-native speedup | 1.774×–1.834× on the token stage `[macOS-CPU advisory]`; derived PASS bar **1.804×**, straddled, graded undecidable locally; full-field **bit-exact** vs the contest receipt | pq4 L1 citing `.omx/research/ddm_wc2c_native_split_identity_and_speedup_20260820.md` | RELAYED |
| Missing lineage assertion | a content-addressed GT registry exists; nothing consults it at load; **five of ten** reopened rows exist only because of that | na10 §8 | SOURCE |

---

## §3b — §B "Directions we measured and closed" (added after the depth calibration)

| Claim | Value | Receipt | Grade |
|---|---|---|---|
| Four mixture architectures lost | best **+359 B** | pq4 A14 citing `ddm_me1` | RELAYED |
| Mixing the prior's own log-odds lost twice | **+552.32 B** and **+253.28 B** | fx1 §4 race table rows `base_k1`, `base_k1_ctxbnd` | SOURCE |
| Its mechanism | prior's odds span ~1e-9 to 1e9; any exponent ≠ 1 moves the confident tail, and 70% of bits live there. fx1 labels this **INFERRED, not measured** — SECTIONS says so | fx1 §4 bullet on `base_odds` | SOURCE (as an inference) |
| Learning rate 2^0 diverges | **+495.88 B**; 2^-2 and 2^-4 within **3 B** of each other | fx1 §4 race table + bullet | SOURCE |
| Secondary adaptive output map dead | loses in **6 of 6** formulations | pq4 A9 citing `ddm_fx2` | RELAYED |
| Correction-channel coder gate and realization bar | 32,270 B vs 35,117 B bar; η 0.6235 at n=9, 0 of 9 above 0.753; net **+0.00252 S** | rt1 §5, §6 | SOURCE |
| Why a coder cannot rescue it | flips are isolated single pixels, **mean run 1.110**; best real coder beats i.i.d. by **2.5%**; ceiling of all free conditioning **12.2%**. This **refuted rt1's own §3.3 guess** | rt1 §5 | SOURCE |
| Flat repaint mechanism | `+1.3808 S`; worse than repainting the whole frame flat because a local flat patch manufactures an edge the scorer reads as real | rt1 item 7 | SOURCE |
| Basis re-orientation is a measured NULL | invariant to **1.9e-08** (machine precision, 24 random pairs) | pq4 P3 citing `ddm_br1` | RELAYED |
| The real reason ±2 could not reach | residual demands a **multi-coordinate** step of **57 to 14,079** int12 code units; ±2 single-coordinate search cannot travel that far | pq4 P3 + O14 | RELAYED |
| The replacement | damped Gauss-Newton on the same basis, lattice and bytes: `d_pose` → **6.993157e-06** at **+9 B**; 204/600 improved, **0 worsened**; net ΔS −3.774950e-04 | pq4 O14 citing `ddm_br1` | RELAYED |
| Serialization split dropped | was −515/−520 B; the row-prune changed the semantic body length, so re-measured on the edited body the split is **negative**; `reserved = 0`, `semantic_split = false`; support ships inert | pq4 P4 | RELAYED |
| Rider re-measured lower and then declined | **183 B = −1.2185e-4 S** against a budgeted −1.85e-4 (**66%**); declined because folding a lossless change still moves archive bytes, making the score DERIVED | pq4 P6; jg5 §7 | RELAYED (number) / SOURCE (decline reason) |
| Token-search headroom priced out | under-convergence **3.238e-04 S**, basin-trapping **6.104e-05 S**, total ≈ **0.06% of gap**, so a lever must clear ~**1000×**; exact `L∞ ≤ 2` joint optimum beats coordinate greedy on **5 of 7** in-box instances; greedy escaped the box on **9 of 16** | `.omx/state/canonical_task_status.jsonl` row `sm1_seg_search_headroom_threshold`, sourced to `.omx/research/ddm_sm1_seg_search_transfer_20260803.md`. **INSTANCE-scoped** to the TR1 base named in that row | SOURCE |
| Deep prune closed pending a probe, not refuted | rate −2,051 B measured; pose leg **unmeasured**, slope from **n=2** | ledger `qw1_mp2_deep_prune_...` | SOURCE |
| Distillation reversal | 0.0050507 → 0.0054967 at `+1.37e-5`/epoch vs control 0.0051147 at `−6.80e-6`; deficit 12.8× the 2.99e-5 noise floor; FORMULATION-scoped | hr1 "DW1 and QA75/KD-#74" | RELAYED |

---

## §4 — §C "Canonicalized move classes"

| Claim | Value | Receipt | Grade |
|---|---|---|---|
| Five families, named and modularised | F1 realized-acceptance descent · F2 terminal joint compile · F3 container/re-encode · F4 model-axis recoder · F5 local authority instruments | cw1 §1 table + §3 table | SOURCE |
| Family definitions quoted in SECTIONS | verbatim mechanism column | cw1 §1 | SOURCE |
| Test count | **239 tests, all passing** | cw1 §3 | SOURCE |
| F1 measured worth | `d_pose` 7.769484e-06 → 7.649247e-06 at **ΔB = 0**; 429 improved / **0** worsened | cw1 §1; pq4 O12 citing `.omx/research/ddm_up2_shipping_object_pose_solve_20260819.md` | SOURCE |
| F3 measured worth | −657 B + −105 B = −762 B = **−5.074e-04 S** at zero distortion | cw1 §1 | SOURCE |
| F4 measured worth | −560 B; −104.584 B | cw1 §1; fx1 §9.1; ma1 | SOURCE |
| F5 gate fidelity | pose gate **0.99993×** of T4; seg gate **0.99995×** | cw1 §1 | SOURCE |
| Realized acceptance registered as law | `cw1_realized_acceptance_monotonicity_v1`, 2 anchors, **max residual 0.0** | cw1 §6 table; `.omx/state/canonical_equations_registry.jsonl` | SOURCE |
| Micro-edit engine's four architectures lost | best **+359 B** | pq4 A14 citing `.omx/research/ddm_me1_micro_edit_engine_20260817.md` | RELAYED |
| The arithmetic-mean theorem | `min_k m_k ≤ (Σ w_k m_k)/(Σ w_k) ≤ max_k m_k`; an average cannot beat its best member | `experiments/ddm_fx1_logistic_mixer_corrector.py` module docstring, **read at source this turn**; fx1 §1 | SOURCE |
| me1's rows really are arithmetic-mean | `experiments/ddm_me1_mixed_context_corrector.py:237-252`, `odds_multiplier = numerator/denominator`, docstring "COUNT-WEIGHTED ARITHMETIC MEAN IN ODDS SPACE"; AST grep for `log\|exp\|pow\|**\|sqrt` returns only imports and prose | fx1 §1, which states it verified this **at source, not relayed** | SOURCE (fx1) |
| **Log-odds mixing IS a weighted geometric mean** | `m_mix = Π_k m_k ** w_k`, weights need not sum to 1 | **VERIFIED AT SOURCE BY THIS ARM**: `experiments/ddm_fx1_logistic_mixer_corrector.py` docstring derives it explicitly ("That is a weighted GEOMETRIC mean of the odds multipliers"); fx1 §2 | SOURCE |
| The radical construction | `m**(W/2^b) = m**k · Π_{i∈bits(j)} m**(1/2^(i+1))`, `W = k·2^b + j` | same docstring; fx1 §2 | SOURCE |
| Why radicals and not log/exp | IEEE-754 **requires** correctly-rounded `sqrt`, not `log`/`exp`; one ULP at p≈0.5 moves an RC64 integer frequency by 128 counts and desynchronises the decoder | same docstring; fx1 §2 | SOURCE |
| The prior desync it cures | a device-scoped decoder desync scored **S = 27.83** | fx1 §2; pq4 O16 ("rr2's S 27.83") | SOURCE |
| No transcendental on the decision path | asserted by a test that walks the module AST | fx1 §2 + Artifacts (107 tests; 3 injected mutations all caught) | SOURCE |
| Nesting controls | weight exactly 1.0 → multiplier returned **bit-identically**, mixer collapses onto the shipped law at **0.000000 bits**; all-zero weights → `m = 1.0` exactly | fx1 §2 + §3 control table row 4 | SOURCE |
| The sign flip | `temporal_spatial` **+359.47 B** under arithmetic mean → **saves 36 B** under geometric; `surprise_only` **+689.11 B** → saves **22.6 B** | fx1 §4 "The charter's core claim, confirmed by a sign flip" | SOURCE |
| Learned weights indict the shipped law | optimum ranges **0.23 → 1.87** across cells; implicit weight of exactly 1.0 wrong nearly everywhere | fx1 §4 "What the learned weights actually say" | SOURCE |
| Race outcome | **30 of 33** rows negative or exactly zero; 3 refused | fx1 §4 | SOURCE |
| Where the bits are | 223,694 of 117,964,800 positions = **0.190%** carry **70.01%** of the stream; `q` governs 77,241 B = **98.4%** of miss cost, **69%** of the whole stream | fx1 §5 table | SOURCE |
| Bit share is prior-invariant, fraction is not | share **68.92–76.56%** across four prior families while the miss fraction spans **0.190% → 50.48% (266×)** | memory `miss_bit_share_is_prior_invariant_and_shared_payloads_are_one_sample_20260818`, sourced to `.omx/research/ddm_pd1_pr_archive_decomposition_priors_20260818.md` (60 competitor archives) | SOURCE (memory) |
| Within-miss realized share | 104.584 B = **58%** of the best hindsight bound (~180 B) | ma1 §Conclusion table | SOURCE |
| Re-encoder is the exact inverse | decode replaced by encode, importing model / 190-group plan / boundary map / fixed table / corrector from the shipped runtime; identity control **byte-identical at 109,696 B** | pq4 O4 citing `.omx/research/ddm_jg2_sub015_chain_20260819.md` §S1b–S1i | RELAYED |
| Modelling the rate cost was a real error | modelled +4.718 bits/token; measured +30 B at 4.1379 bits/token; headline moved from −0.0104 to an honest gap of 0.006526 | cw1 §4.2 | SOURCE |
| Shipped edit cost | **3.8373 measured bits per changed token**, 8,654 tokens changed, `delta_trustworthy=true` | jg5 §6.2 | SOURCE |
| Seal contract contents | archive/member/runtime FILES digest, per-file receiver pins, `admit_bar` **with derivation**, axis, retained payloads, pre-registered falsifiers, `seal_sha256`; validator re-derives every pin at consumption; firer refuses on drift; sealed values removed from the command line | pq4 O16 citing `.omx/research/ddm_seal1_candidate_seal_contract_20260818.md` + `src/tac/candidate_seal.py` | RELAYED |
| Seal control count | **36 new + 49 sister controls green; 9 typed verdicts**; packet seal `96e9860a…` `SEAL_VALID` at fire time | pq4 O16 | RELAYED |

---

## §5 — §D "Repair and compensation"

| Claim | Value | Receipt | Grade |
|---|---|---|---|
| Edits bought seg, cost pose | seg **−0.012847 S**, pose **+0.172 S**, a **13.4×** loss | jg5 §1; pq4 O3 | SOURCE |
| 571 of 573 edited pairs worse on pose | 2 better | jg5 §1 / §2 | SOURCE |
| Composed candidate was not sub-0.15 | `d_pose` 0.0032680447584351262 = **467.3×** the pointer's 6.99315662169577e-06; composed S **0.3192** | jg5 §1 | SOURCE |
| Instrument control that makes it mean something | swapping only ODD frames back to base reproduces the banked `d_pose` to 6 sig figs (2.2e-11 abs) — so the 467× is the token edit's effect, not a lineage or decode change | jg5 §2 | SOURCE |
| Seg edits multiply `d_pose` ×387 | through the photometric frame | na10 §0 law **L2**, sourced to `ddm_jg1` ANSWER-5/6 §S2 | SOURCE (na10) |
| Advisory subset: seg-only descent blew pose up | `d_pose` 0.000801428562340 → **0.033106106524428** (41.3×), repaired to 0.000791809037082 — **below** the same-row baseline | `.omx/research/ddm_od3_20260805/OD3_TERMINALITY_RECEIPT.md` pose table | SOURCE |
| Sister run, same shape | 0.0008014285623403339 → 0.0058411338650330435 → 0.0007588698333620414 | `.omx/research/ddm_od2_20260805/OD2_STAGE12_RECEIPT.md` | SOURCE |
| **Scope caveat carried in the text** | n32, `[macOS-CPU frozen-scorer advisory]`, and the subset is **pose-EASY at 0.42628664334579025× population** | OD2 receipt "Answer First" + OD3 "m96 carry-forward" | SOURCE |
| Direct quantization costs ~29× on pose | compensation is part of every hard object | hr1 RECALL EVIDENCE, quoting `.omx/research/ddm_pz4p_pose_gauge_preproof_20260811.md` | RELAYED |
| Prune with measured rate credit, wrong-signed pose | keep25 **−2,051 B = −1.3657e-3 S**; projected pose leg **+0.0264 S = 2.8× the gap, wrong sign**; pose leg **UNMEASURED**, slope from n=2 | `.omx/state/canonical_task_status.jsonl` row `qw1_mp2_deep_prune_...`, sourced to `.omx/research/ddm_qw1_unfired_wins_inventory_20260816.md` | SOURCE |
| Both scorers share one resize | PoseNet resizes **first**, converts to YUV6 second, to the identical SegNet target size | `upstream/modules.py:72-73` (pose) and `:109` (seg), as read by `.omx/research/ddm_us2_20260805/RECEIPT.md` ("shared resize `D` and blind geometry") and corrected by `.omx/research/ddm_pz1_pose_axis_cx1_base_20260803.md`; also restated in CLAUDE.md | SOURCE (us2) |
| Null-space membership does not survive a lattice change | measured attenuation **1.662×**; frame-0 scorer-plane delta **2.12×** the frame-1 debt | CLAUDE.md `ddm_pz1` clause; `.omx/research/ddm_pz1_*_20260803.json` receipts | RELAYED |
| **The counting argument** (why the repair exists before it is built) | pose scorer emits **6** scored numbers per pair; the carrier holds **12** free integer coefficients per pair, so the cancellation system is underdetermined by 2× | na10 §5.1, which quotes rc4's structural derivation verbatim — *"6 pose equations, 12 free coefficients per pair"* — and records that `jg1` later confirmed it | SOURCE (na10) |
| Schur compensation cancellation | **99.995054%** pose-energy cancellation; rate route turns ~7,000 B of sidecar into **41 B** | pq4 A10 citing `.omx/research/ddm_qs1_frame0_schur_coupled_solve_20260813.md` + `ddm_qs5_resolve_compensation_20260813.md` | RELAYED |
| Why the fingerprint binding is load-bearing | a Schur solve is a linearisation about **one** frame-1 token stream; moving the tokens invalidates the Jacobian it inverted | mechanism stated in `ddm_qs5` "Compensation is now bound to the compiled object"; the refusal of the earlier archive is the measured instance | SOURCE |
| Concavity ⇒ no fixed-ratio greedy | pose enters as `√(10·d_pose)` over the mean, so the marginal score cost of `d_pose` **falls** as total `d_pose` rises; an exchange rate computed at one operating point mis-prices every other one | `upstream/evaluate.py:92` score form; jg5 §5; memory `concavity_helps_when_you_pay_the_axis_upward_20260818` | DERIVED from receipted form |
| Three exact solves | 98.33% / 99.94% / 99.93% leakage-energy cancellation, `[macOS-CPU advisory frozen CPU-torch PoseNet] NON-PROMOTABLE` | `.omx/research/ddm_qs5_resolve_compensation_20260813.md` Outcome table | SOURCE |
| Compensation is content-bound and fails closed | fingerprint over pair index, exact semantic-token bytes, exact camera-uint8 master, archive identity; re-checked before the lattice moves; an earlier archive is refused because its frame-0 compensation was solved for a different frame-1 token stream | qs5 "Compensation is now bound to the compiled object" | SOURCE |
| Credit for the pattern | the edit-then-recompensate pattern is **PR #135's**, not ours; ours is the fingerprint binding, the disjointness argument, the step-matched Jacobian, the rate route | pq4 A10 | RELAYED |
| **Known defect disclosed** | the compensation implementation targets `qs1.GT_POSE`, the **PyAV** table, while the shipped object scores on the DALI/CUDA axis; any reuse must repoint it | na10 §7 item 1 (`experiments/ddm_qs5_resolve_compensation.py:183,783`); cw1 §4.2 | SOURCE |
| Carrier re-solve recovery | all 573 edits kept: `d_pose` **3.268e-3 → 4.089e-4**, an **8.0×** recovery; still 58× the pointer; S still 0.2023 | jg5 §6.1 | SOURCE |
| Three-pair recovery to 1.073× at ~0 bytes | 9–12 of 12 coefficients moved | na10 §0 L2 refinement | SOURCE |
| Recovery is bimodal | pair 0 lands **below** its own base value; pair 10 barely moves; biggest recoveries pair 240 2.5345e-03→5.4316e-06 (467×), pair 221 2.1402e-03→1.0997e-05 (195×), both flip DROP→KEEP | jg5 §3 table + §6.1 | SOURCE |
| The inherited ceiling that did not transfer | a free 2304-dof pose ceiling of **0.7347** would have predicted ≤27% recovery and closed the arm; it is a **near-floor** constant | jg5 §3 | SOURCE |
| Two states per pair, both legs measured | DROP = base frame 1 + incumbent carrier codes, 0 tokens; KEEP = edited frame 1 + re-solved carrier | jg5 §5; pq4 O1 | SOURCE |
| DROP branch is measured, not modelled | on all 8 unedited pairs checked, the odd frame is **byte-identical** between base and candidate decode | jg5 §2 | SOURCE |
| Concavity forbids a fixed-ratio greedy | pose leg is √-concave; per-pair pose costs do not add in score units; admission swept over a Lagrange multiplier, every subset scored through the exact formula | jg5 §5; memory `concavity_helps_when_you_pay_the_axis_upward_20260818` | SOURCE |
| Admission result | **455 of 573**; net **−0.00776976** vs the pointer 0.15615243; **0 counted archive bytes** for the admission | jg5 §5 + §6.2; pq4 O1 | SOURCE |
| Sweep arithmetic controls | drop-everything model 0.15614834772046404 vs pointer 0.15615242950573233; keep-all model 0.31917539282632396 vs measured 0.3191825455409289; both offsets quoted, neither folded | jg5 §5 | SOURCE |
| The inherited counter | `iterations=6`; accepted-step histogram **[98, 131, 146, 111, 63, 35, 16]**; all **16** pairs at k=6 accepted an improving step every iteration and were still improving **>2%** | jg5 §4(a) | SOURCE |
| The derived stop rule | iterate while `remaining_dd · dS/dd_i > DELTA_FLOOR_S`; `dS/dd_i = 10/(2·600·√(10·m)) = 1.043784`; `DELTA_FLOOR_S = 3.5e-6/600 = 5.833333e-09`; **threshold 5.588639e-09** `d_pose` units; `remaining_dd` from each pair's **own** decay ratio | jg5 §4 table | SOURCE |
| The rule never bound | 600/600 `no_improving_step`, 0 budget hits, 0 materiality-floor stops | jg5 §6.1; pq4 O2 honest caveat | SOURCE |
| A superseded arbitrary rule was retained, not merged | first replacement used a 0.5%/step tolerance and a hand-set budget; retained with `WHY_SUPERSEDED.txt`, merged into nothing | jg5 §4 | SOURCE |
| Rate costs ADD | union/sum **1.0258**; `10+6+14` = **measured 30 exactly**; causality control **0.000000 bits** | pq4 O5 citing `ddm_jg2` §S1h | RELAYED |
| Compensation does NOT add | joint beat naive union by **3.705×** | pq4 O6 citing `.omx/research/ddm_bu1_bank_union_compile_20260817.md`; memory `never-price-a-union-as-the-sum-of-its-legs` | RELAYED |
| Reach ≠ magnitude | a global cascade can be a CREDIT; dependency arguments bound WHERE, never HOW BIG | pq4 O7; memory `cascade_reach_is_not_cascade_magnitude_20260819` | RELAYED |
| Batch shape is part of the instrument | deterministic at fixed shape; value moves with shape — spread to **7.7e-3** (pair 299) across shapes 1/8/32; final KEEP/DROP made at one declared shape (batch 8) for both code sets | jg5 §4b table | SOURCE |

---

## §6 — §E "Realization walls, QAT, in-loop"

| Claim | Value | Receipt | Grade |
|---|---|---|---|
| The realization path | hard tokens → renderer at 384×512 float RGB → bicubic to 874×1164 camera RGB → clamp + round to uint8 → bilinear to 384×512 → frozen scorer | hr1 "Binding receiver and round-trip contract" | SOURCE |
| Three independently enforced placements | R in loop · uint8 in loop (exact forward, registered STE backward) · YUV6 in loop (no `no_grad`, no in-place clamp) | hr1, same section, items 1–3 | SOURCE |
| Validity condition | training hard-forward camera bytes must equal the public receiver's camera bytes on **real** frames; receiver ordered argmax must equal the CPU-Torch reference at the pinned batch shape | hr1, same section | SOURCE |
| Ladder result: representability ≠ survival | exact target grid `d_seg = 0.000282948812`; repaired realized output ≈ **0.0274**; one horizon realized **5.2867%** of its forecast cell gain | hr1 "V14 realization ladder", quoting `.omx/research/ddm_v14_realization_fidelity_603_DAG_FEED_20260722.md` | RELAYED |
| Realization efficiency measured | η = **0.6235**, n=9 seeded-random, 0 of 9 above the required **0.753** | rt1 §6 | SOURCE |
| The resize supplies zero flips | S3 (the R operator) supplies **exactly zero** flips on piecewise-constant content | rt1 ANSWER FIRST item 5 | SOURCE |
| Flat paint is a ceiling | S2 flat-prototype paint reads back **35.4×** worse than the trained render | rt1 item 5 | SOURCE |
| Post-hoc repaint closed | +1.3808 S, 47.6× worse; worse than repainting the whole frame flat because a local flat patch manufactures an edge the scorer believes | rt1 item 7 | SOURCE |
| QAT is inherited | the source training family used CE, softplus margin, expected-flip polishing, **quantization-aware rendering**, and the real token plane | hr1 "PR135 renderer and HPAC" | SOURCE |
| Weight-MSE is the wrong sensitivity metric | amplification 38,700× on "dead" tensors vs 2,518× on "live" (~15× spread); the ladder's **mildest** rung 90× underwater on seg; damage ∝ `weight_mse^~0.4`; contrast +2e-6 (rendered-ranked recipe, survived T4) vs +3.4e-3 (weight-MSE-ranked, same depth) | memory `weight_mse_is_the_wrong_sensitivity_metric_20260818`, sourced to `.omx/research/ddm_sa1_vseries_v7_dead_v2_routing_20260818.md` §5 | SOURCE (memory) |
| Ratcheting realized seg floor | best accepted full-n600 realized `d_seg` may not increase on the same scorer/device/batch; no smooth-loss allowance substitutes | hr1 "Seg hold, pose guard" rule 1 | SOURCE |
| Joint distortion guard before pose engages | must also improve `100·d_seg + √(10·d_pose)` vs the accepted parent | hr1, same, rule 2 | SOURCE |
| Loss-space holds are closed | one joint run let live `d_seg` worsen from ~0.00357 to ~0.00599 while pose improved | hr1 "JD-line joint descent", quoting JD1 | RELAYED |
| A short race picks the wrong objective | 12-epoch winner `d_seg = 0.0050507`; matched window reversed to **0.0054967** at slope **+1.37e-5/epoch**; control ended **0.0051147** at **−6.80e-6/epoch**; deficit 3.82e-4 = **12.8×** the measured **2.99e-5** noise floor; verdict FORMULATION-scoped | hr1 "DW1 and QA75/KD-#74" | RELAYED |
| Scalar loss parity is not parity | identical loss while **40 of 41** arrays diverged | hr1 "Binding receiver…", citing #903 | RELAYED |
| Structured prune + mixed precision | 6,713 of 7,200 carrier coordinates changed; that generation lost byte-identity to the lineage | pq4 A12 | RELAYED |

---

## §7 — §F "Upstream dynamics → order"

| Claim | Value | Receipt | Grade |
|---|---|---|---|
| Method | predict-then-diff per surface, read line by line, classify `{MATCHES-RECALL, DRIFTED, FORGOTTEN, NEW-SIGNAL}` with file:line receipts | `.omx/research/ddm_us1_upstream_reread_20260731.md` §0 | SOURCE |
| Zero critical law-basis failures | 13 core laws re-derived from the primary artifact; 2 DRIFTED, 4 FORGOTTEN, 11 NEW-SIGNAL, **0 CRITICAL** | us1 §0 + §1 | SOURCE |
| Dynamic rate denominator | numerator is `archive.zip` `stat().st_size`; denominator is a live `rglob` sum; equals 37,545,489 B today — a filesystem fact, not a law | `.omx/research/ddm_us2_20260805/RECEIPT.md` typed-findings row | SOURCE |
| Report precision | components printed at 8 dp, final at 2 dp; the unrounded internal score is unrecoverable from the report | us2 typed-findings row; `REPORT_PUBLIC.txt` carries the worst-case bound 3.63296497868841e-06 | SOURCE |
| Loader truncation | `zip(dl_gt, dl_comp)` truncates to the shorter iterator; `N = file_size // frame_bytes` | us2 typed-findings row | SOURCE |
| Scorer weights counted at decode, topology free | compress-time use is legal | us2 typed-findings row ("scorer checkpoint objects") | SOURCE |
| Evaluator split | `evaluate.py` is scoring-only; `evaluate.sh` unzips, calls `inflate.sh`, existence-checks raw output, then scores | us2 typed-findings row | SOURCE |
| Realized-acceptance beats gradient/blind-search on quantized values | fired on **two axes with two actuators** (pose coefficients; seg tokens) | na10 §0 law **L3** | SOURCE |
| Pose estimate band | median **13.4×** seg's at equal n; wider in **100.0%** of 2,000 shuffles (p10 5.9×, p90 27.9×) | na10 §0 law **L5**, sourced to `fo2h` §4.3 | SOURCE (na10) |
| Prefix bias inverts by axis | pose prefixes **2.54–4.21× harder**; seg **0.95–0.97× easier**; `select_pairs` never returns `[0..n)` | cw1 §3; CLAUDE.md `[[m96]]` | SOURCE |
| Rate was the binding leg on the token axis | seg −2.712674e-5, pose +1.126177e-7, rate +5.127114e-5 — rate **455×** pose | na10 §7 item 3, at the qs1 receipt | SOURCE |
| Compensation cost per pair fell an order of magnitude | qs1 12.83 B/pair → qs2 5.667 → the re-solve moves already-shipped coefficients the Rice stream absorbs at +5 B for all 7,200 ≈ **0.83 B/pair** | na10 §7 item 3 closing note | SOURCE |
| Container step has zero distortion **by construction** | the receiver restores each section body byte-for-byte before parsing | pq4 O8 citing `.omx/research/ddm_ck2_container_plane2_eleventh_move_20260819.md`; measured −657 B, `d_seg` 0, `d_pose` 0 | RELAYED |
| Per-candidate proof that the seg leg cannot move under a carrier splice | `mode=close` byte-diffs all four sections of every candidate against the body it was built from and refuses the level if any frame-1 section moved; measured on **all five** levels: hpac / semantic / token tail identical | jg5 §6.3 | SOURCE |

---

## §8 — §G "Honest limits"

| Claim | Value | Receipt | Grade |
|---|---|---|---|
| Toy/naive audit grades | 37 groups: **6 TOY-NAMED · 15 NAIVE-NAMED · 13 OPTIMAL-FORM · 3 not built**; explicitly **not** a global exhaustion claim (two machine corpora unadjudicated) | `.omx/research/ddm_ty1_20260806/RECEIPT.md` "Denominator First" + grade table; machine ledger `TOY_LEDGER.jsonl` | SOURCE |
| "Fewer than half licensed their verdict" | 13 OPTIMAL-FORM of 34 implemented (37 minus 3 not built) | this arm | DERIVED |
| Five charter premises falsified | four of them would have produced an over-claim | pq4 `CONTRIBUTIONS_INVENTORY.md` §6 table; `MAIN_HANDOFF.md` §5 | SOURCE |
| Rediscovery at three days | the pose law was measured 2026-08-16 in a better functional form and re-derived 2026-08-19 without citation; citation audit `grep -c "pi2"`: up2 **1** (dismissive), jg1 **0**, fo2h **0**, jg2 **0**, the carrying memory **0** | na10 §4.1 + citation audit | SOURCE |
| Multiplicative restatement wrong by up to 52% | error systematic in the direction that matters — the ratio inflates as a candidate improves | na10 §4.2, sourced to `.omx/research/ddm_pi2_pose_axis_attribution_20260816.md` §0.4 | SOURCE |
| Three agents by measurement, none by recall | including the auditing arm itself | na10 §4.4 | SOURCE |
| Re-grade tally | 24 verdicts adjudicated at receipt: **13 STANDS · 10 REOPENED · 1 SHARPENED** | na10 §6 tally line | SOURCE |
| REOPENED semantics | the negative no longer binds; **not** that the positive is proven | na10 frontmatter `verdict_scope` | SOURCE |
| 24 of 34 shipping-tree files have no VCS source | including the receiver, the residual archive, the corrector, and `inflate.py`/`inflate.sh` | pq4 `MAIN_HANDOFF.md` §4 item 2 + `CONTRIBUTIONS_INVENTORY.md` §5 near-miss 3, measured by the wc2c 34-file census | SOURCE |
| Two wall-clock memos disagree | `[890.6, 1430.6] s` graded WARN vs `[822, 1302] s` graded REFUSE on the same body; both ours; unreconciled | pq4 `MAIN_HANDOFF.md` §4 item 1 | SOURCE |
| Stager has no tests | named twice, still owed | pq4 O18 + Owed item 1 | SOURCE |
| Native port unbuilt, critical path | projected ~331 s is **DERIVED, not measured** | pq4 L6 + Owed item 2 | SOURCE |
| Residual-payload provenance unresolved | claim withdrawn | pq4 A16 + Owed item 6 | SOURCE |
| Device-refusal cause OPEN | chartered explanation falsified; round trip perturbs **50.086%** of positions by ~1 ULP; differential test **refused to convict** (0 changed coding rows) | pq4 A13 + Owed item 8 | SOURCE |
| The measured NULL | 12-dim basis re-orientation leaves reachable pose correction invariant to **1.9e-08** (machine precision, 24 random pairs); it **ships nothing** | pq4 P3 citing `.omx/research/ddm_br1_pose_basis_reorientation_20260819.md` | RELAYED |
| The dropped split | `reserved = 0`, `semantic_split = false`; receiver support ships and is **inert**; 520 of a cumulative 2,829 B are **not** in these bytes | pq4 P4 | RELAYED |
| Selection on the scored clip | member set, mixer context and learning rate chosen by racing on the scored video; family robustly negative at **30 of 33**; fallback (1 member, 1 context) still **−340.82 B**; optimality on another clip **not claimed** | fx1 §7 first bullet + §4 race table row `k1_cb16` | SOURCE |
| Floor is a band | 0.07–0.13 ESTIMATE, rate-dominated; no proven nontrivial floor; `0.11797` is the OLD carrier's own iid coding floor | memory `theoretical_floor_is_below_the_goal_20260817` | SOURCE |

---

## §9 — Top recorded follow-ons, priced (for MAIN's routing, not only the writeup)

| # | Follow-on | Projected ΔS | Grade of that number | Blocker / cost | Receipt |
|---|---|---:|---|---|---|
| 1 | Reopened rung-4 token drop, with the compensated pose leg | **−3.243e-3** (rate leg exact) | MEASURED for rate+seg; pose leg **UNMEASURED at n≥60** | $0, ~40–65 min at n=60; 6.5–10.7 h at n600; priced on a different body (182,759 B) | na10 §5.1, §7 item 2 |
| 2 | Third admission branch `drop` | **−0.002929** | MODELLED/priced | needs a **receiver** path; invalidates the current byte-identity chain | pq4 A15; jg3 |
| 3 | Model half as a representation (66,528 B = 37.7%) | **unpriced** | — | no design exists; largest never-attacked section | tx1 §0 |
| 4 | CPR1 inner-coder repack | **[−1.751e-4, −1.13e-4]** | MEASURED raw (+263 B), realized PROVISIONAL (~230 B, ±117 B) | lossless ⇒ no scorer run; measured on a **superseded** body, needs re-measure on the shipped one | `.omx/state/canonical_task_status.jsonl` `qw1_ra2_cpr1_inner_coder_repack_byteclose_20260816`; `ddm_ra2` §5 |
| 5 | CPR1 lossless rider | **−1.2185e-4** (183 B) | MEASURED, then **DECLINED** | folding changes archive bytes ⇒ the score becomes DERIVED, not measured; must re-run **last** on the final body with its two-line receiver patch; **not** proven across a carrier re-solve | pq4 P6; jg5 §7 third bullet |
| — | *Named, but not headroom:* deep prune keep25 | rate −1.3657e-3, **pose +0.0264** | rate MEASURED, pose PROJECTED from n=2 | net **2.8× the gap, wrong sign** | ledger `qw1_mp2_deep_prune_...` |
| — | *Named, but nearly spent:* miss-sector remainder | ≈ **−4e-4** at a few hundred bytes | ceiling MEASURED (1,247.19 B perfect-oracle), realized 104.584 B | the 77,241 B framing is withdrawn | ma1; fx1 §5 |

---

## §X — Claims I could NOT receipt, and what I did about them

1. **"95.9% render-loss" vs "96.6% of the seg axis."** My charter named 95.9%. Two artifacts
   measure adjacent quantities on **different vehicles**: `na10` L4 (via `jg1`) gives **95.9%** of
   the seg leg as render/re-segment loss on the jg2/up3 base; `rt1` gives **96.6%** of the seg axis
   as round-trip loss on the `hv1 ep0634` base; `td1` gives "~95%" with 1,717 label errors. **Action:**
   SECTIONS quotes 95.9% for the general claim with na10 as receipt, quotes rt1's decomposition
   separately with its vehicle named, and states that the shape transfers and the magnitude does not.
   Nothing is averaged.
2. **The rt1 decomposition is not measured on the shipped body.** Carried in the text as an explicit
   caveat rather than dropped, because the structural facts (boundary concentration, runner-up
   margins) are the load-bearing part and no shipped-body equivalent exists.
3. **`0.14838267` vs `0.14839100138338618`.** The build memo's advisory-projected S differs from the
   contest row in the 5th decimal. **Action:** SECTIONS uses **only** the contest row and the report's
   own legs. The advisory figure is not quoted.
4. **"~29× pose loss from direct quantization"** is RELAYED through `hr1`; I did not open `pz4p`.
   Kept, marked RELAYED, and stated in SECTIONS as "about 29×" rather than to more precision.
5. **Task-ledger coverage.** `cw1` reports the numeric task ledger as document-abandoned (IDs
   1103/969/1142/1143 return zero rows). I re-checked: `.omx/state/canonical_task_status.jsonl` has
   **216 unique ids, 104 open, 83 of them written in 2026-08** — so the ledger is live but has moved
   to **string** ids. **Action:** SECTIONS makes no claim about ledger completeness; §9 above cites
   individual rows by content, never by bare numeric id.
6. **NEON.** Not receipted anywhere in scope. **CUT** — SECTIONS says "split-native port" and names
   only the forced-scalar parity twin, which is receipted.
7. **A pose figure of `3.4e-5`** appears in older doctrine. It is an **ancestor-vehicle** number and
   does not transfer. **CUT** — it appears nowhere in SECTIONS.
8. **`d_seg` "one edge is 43.4%" on the shipped body.** Only measured on `hv1`. Carried with the
   vehicle named.
9. **My own denominator error, caught by a sister arm and corrected here.** My first draft said
   token decode is "95.72% of inflation". That is wrong: 95.72% is the share against the
   **1,401.58 s instrumented-stage sum**; against the **1,419.9 s inflation** the figure is
   **94.5%**. `ddm_pq5` had already fixed this in the packet (commit `4226017206`) before I wrote
   it. SECTIONS now carries 94.5%. This is the same class as the sm1 "the floor you divide by
   decides the answer" finding, and it happened inside this arm.
10. **The mechanism for the refused `base_odds` member is INFERRED, not measured.** fx1 labels it
    so. SECTIONS states it as an inference in the same sentence.
11. **`sm1`'s search-headroom threshold is INSTANCE-scoped** to a base other than the shipped one.
    SECTIONS quotes the percentages and the ~1000× bar without implying they were re-measured here.

**Cut count: 3 claims cut (NEON, the 3.4e-5 pose figure, any averaged render-loss share).
One number corrected against a sister arm (95.72% → 94.5%). Flagged-but-kept with explicit
scope: 8. Everything else in SECTIONS.md has a row above.**
