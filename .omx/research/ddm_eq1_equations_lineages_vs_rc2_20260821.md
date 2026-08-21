# ddm_eq1 — 449 canonical equations and every past lineage, re-priced against rc2

**arm** `ddm_eq1` (canonical-equations × vehicle-lineage crosswalk) · **date** 2026-08-21
**axis** every re-pricing below is `DERIVED-EXACT` arithmetic over the score law at the live
pointer. This arm fired **no** scorer, **no** Modal job, **no** training run. It produced **no**
byte-closed candidate. `score_claim=false`, `promotion_eligible=false`.
**Own-vehicle frontier: S 0.14827847122030852 @ 180,456 B `[contest-CUDA T4, n600]`, archive
`df7fd266…` — UNMOVED by this arm.**

STORES CONSULTED: `tools/list_canonical_equations.py --json` (449 equations,
`.omx/state/canonical_equations_registry.jsonl`) · `.omx/state/canonical_frontier_pointer.json` ·
`.omx/research/ddm_wl1_20260805/TRANSFER_TABLE.md` · `ddm_asym1_three_axis_asymmetry_and_dynamics_20260818.md` ·
`ddm_rc2_t4_row_sixteenth_move_20260820.md` · `ddm_up2_shipping_object_pose_solve_20260819.md` ·
`ddm_os1_optimization_sweep_termination_census_20260802.md` · `ddm_fs2_rc4_drop_carrier_resolve_20260820.md` ·
`ddm_fs3_jg5_real_price_reopen_20260820.md` · two sub-arms (TR1/HPAC receipts; PR130/135 intake).

---

## ANSWER FIRST

**The largest credible un-cashed claim is on the SEG axis, mechanism class CONDITIONING:
`ddm_ec1` implicit edge-conditioned seg adjustment.** Ceiling **−0.019002 S** net of its own
rate price — **104% of the gap to the floor band's upper edge**. Its break-even is **1.1367e-5
d_seg = 5.64% of rc2's flips**. Its context selectivity is MEASURED — AUROC **0.9956550**,
**808.39×** lift `[contest-CUDA T4 retained-field, n600, scorer-free]` — and its design price is
MEASURED at **1,707 B**. **Its realized Δd_seg has never been measured.** The design landed
2026-08-14 and never got a training slot.

Seg is only 13.6% of S, and that is exactly why this is the finding: **seg is the only axis whose
ceiling (−0.020139) exceeds the gap to 0.13 (0.018278).** Pose cannot (43.7% of that gap). Rate
can in principle, but every rate mechanism measured on this vehicle realizes **9–21%** of its
first-order model, and the coding sub-axis is measured **closed at −5 B**.

**And one correction, first, because it was in my charter as a premise.** The TR1/HPAC claim —
"the e960/hv1 endpoint reached ~130,875 B, 49,581 B below rc2, worth 0.033 S" — is a **category
error, not a small error**. `estimated_joint_bytes` is a trainer-internal MPS telemetry proxy
(`tools/train_ddm_cl1_hpac_capacity.py:1079`), stamped `byte_authority:
"ADVISORY_ESTIMATE_NOT_SERIALIZED"`, carrying **no pose term and no archive**. It omits 52,366 B
of frozen semantic (34,763 B), carrier (22,161 B), residual and wrapper sections. hv1's **real**
byte-closed archive is **182,759 B — 2,303 B worse than rc2**, and it loses to rc2 on **all three
components**. There is no rate curve there to trade against. §2.2.

---

## 1. PART 1 — the equations, re-priced at rc2

### 1.0 The operating point, re-derived (not copied)

`S = 100·d_seg + √(10·d_pose) + 25·B/37,545,489`

| term | value | S | share |
|---|---:|---:|---:|
| d_seg | 0.00020139 | 0.020139000 | 13.6% |
| d_pose | 6.37e-06 | 0.007981228 | 5.4% |
| rate | 180,456 B | 0.120158243 | **81.0%** |
| **sum** | | **0.14827847122** | matches the pointer to 11 dp |

Exchange rates, all re-derived from the score law, all confirming the charter's figures:

| quantity | derived | charter |
|---|---:|---:|
| 1,000 B | 6.6586e-04 S | 6.66e-4 ✓ |
| 1 seg flip (n600 × 512×384 = 117,964,800 px) | 8.4771e-07 S = **1.273 B** | 1.27 B ✓ |
| 1e-7 d_pose | 6.2647e-05 S = **94.1 B** | 94 B ✓ |
| rc2 flip count | **23,757 flips** | — |

**Byte bars** (`sub015_pure_rate_archive_byte_bar_v1`, called rather than copied — the law is
stale-proof by construction and invalid the moment a candidate moves distortion):

| target | bar_bytes | cut required |
|---|---:|---:|
| 0.15 | 183,041 | **already met by 2,585 B** |
| 0.13 (floor band upper) | 153,005 | 27,451 B = 15.2% |
| 0.07 (floor band lower) | 62,896 | 117,560 B = 65.1% |

### 1.1 The operating-point flip — the 77× precedent has fired again, and it now cuts BOTH ways

CLAUDE.md records the seg-vs-pose marginal flip at PR106 (pose_avg 3.4e-5 → pose marginal 2.71×
seg's). Re-derived at rc2: `dS/d(d_pose) = 5/√(10·d_pose) = 626.5` against seg's constant `100`.

| operating point | d_pose | pose marginal | vs seg |
|---|---:|---:|---:|
| old 1.x scores | ~0.18 | ~12 | 0.12× (seg 77× more important) |
| PR106 frontier | 3.4e-05 | 271.2 | 2.71× |
| **rc2** | **6.37e-06** | **626.5** | **6.26×** |

rc2 sits **39× below** the 2.5e-4 crossover. **CLAUDE.md's table owes a new row.**

**But the routing conclusion INVERTS against the marginal, and this is the load-bearing part.**
`ddm_asym1` (2026-08-18, at S 0.15771) concluded *"pose is 107.5% of the gap"* and routed
pose-first. Re-derived at rc2 against the floor band:

| route | share of gap to **0.13** | share of gap to **0.07** |
|---|---:|---:|
| rate (drive bytes) | unbounded — needs 27,451 B (15.2%) | needs 117,560 B (65.1%) |
| **seg → 0** | **110.2%** — the only axis that can close it alone | 25.7% |
| **pose → 0** | **43.7% — CANNOT close it alone** | **10.2%** |

Driving **all** distortion to zero leaves S = 0.120158. The pose-first doctrine is retired **for
routing** while remaining correct **for local admission** — asym1's own §3 warned of exactly this
confusion ("cite the marginal for local decisions and the total for routing; using the marginal
for routing is the trap"). It is now measurable rather than hypothetical.

The PR130/135 intake carried the same premise harder — both `fd135` and `pi135` led with *"the
remaining gap is pose-first, 95.1% pose."* That was true at lc2 (`d_pose 2.332e-5`). **rc2's
6.37e-6 is already below PR135's own 6.88e-6. That gap no longer exists. Do not re-route on
"95% pose."**

### 1.2 Laws that are STALE or SATISFIED at rc2 — retire, do not cite

| law | derived at | status at rc2 |
|---|---|---|
| `pose_second_wall_t1_feasibility_bound_v1` (07-07) | d_seg 0.00092 / rate 0.0602 | **SATISFIED.** Its T_3 bound is d_pose ≤ 3.2e-5; rc2 is **5.0× inside**. Pose is not "the second wall" any more. |
| `gt_scoredframe_spike_rate_equals_witness_flicker_floor_v1` (07-15) | label-smooth witness flow | **NOT A FLOOR HERE.** rc2's d_seg is **26.4× BELOW** the claimed 0.005318 "floor". It was lineage-scoped to the witness vehicle. Citing it as a floor on rc2 is a lineage transfer error. |
| `horizon_weighted_margin_hinge_v1` (07-09) | d_seg ~26× higher | **STALE.** Its ΔS ceiling 0.012–0.024 at 0 bytes is 60–119% of rc2's *entire* seg budget. Not credible at this operating point; re-derive or drop. |
| `ddm_gc9_seg_rate_product_law_v1` (07-30) | knee-B / QA24 | **PASSES, non-binding.** c = (100·d_seg)·(rate term) = 2.4199e-3 against a 5.05e-3 bar — 2.09× margin. |
| `sfess_fixed_k_cached_replay_ranking_v1` (07-13) | S ≈ 0.19080 | **STALE** operating point. |
| `seg_rate_breakeven_v1` (07-19) | — | **STILL EXACTLY VALID.** B* = 150.18 B per 1e-6 d_seg. Both terms are LINEAR, so this one is operating-point **independent**. Re-derived: 150.181956 ✓ |

**The general rule this sweep confirms:** seg↔rate is linear and therefore transfers across
operating points; pose↔rate is √-concave and therefore never does. Any law quoting a pose ΔS
must carry its reduction *factor*, not its ΔS.

### 1.3 The per-axis headroom map

`consumed` = a live file/module on disk consumes it. `orphaned-as-code` = the equation's
`canonical_consumers` field holds prose or memo references only, so no code enforces it. Of 449
equations, **27 have no resolvable code consumer; 17 of those are rate-axis.**

#### RATE — 0.120158 S (81.0%)

| law | claim re-priced at rc2 | evidence body | consumed? |
|---|---|---|---|
| `section_coding_axis_closure_v1` | **CODING IS CLOSED**: 4 mechanisms × 4 sections leave **−5 B = −3.3e-6 S** | `[byte-exact through the real coders]`, no entropy estimates | orphaned-as-code |
| `carrier_rate_credit_pose_affordance_v1` | carrier 22,161 B; full delete **−0.014756 S**; pose bar **LOOSENS to 8.12×** at rc2 (7.72× at the 182,759 B frontier) | `[macOS-CPU advisory]`; explicitly **does not license** any claim about where a rung lands on d_pose(fidelity) | orphaned-as-code |
| `realization_breakeven_bytes_v1` | realization **9.46%** (r2b) and **21.27%** (pz4r, measured on exactly the CPR1 basis+coeff-map section: 19,221 B envelope → 4,089 B realized, 15,132 B decoder-required) | `[macOS-CPU advisory]` + `[macOS-CPU scorer-free receiver build]` | consumed |
| `compensated_semantic_edit_exchange_v1` | per S of rate credit: 0.508 back as pose, 0.388 as seg, **0.105 net**; retention RISES with mass. To close the gap to 0.13 alone needs **261,438 B = 7.52× the semantic section**. INFEASIBLE alone | `[contest-CUDA T4 n600]` legs, exchange DERIVED-EXACT | orphaned-as-code |
| `ddm_et5` / `ddm_et4` | additive carriage **rate-dead**: 84.48 B/flip against the 1.2731 B/flip waterline = **66.35× over**; waterfill selected **0/32** (stratified, not prefix) | `[macOS-CPU advisory]` | orphaned-as-code |
| `ddm_dc1_correction_stream_label_cost_v1` | correction break-even density **1.2853e-3**; rc2's global flip density 2.0139e-4 is **6.4× below** it — and below the bottom of dc1's own swept band (2.2e-4). *Caveat: dc1's ρ is support-local, not global, so this is corroboration for ET5's measured kill, not an independent one.* | `[macOS-CPU advisory]` | consumed |
| `weight_entropy_rate_in_loss_lever_v1` | **−16,007 B (−19.6%)** MEASURED — but its own anchor reads *"TORCH-VEHICLE anchor; NOT transferable"*; MLX port **NEVER-FIRED** | `[contest-CPU advisory]`, torch_split_by_head_basin | orphaned-as-code |
| `token_rate_model_direction_dependence_v1` (08-21) | −log2 p trust factor is **direction-dependent**: 0.92× away from argmax, 0.09× toward it | `[macOS-CPU advisory / scorer-free EXACT byte]` | consumed |
| `greedy_set_average_vs_marginal_price_v1` (08-21) | a greedy set's average is **not** its margin: 2.24× the average at 3.98× less yield | same | consumed |
| `sub015_pure_rate_archive_byte_bar_v1` | bars in §1.0 | DERIVED-EXACT | orphaned-as-code |

#### SEG — 0.020139 S (13.6%)

| law | claim re-priced at rc2 | evidence body | consumed? |
|---|---|---|---|
| `seg_rate_breakeven_v1` | 150.18 B per 1e-6 d_seg; **operating-point independent** | DERIVED identity | orphaned-as-code |
| `ddm_pw1_menu_saturation_discriminator_v1` | occupancy at a menu bound ⇒ **our** menu clips the solution. Freeing two menus realized exact **ΔS −0.0163787 at +85 B** — *on the v4d/tq1c body at S 0.9476, a retired vehicle. The number does not transfer; the DISCRIMINATOR does.* | `[macOS-CPU advisory]` | consumed |
| `ddm_v19c_correction_saturation_asymptote_v1` | zero-byte corrections saturate at d_seg 0.024787 — **123× above rc2's d_seg**. Saturation is not the binding constraint here | `[macOS-CPU advisory]` | orphaned-as-code |
| `ddm_ec1` implicit edge conditioning | **the largest un-cashed item** — §3 | selectivity `[contest-CUDA T4 retained-field, n600, scorer-free]`; realized Δd_seg **NEVER MEASURED** | not built |

#### POSE — 0.007981 S (5.4%)

| law | claim re-priced at rc2 | evidence body | consumed? |
|---|---|---|---|
| `ddm_up2` converged solve | uncapped convergence PROOF on **600/600** pairs; 429 improved, **0 worsened**; **0 archive bytes**, **0 d_seg**; ΔS −6.847e-05. Carrier within **1.5% of its own optimum** | `[macOS-CPU advisory, frozen CPU-torch PoseNet, DALI GT]`, on the **to1** body (176,420 B), not rc2 | byte-close **BLOCKED** by a tool bug |
| `ddm_up2` §5 — the basis wall | not rate (all 7,200 coefficients ±4 = **+5 B**), not the int12 lattice (headroom 1.000–1.020×). **The wall is the 12-dim basis**: 6.4× median demanded-step penalty. A re-fit delivering 6.4× on d_pose would be **−0.004826 S** | up2 states the 6.4× bounds the **demanded step**, not the realized gain — treat the −0.004826 as a ceiling, never a projection | **never built** |
| `cw1_gt_lineage_additive_pose_offset_v1` | d_pose_pyav = d_pose_dali + **1.4061e-04**, ADDITIVE | `[macOS-CPU advisory]` | consumed |
| `ddm_up2` §4c | improving CUDA pose by 1.55% made the **CPU axis 0.041% worse** — the two axes pull in **opposite** directions under this actuator | measured, both GT caches, n600 | consumed |
| `pose_stack_exact_budget_v1` | budget the whole stack, never each proposal independently | `[derived from CP135 contest-CUDA T4 n600]` | consumed |
| `ddm_os1_termination_census` | the pfs1 GN solve was censored **600/600 stopped-on-bound, 0/600 converged** — but that is the **retired v4c/TR1** chain. up2's solve on the shipping object **did** converge | `[macOS-CPU frozen-PoseNet advisory]` | consumed |

---

## 2. PART 2 — the lineages, component-wise against rc2

A lineage beats the frontier if it holds **any** component cheaper at equal-or-better others.

### 2.1 Own witness line v1–v10 (`ddm_wl1` transfer table, 2026-08-05)

wl1 ranked 35 rows: 9 PORT-NOW, 9 RACE, 6 LESSON-ONLY, 5 DEAD, 6 PRECONDITION-CHANGED. It was
written when the own-vehicle frontier was **S 0.7539807296911207 @ 357,836 B `[macOS-CPU
advisory]`** — a vehicle **5.1× worse** than rc2 and since retired.

**Not one of its 9 PORT-NOW rows targets the RATE axis.** They are: exact S_R reachability weight
(seg), v10 readiness surface (apparatus), annulus dwell satisfiability (apparatus), parser-truth
review rule (apparatus), P0-force isolation guard (apparatus), v10 per-cell flat cells (design
guard), v9c3 fork geometry (A/B template), plus two folded. wl1's own top-5 is **4 apparatus
guards and 1 seg lever**.

The rate-adjacent rows sit in RACE / PRECONDITION-CHANGED and are all witness-lineage-scoped:
contour arithmetic re-price (rank 31), structured code-table locality probe (rank 33), v10
content-priced coder (rank 7). Their byte numbers — LBND2 lane-band 41,526 → 30,892 B with a
26,179 B floor; store-nothing carrier 697,941 → 1,049 B — are on a carrier rc2 does not ship.

**Verdict: LESSON-ONLY as numbers, and the ported items are apparatus, not score.** The one
genuinely un-ported score row is rank 2 (exact through-R `S_R` reachability weight — built,
cache-ready, never fired, seg axis). It needs a trainer this vehicle no longer runs.

### 2.2 TR1 / HPAC own-trained burn — the 130,875 B claim is a CATEGORY ERROR

| point | archive B | d_seg | d_pose | S | axis | byte-closed? |
|---|---:|---:|---:|---:|---|---|
| **rc2 (frontier)** | **180,456** | **2.0139e-4** | **6.37e-6** | **0.148278471** | contest-CUDA T4 n600 | yes `df7fd266…` |
| hv1 ep634 `s1p25_c1p0` | 182,759 | 2.9611e-4 | 6.88e-6 | 0.159597293 | contest-CUDA T4 n600 | yes `80d9c8c6…` |
| e480b v2 (ep480) | 183,502 | 2.9611e-4 | 6.88e-6 | 0.160092026 | contest-CUDA T4 n600 | yes |
| ep508 telemetry **"130,875"** | *est. only* | — | **absent** | — | **macOS-MPS proxy** | **NO — never serialized** |
| TR1 pfs1 D1 (best-ever d_seg) | 569,996 | 3.89011e-3 | predicted | 0.7768 = 5.24× rc2 | n600 evaluate.py | yes |
| TR1 cx1/ix2 (smallest byte-closed) | 353,805 | 4.31179e-3 | predicted | 0.6751 = 4.55× rc2 | n600 | yes |

`estimated_joint_bytes` = `estimated_token_bytes + estimated_model_bytes` only. Its own selector
receipt reads `authority: "[macOS-MPS advisory telemetry proxy; no contest score]"`,
`score_claim: false`, `pose_term_included: false`. It has **no d_seg, no d_pose, no
archive-bytes field anywhere** across 81 candidates. 130,875 is not even its minimum (ep634's
130,393 is), and `av1` F3 separately declared the fit that pinned `y_inf = 130875` **defective**.

**Component-wise, hv1 loses to rc2 on all three legs:** rate −2,303 B (−0.0015335 S), d_seg
−0.009472 S, d_pose −0.0003133 S. The three deltas sum to **0.011318822**, matching the total S
gap to the last digit. **No component of this lineage is cheaper than rc2 at equal-or-better
others.**

**The hybrid is moot because rc2 already IS it.** `rc2_composed` is hv1 ep0634's own descendant
on the same `pr130_lift` semantically-quantized vehicle; it kept the rate machinery and improved
distortion on top. TR1 proper is d_seg-bound, not rate-bound: granting it a **zero-byte archive**
and hv1's pose for free, its best-ever d_seg alone gives S = 0.397306 = **2.68× rc2**.

**Cheapest falsifier (already run, $0):** sum the shipped section bytes —
`112,110 + 13,515 + 34,763 + 22,161 + 14 + 96 + 100 = 182,759` against est_joint 130,393. Anyone
claiming a ~130 KB archive here must name which of the frozen semantic (34,763 B) and carrier
(22,161 B) sections they intend not to ship. Both are proven byte-identical to the incumbent's
and are load-bearing for the decode.

**Related registered law, now correctly scoped:** `hpac_mc36_joint_descent_law_v1` (cont floor
135,248 B, QAT floor 132,798 B) is `[macOS-MLX research-signal]` fitted to **that same telemetry
proxy**. It is a fit to a non-archive quantity. It must never be quoted as an archive floor.

### 2.3 PR130 / PR135 intake — mostly CONSUMED, because we adopted the F26 receiver wholesale

The shipping tree `submissions/robust_current/jg5_sub015_runtime/runtime/` **is** an F26
derivative, and rc2's 180,456 B already sits **6,268 B below PR135's own 186,724 B**.

**CONSUMED** (each with a live file): F24S schema (`residual_archive.py`) · RCF1 (same) · IHS2 v3
+ Gate-A (`ihs2.py`, `ihs2_gate_a.py`) · WANS1 weight codec
(`entropy/renderer_weight_codec.py`) · CAP1 AR(1)+Rice (`entropy/coefficient_ar1_codec.py`) ·
RC64 range coder (`entropy/rc64.py` + `rc64_backend.c`, natively lowered by us) · CBQ basis
quantization (as archive **state** — no code surface) · sparse frame-0 selector
(`frame0_selector.py`) · deterministic HPAC (`hpac_inference.py` + our `f26_hpac_native.c`).

**RF-7 — resolved, and it is not a PR130/135 mechanism at all.** It is a **PR86** architectural
claim (receptive field = 7: SegNet's class boundaries are decided by the stem + Stage-0
7-input-pixel window, so a renderer with RF exactly 7 — about 3 conv layers — suffices to
*place* boundaries). **Axis: SEG.** Its intakes attach **no ΔS, no bytes, no d_seg estimate** of
any kind, and tag it verbatim *"their derivation … not consumed as fact"* and *"LESSON-ONLY,
UNVERIFIED."* `grep -rn "renderer_conv_depth"` over `src/ tools/ experiments/` → **0 hits**; the
landed DSL `spec_tr1_renderer_20260728.py` carries **width, never depth**. **Double-orphaned**:
the lever was never built, and the vehicle it was designed for is not the vehicle we ship.

**UNHARVESTED, ranked (rate axis first — 81% of S):**

| # | item | axis | value at rc2 | body |
|---:|---|---|---|---|
| 1 | **CAP1 fixed metadata pack** | R | ≈ **−5.3e-5 S** (−79 B) if it reproduces | **MEASURED** `[custodied exact bytes]` — but on CP135's 186,252 B body; rc2's carrier is 22,316 B of *our* re-solved state, not F26's 22,222 B. An anchor, not a promise. |
| 2 | **Global starts beyond the ±1 Jacobian shell** | P | **UNPRICED — do not invent one** | jg5 §6.1 measured a **fixed point** (600/600 `no_improving_step`, 0 budget-limited). A fixed point of a *local* search is exactly when a global restart is the only remaining move. `radius2_multistart_singleton_escape_v1` measured escape on **597/600** rows; CUDA transfer open. |
| 3 | Rate-aware carrier gauge / QAT on F26 learned state | R+P | UNPRICED | credible only in *learned* form — the frozen post-hoc form is closed twice (PK3 saved 64 B against a 2,000 B trigger; pk2 cost +4,316 B) |
| 4 | mp2 keep25 mixed precision | R/P | **NOT CREDIBLE** — its −2,051 B credit was measured on hv1's 182,759 B; rc2 is already **252 B smaller than keep25's own 180,708 B**, so the credit is now negative; pose leg projects +0.0278 S from an **n=2** dose-response | — |
| 5 | lc2 same-state ANS recode | R | **CLOSED** — measured negative twice (+6 B, +9 B) against RC64 | `[custodied exact bytes]` |
| — | **`ddm_ec1` implicit edge conditioning** | **S** | **the largest un-cashed item — §3** | selectivity MEASURED, realized effect NOT |

**Operating-point transfer flags (numbers that do NOT carry to rc2):** the "95.1% pose" routing
premise (§1.1) · mp2's byte credit · CAP1's and ec1's byte prices (both measured on CP135's
186,252 B body) · **all** eh1 EUREKA projections (priced against the pre-F26 flat-paint vehicle;
row 1's headline 0.80084 is dead arithmetic) · #869's −113,555 B (IX2, task-lossy, later
scorer-harmful). The invariant that *does* carry: `dS/dB = 6.658590e-07`.

### 2.4 The micro-edit / composition line (cp135 → … → rc2) — what its own trajectory says

The sixteenth move is the cleanest read of this line's shape: **d_seg IDENTICAL, d_pose
IDENTICAL, −169 B — the entire ΔS of −1.125302e-4 came from rate.** Moves 12–16 are all ~1e-4
class.

The line is **not** diminishing because it ran out of ideas; it is diminishing because its
mechanisms keep meeting the same measured wall:

| rung | first-order model | realized | ratio |
|---|---:|---:|---:|
| fs2 token drop (Path B, u = 7.75) | −1.6058e-03 S (11,716.7 B) | **1,022 B** | **8.72%** |
| r2b sparse repair stream | 0.0142 S scheduled | 0.0012332 S | **9.46%** |
| compensated semantic edit | 1.000 S of rate credit | 0.105 net | **10.5%** |
| pz4p→pz4r CPR1 envelope | 19,221 B | 4,089 B | **21.27%** |

Four independent mechanisms, four independent instruments, **realization 8.7%–21.3%**. That is
the extrapolation: **discount any first-order rate claim on this vehicle by 5–10× before
believing it.** fs3 then measured the second-order form of the same law — a greedy set's
**marginal** price is 2.24× its average at 3.98× less yield, so the tail of any admitted set is
where the credit dies.

### 2.5 The solve line (up2 / sq2 / C1) — does any solve asset beat rc2's distortion at ≤ its bytes?

**Pose: one asset, real, small, and packaging-blocked.** up2's uncapped solve reached a
convergence **proof** on 600/600 pairs — 429 improved, **0 worsened**, **0 archive bytes**, **0
d_seg**, ΔS **−6.847e-05**, with a bit-identical determinism control on 10 re-solved pairs across
4 origin processes. Byte-close is **BLOCKED by a tool bug, not a physics limit**:
`ddm_t1h_compose_pass1.py:108` reads `k_base = packed[139]`, a body-specific hardcoded offset (177
on the to1 body), so the tool refuses **every** candidate including the shipped codes themselves —
a failed identity control that proves it is a tool bug. up2 §8.7: *"No re-solve and no new
measurement is needed — only packaging."* **Caveat: the solve ran on the to1 body (176,420 B,
d_pose 7.77e-6), not rc2 (d_pose 6.37e-6).** The −6.847e-05 is that body's number.

**Pose, the honest ceiling:** the carrier is within **1.5% of its own optimum**. up2's own verdict
is *"the pose carrier is not where sub-0.15 comes from."* The whole axis holds 0.007981 S.

**Seg: no solve asset beats 2.0139e-4 at ≤ current bytes** — because at rc2's flip density every
*additive* corrective family is measured rate-dead (§1.3, ET5 66×). `ddm_pw1`'s −0.0163787 is a
**retired-vehicle** number (v4d/tq1c at S 0.9476); only its discriminator transfers.

---

## 3. The largest un-cashed claim, priced

**`ddm_ec1` implicit edge-conditioned seg adjustment.** Oriented four-neighbour context adapter,
1,424 parameters, design complete 2026-08-14, **never trained**.

| quantity | value | body |
|---|---:|---|
| context selectivity | AUROC **0.9956550**, **808.39×** lift | **MEASURED** `[contest-CUDA T4 retained-field, n600, scorer-free]` |
| design price | **1,707 B** = +1.1367e-03 S | **MEASURED** exact archive delta — *on CP135's 186,252 B body; owes a re-pin to rc2* |
| **break-even** | Δd_seg **1.1367e-05** = **5.64% of rc2's 23,757 flips** | **DERIVED-EXACT** from `seg_rate_breakeven_v1` (operating-point independent) |
| **ceiling** | **−0.019002 S** = **104% of the gap to 0.13** | DERIVED (d_seg → 0, net of its own rate price) |
| **realized Δd_seg** | **NEVER MEASURED** | — |

**Why it is credible rather than merely large.** It is the one mechanism in the whole inventory
that is *conditioning* rather than *additive correction* — it changes how the existing tokens are
coded, so it does not pay the 66× carriage tax that killed ET4/ET5, and it is not exposed to the
8.7–21.3% realization discount that governs subtractive rate rungs, because it buys distortion
directly rather than buying bytes and paying distortion back.

**Why it is not a promise.** A measured *selectivity* is not a measured *effect*. 808× lift says
the context predicts where flips are; it says nothing about how many the adapter can remove
through the real receiver. The break-even is low (5.64%), but the realized number could be zero.

**Cheapest falsifier:** train the 1,424-parameter oriented adapter on the current base and measure
realized Δd_seg through the real receiver forward. Re-pin the 1,707 B against rc2's carrier first,
or the price answers about a body we no longer ship.

**Runner-up (rate axis).** Carrier shave/deletion under
`carrier_rate_credit_pose_affordance_v1`: first-order **−0.014756 S** (22,161 B), pose bar
**LOOSENED to 8.12×** at rc2 (from 7.72×) — the bar is quadratic in the credit, so the
largest-credit rung sits under the loosest bar and bounds the whole curve in one decode. But
`pz4r` already measured **21.27%** realization on exactly this CPR1 basis + coefficient-map
section (15,132 B proved decoder-required), which discounts it to **≈ −0.0031 S**. And the law
explicitly does not license any claim about where the rung lands on d_pose(fidelity) — that is
unmeasured. **Fire order: measure the largest rung first.**

---

## 4. PART 3 — the five-line pattern read

1. **The score is 81% rate, but rate has split in two: CODING is measured CLOSED (−5 B across 4
   mechanisms × 4 sections, byte-exact through the real coders), so all remaining rate headroom is
   REPRESENTATION.**
2. **Representation realizes 8.7%–21.3% of its first-order model** — four independent mechanisms,
   four instruments, one number. Discount every first-order rate claim by 5–10× before believing
   it, and price the margin, not the average (2.24×).
3. **Every ADDITIVE mechanism is now rate-dead at rc2's distortion density.** ET5 measured 84.48
   B/flip against a 1.2731 B/flip waterline (66×, waterfill 0/32); up2's 15-pair overlay refused;
   fs3's 137-pair drop refused at 81× its credit. The flips got too sparse to be worth addressing
   one at a time. **The live classes are SUBTRACTIVE (remove what the receiver does not need) and
   CONDITIONING (same bytes, better basis or context) — never sidecars.**
4. **The pose-first doctrine is arithmetically retired for ROUTING** (43.7% of the gap to 0.13,
   10.2% to 0.07 — it cannot close either alone) **while remaining correct for local admission**
   (its marginal is now 6.26× seg's, up from 2.71× at PR106). Both statements are true; the trap
   is using the marginal to route.
5. **So the largest credible un-cashed claim is CONDITIONING on the SEG axis — because seg is the
   only axis whose ceiling (−0.020139) exceeds the gap to the floor band (0.018278), and `ec1` is
   the only conditioning mechanism with measured selectivity, a measured price, and a 5.64%
   break-even.** It has never been trained.

---

## 5. My own round-1 adversarial review

1. **Am I transferring ancestor numbers?** Every lineage number in §2 carries its body and its
   axis, and §1.2 / §2.3 list the transfers I explicitly refuse. The ec1 1,707 B and the CAP1
   −79 B are both flagged as CP135-body measurements owing a re-pin — I did not silently rebase
   them onto rc2.
2. **Is "orphaned-as-code" a real finding or an artifact of my resolver?** Partly an artifact, and
   I have labelled it as such: my check resolves a consumer only if it names a file or importable
   module. Many `canonical_consumers` entries are prose sentences ("any successor rung of the
   compensated-edit family…"). That is *itself* the orphan-grade debt the registry is supposed to
   prevent, but it is weaker than "no code consumes this." I report the denominator: 27 of 449
   unresolvable, 17 rate-axis.
3. **Is the ec1 ceiling honest?** It is a ceiling, not a projection, and I say so three times. The
   only measured quantities are selectivity and price. A reader who takes −0.019002 as an expected
   value has misread me.
4. **Did I check whether the floor band is reachable at all?** Yes, and the arithmetic is
   uncomfortable: zeroing **all** distortion leaves S = 0.120158, which clears 0.13 but not 0.07.
   Reaching 0.07 requires 117,560 B of cuts (65.1% of the archive) *plus* distortion work. Nothing
   in the registry predicts a mechanism of that size on this vehicle.
5. **Negative-existence claims.** "Not one wl1 PORT-NOW row targets rate" is over the 9 rows in
   `TRANSFER_TABLE.md` §Ranked, read in full. "RF-7 has no code consumer" is over
   `src/ tools/ runtime-rs/ experiments/ scripts/ .omx/state/` — one hit, a base64 substring. Both
   are scoped searches, not universal claims.
6. **What would change my mind fastest?** If ec1's realized Δd_seg comes back below 5.64% of the
   flips, the seg-conditioning route is refuted and the ranking inverts to the carrier shave. That
   is one training run, and it is the cheapest decisive measurement in this memo.

---

## 6. Owed, with owners

1. **Train ec1 and measure realized Δd_seg through the real receiver.** Re-pin the 1,707 B against
   rc2's carrier first. The decisive measurement in this memo. **Unowned.**
2. **Fix `ddm_t1h_compose_pass1.py:108`** (hardcoded `packed[139]`) and byte-close up2's converged
   pose codes. Packaging only, no re-solve. **Unowned.**
3. **Measure the largest carrier rung first** under the affordance law — it sits under the loosest
   bar (8.12× at rc2) and bounds the whole curve in one decode.
4. **CLAUDE.md owes a new row** in its seg-vs-pose operating-point table: rc2, d_pose 6.37e-6,
   pose marginal 626.5 = 6.26× seg — with asym1's warning attached, because the routing conclusion
   inverts against the marginal.
5. **Retire three laws from citation** (§1.2): the pose second wall (satisfied), the witness
   flicker floor (lineage-scoped, rc2 is 26.4× below it), the HWM ΔS ceiling (stale by ~26× in
   d_seg). And never quote `hpac_mc36_joint_descent_law_v1`'s 132,798 B as an archive floor — it
   is a fit to a non-archive telemetry proxy.

`verdict_scope`: **DERIVED-EXACT** re-pricing over the registry and the named lineage receipts.
No new measurement was taken. Every ΔS above is priced against
**baseline rc2 = 0.14827847122030852**.
