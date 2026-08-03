# ddm_rs2 — resuming five killed arms: one re-scope that survives, two premises that were already stale, and a false claim that stood for a day

**UTC** 2026-08-03 · **arm** `ddm_rs2_orphan_resumption` · **axis** `[macOS-CPU advisory]` /
`[byte-closed, scorer-free]` · `score_claim=false`, `promotion_eligible=false`.
**Pointer UNMOVED. This unit fired no gate and produced no exact row.**

**Denominator** (every ΔS below): gap to the PR130 demonstrated floor **0.7262358**;
1% of gap = **10,907 B**; PR130 = 191,052 B (`ddm_na1`).
Live own-vehicle base: `dc1_fold` **S = 0.8983775** @ 360,309 B.
(`tac.canonical_equations.gap_decomposition_against_floor_20260802`.)

---

## The answer first

1. **`ddm_pb3` §5 SURVIVES its re-scope, and the falsifier hardens by a factor I can state
   exactly.** `ddm_pj2` cut the pose contribution 0.227293 → 0.159726, so pb3's break-even
   hardens by **exactly the contribution ratio, 1.7308×** (mean d_pose reduction needed
   2.0130% → **3.4711%**) and its ceiling shrinks **−0.227754 → −0.126923**. But the ceiling
   is still **45–47× the price**, so it does *not* drop below the price and §5 is *not* moot.
   The operator's pre-registered close condition was checked and did **not** fire.
2. **pb3 and pj2 are neither the same win nor orthogonal — MEASURED.** They move disjoint
   DOF (pj2: 6 pose parameters + `s_t`; pb3: 692,712 seg-blind pixels) but are ranked by the
   same difficulty: `spearman(pj2 per-pair gain, blind-set reach g1) = +0.5320`, and pj2's
   top-30 gains overlap pb3's top-30 reach on **14/30 = 47%** against a 5% random expectation.
   **pj2 discounts pb3 by 1.73× and cannibalises ~47% of its best target set** — it does not
   refute it, because the debt that *remains* after pj2 is still concentrated exactly where the
   blind box reaches (`spearman = +0.3740`).
3. **`ddm_bo1`'s "live unresolved contradiction" is NOT live.** The recovery commit
   `eed9c61c81` already landed the §0 revision *and* the §7.2 reconciliation. Verified at
   source; the resolution is sound. **No correction is owed — landing one would have been
   duplicate honesty debt.** §4.
4. **A false claim stood in the record for a day and is now closed.** `ddm_bs2`'s
   `lever_lane_guard_ratchet` was **uncommitted** while commit `9f45920dca`'s message *and*
   its memo §6 both asserted it had landed. Landed in `3a90117ef0`. §6.
5. **The structural cause is paid** (`09fca46f37`): the checkpoint store now has a `findings`
   field and a `read --findings` knowledge log. §7.
6. **pj2's own headline is tail-concentrated, exactly as pb3 diagnosed in bp2** — top 1% of
   pairs carry **70.33%** of its reduction; the median pair improves **11.70%** against a
   **50.62%** mean. Its break-even is quoted in mean terms. §3.4.

---

## 1. What was resumed, and what state each arm was actually in

| arm | checkpoint at death | premise in my brief | **actual state on disk** |
|---|---|---|---|
| `ddm_pb3` | step 2 | §5 owed, re-scope it | correct — §5 genuinely never taken |
| `ddm_bo1` | step 3 | contradiction live | **STALE** — resolved by `eed9c61c81` |
| `ddm_cb2` | step 5 | died building the risk ladder | **STALE** — ladder present (7 rows) |
| `a6987ed5e93b04def` | **none** | unknown | is `ddm_bs2`; never checkpointed |
| `ddm_gd2` | step 1 | blocker permanently lost | **STALE** — `ddm_gd3` re-derived it |

**Three of five briefs were stale.** Not a complaint — a finding about the resumption
protocol: a recovery pass ran between the kills and this unit, and its work was invisible to
the brief. §4 of the operating manual applies to briefs about state as much as to numbers:
*resolve live state from the artifact, not from the notes.* Had I acted on the brief I would
have re-landed a correction bo1 already carried and re-derived a ladder cb2 already had.

`ddm_bs2` never checkpointed at all, which is why its label ("Lane guard P0 + binary-operations
sweep") was the only thing known about it. Its real task: replace the lane guard's constant
Lane budget with a *schedule*, and apply pw1's occupancy discriminator to every discrete choice
reachable from the live v4d chain.

---

## 2. `ddm_pb3` §5 re-scoped — the arithmetic

**Positive control first.** My harness reproduces pb3's published numbers before re-scoping
anything: ceiling **−0.227754** (memo: −0.22775), break-even **2.0130%** (memo: 2.01%),
`η_breakeven` **0.004726** (memo: 0.0047). Re-derived from
`reports/ddm_bp2/reach_n600.jsonl`, joined n600 against
`/Volumes/VertigoDataTier/pact/ddm_pj2_20260802/{final_pj2,final_ms8_reference}.jsonl`.

### 2.1 There are THREE pose bases in play, and pb3's is not the live one

| vehicle | mean d_pose | pose contribution `√(10·d)` |
|---|---:|---:|
| `v4d_composed_pb2_bestof` — **pb3/bp2's vehicle** | 0.0076424674 | **0.276450** |
| `ms8` — the prior own-vehicle frontier | 0.0051661970 | 0.227293 |
| `pj2` — this session's pose result | **0.0025512505** | **0.159726** |

**pb3 priced on a vehicle whose pose is worse than the frontier it was measured against.**
Everything in pb3 §1–§4 is scoped to that archive. That is the re-scope's whole content.

### 2.2 Break-even (EXACT — the rate side is exact, the base is measured)

7 scalars/pair × 600 = 4,200 B ⇒ `ΔS_rate = 25·4200/37,545,489 = +0.0027966`.

| base | pose contribution | mean d_pose reduction needed | `η_breakeven` |
|---|---:|---:|---:|
| pb3 as-published | 0.276450 | **2.0130%** | 0.004726 |
| ms8 | 0.227293 | 2.4457% | — |
| **pj2 (new base)** | **0.159726** | **3.4711%** | **0.006124** |

**Break-even hardens by exactly the pose-contribution ratio, 1.7308×** (to second order —
`r ≈ 2·ΔS_rate/C`, so `r ∝ 1/C`). `η` hardens **1.296×** (it enters nonlinearly).

### 2.3 The ceiling shrinks 44% — and still does not fall below the price

Floor: `d_floor ≥ d·(1 − γ)²`, `γ = g1/(2d) = h_Z/√(6d)`, else 0 when `γ ≥ 1`.

| | base contribution | floor contribution | **ΔS_pose ceiling** | full-cancel frac |
|---|---:|---:|---:|---:|
| pb3 as-published | 0.276450 | 0.048696 | **−0.227754** | 53.67% |
| pj2 base, `h_Z`-invariant | 0.159726 | 0.032804 | **−0.126923** | 58.33% |
| pj2 base, `γ`-invariant | 0.159726 | 0.028135 | −0.131591 | — |

`h_Z` is the box's reach *along* `e`; it is a property of the image gradients and the warp,
not of the pose residual's magnitude. Whether it is held fixed (`h_Z`-invariant) or scaled with
`‖e‖` (`γ`-invariant) is an **ASSUMPTION**, so I bracketed both extremes rather than picking one.

> **The bracket does not change the verdict: the ceiling is 45×–47× the price either way.**
> The operator's close condition — *"if it drops below the price, §5 becomes moot"* — **did not
> fire.** Even the hard bound (floor = 0 on every pair) is −0.159726 = 57× the price.

### 2.4 What the re-scope *does* kill

bp2's own step direction realizes a 0.1%–0.35% mean reduction (`cos ≈ −0.027…−0.059`,
so `1−cos² ≈ 0.1%–0.35%`). Against the **new** break-even that is
`η ≈ 0.000171…0.000600` = **0.028×–0.098× of 0.006124 — i.e. 10×–36× short.**
It was 10×–35× short before; the re-scope does not rescue it and does not condemn it further.
**The parametric 7-scalar field is the only thing in this family that can clear the bar**, and
whether it does is still §5, still unmeasured, still the number that decides the arm.

### 2.5 The subset payload survives and is the better shape

| subset | ceiling gain captured | bytes | ΔS_pose | **net ΔS at ceiling** |
|---:|---:|---:|---:|---:|
| k=30 + colex index | 77.3% | **232** | −0.078390 | **−0.078235** |
| k=60 | 81.1% | 455 | −0.084213 | −0.083910 |
| k=120 | 86.3% | 894 | −0.093161 | −0.092565 |
| all 600 (7 scalars) | 100% | 4,200 | −0.126923 | −0.124126 |

k=30 is **10.77% of the total gap for 232 B**. pb3's §6 item 5 conclusion — *"the payload is
not the binding constraint anywhere in this family"* — **survives the re-scope intact.**

---

## 3. Same win twice, orthogonal, or does pj2 dominate?

**None of the three cleanly. Answer: mechanistically disjoint, statistically correlated,
and pj2 dominates on price.**

**3.1 Disjoint DOF (DERIVED from the two mechanisms).** pj2 exploits an *exact algebraic
degeneracy in shipped coordinates* — `t = s_t·[p2,p1,p0]`, so `H(p,s) = H(λp, s/λ)` — and moves
the 6 pose parameters. pb3 perturbs 692,712 *seg-blind camera pixels* at 1 LSB. Neither can
produce the other's move. **They are not the same win twice.**

**3.2 But strongly correlated (MEASURED).**

| | value | random expectation |
|---|---:|---:|
| spearman(pj2 per-pair gain, blind reach `g1`) | **+0.5320** | 0 |
| spearman(pj2 *remaining* d_pose, `g1`) | **+0.3740** | 0 |
| top-30 remaining-debt ∩ top-30 blind-reach | **14/30 = 47%** | 5% |
| top-60 ∩ top-60 | 25/60 = 42% | 10% |

pj2 preferentially fixed the pairs where the blind box reaches most — a **9.4× enrichment** at
k=30. It eats into pb3's best target set. It does **not** empty it: the `+0.3740` on the
*residual* says the pairs still carrying pose debt are still the ones the box reaches best.

**3.3 pj2 dominates on price, by a lot.**

| move | ΔS | bytes | **S per byte** | status |
|---|---:|---:|---:|---|
| pj2 | −0.0675451 | +32 | **−2.11e-3** | realized, byte-closed |
| ms8 | −0.0491770 | +51 | −9.64e-4 | realized, byte-closed |
| pb3 k=30 subset | −0.078235 | 232 | −3.37e-4 | **ceiling, unreachable** |
| pb3 full field | −0.124126 | 4,200 | −2.96e-5 | **ceiling, unreachable** |

**pj2 is 6.3×–71× better per byte than pb3 at pb3's own unreachable ceiling.** The caveat
travels: pb3's rows are ceilings at `η=1`, pj2's is a realized measurement. pb3's *absolute*
ceiling (−0.127 = 17.5% of the gap) remains far larger than pj2's realized −0.0675, which is
why the family is not closed — but any ranking by price puts pj2 first and it is not close.

**3.4 pj2's headline is tail-concentrated — the pb3 lesson, applied to pj2.**

| | share of pj2's total d_pose reduction |
|---|---:|
| top 1% of pairs (6 of 600) | **70.33%** |
| top 5% (30) | 89.68% |
| top 10% (60) | 94.41% |

Median per-pair reduction **11.70%**; mean-of-means **50.62%**. pb3 §6 item 2 warned that *"any
future arm quoting a mean over this distribution is quoting the tail."* pj2's win is real and
byte-closed — the point is narrower: **its 49–51% is not a typical pair's experience**, and
since pb3's break-even is stated as a *mean* d_pose reduction, a mean-vs-median confusion here
would mis-price the whole family. Flagged for pj2's owner; not a defect in pj2's result.

---

## 4. `ddm_bo1` — the contradiction was already resolved. Verified, not re-landed.

My brief said a landed memo asserts "0.00% irreducible" while `ddm_cr2r`/#889 measures a
seg-only base 6.36–148.98× pose-hostile. **Checked at source: `eed9c61c81` already landed both
the §0 revision and §7.2.** The resolution:

> §1.3's optimum is over **range(A_0)** — `δ_0` free to be anything. `cr2r`'s penalty is
> measured with the **shipped carrier family F** (~8-dim: warp ξ + a,b). Both true ⇒ the
> 6.36–148.98× lives entirely in the gap between F and range(A_0), not in the scorer.

**This is sound and it is the right resolution.** They measure different sets: bo1 bounds
cancellability in a **rank-≤6 readout with δ_0 unrestricted**; cr2r measures what an ~8-dim
shipped family actually achieves. Different feasible sets, different answers, no contradiction.
bo1's §0 already carries the caveat verbatim ("for a δ_0 free to be *anything in range(A_0)*").

**Landing another correction would have been duplicate honesty debt.** What *is* still owed is
bo1's own §8 item 2: **F1, the rung-3 re-measurement, $0, and "the only thing that can overturn
§1.3."** It is unfired. That is the live item, not the reconciliation.

---

## 5. `ddm_cb2` — the ladder exists; what is owed is a scorer slot and a better proxy

The photometric-risk ladder is **present** (7 rows, 5 pairs, full camera resolution,
L ∈ {12,10,9,8,7,6,5}), recovered by `eed9c61c81`. §8's n600 gate on
`cb2_levels08_archive.zip` (254,652 B, sha `a9e99a69…9d67a`) is **BLOCKED-ON-SLOT** — the single
n600 scorer slot was held by `ddm_gd3` throughout this unit. I did not fire it.

**Coordination with `ddm_ix1`: no duplication, and they compose.** ix1 has **landed**
(`fd8897abff`) and is a *different axis* — LAYOUT/packing, lossless, `d_seg`/`d_pose` invariant
by construction, −6,144 B on `dc1_fold`. cb2 is GRANULARITY, lossy, −105,657 B at L=8 with a
d_seg cost that is exactly what the blocked gate would measure. Lossless layout composes with
lossy granularity; neither arm's result is contained in the other's.

**The ladder's real defect, and the better ladder (pre-registered, not measured here).** cb2
self-labels the proxy `ASSUMED` and says so plainly: *"SegNet reads regions and its argmax flips
at boundaries, so photometric mean is a weak predictor."* `ddm_pc2` has since made that precise:
interiors contribute **0.058%** of flips; **Road participates in 87.8%** of all 458,738 flips and
**Road↔Lane alone is 49.2%**; `err_rate ∝ area^−1.26`. So a **global mean |Δ| over all pixels is
the wrong statistic** — the scorer only reads a codim-1 separatrix. The ladder should be re-run
weighting |Δ| by distance to the nearest GT class boundary (and per-EDGE, not per-class, since a
per-class table splits one separatrix across two rows). That is scorer-free given a cached GT
argmax, and it would convert cb2's `ASSUMED` row into a `DERIVED` one. **I did not run it** — it
needs per-L camera renders I did not produce, and inventing an ordering here would be worse than
handing over the specification.

---

## 6. `a6987ed5e93b04def` = `ddm_bs2` — harvested, and it was carrying a false claim

Never checkpointed; ran 2026-08-02T00:48Z→01:21Z (~33 min); zero git commands. Recovered from
transcript and **verified against disk** in three buckets.

**LANDED** (`9f45920dca`, a recovery pass 25 min after it died): `lane_guard.py` (+468/−8),
`test_lane_guard.py` (+288, 25→44 tests), its memo, a DAG update. Trainer wiring landed too but
was **absorbed into `06fa0ad37d`** (`ddm_df1`'s unrelated CLI fix) — bs2's wire-in has no commit
of its own.

**WAS UNCOMMITTED — now landed (`3a90117ef0`).**
`src/tac/witness_dsl/spec_tr1_renderer_20260728.py::lever_lane_guard_ratchet`, +52/−0.
**Both `9f45920dca`'s message and the memo §6 assert this landed. It had not.** Verified before
landing: one hunk, byte-identical to bs2's own edit, no sibling content absorbed, imports and
constructs. Per the triality rule a lever is not built until it is a `Lever` factory — so the
DSL leg was missing for a day while the record claimed otherwise.

**RECOVERED FROM THE TRANSCRIPT — and one of them was wrong.** bs2's round-1 self-review found
two real defects and *fixed* them, but the heredoc appending their regression guards was blocked
and then killed, so **both fixes shipped unguarded.**

- non-finite guard: **passed as written**, landed as written.
- `lambda_max` guard: **failed — and the TEST was wrong, not the code.** The fix is plumbed
  (`lane_guard.py:775`, `lambda_max=cfg.lambda_max`). Two defects in bs2's test, both MEASURED:
  **(a)** it flipped exactly one pixel per gate, holding `realized_lane_s` constant ⇒ σ=0 ⇒ the
  calibrator short-circuits to `'degenerate input'`, k=0.0 for every `lambda_max`;
  **(b)** after repairing the fixture it *still* failed, because **k's `lambda_max` dependence
  SATURATES** — `k(0.2)/k(5)/k(50) = {1.975543542, 2.269919527, 2.269919527}` at burn-4
  σ=0.00142148 (2 distinct) but `{2.915180906}`×3 at the σ an 8×8 fixture can reach (1 distinct).
  Requiring k to differ is regime-dependent and false-fails correct code.
  **Repaired** to assert the actual defect — a dropped kwarg — by spying the
  `derive_ratchet_budget` call. **MUTATION-TESTED:** deleting `lambda_max=cfg.lambda_max` turns
  it RED; `lane_guard.py` restored byte-identical. 46 tests pass.

This is the second instance this session of *a landed record asserting more than the disk holds*
(the first being pb3's own §5, honestly stamped OWED). bs2's memo §7 also leaves standing: the
ratchet has **never run inside a real trainer**, and the **distortion side of every rate row is
unmeasured**.

---

## 7. The structural cause, paid (`09fca46f37`)

`tools/subagent_checkpoint.py` carried `step · status · files_touched · next_action · notes` —
five fields that all answer *"where do I resume"* and none that answers *"what did we learn."*

Added `findings`: additive, legacy-compatible (pre-existing rows load unchanged through every
reader — tested), `--finding` repeatable and **deliberately not comma-split** (findings are
prose; prose contains commas — `--files-touched`'s split is wrong for them), plus
`read --findings`, the knowledge log the store could not produce, per agent or fleet-wide.
`CHECKPOINT_FINDINGS` composed into `tac.subagent_contract.standard_contract`, so every future
dispatch carries the requirement. 28 new tests; 143 pass; ruff clean; contract-integrity gate
PASSes; the deliberate block-count assertion updated 28→29 with its reason recorded inline.

**I corrected my own motivating claim mid-flight.** I first wrote that `ddm_gd2`'s blocker was
"permanently lost." It was not: `ddm_gd3` spent a whole unit re-deriving it (`db3abc5b4a`).
**The bug class is PAID REDISCOVERY, not permanent loss** — the cheaper claim and the true one.
Dogfooded: this arm's own checkpoints carry findings.

---

## 8. Verdict-scope ladder

| claim | scope | basis |
|---|---|---|
| pb3 break-even hardens **exactly 1.7308×** vs pj2's base | **DERIVED EXACT** | rate term exact; both bases MEASURED n600 |
| pb3 ceiling −0.227754 → −0.126923 | **MEASURED + one ASSUMPTION** | n600 join; `h_Z`-invariance, bracketed 45×–47× |
| ceiling does **not** fall below the price ⇒ §5 survives | **DERIVED** | robust across both bracket ends |
| pj2 ∩ pb3 correlation `spearman +0.5320` / 47% top-30 overlap | **MEASURED** | n600 join, INSTANCE (this vehicle pair) |
| pj2's reduction is tail-concentrated (top 1% = 70.33%) | **MEASURED EXACT** | `final_pj2` vs `final_ms8_reference`, n600 |
| pj2 is 6.3×–71× better per byte than pb3's ceiling | **DERIVED** | realized-vs-ceiling; caveat travels |
| bo1's contradiction is resolved and the resolution is sound | **VERIFIED_VIA_SOURCE_INSPECTION** | `eed9c61c81` §0 + §7.2 |
| bs2's lever was uncommitted while two artifacts claimed it landed | **MEASURED EXACT** | `git diff` + import check |
| bs2's `lambda_max` guard was over-asserting, not detecting | **MEASURED** | k-saturation table + mutation test |
| a boundary-weighted ladder would beat cb2's global-mean proxy | **INFERRED** | from `ddm_pc2`'s separatrix geometry; **unmeasured** |
| pb3 §5 realized `η` | **UNMEASURED — still OWED** | needs the n600 scorer slot |

---

## 9. NEXT-IF-RESUMED

1. **pb3 §5, at the re-scoped bar.** The falsifier is now **`η ≥ 0.006124`** (was 0.0047) and
   the ceiling is **−0.126923** (bracket −0.1269…−0.1316). **Do not soften it, and do not
   report the ceiling as the result** — that is bp2's error one level up, and it is now also
   pj2's tail-vs-median trap. **Measure on the archive that will actually ship**, not on
   `v4d_composed_pb2_bestof`: pb3's entire §1–§4 is scoped to a vehicle whose pose is 1.73×
   worse than pj2's.
2. **Start with the k=30 subset, not the full field.** 232 B for 77.3% of the ceiling gain —
   18× cheaper, and it needs `η` to clear a bar ~18× lower. If the subset cannot clear it, the
   full field cannot either, and one measurement closes the family.
3. **Re-measure `g1` on the shipping vehicle.** Every `γ` above inherits bp2's `g1`, measured
   against a *different* pose residual. The `h_Z`-invariance assumption is bracketed, not
   verified; one re-measure of `grad_blind_l1` on the live base retires it.
4. **`ddm_bo1` F1** — $0, rung-3 re-measurement, "the only thing that can overturn §1.3."
   Unfired, and bo1's §8 ranks it above everything else in that memo except F3.
5. **`ddm_cb2` §8's n600 gate** the moment the slot frees — `cb2_levels08_archive.zip`,
   ADOPT if `d_seg < 0.00501579`. Then the boundary-weighted ladder of §5 above, which is
   scorer-free and converts cb2's one `ASSUMED` row into a `DERIVED` one.
6. **Task-ledger debt, unowned:** `#871` still `in_progress` and `#822` still `pending` in
   `canonical_task_status.jsonl` though `#822` was **tested and REFUTED** (r=+0.9697 over 64
   paired gates). bs2 could not close them. Per `m89` the harness TaskList and the repo ledger
   are different stores — cite content, not bare ids.
7. **Two guards I filed rather than fixed.**
   (a) `check_subagent_dispatches_use_checkpoint_discipline` has 3 live violations
   (`gd3`/`MAIN`/`ix1` commits) and its threshold test sits at **50 against a 30 cap** —
   pre-existing, and it is the exact symptom §7 addresses at the root. **Raising the threshold
   would be the silent-guard anti-pattern.**
   (b) **The Catalog #340 staging guard SELF-MATCHES.** It refused my commit citing
   `sister='ddm_rs2'` against **my own** checkpoint 0.8 min old. Same genus as the `pgrep`
   self-match (`m50`): *a probe that counts the prober.* Committed via the documented
   paired-env bypass with the self-match as the rationale. The fix is one predicate — exclude
   the committing agent's own `subagent_id` — but it is a shared hot file and belongs to
   whoever owns #340.
8. **A pre-existing DSL suite failure needs a positive control before anyone calls it green:**
   `test_taskspace_inverse_stack_receipt::test_canonical_sources_strictly_reopen_after_regeneration`
   → `CensusError: exact V9/PBR2 pair window or renderer identity differs`. INFERRED unrelated
   to bs2's additive lever; **unconfirmed** — nobody stash-and-reran it.

---

*STORES CONSULTED:* `ddm_pb3_parametric_blind_set_20260802.md` + `reports/ddm_pb3/ceiling_n600.json`
· `reports/ddm_bp2/reach_n600.jsonl` (re-derived, not recalled) ·
`ddm_pj2_pose_scale_degeneracy_20260802.md` + `/Volumes/VertigoDataTier/pact/ddm_pj2_20260802/`
· `ddm_bo1_base_objective_menu_order_20260802.md` §0/§7.2 ·
`ddm_cr2r_ep854_pose_resolve_refuted_matched_control_20260802.md` ·
`ddm_cb2_codebook_race_20260802.md` §5/§8/§9 · `ddm_ix1_index_compaction_ladder_20260802.md` ·
`ddm_pc2` (separatrix geometry) · `ddm_na1` (PR130 = 191,052 B) ·
`ddm_main_friction_audit_20260802.md` §1 · `src/tac/optimization/lane_guard.py` ·
`docs/operating_manual_craft_handoff.md` · `tac.canonical_equations.gap_decomposition_against_floor_20260802`.

*COMMITS THIS UNIT:* `09fca46f37` (checkpoint `findings` field) · `3a90117ef0` (ddm_bs2 recovery).
**Pointer UNMOVED at `dc1_fold` S = 0.8983775.**
