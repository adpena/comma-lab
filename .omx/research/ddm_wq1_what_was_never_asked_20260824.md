# ddm_wq1 — twenty arms shared one probe shape, and it cannot see the training objective

**Type: DERIVED synthesis over MEASURED receipts.** This arm measures nothing on a scorer. No Modal,
no Metal, no scorer run, no dispatch. Cost $0.
`score_claim: false` · `promotion_eligible: false` · `pointer_moved: false`.

`verdict_scope`: **the dx2 object** (archive sha
`976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`, 180,368 B) and the 96-memo
dx2-era corpus `.omx/research/ddm_*2026082{1,2,3,4}*.md`. A different object has a different residue
map and a different table.

**Loop-until-dry: 6 rounds. Round 6 produced no new rows.**

STORES CONSULTED: `.omx/research/` (10,710 files; the dx2-era subset enumerated as
`ddm_*20260821*.md` … `ddm_*20260824*.md`, 96 memos) · `ddm_fb1_sub012_feasibility_bound_20260823.md`
@ `9c137a91ed` · `ddm_tac1_two_axis_composition_20260823.md` ·
`ddm_sy2_composition_synergy_deep_pass_20260823.md` @ `fe2ba12dc2` ·
`ddm_ar1b_archive_residue_purchase_20260822.md` @ `e864cb4ab4` ·
`ddm_dg2_diagonal_distortion_verdict_20260824.md` · `ddm_dc1_decode_time_compute_20260821.md` ·
`ddm_mst1_manufactured_stage_split_20260822.md` · `ddm_msr1_manufactured_seg_reduction_20260823.md` ·
`ddm_tx1_toolbox_crosswalk_20260819.md` §0 (exchange rate **CITED, not re-derived**, per `#1207`) ·
`ddm_rb1` · `ddm_vf1` · `ddm_nl1` · `ddm_db1` · `ddm_hg1` · `ddm_et1` · `ddm_ws0` · `ddm_ws1` ·
`ddm_hr3` · `ddm_lq1` · `ddm_ap1` · `ddm_ld1` · `ddm_nt1` · `ddm_jo1` ·
`ddm_tc1_tr1_lifecycle_spec_20260817.md` · `ddm_rg5_rate_gradient_sign_20260801.md` ·
`ddm_rsf1_rate_surrogate_fidelity_20260801.md` · `ddm_gd1_generic_default_census_20260731.md` ·
`ddm_gc15` · `ddm_gc17` · `ddm_mt1` · `ddm_na7` · `ddm_pu1` · `ddm_sp1` · `ddm_rp1` ·
sources `src/tac/pr130_lift/train_semantic_quantized_resumable.py`,
`experiments/train_tr1_partition_renderer_mlx.py`, `src/tac/witness_dsl/spec_tr1_renderer_20260728.py` ·
`.omx/state/lever_activation_ledger.jsonl` · memories
`[[cross-regime-constant-transfer-genus-finishing-stage]]`,
`[[perfect-localization-is-worthless-the-address-is-the-tax]]`,
`[[object-change-not-jointness-is-the-composition-law]]`.

---

## 1. Answer first

**The twenty arms converged because they share a probe shape, and the shape's blind spot is exactly
the thing that produces the 250–800× signature.**

`dx2` was trained by `src/tac/pr130_lift/train_semantic_quantized_resumable.py`, whose loss is
**distortion-only** — `F.cross_entropy(logits, target)` with an optional band reweighting, plus
fixed-lattice int4 QAT. A full grep of that file for
`w_rate|rate_loss|rate_term|lambda_rate|entropy|rate_model|posenet|pose_loss|d_pose|w_pose` returns
**zero rate-penalty sites and zero pose sites**. Packed size is *measured* post-hoc
(`qat.packed_size`, `:1446`), never differentiated. **MEASURED (source read).**

Recomputed from components, never a display (`#877`):

| term | value | **share of shipped S** |
|---|---:|---:|
| rate `25·180,368/37,545,489` | 0.1200996476567398 | **81.028%** |
| seg `100·0.00020139` | 0.020139 | 13.587% |
| pose `√(10·0.00000637)` | 0.007981227975693965 | 5.385% |
| **S** | **0.14821987563243377** | |

**86.413% of the shipped score is produced by two terms the training objective never saw.**
That is the finding. Everything below is its consequences.

---

## 2. The probe shape, stated as a constraint (with evidence, not vibes)

> **Take the CONVERGED dx2 object; hold its TRAINING OBJECTIVE fixed; perturb one section,
> parameter, ordering, or address POST-HOC; byte-close; measure.**

Three legs, each with its receipt:

**Leg 1 — POST-HOC.** No arm re-trains under a changed objective. `jf1`/`dg2` refit the HPAC
*probability model* to a changed field; that is fitting a coder to an output, not changing the loss
that produced the output. The one arm that built a joint objective, `ddm_jo1`, is **BLOCKED** on five
named blockers and never ran, on a different vehicle (rc2). *MEASURED.*

**Leg 2 — SINGLE-OBJECT.** The comparison is dx2 vs dx2-perturbed. The four whole-body replacements
(`rc1`/`ri1`/`nr1`/`ni1`) did change the object and failed on **distortion** (S = 17.31 / 27.80) — a
different failure mode from the rate-vs-distortion refusals, not a counter-example to the shape.
*MEASURED (sy2 inventory).*

**Leg 3 — EXACT-FIELD REPRODUCTION.** The alternative-representation arms were required to reproduce
dx2's exact 117,964,800-position categorical field. The common killer is **not the generator**:

| arm | best receiver-closed container | vs the 137,986 B cap | the killer |
|---|---:|---:|---|
| `hg1` heterogeneous analytic generator | 460,408 B | +322,422 | **359,280 B unique-home residual** |
| `hr3` Fourier-coordinate INR (w8/16) | 463,601 B | +325,615 | **362,473 B residual** |
| `et1` implicit BSP generator tree | 535,761 B | +397,775 | 482,678 B topology packet |
| `ws1` explicit boundary-curve worldsheet | 918,904 B | +780,918 | `FAMILY_NO_GO` |

Every generator was priced *plus the cost of exactly reproducing the incumbent field*. **The contest
requires evaluator-cell equivalence, not token identity.** *MEASURED.*

### Why the shape produces 250–800×, derived — and the derivation's prediction was already measured

`dx2` sits at a stationary point of a loss containing only the seg term. Any post-hoc perturbation
that buys rate moves off that minimum, and the damage lands wherever nothing was protecting.
**DERIVED prediction: the damage should be pose-dominated.** Measured signature:

| arm | pose share of damage | ratio to credit |
|---|---:|---:|
| `dg2` k060000 | **93.3%** | 687.3× |
| `dg2` k040000 | **93.4%** | 791.7× |
| `w72` renderer | 65.3% | 46.3× (d_pose ×303,989) |
| `ap1` carrier L1–L3 | **100%** — SegNet-inert, 0 flips in all 5 classes | 167.8× / … |

The prediction and the measurement agree. This is not a re-description of the refusals; it names
their cause and it is falsifiable (§7).

### The marginal asymmetry that makes pose the expensive axis (DERIVED)

`d(seg term)/d(d_seg) = 100` exactly. `d(pose term)/d(d_pose) = 5/√(10·d_pose) = **626.470**` at
dx2's operating point — **6.2647× the seg marginal.** The crossover is `d_pose = 2.5e-4`; dx2 sits
**39.246× below it**. So the one term nothing protects is also the one term that is marginally most
expensive to disturb. That is the whole 250–800× phenomenon in one line.

---

## 3. A corpus correction this arm owes: `tc1`'s DOMINATED label does not transfer to dx2

`ddm_tc1_tr1_lifecycle_spec_20260817.md:196,197,264` marks `--w-rate`, `--rate-model
smevr_surrogate`, and the whole jd1 pose family **DOMINATED (§1.2)**. §1.2's binding premise is
*"A zero-byte archive does not save TR1 … while d_seg sits at 3.89e-3."* Applying tc1's own method to
dx2 (**DERIVED**, exact arithmetic over two cited measured rows):

| check | tc1's TR1 object | **dx2** |
|---|---:|---:|
| best-ever `d_seg` | 0.00389011 | **0.00020139** (**19.316× better**) |
| best row, × its seg+rate budget | 3.818× | **1.2519×** |
| **free archive** (B = 0) | S = 0.397306 = **2.489×** its frontier | S = **0.028120 = 0.2343×** of 0.12 |

**tc1's premise is TRUE for TR1 and FALSE for dx2.** Carrying the label across is the
cross-regime constant-transfer genus. *Two honest caveats:* TR1 is additionally a **retired** vehicle
(tc1 point 5), and the live trainer is `pr130_lift` — so rate-in-loss on the live vehicle is a
**BUILD, not a flag flip**. The lever's absence is what matters, not the retired flag.

---

## 4. The full enumeration

| # | direction | status | evidence |
|---|---|---|---|
| **D1** | **Rate absent from the LIVE training objective** | **UNASKED on the live vehicle**; NAMED-NEVER-RACED on the retired one | live trainer grep = 0 rate sites. On retired TR1: `w_rate=0.05` fired lineage-wide; `rsf1` MEASURED the live `entropy` surrogate **anti-correlated** with shipped bytes (ρ = −0.7235, CI [−0.943, −0.227]); permutation moves the surrogate ≤4.8e-07 bits and real bytes **+16,062…+18,339 B**; `rg5` MEASURED `smevr_surrogate` **1.5–3.6× stronger** byte-descent with `cos(entropy, smevr) ∈ [−0.066, +0.020]` — **near-orthogonal**; `entropy` in **29/29** landed configs, `smevr_surrogate` **never fired**, the sum arm **does not exist** |
| **D2** | **Pose absent from the LIVE training objective** | **UNASKED on the live vehicle**; TR1's 15 jd1 pose flags **never fired** | live trainer: no `posenet`/`pose_loss`/`w_pose`. TR1 `w_pose=0.0`, `compute_pose=False` (`rg5` re-derived at file:line). `tc1:203,264` — *"all 15 flags never fired … this is why raw TR1 ships an inert 83 B pose stub"* |
| **D3** | **Exact-field reproduction is a self-imposed constraint** | **NAMED, NEVER MEASURED** | `vf1`: classification denominator **0 / 117,964,800 positions**, **0 / 113,777 coded bytes** ⇒ **0 B of 42,382 B = 0%** measured evaluator-visible credit. `db1` L120: *"scorer-cell equivalence is a weaker requirement than exact token identity; the 42,382 B demand fits inside the token member's mass … unmeasured"*. Adjacent slack: `msr1` measured **63.4%** of manufactured pixels had a 17×17 token window **exactly equal to GT** |
| **D4** | **Decode-time compute** (~1,300 s of 1,800 s unspent) | **ALREADY MEASURED, INSUFFICIENT** (family A); family B **BLOCKED** | `dc1`: real solver, **510/510** blocks exact, group-0 **+4.488810 B**; five-group bounded sample **−4.907404 B**; dc1's own best-case 600-frame projection = **2,693.286 B = 6.34%** of the ask. Family B ceiling is the whole 113,777 B but blocked on one named artifact |
| **D5** | **sy2's rank-3 born-small edge/topology carrier** | **UNASKED — the queue was never drained after both its gates closed** | `sy2`: *"the only composition with enough object mass to target sub-0.12"*, dispositioned `QUEUED-BEHIND-JF1/W96`. Both gates have since been measured and refused (`dg2` 687×/792× monotone; `w72` 46.3×). No arm re-ran the queue |
| **D6** | **Buy pose WITH bytes** | **UNASKED, bounded** | `ap1` sampled the carrier at three **coarsening** levels only (−2,742 / −5,875 / −9,035 B). The **+ direction is unsampled.** Ceiling = pose→0 = **11,986 B = 28.28%** of demand |
| **D7** | Per-pair / temporal byte allocation | **WEAKLY ALREADY MEASURED — downranked** | `hr3` line 65: per-**frame** mismatch **Gini 0.211095**, *"much less concentrated across frames than BL1's per-position cost field"* (Gini 0.99516). Concentration is **spatial, not temporal** |
| **D8** | Seed / run-to-run variance of the shipped S | **UNASKED** | **SEARCH SCOPE:** `.omx/research/ddm_*2026082*.md`. Every run-to-run figure found (`cd1`, `rr6`, `rr8`) is **wall-clock** variance, not S variance. NOT searched: `.json` receipts, the task ledger, SSD-only artifacts |

**Negative-existence claims, with scope (m53).** Round 6 searched `.omx/research/*.md` for any statement
that the live training objective omits rate and pose, and for any arm that ADDS bytes to buy pose.
**NOT FOUND within that scope.** Not searched: charters, `.json`/`.jsonl` receipts, `.py` sources
beyond the two trainers named, SSD-only artifacts.

---

## 5. The ranked table — reach ÷ cost-to-falsify

| rank | direction | plausible reach toward 42,382 B | cost to falsify |
|---:|---|---|---|
| **1** | **D1 + D2 — put rate and pose in the live loss** | **42,382 B in play.** It is an object change (sy2's law) moving ≥2 axes (fb1's arithmetic requirement), and it admits **86.413% of S** into the objective for the first time | **MEDIUM** — one live-trainer run + byte-close + one advisory n600 row |
| **2** | **D3 — measure the field's evaluator tolerance** | **113,777 B ceiling** (the whole token stream); a cheap discriminator exists | **LOW** — ~5 advisory rows, $0 local, no Modal |
| **3** | **D5 — drain sy2's rank-3 queue** | 42,382 B; sy2's own named object-mass candidate | **HIGH** — a born-small build + materialization + n600 row |
| 4 | D6 — buy pose with bytes | 11,986 B (28.28%) — second axis only, never a close | LOW-MEDIUM — one carrier re-solve |
| 5 | D8 — seed variance | unquantified; recalibrates every prior verdict | **FREE** if it rides a retrain |
| 6 | D4 — decode-time compute | 2,693 B measured ceiling — **cannot close** | n/a, adjudicated |
| 7 | D7 — per-pair allocation | downranked on hr3's Gini 0.211 | n/a, downranked |

---

## 6. Charter-grade next-arm specs

### SPEC A (rank 1) — `ddm_ro1`: the three-term objective, one run

**Measurement.** Add two terms to `train_semantic_quantized_resumable.py`'s loss: a differentiable
code-length surrogate for the *shipped* coder, and the PoseNet term. Train one arm at matched budget
against a byte-identical distortion-only control (`w_rate = 0`, `w_pose = 0` ⇒ byte-identical, the
control the TR1 lever already documents). Byte-close both. One advisory n600 row each.
**OPTIMAL FORM:** the surrogate must be **coder-matched**, not a marginal histogram — `rsf1`/`rg5`
measured the marginal form permutation-blind (≤4.8e-07 bits vs +16,062…+18,339 B) and near-orthogonal
to the matched form; and `rg5` §5 names the **sum** arm as the indicated third variant. Shipping the
marginal form would reproduce a known-defective mechanism (charter-time optimal-form law).

**Bar.** ΔS < 0 on the byte-closed advisory pair, with rate credit exceeding distortion damage — i.e.
a refusal ratio **< 1**, against a corpus where the best measured ratio on any dx2 lever is **46.3×**
and the cheapest residue rung is **167.8×**.

**Falsifier.** The rate-aware arm's byte-closed archive is not smaller than the control's at matched
`d_seg`, **or** the joint arm's ratio stays above 46.3× — in which case the objective is not the
crux and the 250–800× law is a property of the representation, not the training. Either result is
decisive and neither has ever been measured on this vehicle.

**Cost.** One training run + two byte-closes + two advisory n600 rows. No Modal.

**Prior-law prediction to pre-register.** The refusal ratio falls by at least an order of magnitude,
because the damage is currently **93%** pose and pose enters the loss for the first time.

### SPEC B (rank 2) — `ddm_tv1`: the field's evaluator tolerance curve

**Measurement.** Reassign `k` uniformly-random token positions for `k` on a log ladder
(10³, 10⁴, 10⁵, 10⁶, 10⁷ of 117,964,800), push each through the **real receiver**, and measure `d_seg`
and `d_pose`. No addressing, no coding — this measures **tolerance alone**, decoupled from the address
tax that killed `mf1` (+35,969 B) and `ld1` (every rung bigger).

**Bar.** Fill `vf1`'s empty denominator. A curve flat out to `k ≈ 10⁶` means ≥0.85% of the field is
evaluator-inert and the exact-reproduction constraint that killed all four alternative
representations is **provably over-strict**.

**Falsifier.** `d_seg` rises approximately linearly in `k` from the first rung — the field is
everywhere load-bearing, D3 is dead, and the `hg1`/`hr3`/`et1`/`ws1` residuals were an honest price,
not a self-imposed one.

**Cost.** ~5 advisory n600 rows, $0 local, no Modal, no Metal.

**Honest counter-evidence to carry into it.** `ld1` and `mf1` measured that tolerance does **not** pay
once the changed subset must be **named**. SPEC B does not contradict them; it measures the numerator
they never isolated. A positive result licenses a *representation whose coding units are already
cells* — not a lossy edit to the current field.

### SPEC C (rank 3) — `ddm_bs1`: drain sy2's rank-3 queue

**Measurement.** One complete born-small edge/topology carrier on the current object: actual coder,
real receiver, exact B/H/W, Seg, Pose, bytes. No `nr1`/`rc1` agreement proxy is admissible (sy2's own
condition). **Bar:** complete archive ≤ 137,986 B at its own measured distortion. **Falsifier:** the
container clears 300,000 B before a scorer is fired — the gate that closed `et1` (535,761 B) and
`hg1` (460,408 B). **Cost:** one training/materialization + one n600 row; dollar cost unestimated —
sy2's own words, not softened here.

---

## 7. Prior-law prediction — adjudicated

> **Predicted:** at least **2** genuinely-unasked directions survive the corpus check, and at least
> one is about the OBJECTIVE or the REPRESENTATION CLASS rather than a perturbation of dx2.
> **Falsifier:** every direction turns out already-measured or excluded, in which case the honest
> verdict is that the *problem*, not the method, is closed.

**CONFIRMED. Count = 5** (D1, D2, D3, D5, D6). Two of the five — D1 and D2 — are about the
**objective**; one — D3 — is about the **representation class**. The falsifier did not fire.

**Against my own verdict.** Three of the five (D3, D5, D6) were **named by prior arms** (`vf1`,
`db1`, `sy2`, `ap1`) and are unasked only in the sense that no measurement exists. I did not discover
them; I found them un-drained. Only **D1 and D2 are genuinely unnamed** — round 6 searched
`.omx/research/*.md` for any statement that the live training objective omits rate and pose and found
none. "Five survive" should be read as "two unnamed, three un-drained."

---

## 8. NOT CLAIMED

- This does **not** prove sub-0.12 reachable, and it prices no route. Every reach figure is a
  **ceiling**, not an achievement. **No arm has achieved any of them.**
- It does **not** claim the objective is *the* crux. It claims the objective is the one thing the
  probe shape structurally cannot see, and it names the measurement that decides it.
- **D1's reach is DERIVED, not measured.** No rate-aware or pose-aware training run exists on this
  vehicle. The rate-surrogate fidelity numbers (ρ = −0.7235, 1.5–3.6×, cos ≈ 0) are `[macOS-CPU
  advisory]` static-field byte responses on the **retired TR1** lineage across **4 fields**, and
  `rg5`'s own residual #2 states *"v4d — our actual frontier vehicle — is in neither rsf1's field set
  nor mine."* They are evidence about a direction's sign and strength, nothing more, and **no effect
  size transfers** (`na7:219` says exactly this).
- `rg5`'s own binding residual #3 — *"the `d_seg` price of the byte reduction is unmeasured"* —
  is **not** discharged by this memo.
- The `ap1` carrier marginal (3,388× break-even) is a **SECANT over 2,742 B on a convex response**
  (L1/L2/L3 at 1.117e-4 / 1.245e-4 / 4.348e-4 S/B). A secant on a convex curve **overstates** the
  local slope, so it is an **upper bound** on the shipped carrier's distance from its own optimum.
  Label: DERIVED-FROM-SECANT.
- `tc1`'s §1.2 dominance verdict is **correct for TR1**. Only its **transfer** to dx2 is refuted.
- Triples, and any object other than dx2, are out of scope.
- `fb1` §9.1's retraction of the "rj1 W64 refused 3.51× / d_pose 97.70%" figures propagates:
  **`ddm_tac1` §6 still carries both**, inherited before the retraction landed. Flagged here, not
  corrected — the correction belongs in tac1's own append-only block.
- No score, no pointer claim, no promotion.

---

## 9. Payload custody (P0 — persisted, not prose-only)

`/Volumes/APDataStore/pact/ddm_wq1_never_asked/` (Vertigo NOT used — it is at 100%).

| file | bytes | sha256 |
|---|---:|---|
| `ddm_wq1_never_asked_table.json` | 16,745 | `b3158a8669f6ed4d23ed87a9dfeb520aed82132d22f7c70968ab6b83b8f03841` |
| `ddm_wq1_ranked.csv` | 753 | `1a82c88a5298fe85fbab4445354c32a4a5a0753620cd3d2fbc165ce5bc457a2b` |
| `MANIFEST.json` | 277 | `fe92f4346d9d7bd49687ac184cee66b3db27f3c351a863c480b254cf4ed4bfd9` |

Generator `experiments/ddm_wq1_never_asked_table.py`. **Determinism verified:** a repeat run produced
byte-identical output (same three sha256).

---

## 10. Own-vehicle frontier

**dx2 — S 0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]`** — **UNMOVED by this memo**
($0, no measurement fired). Gap to 0.12 = 0.028220 ⇒ shed **42,382 B** at fixed distortion, or
**150 B** at zero distortion.
