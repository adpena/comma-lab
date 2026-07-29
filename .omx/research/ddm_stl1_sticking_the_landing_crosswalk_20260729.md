# ddm_stl1 — "Sticking the Landing" (Roeder/Wu/Duvenaud, NeurIPS 2017) crosswalk

UTC: 2026-07-29 · Arm: ddm_stl1 (bounded $0 paper-crosswalk; NO launches, NO scorer jobs) ·
Harvested by: ddm_stl1 (Opus) · Evidence class: FROM-LITERATURE + retro-typing of ON-DISK receipts.
`research_only=true` · `score_claim=false` · `promotion_eligible=false`.
**Pointer 0.1910828242 [contest-CPU] UNMOVED — this is MEANS, not a score move.**

Paper: Roeder, Wu, Duvenaud. *Sticking the Landing: Simple, Lower-Variance Gradient Estimators
for Variational Inference.* NeurIPS 2017. arXiv:1703.09194.
Operator drop (07-29): proceedings.neurips.cc/paper/2017/hash/e91068fff3d7fa1594dfdf3b4308433a.

---

## §0 RECALL-FIRST — what is already in our corpus (honest diff, done BEFORE claiming novelty)

Grepped `.omx/research/`, `.ralph/`, `docs/` for `sticking the landing | roeder | pathwise |
reparameteriz* | score.function | DReG | 1703.09194`. Findings:

- **STL itself (Roeder 2017), the variance-at-optimum theorem, the "drop the score term"
  trick, and DReG/STL-bias follow-ups: ABSENT.** No prior consultation. This crosswalk is NEW.
- **The score-function-vs-pathwise DICHOTOMY is ALREADY in corpus**, in
  `policy_gradient_variance_reduction_survey_20260712.md` (34 KB, Fable/MAIN 07-12):
  - line 69 cites Mohamed et al. *Monte Carlo Gradient Estimation in ML* (JMLR 2020,
    arXiv:1906.10652) — the score-function / pathwise / measure-valued taxonomy.
  - line 15 / 155: the adopted "division of labor" — **pathwise/adjoint where the rollout is
    differentiable and known; likelihood-ratio only at genuinely discrete interfaces**; and the
    explicit guardrail "pathwise is **not universally strictly lower variance**."
  - **line 140 [DERIVED]: "Both ES and policy gradients are likelihood-ratio gradients of
    different sampling distributions"** — i.e. our corpus ALREADY types ES as
    score-function-class. The P2b retro-typing (§2) does NOT get to claim that as novel.
- The `codex_findings_onpolicy_forward_surrogate_*` + `warmstart_organ_n1_rl_*` cluster uses
  adjoint/pathwise costate gradients but never names STL or the variance-at-optimum result.

**Net recall verdict:** the DICHOTOMY (ES = likelihood-ratio class; use pathwise where
differentiable) is established. What STL ADDS, and what this memo contributes, is the
**variance-AT-OPTIMUM axis** the existing crossover lacked: the existing survey keys its
ES-vs-structured crossover on horizon / credit-assignment / active-support size; STL supplies
the **proximity-to-optimum (operating-point |gradient|)** axis and a THEOREM basis for why
score-class search collapses at a *plateau* specifically. That axis is exactly the regime our
post-burn endpoints (P2b, P2c, terminal polish) live in. This memo is a *sharpening*, not a
rediscovery — and it names the falsifier.

---

## §1 DELIVERABLE 1 — mechanism, variance-at-optimum theorem, known failure modes [FROM-LITERATURE]

### 1.1 The total-derivative decomposition
Reparameterized ELBO with `z = t(ε, φ)`, `ε ~ s(ε)` (base noise, φ-free), variational params φ:

```
L(φ) = E_ε[ log p(x, t(ε,φ)) − log q_φ(t(ε,φ)) ]
```

The reparameterization gradient of the integrand splits into two pieces by the chain/total
derivative w.r.t. φ (Roeder eq. 2–4):

```
∇_φ [ log p − log q_φ(z) ]  =  (∂/∂z)[log p − log q_φ(z)] · (∂z/∂φ)     ← PATH term (pathwise)
                              −  (∂/∂φ) log q_φ(z) |_{z held fixed}       ← SCORE term (direct)
```

- **PATH term:** how the objective moves because the *sample* `z = t(ε,φ)` moves with φ.
- **SCORE term:** the *explicit* dependence of `log q_φ` on φ at a frozen sample =
  `∇_φ log q_φ(z)`, the classical score function.

### 1.2 Why dropping the SCORE term keeps unbiasedness
The score function has zero expectation under its own density:
`E_{q_φ}[ ∇_φ log q_φ(z) ] = ∫ q_φ ∇_φ log q_φ = ∇_φ ∫ q_φ = ∇_φ 1 = 0`. So the STL estimator
(PATH term only) has the **same expectation** as the full estimator — **unbiased for any φ.**

### 1.3 The variance-at-optimum theorem (the "landing")
At the optimum `q_φ = p(z|x)` (family contains the posterior), the *true* gradient is 0. There:
- the **full** estimator still fires a per-sample nonzero SCORE term that only averages to 0 →
  **nonzero variance at the optimum** (it "jitters over the landing");
- the **STL/path-only** estimator evaluates to **exactly 0 for every ε** at `q_φ=p` → **zero
  variance at the optimum.** As φ approaches the optimum the estimator variance → 0 smoothly:
  it *sticks the landing*. This is the paper's core result and the whole benefit.

### 1.4 Generalizations
- **Mixture / flow / IWAE (importance-weighted) posteriors:** Roeder gives the path-only form
  for these; the score term is dropped per-component.
- The estimator is a special case of a **control variate**: STL = full estimator minus the
  score term used as a zero-mean control variate with **coefficient fixed at 1** (Geffner-Domke
  framing, §1.5).

### 1.5 KNOWN FAILURE MODES — adopt with eyes open (the follow-up literature)
1. **Naive STL on IWAE / multi-sample objectives can INCREASE variance.** Rainforth et al.,
   *Tighter Variational Bounds are Not Necessarily Better* (ICML 2018, arXiv:1802.04537) showed
   the IWAE inference-network gradient SNR degrades as O(1/√K). Tucker et al., **DReG — Doubly
   Reparameterized Gradient Estimators** (ICLR 2019, arXiv:1810.04152) showed the *naive* STL
   score-dropping on IWAE fails because the score term **re-enters through the importance
   weights**; the fix is to reparameterize a SECOND time, giving a strictly-lower-variance
   estimator. **Lesson: any multi-sample / importance-weighted rate objective must use DReG,
   not bare STL.**
2. **Far from the optimum / model misspecification, the SCORE term carries SIGNAL, not just
   noise.** Geffner & Domke (*Approximation-based variance reduction for reparameterization
   gradients*, NeurIPS 2020, arXiv:2007.14634; and the large-ensemble control-variate line,
   arXiv:1810.12482) show the **variance-minimizing coefficient on the score term is generally
   between 0 (STL) and 1 (full)**, and equals 0 **only** at/near the optimum or when the family
   contains the posterior. Under misspecification (posterior NOT in the family — our usual
   world), fixed-coefficient STL can be *higher* variance than full or than the learned-
   coefficient "double control variate." **Lesson: STL is a NEAR-OPTIMUM tool; do not hard-wire
   it ON early in training or under a misspecified family.**

These two failure modes are exactly why the forward-row default (§3) is derived, not
cargo-culted.

---

## §2 DELIVERABLE 2 — retro-typing the P2b ES negative through the STL lens [$0, receipt on disk]

**Receipt:** `/Volumes/VertigoDataTier/pact/ddm_pb1_20260729/p2b_mc400_diagonal_receipt.json`
(schema `ddm_pb1_p2b_mc400_diagonal_receipt.v1`; `[macOS-CPU advisory]`, `score_claim=false`).
Design source: rv1-R1 #400 `mc_finisher` diagonal on the tr1 renderer stream, bounded
instantiation. Seed 20260729, budget 8, σ_rel_init 1.953e-3.

**Measured facts (verbatim from receipt):**
- base `joint_action = 28.86390`, best `28.82283`, `delta_vs_base = −0.0410733` (`strict_improvement_found=true`).
- **the entire gain is the pose √-term:** base `d_pose 78.19574` → best `77.96472`
  (Δ = −0.231, **0.30 % relative**); `√(10·78.19574)=27.9635` → `√(10·77.96472)=27.9222`,
  Δ = −0.0413 ≈ the whole −0.0411. (`joint_action` at this pre-pose-solve endpoint ≈ contest S,
  dominated by the catastrophic √(10·d_pose)≈27.9 term.)
- **seg moved the WRONG way:** base `d_seg 3.888796e-3` → best `3.891432e-3` (**+2.6e-6, worse**).
- 8 trials, **only 2 accepted** (σ_scale 0.6, 0.9), both pose-only; total wall **2023.15 s**.

**Typing [DERIVED]:** the `#400 mc_finisher` diagonal is an **evolution-strategy / Monte-Carlo
estimator** = Gaussian smoothing of the frozen-scorer objective. Its update is a **likelihood-
ratio (score-function-class)** gradient of a *sampling* distribution
`g_ES = E_{u~N(0,σ²)}[(u/σ²) f(θ+u)]` — precisely the class our own corpus already assigns to ES
(`policy_gradient_variance_reduction_survey_20260712.md` line 140). By contrast the aimed
Jacobian edits queued for P2c (atlas-aimed channel-sign singles; GN direction rounded to
lattice) use the **exact rank-4 head Jacobian** (`segnet_recursive_fractal_factorization`) —
a **pathwise/adjoint** derivative of the deterministic frozen scorer.

**The STL variance-at-optimum theorem, read as a search law at a PLATEAU:** at a low-|gradient|
endpoint the *mean* signal a score-function estimator recovers → 0, but its **variance does NOT
vanish** — it retains an O(f/σ) floor set by the objective's second moment. The path-only /
aimed estimator, exploiting the exact analytic Jacobian, still points along the true descent
direction with no such floor. The receipt IS the theorem realized: **2023 s of ES bought a
0.30 %-relative pose nudge and moved seg backwards** — full variance cost paid for a near-zero
mean at the plateau.

### 2.1 LAW CANDIDATE (for the organ SENSE layer / canonical-equations leg)
> **Score-class search (ES / MC / likelihood-ratio) is variance-dominated and is dominated by
> aimed pathwise edits at low-|gradient| operating points (post-burn plateaus / terminal
> polish). Route the byte-budget to exact-Jacobian aimed edits there; spend score-class search
> only where the local gradient is large or the interface is genuinely non-differentiable.**

- Honesty label: **DERIVED** (from STL's variance-at-optimum theorem + the P2b INSTANCE), a
  **sharpening** of the existing crossover in `policy_gradient_variance_reduction_survey`
  (adds the proximity-to-optimum axis the horizon/support crossover lacked). NOT a novel claim
  that ES=likelihood-ratio (already in corpus).
- Verdict-scope: the P2b evidence is **INSTANCE** (this budget, this σ schedule, this stream);
  the LAW is a **FORMULATION-level** candidate until a second independent plateau instance
  confirms it.
- **FALSIFIER (preregistered):** an ES / MC / likelihood-ratio-family row that **BEATS aimed
  pathwise (exact-Jacobian) edits at matched full-pair evals at this endpoint** (P2c round-2 is
  the natural head-to-head: SparseRS/Square patches vs atlas-aimed singles at matched Q). If
  that fires, the LAW retires to "large-gradient / non-differentiable only."
- Consumer: **E2 boundary** (types why P2b booked ~0 seg and negligible pose) · **P2c round-2
  budget split** (deprioritize blind ES relative to atlas-aimed singles) · **canonical-equations
  leg** IF the second instance confirms (co9 SENSE ingestion, gc6 row 12).

---

## §3 DELIVERABLE 3 — the forward row: exact gradient decomposition for OUR rate term (gc6 row 8)

**Consumer:** gc6 §4 **row 8** — "Rate-in-loss raced arm (Ballé noise-proxy on mode-deltas)
inside row 7's window," EVENT_GATED(row 7 fires), consumers `E2` + next-vehicle SPEC; Hotz
dissent adopted as ordering ("run the waterfill first"). Also the T4 Ballé-seat question
(gc6 §3): "*soft-round + factorized prior over mode-deltas, or skip to post-hoc waterfill?*"

### 3.1 The exact decomposition — and a PREMISE REFINEMENT [DERIVED]
Charter hypothesis: "per-dim LEARNED quantization scales make the Ballé noise distribution
parameterized → the score term appears in ∂rate/∂scale → STL path-only is the candidate
default." Working the algebra on the charter's own form exposes a subtlety worth banking:

Let `d` = a mode-delta we code, `Δ` = learned per-dim step (scale), `u ~ U(−½,½)` the additive-
uniform-noise proxy sample, `p_ψ` the factorized prior. The charter's rate term:

```
R(d; Δ, ψ) = −log₂ p_ψ( (d + u·Δ)/Δ )   [+ log₂ Δ  change-of-vars, if rate measured in d-units]
```

**Key algebra:** `(d + u·Δ)/Δ = d/Δ + u`. The noise `u` enters **additively and Δ-independently**
inside `p_ψ`. So the *sampling density over the argument* `s = d/Δ + u` is a Δ-shift of a FIXED
`U(−½,½)` — its random part has **no Δ dependence**. Therefore:

```
∂R/∂Δ = −(∂ log₂ p_ψ(s)/∂s)·(∂s/∂Δ)  +  ∂(log₂Δ)/∂Δ
      =  (d/Δ²)·(p'_ψ(s)/p_ψ(s))/ln2  +  1/(Δ ln2)
         └────── PATHWISE (through s=d/Δ) ──────┘   └─ DETERMINISTIC Jacobian ─┘
```

**There is NO stochastic score term w.r.t. Δ in the uniform-noise proxy.** Equivalently: the
standard Ballé proxy posterior `q = U(width Δ)` has entropy `−log Δ` that is **constant in the
sample** — its `∇_φ log q` (the STL "score term") is either identically 0 (unit width) or
**deterministic** (learned Δ), hence **zero gradient variance either way.**

**Consequences (this is the deliverable):**
- **Fixed Δ:** `∂/∂Δ` absent entirely → pure pathwise → **STL is a no-op.**
- **Learned Δ, uniform proxy:** the "score term" is the **deterministic** `1/(Δ ln2)` Jacobian.
  Its variance is 0, so STL **cannot reduce variance** here, and *dropping* it would **BIAS**
  the scale gradient (you'd delete a real, non-noisy contribution). **STL OFF is correct;
  turning it ON is a bug, not a variance win.** The charter's premise ("learned Δ ⇒ stochastic
  score term") is directionally reasonable but the ÷Δ structure **cancels** the Δ-in-noise
  dependence — bank this so nobody re-derives it wrong.
- **Where a genuine STOCHASTIC, STL-relevant score term DOES appear:** when the *entropy model
  itself is variational/sampled*, not the main-latent scale. Two concrete variants:
  1. a **scale hyperprior** (Ballé et al. 2018): `p_ψ(s)` with `ψ = h(v + u_v)` from a
     reparameterized hyper-latent `v`. The hyper-latent's ELBO gradient is the ORIGINAL STL
     setting — `∇ log q(v)` is a per-sample nonzero score term with nonvanishing variance at
     the RD optimum → **STL path-only sticks the landing** on the hyper-encoder.
  2. a **Gaussian entropy-model** relaxation `q = N(μ_φ, σ_φ²)` on the code (some NVC variants):
     `∇_φ log q(z)` has a z-dependent (stochastic) part whose variance is nonzero at the optimum
     → **STL applies.** (Note: the naive `(d+u·Δ)/Δ` Gaussian analogue with `u~N(0,1)` also
     self-cancels; the score term needs an entropy scale NOT divided back out — i.e. an actual
     variational posterior, not a co-scaled noise.)

### 3.2 DSL lever stub spec (derived default; NOT a code landing — $0 arm)
```
lever id:        rate_stl_path_only              (next-vehicle / v10 rate-in-loss DSL)
type:            bool, score-affecting → DEFAULT OFF (safe-config discipline)
DERIVED default reason (value-provenance ladder, NOT cargo-cult):
   OFF for the additive-UNIFORM Ballé proxy (the row-8 baseline): the STL "score term" is a
   deterministic Jacobian with ZERO gradient variance → STL gives no variance win and biases
   the scale gradient (§3.1). ON is *only* admissible when the row-8 arm uses a GAUSSIAN
   entropy model OR a scale-hyperprior reparameterized latent (§3.1 variants 1–2), where a
   stochastic score term genuinely exists.
duty-to-measure (default-off-is-orphan discipline): if row 8's arm ships a Gaussian/hyperprior
   entropy model, the lever is REQUIRED to be A/B-measured (STL on vs full) at matched wall;
   else it is a documented NO-OP and MUST NOT be wired live (NO-FAKE #17 inert-flag guard).
DReG guard: if the row-8 rate objective becomes multi-sample / importance-weighted, STL-naive
   is FORBIDDEN — use DReG (§1.5.1). Register as a sibling flag `rate_dreg` gated on K>1.
ordering: honor Hotz — post-hoc waterfill (gc6 row 6) races FIRST; row-8 in-loss is the
   EVENT_GATED extension arm. The lever only matters once row 7 fires and the arm is Gaussian/
   hyperprior.
lands as: a `Lever` factory in the v10 vehicle DSL (triality: DSL=SoT, never a hand-added
   trainer flag), with `constant_refs` → this memo's §3.1 derivation.
```

### 3.3 Consumer-routing note — the N1=NO update makes this window MORE likely to fire
gc6 §5 E2 tree: node N4 (extend-vs-re-race, which carries "row-8 rate-in-loss arm" in its fold
list) was gated "only reachable with N1=YES." **NEW (07-29): P3 terminal pose landed N1=NO
(d_pose 38.06 — photometric wall confirmed).** Per the tree, N1=NO routes **pose-in-burn
conditioning to the v10 SPEC (row 12) = a next-vehicle RE-BURN.** A re-burn is an actual
training loop with in-loss terms — which is exactly the habitat of a rate-in-loss arm. So the
row-8 window shifts from "extension-only, gated on N1=YES" toward "**native to the v10 re-burn
SPEC.**" This spec is therefore *more* likely to be consumed than the E2-extension reading
alone suggested. Route this stub to the **v10 SPEC rate section**, not only to the E2 extension
ledger.

---

## §4 DELIVERABLE 4 — honest N-A sweep (where STL does NOT apply; do not re-derive this)

STL is a variance-reduction trick for **sampled reparameterization (pathwise) gradients of a
variational objective.** It is N/A wherever there is no such sampled ELBO gradient:

| surface | why STL is N/A |
|---|---|
| **Deterministic seg loss (d_seg via frozen argmax)** | No sampling in the loss path; the objective is piecewise-constant argmax disagreement. Gradients come from the **exact rank-4 head linearization (pathwise/adjoint)**, not a sampled ELBO. Nothing to "drop." STL N/A. |
| **Terminal pose solve (GN on frozen frames, e_p rank-1)** | A deterministic Gauss-Newton least-squares solve, not variational, no sampling. STL N/A. |
| **Costate / Pontryagin organ (supervised, advisory)** | Deterministic adjoint gradient — **already pathwise by construction**, no sampled entropy term. STL N/A (there is no score term present to drop; the organ is *already* in STL's preferred regime). |
| **Gradient-free search (P2b ES, SparseRS, Square, mc400)** | STL the ALGORITHM is N/A — these are not reparameterizable ELBO gradients; the score term IS the whole estimator, you cannot drop it. **But STL's variance-at-optimum THEOREM types WHY they stall at plateaus (§2) — that lesson DOES transfer.** Keep the distinction: algorithm N/A, theorem applies. |
| **warp-PREDICT / flip-label coding / exporter / partition coding** | Deterministic combinatorial coding, no sampling. STL N/A. |

**One-line guard for future arms:** STL only ever bites where the rate/entropy term is a
**sampled variational** quantity (a learned Gaussian/hyperprior entropy model). Everywhere else
in this stack the gradient is already pathwise/adjoint or the surface is deterministic — do not
re-open this crosswalk.

---

## §5 Honesty labels · falsifiers · scope · STORES CONSULTED

- **Pointer 0.1910828242 [contest-CPU] UNMOVED.** This memo is MEANS: a paper crosswalk + a
  retro-typing of an on-disk advisory receipt + a forward-row spec stub. Zero score claim, zero
  bytes, zero launches.
- Evidence: §1 FROM-LITERATURE (cited); §2 retro-typing of `[macOS-CPU advisory]` receipt
  (`score_claim=false`); §3 DERIVED spec; §4 DERIVED N-A.
- Verdict-scope: P2b = **INSTANCE**; the §2.1 LAW = **FORMULATION candidate** (falsifier named,
  needs a 2nd plateau instance); §3 refinement = **DERIVED** (algebra, reproducible).
- Falsifiers: §2.1 — an ES/MC row that beats aimed pathwise at matched evals at this endpoint;
  §3 — a measured variance A/B where STL-on beats full on the UNIFORM proxy (would contradict
  the zero-variance-Jacobian derivation → re-open §3.1).
- STORES CONSULTED: `policy_gradient_variance_reduction_survey_20260712.md` (the prior
  score-vs-pathwise corpus — line 140 ES=likelihood-ratio, line 155 pathwise-not-universally-
  lower-variance) · `ddm_gc6_from_endpoint_convocation_20260729.md` (row 8, T4 Ballé seat,
  §5 E2 tree N4) · `ddm_e2_pose_stream_and_doctrine_export_DAG_FEED_20260723.md` (E2 boundary) ·
  `p2b_mc400_diagonal_receipt.json` (SSD) · `segnet_recursive_fractal_factorization_20260715`
  (exact rank-4 head Jacobian = the pathwise operator) · MEMORY costate/organ + default-off-
  orphan + verdict-scope-ladder rows.

## §AMENDMENT (2026-07-29, MAIN — #404 relative-significance cure for the "negligible pose / deprioritize" routing language)

The consumer-routing rows above use magnitude words ("negligible pose", "deprioritize blind") — here
is the RELATIVE arithmetic they rest on, at the current operating point (post-P3 composed row:
S ≈ 20.3 advisory vs target 0.172 → remaining gap ≈ 20.1):

- P2b ES yield: ΔS = −0.0411 (100% pose √-term; d_seg +2.6e-6 wrong-way) = **0.20% of the remaining
  gap for 2,023 s of wall** [MEASURED, p2b receipt]. The dismissal is NOT absolute-magnitude: it is
  DOMINANCE — the aimed pathwise alternative books −0.046 (Contrarian bound, ceiling −0.138) at ~$0
  bytes and comparable wall [ru1/gc6 receipts], i.e. ≥1.1× the yield with a measured +24-flips/quantum
  mechanism vs a variance-dominated one. verdict_scope: INSTANCE (this endpoint, this budget).
- "Deprioritize blind edits" in the P2c split likewise rests on a MEASURED dominance ratio, not
  eyeball: blind single-token edits median −1 flips (65% net-negative) vs aimed best-of-8 positive in
  17/18 hotspot cells [ru1 receipt] — a sign flip, not a small number.
- Un-retired: the LAW's own falsifier (an ES/MC row beating aimed edits at matched evals at this
  endpoint) is the standing exit criterion; nothing here is a family kill.
