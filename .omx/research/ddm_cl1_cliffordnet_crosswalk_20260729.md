# ddm_cl1 — CliffordNet (arXiv 2601.06793) deep-read + crosswalk vs the post-burn campaign

**Arm:** `ddm_cl1` (bounded $0 paper-crosswalk; no code, no launches, no scorer jobs — pb1 owns the n600 slot).
**Date:** 2026-07-29. **Evidence axis:** `[advisory]` throughout — every number below is MEASURED-from-receipt,
DERIVED-arithmetic, or clearly labeled INFERRED/ASSUMED. **Nothing here is a score.**

## POINTER HONESTY FIRST

`0.1910828242 [contest-CPU]` **UNMOVED.** Everything in this memo is advisory MEANS. The competitive bar is
`effective_frontier = min(0.15, official ~0.172)` (PR130 real 190,952 B measured 0.172141). The pointer moves
ONLY through a byte-closed `upstream/evaluate.py` row (§R6 chain), which this arm does not touch. A paper
crosswalk that ends with the pointer unmoved has not moved the goal — this memo's job is to say, with
MEASURED arithmetic, whether CliffordNet shortens the distance to a lower exact row. **The answer for the
renderer axis is NO (dominated by ~58×); the one live thread is a $0 token-field coder-race probe.**

## STORES CONSULTED (recall-first, multi-pass)

CLAUDE.md + AGENTS.md (NO-FAKE #6/#7/#8; rule-118; verdict-scope ladder; pools law non-additive;
constants-are-poison; no-old-lineage) · MEMORY.md current-state (box-retired/warp-closed endgame; pose=terminal
6-eq; TOKEN STREAM = binding rate axis; realization-gated) · charter (scratchpad `cl1_charter.md`) ·
`ddm_tb1_renderer_build_20260728.md` (THE renderer build — T1/T2 MEASURED byte ledgers + adjudication) ·
burn endpoint receipt `t3_long_burn_lotto_v2/tr1_window_receipt.json` (full n600 realized d_seg 0.0038892 +
byte ledger) · eg1 rehearsal ZIP `ddm_eg1_tr1_rehearsal_20260728/submission/{archive.zip,archive/manifest.json,report.txt}`
(byte-closed 504,736 B) · `c_token_stack_race/receipt.json` (lv1/r7 coder race on the T2 payload) · lv1 INDEX.md ·
`ddm_co8_organ_round8_20260728.md` (rv1 conditional-validity reactivation table = the extension queue) ·
paper: arXiv abs + GitHub `ParaMind2025/CAN` + second-source aggregators (researchgate/academia/aimodels/emergentmind).

---

## 1. DEEP READ + MECHANISM + EVIDENCE-GRADE AUDIT

### Identification
**CliffordNet: All You Need is Geometric Algebra** — Zhongping Ji, **single author**, arXiv:**2601.06793v2**
(Jan 2026), cs.CV / cs.LG, 16 pages. Repo name **CAN** (Clifford Algebra Network).

### Mechanism (as recovered from the abs + repo docstrings + second sources — the PDF body did not extract to text; labeled INFERRED where the extraction was indirect)
- **Core primitive (MEASURED from abstract/verbatim):** the Clifford **geometric product** `uv = u·v + u∧v` used
  as the *sole* interaction layer. The inner product `u·v` carries **feature coherence** (symmetric/alignment);
  the exterior wedge `u∧v` carries **structural variation** (antisymmetric/bivector). The paper's "No-FFN"
  thesis: this single algebraically-complete interaction makes the Transformer/CNN FFN (channel-mixer MLP)
  **redundant**.
- **Layer realization (INFERRED from repo `model.py`/docstrings):** the product is decomposed into a **Scalar
  Component** and a **Bivector Component**; spatial variation is injected by a **`shifts` / "sparse rolling"**
  parameter — a channel-wise cyclic roll that gives each channel a different spatial offset, so the
  geometric-product interaction mixes spatially-displaced features at **O(N)** (linear, no attention quadratic).
- **Param counts / models (MEASURED from abs + second source):** Nano **1.4M** params → **77.82%** CIFAR-100
  (vs ResNet-18 ~11.2M); Lite **2.6M** → **79.05%** (rivals ResNet-50). Headline: **~8× fewer params** at
  ResNet-18-class accuracy. `model_hier.py` adds a hierarchical/pyramid backbone (2026-03-01).

### Evidence-grade audit (the load-bearing part for us)
| axis | finding | grade |
|---|---|---|
| authorship | **single author** (Zhongping Ji) | thin — no co-author adversarial check |
| scale | **CIFAR-10/100 only** in the headline; no ImageNet result recovered | small-scale |
| task | **classification-only** — NO segmentation / detection / **dense-prediction** result anywhere | **the decisive gap for us** |
| code | **RELEASED**: `github.com/ParaMind2025/CAN`, **MIT**, `model.py`/`model_hier.py`/`train.py`/`gffn.py`/`cuda/` Triton wheels; 178★/24 forks; last update 2026-03-01 | good |
| reproducibility | **NO pretrained weights**; **NO exact-command README** (inline docstrings only) | partial |
| independent replication | **NONE found** — second sources (researchgate, academia.edu "review", aimodels.fyi, emergentmind) are paper-derived aggregators, not replications | unverified |

**Verdict-scope: PROMISING-BUT-UNVERIFIED-FOR-OUR-TASK.** The param-efficiency claim is real and code-backed
*for CIFAR classification*. The transfer to our task — a **dense per-pixel argmax realization** paint — is
**UNPROVEN in the paper** (no dense head, no segmentation number). This is not a paradigm kill; it is a
formulation-level "transfer risk HIGH, evidence CIFAR-only" tag.

---

## 2. THE ONE ARITHMETIC THAT MATTERS — renderer-stream bytes TODAY (all MEASURED from receipts)

### The counted renderer stream (LOTTO = the ADOPTED arm), from the burn endpoint receipt
`t3_long_burn_lotto_v2/tr1_window_receipt.json` (full n600 realized d_seg **0.0038892**, ep399, `[macOS-CPU/MLX advisory]`, COUNTED-ESTIMATE zlib, no container):

| stream | bytes | share of total |
|---|---:|---:|
| **tokens** | **875,171** | **99.60%** |
| renderer (LOTTO mask+mods) | **3,284** | 0.374% |
| selector ledger | 216 | 0.025% |
| **total_counted** | **878,671** | 100% |

Cross-checks (MEASURED): T2 40-ep window — renderer **3,284** (lotto) vs **20,214** (plain), tokens 531,097,
total 534,597 (the "6.2×" = 20,214/3,284 = **lotto-vs-plain renderer**, not renderer-vs-token). Byte-closed
eg1 rehearsal ZIP = **504,736 B** total (payload `state/tr1.ddt1` 504,249 + `manifest.json` 259 + **228 B** ZIP
container overhead) — a single opaque blob that confirms the byte-closed *total* but does not itself split
streams; the split is the tb1 receipt above.

### The arithmetic (DERIVED; `S`-rate term = 25·B / 37,545,489)
- Renderer stream **entire, reduced to zero**: ΔS_rate = 25·3,284 / 37,545,489 = **2.19e-3** ← the *absolute
  ceiling* of what any renderer improvement can ever buy.
- Clifford renderer at the paper's **8×** param claim (best case, transfer assumed): saves 3,284·7/8 = 2,874 B →
  ΔS_rate = **1.91e-3**.
- **Contrast — the binding token axis (MEASURED, `c_token_stack_race`):** lv1 **lossless** factorize
  (static-base + KT-prev1 delta) took the T2 token stream 531,097 → **364,581.7 B** = −166,515 B →
  ΔS_rate = **0.1109**.

**Leverage ratio: the lossless token factorize alone is 58× the entire Clifford-8× renderer gain**
(0.1109 / 0.00191). And it is *already measured and lossless* (realized d_seg custody exact 0.013833 at T2),
while the lossy token truncations (T1_revert 124 KB) currently FAIL the realized gate (0.023111 > baseline).

### VERDICT (deliverable 2)
**A Clifford renderer race does NOT clear the bar "shortens distance to a lower exact row."** The renderer
stream is **0.37% of the counted total** and its *entire* removal is worth **2.2 millipoints** of S; the token
stream (875 KB at the burn endpoint — it GREW from 531 KB as the long burn densified the deltas) is the axis,
and the gap to the 0.172 bar is dominated by it. The renderer bytes are **NON-BINDING**. Against the already-queued
next-vehicle work (the rv1 reactivation table in `co8`: R7 coder race + R4 token probe armed on the token axis),
a Clifford renderer is dominated by ~1.5–2 orders of magnitude. **Do not spend a vehicle slot on it now.** The
tb1 memo's own note ("at G4 geometry renderer share grows") does NOT rescue it: LOTTO renderer is already 3.3 KB,
nowhere near the ≤64 KB budget ceiling — shrinking a 3.3 KB stream is not where S lives.

---

## 3. GA-NATIVE CHEAP CHECKS ($0, design-level)

### (a) Rotor/motor GA pose chart vs `tac.lie` se(3) — **honest NO (ALREADY-COVERED)**
A PGA **motor** (even subalgebra of Cl(3,0,1)) is *identically* the SE(3) exponential of a bivector screw axis —
i.e. **motor ≡ exp(se(3)) = Chasles screw**, which is exactly what `tac.lie` already implements for the terminal
per-pair 6-eq GN pose solve (e_p rank-1 ~2 KB). Same 6-DOF manifold, same chart, same numerics. GN conditioning
is set by the **reprojection Jacobian**, not by the group chart, so the motor gives **no conditioning advantage**
and **no byte advantage** (8 motor components for a 6-DOF pose is if anything a worse packing than the minimal
twist). **Verdict: N-A / ALREADY-COVERED.** Expected-NO confirmed, stated honestly.

### (b) THE SLEEPER — token-field `u·v + u∧v` coherence/variation factorization as a coder-race transform
**Structural coincidence (real):** the token field is shape **[600, 24, 32, 4]** — c=**4** channels =
2² = dim of the **Cl(2,0)** multivector `{1, e1, e2, e12}` (scalar + 2 vectors + 1 bivector). Each lattice
cell's 4-vector **can be read as a Cl(2) multivector**, so the geometric product of adjacent cells (spatial OR
temporal) splits into a **coherence** part (`u·v`, symmetric/scalar) and a **variation** part (`u∧v`,
antisymmetric/bivector). The most GA-native form: code the **inter-frame rotor** `M_t · M_{t+1}~` — its scalar
part is temporal coherence, its bivector part the small inter-frame "rotation" — instead of the raw 4-channel
temporal delta.

**Honest transfer risk (the reason the prior is LOW, not zero):**
1. The c=4 channels are **learned latents** with NO GA structure imposed at train time — no reason the channel
   basis aligns with GA blades. A learned **KLT/PCA** channel rebasis is the stronger decorrelation baseline the
   GA transform must beat, and a 4×4 KLT basis costs ~32 B (negligible), so the "fixed transform = zero basis
   bytes" argument is weak here.
2. The current coder **already** does a coherence/variation split (static base + temporal delta = `F1`), and
   KT-prev1 (order-1 context) already exploits temporal coherence — so a GA transform must **beat** KT-prev1's
   **364.6 KB**, not add to it (**pools law: same-pool coders COMPETE, never sum**).
3. Gain is entropy-only, in the same coder pool.

**Verdict: PROBE-WORTHY (LOW prior, $0).** Spec the $0 probe: apply the Cl(2) geometric-product transform to the
burn-checkpoint token field (both spatial-adjacent and inter-frame-rotor variants), measure zlib/entropy of the
transformed field vs (i) KT-prev1 lossless 364.6 KB and (ii) a KLT-rebasis control. **Consumer: r7 coder race
(pb1-P5 owns the n600 coder slot).** **Falsifier: transformed-field entropy ≥ min(KT-prev1, KLT-control)** → dead,
file with reason. This is a **lossless algebraic re-factorization of the token field = a coder-race candidate,
NOT an architecture change** — correctly scoped, and it lands on the *binding* (token) axis, which is why it is
the one live thread out of this whole crosswalk. **Caveat: the −31% was measured on the T2 (531 KB) checkpoint;
the burn field is 875 KB and denser — the probe (and the whole r7 race) must re-run on the burn payload.**

---

## 4. RANKED CROSSWALK TABLE

| # | CliffordNet element | disposition | named consumer | falsifier | price / trigger |
|---|---|---|---|---|---|
| 1 | **Token-field `u·v`+`u∧v` (Cl(2) coherence/variation) lossless transform** | **DESIGN-INPUT → $0 PROBE (the sleeper)** | **r7 coder race / pb1-P5** | transformed-field entropy ≥ min(KT-prev1 364.6 KB, KLT-rebasis control) | **$0**, LOW prior; trigger = r7 race owner admits a transform stage; re-run on burn payload |
| 2 | Clifford geometric-product **renderer block** (param-efficient dense paint) | **DESIGN-INPUT (weak) — NOT a race row now** | tb1/tr1 renderer-variant pool (vs LOTTO supermask) | matched-byte Clifford renderer does **not** reduce realized **Lane Betti-0 erasure** vs LOTTO | **dominated**: renderer ΔS ceiling 2.2e-3 even at 0 B; trigger = renderer capacity proven to bind Lane nucleation (currently a LOSS/pool issue, not renderer arch) |
| 3 | **Rotor/motor** SE(3) pose chart | **ALREADY-COVERED / N-A** | none (pose terminal, solved) | n/a — motor ≡ exp(se(3)) = Chasles = `tac.lie` | 0 |
| 4 | **"No-FFN"** thesis (geometric product replaces MLP) | **DESIGN-INPUT (philosophical)** | frontier "reallocate bytes → effective capacity" thesis | n/a (not a byte lever) | 0 — aligns with, does not add to, the capstone thesis |
| 5 | **Sparse rolling** O(N) channel-shift op | **N-A** | none | n/a | our renderer is already 3.3 KB conv — not a compute/param bottleneck |

**Nothing-unreachable discipline:** rows 1–2 are filed **priced-above-current-water with explicit triggers** —
no abandon bucket. Row 1 enters the r7/pb1-P5 coder-race queue as a $0 transform candidate; row 2 sits in the
tb1 renderer-variant DUTY_TO_MEASURE pool, trigger-gated on a future finding that renderer capacity (not the
Lane-pool loss levers) binds Lane nucleation.

## Honest boundaries
- The paper PDF body did not extract to text (arXiv PDF is image/stream-encoded); mechanism details in §1 marked
  INFERRED come from the abstract (verbatim), the GitHub repo docstrings, and second-source aggregators — **not**
  from a line-by-line read of the method section. Sources actually read: arXiv abs page, `github.com/ParaMind2025/CAN`
  repo page, WebSearch aggregators (researchgate/academia/aimodels/emergentmind). No independent replication exists.
- All byte numbers are **COUNTED-ESTIMATE** (zlib, no final entropy coder, no archive container) from the tb1
  receipts, except the eg1 byte-closed ZIP (504,736 B, real container). The token axis moved (531 KB → 875 KB)
  between T2 and the burn — the coder-race numbers (364.6 KB, −31%) are the **T2** payload and must be re-measured
  on the burn checkpoint before any next-vehicle claim.
- The next-vehicle queue I located is the **rv1 conditional-validity reactivation table** in `ddm_co8_organ_round8`
  (8 typed reactivation rows; R7 coder race + R4 token probe armed). The charter's "sc2 folds / per-type optimizer
  race" tags were **not** re-verified in receipts this arm — INFERRED-present, not confirmed.
- verdict_scope: all realized d_seg rows are single-seed INSTANCE, `[macOS-CPU/MLX advisory]`, no noise floor.
