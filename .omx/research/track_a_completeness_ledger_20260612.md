# Track-A Completeness Ledger — the no-signal-loss source of truth

**Operator directive 2026-06-12 (verbatim, binding):** *"track A completeness queue top
priority"* + *"all items no signal loss nothing fall through the cracks all fully iterating and
optimized and recursively adversarially reviewed by the owner as it's built."*

**What Track A is.** The legacy-HNeRV `base_ch=20` vehicle (the smaller-basis defensive bank)
PLUS our techniques (the 5 Layer-2 levers, the L3 finishing-kit bolt-ons, the rate recodes),
fully polished / tested / hardened / wired / integrated so that **the moment the distortion arm
finishes, a one-command path takes the trained checkpoint → byte-closed archive → bolt-on stack
→ exact eval** with every win banked and stacked.

**Why it matters (the stakes, recomputed honestly):**
- Frontier (pointer-only): contest-CPU **0.19109982** (177,169 B). Its breakdown: rate term
  `25·177169/37_545_489 = 0.118`, so its `d_seg+d_pose ≈ 0.073` (d_seg ~0.0006).
- Track-A `base_ch=20` basin best (advisory, NON-PROMOTABLE): **S≈0.378** (ep2236), d_seg
  **0.0026**, d_pose 0.00034, archive **89,628 B** → rate term only **0.0597** (HALF the
  frontier's rate). **base_ch20's rate is already a win; its d_seg (0.26 contribution) is 4×
  worse than frontier and is THE binding term.** The distortion arm (levers 2,3,5) attacks
  exactly that. If d_seg → ~0.0006 at the base_ch20 rate, base_ch20 ≈ 0.06 rate + 0.06 d_seg
  ≈ **0.12 — under T_3 (0.15) and under the frontier.** THAT is the Track-A thesis.

**The reusability contract (operator binding):** every algorithm is built against an
**abstract interface** — `Renderer`, `latents`, `weights` (state_dict), `Scorer` (the real
frozen SegNet/PoseNet) — so it works on Track A **or** Track B. The ONLY Track-A-specific
surface is the **thin `base_ch=20` archive-grammar adapter** (section offsets / blob layout).
Each item below names its `[CORE]` (agnostic) and `[ADAPTER]` (base_ch20) parts.

**The recursive-review gate (operator binding):** NO item is SEALED until it has passed a
genuine recursive adversarial review (the R1-style council / 3-clean-pass discipline applied to
THAT component). A finding resets the count. The instrument bug found in R8 (lever review) is
the proof this is live, not ceremonial.

---

## STATUS LEGEND
`BUILT` code exists + tests · `WIRED` a production caller invokes it · `REVIEWED n/3` recursive
clean-pass count · `SEALED` 3/3 clean + default byte-identical proven · `MEASURED` real-scorer
advisory number exists · `UNBUILT` not started.

---

## ITEM A — L2 distortion levers (2 seg-surrogate, 3 pose-FiLM, 5 margin-weight)
- **State:** BUILT + WIRED + MEASURED (pose-FiLM −0.0769, d_pose 7.8× reduction). **REVIEWED 1/3**
  (R7 CLEAN → 1/3; R8 NOT-CLEAN, found+fixed a probe measurement-validity bug — finding was in the
  review *instrument*, not the levers, so no reset, but the 3-clean gate is NOT met).
- **`[CORE]`** the levers are pure training-loop terms over `(renderer, latents, scorer)` — fully
  agnostic; reusable on Track B unchanged.
- **`[ADAPTER]`** none (levers never touch archive grammar).
- **GATE TO SEAL:** R9 + R10 + R11 three consecutive genuine clean lenses (fresh count, post-R8).
  Distinct lenses remaining: (i) **numerical-stability under temperature-anneal→0.05** (soft_cosine
  denominator / margin exp overflow at the annealed tail); (ii) **lever-interaction sign**
  (do 2+3+5 compose monotonically, or does margin-weight fight the pose term?); (iii) **resume ×
  anneal-schedule** (does the annealed temperature restore correctly mid-stage on a daemon resume?).
- **NO-SIGNAL-LOSS:** the distortion arm is LIVE-MEASURING these on the real scorer right now;
  its eval rows are the empirical SEAL evidence.

## ITEM B — L2 rate levers (1 rate-surrogate, 4 score-aware QAT) + variable-level codec
- **State (CORRECTED 2026-06-12 post-engineering-audit — supersedes the earlier "−0.0005..−0.0011
  ready win" framing, which was an OVERSTATEMENT):** Lever 1 BUILT+WIRED (MED-1 `codec_scan_order`
  fix → Spearman 0.90/0.999 vs real brotli). Lever 4 BUILT+WIRED but **net-score BYTE_DIRECTION_ONLY**
  (CERTIFIED real negative: vendored 127-requant ALWAYS re-quantizes, so a train-only grid is erased;
  −3263 B snap → −7 B trained; mechanism structural). The variable-level codec
  (`src/tac/losses/variable_level_codec.py`, 9 NO-FAKE tests) fixes the ERASURE — the per-tensor grid
  IS the deployed grid, so the byte win SURVIVES inflate (unlike Lever-4). **BUT the MEASURED net is
  NOT yet a win:** `probe_variable_level_codec_byte_distortion.py` on the real basin EMA + real scorer
  returns **BYTE_WIN_DISTORTION_NEAR_BREAKEVEN** — byte −789..−1721 B but **net advisory S Δ
  = +0.001053 / +0.006030 (WORSE)** because coarsening 27/28 tensors hurts distortion (the d_pose
  sqrt-term near the low operating point). REVIEWED 0/3.
- **`[CORE]`** the variable-level reverse-waterfill operates on weight tensors + a sensitivity
  vector — agnostic.
- **`[ADAPTER]`** `build_decoder_blob_variable_or_vendored` IS the thin base_ch20 grammar adapter
  (1-byte format flag + per-tensor `u8 n_levels`); default-preserving (uniform ⇒ vendored bytes).
- **GATE TO NET-SCORE WIN (the codec is a real BYTE lever; the SCORE win is TRAINING-GATED):** two
  untried reactivation paths, neither yet attempted — (a) **tighten the allocation**: coarsen ONLY
  the truly-lowest-sensitivity tensors and keep the pose-sensitive ones at 127, search for a
  net-non-worse operating point on the real scorer (no retrain); (b) **train the decoder at the
  variable grid** (QAT with the variable grid as the deployed grid, eval_roundtrip-style, so the
  decoder learns coarse-grid robustness and RECOVERS the distortion). Path (b) folds into a
  distortion-arm QAT stage — it is NOT a standalone bolt-on. Then: recursive review (3 clean) +
  MEASURE deployed net on a real base_ch20 archive.
- **HONEST VALUE: byte lever −789..−1721 B is real and survives inflate, but NET score is currently
  +worse. Expected net win is 0 until path (a) or (b) lands — do NOT count it in the stack yet.**

## ITEM C — L3 distortion finishing-kit (PR98 color-bias / T10 boundary / S12 null-preimage / Lever-D margin-residual)
- **State:** BUILDING (background agent `ae43ac8d`). `distortion_finishing_kit.py` + probe
  in-flight. PR98 re-fit correctly re-fits all 6 (frame,channel) bias slots on the REAL scorer
  (exploits "SegNet reads ONLY frame_1 ⇒ a frame_0 bias moves ONLY d_pose"). NOT landed.
- **`[CORE]`** all four operate on decoded frames + the real scorer at inflate-time — agnostic.
- **`[ADAPTER]`** applied post-render in the inflate path; the only base_ch20 coupling is where the
  bias/residual sidecar bytes live in the archive grammar.
- **GATE TO SEAL:** on land → recursive review (3 clean) → prove default-OFF byte-identical →
  MEASURE each bolt-on's real ΔS-per-byte (admit only if it pays rent).
- **Cumulative est −0.0003 to −0.0015 (PR98 + T10 + S12 + Lever-D), stacked.**

## ITEM D — L3 rate recodes
- **D1 — T1 cross-pair latent dedup:** **BUILT + SEALED — MEASURED HONEST NEGATIVE on the
  current base_ch20 latents** (commit `434d65d35`; memo
  `.omx/research/track_a_d1_cross_pair_latent_dedup_landed_20260612T214737Z.md`). CORE
  `src/tac/losses/cross_pair_latent_codec.py` (carrier-agnostic measure-and-select over dedup /
  codebook-VQ / framed-delta; AST-verified no base_ch20 import) + ADAPTER
  `build_latent_blob_dedup_or_vendored` (default-preserving: byte-identical vendored bytes unless
  a STRUCTURAL candidate strictly wins) + 27 NO-FAKE tests. **MEASURED on the 3 real latent
  tensors: 0 B deployed delta, byte-identical, ΔS=+0.0000000** `[macOS-CPU advisory]
  NON-PROMOTABLE`. The estimated −0.003..−0.006 did NOT materialize: 0 exact dups (600→600
  unique); VQ +1584 B; 2nd-order +1747 B; static range coder +113 B — the vendored 1st-order
  delta+lo/hi+brotli is already ~1.3% above the symbol-entropy floor and beats every alternative.
  The apparatus IS real (control: −616 B on structured latents, survives parse-back), so it fires
  automatically if Track B / future latents gain temporal structure. Recursive review: 1 finding
  (Lens-2 framed-delta brotli-alignment artifact) fixed + regression-tested → counter reset → 3
  fresh clean lenses → **SEALED**. NOT driver-wired (ITEM E gate). Honest next-mechanism rec: only
  ~205 B headroom remains (gap to entropy floor via an adaptive context coder) — small/uncertain;
  the binding base_ch20 term is d_seg distortion, not latent rate.
- **D2 — variable-level-codec driver wire-in:** overlaps ITEM B gate (1)-(2). Track here so it
  isn't double-counted-then-dropped.
- **D3 — WRQ (score-aware weight re-quant, task #69):** analysis DONE; needs driver wiring +
  reconciliation with the variable-level codec (they are the same lever family — UNIFY, don't
  duplicate). GATE: decide WRQ-vs-variable-codec canonical surface → wire ONE → review → MEASURE.
- **D4 — R1/R2/S12 lossless materializer stack (task #64):** components exist; verify they
  byte-close on a base_ch20 archive + fold into the one-command pipeline (ITEM E).

## ITEM E — byte-close → bolt-on-stack → exact-eval pipeline (THE actuator)
- **State:** to AUDIT (parts exist: `_build_archive_and_eval_decoder` in the driver, the bolt-on
  helpers). Needs to be a **single command** that: takes the distortion-arm BEST checkpoint →
  builds the byte-closed base_ch20 archive → applies the SEALED bolt-on stack (B + C + D, each
  default-OFF unless it paid rent) → runs `upstream/evaluate.py` (the real authority) → reports the
  exact score + per-bolt-on attribution.
- **`[CORE]`** the stack-compose + eval-driver loop is agnostic (consumes the abstract archive +
  scorer). **`[ADAPTER]`** the base_ch20 materializer.
- **GATE TO SEAL:** integration test (one command, deterministic, byte-identical with empty bolt-on
  set) + recursive review. **This is the "ready when training finishes" deliverable** — without it,
  a promising distortion-arm result has no path to an exact row (= signal loss).

## ITEM F — carrier-agnostic core + thin-adapter refactor (cross-cutting)
- **State:** PARTIAL (levers are already agnostic; codec/finishing-kit need the interface seam made
  explicit). GATE: each item A-E names + honors its `[CORE]`/`[ADAPTER]` split (done above); a
  final pass confirms the CORE modules import NO base_ch20 constant (the seam is clean) so Track B
  can consume them unchanged.

---

## DRIVE ORDER (highest-leverage first, parallel-safe with the live distortion arm)
1. **ITEM A → SEAL** (R9/R10/R11) — most advanced; closes the levers powering the live arm.
2. **ITEM C review** when `ae43ac8d` lands (don't duplicate; review + byte-identical proof).
3. **ITEM B/D2/D3 unify + wire + review** (variable-level codec is the canonical rate lever; fold WRQ).
4. **ITEM E pipeline** — build + test the one-command actuator (the readiness deliverable).
5. **ITEM D1 (T1 dedup)** — biggest unbuilt rate lever; CORE + review + adapter + MEASURE.
6. **ITEM F final seam audit** — CORE imports no base_ch20 constant.

**Every build:** default-OFF byte-identical (must NOT perturb the running distortion arm),
real-scorer-MEASURED (no synthetic-fixture score claims), committed via the serializer with
review-gate compliance, and **recursively adversarially reviewed before SEAL**. No item is "done"
until SEALED here.

**Authority discipline:** every number on this page is `[contest-CPU advisory] / [macOS-CPU
advisory]` NON-PROMOTABLE until `upstream/evaluate.py` on the byte-closed archive (ITEM E). MPS is
the train-gradient device only, never authority.
