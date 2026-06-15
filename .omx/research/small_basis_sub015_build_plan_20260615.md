# Sub-0.15 build plan — full lifecycle (design→research→build→test→adversarial-review→iterate→optimize)

**Date:** 2026-06-15. **Source:** `.omx/research/small_basis_optimization_register_20260615.md` (the micro→macro
audit). **Goal:** the scaled base_ch=20 campaign that crosses sub-0.15. **Reachability:** d_seg=0.0006 +
FP4 + FiLM-pose ⇒ S≈0.105. **All advisory until byte-closed exact CPU+CUDA eval (the only pointer-mover).**

Status key: ✅ done · 🔵 partial/exists-unrouted · ⬜ todo · 🔬 research-gated (unknown answer).

## CRITICAL PATH (dependency-sequenced)
```
[live] from0 A/B verdict (lever-refinement accelerator: does it help d_seg?)
   │
   ├─► WS-A meso/micro correctness fixes  ──┐ (cheap, build now, become the new baseline)
   ├─► WS-D exact-eval apparatus  ──────────┤ (the gate; partially exists)
   │                                        │
   ├─► WS-B FP4 codec (rate maker)  ────────┤ 🔬 d_seg-hold is the research gate
   ├─► WS-C taper realloc (d_seg accel) ────┤ 🔬 the 192×256-band hypothesis is the research gate
   │                                        ▼
   └─────────────────► WS-E scaled campaign: base_ch20 + A+B+C+FiLM → byte-close → exact CPU+CUDA → pointer
```
Each of WS-A/B/C needs its own MVP-first $0 smoke BEFORE the paid scaled run (see gates). Adversarial
review (3-clean-pass) on every code change to the training/codec path before it enters the scaled run.

---

## ⚑ WS-A LANDING (2026-06-15) — 2 code fixes BUILT (default-off), 2 config-only, 1 deferred
- **margin-renorm (M7/S10):** ✅ BUILT — `StageSpec.margin_weight_renorm` (default False → `.mean()`
  byte-identical; True → `(per_pixel·w).sum()/Σw`). `driver._seg_loss_for_spec` + `curriculum.StageSpec`.
- **Muon LR-floor (M3/S18):** ✅ BUILT — `TorchVehicleConfig.muon_lr_floor_fix` (default False → Muon
  shares AdamW lambda; True → own cosine floor keyed to muon_lr). `driver._build_stage_runtime`.
- Both default-OFF (live A/B apples-to-apples preserved); 6 NO-FAKE tests + 22 regression byte-identical.
- **EMA window (①/S17):** CONFIG-ONLY — scaled run passes `--ema-decay 0.99` (window ~100 ≪ 800;
  captures ~90% of the warmup benefit, zero code). The warmup-decay code refinement (#86) is deferred.
- **eval_every scaling (M2/S19):** CONFIG-ONLY — scaled run passes `--eval-every 2` (the launcher already
  threads it into `build_curriculum`); evaluates stage-4 QAT onset. Zero code.
- **eval-roundtrip op-order (μ3/S20):** DEFERRED — most delicate (core render path + the eval_roundtrip
  non-negotiable); build as a careful, separately-reviewed follow-up.

## WS-A — meso/micro correctness fixes (the new baseline; cheap, certain)
The "PR95-faithful-scaled-without-re-deriving-constants" cluster. Batch ALL into the scaled-run baseline.
- **EMA window** (①/M1): **DESIGN** = warmup-decay `min(decay,(1+t)/(10+t))` (task #86) or constant 0.99.
  **BUILD** = wire into `tac` torch EMA (the vendored `ema_update` is constant-decay). **TEST** = shadow
  tracks live within N ep; byte-identity when decay unchanged; the capstone-shadow-lag regression.
  **REVIEW** = does it move the basin? **OPTIMIZE** = warmup rate vs budget.
- **Muon LR-floor** (M3): **BUILD** = separate `muon_lr_lambda` keyed to `muon_lr` (2-line, `driver.py:770`).
  **TEST** = Muon LR anneals to the intended abs floor at stage 8. **REVIEW** = trivial/low-risk.
- **eval_every scaling** (M2): **BUILD** = `eval_every=max(1,round(25·budget/29650))`. **TEST** = stage 4
  (QAT onset) now evaluated. **REVIEW** = selection-bias closed.
- **margin-renorm** (③/M7): **DESIGN** = `(per_pixel·w).sum()/w.sum().clamp_min(eps)` vs current `.mean()`.
  **BUILD** + **TEST** = gradient *direction* preserved, magnitude no longer decays as margins grow.
  **REVIEW** = does it accelerate the tail or destabilize? **ITERATE** = renorm vs τ-only.
- **eval-roundtrip op-order** (μ3): **DESIGN** = round→uint8 at 874 (STE) THEN bilinear↓384, matching the
  authority pixel path. **BUILD** + **TEST** = proxy pixels == authority pixels (the 0.43/16.75px drift → 0).
  **REVIEW** = honors the eval_roundtrip non-negotiable; **does NOT change the authority eval** (already faithful).

## ⚑ WS-B SMOKE RESULT (2026-06-15, MEASURED — research gate 1 ANSWERED)
`experiments/probe_fp4_dseg_hold_smoke.py` on the real control ckpt (`reports/fp4_dseg_hold_smoke.json`):
fp32 d_seg=0.003578 · **int8 0.003591 (Δ+0.4% — int8 is d_seg-LOSSLESS)** · **fp4_mixed 0.005610 (Δ+56% — NO-GO)** ·
fp4_all 0.006258. ΔS int8→fp4_mixed = **+0.2531** (rate −0.0224 but seg +0.2020 → NET WORSE).
**VERDICT: post-hoc FP4 NO-GO.** Falsifies the μ4 "d_seg HF-blind ⇒ FP4-safe" bridge — FP4 *weight* error
is BROADBAND (corrupts the low-freq structure SegNet reads), not HF-only; it flips boundaries (d_seg +56%,
d_pose +588%). ⇒ **FP4 requires FP4-QAT** (train weights to be 4-bit-robust, PR95 path), NOT a free codec
swap. WS-B re-ranks BELOW WS-C (taper is rate-neutral, no QAT).

**FREE-FLOOR follow-up (COMPLETE — gate 1 fully closed):** the full d_seg-vs-bits curve (post-hoc, no QAT):
int8 0.003591 (S 0.4553, lossless) · int6 0.003972 (Δ+10.6%, S 0.4870 net **+0.032 WORSE**) · int5 0.004942
(S 0.6089) · int4u 0.008032 (S 1.0426) · fp4_mixed 0.005610 (S 0.7084). **There is NO free post-hoc
sub-8-bit win** — the curve is steeply convex, so 100·Δd_seg always exceeds the rate saving; int8 is the
correct post-hoc ship. **The ENTIRE rate lever is QAT-gated.** Concrete WS-B success criterion: **FP4-QAT
must pull d_seg from 0.00561 → ≤ 0.00381** (recover ~0.0018 so 100·Δd_seg < the −0.0224 rate save). If QAT
lands d_seg ≤ 0.0038 at FP4 bytes → the −0.022 win banks; else FP4 is out, int8 stays. (`reports/fp4_dseg_hold_smoke.json`)

## WS-B — FP4 mixed-precision decoder codec (the rate MAKER, Δrate −0.022)  🔬 [now QAT-gated, not free]
- **DESIGN:** mixed partition (FP4 interior convs ~80K / int8 output-proximal heads ~3.4K); FP4 format
  (E2M1 nonuniform vs int4 uniform — measured E2M1 38.3KB / mixed 40KB); the **packed 4-bit container**
  (not just a smaller int8 alphabet — that's the current gap); archive-grammar update (build + inflate agree
  on the packed format); per-channel scales (L29) + per-tensor byte-maps (L21).
- **RESEARCH 🔬 (the gate):** does FP4-QAT hold d_seg? μ4 predicts YES (d_seg blind to HF >192×256, FP4 noise
  is HF) but it's unmeasured. Map relL2→d_seg per tensor; identify d_seg-high-leverage tensors that MUST stay
  int8 (the sensitivity map — shared with WS-C + Lever-4). The 18× relL2 ratio of int4 — does it translate
  to d_seg or wash out?
- **BUILD:** route `src/tac/quantization.py` FP4/per-channel into the torch-vehicle build (it exists, unrouted);
  FP4 fake-quant STE in the QAT stages; packed-FP4 archive writer + **numpy-portable inflate reader** with
  byte-parity; the mixed-precision tensor partition driven by the sensitivity map.
- **TEST:** build→inflate bit-identical round-trip; FP4-STE gradient finite/nonzero; **paired d_seg-hold smoke
  (FP4 vs int8 on the SAME trained ckpt)**; archive parse-back; no-op detector (bytes changed AND consumed).
- **ADVERSARIAL REVIEW:** the d_seg-hold claim (boundary spill?); bit-pack correctness (endianness/off-by-one);
  proxy↔auth gap on the FP4 archive; per Catalog #220 (operational, not scaffold).
- **ITERATE:** per-tensor bit-width by sensitivity (mixed 4/6/8); the int8-vs-FP4 threshold; QAT schedule.
- **OPTIMIZE:** range/Categorical coding on the FP4 stream (L30), split-brotli (L23), variable-level allocation
  via Lever-4 online sensitivity EMA (the WRQ unification).

## ⚑ WS-C SMOKE RESULT (2026-06-15, MEASURED — research gate 2 ANSWERED: Φ3 CONFIRMED)
`experiments/probe_dseg_sensitivity_map.py` (`reports/dseg_sensitivity_map.json`), 20% relative-RMS
per-tensor perturbation → real Δd_seg:
- **LOW** (stem+blocks.0-1) **68% params / 25% sensitivity → 0.36× density (OVER-provisioned → SHRINK)**
- **MID** (blocks.2-3, skips) 21% / 42% → 1.97× (under) · **HIGH** (blocks.4-5, refine, rgb) **11% / 34% → 3.07× (UNDER → WIDEN)**
- **rgb_1 (last-frame head) is THE d_seg bottleneck** (Δ+0.0037, 270 params) · **rgb_0 Δ=0.000000** (contest reads only the last frame!) · refine Δ~0 (d_seg-irrelevant) · skips.3/4 high-density.
**VERDICT: Φ3 CONFIRMED — HIGH-res band 3× under-provisioned per d_seg sensitivity.**
**TAPER DESIGN (WS-C):** shrink stem/blocks.0-1 → widen rgb_1 + skips.3/4 + blocks.4; keep rgb_0/refine lean.
**BRIDGE → gate 1 (WS-B):** the unified sensitivity map = the FP4-QAT partition. Keep **rgb_1 + skips + blocks.4 int8**
(d_seg-critical); FP4-QAT the insensitive bulk (stem/blocks.0-3/rgb_0/refine). It also EXPLAINS gate-1's NO-GO:
the rate win needs FP4 on the big low-res tensors (stem/blocks, ~0.0013 Δ each) → post-hoc spills → QAT.
**CAVEAT:** perturbation-sensitivity ⇒ under-provisioned is a HYPOTHESIS; the WS-C short-budget taper smoke
must confirm that *adding capacity* there actually lowers d_seg (vs saturated-critical).

## WS-C — d_seg-aware taper reallocation (the d_seg ACCELERATOR; NOVEL)  🔬 [gate 2 DONE → design ready]
- **DESIGN:** new taper moving capacity from the 33% stem-Linear (27.8K params, low-freq) → a wider/deeper
  192×256→384×512 refine head, **at FIXED total param count** (rate-neutral). Thin decoder subclass (do NOT
  edit the pristine vendored clone; `base_channels` already threaded; the taper list is the knob). Preserve
  the PR95 PixelShuffle/sin/bilinear-skip structure (L18).
- **RESEARCH 🔬 (the gate):** WHERE exactly does d_seg live? Build the d_seg-sensitivity map over
  (tensor × resolution × spatial-frequency) — confirm the 192×256-boundary-band hypothesis empirically
  (∂d_seg/∂(layer activations)). Is more refine-head capacity the right actuator, or skip-to-high-res /
  boundary-attention? Re-measure the power-law exponent under the new taper (does it accelerate the descent?).
- **BUILD:** the reallocated-taper decoder subclass; param-count-matched assertion; codec/inflate handle the
  new tensor shapes (couples to WS-B).
- **TEST:** total-param parity; fwd/bwd correctness; export/inflate round-trip with new shapes; **short-budget
  d_seg smoke — does the new taper descend faster than baseline at matched budget?** (the decisive measurement).
- **ADVERSARIAL REVIEW:** is it really targeting d_seg or just shuffling params? (validate the 192×256
  hypothesis, don't assume it); apples-to-apples vs baseline taper (same init/seed/budget); does it break
  the renderer's coarse-structure capacity?
- **ITERATE:** multiple taper candidates A/B'd (how much to move, where); the refine-head depth/width.
- **OPTIMIZE:** joint with WS-B — put both BITS (FP4) and CAPACITY (taper) at the d_seg band (the unified
  sensitivity-map allocation).

## WS-D — exact-eval apparatus (the GATE that moves the pointer)  🔵
- **BUILD/COMPLETE:** the one-command byte-close → exact CPU+CUDA eval pipeline (Track-A item E, partial).
  **TEST:** archive byte-parity, dual-axis (contest-CPU Linux x86_64 + contest-CUDA T4) per the dual-eval
  non-negotiable. **REVIEW:** apples-to-apples (source-vs-candidate same runtime); no MPS authority.
- The d_seg-hold validation harness (WS-B) + the taper A/B Pareto harness (WS-C) plug into this.

## WS-E — the scaled sub-0.15 campaign (integration)
Once A (fixes) + B (FP4, d_seg-hold proven) + C (taper, descent-accel proven) each pass their $0 smoke:
base_ch=20 + all WS-A fixes + WS-B FP4 + WS-C taper + FiLM-pose (✅ live) → full PR95 curriculum at the
budget the new power-law exponent says reaches basin → byte-close (WS-D) → exact CPU+CUDA → pointer update.
**ITERATE/OPTIMIZE:** the full-stack is itself A/B'd against the current frontier; re-seed the power-law +
sensitivity map from each exact row (continual learning).

## MVP-FIRST GATES ($0 smoke before any paid/long run — Carmack phasing)
- WS-A: each fix has a unit/regression test (free) + a short-budget byte-identity-when-off proof.
- WS-B: the **FP4 d_seg-hold smoke** ($0, on an existing trained ckpt — re-quant int8→FP4, re-measure d_seg)
  is the GO/NO-GO before any FP4-QAT retrain.
- WS-C: the **d_seg-sensitivity map + short-budget taper smoke** ($0/cheap) is the GO/NO-GO before the
  scaled taper retrain.
- WS-E: paired contest-CPU/CUDA on the byte-closed candidate; no pointer claim without it.

## ADVERSARIAL REVIEW CADENCE
Every change to the training-loop / codec / archive-grammar path = the recursive 3-clean-pass protocol
(council lenses + the 8 axes incl. assumption-challenge) BEFORE it enters the scaled run. The d_seg-hold
(WS-B) and the 192×256-band (WS-C) hypotheses are the two highest-risk assumptions — each must be
EMPIRICALLY verified ($0 smoke), not assumed, per the NO-FAKE + measurement-first discipline.

## NO-SIGNAL-LOSS SESSION REGISTER — every finding this session, with disposition + lifecycle home
Per CLAUDE.md ANTI-SIGNAL-LOSS + "Results must become system intelligence." Disposition ∈
{✅DONE · 🟢LIVE · 🔨BUILD-NOW · 🔬RESEARCH-GATED · ⏸DEFERRED(+criteria) · ⛔DEAD}.

### Training-signal / lever findings
| # | finding | disposition | lifecycle home |
|---|---|---|---|
| S1 | **Anneal calibration #119**: fast-cool T1.0→0.3 hold_frac=0.3 beats adaptive; weighted-mean fixed point clamps to the 0.30 floor; win = trajectory value (warm phase fixes deep flips) | ✅DONE (in use) | iterate via S12 (per-pixel-resonant-T) |
| S2 | **soft_cosine vs CE gradient math**: soft_cosine 1/(TC)=5× weaker at init (vanishing q_g factor); CE log-barrier unconditionally strong; soft_cosine→d_seg as T→0 (metric-aligned) | ✅DONE (derived) | drives S3 design |
| S3 | **CE-coarse → soft_cosine-refine** (stack_ce_early) | 🟢LIVE (A/B running) | verdict gates WS-E |
| S4 | **l235: levers DO break the d_seg plateau** 0.00374→0.00247 (600pair ep2400, S 0.351) then OVERFIT (no FiLM-v2) | ✅BANKED finding | motivates FiLM-v2 + the campaign |
| S5 | stack (soft_cosine-from-0) loses to control (d_seg 0.022 vs 0.0036) — confirms S2 | ⏸DEFERRED @ep369 (resumable) | reactivate only w/ S2 fix |
| S6 | **lovász arm** (Jaccard convex envelope; submodular complement to soft_cosine) | ✅BUILT+committed, 🟢QUEUED (after stack_ce_early) | A/B result feeds WS-E |
| S7 | Lovász analysis: modular d_seg ⇒ its Lovász≈soft_cosine; Jaccard/IoU submodular ⇒ complementary; don't add a separate Jaccard term | ✅DONE | — |
| S8 | **pose-throttle** k=4/thr=0.001: pose 51–92% of epoch, solved ⇒ **3.2× speedup**; d_seg every epoch (verdict valid) | ✅BUILT+DEPLOYED (`278227ae3`) | live; OPTIMIZE thr/k post-verdict |
| S9 | **τ coherence** 0.3→1.0 (soft_cosine self-concentrates q_g(1-q_g)/T; τ=0.3 double-concentrated+mis-scaled vs 0.5–2 logit flip band) | ✅FIXED (`7e4f83f37`) | — |
| S10 | margin-weight non-renorm (③) — compounding gradient-decay | 🔨BUILD-NOW | WS-A |
| S11 | Lever-5 margin-weight → feed survival-robust flip set (Track-A #114) | ⏸DEFERRED | post-verdict lever |

### The deferred "other items" (the user's earlier "we can talk about the other items" — NOT lost)
| # | item | disposition | home |
|---|---|---|---|
| S12 | **per-pixel-resonant-T** (each flip's T*=its own margin Δ, vs one global T) — the resonance math from the anneal analysis | 🔬RESEARCH-GATED | WS-A iterate / a $0 smoke: does per-pixel T beat global fast-cool? |
| S13 | **renorm** the surrogate (sister of S10) | 🔨BUILD-NOW | WS-A (margin-renorm) |
| S14 | **flip-conditional weight** (weight only the pixels that actually flip vs all small-margin) | 🔬RESEARCH-GATED | WS-A iterate (couples to Lever-D survival flips S11) |
| S15 | **rate-spend-on-d_seg-capacity** (trade rate budget directly for d_seg capacity) | 🔬RESEARCH-GATED | = WS-C taper-realloc + WS-B FP4 (this IS the capacity↔rate trade, now concrete) |
| S16 | **round-trip-margin objective** (optimize the margin AFTER the uint8 roundtrip, not before) | 🔬RESEARCH-GATED | couples to WS-A μ3 (eval-roundtrip op-order) — same root |

### Measurement / apparatus findings
| # | finding | disposition | home |
|---|---|---|---|
| S17 | EMA-shadow-lag (const 0.999, no warmup, window 1000>800) | 🔨BUILD-NOW | WS-A ① |
| S18 | Muon LR-floor mis-keyed (shares AdamW lr_lambda) | 🔨BUILD-NOW | WS-A M3 |
| S19 | eval_every not budget-scaled (stage-4 QAT 0 evals) | 🔨BUILD-NOW | WS-A M2 |
| S20 | eval-roundtrip op-order mismatch (0.43/16.75px) | 🔨BUILD-NOW | WS-A μ3 |
| S21 | 96-pair = "first 96" deterministic/shared (relative-ranking signal, not absolute; biased subset) | ✅KNOWN (discipline) | WS-D (600-pair confirm before exact) |
| S22 | 800-budget = first read; power-law-slow ⇒ may need extend | ✅KNOWN | WS-E budget sizing |

### Rate findings
| # | finding | disposition | home |
|---|---|---|---|
| S23 | decoder int8 ~7.35 b/param; **FP4 packing NOT wired** (variable-level only shrinks the alphabet; int8 entropy at brotli floor) | 🔬RESEARCH-GATED (d_seg-hold) | WS-B |
| S24 | per-channel scales + byte-maps (L21/L22/L29) unused | 🔨BUILD (enabler) | WS-B |
| S25 | latents near-optimal (~250B headroom, 3.7% of archive) | ✅DONE (no action, low-EV) | — |
| S26 | T1 cross-pair latent dedup (−0.003 to −0.006 rate, unbuilt) | ⏸DEFERRED (n600-regime) | post-basin, 600-pair campaign |

### Macro / strategy
| # | finding | disposition | home |
|---|---|---|---|
| S27 | **d_seg power-law-slow** 0.0367·ep^-0.351; 1.6× capacity penalty vs ch=36; NOT a wall | ✅KEY finding | the campaign thesis |
| S28 | **taper realloc** (81% params low-freq, 1.75% at the 192×256 d_seg band) | 🔬RESEARCH-GATED | WS-C |
| S29 | capacity↔d_seg↔rate Pareto; sub-0.15 reachability S≈0.105 | ✅KEY arithmetic | WS-E target |
| S30 | **μ4 bridge**: d_seg blind to HF >192×256 ⇒ FP4 safe (de-risks WS-B) | ✅KEY bridge | WS-B research gate |
| S31 | mask-grammar storage for d_seg | ⛔DEAD (rate limit; 95% flips 1-px salt-and-pepper unrepairable) | terminal |
| S32 | basis alternatives (Cool-Chic/C3/VQ/Mamba-Z7/Dreamer-Z8) | ⏸DEFERRED (all pre-measurement; means-not-ends) | reactivate only at the measurement gate |
| S33 | FiLM-v2 pose store (pose 0.045→0.017) | ✅DONE/🟢LIVE | WS-E (keep on) |

### Operational (lessons — durable, not lost)
| # | finding | disposition |
|---|---|---|
| S34 | orphan-contention (l235 + `ps aux` false-death → use `pgrep -fl`; kill superseded MPS runs) | ✅LESSON ([[l235-...]]) |
| S35 | native-accel honesty: the scorer forward (EfficientNet+FastViT on GPU) is the train-time bottleneck, **NOT Rust-able**; codec IS Rust-able but is NOT the bottleneck (μ7) ⇒ speed levers = fp16/throttle/compile, not native | ✅KNOWN |
| S36 | dashboard + Cloudflare tunnel (iPhone-offline-Tailscale diagnosis) + caffeinate keep-awake | ✅DONE (operational) |

**Completeness check:** S1–S36 cover every finding from this session's turns (anneal → v2 wire-in → throttle →
coherence review → dashboard → micro/meso/macro audit → this plan). DONE/LIVE = 12 · BUILD-NOW = 6 ·
RESEARCH-GATED = 6 · DEFERRED+criteria = 5 · DEAD = 1 · KNOWN/LESSON = 6. No finding is orphaned.

## Cross-refs
`.omx/research/small_basis_optimization_register_20260615.md` (the audit) ·
[[small-basis-micro-macro-audit-sub015-path]] · CLAUDE.md "THE GOAL — SUB-0.15", "canonical leaderboard
binding-depth discipline" L18/L21–L32, "Carmack MVP-first phasing", "Recursive adversarial review protocol",
"Submission auth eval — BOTH CPU AND CUDA".
