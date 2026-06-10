# Frontier DECODER-AXIS QAT/LSQ recovery — coarsen THEN retrain (reactivation #2)

**Subagent:** `frontier_decoder_qat_recovery_20260610` · UTC 2026-06-10.
**Lane:** `lane_frontier_decoder_qat_recovery_20260610`.
**Mission (decoder-axis verdict §6 reactivation criterion #2):** the naive sensitivity-aware
q-coarsening of the 0.19199 frontier decoder was FALSIFIED-AT-IMPLEMENTATION
(`.omx/research/frontier_decoder_axis_waterfill_verdict_20260610.md`): even the gentlest step=2
candidate landed **+0.0709** on the contest-CPU authority because d_seg MORE THAN DOUBLED
(5.6e-4 → 1.25e-3) while the rate gain was only −0.0062. The named lever to recover: **coarsen
the bottom-decile tensors, THEN fine-tune (QAT/LSQ) the coarsened tensors' scales + codes** so
d_seg is recovered AT the smaller byte count. This memo executes that lever.

**Axis discipline:** the recovery loop + advisory screen are `[macOS-CPU advisory]` candidate-
generator priors — NOT score claims (CLAUDE.md "MPS auth eval is NOISE" + "Submission auth eval";
CPU-torch only, NEVER MPS). The paired contest-CPU eval (`upstream/evaluate.py --device cpu`,
Linux-x86_64 Modal CPU container, 1:1 with the contest GHA runner) is the score authority.

---

## 0. Parity check (the fidelity oracle) — PASS

`parity_check.py` validates that the in-process state_dict reconstruction + `HNeRVDecoder` render
produces the EXACT pixels the contest-CPU eval scores. Reconstruction reuses the canonical engine
(`fec6_decoder_mutations.prepare_decoder_blob` → per-tensor int8 q-codes + fp16 scale →
`decode_decoder_compact`-equivalent dequant with CONV4 perm inversion). Render reuses the frontier
runtime's own `model.HNeRVDecoder.forward` + the full eval pixel chain (bicubic 874×1164 + PR98
channel postprocess).

**Result:** on the 24-pair subset, fully-clean pairs (selector code 0, no DQS1: pairs 0/5/23)
match the byte-faithful inflate with worst per-pixel |diff| = **1 LSB**, mean 0.0000. Frame-1
(decoder-only for ALL pairs; the selector touches frame-0 only) matches within **1 LSB** across all
24 pairs. **PARITY PASS** — the in-process recovery loop optimizes the same pixels the eval scores.
The frame-0 selector perturbations (compact modes: luma/rgb/blue-chroma biases, ≤±4) are applied
byte-identically at byte-close time, so they are preserved in the final archive.

---

## 1. The recovery design (UNIQUE-AND-COMPLETE-PER-METHOD)

**The rate-gain mechanism that MUST be preserved:** coarsening rounds q-codes to multiples of
`step` → fewer distinct symbols → lower entropy → smaller brotli. For the recovery to keep the rate
gain, the fine-tuned codes must stay on the coarse grid (multiples of `step`).

**Recovery levers (both, per the verdict "scales+codes"), on the 3 bottom-decile tensors
(blocks.0.weight / blocks.1.weight / stem.weight = 62% of decoder bytes):**
1. **Per-tensor fp16 SCALE** — 1 scalar/tensor, 0 byte-count change, learnable (`scale_only` mode).
2. **Coarse CODES via LSQ straight-through round-to-`step`-grid** — learnable continuous codes,
   STE-rounded to multiples of `step` so q-entropy stays low (rate preserved); `codes_and_scale`.

**The key design correction — TEACHER DISTILLATION, not GT-fit:** the frontier is a MEMORIZED
single-video renderer already at d_seg=5.6e-4. The d_seg rise from coarsening is the weight
PERTURBATION, NOT a GT-fidelity gap. So the recovery TEACHER is the FRONTIER decoder's own render
(the full-precision model the quantized student must match) — canonical QAT teacher/student. The
loss pulls the coarse-recovered render back toward the frontier render (both through the same
eval_roundtrip). A first attempt at GT-fit (score-aware-vs-GT only) DIVERGED (d_seg → 3.1e-3),
because optimizing the soft-cosine seg surrogate toward GT moves AWAY from the memorized point.

**Loss** `= w_distill·MSE(student_render_rt, teacher_render_rt) + w_score·(100·d_seg_surrogate +
√10·√d_pose_surrogate)`, eval_roundtrip applied to both renders (NON-NEGOTIABLE), differentiable
rgb_to_yuv6 patched before scorer construction (PR#95/#106 contract), score weights NONZERO when
w_score>0 (Catalog #384).

### Canonical-vs-unique decision per layer
- **Reconstruction engine (fec6_decoder_mutations + feca_selector_reparameterize):** ADOPT_CANONICAL
  (reuses the prior lane's exact split/prepare/recompress engine; no duplicative code).
- **Render path (frontier runtime model.HNeRVDecoder):** ADOPT_CANONICAL (the eval's own decoder).
- **eval_roundtrip + differentiable scorers + score_pair_components_dispatch:** ADOPT_CANONICAL
  (the CLAUDE.md-mandated training-signal contract).
- **Teacher-distillation target + STE round-to-grid + per-tensor scale params:** FORK_PRINCIPLED
  (the coarse-grid constraint + memorized-teacher target are specific to this recovery; no canonical
  QAT helper targets a packed split-brotli int8 decoder on a coarse grid).

---

## 2. The recovery curve (advisory screen) — the lever has ~ZERO d_seg-recovery capacity

All runs target the 3 bottom-decile tensors at step=2 (the gentlest, most-recoverable operating
point — c1 in the prior lane). The advisory screen byte-closes each recovered candidate and scores
EXACT d_seg/d_pose via `DistortionNet.compute_distortion` on a held FULLY-CLEAN eval subset (the
[macOS-CPU advisory] kill-gate; the prior lane validated this gate prices large decoder distortion
changes within ~9% of the contest-CPU authority).

The frontier-decoder eval-subset baseline is **d_seg≈5.4e-4**; coarsening (no fine-tune) pushes it
to **d_seg≈1.2e-3** (the +0.069 problem). The recovery target is to pull d_seg back toward 5.4e-4
while keeping the ~9,300-byte rate gain.

| mode | LR (codes/scale) | budget | best d_seg | vs coarse | bytes saved | best dS_total advisory |
|---|---|---:|---:|---|---:|---:|
| scale_only | —/5e-3 | 4 | 1.110e-3 | 0% (stuck at coarse) | 9,308 | +0.060 |
| codes_and_scale | 5e-3/5e-3 | 8 | 1.131e-3 | ~0.8% recovered | 9,308 | +0.056 |
| codes_and_scale | 5e-2/1e-2 (foreground) | 30 | 1.227e-3 (it6) | 0% | drops 9,306→2,532 | +0.069 |
| codes_and_scale | 5e-2/1e-2 (detached) | 40 | 1.201e-3 (it10) | ~8% | drops 9,308→3,328 | +0.072 |

**Two independent failure mechanisms, both fatal:**

1. **The even-code grid cannot represent the precision the un-coarsened decoder uses.** Per-weight,
   the coarsening (round-to-nearest-even) is ALREADY the optimal even code; LSQ can only exploit
   cross-weight coordination + the global scale, which recover **<1% of the d_seg rise** (within the
   eval-subset noise). Gradient flow is healthy (code grads nonzero on ~all 46,656/48,384 weights,
   log-scale grads strong — verified) — this is a genuine REPRESENTATIONAL capacity limit, not a
   training bug. scale_only (3 global fp16 scalars) recovers exactly 0% — the perturbation is
   per-weight, not a global-scale error.

2. **As the distill MSE "improves", the codes drift OFF the low-entropy grid distribution and the
   RATE GAIN EVAPORATES.** At lr_codes=5e-2 the foreground run drove distill loss 28.3→15.6, but the
   recovered codes spread to more distinct even-multiples → brotli compresses worse → bytes-saved
   COLLAPSES from 9,306 to 2,532, while d_seg stayed WORSE than coarse (1.23e-3→1.28e-3). The MSE
   reduction is a pixel proxy that does not translate to fewer SegNet argmax-flips. So training
   harder makes the candidate STRICTLY WORSE on both axes (loses rate AND keeps the d_seg penalty).

**Best advisory dS_total across the entire sweep = +0.056** — ~9× the rate gain (−0.0062), the same
order as the prior lane's naive-coarsen verdict (+0.0709). No configuration approached favorable.

---

## 3. Verdict + routing (Catalog #125)

### Verdict: coarsen-then-grid-LSQ-retrain recovery is FALSIFIED-AT-IMPLEMENTATION
(NOT a paradigm kill — Catalog #307; reactivation criterion #2 of the decoder-axis verdict tested
the SPECIFIC "fine-tune the coarsened scales+codes" implementation; it is falsified, the broader
decoder-distortion-recovery paradigm is intact). The lever has **~zero d_seg-recovery capacity**:
the coarse even-code grid cannot represent the memorized renderer's required precision, and pushing
the distill objective harder erodes the rate gain faster than it recovers fidelity. The frontier
remains **0.19198534 [contest-CPU]**; no candidate promoted; no submission.

### No paid Modal eval was spent (MVP-first kill-gate fired)
Per CLAUDE.md "Carmack MVP-first phasing": the [macOS-CPU advisory] kill-gate fired DECISIVELY at
+0.056 (best of the entire sweep, ~9× the rate gain), and the prior lane empirically validated that
this advisory gate prices large decoder distortion changes within ~9% of the contest-CPU authority.
A $0.3 Modal eval of a candidate the advisory screen rules out at +0.056 (4 orders of magnitude
above the eval reproducibility floor) would be paid-dispatch-first against a fired kill-gate. The
parity check (§0) is the apples-to-apples bridge: the in-process render == the byte-faithful inflate
== the eval pixels (≤1 LSB).

### Why this is the DEEP finding (permanent knowledge): the memorized renderer has NO redundant precision
The decoder-axis verdict found the decoder is already INT8 + already at the brotli entropy floor, so
distortion is the only lever. THIS lane finds the distortion lever has no recovery headroom either:
1. **A memorized single-video renderer's weights carry NO quantization-redundant precision.** Unlike
   a generalizing codec (where QAT recovers low-bit PTQ collapse because the task tolerates weight
   noise), every frontier decoder weight is overfit to the 1200 contest frames; the LSB of each INT8
   code is doing real frame-specific work. Coarsening destroys it; no coarser-grid retrain restores
   it because the coarser grid literally cannot hold the value.
2. **The canonical low-bit-PTQ-collapse fix (QAT/LSQ) ASSUMES the model can re-learn at lower
   precision.** It cannot here, because there is nothing to re-learn TO — the model is already at its
   memorized optimum, and the constraint is representational (grid resolution), not optimization.
3. **Rate-distortion on a memorized decoder is a knife-edge:** the frontier sits exactly on its
   decoder-distortion Pareto vertex. Any precision reduction (uniform OR sensitivity-aware OR
   retrained) moves UP the d_seg axis faster than it moves LEFT the rate axis, because the 100·d_seg
   coefficient × the all-frames weight-sharing of early layers dominates the rate waterline by ~75×.

### Routing: DEFER-pending-research (per CLAUDE.md "Forbidden premature KILL")
The decoder-axis distortion lever is NOT killed; two of the three implementations are now falsified
(naive uniform coarsening per the prior lane; grid-LSQ retrain per THIS lane). The remaining
reactivation criteria (each tests a distinct assumption the falsified two did not):
1. **Score-aware PER-WEIGHT quant (free int8 + entropy/rate penalty, NOT grid-constrained):** keep
   the high-sensitivity weights at full INT8, coarsen ONLY a sparse low-per-weight-gradient subset,
   with a differentiable rate (brotli-proxy) penalty so the rate gain is OPTIMIZED rather than
   destroyed. Tests: "is the distortion concentrated in a sparse weight subset that can be coarsened
   without the all-frames d_seg propagation?" This is decoder-axis verdict reactivation #1 + the
   THIS-lane finding that the grid constraint (not the retrain) is the binding wall. Requires
   per-WEIGHT score-gradient extraction + a differentiable rate proxy.
2. **Latent-axis instead of decoder-axis** (15,387 B, 8.6% of bytes, ~0.02% of sensitivity per the
   map) — the lowest-sensitivity remaining surface (R3 selector + the per-tensor map both point here).
   Small absolute bytes but a far more favorable sensitivity-per-byte target; no memorized-renderer
   d_seg knife-edge.
3. **Full-decoder score-aware re-training from the frontier checkpoint at a NEW smaller architecture**
   (fewer channels) — NOT a byte-transform; a fresh score-aware training run that re-memorizes the
   video at a smaller decoder. This is the only path that can actually shrink the decoder without the
   precision wall, but it is a multi-hour training campaign, not a recovery loop.

### The frontier remains 0.19198534 [contest-CPU]; no candidate promoted; no submission.

### Wire-in (Catalog #125)
- **Hook #1 sensitivity-map:** N/A — reuses the prior lane's per-tensor byte×sensitivity map; this
  lane adds no new sensitivity surface (it consumes the bottom-decile ranking).
- **Hook #2 Pareto:** confirms (a second, independent way) that the frontier is on its decoder-
  distortion Pareto VERTEX — d_seg cannot be traded for rate via coarsen-then-retrain any more than
  via naive coarsen. The decoder is rate-AND-distortion saturated for any grid-quantized mutation.
- **Hook #3 bit-allocator:** the bit-allocator must NOT treat the bottom-decile tensors as
  coarsenable — even WITH a retrain-recovery model, the realized recovery is ~0% and the rate gain
  is unstable. The allocator's quant decision needs per-WEIGHT gradient + a differentiable rate
  penalty + a free (non-grid) code space, per reactivation #1.
- **Hook #5 continual-learning:** the recovery curve (`recovery_result_*.json`) is the permanent
  empirical anchor that grid-LSQ decoder recovery is dominated; it reseeds the V3 ΔS-judge that the
  decoder-distortion lever has no retrain headroom on a memorized renderer.
- **Hook #6 probe-disambiguator:** RESOLVED "is the coarsened frontier decoder's d_seg recoverable
  by fine-tuning?" → NO (the canonical QAT assumption fails on a memorized renderer; the binding
  constraint is grid representational capacity, not optimization). Also RESOLVED that distill-MSE
  reduction does NOT track d_seg recovery (pixel proxy ≠ argmax-flip rate) — a reusable caution for
  any future render-distillation recovery.
- **Hook #4 cathedral-autopilot:** N/A — no promotable archive emitted (all candidates above frontier).

### Cost
Parity check + recovery sweep (scale_only + codes_and_scale at 3 LR/budget points) + capacity probe
= **$0 (macOS-CPU)**. No paid Modal eval (the advisory kill-gate fired at +0.056; spending against a
fired gate violates MVP-first). Under the $5 STOP gate by construction.
