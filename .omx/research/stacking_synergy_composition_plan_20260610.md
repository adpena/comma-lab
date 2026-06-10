# Stacking / synergy / positive-externality composition plan (2026-06-10)

Operator: "think about stacking and composing and synergy and positive externalities and more."
The composition ALGEBRA (`composition_algebra_coherence_law_20260610.md`) says HOW to admit a bundle
correctly (distortion-domain, sequential, ledger-debited). THIS memo is the PLANNER: which axes
exist, which compose ADDITIVELY (orthogonal → stack freely), which INTERACT (need a measured joint
row), and the maximum-synergy bundle to assemble. Authority: every composed bundle is admitted only
by exact paired eval; predictions here are derivations to be confirmed.

## The orthogonality map (the key to free stacking)
Two moves stack ADDITIVELY iff they touch disjoint score-mechanisms OR disjoint archive sections
separated beyond the receptive-field radius. The axes, with PROVEN orthogonality:

| Axis | Touches | Section | Orthogonal to |
|---|---|---|---|
| R3 selector/framing recode (−22B, ours) | rate only | selector/framing | EVERYTHING (rate, disjoint section) |
| R1 decoder entropy recode (PR#112, −1060B) | rate only, LOSSLESS | decoder weights | selector, latents, all distortion axes |
| R2 latent AR+range recode (−317B) | rate only, LOSSLESS | latents | selector, decoder, all distortion axes |
| S12 resize-null preimage (−10-19.5% coded) | rate only, CERTIFIED zero distortion | any frame bytes | ALL (touches only invisible DOF) |
| Class-3 seg-repair atoms | d_seg (−) | sidecar/correction | rate axes; pose IF in pose-null cone |
| frame-0 pose selector | d_pose only (SegNet-blind, EXACT) | selector | d_seg entirely |
| AFSR-1 retrained weights | d_seg + d_pose + rate | decoder | replaces the base — everything downstream re-stacks |

**The four lossless rate moves (R1+R2+R3+S12) are MUTUALLY ORTHOGONAL** — different sections, all
zero-distortion → they stack ADDITIVELY with proof-by-construction (lossless = identical pixels =
identical d_seg/d_pose; the byte savings simply sum). This is the immediate free-stack: not −1,381B
(PR#112's decoder+latent) but potentially −1,381 −22 (R3 selector) − S12(invisible-DOF on the frame
payloads) — a STRICTLY LARGER win than any competitor has, assembled from orthogonal axes.

## The interaction terms (where naive stacking fails — measure, don't sum)
- **Class-3 atoms × each other**: same-class-region atoms SATURATE (both fix the same flips) →
  sequential admission + cone-ledger debit (the coherence law). NOT additive.
- **Class-3 (seg, frame1) × frame-0 pose selector**: the both-frame composite — frame1 seg gain may
  cost pose; frame0 compensates at zero seg cost (the asymmetry used CONSTRUCTIVELY). Measured
  commutator required; this is positive-externality stacking (each enables the other's budget).
- **AFSR-1 retrain × everything**: a new base INVALIDATES all atom/recode rows minted on the old base
  (sha staleness). But POSITIVE EXTERNALITY: a better-trained decoder has DIFFERENT (likely fewer,
  differently-placed) residual flips → the flip map / atlas / cone REGENERATE on the new base and the
  whole atom+recode stack re-applies, compounding. Retrain → re-map → re-stack is the master loop.
- **E3 cross-pair pose fungibility**: pose is pooled-mean-before-sqrt → pose moves on ANY pair trade
  1:1 across pairs. This makes pose a GLOBAL budget the planner allocates, not a per-pair constraint —
  a synergy multiplier (spend pose where it's cheapest, anywhere in the 600).

## Positive externalities (moves that make OTHER moves cheaper/possible)
1. **S12 preimage BEFORE entropy recode**: filling the 22.7% invisible pixels with maximally-
   compressible values makes R1/R2's entropy coders MORE effective (lower-entropy input) — S12 is a
   force-multiplier on every rate move, not just an independent saving. (Test the joint, not the sum.)
2. **AFSR-1's flip reduction shrinks the seg pool** → Class-3 atoms become CHEAPER (fewer flips to
   address → the 1.525 B/flip floor that blocked the sidecar may drop below break-even on a
   smaller flip set).
3. **The invisibility basis as a TRAINING constraint** (not just a postprocess): train AFSR-1 to put
   its representation error INTO the null space → certified-free error. The strongest synergy —
   architecture trained to be cheap-to-encode by construction.
4. **Per-axis tuning (the CUDA lesson)**: the same stack re-tuned per authority axis — the CPU bundle
   and CUDA bundle are different optimal stacks; building both is required for the dual-axis gate.

## The planner (what V3's stacking loop should do)
Maintain a typed candidate pool tagged {axis, section, distortion-touched, orthogonality-class}.
Greedy-by-value with the coherence law: (1) admit all MUTUALLY-ORTHOGONAL lossless rate moves as one
batch (proof-by-construction, one re-measure) — the free stack; (2) sequential-admit distortion moves
with re-measure + ledger debit; (3) measure commutators only for same-section / same-region / both-
frame pairs; (4) on any base change (AFSR-1), invalidate + re-map + re-stack. Every bundle → exact
paired eval → frontier pointer only on a proven beat.

## Immediate synergy bundle (assemble NOW, all in-flight or ready)
R1 (leapfrog building) ⊕ R2 (latent recode, small-build) ⊕ R3 (our selector win, landed) ⊕ S12
(preimage, landed) — four orthogonal lossless moves, predicted strictly < PR#112's 177,136 B,
zero fidelity risk, one paired eval to ratify. THEN the distortion axis (Class-3 + AFSR-1) stacks on
the smaller-byte base. Consumers: the leapfrog agent (fold R3+S12 into its recode bundle, not just R1).
