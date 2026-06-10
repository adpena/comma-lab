# Innovation mandate + original-direction menu (operator, 2026-06-10)

Operator: the recoded-R3 win (0.19109982, −2.59e-5 over PR#112, built FROM PR#112's codec) is within
contest reporting precision and not original — submitting it makes the "competitive OR innovative"
statement questionable. **New binding bar: a submission must be genuinely ORIGINAL/INNOVATIVE (a class
shift or a novel method) AND meaningfully below frontier (not a noise-margin absorb-recode).** The
recoded-R3 frontier is a DEFENSIVE HOLD (banked, CUDA-paired for readiness) — NOT the offensive
submission. The offensive play is one of the directions below. This memo seeds them; a research+
ideation agent expands/grounds/adds what we haven't conceived.

## The reframe that unlocks originality
The contest measures TWO things and a byte cost: PoseNet = 6-dof ego-motion (first 6 of 12 dims),
SegNet = 5-class semantic layout on frame1. HNeRV-family carriers (ours + every leaderboard entry)
compress the VIDEO (RGB pixels) and let the scorer derive its measurements. That is pixel-native, and
it is why the whole field clusters at 0.19-0.20 — everyone is solving a harder problem than the score
asks. The original move is to be **SCORE-NATIVE**: store/synthesize only what the evaluator measures.

## The original directions (mine — ranked by innovation × headroom × feasibility)

### A. Evaluator-equivalence quotient compiler (the V6 thesis, realized) — HIGHEST innovation
Stop compressing the video; directly CONSTRUCT the minimum-description archive whose inflate output
lands in the same evaluator CELL — SegNet-argmax-identical on frame1 ∧ PoseNet within-tube on the
pair. New problem FORMULATION (task-conditioned MDL under a frozen oracle), not a new codec. We have
the pieces: invisibility basis (80.67% of pixel-DOF the scorer can't see), the cone/flip-map (the
argmax-cell boundary), the preimage compiler, the composition algebra, the V6 FrozenEvaluatorContract
scaffold. Headroom: potentially large (pay zero for everything outside the cell). "Innovative" is
unquestionable — it is a reframing of the problem.

### B. Score-native decomposition carrier — HIGH innovation, the cleanest class shift
A carrier whose SECTIONS are the measured quantities, not pixels:
  (1) ego-motion trajectory — ~600×6 pose floats, kilobytes, the literal PoseNet target;
  (2) semantic layout — the SegNet argmax structure (5-class), which IS the scored seg quantity;
  (3) minimal appearance — only enough to keep the argmax from flipping (the margin/cone budget) and
      to keep PoseNet's YUV6 tube (the luma motion).
This stores WHAT IS MEASURED + the minimal carrier to reproduce the measurement, vs a full RGB
renderer. Fundamentally different decomposition; directly attacks all three score terms at their
source. Feasibility: real engineering, all aiming surfaces exist.

### C. Fresh-init score-aware NAS/training, null-space-primary — HIGH (the AFSR-1 reactivation)
The killed AFSR-1 continuation degraded because the memorized point has no slack. A FRESH small
architecture, searched/trained under the byte+scorer Lagrangian with the null-space constraint PRIMARY
(error lives in certified-invisible DOF by construction), aimed by flip-map/atlas/cone, QAT-in-loop,
MLX-first. This is the "long training + class shift" path. Innovation: the training METHODOLOGY
(geometry-aimed, null-space-native objective) is novel even if the arch family isn't.

### D. Inverse-steganalysis-native coding — HIGH, uses the contest's OWN theory
The contest IS inverse steganalysis (Yousfi = Fridrich's student; SegNet/PoseNet are the "steganalysis
detectors"). STC / syndrome-trellis coding minimizes embedding cost subject to a payload — invert it:
minimize ARCHIVE BYTES subject to staying inside the argmax cell, with per-pixel cost = the margin/
cone budget (UNIWARD-style). We have STC + UNIWARD surfaces (deferred). This is the theoretically
principled version of the seg-repair/atom work that the sidecar floor blocked — STC's coding-theoretic
efficiency may beat the 1.525 B/flip naive floor.

### E. Generative/implicit micro-prior — HIGH innovation, higher risk
A tiny conditional generator (diffusion-distilled or coordinate-INR) that produces evaluator-valid
frames from minimal latents, trained to land in the cell rather than match pixels. Feasibility risk
(inflate runtime + drift), but a genuinely novel carrier.

### F. Information-theoretic floor derivation — original ANALYSIS (sets the target)
Compute the actual minimum bits to specify a member of the evaluator's equivalence class of THIS
video (the lower-bound ledger's T_floor). Not a score move alone, but it proves how much headroom
exists, makes the goal's threshold ladder principled, and is publishable original analysis.

### G. Engineered deterministic corrections — ZERO-byte distortion lever (operator, 2026-06-10)
inflate.py-resident DETERMINISTIC transforms on the decoded frames (NO scorer loaded at inflate per the
strict-scorer rule; NO archive bytes — pure code in the rate-free inflate program) that nudge frame1
toward the SegNet-argmax cell and the pair toward the PoseNet tube. **Canonical proof it works:** the
PR95-family L28 decode-side channel postprocess (subtract 1.0 from specific RGB channels; 0 archive
bytes; ~−0.0001 to −0.0005 [contest]). The MATH (must be a real solve, not a per-pixel search — class-6
fake): a fixed correction field c(x) DERIVED from the measured flip-map / atlas / cone — flip the
most-flip-prone boundary pixels back (where the SegNet top-2 margin is smallest, our 91% margin<0.5
set) and shrink the dominant pose residual direction — applied as one closed-form decode-time op.
Because it costs ZERO archive bytes and touches only decoded frames, it stacks ORTHOGONALLY on EVERY
carrier and is the cleanest move on the "distortion threshold at CONSTANT bytes" path to sub-0.15.
**Reuse (verified present):** `src/tac/engineered_corrections.py` + `engineered_corrections_v2.py` +
`engineered_correction_readiness.py` (+ tests). Risk: must be DERIVED from the oracle geometry and
exact-eval-ratified, never a brute-force per-pixel sweep dressed up as a "correction."

### H. Super-cheap small postfilters — low-byte distortion lever (operator, 2026-06-10)
A TINY (target ≤ a few KB quantized) learned residual postfilter run at inflate time on decoded frames,
trained SCORE-AWARE (eval_roundtrip + differentiable YUV6; NO scorer at inflate; within the 30-min T4
budget) to reduce d_seg/d_pose. Distinct from G: G is zero-byte deterministic, H is small-byte learned —
so H MUST PAY RENT (its quantized weight bytes enter the archive; admit only if ΔS_distortion beats
Δrate, the pays-rent gate). Stack G (free) FIRST, then H only if it still pays on the corrected base.
**Reuse (verified present):** `experiments/train_postfilter_on_renderer.py` +
`experiments/modal_hdm8_postfilter_sweep.py` + `experiments/postfilter_weights/` + the postfilter
loader/parity/no-op tests. MLX-first for training; QAT-in-loop so the deployed int8 weights are what was
scored. Both G and H are sequential-admit distortion moves under the composition algebra (re-measure +
cone-ledger debit), and both compose on top of the carrier the offensive lever (B/C) produces.

## Routing under goal v2
The recoded-R3 hold is banked (CUDA pairing in flight = readiness, NOT the submission we want). The
offensive lever is B or C (the cleanest class shifts with all aiming surfaces in hand), with A as the
unifying frame and D as the principled atom-coding upgrade. The research+ideation agent ranks these +
adds the literature directions (neural video compression SOTA, task-aware/perceptual compression,
INR/NeRV frontier, the steganalysis lineage) + whatever we haven't conceived. Then the top direction
gets a $0 descent-smoke before any campaign spend (MVP-first).
