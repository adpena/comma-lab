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

## Routing under goal v2
The recoded-R3 hold is banked (CUDA pairing in flight = readiness, NOT the submission we want). The
offensive lever is B or C (the cleanest class shifts with all aiming surfaces in hand), with A as the
unifying frame and D as the principled atom-coding upgrade. The research+ideation agent ranks these +
adds the literature directions (neural video compression SOTA, task-aware/perceptual compression,
INR/NeRV frontier, the steganalysis lineage) + whatever we haven't conceived. Then the top direction
gets a $0 descent-smoke before any campaign spend (MVP-first).
