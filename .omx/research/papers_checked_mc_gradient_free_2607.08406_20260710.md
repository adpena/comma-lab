# Papers-checked: arXiv 2607.08406 — "Beyond Backpropagation: Monte Carlo Method Can Train Deep Neural Networks" (Hong Zhao)

Date: 2026-07-10 · operator-supplied link · anti-re-research ledger (sister of papers_checked_* series).
STORES CONSULTED: #350 (deterministic-differentiable-GPU decode + payload-space TTO + exact-attribution
harness, completed) · #341 (quadratic basin GN/CG finisher, completed) · #391 flip solver (exact composite-R
adjoint + flip ledger + corrected step law) · #157 sensitivity waterfill · Unit B (int8 past RD knee;
per-class waterfill dominated) · surrogate≠authority discipline (NO-FAKE #8) · #310 unswept gauss/step arm.

## The paper
Gradient-free training: randomly mutate ONE parameter, keep if loss decreases, else retry ((1+1)-ES-style
stochastic local search). Trains >20-layer nets, 16K-neuron single layers, a small Transformer — on MNIST /
Tiny-Shakespeare (toy scale, honest). Notable capabilities: DISCRETE weight support (train IN the quantized
space, no STE), non-standard activations (Gaussian), pure pruning during training.

## Verdict — RELEVANT, one genuinely novel-to-us lever concept (no kill, nothing re-opened)
**The exact-metric Monte-Carlo FINISHER.** Our binding functional is the DISCRETE argmax d_seg — every
training path optimizes a smoothed surrogate and pays the surrogate↔exact gap at the finish. An
accept/reject loop needs NO gradient ⇒ it can optimize the EXACT through-R n600 d_seg directly, where our
gradient signal is weakest (terminal stage, a few thousand residual flips).
Made practical by three things we ALREADY BUILT:
1. **Guided proposals, not blind mutation:** the #391 flip ledger + corrected step law (d*=Dᵀ∇m) + #157
   per-tensor sensitivity rank WHICH parameters/directions can flip pixels — proposal distribution
   concentrated on the flip-adjacent subspace (acceptance rate ≫ blind random).
2. **Cheap exact-ish accept tests:** the #350 exact-attribution harness + content-addressed search were
   built exactly for payload-space accept/reject; ladder = subsample screen → n600 through-R confirm
   (the n600 confirm is the authority; subsample is a SCREEN, never the verdict — P9).
3. **Discrete-space composition:** mutate IN int8/discrete weight space (the paper's discrete support) ⇒
   the finisher output is already byte-closed; pairs with Unit B's "int8 past the RD knee" — an MC finisher
   could claw back quantized d_seg at FIXED bits (fine-tune where STE can't follow).
Scope-honest caveats: paper is MNIST-scale; per-eval cost is OUR wall (exact n600 verdict ~23 min ⇒ the
finisher must mutate few high-saliency params in batches with subsample screening); this is a TERMINAL
micro-finisher (out_sdf head / palette / out_tex — thousands of params), not a training paradigm swap.
Secondary: gradient-free training un-blocks the UNSWEPT gauss/step activation arm (#310) where fixed-β
saturation kills backprop — an MC arm needs no gradient through the step.

## Disposition
- Lever concept registered as task (exact-metric MC finisher, gated behind owed-16/#385 + the texture-trunk
  P0 — machine-bound). No equation yet (no measured row of ours — registered on first measurement).
- MEMORY L55 papers-checked line: covered by this memo (index updated at next MEMORY.md touch).
- means≠ends: pointer 0.19110 UNMOVED; this is anti-re-research banking + a lever concept.
