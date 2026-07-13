# DIRECTIVE (operator-routed, 2026-07-13) → muonh_manifold_muon_dig: AdamW is IN SCOPE — "the adamw implementation is also likely not optimal"

Your module-to-norm derivation covers the WHOLE optimizer stack, not just the Muon finishing stage. The witness
trains stages CE→τ→l7 under AdamW (+ AdamW on non-matrix params during Muon, PR95 heritage L15: 177K Muon / 51.8K
AdamW). Audit + derive:

1. MODULAR-NORM VIEW OF THE ADAMW STAGES: Adam ≈ steepest descent under a max-norm/sign geometry (Bernstein's own
   analysis). Derive what norm the AdamW stages SHOULD use per module from the same theory as your Muon table —
   if the early stages' natural geometry differs from Adam's implied one, that mismatch is a candidate epochs-to-
   target lever exactly like the Muon delta.
2. IMPLEMENTATION SEMANTICS AUDIT (concrete, checkable): does the trainer's AdamW (MLX mlx.optimizers or custom —
   READ the levelset trainer's optimizer construction) match torch AdamW semantics: (a) decoupled weight decay
   scaled by lr or not; (b) bias-correction form; (c) eps INSIDE vs OUTSIDE the sqrt (a known cross-framework
   divergence that changes small-gradient behavior — our island/lane gradients ARE small); (d) which param groups
   get decay (biases/gains/FiLM excluded or not — decay on gains is usually wrong).
3. HYPERPARAMETER PROVENANCE: β₁=0.9/β₂=0.999/eps defaults are LLM-tuned. We have BANKED-NEVER-FIRED levers here:
   #222 (β₂-sweep disambiguator + β₁<√β₂ guard, arXiv 2603.02092) and #223's β₂-from-n derivation (Tier-0). Fold
   them: DERIVE β₂ from our actual n600/micro-batch noise scale rather than defaulting; state the derived value with
   provenance.
4. WEIGHT DECAY ON AN OVERFIT-ONE-CLIP TASK: decay fights memorization — for the witness (deliberate single-clip
   overfit) is wd>0 harmful, neutral, or actually a RATE regularizer (flat-minima/MDL, #242)? Derive the answer from
   the objective; if it is a rate lever, say what the decay coefficient buys in bytes vs d_seg.
Same deliverable/A-B-ticket protocol as your Muon scope; same adversarial bar (beat the TUNED incumbent). The
combined table (Muon modules + AdamW modules + derived β₂ + decay verdict) is the full optimizer-stack optimality
answer.
