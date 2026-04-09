# endgame checklist

When active experimentation winds down (or we want to set up multi-day unattended runs), execute these in order. Each is designed to squeeze the last drops from every axis.

## Phase 1: Max-width scaling (1-3 days unattended)

- [ ] PixelShuffle+Dilated at h=96 (1500 epochs) — LeCun crossover test
- [ ] PixelShuffle+Dilated at h=128 (2000 epochs) — rate penalty boundary
- [ ] PixelShuffle+Dilated at h=48 (1000 epochs) — byte-optimal sweet spot
- [ ] Plain h=128 (2000 epochs) — does vanilla ever catch PixelShuffle?
- [ ] Run ALL on bat00 CUDA for faster iteration

## Phase 2: Hyperparameter meta-sweep (1-2 days)

- [ ] Alpha sweep on best architecture: {2, 5, 10, 15, 20, 30, 40}
- [ ] EMA decay sweep: {0.99, 0.995, 0.997, 0.999, 0.9995}
- [ ] Learning rate sweep: {1e-4, 3e-4, 5e-4, 1e-3}
- [ ] Warmup epoch sweep: {0, 5, 10, 20, 50}
- [ ] Gradient clip sweep: {0.3, 0.5, 1.0, 2.0, none}
- [ ] Saliency lambda sweep: {0.01, 0.05, 0.1, 0.2, 0.5}
- [ ] Accum steps sweep: {1, 2, 4, 8}
- [ ] Train subsample sweep: {2, 4, 8, 16} (more pairs per epoch)

## Phase 3: Quantization optimization (hours)

- [ ] Per-channel int8 on every candidate
- [ ] LSQ (learned scale) on the best architecture
- [ ] Mixed precision: fp32 biases + per-channel int8 weights
- [ ] Prune 10%/20%/30% smallest weights then retrain 200 epochs
- [ ] Huffman entropy coding of int8 weights (custom packer)

## Phase 4: Architecture exhaustive search (2-5 days)

- [ ] Kernel size sweep: {3, 5, 7} on best PixelShuffle variant
- [ ] Dilation pattern sweep: {[1,1,1,1], [1,2,1,1], [1,2,2,1], [1,1,2,1]}
- [ ] Depth sweep: {3, 4, 5, 6} layers in PixelShuffle body
- [ ] PixelUnshuffle factor: {2, 3, 4} (quarter-res, ninth-res)
- [ ] Squeeze-excite on best variant (+128 params)
- [ ] Pair-aware 6-channel input on best variant
- [ ] FiLM per-scene conditioning on best variant
- [ ] Residual scaling factor sweep: {0.1, 0.25, 0.5, 1.0}

## Phase 5: Multi-seed validation (1 day)

- [ ] Train best architecture with 5 different random seeds
- [ ] Report median ± std of scorer score
- [ ] Promote the MEDIAN seed, not the best (von Neumann rigor)
- [ ] Cross-validate on hold-out videos if available

## Phase 6: Submission hardening (hours)

- [ ] Rebuild archive.zip with best weights bundled
- [ ] Dry-run inflate.sh end-to-end
- [ ] Authoritative CPU scorer confirmation
- [ ] fp64 scorer recompute for bit-exact verification
- [ ] Cross-CRF robustness check (CRF 32, 34, 36)
- [ ] Update PR draft with final numbers
- [ ] Update writeup with complete chain of insights
- [ ] Review all durable state for consistency
- [ ] Final compliance audit (bug review agent)

## Phase 7: Submit (minutes)

- [ ] File PR with best verified score
- [ ] Include writeup in PR body
- [ ] Link to archive.zip download
- [ ] Wait for official evaluation
- [ ] Celebrate 🎉

## Kill criteria for each phase

- If a phase produces <0.005 improvement after full sweep, move to next
- If load average >40 for >1 hour, kill lowest-priority experiment
- If any experiment diverges (NaN, pose >0.10), kill immediately
- If deadline is <5 days away, jump to Phase 6
