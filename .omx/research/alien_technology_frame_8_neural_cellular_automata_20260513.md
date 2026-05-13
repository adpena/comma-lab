# Frame 8 — Neural Cellular Automata renderer (alien-tech ledger 2026-05-13)

**Parent memo**: `.omx/research/alien_technology_unknown_unknowns_research_20260513.md` §8.4.
**Lane**: `lane_alien_technology_unknown_unknowns_research_20260513` (L0).

## Worldview

A civilization that took Wolfram's "A New Kind of Science" (2002) as foundational. They write programs as **local rules** + initial conditions; complexity emerges from iteration.

## Core inductive bias

**Local rules produce global patterns.** Encode the rule + seed; let computation produce the artifact.

## Concrete technique 8.4 — Neural Cellular Automata renderer

[Mordvintsev-Niklasson-Randazzo 2020](https://arxiv.org/abs/2009.01410), [Distill 2020](https://distill.pub/2020/growing-ca/).

Architecture: a small CNN (4-8 layers, ~10K parameters ≈ 5 KB) defines a CELL UPDATE RULE. The rule is applied 32-64 times to a SEED (a small image) to produce the target output.

```python
def neural_ca_step(state, rule_cnn):
    # state: (B, C, H, W); C = hidden channels (e.g., 16)
    perceived = perceive(state)           # 3x3 filter producing dx, dy, dxx, dyy etc.
    update = rule_cnn(perceived)          # tiny CNN
    return state + update * alpha          # residual update

def render_frame(seed, rule_cnn, n_steps=48):
    state = pad_state(seed, hidden_channels=16)
    for _ in range(n_steps):
        state = neural_ca_step(state, rule_cnn)
    return state[:, :6, :, :]              # output YUV6 channels
```

**Empirical results** (Distill 2020): NCA grows complex images from a single seed cell after 1000-10000 training iterations. For driving video, the rule + seed approach naturally exploits **scale-invariance** + **local patterns** (lane markings, road texture).

## Concrete byte budget

| Component | Bytes | Reasoning |
|-----------|------:|-----------|
| Rule CNN | ~10 KB | 4-layer CNN, ~5-10K params at FP4-FP8 |
| Per-frame seed | ~16 bytes | 16-byte seed embedding |
| 1200 seeds | ~19 KB | 1200 × 16 bytes |
| Header / format | ~50 bytes | |
| **TOTAL** | **~29 KB** | |

**At PR106 frontier (187 KB), this is 85% reduction.**

**Rate Δscore**: (187-29) × 6.66e-7 = 0.000105 = -0.000105 (negligible at PR106 frontier).

**Distortion Δscore**: indeterminate. NCA produces emergent textures that may NOT match contest video at sufficient fidelity. **The hypothesis is that for SEGMENTATION-RELEVANT features (smooth road + lane markings + horizon), NCA is sufficient.**

## Tractability assessment

**Cost to build smoke**:
- ~1 GPU-hour to train NCA on 100 frames (~$0.50)
- ~$1 to evaluate proxy-MSE on next 100 frames
- ~$0.50 to test FULL contest scoring via inflate.sh

**Total smoke probe: ~$2-3.**

## Implementation notes

- Use the [Growing-NCA implementation](https://github.com/google-research/self-organising-systems/tree/master/notebooks) as scaffold.
- Critical: NCA training requires regularizers for **convergence stability** (resampling, stochastic-update masking).
- The seed embedding can be initialized as the first frame's downsampled version; learned via gradient descent.
- Hybrid: NCA + small per-frame residual neural network for pose-relevant detail.

## Risk assessment

**HIGH RISK**: NCA may produce blurry / texture-deficient frames → score regression.
**HIGH REWARD**: 85% byte reduction if it works.

## Recommended smoke probe protocol

```python
# Stage 1: train NCA on 100 frames
nca = NeuralCellularAutomaton(channels=16, rule_cnn_layers=4)
for epoch in range(1000):
    seeds = torch.randn(BATCH, 16, 16, 16)  # 16x16 seeds
    targets = videos_0_mkv[batch_indices, :6, :, :]
    output = nca.render(seeds, n_steps=48)
    loss = mse(output, targets) + scorer_distill_loss(output, targets)
    loss.backward()
    optimizer.step()

# Stage 2: evaluate proxy MSE on next 100 frames
test_seeds = optimize_seeds_per_frame(nca, videos_0_mkv[100:200])
proxy_score = evaluate_proxy_score(nca, test_seeds, videos_0_mkv[100:200])

# Stage 3: build archive with rule + seeds; auth-eval
archive_bytes = build_nca_archive(nca, test_seeds)
score = inflate_and_evaluate(archive_bytes)
```

## Wire-in declaration

All 6 hooks: N/A pending operator approval of Decision A (NCA substrate investment) in parent memo §15.1.

## Research-only tag

`research_only=true`.

## Closest extant work

- [Distill 2020 — Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/)
- [Mordvintsev 2020 — Neural Cellular Automata](https://arxiv.org/abs/2009.01410)
- [Google research self-organising systems](https://github.com/google-research/self-organising-systems)
- [Wolfram 2002 — A New Kind of Science](https://www.wolframscience.com/nks/)
- [Cellular automata image compression — Cloudinary blog](https://cloudinary.com/blog/compressing_cellular_automata)

## SHOCK-AND-AWE candidate ranking

**#1 of 5** in parent memo §13. Worth approving the smoke probe.
