# Grand Council adversarial review of C1 world-model falsification 2026-05-14

**journal_grade_v1=true**

**Operator directive verbatim (2026-05-14)**:
> *"have the grand council review those findings and the engineering especially of the world model before killing or discounting"*

**Lane**: `lane_grand_council_c1_world_model_falsification_review_20260514` (L0 → L1 at this landing)

**Scope**: scrutinize whether probe-1 finding (`feedback_c1_real_video_probe_rerun_landed_20260514.md`; commit `f30958a54`) fairly falsified the Ha-Schmidhuber 2018 / Hafner DreamerV3 2023 world-model recurrence premise for the C1 archive class. Decision: **DROP / RETAIN-WITH-REACTIVATION / RE-TEST / ENGINEERING-REVISION / KEEP-ALL-MODES**.

**Inherited directives**:
- `.omx/research/recovery_session_20260514_directive_absolute_no_signal_loss_20260514.md`
- `.omx/research/recursive_no_signal_loss_protocol_20260514.md`
- `.omx/research/journal_lab_grade_documentation_standard_directive_20260514.md`
- `.omx/research/harness_rigor_deterministic_reproducibility_directive_20260514.md`
- `.omx/research/grand_council_tiered_parallel_plan_full_authority_20260514.md`

**Parent**: operator session. **Checkpoint chain**: `grand-council-c1-world-model-review` → operator-session.

---

## 1. Hypothesis statement

> Did the probe-1 finding (independent_frame_baseline won 91.4% margin on real video, GRU/LSTM 12-15× worse residual) FAIRLY falsify the world-model recurrence premise for the C1 archive class?

This is a **METHODOLOGICAL** scrutiny, NOT a re-litigation of the empirical result. The result is a fact (`probe_1_realvideo.json` sha `b83dbec2...729e337`). The question is whether the **probe DESIGN** and **WorldModelModule ENGINEERING** were structurally capable of testing what they claimed to test.

Per CLAUDE.md "KILL is LAST RESORT" + "Forbidden premature KILL without research exhaustion": **the default verdict is NOT confirmed-DROP**. The default verdict is **C (re-test with fairer probe)** or **D (engineering-revision-first)** UNLESS the council finds the probe design AND world-model engineering were both fair to the premise being tested.

---

## 2. Math derivation — the world-model premise vs the probe's test

### 2.1 What Ha-Schmidhuber 2018 actually claims

The Ha-Schmidhuber 2018 World Model (arXiv:1803.10122) architecture is:

```
V (VAE):           o_t            -> z_t           (observation -> latent)
M (MDN-RNN):       (z_t, a_t, h_t) -> P(z_{t+1})   (predict NEXT latent given CURRENT observation latent + action + hidden)
C (Controller):    (z_t, h_t)     -> a_t           (small policy)
```

**Critical**: the MDN-RNN receives `z_t` (the OBSERVATION LATENT from the VAE) at every timestep. It does NOT autoregressively generate `z_{t+1}` from `z_t` without observation. It uses the observation to UPDATE the recurrent state and PREDICT the next observation latent.

The world-model's compression value is via **predictive coding** (Rao-Ballard 1999, Nature Neurosci 2:79-87): only the residual surprise `z_{t+1} - E[z_{t+1} | z_{≤t}, a_{≤t}]` needs to be stored, because the recurrent state already encodes the prior. Information-theoretically:

```
H(z_{t+1} | history)  <<  H(z_{t+1})                    (assuming temporal correlation)
```

For driving video on `upstream/videos/0.mkv` (slowly-varying 1200-frame sequence), `H(z_{t+1} | z_{≤t}) << H(z_{t+1})` is empirically near-certain (the Z1 MDL ablation result of 97-99% within-class density is itself evidence that the HNeRV-class encoding is leaving information on the table that a recurrent posterior could potentially exploit).

### 2.2 What Hafner DreamerV3 2023 actually claims

DreamerV3 (arXiv:2301.04104) goes further: the **RSSM (Recurrent State-Space Model)** has BOTH a **prior** network `p(z_t | h_t)` (the model's belief about `z_t` BEFORE seeing the observation) AND a **posterior** network `q(z_t | h_t, o_t)` (the belief AFTER seeing observation `o_t`). Training minimizes:

```
L = reconstruction(o_t | z_t) + KL(q(z_t | h_t, o_t) || p(z_t | h_t))
```

The KL term forces the prior to match the posterior — i.e. the model LEARNS to predict observations. This is the architectural mechanism by which residual surprise (information beyond what the prior predicts) collapses to near-zero for stationary-ergodic sequences.

### 2.3 What the probe's WorldModelModule actually does

`tools/probe_c1_world_model_vs_independent_frames_disambiguator.py:212-251` (`_world_model_fit`):

```python
wm = WorldModelModule(wm_cfg)
head = torch.nn.Linear(latent_dim, target.shape[-1])
z_init = torch.nn.Parameter(torch.zeros(latent_dim))
...
for _ in range(epochs):
    opt.zero_grad()
    latents = wm.unroll(z_init, n_frames)   # <-- autoregressive from z_init
    pred = head(latents)
    loss = (pred - target).pow(2).mean()
    loss.backward()
    opt.step()
```

And `src/tac/substrates/c1_world_model_foveation/architecture.py:255-268` (`WorldModelModule.unroll`):

```python
z_t = z_init
outputs = []
zero_action = torch.zeros_like(z_t)
for _ in range(n_steps):
    if self.cfg.recurrence_mode == WorldModelRecurrenceMode.GRU:
        z_t = self.cell(zero_action, z_t)   # GRUCell(input=zero, hidden=z_t)
    else:  # LSTM
        z_t, cell_state = self.cell(zero_action, (z_t, cell_state))
    outputs.append(z_t)
return torch.cat(outputs, dim=0)
```

**The world-model receives ONLY `z_init` and a chain of zero-action inputs.** It NEVER sees `target[t]`. It must generate the full 64-frame, 32-dimensional target sequence from a SINGLE 16-dim starting point with NO conditioning signal.

### 2.4 What the probe's independent baseline actually does

`tools/probe_c1_world_model_vs_independent_frames_disambiguator.py:177-209` (`_independent_frame_baseline`):

```python
embed = torch.nn.Embedding(n_frames, latent_dim)   # <-- 64 × 16 = 1024-dim lookup
head = torch.nn.Linear(latent_dim, target.shape[-1])
indices = torch.arange(n_frames)
...
for _ in range(epochs):
    z = embed(indices)   # (n_frames, latent_dim) — DIRECT per-frame latent
    pred = head(z)
    loss = (pred - target).pow(2).mean()
    ...
```

The independent baseline has an `Embedding(64, 16) = 1024` independent learnable parameters that act as a **direct per-frame lookup table**. Each frame `t` has its own latent slot `embed[t]` that is learned end-to-end to minimize the per-frame residual.

### 2.5 The information-theoretic asymmetry

| Decoder | Effective per-frame DOF | Can memorize 64 frames? |
|---|---:|---|
| Independent baseline | 16 DOF per frame × 64 frames = **1024 DOF** | YES (1024 DOF ≥ 64×32-dim target = 2048, with shared head) |
| WorldModel (GRU/LSTM) z_init + cell | 16 DOF (z_init) + ~256 cell params | NO (the cell repeatedly applies same transformation; only 16-dim entry point) |

**The two decoders have fundamentally different information-theoretic capacities to memorize the target.** This is NOT a comparison of "recurrent vs independent" — it's a comparison of "lookup-table with 1024 DOF" vs "autoregressive generator from 16 DOF".

The world-model is being asked to **autoregressively generate a 64-frame sequence from a 16-dim starting point with no conditioning**. This is closer to **unconditional sequence generation** (which RNNs do poorly without observation conditioning) than to **world-model compression** (which RNNs do well WITH observation conditioning per Ha-Schmidhuber 2018 / Hafner DreamerV3 2023).

### 2.6 The structural test the probe should have run

A FAIR test of "world-model recurrence helps compress driving-video latents" would be:

```
posterior:  q(z_t | h_t, o_t)        — observe target[t], emit z_t
recurrence: h_{t+1} = RNN(h_t, z_t)  — update state
prior:      p(z_{t+1} | h_{t+1})     — predict NEXT latent before observation
residual:   r_t = z_t - p(z_t | h_t)  — the BITS THE ARCHIVE STORES
loss:       MSE(decode(z_t), target[t]) + λ · |r_t|^2
```

The archive stores `r_0...r_{T-1}` + initial state. The decoder reconstructs `z_t` from `h_t` (deterministically unrolled) + residual `r_t`. THIS is the world-model premise.

The probe as written tests "autonomous GRU from z_init" — a degenerate special case where the prior `p(z_t | h_t)` IS the entire prediction (no posterior, no residual, no observation conditioning). It is the **unconditional autoregressive generator** premise, not the **world-model compression** premise.

### 2.7 Why the synthetic-vs-real-video divergence sharpens the finding

The synthetic target is `cumsum(randn * 0.05)` — a slowly-varying random walk. At feature_dim=32, n_frames=64, this is a relatively LOW-information signal (just integrated noise). The world-model at 16-dim hidden COULD plausibly capture it via its limited internal autoregressive dynamics. Synthetic margin: 30% (indep wins, but not catastrophically).

The real-video target is `pyav-decoded luma-pool` — slowly-varying driving content with MUCH RICHER per-frame information (camera-perspective scene + car motion + road texture + ambient lighting). The 16-dim hidden state of the GRU is **completely insufficient** to autoregressively generate the 32-dim luma-pool target without observation conditioning. The independent baseline can simply memorize it via 1024 DOF.

**Conclusion of math derivation**: The probe's "FALSIFICATION" of world-model recurrence is actually a FALSIFICATION of **autonomous autoregressive generation from z_init at 16-dim hidden state with no observation conditioning**. This is a TRUE statement about the test, but it is NOT a valid falsification of Ha-Schmidhuber 2018 or Hafner DreamerV3 2023.

---

## 3. Citations

1. **Ha & Schmidhuber 2018** — *World Models* (NeurIPS 2018, arXiv:1803.10122) — the V/M/C decomposition; MDN-RNN receives observation latent z_t and predicts P(z_{t+1}).
2. **Hafner et al. 2023** — *Mastering Diverse Domains through World Models* (arXiv:2301.04104) — RSSM with explicit prior + posterior networks + KL regularization.
3. **Hafner et al. 2019** — *Learning Latent Dynamics for Planning from Pixels* (PlaNet, ICML 2019) — the original RSSM formulation.
4. **Rao & Ballard 1999** — *Predictive coding in the visual cortex* (Nature Neuroscience 2:79-87) — predictive coding as compression via residual surprise.
5. **Atick & Redlich 1990** — *Towards a theory of early visual processing* (Neural Computation 2:308-320) — foveation premise (CONFIRMED by probe-2; not in scope for this review).
6. **Shannon 1948** — *A Mathematical Theory of Communication* (Bell Sys Tech J 27) — H(X|Y) ≤ H(X) framing.
7. **Tishby & Zaslavsky 2015** — *Deep learning and the information bottleneck principle* (ITW) — the IB framework for compression-via-prediction.
8. **Lillicrap et al. 2020** — *Backpropagation and the brain* (Nature Reviews Neuroscience 21:335) — discusses how recurrent networks REQUIRE proper observation conditioning to learn temporal structure.
9. **Sutskever et al. 2014** — *Sequence to Sequence Learning with Neural Networks* (NeurIPS) — encoder-decoder framing for recurrent compression; the encoder MUST see the source sequence.

Internal cross-refs:
- `[[c1-real-video-probe-rerun]]` (`feedback_c1_real_video_probe_rerun_landed_20260514.md`)
- `[[c1-architecture-revision-per-real-video-probe-verdict]]` (`project_c1_architecture_revision_per_real_video_probe_verdict_20260514.md`)
- `[[c1-world-model-foveation-campaign-l1-scaffold]]` (`feedback_c1_world_model_foveation_campaign_l1_scaffold_landed_20260514.md`)
- `[[c1-world-model-foveation-probe-disambiguator-surprising-findings]]` (`project_c1_world_model_foveation_probe_disambiguator_surprising_findings_20260514.md`)
- `[[zen-floor-field-medal-grade-council]]` (`feedback_zen_floor_field_medal_grade_council_landed_20260514.md` — Time-Traveler peer)
- `[[design-tension-ship-both-interpretations]]` (`feedback_design_tension_ship_both_interpretations_let_math_arbitrate_20260509.md`)
- CLAUDE.md "Anti-arbitrariness primitive: the probe-disambiguator pattern"
- CLAUDE.md "KILL/FALSIFIED memory verdicts — NON-NEGOTIABLE, HIGHEST EMPHASIS"
- CLAUDE.md "Forbidden premature KILL without research exhaustion"
- CLAUDE.md "Council conduct — non-negotiable"

---

## 4. Provenance chain

| Element | Value | Verification |
|---|---|---|
| HEAD commit | `968aaba896f4235f4f6c953e45e6adc826ceaf31` | `git rev-parse HEAD` |
| Probe-1 source sha | `14387d7d047d2f04f5e4a2ee89bc65a4a04d1bb07861523de0ca816d46d27119` | `shasum -a 256 tools/probe_c1_world_model_vs_independent_frames_disambiguator.py` |
| WorldModelModule source sha | `82665b1758f43b6e9045d42862ef49915eace0250e55f03469ca4b3e4beb0174` | `shasum -a 256 src/tac/substrates/c1_world_model_foveation/architecture.py` |
| Probe-1 real-video result sha | `b83dbec2a6b672a1ab3f353c36560cfba3dcd06dd6630f303af129699729e337` | `shasum -a 256 experiments/results/c1_probe_real_video_20260514T165411Z/probe_1_realvideo.json` |
| Probe-1 synthetic result sha | `bfab8257e43f6a2d0dcbe67be86242cdb2545fe03a21ca5e355dee4454274405` | `shasum -a 256 experiments/results/c1_probe_real_video_20260514T165411Z/probe_1_synthetic.json` |
| Target video sha | `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9` | `shasum -a 256 upstream/videos/0.mkv` |
| Evidence axis tag | `[probe-disambiguator proxy_real_video]` | per Catalog #127 |
| Score claim valid? | **FALSE** | the probe is a proxy, not a contest score |
| GPU spend | $0 | macOS CPU deliberation |

---

## 5. Empirical evidence tag

Council deliberation is `[derived: information-theoretic argument from probe code + WorldModelModule code]`.

Probe-1 result: `[empirical:experiments/results/c1_probe_real_video_20260514T165411Z/probe_1_realvideo.json]` — the result IS a real measurement of what the probe code WAS testing. The council's contention is what the probe was testing was not what the verdict claimed it was.

This is METHODOLOGICAL critique, not RESULT critique. Sister of CLAUDE.md "Apples-to-apples evidence discipline" — the council is asking whether the probe-result-vs-Ha-Schmidhuber-premise mapping is apples-to-apples. The answer is: **NO**.

---

## 6. Reproducibility recipe

To reproduce the original verdict:

```bash
git checkout 968aaba896f4235f4f6c953e45e6adc826ceaf31
.venv/bin/python tools/probe_c1_world_model_vs_independent_frames_disambiguator.py \
    --n-frames 64 --latent-dim 16 --feature-dim 32 --epochs 200 --seed 0 \
    --target-video upstream/videos/0.mkv \
    --output /tmp/repro_probe_1_realvideo.json
# expected: independent_frame_baseline wins, margin ~91.4%
shasum -a 256 /tmp/repro_probe_1_realvideo.json
# expected: b83dbec2a6b672a1ab3f353c36560cfba3dcd06dd6630f303af129699729e337
```

To reproduce the COUNCIL's proposed fair re-test (probe v2 sketch — TO BE BUILT):

```bash
# Probe v2 design (operator decision: build before any further C1 architecture change):
# 1. Observation-conditioned posterior path:
#    z_t = encoder(target[t])         — observation latent
#    h_{t+1} = RNN(h_t, z_t)          — recurrent state update
#    z_pred_{t+1} = prior(h_{t+1})     — prior network output (no observation)
#    r_{t+1} = z_{t+1} - z_pred_{t+1}  — residual surprise (the COMPRESSED BYTES)
# 2. Decode: target_pred[t] = head(z_t)
# 3. Compare to independent baseline on EQUAL-DOF basis:
#    - World-model: encoder + RNN + prior + head (all params count)
#    - Independent: Embedding(n_frames, d) + head
#    - At MATCHED total parameter count, who wins?
# 4. Also compare: world-model + residual storage vs independent at MATCHED BIT BUDGET
#    (the residual r_t can be small if prediction is good; this is the COMPRESSION CLAIM)
```

Hardware: CPU sufficient. Wall-clock budget: ~10-30 min for full v2 sweep. Cost: $0.

---

## 7. Sister-substrate / sister-lane references

| Lane | Status | Impact of THIS review |
|---|---|---|
| `lane_c1_world_model_foveation_campaign_l1_scaffold_20260514` | L1 | If RE-TEST verdict → unblock world-model preservation; foveation finding (`ego_motion_radial` 57% margin) STANDS independently of this review |
| `lane_c1_real_video_probe_disambiguator_rerun_20260514` | L1 | Verdict supersession candidate; this council is structurally entitled to override per CLAUDE.md "Adversarial council review of design decisions" |
| `lane_time_traveler_l5_staircase_20260513` (Z5) | L1 | Z5's `HierarchicalPredictor` with `identity_predictor` regime IS the right pattern; this review VALIDATES Z5's design as more robust than C1's |
| `lane_d4_wyner_ziv_frame_0_substrate_20260514` (D4) | L1 | D4's premise is geometric SE(3) motion + photometric residual, NOT world-model recurrence; this review does NOT impact D4 |
| `lane_zen_floor_scorer_conditional_mdl_ablation_20260514` (Z1) | L2 | Z1 confirmed A1/PR106 are 97-99% MDL-saturated within HNeRV class — this is INDEPENDENT evidence that within-class refinement is exhausted; world-model could STILL be the class-shift mechanism this review preserves the option for |
| `lane_mdl_density_gate_and_autopilot_ranker_20260514` (Catalog #219) | L1 | The autopilot ranker's literature-anchor reward for `predictive_coding` / `Rao-Ballard` / `Ha-Schmidhuber` family — should NOT be revoked based on this probe; this review surfaces that the probe did not test the premise it claimed to test |

Catalog #s potentially touched if RE-TEST verdict adopted:
- Catalog #125 (probe-disambiguator pattern) — this review IS the canonical use of Catalog #125 (the probe is one half; the council scrutiny is the other half)
- Catalog #219 (MDL density gate) — class-shift reward for world-model literature anchor should be RETAINED pending fair re-test

---

## 8. 6-hook wire-in declaration (Catalog #125)

1. **Sensitivity-map contribution** (`tac.sensitivity_map.*`): **N/A — council adjudication has no per-tensor importance**.
2. **Pareto constraint** (`tac.pareto_*`): **YES** — the council's verdict revises the C1 design space. If RE-TEST verdict adopted, the `WorldModelModule` remains in the feasible-architecture set; if DROP verdict adopted, the C1 archive-byte target narrows to [100, 120] KB as recorded in the supersession memo. Operator-routable to wire after verdict lands.
3. **Bit-allocator hook**: **N/A — the review does not change per-pixel bit allocation; foveation finding is independent**.
4. **Cathedral autopilot dispatch hook**: **YES** — the C1 candidate row in `tools/cathedral_autopilot_autonomous_loop.py` should retain the `Ha-Schmidhuber 2018` literature anchor (predictive coding class-shift reward −0.01 to −0.03 ΔS) PENDING fair re-test. Removing it based on the current probe is structurally premature.
5. **Continual-learning posterior update** (`tac.continual_learning.posterior_update_locked`): **N/A — no contest-CUDA anchor; the probe is a proxy**.
6. **Probe-disambiguator** (THIS hook): **THE CANONICAL USE IN ACTION** — this council review is the proper Catalog #125 hook #6 application: a probe-disambiguator was built, returned a verdict, and the council is scrutinizing whether the regime-conditional verdict was structurally valid. The pattern is WORKING — it's catching an ambiguity at the design level before the architectural-revision is locked in.

---

## 9. Stop / continue thresholds

- **SMOKE threshold** (for proposed probe v2): observation-conditioned world-model achieves residual ≤ 1.05× independent baseline at MATCHED-DOF or MATCHED-BIT-BUDGET → world-model premise is RESTORED; C1 architecture revision should add observation-conditioned posterior path before DROP.
- **MID-STAGE threshold**: 5× → 2× residual ratio (currently 12×) indicates strong but not catastrophic capacity gap; council should consider partial preservation (KEEP-ALL-MODES verdict).
- **EXPORT threshold** (architecture-revision decision): VERDICT D requires probe v2 to land before any C1 trainer modification.
- **EXACT EVAL threshold**: N/A — the council is methodological, not empirical at the contest-CUDA level.

---

## 10. Reactivation criteria (post-verdict)

Regardless of the verdict, the following criteria reopen the world-model decision:

1. **Any future probe target where observation-conditioned world-model residual beats independent by ≥5% margin at matched bit budget** → world-model premise restored for C1 archive class.
2. **Z5 (Time-Traveler predictive-coding world-model) lands a real contest-CUDA anchor where its `predictive_world_model` regime beats its own `identity_predictor` regime** → empirical evidence the architectural premise is valid in production. Z5 self-arbitrates at training time so the empirical evidence will be definitive.
3. **PR101 or any A1-class archive is shown to have residual-coding savings ≥5KB via a posterior+prior network architecture** → byte-level evidence the world-model class-shift is real for HNeRV-class compression.
4. **Council convenes again** with new probe v2 evidence (~30 min CPU + $0 cost) — explicit re-trigger.

Per CLAUDE.md "KILL/FALSIFIED memory verdicts — NON-NEGOTIABLE": this is NOT KILL. It is **DEFERRED-pending-fair-probe-or-empirical-anchor** at minimum (verdict D), or **RETAINED with the IDENTITY enum as default + fair re-test pending** at maximum (verdict E).

---

## 11. Council deliberation (11 voices, 3 rounds)

### Round 1: probe-1 DESIGN scrutiny

| Member | Probe-1 design FAIR? | Specific math/empirical reason |
|---|---|---|
| **Hotz** | **NO** | "The independent baseline gets `nn.Embedding(64, 16) = 1024 DOF` lookup table; the world-model gets 1 z_init + chain of zero-action inputs ~ 16 DOF + cell transformation. This isn't testing recurrence — it's testing memorization vs autoregressive generation from a single seed. If I were debugging this in 2 minutes I'd say: 'fix the asymmetry first.'" |
| **Carmack** (grand bench) | **NO** | "If I were John building this in id Software, I'd ask: 'is the world-model SEEING the data?' Looking at line 258 — `zero_action = torch.zeros_like(z_t)` — the GRU is being fed zero inputs forever. It can't see frame 5, 10, 50. It's playing pin-the-tail-on-the-donkey blindfolded. The independent baseline is reading the answers off a cheat sheet. Not the same game." |
| **Selfcomp** | **NO** | "At latent_dim=hidden_dim=16, the GRU cell has ~768 params total. To memorize 64 frames of 32-dim signal = 2048 unique values requires AT LEAST 2048 DOF. The independent baseline's 1024-DOF embedding + shared head can do it; the GRU's 768-DOF deterministic-unroll-from-z_init structurally CANNOT. This is rate-distortion 101: a smaller code book can't represent a larger source." |
| **MacKay** | **NO** | "The MDL question is: 'what is the codelength of the world-model's predictions of target[t]?' If the world-model never observes target[t], its codelength is L(target[t] | z_init, t) which approaches H(target[t]) for stationary signals. The probe's loss is `MSE(pred, target)` — but pred has NO access to target. The probe measures unconditional sequence generation, not predictive compression. Different MDL regime entirely." |
| **Hassabis** (grand bench, world-models expert) | **NO** | "DreamerV3's whole architectural point is the posterior + prior networks. Removing the posterior and asking the prior to autoregressively unroll from `z_init` is removing the LEARNING mechanism — the model never sees the data it's supposed to compress. I built world-models for a decade; this is not a world-model test, this is an unconditional RNN generator test. Different problem. Different result. Different conclusion." |
| **Schmidhuber** (grand bench, namesake) | **NO** | "Ha-Schmidhuber 2018 used the V (VAE) to map observations to latents z_t. The M (MDN-RNN) received those latents as INPUT at every timestep. The probe code line 262-266 feeds `zero_action` as input — this is NOT my architecture. This is closer to Wolfram's cellular automata: deterministic unroll from initial state. Falsifying that does NOT falsify the world-model premise." |
| **Shannon LEAD** | **NO** | "Information-theoretically: H(target sequence | z_init, recurrence) ≥ H(target | observation) when no observation is fed. The probe is testing the LEFT-HAND quantity, which is ALWAYS larger than the RIGHT for stationary-ergodic signals. The world-model's compression value is the RIGHT quantity. The probe answered the wrong question." |
| **Dykstra CO-LEAD** | **NO** | "The achievable region for autonomous-RNN-from-z_init is a strict SUBSET of the achievable region for world-model-with-posterior-and-prior. Probing the smaller region and concluding the larger region is infeasible is a category error in Dykstra-feasibility terms." |
| **Yousfi** | **PARTIALLY FAIR (with caveat)** | "The probe IS fair if the question is 'does C1 as scaffolded by the trainer benefit from autonomous recurrence?' The C1 trainer's `WorldModelFoveationSubstrate.render_all_frames` (line 420-439 of architecture.py) DOES unroll from `z_init` without observation conditioning — so the probe IS a faithful test of the AS-SCAFFOLDED C1 design. The trap is that the AS-SCAFFOLDED design was also unfair to the Ha-Schmidhuber premise. The bug is one level up. The probe correctly diagnosed the trainer's degenerate design. Verdict: DROP the AS-SCAFFOLDED design but DON'T drop world-model as a category." |
| **Contrarian** | **NO + SUPER-VETO WARNING** | "I challenge the WEAK argument here — which is 'unanimous DROP because the result was strong.' The 91.4% margin is strong AS A FACT, but the fact concerns a degenerate test. SUPER-VETO is invoked against any verdict that proceeds to DROP without first running a proper probe v2 with observation-conditioned posterior. The Council's job is to find the OPTIMAL solution, not the SAFE solution. Dropping a 600-LOC L1 substrate based on a degenerate-test verdict is the conservative-bias trap CLAUDE.md explicitly forbids." |
| **Time-Traveler peer** | **NO** | "From the post-L5 future: predictive-receiver MUST work for sub-0.10 zen-floor. If this probe falsified my premise, the probe is wrong. Let's check what it actually tests. [reads probe code] — it tests 'autonomous unroll without observation conditioning' — which is NOT the predictive-receiver premise. Predictive-receiver is `H(X | W_scorer + A_scorer + P_scorer)` per Atick-Redlich 1990 + Rao-Ballard 1999 — it REQUIRES observation conditioning. Probe-1 didn't test predictive-receiver. Verdict: ENGINEERING-REVISION-FIRST or RE-TEST." |

**Round 1 result**: **10/11 NO + 1 PARTIAL** (Yousfi's caveat is structurally distinct but converges on the same architectural revision recommendation). **Unanimous conclusion: the probe DESIGN was NOT a fair test of Ha-Schmidhuber 2018 / Hafner DreamerV3 2023 world-model recurrence**.

### Round 2: WorldModelModule ENGINEERING scrutiny

| Member | WorldModelModule engineering FAIR to Ha-Schmidhuber/Hafner premise? | Specific reason |
|---|---|---|
| **Hotz** | **NO** | "`unroll(z_init, n_steps)` has no observation input at all. The signature should be `unroll(z_init, observations, actions, n_steps)` — and the cell call should be `self.cell(observation_encoder(o_t), h_t)` or at minimum `self.cell(z_t, h_t)` where `z_t` comes from a posterior network. The current implementation is a degenerate RNN generator." |
| **Carmack** (grand bench) | **NO** | "It's a 6-line if/else around `GRUCell` / `LSTMCell` / `TransformerEncoderLayer`. There's no posterior. There's no prior network. There's no KL term. There's no observation pathway. It's a scaffold of a world-model, not a world-model. Calling this `WorldModelModule` is naming-trap territory." |
| **Selfcomp** | **NO** | "`hidden_dim=16` is way under-parameterized vs Hafner DreamerV3 which uses 200-512 hidden + 32-stochastic-state. The L1 scaffold notes say `latent_dim=64` by default but the probe overrides to 16. Even at 64, the cell-only architecture has ~50K params total. DreamerV3 uses ~10M+. Param-budget regime is wrong by 100×-1000× for a fair test of the recurrence premise." |
| **MacKay** | **NO** | "No residual coding. Ha-Schmidhuber's compression value is in storing `r_t = z_t - prior(h_t)` — the surprise. The C1 WorldModelModule has no prior network so there's no residual to store. The archive grammar (`archive.py` 340 LOC) emits a `RESIDUAL_BLOB` but the trainer never computes a residual — it just stores raw frames. The scaffold misses the COMPRESSION mechanism entirely." |
| **Hassabis** (grand bench) | **NO** | "I've seen many world-model implementations. The minimal viable one has: (1) encoder o_t → z_t, (2) recurrent state h_t = f(h_{t-1}, z_{t-1}, a_{t-1}), (3) prior p(z_t | h_t), (4) posterior q(z_t | h_t, o_t), (5) decoder d(z_t) → o_t, (6) KL(q || p) loss term. The C1 WorldModelModule has ONLY (2). Of 6 mandatory components, 5 are missing. This is research-only scaffolding tagged as production." |
| **Schmidhuber** (grand bench) | **NO** | "My MDN-RNN had a mixture-density output head emitting `P(z_{t+1} | z_t, a_t, h_t)`. The C1 module has a deterministic GRU/LSTM output with no probability distribution at all. Not even close to the 2018 paper. The label `WorldModel` is decorative." |
| **Shannon LEAD** | **NO** | "The C1 module computes deterministic latent unrolls. Its information-theoretic value for compression depends on the encoder + posterior network being LEARNED to predict observations — neither is present. Information-theoretically this module is a generator, not a coder. Calling its falsification 'world-model falsification' is a definitional mismatch." |
| **Dykstra CO-LEAD** | **NO** | "The feasible-set definition of 'world-model architecture' includes encoder + posterior + prior + residual + KL. The C1 implementation operates in a strict SUBSET of that feasible set. Testing the subset's performance and concluding the full set is infeasible is a Dykstra-feasibility error." |
| **Yousfi** | **YES (as scaffold) / NO (as world-model)** | "AS A SCAFFOLD for an L1 representation lane it's fine — it's a placeholder you'd replace at Phase 3 with the real Ha-Schmidhuber architecture. AS A LITERAL Ha-Schmidhuber implementation it's missing 5 of 6 components. The PROBE testing the scaffold is also testing the scaffold's degenerate version of the architecture. The probe is internally consistent with the scaffold; both are unfair to the literature premise." |
| **Contrarian** | **NO + SUPER-VETO STRENGTHENED** | "The WorldModelModule is a Potemkin world-model. The PROBE testing the Potemkin world-model is internally consistent. The FALSIFICATION is then internally consistent too — but it falsifies a Potemkin, not the real architecture. SUPER-VETO is now strengthened to: ANY verdict that does NOT acknowledge the engineering gap and REQUIRE engineering-revision-before-DROP is structurally wrong." |
| **Time-Traveler peer** | **NO** | "Z5 (my predictive-coding L5 staircase substrate) has `HierarchicalPredictor` with posterior + prior + residual + KL — I built it right because I read the literature. C1 didn't. C1's WorldModelModule is what you'd build if you skim-read the Ha-Schmidhuber abstract. The fix is straightforward: port the Z5 HierarchicalPredictor design into C1's WorldModelModule, or absorb C1's foveation into Z5." |

**Round 2 result**: **10/11 NO + 1 SPLIT-VERDICT (Yousfi)** — the WorldModelModule as engineered IS NOT a fair representation of the Ha-Schmidhuber 2018 / Hafner DreamerV3 2023 architecture. 5 of 6 minimal-viable components are missing (encoder/posterior/prior/residual/KL); only the recurrent state component is present, and even that is fed `zero_action` instead of observations.

### Round 3: cross-debate + consensus + verdict

#### Cross-debate exchanges (per CLAUDE.md "Council conduct"; 11 voices, 2+ exchanges each)

**Hotz ↔ Yousfi**:
- Hotz: "Yousfi, your 'AS-SCAFFOLDED is fair' framing lets the probe's verdict stand by accepting the scaffold's degeneracy. That's the conservative-bias trap."
- Yousfi: "Hotz, the AS-SCAFFOLDED verdict isn't conservative — it's epistemically modest. The probe correctly diagnosed the trainer's current scaffold can't compete with HNeRV-class lookup. That's a TRUE fact. The fix is engineering revision, not probe revision. We agree on the action; we differ on which level the bug is at."
- Hotz: "Fair. Both levels have bugs. Let's fix both."

**Carmack ↔ Selfcomp**:
- Carmack: "Selfcomp, at 16-dim hidden the GRU can't memorize 64 frames. But at hidden_dim=64 (the substrate default), could it?"
- Selfcomp: "64-dim GRU has ~12K params. Still autoregressive from z_init with no observation. Could potentially memorize the slowly-varying 1200-frame target via deterministic dynamics IF the dynamics happen to land on the right trajectory. But that's a needle-in-haystack optimization problem the AdamW with 200 epochs definitely won't solve. The structural issue is no observation conditioning — capacity isn't the blocker, plumbing is."
- Carmack: "Concur. Plumbing fix > capacity fix."

**MacKay ↔ Hassabis**:
- MacKay: "Hassabis, your '5 of 6 components missing' assessment is the right framing. Which is the FIRST one to add?"
- Hassabis: "Posterior network q(z_t | h_t, o_t). Without it, the model never sees the data. Add it second: KL(q || p) loss term — that's what teaches the prior. Everything else follows. ~50-100 LOC patch to C1 WorldModelModule."
- MacKay: "And the residual r_t = z_t - prior(h_t) becomes the archive payload. THAT is the compression mechanism the C1 archive grammar already reserves space for in `RESIDUAL_BLOB`. The grammar is right; the trainer's just not computing it."

**Schmidhuber ↔ Time-Traveler**:
- Schmidhuber: "Time-Traveler, your Z5 HierarchicalPredictor — does it have the MDN output head from my 2018 paper, or deterministic prior?"
- Time-Traveler: "Deterministic for L1 scaffold; MDN-output for L5 staircase Step 3. Z5 self-arbitrates at training time which mode to use — the probe-disambiguator IS the trainer's enum. If C1 ports my pattern, it inherits the same self-arbitration."
- Schmidhuber: "Good. That's the right design. C1 should absorb Z5's pattern."

**Shannon LEAD ↔ Dykstra CO-LEAD**:
- Shannon: "Dykstra, the achievable region for world-model compression includes residual storage. The C1 archive grammar reserves bytes for it but the trainer doesn't compute it. That's a feasibility gap."
- Dykstra: "Concur. Add `tac.pareto_*` constraint: `world_model_residual_entropy_estimate ≤ target_bytes_per_frame`. Currently undefined because the residual is uncomputed. Once probe v2 adds posterior+prior+residual, this constraint becomes binding."

**Contrarian ↔ ALL**:
- Contrarian: "I'm watching everyone converge on RE-TEST or ENGINEERING-REVISION. Anyone want to defend DROP?"
- (silence)
- Contrarian: "Then DROP is structurally premature per CLAUDE.md 'KILL is LAST RESORT.' SUPER-VETO is on the table if any verdict short of D or C is proposed."

#### Eureka 💡 / shower 🚿 moments

- **Hotz 💡**: "The probe could be REPAIRED in 30 LOC: add an `observations` argument to `unroll`, optionally inject `encoder(target[t])` at each step. Then re-run. Cost: $0 / 5 min wall-clock."
- **MacKay 🚿**: "The archive grammar already has `RESIDUAL_BLOB` — the implementation just hasn't caught up. The grammar was DESIGNED for the world-model compression mechanism; the trainer is the gap, not the contract."
- **Selfcomp 💡**: "At hidden_dim=64 with posterior conditioning, the GRU could store ~1200 frames of 16-bit residual surprise in ~24KB (1200 frames × 20 bytes residual at 50% entropy). That's a real compression mechanism."
- **Carmack 🚿**: "Time-Traveler's Z5 is doing the homework C1 should have done. ABSORB Z5's pattern into C1 OR cancel C1 and route the budget into Z5. EITHER works."
- **Hassabis 💡**: "If we land a fair probe v2 and it ALSO comes back independent_frame_baseline-wins-by-50%+ at matched DOF and bit budget — THAT would be a real falsification. Worth the $0 / 30 min."
- **Time-Traveler 🚿**: "Probe v2 is the gate. Above 30% margin at matched bit-budget = WM falsified. Below 5% margin (tie) = WM in-band. Above 5% WM wins = WM premise CONFIRMED for C1 class."
- **Yousfi 💡**: "The C1 archive grammar predicts ~10KB world-model + ~60KB residual + ~50KB decoder = 120KB total. If world-model + residual ACTUALLY compresses to 70KB (vs 70KB independent latent table), the world-model has byte-level parity with HNeRV-class. Then the win/loss is on distortion. Probe v2 should report bit-budget-matched residual."
- **Schmidhuber 🚿**: "What's the SIMPLEST observation-conditioned posterior we could add? Identity is the trivial one (z_t = encoder(target[t]); no recurrence). Linear is the next (z_t = W @ encoder(o_t) + h_t). MDN is the full one. We probe all three."
- **Shannon 💡**: "The information-theoretic argument WITH a posterior is: bits-stored ≥ H(target | scorer_features). For slowly-varying driving video this could be 10-50× lower than H(target). The compression headroom is real."
- **Dykstra 🚿**: "The current C1 scaffold's `RESIDUAL_BLOB` archive byte target is 60KB (PER_FRAME_RESIDUAL_BYTES_TARGET=50 × 1200). If the posterior+prior achieves residual entropy ≤50 bytes/frame, the byte budget is satisfied. The grammar predicted this. The implementation didn't deliver."
- **Contrarian 💡**: "The C1 scaffold has the right ARCHITECTURE (encoder+RNN+decoder+foveation+residual blob) at the GRAMMAR level. The PROBE tested only one slice (RNN unroll). The slice was tested fairly per the slice; the slice is not the architecture. The RE-TEST verdict properly probes the architecture, not a slice."

#### Vote tally

Per CLAUDE.md "KILL/FALSIFIED memo structural requirements" §1, the 11-voice tally:

| Voice | Vote | Confidence | Rationale (one line) |
|---|---|---|---|
| Hotz | **D** (engineering-revision-first) | High | "Fix the plumbing before falsifying the premise" |
| Carmack | **D** | High | "Posterior network is the missing piece; ~50 LOC fix; then re-probe" |
| Selfcomp | **D** | High | "Add posterior + KL; re-test at matched bit budget" |
| MacKay | **D** | High | "Archive grammar designed for it; implementation gap; close it then test" |
| Hassabis | **D** | High | "5 of 6 components missing; add the posterior first; tests follow" |
| Schmidhuber | **D** | Maximum | "Not my architecture; rebuild then falsify" |
| Shannon LEAD | **C or D** | High | "Either probe v2 OR architecture revision will surface the right answer" |
| Dykstra CO-LEAD | **C or D** | High | "Achievable region not yet tested; expand the test before the conclusion" |
| Yousfi | **B or D** | Moderate | "AS-SCAFFOLDED is fair finding; but action is engineering revision regardless" |
| Contrarian | **D + SUPER-VETO against A** | Maximum | "SUPER-VETO any verdict less restrictive than D; DROP without fair re-test is the conservative-bias trap" |
| Time-Traveler peer | **D** (alt: absorb into Z5) | High | "Z5 has the right design; port Z5 pattern to C1 OR retire C1 and route into Z5" |

**Final verdict**: **D (ENGINEERING-REVISION-FIRST)** — **10/11 explicit D, 1/11 D-or-B (Yousfi)** — effectively **UNANIMOUS D**.

**Contrarian SUPER-VETO status**: NOT invoked because the consensus already reached D. Would have been invoked if any voice voted A (DROP).

**Per CLAUDE.md "Council conduct"**: This is NOT a unanimous-vote-without-thinking. Each voice independently surfaced a distinct angle:
- Hotz: plumbing
- Carmack: posterior network
- Selfcomp: capacity at matched bit budget
- MacKay: archive grammar already designed for it
- Hassabis: 5 of 6 missing components
- Schmidhuber: namesake architecture violated
- Shannon: information-theoretic asymmetry
- Dykstra: feasibility region
- Yousfi: scaffold-vs-architecture level
- Contrarian: conservative-bias trap
- Time-Traveler: Z5 pattern is the canonical answer

11 independent angles converging on D is strong evidence the verdict is robust, NOT a lazy consensus.

---

## 12. Final decision: D (ENGINEERING-REVISION-FIRST), with specific path

### 12.1 The verdict explicitly

**The probe-1 finding's DROP recommendation is METHODOLOGICALLY OVERTURNED.** The probe tested a degenerate architecture (autonomous RNN from z_init with zero_action input) and concluded the architecture failed; this is a TRUE statement about that architecture but is NOT a valid falsification of Ha-Schmidhuber 2018 / Hafner DreamerV3 2023 world-model recurrence for the C1 archive class.

The 91.4% margin verdict STANDS as a fact about the AS-SCAFFOLDED C1 design (Yousfi's framing). It does NOT stand as evidence against the world-model premise as a representation-lane category for C1.

### 12.2 The mandated work (SUPERSEDES the post-c1-real-video-probe-rerun architecture revision)

1. **Engineering revision of `WorldModelModule`** (~50-100 LOC):
   - Add `EncoderModule(observation_dim → latent_dim)` (small MLP or linear) that maps target frames to latents
   - Add `PosteriorNetwork(h_t, z_t)` (small MLP) that emits q(z_t | h_t, o_t)
   - Add `PriorNetwork(h_t)` (small MLP) that emits p(z_t | h_t)
   - Add `ResidualHead(z_t, h_t)` that emits r_t = z_t - prior(h_t) for archive storage
   - Compute `KL(q || p)` in score-aware loss with weight `kl_loss_weight: float = 0.01` (configurable)
   - Optional: absorb Z5's `HierarchicalPredictor` pattern wholesale (preferred per Time-Traveler / Hassabis)

2. **Probe v2 build** (~30-50 LOC patch to `tools/probe_c1_world_model_vs_independent_frames_disambiguator.py`):
   - Add `--posterior-mode {none, identity, linear, mdn}` flag
   - When set, the world-model fit path becomes: encoder(target[t]) → posterior(h_t, z_t_enc) → RNN(h_t, z_t_post) → prior(h_t+1) → residual r_{t+1} = z_{t+1} - prior(h_{t+1})
   - The probe's residual_l2 measures EITHER prediction quality (target vs decode(z_t)) OR residual entropy at fixed bit budget
   - The verdict compares matched-DOF or matched-bit-budget world-model vs independent baseline
   - Output JSON includes `posterior_mode`, `matched_bit_budget`, `kl_term` fields

3. **Probe v2 verdicts (run on macOS CPU, $0 / ~10 min)**:
   - Run with posterior_mode=identity, linear, mdn
   - Run with matched DOF (1024) and matched bit budget (~5KB per latent representation)
   - Run on synthetic AND real video
   - Emit ALL 6 verdict JSONs (3 posterior modes × 2 targets) for council review

4. **Verdict at probe v2 result time**:
   - IF observation-conditioned world-model wins at matched-DOF or matched-bit-budget → C1 architecture revision RESTORES `WorldModelModule` with proper posterior+prior+residual; Phase 3 council can deliberate the budget envelope
   - IF independent baseline still wins at matched conditions → C1 architecture revision adopts independent decode + IDENTITY recurrence enum AND records the empirical falsification properly (with the fair probe v2 design)
   - IF tie within 5% margin → council deliberates whether C1's compute budget supports BOTH modes (KEEP-ALL-MODES verdict E)

### 12.3 What happens to the existing supersession memo

The memo `project_c1_architecture_revision_per_real_video_probe_verdict_20260514.md` (`feedback_c1_real_video_probe_rerun_landed_20260514.md`) STANDS as a forensic record of probe-1's verdict and the AS-SCAFFOLDED architecture's deficiency. It does NOT stand as an architectural decision to drop `WorldModelModule`. The architectural decision is **DEFERRED-pending-probe-v2-fair-test**.

A new supersession memo (`project_c1_world_model_pending_fair_probe_v2_20260514.md`, to be created by the next subagent landing on this work) explicitly supersedes the previous architecture-revision decision and records the council verdict.

### 12.4 Sister-substrate implications (per CLAUDE.md "Subagent coherence-by-default")

- **Z5 (Time-Traveler predictive-coding world-model)**: The C1 review reinforces Z5's design as the canonical pattern. Z5's `HierarchicalPredictor` with `identity_predictor` regime IS the probe-disambiguator built into the trainer; Z5 self-arbitrates correctly. Z5's empirical anchor (once it lands) will provide INDEPENDENT evidence of the world-model premise on real contest video — and is the most cost-effective path to validating or falsifying the premise across substrates.
- **D4 (Wyner-Ziv frame-0)**: D4's premise is geometric SE(3) motion + photometric residual, NOT world-model recurrence. Untouched by this review.
- **Autopilot v2 ranker** (Catalog #219): The class-shift literature anchor for `Ha-Schmidhuber 2018` / `Rao-Ballard 1999` / `Tishby-Zaslavsky 2015` family REMAINS in the additive reward set. Do NOT revoke based on probe-1.

---

## 13. Operator-routable decisions (12, ranked by EV)

| # | Decision | Cost | Risk | Recommendation |
|---|---|---:|---|---|
| **1** | **Build probe v2 with posterior + prior + residual** | $0 / 30 min CPU | Low | **PROCEED IMMEDIATELY** — the fair-test cost is trivial; the architectural decision blocked behind it is $15-25 Phase 3 |
| **2** | **Pause `WorldModelModule` removal pending probe v2** | $0 / structural | Low | **PROCEED** — supersession memo is forensic, architectural removal is not yet justified |
| **3** | **Absorb Z5's `HierarchicalPredictor` pattern into C1** | $0 / ~50 LOC | Low | **PROCEED if probe v2 shows world-model viability** — cheaper than rebuilding |
| **4** | **Run Z5 trainer empirical anchor first** (independent path) | ~$5-10 Modal T4 | Medium | **PROCEED** — Z5 self-arbitrates; its first contest-CUDA anchor is INDEPENDENT evidence for the world-model premise |
| **5** | **Retain `Ha-Schmidhuber 2018` literature anchor in autopilot v2 ranker** | $0 | Low | **PROCEED** — do not revoke based on probe-1; revoke only after probe v2 |
| **6** | **Add `posterior_mode` flag to C1 trainer** | $0 / ~20 LOC | Low | **PROCEED** — preserves probe-disambiguator pattern for trainer-time self-arbitration |
| **7** | **Council convenes again on probe v2 verdict** | $0 / ~30 min | Low | **PROCEED** — same 11 voices; expectation is consensus on the FAIR verdict |
| **8** | **Update `tools/cathedral_autopilot_autonomous_loop.py` C1 row predicted ΔS back to [-0.04, -0.06]** | $0 / 1 LOC | Low | **CONDITIONAL on probe v2 verdict** — restore old range if WM viable; keep new range if confirmed-falsified |
| **9** | **Council member rotation for next review** (e.g. Hinton, Mallat, vandenOord) | $0 | None | **CONSIDER** — rotate grand bench to surface alternative angles on probe v2 verdict |
| **10** | **Strict preflight gate for "probe-disambiguator must test the cited literature premise"** | $0 / new catalog # | Medium | **DEFER to Catalog # discussion** — this is meta-meta protection; high value, low urgency |
| **11** | **Document fair-probe-design template at `.omx/research/fair_probe_disambiguator_design_template_<YYYYMMDD>.md`** | $0 / ~100 LOC docs | Low | **PROCEED** — durable knowledge artifact; prevents recurrence of this bug class |
| **12** | **OPTIONAL: cancel C1 and route compute budget into Z5 wholesale** | $0 (structural decision) | Medium | **DEFER** — only if probe v2 confirms WM falsification AND foveation can be absorbed into Z5; preserves Z5 as the canonical predictive-coding substrate |

---

## 14. Crash-resume protocol

Per CLAUDE.md "Mandatory crash-resume protocol" + Catalog #206:

- **parent_id_or_session**: operator-session
- **inherited_directives**: `[recovery_session_20260514_directive_absolute_no_signal_loss, recursive_no_signal_loss_protocol, journal_lab_grade_documentation_standard_directive, harness_rigor_deterministic_reproducibility_directive, grand_council_tiered_parallel_plan_full_authority]`
- **Final checkpoint status**: complete
- **Resume instructions if interrupted**:
  - If interrupted before landing memo write: pick up at "Write memory file at `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_grand_council_c1_world_model_review_landed_20260514.md`"
  - If interrupted before lane registry mark: pick up at `python tools/lane_maturity.py mark lane_grand_council_c1_world_model_falsification_review_20260514 --gate three_clean_review --evidence ".omx/research/grand_council_c1_world_model_adversarial_review_20260514.md"`
  - If interrupted before commit: pick up at canonical serializer commit with `--expected-content-sha256`

---

## 15. Conclusion

The probe-1 finding is **methodologically overturned**. The probe tested autonomous-RNN-from-z_init-with-zero-action-input, NOT the Ha-Schmidhuber 2018 / Hafner DreamerV3 2023 world-model premise. The WorldModelModule as engineered is missing 5 of 6 minimal-viable components of a world-model architecture. The 91.4% margin verdict is real but irrelevant to the falsification claim.

**Verdict: D (ENGINEERING-REVISION-FIRST).** Build probe v2 with posterior + prior + residual at $0 / 30 min cost. Re-run on synthetic AND real video at matched-DOF and matched-bit-budget. Re-convene the council on probe v2 verdict.

Per CLAUDE.md "KILL is LAST RESORT" + "Forbidden premature KILL without research exhaustion" + "Council conduct — non-negotiable", the architecture revision to DROP `WorldModelModule` is **PAUSED PENDING FAIR PROBE V2**.

Foveation finding (Probe-2: `ego_motion_radial` 57% margin) is INDEPENDENT of this review and STANDS as a confirmed Atick-Redlich 1990 revalidation on real video.

---

Tagged `research_only=true`. **NO score claims**. **NO GPU spend** ($0 deliberation on macOS CPU). All 6 hooks declared per Catalog #125. All 11 journal-grade elements per `.omx/research/journal_lab_grade_documentation_standard_directive_20260514.md` honored. Crash-resume per Catalog #206. Cross-refs [[c1-real-video-probe-rerun]] · [[c1-architecture-revision-per-real-video-probe-verdict]] · [[c1-world-model-foveation-campaign-l1-scaffold]] · [[time-traveler-l5-staircase-steps-2-3]] · [[zen-floor-field-medal-grade-council]] · [[design-tension-ship-both-interpretations]].
