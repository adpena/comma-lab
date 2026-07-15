# AdamC / MuonC research + reconciliation against our measured phenomena (2026-07-15)

**Operator directive:** "We should research adamc and muonc as well." Doctrine:
PAPER_WARM_START_FROM_DIVERGENCE (`paper_warm_start_from_assumption_divergence_not_route_or_dismiss_20260714`) —
trace the assumption fork, import what survives OUR premises (n=1 overfit, MLX fp32, single frozen
instance, Muon finishing stage, EMA-shadow inference).

**STORES CONSULTED:** DAG FEED-509-b3r/b3x (maglaw 3-arm A/B + ep75 overrun) · memory
`perparam_normalize_masks_all_norm_clipping_c0_confound_20260715` · canonical equation
`autoclip_percentile_threshold_v1` (anchor `autoclip_descent_speed_effect_n24_ab_measured_20260715`) ·
`src/tac/optimization/muon_finisher_mlx.py` · installed `mlx.optimizers` source ·
live launch configs (`experiments/results/levelset_n24_maglaw_arm*/launch.sh`) ·
`src/tac/witness_dsl/{curriculum_dsl,adaptivization_tickets_20260715}.py`.

means != ends: everything below is MEANS. **Pointer 0.19108 UNMOVED.**

---

## 1. AdamC — the exact result (source: full PDF read, arXiv:2506.02285v2)

**Paper:** Aaron Defazio (solo author), FAIR at Meta, "Why Gradients Rapidly Increase Near the End
of Training", arXiv:2506.02285v2 (dated 2025-06-14). 11 pages; read in full 2026-07-15.

### 1a. The equilibrium law (Van Laarhoven 2017 framework, recapped + extended in §3)

For a weight tensor `x` **immediately followed by a normalization layer** (LayerNorm/BatchNorm),
scale-invariance gives `⟨g_t, x_t⟩ = 0` (gradients orthogonal to weights). The SGD+decoupled-wd
norm recursion is then EXACT:

    ‖x_{t+1}‖² = (1−λγ)²‖x_t‖² + γ²‖g_t‖²        (paper Eq. 1)

Setting ‖x_{t+1}‖ = ‖x_t‖ and dropping the O(λ²γ²) term gives the steady state:

    ‖g_t‖ / ‖x_t‖ = sqrt(2λ / γ_t)                 (paper Eq. 2)

With a **decaying schedule γ_t → 0 the steady-state TARGET rises like 1/sqrt(γ_t)** — the measured
tail gradient-norm blow-up. Three stages (paper §3.1): burn-in → stationary tracking → tail
blow-up (target rises while the shrinking lr slows the chase). Dual symptom: **weight-norm
collapse** (constant λ keeps eating while gradient replenishment shrinks) — Fig. 4 shows AdamW
weight norm falling ~6500 → ~2000 over the cosine tail while AdamC stays ~flat.

For AdamW the same argument holds in the `A_t = diag(sqrt(v̂)+ε)` weighted norm
(`‖g‖_{A⁻¹}/‖x‖_A = sqrt(2λ/γ)`), and §4.2 shows AdamW ≈ balancing per-layer `‖x‖_∞ ≈ sqrt(γ/2λ)`.
§4.1: **original-Adam-style coupled wd (decay passed through the preconditioner) breaks the
derivation** — the norm terms live in different metrics, layers no longer converge to one ratio.
This is their conjectured root of the Adam-vs-AdamW gap. (This clause is load-bearing for OUR Muon
finding in §2/§4c below.)

### 1b. The exact AdamC correction (paper §5 + Algorithm 1)

    λ̂_t = λ · γ_t / γ_max

applied **only to layers followed by normalization** (in transformers: every linear layer treated
as normalized, EXCLUDING the output layer). Since MLX/PyTorch AdamW's applied decay term is
`x ← x·(1 − γ_t·wd)`, the corrected APPLIED term is `(γ_t²/γ_max)·λ·x` (Alg. 1 line 12:
`x ← x − γ_t·m̂/(sqrt(v̂)+ε) − (γ_t²/γ_max)·λ·x`). The gradient term is UNCHANGED; γ_max keeps λ's
scale comparable to the uncorrected default. Steady state becomes

    ‖g‖/‖x‖ = sqrt(2λ/γ_max)   — schedule-INDEPENDENT (flat gradient-norm trajectory).

Uncoupled wd (λ_t not multiplied by γ) does NOT fix it (§5: ratio still `sqrt(2λ_t)/γ_t`).
SGDC = the same correction for SGD+momentum.

### 1c. Measured effects (paper §6)

- 120M-param Llama-3-architecture LLM, FineWeb-Edu, 200B tokens, wd λ=0.05, lr swept on a
  power-of-two grid in [0.001, 0.02], cosine schedule: final loss **AdamW 2.461 → AdamC 2.457**,
  with a visibly lower loss curve THROUGHOUT the mid-run, tail gradient-norm blow-up removed,
  weight norm stable-vs-collapsing.
- ResNet-50/ImageNet SGD-momentum: **SGDC 77.07±0.10% vs SGDM 76.95±0.14%** (5 seeds); tail
  gradient blow-up eliminated; a separate slow linear gradient-norm drift REMAINS (their theory
  covers only the wd×schedule component).
- Closing question (verbatim relevance to us): *"Can any remaining drift in the gradient norm be
  eliminated without using stronger corrections such as projection?"* — our incumbent
  `--grad-normalize per-param` IS such a stronger (projection-class) correction; see §3a.

### 1d. Follow-up: Chou, "Correction of Decoupled Weight Decay" (arXiv:2512.08217, OpenReview)

Generalizes the correction beyond normalized layers using the weaker assumption
**"updates become independent of the weights at steady state"** (E⟨θ_{t−1}, u_t⟩ ≈ 0):

    E‖θ‖² = γ·C/(2λ)  at steady state (C = E‖u‖²)  ⟹  λ_t ∝ γ_t keeps norms schedule-stable,

i.e. the same effective-γ² decay term, derived for ALL optimizers with decoupled wd. Applied to
constrained Scion with a momentum-dependent layer-wise coefficient
`λ_{t,l} = (2−α)/(2α·C²_{t,l}) · γ_{t,l}` → the **ScionC** variant (124M NanoGPT/FineWeb-Edu:
val 2.838 vs 2.846; ViT-S/16 ImageNet-1k 90ep: 78.74±0.09 vs 78.68±0.09 — parity-or-better with
much more stable weight/gradient/spectral norms). Notes verbatim-relevantly that **Muon "can be
considered the Spectral-norm subset of unconstrained Scion" and "was proposed without weight
decay"**.

## 2. "MuonC" — does it exist?

**NO — "MuonC" is not a named optimizer in the literature as of 2026-07** (web sweep + both
papers read; Defazio defines only AdamC and SGDC; Chou defines ScionC). What the operator most
plausibly means, in decreasing order of fit:

1. **Muon with AdamC-style schedule-corrected decoupled weight decay** — the corrected-wd analog:
   Muon-group update `W ← W·(1 − γ_t·λ̂_t) − γ_t·NS(m_t)` with `λ̂_t = λ·γ_t/γ_max^muon`. No paper
   names this "MuonC"; ScionC (2512.08217) is the nearest published relative (Muon = spectral-norm
   Scion subset), and its independence-assumption derivation is **near-exact for Muon**: the
   Newton-Schulz-orthogonalized update has weight-independent norm by construction
   (`‖NS(m)‖_F ≈ sqrt(min(n,m))`), so `E‖u‖² = C` holds without any normalization-layer argument.
2. **Moonlight's "Muon with (plain) decoupled weight decay"** ("Muon is Scalable for LLM
   Training", arXiv:2502.16982): adds standard AdamW-style decoupled wd `W ← W(1−γλ) − γ·O` plus
   the `0.2·sqrt(max(fan_in,fan_out))` update-RMS matching — "consistent wd", not
   schedule-corrected.

We proceed with meaning (1) and say so explicitly. **Do not cite "MuonC (paper)" anywhere — no
such citation exists.**

### The sharp LOCAL finding this research surfaced (source-verified, MLX ≥ our pinned version)

`mlx.optimizers.Muon.apply_single` applies weight decay **COUPLED — added to the raw gradient
BEFORE momentum and Newton-Schulz orthogonalization** (`gradient = gradient + wd·parameter`),
NOT decoupled. Consequences for OUR finisher (`build_muon_finisher_optimizer`, muon_wd_eff
defaults to `--weight-decay` = 1e-4):

- Magnitude: the wd term enters at `λ·‖W‖ ~ 1e-4·O(1–10) ≈ 1e-4–1e-3` against raw gradient norms
  ~O(1–17) (C0 measured band 5.9–17.5): **3–4 orders down**, and NS then re-normalizes the whole
  update anyway (orthogonalization is scale-invariant, the direction perturbation is negligible).
  ⟹ **the Muon group's weight decay is effectively INERT — the finisher has NO weight-norm control
  at all** (Defazio §4.1's "coupled decay through the preconditioner loses the damping role" is
  exactly this, one step worse: NS destroys magnitude information entirely).
- Our `build_muon_finisher_optimizer` docstring said "Decoupled weight decay for the Muon group" —
  a **documentation bug** (fixed this landing; the #417 counted-but-inert shape at the doc surface).
- Under a flat-then-annealed Muon lr with no effective decay, Muon-group `‖W‖` follows undamped
  random-walk growth `d‖W‖² ≈ γ_t²·C per step` — no AdamC-type blow-up mechanism exists (no
  decay-vs-step equilibrium to destabilize) but norms drift monotonically UP, which the 0.997
  EMA shadow tracks with ~333-step lag. Whether this drift is score-relevant is UNMEASURED
  (prediction P3 below).

**The honest MuonC import is therefore NOT "scale muon wd by γ_t/γ_max"** (scaling an inert
coupled term = arming a no-op = the C0/#417 confound class again). It is: (a) first build a
DECOUPLED Muon-group wd path (Moonlight form), (b) then apply the AdamC correction to it.
Registered as an `AdaptivizationTicket` (never a hand flag; a fake Lever would violate NO-FAKE).

## 3. Assumption fork — their setting vs our premises

| Axis | Defazio/Chou setting | Our setting | Transfer |
|---|---|---|---|
| Weight decay λ | 0.05 (LLM), 5e-5–1e-2 (ResNet) | **1e-4** (trunk + finisher, sealed) | mechanism strength ∝ λ·Σγ_t: theirs ≳ O(1) (weight norm moved 70%); ours ≈ **1.8e-4 (n24 ep1–75)**, ≈ 0.011 (n600×3000ep @ ~75 opt-steps/ep) → ≲1% weight-norm effect. **500× weaker.** |
| Normalization layers | Load-bearing (scale-invariance ⟨g,x⟩=0) | **NONE in the witness** (coord-INR + FiLM; the frozen scorers' BatchNorm is eval-mode, untrained) | Defazio's exact derivation does NOT apply; Chou's independence form partially transfers — and is near-exact for our per-param-normalized updates (unit-norm per tensor ⟹ E‖u‖² weight-independent) and for Muon (NS-orthonormalized). |
| Schedule position | Cosine tail (γ_t → 0, second half of run) | maglaw window ep1–39/75 of a 3000-ep cosine: γ decayed ≲ a few % (window sits in warmup/early-cosine) | The blow-up regime was NEVER ENTERED in the A/B window. |
| Regime | Large-scale single-pass pretraining, noisy gradients, generalization | n=1 overfit, deterministic data, MLX fp32, EMA-shadow inference | Their "classical MNIST/CIFAR near-zero-gradient regime is different" caveat cuts toward US: overfit runs drive raw ‖g‖ DOWN near the minimum — a third behavior their theory brackets but does not model. |
| Optimizer | AdamW/SGD (+Muon absent) | AdamW trunk → Muon+AdamW finisher; **per-param grad normalize upstream** | Per-param normalize makes the applied update magnitude-STATIONARY by construction — the "stronger correction (projection)" their conclusion names. It sits closer to the fixed point AdamC approaches asymptotically. |

**Net import verdict: the MECHANISM is quantitatively negligible at our λ and run lengths; the
FRAME (schedule-stationarity of the applied update law) transfers and is load-bearing** — it
explains the maglaw durability split (§4a) and exposes the Muon-wd inertness (§2).

## 4. Reconciliation verdicts against our three measured phenomena

### 4a. Maglaw 3-arm n24 A/B — AutoClip wins early, REVERSES post-ep25; incumbent durable

(FEED-509-b3x anchor `autoclip_descent_speed_effect_n24_ab_measured_20260715`: ep25 d_seg
A 0.018045 / B 0.015902 / C 0.017737; AutoClip law effect B−C = −10.35%; BONUS overrun: armB ep75
0.018644 REGRESSING vs armA 0.015325 still descending.)

**VERDICT: DOES-NOT-APPLY as the specific wd×schedule mechanism; CONSISTENT (and sharpening) at
the stationarity-frame level.**

- Mechanism arithmetic: λ·Σγ_t over ep1–75 at n24 ≈ 1e-4 × (1e-3 × ~24 steps/ep × 75) ≈ **1.8e-4**
  total multiplicative weight-norm pressure — nothing. lr had decayed ≲ a few % in the window
  (early cosine). No normalization layers. The paper's equilibrium timescale ~1/(2λγ) ≈ 5e6 steps
  is 2–3 orders beyond the whole run. The AdamC mechanism cannot have caused the reversal.
- What the AdamC FRAME says instead: the two arms differ in whether the applied update law is
  **stationary**. Arm A (per-param normalize) pins per-tensor update norm = lr exactly — a
  projection-class correction, immune to ANY raw-‖g‖ drift (this is why it is durable). Arm B
  re-exposes raw ‖g‖ and inserts a **trailing-percentile estimator (AutoClip p10/w1000) of a
  NON-STATIONARY quantity**: near the overfit minimum raw norms drift/spike (spike share was
  already measured 44% lane-flicker-driven on this vehicle), the window statistics lag the drift,
  and the effective step wanders. Defazio's lesson generalized: *any magnitude law referenced to a
  schedule- or phase-dependent gradient scale inherits that nonstationarity unless corrected.*
  AutoClip's window IS an uncorrected reference.
- The C−A leg (lineage −1.71% ≈ neutral) already told us normalize-vs-not barely moves the ep1–39
  descent; the B-vs-A durability split is therefore about the CLIP LAW's stationarity, which is
  the AdamC frame verbatim, not the AdamC formula.

### 4b. C0 grad-clip frac_clipped=1.0 saturation

**VERDICT: DOES-NOT-APPLY.** The saturation signature is wrong for the AdamC mechanism in three
independent ways: (i) it is present at EVERY accum step from ep1 (a 12× level mismatch between
clip 0.5 and the natural gnorm scale ~6 of this loss stack — a set-point miscalibration, not
late-training growth); (ii) the AdamC mechanism needs the schedule tail + λ strong enough to move
norms (λΣγ ≈ 1e-4-scale in the window — see 4a); (iii) no normalization layers. And the clip was
INERT anyway (per-param normalize downstream divides out any uniform scale — the confound memory).
AdamC would predict a slow ‖g‖/‖w‖ RISE toward the tail; C0's telemetry shows a LEVEL offset from
step one. Different phenomenon, different cause.

### 4c. Muon finishing stage (#269/#270: lr-anneal ×0.1 + warm-start momentum; wd 1e-4; EMA 0.997)

**VERDICT: CONSISTENT-WITH-A-TWIST — the AdamC lens exposes a REAL latent defect, but with the
OPPOSITE sign to the paper's.** The paper's tail pathology needs an ACTIVE decoupled wd fighting a
decaying lr. Our finisher has (source-verified, §2): a Muon group whose wd is coupled-through-NS ≈
**inert** (no norm control at all → slow undamped ‖W‖ growth, EMA-tracked), an AdamW rest group at
FLAT lr (no schedule → its equilibrium target sqrt(2λ/γ) is STATIC → no AdamC drift), and a trunk
whose λΣγ ≈ 1% at n600 full length (marginal). So: no AdamC-type blow-up anywhere in the finisher —
but also no functioning weight-norm control in exactly the stage that polishes the shipped
EMA shadow. The `--muon-lr-final-frac 0.1` anneal shrinks the growth increments γ_t²C as it decays
(a partial accidental mitigation). The honest fix ladder: decoupled Muon wd (Moonlight) → then the
AdamC/Chou correction λ̂_t = λ·γ_t/γ_max^muon (which for Muon is theoretically CLEANER than for
Adam, since E‖u‖² is exactly weight-independent). Ticketed, not armed (§2).

### 4d. Slot into the DE-derivation framework (#318) + EoS sweep (#305)

**YES — this is a derived LAW, exactly the constants-are-poison shape.** The norm recursion is a
linear ODE in ‖x‖²:  d‖x‖²/dt = −2γ(t)λ‖x‖² + γ(t)²·E‖u‖², with closed-form envelope and
time-constant 1/(2λγ). The registered equation (`adamc_wd_lr_equilibrium_v1`) carries: the
steady-state ratio law, the corrected-λ̂ law, the time-constant, and the **mechanism-strength
scalar λ·Σγ_t** — a config-computable diagnostic that says UP FRONT whether the AdamC mechanism
can matter for a given run (ours: 1.8e-4 at n24, ~0.011 at n600 → predicted-null). It composes
with the EoS/training-dynamics sweep as the wd-channel term that the EoS literature (curvature
channel) does not cover; the two channels are additive candidate explanations for any late-run
gradient-norm drift we observe, and the λΣγ scalar cleanly deconfounds them (if λΣγ ≪ 1, observed
drift is NOT the wd channel).

## 5. Derived predictions the next n24 A/B can falsify

- **P1 (AdamC-null, the cargo-cult guard):** `CorrectedWeightDecay` ON vs OFF, same seed/config,
  n24 ≥150 ep: |Δd_seg| within seed noise AND per-tensor ‖W‖ trajectories differ < 0.1%. Derived
  from λΣγ ≈ 2e-4–1e-2 ≪ 1. **If the arms separate, the law is WRONG about our effective decay
  channel** (would imply another implicit-decay path — e.g. eikonal/length regularizers acting as
  weight shrinkage — a finding worth more than the lever).
- **P2 (AutoClip-throttle mechanism for the armB reversal):** in the armB overrun telemetry,
  post-ep25 `frac_clipped` rises and clip_t trails a drifting/spiking ‖g‖ (already-logged fields).
  A ≥150-ep A/B with a shorter window w or higher percentile should shift the reversal epoch.
  If frac_clipped does NOT rise while d_seg regresses, the throttle story is falsified and the
  reversal is landscape-side (spike-guard interaction next suspect).
- **P3 (Muon-norm-drift):** across any finisher-phase checkpoint sequence, Muon-group per-tensor
  ‖W‖ grows monotonically (undamped random walk, slope ∝ γ_t²) — checkable for $0 from existing
  stage checkpoints. If growth is material AND late-finisher d_seg wobble correlates, the
  decoupled-Muon-wd ticket gets promoted to a build.

## 6. What landed (this unit)

1. This memo.
2. Trainer lever (default-OFF, byte-identical off): `--weight-decay-corrected` in
   `experiments/train_levelset_witness_realized_through_R_mlx.py` — per-epoch
   `opt.weight_decay = λ·(lr_t/lr_max)` on the AdamW TRUNK (exact AdamC Alg. 1 form under MLX's
   `x·(1−lr·wd)` application; γ_max = `--lr`). TRUNK-ONLY scope by construction (inside the
   pre-Muon lr-schedule block); the Muon side is ticketed, not faked (§2).
3. DSL Lever factory `CorrectedWeightDecay()` (curriculum_dsl) — arms the flag; duty-to-measure
   via the never-fired activation ledger; A/B staged below.
4. `AdaptivizationTicket` `--muon-weight-decay` (adaptivization_tickets_20260715): MLX-coupled-
   through-NS inertness evidence + the decoupled+corrected law + the named unlock.
5. Docstring fix in `src/tac/optimization/muon_finisher_mlx.py` (was: "Decoupled weight decay for
   the Muon group"; is: MLX-coupled mechanism stated honestly).
6. Canonical equation `adamc_wd_lr_equilibrium_v1`
   (`src/tac/canonical_equations/adamc_wd_lr_equilibrium_20260715.py`), registered append-only:
   paper anchor (literature) + our config-derived mechanism-strength anchor (λΣγ) + P1 as the
   owed local anchor in the recalibration criteria.
7. DAG FEED row.

## 7. Staged A/B (do NOT fire — GPU belongs to the live 507/r6 chain; bf16-QC staging pattern)

```bash
# P1 arm (CorrectedWeightDecay ON), bounded n24, >=150 ep window per the durability follow-up;
# control = same config without --dsl-lever. FIRE ONLY when the r6 chain frees the GPU + admission passes.
.venv/bin/python tools/launch_witness_run.py \
  --config v9_cgauge_ideal_mod19 \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n24.npz \
  --num-pairs 24 --dry-start 3 \
  --dsl-lever CorrectedWeightDecay \
  --out-dir experiments/results/levelset_n24_adamc_wdc_<utc>
```

Admission bar: gradient-quality + no-flicker (relaxed-identity directive); verdict metric: d_seg
descent per wall-clock + per-tensor ‖W‖ trajectories (P1). Predicted verdict: NULL (that is the
point — the null guards the config against cargo-culting a 500×-out-of-regime correction, and a
non-null is a bigger finding than the lever).

**verdict_scope of everything here: FORMULATION (v9_cgauge lineage, n24/n600 MLX advisory).
[macOS-MLX research-signal], NON-PROMOTABLE. Pointer 0.19108 UNMOVED.**
