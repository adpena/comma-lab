# Local→Contest scorer drift ladder + drift-aware correction/safety-margin

**Date:** 2026-06-11
**Author:** drift-ladder subagent (`drift-ladder-20260611`)
**Operator ask (2026-06-11):** *"there will still be drift even with the best implementation vs upstream
Linux x86_64 CPU or CUDA T4 — smaller, but we can test it and correct/engineer for it."* Characterize the
full local→contest drift ladder and design a drift-aware correction/safety-margin so the fast LOCAL
advisory (MLX / macOS-torch) is a trustworthy *predictor* of the TRUE contest score (Linux x86_64 CPU
leaderboard + CUDA T4), with the contest exact eval staying the final arbiter.

**Evidence grade:** all numbers below are assembled from EXISTING measured anchors. macOS (torch OR MLX) is
`[macOS-CPU advisory]` / `[macOS-MLX research-signal]`, NON-PROMOTABLE. Linux x86_64 CPU is `[contest-CPU]`
(leaderboard authority); NVIDIA T4 is `[contest-CUDA]` (CUDA authority). No MPS used anywhere. No score
claim / promotion / rank / kill is made here.
**Did the exact frontier pointer move?** No. This is a *correction/safety-margin design* + a *test plan*,
not an exact-eval row. The exact contest eval remains the only arbiter.

**Scope discipline:** This memo did NOT touch the running daemons (capstone, atlas) or the MLX scorer
files (`mlx_scorer_adapters.py` / `capstone_trainer.py` — other subagents own those). No paid dispatch
fired (estimate-first; see §4).

---

## 0. TL;DR — the ladder, the dominant rung, the rule

```
   contest Linux-x86_64-CPU  (LEADERBOARD AUTHORITY)        contest CUDA-T4 (CUDA AUTHORITY)
        ▲                                                        ▲
        │  RUNG C: cross-AXIS gap  CUDA−CPU = +0.0330 ± 0.0004 score (THE DOMINANT RUNG, ~3000× B)
        │           (R_pose=5.04±0.10, R_seg=1.17±0.01; 70% pose / 30% seg; rate unchanged)
        │
   macOS-torch-CPU  (our fast proxy for the CPU leaderboard axis)
        ▲
        │  RUNG B: macOS↔Linux-x86_64 CPU, SAME device mode  = +1.05e-5 score (SegNet-only bias)
        │           (5 same-archive anchors; aggregate PR107 = 6e-6; per-component = GAP, see §2.2)
        ▼
   MLX-CPU / MLX-GPU / ExecuTorch  (the GPU-fast fidelity scorer, below the proxy)
            RUNG A: MLX-CPU↔macOS-torch-CPU = ~0 (2 flips/19.66M; d_pose 8.7e-11)   ← negligible
                    MLX-GPU↔macOS-torch-CPU = 243 flips/19.66M (d_seg 1.24e-5); d_pose 2.76e-4 ← pose caution
                    ExecuTorch-MLX-GPU PoseNet = FP32-exact (rel_mse 5e-14); SegNet blocked (kernel bug)
```

**Dominant rung by a wide margin: RUNG C, the CUDA−CPU cross-AXIS gap (+0.0330).** It is ~3,100× larger
than RUNG B (macOS↔Linux CPU, +1.05e-5) and ~2,700× larger than the worst RUNG-A pose drift expressed in
score points (~1.2e-4). The two within-CPU-axis rungs (A + B) are essentially noise; the *cross-axis* rung
is a structural, near-constant bias. **Consequence:** a local advisory that targets the *CPU leaderboard*
axis needs only a tiny one-sided safety band (~3e-5 score); a local advisory that targets the *CUDA* axis
needs a large band (~0.033) — but in practice we should NOT predict the CUDA axis from a CPU-axis local
proxy at all; the two axes are separate evidence spaces and each needs its own exact eval.

**The rule (the engineer-for-it part, §3):** a fast LOCAL macOS-torch-CPU advisory `S_local` qualifies as a
*contest-CPU candidate worth spending an exact eval on* when
`S_local − bias_B + guard_B  <  S_frontier_CPU` (conservative projection beats the CPU frontier).
With `bias_B = +1.0e-5` and `guard_B = +3e-6` this is the existing eureka rule; the *required beat* over
T_1 = 0.19 is therefore only ~**1.3e-5 score** (the bias minus guard is ~+7e-6, so a local 0.18999 already
conservatively projects below 0.19). MLX-GPU adds a pose-axis caution that *widens* the band near the
frontier (§3.3). **The CUDA axis is never projected from a CPU-axis local proxy.**

---

## 1. RUNG C — the CUDA−CPU cross-axis gap (THE DOMINANT RUNG)

**Source:** `reports/public_pr100_108_cpu_cuda_drift_analysis_20260508.json` (5 paired contest CI bot
comments PR100/101/102/103/105) + `docs/findings/cuda_cpu_auth_eval_split_20260508.md` (canonical
write-up). These are the contest's OWN paired `--device cuda` / `--device cpu` outputs on the same archive
bytes — the highest-authority cross-axis anchor we have.

| Quantity (CUDA vs CPU, same archive) | min | median | max | σ |
|---|---:|---:|---:|---:|
| Δscore (CUDA − CPU) | +0.0325 | **+0.0329** | +0.0335 | 4e-4 |
| R_pose = pose_cuda / pose_cpu | 4.97 | **5.04** | 5.21 | 0.10 |
| R_seg = seg_cuda / seg_cpu | 1.16 | **1.17** | 1.18 | 0.01 |
| pose share of the gap | — | **70%** | — | — |
| seg share of the gap | — | **30%** | — | — |
| rate share of the gap | — | **0%** | — | — |

Score-points decomposition of the +0.0330 gap (medal-band operating point, pose_cpu ≈ 3.4e-5):
- **Pose ≈ +0.0230 (70%)** — amplified through `sqrt(10·d_pose)`: the 5.04× raw-pose ratio softens to a
  √5 ≈ 2.24× *term* ratio (0.0417 CUDA vs 0.0187 CPU).
- **Seg ≈ +0.0098 (30%)** — `100·d_seg` with R_seg = 1.17.
- **Rate = 0** — bit-identical archive, no decode in the rate term.

**Constancy:** for 5 independently-engineered HNeRV-class archives across 4 authors and 3 codec strategies,
Δscore sits in [0.0325, 0.0335] with σ ≈ 4e-4. This is a *structural property of the scorer*, not a
statistical artifact — so it is a candidate for a *calibration offset* (a stable bias), not just a noise
band, **within the HNeRV medal-band class**. Mechanism still unresolved (see §1.1).

**Why CUDA > CPU:** `evaluate.py --device` switches BOTH (a) the GT decoder (CUDA→DALI/NVDEC,
CPU→PyAV/libav) and (b) the perception-net numeric kernels. The leading additive-precision model
(`σ²_cuda ≈ K·L·ε²·||x||²`, L≈50 conv ops, ε≈1.7e-3 → σ²≈1.4e-4) reproduces the observed R_pose ≈ 5.0
numerically, and regression heads (PoseNet) are quadratically noise-sensitive while argmax classification
(SegNet) is stable under small logit perturbations — explaining the 5× pose / 1.17× seg asymmetry.

### 1.1 RUNG-C caveats (what is NOT proven)
- **Class-bounded.** The +0.0330 / R_pose ≈ 5.04 / R_seg ≈ 1.17 constants are descriptive for the
  *HNeRV medal-band cluster* (pose_cuda ≈ 1.7e-4, pose_cpu ≈ 3.4e-5) ONLY. They are NOT validated
  cross-family; at high pose_cuda the additive-noise model saturates and R_pose → 1 (untested). Do NOT
  apply this offset to a non-HNeRV substrate (e.g. wavelet/Z8, PR106 format0d) without its own paired
  anchor. The FALSE-AUTHORITY firewall: this is a *spend trigger* / *projection prior*, NEVER a CUDA→CPU
  or CPU→CUDA conversion for promotion/rank/kill.
- **Mechanism unresolved (H1 decoder / H2 network-kernel / H3 mixed).** The 25/75 decoder/network split is
  a *measurement target*, not a settled fact. The shared-tensor 2×2 matrix (PyAV+CPU-fwd, DALI+CUDA-fwd,
  cross cells) is the rigorous test (§4, optional). It does NOT change the +0.0330 offset's usability as a
  class-bounded projection prior.

---

## 2. RUNG B — macOS-torch-CPU ↔ Linux-x86_64-CPU (same device mode)

**Source:** `.omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json` (5 same-archive
paired anchors, fitted via `tac.optimization.local_cpu_contest_drift`) + the PR107 aggregate anchor
(`docs/findings/cuda_cpu_auth_eval_split_20260508.md` §0).

This rung is the **macOS-vs-Linux x86_64 floating-point drift on the SAME `--device cpu` mode** — the price
of macOS not being 1:1 contest-compliant CPU hardware.

| Anchor / fit | local − contest score | mechanism |
|---|---:|---|
| PR101 fec6 `6bae0201` (600-pair) | +1.00e-5 | SegNet-only: seg_local higher by 1.0e-7 |
| DQS1 top32 `3c4e15bf` | +1.00e-5 | SegNet-only |
| DQS1 diversity k002 `4432525d` | +1.20e-5 | SegNet-only (seg 1.2e-7) |
| DQS1 drop-one `088c17e2` | +1.10e-5 | SegNet-only |
| PR107 apogee `7ecb0df1` (aggregate) | +6.0e-6 (= 0.19664189 macOS − 0.1966358879 GHA Linux) | aggregate only |
| **Fitted stable-core band** | **median +1.05e-5, range [+1.0e-5, +1.2e-5], σ ≈ 8.3e-7, guard 3e-6** | **SegNet rounding, PoseNet & rate IDENTICAL** |

**Interpretation:**
- macOS-torch-CPU is an **excellent** CPU-leaderboard proxy: the bias is +1.05e-5 score and it is
  *entirely SegNet rounding* (PoseNet and rate are bit-identical local-vs-contest across all 5 anchors).
  The drift is one-sided (macOS scores slightly *worse* than Linux), which is the safe direction for a
  conservative projection.
- **Trust region matters.** The fit is `dqs1_fec6_like_same_archive_segnet_rounding`. Out-of-class older
  rows show MUCH larger offsets (+1.36e-4, +2.88e-4) and must be treated as `wide_or_mixed` until
  separately calibrated by substrate class / raw-output identity / scorer path. The +1.05e-5 constant is
  HNeRV-medal-band-bounded, same caveat as RUNG C.

### 2.2 The GAP this memo flags (per-component macOS↔Linux drift)
The PR107 macOS↔Linux anchor is **aggregate only** (6e-6 score); the 5 calibration anchors give the
*aggregate* macOS-advisory-vs-Linux-CPU delta (+1.05e-5) and *attribute it to SegNet* because their local
rows happen to have pose/rate bit-identical to contest. But there is **no dedicated same-archive
macOS-CPU vs Linux-CPU run that isolates d_seg argmax-flip count and d_pose MSE as a *direct* paired
component measurement** the way RUNG A does (MLX vs torch). The +1.05e-5 SegNet attribution is inferred
from the per-anchor component rows, not from a controlled macOS-vs-Linux component probe. **Closing this
gap is exactly what the §4 test plan does** (run the exact scorer on a frame sample on Linux x86_64 CPU and
diff d_seg/d_pose against the local macOS-torch-CPU components). It is cheap and it would convert RUNG B
from "aggregate + inferred attribution" to "measured per-component."

---

## 3. RUNG A — MLX (CPU / GPU / ExecuTorch) ↔ macOS-torch-CPU

**Source:** `.omx/research/mlx_scorer_port_drift_audit_20260611.md` (real 0.mkv 100-pair sample,
19.66M SegNet argmax pixels) + `.omx/research/executorch_mlx_delegate_scorer_spike_20260611.md`.

| Rung-A path | d_seg argmax flips / 19.66M | d_seg flip rate | d_pose component abs max | verdict |
|---|---:|---:|---:|---|
| **MLX-CPU** vs torch-CPU | **2** (both genuine ties, margin 2.4e-7) | 1.0e-7 | **8.7e-11** | bit-faithful — negligible |
| **MLX-GPU (Metal)** vs torch-CPU | **243** (all boundary near-ties, margin 5.2e-5) | 1.24e-5 | **2.76e-4** | d_seg negligible; **pose caution near frontier** |
| **ExecuTorch-MLX-GPU PoseNet** | n/a | n/a | rel_mse 5e-14, max_err 7.6e-6 (FP32-exact) | GO (zero-port) |
| ExecuTorch-MLX-GPU SegNet | unmeasured (steel_gemm JIT bug) | — | — | conditional NO-GO (fixable kernel bug, not arch) |

**Interpretation:**
- **MLX-CPU = torch-CPU** at the precision the contest charges → MLX-CPU is a bit-faithful near-authority
  *cross-check* for the CPU axis (still macOS, still advisory, but it doesn't *add* drift on top of
  macOS-torch-CPU).
- **MLX-GPU d_seg drift is negligible** (1.24e-5 flip rate, all at decision-boundary near-ties — classic
  GPU/CPU fp32 reduction-order non-associativity, first amplified at the SE global-avg-pool). Trustworthy
  for training/atlas SegNet gradients + sensitivities.
- **MLX-GPU pose is the one real caution.** The pose-component abs drift 2.76e-4 can *exceed the pose
  signal itself* near the PR106 frontier (d_pose ≈ 3.4e-5). MLX-GPU pose is fine as a *relative*
  training/ranking signal, but NOT a trustworthy *absolute* d_pose readout near the frontier without a
  torch-CPU re-score. **ExecuTorch-MLX-GPU PoseNet is FP32-exact** (rel_mse 5e-14) → the zero-port path
  *eliminates* RUNG-A pose drift if/when the SegNet steel_gemm kernel bug is resolved (or PoseNet runs on
  ExecuTorch-GPU while SegNet stays torch-CPU).

**RUNG A in score points:** worst-case MLX-GPU contributions are d_seg ≈ 100·(1.24e-5) ≈ 1.2e-3 and a
pose-term swing of order √(10·2.76e-4) ≈ 0.05 *if* the absolute d_pose were read off MLX-GPU directly near
the frontier — which is exactly why the rule (§3.3) forbids absolute MLX-GPU pose readout near the
frontier. With the torch-CPU pose re-score in place, RUNG A collapses to ≈ the MLX-CPU level (negligible).

---

## 4 (renumbered to 3). The drift-aware correction / safety-margin DESIGN

This is the "engineer-for-it" deliverable: a **drift-aware threshold gate** so a fast LOCAL advisory `S`
must beat the threshold by the *cumulative* one-sided drift band before it qualifies as a contest
candidate, plus the *calibration offset* where the drift is a stable bias rather than noise.

### 3.1 Two distinct prediction targets — never mix them
1. **Predicting the CPU leaderboard axis from a CPU-axis local proxy (macOS-torch-CPU or MLX-CPU).**
   Cumulative band = RUNG A (if MLX) + RUNG B. Both are tiny and one-sided. **This is the supported path.**
2. **Predicting the CUDA axis.** The cross-axis RUNG C gap (+0.0330) is large *and class-bounded*. We do
   NOT predict the CUDA axis from a CPU-axis local proxy. If a CUDA-axis number is needed, it requires its
   own `[contest-CUDA]` exact eval. RUNG C is usable only as a *class-bounded projection prior* (e.g.
   "this HNeRV archive's CUDA score will be ≈ CPU + 0.033") for triage, never for a claim.

### 3.2 The CPU-leaderboard safety-margin rule (calibration-offset form)
Use the existing eureka math (`tac.optimization.local_cpu_contest_drift`), generalized for the local rung:

```
bias       = bias_B  (+ bias_A if the local rung is MLX-CPU; MLX-CPU bias_A ≈ 0)
guard      = guard_B (+ guard_A) ; guard_B = 3e-6
conservative_projected_contest_CPU = S_local − bias + guard
QUALIFIES_AS_CPU_CANDIDATE  ⇔  conservative_projected_contest_CPU < S_frontier_CPU
```

- For macOS-torch-CPU on an **HNeRV-medal-band, same-archive** candidate: `bias = +1.0e-5`, `guard = 3e-6`.
  The conservative projection subtracts the bias (macOS reads high) then adds the guard back → net required
  *beat* over a target T is **~+1.3e-5 score**. So a local macOS-CPU advisory of **0.189987 or lower**
  conservatively projects below **T_1 = 0.19** and warrants a `[contest-CPU]` exact eval. (Current CPU
  frontier is **0.191099824**, so the live trigger threshold is local < **0.191093** ≈ frontier − 7e-6.)
- For MLX-CPU as the local rung: add nothing meaningful (`bias_A ≈ 0`, MLX-CPU is bit-faithful). MLX-CPU is
  a valid fast near-authority cross-check before spending.

### 3.3 The pose caution that WIDENS the band (MLX-GPU near the frontier)
If the local advisory `S_local` was produced with **MLX-GPU** as the scorer (the fast training/atlas
path), the *absolute d_pose* is untrustworthy near the frontier (drift 2.76e-4 ≳ signal 3.4e-5). Rule:
- **Do not** form `S_local` from an MLX-GPU *absolute* d_pose near the frontier. Recompute the absolute
  d_pose on **torch-CPU** (or ExecuTorch-MLX-GPU PoseNet, which is FP32-exact) before applying §3.2.
- The MLX-GPU SegNet term IS usable directly (flip rate 1.24e-5 negligible).
- Equivalently: the cumulative band when the pose term comes from MLX-GPU is dominated by the pose drift
  (~√(10·2.76e-4) ≈ 0.05 in the worst case), which is *larger than RUNG C* — so MLX-GPU absolute pose near
  the frontier is the WORST predictor and must be re-scored. This is the single most important "engineer
  for it" guardrail: **the fast scorer's pose readout is the thing to distrust, and the fix already exists
  (torch-CPU re-score every N epochs, or ExecuTorch-GPU FP32-exact PoseNet).**

### 3.4 Stable-bias vs noise — which rungs get an *offset* vs a *band*
| Rung | bias (offset) | guard (noise band) | use as |
|---|---:|---:|---|
| A (MLX-CPU) | ≈ 0 | ≈ 0 (2 flips, ties) | bit-faithful cross-check |
| A (MLX-GPU SegNet) | ≈ 0 | +1.2e-3 score | usable direct |
| A (MLX-GPU pose, near frontier) | — | **untrustworthy** → re-score on torch-CPU/ExecuTorch | DO NOT read absolute |
| B (macOS↔Linux CPU) | **+1.05e-5 (offset)** | +3e-6 (guard) | conservative CPU projection |
| C (CUDA−CPU cross-axis) | **+0.0330 (offset, HNeRV-class only)** | ±4e-4 | class-bounded triage prior; never a claim |

### 3.5 Tie to the GOAL threshold ladder (T_1 = 0.19, T_3 = 0.15)
- A local macOS-CPU advisory qualifies as a **real contest-CPU candidate worth an exact eval** when its
  conservative projection (§3.2) crosses the *target it claims*. For T_1 = 0.19 the required local value is
  ≤ ~0.189987 (HNeRV-class, same-archive). For the live CPU frontier 0.191099824, local < ~0.191093.
- "Any score sub-0.19 is good progress" (operator) → a local advisory that conservatively projects below
  0.19 is precisely the eureka spend trigger: claim the lane, dispatch a `[contest-CPU]` exact eval, and
  (if the candidate is shippable) a paired `[contest-CUDA]` eval too — RUNG C tells you the CUDA number
  will be ≈ +0.033 higher for HNeRV-class, which is informational only.
- The bands are tiny relative to the T_1→T_3 distance (0.04). The drift ladder does NOT obstruct the goal;
  it only sets a ~1.3e-5 conservative cushion on the CPU axis. The real obstacle remains lowering the exact
  CPU score, not the local↔contest drift.

---

## 5 (renumbered to 4). The TEST plan (estimate-first)

**Goal of the test:** convert RUNG B from "aggregate + inferred SegNet attribution" (§2.2) to a *measured*
per-component macOS-torch-CPU ↔ Linux-x86_64-CPU drift (d_seg argmax-flip count + d_pose MSE on a frame
sample), and (optionally) refresh a single RUNG-C paired point on a current-class archive.

### 4.1 Smallest faithful measurement (RUNG B per-component) — ESTIMATE
- **What:** run `upstream/evaluate.py --device cpu` (or the exact scorer forward on the cached
  `segnet_last_rgb` + `posenet_yuv6_pair` inputs) on a **Linux x86_64 CPU Modal container**, on the SAME
  archive + SAME 600-pair frame sample we score locally, and diff the per-component d_seg (argmax-flip
  count) + d_pose against the local macOS-torch-CPU run. Reuse `experiments/modal_auth_eval_cpu.py`
  (Linux x86_64, the canonical CPU eval path) on the current CPU frontier archive
  `b46897267ded…` (or the well-anchored fec6 `6bae0201`).
- **Cost estimate:** Modal CPU container ≈ **$0.06/hr**, 600-sample eval runs **60–120 min** + image
  start/safety margin ⇒ **≈ $0.12** per full CPU eval (per `docs/findings/cuda_cpu_auth_eval_split_20260508.md`
  §6.1 + §11). A frame-sample (e.g. first 100 pairs) is cheaper (~$0.04–0.06) but the canonical
  same-archive comparison wants the full 600 to match the local 600-pair components exactly.
- **Within the standing <$5 budget and <~$1 trivial threshold?** YES (~$0.12). **But I did NOT run it**
  because: (a) the operator instruction is estimate-first and the marginal value is *characterization*, not
  an exact-score row that moves the frontier; (b) a paid Modal CPU eval requires a lane-claim
  (`tools/claim_lane_dispatch.py`) + HARVEST-OR-LOSE harvest discipline, which is a real dispatch the
  operator should greenlight rather than have a characterization subagent fire unattended; (c) the
  existing 5-anchor calibration already pins the aggregate macOS↔Linux bias at +1.05e-5 with σ 8.3e-7, so
  the per-component refinement is a *nice-to-have* (it confirms the SegNet-only attribution directly) not a
  blocker. **Recommendation: GREENLIT-ON-REQUEST — it is trivially cheap (~$0.12) and within budget; fire
  it when an operator wants RUNG B promoted from inferred to measured.** Command sketch (verify argparse
  before running):
  ```bash
  # lane-claim FIRST (non-negotiable)
  .venv/bin/python tools/claim_lane_dispatch.py claim --lane-id lane_drift_rung_b_per_component_cpu_20260611 \
      --instance modal_cpu --status active --notes "RUNG B per-component macOS<->Linux CPU drift measure"
  # then the canonical Linux x86_64 CPU eval on the SAME archive we score locally:
  PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach experiments/modal_auth_eval_cpu.py \
      --archive <CPU-frontier archive.zip> --device cpu   # ~$0.12, 60-120 min, Linux x86_64
  # harvest within 24h (HARVEST-OR-LOSE) + diff per-component vs local macOS-torch-CPU run.
  ```

### 4.2 RUNG-C refresh on a current-class archive (OPTIONAL) — ESTIMATE
- **What:** paired `[contest-CPU]` (Modal CPU ~$0.12) + `[contest-CUDA]` (Lightning/Vast T4 ~$0.30) on ONE
  current archive to confirm whether R_pose ≈ 5.04 / Δscore ≈ +0.033 still holds for our *current*
  substrate class (the 5 anchors are 2026-05 HNeRV public PRs; our CPU frontier is now
  `lane_pr110_payload_entropy_recode`). **Cost ≈ $0.42 combined** (per the findings-doc §11 recipe).
- **Recommendation: PROPOSE, don't auto-run.** It is >$0.30 and its value is *cross-class validation of a
  prior*, not a frontier row. If RUNG C is to be used as a triage offset on the *current* class, this
  $0.42 paired eval is the right de-risk — operator greenlight + lane-claim.

### 4.3 What I did NOT do (and why)
- No paid dispatch fired (estimate-first; both measurements are characterization, not frontier rows).
- Did not touch capstone/atlas daemons or the MLX scorer files (other subagents own them).
- The shared-tensor 2×2 decoder/network mechanism matrix (RUNG-C mechanism isolation) is out of scope for a
  drift *characterization+correction* memo; it's a separate mechanism-localization program (§1.1) and does
  not change the +0.0330 offset's usability as a class-bounded prior.

---

## 6 (renumbered to 5). Solver / system wire-in (results become intelligence)

Per CLAUDE.md "Results must become system intelligence" — the 6 unified-Lagrangian hooks:
1. **Sensitivity-map** — ACTIVE (prior). The cross-hardware top-K leverage drift is already a documented
   sensitivity signal (`docs/per_byte_sensitivity_comparative_analysis_methodology.md`: 6.4% [macOS-CPU]
   vs 11.1% [contest-CUDA T4] on the SAME archive). The §3 rule formalizes the *axis* on which sensitivity
   is read.
2. **Pareto constraint** — N/A (this memo sets a *projection band*, not a feasible-set constraint; the band
   informs the spend-trigger, not the rate/seg/pose polytope).
3. **Bit-allocator hook** — N/A (no per-tensor importance change).
4. **Cathedral autopilot dispatch hook** — the §3.2 eureka rule IS the autopilot's CPU-axis spend trigger
   (`tac.optimization.local_cpu_contest_drift` already emits `local_cpu_contest_drift_eureka_signal.v1`).
   This memo documents the *ladder* feeding that trigger; no new code landed (research+design only).
5. **Continual-learning posterior** — the §4.1 measurement, when run, adds a per-component RUNG-B anchor +
   (§4.2) a current-class RUNG-C anchor to the calibration; both reseed `local_cpu_contest_drift` /
   the CUDA-CPU ratio prior. Triggered on the empirical anchor when a dispatch fires.
6. **Probe-disambiguator** — the §4 plan IS the disambiguator between "RUNG B is SegNet-only +1.05e-5
   inferred" vs "measured per-component"; the RUNG-C mechanism (H1/H2/H3) disambiguator is the separate
   shared-tensor 2×2 program (§1.1, out of scope here).

**`research_only=true`** for this memo: no code landed, no exact-eval row, no promotion. It is a
characterization + correction-DESIGN + estimate-first test plan. Integration blocker for promoting RUNG B
to "measured": the ~$0.12 Modal CPU dispatch in §4.1 (operator-greenlight-on-request).

---

## 7 (renumbered to 6). Reproduce / sources
- RUNG C: `reports/public_pr100_108_cpu_cuda_drift_analysis_20260508.json`,
  `docs/findings/cuda_cpu_auth_eval_split_20260508.md`.
- RUNG B: `.omx/research/local_cpu_contest_drift_calibration_dqs1_fec6_20260522T194800Z.json`,
  `.omx/research/codex_findings_local_cpu_contest_drift_eureka_20260522T194925Z_codex.md`,
  `src/tac/optimization/local_cpu_contest_drift.py`, `tools/calibrate_local_cpu_contest_drift.py`.
- RUNG A: `.omx/research/mlx_scorer_port_drift_audit_20260611.md` (+ `…_artifacts/`),
  `.omx/research/executorch_mlx_delegate_scorer_spike_20260611.md`.
- Tooling: `tools/plan_dual_device_auth_eval.py` (paired CPU/CUDA plan/execute),
  `tools/xray_paired_cpu_cuda_axis_delta.py` (per-component axis delta),
  `experiments/contest_auth_eval.py` / `experiments/modal_auth_eval_cpu.py` (Linux x86_64 CPU eval),
  `tools/claim_lane_dispatch.py` (lane-claim before any dispatch).
- Frontier: `.omx/state/canonical_frontier_pointer.json` (CPU 0.191099824 `b4689726…`,
  CUDA 0.192… `9cb989ce…`).

## 8 (renumbered to 7). NO-FAKE / authority notes
- All local (macOS torch + MLX) numbers are `[macOS-CPU advisory]` / `[macOS-MLX research-signal]`,
  NON-PROMOTABLE. Only Linux x86_64 = `[contest-CPU]` (leaderboard authority); only NVIDIA T4 =
  `[contest-CUDA]`. No MPS anywhere.
- The +0.0330 (RUNG C) and +1.05e-5 (RUNG B) offsets are HNeRV-medal-band-class-bounded *projection priors
  / spend triggers*, NEVER score claims, conversions, promotions, ranks, or kills. The contest exact eval
  is the only arbiter.
- The per-component macOS↔Linux drift (RUNG B) is currently *inferred SegNet-only* from same-archive
  component rows; the §4.1 dispatch would make it *measured*. This is honestly flagged as a gap, not
  presented as a measured fact.
