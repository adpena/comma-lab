# Codex 30k Strategy Review: CPU/GPU, Loader Drift, And Optimal Floor

Date: `2026-05-08`
Owner: `codex`
Branch observed: `main`
Write scope: this memo only
Dispatch performed: `false`
Score claim: `false`
Promotion/rank/kill claim: `false`

## Executive Verdict

Today's durable update is not "CPU replaces CUDA" and not "CUDA was wrong."
It is stricter: the project now has two score axes that must be labeled and
used for different decisions.

- `[contest-CUDA]` remains the internal promotion, regression, and retirement
  axis for exact archive/runtime work unless the operator changes that rule.
- `[contest-CPU]` is the public leaderboard reproduction axis and is now
  mandatory for any medal-band or public-rank statement.
- macOS CPU, MPS, public rounded comments, byte proxies, predicted bands, and
  xray probes are planning signals only.
- The score-lowering path is a byte-closed packet compiler plus calibrated
  solver loop, not another naked byte proxy or another uncalibrated solver
  ranking.

The immediate action bias should be: harvest active exact jobs, build real
runtime-consumed packets from parser-proven monolithic sections, and keep every
new learning signal in a machine-readable evidence schema so failures improve
the next build instead of disappearing into chat.

## 1. Proven Versus Hypothesis

### Proven

1. The CPU/CUDA auth-eval split is real.

   `upstream/evaluate.py --device cuda` and `--device cpu` can produce
   materially different component values for the same archive bytes. The
   public HNeRV comment cluster shows a stable external pattern across PR100,
   PR101, PR102, PR103, and PR105: pose CUDA/CPU ratio near `5`, seg CUDA/CPU
   ratio near `1.17`, and score gap near `0.033`. Public comments are not
   promotion evidence, but the paired pattern is too stable to ignore.

2. PR102 is a closed anchor on both axes.

   PR102 Linux x86_64 CPU replay via GitHub Actions produced:
   score `0.19537807523773826`, seg `0.00057601`, pose `0.00003460`,
   rate `0.00476704`, bytes `178981`, archive SHA-256
   `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`.
   The local hardened T4 CUDA replay at about `0.22839372989108092` matches
   the public CUDA band. Therefore the PR102 gap is device-axis semantics, not
   a broken local replay.

3. PR107 now has a Linux CPU anchor.

   PR107 apogee Linux x86_64 GitHub Actions replay produced score
   `0.1966358879`, seg `0.00058931`, pose `0.00003580`, rate `0.00475136`,
   bytes `178392`, archive SHA-256
   `7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb`.
   The M5 Max CPU replay was only about `6e-6` higher, so macOS CPU is useful
   for HNeRV-class development sweeps. It is still not a promotion tag.

4. The loader split is code-real and asymmetric.

   CUDA eval uses DALI/NVDEC for the ground-truth video path; CPU eval uses
   PyAV/libav. The compressed output path uses raw tensors. That means loader
   drift affects the GT side differently from the compressed side. Any
   mechanism story that ignores this split is incomplete.

5. The first precision story was wrong.

   FastViT-T12 in the scorer is RepMixer/convolutional, not attention-based.
   T4 is Turing and has no TF32 datapath. "TF32 attention compounding" is not
   a valid explanation for the observed pose ratio.

6. Xray tooling exists but is diagnostic only.

   `tools/probe_eval_loader_drift.py` records the intended `CPU+AV`,
   `CUDA+DALI`, `CUDA+AV/shared-input`, and `CPU+DALI` cells and keeps them
   non-promotable. `tools/probe_posenet_layer_drift.py` captures shared-input
   PoseNet activation drift. `tools/all_lanes_preflight.py` includes the
   loader-drift probe as Gate #22. None of these tools produces a score claim.

7. The active internal exact CUDA floor is not a public CPU medal row.

   The current inspected HNeRV CUDA anchor remains PR103-on-PR106 AC repack at
   about `0.20898105`, bytes `185578`, archive SHA-256
   `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`.
   Rate-only work must beat that byte anchor or prove scorer-component
   improvement with charged runtime consumption.

8. Several bad outcomes are scoped negatives, not family kills.

   PR101 lossy coarsening near `0.3517`, PR106 UNIWARD near `0.3372`, and
   PR107 lossy-coarsening CPU sweeps are measured-config retirements. They
   update trust regions, substrate sensitivity, and reactivation criteria;
   they do not retire lossy compression, UNIWARD-style priors, or K allocation
   as families.

### Still Hypothesis

1. Decoder-dominant drift.

   DALI/NVDEC versus PyAV/libav byte drift of 1 to 2 LSB can plausibly explain
   most of the PoseNet gap under reasonable Lipschitz assumptions. This is
   plausible, not proven, until DALI bytes and AV bytes are run through the
   cross cells.

2. Network-kernel-dominant drift.

   CPU/CUDA FP32 conv, GELU, linear, and reduction-order differences can
   plausibly generate shared-input PoseNet drift, especially because PoseNet
   is regression/MSE and SegNet is argmax/classification. This is plausible,
   not localized.

3. Mixed drift.

   The safest mechanism hypothesis is mixed decoder plus network drift. The
   share may differ by architecture class and pose operating point.

4. Global CPU/CUDA conversion ratios.

   The `R_pose ~= 5` and `R_seg ~= 1.17` values are strong for the HNeRV
   public-comment cluster. They must not be blindly applied to qhnerv, AV1,
   HPAC, categorical, foveation, self-compress, or future substrates without
   paired anchors.

5. True optimal floor.

   Numbers such as `0.155`, `0.140-0.180`, or `0.125-0.155` are roadmap bands,
   not floors. The true floor is task-aware rate-distortion under the contest
   scorer and exact packet constraints. It will be discovered by executable
   packet milestones, not by declaring a bound.

## 2. Immediate Score-Lowering Dispatch/Build Strategy

1. Harvest before launching.

   `arch_shrink_x0.4_lightning` is the highest-upside active lane already
   claimed/running in the inspected ledgers. Do not duplicate it. Harvest and
   adjudicate terminal artifacts when present: archive SHA, runtime-tree SHA,
   sample count, CUDA components, recomputed formula, logs, terminal claim row,
   and adversarial result review.

2. Build the corrected cross-paradigm ADMM-K plus Op1 path into a real packet.

   The `137469`-byte corrected substrate row is high-upside but not a scored
   archive. The next build is not exact CUDA; it is a byte-closed contest
   packet whose `inflate.sh` consumes the corrected stream, charges all side
   information, emits old/new section SHA boundaries, and passes strict
   compliance.

3. Requalify Path-B/Jacobian-Fisher before exact spend.

   The no-dead-K archive around `153671` bytes is closer to dispatchable
   because a real archive exists, but its rel_err is worse than an exact
   negative lossy-coarsening config. It needs scorer-aware reactivation
   evidence, tensor whitelist/blacklist or QAT recovery, strict compliance,
   and only then a lane claim plus exact CUDA.

4. Treat PR102 and PR107 CPU anchors as calibration assets, not automatic
   score wins.

   PR102 explains the public CPU/CUDA split; PR107 gives our apogee CPU
   baseline and a sharp distortion-vs-bytes warning. Neither authorizes
   promotion of CPU-built candidates on the CUDA axis.

5. Keep monolithic section discipline.

   PR101/PR103/PR106-style HNeRV packets are single-member payloads with
   internal sections. Candidate accounting must name parser-proven offsets,
   lengths, section SHA-256s, tail/padding rules, and runtime consumers. ZIP
   member narratives are invalid unless the packet really has separate
   members.

6. Do not spend exact CUDA on known non-candidates.

   Refuse rate-only packets above `185578` bytes, current `137469` byte-proxy
   rows without runtime packets, exact-negative measured configs without
   reactivation evidence, and any solver prediction without archive bytes,
   archive SHA, runtime SHA, and consumed changed bytes.

## 3. Solver And Autopilot Integration Requirements

The solver stack should become an evidence router, not a score authority.

Required contract for any row that can rank as active promotable exact CUDA:

- `score_claim=true`, `promotion_eligible=true`,
  `rank_or_kill_eligible=true`, and `ready_for_exact_eval_dispatch=true`;
- exact CUDA evidence marker and no proxy/planning/research marker;
- positive archive bytes;
- valid archive SHA-256 and runtime-tree SHA-256;
- CUDA score/component fields;
- no unresolved dispatch blockers;
- terminal or active dispatch-claim linkage when applicable.

Rows that fail this contract are still useful, but they must be typed as one
of: planning signal, diagnostic, exact negative, CPU public-axis anchor,
macOS advisory, byte proxy, runtime blocker, or unknown technique.

Meta-Lagrangian and Pareto requirements:

- Pareto dominance must be scope-local. Do not let one family suppress an
  orthogonal atom before stack interaction review.
- Pareto objectives should stay in component space:
  `(seg_dist, pose_dist, archive_bytes)`, then add cost/time and confidence.
- CPU/CUDA calibration may reweight planning marginals under a
  `target_axis="cpu_leaderboard"` or equivalent flag, but it cannot promote a
  packet without exact custody.
- Every predicted lower score must map to a build recipe that can emit a
  deterministic packet. "Predicted score" alone is not a candidate.
- Unknown technique rows must be visible and unrated until they receive either
  catalog coverage or an explicit planning-only disposition.

Autopilot should prioritize expected information gain among packet-closable
actions:

1. active exact harvests;
2. byte-closed runtime packets with strict compliance blockers only;
3. high-upside byte proxies that can be converted into runtime-consumed
   packets locally;
4. architecture/retrain studies only after their predispatch sanity ladder is
   satisfied;
5. broad research bands last.

## 4. Adversarial Falsification Tests

1. Paired exact CPU/CUDA eval on identical archive/runtime custody.

   For PR101, PR103, PR105, PR107 variants, and any shippable candidate, run
   both axes with the same archive SHA and runtime tree. Output must include
   components, sample count, hardware, evaluator hash, command/logs, and
   recomputed formula.

2. 2x2 loader/kernel discriminator on a CUDA+DALI host.

   Fill `CPU+AV`, `CUDA+DALI`, `CUDA+AV/shared-input`, and `CPU+DALI`.
   This is the decisive split between decoded-byte drift and network-forward
   drift. Result tag remains `[diagnostic]`.

3. DALI bytes through CPU and AV bytes through CUDA.

   If CPU forward on DALI bytes moves toward CUDA, loader is dominant. If CUDA
   forward on AV bytes stays near CUDA, network kernels dominate. If both move,
   mixed drift wins.

4. Controlled LSB injection.

   Inject 1, 2, and 3 LSB RGB perturbations into AV-decoded tensors and run
   PoseNet/SegNet. If about 1.5 LSB reproduces the observed pose gap, decoder
   drift becomes the leading mechanism.

5. PoseNet/Hydra layer xray.

   Run shared-input layer hooks across CPU and CUDA. Gradual accumulation
   supports additive precision noise; sudden jumps identify modules; flat low
   drift pushes blame back to loader bytes.

6. Cross-family calibration sweep.

   Learn CPU/CUDA profiles separately for HNeRV, qhnerv, AV1/H265, HPAC,
   categorical/HPM1, self-compress, and future substrates. Widen bands for
   classes with fewer than three paired exact anchors.

7. Packet closure/no-op falsification.

   Any byte-level candidate must prove target payload changed and scored
   runtime consumed the changed bytes. Reuse, decode/re-encode, cosmetic ZIP
   changes, and unconsumed side information must fail closed.

8. Solver backtest.

   Replay prior solver recommendations against exact anchors and exact
   negatives. Calibrate confidence by family and substrate. A solver that
   predicts the right byte count but the wrong distortion cliff should lose
   dispatch priority.

## 5. OSS, Paper, And Compliance Guardrails

OSS:

- Expose evidence contracts, diagnostic probes, packet grammar parsers, and
  conformance vectors as reusable APIs.
- Keep raw provider logs, private account metadata, local absolute paths, and
  unreviewed forecast bands out of public surfaces.
- Make diagnostic tools emit explicit `score_claim=false` and
  `promotion_eligible=false` fields by default.

Paper:

- Center the evidence-grade discipline: dual-axis eval, loader split,
  exact-negative scope, public-frontier forensics, and packet compiler design.
- Separate exact rows, public/external rows, diagnostics, and predictions in
  tables. Do not rank mixed evidence grades in one frontier table.
- State that CPU/CUDA ratios are currently proven for the inspected HNeRV
  cluster and require cross-family validation.

Compliance:

- Do not edit upstream scorer files for contest claims.
- `inflate.sh` must remain scorer-free and must not depend on external state,
  network access, hidden sidecars, or local paths.
- All score-affecting bytes must be charged inside the archive or fixed
  contest code and consumed by the scored runtime.
- Every dispatch needs a Level-2 lane claim before GPU/remote spend and a
  terminal claim row after success/failure/refusal.
- `scripts/pre_submission_compliance_check.py --contest-final --strict` is the
  upload-surface gate for judge-facing packets.

## 6. Keep Learning Without Signal Loss

No signal loss means every artifact gets a durable disposition:

- exact positives: promote only with full custody and adversarial review;
- exact negatives: retire measured configs with reactivation criteria;
- CPU anchors: public-axis calibration, not CUDA promotion;
- macOS/MPS/proxy rows: development priors, never rank/kill;
- byte proxies: packet-build TODOs, not dispatch candidates;
- runtime failures: blocker class plus next unblock action;
- unknown technique rows: catalog or explicit planning-only disposition;
- stale predictions: supersede rather than delete.

The working mental model should be a compiler:

`representation -> prediction -> quantization -> hyperprior -> arithmetic -> pack -> inflate -> exact eval`

Every stage must preserve typed contracts, old/new SHA boundaries, charged-byte
accounting, no-op proof, and runtime consumption. The true floor will move only
when this compiler emits deterministic byte-different packets and exact eval
turns them into evidence.

## Control Decisions

1. Use dual-axis evidence language everywhere from now on.
2. Treat DALI/PyAV loader drift and PoseNet/SegNet/Hydra precision drift as
   separate mechanisms until the 2x2 discriminator closes them.
3. Prefer packet closure work over naked byte optimization.
4. Route solver output through custody gates before ranking or dispatch.
5. Preserve scoped negatives and proxy positives with reactivation criteria.
6. Keep the long-horizon target as task-aware packet optimization, not a
   declared numeric floor.
