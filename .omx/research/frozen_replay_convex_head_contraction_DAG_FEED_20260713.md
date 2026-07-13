# Standalone DAG FEED — Frozen-replay convex-head contraction

**Date:** 2026-07-13 UTC  
**Lane:** `lane_95kill_round2_frozen_replay_convex_head_20260713`  
**Node:** `FEED-95kill-fleet-round1/round2-frozen-replay-convex-head`  
**Status at preregistration:** `BUILD + LOCAL-MEASURE`, `research_only=true`  
**Measured verdict:** `GO` at `n600-real`, local macOS CPU training-gradient axis; `score_claim=false`  
**Shared-DAG append:** `DEFERRED_MAIN`; the canonical `sub015_DAG` is a live sibling surface.  
**Pointer delta:** `UNMOVED`

## Proactive recall — settled parents, not re-derived

- `#455` / `onpolicy_surrogate_95kill_20260713.md`: nonlinear on-policy formulation drifts by steps 2–3; negative scope is `INSTANCE-to-FORMULATION`, not family death.
- `#462` / `vrghal_95kill_fixedpoint_20260713.md`: admissible family remains open exactly for fixed replay, a convex head, explicit norm/curvature/variance/residual fidelity, and complete teacher-call custody.
- `#463` / `tofupov_ranker_allocation_20260713.md`: exact labels on frozen replay may dominate economically; any adaptive sampling interpretation separately owes frozen epochs, positive propensities, and IPW correction.
- `FEED-95kill-fleet-round1` scope addendum: queue items 1 and 2 compose and are the round-2 replacement arm.

## Preregistered edge

```text
three sealed, cold V9 trajectory checkpoints
  -> deterministic n600 unique-pair state assignment (one state per pair)
  -> fixed 480-train / 120-heldout split
  -> exact CPU SegNet input-costate call exactly once per cached state
  -> content-addressed objective-exact {X'X, X'Y, Y'Y} state cache
  -> fixed RGB/geometry/source-label chart with exact target-costate tensor absent
  -> convex linear ridge head in Euclidean/Frobenius geometry
  -> lambda := lambda_max(X'X/n)
  -> mu := lambda_min(H), L := lambda_max(H), eta := 2/(mu+L)
  -> ideal eta* := 2/(mu+L), gamma_ideal := (L-mu)/(L+mu) <= 1/3
  -> seal eta_fp32, then derive gamma_executed := ||I-eta_fp32 H_fp32||_2 < 1
  -> heldout exact-costate cosine + renderer-gradient dot/cosine
  -> C_teacher = A + c_label*D with c_label=0 for cached same-state differences
  -> preregistered {GO | scoped NO-GO + reformulation queue}
```

The fixed-operator hypothesis is structural here: neither replay membership nor `X` changes during fit. The `1/3` theorem is the ideal real-arithmetic bound on the spectrum of the realized fp32 Hessian. The executed fp32 step is rounded first and gets its own derived operator norm plus observed iterate ratios above the numeric floor; it is never silently called exactly `<=1/3`. Per-state gradient variance is reported separately and never substituted for the full-batch theorem.

## Frozen real-n600 custody

| Stage | Source checkpoint | Epoch | Bytes | SHA-256 | Assignment |
|---|---|---:|---:|---|---|
| early | `levelset_witness_ema_BEST.npz` | 150 | 379776 | `2599ad8b396af2af220a3bdbeee2ade92f194771ae6ef01a6faa15d39333484c` | `pair % 3 == 0` |
| middle | `levelset_ckpt_stageOctave1_ep251.npz` | 251 | 380136 | `c59cdec6eec16677c0a2eb5667979dd1c8f883bcd1cf5532302d67acd633c758` | `pair % 3 == 1` |
| late | `levelset_witness_ema_mlx.npz` | 275 | 380136 | `1676e4d45e180c7a28ec2ecce2b932d0e5087a2cfec2636ff2efe1673dbbcbf0` | `pair % 3 == 2` |

Exact GT labels/margins come from read-only `gt_n600.npz`, 5078017610 bytes, SHA-256 `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`. The exact teacher is frozen CPU SegNet SHA-256 `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`. No raw 1.4-GB costate tree is materialized: the train cache preserves the exact sufficient statistics for this registered objective and per-state target hashes; heldout evidence preserves reduced fp64 dot/norm custody.

## Decision rule — preregistered before measurement

`GO` iff both:

1. aggregate heldout costate cosine is at least the literal round-1 measured early saved-regime bar `-0.16153190769629602`; the earlier work never reached a full K20 regime, so no K20 measurement is claimed; and
2. inclusive exact-teacher state-call amortization is at least `5x` against one teacher state-call per effective cached-state training use.

The receipt must report effective cached-state uses and optimizer steps separately. Batching changes wall time only; it never reduces the counted teacher state calls. All cache-build, validation, retry, and crashed-attempt calls are included. Renderer-gradient dot/cosine is a required diagnostic but has no operator-specified GO threshold in this round.

## Constant/value provenance

| Value | Label | Provenance |
|---|---|---|
| `n=600` | `SOURCE` | operator evidence rule and sealed GT/checkpoint geometry |
| seed `455` | `SOURCE` | inherited committed task-455 lineage |
| 3 checkpoint stages | `DERIVED` | available cold trajectory stages with n600 code rows |
| 20% heldout residue (`pair % 5`) | `ASSUMED` | fixed before measurement; not tuned |
| training lattice stride `8` | `ASSUMED` | compute-bounded fixed chart; not tuned; narrows verdict scope |
| fit epochs `15` | `DERIVED` | `ceil(23 ln(2)/ln(3))`, enough ideal `gamma<=1/3` contractions to cross one binary32 fraction ulp |
| ridge, `mu`, `L`, ideal `eta*`/`gamma_ideal`, executed `eta_fp32`/`gamma_executed` | `DERIVED` | realized NumPy-fp32 covariance/Hessian plus explicit fp32 step rounding |
| teacher batch size `1` | `DERIVED` | committed exact teacher averages CE over batch; per-state label parity requires one state |
| cosine decision bar `-0.16153190769629602`, amortization floor `5x` | `SOURCE` | operator's literal round-1 early saved-regime comparator plus preregistration |

The old round-1 nonnegative (`0.0`) admission predicate is reported as a separate stricter diagnostic overlay. It is not silently substituted for the operator's round-2 early-regime decision bar.

## Negative verdict scope and reformulation queue

Any `NO-GO` is scoped to the fixed three-checkpoint V9 n600 replay, deterministic feature chart, spectral-scale ridge head, seed 455, and local macOS CPU measurement. It does not reject frozen replay, cached labels, convex heads, fixed SegNet features, input-convex networks, contest-CPU, or CUDA as families/axes.

Queue on `NO-GO`:

1. replace hand-built fixed features with frozen SegNet-stem features, retaining the convex output head and exact same cache/call law;
2. use a fixed multiscale random/Fourier lift with the same explicit Hessian certificate;
3. use class-block convex heads calibrated on heldout renderer-VJP direction.

## Triality and wire-in

- DAG: this collision-free standalone FEED; shared canonical append deferred for main review.
- Equation: `tac.canonical_equations.frozen_replay_convex_head_contraction_20260713`.
- DSL: `tac.witness_dsl.frozen_replay_convex_head_policy`; default-off, `live_trainer_argv=[]`.
- Sensitivity contribution: heldout exact/predicted costate and renderer-VJP dot/norm reductions.
- Pareto constraint: teacher state-calls per effective cached-state use; no archive-rate claim.
- Bit allocator: non-binding until exact evaluator-cell debt and bytes are measured.
- Cathedral dispatch: none; research-only local probe, no live/paid launch.
- Continual-learning update: measured receipt, dated memo, and advisory canonical probe-outcome row `frozen_replay_convex_head_v9_n600_seed455_20260713`; shared equation-registry append deferred.
- Probe disambiguator: this probe arbitrates manual fixed features now; frozen-stem and fixed-RFF interpretations remain explicit queued modes if falsified.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `PROGRAM.md`
- v7.5 and v8 canonical SPECs
- `reports/latest.md`, lane registry, subagent-progress ledger, gradient anchors, Modal ledger, cost-band and continual-learning posteriors, probe-outcome ledger
- latest Codex findings/session summary, council T3, design memos, and last-24h directives
- the three named proactive-recall memos and committed task-455 scorer-surrogate stack
- `FEED-95kill-fleet-round1` and its scope addendum in the canonical `sub015_DAG`

## Measurement append

### Immutable receipt and authority

- Receipt: `experiments/results/frozen_replay_convex_head_95kill_n600_20260713/measurement_receipt.json`
- Receipt bytes / SHA-256: `136633` / `067ce197d30fa9e2c7c4bda48ac671af550e0a00f126289ba5b30946d44fc4b1`
- Completion timestamp: `2026-07-13T17:05:25.719725Z`
- Axis: `[macOS-CPU advisory; numpy-fp32 training-gradient evidence; no score authority]`
- Replay: `600` unique real V9 pair states, exactly `480` train and `120` heldout; each checkpoint contributes `160/40`.
- Receiver parity: `120/120` heldout chart renders bit-equal to the settled renderer, zero differing elements.
- Pointer: `UNMOVED`; live activation, archive promotion, evaluator score, contest-CPU, CUDA, and MPS authority remain absent.

### Preregistered decision readback

| Quantity | Label | Measured value | Gate / result |
|---|---|---:|---|
| heldout aggregate costate cosine | `MEASURED` | `0.0014157933865487525` | `>= -0.16153190769629602`; **PASS** by `0.16294770108284477` |
| heldout costate dot | `MEASURED` | `6.739422408392844e-09` | diagnostic |
| heldout costate relative L2 | `MEASURED` | `1.0000018705777456` | diagnostic; no fidelity-magnitude claim |
| positive costate-dot state fraction | `MEASURED` | `0.825` | diagnostic |
| renderer-gradient dot | `MEASURED` | `0.1096160079189985` | required diagnostic; positive |
| renderer-gradient cosine | `MEASURED` | `0.017697414591996724` | required diagnostic |
| positive renderer-dot state fraction | `MEASURED` | `0.6083333333333333` | diagnostic |
| teacher-call amortization | `MEASURED` | `12.0x` | `>=5x`; **PASS** |
| old nonnegative policy overlay | `MEASURED` | cosine `>0` | diagnostic **PASS**, not the decision gate |

Both preregistered gates pass, so the verdict is **GO**. This is deliberately a weak-fidelity GO under the operator's literal early-regime comparator, not a claim that the head reconstructs costate magnitude: relative L2 remains approximately one. Verdict scope is exactly `FORMULATION x INSTANCE`: fixed three-checkpoint V9 n600 distribution, this 31-feature chart (exact target-costate tensor absent, source labels/margins present), spectral-scale ridge, full-batch 15-step fp32 solve, seed 455, local macOS CPU.

### Teacher-call law and custody

`MEASURED`: `A=600` unique exact labeled states (`480` cache-build + `120` heldout), `D=7200` effective cached-state uses, `c_label=0`, hence

`C_teacher = A + c_label*D = 600 + 0*7200 = 600`.

The naive one-call-per-effective-use baseline is `7200`; measured calls per effective use are `600/7200 = 1/12 = 0.08333333333333333`, or `12.0x` amortization and `6600` saved calls (`91.6667%`). The atomic ledger has exactly `1800` events: `600` starts, `600` completions, `600` unit-batch completions; zero pending/crashed attempts, zero restore calls, and exact assignment reconciliation `PASS`. Resume after the verifier repair repeated zero training teacher calls.

### Contraction and residual certificate

| Quantity | Label | Value |
|---|---|---:|
| `lambda` | `DERIVED` | `3.2247040271759033` |
| `mu` | `DERIVED` | `3.2247038851557344` |
| `L` | `DERIVED` | `6.449407796557402` |
| ideal `eta*` | `DERIVED` | `0.20673732801540606` |
| executed fp32 `eta` | `MEASURED-from-realized-operator` | `0.20673732459545135` |
| ideal `gamma` | `DERIVED` | `0.33333333514200475` |
| executed fp32 `gamma = ||I-eta H||_2` | `DERIVED-from-realized-fp32-operator` | `0.3333333461703458` |
| max admitted parameter ratio | `MEASURED` | `0.32923753849768017 < gamma` |
| max admitted objective-gap ratio | `MEASURED` | `0.10413857661064749 < gamma^2=0.1111111196691196` |
| terminal gradient norm | `MEASURED` | `8.38783763098216e-15` |
| parameter residual / bound | `MEASURED/DERIVED` | `2.2703186949787158e-15 <= 2.601118716541341e-15` |
| prediction RMSE residual / bound | `MEASURED/DERIVED` | `1.3920579326014151e-15 <= 4.670948667757252e-15` |
| per-state gradient variance | `MEASURED` | `7.21498595203425e-14` |

The ideal real-arithmetic theorem and the executed fp32 operator remain separate. The tiny negative measured data-curvature minimum (`-1.219715371858462e-08`) is fp32 eigensolver/accumulation drift; curvature and contraction use the realized ridge Hessian spectrum, not a guessed PSD clamp.

### Fail-closed repair history

1. A first batch-4 attempt was invalidated after `172/480` train records because the committed teacher averages CE over batch and therefore scaled each costate by `1/4`. It is preserved, never cited as evidence, at `experiments/results/frozen_replay_convex_head_95kill_n600_batch4_INVALID_20260713T160523Z` with verdict `INVALIDATED_NO_EVIDENCE`.
2. The corrected batch-1 run completed all `480` cache records, then the verifier falsely refused because an absolute ratio floor `128*eps32=1.52587890625e-05` exceeded the whole initial parameter error `3.132128315313952e-08` by `487.17x`. This was a verifier implementation bug, not a formulation negative. A scale-relative floor `4.779248528005908e-13` admits ten ratios, all below executed `gamma`.
3. Recovery used append-only amendment `fit-ratio-scale-floor-v1`, SHA-256 `bba2729d7fe73385b44af875004d16018ee36a51fb07fa87925740a5fb1beabb`. It allowed exactly the scorer module and probe source deltas, rehashed all 480 cache records and the exact 1440-event train boundary, re-derived the fitted arrays, preserved the existing NPZ bytes, and made zero new train teacher calls.
4. A terminal `--resume` re-open hashed all run files before and after to the identical tree SHA-256 `00777398fcf874a59cc06cc7e288d90b20f164a89170c23f51370e6a9b41769e`; sacred completed bytes were not mutated. Reproduction algorithm and file/byte counts are in `.omx/research/frozen_replay_convex_head_terminal_resume_20260713.json`.
5. Post-measure source hardening suppresses only stale Torch IEEE warning flags around prediction matmul and then explicitly refuses any actual non-finite output; a forced-overflow regression covers the guard. It postdates and does not rewrite the immutable measurement bundle.

### Outcome and next edge

The round-2 composed formulation stays open and clears its preregistered replacement gate. The NO-GO reformulation queue is therefore not activated (`[]`). Before any live lever or promotion interpretation, a separate gate still owes evaluator-cell debt movement, stability beyond this fixed replay/chart, and contest-axis evidence. The DSL remains default-off with no trainer argv; main review owns any shared DAG or canonical-registry append.
