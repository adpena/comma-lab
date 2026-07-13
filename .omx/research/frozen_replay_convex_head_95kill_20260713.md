---
title: "Round-2 95%-kill frozen-replay convex head on cached exact labels"
date_utc: "2026-07-13T17:05:25.719725Z"
lane_id: "lane_95kill_round2_frozen_replay_convex_head_20260713"
feed: "FEED-95kill-fleet-round1/round2-frozen-replay-convex-head"
research_only: true
verdict: "GO"
verdict_scope: "fixed three-checkpoint V9 n600 replay, 31-feature chart, spectral-scale convex ridge head, seed455, local macOS CPU"
authority: "macOS-CPU advisory numpy-fp32 training-gradient evidence; no evaluator-score authority"
pointer_delta: "NONE"
---

# Round-2 95%-kill replacement — frozen replay + cached exact labels + convex head

**ONE-LINE OUTCOME:** **`GO`** under the preregistered round-2 rule. On `600` unique real V9
states (`480` train / `120` heldout), the heldout aggregate costate cosine is **`0.0014157933865487525`**,
the renderer-gradient dot is **`0.1096160079189985`**, exact teacher calls amortize **`12.0x`**
(`0.08333333333333333` call per effective cached-state step), and the realized executed-fp32
contraction constant is **`gamma=0.3333333461703458`**. This is local training-gradient evidence,
not `d_seg`, `d_pose`, archive score, contest-CPU/CUDA, promotion, or live-activation authority.

## Verdict and literal decision readback

The operator's preregistered conjunction is satisfied:

| Gate | Comparator | `MEASURED` | Result |
|---|---:|---:|---|
| heldout costate cosine | `>= -0.16153190769629602` | `0.0014157933865487525` | **PASS**, margin `0.16294770108284477` |
| teacher-call amortization | `>= 5x` | `12.0x` | **PASS**, `2.4x` the required factor |

The old round-1 `0.0` policy overlay also passes, but it is diagnostic only and was not substituted
for the operator's literal early saved-regime comparator. The renderer-gradient diagnostics are
positive (`dot=0.1096160079189985`, cosine `0.017697414591996724`) but had no preregistered threshold.

This is an intentionally narrow **GO**. The costate relative L2 is `1.0000018705777456`, so the
result does not establish useful magnitude reconstruction or authorize replacing the teacher in a
live controller. It establishes exactly what round 2 asked: fixed-distribution directional fidelity
above the inherited early-regime bar plus at least fivefold inclusive teacher-call amortization, with
an explicit convex contraction certificate.

`verdict_scope`: `FORMULATION x INSTANCE` — three read-only V9 checkpoints at epochs `150`, `251`,
and `275`; one deterministic state for each of the `600` witness pairs; fixed seed-455 split;
31-channel RGB/geometry/source-label chart with the exact target-costate tensor absent; stride-8 train lattice; full-grid
heldout evaluation; spectral-scale ridge on every linear-head coordinate; 15 deterministic full-batch
NumPy-fp32 steps; CPU Torch exact teacher; local macOS arm64. It is not a family-wide verdict and it
does not transfer to on-policy replay, nonlinear learners, frozen-stem features, RFF lifts, another
seed/chart, contest axes, or evaluator score.

## Immutable evidence

- Run directory: `experiments/results/frozen_replay_convex_head_95kill_n600_20260713`
- Receipt: `experiments/results/frozen_replay_convex_head_95kill_n600_20260713/measurement_receipt.json`
- Receipt bytes: `136633`
- Receipt SHA-256: `067ce197d30fa9e2c7c4bda48ac671af550e0a00f126289ba5b30946d44fc4b1`
- Completion timestamp: `2026-07-13T17:05:25.719725Z`
- Completion-record SHA-256: `a33aa873b5b1c6e9c5dc625b8f1b30f79a0bc61d906f9d342fd724b5557c7cae`
- Teacher ledger: `600` starts + `600` completions + `600` unit-batch completions; SHA-256
  `14bfa8f4ca51f15ffbe4815cfd1af8109eca09cb2b0ea432a5c9fdc2e3efef16`;
  event-tree SHA-256 `0d5863a7ea6ba29bc0c917fe7de54fa7f034cfc7384a5623c963a57017bced7e`.
- Source amendment: `fit-ratio-scale-floor-v1`, SHA-256
  `bba2729d7fe73385b44af875004d16018ee36a51fb07fa87925740a5fb1beabb`.
- Storage cleanup: `COMPLETE_NO_BULK`, no scratch blocker and no raw costate/frame tree retained.
- Terminal resume: complete run tree SHA-256 remained
  `00777398fcf874a59cc06cc7e288d90b20f164a89170c23f51370e6a9b41769e`
  before and after `--resume`; no sacred byte changed. Machine-readable reproduction command,
  hash definition, `2451`-file count, and `7572854`-byte count are in
  `.omx/research/frozen_replay_convex_head_terminal_resume_20260713.json`.

### Real-n600 input custody

| Input | Bytes | SHA-256 | Use |
|---|---:|---|---|
| V9 epoch-150 EMA-best checkpoint | `379776` | `2599ad8b396af2af220a3bdbeee2ade92f194771ae6ef01a6faa15d39333484c` | 160 train / 40 heldout states |
| V9 epoch-251 Octave-1 checkpoint | `380136` | `c59cdec6eec16677c0a2eb5667979dd1c8f883bcd1cf5532302d67acd633c758` | 160 train / 40 heldout states |
| V9 epoch-275 EMA-final checkpoint | `380136` | `1676e4d45e180c7a28ec2ecce2b932d0e5087a2cfec2636ff2efe1673dbbcbf0` | 160 train / 40 heldout states |
| `gt_n600.npz` | `5078017610` | `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6` | read-only labels/margins, ZIP_STORED memmap |
| frozen CPU SegNet | `38502892` | `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6` | exact input-costate teacher |

No live run, paid provider, GPU, MPS, evaluator, archive, or submission surface was actuated. The
three source checkpoints and the 5.08-GB GT cache were read-only. The compact train cache stores
per-state `X'X`, `X'Y`, `Y'Y`, row count, and feature/target hashes—not a raw costate tensor tree.

## What was built

### Fixed replay and exact-label cache

Pair `i` is assigned to checkpoint `i mod 3`; heldout membership is the seed-455-fixed `i mod 5`
residue. This yields exactly `480/120`, with `160/40` per checkpoint. Membership and rendered states
never change during fitting, so the operator is fixed by construction. Every state gets one exact
batch-size-1 CPU SegNet input-costate call. Batch size 1 is `DERIVED`, not tuned: the committed
teacher uses mean CE across batch and pixels, so larger batches rescale the per-state target.

The fixed chart has `31` features: bias; RGB; RGB squared; finite-difference `dx`, `dy`, and
Laplacian channels; normalized coordinates and four Fourier coordinate channels; cached five-class
source labels; cached margin; and checkpoint one-hot. The chart never consumes the exact target
costate. It is sufficient for this research probe but is not archive-legal or live-policy authority;
future activation would need separate custody for every inference-time feature.

### Convex head and explicit norm

For fixed `X` and exact cached target `Y`, the only trainable object is a linear head `W`:

```text
F(W) = ||XW-Y||_F^2/(2n) + lambda ||W||_F^2/2
H    = X'X/n + lambda I
lambda := lambda_max(X'X/n)
mu   := lambda_min(H)
L    := lambda_max(H)
eta* := 2/(mu+L)
gamma_ideal := (L-mu)/(L+mu) <= 1/3
```

The norm is the Euclidean parameter norm, equivalently the Frobenius head norm. The realized
NumPy-fp32 `H` is sealed first. `eta` is then rounded to fp32 and the executed constant is re-derived
as `||I-eta_fp32 H_fp32||_2`; it is not mislabeled as the ideal `<=1/3` theorem.

| Certificate term | Label | Value |
|---|---|---:|
| data curvature max | `MEASURED-from-fp32-X` | `3.2247038856827617` |
| ridge `lambda` | `DERIVED` | `3.2247040271759033` |
| strong curvature `mu` | `DERIVED` | `3.2247038851557344` |
| smoothness `L` | `DERIVED` | `6.449407796557402` |
| ideal `eta*` | `DERIVED` | `0.20673732801540606` |
| executed fp32 `eta` | `MEASURED-from-realized-operator` | `0.20673732459545135` |
| ideal `gamma` | `DERIVED` | `0.33333333514200475` |
| executed fp32 `gamma` | `DERIVED-from-realized-operator` | `0.3333333461703458` |
| max admitted parameter ratio | `MEASURED` | `0.32923753849768017` |
| max admitted objective ratio | `MEASURED` | `0.10413857661064749` |

The observed ratios obey `0.32923753849768017 < gamma`; objective gaps obey
`0.10413857661064749 < gamma^2 = 0.1111111196691196`. Terminal residual certificates also close:

- parameter: `2.2703186949787158e-15 <= 2.601118716541341e-15`;
- prediction RMSE: `1.3920579326014151e-15 <= 4.670948667757252e-15`;
- objective gap: `0 <= 1.0908880726628738e-29`;
- terminal gradient norm: `8.38783763098216e-15`;
- per-state gradient variance: `7.21498595203425e-14` (reported separately, never used as the
  deterministic contraction theorem).

### Cached-label economics

For a cached state `s`,

```text
g_s(W) - g_s(V) = X_s'X_s (W-V),
```

so the exact label cancels from same-state gradient differences and `c_label=0`. With `A=600`
fresh unique-state labels and `D=480*15=7200` effective cached-state uses:

```text
C_teacher = A + c_label*D = 600 + 0*7200 = 600.
```

The one-call-per-effective-use baseline is `7200`, hence `600/7200=1/12` teacher call per effective
step, `12.0x` amortization, `6600` calls saved, and `91.66666666666666%` saving. The receipt includes
all cache-build and heldout-validation calls; batching does not discount state calls. There were
zero retries, pending calls, or resume-restore calls.

## Heldout fidelity — `MEASURED`, n600-real

| Metric | Aggregate | Mean per state / fraction |
|---|---:|---:|
| costate cosine | `0.0014157933865487525` | `0.001337415693389649` |
| costate dot | `6.739422408392844e-09` | positive state fraction `0.825` |
| costate relative L2 | `1.0000018705777456` | — |
| renderer-gradient cosine | `0.017697414591996724` | `0.021366418938148782` |
| renderer-gradient dot | `0.1096160079189985` | positive state fraction `0.6083333333333333` |
| compared elements | `70,778,880` costate / `2,280` renderer-gradient | fp64 reduction |

All `120` full-grid feature renders were bit-equal to the settled renderer (`different_elements=0`).
The positive direction is real under the fixed axis, but the near-zero cosine and unit relative L2
are an explicit warning against reading the preregistered GO as a strong surrogate-quality claim.

## Fail-closed falsifications and recovery

1. **Batch-4 attempt — `INVALIDATED_NO_EVIDENCE`.** After `172/480` train records, review found
   that mean-CE batch scaling made train costates one quarter of heldout batch-1 costates. The bytes
   are preserved under
   `experiments/results/frozen_replay_convex_head_95kill_n600_batch4_INVALID_20260713T160523Z` with
   an invalidation record. No metric from it is used.
2. **Absolute numeric-floor bug — verifier only.** The corrected batch-1 run sealed all `480`
   records and `1440` train events, then the fit verifier refused because its absolute floor
   `128*eps32=1.52587890625e-05` was `487.17x` larger than the entire initial parameter error
   `3.132128315313952e-08`. Scale-relative floor `4.779248528005908e-13` admits ten parameter
   ratios (`0.2250444486` through `0.3292375385`), all under `gamma`. Verdict scope for this negative
   is `IMPLEMENTATION INSTANCE`, not formulation.
3. **Append-only source amendment.** Recovery allowed exactly the scorer module and probe to change
   from their sealed old hashes, rehashed all `480` records and the exact `1440` train-event subset,
   re-derived both weight arrays, preserved the existing NPZ bytes/SHA, and issued zero new training
   teacher calls. Ordinary source drift still refuses. Forty-eight focused tests plus an independent
   disposable recovery exercise passed before resume.
4. **Post-measure warning hardening.** Torch left stale IEEE exception flags that made NumPy warn on
   finite heldout matrix products; an audit of all stored numeric fields remained finite. The final
   source suppresses stale flags only around the matmul and then explicitly refuses any non-finite
   prediction. A regression forces real overflow and proves the refusal. This hardening postdates the
   immutable measured source bundle and does not alter receipt bytes or metrics.

## Triality and system wire-in

- **Equation:** `src/tac/canonical_equations/frozen_replay_convex_head_contraction_20260713.py`
  now carries this empirical anchor, receipt hash, explicit executed-fp32 constant, residuals, and
  amortization law. Shared hot-registry population remains deferred to main review.
- **DAG:** `.omx/research/frozen_replay_convex_head_contraction_DAG_FEED_20260713.md` carries the
  preregistered edge, measured append, custody, fail-closed history, verdict scope, and pointer honesty.
- **DSL:** `src/tac/witness_dsl/frozen_replay_convex_head_policy.py` compiles the fixed split,
  derived 15 epochs, batch-size-1 custody, literal decision bar, and default-off research policy.
  `live_trainer_argv=[]`; no live lever was wired.
- **Sensitivity:** heldout exact/predicted costate and matched renderer-gradient reductions are the
  directional signal.
- **Pareto constraint:** inclusive teacher calls per effective cached-state use; no rate/score claim.
- **Bit allocator:** non-binding until evaluator-cell debt and archive bytes are measured.
- **Cathedral/autopilot:** no dispatch hook; research-only evidence must not actuate a run.
- **Continual learning:** the canonical equation anchor plus this memo/receipt prevent rediscovery;
  canonical probe outcome `frozen_replay_convex_head_v9_n600_seed455_20260713` records advisory
  `PROCEED` with the exact receipt hash; shared equation-registry/DAG integration is explicitly
  `DEFERRED_MAIN`.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `PROGRAM.md`, and the v7.5/v8
  canonical specs.
- `reports/latest.md`; lane registry; subagent-progress ledger; master gradient anchors; Modal ledger;
  cost-band and continual-learning posteriors; probe-outcome ledger; latest Codex summary/findings,
  council-T3, design, and recent directive surfaces.
- `.omx/research/onpolicy_surrogate_95kill_20260713.md` (`#455`),
  `.omx/research/vrghal_95kill_fixedpoint_20260713.md` (`#462`), and
  `.omx/research/tofupov_ranker_allocation_20260713.md` (`#463`).
- The committed task-455 `src/tac/scorer_surrogate/*` harness/verdict stack and
  `FEED-95kill-fleet-round1` scope addendum in the canonical `sub015_DAG`.
- The three sealed V9 checkpoint files, `gt_n600.npz`, frozen SegNet, all stage manifests, atomic
  event ledgers, source bundles, source amendment, cleanup record, and immutable terminal receipt.

## Pointer delta and main-review handoff

`pointer_delta=NONE`. This lane is left uncommitted as requested. Main review may inspect and then
decide whether to populate the shared equation registry / append the shared DAG. Any future live
interpretation must first replace or explicitly custody inference-time source-label features, measure
actual evaluator-cell debt rather than costate proxies, and obtain separate contest-CPU/CUDA evidence.
