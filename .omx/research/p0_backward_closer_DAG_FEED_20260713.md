# DAG FEED — P0 backward closer — 2026-07-13

`research_only=true` · `score_claim=false` · `pointer_moved=false` · `$0 cached/local` ·
`training_performed=false` · `live_run_mutated=false` · `shared_DAG_append=DEFERRED_MAIN`

Lane: `lane_p0_backward_closer_20260713`  
Primary equation: `exact_costate_reuse_k2_guarded_v1`  
Sparse equation: `sparse_adjoint_mask_error_and_se_support_closure_v1`

## Terminal state

```text
guarded K2 exact-costate reuse     PENDING_SEALED_N600
masked/fixed-low-rank adjoint      PRIMITIVE_LANDED_DEFAULT_OFF; accuracy NO_GO;
                                    Metal wall BLOCKED; measured current factor 1.0x
#396 exact-metric terminal skip    LANDED_ROUTE_LOCAL (bulk factor unquantified)
whole-epoch wall clock             UNKNOWN_OPERATOR_GO_TIMER_OWED
```

## Executable dependency graph

```text
immutable V9/task455 cached n600 states + frozen SegNet/PoseNet + exact objective
  |
  +-> K2 event-controlled reuse probe
  |     -> exact anchor costate + durable payload identity
  |     -> one changed-frame stale attempt
  |     -> exact forward-only CE/d_seg/d_pose guard
  |          fail -> byte-exact rollback + full exact refresh
  |          pass -> accepted-row renderer-gradient relL2<1 + d_seg regret<=0
  |     -> 600 unique self-hashed pair records
  |     -> 3 x 200 byte-bound stage manifests
  |     -> trusted receipt + strict p>0.4344985574 gate
  |     -> PENDING_SEALED_N600_VERDICT
  |
  +-> sparse / low-rank flagship
  |     -> 600/600 regenerated costate hash matches
  |     -> no exact zeros + SE/global support
  |     -> 4.7366% masked VJP error 0.363536 oracle / 0.793434 source-margin
  |     -> r95=68, r99=100, rank64 error=0.238192
  |     -> custom ideal ceiling 2.208577x BUT dense realized 1.0x
  |     -> compact Conv2d VJP child: CPU parity 40/40
  |          -> DERIVED nominal ceiling 2.208615x / exact-tap 2.200230x
  |          -> Metal parity+wall BLOCKED; current-mask predictor gap 0.513967
  |     -> PRIMITIVE LANDED DEFAULT-OFF; NO_GO_DENSE_FULLRANK admission
  |     -> fallback full dense VJP
  |
  +-> terminal #396 exact-metric route
        -> STOP-sealed candidate + 600 stored base Seg/Pose cells
        -> six pair-local candidate cells; nonlinear n600 score recomposition
        -> measured (1+1)-ES Delta S=-8.229284904293088e-6 at 64 calls
        -> costate calls inside route = 0
        -> fixed-SHA handoff + default-off DSL policy
        -> LANDED route-local skip
        -> SPSA/ES gradient estimator REFUSED without deterministic r_eff<=2

admitted quantified composition
  = guarded K2 PENDING
    x sparse realized 1.0x
    x no numeric terminal bulk factor
  -> PENDING_K2_ONLY exact-call reduction
  -> whole-epoch timing edge remains OPERATOR-GO REQUIRED
```

No edge authorizes heavy/paid training, live trainer mutation, contest score, or pointer movement.

## Verdict scopes

- K2: direct raw zero-order-hold of one exact input costate, one changed-frame attempt, Kmax=2,
  event/stage/custody refresh, exact full-facet forward guard. Fixed cadence and K>2 remain refused.
- Sparse: frozen task455 EfficientNet-B2 U-Net, 4.7366% output masks, and fixed high-fidelity cohort
  bases. Pre-SE/local providers, larger learned current masks with custom kernels, and replacement
  providers remain open.
- Terminal: post-training #396 exact-metric accept/reject over the pinned n600-composed objective.
  It does not admit bulk SPSA/ES, a training-loop factor, contest timing, or a fresh full-n600 scorer
  replay claim.

## Triality and six-hook wire-in

- **DSL:** named K2 and terminal factories; argv-inert/default-off until reviewed integration.
- **DAG:** this standalone FEED; shared append deferred on the hot surface.
- **Equations:** guarded K2/terminal law plus sparse support/rank closure.
- **Sensitivity/Pareto:** fidelity regret versus calls saved; mask/rank error versus ideal FLOPs;
  terminal exact objective progress versus function calls.
- **Bit allocator:** no new byte actuator; #396 uses its exact repack/re-score gate.
- **Cathedral/autopilot:** trusted receipt only; every refusal routes to full teacher or ordinary #396.
- **Continual learning:** synthesis, receipts, tests, equations, FEED, and GO packet.
- **Probe disambiguators:** exact/stale matched-step K2; oracle/source-margin sparse masks; exact-metric
  #396 versus true SPSA/ES.

## Custody and pointer

The K2 replay writes only small atomic pair/stage metadata and is resumable from verified stage
boundaries. Sparse scratch was certified before cleanup by its sibling landing. Terminal source
bytes and the STOP-sealed candidate remain local durable evidence bound by the committed handoff
manifest. No operator-facing authority path cites `/tmp`. Pointer delta is exactly `NONE`.

---

## 2026-07-14 CORRECTION / SUPERSESSION — sealed K2 DAG state

The entire 2026-07-13 body above is **HISTORICAL_PROVENANCE** and remains byte-for-byte intact.
Its `PENDING_K2_ONLY`, placeholder terms, and “shared append deferred” statements were true when
written. The shared DAG feed is **NOW LANDED** as `FEED-p0-backward-closer-20260713`; this block is
the current DAG authority.

**Actual verdict:** `K2_CORRECTED_NOT_ADMITTED_DEFAULT_OFF`. **verdict_scope=FORMULATION:** bounded
direct raw-input-costate zero-order-hold K2, strict all-accepted stale-minus-exact `d_seg<=0`, three
V9 checkpoint states (`v9_ep150_ema_best`, `v9_ep251_stage_octave1`, `v9_ep275_ema_final`; 200 rows
each), and `[macOS-CPU advisory; Torch/NumPy-fp32 training-gradient MEANS only]`. This formulation is
NO-GO/default-off; the family remains intact under a new preregistered guard/tolerance.

```text
sealed 600 rows
  -> eligible 523
     -> behavioral full-facet accept 456 (p=0.76)
     -> actual guard fallback 67
  -> terminal/blocked 77
  -> total nonaccept 144 = 67 + 77 (q=0.24)
  -> accepted renderer relL2: 456/456 < 1
  -> accepted stale-minus-exact d_seg<=0: 308/456
  -> corrected gate false -> NOT_ADMITTED
```

The wrapper file SHA is `2102912bc8bd9711f00869746414fb21ea723729bcd26e612274547c6ca73d59`
and its content SHA is `4f7f2a6ef95f9989b734cd6e785d8b55dca7a77d7d9f03c693ff18287dea6e6e`.
Full source custody is indexed by the compact tracked summary
`.omx/research/p0_costate_reuse_k2_corrected_adjudication_receipt_20260714.json`; the full wrapper
and rows are absent from git, so clean-clone revalidation requires certified restoration. The cold-
store destination is not recorded and remains an explicit custody gap.

The corrected diagnostic law is `baseline=2.0` and
`guarded_cost=2+alpha-p=1.4184755862999998`. The `1.6129032258064517x` exact-backward factor and
`0.38` reduction are **DERIVED COUNTERFACTUALS BEHIND THE FAILED GATE**. Current admitted factors
are K2 `1.0x` and sparse `1.0x`; terminal #396 has no bulk multiplier; admitted bulk teacher-cost
reduction is `0%`; whole-epoch wall-clock is `UNKNOWN`.

Commit `e59f69a79cb2d974ec29fcaf75c6c855bd782a7a` plus
`.omx/research/frozen_scorer_verdict_wallclock_n96_20260714.json` gives **MEASURED** n96
`0.621 s/pair` on `[macOS-CPU-torch 1-thread advisory wall-clock] NON-PROMOTABLE`, with SegNet/PoseNet
forward shares `0.774/0.226`; `6.21 min/n600` is **DERIVED** by linear extrapolation. It does not
alter the K2 measurements, admit a whole-epoch benefit, or authorize activation. The timer packet
is `FIDELITY_BLOCKED_FUTURE_TEMPLATE`, diagnostic-only after a newly admitted formulation/provider.
