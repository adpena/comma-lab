---
schema: codex_findings_v1
lane_id: lane_ddm_j12_366_receiver_coordinate_custody_producers_20260725
authority_sha256: fc5e7e1d5ac00cefb4b1079464f226b431cdfd2b75c79c2212adf380d0802bcf
research_only: true
score_claim: false
pointer_moved: false
main_review_required: true
verdict: MEASURED_J12_REHOMED_PC1_NEGATIVE_CONDITIONAL_SMOKE_COMPLETE
verdict_scope: MEASURED_LOCAL_MACOS_CPU_FROZEN_SCORER_ADVISORY
---

# DDM J12 #366 receiver-coordinate custody producers — findings

## Disposition

J12 closes J11's four receiver-custody reopener obligations:

- full SHA-bound n600/batch32 Pose6 and Seg rank4-inner Jacobians exist for every sealed J10
  proposal ray;
- the PC1 active-zero adapter reemits the 138,813-byte source exactly;
- 16 singles and 8 composites were integer-realized, receiver-closed, and exact-priced from
  both named bases;
- negative realized composites triggered the required resumable four-step live/EMA smoke.

The exact result is mixed:

1. Every one-dimensional sealed proposal ray has rank-one Pose and Seg Jacobians, hence both
   null projectors are exactly zero. All 16 singles are active-zero with `delta_S=0`.
2. The source-preserving PC1 coordinate is strongly negative: `-2.761204260556886` from
   W_joint and `-3.5711431248357903` from W_seg.
3. Four live J10 steps regress the initial W_joint+PC1 endpoint by
   `+0.12759259096760986`; short-horizon EMA remains byte-identical (`0.0`).

This is a `[macOS-CPU frozen-scorer advisory]` local measurement. It is **not** a contest score,
promotion, reseal, `READY_TO_FIRE_UNDER_STANDING_GO`, or FIRE claim. Pointer
`0.1910828242 [contest-CPU]` remains **UNMOVED**.

## Lawful receiver-coordinate Jacobians

The preregistered central secant was tested first and failed closed: RG1 makes three negative
reflections differ from the requested inverse proposal. That attempt is preserved at:

`/Volumes/VertigoDataTier/pact/experiments/results/ddm_j12_366_receiver_coordinate_custody_producers_20260725T150405Z_REFUSED_CENTRAL_RG1_ASYMMETRY`.

J12 then used the only common lawful coordinate, each sealed boundary ray
`W(alpha)=W0+alpha*delta_p`, `alpha in [0,1]`, measured from exact source alpha 0 to sealed
proposal alpha 1.

| Proposal | Proposal SHA prefix | Pose Gram | Seg rank4-inner Gram | Pose/Seg rank | Nullity |
|---|---|---:|---:|---:|---:|
| `x+` | `679b096b` | 59.01277788759191 | 126039.39267437292 | 1 / 1 | 0 / 0 |
| `x-` | `e4103eec` | 178.13413920850263 | 771651.5078735556 | 1 / 1 | 0 / 0 |
| `y-` | `1e539c62` | 204.03566571388268 | 1216349.2417902579 | 1 / 1 | 0 / 0 |
| `local_exact_gradient` | `010ae7df` | 132.3679583206048 | 450801.6832600525 | 1 / 1 | 0 / 0 |

Pose arrays have shape `[600,6,1]`/float64. Seg arrays have shape
`[600,4,384,512,1]`/float32. Every batch32 chunk carries proposal/source/camera/NPZ hashes,
pair IDs, receiver-chain identity, and exact array shapes. The index SHA is
`803b11013a540288179ee387274c854e5a8858dc995757e59aa1478b56071dd5`.

Because each coordinate count is one and each exact Gram is positive, both null projectors
are `[0]`. This is not a claim that the broader actuator family has no null space; it is an
exact result on the four named scalar sealed rays.

## Source-preserving PC1 adapter

The landed adapter uses:

`parent + receive_pc1(parent, packet) - receive_pc1(parent, active_zero)`, followed by the
actual uint8 clip/receiver path.

Inactive or active-zero encoding returns the raw parent bytes without a wrapper:

- 138,813 bytes;
- SHA-256 `2a2c0367150f8c8c0953dfb5c1485e238bbc9995c37385e149e52ae22f506241`;
- exact parser reemit and receiver parseback.

The old ratio from a different PC1 home was not transferred. Fresh exact local descent from
this home is:

| Accepted step | Bytes | d_seg | d_pose | Exact joint delta_S |
|---:|---:|---:|---:|---:|
| 0 | 138813 | 0.06974277072482639 | 35.49982080959101 | 0 |
| 8 | 139693 | 0.06472418891059027 | 31.7612616254476 | -1.5209788653637777 |
| 16 | 139701 | 0.0627319590250651 | 28.15912801903941 | -2.761204260556886 |

## Exact two-base pricing

No fixed `R*` was used. Acceptance is exclusively the sign of exact realized joint
`delta_S`; operating-point pose marginals are `5/sqrt(10*d_pose)`:

| Base | Pose marginal | Singles | Single outcome | Composites | Unique composite delta_S |
|---|---:|---:|---|---:|---:|
| W_joint step50 live | 0.2653731159689291 | 8 | active-zero, 0 | 4 | -2.761204260556886 |
| W_seg | 0.13069274688421467 | 8 | active-zero, 0 | 4 | -3.5711431248357903 |

The four composites on each base share one endpoint because every proposal's pose-null Seg
component is zero. W_seg's unique endpoint is 138,919 bytes, SHA
`b8013857fae78b129799cb620a56df7f9aaed4937b52c97cea4c3c77cb555f97`,
d_seg `0.024523688422309026`, and d_pose `120.03465558155271`.

The historical raw W_seg x+ `-0.0004297730820253919` is retained only as pre-decomposition
signal and is explicitly non-authoritative for J12 projection. No objective-gate
contradictions occurred.

## Conditional live/EMA smoke

The runner executed exactly four governed J10-engine steps:

- one complete checkpoint per step, preserving theta, EMA, both Adam moments, step cursor,
  and full telemetry;
- exact J10 schedule, Adam/Q8/RG1 geometry, and local exact-gradient proposal source;
- no acceptance-rule change;
- PoseFinish remained closed because component-safe residual admission was not proven.

Training proxy loss fell from `25.45241928100586` to `23.06488037109375`, but proxy direction
has no acceptance authority. Exact n600 endpoints show:

| Shadow | Parent SHA prefix | Merged SHA prefix | Exact delta_S vs initial merged | Disposition |
|---|---|---|---:|---|
| live | `815547e5` | `d48f9ac7` | +0.12759259096760986 | regression |
| EMA | `2a2c0367` | `9d1e599f` | 0.0 | byte-identical replay |

Thus the smoke completes the obligation but adds no stronger candidate. The initial rehomed
PC1 negative remains the measured signal.

## Apparatus warning and reseal boundary

The PC1 warp emitted divide-by-zero, overflow, and invalid-value NumPy warnings in
`_warp_scorer_frame` matmul. The exact receiver still produced finite uint8 chunks and both
n600 endpoints completed deterministically. The warnings were not suppressed. MAIN should
review whether this surface needs explicit sanitization or fail-closed handling before any
promotion-grade use.

`tools/reseal_ddm_j7_366_ticket.py` cannot lawfully reseal this branch result: it has no J12
profile and does not recognize the current delegated-authority hash. J12 therefore records
`PREPARED_REVIEW_REQUIRED` with the merged archive SHA, J10 parent SHA, PC1 packet SHA,
scorer custody, memory preflight, and worst-geometry contract. MAIN must first review/land the
branch, then add or authorize a source-bound J12 reseal against merged-main SHAs and regenerate
the worst-geometry receipt. FIRE stays MAIN-only.

## Durable artifacts and storage hygiene

- Compact receipt:
  `.omx/research/ddm_j12_366_receiver_coordinate_custody_receipt_20260725.json`,
  SHA `71b4ce5932be07483892e4c6627106da34ce2d142a9c83f05ba526c9cf913b64`.
- Full result root: 5.3 GiB under
  `/Volumes/VertigoDataTier/pact/experiments/results/ddm_j12_366_receiver_coordinate_custody_producers_20260725T150405Z`.
- Full receipt SHA:
  `0135dca6ad1bdd4691041e66161b4ccfed71d2cce3258d86b23a0345e2591c6d`.
- Refused central-secant attempt was preserved with a reproducibility manifest.
- One interrupted 72-MiB temporary partial chunk was removed before the lawful rerun; it had
  no final receipt or unique measurement authority.

No source archive, live run, frontier pointer, paid provider, or contest-eval surface was
modified.

## Verification

- Three consecutive focused passes: `51 passed in 95.00s`, `51 passed in 96.55s`, and
  `51 passed in 95.90s`.
- Ruff lint and format checks pass on all changed/new Python surfaces.
- Python compilation, JSON parse, and `git diff --check` pass.
- Review tracker: 89 entities recorded across four files at `j12-clean-3`.
- Exact EMA rerun reproduced both the merged archive SHA and n600 endpoint byte/metric identity.
- `tools/codex_landing_review_gate.py status` reports no pending landing in this isolated
  worktree; terminal disposition remains a MAIN merge-boundary obligation.

## MAIN landing review checklist

MAIN must review:

1. the RG1-asymmetry refusal and lawful forward-ray reformulation;
2. full per-pair Jacobian custody and rank/null derivation;
3. exact active-zero byte identity of the PC1 adapter;
4. unchanged S-primary pricing, including the W_seg Seg regression admitted by stronger Pose;
5. the fact that eight accepted composite rows represent only two unique PC1 endpoints;
6. live smoke regression versus EMA identity;
7. unsuppressed PC1 warp numerical warnings;
8. the absence of a current J12 resealer profile, then perform the merged-main SHA/worst-
   geometry reseal before any READY/FIRE disposition.

## Triality, system intelligence, and stores consulted

- DSL: typed config, adapter/parser/receiver, exact rank helper, resumable producer, and tests.
- DAG: `FEED-603-j12`.
- Equations:
  `.omx/research/ddm_j12_receiver_coordinate_custody_canonical_equations_20260725.md`.
- Sensitivity map: full exact arrays and rank certificates are reusable foreign-keyed inputs.
- Pareto/bit allocator: exact two-base marginal/byte/Seg/Pose tables are available.
- Cathedral/autopilot: no dispatch or FIRE edge; review/reseal is the only route.
- Continual learning: scalar-ray decomposition is null; PC1 rehome is productive; short live
  co-optimization is adverse.
- Probe disambiguator: forward boundary-ray measurement is the measured resolution to central
  RG1 infeasibility.
- Stores consulted: delegated authority; CLAUDE.md; AGENTS.md; craft operating manual; top-10
  Claude memory; last-24-hour directives; canonical frontier/lane/ownership/probe/equation/
  cost/posterior stores; J11 findings/refusal; WS2/WS4/J10/PC1 receipts; rank-4 scorer
  recursion; #580; exact target cache and frozen scorer custody.
- Quarantine waiver: `HARVEST-SIGNAL-ONLY`; no quarantined bytes or weights were composed.
- Inbox directives: no newer task-window directive arrived; the last broadcast remained
  `2026-07-24T23:09:25Z`.
