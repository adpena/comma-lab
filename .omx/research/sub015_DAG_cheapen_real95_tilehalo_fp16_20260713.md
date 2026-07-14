# FEED-cheapen-real95-tilehalo-fp16-20260713 — measured-wall / tile-halo / precision DAG

`research_only=true` · `score_claim=false` · `pointer_moved=false` · `$0 LOCAL` · `n600 required`

## Executable dependency graph

```text
sacred v7.5.2 n600 run (launch + daemon log + EMA; READ ONLY)
  ├─> current-wall receipt
  │     ├─ aggregate ep wall from measured timestamps
  │     └─ component split BLOCKED (live profiler OFF + no Metal device)
  ├─> Lever A exactness receipt
  │     ├─ actual frozen B2 U-Net topology
  │     ├─ phase-aware skip-inclusive halo derivation
  │     ├─ 23 global squeeze-excite reductions
  │     └─ n600 boundary coverage
  │           └─ NO-GO: full-frame dependency, exact upper bound 1.00x
  └─> Lever B real-state precision probe
        ├─ fp32 / fp16 / bf16 weights+activations
        ├─ fwd+bwd timing
        ├─ exact n600 render-pixel cotangent fidelity
        └─ BLOCKED here: Metal device unavailable
              ↓
      measured-only disjoint Amdahl equation
              ↓
      joint A+B receipt required because scorer-forward work overlaps
              ↓
      REFUSE numeric composition until all upstream measurements exist
```

## Canonical task rows

| Node | State | Producer | Consumer | Hard gate / verdict scope |
|---|---|---|---|---|
| `real95_current_wall_split` | `BLOCKED_NOT_MEASURED` | `tools/probe_cheapen_real95_current_wall.py` | composition law | Current aggregate is DERIVED 295.352 s/epoch; scorer fwd/bwd/render/R/loss remain null. Stale 78/22 cannot fill them. |
| `real95_tile_halo_exact` | `NO_GO_SCOPED` | `tools/probe_tile_halo_exactness_n600.py` | waterfill selector | Finite input-crop tile-halo exact forward for frozen `tu-efficientnet_b2` at 384×512 only. Halo 685 + global SE closes to full frame. |
| `real95_tile_waterfill_approx` | `DEFAULT_OFF_REFORMULATION_QUEUE` | typed DSL proposal | future probe | Tier-1 staleness is training tolerance, not authority; n600 gradient + timing receipt owed. |
| `real95_mlx_precision_fp16_bf16` | `NO_VERDICT_BLOCKED` | `tools/probe_mlx_real_n600_precision.py` | precision selector | Environment only. Requires speed >=1.5x, global/min-pair cosine >=0.99, exact n600 quality coverage. |
| `real95_joint_ab` | `NOT_MEASURED` | future joint forward/grad probe | composition law | A and B overlap scorer forward; isolated factors may not be multiplied. |
| `real95_composition` | `REFUSED_INCOMPLETE_MEASUREMENTS` | `tools/probe_cheapen_real95_composition.py` | main review | Numeric total remains null until component and joint receipts are measured. |
| `real95_trainer_wiring` | `NOT_AUTHORIZED` | main integration | governed launcher | Both DSL levers remain `OFF_UNWIRED`; this lane owns no trainer edits. |

## Triality

- DSL: `src/tac/witness_dsl/tile_halo_mixed_precision_proposal.py`, default OFF, no argv emission.
- Canonical equation: `amdahl_measured_disjoint_wall_split_with_async_cpu_verdict_v2` in `src/tac/canonical_equations/amdahl_measured_wall_split_20260713.py`.
- DAG: this FEED. Shared DAG/ledger files were not appended because they are hot and this task is new-files-only.

## Receipt custody

| Receipt | Bytes | SHA-256 |
|---|---:|---|
| `current_wall_receipt.json` | 5,673 | `c9ec6b2d7154a69b98dddd5c8a6a47455187fcdd3c0f4ea6afbff28554ac3614` |
| `tile_halo_receipt.json` | 10,615 | `b9f264166fea40224966c1902065eebd3fb34949750f87d7fd020e963bb99465` |
| `mlx_precision_receipt.json` | 3,252 | `78982aa3223d9a327c326b9ceb95cb4231bcf2ba3ba4f49e576e55d1e10b2240` |
| `composition_receipt.json` | 2,441 | `8115f8292c975f2f97e5a50f6da25f8e86a7fdc9858be0ff2c3a0d682837f99c` |

All live run inputs were read only. No score authority, promotion authority, GPU/paid dispatch, training launch, or pointer movement was created.

## 6-hook wire-in

Sensitivity map: ACTIVE (margin × class-pair sensitivity) · Pareto: ACTIVE (exactness/cosine/wall gates) · bit allocator: SAME sensitivity object, no byte mutation here · cathedral/autopilot: REFUSE until GO receipts · continual learning: memo + receipts + this FEED · probe disambiguator: exact binary-tile control vs approximate waterfill and fp16 vs bf16.

## verdict_scope addendum (main, ladder conformance)
verdict_scope: FAMILY — exact spatial sparsity (any tile/crop/mask formulation) on THIS exact scorer
(EfficientNet-B2 UNet): the 23 global squeeze-excite reductions make every output pixel depend on every input
pixel — a PROOF, so all exact-tiling formulations fail, not just the halo one tested. NOT paradigm: spatial
sparsity remains open for (a) SE-free architectures — the surrogate (directive routed to replace_round2), and
(b) NON-exact low-cadence tiling (training-path tolerance, unmeasured). untested formulations / alternatives:
SE-free surrogate tiling · approximate (stale-SE-statistics) tiling at pre-registered cadence · fp16 Lever-B
(probe running locally on main's Metal, receipt pending).

---
## FEED-bregman-review + apparatus + paper-intake (2026-07-14, pointer 0.19108/0.18804 UNMOVED)

**Bregman review (arm bregman_v9_all_surfaces, reviewed_committed) — LOAD-BEARING geometric correction.**
The proposed NO-SOLVE dual metric `ρ=‖η1−η2‖₂=√(Δθᵀ H Δθ)` is FALSE for a general PD Hessian. Re-derived +
MEASURED (600/600 synthetic SPD states, false-equality err ~9e-13): raw dual-Euclidean `‖Δη‖₂²=Δθᵀ H² Δθ`
is a SQUARED-HESSIAN geometry, NOT the ordinary/Fisher-natural Hessian metric `Δθᵀ H Δθ = Δηᵀ H⁻¹ Δη`
(Crouzeix H_F H_F*=I gives the INVERSE-Hessian dual, not the identity). ⇒ the Fisher-natural cotangent dual
REQUIRES the typed H⁻¹ solve; a no-solve `‖Δη‖` shortcut would silently corrupt `argmax_native_vjp_fidelity_v1`
while preserving its name (a caught name-preserving FAKE). Routes to #500 (metric design), #501 (fake-audit
candidate — provenance arm owns), #504 (grounding). Also banked: KL batchmean bug fix (nonneg log(p/q)+q/p−1
estimator, live consumer patched) + 120× exact sigma-point reduction (600→5 Caratheodory). 7 .py HELD for the
live provenance owner (canonical_equations/witness_dsl). All synthetic numbers tagged local-CPU-fixture, NOT
through-R n600. eq candidates (held): local_hessian_dual_geometry / affine_legendre_gauge_covariance /
exp_family_sigma_point_kl / categorical_chernoff_bisector / curved_bregman_centroid_projection.

**Apparatus (less-fragile/verbose, operator):** serializer `--files` now `action=extend` — repeated flags
ACCUMULATE (silent-drop fragility that wasted 5 commits, fixed; single-flag unchanged, 1cb053f53e). codex_status
= high-signal digest (folds landing-gate disposition + NEEDS_REVIEW surfacing) + delegate de-conflict preflight
(refuse duplicate-live-label) + robust-liveness discipline (never pgrep-fl|head — 1-of-8 undercount). Codex
landing review gate now enforced (Stop hook) — every arm dispositioned before trust.

**Paper intake dispositions (warm-start-from-divergence):** 2607.11883 Requential Coding → organ n=1
self-generated-data + MDL parent of margin-conditional flip coder (#226/#307); 2607.09197 routing-meaningfulness
→ HSE diagnostic for organ #436 regime-dispatch (vacuous-vs-real); 2607.10109 AMC saliency compression → margin-
saliency bit/rank tiering RATE lever (Fable arm live, macOS=all-surfaces); 2607.10765 Lee-Yang spectral-gap =
COOL-BUT-DISTANT (quantum; one thread = size-independent-gap↔anneal-rate floor #318/log-Sobolev, no arm).

---
## FEED-fable-AMC-saliency (2026-07-14, pointer 0.19108/0.18804 UNMOVED, advisory axis)

**Fable arm (Apple adaptive-compression warm-start, macOS=all-surfaces; committed 7c99b52b75) — a BANKED
advisory RATE lever + an honest seed-proxy NEGATIVE.**
STRUCTURAL KEY (verified in render path): pair i's d_seg depends ONLY on its own frame_1 code row ⇒ per-row
QDQ composition is EXACTLY ADDITIVE ⇒ escapes the 07-13 cross-tensor joint-REJECT by construction; per-row QDQ
on the tensor-global absmax grid needs ZERO receiver changes + ZERO side info.
MEASURED (real brotli, exact grammar; d_seg = DERIVED-exact recombination of measured per-pair rung rows,
pre-registered): role arms −5,704/−8,461 B at unchanged d_seg. **pairkkt (exact per-pair KKT from measured
responses): d_seg 0.031523 BELOW the int8 baseline at 52,981 B = −10,683 B** — encoder-side per-video search,
receiver unchanged, PR101-sidecar lineage. NEGATIVE: naive AMC error-mass saliency tiers DOMINATED (d_seg
0.034019, worse than its own random falsifier 0.033926 AND uniform int4 0.033804) → the seed's saliency proxy
is INSTANCE-level falsified (NOT a family NO-GO). Tiering-changes-allocation proven (230 B + sha delta for
same tier sizes, different assignment). Custody: baseline/uniform re-derived byte-identical 6/6 sha vs 07-13
n600 artifact. VERDICT SCOPE: INSTANCE×FORMULATION. AXIS: [macOS-CPU; NumPy-fp32; CPU frozen scorers],
score_claim=false — NOT contest-CPU.
OWED (operator-GO): fresh joint n600 scoring (d_pose NOT recombinable + fresh d_seg confirmation) was REFUSED
TWICE by the governed admission gate (+50 GiB phantom growth charged to 2 UNREGISTERED sibling arms
click_polish_block_loop / probe_genuine_frame_nterm_n600; no override — operator-verbatim required). Resumable
~50 min command in .omx/research/fable_amc_saliency_codex.md §6. PAYS ONLY at a competitive witness checkpoint
(witness still S≈4+ advisory; allocation re-solves per checkpoint from its own response rows).
HELD for provenance owner: canonical eq amc_perrow_tiered_code_bitalloc_v1 (law: pair-local rows ⇒ additive ⇒
measured-response allocation dominates proxy-saliency tiers); DSL TieredCodeQATLever spec (train-time, born-through-DSL).
ROUTES: #406 rate/pose apply-pass + #336 bit-alloc (banked); #501 fake-audit (the seed-proxy negative).
APPARATUS FLAG: 2 unregistered heavy sibling arms are blowing +50GiB phantom RSS + refusing legit launches —
governor contained (refused, no override); operator-decide register-or-stop.
