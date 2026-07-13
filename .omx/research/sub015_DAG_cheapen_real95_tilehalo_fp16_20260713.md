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
