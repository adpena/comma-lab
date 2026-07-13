# DAG FEED — Manifold-Muon round 2 fireability — 2026-07-13

**Lane:** `muon_round2_wire`  
**Authority:** BUILD + local `$0` measurement only; no paid/heavy launch  
**Verdict scope:** optimizer/DSL/resume mechanics and a checkpoint-bound NumPy-fp32 proxy; no through-R, `d_seg`, `d_pose`, archive, or `n600` score authority

## Executable dependency state

| Node | Dependency | State | Durable surface / exact next gate |
|---|---|---|---|
| `MM2-B0` | Function-preserving polar chart `W=QH0` | `COMPLETE` | `tac.optimization.film_polar_chart_spel_mlx.polar_chart_numpy`; checkpoint-bound relative reconstruction error `8.231831571947623e-08` (`MEASURED`) |
| `MM2-B1-exact` | Exact Bernstein tangent-dual ascent | `NOT_FIREABLE / VERDICT_SCOPE=formulation implementation` | Nested dual state and its convergence/stopping proof are not implemented. This is not a negative verdict on exact Manifold Muon. |
| `MM2-B1-spel` | MCSD/SPEL single-loop tangent spectral update | `FIREABLE` | `FilmPolarChartSPELState.step`; tangent projection, NS5 spectral LMO approximation, positive-diagonal QR retraction |
| `MM2-B2` | AdamW-to-tangent momentum transfer | `COMPLETE` | `warm_start_tangent_momentum`; pull back `m_W` by `H0.T`, then project to `T_Q St` |
| `MM2-B3` | Additive resume schema | `COMPLETE` | direct controller `film_polar_chart_spel`; persists `Q`, frozen `H0`, tangent momentum, `Q_ema`, step, method/schema/source custody; missing active state fails closed |
| `MM2-B4` | NumPy/MLX update parity | `BLOCKED_ON_HOST` | NumPy deterministic split-resume verified. MLX device execution refused with `[metal::load_device] No Metal device available`; parity remains owed on a Metal-capable local host. |
| `MM2-B5` | Typed DSL + trainer route | `COMPLETE` | `FilmPolarChartSPELManifoldMuon`, `MuonAtCheckpointBoundary`, `AdamWReferenceSemantics`; all default OFF; schedule-valid governed dry-run (`epochs=727`, Muon cap `726`) compiled 200/200 trainer flags and spawned no process |
| `MM2-B6` | Matched local MLX micro-A/B from newest V9 checkpoint | `BLOCKED` | Metal unavailable. Additionally, epoch 275 precedes the governed ladder's derived first safe Muon boundary; do not bypass the stagger validator. Execute at the next real finishing boundary. |
| `MM2-B7` | Checkpoint-bound proxy | `COMPLETE / NON_PROMOTABLE` | `.omx/research/muon_round2_local_micro_probe_20260713.json`; treatment `+15.1285%` loss vs control `+88.4975%` after 8 constructed-regression steps; both are adverse. |
| `MM2-B8` | Holistic evaluator verdict | `OWED` | matched seed/step treatment-control at a governed real finishing boundary, then `n600` through-R `d_seg`, per-class facets, `d_pose`, rate, and receiver survival |

## AdamW audit feed

| Node | Finding | State | Severity / disposition |
|---|---|---|---|
| `MM2-A1` | MLX AdamW defaults to `bias_correction=False`, while reference AdamW uses first- and second-moment bias correction | `FIXED_DEFAULT_OFF` | **HIGH for cold/restarted short stages;** typed `AdamWReferenceSemantics` applies the reference treatment consistently to main AdamW, Muon fallback, and optimizer rebuilds. Legacy checkpoints retain old behavior unless explicitly armed. |
| `MM2-A2` | A single Euclidean AdamW fallback group does not encode module-specific metric/sensitivity budgets for codes, pose, SDF/RGB heads, biases, and palette | `TICKETED` | **MEDIUM optimality gap, not a correctness bug.** Ticket `adamw_module_metric_groups_round3`: calibrate sensitivities before per-module groups; include an SDF-head-only margin-aware trust-region probe. |
| `MM2-A3` | Epsilon placement | `CONFORMING` | `sqrt(v_hat)+eps`, reference form |
| `MM2-A4` | Weight decay | `CONFORMING` | decoupled, learning-rate-scaled predecay, algebraically equivalent to AdamW |
| `MM2-A5` | Fused/predecay ordering | `CONFORMING` | the MLX expression does not couple decay into moments |

## Triality and consumer feed

- **DSL:** the three typed factories above are the only admitted argv construction surface.
- **DAG:** `MM2-B0..B8` encode the exact fireability, blocker, and measurement order.
- **Equations:** `witness_film_polar_chart_spel_v1` records the shipped approximation and explicitly excludes exact tangent-dual authority; round-1 `witness_modular_norm_assignment_v1` remains the module-metric law.
- **Sensitivity map:** no scalar sensitivity is inferred from the proxy. `adamw_module_metric_groups_round3` remains blocked on calibrated evaluator pullbacks.
- **Pareto constraint:** the `n600` holistic gate is binding; a local proxy may not promote the arm.
- **Bit allocator:** non-binding for this training-only optimizer because the folded `QH0` does not add archive payload; any eventual checkpoint/decoder byte delta must still be measured.
- **Autopilot:** launcher duty-to-measure is registered through the DSL lever; actual launch remains governed and boundary-gated.
- **Continual learning:** the blocked MLX result and non-positive proxy are durable receipts, not a promotion posterior update.
- **Probe disambiguator:** `MuonAtCheckpointBoundary` and `FilmPolarChartSPELManifoldMuon` define matched control/treatment argv without invented flags.

## Required next execution

On a Metal-capable local host, at the first governed V9 finishing-stage checkpoint:

1. run the NumPy/MLX one-step parity test;
2. fork the same read-only checkpoint, seed, data order, and exact step count;
3. compile control with `MuonAtCheckpointBoundary(start_epoch=E)` and treatment with `FilmPolarChartSPELManifoldMuon(start_epoch=E)`;
4. preserve every stage checkpoint and canonical resume state;
5. harvest the short optimizer telemetry, then run the owed `n600` holistic through-R comparison.

No node authorizes bypassing the ladder stagger, governor, resume registry, or exact scorer custody.
