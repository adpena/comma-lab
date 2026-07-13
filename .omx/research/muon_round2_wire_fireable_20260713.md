# Manifold-Muon round 2 — wired, resumable MCSD/SPEL finisher — 2026-07-13

**Verdict capsule:** `{exact tangent-dual FIREABLE N; MCSD/SPEL FIREABLE Y; resumable Y; requested MLX micro-A/B BLOCKED, checkpoint proxy non-positive (+15.13% treatment vs +88.50% control); AdamW defects 2 (1 fixed / 1 ticketed)}`

**Lane:** `muon_round2_wire`  
**Mode:** BUILD + local `$0` verification; no heavy or paid launch  
**Pointer delta:** `NONE` — no contest score, candidate archive, or frontier pointer moved  
**Maturity:** `L0` intentionally — the build is uncommitted for main review and has no Metal parity/real-archive empirical gate; `impl_complete` was not falsely marked  
**Primary verdict scope:** fireability/resume/DSL mechanics and checkpoint-bound NumPy-fp32 optimizer behavior; **not** through-R, `d_seg`, `d_pose`, rate, receiver survival, or `n600`

## 1. What ships

The exact Bernstein tangent-dual formulation does **not** ship this round. Its nested dual ascent would add an inner convergence loop and additional resumable dual state for every FiLM step. No iteration count or truncation error has been measured for this witness, so calling a hand-truncated inner solve “exact” would violate NO-FAKE.

The shipped treatment is the named **MCSD/SPEL fallback** from the round-1 ticket:

\[
W_0=Q_0H_0,\qquad H_0\ \text{frozen},\qquad
G_Q=G_WH_0^\top,
\]

\[
G_T=G_Q-Q\operatorname{sym}(Q^\top G_Q),\qquad
A=\operatorname{NS5}(G_T),\qquad
M^+=\mu M+(1-\mu)A,
\]

\[
Q^+=\operatorname{qf}_{+}(Q-\eta M^+),\qquad W^+=Q^+H_0.
\]

`qf_+` is reduced QR with deterministic positive diagonal. This is a single-loop tangent spectral approximation with QR retraction. It is **not algebraically identical** to the exact tangent-constrained LMO.

### Landed surfaces

- `src/tac/optimization/film_polar_chart_spel_mlx.py`: NumPy-fp32 reference, MLX twin, chart/retraction/tangent math, EMA folding, deterministic telemetry, and resume provider.
- `src/tac/canonical_equations/witness_film_polar_chart_spel_20260713.py`: canonical shipped law and authority boundary.
- `src/tac/witness_control/resume_registry.py`: additive direct-controller registration `film_polar_chart_spel`.
- `src/tac/witness_dsl/curriculum_dsl.py`: typed `FilmPolarChartSPELManifoldMuon`, `MuonAtCheckpointBoundary`, and `AdamWReferenceSemantics` factories.
- `experiments/train_levelset_witness_realized_through_R_mlx.py`: default-OFF treatment wiring, switch-boundary initialization, live/deploy folding, spike rollback, EMA, and fail-closed resume checks.
- `tools/probe_film_polar_chart_spel.py`: deterministic checkpoint-bound proxy and MLX availability receipt.
- `src/tac/tests/test_film_polar_chart_spel_mlx.py`: chart, tangent, deterministic split-resume, config drift, DSL, equation, and conditional MLX parity tests.
- `.omx/research/muon_round2_wire_fireable_DAG_FEED_20260713.md`: executable dependency and consumer feed.

## 2. Resumability is P0

The treatment registers only when armed. Its canonical state is additive and legacy-compatible:

- `Q` — live Stiefel coordinate;
- `H0` — frozen function-preserving polar magnitude;
- tangent momentum;
- `Q_ema` — deploy coordinate, folded as `Q_ema H0`;
- optimizer step;
- schema/method identifier and source-weight SHA-256.

At the Muon boundary, initialization is followed by the existing atomic stage checkpoint. Later stage and periodic checkpoints use the same canonical registry. Active-treatment resume fails closed if its state is absent, if config custody differs, if EMA/live semantics are ambiguous, or if restored `QH0` does not reconstruct the checkpoint's live FiLM weight within tolerance. A deliberate weights-only warm start may initialize a new chart; it is reported as a new treatment boundary, not silently treated as a stateful resume.

If warm-start momentum is armed, the outgoing AdamW first moment is pulled back as `m_W H0.T` and tangent-projected. This preserves the intended coordinate transition without claiming second-moment transport.

## 3. Local measurements and honest blocker

Source custody:

- newest V9 deploy checkpoint found: `experiments/results/v9_cgauge_432_coherent_arm_20260711/levelset_witness_ema_mlx.npz`;
- epoch `275` (`MEASURED` metadata);
- bytes `380136` (`MEASURED`);
- SHA-256 `1676e4d45e180c7a28ec2ecce2b932d0e5087a2cfec2636ff2efe1673dbbcbf0` (`MEASURED`);
- FiLM shape `768x19` (`MEASURED`).

### Boundary and state mechanics

| Quantity | Result | Label / scope |
|---|---:|---|
| `||QH0-W||F / ||W||F` | `8.231831571947623e-08` | `MEASURED`, NumPy-fp32 checkpoint boundary |
| direct unit projection relative movement | `0.8698047399520874` | `MEASURED`; confirms round-1 rejection of naive projection |
| `||Q.T Q-I||F` | `1.7445133428197232e-06` | `MEASURED`, NumPy-fp32 |
| `sigma_min(H0) / sigma_max(H0)` | `6.571900542517128 / 9.126911748215987` | `MEASURED` |
| split-resume trajectory | exact array equality | `MEASURED` by deterministic unit test |
| Muon aspect multiplier | `6.35775531391221` | `DERIVED` from `sqrt(768/19)` to match incumbent MLX Muon |

### Requested micro-A/B

The requested local MLX fine-tune did **not** execute. MLX failed before a training step with:

`[metal::load_device] No Metal device available. This typically occurs in headless, sandboxed, or virtualized macOS sessions where the GPU is not accessible.`

There is a second apparatus constraint: epoch 275 is before the V9 governed ladder's derived safe Muon boundary. An immediate epoch-276 finisher would be refused by the stagger validator. Neither the device gate nor the scientific boundary was bypassed.

The non-promotable constructed local regression produced a first signal only:

| Arm, 8 matched steps, seed 469 | Initial loss | Final loss | Ratio | `H` relative drift |
|---|---:|---:|---:|---:|
| ambient Muon control | `0.00014330267731565982` | `0.0002701219927985221` | `1.8849751997550694` | `0.009847401641309261` |
| polar SPEL treatment | same | `0.00016498226614203304` | `1.1512853020786105` | `0.0` |

**MEASURED / proxy verdict:** both arms worsened this constructed FiLM-map objective; treatment was less adverse and exactly froze `H0`, but it is still a **non-positive signal** (`+15.13%`, not an improvement).  
**VERDICT-SCOPE:** checkpoint-bound chart/reproducibility mechanics and a constructed local regression only. It is not a trainer fine-tune and has no evaluator-cell authority. The matched real MLX micro-A/B and holistic `n600` finishing-stage verdict remain owed.

## 4. AdamW optimality audit

### Defect 1 — bias-correction semantics (`FIXED`, severity HIGH for short/restarted stages)

**Source locations:** `.venv/lib/python3.13/site-packages/mlx/optimizers/optimizers.py:493-505,528-535,564-577`; `experiments/train_levelset_witness_realized_through_R_mlx.py:2351-2373,4386-4394`; `src/tac/optimization/muon_finisher_mlx.py:161-173,286-297`.

MLX `AdamW` inherits `Adam` with `bias_correction=False` by default. The incumbent trainer selected that behavior for `beta2=0.999`; reference AdamW applies both first- and second-moment correction. The discrepancy is most material at cold or reset boundaries—the exact regime used by a short finisher probe.

The fix is additive and default OFF to preserve legacy trajectories. Typed DSL lever `AdamWReferenceSemantics` sets the reference treatment; the setting is persisted and guarded on resume. It is threaded through the main AdamW, Muon fallback AdamW, and every stage/rung optimizer rebuild. The implementation selects correction by the actual `beta2`, not a hardcoded optimizer name.

The existing Wave-C self-protect was also hardened: it now AST-inspects every trainer `AdamW` construction carrying `adam_beta2`, rather than asserting a stale hard-coded site count, and it distinguishes an installed MLX package from an executable Metal device.

### Defect 2 — modular metric grouping (`TICKETED`, severity MEDIUM optimality gap)

**Source locations:** `src/tac/optimization/muon_finisher_mlx.py:153-157,286-297`; `src/tac/canonical_equations/witness_modular_norm_assignment_20260713.py:117-140,143-231`.

The fallback still uses one Euclidean AdamW group for code, pose, SDF/RGB heads, biases, and palette. Autodiff already supplies task pullback gradients, so this is not a gradient-correctness bug. The unresolved defect is that the optimizer does not encode the round-1 module-specific norms or calibrated inter-module sensitivities.

Ticket `adamw_module_metric_groups_round3` is `WIRING_NEEDED`: first measure evaluator-pullback sensitivities, then admit per-module metric/LR/decay groups. Its first disambiguator should include a **head-only margin-aware trust region** for `out_sdf`: the derived RMS-to-`linf` geometry supports the family, but no safe radius or margin threshold is yet measured. No guessed trust-region constant was added.

### Reference-conforming paths

- epsilon is outside the square root: `sqrt(v_hat)+eps`;
- decay is decoupled and scaled by the scheduled learning rate;
- MLX's predecay expression is algebraically equivalent to the Loshchilov-Hutter update and does not feed decay into the moments;
- no separate fused-kernel semantic discrepancy was found.

**Count:** `2 defects = 1 fixed + 1 ticketed`; three audited semantics conform.

## 5. Verification receipt

- final combined relevant suite: `141 passed, 2 skipped`; both skips carry the exact unavailable-Metal blocker (optimizer parity and real MLX AdamW numerics).
- `36 passed, 1 skipped` — focused finisher plus resume-registry suite; the sole skip is MLX parity because Metal is unavailable.
- `108 passed, 1 skipped` — full witness curriculum DSL suite plus finisher suite.
- `1 passed` — targeted witness autoconfig parser-default test.
- `6 passed, 1 skipped` — AdamW Wave-C numerical/source self-protect; the real-MLX numerical case is the same exact no-Metal skip.
- schedule-valid governed launcher dry-run (`epochs=727`, sealed Muon cap `726`): `200/200` flags recognized, DSL and schedule-provenance gates passed, no process spawned; artifacts are under `experiments/results/muon_round2_dryrun_treatment_schedulevalid_20260713/`.
- new module, canonical equation, test, and probe: `ruff clean`.
- changed Python sources: bytecode compile succeeds.
- existing MLX Muon suite cannot collect on this host because importing its test module initializes the unavailable Metal device; this is recorded as an environment blocker, not a passing test.

## 6. Triality and six-hook wire-in

- **DSL:** typed factories compile the treatment, matched control, and reference-AdamW option; no trainer argv was invented by hand.
- **DAG:** the companion FEED orders chart, optimizer state, registry, parity, governed A/B, and `n600` authority.
- **Equations:** `witness_film_polar_chart_spel_v1` refines the round-1 law to the exact shipped approximation; `witness_modular_norm_assignment_v1` remains authoritative for module roles.
- **Sensitivity map:** no scalar sensitivity is inferred from the proxy. The AdamW grouping ticket owns this missing calibration.
- **Pareto:** the holistic `n600` through-R gate remains binding.
- **Bit allocator:** non-binding now because the optimizer adds no decode payload; eventual emitted checkpoint/archive bytes still require exact accounting.
- **Cathedral/autopilot:** duty-to-measure and launcher compilation are wired; the governor and stage boundary remain authoritative.
- **Continual-learning posterior:** no score posterior update is admissible from a blocked/non-authority probe; the negative proxy and blocker are durable receipts.
- **Probe disambiguator:** matched control/treatment DSL factories preserve the only defensible comparison.

**INT8 sibling handoff:** Q, H0, tangent momentum, and Q-EMA remain fp32 optimizer/resume state and are not an archive payload. Only the folded deploy matrix `qf(Q_ema)H0` enters the existing quantize/dequantize path. The int8 lane should measure whether retraction-plus-fold changes `film.weight` quantization error versus ambient Muon; no precision-path change is made here.

## 7. STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`, `PROGRAM.md`;
- v7.5 and v8 canonical SPECs, including v7.5 §8;
- round-1 `muonh_manifold_muon_dig_20260713.md` and `witness_modular_norm_assignment_v1`;
- `reports/latest.md`, lane registry, subagent progress, canonical task/pointer surfaces, and current V9 launch/checkpoint artifacts;
- installed MLX `Adam`/`AdamW` source and the trainer's every AdamW construction/rebuild path;
- latest sister findings/session summaries, latest T3/design memos, current directives, and top Claude-memory entries.

## 8. Next authority gate

At the next real V9 finishing boundary on a Metal-capable local host: first establish one-step NumPy/MLX parity, then run the matched governed control/treatment fork with the same seed, data order, steps, resume custody, and preserved per-stage checkpoints. Only then harvest holistic `n600` through-R facets. Exact tangent-dual ascent remains a separate formulation ticket; the shipped treatment must continue to be named MCSD/SPEL.
