# Codex Findings - Optimizer, MLX, And Master-Gradient Integration Sweep

UTC: 2026-05-22T21:33:06Z
Agent: Codex
Review kind: adversarial integration sweep
Scope: PR95 HNeRV/Muon local training, PR95 variants, new HNeRV variants, broader NeRV-family and non-NeRV representation training probes, optimizer-guided queues, MLX dynamic sweeps, master-gradient/X-ray/Pareto/atom/cathedral integration, and exact-auth gates.

## Executive Verdict

The next engineering layer is not a new contest runtime. It is a canonical
training-and-planning signal carrier that lets local PR95/HNeRV optimizer
smokes, other PR95 variants, new HNeRV variants, broader NeRV-family models,
non-NeRV learned substrates, non-neural representation probes, MLX/local sweep
plans, master-gradient features, and exact-eval custody gates converge through
one queue surface.

Landed in this pass:

- `src/tac/optimization/optimizer_training_signal_bridge.py` creates a typed,
  false-authority solver-stack wire-in payload for representation-training
  rows. It now carries explicit `representation_family`, `substrate_family`,
  `training_signal_kind`, `variant_axes`, and paired probe modes so PR95 is one
  profile, not the abstraction.
- `src/tac/optimization/optimizer_guided_candidate_generation.py` now includes
  a `pr95_hnerv_muon_training_smoke` profile and emits master-gradient, X-ray,
  canonical-equation, atom, deterministic-solution, Pareto, bit allocator,
  cathedral autopilot, continual-learning, and probe-disambiguator metadata.
- `src/tac/optimization/representation_training_probe_integration.py` accepts
  `representation_training_probe_manifest_v1` for any representation family,
  including PR95 variants, new HNeRV, NeRV-family, non-NeRV learned codecs, and
  non-neural signal-processing/substrate probes.
- `src/tac/optimization/pr95_muon_local_training_integration.py` adapts PR95
  local-training manifests into optimizer candidate queue rows while stamping
  HNeRV/NeRV-family metadata.
- `src/tac/optimizer/candidate_queue.py` now consumes both PR95 training
  manifests, generic representation-training manifests, and
  `mlx_dynamic_learned_sweep_plan.v1` rows as planning-only queue inputs.
- `tools/cathedral_autopilot_autonomous_loop.py` now rejects
  `consumer_payload.rank_or_kill_eligible=true` and
  `consumer_payload.promotable=true`, closing an authority-leak gap.

All new rows remain non-promotional:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- exact CPU/CUDA auth eval still required for score authority

## Layer Sweep

### Layer 0 - Authority And Git Custody

Status: partially hardened; continue.

Already present:

- Proxy false-authority contract in `tac.optimization.proxy_candidate_contract`.
- Exact-readiness promotion gate in `tools/promote_optimizer_candidate_for_exact_eval.py`.
- Queue adapters force planning-only rows through `apply_proxy_evidence_boundary`.

Remaining work:

1. Add a central "consumer payload authority" helper so every cathedral,
   candidate-queue, and observer payload rejects the same authority fields.
   Current hardening landed for the main cathedral payload path; other loaders
   still have local field loops.
2. Add a preflight scan for `consumer_payload` authority leaks across JSON
   artifact producers.
3. Register a retroactive sweep once that gate lands, because any previous
   `consumer_payload.promotable` or `rank_or_kill_eligible` leak could have
   tainted autopilot prioritization.

### Layer 1 - Local Queue And DQS1 Advisory Cascade

Status: active and useful.

Current queue after rank016 local advisory:

- rank016 local macOS CPU advisory: `0.19204028295713674`
- calibrated projected contest CPU: `0.19203028295713673`
- conservative projected contest CPU: `0.19203328295713673`
- eureka trigger: `false`
- action: observe only
- queue now points at rank025 / pair0026 for the next local-first step

Remaining work:

1. Continue bounded local-first rank sweep: rank025 -> next learned-reroute target -> group/null
   remove candidates until calibrated eureka flips or marginal curve saturates.
2. Make the queue builder record the reason each prior candidate was skipped
   in a committed research summary, not only in JSON output.
3. Add a small `tools/harvest_dqs1_local_first_result.py` helper that reads the
   local advisory JSON, writes the drift/eureka JSON, updates latest report
   surfaces, and optionally rebuilds the next queue target.

### Layer 2 - MLX Scorer And Dynamic Learned Sweep

Status: proxy useful; not authority.

Already present:

- MLX scorer rows must pass exact auth-axis custody gates before calibration.
- MLX dynamic learned sweep plans carry recursive pass rows and observation
  feedback.
- This pass wires `mlx_dynamic_learned_sweep_plan.v1` into the optimizer
  candidate queue.

Remaining work:

1. Implement batch-invariance trace: compare local CPU, MLX, and contest CPU
   by batch size and component deltas, not just final score.
2. Add a local "same archive, same inflated raw aggregate" verifier for every
   MLX calibration target.
3. Teach the learned sweep to choose follow-up region types: pair drop, group
   drop, null remove, frame subset, byte/pixel atom, optimizer config, or
   substrate-training smoke.
4. Add a queue-backed SQLite store for long-running learned sweeps with pause,
   freeze, rewind, concurrency, and observation telemetry.

### Layer 3 - Master Gradient, X-Ray, Atoms, And Canonical Equations

Status: hook surfaces exist; row-level integration now started.

Already present:

- `tac.master_gradient_consumers`
- `tools/master_gradient_xray.py`
- `tac.xray.*`
- `tac.atom.*`
- `tac.canonical_equations.*`
- `tac.unified_action`

This pass:

- Optimizer training rows now reference master-gradient features:
  `pairset_component_marginal`, hard pairs, SegNet/PoseNet dominance, per-pair
  training weights, and optimizer recipe component marginal.
- Rows declare X-ray primitives and canonical equation refs instead of burying
  them in prose.
- Generic representation-training rows carry the same wire-in surfaces, so a
  SIREN, WIRE, Cool-Chic/C3, Ballé/CompressAI, wavelet, null-codebook,
  arithmetic-coder, deterministic compiler, or future substrate probe can
  enter the same queue without PR95/HNeRV naming leakage.

Remaining work:

1. Convert the optimizer signal wire-in payload into actual Atom rows via
   `tac.atom.builders.build_meta_lagrangian_atom`.
2. Add a master-gradient feature extractor for training probes:
   common seed, per-pair sample weights, hard-pair hit rate, SegNet/PoseNet
   axis loss contribution, and export-time archive identity.
3. Add a canonical equation only after empirical anchors exist. Do not invent
   a Muon/HNeRV training-efficiency equation from literature alone.
4. Extend X-ray with a local-training portability primitive only after generic
   representation-training manifests carry enough stable telemetry across at
   least one HNeRV-family and one non-NeRV-family probe.

### Layer 4 - PR95 HNeRV/Muon Optimizer Program

Status: queueable planning layer landed; trainer flag integration remains.

Already present:

- PR95 source-faithful local probe harness.
- `tac.optimization.muon` with PR95-aware parameter partitioning.
- New `pr95_hnerv_muon_training_smoke` optimizer queue profile.
- New PR95 manifest adapter into the optimizer candidate queue.

Immediate engineering order:

1. Add `src/tac/training/orthogonalized_optimizer_registry.py`:
   `adamw`, `torch_muon`, local `MuonOptimizer`, `mars`, `mars_m`,
   `polar_express`, `normuon` as training-only adapters with explicit
   dependency/license metadata.
2. Add `src/tac/training/schedule_registry.py`:
   cosine, WSD, linear/power decay, schedule-free wrapper if dependency is
   reviewed, SWA/Polyak tail averaging.
3. Add `tools/plan_hnerv_optimizer_schedule_sweep.py` to emit PR95/HNeRV
   optimizer schedule plans through the same queue path.
4. Modify `tools/run_pr95_local_training_probe.py` to accept optimizer and
   scheduler recipe IDs, persist optimizer config SHA, parameter partition
   digest, update-RMS telemetry, and memory estimates.
5. Only after paired smokes: export archive/runtime and use exact-readiness
   promotion gate.

### Layer 4B - Other Representation Families

Status: generic queue ingress landed; family-specific builders remain.

The generic `representation_training_probe_manifest_v1` adapter is the entry
point for:

- PR95 variants that are not exactly the current HNeRV/Muon local probe.
- New HNeRV/HNeRV++/HNeRV-with-codec-head variants.
- NeRV-family variants: NeRV, E-NeRV, FFNeRV, CNeRV, SIREN/FINER/WIRE/BACON
  adjacent coordinate networks.
- Non-NeRV learned codecs: CompressAI/Ballé, Cool-Chic/C3, VQ/VAE, frame/pair
  segmentation or pose-conditioned learned residuals.
- Non-neural or hybrid substrates: wavelet residuals, arithmetic/range/ANS
  coder passes, null/procedural codebooks, deterministic compiler candidates.

Immediate engineering order:

1. Add manifest emitters to each family-specific smoke/probe so all of them
   produce `representation_training_probe_manifest_v1`.
2. Keep family-specific recipe details in `training_recipe`,
   `optimizer_recipe`, `scheduler_recipe`, and `candidate_params`; do not
   fork the queue schema per family.
3. Add byte-closed export adapters only after paired local smokes produce a
   measurable positive proxy and stable archive/runtime custody.
4. Add exact-readiness promotion from the generic queue row, not from the
   family-specific smoke output.

Do not:

- put optimizer dependencies into inflate runtime;
- use Muon on 1-D/bias/norm/stem/RGB/head params;
- promote local training score or MLX/CPU advisory score;
- exact-eval optimizer variants before byte-closed archive/runtime custody.

### Layer 5 - Sequence Models And Recurrent Planners

Status: research signal only.

Current research summary:

- Mamba-3, Gated DeltaNet-2, KDA/Kimi Linear, Preconditioned DeltaNet, and
  related linear-attention/recurrent methods are relevant to planner memory,
  not contest inflate runtime at this stage.
- Their best use is as learned acquisition over pair/group/null/frame rows and
  recursive sweep observations.

Engineering order:

1. Start with tiny deterministic recurrent features over existing observation
   ledgers.
2. Compare against the current Bayesian/acquisition planner.
3. Keep all sequence models training/planner-only unless an export-first byte
   grammar proves runtime value.

## Sources Used By Subagents

- Muon: https://kellerjordan.github.io/posts/muon/
- PyTorch Muon API: https://docs.pytorch.org/docs/2.9/generated/torch.optim.Muon.html
- Moonshot Muon scaling: https://arxiv.org/abs/2502.16982
- PolarExpress: https://arxiv.org/abs/2505.16932
- MARS: https://arxiv.org/abs/2411.10438
- MARS-M: https://arxiv.org/abs/2510.21800
- Schedule-Free: https://arxiv.org/abs/2405.15682
- Schedule-Free repo: https://github.com/facebookresearch/schedule_free
- WSD schedule: https://arxiv.org/abs/2410.05192
- Mamba-3: https://arxiv.org/abs/2603.15569
- Gated DeltaNet-2: https://arxiv.org/abs/2605.22791
- Preconditioned DeltaNet: https://arxiv.org/abs/2604.21100
- Kimi Linear / KDA: https://arxiv.org/abs/2510.26692

## Verification

Focused tests run:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_representation_training_probe_integration.py \
  src/tac/tests/test_pr95_muon_local_training_integration.py \
  src/tac/tests/test_optimizer_candidate_queue.py \
  src/tac/tests/test_optimizer_guided_candidate_generation.py \
  src/tac/tests/test_cathedral_consumer_payload_passthrough.py
```

Result: `28 passed in 0.74s`.

Also passed:

```bash
.venv/bin/python -m ruff check \
  src/tac/optimization/optimizer_training_signal_bridge.py \
  src/tac/optimization/optimizer_guided_candidate_generation.py \
  src/tac/optimization/pr95_muon_local_training_integration.py \
  src/tac/optimization/representation_training_probe_integration.py \
  src/tac/optimizer/candidate_queue.py \
  tools/build_optimizer_guided_candidate_queue.py \
  tools/cathedral_autopilot_autonomous_loop.py \
  src/tac/tests/test_optimizer_guided_candidate_generation.py \
  src/tac/tests/test_optimizer_candidate_queue.py \
  src/tac/tests/test_pr95_muon_local_training_integration.py \
  src/tac/tests/test_representation_training_probe_integration.py \
  src/tac/tests/test_cathedral_consumer_payload_passthrough.py

.venv/bin/python -m py_compile \
  src/tac/optimization/optimizer_training_signal_bridge.py \
  src/tac/optimization/optimizer_guided_candidate_generation.py \
  src/tac/optimization/pr95_muon_local_training_integration.py \
  src/tac/optimization/representation_training_probe_integration.py \
  src/tac/optimizer/candidate_queue.py \
  tools/cathedral_autopilot_autonomous_loop.py

git diff --check
```

## Next Two-Week Parallel Tranche

1. Local DQS1 bounded sweep: keep running local-first rank/group/null candidates
   and exact-eval only calibrated eureka positives.
2. PR95 optimizer registry: implement orthogonalized optimizer registry and
   schedule registry; add PR95 probe flags; run paired source-faithful vs
   variant smokes.
3. Generic representation-training emitters: wire PR95 variants, new HNeRV,
   NeRV-family, CompressAI/Ballé/Cool-Chic, SIREN/WIRE/BACON, wavelet/null/
   procedural-codebook, and deterministic compiler probes into
   `representation_training_probe_manifest_v1`.
4. Master-gradient training telemetry: emit per-pair weights, hard-pair hit
   rate, and SegNet/PoseNet axis deltas from training probes.
5. MLX batch drift: run batch-size trace and identify whether drift is fixed,
   batch-dependent, device-dependent, or scorer-cache-dependent.
6. Queue store: move dynamic learned sweeps into SQLite with pause/freeze/
   rewind/concurrency and observation telemetry.
7. Cathedral hardening: centralize consumer-payload authority validation and
   add a preflight guard.
8. Exact-gate discipline: every byte-closed candidate enters exact-readiness
   promotion before any CPU/CUDA auth dispatch.
