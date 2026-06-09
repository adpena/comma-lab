# B1 baseline discipline — operator directives (2026-06-09), binding for the B1 launch

Operator sharpening (verbatim spine): "the first exact score beats all optimizer speculation."
CRITICAL RISK flagged: "inner-loop speed work becomes another rabbit hole unless it
immediately produces b1_large_batch_timing_sweep.v1 and a launch decision." These 8 points
are binding for the B1-baseline launch; the B1 agent + orchestrator honor them.

## The 3 optimizer layers (settled)
- **Layer 1 — inner differentiable trainer** (proposes params): PR95-family. Stages 1-7
  AdamW/Adam staged CE→margin→smooth→QAT→hard-pixel/C1a→λ→σ; stage 8 vanilla Muon
  continuation on matrix-like decoder weights ONLY. Use MLX MultiOptimizer / param groups.
- **Layer 2 — archive/action waterfiller** (the REAL contest optimizer, already built):
  admit σ iff S(P+σ) < S(P) by exact evaluate.py. `tac.optimization.evaluator_action_waterfill`.
- **Layer 3 — commutator/global selector** (Vehicle 3): comm(a,b)=ΔS(a∘b)−ΔS(a)−ΔS(b);
  comp-muon's composed-operator philosophy lives HERE, not B1.

## The 8 binding points
1. **Outer waterfiller is THE canonical optimizer.** Inner-optimizer changes are variants.
2. **B1 baseline = PR95-family; NO optimizer novelty in baseline.** No Aurora/comp-muon/Lion/
   LionMuon in the baseline. They are POST-baseline controlled variants only.
3. **Large-batch timing measures speed AND improvement** — emit `b1_large_batch_timing_sweep.v1`;
   choose batch by `proxy_score_improvement_per_wall_clock_second`, NOT seconds/epoch alone.
4. **Full batch may be BAD for early argmax chamber discovery** (it averages away the
   stochasticity the discontinuous argmax objective needs). Hypothesis: medium batch (64-256)
   stages 1-2 for chamber discovery; large/full stages 3-8 for margin/QAT/rate/Muon. Let the
   sweep decide.
5. **MultiOptimizer param groups MANDATORY.** Muon group = matrix-like decoder weights only.
   EXCLUDED from Muon (→ AdamW/Adam): latents, biases, norms, entropy/QAT scalars,
   scalar schedules, sidecar/action/waterfill params, stem, rgb head. (= the verified PR95
   selective partition `ndim>=2 ∧ not stem ∧ not rgb`, applied to decoder weights.)
6. **comp-Muon = Vehicle-3 research note** (composed evaluator-action operators:
   fine_injector∘head_rgb_1, frame1-birth∘frame0-pose, SNeRV-source∘receiver), NOT B1.
7. **Lion = post-baseline variant only** (MLX has it); not a baseline/launch dependency.
8. **First exact score beats optimizer speculation.** Get the 229K PR95-family backend-only
   exact CPU score (then CUDA if promising) BEFORE any optimizer variant is worth considering.

## Sequencing (one line)
First exact backend-only HiNeRV score → then pass fleet → then atoms → then global waterfiller
→ then SNeRV/semantic/interpreter composition.

## Concrete launch chain
1. throughput-fix lands → `b1_large_batch_timing_sweep.v1` + a LAUNCH DECISION (anti-rabbit-hole:
   even a PARTIAL speedup is fine; at the current ~8 s/epoch the reduced 3000-ep pilot is ~6.5h
   local $0 — already launchable; do NOT block the first score on a perfect speedup).
2. `b1_launch_manifest.json` (param_count=228903, decoder_channels=[36,30,23,17,14,11,8],
   pay_rent_gate_active=true, sidecar_exported=false-unless-pays_rent, stages 1-7 muon_active=false,
   stage 8 muon_active=true, Muon groups restricted per point 5, exact-eval command stub,
   telemetry path, checkpoint/resume path).
3. Launch 229K PR95-family backend-only baseline (reduced pilot first).
4. Exact `evaluate.py --device cpu` (Linux x86_64 authoritative; macOS-CPU = advisory only),
   CUDA/T4 if promising → the B2-bridge arbiter.
5. ONLY THEN optimizer variants (Lion, stage-8 Muon variants, the verified-but-predicted-null
   Aurora arm, comp-muon-like composed-operator research if a composed pair exists).

## Burning question this chain answers
Does the first 229K PR95-family backend-only pass beat the 0.19199 frontier — before any
optimizer variant is even worth considering? Get to that exact number first.
