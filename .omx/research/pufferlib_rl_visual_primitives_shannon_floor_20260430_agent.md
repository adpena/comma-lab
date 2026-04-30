# PufferLib, RL, Local-Model, and Visual-Primitive Applicability to Shannon-Floor Search

Date: 2026-04-30
Author: Codex agent
Scope: research ledger only. No code changes. No GPU dispatch. No score claims.

## Controlling Policy

CUDA exact auth eval on exact archive bytes remains the only promotion,
ranking, or retirement truth:

```text
archive.zip -> inflate.sh -> upstream/evaluate.py
```

All PufferLib, RL, local-model, LM Studio, and visual-primitive outputs in this
document are `external`, `prediction`, `derivation`, or `empirical-diagnostic`
until a candidate archive is evaluated through `experiments/contest_auth_eval.py
--device cuda` with full custody.

This report is also scoped by the current frontier and claims matrix:

- PFP16 is current A++ deploy baseline:
  `score=1.043987524793892`, `archive_bytes=686635`,
  `archive_sha256=0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`.
- Lane 12 NeRV `jsonfix40` retired only as a measured implementation/config.
  Alpha/mask compression remains open, but must preserve decoded baseline
  geometry and PoseNet.
- OWV3/Fisher is byte-blocked until ASYM-preserving or byte-positive before
  exact eval.
- Primary scorer KL remains forensic-only; SegNet-aux KL remains gated.

## External Sources Used

Fetched/consulted on 2026-04-30:

1. PufferLib docs: https://puffer.ai/docs.html
   - Claims current PufferLib 4.0 has a native CUDA backend, a PyTorch fallback,
     Ocean environments, Constellation visualization, and Protein tuning.
   - Docs report PuffeRL up to 20M steps/s in CUDA and 5M steps/s in Torch.
   - Docs describe Protein as a hyperparameter tuning algorithm combining
     Gaussian processes and a simple genetic algorithm over a Pareto frontier
     of cost and score.
2. PufferLib homepage/GitHub:
   - https://puffer.ai/
   - https://github.com/PufferAI/PufferLib
   - GitHub page showed branch/tag context around 4.0 and latest release
     "4.0 Experiments" dated 2026-04-05 at time of fetch.
3. PufferLib paper:
   - https://arxiv.org/abs/2406.12905
   - Frames PufferLib as a compatibility/vectorization layer for RL libraries
     and environments.
4. PufferLib 2.0 RLC page:
   - https://rlj.cs.umass.edu/2025/papers/Paper151.html
   - Reports first-party environments running at 1M steps/s and broad
     environment compatibility.
5. LM Studio docs:
   - Local server: https://lmstudio.ai/docs/developer/core/server
   - System requirements: https://lmstudio.ai/docs/app/system-requirements
   - OpenAI-compatible chat: https://lmstudio.ai/docs/developer/openai-compat/chat-completions
   - Structured output: https://lmstudio.ai/docs/developer/openai-compat/structured-output
   - Tool use: https://lmstudio.ai/docs/developer/openai-compat/tools
   - Parallel requests: https://lmstudio.ai/docs/app/advanced/parallel-requests
   - Speculative decoding: https://lmstudio.ai/docs/app/advanced/speculative-decoding
   - LM Link: https://lmstudio.ai/docs/developer/core/lmlink
6. Local reasoning-model references:
   - LM Studio DeepSeek R1 blog: https://lmstudio.ai/blog/deepseek-r1
   - DeepSeek R1 repo: https://github.com/deepseek-ai/DeepSeek-R1
   - LM Studio Community DeepSeek-R1-Distill-Qwen-32B GGUF:
     https://huggingface.co/lmstudio-community/DeepSeek-R1-Distill-Qwen-32B-GGUF
   - Bartowski DeepSeek-R1-Distill-Qwen-32B GGUF sizes:
     https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF
7. DeepSeek "Thinking with Visual Primitives" PDF supplied by user:
   - https://huggingface.co/datasets/NodeLinker/deepseek-ai-Thinking-with-Visual-Primitives-deleted-repo/resolve/main/Thinking_with_Visual_Primitives.pdf
   - Downloaded PDF SHA-256:
     `1951785873385498b608b7aa66868f9915ecd78f6c578d42ef52d80ad25f7153`
   - PDF size: `4739600` bytes, 25 pages.
   - Text extraction SHA-256:
     `a869ab913b5ff9630cbb4e20b4490d46176c0b0670aa66ef44548e65f7654cd9`

## Executive Verdict

### Highest-Value Adoption

1. Use visual-primitives thinking immediately as a design pattern for Alpha
   geometry preservation:
   - Treat object boxes, lane/boundary polylines, component centroids, temporal
     point tracks, and pair-diff trajectories as explicit state in diagnostics,
     training targets, and charged residual planning.
   - This directly addresses the observed Lane 12 failure mode: rate mechanics
     worked, but geometry and PoseNet reference stability collapsed.

2. Use local-model tooling for read-only harness triage and proposal
   generation:
   - LM Studio's local OpenAI-compatible server, JSON schema output, and tool-use
     format are useful for offline extraction of structured findings from logs,
     manifests, and research ledgers.
   - Local models must not make score, promotion, kill, deletion, or dispatch
     decisions. They produce candidate JSON that deterministic validators must
     check.

3. Use bandits/Bayesian optimization before PufferLib/PPO:
   - Current codec lane choice is closer to a low-dimensional, expensive,
     one-step decision problem than a high-throughput RL environment.
   - PufferLib is compelling only after a cheap deterministic surrogate
     environment exists whose steps are at least 100x cheaper than archive eval
     and whose reward correlates with exact CUDA results.

### Defer or Stop

1. Defer full PufferLib PPO over exact archive eval.
   - Direct policy-gradient search over archive candidates is too expensive:
     PPO needs many rollouts, while exact archive eval is minutes and dollars
     per step.
   - This is not a PufferLib family kill. It is a scoped "wrong first tool"
     verdict for direct exact-eval control.

2. Do not use DeepSeek TWP as an available model/tooling dependency.
   - The supplied PDF is usable research input, but the repository appears
     deleted from the URL path and no code/model artifact was verified here.
   - Adopt the method pattern, not an unpinned runtime dependency.

3. Do not use local 128GB model outputs as evidence.
   - They can reduce human wall-clock on log triage, experiment design, and
     structured proposal generation.
   - They cannot validate score, rank lanes, or retire methods.

## Applicability Matrix

| Candidate | Immediate value | Main use | Not allowed for | Continue condition |
|---|---:|---|---|---|
| PufferLib native CUDA RL | Medium later | High-throughput surrogate env once cheap reward exists | Direct exact-eval PPO | Surrogate reward correlates with exact CUDA deltas across >= 20 candidates |
| PufferLib Protein/Pareto tuner | Medium | Cost-aware sweep scheduler for lane hyperparameters | Score claims | Beats current hand/codex sweep in exact eval efficiency |
| Contextual bandit / BO / CMA-ES | High now | First-line codec and controller search | Promotion without exact eval | Finds exact-eval candidate >= 0.005 better than hand sweep within cap |
| Local LM Studio 128GB | High now | Offline log/research triage, structured JSON, prompt batching | Score/rank/kill/dispatch authority | Validator precision/recall on known harness incidents >= 0.95 |
| DeepSeek visual primitives | High now | Alpha geometry, mask residuals, pose/track diagnostics | Runtime dependency or score evidence | Improves Alpha diagnostics and exact PoseNet recovery |
| DeepSeek-style RL rewards | Medium | Dense reward shaping for primitive-preserving mask/pose policies | Replacement for exact eval | Reward correlation to exact components is measured and stable |

## Key Source Takeaways

### PufferLib/RL

PufferLib's strongest claims are speed, vectorization, and sweep-first tooling:
CUDA/Torch training speed, C/CUDA-oriented environments, static memory,
cudagraph tracing, and Protein search over cost/score Pareto points. That maps
well to simulator-style lane search, not to direct exact-eval archive search.

For this repo, an "environment step" can be one of three things:

1. Cheap diagnostic step:
   - Build no archive.
   - Use deterministic byte accounting, mask diagnostics, sensitivity-map
     summaries, or round-trip metrics.
   - Good fit for PufferLib or bandit search.

2. Medium candidate step:
   - Build deterministic archive and run local shape/manifest/byte checks.
   - Still not score evidence.
   - Good fit for BO/Pareto scheduling.

3. Exact score step:
   - Run full CUDA auth eval.
   - Expensive and too sparse for PPO rollouts.
   - Use only for finalists selected by the cheap stages.

Conclusion: PufferLib is not the first move. The first move is a smaller
bandit/BO harness over existing deterministic candidate builders and diagnostic
artifacts. PufferLib becomes attractive if we can vectorize a surrogate
environment with thousands to millions of cheap steps.

### LM Studio / Local 128GB

LM Studio is useful because it can run local models via REST/OpenAI-compatible
endpoints, structured JSON schemas, tool-use format, parallel requests via
continuous batching for GGUF, and speculative decoding with a draft model. A
128GB local machine can run useful 32B-class reasoning models and some larger
quantized models, but model size is not the gating issue for contest evidence.

Recommended local roles:

- Convert messy logs into schema-validated harvest summaries.
- Extract archive-custody gaps from `contest_auth_eval.json`,
  `remote_provenance.json`, manifests, and run logs.
- Generate candidate experiment cards from current ledgers.
- Cross-check stale tracker entries against live-state JSON, in read-only mode.
- Draft "why this exact eval is worth spending" review packets.

Forbidden roles:

- No autonomous archive deletion.
- No score parsing from human logs when JSON exists.
- No rank/promotion/retirement decisions.
- No use of MCP integrations in this project unless explicitly re-enabled.
- No score-affecting sidecars or hidden local-model outputs.

### DeepSeek Thinking with Visual Primitives

The PDF's core contribution is a "reference gap" diagnosis: language-only
reasoning is too ambiguous for complex spatial layouts. It proposes interleaved
points and boxes as minimal thought units. The training pipeline uses verified
box/point data, specialized SFT, specialized RL via GRPO, reward models for
format/quality/accuracy, and dense rule-based rewards for tasks such as maze
navigation and path tracing.

The relevant transfer is not the benchmark claim. It is the engineering pattern:

- Use explicit coordinates as state, not prose labels.
- Use rule-based verifiers wherever geometry is deterministic.
- Reward trajectory continuity, endpoint accuracy, coverage, and violation
  penalties instead of only final answer correctness.
- Separate specialists first, then merge/unify only after each specialist is
  validated.

This maps directly to Alpha and mask/pose decisions:

- Boxes: connected components, vehicles, lane regions, high-risk object masks.
- Points: component centroids, pose-sensitive landmarks, trajectory knots.
- Polylines: lane boundaries, temporal object tracks, path-like mask changes.
- Dense rewards: boundary-ring agreement, temporal-diff continuity, centroid
  stability, renderer-embedding drift, PoseNet component preservation.

## Concrete Experiment Designs

### E1. Bandit Codec Search Before PufferLib

Goal:
Find whether a cheap contextual bandit/BO/CMA-ES search can discover better
codec settings than hand/codex sweeps without building full RL infrastructure.

State:

- Archive byte accounting by member.
- Current PFP16 baseline bytes/components.
- Sensitivity artifact summaries if available.
- Alpha diagnostics: global Hamming, boundary-ring disagreement, temporal
  pair-diff disagreement, component centroid drift.
- Renderer action summaries for OWV3/NWCS/IMP candidates.

Actions:

- Mask/residual knobs: residual budget, boundary-ring radius, component class
  weights, temporal correction density.
- Renderer knobs: qint budget, protected channel threshold, keep-asym vs
  low-bit vs fp16-protect action.
- Pose knobs: fp16, arithmetic delta, regeneration toggle.
- Eval knobs: whether candidate deserves exact eval.

Reward:

- Diagnostic reward:
  `-(predicted_score) - penalty(noncompliance) - penalty(component_risk)`.
- Exact reward only when CUDA eval exists:
  `-score_recomputed_from_components`.
- Exact eval results must be stored as sparse anchor points; diagnostic reward
  remains non-promotable.

Wall-clock and cost:

- 1 evening design/code once implementation is approved.
- Local candidate scoring: minutes to hours, $0.
- Exact finalists: 5-20 CUDA evals, roughly $2.50-$12 depending backend and
  wall-clock.
- Initial cap: $5 or 4 hours.

Deterministic reproducibility:

- Fixed random seed and sampled action list.
- One JSONL row per trial with config, source archive SHA, candidate manifest,
  byte accounting, diagnostic metrics, and exact eval path if used.
- Deterministic archive rebuild check before exact eval.
- No human-log score parsing.

Kill/continue gates:

- Continue if within $5 it finds at least one exact CUDA candidate that improves
  over the hand baseline by >= 0.005 score with no component collapse.
- Continue if it equalizes score while saving >= 2% archive bytes and exact
  PoseNet/SegNet stay within gates.
- Stop the measured pilot if no exact finalist beats a random/hand sweep after
  50-100 trials.
- Stop the measured pilot if its top diagnostic candidates repeatedly fail exact
  CUDA due to the same unmodeled component risk.

### E2. PufferLib Surrogate Environment, Not Exact-Eval PPO

Goal:
Use PufferLib only after a cheap surrogate is validated, so policy search is
performed over thousands of low-cost steps rather than exact archive evals.

Environment:

- Episode = allocate a fixed byte budget across mask, renderer, pose, residual,
  and side-info streams.
- Step = choose one stream/action update.
- Terminal = predicted archive score plus hard compliance penalties.
- Optional exact-eval action = terminal "spend" action, heavily penalized in
  training and used only in evaluation.

Why PufferLib:

- Vectorized environment execution if the reward is cheap.
- Protein/Pareto sweep can optimize reward against wall-clock.
- Constellation could help visualize Pareto movement, but visualization is not
  evidence.

Wall-clock and cost:

- Prototype surrogate: 1-2 days after bandit success.
- PufferLib install/build via pinned Docker/GitHub commit: 0.5 day.
- Training: local/cheap GPU if surrogate steps are sub-second; $0-$10.
- Exact eval finalists: separate budget, $5-$25.

Deterministic reproducibility:

- Pin PufferLib commit/release, Docker image digest, CUDA version, seeds.
- Store every environment version, reward coefficients, and feature schema.
- No PufferLib output promoted without independent exact archive eval.

Kill/continue gates:

- Continue only if surrogate top-20 ranking has positive Spearman correlation
  with exact CUDA deltas on a validation set of >= 20 candidates.
- Continue if PufferLib/Protein finds candidates with fewer exact evals than
  bandit/BO at the same score threshold.
- Stop this measured implementation if surrogate correlation is unstable,
  negative, or dominated by byte-only predictions that miss PoseNet/SegNet.
- Do not make a broad PufferLib/RL kill from one failed surrogate.

### E3. Visual-Primitive Alpha Geometry Rescue

Goal:
Prevent a repeat of Lane 12 `jsonfix40`: strong byte reduction but catastrophic
PoseNet/geometry drift.

Primitive representation:

- Boxes for connected components and object/lane regions.
- Points for centroids, endpoints, high-curvature boundary knots, and temporal
  landmarks.
- Polylines for lane boundaries, moving-object tracks, and temporal-diff paths.
- Residual tiles only where primitives indicate high scorer risk.

Candidate sequence:

1. Extract primitives from decoded baseline `masks.mkv`, not fresh SegNet labels.
2. Train or configure the mask codec to preserve primitive state.
3. Add charged sparse corrections only for violated primitive constraints.
4. Regenerate poses against decoded candidate masks before exact eval.
5. Build deterministic archive and run CUDA auth eval only after diagnostics pass.

Diagnostics:

- Global Hamming <= 0.003 exploratory, <= 0.001 promotion diagnostic.
- 2px boundary-ring disagreement <= 0.005 exploratory, <= 0.002 promotion
  diagnostic.
- Temporal pair-diff disagreement <= 0.004 exploratory, <= 0.002 promotion
  diagnostic.
- Connected-component centroid jump <= 1 px promotion diagnostic.
- Renderer embedding drift tracked against `jsonfix40` and baseline.

Wall-clock and cost:

- Primitive extraction/diagnostics: 2-6 local hours.
- Training small candidate: 4-12 hours on 4090/A10G, roughly $1-$12 depending
  backend.
- Exact eval: one CUDA eval per byte-plausible finalist, roughly $0.50-$3.

Deterministic reproducibility:

- Store baseline mask stream SHA, primitive extraction config, normalization
  rules, candidate mask SHA, residual side-info bytes, pose-regeneration
  provenance, archive manifest, and exact eval JSON.
- All primitive side information charged inside `archive.zip` for candidates.

Kill/continue gates:

- Continue if diagnostics improve versus `jsonfix40` and archive stays below
  PFP16 bytes by a meaningful margin.
- Continue to exact eval only if PoseNet risk diagnostics are below exploratory
  gates and deterministic archive closure is clean.
- Retire the measured config if exact CUDA PoseNet > 0.01 or if score fails to
  beat PFP16 after two geometry-preserving variants with the same root cause.
- Do not kill NeRV/INR/mask compression family from this result.

### E4. Visual-Primitive Reward Model for Mask/Pose Decisions

Goal:
Borrow DeepSeek's verifier-heavy reward design to train or search mask/pose
controllers with dense, interpretable rewards.

Reward components:

- Format reward: primitive schema parses, coordinates in range, no duplicate
  boxes/points unless allowed.
- Quality reward: primitives refer to meaningful connected components, not
  arbitrary singleton noise.
- Accuracy reward:
  - boundary coverage,
  - bidirectional polyline distance,
  - endpoint/centroid accuracy,
  - trajectory continuity,
  - temporal-diff recall,
  - no illegal jumps across disconnected components,
  - final exact component reward only when CUDA eval exists.

Use cases:

- Select sparse residual correction regions for Alpha.
- Decide pose regeneration vs reuse.
- Route bits between boundary and interior masks.
- Identify candidate archives that deserve exact eval.

Wall-clock and cost:

- Rule-based reward prototype: 1 day local.
- Controller search via E1 bandit: $0-$5 initial.
- RL training only after bandit and reward validation: $10-$50 exploratory.

Deterministic reproducibility:

- Store reward component values separately, not only aggregate reward.
- Fixed weights or documented weight search.
- Calibration/holdout split by frame pairs.
- Exact eval anchors used only as held-out validation.

Kill/continue gates:

- Continue if reward predicts exact component preservation on holdout candidates
  better than current Hamming/boundary-only diagnostics.
- Continue if it reduces wasted exact evals by >= 30% over current review.
- Stop measured reward if it can be gamed by trivial large boxes, all-boundary
  residuals, or byte-only solutions.

### E5. Local LM Studio Harness Triage

Goal:
Use a local model as a read-only assistant for contest custody and harness
automation, with deterministic validators as authority.

Inputs:

- `contest_auth_eval.json`
- `remote_provenance.json`
- archive manifest
- logs
- source/staged-tree manifests
- dispatch state ledgers

Outputs:

- Strict JSON summary:
  - archive SHA/bytes consistency,
  - CUDA/full-sample status,
  - component gates,
  - stale parser fields,
  - missing provenance,
  - hidden sidecar risks,
  - recommended next deterministic check.

Local model configuration:

- Use LM Studio local server on `localhost` by default.
- Use structured output JSON schema.
- Prefer 32B-class reasoning models for cost/speed; 32B Q4_K_M is about
  19.9GB in one referenced GGUF card, with 128k context support in the LM Studio
  community card.
- Record model identifier, quant file SHA, context size, seed, temperature, and
  prompt template.
- Do not use MCP integrations unless the user re-enables MCP for this project.

Wall-clock and cost:

- Setup: 1-2 hours.
- Per harvest packet: seconds to minutes.
- GPU/cloud cost: $0 if local.

Deterministic reproducibility:

- Every local-model output is advisory and stored next to the deterministic
  validator output.
- Validator must be able to reproduce the final custody decision without the
  model.
- No hidden prompts or model-generated sidecars in candidate archives.

Kill/continue gates:

- Continue if local model plus validator catches >= 95% of known historical
  custody/harness issues with <= 5% false blockers on a replay set.
- Continue if it reduces manual triage wall-clock by >= 2x without changing
  deterministic decisions.
- Stop measured workflow if it invents file paths, misses CUDA/CPU distinctions,
  or encourages log-score parsing over JSON.

### E6. Local Model Research Synthesizer for Lane Cards

Goal:
Generate standardized experiment cards from research ledgers so humans/agents
spend exact eval budget on better-scoped candidates.

Card schema:

- hypothesis,
- evidence grade,
- source docs,
- expected byte delta,
- expected component risk,
- exact eval prerequisites,
- deterministic manifest requirements,
- cost estimate,
- scoped kill/continue gates.

Wall-clock and cost:

- Setup reused from E5.
- 5-20 cards/hour local, $0 cloud.

Deterministic reproducibility:

- Store prompt, model ID, source file SHAs, output JSON, and human/validator
  edits.
- Cards cannot enqueue dispatch automatically.

Kill/continue gates:

- Continue if card quality reduces ambiguous dispatches and missing provenance.
- Stop measured workflow if it increases stale/prediction-only dispatches.

### E7. Protein/Pareto Scheduler for Remote Spend

Goal:
Use the PufferLib Protein idea, or a repo-native equivalent, to optimize
score-improvement probability versus wall-clock/cost across lanes.

State:

- Lane readiness,
- dependency graph,
- live remote capacity,
- cost platform table,
- exact-evidence availability,
- expected component risk,
- freshness/staleness of trackers.

Action:

- Choose next diagnostic, build-only, or exact-eval spend.

Reward:

- Exact score improvement only when exact eval exists.
- Otherwise reward is information gain: reduced uncertainty, proven compliance,
  or retired measured config.

Wall-clock and cost:

- Initial scheduler simulation: 1 day local.
- No GPU required.
- If it dispatches candidates later, exact eval budget stays explicit.

Deterministic reproducibility:

- Immutable decision log.
- State snapshot before each recommendation.
- No mutation of `.omx/state` without explicit operator command.

Kill/continue gates:

- Continue if it improves harvest/eval throughput or avoids duplicate dispatches
  on replay.
- Stop measured scheduler if it conflicts with live API evidence, ignores locks,
  or recommends prediction-only promotion.

### E8. Visual-Primitive Component Sensitivity Audit

Goal:
Tie component sensitivity maps to human-inspectable geometric primitives so
OWV3/Alpha decisions are less opaque.

Method:

- Overlay component-sensitivity maps with boxes, centroids, boundary polylines,
  and temporal tracks.
- Aggregate PoseNet/SegNet sensitivity by primitive class.
- Use this to decide which regions/channels get protected or corrected.

Wall-clock and cost:

- Local analysis after CUDA sensitivity artifact exists: 2-6 hours.
- No extra exact eval until it proposes a candidate archive.

Deterministic reproducibility:

- Requires CUDA-authored `component_sensitivity_v1` artifact.
- Store primitive extraction config and aggregation code version.
- Separate SegNet, PoseNet, and combined maps.

Kill/continue gates:

- Continue if primitive groups show stable calibration/holdout rank ordering.
- Continue if region/channel protection reduces archive bytes or component risk
  in a byte-plausible build.
- Stop measured config if primitive grouping is unstable or redundant with
  existing sensitivity maps.

## Expected Wall-Clock and Compute Cost Summary

| Experiment | First useful signal | GPU/cloud cost | Wall-clock | Promotion path |
|---|---:|---:|---:|---|
| E1 bandit codec search | 4 hours | $0-$5 cap | 0.5-1 day | Exact CUDA finalists only |
| E2 PufferLib surrogate | 2-4 days | $0-$10 plus exact finalists | 2-5 days | Only after surrogate/exact correlation |
| E3 visual-primitive Alpha rescue | 1-2 days | $1-$15 first wave | 1-3 days | Exact CUDA archive with pose gates |
| E4 primitive reward model | 1 day | $0-$5 first wave | 1-2 days | Exact CUDA anchors validate reward |
| E5 local harness triage | same day | $0 | 1 day setup | Validator output, not model output |
| E6 local lane-card synthesizer | same day | $0 | 0.5-1 day | Human/validator gated |
| E7 Pareto scheduler | 1 day | $0 | 1-2 days | Recommendations only |
| E8 primitive sensitivity audit | after sensitivity artifact | $0 extra | 0.5-1 day | Exact CUDA candidate archive |

## Reproducibility Requirements

For every experiment above:

- Record source docs and fetched source hashes where applicable.
- Record repo git status without reverting unrelated work.
- Record exact config JSON, random seeds, model identifiers, quantization files,
  prompt templates, and environment variables.
- Record all diagnostic metrics in structured JSON.
- Keep local model outputs as advisory artifacts, never as hidden sidecars.
- Build candidate archives deterministically:
  fixed member ordering, timestamps, permissions, compression settings, manifest
  records, no hidden files, no resource forks, no traversal paths.
- Exact score claims require:
  - exact archive bytes,
  - archive SHA-256,
  - full manifest,
  - `contest_auth_eval.json`,
  - CUDA device proof,
  - full sample count,
  - component distances,
  - recomputed score,
  - logs,
  - hardware provenance,
  - source/staged-tree manifest,
  - adversarial review status.
- CPU/MPS/local model/proxy/smoke/byte-only evidence remains diagnostic.

## Recommended Order

1. Implement or plan E5/E6 first if the goal is immediate wall-clock reduction
   without touching scoring code: local structured triage and lane-card
   generation are safe, cheap, and reversible.
2. Run E1 before any PufferLib setup. If bandit/BO fails under a $5 cap, direct
   RL is not justified yet.
3. Start E3/E4 in parallel with sensitivity work because visual primitives map
   directly to the known Alpha failure mode.
4. Use E8 after a valid CUDA component-sensitivity artifact exists.
5. Graduate to E2 only after E1 plus E4/E8 provide a cheap reward with measured
   exact-eval correlation.

## Final Position

PufferLib and RL are useful as search infrastructure only after the reward
surface is cheap and validated. For the current contest state, the fastest
scientifically defensible path is:

```text
visual primitives for Alpha geometry
+ deterministic component sensitivity
+ bandit/BO controller search
+ local-model structured triage
+ exact CUDA archive eval for finalists
```

This can improve lane search, controller policies, mask/pose/codec decisions,
and harness automation without weakening contest evidence discipline. The only
outputs that can move the frontier remain exact CUDA auth eval artifacts on
deterministic archive bytes.
