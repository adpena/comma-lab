# True Strategy Review: CPU/GPU, Loader Drift, Solvers, And The Optimal Floor

Date: `2026-05-08`
Owner: `codex`
Requested mode: `30k-foot xhigh strategy review`
Branch observed: `main`
Write scope: this memo only
Dispatch performed: `false`
Lane claim performed: `false`
Score claim: `false`
Promotion/rank/kill claim: `false`

## Scope And Non-Negotiable Corrections

This memo synthesizes the 2026-05-08 CPU/GPU drift, loader-drift,
introspection, solver/autopilot, monolithic-packet, paper/OSS/site, and
score-lowering roadmap findings. It is a strategic control ledger, not a
promotion ledger.

The critical corrections are:

1. CPU, macOS CPU, GitHub Actions CPU, and MPS can accelerate development,
   reproduce public behavior, and generate candidate priors. They do not
   promote, rank, kill, or retire internal CUDA lanes unless the repository's
   evidence rules are explicitly changed.
2. `[contest-CUDA]` remains the internal promotion/regression/retirement axis:
   exact archive/runtime custody through `archive.zip -> inflate.sh ->
   upstream/evaluate.py`, full sample count, component recomputation, archive
   SHA, runtime-tree SHA, logs, hardware, and review packet.
3. `[contest-CPU]` is a separate Linux x86_64 public-reproduction axis. It is
   not interchangeable with CUDA, and local macOS CPU is not interchangeable
   with Linux x86_64 CPU even when close.
4. DALI/NVDEC versus PyAV/libav is a decoder/input-byte axis. It must be
   separated from PoseNet/SegNet network-kernel precision. Do not collapse
   both into "GPU precision" or "FastViT drift".
5. No signal loss is mandatory: proxy positives, exact negatives, stale
   overclaims, runtime failures, and public-comment evidence all remain useful
   only with explicit scope, evidence grade, and reactivation criteria.
6. Composability remains the architecture rule: every candidate must fit the
   typed chain `representation -> prediction -> quantization -> hyperprior ->
   arithmetic -> pack`, and the scored runtime must consume the changed bytes.

## Context Read

Primary local context inspected before writing:

- `AGENTS.md`
- `docs/hardware_layout.md`
- `reports/latest.md`
- `.omx/research/public_pr_auth_eval_comment_drift_and_dual_axis_protocol_20260508_codex.md`
- `.omx/research/public_pr_cpu_cuda_drift_analysis_20260508_codex.md`
- `.omx/research/cpu_cuda_drift_adversarial_review_20260508_codex.md`
- `.omx/research/public_replay_drift_hypothesis_20260508_codex.md`
- `.omx/research/decoder_drift_introspection_design_20260508_claude.md`
- `.omx/research/loader_drift_discriminator_hardening_20260508_worker_b.md`
- `.omx/research/cuda_cpu_drift_sweep_design_20260508_claude.md`
- `.omx/research/cuda_cpu_pose_drift_mechanism_deep_dive_20260508_claude.md`
- `.omx/research/cuda_cpu_axis_profile_learning_layer_20260508_claude.md`
- `.omx/research/score_lowering_action_ledger_20260508_codex.md`
- `.omx/research/frontier_roadmap_evidence_correction_20260508_worker_a.md`
- `.omx/research/proxy_signal_and_entropy_oracle_guard_20260508_codex.md`
- `.omx/research/representation_integration_gap_audit_20260508_codex.md`
- `.omx/research/autopilot_evidence_semantics_review_20260508_worker_b.md`
- `.omx/research/cathedral_autopilot_candidate_custody_guard_20260508_codex.md`
- `.omx/research/autopilot_post_session_refresh_planning_memo_20260508.md`
- `.omx/research/meta_lagrangian_pareto_gate_20260506_codex.md`
- `.omx/research/strategic_frameworks_roadmap_rollup_20260507.md`
- `.omx/research/grand_council_optimal_path_to_shannon_floor_20260507.md`
- `.omx/research/lossy_falsification_scope_audit_20260508_codex.md`
- `.omx/research/implementation_vs_model_verdict_chain_review_20260508_codex.md`
- `.omx/research/no_signal_loss_canonicalization_plan_20260508_worker_c2.md`
- `.omx/research/paper_proxy_claim_language_audit_20260508_worker_d.md`
- `tools/probe_eval_loader_drift.py`
- `tools/probe_posenet_layer_drift.py`
- `tools/all_lanes_preflight.py`

## Scoring And Custody Tags

Use these tags without dilution:

| Tag | Meaning | May rank/promote/kill? | Strategic use |
| --- | --- | ---: | --- |
| `[contest-CUDA]` / `A++` | Exact full-sample archive/runtime CUDA custody, T4/equivalent, formula recomputed | Yes, within exact review scope | Internal score truth and release gate |
| `[contest-CUDA A-negative]` | Exact CUDA regression for one archive/runtime/config | No broad kill | Retire measured config; define reactivation |
| `[contest-CPU]` | Linux x86_64 CPU auth eval on exact archive/runtime path | Not for CUDA lanes | Public-axis reproduction and CPU-specific calibration |
| `[macOS-CPU advisory]` | Local Apple CPU run | No | Fast development proxy only |
| `[MPS-research-signal]` | Apple MPS smoke/proxy curve | No | Candidate-generation prior only |
| `[CPU-prep]` | CPU byte build, rel_err, smoke, manifest, or packet-construction artifact | No | Packet staging, candidate filtering |
| `[external]` | Public title/comment/leaderboard/source evidence without full local custody | No | Public pressure, intake, reproduction target |
| `[diagnostic]` | Mechanism probe, introspection, or loader/kernel discriminator | No | Falsify mechanism hypotheses |
| `[prediction]` / `[predicted-band]` | Model forecast or solver output | No | Dispatch planning only after custody gates |

## Measured Facts

The CPU/CUDA split is real in the inspected public-comment cluster. PR100,
PR101, PR102, PR103, and PR105 have paired public rows with median
`pose_cuda / pose_cpu` about `4.995`, median `seg_cuda / seg_cpu` about
`1.173`, and an average score gap around `0.033`. Public rounded comments are
not enough for promotion, but they are strong evidence that public PR/title
scores and local CUDA replay scores can refer to different device axes.

PR102 is the cleanest drift example. Local hardened T4 CUDA replay scored
`0.22839372989108092`, matching the public CUDA comment band, while the public
CPU comment band recomputes near `0.195376176526`. That falsifies "local CUDA
replay is broken" for that case and shifts the problem to device-axis
semantics and mechanism closure.

PR104 is no longer an evidence hole on the CUDA axis. Its local T4 replay
scored `0.23113446620399658`, matching the public CUDA comment band within
rounded-component noise and not changing the local frontier.

PR107 now has a Linux x86_64 GitHub Actions CPU replay at `0.1966358879`.
The M5 Max CPU replay was about `6e-6` higher. That makes macOS CPU a useful
development proxy for HNeRV-class CPU-axis sweeps, but it does not upgrade
macOS CPU into `[contest-CPU]` evidence.

The active exact local HNeRV rate anchor is PR103-on-PR106 at about
`0.20898105`, `185578` bytes, archive SHA
`ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`,
full-sample T4 CUDA. Older `0.20454` / `178873` wording is an unanchored
formula projection unless a matching exact CUDA artifact is produced.

Several attractive byte/proxy rows are not score evidence. UNIWARD,
cross-paradigm, ADMM, lossy-int4, GPTQ/AWQ, and lossy-coarsening rows are
valuable only with their current scopes. The PR101 lossy-coarsening exact CUDA
row near `0.3517` and PR106 UNIWARD row near `0.3372` retire measured
configs, not method families.

The public HNeRV-family packets are monolithic at the ZIP layer. PR101 uses
member `x` with parser-proven sections such as `decoder_blob`, `latent_blob`,
and `sidecar_blob`; PR106 uses member `0.bin` with packed internal sections.
Strategy based on separate ZIP members for masks, poses, or renderer streams
is invalid for these packets unless a new runtime packet with consumed bytes is
built and proven.

The xray/introspection tool surface now exists. `tools/probe_eval_loader_drift.py`
records the four intended cells `CPU+AV`, `CUDA+DALI`, `CUDA+AV/shared-input`,
and `CPU+DALI`; `tools/probe_posenet_layer_drift.py` captures shared-input
PoseNet layer drift; `tools/all_lanes_preflight.py` includes the loader-drift
probe as Gate #22. These are diagnostic-only surfaces.

The "FastViT TF32 attention" explanation is overclaimed. The inspected
PoseNet uses FastViT-T12 with RepMixer-style blocks, not attention blocks, and
T4 does not have TF32 hardware. Viable mechanism hypotheses are DALI/PyAV
input-byte differences, CPU/CUDA FP32 convolution/GELU/linear kernel
differences, preprocessing differences, and mixed interactions.

Autopilot and meta-Lagrangian tooling has improved, but solver output is not
score evidence. Non-promotable rows are now blocked from active ranking,
candidate custody requires archive/runtime SHA and score fields, and Pareto
dominance is planning-order metadata only.

## Hypotheses

H1 - decoder-dominant drift:
DALI/NVDEC and PyAV/libav produce small but scorer-visible RGB differences on
the ground-truth side. Because `ds_gt` switches loader by device and
`ds_comp` is raw tensor input, the drift is asymmetric. A 1 to 2 LSB RGB
difference can plausibly explain most of the PoseNet gap.

H2 - network-kernel-dominant drift:
On identical input bytes, CPU and CUDA PoseNet/SegNet forwards differ because
of FP32 kernel accumulation order, cuDNN/oneDNN choices, GELU implementation,
and layer layout. PoseNet regression exposes this as MSE drift; SegNet argmax
is more stable and shows a smaller ratio.

H3 - mixed drift:
Decoder/input drift and network-kernel drift both contribute. This is the
current safest hypothesis because the 2x2 discriminator has not been filled on
a CUDA+DALI host.

H4 - architecture and operating point dependence:
The HNeRV cluster has a tight CPU/CUDA relationship, but qhnerv, H3/AV1,
HPAC, raw AV1, self-compress, Balle-style hyperprior, and future substrates may
have different ratios and floors. A single global CPU predictor is unsafe.

H5 - public-axis strategy changes marginal priorities:
If the CPU axis is the public ranking axis for a given archive family, pose
improvements may transfer with lower marginal utility than CUDA suggests, while
SegNet improvements transfer more strongly. This is a planning hypothesis, not
a license to weaken CUDA custody.

H6 - true floor is solver-and-packet limited, not idea limited:
The best path is not another isolated codec idea. It is a closed packet
compiler plus bilevel optimizer: substrate training, meta-Lagrangian atom
selection, joint rate/distortion allocation, entropy coding, runtime closure,
and exact dual-axis review.

## Falsification Tests

Test 1 - paired dual-axis exact eval:
For the same archive SHA and runtime-tree SHA, run full `[contest-CUDA]` and
Linux x86_64 `[contest-CPU]` auth eval. Required outputs are full JSON,
component fields, archive bytes/SHA, runtime-tree SHA, evaluator hash, sample
count, logs, and recomputation. This calibrates the dual-axis semantics and
prevents public-comment inference from becoming folklore.

Test 2 - 2x2 loader/kernel discriminator:
Run `tools/probe_eval_loader_drift.py --run-forward-cells` on a CUDA host with
DALI/NVDEC. Fill `CPU+AV`, `CUDA+DALI`, `CUDA+AV/shared-input`, and `CPU+DALI`.
This separates raw decoder-byte drift from shared-input PoseNet/SegNet kernel
drift. It remains diagnostic, not a score.

Test 3 - DALI bytes through CPU, AV bytes through CUDA:
Use dumped decoded tensors as shared inputs. If CPU forward on DALI bytes moves
toward CUDA score, decoder is dominant. If CUDA forward on AV bytes stays near
CUDA score, network kernels dominate. If both move, mixed drift wins.

Test 4 - controlled LSB noise:
Inject 1, 2, and 3 LSB perturbations into decoded RGB and run PoseNet/SegNet
for component sensitivity. This estimates how much DALI/PyAV byte drift is
needed to explain the observed ratios.

Test 5 - layer-by-layer PoseNet drift:
Run `tools/probe_posenet_layer_drift.py` on shared input across CPU and CUDA.
Linear growth supports random-walk precision noise; sudden jumps identify
specific modules; flat low drift pushes blame back to loader/input bytes.

Test 6 - architecture-class paired sweep:
Calibrate HNeRV, qhnerv, H3/AV1, HPAC, qzs3/qpose, selfcomp/Quantizr, and raw
AV1 families separately. The learning layer should widen prediction bands for
classes with fewer than three paired anchors.

Test 7 - monolithic packet closure:
Before exact eval of cross-paradigm/ADMM/UNIWARD/Omega candidates, require
parser section offsets, section SHA-256s, old/new archive SHA, runtime-tree
SHA, no-op proof, and proof that `inflate.sh` consumes the changed bytes.

Test 8 - solver calibration:
Every meta-Lagrangian or autopilot recommendation that claims a score delta
must be backtested against exact anchors. Predicted-band rows should be ranked
by expected information gain and custody readiness, not by fake score.

## Engineering Actions

Immediate local-only actions, no dispatch required:

1. Keep all CPU/MPS/proxy rows fail-closed. Any generator that emits planning
   rows must set `score_claim=false`, `promotion_eligible=false`,
   `rank_or_kill_eligible=false`, and `ready_for_exact_eval_dispatch=false`
   unless it has exact custody.
2. Treat `tools/probe_eval_loader_drift.py` and `tools/probe_posenet_layer_drift.py`
   as first-class xray tools. They should write diagnostic artifacts with
   explicit non-promotable tags and should remain wired into preflight or
   operator briefing surfaces.
3. Extend per-class CPU/CUDA calibration only from paired artifacts, not from
   rounded comments alone. Low-anchor families get widened bands.
4. Continue packet compiler work on monolithic archives. Candidate builders
   must operate on parser-proven sections, not invented ZIP members.
5. Grow the exact-evidence corpus loader for meta-Lagrangian/Pareto from
   historical exact artifacts. This is higher value than another speculative
   predicted-band solver pass.
6. Preserve A-negative rows with measured scope and reactivation criteria.
   Do not convert failed exact CUDA configs into family kills.
7. Update public/paper/site wording to say "public-axis reproduction" or
   "external public context" where exact local CUDA custody is absent.

GPU or remote actions, explicitly outside this memo:

1. Paired CPU/CUDA exact eval for selected archives after lane claim and
   operator approval.
2. CUDA+DALI 2x2 discriminator run after lane claim and operator approval if it
   consumes remote GPU time.
3. Harvest-only review of active remote jobs when terminal artifacts exist;
   do not duplicate active claimed lanes.

## Solver Strategy

The solver stack is useful only if it is evidence-gated:

- Autopilot should surface candidates, blockers, and unknown techniques. It
  must not let byte-only, CPU-prep, MPS, or malformed exact rows dominate
  active ranking.
- Meta-Lagrangian search should choose typed atoms and preserve Pareto
  dominance within replacement scopes. Orthogonal atoms must not be discarded
  simply because another family wins on bytes.
- Pareto tracing should operate over `(seg_dist, pose_dist, archive_bytes)`
  and eventually cost/time. It is planning math until populated with exact
  full-custody rows.
- The bilevel path remains the most coherent long-horizon design:
  outer substrate training, middle atom selection, inner allocation. But each
  milestone must produce a byte-closed archive and exact eval, not only a
  better predicted band.
- Per-axis CPU/CUDA calibration can adjust planning weights, but it cannot
  replace exact CUDA promotion gates.

The most important solver correction is epistemic: predicted lower score is
not a candidate. A candidate is a charged packet plus a runtime consumer plus
custody. The solver's job is to decide which packets deserve scarce exact eval
after blockers are closed.

## True Optimal Floor

There is no single proven "true floor" number today. There are three floors:

1. Measured exact floor:
   The exact archive/runtime frontier currently supported by full local CUDA
   custody. This is the only floor that may rank.
2. Practical packet floor:
   The best score likely reachable by composing existing HNeRV/public-substrate
   bolt-ons, monolithic section rewrites, exact entropy coding, guarded
   lossy/coarsening, and runtime-closed packet surgery. This is constrained by
   archive grammar and component drift, not by lack of ideas.
3. Theoretical task-aware floor:
   The Yousfi-Fridrich / Shannon / MDL target, where the substrate is trained
   to make scorer-visible information cheap and scorer-invisible information
   sparse. Existing numbers such as `0.155`, `0.140-0.180`, or lower are
   roadmap bands unless exact archive evidence lands.

The 30k-foot conclusion: the road to the true floor is a deterministic packet
compiler plus a calibrated solver loop. It must ingest public packets, expose
typed streams, emit byte-identical identity packets, then emit intentionally
byte-different optimized packets with exact old/new SHA boundaries and runtime
consumption proofs. Only then should exact CUDA decide whether the theoretical
floor is being approached.

## Score-Lowering Roadmap

P0 - correct the evidence model:
Use dual-axis language everywhere. Exact CUDA remains internal truth; Linux
x86_64 CPU is public-axis reproduction; macOS CPU and MPS are proxy/advisory.
This prevents wrong strategy from being driven by mislabeled scores.

P1 - close the mechanism:
Run the 2x2 loader/kernel discriminator and layer xray when remote work is
authorized. Until then, route DALI/PyAV, PoseNet/Hydra, and SegNet claims as
hypotheses.

P2 - build byte-closed monolithic candidates:
Prioritize PR101/PR106/PR103-on-PR106 section-aware packet work with explicit
parser sections and runtime consumption. Do not spend exact eval on naked
byte proxies.

P3 - harvest and review exact active work:
If an active claimed job lands terminal artifacts, perform the full custody,
recompute, classification, adversarial review, and reactivation packet before
changing status. No duplicate launch while an active claim is open.

P4 - calibrate architecture classes:
Use paired exact artifacts to learn CPU/CUDA profiles by architecture family.
Do not assume HNeRV ratios apply to qhnerv, HPAC, AV1, self-compress, or future
substrates.

P5 - extend solver corpus:
Migrate legacy exact evidence into the canonical schema so Pareto/autopilot
has more than a thin candidate set. This is CPU-only leverage.

P6 - execute bilevel milestones:
Only after packet closure and evidence gates, run small exact-evaluable
milestones: known-anchor replay, one orthogonal atom, one stack, one retrain.
Each must produce an exact result or scoped negative.

## OSS, Paper, And Site Documentation

OSS:
Public APIs should expose the evidence contract, packet compiler primitives,
diagnostic probes, and conformance vectors. Do not publish private provider
state, raw custody logs, local absolute paths, or unreviewed forecast bands.

Paper:
The paper should make evidence-grade discipline the central contribution, not
hide it. CPU/CUDA split, DALI/PyAV split, exact-negative scoping, public
frontier forensics, and packet-custody compiler design are publishable rigor.
Predicted floors and proxy rows belong in roadmap/future-work tables with
explicit falsification tests.

Site:
The site should present exact rows, external rows, proxy rows, and predictions
as separate visual classes. It must not show CPU/macOS/MPS/proxy numbers in a
ranked frontier card unless the card says exactly what axis and evidence grade
it represents. Public release hygiene must run on the exact publish bundle.

## Integration Points

Diagnostic and evidence gates:

- `tools/probe_eval_loader_drift.py`
- `tools/probe_posenet_layer_drift.py`
- `tools/all_lanes_preflight.py` Gate #22
- `tools/public_pr_eval_comment_scorecard.py`
- `tools/plan_dual_device_auth_eval.py`
- `tools/cathedral_autopilot.py`
- `src/tac/optimization/candidate_evidence_contract.py`
- `src/tac/optimization/meta_lagrangian_allocator`

Packet and runtime closure:

- `tools/pr106_archive_decomposition.py`
- `scripts/pre_submission_compliance_check.py --contest-final --strict`
- monolithic section manifests with offsets, lengths, section SHA-256s, and
  old/new archive SHA-256s

Documentation:

- `docs/hardware_layout.md`
- `docs/paper/SUBMISSION_CHECKLIST.md`
- `docs/paper/04_results.md`
- `docs/paper/phase4_paper_harness_blueprint_20260508.md`
- public site bundle hygiene via the existing release-hygiene checks

## Decisions

1. Adopt a dual-axis evidence model: `[contest-CUDA]` for internal exact score
   truth and `[contest-CPU]` for Linux public-axis reproduction.
2. Keep macOS CPU and MPS as acceleration tools only; they can prioritize
   experiments but cannot rank, promote, kill, or dispatch.
3. Treat DALI/PyAV as a separate decoder/input-byte mechanism from network
   precision. All mechanism language must say which axis it means.
4. Use xray diagnostics to falsify mechanisms, not to claim scores.
5. Prefer byte-closed monolithic packet work over naked byte proxies.
6. Use meta-Lagrangian/Pareto/autopilot as custody-aware planning layers, not
   as score authority.
7. Preserve exact negatives as scoped config evidence with reactivation tests.
8. Make OSS/paper/site language evidence-grade native.

## Proposed Tests Summary

| Test | Purpose | Output tag |
| --- | --- | --- |
| Paired exact CPU/CUDA eval | Calibrate same archive/runtime on both official axes | `[contest-CUDA]` and `[contest-CPU]` if full custody |
| 2x2 loader/kernel probe | Separate DALI/PyAV decoder drift from forward-kernel drift | `[diagnostic]` |
| Layer xray | Locate PoseNet/Hydra drift inside shared-input forward | `[diagnostic]` |
| Controlled LSB injection | Estimate scorer sensitivity to decoder-byte drift | `[diagnostic]` |
| Per-class paired sweep | Learn architecture-specific CPU/CUDA profiles | `[contest-*]` only if exact, otherwise `[prediction]` |
| Monolithic packet closure | Prove changed section bytes are charged and consumed | `[CPU-prep]` until exact eval |
| Solver backtest | Check autopilot/Pareto predictions against exact anchors | `[diagnostic]` / `[planning]` |

## Final Position

The strategy after 2026-05-08 is not "optimize CPU instead of CUDA" and not
"ignore CPU because CUDA is internal truth." The correct strategy is stricter:
measure and label both official axes, use cheap local proxies for velocity,
close the DALI/PyAV versus network-kernel mechanism with xray tools, and route
all score-lowering work through byte-closed, runtime-consumed, composable
packets before exact CUDA and exact CPU are allowed to change status.
