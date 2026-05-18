---
schema: pact_design_memo_v1
memo_id: rate_attack_novel_vectors_design_memo_20260518
review_date: "2026-05-18"
author: codex
write_scope: ".omx/research only"
score_claim: false
promotion_eligible: false
provider_spend: false
research_only: true
canonical_consumer: "tools/cathedral_autopilot_autonomous_loop.py"
frontier_anchor:
  contest_cpu: "0.1920513169 [contest-CPU GHA Linux x86_64], archive 6bae0201fb08, per reports/latest.md Catalog #316 section"
  contest_cuda: "0.2053300290 [contest-CUDA T4], archive 9cb989cef519, per reports/latest.md Catalog #316 section"
predicted_band_semantics: "prediction_only; no score claim; no promotion authority"
state_mutation: "none"
---

# Rate-Attack Novel Vectors Design Memo - 2026-05-18

## 0. Executive Verdict

This memo turns the operator's 13-vector rate-attack seed into a canonical
Codex-goal-loop routing surface. It is intentionally design-only: no state
files, source files, tools, git state, or dispatch claims were edited.

The central adversarial finding is simple: a pure post-hoc recompression lane is
not the high-EV path right now. `reports/latest.md` records that 8 validated
contest archives are already at compression-ratio saturation and that the best
aggregate pre-entropy Wyner-Ziv sweep was below the 0.001 leaderboard precision
floor. The rate term is:

```text
rate_score_per_byte = 25 / 37,545,489 = 6.657e-7
1 KiB saved  -> 0.00068 score
7.5 KiB saved -> 0.0050 score
15 KiB saved  -> 0.0100 score
30 KiB saved  -> 0.0200 score
```

Therefore the TOP-3 should attack bytes before generic entropy coding, not
repack already-saturated archives:

1. **RATE-OP-1: A1+B3+M3 stable-orbit packet diet.** Use SegNet argmax-margin
   interior pixels (A1), STC/Brotli marginal costs (B3), and score-Fisher
   per-bit allocation (M3) to remove or down-code low-scorer-impact pre-entropy
   payload before compression. Predicted DeltaS: `[-0.010, -0.003]`
   prediction only.
2. **RATE-OP-2: M2 tropical argmax boundary grammar.** Convert the discrete
   SegNet scorer surface into a tropical/polyhedral label-boundary grammar,
   only for archives/substrates whose bytes actually drive SegNet-visible
   frame-1 content. Predicted DeltaS: `[-0.012, -0.004]` prediction only.
3. **RATE-OP-3: B1+B2 decoy/mosaic residual basis.** Split the renderer into
   expected-statistics decoy baseline plus small per-pair specialist residuals,
   with a shared-backbone mosaic runtime that stays within exact-inflate limits.
   Predicted DeltaS: `[-0.018, -0.006]` prediction only, higher engineering
   risk.

The memo's canonical consumer is the cathedral autopilot. The output below is
written as extractable task blocks; the next Codex goal loop can ingest them as
frontier-routing work without treating any predicted band as score evidence.

## 1. Preflight Surfaces Read

Read before writing:

- `AGENTS.md`: role split, append-only research memo discipline, no state
  mutation without lane preflight, exact-eval custody, no phantom score claims.
- `CLAUDE.md` top contracts: race-mode inversion, HNeRV parity 13 lessons,
  unique-and-complete per method, scaffold must be complete or research-only,
  predicted-band Dykstra discipline, phantom-score/research-sidecar failures.
- `PROGRAM.md`: mission, score formula, contest-compliance boundaries.
- `reports/latest.md` frontier section: current CPU/CUDA split, Q4 saturation
  result, phantom-score warning, asymptotic-pursuit readiness matrix.
- Latest directives/memos: deeper granularity directive, deterministic optimizer
  design-constraint directive, cross-stack synthesis, cheap-probe pose-axis
  routing directive, OP-SYN-1 Codex findings.

This memo does not register a lane because the operator explicitly limited
this turn to exactly one `.omx/research` memo and prohibited `.omx/state`
edits. The proposed future lane id is:

```text
lane_rate_attack_novel_vectors_design_20260518
```

## 2. Seed Vector Normalization

The prompt supplied IDs A1-A3, B1-B4, C1-C3, and M1-M3. I normalized them to
the latest in-repo meanings that match the rate-attack line:

| ID | Normalized vector | Primary role | Immediate verdict |
|---|---|---|---|
| A1 | SABOR / stable-argmax boundary-orbit encoding | SegNet-invisible rate removal | TOP-3 core |
| A2 | Stride-2 stem blindspot / high-frequency allocation | SegNet blindspot search | Supporting probe |
| A3 | Continuous-curvature operating-point sweep | Proxy/auth drift hardening | Supporting training guard |
| B1 | Decoy-target rendering | Expected-statistics residual coding | TOP-3 with B2 |
| B2 | Mosaic encoder swarm | Per-pair specialist renderer routing | TOP-3 with B1 |
| B3 | STC/Brotli-incompressible-aware byte allocation | Pre-entropy rate allocator | TOP-3 core |
| B4 | Cheap-prototype probe discipline | Goal-loop sequencing | Mandatory process hook |
| C1 | ACH-driven substrate ranking | Hypothesis ranking | Autopilot feature |
| C2 | Key-assumption matrix | Fail-closed premise surfacing | Autopilot blocker |
| C3 | Rotating devil's advocate | Adversarial review process | Council hygiene |
| M1 | Brotli + cooperative-receiver dictionary | Universal low-risk rate helper | Secondary |
| M2 | Tropical SegNet argmax encoding | Discrete scorer grammar | TOP-3 core |
| M3 | Per-bit Fisher-weighted FP4 lattice | Score-aware pre-entropy quantization | TOP-3 core |

Alternative M-series meanings exist in older MIT/LIDS signal-processing memos
(RLNC, BATS, compressed sensing). Those remain related but are not the primary
rate-attack seed here because this turn asks for a top-priority design memo for
the Codex goal loop, and the deep-math M1-M3 mechanisms map more directly onto
current archive bytes and scorer geometry.

## 3. Contest-Compliance Envelope

Any selected vector must satisfy all of the following before it can leave
research-only status:

1. **Scorer-free inflate.** No SegNet/PoseNet/scorer imports or learned scorer
   inference at inflate time. Scorers are compress-time/training-time only.
2. **Archive-charged bytes only.** Positive rate savings must be on
   `archive.zip` members or pre-entropy bytes that become `archive.zip` bytes.
   `.pt` research sidecars, state dicts, `.omx/tmp` payloads, and analysis
   artifacts are not score bytes.
3. **Exact runtime signature.** Future implementations must preserve
   `inflate.sh archive_dir output_dir file_list` and deterministic output
   across CPU/CUDA promotion axes.
4. **No hidden runtime.** No network downloads, local paths, unpinned
   dependencies, or runtime-scored side channels.
5. **Byte consumption proof.** A targeted byte mutation must change downstream
   output in the intended section, or the vector is a no-op/phantom candidate.
6. **Axis separation.** CPU/CUDA/macOS advisory results are separate evidence
   spaces; this memo contains no score result on any axis.
7. **Dykstra check.** Every prediction below is conditional on the intersection
   of rate, SegNet, PoseNet, inflate LOC, dependency closure, and runtime
   determinism constraints being non-empty. Dykstra feasibility is a screening
   discipline, not proof of future score.

## 4. Adversarial Feasibility Matrix

| Vector | Feasibility | Contest risk | Rate impact | Scorer-geometry fit | Implementation surface |
|---|---|---|---|---|---|
| A1 SABOR | Medium-high | Low if compress-time only | High only if it removes/downcodes pre-entropy pixels, not if it appends masks | Strong: SegNet argmax is discrete and margin-rich | Sensitivity map + encoder bit allocator |
| A2 S2SBS | Medium | Medium: high-frequency noise can perturb PoseNet | Medium-low | Good for SegNet stem; weak for PoseNet | Frequency-band probe, not standalone |
| A3 curvature sweep | High | Low | Indirect | Good proxy/auth stabilizer | Training/loss schedule, not rate lane |
| B1 decoy | Medium | Medium: decoy baseline must still reconstruct scorer-visible content | Medium-high | Good if baseline captures road/pose statistics | New substrate grammar |
| B2 mosaic | Medium | Medium-high: inflate LOC and class-label overhead | High if shared backbone is real | Strong for hard-pair routing | Runtime dispatch + per-pair class labels |
| B3 STC/Brotli allocator | High | Low | Low alone; high when upstream payload changes | Good cost-allocation lens | Bit allocator / packet compiler |
| B4 cheap probe | High | None | Indirect | Process-only | Cathedral task gate |
| C1 ACH ranking | High | None | Indirect | Process-only | Autopilot feature |
| C2 assumptions | High | None | Indirect | Process-only | Autopilot blocker |
| C3 devil advocate | High | None | Indirect | Process-only | Review discipline |
| M1 BR-CR-DICT | Medium | Low if dictionary code is reviewable and counted correctly | Low on saturated archives | Weak direct scorer fit | Packet compiler / brotli strategy |
| M2 TROP-ARGMAX | Medium | Medium: only valid where label/argmax grammar is real contest payload | High if paired with boundary grammar | Very strong: SegNet is argmax | Substrate-engineering grammar |
| M3 FW-FP4 | Medium-high | Low if Fisher is score-derived and archive bytes shrink | Medium | Strong if Fisher is computed through scorer loss | Quantizer / packet compiler |

## 5. Why These TOP-3

### TOP-1: RATE-OP-1 A1+B3+M3 stable-orbit packet diet

This is the highest EV because it connects three already-compatible surfaces:

- A1 says many SegNet-visible pixels live deep inside the same argmax class.
- B3 says byte allocation should be a cost minimization problem, like STC.
- M3 says bit widths should be Fisher/score-gradient weighted, not weight-MSE
  weighted.

This avoids the known failure mode of recompressing saturated archive bytes.
The target is the pre-entropy representation that generates those bytes. The
first artifact should be an xray report over one real frontier-family archive:
which sections, tensors, pixels, or selector modes can be downcoded before
Brotli/LZMA without changing scorer-relevant output?

**Predicted DeltaS:** `[-0.010, -0.003]` prediction only.

**Dykstra caveat:** this band requires non-empty intersection of:

- at least 4.5-15 KiB pre-entropy reduction after final compression;
- no SegNet argmax flip on frame 1 beyond the baseline tolerance;
- no PoseNet regression from high-frequency or low-Fisher perturbations;
- unchanged exact inflate runtime or a <=100/200 LOC reviewable helper;
- byte mutation proof on the consumed archive section.

### TOP-2: RATE-OP-2 M2 tropical argmax boundary grammar

SegNet's contribution is a 5-class argmax disagreement, not continuous RGB
MSE. A tropical/polyhedral grammar is mathematically aligned with that scorer:
encode the cell/boundary structure needed to preserve argmax, not the raw
RGB/value structure. This can dominate smooth proxy losses if it is bound to
real contest bytes.

The risk is integration. If the current frontier packet has no explicit
mask/logit/argmax payload, this becomes substrate engineering rather than a
bolt-on. It must be export-first and monolithic `0.bin`-first.

**Predicted DeltaS:** `[-0.012, -0.004]` prediction only.

**Dykstra caveat:** feasibility fails if the tropical grammar either:

- requires a large boundary sidecar that erases its rate savings;
- changes frame outputs without full-frame parity tests;
- relies on scorer inference or learned segmentation at inflate time;
- applies only to research labels rather than contest archive payload.

### TOP-3: RATE-OP-3 B1+B2 decoy/mosaic residual basis

The current HNeRV-family plateau suggests a single generic renderer may be a
bad rate allocation for all 600 pairs. B1 removes expected statistics first;
B2 routes remaining residuals to small specialist heads. This is not generic
recompression. It changes the modeling basis before entropy coding.

The risk is runtime complexity. A naive four-renderer swarm violates the
reviewability and inflate-LOC discipline. The only viable design is a shared
backbone with tiny per-class heads and a 2-bit per-pair route table inside
`0.bin`.

**Predicted DeltaS:** `[-0.018, -0.006]` prediction only.

**Dykstra caveat:** feasibility requires the route table plus specialist heads
to save at least 9-27 KiB net after final compression while keeping both
SegNet and PoseNet distortions within baseline movement. If route labels,
heads, or dispatch code exceed the savings, this drops below RATE-OP-1.

## 6. Dykstra Feasibility Envelope

Treat each candidate as a projection into this constraint intersection:

```text
C_rate:      final archive bytes lower by enough to matter
C_seg:       frame-1 argmax disagreement does not increase
C_pose:      pair YUV6 PoseNet first-6 MSE does not increase materially
C_inflate:   exact inflate signature, no scorer loads, deterministic output
C_runtime:   <=100 LOC target or <=200 LOC with waiver; <=2 dependencies
C_custody:   archive SHA, member SHA, runtime tree SHA, byte mutation proof
C_consumer:  cathedral autopilot can consume the result without custom prose
```

The alternating-projection screen should use this order:

1. Project onto `C_custody` first: reject any research sidecar or phantom byte.
2. Project onto `C_rate`: require at least 4.5 KiB plausible net reduction for
   a top-priority pure-rate branch, or a distortion-improvement reason.
3. Project onto `C_seg` and `C_pose`: use cheap local scorer/proxy only as
   prediction; no score claim.
4. Project onto `C_inflate` and `C_runtime`: reject if the idea needs scorer
   imports, network, hidden dependencies, or opaque runtime.
5. Project onto `C_consumer`: emit typed artifacts that cathedral autopilot can
   rank.

Any vector that cannot survive these projections remains `research_only=true`.

## 7. Canonical-vs-Unique Decision Per Layer

| Layer | Decision | Rationale |
|---|---|---|
| Frontier scan and axis labels | Adopt canonical | Catalog #316 is the authority for current best CPU/CUDA anchors. |
| Archive grammar | Unique per method | A1/M2/B2 each needs method-specific pre-entropy grammar; forcing one generic grammar would suppress signal. |
| Runtime/inflate discipline | Adopt canonical | Exact signature, scorer-free inflate, LOC/dependency budget are universal bug-class guards. |
| Sensitivity/Fisher inputs | Adopt canonical where available | `master_gradient`, `sensitivity_map`, and xray outputs are existing score-derived signals. |
| Bit allocation | Fork locally for RATE-OP-1 | B3+M3 needs a rate-attack-specific cost model over pre-entropy bytes, then can feed canonical allocator outputs. |
| Cathedral autopilot consumer | Adopt canonical as primary consumer | The memo is useful only if autopilot can rank and route the next work. |
| Probe discipline | Adopt B4 as mandatory | Cheap-probe-first protects operator attention and prevents large substrate builds from running on untested premises. |

## 8. Consumer Hooks

Cathedral autopilot is the canonical consumer. All other hooks feed it.

1. **Sensitivity map:** RATE-OP-1 contributes a joint map:
   `seg_margin * fisher_inverse * brotli_marginal_cost`; RATE-OP-2 contributes
   tropical boundary density; RATE-OP-3 contributes per-pair route hardness.
2. **Pareto constraint:** each candidate emits `(delta_bytes_pred,
   delta_seg_pred, delta_pose_pred, runtime_risk, custody_status)` so Pareto
   blockers can reject non-feasible rows.
3. **Bit allocator:** RATE-OP-1 and RATE-OP-2 directly feed bit allocation.
   RATE-OP-3 feeds per-pair/head allocation.
4. **Cathedral autopilot:** add rows as `rate_attack_candidate` entries with
   `prediction_only=true`, `score_claim=false`, `promotion_eligible=false`.
5. **Continual learning posterior:** only after empirical probes land, append
   anchor rows with exact archive/runtime custody. This memo writes no posterior.
6. **Probe disambiguator:** first disambiguator compares RATE-OP-1 vs RATE-OP-2
   on the same archive family and byte budget; second compares RATE-OP-3 against
   a single-monolith control.

## 9. Canonical Task Blocks

These blocks are deliberately regular so `canonical_task_status` extraction can
pull them into the Codex goal loop later. They are not state updates.

```yaml
TASK_BLOCK:
  task_id: rate_attack_op_1_stable_orbit_packet_diet
  priority: P0
  owner_suggestion: codex
  canonical_consumer: cathedral_autopilot
  source_vectors: [A1_SABOR, B3_STC_BROTLI_ALLOCATOR, M3_FISHER_FP4]
  status: proposed
  score_claim: false
  promotion_eligible: false
  provider_spend: false
  predicted_delta_s: "[-0.010, -0.003] prediction_only"
  dykstra_gate: "C_rate and C_seg and C_pose and C_inflate and C_custody must all be nonempty"
  first_artifact: "experiments/results/rate_attack_op1_stable_orbit_packet_diet_*/xray_manifest.json"
  implementation_surface_future:
    - "tac.sensitivity_map"
    - "tac.bit_allocator"
    - "packet compiler or archive grammar builder"
    - "tools/cathedral_autopilot_autonomous_loop.py consumer row"
  acceptance_checks_future:
    - "proves >=4.5 KiB plausible final archive-byte reduction or explicit distortion benefit"
    - "byte mutation changes consumed output section"
    - "no scorer import at inflate"
    - "CPU/CUDA axis labels remain prediction-only until exact eval"
```

```yaml
TASK_BLOCK:
  task_id: rate_attack_op_2_tropical_argmax_boundary_grammar
  priority: P1
  owner_suggestion: codex
  canonical_consumer: cathedral_autopilot
  source_vectors: [M2_TROPICAL_ARGMAX, A1_SABOR, C2_KEY_ASSUMPTION_CHECK]
  status: proposed
  score_claim: false
  promotion_eligible: false
  provider_spend: false
  predicted_delta_s: "[-0.012, -0.004] prediction_only"
  dykstra_gate: "valid only if argmax/boundary grammar maps to contest archive bytes, not research labels"
  first_artifact: "experiments/results/rate_attack_op2_tropical_argmax_boundary_*/feasibility_report.json"
  implementation_surface_future:
    - "substrate-specific monolithic 0.bin grammar"
    - "tac.xray segnet boundary primitive"
    - "bit allocator boundary tiers"
    - "cathedral autopilot rate_attack_candidate row"
  acceptance_checks_future:
    - "declares export-first archive grammar before trainer changes"
    - "full-frame inflate parity checked before any method verdict"
    - "boundary payload overhead is below predicted savings"
    - "no scorer loads or labels at inflate"
```

```yaml
TASK_BLOCK:
  task_id: rate_attack_op_3_decoy_mosaic_residual_basis
  priority: P2
  owner_suggestion: codex
  canonical_consumer: cathedral_autopilot
  source_vectors: [B1_DECOY_RENDERING, B2_MOSAIC_ENCODER_SWARM, B4_CHEAP_PROBE]
  status: proposed
  score_claim: false
  promotion_eligible: false
  provider_spend: false
  predicted_delta_s: "[-0.018, -0.006] prediction_only"
  dykstra_gate: "route table plus specialist heads must save net bytes and fit exact inflate reviewability"
  first_artifact: "experiments/results/rate_attack_op3_decoy_mosaic_probe_*/route_entropy_report.json"
  implementation_surface_future:
    - "shared-backbone plus tiny-head substrate design memo"
    - "2-bit per-pair route table in 0.bin"
    - "single-monolith control probe"
    - "cathedral autopilot mosaic-vs-monolith feature"
  acceptance_checks_future:
    - "10-50 pair cheap probe before full trainer"
    - "inflate runtime <=200 LOC with explicit waiver if >100"
    - "class labels are archive-charged and deterministic"
    - "single-monolith control included"
```

```yaml
TASK_BLOCK:
  task_id: rate_attack_process_ach_assumption_autopilot_features
  priority: P3
  owner_suggestion: codex
  canonical_consumer: cathedral_autopilot
  source_vectors: [C1_ACH_RANKING, C2_KEY_ASSUMPTION_MATRIX, C3_ROTATING_DEVIL_ADVOCATE, B4_CHEAP_PROBE]
  status: proposed
  score_claim: false
  promotion_eligible: false
  provider_spend: false
  predicted_delta_s: "indirect; no standalone score prediction"
  dykstra_gate: "process features must block bad candidates without suppressing measured frontier movement"
  first_artifact: "reports/rate_attack_autopilot_feature_matrix_*.json"
  implementation_surface_future:
    - "cathedral autopilot ranker features"
    - "canonical_task_status blocker fields"
    - "probe-outcome rows after empirical anchors only"
  acceptance_checks_future:
    - "every rate candidate has explicit disconfirming assumptions"
    - "cheap-probe verdict exists before >$1 provider spend"
    - "autopilot can sort without reading prose memo"
```

## 10. Vector-by-Vector Notes

### A1 SABOR

High promise, but only if converted from a margin observation into an archive
grammar. Do not ship a SABOR mask sidecar unless the sidecar replaces more
bytes than it adds. The correct first product is a margin-stratified byte
allocation xray over a real frontier-family packet.

### A2 S2SBS

Useful as a secondary feature inside RATE-OP-1. Do not rely on high-frequency
noise as "free": PoseNet sees pairwise YUV6 and may be more sensitive than
SegNet in the current operating point. A2 needs a PoseNet regression guard.

### A3 Continuous-Curvature Sweep

Not a rate attack by itself. It is a training guard that prevents a candidate
from optimizing one fragile point on the proxy/auth curve. Use it as a
requirement for RATE-OP-3 or any full substrate build.

### B1 Decoy Rendering

Potentially strong when paired with B2. Alone, "expected statistics" can become
a vague baseline that loses scorer-specific detail. The cheap probe must report
route entropy and residual entropy, not just visuals.

### B2 Mosaic Encoder Swarm

High upside, high integration risk. The runtime must be a shared backbone with
small heads, not four separate renderers. If the route table consumes more than
the specialist savings, defer.

### B3 STC/Brotli Allocator

Strong engineering discipline. Weak as post-hoc recompression because current
archives are saturated. Strong as a pre-entropy cost model that decides which
bytes should exist before Brotli sees them.

### B4 Cheap-Probe Discipline

Mandatory. Any rate-attack branch above P2 must have a cheap principle probe
before provider spend above $1.

### C1-C3 Structured Analytic Process

Do not count these as score-moving vectors. Their value is to become autopilot
features and blockers: ACH inconsistency count, key-assumption risk, and
rotating adversarial review status.

### M1 Brotli + Cooperative-Receiver Dictionary

Low-risk but likely low-yield on current packets. Elevate only if the dictionary
is trained for a new pre-entropy stream, not after final archive saturation.

### M2 Tropical Argmax

Best theoretical scorer-geometry fit. Risk is byte reality: if there is no
contest-charged argmax-like payload, it becomes substrate engineering. Keep it
export-first.

### M3 Fisher-Weighted FP4

Good because it directly repairs the Catalog #123 failure mode: not weight-MSE
saliency, but score-derived Fisher/gradient saliency. Pure byte savings must
clear the 4.5-15 KiB materiality threshold or be paired with distortion gains.

## 11. Citations and Source Pointers

Local canonical sources:

- `reports/latest.md` Catalog #316 frontier section and Q4 pivot.
- `.omx/research/expert_team_aerospace_stealth_analytic_alien_tech_20260513.md`
  plus ledgers 01-03 for A/B/C seed vectors.
- `.omx/research/deep_math_geometry_manifolds_synthesis_20260514.md` for
  M1-M3 rate mechanisms.
- `.omx/research/comprehensive_analytical_surfaces_inventory_plus_synthesis_design_memo_20260518.md`
  for master-gradient/sensitivity/cathedral hook gaps.
- `.omx/research/codex_findings_op_syn_1_extract_all_manifest_20260518T202947Z_codex.md`
  for current master-gradient projector authority limits.

External primary/paper/doc sources browsed for this memo:

- Brotli format: IETF RFC 7932, <https://datatracker.ietf.org/doc/rfc7932/>.
- Syndrome-trellis codes: Filler, Judas, Fridrich, IEEE TIFS 2011,
  DOI <https://doi.org/10.1109/TIFS.2011.2134094>, author PDF
  <https://dde.binghamton.edu/filler/pdf/Fill10tifs-stc.pdf>.
- Wyner-Ziv source coding with decoder side information: IEEE T-IT 1976
  mirror PDF <https://www.mit.edu/~6.454/www_fall_2001/kusuma/wynerziv.pdf>.
- Compressed sensing / sparse recovery: Candes, Romberg, Tao arXiv
  <https://arxiv.org/abs/math/0409186>; Donoho DOI
  <https://doi.org/10.1109/TIT.2006.871582>.
- Network information flow / coding: Ahlswede, Cai, Li, Yeung DOI
  <https://doi.org/10.1109/18.850663>.
- BATS codes: Yang and Yeung arXiv <https://arxiv.org/abs/1206.5365>.
- Dykstra-style feasibility background: arXiv overview of Dykstra variants
  <https://arxiv.org/abs/2001.06747>.

## 12. Closeout

This memo deliberately produces one artifact: the research/design memo itself.
It does not claim score movement. It does not promote a lane. It does not write
state. The intended next consumer is the Codex goal loop via cathedral autopilot:
ingest the task blocks, run RATE-OP-1 first, and keep all predicted bands tagged
as prediction-only until exact archive/runtime custody and paired axis evidence
exist.
