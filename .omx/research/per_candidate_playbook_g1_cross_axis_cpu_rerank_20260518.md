# Per-Candidate Playbook: G1 — Cross-Axis CPU Re-Rank (SATURATION-INDEPENDENT)

**Parent framework**: `.omx/research/dynamic_per_candidate_composition_framework_all_canonical_apparatus_composed_20260518.md` §14.1
**Lane**: per parent framework `lane_dynamic_per_candidate_composition_framework_20260518` derived
**Routing**: Codex (already routed via `83440e8a5`)
**Mission alignment**: frontier-directed authority hardening per CLAUDE.md "Mission alignment" + Hotz binding directive PROCEED IMMEDIATELY

---

## Header

- **Candidate type**: any of OUR submission candidates with paired dual-eval (CPU + CUDA) anchors
- **Candidate archive_sha256 examples**: `6bae0201...` (fec6_fixed_huffman_k16; current CPU frontier 0.19205) + sister archives in `.omx/state/continual_learning_posterior.jsonl`
- **Candidate lane_id**: `pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean` + sister lanes
- **Asymptotic approach**: PLATEAU_ADJACENT per Catalog #309 (cross-axis re-rank is within-class optimization)
- **Budget**: $0 (re-ranking uses existing dual-eval data; no GPU spend)
- **Realized ΔS**: `0.0` [contest-CPU] for the existing-anchor rerank in `experiments/results/g1_cpu_axis_re_rank_20260518T214250Z/report.json`; no new score claim
- **Future upside condition**: requires a new paired Linux x86_64 CPU anchor or a candidate set where CPU-optimal differs from CUDA-optimal
- **Per-axis hardware classification**: AXIS-INVARIANT mechanism + CPU-axis-OPTIMAL operationally per primitive 14
- **Mathematical sub-paradigm (Tao partition)**: (c) Pareto-tightening — axis-conditioned RD frontier optimization
- **Predecessor probe outcome**: no INDEPENDENT/KILL/DEFER per `.omx/state/probe_outcomes.jsonl`; G1 is novel territory per SYNTHESIS-V2

## Empirical anchor (per primitive 1 = master_gradient NOT required; per primitive 14 = per-axis matrix is PRIMARY)

The G1 probe is grounded in two authority layers:

1. **Paired-axis priors**: prior HNeRV-family archives show material CPU/CUDA
   gaps, useful for deciding what to probe.
2. **Current execution artifact**: Codex's 2026-05-18 existing-anchor rerank
   found no current CPU-axis frontier improvement (`actual_delta_s=0.0`).

The paired-axis priors are NOT a CUDA-to-CPU conversion factor and are NOT a
score claim.

| Archive | Lane | CPU [contest-CPU] | CUDA [contest-CUDA] | Δ (CUDA-CPU) |
|---|---|---|---|---|
| PR102 | rem2 | 0.19538 | 0.22839 | +0.033 |
| PR107 | apogee | 0.19664 | 0.22936 | +0.033 |
| PR101 GOLD | quantizr | 0.193 | ~0.226 (implied) | ~+0.033 |

**Current G1 artifact**: `experiments/results/g1_cpu_axis_re_rank_20260518T214250Z/report.json` scanned 194 canonical anchors, found 55 qualifying CPU anchors, and returned `FRONTIER_STABLE_VIA_RE_RANK` with `delta_vs_current_frontier=0.0`.

## Composition plan (per the framework's CompositionPlan output for this candidate)

| Primitive # | Primitive name | Role for G1 | Per-candidate adaptation |
|---|---|---|---|
| 1 | master_gradient | NOT-REQUIRED (re-rank uses whole-archive scores, not per-byte gradient) | n/a |
| 2 | Venn classifier | NOT-REQUIRED (G1 is per-archive, not per-byte) | n/a |
| 3 | per-pair / per-frame / byte-level granularity | NOT-REQUIRED (per-archive granularity) | n/a |
| 4 | hard-pair atlas + sensitivity_map | NOT-REQUIRED | n/a |
| 5 | composition_alpha N-way | TRIVIAL (single-primitive G1; α=1.0) | n/a |
| 6 | Wyner-Ziv deliverability | NOT-REQUIRED (G1 doesn't modify archive bytes) | n/a |
| 7 | probe_outcomes ledger | ACTIVE (G1 outcome registers new entry per Catalog #313) | per candidate set |
| 8 | xray observability | NOT-REQUIRED at the primitive level | n/a |
| 9 | cathedral autopilot v2 cascade | ACTIVE (cascade re-ranks per-archive candidates by CPU-axis score per Catalog #319 Q3) | per archive set |
| 10 | null_space_exploiter | NOT-REQUIRED (G1 doesn't touch null-space) | n/a |
| 11 | procedural_codebook_generator | NOT-REQUIRED | n/a |
| 12 | freezing exploits | NOT-REQUIRED | n/a |
| 13 | A1-SPECIALIZED binary | NOT-REQUIRED | n/a |
| 14 | per-axis hardware exploit matrix | **PRIMARY** (G1 IS the matrix classifier's CPU-axis routing application) | per archive set |

**Key insight**: G1 is the simplest possible composition — primitives 7 + 9 + 14 ACTIVE; all 11 others NOT-REQUIRED. The composition is trivial; the current artifact is a zero-delta authority guardrail, not a frontier move.

## Layer 3 bilevel optimizer state

- **OUTER tier verdict** (§4.2): codec_config = N/A (G1 doesn't change codec); per_primitive_alpha = 1.0 (single-primitive)
- **MIDDLE tier verdict** (§4.3): class_shift_required = False (within-class optimization per PLATEAU_ADJACENT default)
- **INNER tier verdict** (§4.4): N/A (G1 is per-archive whole-score re-rank, not per-substrate parameter optimization)
- **INNERMOST tier verdict** (§4.5): N/A (G1 doesn't touch per-pair byte-level decisions)

The G1 plan is structurally LAYER-3-SKIPPABLE; the framework's per-candidate composition algorithm fast-paths candidates whose only applicable primitive is at OUTER tier (per-archive re-rank).

## Anti-arbitrariness foundation per parent framework §6

### (a) HARD-EARNED vs CARGO-CULTED classification per Catalog #303

**HARD-EARNED-EMPIRICAL-PROBE**: paired-axis gap priors justify the probe; the executed existing-anchor rerank measured `actual_delta_s=0.0`.

### (b) Empirical anchor cite per Catalog #287

- `[empirical:experiments/results/g1_cpu_axis_re_rank_20260518T214250Z/report.json]`
- `[contest-CPU]` current frontier 0.1920513168811056
- G1 realized delta 0.0 on existing anchors
- Paired CPU/CUDA gaps remain per-archive priors only

### (c) Per-candidate adaptation evidence

Re-rank per-candidate's existing dual-eval data. Each candidate set produces a different CPU-axis ranking; the framework's per-candidate orchestration adapts per the candidate set membership.

**Current example**: for the canonical anchor set, the CPU-best existing
archive is still PR101/fec6 (`6bae0201...`) at 0.1920513168811056
[contest-CPU]. G1 outcome: keep the current frontier; register the zero-delta
guardrail so future consumers do not infer a hidden rerank win.

### (d) Legal-receiver-path classification per Catalog #6 + HNeRV parity L4

**NO_RECEIVER_NEEDED** — G1 doesn't modify archives; no inflate code change; no rate hit; no Catalog #6 strict-scorer-rule concern.

### (e) Dykstra-feasibility intersection per Catalog #296

**TRIVIAL** — single-primitive (G1 only); no composition; Pareto-feasibility is trivially satisfied.

## Cost + time

- **Cost estimate**: $0 (local re-ranking using existing dual-eval state)
- **Time estimate**: completed by Codex in this session
- **Predecessor**: none (G1 is operator-routable IMMEDIATELY per Hotz binding directive)
- **Successors**: future G1 authority-upgrade work should replace metadata-bucket grouping with genuine family classification via provenance/custody validators before reranking new candidate sets

## Reactivation criteria per CLAUDE.md "Forbidden premature KILL"

If G1 returns INDEPENDENT verdict (no candidate set has best-CPU ≠ best-CUDA):

1. **Reactivation Path 1**: Re-run G1 with extended per-candidate set (include public PR archives via reverse-engineered dual-eval if license-compatible)
2. **Reactivation Path 2**: Re-run G1 with per-cell-Venn-class-conditioned re-rank (per primitive 2) — escalate to non-trivial Layer 2 composition
3. **Reactivation Path 3**: Re-run G1 with per-hardest-pair-conditioned re-rank (per primitive 4) — escalate to per-pair granularity per primitive 3
4. **Reactivation Path 4**: Pivot to F1-as-A2 playbook (SATURATION-INDEPENDENT sister)

If G1 returns DEFER (compute infrastructure issue):

1. Re-run G1 with snapshot-of-state preserved
2. Re-run G1 with explicit per-candidate set override
3. Escalate to operator review per Catalog #325

## Operator-routable next action

- **One-line command**: `.venv/bin/python tools/probe_g1_cpu_axis_re_rank.py --output-dir experiments/results/g1_cpu_axis_re_rank_20260518T214250Z --json`
- **Routing destination**: Codex
- **Sister directive**: `83440e8a5`
- **Harvest evidence**: `experiments/results/g1_cpu_axis_re_rank_20260518T214250Z/report.json`
- **Verdict**: `FRONTIER_STABLE_VIA_RE_RANK`; no new score claim

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution**: N/A (G1 is per-archive whole-score, not per-byte)
2. **Pareto constraint**: ACTIVE (G1 IS a Pareto-tightening primitive per Tao partition (c))
3. **Bit-allocator hook**: N/A (G1 doesn't allocate bits)
4. **Cathedral autopilot dispatch hook**: **PRIMARY** (G1 IS the canonical CPU-axis-ranking application of cathedral autopilot v2 cascade)
5. **Continual-learning posterior update**: ACTIVE (G1 outcome registers via probe_outcomes ledger per Catalog #313)
6. **Probe-disambiguator**: ACTIVE (G1 disambiguates between CUDA-axis vs CPU-axis optimal candidate selection)

## Observability surface per Catalog #305

1. **Inspectable per layer**: G1 produces per-candidate CPU-axis ranking; inspectable per `tools/scan_best_anchor_per_axis.py` per Catalog #316
2. **Decomposable per signal**: G1 decomposes per-candidate score into CPU + CUDA + Δ (CUDA-CPU); per-archive cited
3. **Diff-able across runs**: Two G1 invocations on different candidate sets can be diffed
4. **Queryable post-hoc**: `.omx/state/continual_learning_posterior.jsonl` query per archive_sha256 per axis
5. **Cite-able**: `[contest-CUDA]` + `[contest-CPU]` axis tags per CLAUDE.md "Submission auth eval"
6. **Counterfactual-able**: "what if we submit CUDA-best instead of CPU-best?" → re-run G1 with `--rank-by cuda` instead of `--rank-by cpu`

## Cargo-cult audit per Catalog #303

| Assumption | Classification | Rationale |
|---|---|---|
| PR102 +0.033 generalizes to all HNeRV-family | FALSE_AS_CONVERSION__HARD_EARNED_AS_PER_ARCHIVE_PRIOR | Paired-axis gaps justify probes only; Codex G1 existing-anchor rerank measured 0.0 |
| CPU-axis is the leaderboard | HARD-EARNED-VERIFIED | per CLAUDE.md "Submission auth eval" non-negotiable + Hotz binding |
| G1 re-rank uses existing data | HARD-EARNED-VERIFIED | per `.omx/state/continual_learning_posterior.jsonl` + G1 report |
| G1 current score movement | HARD-EARNED-ZERO-DELTA | `FRONTIER_STABLE_VIA_RE_RANK`; `actual_delta_s=0.0` |

— Per-candidate playbook for G1 cross-axis CPU re-rank per parent framework §14.1
