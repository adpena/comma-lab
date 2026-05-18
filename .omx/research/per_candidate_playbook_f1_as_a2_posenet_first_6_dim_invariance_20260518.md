# Per-Candidate Playbook: F1-as-A2 — PoseNet First-6-Dim Invariance Steganography (SATURATION-INDEPENDENT)

**Parent framework**: `.omx/research/dynamic_per_candidate_composition_framework_all_canonical_apparatus_composed_20260518.md` §14.2
**Lane**: per parent framework derived
**Routing**: Codex (already routed via `83440e8a5` probe; build conditional on probe PASS)
**Mission alignment**: frontier_breaking per CLAUDE.md "Mission alignment" + META-audit CONFLATE_DECLARATIVE_WITH_PHYSICAL correction

---

## Header

- **Candidate type**: any per-frame RGB sequence that PoseNet's first-6-dim is invariant to (NOT "free bytes in dim 7-12"; this IS adversarial-steganographic perturbations to RGB that produce same PoseNet first-6-dim output per META-audit's correction)
- **Candidate archive_sha256 examples**: any frontier archive; F1-as-A2 is a per-pair perturbation overlay
- **Candidate lane_id**: TBD per probe verdict landing
- **Asymptotic approach**: PLATEAU_ADJACENT per Catalog #309 (scorer-feature-invariance is within-class; no substrate class shift)
- **Budget**: $0 probe + $1-3 build (conditional on probe PASS)
- **Predicted aggregate ΔS band**: `[-0.012, -0.004]` [contest-CPU] per HARD-EARNED-STRUCTURAL via upstream/modules.py:84 source-verification
- **Per-axis hardware classification**: AXIS-INVARIANT per primitive 14 (Hydra dims 7-12 are scorer-invariant on both CPU and CUDA per source line `[..., : h.out // 2]`)
- **Mathematical sub-paradigm (Tao partition)**: (a) MI-min Wyner-Ziv — zero-rate side channel through scorer-feature-invariance
- **Predecessor probe outcome**: none per `.omx/state/probe_outcomes.jsonl`; F1-as-A2 is novel territory per SYNTHESIS-V2

## CRITICAL: META-audit CONFLATE_DECLARATIVE_WITH_PHYSICAL correction

**Original F1 framing (BROKEN per META-audit Claim #1)**: "PoseNet dim 7-12 are unscored → free byte channel in archive"

**Why broken**: dim 7-12 are OUTPUTS of PoseNet's forward pass on RGB inputs; they don't exist in archive bytes. There's no encoder path FROM archive bytes TO "dim 7-12 territory".

**Canonical F1-as-A2 framing (WORKS per META-audit + framework Layer 1 §6.1.(d))**: "find RGB perturbations whose corresponding PoseNet first-6-dim output is INVARIANT (i.e., perturbations that change dim 7-12 but NOT dim 1-6). Those perturbations are SCORE-INVARIANT but pixel-level-distinct. Use them as steganographic carriers — encode information as choice of which scorer-invariant perturbation to apply."

This is STRUCTURALLY IDENTICAL to vector A2 (Adversarial steganography on scorer blind-spots — Fridrich + Yousfi PhD territory) applied specifically to the PoseNet first-6-dim invariance manifold.

## Source-level verification per upstream/modules.py:84

```python
# upstream/modules.py:84 (verified per Catalog #229 premise verification + Codex F1 finding):
return sum(
    (out1[h.name][..., : h.out // 2] - out2[h.name][..., : h.out // 2]).pow(2).mean(...)
    for h in self.hydra.heads if h.name in distortion_heads
)
```

Combined with `upstream/modules.py:26` (`HEADS = [Head('pose', 32, 12)]`): PoseNet outputs 12 dims; `compute_distortion` uses `[..., : h.out // 2]` slice with `h.out = 12` → only first 6 dims scored.

**Verdict per Catalog #229 PV**: HARD-EARNED-VERIFIED-FROM-SOURCE.

## Composition plan (per the framework's CompositionPlan output for this candidate)

| Primitive # | Primitive name | Role for F1-as-A2 | Per-candidate adaptation |
|---|---|---|---|
| 1 | master_gradient | **ACTIVE** (use per-pair gradient signature on Hydra dims 7-12 region to identify RGB perturbation directions that maximize dim 7-12 magnitude while preserving dim 0-5) | per per-pair fp64 gradient |
| 2 | Venn classifier | **ACTIVE** (classify per-pair RGB positions by their gradient-to-dim-7-12 vs gradient-to-dim-0-5 ratio; HIGH_PAIR_INVARIANT for high-ratio pairs) | per per-pair classification |
| 3 | per-pair / per-frame / byte-level granularity | **ACTIVE** at per-pair granularity | per per-pair perturbation magnitude |
| 4 | hard-pair atlas + sensitivity_map | **ACTIVE** (identify hardest pairs for dim 7-12 channel encoding capacity) | per hardest 100 pairs |
| 5 | composition_alpha N-way | **ACTIVE** (sub-additive composition for primitives 1 × 6 × 10) | α ≈ 0.7 expected per parent memo |
| 6 | Wyner-Ziv deliverability | **PRIMARY** (F1-as-A2 IS the canonical Wyner-Ziv example; bits ship through scorer-feature-invariance channel at zero rate cost) | per per-substrate proof |
| 7 | probe_outcomes ledger | ACTIVE (F1-as-A2 outcome registers per Catalog #313) | per candidate |
| 8 | xray observability | **ACTIVE** (xray scorer output to verify first-6-dim invariance per perturbation) | per perturbation magnitude |
| 9 | cathedral autopilot v2 cascade | ACTIVE (per-pair routing per Catalog #319 Q3) | per per-pair classification |
| 10 | null_space_exploiter | **ACTIVE** (RGB perturbations that produce identical PoseNet first-6-dim ARE null-space directions; the canonical null_space_exploit per Codex landing) | per per-pair null-space directions |
| 11 | procedural_codebook_generator | OPTIONAL (codebook of canonical perturbation directions) | per per-substrate codebook |
| 12 | freezing exploits | NOT-REQUIRED (F1-as-A2 doesn't freeze scorer) | n/a |
| 13 | A1-SPECIALIZED binary | NOT-REQUIRED (F1-as-A2 doesn't require binary distillation; perturbations modify RGB directly) | n/a |
| 14 | per-axis hardware exploit matrix | ACTIVE (AXIS-INVARIANT per Hydra dims invariant on both CPU and CUDA) | per axis routing |

**Key insight**: F1-as-A2 is the RICHEST composition in the framework — 11 primitives ACTIVE; only 3 NOT-REQUIRED. The composition leverages master_gradient + Venn classifier + sensitivity_map + Wyner-Ziv deliverability + null_space_exploiter jointly.

## Layer 3 bilevel optimizer state

- **OUTER tier verdict** (§4.2): codec_config = current; per_primitive_alpha = SUB-ADDITIVE composition (primitives 1 × 6 × 10 expected at α ≈ 0.7 per Catalog #322 v2)
- **MIDDLE tier verdict** (§4.3): class_shift_required = False (within-class per PLATEAU_ADJACENT default)
- **INNER tier verdict** (§4.4): Fisher matrix per-pair gradient signature on Hydra dims; Riem-Newton converged perturbation directions
- **INNERMOST tier verdict** (§4.5): per-pair RGB perturbation magnitude per cathedral autopilot v2 cascade routing

## Anti-arbitrariness foundation per parent framework §6

### (a) HARD-EARNED vs CARGO-CULTED classification per Catalog #303

**HARD-EARNED-VERIFIED-FROM-SOURCE** per upstream/modules.py:84 line `[..., : h.out // 2]` with h.out=12.

**Recoverability claim is CARGO-CULTED-PENDING-PROBE**: "encoder can freely set dims 7-12 without affecting forward pass" requires probe verification per Fridrich+Contrarian revision in PRIMARY § (probe gate at step 1).

### (b) Empirical anchor cite per Catalog #287

- `[source-trace:upstream/modules.py:84]` per Codex F1 finding source verification
- `[empirical:.omx/state/master_gradient_anchors.jsonl]` per per-pair gradient signature (PR101_lc_v2 `f174192aeadf...`)
- `[prediction]` per predicted ΔS band `[-0.012, -0.004]` pending probe PASS

### (c) Per-candidate adaptation evidence

Per-pair RGB perturbation directions differ per per-pair gradient signature. Two candidates with different archive bytes produce different per-pair master_gradient signatures → different perturbation directions.

**Example for candidate PR101_lc_v2** (sha `f174192aeadf...`):
- Per-pair |∇_θ Hydra-dims-7-12 / Hydra-dims-0-5| ratio distribution
- Top-100 hardest pairs by ratio identified per primitive 4 atlas
- Per-pair perturbation directions emitted via primitive 10 null_space_exploiter

### (d) Legal-receiver-path classification per Catalog #6 + HNeRV parity L4

**NO_RECEIVER_NEEDED** — perturbations modify RGB in-place; inflate.py reads modified RGB identically; scorer first-6-dim output unchanged. ZERO additional inflate code or deps. Per the META-audit's CONFLATE_DECLARATIVE_WITH_PHYSICAL correction: F1-as-A2 is a steganographic-perturbation primitive, not a free-bytes-in-archive primitive.

### (e) Dykstra-feasibility intersection per Catalog #296

Per primitives 1 + 6 + 10 composition; Boyd's alternating projections converge per the SUB-ADDITIVE prior per Catalog #322. Expected α ≈ 0.7 per the parent memo.

## Cost + time

- **Cost estimate**: $0 probe (CPU-only RGB perturbation generation + scorer forward) + $1-3 build (Modal A10G smoke for full per-pair encoding)
- **Time estimate**: ~1-2 days probe + ~2-3 days build conditional on probe PASS
- **Predecessor**: none (F1-as-A2 is operator-routable per Fridrich+Contrarian probe-FIRST revision per SYNTHESIS-V2 §3.2)
- **Successors**: F1-as-A2 verdict informs sister vector A2 (Adversarial steganography on scorer blind-spots; Fridrich + Yousfi PhD territory)

## Reactivation criteria per CLAUDE.md "Forbidden premature KILL"

If probe returns INDEPENDENT verdict (cannot find perturbations with first-6-dim invariance):

1. **Reactivation Path 1**: Re-run probe with extended perturbation magnitude range (test smaller / larger perturbations; UNIWARD-style L∞ penalty)
2. **Reactivation Path 2**: Re-run probe with per-pair Venn-class-conditioned perturbation (use HIGH_PAIR_INVARIANT pairs only per primitive 2)
3. **Reactivation Path 3**: Pivot to F3-F6 (deeper scorer layer invariance per META-audit reclaimability table; PoseNet vision/summary/ResBlock/Hydra trunk inverters)
4. **Reactivation Path 4**: Pivot to A2 direct (Fridrich+Yousfi adversarial steganography literature; canonical inverse-steganalysis framework per CLAUDE.md "Fridrich inverse steganalysis")

If build returns DEFER (perturbations work in probe but build doesn't ship bits at predicted ΔS):

1. Re-run build with extended per-pair perturbation set
2. Re-run build with composition_alpha override per primitive 5 (force ADDITIVE if ORTHO empirically verified)
3. Escalate to operator review per Catalog #325

## Operator-routable next action

- **One-line command**: routing directive `83440e8a5` (probe+build for F1) already landed; probe gate at step 1; Codex /goal LOOP picks up
- **Routing destination**: Codex (CPU-only probe) → Modal A10G (conditional build)
- **Sister directive**: `83440e8a5` (sister to G1 directive)
- **Expected harvest**: `experiments/results/f1_as_a2_probe_<utc>/report.json` (probe verdict) + `experiments/results/f1_as_a2_build_<utc>/report.json` (build score)

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution**: **PRIMARY** (F1-as-A2 IS the canonical scorer-feature-invariance sensitivity map at per-pair granularity)
2. **Pareto constraint**: ACTIVE (Wyner-Ziv deliverability per primitive 6 enforces Pareto-feasibility for zero-rate side channel)
3. **Bit-allocator hook**: **PRIMARY** (F1-as-A2's primary purpose is per-pair bit allocation to scorer-feature-invariant channel)
4. **Cathedral autopilot dispatch hook**: ACTIVE (per-pair cascade routing per Catalog #319 Q3)
5. **Continual-learning posterior update**: ACTIVE (per-pair perturbation directions accumulate per substrate composition matrix)
6. **Probe-disambiguator**: **PRIMARY** (F1-as-A2's probe gate at step 1 IS the canonical disambiguator between scorer-feature-invariant vs scorer-feature-sensitive RGB perturbations)

## Observability surface per Catalog #305

1. **Inspectable per layer**: xray scorer output at per-pair granularity per primitive 8 + per-pair perturbation directions inspectable
2. **Decomposable per signal**: per-pair |∇| ratio dim-7-12 / dim-0-5 + per-pair perturbation magnitude decomposable
3. **Diff-able across runs**: two probe invocations on different magnitude ranges can be diffed
4. **Queryable post-hoc**: per-pair perturbation directions queryable per `.omx/state/composition_plans/<archive_sha[:12]>_<lane_id>_<utc>.json`
5. **Cite-able**: source-trace cite per upstream/modules.py:84 + empirical cite per master_gradient
6. **Counterfactual-able**: "what if we increase perturbation magnitude?" → re-run probe with extended range

## Cargo-cult audit per Catalog #303

| Assumption | Classification | Rationale |
|---|---|---|
| PoseNet dim 7-12 are unscored | HARD-EARNED-VERIFIED-FROM-SOURCE | upstream/modules.py:84 |
| Encoder can freely set dims 7-12 | CARGO-CULTED-PENDING-PROBE | probe at step 1 is the disambiguator |
| RGB perturbations producing first-6-dim invariance EXIST | CARGO-CULTED-PENDING-PROBE | probe at step 1 is the disambiguator |
| Recoverability of arbitrary bits at full fidelity | CARGO-CULTED-PENDING-PROBE | post-probe build verifies |
| Composition with primitives 1 + 6 + 10 is sub-additive | HARD-EARNED-PARTIAL per Catalog #322 | per α ≈ 0.7 expected |

## Cross-references

- META-audit `meta_audit_conflate_declarative_with_physical_error_pattern_*.md` Section 1 Claim #1 (the canonical correction)
- Codex F1 finding `rate_attack_legal_receiver_path_audit_codex_f1_finding_relay_*.md` (source verification)
- SYNTHESIS-V2 §3.2 (HARD-EARNED-VERIFIED-FROM-SOURCE classification)
- CLAUDE.md "Fridrich inverse steganalysis" (Fridrich+Yousfi adversarial steganography framework)
- Catalog #6 strict-scorer-rule (NO_RECEIVER_NEEDED satisfies)
- Catalog #319 Q3 v2 cascade (cathedral autopilot routing)

— Per-candidate playbook for F1-as-A2 PoseNet first-6-dim invariance steganography per parent framework §14.2
