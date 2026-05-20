<!-- SPDX-License-Identifier: MIT -->
# WAVE-3 DWT Detail-Subband Procedural Codebook Local CPU Smoke — landed 2026-05-20

**Lane**: `lane_wave_3_dwt_detail_subband_procedural_cpu_smoke_20260520`
**Council anchor**: `grand_council_dwt_hnerv_world_model_bind_20260520` (commit `9ef3eee22`, T3 PROCEED_WITH_REVISIONS)
**Op-routable**: Carmack MVP-first op-routable #2 — Catalog #272 byte-mutation smoke on DWT detail-subbands with procedural codebook substitution
**Canonical equation**: `procedural_codebook_from_seed_compression_savings_v1` (registry #26)
**Axis tag**: `[macOS-CPU advisory]` (NEVER promotable per Catalog #192 + #127 + #323)
**$$ spent**: $0 (LOCAL macOS-CPU smoke)
**Wall clock**: <2 hours

`<!-- HISTORICAL_SCORE_LITERAL_OK:dwt_smoke_distributional_residual_anchor_2026-05-20_macos_cpu_advisory_not_score_truth -->`

## 1. Empirical headline

The T3 DWT BIND SYMPOSIUM Assumption-Adversary verdict #1 ("DWT-2-level decomposition is canonical for 384x512 contest video given PoseNet's frequency-response profile" = CARGO-CULTED) is **empirically vindicated** at the distributional surface. Uniform-PRNG-derived bytes do NOT match Laplacian-peaked detail-subband statistics on contest video frame 300 (Y plane, 874x1164):

| Metric | Value | Verdict |
|---|---|---|
| Aggregate KL(empirical \|\| seed-derived) mean | **1.638 nats** | Far from H0 (KL ≈ 0) |
| Aggregate Wasserstein-1 mean | **43.36** (int8 units) | High CDF separation |
| residual_zscore vs H0 threshold 0.5 nats | **3.28** | > 2σ |
| Canonical equation #26 verdict at 2σ | **CARGO-CULTED** | Assumption #1 vindicated |
| Catalog #272 byte-mutation smoke (3 subbands) | **PASSED** | All seed-sensitive |

Per-subband detail (haar DWT level 2 on Y plane; each subband = 63,729 pixels):

| Subband | KL nats | Wasserstein-1 | Mutation KL (syn vs syn-mutated) | Bytes differ on 1-byte seed flip | Seed-sensitive |
|---|---:|---:|---:|---:|---|
| LH | 1.607 | 47.38 | 0.0040 | 63,488 / 63,729 (99.6%) | YES |
| HL | 1.311 | 39.52 | 0.0038 | 63,476 / 63,729 (99.6%) | YES |
| HH | 1.997 | 43.18 | 0.0044 | 63,468 / 63,729 (99.6%) | YES |

## 2. Interpretation

The smoke separates TWO distinct claims that the symposium memo conflated:

1. **Distributional fit claim** (Assumption #1 CARGO-CULTED): seed-derived uniform int8 bytes do NOT statistically match Laplacian-peaked detail-subband bytes. The two distributions differ by 1.6 nats KL on average — equivalent to predicting the wrong symbol with probability ~80% if used as a direct codec substitute. **Direct procedural substitution of detail-subband bytes with `derive_codebook_from_seed(..., dtype=int8)` would corrupt the per-frame DWT inverse and almost certainly destroy the rendered RGB.**

2. **Seed-sensitivity claim** (Catalog #272 byte-mutation smoke PASSED): mutating 1 byte of the 32-byte seed flips ~99.6% of the 63,729 derived bytes per subband. The procedural codebook is **structurally not** a null-byte / no-op trap — the seed IS the entropy source.

The combination matters: the **producer** (`derive_codebook_from_seed`) is structurally sound (seed-sensitive, deterministic, byte-stable across PRNG kinds), but the **predicted savings formula** (`-25 * (N_codebook - K_seed) / 37_545_489`) is only achievable if the substrate **does not directly substitute** detail-subband bytes. The canonical equation's domain-of-validity must restrict to substrates where the procedural codebook is consumed as an *intermediate transform* (e.g., as a quantizer codebook fed through a learned dequantizer) rather than as the final byte stream.

## 3. Operator-routable implications

### 3.1 For the T3 DWT BIND SYMPOSIUM (op-routable #1 paid substrate BUILD)

**RECOMMENDATION: DEFER op-routable #1 (`dwt_hnerv_ll` substrate $1 paired smoke) until the BIND design accounts for non-direct procedural substitution.** The Contrarian's dissent verbatim demanded the DWT-only paired smoke FIRST; this distributional smoke is a CHEAPER ($0) sister probe that surfaces a structural objection BEFORE the $1-$2 paid GPU smoke would have. The bind as currently scoped assumes detail-subband substitutability — that assumption is now empirically refuted at the distributional surface.

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": this is **DEFER-pending-redesign**, NOT KILL. Reactivation criteria:

- Operator approves a redesigned BIND that consumes the procedural codebook as a *quantizer / dequantizer intermediate* rather than direct byte substitution
- OR a paired smoke confirms that residual-after-direct-substitution KL ≪ 0.5 nats on the LL-only HNeRV path (where detail subbands are bilinearly-upsampled, not byte-substituted)

### 3.2 For canonical equation #26 (`procedural_codebook_from_seed_compression_savings_v1`)

The first empirical anchor lands `residual = 1.638 nats` against the predicted-only hypothesis. The aggregate predicted ΔS of -0.013 across 5 substrates per the existing hypothesis anchor remains a hypothesis-pending-substrate-specific-empirical (NSCS06 v8 chroma LUT / ATW V2 / TT5L / DP1 / sister); this smoke does NOT falsify the aggregate prediction (those substrates apply procedural codebooks at DIFFERENT substrate-surfaces than direct DWT detail-subband substitution). The canonical equation's `domain_of_validity` should be amended in a sister landing to explicitly EXCLUDE direct-DWT-detail-subband-byte-substitution (the present smoke's substrate context) so future autopilot consumers do not over-extrapolate.

### 3.3 For canonical equation #26's verdict resolution

The residual_zscore = 3.28 > 2σ verdict is correctly CARGO-CULTED for **this specific substrate context** (direct procedural substitution of detail-subband int8 bytes). The same canonical equation may resolve to HARD-EARNED for the operator-intended substrate contexts (NSCS06 v8 chroma LUT replacement, DP1 codebook-byte replacement, etc.) where the procedural codebook does NOT directly substitute scorer-relevant pixel bytes. The CARGO-CULTED verdict here is **per-substrate-context, not universal**.

## 4. Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| pyav frame decode | ADOPT canonical (`av.open` per CLAUDE.md "Forbidden patterns" — explicit pyav per HNeRV parity L1) | Standard contest tooling |
| pywt DWT 2-level haar | ADOPT canonical (per Catalog #277 wavelet multi-scale helper + Mallat 1989 matched-filter framework) | Reference implementation matches symposium's claimed structure |
| `derive_codebook_from_seed` | ADOPT canonical (per Catalog #344 equation #26 producer) | The smoke IS the equation's first empirical pairing |
| `EmpiricalAnchor` + `update_equation_with_empirical_anchor` | ADOPT canonical (Catalog #344 sister) | Single source of truth for the registry |
| `build_provenance_for_macos_cpu_advisory` | ADOPT canonical (per Catalog #192 + #323 + sister #127) | macOS-CPU advisory non-promotable contract |
| KL + Wasserstein-1 metrics | UNIQUE-PER-METHOD (inline implementations) | Reference scipy KL has different smoothing default; W1 needs identical int8 support; ~20 LOC of arithmetic |
| Per-subband seed derivation (`sha256(base \|\| name)`) | UNIQUE-PER-METHOD | Deterministic per-subband seed pattern with no canonical existing helper |
| Catalog #272 byte-mutation smoke | UNIQUE-PER-METHOD (inline; 1-byte XOR of seed[0]) | Lightweight version of the canonical helper at `tools/verify_distinguishing_feature_byte_mutation.py` since we don't ship an archive |

## 5. 9-dimension success checklist evidence

1. **UNIQUENESS**: cheap-signal-first probe at $0 cost separates distributional fit from seed-sensitivity (orthogonal claims neither prior subagent isolated).
2. **BEAUTY + ELEGANCE**: 590 LOC tool + 173 LOC tests; reviewable in <30 sec per panel.
3. **DISTINCTNESS**: explicitly different from op-routable #1 (paid substrate BUILD) and op-routable #3 (world-model probe); first probe to apply the canonical Catalog #344 equation #26 to real DWT subband statistics.
4. **RIGOR**: 10 dedicated tests (deterministic seed, flat-subband edge case, int8 clipping, KL self-zero, KL disjoint-positive, W1 self-zero, per-subband seed determinism, pywt level-2 shape verification, end-to-end canonical-keys check, CLI subprocess artifact emission). Sister regression: 73/73 canonical equations + procedural codebook generator tests pass. Catalog #185 META-meta drift gate clean.
5. **OPTIMIZATION PER TECHNIQUE**: per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" — distributional metrics tailored to int8 PRNG output domain; not generic numpy/scipy stats.
6. **STACK-OF-STACKS COMPOSABILITY**: smoke output JSON + canonical equation anchor row both feed downstream cathedral autopilot consumers (Catalog #335 contract); the canonical-equation-lookup-consumer (Catalog #344) auto-discovers the new anchor.
7. **DETERMINISTIC REPRODUCIBILITY**: every byte deterministic given (video sha, frame_index, base_seed, wavelet, level, generator_kind); per-frame Y-plane sha256 + per-subband empirical-int8 sha256 + per-subband synthetic-int8 sha256 all recorded.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: ~1 second total wall-clock (single-frame decode + 3 DWT subbands + 3×63K-byte PRNG + 3 KL + 3 W1).
9. **OPTIMAL MINIMAL CONTEST SCORE**: not the smoke's objective (advisory-only per Catalog #192); the smoke INFORMS the BIND design which targets predicted band [-0.015, -0.005] per horizon_class=`asymptotic_pursuit`.

## 6. Cargo-cult audit per assumption

| Assumption (this smoke's design) | Classification | Unwind path |
|---|---|---|
| Haar wavelet is canonical for contest video DWT | HARD-EARNED (Mallat 1989 + Daubechies 1988 + Catalog #277) | None needed; default canonical reference for natural images |
| Detail-subband distributions are Laplacian-peaked | HARD-EARNED (Mallat 1999 + Daubechies matched-filter analysis) | Smoke empirically confirms KL > 0 vs uniform int8 |
| H0 threshold KL = 0.5 nats for "near-uniform" | CARGO-CULTED-EMPIRICAL | Symposium memo cited this as Daubechies matched-filter recovery analysis but the specific 0.5 nats value is a CONSERVATIVE heuristic; a sister analytical derivation would unwind via R(D) for Laplacian sources |
| 1-byte seed mutation produces "different distribution" | HARD-EARNED (avalanche property of PCG64 per O'Neill 2014) | Smoke empirically confirms ~99.6% bytes-differ on 1-byte flip |
| Single-frame smoke generalizes to all 1200 contest frames | CARGO-CULTED | Per CLAUDE.md "Apples-to-apples evidence discipline": only frame 300 measured; mean-over-N-frames sister smoke would unwind |

## 7. Observability surface

- **Inspectable per layer**: smoke_result.json carries per-subband (KL, W1, byte-mutation results) + Y-plane sha256 + per-subband seed sha256s
- **Decomposable per signal**: aggregate residual decomposes into LH + HL + HH contributions
- **Diff-able across runs**: byte-stable smoke output given same (video, frame_index, base_seed, wavelet, level, generator_kind)
- **Queryable post-hoc**: JSON + canonical-equation anchor row both fcntl-locked-append per Catalog #131 / #344
- **Cite-able**: canonical equation #26 anchor carries provenance via `build_provenance_for_macos_cpu_advisory`; the source_artifact field points at the canonical JSON
- **Counterfactual-able**: byte-mutation smoke IS the counterfactual surface (mutate seed → re-derive → compare); could extend to mutate-empirical (substitute LH bytes → re-inverse-DWT → measure rendered-frame diff) as a sister landing

## 8. Top-3 operator-routable next-actions

1. **(PRIMARY)** AMEND canonical equation #26 (`procedural_codebook_from_seed_compression_savings_v1`) `domain_of_validity` to EXCLUDE `direct_dwt_detail_subband_byte_substitution` context; add the smoke's first empirical anchor as a structural example of what the equation's prediction does NOT apply to. Sister subagent task; ~50 LOC delta in `src/tac/canonical_equations/procedural_codebook_savings.py`.
2. **(SECONDARY)** RE-SCOPE T3 DWT BIND SYMPOSIUM op-routable #1 (`dwt_hnerv_ll` substrate $1 paired smoke) to NOT depend on procedural detail-subband substitution; instead train HNeRV on the LL subband while detail subbands are either (a) bilinearly upsampled at inflate-time (i.e., zero-byte; pure DWT downsampling savings) OR (b) entropy-coded via brotli/STC per Catalog #344 sister equations. This avoids the empirically-vindicated distributional mismatch.
3. **(TERTIARY)** Spawn sister mean-over-N-frames smoke (frames 0, 200, 400, 600, 800, 1000) to confirm the single-frame distributional verdict generalizes; cost ~$0 + ~6 sec local CPU; would unwind Cargo-cult #5 in §6 above.

## 9. Discipline citations

- CLAUDE.md "MPS auth eval is NOISE" — macOS-CPU is NEVER score truth
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" — `[macOS-CPU advisory]` non-promotable
- CLAUDE.md "Apples-to-apples evidence discipline" — every metric carries axis label + hardware substrate
- CLAUDE.md "Bit-level deconstruction and entropy discipline" — distributional analysis at byte level
- CLAUDE.md "Forbidden premature KILL without research exhaustion" — verdict CARGO-CULTED is per-substrate-context DEFER, not universal KILL
- CLAUDE.md "Subagent coherence-by-default" — 6-hook wire-in declaration below
- CLAUDE.md "META-ASSUMPTION ADVERSARIAL REVIEW" — Assumption-Adversary #1 empirically vindicated; sister #5 surfaced
- Catalog #127 — custody triple axis × hardware × evidence_grade enforced
- Catalog #192 — macOS-CPU advisory not promotable without Linux x86_64 verification
- Catalog #206 — 3 crash-resume checkpoints emitted
- Catalog #229 — premise verification before write (canonical helpers + symposium memo + prototypes verified)
- Catalog #230 — sister-subagent ownership map (zero collision with in-flight DP1 / D1 / D4 / Z6 etc. subagents per scope analysis)
- Catalog #272 — distinguishing-feature byte-mutation smoke (PASSED, all 3 subbands seed-sensitive)
- Catalog #277 — wavelet multi-scale canonical helper (haar DWT 2-level)
- Catalog #287 — placeholder-rationale rejection (zero `<rationale>` / `<reason>` literals)
- Catalog #305 — observability surface section (this memo + JSON + MD)
- Catalog #309 — horizon_class=`frontier_pursuit` (predicted-band [-0.015, -0.005] would break 0.18 floor per symposium memo)
- Catalog #323 — canonical Provenance umbrella (`build_provenance_for_macos_cpu_advisory`)
- Catalog #335 — canonical consumer contract (this anchor surface auto-discovered by `canonical_equation_lookup_consumer`)
- Catalog #340 — sister-checkpoint guard (PROCEED at start; no overlap)
- Catalog #344 — canonical equation FIRST `anchor_appended` event (verified via `get_equation_by_id` → 2 anchors: pending hypothesis + this empirical anchor)
- Catalog #110 + #113 — APPEND-ONLY HISTORICAL_PROVENANCE (zero mutation of existing memos or anchors)

## 10. 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution** — N/A (this smoke is a distributional probe; no per-axis sensitivity weights produced)
2. **Pareto constraint** — N/A (no Pareto-relevant signal; the smoke validates a precondition to a future Pareto computation)
3. **Bit-allocator hook** — N/A (no bit-allocator signal; the smoke surfaces a distributional VETO against direct-substitution bit allocation)
4. **Cathedral autopilot dispatch hook** — N/A directly; INDIRECTLY ACTIVE via the canonical equation #26 anchor which the `canonical_equation_lookup_consumer` (Catalog #344 sister consumer) auto-ingests
5. **Continual-learning posterior update** — **ACTIVE**: `update_equation_with_empirical_anchor("procedural_codebook_from_seed_compression_savings_v1", anchor)` appends to `.omx/state/canonical_equations_registry.jsonl` via canonical fcntl-locked helper per Catalog #131 / #138 / #344
6. **Probe-disambiguator** — **ACTIVE**: the smoke IS the canonical disambiguator between Assumption #1 HARD-EARNED-vs-CARGO-CULTED at the distributional surface (vindicated CARGO-CULTED with residual_zscore = 3.28)

## 11. Sister-collision verdict

**ZERO collision** with in-flight DP1 PAIRED-SMOKE PRE-DISPATCH DESIGN slot 2:

- DP1 substrate context = Comma2k19-derived PCA basis (4 KB OOD-derived codebook bytes) per `src/tac/substrates/pretrained_driving_prior/`
- THIS smoke substrate context = DWT-detail-subband bytes (in-distribution detail-subband statistics)
- Both contribute orthogonal empirical anchors to canonical equation #26 per the equation's `domain_of_validity` enumeration (DP1 substrate explicitly distinguished from `nscs06_v8_chroma_lut` / `atw_v2_codec` / `tt5l_transformer_tokens` / `sister_substrate_pending_identification`)
- Catalog #340 sister-checkpoint guard reported PROCEED at smoke launch (no sister edited any of my target files within 6h lookback)
- Both subagents communicate ONLY via canonical equation #26 anchor append (single source of truth per Catalog #344)

## 12. Mission contribution per Catalog #300

**`frontier_breaking`** — the smoke surfaces a structural objection to the T3 DWT BIND op-routable #1 BEFORE the $1-$2 paid GPU smoke would have, and lands the first empirical anchor for canonical equation #26 that anchors the canonical-equation-lookup-consumer's downstream routing. The objection is per-substrate-context (not paradigm KILL); the BIND remains DEFERRED-pending-redesign with explicit reactivation criteria.

## 13. Cross-references

- T3 DWT BIND SYMPOSIUM: `.omx/research/grand_council_symposium_dwt_hnerv_world_model_bind_20260520.md`
- Canonical equation builder: `src/tac/canonical_equations/procedural_codebook_savings.py`
- Canonical producer: `src/tac/procedural_codebook_generator/seed_derived_codebook.py`
- Smoke output: `experiments/results/dwt_detail_subband_procedural_smoke_20260520T232239Z/smoke_result.{json,md}`
- Canonical equations registry: `.omx/state/canonical_equations_registry.jsonl` (now carries 2 anchors for equation #26)
- Lane: `lane_wave_3_dwt_detail_subband_procedural_cpu_smoke_20260520` L1 (impl_complete + memory_entry)
