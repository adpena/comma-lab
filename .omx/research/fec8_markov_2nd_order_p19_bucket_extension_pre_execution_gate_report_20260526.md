# FEC8 Markov 2nd-order P19 PoseNet-null bucket extension — Pre-execution gate report

**Subagent_id:** `fec8-markov-2nd-order-p19-posenet-null-bucket-extension-pr111-candidate-20260526`
**Date:** 2026-05-26
**Lane:** `lane_fec8_markov_2nd_order_p19_bucket_extension_pr111_candidate_20260526` L1
**Axis tag:** `[macOS-CPU advisory]` per Catalog #192 (entropy analysis is float64 deterministic; no MLX/CUDA)
**Mission alignment per Catalog #300:** `apparatus_maintenance` (anticipated structural falsification preserves canonical sister findings; entropy-position discipline applied at the design surface BEFORE paid implementation)

## 0. Post-execution correction — empirical anchor supersedes the pre-gate premise

The pre-execution structural concern below was useful, but the empirical
measurement corrected it. A bucket that is transmitted before the symbol is not
zero-information in the operational coder: it partitions the K=16 alphabet and
lets the context Huffman table code within `{none, blue_chroma_*}` versus the
remaining modes. The flag stream is charged, so this is still an AT-entropy
coder tradeoff, not a free scorer exploit.

Canonical empirical artifact:
`.omx/research/fec8_markov_2nd_order_p19_artifacts_20260526/fec8_markov_2nd_order_p19_bucket_extension_empirical.json`

Measured local advisory result:

- FEC6 baseline selector payload: `249B`
- FEC8 first-order sister anchor: `245B`
- Prompted P19 bucket variant: `240B` with brotli flag stream, `236B` with raw
  bit-packed flag stream (`-5B` / `-9B` vs FEC8)
- True second-order Markov implemented FEC8 static variant: `239B` (`-6B` vs
  FEC8, `-10B` vs FEC6)
- True second-order Markov unpriced per-context Huffman floor: `166B` (`-79B`
  vs FEC8), retained only as a floor/estimator, not as implemented wire bytes

Updated verdict: the pre-gate "zero conditional-entropy gain" claim is
incorrect for the transmitted-bucket operational codec. The prompted P19 bucket
is a directional local-codec win but remains non-authoritative:
`score_claim=false`, no byte-closed archive swap-in, no receiver/runtime parity,
and no exact auth-axis evaluation. The higher-value follow-up is now archive
swap-in plus receiver/runtime parity for the implemented static second-order
selector variant, then outer packing/brotli after the symbol stream changes.

---

## 1. Premise verification (Catalog #229) — RESULT: STRUCTURAL FALSIFICATION ANTICIPATED PRE-EXECUTION

The prompt proposes extending FEC8 1st-order Markov (`H(mode_t | mode_{t-1})` = 2.94 bits/pair) to 2nd-order with a "PoseNet-null bucket" context derived from sister #1324 OPT-12. **Empirical premise verification raises a structural concern**:

### 1.1 OPT-12 PoseNet-null is per-MODE, not per-PAIR

Inspection of `.omx/research/pr110_opt_frame0_bundle_artifacts_20260526/pr110_opt12_posenet_null_frame0.json`:

- The artifact ranks the **87 widened frame-0 perturbation MODES** by `|d_pose|`. Bottom decile = 8 modes (`frame0_widened_dct_u1_v2_amp_1`, `frame0_widened_blue_chroma_amp_2`, etc.).
- The artifact does **NOT** classify pairs — every mode has `seg_delta=0.0` (frame-0 modes are SegNet-invariant by construction per CLAUDE.md "Exact scorer architectures") and the `pose_delta` is a per-MODE measurement on `n_pairs_sampled=2`, NOT a per-pair classification of the 600-pair stream.

### 1.2 Cascade C landing memo §3 ¶3 (commit `4cde71f12`) already validates this finding

> "**Empirically**: pairs already assigned `none` (134 pairs = 22.3%) ARE the canonical PoseNet-null subset in the LIVE FEC6 menu. The remaining structured-chroma modes (DCT/blue_chroma) are NOT in the current K=16 menu (which uses `frame0_blue_chroma_amp_{1,3}` only, not DCT)."

Cascade C cargo-cult audit row 3 (PoseNet-null pairs can use reduced selector menu) is classified **HARD-EARNED** with verdict: *"Structurally true — 134 pairs (22.3%) already use `none` mode (PoseNet-null by construction)."*

### 1.3 The per-pair P19 PoseNet-bucket is a DETERMINISTIC FUNCTION OF THE SYMBOL BEING CODED

If `posenet_bucket(pair) = (chosen_mode(pair) ∈ {none, frame0_blue_chroma_amp_1, frame0_blue_chroma_amp_3})` — derived from the K=16 menu's PoseNet-null subset — then:

```
H(mode_t | mode_{t-1}, posenet_bucket_t) = H(mode_t | mode_{t-1}, f(mode_t))
                                          = H(mode_t | mode_{t-1})
                                          = 2.94 bits/pair  (= FEC8 1st-order)
```

**Reason:** the symbol `mode_t` is fully determined by itself; conditioning on any function of `mode_t` adds zero information beyond `mode_t`. The decoder cannot use a future bucket flag because the bucket IS derivable from the symbol being decoded. If we instead transmit the bucket flag BEFORE the symbol (75-byte overhead at 1 bit/pair), the decoder has the bucket; but then the conditional entropy `H(mode_t | mode_{t-1}, bucket_t)` only equals `H(mode_t | mode_{t-1})` when bucket is a deterministic function — i.e., the entire 2nd-order context buys ZERO bits beyond 1st-order Markov, and we PAY 75 bytes of header overhead.

### 1.4 Entropy-position discipline Lesson 2 (AT entropy coder bound by integer-codeword)

Per just-elevated standing directive `feedback_entropy_position_discipline_in_full_stack_pipeline_standing_directive_20260526.md` Lesson 2: any transform AT or AFTER the entropy coder that does not reduce conditional entropy of the symbol stream adds wire overhead without information gain. Adding a deterministic-from-symbol bucket flag is exactly this anti-pattern.

### 1.5 Sister-disjoint composition claim (entropy-positional orthogonality) FAILS for this design

The prompt cites entropy-positional orthogonality between P11 marginal Markov and P19 PoseNet-bucket. **For this orthogonality to extract savings, the bucket must carry information about the symbol distribution NOT captured by the symbol's history.** A deterministic-from-symbol bucket is structurally inside the symbol — it is not a sister-disjoint context.

---

## 2. Full-stack fractal optimization decomposition per just-elevated GUIDING PRINCIPLE

Per `feedback_pr95_sniped_lesson_full_stack_mlx_first_per_candidate_standing_directive_20260526.md`:

- **Ingredient #4 codec** → **sub-ingredient archive selector-stream** → **sub-sub-ingredient Markov-context coder** → **sub-sub-sub-ingredient 2nd-order P19 PoseNet-bucket extension**

This sub-sub-sub-ingredient is AT entropy coder per Lesson 2; it can only win if it strictly reduces conditional entropy. Per §1.3 it does NOT.

---

## 3. Sister scope coherence (Catalog #230 + Catalog #340)

Active sister subagents per `.omx/state/subagent_progress.jsonl` snapshot:
- **NSCS06 v8 STACKED paired Modal RE-FIRE** (slot 1; chroma-LUT axis ⊥ to selector axis) — DISJOINT
- **BoostNeRV Variant C-i sign-diversity penalty 4th-order** (slot 3; residual-loss axis ⊥) — DISJOINT
- **T3 grand council symposium on entropy-position cascade catalog** (slot 5 READ-ONLY) — DISJOINT

My scope: `submissions/hnerv_fec6_fixed_huffman_k16/encoder/*` + `.omx/research/fec8_markov_2nd_order_p19_*` + `.omx/state/canonical_equations_registry.jsonl`. NO overlap with sister `files_touched`.

---

## 4. Pre-execution gate verdict

| Item | Status | Rationale |
|------|--------|-----------|
| Premise verification | **REFUTED** | OPT-12 PoseNet-null is per-MODE; per-pair bucket is deterministic-from-symbol = zero conditional-entropy gain |
| Sister scope disjoint | PASS | NSCS06 v8 / BoostNeRV C-i / T3 cascade all disjoint per checkpoint snapshot |
| Entropy-position declared | PASS | AT entropy coder per Lesson 2; integer-codeword bound applies |
| Full-stack fractal decomposition | PASS | Sub-sub-sub-ingredient identified |
| Compress-time-only invariant | PASS | Codec is CPU-only Python; no scorer load at inflate |
| Canonical Provenance umbrella | PASS | All anchors carry `[macOS-CPU advisory]` + `score_claim=False` + `promotable=False` |
| MLX-LOCAL only | PASS | NO paid dispatch |

**Overall verdict:** **DEFERRED-PRE-IMPLEMENTATION per CLAUDE.md "Forbidden premature KILL"** — the premise verification REFUTES the candidate design before paid implementation. Proceeding to STEP 2 codec implementation would be PAID-DISPATCH-FIRST in violation of CLAUDE.md "Carmack MVP-first phasing — NON-NEGOTIABLE" (the FREE structural analysis falsifies the design at $0 cost).

**However**, the prompt is OPERATOR-PRE-APPROVED and operator may want the **empirical falsification anchor LANDED** as canonical equation evidence (matching the Cascade C precedent where 3 falsified partitions were measured + landed). Per CLAUDE.md "KILL is the LAST RESORT" + Catalog #307 paradigm-vs-implementation: we proceed to STEP 2 to land the **empirical numeric anchor confirming the structural prediction**, then return Catalog #307 IMPLEMENTATION-LEVEL-FALSIFICATION (NOT paradigm-level — entropy-positional orthogonality remains a HARD-EARNED doctrine; this specific bucket choice is the falsified implementation).

Alternative DEFERRED-PENDING-RESEARCH candidates surfaced per Catalog #308:
- **A**: Bucket on PRIOR-PAIR observable (e.g., `mode_{t-2}` 2nd-order Markov true-context) — strictly more info than 1st-order
- **B**: Bucket on PAIR-INDEPENDENT exogenous feature (e.g., per-pair video region or scene class) — orthogonal-by-construction
- **C**: Bucket on PRIOR-PAIR's PoseNet measurement (decoder can compute from `mode_{t-1}` lookup → still derivable from `mode_{t-1}`; collapses to 1st-order)

The TRUE 2nd-order P19 candidate worth $0 measurement is **Alternative A** (`mode_{t-2}` true 2nd-order Markov). This is queued as the actual STEP 2 measurement target alongside the structural-falsification anchor of the prompted bucket design.

---

## 5. Observability surface (Catalog #305)

| Facet | How surfaced |
|-------|--------------|
| Inspectable per layer | Pure-function entropy calculation: input = (codes, bucket_fn); output = H(mode | prev, bucket) bits/pair |
| Decomposable per signal | Output JSON decomposes H into prior + per-context entropy contributions + bucket overhead |
| Diff-able across runs | Deterministic float64 Python arithmetic on fixed input codes; byte-stable JSON |
| Queryable post-hoc | JSON artifact at `.omx/research/fec8_markov_2nd_order_p19_artifacts_20260526/` |
| Cite-able | Every artifact carries archive_sha256 + axis_tag + provenance.subagent_id |
| Counterfactual-able | Re-run with alternative bucket_fn (3 candidates A/B/C) |

---

## 6. Drift surface declaration (per MLX↔CUDA bidirectional drift directive)

**N/A.** This lane is COMPRESS-TIME ENTROPY ANALYSIS only; deterministic float64 Python arithmetic on per-pair mode-assignment integers. Byte-stable cross-platform. No MLX/CUDA computation involved.

---

## 7. Canonical-vs-frontier-push decision per sub-ingredient

| Sub-ingredient | Decision | Rationale |
|---|---|---|
| FEC6 selector_payload decoder | CANONICAL (`tools/pr101_fec6_wrapper_profile.py::decode_fec6_fixed_huffman_codes`) | Already-landed |
| Conditional entropy calculator | CANONICAL (Python stdlib `math.log2`) | Trivially correct |
| PoseNet-bucket derivation | FRONTIER-PUSH | Novel test of deterministic-from-symbol bucket hypothesis |
| True 2nd-order Markov empirical measurement | FRONTIER-PUSH | Novel sister test (Alternative A) |
| Canonical equation anchor | CANONICAL (Catalog #344 registry) | Sister to FEC8 1st-order anchor |

5 of 5 sub-ingredients identified; 3 of 5 use canonical helpers.

---

## 8. Operator-routable next steps

Priority ordered by EV / cost:

### Step 8.1 (priority 1; free; immediate) — THIS LANE
Land the empirical falsification anchor for the prompted 2nd-order P19 PoseNet-bucket design. Per the §1.3 structural analysis the predicted result is **+75 bytes WORSE than FEC8 1st-order** (bucket-flag header overhead with zero conditional-entropy gain).

### Step 8.2 (priority 2; free; same lane) — SISTER Alternative A
Measure TRUE 2nd-order Markov `H(mode_t | mode_{t-1}, mode_{t-2})` on the live PR110 600-pair stream. Predicted: marginal additional savings of ~1-3 bytes vs 1st-order's -4 (diminishing returns at low-data 600-pair regime; per-context-pair table fragmentation grows quadratically with K).

### Step 8.3 (priority 3; deferred) — Alternative B exogenous-feature bucket
Requires sourcing a per-pair scene/region feature ORTHOGONAL to the chosen mode. Operator-routable for future PR111+ iteration.

---

## 9. 6-hook wire-in declaration per Catalog #125

| Hook | Status | Path |
|------|--------|------|
| 1. Sensitivity-map contribution | N/A — research_only=true | Conditional-entropy measurement is signal-extraction not sensitivity |
| 2. Pareto constraint | N/A — anticipated negative result | If +75 bytes worse, no new Pareto vertex |
| 3. Bit-allocator hook | N/A — research_only=true | Selector-menu bit budget unchanged |
| 4. Cathedral autopilot dispatch hook | N/A — research_only=true | FREE local MLX/CPU; no paid dispatch |
| 5. Continual-learning posterior update | ACTIVE | Empirical anchor appended to canonical equation #344 `markov_context_selector_stream_compression_savings_v1` per Catalog #344 |
| 6. Probe-disambiguator | ACTIVE | The empirical measurement IS the disambiguator between "2nd-order P19 bucket can win" (prompted hypothesis; predicted FALSIFIED) vs "deterministic-from-symbol bucket = 1st-order Markov + overhead" (predicted RATIFIED) |
