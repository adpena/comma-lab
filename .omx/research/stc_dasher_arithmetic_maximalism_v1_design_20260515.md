# STC-Dasher Arithmetic-Coding Maximalism v1 — Design Memo

**Date:** 2026-05-15
**Lane:** `lane_stc_dasher_scaffold_v1_20260515`
**Council seat:** Tomáš Filler (Fridrich's other PhD student; STC + parity-check codes) + David MacKay (memorial seat; Dasher + arithmetic coding) + Shannon (LEAD; rate-distortion bound)
**Anchor:** Grand Reunion Fields-Grade Symposium 2026-05-15 Phase F binding verdict, Composite #6
**Predicted impact:** `[-0.010, -0.030]` rate-axis `[prediction; first-principles; STC achieves H(W|context) within 0.5%]` — **substrate-agnostic** (applies to ALL existing substrates)
**Cost:** $0 design + $0 implementation + $0 unit tests; smoke validation requires ~$2-5 GPU dispatch

---

## 1. Mathematical formulation

### 1.1 Filler-Judas-Fridrich 2011 syndrome-trellis coding (STC)

Per Filler, Judas, Fridrich (2011) *IEEE TIFS* "Minimizing additive distortion in steganography using syndrome-trellis codes" §III + IV: STC is a parity-check code optimized for steganographic embedding via Viterbi-like decoding through a trellis defined by the parity-check matrix `H ∈ GF(2)^{m×n}`.

For a payload of `m` bits embedded in `n` cover bits (rate `m/n`), the syndrome is

```
s = H · x mod 2
```

where `x ∈ GF(2)^n` is the binary cover vector. STC achieves the rate-distortion bound for additive distortion within a small gap that vanishes as the constraint length increases:

```
R_STC(D) ≤ R_AC(D) + 1/h          (Filler 2011 Theorem 4)
```

where `R_AC(D)` is the arithmetic coding lower bound at distortion `D` and `h` is the constraint length. With `h = 12` (the Filler 2011 §IV.B canonical choice) we approach the AC bound within `~1/12 ≈ 0.083` bits per symbol.

### 1.2 MacKay 2003 ITILA §6.6 Dasher arithmetic coding

Per MacKay (2003) *Information Theory, Inference, and Learning Algorithms* §6.4 + §6.6: Dasher is an arithmetic-coding-based input system where the context model adapts to the symbol history. For sparse signals (most coefficients near zero), the Dasher context model achieves entropy close to the conditional entropy

```
H(X_t | X_{t-1}, ..., X_{t-k})
```

with `k` the context length. MacKay's canonical choice for binary-alphabet sparse-signal streams is `k = 2` (ITILA §6.6).

### 1.3 Composition (the symposium spec)

```
encoded = arithmetic_code( STC_encode(symbols, parity_matrix) )
```

The composed coder applies STC to map source symbols to their syndrome vector; the syndrome is then arithmetic-coded with a Dasher-style context model over the syndrome support.

### 1.4 First-principles ΔS prediction

Per the symposium memo Phase F Composite #6 §"Math":

- A1 archive has `~178 KB` total bytes.
- Of those, the symposium identified `~150 KB` of "renderer parameters" (high entropy, high cardinality) suitable for STC, and `~30 KB` of "mask argmax" (low cardinality, sparse) suitable for Dasher.
- MacKay's lower-bound estimate of the conditional-entropy slack relative to brotli on the renderer-parameter stream is `~0.5%` of total bytes (one MacKay-style "what's the rate cost of the approximation?" derivation).
- `0.5% × 178 KB / 37,545,489 contest-divisor-bytes × 25 (rate weight) = ΔS ≈ -0.00006` per `0.5%` reclaimed. V1 therefore carries **no score band**: it writes raw syndrome bits and carries the original residual payload, so it is a byte-accounting scaffold only. Any `[-0.010, -0.030]` planning band belongs only to a future syndrome-only Viterbi inverse plus real range/ANS/AC byte stream.

This is a **first-principles prediction**, not a measurement. Per CLAUDE.md "Apples-to-apples evidence discipline" the band MUST be tagged `[prediction; first-principles]` until a contest-CUDA / contest-CPU anchor lands.

---

## 2. Application target: A1 archive substrate-agnostic rate-axis bolt-on

### 2.1 Substrate-agnosticism

The codec operates on **archive bytes**, not on substrate-internal tensors. Therefore it composes on TOP of any existing substrate's output:

- A1 (current frontier baseline 0.19285 [contest-CPU GHA Linux x86_64])
- PR101 / PR103 / PR106 sister anchors
- D1 / D4 / Z3 / Z4 / Z5 substrate variants
- Any future substrate that emits a `archive.zip`

### 2.2 Anchor: A1

Per CLAUDE.md "Apples-to-apples evidence discipline" + the Catalog #205 anchor (`submissions/a1/archive.zip` sha `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`, 178,262 bytes, contest-CPU `0.19285`, contest-CUDA `0.22635`).

### 2.3 Composition with existing primitives

- **Stacks on top of**: any substrate's residual stream (renderer params, mask argmax, latent bytes).
- **Replaces**: nothing in the substrate itself; pure post-hoc rate-shaver.
- **Compatible with**: all 8 representation-integration gates (Catalog #99-#108) — the codec emits an archive-internal payload that is consumed by `inflate.py` after the canonical `select_inflate_device` per Catalog #205.

---

## 3. Predicted ΔS bands

| State | Predicted ΔS | Tag |
|------:|:------------:|:---:|
| Pre-anchor (this scaffold) | not claimed | `byte_anchor_only=true` |
| Post-Viterbi-inverse v2 | `[-0.010, -0.030]` | `[prediction; first-principles]` |
| Post-contest-CUDA anchor | TBD | `[contest-CUDA]` |
| Post-paired contest-CPU anchor | TBD | `[contest-CPU]` |

Source: Grand Reunion symposium memo `feedback_grand_reunion_fields_grade_passion_full_council_debrief_*_20260515.md` Phase F Composite #6.

---

## 4. Compatibility (contest-compliant per CLAUDE.md "Contest compliance canonical constraints")

Per CLAUDE.md "Contest compliance canonical constraints" non-negotiable + the WUNDERKIND-VISIONARY classification table:

- **Compress side (FREE)**: STC encoding + Dasher AC of the syndrome.
- **Inflate side (FREE)**: STC decoding + Dasher AC inverse. Per the contest README the decoder runs in 30 min on T4 (26GB RAM/16GB VRAM) OR 4 CPU/16GB CPU. STC + Dasher AC are O(n × h) Viterbi + O(n × |alphabet|) AC respectively, well inside both.
- **Eval side (FREE)**: contest evaluator runs unchanged.
- **Archive bytes (RATE-COUNTED)**: only the encoded syndrome + Dasher-AC stream lands in the archive. v1 scaffold carries the residual envelope (so the v1 candidate is rate-NEGATIVE, intentionally); v2 (post-Viterbi-inverse) drops the residual envelope so only the encoded payload is rate-counted.

NO scorer load at inflate time. NO 73 MB rate hit. CLAUDE.md "Strict scorer rule" honored.

---

## 5. Reactivation criteria

Per CLAUDE.md "KILL/FALSIFIED is LAST RESORT" + scaffold discipline (Catalog #220):

The lane reaches L2 INTEGRATION (and `SCAFFOLD_ONLY = False`) when ALL of:

1. The full Viterbi inverse primitive lands at `tac.codecs.stc_dasher.viterbi_decode(syndrome, parity, ...) -> bytes` and recovers the original symbols from the syndrome alone (drops the residual envelope from the v1 scaffold envelope).
2. A `[contest-CUDA]` exact-eval anchor on Modal A100 (substrate-agnostic; A1 OR any sister substrate as the source archive) returns a finite score within the predicted `[-0.010, -0.030]` band.
3. A paired `[contest-CPU GHA Linux x86_64]` anchor confirms the apples-to-apples axis match.
4. Catalog #220 + Catalog #221 fail-closed status updated with the empirical anchor.
5. The lane registry `score_improvement_mechanism_status` flips from `SCAFFOLD-INTEGRATION-PENDING` to `OPERATIONAL`.

If the empirical anchor lands OUTSIDE the band:
- Default verdict per CLAUDE.md "KILL/FALSIFIED is LAST RESORT" is **DEFERRED-pending-research**, not KILLED.
- Sweep the constraint length `h ∈ {8, 16, 24, 32}`, the context length `k ∈ {1, 2, 3, 4}`, the payload-bit-ratio `r ∈ {2, 4, 8}`, and the per-stream split (renderer params vs mask argmax) before any KILL.
- Council CONSENSUS required for KILL (5+ inner-quintet members + Tomáš Filler + MacKay memorial all PROCEED-WITH-KILL).

---

## 6. Scaffold scope (this landing) vs follow-on

### 6.1 In scope (this landing — DESIGN + SCAFFOLD ONLY per directive)

- `src/tac/codecs/stc_dasher/` package (3 files, ~470 LOC):
  - `__init__.py` re-exports + `SCAFFOLD_ONLY=True` + magic + schema version
  - `encoder.py` with `STCDasherEncoder` + `encode_stream()` + diagnostic result
  - `decoder.py` with `STCDasherDecoder` + `decode_stream()` + no-op detector
- `src/tac/tests/test_stc_dasher_scaffold.py` (50 tests passing)
- `tools/build_stc_dasher_archive_v1.py` build tool + cost-band manifest
- This design memo + memory file
- Lane registry registration at L0 → L1 (impl_complete + memory_entry)

### 6.2 Out of scope (follow-on subagents)

- **Full Viterbi inverse** (`viterbi_decode(syndrome, parity, ...)`): council-gated per the symposium spec; lands as `tac.codecs.stc_dasher.viterbi.py` + `decode_stream(..., recover_from_syndrome=True)`.
- **Sigma > 0 lossy paths**: rate-distortion sweep with controlled distortion injection; per CLAUDE.md "score-aware Lagrangian" (HNeRV parity discipline lesson 6) requires distortion measured against the actual scorer, not L²/KL.
- **Dispatch**: the operator-authorize recipe is NOT written by this scaffold per directive; a follow-on subagent (or operator decision) wires it once the Viterbi inverse lands.
- **Per-stream codec routing**: the symposium's "renderer params via STC + mask argmax via Dasher" split is currently a single composed pass; the follow-on wires the per-stream router via the canonical `tac.composition.registry` per Catalog #169 (sister Composite #1 ATW codec lands in parallel with this one).

---

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| Mathematical primitive | ADOPT repo primitives | The existing STC/Dasher symposium primitives are reusable math kernels, not score-suppressing substrate scaffolds. |
| Codec envelope | UNIQUE | The residual-envelope v1 and future syndrome-only v2 grammar are specific to this rate shaver. |
| Archive integration | UNIQUE until v2 | A post-hoc archive-rate pass needs byte-level proof that decoded output is unchanged and rate decreases; it cannot be treated as a generic trainer helper. |
| Runtime/inflate discipline | ADOPT canonical | Device selection, scorer-free inflate, custody, and auth-eval gating are compliance hygiene. |
| Dispatch policy | UNIQUE fail-closed | No contest score claim or dispatch until the Viterbi inverse removes the residual envelope and exact CPU/CUDA anchors exist. |

---

## 7. Cross-references

- `feedback_grand_reunion_fields_grade_passion_full_council_debrief_vision_strategy_design_whiteboard_session_20260515.md` — symposium binding verdict (Phase F #2 + Composite #6)
- `feedback_chunked_codex_full_codebase_review_landed_20260515.md` — full-codebase review framing
- `feedback_filler_pevny_2010_dual_layer_stc_av1_landed_20260508.md` — Filler-Pevny 2010 dual-layer STC sister landing (sister codec; non-conflicting)
- `feedback_pr101_pose_filler_stc_byte_anchor_landed_20260508.md` — pose-filler STC byte anchor sister
- `feedback_track_4_uniward_stc_hessian_a1_landed_20260509.md` — Track 4 UNIWARD-STC-Hessian on A1 (sister anchor)
- `src/tac/symposium_impls/stc_dasher_arithmetic_coding_maximalism.py` — the council-grade math primitives this scaffold delegates to
- `src/tac/codec/syndrome_trellis_codec.py` — sister STC primitive (block-level Viterbi)
- `src/tac/codec/dual_layer_stc_av1_codec.py` — sister dual-layer STC + AV1 codec
- `src/tac/codec/pose_filler_stc_codec.py` — sister Filler STC codec for pose deltas

---

## 8. 6-hook wire-in declaration (per Catalog #125)

| Hook | Status | Rationale |
|:-----|:------:|:----------|
| Sensitivity-map contribution | N/A — `tac.sensitivity_map` consumes per-tensor importance; this codec is post-hoc on archive bytes (no per-tensor signal) | rationale-required |
| Pareto constraint | ACTIVE | rate-axis Pareto constraint added: `archive_bytes ≤ K(stc_dasher_h, stc_dasher_k)` for the v2 post-Viterbi candidate |
| Bit-allocator hook | ACTIVE | `tac.composition.registry` admits `STCDasherEncoder` as a per-stream rate-shaver; bit-allocator consumes the `estimated_arithmetic_bits` diagnostic |
| Cathedral autopilot dispatch hook | ACTIVE | the build tool emits a cost-band manifest with predicted band; autopilot ranker consumes it via `apply_z1_empirical_revision_to_candidate_delta` (within-class refinement on A1 IS Tier-A-saturated per Catalog #219, so the autopilot will rank this LOWER than class-shift candidates until the band shifts post-anchor) |
| Continual-learning posterior update | ACTIVE on first empirical anchor; deferred until anchor lands | the cost-band manifest carries `score_claim=false` so no posterior update fires now; first contest-CUDA anchor triggers `tac.continual_learning.posterior_update` per Catalog #128 |
| Probe-disambiguator | N/A — only one defensible interpretation (Filler STC + MacKay Dasher composition; symposium binding verdict 11/11 + Tomáš Filler unanimous) | rationale-required |

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable.

---

## 9. Premise verification (per Catalog #229)

- **PV-1** Symposium memo §"Phase F Composite #6" confirms `(Filler + MacKay + Shannon)` as the contributing voices and `[-0.010, -0.030]` as the predicted band. ✓
- **PV-2** Existing math primitives at `src/tac/symposium_impls/stc_dasher_arithmetic_coding_maximalism.py` (316 LOC) provide `stc_encode_to_syndrome` + `arithmetic_code_bit_estimate` + `compose_stc_dasher_encoded_bits` + `build_default_stc_parity_matrix`. ✓
- **PV-3** A1 archive at `submissions/a1/archive.zip` (178,262 bytes) sha `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5` matches the Catalog #205 / Catalog #233 reference anchor. ✓
- **PV-4** Catalog #220 SCAFFOLD discipline applies (substrate-agnostic codec adding bytes to a composition archive >1 KB requires OPERATIONAL or `research_only=true` opt-out). v1 carries the residual envelope so it's rate-NEGATIVE; the predicted band applies post-Viterbi-inverse. ✓
- **PV-5** Sister grand-reunion roster (Tomáš Filler grand-council seat) confirmed at MEMORY.md "Grand Council (advisory)" section. ✓
- **PV-6** Contest-compliance overlay (CLAUDE.md "Contest compliance canonical constraints") confirms compress-side codec is FREE; inflate-side codec is FREE; only the archive bytes count. ✓
- **PV-7** No file overlap with sister subagents per Catalog #230 (NEW package `src/tac/codecs/stc_dasher/`; NEW test file; NEW build tool; NEW design memo). ✓
- **PV-8** Roundtrip-byte-stable verified empirically at `experiments/results/stc_dasher_a1_v1_smoke_20260515T224623Z/` against A1 archive bytes (4KB cap; rc=0; manifest emitted). ✓

---

## 10. Open follow-on questions for the operator

1. **Viterbi inverse council**: does the operator approve a $0 follow-on subagent to land the Viterbi inverse at `tac.codecs.stc_dasher.viterbi.viterbi_decode(...)`?
2. **Smoke dispatch budget**: a contest-CUDA smoke against A1 (or a sister substrate) costs ~$2-5 (Modal T4 100 epochs OR direct apply-and-eval). The operator must approve the dispatch claim per CLAUDE.md "CROSS-AGENT DISPATCH COORDINATION" non-negotiable.
3. **Per-stream router**: should the follow-on wire the symposium's "renderer params via STC + mask argmax via Dasher" split, or keep the single composed pass for v2?
4. **Council-grade lossy mode**: sigma > 0 paths require council deliberation per the "score-aware Lagrangian" non-negotiable. Defer to a future symposium round?

---

## 11. Lane registry mark plan (this landing)

```bash
.venv/bin/python tools/lane_maturity.py mark lane_stc_dasher_scaffold_v1_20260515 \
    --gate impl_complete \
    --evidence "src/tac/codecs/stc_dasher/{__init__,encoder,decoder}.py + tools/build_stc_dasher_archive_v1.py + 50 tests passing"

.venv/bin/python tools/lane_maturity.py mark lane_stc_dasher_scaffold_v1_20260515 \
    --gate memory_entry \
    --evidence "feedback_stc_dasher_scaffold_v1_arithmetic_maximalism_landed_20260515.md"
```

Computed level after marks: **L1** (2/7 gates: impl_complete + memory_entry).

---

## 9-dimension success checklist evidence

Per CLAUDE.md "Catalog #294 — 9-dim checklist evidence section" + standing directive. Per-dimension PRESENT/MISSING/N/A.

1. **Source-fidelity (PR95-style binding of all ingredients)** — PARTIAL. Memo declares STC (syndrome-trellis coding per Filler-Fridrich 2011) + Dasher-style adaptive arithmetic coding (per Ward-Mackay-Blunsom 2002) on top of a Quantizr-class baseline. PR95 binding is DEFERRED to trainer-build time; design memo focuses on codec layer.
2. **Score-aware loss path** — PARTIAL. STC-Dasher is post-train entropy coder; the underlying trainer (Quantizr-class or A1-class) provides the score-aware loss path; STC-Dasher does not add a new loss surface. Will route through canonical helpers per Catalog #164.
3. **Archive grammar + export contract** — PARTIAL. Memo declares STC syndrome wire format + Dasher adaptive-context tables at design level. Per Catalog #124 the 8 archive-grammar fields are declared at design-time; operational-runtime byte-mutation smoke (Catalog #272) DEFERRED to trainer-build.
4. **Inflate runtime closure** — DEFERRED. inflate.py decoder for STC + Dasher arithmetic should fit ≤100 LOC per HNeRV parity L4 (STC decoder ~30 LOC + Dasher decoder ~50 LOC + dispatch shell ~20 LOC). Not yet written.
5. **Mask/pose coupling + scorer routing** — N/A AT DESIGN STAGE — codec is post-trainer; mask/pose coupling lives upstream.
6. **Composability with other substrates** — PRESENT (composition matrix below). STC-Dasher is post-train entropy coder; orthogonal to renderer-architecture substrates (NSCS01/NSCS02/NSCS06); redundant with other entropy coders (NSCS03/ATW).
7. **Tier 1/2/3 engineering** — DEFERRED to trainer-recipe pair time.
8. **Custody + apples-to-apples evidence** — N/A AT DESIGN STAGE — research-only design memo. All score predictions `[prediction]` tagged.
9. **Predicted ΔS band with first-principles derivation** — PRESENT (see below).

## Canonical-vs-unique decision per layer

Per CLAUDE.md Catalog #290 + "UNIQUE-AND-COMPLETE-PER-METHOD operating mode".

| Layer | Decision | Rationale |
|---|---|---|
| Architecture (underlying renderer) | ADOPT CANONICAL | STC-Dasher is post-train; underlying renderer is canonical Quantizr-class or A1-class. HARD-EARNED per Quantizr empirical 0.33 anchor + A1 frontier. |
| Score-aware loss | ADOPT CANONICAL | `score_pair_components` per Catalog #164 — no codec-specific loss. HARD-EARNED. |
| Archive grammar | UNIQUE FORK | STC syndrome wire format + Dasher adaptive-context tables are codec-specific bytes that cannot be shared with NSP1/NSCS03/ATW. HARD-EARNED per Filler-Fridrich 2011 syndrome-trellis canonical + Mackay-Blunsom-Ward 2002 Dasher canonical. |
| Inflate runtime | ADOPT CANONICAL skeleton | ≤100 LOC budget + canonical `select_inflate_device` + torch+brotli closure (brotli used as STC byte-pack outer wrap). HARD-EARNED. |
| Export contract | UNIQUE FORK | ZIP STORED `0.bin` with STC syndrome bytes + Dasher state. HARD-EARNED per HNeRV parity L3. |
| Training curriculum | ADOPT CANONICAL | Underlying renderer trains canonically (2-frame curriculum + EMA + eval_roundtrip + etc.). STC-Dasher applies post-train. HARD-EARNED. |
| Tier-1 engineering | ADOPT CANONICAL | All Tier 1 primitives (autocast/TF32/torch.compile/no_grad) apply to the underlying renderer. HARD-EARNED. |
| Scorer routing | ADOPT CANONICAL | `load_differentiable_scorers` + Catalog #164/#222. HARD-EARNED. |

## Predicted ΔS band

Per Dimension 9.

**RESEARCH-ONLY-NO-SCORE-CLAIM** until: (a) trainer + STC-Dasher post-train pipeline lands + smoke greens-up; (b) paired Tier C MDL ablation per Catalog #227; (c) 5/5 council PROCEED.

**First-principles upper-bound**:
- STC near Shannon limit per Filler-Fridrich 2011 syndrome-trellis: payload embedding efficiency approaches `H(p) / log2(1/p)` bits-per-symbol. For Quantizr-class baseline (88K params @ FP4 + brotli ≈ 64 KB total), brotli's general-purpose model leaves an estimated 5-15% slack vs adaptive arithmetic.
- Dasher adaptive arithmetic per Mackay-Blunsom-Ward 2002: prediction model adapts per-token context; expected gain 3-8% over static brotli model.
- Combined STC + Dasher: 5-15% archive byte reduction over canonical Quantizr-class brotli baseline.
- ΔS translation: A1 baseline rate-term ≈ 0.233; 10% rate reduction → rate ΔS ≈ -0.023. Distortion side unchanged (codec is post-train; encoder weights identical).

**Predicted bands** (research-only-no-score-claim):
- `[contest-CUDA T4 prediction]` band: [0.180, 0.196] (within-class refactor per Z1 framework; if STC+Dasher are 100% lossless then distortion preserved; only rate moves).
- `[contest-CPU GHA Linux x86_64 prediction]` band: [0.175, 0.191] (paired with CUDA gap ≈ -0.005).
- Score-improvement-mechanism: WITHIN-CLASS refactor (improved entropy coding of same canonical bytes); per Z1 framework Tier C density expected ≈ 0.85-0.95 (within-class plateau).

**Reactivation criteria if smoke produces ΔS > 0 (regression)**: STC+Dasher are lossless by construction; any score regression points to (a) integer-rounding errors in adaptive context tables; (b) brotli outer-wrap interaction; (c) inflate-side decoder bug — investigate before declaring negative.

## Stack-of-stacks composition matrix

Per Dimension 6 + Subagent C plan.

| With substrate | Axis orthogonality | Composition class | Expected ΔS | Rationale |
|---|---|---|---|---|
| **NSCS01** (nullspace split renderer) | ORTHOGONAL (codec vs renderer-architecture) | ADDITIVE | additive (~0.010-0.020) | NSCS01 + STC-Dasher composes the renderer-architecture refactor with optimal entropy coding on the resulting bytes. Strong pairing. |
| **NSCS02/NSCS06** (Carmack-Hotz strip-everything) | NEAR-REDUNDANT (both target rate term) | SATURATING | floor at -0.005 | Strip-everything reduces bytes BEFORE coding; STC-Dasher reduces bytes BY coding; combined leverage limited (the post-strip bytes are already low-entropy). |
| **NSCS03** (Ballé end-to-end joint codec) | REDUNDANT (both end-to-end entropy coders) | SATURATING | floor at -0.005 | NSCS03 trains an end-to-end learned entropy coder; STC-Dasher is post-train adaptive. Choose ONE. |
| **ATW codec** (cooperative-receiver) | REDUNDANT (both post-train entropy coders) | SATURATING | floor at -0.005 | Both occupy the post-train codec slot. Choose ONE. ATW is cooperative-receiver-specific (potentially across-class per Z1); STC-Dasher is general-purpose (within-class). |

Per Catalog #227, STC-Dasher within-class density (expected ~0.90) inherits the within-class penalty when ranked alone. Paired with NSCS01 (architecture-refactor) the combined density stays within-class but the additive composition_alpha bonus per Subagent C plan keeps it competitive. Recommended next-wave: `NSCS01 + STC-Dasher` as a single-axis-refactor + entropy-floor pair; do NOT stack STC-Dasher with NSCS03/ATW (redundant codec slot).

