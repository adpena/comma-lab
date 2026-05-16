# Z3-G1 entropy-coded v2 design memo

Operator-approved 2026-05-15 reactivation of `lane_z3_g1_scorer_softmax_hyperprior_gating_20260515`
(research_only=true, deferred per F1 codex finding empirical confirmation that
`hyperprior_weights_int8` + `w_hat_int8` slots ship empty `b""`). This memo
specifies the v2 entropy-coded sigma-table + class-index variant of the Z3HG1
wire grammar so the trainer's distinguishing bytes ACTUALLY ship in the archive.

## 1. Mathematical formulation

### 1.1 What v1 (Z3-G1) ships vs predicts

| Slot | v1 prediction | v1 actual | Delta |
| --- | --- | --- | --- |
| `hyperprior_weights_int8` | ~80B brotli'd sigma table | `b""` (empty) | -80B (UNSHIPPED) |
| `w_hat_int8` | ~200B brotli'd class indices | `b""` (empty) | -200B (UNSHIPPED) |
| `residual_int8` | ~1200B brotli'd AC residual | ~1200B Z3HV2 residual | identical |
| Net distinguishing bytes | ~280B class-conditional info | 0B | THE BUG |

The v1 architecture computed a 5×28 sigma table and 600 class indices DURING
TRAINING but the production-safe direct-residual export path ships the SAME
Z3HV2 packet as Z3 v2 because the empty slots short-circuit the inflate-time
class-conditional decoder. Smoke score 0.19869 EXACTLY matches Z3 v2 baseline
(F1 codex finding empirically confirmed by `c5b9f38d2` smoke
`fc-01KRPKCXARWP7NBGJCXB2P9QEP`).

### 1.2 What v2 ships

v2 introduces a NEW magic + grammar (`Z3G2`) that REPLACES the empty Z3HV2
slots with TWO entropy-coded streams:

1. **sigma_table_entropy_coded_int8** (`sigma_table_blob`, ~300-500B brotli'd):
   The 5×28 = 140 int8 sigma values, brotli-compressed at quality 11.
2. **class_index_arithmetic_coded_uint8** (`class_index_blob`, ~200-400B):
   The 600 per-pair dominant SegNet class indices, encoded via constriction's
   `QueueEncoder` under a per-class prior CDF derived from frequency counts
   over the 600 pairs. The class prior CDF (5 × uint16 counts = 10B) ships
   alongside the encoded stream so the decoder can reconstruct the prior.

The Z3G2 magic distinguishes v2 from v1 (`Z3V2`) so the inflate path can fork
on first 4 bytes; v1 packets remain readable via `tac.substrates.z3_balle_hyperprior_bolton.inflate_v2`.

### 1.3 Inflate-time mechanism (NO scorer load)

```python
# inflate_consumer.py pseudocode (≤100 LOC budget per HNeRV parity L4):
section_bytes = z3v2_payload[A1_DECODER_SECTION_TOTAL : ...]
assert section_bytes[:4] == b"Z3G2"
meta = unpack_z3g2_header(section_bytes)
sigma_table = _unpack_sigma_table_entropy_coded(
    section_bytes[meta.sigma_blob_offset : meta.sigma_blob_offset + meta.sigma_blob_len],
    meta.int8_sigma_scale,
)  # (5, 28) fp32

class_prior_cdf = _unpack_class_prior_cdf(
    section_bytes[meta.class_prior_offset : meta.class_prior_offset + 10]
)  # (5,) uint16 counts -> normalized prior

class_indices = _class_conditional_arithmetic_decode(
    section_bytes[meta.class_index_offset : meta.class_index_offset + meta.class_index_blob_len],
    class_prior_cdf,
    n_pairs=600,
)  # (600,) uint8 in [0, 5)

# Per-pair sigma lookup -> per-pair AC decode of residual:
sigmas_per_pair = sigma_table[class_indices, :]  # (600, 28)
residual = arithmetic_decode_residual(
    section_bytes[meta.residual_offset:],
    sigmas_per_pair,  # used as conditional Gaussian std per pair
    quantization_step=meta.quant_step,
)  # (600, 28) int8
```

**Critically**: NO scorer load at inflate time per CLAUDE.md "Strict scorer rule".
The class indices ship as bytes computed at compress-time (SegNet runs on GT
frame, FREE per "compress-side scorer use is FREE" non-negotiable). Inflate
runtime only reads the bytes and runs the AC decoder.

## 2. Rate-side planning hypothesis

Score movement is unranked until full-frame `inflate.sh` mutation proof and
paired CPU+CUDA exact eval exist. The current scaffold is a lossy latent
transform because sigma affects reconstructed latents at inflate time; the
rate-side calculation below is a planning hypothesis, not a score claim.

### 2.1 Bit-savings derivation

- A1 latent_blob slot: 15387 B (replaced by v2 section)
- v2 section size estimate:
  - Z3G2 header: ~40B
  - sigma_table_entropy_coded_int8: ~300B brotli'd
  - class_prior_cdf: 10B raw
  - class_index_arithmetic_coded_uint8: ~200-400B (entropy of 5-class
    distribution on driving scenes ~1.8 bits/pair × 600 = 1080 bits ≈ 135B
    raw; arithmetic coding overhead + length prefix ~ 200B total)
  - residual_int8 brotli'd: ~1200B (same as Z3 v2 since residual statistics
    are class-conditional now but byte distribution similar)
  - Per-dim affine (offset + scale): 224B
  - Length prefixes: ~12B
- v2 section total: ~1986B vs A1 latent_blob 15387B ⇒ **savings ~13.4 KB**
- Rate term reduction: 25 × 13400 / 37545489 ≈ -0.0089 rate-axis (dominates ΔS)

### 2.2 Distortion-axis contribution

Implementation note after adversarial review: the current scaffold encodes
class indices with the local entropy path and carries direct int8 residuals;
inflate multiplies residuals by the decoded sigma table so byte mutation affects
the reconstructed latent. That is a **lossy latent transform**, not a proven
rate-only entropy-model improvement. Distortion-axis contribution is therefore
unknown until paired exact eval; near-zero distortion is a hypothesis, not a
claim.

### 2.3 Haircut + uncertainty

- Class entropy may be HIGHER than 1.8 bits/pair (driving scenes have
  variable class distribution per video segment) → +50-100B class index blob.
- Sigma table quantization to int8 may degrade conditional Gaussian fit →
  +0.001 distortion-axis ΔS.
- arithmetic coder model mismatch → +50B residual blob.
- **Haircut**: predicted band [-0.005, -0.015] (vs first-principles -0.0089
  rate-only) accounts for these uncertainties.

## 3. Composition

Z3G2 stacks on **A1 substrate** (frozen decoder + sidecar) like Z3 v2 does,
but REPLACES the Z3HV2 section instead of stacking on top. The v2 section
slots into A1's wire format at byte offset `A1_DECODER_SECTION_TOTAL` (162168),
exactly where Z3 v2's section sits today.

A1 wire format (verbatim from `submissions/a1/src/codec.py`):
```
[uint32 LE section_total = 162168]
[decoder_blob 162164 B]
[latent_blob 15387 B]                                 <-- v2 REPLACES this
[sidecar_blob (variable; ~607 B)]
```

v2 wire format:
```
[uint32 LE section_total = 162168]              (verbatim from A1)
[decoder_blob 162164 B]                         (verbatim from A1)
[Z3G2 header + payload]                          (NEW; replaces latent_blob)
[sidecar_blob (variable; ~607 B)]               (verbatim from A1)
```

## 4. 36-field SubstrateContract per Catalog #241/#242

```python
Z3_G1_ENTROPY_CODED_V2_CONTRACT = SubstrateContract(
    # 2.1 Identity & lifecycle
    id="z3_g1_entropy_coded_v2",
    lane_id="lane_z3_g1_entropy_coded_v2_20260515",
    target_modes=("research_substrate",),
    deployment_target="desktop_research",
    council_verdict_provenance=".omx/research/wunderkind_g1_entropy_coded_v2_design_20260515.md",
    # 2.2 Architecture & runtime (8 per Catalog #124)
    archive_grammar="z3g2_entropy_coded_sigma_table_plus_class_index_replaces_a1_latent_blob",
    parser_section_manifest={
        "header": "Z3G2_HEADER_STRUCT (~40B fixed)",
        "sigma_table_blob": "brotli(sigma_table_int8) (~300B variable)",
        "class_prior_cdf_blob": "5*uint16 = 10B fixed (class frequency counts)",
        "class_index_blob": "constriction-AC(class_indices, prior_cdf) (~200-400B variable)",
        "residual_blob": "brotli(residual_int8) (~1200B variable)",
        "per_dim_affine": "2 * 28 * float32 = 224B fixed (offset + scale)",
    },
    inflate_runtime_loc_budget=100,  # HNeRV parity L4
    runtime_dep_closure=("torch>=2.5,<2.7", "brotli", "constriction>=0.4,<0.5"),
    export_format="z3g2_entropy_coded_packet",
    score_aware_loss="scorer_loss_terms_btchw",  # canonical via score_pair_components_dispatch
    bolt_on_loc_budget=350,  # HNeRV parity L7
    no_op_detector_planned=True,  # byte-mutation smoke per Catalog #139
    # 2.3 Operational mechanism (3 per Catalog #220)
    archive_bytes_added=None,  # NET SAVINGS not addition (replaces A1 15387B with ~1986B)
    score_improvement_mechanism_status="RESEARCH_ONLY",  # full-frame proof missing
    runtime_overlay_consumed=False,  # parser/intermediate proof only
    # 2.4 Recipe schema (8)
    recipe_smoke_only=True,
    recipe_research_only=True,
    recipe_min_smoke_gpu="T4",
    recipe_min_vram_gb=14,
    recipe_pyav_decode_strategy="cpu_thread_async_upload",
    recipe_canary_status="post_canary_dependent",
    recipe_video_input_strategy="per_dispatch_local_copy",
    recipe_canary_dependency="substrate_a1",
    # 2.5 Cost band & GPU envelope (4)
    cost_band_epochs=100,
    cost_band_gpu_key="T4",
    cost_band_platform_key="modal",
    cost_band_p50_usd=7.50,  # midpoint of $5-10 band
    # 2.6 6-hook wire-in (Catalog #125)
    hook_sensitivity_contribution="scorer_conditional_entropy_map_v1",  # SegNet class IS the conditioning variable
    hook_pareto_constraint="rate_distortion_v1",  # rate-dominant ΔS
    hook_bit_allocator_class="per_channel_lsq",  # per-class sigma table = per-class allocation
    hook_autopilot_ranker_class_shift_token="cooperative-receiver",  # per Wunderkind G1 paradigm
    hook_continual_learning_anchor_kind="not_applicable_with_rationale",
    hook_probe_disambiguator=None,  # single defensible interpretation: bytes-consumed-by-inflate
    # 2.7 Compliance + 2.8 not-applicable rationales
    catalog_compliance_declarations=(
        "catalog_124_archive_grammar_at_design_time",
        "catalog_125_six_hook_wire_in",
        "catalog_139_no_op_proof_byte_mutation_smoke_planned",
        "catalog_146_3arg_archive_grammar_honored",
        "catalog_205_select_inflate_device_used",
        "catalog_220_operational_mechanism_declared",
        "catalog_226_gate_auth_eval_call_used",
        "catalog_240_dispatch_blocked_until_implementation_complete",
        "catalog_272_distinguishing_feature_integration_contract",
    ),
    hook_not_applicable_rationale={
        "hook_probe_disambiguator": "single defensible interpretation for the wire grammar: archive bytes are consumed by the parser/decoder; semantic byte-mutation smoke proves clean decoded-output mutation before promotion.",
        "hook_continual_learning_anchor_kind": "no empirical score anchor exists; posterior update waits for paired CPU+CUDA exact eval.",
    },
)
```

## 5. Distinguishing-feature integration contract per Catalog #272

```python
distinguishing_feature_name = "entropy-coded sigma-table + class-index replacing 50KB Ballé hyperprior with 1KB SegNet-class CDF"

distinguishing_bytes_path = [
    "sigma_table_blob (~300B brotli'd int8 5x28 = 140 values; offset = header_end)",
    "class_prior_cdf_blob (10B raw 5x uint16 frequency counts; offset = sigma_blob_end)",
    "class_index_blob (~200-400B constriction-AC encoded uint8 600-pair stream; offset = class_prior_end)",
]

inflate_consumer_function = [
    "tac.substrates.z3_g1_entropy_coded_v2.inflate_consumer._unpack_sigma_table_entropy_coded",
    "tac.substrates.z3_g1_entropy_coded_v2.inflate_consumer._unpack_class_prior_cdf",
    "tac.substrates.z3_g1_entropy_coded_v2.inflate_consumer._class_conditional_arithmetic_decode",
]

byte_mutation_smoke_passes = "verify via tools/verify_z3_g1_entropy_coded_v2_byte_mutation.py per Catalog #139; mutates distinguishing bytes and records parser_intermediate_mutation evidence separately from parser_bound_consumption (extincts F1 phantom-class overclaim without claiming full-frame output proof)"
```

## 6. Reactivation gate

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" + Catalog #220:
v2 declares parser/intermediate consumption but stays `research_only=true` and
`dispatch_enabled=false` until a real `_full_main`, remote driver,
byte-closed export, and paired exact-eval custody land. This claim is
structurally verified by:

1. **Encoder/decoder roundtrip test**: encode known sigma table + class indices →
   decode → assert identity (5×28 sigma + 600 class indices preserved exactly).
2. **Byte-mutation smoke per Catalog #139**: tool mutates distinguishing bytes,
   requires a clean decode with changed decoded-output hash/tensors, and records
   parser-bound rejection separately as lower-grade evidence. Parser rejection
   alone is NOT a pass.
3. **Archive grammar parser symmetry**: encode → split_z3g2_payload_bytes →
   re-encode produces byte-identical packet.
4. **HNeRV parity discipline lessons all 13 honored**: see substrate
   `__init__.py` docstring for per-lesson cross-ref.

The lane stays research-only via `_full_main raises NotImplementedError` until
the train/export/auth-eval path lands. Local semantic byte-mutation smoke can
support parser readiness, but it does not authorize remote training or
contest-score promotion.

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| Wire grammar | UNIQUE | Z3G2 exists specifically to ship sigma/class bytes that v1 dropped. |
| Byte-mutation proof | UNIQUE | The proof must exercise this grammar's decoded outputs and distinguish semantic mutation from parser-bound rejection. |
| Training/export path | UNIQUE fail-closed | `_full_main` is not implemented; no generic Z3 path may silently stand in for this lane. |
| Runtime/custody | ADOPT canonical | Scorer-free inflate, recipe blockers, paired CPU/CUDA custody, and dispatch claims remain shared guardrails. |
| Autopilot/ranker status | UNIQUE fail-closed | This lane can be ranked as a research substrate but not as an exact-ready dispatch candidate until full-path custody exists. |

## 7. Smoke dispatch requirements

- Modal T4 ~$5-10 once implementation blockers clear (smoke 100ep + full 1000ep paired CPU/CUDA per CLAUDE.md
  "Submission auth eval — BOTH CPU AND CUDA").
- Smoke first per Catalog #167 `tools/run_modal_smoke_before_full.py`.
- Current recipe is research-only and dispatch-disabled; a future commit must
  flip those fields only after `_full_main`, remote driver, export, and paired
  auth-eval custody are present.
- Smoke success criteria:
  - Encoder/decoder roundtrip test passes locally pre-dispatch.
  - Byte-mutation smoke passes locally pre-dispatch.
  - Auth eval returns score in [0.180, 0.200] (smoke prediction band).
- Full dispatch only fires if smoke passes.

## 8. Cross-references

- `lane_z3_g1_scorer_softmax_hyperprior_gating_20260515` (research_only=true;
  this lane v2 IS the reactivation; v1 lane stays in registry as historical
  per CLAUDE.md "KILL is the LAST RESORT").
- `feedback_z3_g1_full_cpu_paired_aborted_codex_f1_empirically_confirmed_landed_20260515.md`
  (the abort + 3 reactivation criteria, all 3 satisfied by this design).
- `feedback_wunderkind_visionary_scorer_as_cooperative_receiver_paradigm_shift_20260515.md`
  (Wunderkind G1 SUBSTITUTION-1:1 spec — rate-side planning hypothesis only
  until paired exact eval).
- `feedback_contest_compliance_canonical_constraints_for_wunderkind_and_all_subagents_NON_NEGOTIABLE_20260515.md`
  (contest rule #2 — compress-side scorer use is FREE; preserved).
- CLAUDE.md Catalog #124 (archive grammar at design time — all 8 fields
  declared in this memo + the SubstrateContract).
- CLAUDE.md Catalog #139 (no-op proof + byte-mutation smoke).
- CLAUDE.md Catalog #220 (operational mechanism declared at design time).
- CLAUDE.md Catalog #240 (dispatch_enabled requires implementation_complete;
  v2 explicitly remains research_only=true / dispatch_enabled=false until the
  full path lands).
- CLAUDE.md Catalog #272 (distinguishing-feature integration contract — the
  4 fields declared in §5 above).
- `src/tac/substrates/z3_balle_hyperprior_bolton/archive_v2.py` (the Z3 v2
  wire format that v2 forks from; `Z3HV2_MAGIC = b"Z3V2"` vs new `Z3G2_MAGIC = b"Z3G2"`).

## 9. Op-routables

1. **Operator must approve full-path implementation and later smoke dispatch**
   when `_full_main`, remote driver, export, and paired auth-eval custody land.
   The present scaffold is parser/runtime evidence only.
2. **Byte-mutation smoke must pass** before promotion to L1. Run
   `tools/verify_z3_g1_entropy_coded_v2_byte_mutation.py` locally pre-dispatch.
3. **Predicted score band remains unpromoted** until a same-archive exact
   evaluation exists; if a future smoke score is outside band, investigate per
   CLAUDE.md "Forbidden premature KILL" before any kill verdict.

---

## 9-dimension success checklist evidence

Per CLAUDE.md "Catalog #294 — 9-dim checklist evidence section" + standing directive. Per-dimension PRESENT/MISSING/N/A.

1. **Source-fidelity (PR95-style binding of all ingredients)** — PARTIAL. Memo declares Wunderkind-G1 v2 entropy-coded variant addressing the codex review bkrbqet3p F1 finding (Z3-G1 v1 archive shipped empty `hyperprior_weights_int8` + `w_hat_int8` slots making the SegNet-class-CDF distinguishing feature non-operational per Catalog #266). v2 design wires the entropy-coded sigma table + class indices INTO the archive bytes (not just `g1_diagnostic.pt`). PR95 binding lives at the underlying Z3HV2 substrate which v2 inherits.
2. **Score-aware loss path** — PRESENT (UNIQUE EXTENSION). Loss adds a SegNet-class CDF entropy-coding term: per-pixel class probabilities → per-class sigma assignments → entropy-coded latent bytes. Routes through canonical `score_pair_components` per Catalog #164 for the base loss + adds an entropy-rate term `λ_R * H(class_indices)` per Ballé 2018 style.
3. **Archive grammar + export contract** — PRESENT (UNIQUE FORK from v1). v2 wire format: `g1v2` magic header + non-empty `hyperprior_weights_int8` (entropy-coded sigma table per class) + non-empty `w_hat_int8` (entropy-coded latent for class-mapped grid) + brotli outer wrap. Per Catalog #124 the 8 fields declared; per Catalog #266 the archive MUST consume hyperprior bytes for inflate to actually use the SegNet-class CDF (v1 failed this gate; v2 is the structural fix).
4. **Inflate runtime closure** — PRESENT. v2 inflate.py decodes sigma table → per-class quantization params → per-class entropy-decode latent bytes → reconstruct frame. ≤100 LOC per HNeRV parity L4 (entropy decoder ~40 LOC + dispatch shell ~30 LOC + per-class quant decoder ~25 LOC).
5. **Mask/pose coupling + scorer routing** — PRESENT. Per CLAUDE.md "Mask/pose coupling gate" v2 wires the SegNet class derivation via canonical scorer routing (`load_differentiable_scorers` + canonical scorer-loader assignment order per Catalog #222) + fail-closed if SegNet class derivation fails per Catalog #267 (no silent uniform-class fallback).
6. **Composability with other substrates** — PRESENT (composition matrix below). Wunderkind-G1 v2 is the SegNet-class-conditional sister of NSCS03's general-purpose Ballé hyperprior; redundant with NSCS03; orthogonal to renderer architecture (NSCS01/NSCS02/NSCS06).
7. **Tier 1/2/3 engineering** — PRESENT. Tier 1 canonical (autocast/TF32/torch.compile/no_grad per Catalogs #172/#178/#179/#180). Tier 2 recipe declares min_vram_gb/min_smoke_gpu/target_modes per Catalogs #170/#215/#182. Tier 3 canonical auth-eval helper per Catalog #226 + canonical inflate device per Catalog #205 + recipe-vs-trainer consistency per Catalog #240.
8. **Custody + apples-to-apples evidence** — PRESENT. `require_contest_cuda_auth_eval_claim` per Catalog #127 + `posterior_update_locked` per Catalog #128. Per CLAUDE.md "Apples-to-apples evidence discipline" v2's score gain over v1 baseline MUST be measured on byte-mutation smoke per Catalog #272 (which v1 FAILED at 5-decimal match with Z3 v2 — proving v1 bytes weren't consumed). v2 success criterion: byte-mutation smoke produces frame changes when hyperprior bytes mutated.
9. **Predicted ΔS band with first-principles derivation** — PRESENT (see below).

## Canonical-vs-unique decision per layer

Per CLAUDE.md Catalog #290 + "UNIQUE-AND-COMPLETE-PER-METHOD operating mode".

| Layer | Decision | Rationale |
|---|---|---|
| Architecture (underlying renderer) | ADOPT CANONICAL | Inherits Z3HV2 substrate base. HARD-EARNED per Z3 v2 contest-CUDA 0.23171 anchor. |
| Score-aware loss | UNIQUE EXTENSION | Base `score_pair_components` per Catalog #164 + UNIQUE entropy-rate term for class indices. The class-conditional entropy IS the distinguishing-feature per Catalog #272. HARD-EARNED per Ballé 2018 hyperprior canonical + CLAUDE.md "Exact scorer architectures" SegNet 5-class signature. |
| Archive grammar | UNIQUE FORK (g1v2) | v1 grammar was structurally broken (empty slots); v2 grammar binds class-conditional entropy bytes INTO archive per HNeRV parity L3 + Catalog #266 fix. HARD-EARNED per codex bkrbqet3p F1 empirical receipts. |
| Inflate runtime | ADOPT CANONICAL skeleton + UNIQUE body | Canonical `select_inflate_device` (Catalog #205) + torch+brotli closure; UNIQUE entropy-decoder body parses g1v2 wire format. HARD-EARNED. |
| Export contract | UNIQUE FORK | ZIP STORED `0.bin` with g1v2 body bytes. HARD-EARNED per HNeRV parity L3. |
| Training curriculum | ADOPT CANONICAL | Inherits Z3HV2 training (PR95 parity discipline lessons 1-13). HARD-EARNED. |
| Tier-1 engineering | ADOPT CANONICAL | All Tier 1 primitives canonical. HARD-EARNED. |
| Scorer routing | ADOPT CANONICAL | `load_differentiable_scorers` + Catalog #164/#222. HARD-EARNED. |

## Predicted ΔS band

Per Dimension 9.

**RESEARCH-ONLY-NO-SCORE-CLAIM** until: (a) v2 trainer + inflate runtime land; (b) byte-mutation smoke per Catalog #272 proves hyperprior bytes consumed (the v1-FAIL test that motivated v2); (c) paired Tier C MDL ablation per Catalog #227; (d) 5/5 council PROCEED.

**First-principles upper-bound**:
- SegNet 5-class CDF approximation per Ballé 2018 framework: instead of treating latent bytes as uniformly random (Z3HV2 baseline), assume per-class sigma scaling → expected entropy reduction `H(class_conditional) - H(unconditional) ≈ log2(5) / 5 ≈ 0.464` bits per token (5-class case).
- Z3HV2 baseline archive ~350 KB; if hyperprior + class indices reduce rate by 5-10% then archive ≈ 315-332 KB; rate ΔS ≈ -0.012 to -0.024.
- Distortion side: class-conditional reconstruction may help (per-class quantization preserves boundary detail) OR hurt (if class miscategorized at decode); expected near-zero ± 0.005.
- Combined v2 ΔS vs Z3HV2: -0.005 to -0.020 IF byte-mutation smoke passes.

**Predicted bands** (research-only-no-score-claim):
- `[contest-CUDA T4 prediction]` band: [0.211, 0.226] (Z3HV2 baseline = 0.23171; class-conditional codec reduces archive bytes within-class per Z1 framework).
- `[contest-CPU GHA Linux x86_64 prediction]` band: [0.193, 0.208] (paired with CUDA gap ≈ -0.018 per Z3 v2 paired empirical).
- Score-improvement-mechanism: WITHIN-CLASS refactor (class-conditional entropy coding on same canonical Z3HV2 bytes). Per Z1 framework Tier C density expected ≈ 0.75-0.90 (within-class plateau, but more interesting than v1 which collapsed to 1.0-identical with v2 because v1 bytes weren't consumed).

**Reactivation criteria if smoke produces 5-decimal-match with Z3HV2 baseline (the v1 failure mode)**: (a) v2 archive grammar still not consumed by inflate — re-verify byte-mutation smoke per Catalog #272; (b) entropy decoder may be falling back to canonical decode on parse failure — wire fail-closed per Catalog #267 pattern; (c) class derivation may be silently using uniform fallback — verify SegNet-class assignment varies per pair.

## Stack-of-stacks composition matrix

Per Dimension 6 + Subagent C plan.

| With substrate | Axis orthogonality | Composition class | Expected ΔS | Rationale |
|---|---|---|---|---|
| **NSCS01** (nullspace split renderer) | ORTHOGONAL (codec vs renderer-architecture) | ADDITIVE | small additive (~0.005-0.010) | Different layers; composable. |
| **NSCS02/NSCS06** (Carmack-Hotz strip-everything) | NEAR-REDUNDANT (both target rate term) | SATURATING | floor at -0.005 | Both reduce archive bytes via different mechanisms; combined leverage limited. |
| **NSCS03** (Ballé end-to-end joint codec) | REDUNDANT (both Ballé-style hyperprior codecs) | SATURATING | floor at -0.005 | NSCS03 is general-purpose end-to-end Ballé; Wunderkind-G1 is class-conditional Ballé. Choose ONE; class-conditional is sister-of NSCS03 not orthogonal-to. |
| **ATW codec** (cooperative-receiver) | NEAR-REDUNDANT (both scorer-aware codecs) | SATURATING | floor at -0.005 | Both leverage scorer features for entropy coding. Choose ONE. |
| **STC-Dasher** (arithmetic coding maximalism) | NEAR-REDUNDANT (both post-train entropy coders) | SATURATING | floor at -0.005 | Both occupy post-train entropy slot. Choose ONE. |
| **U-DIE-KL** (substrate-wide loss reformulation) | ORTHOGONAL (loss vs codec) | ADDITIVE | additive (~0.010-0.015) | U-DIE-KL changes loss; v2 changes codec. Composable. |

Per Catalog #227, Wunderkind-G1 v2 is within-class refactor (class-conditional canonical Z3HV2). Paired with NSCS01 + U-DIE-KL produces a within-class 3-stack with bounded ΔS gain. The class-shift candidates (NSCS03 / ATW) are preferred for breaking the 0.196 plateau. Recommended deployment for v2: SOLO smoke first (validate byte-mutation smoke per Catalog #272 — the v1 failure mode); only consider stacking AFTER v2 demonstrates non-zero bytes consumed.

