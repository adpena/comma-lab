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

## 2. Predicted ΔS

`[prediction; first-principles-bound]` ΔS ∈ [-0.005, -0.015] vs A1 baseline
0.1928 [contest-CPU 1to1].

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

Class-conditional Gaussian prior shapes residual statistics during training.
Distortion-axis contribution is EXPECTED to be near-zero (the same A1
decoder is used; per-pair sigma only affects the rate side). Predicted
distortion-axis ΔS ≈ 0; rate-axis dominates predicted band.

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
    score_improvement_mechanism_status="OPERATIONAL",  # parser/decoder bytes are consumed
    runtime_overlay_consumed=True,  # entropy-coded sigma/class bytes feed AC decoder
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

byte_mutation_smoke_passes = "verify via tools/verify_z3_g1_entropy_coded_v2_byte_mutation.py per Catalog #139; mutates distinguishing bytes and records semantic_output_mutation evidence separately from parser_bound_consumption (extincts F1 phantom-class overclaim)"
```

## 6. Reactivation gate

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" + Catalog #220:
v2 declares parser/decoder consumption but stays `research_only=true` and
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
  (Wunderkind G1 SUBSTITUTION-1:1 spec — predicted ΔS [-0.005, -0.015]).
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
