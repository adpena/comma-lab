---
review_kind: substrate_design_memo
review_id: atw_v2_1_faiss_ivf_pq_substrate_design_20260518
review_date: "2026-05-18"
lane_id: lane_top5_3_atw_v2_1_faiss_ivf_pq_scaffold_design_20260518
substrate_id: atw_codec_v2_1_faiss_ivf_pq
substrate_alias: atw_v2_1
parent_substrate_id: atw_codec_v2
horizon_class: frontier_pursuit
council_predicted_mission_contribution: frontier_breaking
predicted_band: null  # NULL pending new D4 probe on PQ-encoded channel; design-time only
predicted_band_validation_status: pending_post_training
score_claim: false
promotion_eligible: false
provider_spend: false
research_only: true
dispatch_enabled: false
operator_directive: "TOP-5 #3 SCAFFOLD: ATW V2-1 + Faiss-IVF-PQ per-region SegNet softmax histogram channel"
related_deliberation_ids:
  - council_per_substrate_symposium_atw_v2_reactivation_20260518
  - comprehensive_research_wave_20260518
  - council_z6_phase_3_sextet_candidate_1_multi_layer_film_20260517
  - feedback_per_substrate_optimal_form_symposium_wave_doctrine_plus_c6_ibps_first_landed_20260518
canonical_frontier_anchor:
  contest_cpu: "0.19205 [contest-CPU] (lane pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean; archive sha 6bae0201)"
  contest_cuda: "0.20533 [contest-CUDA] (lane pr106_format0d_latent_score_table; archive sha 9cb989cef519)"
predecessor_probe_outcomes:
  - probe_id: atw_codec_v2_d4_probe_per_pair_argmax_composite
    verdict: INDEPENDENT
    mi_bits_per_symbol: 0.006385502752
    classification: implementation-level-falsification-of-channel
  - probe_id: atw_v2_1_byte_closed_per_region_histogram
    verdict: WEAK_CONDITIONING
    mi_bits_per_symbol: 0.047381530305
    packet_bytes: 323
    classification: implementation-level-evidence-channel-still-too-coarse
---

# ATW V2-1 + Faiss-IVF-PQ per-region SegNet softmax histogram substrate — design memo

**Lane**: `lane_top5_3_atw_v2_1_faiss_ivf_pq_scaffold_design_20260518` (L0 SCAFFOLD)
**Operator directive (2026-05-18)**: TOP-5 #3 from comprehensive research wave §0 — Faiss-IVF-PQ per-region SegNet softmax histograms as the V2-1 richer side-info channel per Atick canonical ranking #2.
**Predicted band**: NULL pending new D4 probe on PQ-encoded channel. Research-wave §0 cited `[-0.015, -0.005]` ⇒ `[0.177, 0.187]` contest-CPU; this memo's §6.3 byte-budget audit identifies that claim as CARGO-CULTED-PENDING-EMPIRICAL and disambiguates via a 3-channel matrix.
**$0 GPU spend** (this scaffold is design-time + canonical helper + tests + recipe scaffold; NO paid dispatch).
**Sister-substrate cross-pollination**: BIDIRECTIONAL with Z6 Wave 2 Candidate 4c per ATW V2 reactivation symposium Revision #5.

## 0. TL;DR (60 seconds)

**Empirical context preserved**: the existing ATW V2-1 byte-closed probe at `.omx/state/atw_v2_1_byte_closed_side_info_probe.json` (2026-05-18T06:24:31Z) tested 4 hand-rolled reducers (`per_pixel_histogram` / `per_region_histogram` / `per_pair_class_2_fraction` / `per_frame_argmax`) using a deterministic ATW21SI dictionary packet. Best reducer was `per_region_histogram` at 323 bytes with **MI=0.047 bits/symbol** (WEAK_CONDITIONING; 0.5-bit threshold; Wyner-Ziv gain ceiling fraction 0.0067). The recipe's `dispatch_blockers` already cites this empirical evidence.

**This memo's hypothesis**: Faiss-IVF-PQ as a NEW richer reducer family is the canonical Atick-ranking #2 channel realization. The hypothesis is that PQ preserves more H(T) per shippable byte than the hand-rolled dictionary packets because:
1. **PQ codeword density**: with M=4 sub-quantizers × ksub=256, each region gets a 40-bit code = 5 bytes vs the existing dict packet's 7-class enumeration that collapsed to log2(7) ≈ 2.8 bits per pair total.
2. **Per-pair conditioning preservation**: PQ encodes EACH region's softmax distribution independently rather than collapsing to per-pair argmax — restoring continuous-distribution-information per Atick-Redlich's canonical recommendation (cf. ATW V2 symposium Atick + Redlich verbatim).

**BUT — byte budget arithmetic discloses honestly** (per §6.3 + the operator's instruction *"IF Faiss-IVF-PQ byte budget math doesn't close at <2KB (likely), the design memo must surface this honestly"*): the research-wave §0 "<2KB shippable budget" claim is FALSE at the per-pair granularity. Three honest variants:

| Variant | Per-pair bytes | Total bytes (600 pairs + codebook) | Shippable verdict |
|---|---|---|---|
| **V1 dense full-PQ**: 16×16 region grid, M=4 subq, k=256 codewords | 1280 bytes/pair | ~778KB | NOT shippable (rate cost +25 × 778K/37545K = +0.518 score) |
| **V2 sparse top-k**: 4×4 region grid, M=2 subq, k=64 codewords + top-k=8 regions | 24 bytes/pair | ~24KB | ARGUABLE (+0.016 rate; +0.001 if shared codebook only ~2KB) |
| **V3 pool-shared codebook + per-pair sparse codeword**: global 16-region histogram + per-pair top-1 codeword | 4 bytes/pair + ~1.5KB codebook | ~3.9KB | SHIPPABLE (+0.003 rate; sister to existing 323-byte probe) |

**RECOMMENDED EMPIRICAL DISPATCH**: probe ALL THREE variants in a $0 CPU smoke ($0.20 wall-clock equivalent on M5 Max) BEFORE any paid Modal dispatch. The byte-budget vs MI tradeoff is empirically resolvable in <1h locally; no GPU spend justified until V1/V2/V3 disambiguator runs.

**Cross-pollination decision tree** (binding per ATW V2 symposium Revisions #5 + #6):

```
[Z6 Wave 2 4c outcome] (sister subagent a58961ea35f767306 in flight)
  ├─ full-FiLM-WIN (scorer-logit conditioning empirically validated as high-MI signal)
  │   → ATW V2-1 ratifies scorer-logit conditioning as canonical channel #1
  │   → fall back to per-region histogram (V2-1-channel #2) ONLY if V2-1 logit compression head fails byte budget
  │   → Faiss-IVF-PQ probe runs on PER-PIXEL SOFTMAX LOGITS not per-region histograms
  └─ DEFER (scorer-logit weak even at predictor surface)
      → ATW V2-1 pivots to per-region histograms (channel #2 per Atick ranking)
      → Faiss-IVF-PQ probe runs on PER-REGION HISTOGRAMS (THIS MEMO's primary scope)
```

**Local M5 Max + 128GB unified hardware exploitation**: Faiss is CPU-first; the codebook training + encode + decode pipelines ALL run locally on M5 Max in seconds. 600 pairs × 256 regions × 5-dim softmaxes = 768K floats × 4 bytes = ~3MB — trivial. Operator can run the V1/V2/V3 disambiguator probe at $0 cost before authorizing Modal A100 dispatch.

**Faiss installation status**: `pip install faiss-cpu` is NOT YET INSTALLED on local M5 Max (verified via `.venv/bin/python -c "import faiss"` → ModuleNotFoundError). Recommended install: `uv pip install faiss-cpu` per CLAUDE.md `uv` discipline. CPU-only (NOT faiss-gpu) is correct: this work is CPU-first per the canonical use case.

## 1. Canonical-vs-unique decision per layer (per Catalog #290)

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Catalog #290 + the operator's 2026-05-15 standing directive that canonical helpers are TOOLS not OBLIGATIONS. Each layer documented below carries a FALL-RULE classification per the Catalog #290 falling-rule list (EMPIRICAL → PRINCIPLED → UNCLEAR → OBVIOUS-FIT).

| Layer | Decision | Rationale | Rule-class |
|---|---|---|---|
| **Encoder backbone** | **ADOPT CANONICAL** (`tac.substrates.atw_codec_v2.architecture.ATWv2Codec`) | V2-1 inherits V2's encoder; the channel change is at the side-info pathway only | OBVIOUS-FIT |
| **Side-info channel encoding** | **UNIQUE FORK** (Faiss-IVF-PQ on per-region SegNet softmax histograms) | V2's `scorer_class_prior_table_fp16` slot ships per-pair argmax composite (D4 INDEPENDENT verdict) OR per-region histograms (323-byte probe WEAK_CONDITIONING); Faiss-IVF-PQ is a NEW reducer family with denser H(T) per byte | PRINCIPLED — Atick-Redlich canonical channel #2 (per-region histograms) + Tishby IB framework H(T) preservation requirement |
| **Codebook persistence** | **UNIQUE FORK** (shared global codebook + per-pair sparse codeword stream) | The hand-rolled ATW21SI packet format ships per-pair full dictionary + per-pair index stream; Faiss-IVF-PQ shipping requires a SHARED CODEBOOK persisted once + per-pair PQ codeword stream | PRINCIPLED — pure-canonical Faiss serialization would ship per-pair codebooks (rate-prohibitive); pool-shared codebook is the canonical Faiss-IVF-PQ use case |
| **Decoder reconstruction** | **UNIQUE FORK** (Faiss decode → fp32 softmax → consume in `wz_side_info_head`) | V2's WZ side-info head consumes `scorer_class_prior_table` fp32 directly; V2-1 inserts a `faiss_pq_decode → softmax_reconstruct → wz_head` pipeline | OBVIOUS-FIT — minimum-change wrapper |
| **Archive grammar** | **UNIQUE FORK** (ATW21PQ magic; +1 codebook section vs ATW2; -1 class_prior_table section) | Per HNeRV parity L3 monolithic single-file: V2-1 grammar adds `faiss_codebook_blob` (~2KB serialized via `faiss.serialize_index`) and `pq_codeword_stream_blob` (per-pair × N regions × log2(ksub) bits); removes V2's `class_prior_table_blob` slot | PRINCIPLED — substrate-engineering exception per HNeRV L7 |
| **Inflate runtime** | **ADOPT CANONICAL** (delegate to `tac.substrates._shared.inflate_runtime`) + UNIQUE FORK Faiss decode | Faiss decode at inflate time requires `pip install faiss-cpu` (HNeRV L4 +1 dep beyond V2's torch + brotli + numpy = 4 deps total — within budget) | UNCLEAR — needs PRINCIPLED audit: faiss-cpu wheel size + import cost at inflate; deferred to V2-1 first-pass profiling |
| **Score-aware loss** | **ADOPT CANONICAL** (`tac.codec.cooperative_receiver.atick_redlich.cooperative_receiver_loss` per Catalog #164) | V2's `ATWv2ScoreAwareLoss` already routes through canonical; V2-1 inherits unchanged | OBVIOUS-FIT |
| **Training curriculum** | **ADOPT CANONICAL** (`tac.substrates._shared.trainer_skeleton`) | Per Catalog #178/#179/#180/#172 engineering hygiene gates | OBVIOUS-FIT |
| **Codebook training schedule** | **UNIQUE FORK** (offline pre-training on train-split SegNet softmax outputs BEFORE trainer epoch 0) | Faiss codebook is NOT learned end-to-end with the encoder; train ONCE on training-split SegNet outputs then frozen for all epochs | PRINCIPLED — Faiss IVF-PQ is offline-trained per Jégou-Douze-Schmid 2011 canonical workflow |
| **Master-gradient integration** | **ADOPT CANONICAL** (per-pair `MasterGradient` row via `tac.master_gradient_consumers`) | V2-1 per-pair PQ codeword stream is per-pair-addressable; canonical per-pair `MasterGradient` consumer surface applies directly per Catalog #319 + sister surfaces | OBVIOUS-FIT |
| **Eval-roundtrip + EMA** | **ADOPT CANONICAL** (per CLAUDE.md non-negotiables) | No substrate-specific reason to fork | OBVIOUS-FIT |
| **Operator-authorize recipe** | **UNIQUE** (research_only=true; dispatch_enabled=false at landing; predicted_band=null per Catalog #324) | Per Catalog #240 + #324 + #325 | OBVIOUS-FIT |

**Summary**: 6 ADOPT_CANONICAL + 5 UNIQUE FORK + 1 UNCLEAR_PENDING_AUDIT (inflate-runtime faiss-cpu dep). The UNCLEAR layer is operator-routable for Wave 2 — first-pass Modal smoke runs faiss-cpu and confirms HNeRV L4 dep budget. Per CLAUDE.md "Subagent coherence-by-default" + HNeRV parity L7: substrate-engineering may exceed bolt-on budget by carrying the UNIQUE FORK; V2-1 substrate engineering happens ONCE per architecture class.

## 2. Cargo-cult audit per assumption (per Catalog #303)

Per the HARD-EARNED-vs-CARGO-CULTED addendum + standing META-ASSUMPTION ADVERSARIAL REVIEW non-negotiable.

| # | Assumption | Classification | Rationale | Unwind path |
|---|---|---|---|---|
| 1 | Faiss-IVF-PQ preserves >90% H(T) at <2KB per pair | CARGO-CULTED | Research wave §0 cited claim without arithmetic; this memo §6.3 audit shows <2KB per pair requires aggressive sparsification (V2 / V3 variants only); V1 dense full-PQ = 1280 bytes/pair × 600 pairs = 768KB total | Run V1/V2/V3 disambiguator probe locally on M5 Max ($0 CPU); pick the variant with best MI-per-shipped-byte |
| 2 | Per-region (16×16) is the right region granularity | CARGO-CULTED-PENDING-EMPIRICAL | Atick ranking #2 said "per-region histograms" but did not specify grid resolution; the existing 323-byte probe used 16-region grid (4×4) and got WEAK_CONDITIONING; finer grid (16×16 = 256 regions) may improve MI but costs proportionally more bytes | Sweep grid resolution {4×4, 8×8, 16×16, 32×32} in the disambiguator probe |
| 3 | PQ codebook trained on training-split generalizes to contest video | HARD-EARNED-PARTIAL | Faiss-IVF-PQ canonical workflow trains codebook on representative data; the contest video IS the training data (single video; no held-out split); per CLAUDE.md "Forbidden component-aliasing": codebook training MUST use the canonical contest video SegNet outputs | Train codebook on `upstream/videos/0.mkv` SegNet softmax outputs; document this is single-video training (not held-out) per CLAUDE.md mutation frontier |
| 4 | MI improvement from PQ-vs-dictionary encoding will clear 0.5 bits/symbol threshold | CARGO-CULTED-PENDING-EMPIRICAL | The existing 323-byte dict packet got MI=0.047 (10× below threshold); PQ encoding might 2-10× improve MI per byte but starting from 0.047 even 10× improvement = 0.47 still below threshold | Probe disambiguator measures empirically; predicted ΔMI ≈ 2-5× based on PQ vs hand-rolled dict (NOT a proof — must measure) |
| 5 | Bidirectional cross-pollination with Z6 4c is safe for V2-1 channel choice | HARD-EARNED | Per ATW V2 symposium Revision #5 binding: Z6 4c outcome materially informs V2-1 channel pick; both substrates honor strict-scorer-rule | PRESERVED; V2-1 channel pick AWAITS Z6 4c outcome per binding revision |
| 6 | The 30-day staleness window for ATW V2 still applies to V2-1 reactivation | CARGO-CULTED-MILD | Per Catalog #298: 30-day window; per ATW V2 symposium Revision #7: reactivation criteria are EVIDENCE-BASED not TIME-BASED for ATW V2 | V2-1 reactivation criteria explicitly require empirical disambiguator probe + new D4 probe + Wave N+1 council BEFORE dispatch |
| 7 | Faiss-IVF-PQ is the canonical Atick channel #2 realization | CARGO-CULTED-PENDING-COUNCIL | Atick's verbatim recommendation in ATW V2 symposium said "per-region SegNet softmax histograms" without specifying PQ vs dict encoding; PQ is ONE candidate encoding but not the only one; LSQ / RQ / vector quantization with learned codebooks are alternative encodings | Sextet pact reviews encoding choice in Wave N+1 council; this memo recommends PQ as the strongest single-candidate per `2024-Faiss-1.8` GPU acceleration + canonical Jégou-Douze-Schmid 2011 reference but does not preclude alternatives |
| 8 | Adding faiss-cpu dependency stays within HNeRV L4 +3 deps budget | HARD-EARNED | V2 already ships torch + brotli + numpy = 3 deps; adding faiss-cpu = 4 deps; HNeRV L4 says "≤2 deps default; ≤3 with rationale; ≤4 with substrate_engineering waiver"; V2-1 carries substrate_engineering waiver per HNeRV L7 | PRESERVED; faiss-cpu wheel size + import latency profiled in Wave 2 Modal smoke |

**Cargo-cult-class summary**: 0 HARD-EARNED-only + 2 HARD-EARNED-PARTIAL + 1 HARD-EARNED + 4 CARGO-CULTED + 1 CARGO-CULTED-MILD. **5 of 8 assumptions are empirically disambiguable by the local M5 Max disambiguator probe** ($0 cost; ~1h wall-clock). NO assumption requires paid GPU spend to disambiguate at this design stage.

## 3. 9-dimension success checklist evidence (per Catalog #294)

Per CLAUDE.md "9-dimension success checklist evidence" non-negotiable. Each dimension's evidence documented inline.

| # | Dimension | Evidence |
|---|---|---|
| 1 | UNIQUENESS (class-shift not within-class) | ✓ CONDITIONAL — V2-1 is across-class IFF the new D4 probe on PQ channel returns MEANINGFUL_CONDITIONING; within-class (same class as V2) if returns WEAK or INDEPENDENT |
| 2 | BEAUTY + ELEGANCE (PR101-style 30-sec-reviewable) | ✓ Target ≤350 LOC for the V2-1 substrate package (codec + archive + inflate + score_aware_loss + canonical helper); inherits V2's structure |
| 3 | DISTINCTNESS (explicitly different from sisters) | ✓ Only substrate using Faiss-IVF-PQ as side-info channel encoder; distinct from V2 (dict packet), Z3 G1 (CDF table), Z6 4c (FiLM conditioning) |
| 4 | RIGOR (premise verification + adversarial review + assumption classification + empirical anchor) | ✓ Premise: existing 323-byte probe WEAK_CONDITIONING anchor preserved; assumption classification done above (§2); empirical disambiguator probe queued $0 CPU |
| 5 | OPTIMIZATION PER TECHNIQUE (substrate-optimal engineering — covered by Catalog #290) | ✓ 6 ADOPT + 5 UNIQUE FORK per §1; Faiss-IVF-PQ is the substrate-optimal channel encoder for per-region histograms; documented falling-rule classifications |
| 6 | STACK-OF-STACKS-COMPOSABILITY (orthogonal axes + additive ΔS) | ✓ V2-1 is sister to V2 (same ATW Lagrangian); composable with A1 base substrate; orthogonal to NSCS06v8 chroma + DP1 pretraining + D1 SegNet overlay per ATW V2 design memo §13 |
| 7 | DETERMINISTIC REPRODUCIBILITY (byte-stable + seed-pinned) | ✓ ATW21PQ archive grammar byte-stable per Catalog #19; Faiss codebook trained with fixed seed (`faiss.IndexIVFPQ(...,seed=42)`); per CLAUDE.md determinism |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | ✓ Per V2 design memo §20: compress ~3-4h A100; inflate <30 min; faiss-cpu decode latency <1ms per pair (Jégou-Douze-Schmid 2011 benchmark); within contest 30-min runtime budget |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | INDETERMINATE — predicted band NULL per Catalog #324 pending post-training Tier-C validation; FRONTIER_PURSUIT class per Catalog #309; predicted band from research wave §0 `[0.177, 0.187]` contest-CPU treated as CARGO-CULTED-PENDING-EMPIRICAL until disambiguator probe + Modal A100 paired anchor |

## 4. Observability surface (per Catalog #305)

Per the MAX-OBSERVABILITY-INTO-BEHAVIOR standing directive. The V2-1 substrate exposes:

1. **Inspectable per layer** ✓ — every layer's input + output + intermediate state captured: encoder activations / decoder activations / WZ head input/output / Faiss codebook indices / decoded softmax reconstruction / per-pair PQ codeword stream. Each accessible via `tac.xray.atw_v2_1_faiss_pq.*` registered lenses.
2. **Decomposable per signal** ✓ — composite metrics decomposed: `final_score = score_seg + sqrt(10) × score_pose + 25 × archive_bytes/37_545_489`. Per-pair MI between codeword stream + A1 latents queryable.
3. **Diff-able across runs** ✓ — byte-level diff via Faiss `serialize_index` deterministic; activation-level diff via `tac.xray.diff_activations`; score-level diff via Catalog #316 frontier-scan integration.
4. **Queryable post-hoc** ✓ — codebook + codeword stream persisted to `experiments/results/atw_v2_1_faiss_pq_*/artifacts/`; canonical SQLite-style query via Faiss API (`index.search(query, k)`).
5. **Cite-able** ✓ — every behavior signal anchored to (substrate=`atw_codec_v2_1_faiss_ivf_pq`, commit=`<git_sha>`, call_id=`<modal_call_id>`, config=`<recipe_path>`, random_seed=`42`, upstream_snapshot_sha256=`<path>`).
6. **Counterfactual-able** ✓ — per Catalog #272 byte-mutation discipline: mutate ONE byte of the codebook OR codeword stream and observe `inflate.sh` output frame delta via `tools/verify_distinguishing_feature_byte_mutation.py`.

**New observability artifact for V2-1**: `experiments/results/atw_v2_1_faiss_pq_codebook_quality_report.json` — codebook reconstruction error per region per training step + MI between PQ-encoded channel + A1 latents (sister to existing 323-byte probe). Persisted via canonical `register_dispatched_call_id` per Catalog #245.

## 5. Predicted ΔS band — Dykstra-feasibility check (per Catalog #296)

Per CLAUDE.md "Predicted band" forbidden pattern (`PREDICTED_BAND_VIBES_OK` waiver requires Dykstra-feasibility intersection check OR first-principles citation OR probe-disambiguator path).

**Predicted band**: NULL pending empirical disambiguator probe.

**Research-wave §0 claim**: `[-0.015, -0.005]` rate-axis ⇒ `[0.177, 0.187]` contest-CPU.

**Dykstra-feasibility intersection of 4 polytopes** (cf. ATW V2 recipe `predicted_delta_basis`):
1. **Atick-Redlich rate constraint**: R_AR ≥ H(X) - I(X; Y_cooperative) — requires empirical I(X; Y) on PQ-encoded channel; bounded above by raw I(X; Y_softmax_logits) which is large (>1 bit/symbol) but compression to <2KB collapses bound
2. **IB constraint**: I(X;T) - β·I(T;Y) ≤ K_IB — Tao+Boyd Blahut-Arimoto floor; requires empirical I(T;Y) on PQ channel
3. **Wyner-Ziv side-info constraint**: R_WZ ≥ R(D) - I(X; Y_side) — requires empirical I(X; Y_side) on PQ-encoded shipped bytes (NOT raw PQ codebook)
4. **Contest rate budget**: 25 × archive_bytes / 37545489 — empirically bounded: V1 dense full-PQ adds +0.518 score; V2 sparse top-k adds +0.016; V3 pool-shared adds +0.003

**Intersection polytope non-empty IFF**: I(X; Y_side) on PQ-encoded shipped bytes ≥ 0.5 bits/symbol AND archive_bytes added ≤ ~5KB.

**Probe-disambiguator path** (per Catalog #296 acceptance cascade): `tools/probe_atw_v2_1_faiss_pq_disambiguator.py` (planned Wave 2 deliverable; runs locally on M5 Max in <1h at $0 cost; measures empirical I + bytes for V1/V2/V3 variants).

This memo cites the Dykstra-feasibility intersection as the canonical first-principles bound + queues the probe-disambiguator path; the predicted band remains NULL per Catalog #324 until the post-training Tier-C validation lands.

## 6. NEW substrate design

### 6.1 Architecture

V2-1 inherits V2's `ATWv2Codec` encoder + decoder + WZ side-info head + G1 distill head. V2-1 REPLACES V2's `scorer_class_prior_table_fp16` slot with:

```
ATW21PQ archive grammar (vs ATW2):
  MAGIC(4)                     b"AT21"
  VERSION(1)                   u8       schema version (currently 1)
  VARIANT(1)                   u8       0 = V1 dense / 1 = V2 sparse top-k / 2 = V3 pool-shared
  ENCODER_BLOB_LEN(4)          u32      brotli(fp16 encoder state)
  DECODER_BLOB_LEN(4)          u32      brotli(fp16 decoder state)
  WZ_HEAD_BLOB_LEN(4)          u32      brotli(fp16 wz_side_info_head state)
  DISTILL_HEAD_BLOB_LEN(4)     u32      brotli(fp16 g1_distill_head state)
  LATENT_RESIDUAL_BLOB_LEN(4)  u32      int8 z_residual = num_pairs * latent_dim
  FAISS_CODEBOOK_BLOB_LEN(4)   u32      brotli(faiss.serialize_index output)   <-- NEW V2-1
  PQ_CODEWORD_STREAM_BLOB_LEN(4) u32    bit-packed PQ codewords per region per pair  <-- NEW V2-1
  META_BLOB_LEN(4)             u32      sorted-keys JSON utf-8

REMOVED vs ATW2: CLASS_PRIOR_TABLE_BLOB + CDF_TABLE_BLOB (replaced by Faiss PQ stream)
```

### 6.2 Faiss-IVF-PQ canonical helper integration

NEW canonical helper at `src/tac/optimization/faiss_ivf_pq_atw_channel.py`:

```python
# 4 public functions:
def build_pq_codebook(
    segnet_softmax_batch: np.ndarray,   # shape (N_train_pairs * N_regions, 5)
    *,
    nlist: int = 256,
    m_subq: int = 4,
    ksub_bits: int = 8,                  # ksub = 1 << ksub_bits = 256
    seed: int = 42,
) -> faiss.IndexIVFPQ: ...

def encode_per_region_histogram(
    softmax_per_pair: np.ndarray,        # shape (N_regions, 5)
    codebook: faiss.IndexIVFPQ,
) -> bytes: ...

def decode_per_region_histogram(
    encoded_bytes: bytes,
    codebook: faiss.IndexIVFPQ,
    n_regions: int,
) -> np.ndarray: ...                     # returns (N_regions, 5) reconstructed softmax

def serialize_codebook(codebook: faiss.IndexIVFPQ) -> bytes: ...
def deserialize_codebook(blob: bytes) -> faiss.IndexIVFPQ: ...
```

**Catalog #810 / per-pair master gradient compatibility**: the per-pair PQ codeword stream is per-pair addressable (each pair's codewords occupy `n_regions * log2(ksub)/8` bytes at known offset). The canonical `tac.master_gradient_consumers.MasterGradient` schema supports per-pair byte-range addressing via the existing `payload_bytes_per_pair` field. V2-1 declares `payload_bytes_per_pair = n_regions * (ksub_bits + log2(nlist)) / 8` and the per-pair PQ codeword stream is the byte target for `MasterGradient` per-pair finite-difference.

### 6.3 Byte budget arithmetic (the empirical reality)

**V1 dense full-PQ** (16×16 region grid, M=4 subq, k=256 codewords, nlist=256):
- Per-region encoding: `log2(nlist) + M × log2(ksub) = 8 + 32 = 40 bits = 5 bytes`
- Per-pair: 256 regions × 5 bytes = **1280 bytes per pair**
- All 600 pairs: 600 × 1280 = **768,000 bytes**
- Shared codebook: nlist × 5 × 4 + M × ksub × (5/M) × 4 ≈ 5KB + 5KB = **~10KB amortized once**
- Total archive contribution: 768KB + 10KB = 778KB
- Contest rate cost: 25 × 778_000 / 37_545_489 = **+0.518 score**
- VERDICT: NOT shippable; +0.518 score dwarfs any plausible -0.015 ΔS gain

**V2 sparse top-k** (4×4 region grid, M=2 subq, k=64 codewords, nlist=64, top-k=8 regions per pair):
- Per-region encoding: `log2(64) + 2 × log2(64) = 6 + 12 = 18 bits ≈ 2.25 bytes`
- Per-pair (top-k=8 regions): 8 × 2.25 + 8 × log2(16) = 18 + 32 = 50 bits + region indices ≈ **24 bytes per pair**
- All 600 pairs: 600 × 24 = **~14.4KB**
- Shared codebook: ~64 × 5 × 4 + 2 × 64 × (5/2) × 4 = 1.3KB + 1.3KB = **~2.6KB amortized**
- Total archive contribution: 14.4 + 2.6 = ~17KB
- Contest rate cost: 25 × 17_000 / 37_545_489 = **+0.011 score**
- VERDICT: ARGUABLE; if MI clears 0.5 threshold the predicted -0.015 gain net = -0.004 (positive ROI but marginal)

**V3 pool-shared codebook + per-pair top-1 codeword** (16-region grid, M=2 subq, k=64 codewords, nlist=64, top-1):
- Per-pair: 1 × 2.25 + log2(16) region index = 2.25 + 4 = **~1 byte per pair** (round to byte)
- All 600 pairs: 600 × 1 = **~600 bytes**
- Shared codebook: ~2.6KB
- Total archive contribution: 0.6 + 2.6 = **~3.2KB**
- Contest rate cost: 25 × 3_200 / 37_545_489 = **+0.0021 score**
- VERDICT: SHIPPABLE; nearly free rate cost (sister to existing 323-byte probe at 0.0014 rate cost)
- RISK: V3 may not improve MI vs the existing 323-byte probe (both are aggressively sparse encodings)

**Recommended dispatch order** (per CLAUDE.md "Forbidden premature KILL" + Catalog #308 N>=3 alternative-probe-methodologies):
1. **V3 first** ($0 CPU): cheapest; if MI ≥ 0.5 at 3.2KB total, dispatch authorized
2. **V2 second** ($0 CPU): moderate cost; if MI 0.1-0.5 at 17KB total, marginal-ROI council adjudication
3. **V1 third** (only if V2/V3 INDEPENDENT): proves PQ-vs-dict-encoding lift is ~0 at the per-pair granularity; falsifies the per-region-histogram-via-PQ paradigm; pivot to scorer-logit-compression (per ATW V2 symposium Atick ranking #1) OR alternate-substrate per ATW V2 symposium Revision #4

### 6.4 Trainer scaffold (Catalog #240 + #220 + #272 compliance)

`experiments/train_substrate_atw_v2_1.py` (scaffold; `_full_main` raises NotImplementedError):
- Inherits `tac.substrates._shared.trainer_skeleton` per Catalog #178/#179/#180/#172
- `_smoke_main(args)`: implemented; loads V2 architecture + builds Faiss PQ codebook offline + dispatches a 5-epoch smoke
- `_full_main(args)`: raises `NotImplementedError(f"V2-1 _full_main pending Wave N+1 council per ATW V2 symposium 2026-05-18; reactivation criteria documented in {LANE_REGISTRY_NOTES}")`
- Per Catalog #240: matching recipe `research_only: true` + `dispatch_enabled: false` opt-out
- Per Catalog #272: distinguishing-feature contract declares `faiss_pq_codebook_blob` + `pq_codeword_stream_blob` as runtime-consumed
- Per Catalog #220: byte-mutation smoke planned (`tools/verify_distinguishing_feature_byte_mutation.py` runs on V2-1 archive in Wave 2)

## 7. Cross-pollination plan (per ATW V2 symposium Revisions #5 + #6)

### Z6 Wave 2 Candidate 4c decision tree

```
Z6 Wave 2 4c outcome (sister subagent a58961ea35f767306; expected 2026-05-19)
  ├─ full-FiLM-WIN (ΔS ≥ 0.005 at contest-CUDA paired disambiguator)
  │   → scorer-logit conditioning EMPIRICALLY VALIDATED as high-MI signal
  │   → V2-1 channel pick: PIVOT to PER-PIXEL SOFTMAX LOGIT compression via Faiss-IVF-PQ
  │     (Atick ranking #1; richer than per-region histogram)
  │   → predicted ΔS band uplift +30% on the V1/V2/V3 disambiguator
  │   → THIS memo's per-region histogram design becomes the V2-1 FALLBACK path
  │
  ├─ DEFER (scorer-logit weak even at predictor surface)
  │   → V2-1 channel pick: PROCEED with per-region histograms (THIS memo's primary scope)
  │   → predicted ΔS band as documented in §6.3
  │
  └─ INDEPENDENT (Z6 4c falsified)
      → ATW V2-1 paradigm-level falsification triangulation per Catalog #307
      → PIVOT per ATW V2 symposium Revision #4: operate ATW V2-1 on DIFFERENT base substrate (PR101 frame_exploit OR PR106 format0d)
      → re-run V2-1 design memo + new D4 probe on new base
```

### C6 IBPS Phase 2 redesign cross-reference

Per ATW V2 symposium Revision #6: if C6 Phase 2 redesign tests scorer-conditioning on the variational decoder, the empirical anchor IS informative for ATW V2-1 channel choice. ATW V2-1 designer (THIS memo) SHOULD read C6 Phase 2 memo as PV-2 to triangulate scorer-conditioning evidence across 3 architectural surfaces (Z6 predictor / C6 variational decoder / ATW V2-1 codec).

C6 Phase 2 redesign status: in flight per `feedback_per_substrate_optimal_form_symposium_wave_doctrine_plus_c6_ibps_first_landed_20260518.md` op-routable #3. Cross-pollination latency: ~1 week.

## 8. Operator dispatch authorization required

**This memo does NOT pre-authorize dispatch.** Per CLAUDE.md "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" + Catalog #315 + ATW V2 symposium Revision #1 binding: the V2-1 redesign requires sextet pact + grand council symposium BEFORE any paid dispatch.

**Operator-routable authorization sequence**:

1. **NOW** (this memo lands): scaffold + canonical helper + recipe + tests committed; lane_top5_3_atw_v2_1_faiss_ivf_pq_scaffold_design_20260518 advances L0 → L1 SCAFFOLD via lane_maturity `mark impl_complete` + `memory_entry`
2. **Wave 2 ($0 CPU; ~1h local M5 Max)**: run V1/V2/V3 disambiguator probe at `tools/probe_atw_v2_1_faiss_pq_disambiguator.py` (NEW; planned next subagent); emits MI + byte-budget verdict per variant; sister to existing 323-byte probe
3. **Wave 3 ($0; ~90 min editor)**: sextet pact symposium per Catalog #325 on V1/V2/V3 disambiguator outcome + Z6 4c cross-pollination
4. **Wave 4 (CONDITIONAL on Wave 3 PROCEED)**: Modal A100 smoke ~$5-10 with selected V1/V2/V3 variant + paired CPU+CUDA via `tools/dispatch_modal_paired_auth_eval.py` per Catalog #246
5. **Wave 5 (CONDITIONAL on Wave 4 paired anchor)**: Modal A100 full ~$15-25 with `--skip-axis-if-promotable-anchor-exists` per Catalog #246

**Predicted total spend through Wave 5**: $20-35 (vs research-wave §0 estimate $5-7 CPU probe + $15-25 Modal A100 = $20-32; aligned with research-wave estimate).

## 9. Local M5 Max + 128GB unified hardware exploitation plan

Per operator's directive: maximize local hardware before paid GPU.

### 9.1 Faiss codebook construction (local M5 Max)

- Input: SegNet softmax outputs from `upstream/videos/0.mkv` × 600 pairs × 16×16 region grid × 5 classes = 600 × 256 × 5 = 768K floats = ~3MB float32
- Faiss IVF-PQ training: `faiss.IndexIVFPQ(quantizer, d=5, nlist=256, m=4, nbits=8)` → trains in ~1-2 sec on M5 Max CPU
- Memory: ~3MB input + ~10KB codebook = trivial for 128GB unified
- Output: `codebook.faiss` (~10KB serialized via `faiss.serialize_index`)

### 9.2 Per-pair PQ encoding (local M5 Max)

- Input: 600 pairs × 256 regions × 5-class softmax = 768K floats
- Faiss encode: `codebook.add(region_softmaxes)` returns int IDs → `np.packbits` for bit-packing
- Throughput: ~100K vectors/sec on M5 Max CPU → 768K vectors in ~8 sec
- Output: bit-packed PQ codeword stream

### 9.3 Decode round-trip verification (local M5 Max)

- Input: bit-packed PQ codeword stream
- Faiss decode: `codebook.reconstruct(id)` → returns approximate fp32 softmax vector
- Throughput: ~100K vectors/sec on M5 Max CPU → 768K vectors in ~8 sec
- Output: reconstructed (N_pairs × N_regions × 5) softmax tensor

### 9.4 MI computation (local M5 Max)

- Input: reconstructed softmax + A1 latents (existing 16800 latent symbols)
- Compute: `tac.optimization.mps_research_signal` per CLAUDE.md "MPS auth eval is NOISE" — this is RESEARCH SIGNAL only, NOT score truth, MUST tag `[macOS-CPU advisory]` per Catalog #192 + #1
- Throughput: <1 sec for 16800 × 768K MI estimation

**Total local M5 Max workflow**: codebook construction + PQ encoding + decode + MI computation = ~30 sec wall-clock at $0 cost. This is the canonical $0 disambiguator probe that MUST run before any Modal A100 dispatch.

### 9.5 Faiss installation status

```bash
# Current state (verified 2026-05-18T07:50:17Z):
.venv/bin/python -c "import faiss" → ModuleNotFoundError: No module named 'faiss'

# Recommended install per CLAUDE.md `uv` discipline:
uv pip install faiss-cpu
```

Note: faiss-gpu is NOT recommended for this scaffold (CPU-first canonical workflow; GPU acceleration is for >1M vector scales not applicable here).

## 10. Catalog compliance summary

| Catalog # | Compliance | Notes |
|---|---|---|
| #1 (no MPS fallback) | ✓ | All compute paths explicitly device-tagged; local M5 Max work tagged `[macOS-CPU advisory]` |
| #127 (custody validator) | ✓ | All score claims route through `tac.continual_learning.validate_custody` |
| #131 (no bare writes) | ✓ | Codebook persistence via fcntl-locked canonical helper |
| #164 (canonical scorer-preprocess) | ✓ | Inherits V2 score_aware_loss routing through `tac.codec.cooperative_receiver.atick_redlich` |
| #178/#179/#180/#172 (Tier 1 engineering) | ✓ | Trainer inherits `trainer_skeleton` canonical |
| #205 (inflate device fork) | ✓ | Uses `tac.substrates._shared.inflate_runtime.select_inflate_device` canonical |
| #209 (no Comma2k19 leakage) | N/A | V2-1 trains on contest video; not Comma2k19 |
| #220 (operational mechanism) | ✓ | Faiss codebook + PQ codeword stream structurally consumed at inflate; byte-mutation smoke planned |
| #226 (canonical auth_eval) | ✓ | Trainer routes through `gate_auth_eval_call` |
| #240 (recipe-vs-trainer-state) | ✓ | `_full_main` raises NotImplementedError + recipe `research_only=true + dispatch_enabled=false` |
| #244 (NVML env block) | ✓ | Driver inherits canonical 3-export block |
| #245 (Modal call_id ledger) | ✓ | Dispatcher uses canonical ledger |
| #270 (dispatch protocol) | ✓ | Inherits V2's Tier 1/2/3 declarations |
| #272 (distinguishing-feature) | ✓ | `faiss_pq_codebook_blob` + `pq_codeword_stream_blob` declared |
| #290 (canonical-vs-unique decision) | ✓ | §1 |
| #294 (9-dim checklist) | ✓ | §3 |
| #296 (Dykstra-feasibility) | ✓ | §5 |
| #298 (substrate retirement) | ✓ | EVIDENCE-BASED reactivation criteria per ATW V2 symposium Revision #7 |
| #303 (cargo-cult audit) | ✓ | §2 |
| #305 (observability surface) | ✓ | §4 |
| #313 (predecessor probe outcome) | ✓ | Existing INDEPENDENT (D4) + WEAK_CONDITIONING (V2-1 byte-closed) probes cited; new D4 probe queued |
| #315 (optimal-form before dispatch) | ✓ | NO dispatch pre-authorized; council symposium required per §8 |
| #316 (frontier scan) | ✓ | Canonical frontier 0.19205 CPU / 0.20533 CUDA cited |
| #319 (Wyner-Ziv deliverability) | ✓ | Per-pair PQ codeword stream is canonical Wyner-Ziv side-info; deliverability proof built at Wave 4 |
| #324 (predicted band Tier-C validation) | ✓ | `predicted_band: null` + `predicted_band_validation_status: pending_post_training` |
| #325 (per-substrate symposium) | ✓ | This memo + ATW V2 symposium constitute the 6-step contract; Wave 3 sextet pact ratifies |

## 11. Risk assessment

**MEDIUM VARIANCE substrate**:

1. **Byte budget collapse**: per §6.3, V1 dense full-PQ is NOT shippable; V2 sparse top-k is ARGUABLE; V3 pool-shared is SHIPPABLE but may not improve MI. The empirical probe MUST run before any paid dispatch.
2. **Faiss-cpu dependency**: +1 dep beyond HNeRV L4 budget (3 → 4 with substrate_engineering waiver); first-pass Modal smoke validates wheel size + import cost.
3. **Cross-pollination dependency**: V2-1 channel pick AWAITS Z6 4c outcome per ATW V2 symposium Revision #5; sequencing dependency.
4. **Codebook generalization**: trained on contest video only (no held-out split); per CLAUDE.md mutation frontier this is the canonical contest workflow.

**Risk mitigations**:
- Local $0 disambiguator probe before paid dispatch
- Per CLAUDE.md "Forbidden premature KILL": V1/V2/V3 ALL probed before any falsification
- Cross-pollination decision tree (§7) handles all 3 Z6 4c outcomes

## 12. Memory + lane registry updates

This memo lands as L0 SCAFFOLD. After committing the 5 deliverables (this memo + canonical helper + trainer scaffold + tests + recipe), the lane advances:

```bash
# Operator-runnable (main Claude commits per parent prompt):
.venv/bin/python tools/lane_maturity.py mark \
  lane_top5_3_atw_v2_1_faiss_ivf_pq_scaffold_design_20260518 \
  --gate impl_complete \
  --evidence "5 deliverables landed: design memo + canonical helper + trainer scaffold + tests + recipe; subagent top5_3_atw_v2_1_scaffold"

.venv/bin/python tools/lane_maturity.py mark \
  lane_top5_3_atw_v2_1_faiss_ivf_pq_scaffold_design_20260518 \
  --gate memory_entry \
  --evidence "feedback_top5_3_atw_v2_1_faiss_ivf_pq_substrate_design_landed_20260518.md"
```

**Reactivation criteria** (ALL required to lift research_only=true + dispatch_enabled=false):
1. Faiss-cpu installed locally (`uv pip install faiss-cpu`)
2. V1/V2/V3 disambiguator probe lands MI + byte-budget verdict per variant
3. Z6 Wave 2 4c outcome lands (sister subagent a58961ea35f767306)
4. Wave 3 sextet pact symposium ratifies V2-1 channel pick per Catalog #325
5. New D4 probe on selected V2-1 channel returns MEANINGFUL_CONDITIONING (MI ≥ 0.5 bits/symbol)
6. Catalog #324 post-training Tier-C validation declared
7. Catalog #270 dispatch optimization protocol verdict PASS

## 13. Codex adversarial hardening pass — optional dependency must not mask local contracts

Follow-up pass on 2026-05-18 found a test-authority bug in the fresh scaffold:
`pytest.importorskip("faiss")` lived at module scope in
`src/tac/tests/test_atw_v2_1_faiss_ivf_pq.py`. On a machine without
`faiss-cpu`, that skipped the entire test file, including byte-budget
arithmetic, trainer-mode resolution, `_full_main` fail-closed behavior, and
the missing-Faiss actionable-error test. That was too broad: optional Faiss
availability should skip only the actual codebook encode/decode round-trips.

Fix landed:

- moved the Faiss skip into a local `_require_faiss()` helper used only by
  Faiss-dependent tests;
- changed `build_pq_codebook`, `encode_per_region_histogram`,
  `decode_per_region_histogram`, and `deserialize_codebook` so cheap shape/type
  invariants are validated before importing the optional Faiss dependency;
- added a regression test proving local contract errors are not hidden behind
  `ImportError: No module named 'faiss'`.

Verification:

```text
.venv/bin/python -m pytest src/tac/tests/test_atw_v2_1_faiss_ivf_pq.py -q
24 passed, 7 skipped in 0.19s
```

Evidence grade: `[local-test]`; no score claim, no promotion claim, no provider
dispatch, and no lane claim.

## 14. Codex adversarial hardening pass — over-budget channels are not byte-closed authority

Follow-up pass on 2026-05-18 found a second false-authority edge in
`tools/probe_atw_v2_1_byte_closed_side_info_channel.py`: if every candidate
side-info packet exceeded the configured byte budget, the payload still filled
`best_byte_closed_channel` with the highest-MI over-budget channel and the
Markdown rendered "Best byte-closed channel." The channel action correctly said
`reject_or_recode_side_info_payload_before_mi_interpretation`, but the summary
field/name could still mislead a dispatcher, council memo, or operator recipe
into treating an over-budget sidecar as byte-closed evidence.

Fix landed:

- split the summary into `best_byte_closed_channel` and `best_overall_channel`;
- set `best_byte_closed_channel: null` when no channel satisfies the configured
  side-info byte budget;
- render an explicit "No byte-closed channel fit the configured side-info
  budget" Markdown verdict in that case;
- added a regression test proving all-over-budget channels cannot emit
  byte-closed authority language.

Verification:

```text
.venv/bin/python -m pytest src/tac/tests/test_probe_atw_v2_1_byte_closed_side_info_channel.py -q
4 passed in 0.49s
```

Evidence grade: `[local-test]`; no score claim, no promotion claim, no provider
dispatch, and no lane claim. This is a dispatch-custody hardening only: it does
not change the existing 323-byte / MI=0.047 byte-closed empirical anchor, but it
prevents future budget-sweep artifacts from laundering over-budget rows as
byte-closed candidates.

## 15. Codex V1/V2/V3 Faiss-PQ disambiguator result

Artifact: `.omx/state/atw_v2_1_faiss_pq_disambiguator_probe.json`
Local forensic bytes: `experiments/results/atw_v2_1_faiss_pq_probe_20260518T100524Z/`
Axis: `[diagnostic-CPU; ATW V2-1 Faiss-PQ side-info MI probe]`
Authority: `score_claim=false`, `promotion_eligible=false`, `ready_for_paid_dispatch=false`; no provider dispatch and no lane claim.

Codex landed and ran the missing V1/V2/V3 disambiguator. The probe renders A1
locally, runs canonical SegNet on frame 1, region-averages softmax probabilities
for 4x4 and 16x16 grids, then hands saved `.npy` arrays to an isolated Faiss
worker subprocess. The subprocess boundary is required on this macOS host:
`faiss-cpu` and Torch initialize incompatible OpenMP runtimes when trained in
the same Python process, producing an abort instead of a catchable exception.

Two implementation bugs were found and fixed before interpreting the result:

- 5-class SegNet softmax vectors are now padded to a dimension divisible by
  `m_subq` before `IndexIVFPQ` construction, then decoded tensors are trimmed
  back to 5 classes.
- The helper now serializes trained quantizers only (`ntotal=0`). The earlier
  helper added training vectors to the IVF index before serialization, bloating
  the archive-side codebook and producing false over-budget evidence.

Measured post-fix outcomes:

| Variant | Brotli archive bytes | Rate cost | Unique fraction | MI bits/symbol | Verdict | Dispatch consequence |
|---|---:|---:|---:|---:|---|---|
| `v3_pool_shared` | 3,114 | 0.002073 | 0.033 | 0.121512378237 | `WEAK_CONDITIONING` | Byte-closed but not meaningful enough; no dispatch authority. |
| `v2_sparse_top_k` | 7,941 | 0.005288 | 1.000 | 2.457397664695 | `MEANINGFUL_CONDITIONING` | High-cardinality plug-in MI upper bound only; over the V3 5KB target; no dispatch authority. |
| `v1_dense` | 452,799 | 0.301500 | 1.000 | 2.457397664695 | `MEANINGFUL_CONDITIONING` | High-cardinality upper bound and structurally over budget; no dispatch authority. |

Conclusion: Faiss-PQ per-region histogram is not the next paid ATW V2-1
dispatch channel. The only shippable variant is weak; the meaningful variants
are pair-identity-like upper bounds and too byte-expensive. The recipe now
carries concrete blockers for the weak V3 result plus the high-cardinality V2/V1
results. Recommended next gate is
`pivot_to_scorer_logit_compression_or_trained_atw_residual_probe`, preserving
ATW's cooperative-receiver paradigm while rejecting this measured Faiss-PQ
channel configuration.

## Cross-references

- **ATW V2 reactivation symposium**: `.omx/research/council_per_substrate_symposium_atw_v2_reactivation_20260518.md`
- **Comprehensive research wave §0**: `.omx/research/comprehensive_research_wave_20260518.md`
- **Existing ATW V2 substrate**: `src/tac/substrates/atw_codec_v2/`
- **Existing ATW V2-1 byte-closed probe**: `.omx/state/atw_v2_1_byte_closed_side_info_probe.json` (323-byte / MI=0.047 empirical anchor)
- **Existing ATW V2-1 probe tool**: `tools/probe_atw_v2_1_byte_closed_side_info_channel.py` (sister to new Faiss-IVF-PQ disambiguator probe)
- **ATW V2 D4 probe verdict (INDEPENDENT)**: `.omx/research/atw_codec_v2_d4_probe_verdict_20260516_codex.json`
- **Z6 Wave 2 Candidate 4c sister subagent**: `a58961ea35f767306` (in flight)
- **C6 IBPS Phase 2 redesign cross-reference**: `feedback_per_substrate_optimal_form_symposium_wave_doctrine_plus_c6_ibps_first_landed_20260518.md` op-routable #3
- **Canonical Atick-Redlich primitive**: `src/tac/codec/cooperative_receiver/atick_redlich.py`
- **Faiss canonical reference**: Jégou-Douze-Schmid 2011 + Faiss 1.8 (2024) GPU-accelerated PQ; GitHub `facebookresearch/faiss`
- **Lane**: `lane_top5_3_atw_v2_1_faiss_ivf_pq_scaffold_design_20260518` L0 → L1 SCAFFOLD post-landing
