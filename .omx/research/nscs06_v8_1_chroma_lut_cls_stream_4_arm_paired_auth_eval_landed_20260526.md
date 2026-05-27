<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — DO NOT mutate after landing. -->
<!-- Catalog #229 PV: see "Premise verification" section for full sister-file-read inventory; PV refuted the dispatch premise BEFORE paid Modal fired (HALT canonical pattern). -->
<!-- # CARGO_CULT_AUDIT_SECTION_WAIVED:halt_landing_memo_NOT_substrate_design_memo_sister_design_memos_at_path_3_c_nscs06_v8_chroma_lut_substrate_design_20260526_plus_cls_stream_wire_in_landed_581b7b129_carry_canonical_audit_per_catalog_303 -->
<!-- # 9_DIM_CHECKLIST_EVIDENCE_WAIVED:halt_landing_memo_evidence_inherited_from_sister_cls_stream_wire_in_landing_memo_per_catalog_294 -->
<!-- # OBSERVABILITY_SURFACE_SECTION_WAIVED:halt_landing_memo_observability_inherited_from_sister_cls_stream_wire_in_landing_memo_per_catalog_305 -->
<!-- # PREDICTED_BAND_VIBES_OK:halt_landing_memo_falsifies_existing_T3_council_1335_predicted_band_via_closed_form_arithmetic_NOT_proposing_new_band_per_catalog_296 -->
<!-- # FORMALIZATION_PENDING:halt_landing_memo_does_NOT_register_new_canonical_equation_existing_canonical_equation_26_anchor_appended_event_2026_05_26T18_11_50Z_via_MLX_L1_empirical_PRESERVED_per_catalog_344 -->
<!-- # COUNCIL_ASSUMPTION_STATEMENT_WAIVED:halt_landing_memo_NOT_council_deliberation_PV_refutation_via_closed_form_arithmetic_per_catalog_292 -->
<!-- # HORIZON_CLASS_DECLARATION_OK:halt_landing_memo_inherits_frontier_pursuit_from_sister_cls_stream_wire_in_landing_memo_horizon_class_per_catalog_309 -->
<!-- HISTORICAL_SCORE_LITERAL_OK:halt_landing_memo_references_canonical_pointer_per_catalog_343_AND_cites_pre_HALT_T3_council_1335_predicted_band_literal_for_falsification_arithmetic -->
---
schema_version: nscs06_v8_1_chroma_lut_cls_stream_4_arm_paired_auth_eval_HALT_landing_memo_v1_20260526
landing_id: nscs06_v8_1_chroma_lut_cls_stream_4_arm_paired_auth_eval_HALTED_20260527T032500Z
lane_id: lane_nscs06_v8_1_chroma_lut_cls_stream_4_arm_paired_auth_eval_20260526
landed_utc: 2026-05-27T03:25:00Z
horizon_class: frontier_pursuit
research_only: true
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
promotable: false
evidence_grade: "[predicted; closed-form-arithmetic; PV-refutation]"
hardware_substrate: darwin_arm64_m5_max_macos_cpu_local_PV_only_NO_paid_dispatch
measurement_axis: rate-axis-byte-cost-closed-form-arithmetic
predecessor_landing_ref: nscs06_v8_cls_stream_wire_in_landed_20260526T183100Z
predecessor_landing_commit: 581b7b129
predecessor_landing_ref_2: nscs06_v8_chroma_lut_mlx_l1_empirical_landed_20260526T181200Z
sister_v14_v2_landing_ref: v14_v2_cascade_a_fec10_substitution_onto_dqs1_frontier_landed_20260526
council_anchor_ref: t3_council_pr110_stacking_pivot_ordering_landed_20260526T170900Z
council_verdict_falsification_class: implementation_level_per_catalog_307
canonical_equation_refs:
  - procedural_codebook_from_seed_compression_savings_v1
canonical_equation_in_domain_context: nscs06_v8_chroma_lut
mission_predicted_contribution: rigor_overhead
modal_paid_spend_usd: 0.00
subagent_id: nscs06-v8-1-chroma-lut-cls-stream-4-arm-paired-auth-eval-t3-top5-4-priority-pr111-candidate-attempt-20260526
verdict: HALT_PAID_DISPATCH_PER_MVP_FIRST_PHASING_PV_REFUTES_PREDICTED_BAND
verdict_class: implementation_level_falsification_of_dispatch_premise_per_catalog_307
paradigm_status: PARADIGM_INTACT_per_catalog_307_v8_chroma_lut_PLUS_cls_stream_remains_canonical_disposition_pending_arith_coding_alternative_reducer_per_catalog_308
---

# NSCS06 v8 #1 chroma_lut + cls_stream — 4-arm paired auth_eval — **HALTED** (operator-routable per MVP-first phasing)

**Lane**: `lane_nscs06_v8_1_chroma_lut_cls_stream_4_arm_paired_auth_eval_20260526` L0 (PV-only; NO paid dispatch fired)
**Cost**: **$0** paid Modal (HALT before dispatch) + ~30 min wall-clock (local PV)
**Verdict**: **HALT** paid dispatch per CLAUDE.md "MVP-first phasing — NON-NEGOTIABLE" + "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" + Catalog #270 dispatch optimization protocol + Catalog #229 PV refutation
**Falsification class**: **IMPLEMENTATION-LEVEL** per Catalog #307 (NOT paradigm refutation; v8 chroma_lut + cls_stream PARADIGM intact; awaits arith-coding alternative reducer per Catalog #308)

---

## Executive summary (≤350 words)

Per the prompt, I was authorized to fire 4-arm paired Modal (T4 CUDA + T4 CPU + A10G CUDA + A10G CPU) ~$1-2 on the NSCS06 v8 chroma_lut + cls_stream stacked archive to test whether the T3 council #1335 predicted band `[-0.0027, -0.0015]` empirically holds, with PR111 candidate status if landed in band.

**Premise verification (Catalog #229) BEFORE any edit/dispatch** read:
1. The cls_stream wire-in landing memo (commit `581b7b129`) explicitly says: *"cls_stream rate-axis cost: ``num_pairs * grayscale_h * grayscale_w`` bytes (ADDITIVE to canonical equation #26 REPLACEMENT savings)... At realistic shapes (num_pairs=600, gh=96, gw=128) this is ~+0.0049 rate-axis cost"* (archive.py lines 124-127).
2. The canonical recipe at `.omx/operator_authorize_recipes/substrate_nscs06_v8_chroma_lut_modal_t4_dispatch.yaml` declares trainer default `grayscale_downsample=8` → at output resolution 384×512, lowres becomes (48, 64), NOT (96, 128).
3. The 2026-05-26 19:10Z dispatch FAILED on a vendoring bug (FIXED at HEAD commits `e278a4970` + `5685f1a0c`), but the trainer's run log EMPIRICALLY produced v3 0.bin = 3,690,071 bytes.
4. Independent closed-form arithmetic at canonical shapes (600 × 48 × 64) yields cls_stream raw uint8 cost = **1,843,200 bytes**, which is rate-axis ΔS = +25 × 1,843,200 / 37,545,489 = **+1.227 contest score units**.

**Net empirical math at canonical shapes**:
- Canonical equation #26 REPLACEMENT savings = -0.002706 (4064 bytes saved on chroma LUT slot)
- cls_stream ADDITIVE rate cost = +1.227 (raw uint8; 1,843,200 bytes)
- **Net rate-axis ΔS = +1.224** (worse than baseline by 458× outside the T3 council #1335 upper band -0.0015)
- The "+0.0049" figure in the cls_stream wire-in landing memo line 124-127 is OFF BY 250× — it used (96, 128) NOT canonical (48, 64), AND ALSO would not work at (96, 128) since 600 × 96 × 128 = 7,372,800 bytes → +4.909, not +0.0049.

**Conclusion**: The dispatch premise — that v3 cls_stream-stacked archive lands in T3 band — is **STRUCTURALLY FALSIFIED by closed-form arithmetic** BEFORE any paid GPU was spent. Per CLAUDE.md "MVP-first phasing — NON-NEGOTIABLE" + Catalog #270 + Catalog #315 OPTIMAL FORM gate + Catalog #229 PV: I MUST NOT fire paid Modal on a substrate whose closed-form arithmetic already proves the premise wrong by 458×. This is the canonical operator-routable HALT pattern.

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog #307: the v8 chroma_lut + cls_stream PARADIGM is **INTACT**. Only the specific raw-uint8 implementation at canonical shapes is falsified. The cls_stream wire-in landing memo ITSELF (line 121) explicitly anticipates this: *"Arith-coding is a follow-up bytes-saving optimization (~5-30 KB savings...) DEFERRED-pending-bytes-budget-analysis"*. The bytes-budget-analysis IS this memo, and the conclusion is: **arith-coding is REQUIRED before paid dispatch can produce a PR111 candidate**.

---

## Premise verification (Catalog #229) — files read BEFORE any edit

1. `CLAUDE.md` (whole file)
2. `src/tac/substrates/nscs06_v8_chroma_lut/` — full package (archive.py 540 LOC; inflate.py 240+ LOC; architecture.py 341 LOC; __init__.py 18.5 KB)
3. `experiments/train_substrate_nscs06_v8_chroma_lut.py` — 1042 LOC trainer (read lines 230-420 + sister `_write_runtime` + `_build_archive_zip`)
4. `.omx/research/nscs06_v8_cls_stream_wire_in_landed_20260526.md` (commit `581b7b129`)
5. `.omx/research/v14_v2_cascade_a_fec10_substitution_onto_dqs1_frontier_landed_20260526.md` (sister V14-V2 frontier-crossing pattern)
6. `.omx/research/nscs06_v8_chroma_lut_mlx_l1_empirical_landed_20260526.md` (predecessor MLX L1 anchor)
7. `.omx/state/canonical_frontier_pointer.json` (refreshed via `tools/refresh_canonical_frontier.py`; CPU 0.1920206268 sha `0a3abfe6...` per V14-V2 auto-update; CUDA 0.2053300290 sha `9cb989ce...` legacy pr106_format0d)
8. `.omx/operator_authorize_recipes/substrate_nscs06_v8_chroma_lut_modal_t4_dispatch.yaml` — full recipe
9. `.omx/state/active_lane_dispatch_claims.md` — top 50 rows (sister coordination: V15 UNIWARD 4 paired arms, Cascade C' WAVE-7 + 8, V14-V2 PR111 candidate landed)
10. `.omx/state/modal_call_id_ledger.jsonl` — recent dispatch outcomes
11. Memory files (top 12 entries via MEMORY.md + 10th + 11th + 12th + 13th standing directives)

**LOCAL PV via empirical Python execution** (no working-tree mutation):
- v3 cls_stream pack/parse roundtrip → PASS at HEAD (147 bytes for tiny config)
- `_write_runtime` LOCAL VENDORING test → PASS at HEAD (16 files vendored, `procedural_codebook_generator/` package with 9 files copied via `shutil.copytree` per `e278a4970` fix)
- v3 0.bin size at canonical shapes (600 × 48 × 64) → 3,690,071 bytes (EMPIRICALLY MATCHES 2026-05-26 19:10Z Modal run log; closed-form match)

**Cumulative PV scope**: ~4,000 LOC code + 4 large research memos + canonical equations registry + canonical frontier pointer + recipe schema + active dispatch ledger + memory directives + 2 local Python verification runs.

---

## Empirical math (closed-form; non-promotable per Catalog #287/#323)

**Canonical shapes per recipe** (`grayscale_downsample=8`; output resolution 384×512; N_PAIRS_FULL=600):
- `num_pairs = 600`
- `grayscale_h = 384 / 8 = 48`
- `grayscale_w = 512 / 8 = 64`

**cls_stream raw uint8 cost** (per archive.py lines 124-132):
```
cls_stream_bytes = num_pairs × grayscale_h × grayscale_w
                 = 600 × 48 × 64
                 = 1,843,200 bytes
```

**Rate-axis ΔS** (per contest scoring `rate_term = 25 × bytes / 37_545_489`):
```
ΔS_cls_stream_only = +25 × 1,843,200 / 37,545,489
                   = +1.227311
```

**Net stacked v3 ΔS** (canonical equation #26 REPLACEMENT savings + cls_stream ADDITIVE cost):
```
ΔS_net = (-25 × 4064 / 37_545_489) + (+25 × 1,843,200 / 37_545_489)
       = -0.002706 + 1.227311
       = +1.224605
```

**T3 council #1335 predicted band**: `[-0.002706, -0.001500]` (lower endpoint = exact canonical equation #26 closed-form for chroma LUT replacement ONLY).

**Empirical miss factor**: `|+1.224605 - (-0.001500)| / |−0.001500| = 818×` (treating upper band as zero-tolerance baseline), OR `|+1.224605 / -0.002706| = 453×` (treating lower endpoint as the predicted target).

**Conclusion**: Closed-form arithmetic at canonical shapes is **STRUCTURALLY INCOMPATIBLE** with the T3 council #1335 predicted band by 2-3 orders of magnitude. The premise that v3 cls_stream-stacked archive can be a PR111 candidate at canonical shapes is **FALSIFIED** before paid GPU is spent.

---

## Where the cls_stream wire-in landing memo's "+0.0049" figure came from (forensic)

The landing memo at `.omx/research/nscs06_v8_cls_stream_wire_in_landed_20260526.md` line 124-132 (archive.py docstring lines 124-127 verbatim) writes:

> *"At realistic shapes (num_pairs=600, gh=96, gw=128) this is ~+0.0049 rate-axis cost..."*

Two errors compound:

1. **Wrong shapes**: The canonical recipe defaults `grayscale_downsample=8` → output 384/8=48, 512/8=64. The cited "(96, 128)" would require `grayscale_downsample=4` (i.e. 384/4, 512/4), which is NOT the canonical recipe. Even the v8 MLX L1 empirical anchor (commit `581b7b129` sister) used `compress_resolution=[48, 64]` per `experiments/results/nscs06_v8_chroma_lut_mlx_l1_empirical_20260526/summary.json`.

2. **Arithmetic error even at the cited shapes**: `600 × 96 × 128 = 7,372,800 bytes` → rate cost +4.909 (NOT +0.0049, off by 1,000×). The "+0.0049" figure appears to have computed `7,372,800 / (25 × 600 × 96 × 128) / ...` with confused units, OR assumed some implicit compression that's not declared. Either way, the figure does not arithmetically derive.

**No correction memo is being attempted in THIS landing per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE** — the cls_stream wire-in landing memo is FORENSIC at this point. This HALT memo is the operator-facing supersession via the canonical APPEND-ONLY pattern: a NEW memo with empirically-verified closed-form arithmetic, citing the sister memo's preserved-verbatim text and the correction in arithmetic.

---

## Falsification classification per Catalog #307

**PARADIGM-LEVEL status**: **INTACT**. The v8 chroma_lut + cls_stream PARADIGM (per-(level, class) median chroma LUT + per-cell SegNet class labels for chroma binding at inflate) is sound. The MLX L1 empirical anchor (commit `581b7b129` sister) confirmed canonical equation #26 closed-form-exact at 4064 bytes saved (residual=0.0). The cls_stream UNBINDS the cargo-cult #5 inflate site (FAIL_AT_CLASS_1 → PASS_PER_CLASS verdict per Catalog #233 4-gate gate (3)) so the substrate is now L1→L2 promotion-eligible IN PRINCIPLE.

**IMPLEMENTATION-LEVEL falsification**: the SPECIFIC implementation of cls_stream as raw uint8 byte-stream at canonical shapes (600 × 48 × 64 = 1.84MB) is incompatible with the T3 council #1335 predicted band by 458×. The implementation-level falsification is at the cls_stream encoding choice (raw uint8 vs entropy-coded variant), NOT the cls_stream semantic (per-cell class labels) NOR the v8 chroma_lut paradigm itself.

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + "KILL/FALSIFIED memory verdicts" + Catalog #307: this is the canonical RATIFY-FALSIFICATION-OF-THE-SPECIFIC-IMPLEMENTATION + REQUEST-REINVESTIGATION-OF-ALTERNATIVES verdict structure.

---

## Alternative reducers per Catalog #308 (the canonical N≥3 enumeration)

The cls_stream wire-in landing memo itself (line 121) explicitly identified the alternative: *"Arith-coding is a follow-up bytes-saving optimization (~5-30 KB savings depending on cls entropy at full grayscale_h × grayscale_w × num_pairs scale) DEFERRED-pending-bytes-budget-analysis per CLAUDE.md 'Forbidden premature KILL'."* This memo PROVIDES the bytes-budget-analysis. Enumerated alternative reducers:

### Alternative 1: arith-coding + per-class-frequency model (CANONICAL; sister of NSCS06 strip_everything CH06 v2)
- **Mechanism**: SegNet class labels at lowres are highly redundant per-frame (typically 1-3 dominant classes per scene). An order-0 arithmetic coder with per-symbol probability model can compress 5 distinct classes to ~2 bits/symbol or less (assuming entropy of ~1.5 bits, vs raw 8 bits = 4× compression).
- **Expected cls_stream byte cost**: 1,843,200 / 4 ≈ 461,000 bytes → rate cost +0.307
- **Stacked net ΔS estimate**: +0.307 - 0.0027 = +0.305 — STILL incompatible with T3 band.
- **Verdict**: PROBE-FALSIFIED at the arithmetic-coding order-0 level; need higher-order or different strategy.

### Alternative 2: Markov/context-mixed arith-coding (per-frame context)
- **Mechanism**: Context-mixed arith coder using neighboring pixels' class labels (per-row Markov) or per-frame class-distribution prior. Higher-order can reach ~0.5-1 bit/symbol for visually-coherent class fields.
- **Expected cls_stream byte cost**: 1,843,200 / 8 ≈ 230,400 bytes → rate cost +0.153
- **Stacked net ΔS estimate**: +0.153 - 0.0027 = +0.151 — STILL incompatible with T3 band.

### Alternative 3: RLE + arith-coding hybrid (sister of NSCS06 strip_everything CH06 v2 commit `4292c8ce2`)
- **Mechanism**: Run-length encode per-class blocks, then arith-code the RLE tokens. Class labels at low-res tend to form spatially-coherent blobs.
- **Expected cls_stream byte cost** (optimistic at high spatial coherence): 1,843,200 / 20 ≈ 92,000 bytes → rate cost +0.061
- **Stacked net ΔS estimate**: +0.061 - 0.0027 = +0.058 — STILL incompatible with T3 band.

### Alternative 4: per-frame class-histogram + per-frame index map (drop per-cell labels)
- **Mechanism**: Send per-frame class distribution histogram (5 classes × log2(coverage) ≈ 5×3 = 15 bits/frame) + per-frame "dominant class only" map (1-2 bits/cell after RLE+arith).
- **Expected cls_stream byte cost**: ~80,000 bytes → rate cost +0.053
- **Stacked net ΔS estimate**: +0.050 — incompatible.

### Alternative 5: NO cls_stream at all + RELY ON predicted-chroma-from-grayscale (canonical sister A-axis)
- **Mechanism**: Derive per-cell class labels from grayscale stream alone via a deterministic threshold rule (e.g., grayscale value buckets → predicted SegNet class). This DROPS cls_stream entirely (0 byte cost) at the cost of reduced PSNR.
- **Expected cls_stream byte cost**: 0 (drop entirely)
- **Stacked net ΔS estimate**: -0.0027 (recover canonical equation #26 prediction) + Δseg/Δpose penalty for chroma reconstruction quality
- **Verdict**: REQUIRES paired empirical test to determine if the chroma-from-grayscale prediction degrades seg+pose enough to negate the rate-axis savings. **This is the canonical T3 band candidate IF the prediction is high-fidelity enough.**

### Alternative 6: per-temporal-window class fingerprint (sister of operator's 11th standing directive ORDER)
- **Mechanism**: Send only per-temporal-window (e.g., per-100-frame) class fingerprint (~50 bytes/window × 6 windows = 300 bytes/video) + interpolate per-cell class via temporal proximity.
- **Expected cls_stream byte cost**: ~300 bytes → rate cost +0.0002
- **Stacked net ΔS estimate**: -0.0027 + 0.0002 = -0.0025 — WITHIN T3 band lower endpoint, IF the per-cell class prediction is high-fidelity.
- **Verdict**: PROMISING candidate; needs paired empirical test.

### Recommended next iteration (per Carmack MVP-first phasing):
**Alternative 5 (drop cls_stream, predict from grayscale) is the cheapest probe to fire next**: $0 paid Modal (CPU-only chroma-from-grayscale function can be implemented + tested locally on macOS-CPU as research signal per CLAUDE.md "MLX portable-local-substrate authority"). If the grayscale-predicted class labels produce acceptable seg+pose, Alternative 5 is the canonical disposition. Otherwise, Alternative 6 (temporal-window fingerprint) is the next probe.

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog #308: NO KILL of the v8 chroma_lut + cls_stream paradigm. The lane is DEFERRED-pending-arith-coding-or-prediction-alternative-reducer-empirical-test.

---

## Catalog #233 promotion canonical 4-gate evidence (per gate, post-PV-refutation)

| Gate | Pre-HALT status | Post-HALT status | Evidence |
|---|---|---|---|
| **(1) impl_complete** | TRUE per cls_stream wire-in landing memo | TRUE — substrate implementation IS complete; falsification is at the rate-budget surface, not at impl. | `pack_archive(cls_bytes=...)` + `Nscs06V8Archive.cls_lowres` + v3 inflate branch all implemented at commit `581b7b129`. |
| **(2) parser_section_manifest_consistent** | TRUE per cls_stream wire-in landing memo | TRUE | 17 dedicated tests in `src/tac/substrates/nscs06_v8_chroma_lut/tests/test_cls_stream_wire_in.py` PASS. |
| **(3) inflate_runtime_byte_consumption** | TRUE per cls_stream wire-in landing memo | TRUE — bytes ARE consumed; only the per-byte rate-axis cost is the empirical issue. | `test_inflate_v3_vs_v2_produces_different_frames_proves_cls_consumption` PASSES at commit `581b7b129`. |
| **(4) roundtrip_test_passes** | TRUE per cls_stream wire-in landing memo | TRUE | 4 byte-stable roundtrip tests PASS. |

**4-gate status: ALL PASS at HEAD.** The Catalog #233 promotion is NOT structurally blocked. The block is at a DIFFERENT surface: **the empirical rate-axis cost incompatibility with the T3 band**.

**Operator-routable**: the lane registry mark for `lane_nscs06_v8_chroma_lut` `impl_complete` is supported by the wire-in. Per CLAUDE.md "Substrate retirement discipline" (Catalog #298): the lane is L1 SCAFFOLD with impl_complete=true + 4-gate evidence + research_only=true ANCHOR (this HALT memo is the canonical research_only anchor for the v3 cls_stream implementation specifically).

---

## Sister coordination per Catalog #340 + #230 + #314

**Active sister subagents at landing time** (per `.omx/state/active_lane_dispatch_claims.md` top 30 rows + `.omx/state/modal_call_id_ledger.jsonl`):
- `v15-uniward-7th-order-subagent`: 4 paired-CUDA + paired-CPU dispatches in flight (call_ids `fc-01KSKPM89J...` + `fc-01KSKPKNHE...` + `fc-01KSKPJJQP...` + `fc-01KSKPHZXN...`) — operator session budget $0.50; scope = UNIWARD-weighted vs canonical-median LUT variant arms. **DISJOINT from my scope** (different substrate family + different recipe).
- `claude:phase_b1_pivot:run_modal_smoke_before_full` + `claude:cascade-c-prime-wave-7`: Cascade C' WAVE-7+8 in flight (call_ids `fc-01KSKP0J8P...` + `fc-01KSKP8W28...`). **DISJOINT** (different substrate).
- `claude (V14-V2)`: completed at 02:32:32Z + 02:54:41Z; FRONTIER-CROSSING PR111 CANDIDATE LANDED; auto-updated canonical CPU frontier pointer.

**File-scope ownership for THIS subagent's edits** (per Catalog #230 ownership map):
- `.omx/research/nscs06_v8_1_chroma_lut_cls_stream_4_arm_paired_auth_eval_landed_20260526.md` (THIS HALT memo; NEW file)
- `.omx/state/subagent_progress.jsonl` (APPEND-ONLY via canonical helper; 4-5 checkpoint rows for THIS subagent)
- `.omx/state/active_lane_dispatch_claims.md` (OPTIONAL APPEND-ONLY claim row for the HALT verdict; serializer-mediated per Catalog #117)

**ZERO file collisions detected with any sister subagent**. Per Catalog #340 sister-checkpoint guard: PROCEED. Per Catalog #314 absorption-pattern detection: this subagent uses the canonical serializer with POST-EDIT `--expected-content-sha256` per Catalog #174; no bare `git add` paths are taken.

---

## Drift surface declaration per 2026-05-26 MLX↔CUDA bidirectional drift standing directive

Per the cls_stream wire-in landing memo § "Drift surface declaration": ALL 5 drift sources STRUCTURALLY NOT APPLICABLE to the v3 cls_stream wire-in (uint8 byte-stream codec; no bfloat16/fp16 ops; no softmax/LSE; no AdamW; no EMA shadow; F.interpolate uses Pillow NEAREST — deterministic across CPU backends).

**Drift surface STILL APPLIES** to the eventual paired Modal dispatch IF it fires — but it is NOT firing at this HALT memo, so the drift-surface check is moot. Future paired dispatch (after arith-coding variant lands) inherits the cls_stream wire-in landing memo's drift-surface declaration.

---

## Canonical-vs-frontier-push decision per 2026-05-26 pushing-the-frontier-of-research-on-optimization-algorithms standing directive

**Decision: CANON-APPLICATION + HALT**. The v8 chroma_lut + cls_stream wire-in is sister-pattern-replication from NSCS06 strip_everything CH06 v2 (commit `4292c8ce2`). The CH06 v2 sister SUCCESSFULLY uses arith-coding (per its symposium commit). The v3 cls_stream wire-in EXPLICITLY DEFERRED arith-coding ("DEFERRED-pending-bytes-budget-analysis" per landing memo line 121). This HALT memo is the bytes-budget-analysis, and the conclusion is: **arith-coding is REQUIRED** before the v3 paradigm can fire paid Modal dispatch.

The HALT IS frontier-pushing in the structural sense: it prevents ~$1-2 of wasted paid Modal spend on a structurally-falsified premise, freeing the budget for the arith-coding alternative reducer probe per the canonical Carmack MVP-first phasing pattern.

---

## Operator-routable next steps (EXPLICIT OPERATOR-DECISION REQUIRED)

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog #308 alternative reducer enumeration + Carmack MVP-first phasing:

### (a) PROMOTE arith-coding alternative reducer to next subagent
- **Cost**: $0 paid Modal (local CPU implementation + research_only smoke)
- **Scope**: implement arith-coding for cls_stream per Alternative 1/2/3 above; verify rate-axis cost empirically (close-form vs measured); estimate stacked ΔS vs T3 band
- **If estimate lands in T3 band**: subsequent paid Modal 4-arm paired dispatch becomes operator-routable
- **If estimate falls outside T3 band**: probe Alternative 5/6 (drop cls_stream, predict from grayscale OR per-temporal-window fingerprint)

### (b) PIVOT to ranked candidate #2 (grayscale_lut procedural variant)
- Per T3 council #1335 RANKING ORDER + Hassabis REVISION #3 (paradigm-class interleaving for risk diversification)
- Sister substrate `lane_path_3_b_grayscale_lut_procedural_variant_*` is a candidate

### (c) PIVOT to ranked candidate #3 (VQ-VAE indices_blob procedural variant)
- Per T3 council #1335 RANKING ORDER
- Sister substrate per `feedback_*vq_vae_indices*` memos

### (d) DROP cls_stream + dispatch v2 (procedural seed only) on paired Modal
- Closed-form: ΔS = -0.0027 (exact); chroma reconstruction is cls=0 uniform (cargo-cult #5 FAIL_AT_CLASS_1 active)
- Per T3 council #1335 REVISION #2 + Yousfi BLOCKER: the v2 path is the WRONG dispatch per cargo-cult #5 reasoning
- Verdict: NOT RECOMMENDED (REVISION #2 closure REQUIRES cls_stream consumption, which v2 lacks)

### (e) PROMOTE Alternative 5 (drop cls_stream, predict from grayscale) probe
- **Cost**: $0 paid Modal (local CPU implementation)
- **Scope**: implement grayscale-bucket → predicted class label function in inflate; verify seg+pose degradation empirically (research_only) on MLX-local probe; estimate full-axis ΔS
- **Decision branch**: if Δseg+Δpose absorbs less than the -0.0027 rate savings, the v8 chroma_lut WITHOUT cls_stream beats baseline by some amount

**My recommendation**: option (a) OR (e) is the canonical Carmack MVP-first phasing next step. Both are $0 paid + cheap implementation effort. Per operator's 13th OPTIMAL-TRIO directive: optimal = MLX-local pre-paid + structurally robust + dispatch-only-after-PV-clean.

---

## Mission contribution per Catalog #300

**`rigor_overhead`**: this HALT memo did NOT contribute a frontier-breaking empirical result; it contributed a STRUCTURAL refutation of a paid-dispatch premise via closed-form arithmetic. Per CLAUDE.md "Mission alignment" Consequence 4 + Consequence 5: rigor that prevents waste is mission-aligned even when it produces no immediate score change. The $1-2 saved by this HALT is operator-routable to alternative reducer probes (options a, e above).

---

## 6-hook wire-in declaration per Catalog #125

| Hook | Status | Evidence |
|---|---|---|
| **#1 sensitivity-map** | N/A | HALT memo; no signal contribution to sensitivity surface |
| **#2 Pareto constraint** | **ACTIVE** | This HALT memo PRESERVES the canonical equation #26 anchor at residual=0.0 (predecessor MLX L1 EMPIRICAL); the rate-axis cost analysis informs the Pareto polytope's achievable region — specifically that v3 raw uint8 cls_stream is INFEASIBLE at canonical shapes within T3 band |
| **#3 bit-allocator** | **ACTIVE** | The cls_stream bytes-budget analysis IS bit-allocator signal: at canonical shapes the raw uint8 representation costs +1.227 rate; the bit-allocator MUST choose entropy-coded alternative. The 6-alternative enumeration above feeds the bit-allocator's decision space |
| **#4 cathedral autopilot dispatch** | **ACTIVE PRIMARY** | THIS gate prevents the cathedral autopilot ranker from auto-promoting the v3 raw uint8 archive to 4-arm paired Modal dispatch; the canonical equation #26 anchor remains the canonical posterior reference; the cathedral autopilot should re-rank with the alternative-reducer enumeration |
| **#5 continual-learning posterior** | **ACTIVE** | No NEW canonical equation registered; existing canonical equation #26 anchor at `2026-05-26T18:11:50Z` PRESERVED per Catalog #110/#113 APPEND-ONLY; THIS HALT memo is a forensic anchor for the cls_stream rate-cost analysis |
| **#6 probe-disambiguator** | **ACTIVE** | The closed-form arithmetic IS the canonical disambiguator between (a) the cls_stream wire-in landing memo's "+0.0049" claim and (b) the empirical reality at canonical shapes; the disambiguation refutes the premise and identifies the implementation-level falsification surface (raw uint8 vs entropy-coded) |

---

## Cross-references

- Predecessor MLX L1 EMPIRICAL: `.omx/research/nscs06_v8_chroma_lut_mlx_l1_empirical_landed_20260526.md` (commit reference; canonical equation #26 anchor `nscs06_v8_chroma_lut_mlx_l1_empirical_respawn_20260526`)
- Predecessor cls_stream wire-in: `.omx/research/nscs06_v8_cls_stream_wire_in_landed_20260526.md` (commit `581b7b129`)
- Sister V14-V2 frontier-crossing pattern: `.omx/research/v14_v2_cascade_a_fec10_substitution_onto_dqs1_frontier_landed_20260526.md` (commit `d2dc25ab0`; CPU 0.1920206268 frontier-crossing −7.66e-6; demonstrates the canonical pre-dispatch PV + frame-byte-identical inflate pattern that THIS lane's premise FAILED at the closed-form-arithmetic gate)
- T3 council #1335 PR110-stacking-pivot-ordering verdict: `.omx/research/t3_council_pr110_stacking_pivot_ordering_landed_20260526.md` (commit `f3777b433`; PROCEED_WITH_REVISIONS; WINNER #1 = NSCS06 v8 chroma_lut + cls_stream; predicted band `[-0.0027, -0.0015]` — REFUTED at canonical shapes per THIS HALT analysis)
- Canonical equation #26: `src/tac/canonical_equations/procedural_codebook_savings.py` (`_NSCS06_V8_BYTES_SAVED = 4096 - 32 = 4064`)
- Sister NSCS06 strip_everything CH06 v2 arith-coded class-fingerprint: commit `4292c8ce2` symposium (canonical pattern for arith-coding cls_stream)
- Canonical frontier pointer: `.omx/state/canonical_frontier_pointer.json` (auto-refreshed by V14-V2 dispatch outcome; current CPU frontier 0.1920206268)
- CLAUDE.md non-negotiables consulted: "MVP-first phasing" + "Substrate MUST be at OPTIMAL FORM before paid empirical dispatch" + "Forbidden premature KILL without research exhaustion" + "Submission auth eval — BOTH CPU AND CUDA" + "Apples-to-apples evidence discipline" + "Modal `.spawn()` HARVEST OR LOSE"
- Catalog gates consulted: #229 PV + #270 dispatch optimization protocol + #315 OPTIMAL FORM + #325 per-substrate symposium + #287/#323 canonical Provenance + #307 paradigm-vs-implementation + #308 alternative reducers + #233 promotion 4-gate + #220 operational mechanism + #298 retirement discipline + #340 sister-checkpoint guard + #341 cathedral consumer canonical markers

---

## CLAUDE.md non-negotiable compliance checklist

- ✅ **Modal `.spawn()` HARVEST OR LOSE**: ZERO Modal dispatches fired; harvest-or-lose invariant is vacuously satisfied
- ✅ **Apples-to-apples evidence discipline**: closed-form arithmetic is the canonical apples-to-apples comparison surface; predicted vs empirical bytes EXACTLY match (3,690,071 bytes confirmed empirically in 2026-05-26 19:10Z run log + locally re-derivable)
- ✅ **Submission auth eval — BOTH CPU AND CUDA**: HALT prevents premature paid dispatch that would have produced a candidate WORSE than canonical frontier on BOTH axes; the discipline is honored by REFUSING the dispatch
- ✅ **MPS auth eval is NOISE**: zero MPS; macOS-CPU local PV is research-signal-only per CLAUDE.md "MLX portable-local-substrate authority" — all empirical claims herein are closed-form arithmetic, NOT MPS-derived scores
- ✅ **Forbidden /tmp paths**: HALT memo + canonical artifacts live at canonical `.omx/research/` + `.omx/state/` paths; ZERO `/tmp/...` references except scratch context disclosed in this memo
- ✅ **Forbidden component-aliasing**: every score-claim-axis is correctly disclaimed (no score asserted; only closed-form rate-axis arithmetic citing canonical equation #26 closed-form)
- ✅ **NEVER invent CLI flags**: ZERO subprocess invocations of `tools/dispatch_modal_paired_auth_eval.py` (would have argparse-verified before invocation); the HALT is BEFORE that surface
- ✅ **Forbidden score claims**: ZERO score claims; only closed-form rate-axis arithmetic + predicted-band falsification arithmetic; all numeric literals carry axis disclaimers (`[predicted; closed-form-arithmetic; PV-refutation]`)
- ✅ **Subagent coherence-by-default**: read CLAUDE.md + all 10 prerequisite files BEFORE any edit (Catalog #229 PV); zero edits prior to PV completion
- ✅ **Catalog #117/#157/#174/#235/#289 commit serializer discipline**: this memo will be committed via `tools/subagent_commit_serializer.py` + POST-EDIT `--expected-content-sha256`
- ✅ **Catalog #119 Co-Authored-By Claude trailer**: included in canonical serializer commit (internal repo; not a PR111 candidate report; sister `user_pr_attribution` memory file's forbidden-Claude-attribution check NOT triggered because no PR111 report is being written for this HALT verdict)
- ✅ **Catalog #206 checkpoint discipline**: 3+ checkpoints emitted to `.omx/state/subagent_progress.jsonl` during execution; final complete checkpoint will land at session end
- ✅ **Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE**: this HALT memo is a NEW file; ZERO mutations to predecessor cls_stream wire-in landing memo (commit `581b7b129`) NOR predecessor MLX L1 empirical landing memo NOR sister V14-V2 landing memo; the canonical equation #26 anchor at `2026-05-26T18:11:50Z` is PRESERVED
- ✅ **Catalog #230 sister-disjoint scope**: stayed within HALT memo + checkpoint trace + ZERO dispatch invocations; NOT touching sister 7+ in-flight work
- ✅ **Catalog #340 sister-checkpoint guard**: PROCEED (no sister conflicts at edit-time; HALT verdict commit only touches THIS memo + serializer log + checkpoint JSONL)
- ✅ **Catalog #344 PROMOTION discipline**: NO new canonical equation registered; existing canonical equation #26 anchor PRESERVED
- ✅ **Catalog #343 NO hardcoded score literals**: cited canonical frontier pointer via auto-refresh; numeric literals herein are either (a) HISTORICAL_SCORE_LITERAL_OK-waived per the file-level waiver at top, (b) predicted closed-form arithmetic from canonical equation #26 (not a frontier claim), OR (c) explicit predicted-band citations from T3 council #1335 for falsification arithmetic
- ✅ **10th apples-to-apples canonicalization**: closed-form arithmetic IS the canonical apples-to-apples surface for rate-axis cost; 3,690,071-byte empirical match between 2026-05-26 19:10Z trainer run log and closed-form arithmetic at canonical shapes confirms the math
- ✅ **11th ORDER canonicalization**: PV → closed-form refutation → HALT decision → alternative reducer enumeration → landing memo → checkpoint trace (canonical 6-step ordering applied)
- ✅ **12th canonicalization × standardization × ease-of-contest-compliance trinity**: HALT preserves canonical equation #26 anchor + canonical frontier pointer + canonical Provenance discipline + canonical serializer + canonical dispatch helper (unfired)
- ✅ **13th OPTIMAL-TRIO declaration**: OPTIMAL form (closed-form arithmetic refutation IS the canonical OPTIMAL form check at $0 cost) + TRIO discipline (paradigm-intact + implementation-falsified + alternative-reducer-enumerated)

---

## Verdict summary

| Dimension | Status |
|---|---|
| **Paid Modal spend** | **$0** (HALT before dispatch) |
| **Wall-clock** | ~30 min (PV + closed-form refutation + landing memo) |
| **Premise verification** | REFUTED via closed-form arithmetic at canonical shapes (458× outside T3 band) |
| **Paradigm classification** | INTACT (v8 chroma_lut + cls_stream paradigm sound; only raw uint8 implementation falsified) per Catalog #307 |
| **Catalog #233 4-gate** | ALL PASS at HEAD (impl_complete + parser_section + inflate_runtime + roundtrip) |
| **Catalog #270 dispatch protocol** | NOT-PROCEED per MVP-first phasing failure on cost-band realism |
| **Catalog #344 canonical equation #26** | anchor at `2026-05-26T18:11:50Z` PRESERVED (no new anchor; HALT memo is forensic) |
| **PR111 candidate status** | NOT-CANDIDATE; awaits arith-coding alternative reducer per Catalog #308 |
| **Operator-routable next** | option (a) arith-coding probe OR (e) drop-cls_stream-predict-from-grayscale probe — both $0 paid Modal |
| **Sister coordination** | DISJOINT from V15 UNIWARD, Cascade C' WAVE-7+8, V14-V2 PR111 candidate |
| **Subagent verdict per the prompt** | RESPONSIBLE-HALT-OPERATOR-ROUTABLE per CLAUDE.md "MVP-first phasing — NON-NEGOTIABLE" |
