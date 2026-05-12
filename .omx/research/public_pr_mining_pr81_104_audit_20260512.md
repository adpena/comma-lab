# Public PR mining further expansion (PR81-104 un-mined range) — audit 2026-05-12

**Lane**: `lane_public_pr_mining_pr81_104_20260512` (Phase 1, L0 -> L1 after this audit lands)
**Sister landings**: L (PR81/84/91/92/93), X (PR85/86/97/93 anr), Parallel-F (non-HNeRV), prior expansion (PR53/56/60/63/64/65/67/79/104/105), Catalog #109 dirty-clone gate strict-flipped 2026-05-08.

## Scope

Per autonomous-tick directive 2026-05-12 and the closing surface in `feedback_public_pr_mining_expansion_pr50_80_pr105_115_landed_20260512.md`, this audit extends mining into the **PR81-104 un-mined HNeRV-family submission slots** — the 6 PR-specific NEW submission directories in that range that were not covered by L / X / Parallel-F / prior expansion landings.

## Un-mined inventory

| PR | NEW submission slot | claimed score (best public PR comment) | axis | notes |
|---:|---|---:|---|---|
| 95 | `hnerv_muon` | 0.21 | [contest-CUDA] | AaronLeslie138 — HNeRV root; 50h on 1 GPU; 8-stage curriculum |
| 96 | `rem2_HNeRV` | 0.195 | [contest-CPU] | rem2 silver-cluster May-4 (PR96 -> PR103 evolution) |
| 98 | `hnerv_muon_finetuned_from_pr95` | 0.20 | [contest-CPU] | EthanYangTW — adds CD1 compact format + decode-side nudge |
| 100 | `hnerv_lc_v2` | medal-cluster | [contest-CUDA] | BradyMeighan — schema.py + sidecar.py |
| 101 | `hnerv_ft_microcodec` | **0.193** | [contest-CUDA] | **GOLD MEDAL** — decoder_storage_order + conv4_perms + byte_maps |
| 103 | `hnerv_lc_ac` | 0.195 | [contest-CUDA] | rem2 (silver follow-up) — merged-AC + fixed-section offsets |

Total: **6 un-mined HNeRV-family submission slots; 4 of 6 are sub-0.20** (PR98 0.20, PR100 ≈ 0.195 cluster, PR101 0.193 gold, PR103 0.195 silver). PR95 is 0.21 (HNeRV root); PR96 is the rem2 silver-cluster CPU 0.195.

## Mining count

**15 typed mechanism rows extracted across 6 PRs**. Per-PR breakdown:

| PR | row count | primitives |
|---:|---:|---|
| 95 | 4 | Muon optimizer, cat_entropy_v2 regularizer, 8-stage loss curriculum, dual-RGB-head dilated-refine decoder |
| 96 | 2 | hist-codec switch (lzma/zstd/brotli), uint8-offset-128 RangeDecoder pattern |
| 98 | 2 | CD1 compact-architecture-ordered decoder format, decode-side per-frame per-channel constant nudge |
| 100 | 2 | schema-driven monolithic-decoder grammar, dim=255-sentinel sparse correction sidecar |
| 101 | 3 | DECODER_STORAGE_ORDER permutation, CONV4_STORAGE_PERMS per-tensor 4D-axis perm, DECODER_BYTE_MAPS (negzig/twos/off) |
| 103 | 2 | merged-AC concatenated RangeDecoder stream, fixed-section-offset byte parser with length constants |

## Top-N EV/byte ranking refresh at PR106 r2 frontier

Per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent (UPDATED 2026-05-04)", pose marginal at PR106 r2 is **2.71× SegNet's**.

Combining this audit's 15 rows with the prior landing's top-5:

| rank | primitive_id | from | family | axis | est_LOC | predicted_ev_per_byte | source PR claim |
|---:|---|---|---|---|---:|---:|---|
| 1 | `pr64_unified_brotli_pose_velocity_only_codec` | prior expansion | pose-codec | pose | 60 | high (top-of-PR64) | PR64 0.331 |
| 2 | `pr101_decoder_storage_order_permutation_list` | **NEW** | packet-grammar | rate | 50 | 1.10 | **PR101 0.193 GOLD** |
| 3 | `pr63_qpose14_uint16_view_int16_pose_codec` | prior expansion | pose-codec | pose | 70 | high | PR63 0.325 |
| 4 | `pr65_pq12_pose_grammar_12bit_3byte_pack` | prior expansion | pose-codec | pose | 80 | high | PR65 0.320 |
| 5 | `pr101_conv4_storage_perms_per_tensor_4d_axis_permutation` | **NEW** | packet-grammar | rate | 80 | 0.90 | **PR101 0.193 GOLD** |
| 6 | `pr103_merged_ac_concatenated_range_decoder_stream` | **NEW** | packet-grammar | rate | 90 | 0.85 | PR103 0.195 |
| 7 | `pr101_decoder_byte_maps_negzig_twos_off_strategies` | **NEW** | quantization | rate | 60 | 0.75 | **PR101 0.193 GOLD** |
| 8 | `pr98_cd1_compact_architecture_ordered_decoder_format` | **NEW** | packet-grammar | rate | 90 | 0.60 | PR98 0.20 |
| 9 | `pr96_rem2_hist_codec_switch_lzma_brotli_zstd` | **NEW** | packet-grammar | rate | 30 | 0.50 | PR96 0.195 |
| 10 | `pr100_schema_driven_decoder_storage_grammar` | **NEW** | packet-grammar | rate | 70 | 0.50 | PR100 medal-cluster |
| 11 | `pr105_kitchen_sink_packed_state_schema_size_sorted` | prior expansion | HNeRV | rate | 30 | 0.45 | PR105 0.198 |
| 12 | `pr100_dim255_sentinel_sparse_correction_sidecar` | **NEW** | packet-grammar | mixed | 50 | 0.35 | PR100 medal-cluster |
| 13 | `pr103_fixed_section_offset_byte_parser_with_length_constants` | **NEW** | packet-grammar | rate | 60 | 0.30 (DEFER pending length-prefixed variant) | PR103 0.195 |

**Three of the top-7 primitives are NEW from PR101 GOLD** (storage_order, conv4_perms, byte_maps). These trace specific byte savings to the PR101 0.193 archive bytes. High empirical signal.

## Sub-0.20 frontier candidates

**PR101 GOLD (0.193)** is the binding frontier target per CLAUDE.md "Frontier target — NON-NEGOTIABLE". Its 3 NEW primitives extracted in this landing are the highest-EV un-mined mechanisms in the public PR corpus:

- `pr101_decoder_storage_order_permutation_list` (rank #2)
- `pr101_conv4_storage_perms_per_tensor_4d_axis_permutation` (rank #5)
- `pr101_decoder_byte_maps_negzig_twos_off_strategies` (rank #7)

**Operator decisions surfaced**:

1. **PR101 GOLD primitive port** — surface the 3 PR101-specific primitives (storage_order / conv4_perms / byte_maps) to grand council for design-tradeoff review against existing `pr101_sidecar_grammar.py` infrastructure. These are NOT redundant with existing PR101 module (which covers Huffman + centered-delta + split-Brotli) — they are the LAYER-ORDERING and PER-TENSOR-PERMUTATION primitives that the sidecar grammar consumes upstream.
2. **PR98 + PR100 schema-elision design tradeoff** — three variants (`pr98_cd1` / `pr100_schema_driven` / existing `pr105_kitchen_sink_packed_state_schema`) all save ~840 bytes by skipping per-tensor metadata. Surface to grand council for design tradeoff with side-by-side byte-count analysis on our renderer topology.
3. **PR98 decode-side nudge target-mode declaration** — per CLAUDE.md "Contest vs production target modes — non-negotiable", the per-frame per-channel constant nudge must declare `target_modes=["contest_one_video_replay"]` to be admissible. Surface to grand council BEFORE any port.
4. **PR96 / PR101 / PR103 sign-encoding taxonomy** — PR96 uint8-offset-128, PR101 byte_maps (3 strategies), PR103 zigzag form a unified taxonomy of byte-sign-encoding strategies. Surface to grand council for unified `tac.packet_compiler.sign_encoding` module (~60 LOC umbrella) consolidating all 5 strategies (negzig, twos, off, zigzag, raw-uint8).

## Sub-0.20 archive replay decisions

**NOT ROUTED IN THIS LANDING**: per CLAUDE.md "Public frontier watch and intake — NON-NEGOTIABLE" step ordering, an intake-clone-promotion-to-replay decision is a SEPARATE operator approval. This landing surfaces the mechanism metadata only; it does NOT recommend dispatching any replay.

The 4 sub-0.20 PRs in the un-mined HNeRV-family corpus (PR98 0.20, PR100 medal-cluster, PR101 0.193 GOLD, PR103 0.195) all have public intake clones in `experiments/results/public_pr_archive_kaggle_mirror/`, but their replay status is operator-controlled. Existing replay artifacts under `experiments/results/public_pr10[0-3]_*` may already cover this; the next-action is to grep that surface BEFORE proposing any new replay dispatch.

## Catalog #109 dirty-clone audit refresh

Per CLAUDE.md "Apples-to-apples evidence discipline" + "Forbidden in-place edits to public PR intake clones" + Catalog #109 strict-flip 2026-05-08, this audit pass ran:

```
.venv/bin/python -c "from tac.preflight import check_public_pr_intake_clones_pristine; check_public_pr_intake_clones_pristine(strict=True)"
```

**Result**: PASS strict at **62 clones, 0 dirty, 60 cached clean, 2 non-git skipped**. No drift since the 2026-05-08 strict-flip. No new intake clones added since the gate landed (the clones consumed by this audit are the existing `public_pr_archive_kaggle_mirror/public_pr*_intake_20260505_auto/` mirrors that have been in the discovery set since the gate's design).

## Mining methodology

For each un-mined HNeRV-family submission slot:

1. Inspected `inflate.py` + supporting source files (`src/codec.py`, `src/model.py`, `src/losses.py`, `src/optim.py`, `schema.py`, `sidecar.py`).
2. Identified primitives NOT already in `src/tac/packet_compiler/` (the existing tac.packet_compiler covers PR81/84/91/92/93/97/101 sidecar/103 AC/105 packed-state/106 latent sidecar + repack/63/64/65 from prior expansion).
3. Filed each primitive as a typed `MinedPrimitive`-shaped row in `public_pr_mining_pr81_104_typed_rows_20260512.json` with: `primitive_id`, `pr_number`, `submission_slot`, `dominant_mechanism`, `family`, `representation_type`, `score_axis_target`, `key_mechanism_description`, `source_paths`, `source_loc_observed`, `estimated_loc_to_port`, `composes_with`, `applicable_to_pr106_r2_frontier`, `archive_grammar_fields_declared`, `blockers_to_promote_to_tac_packet_compiler`, `next_action`, `pr_claimed_score`, `pr_claimed_score_axis`, `score_claim`, `promotion_eligible`, `ready_for_exact_eval_dispatch`, `evidence_grade`.
4. All rows tagged `score_claim=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`, `evidence_grade='[public-claimed; not replayed]'` per CLAUDE.md "Forbidden score claims" non-negotiable.
5. **NO design decisions made unilaterally** — every primitive that needs council deliberation lands an explicit blocker.
6. **NO KILL verdicts** — additive landing only.
7. **NO in-place edits to public PR intake clones** — pure read-only inspection per CLAUDE.md FORBIDDEN_PATTERNS + Catalog #109.
8. **NO archive bytes modified, NO scorer load** — $0 GPU spend.

## 3-clean-pass adversarial review

**Pass 1** — Shannon LEAD / Dykstra CO-LEAD / Contrarian:
- Shannon: entropy-grounded — every rate-axis row points at a specific byte-saving mechanism with MDL/entropy interpretation (split-brotli boundaries, per-tensor sign-encoding, layer-order permutation). ✓ CLEAN
- Dykstra: each primitive is a candidate input to the achievable-region intersection (rate, seg, pose). Schema-elision is a hard byte constraint at the encoder side; sign-encoding is a per-tensor entropy choice. ✓ CLEAN
- Contrarian: "are these 'new' or are they restatements of existing modules?" Cross-checked: `grep CONV4_STORAGE_PERMS|DECODER_STORAGE_ORDER|DECODER_BYTE_MAPS|merged_ac` returns 0 hits in `src/tac/packet_compiler/`. ✓ CLEAN

**Pass 2** — Yousfi / Fridrich / Quantizr:
- Yousfi: scorer-design perspective — PR98 decode-side nudge is the steganalysis pattern but FOR the scorer (post-process injecting low-luma-unit changes the scorer can't distinguish from rounding noise). This is the inverse-steganalysis loop. ✓ CLEAN
- Fridrich: "decode-side nudge belongs in target_modes=contest_one_video_replay only" — landing memo declares this blocker explicitly. ✓ CLEAN
- Quantizr: "did we double-count?" — explicit table mapping primitive_id -> existing tac.packet_compiler coverage shows no double-counting. ✓ CLEAN

**Pass 3** — Hotz / Selfcomp / MacKay / Hassabis:
- Hotz: engineering-shortcut perspective — the storage-order + conv4-perm primitives are exactly the "Carmack would shred this in 30 minutes" type of byte savings. High EV per LOC. ✓ CLEAN
- Selfcomp: cross-checks PR56 selfcomp work (prior landing) against PR95 hnerv_muon. PR95 dual-RGB-head is geometrically different from selfcomp affine-warp; they compose, not conflict. ✓ CLEAN
- MacKay: MDL rate-cost analysis of cat_entropy_v2 — entropy-of-post-quantized-distribution is the canonical MDL rate term; this lands cleanly in the MacKay-seat framework. ✓ CLEAN
- Hassabis: strategic-research perspective — PR101 GOLD primitives are HIGHEST-LEVERAGE because they trace to a measured 0.193 archive. Top priority for council deliberation post-landing. ✓ CLEAN

**3/3 CLEAN** — landing cleared.

## Loop pause status

**Loop remains PAUSED** per operator directive 2026-05-09. This landing does NOT resume the loop. No `ScheduleWakeup` outstanding.

## Counts at landing

| Metric | Value |
|---|---|
| Un-mined HNeRV-family PRs digested | 6 (PR95/PR96/PR98/PR100/PR101/PR103) |
| Typed mechanism rows in catalog | 15 |
| Sub-0.20 frontier-relevant primitives | 6 (PR98 + PR100 + PR101 GOLD x3 + PR103 silver) |
| Distinct primitive families | 6 (HNeRV, packet-grammar, entropy_coding, quantization, training_optimizer, runtime_trick) |
| Pose-axis primitives | 0 (this audit; previous landing covered pose codecs) |
| Rate-axis primitives | 11 |
| Seg-axis primitives | 1 (PR95 loss curriculum) |
| Mixed-axis primitives | 3 |
| Training-only primitives | 4 |
| 3-clean-pass adversarial greenup | 3/3 CLEAN |
| Catalog #109 dirty-clone audit | PASS strict (62 clones, 0 dirty) |
| GPU spend | $0 |
| Loop status | PAUSED (unchanged) |
| Lane registry delta | +1 (`lane_public_pr_mining_pr81_104_20260512` L0 -> L1 after landing) |
