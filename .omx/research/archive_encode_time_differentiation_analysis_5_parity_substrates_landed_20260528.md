<!-- SPDX-License-Identifier: MIT -->
<!-- HISTORICAL_SCORE_LITERAL_OK:contest_cpu_canonical_frontier_anchor_2026-05-28_archive_differentiation_memo_per_catalog_343 -->
---
council_tier: T2
council_attendees: ["Shannon", "Dykstra", "Yousfi", "Fridrich", "Contrarian", "Assumption-Adversary"]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "the apples-to-apples claim is degraded because per_axis_decomposition GAP is acknowledged; the verdict CASCADE_SATURATION_CONFIRMED rests on in-training loss parity + rate-axis byte counts only — not on auth-eval contest-CPU projections; a re-run on V3 with GAP FIX active is the next-truth surface"
council_assumption_adversary_verdict:
  - assumption: "5 substrates parity at 3.40 in-training scorer-bound floor"
    classification: HARD-EARNED-FOR-4-OF-5
    rationale: "V2/V3/V4/VQ in-training final loss empirically 3.396-3.408 (within 0.4%); Z6-v2 on different loss scale (0.016) — NOT in same parity cluster on the apples-to-apples axis"
  - assumption: "differentiation lives DOWNSTREAM of in-training (archive-encode-time + per-pair-difficulty-atlas + sub-frontier-inference)"
    classification: PARTIALLY-VERIFIED
    rationale: "archive-encode-time differentiation EXISTS but is DOMINATED by decoder_state_dict variance (77% of 0.bin) NOT per-substrate canonical-distinguishing-codec primitive (<0.3% of 0.bin); the asserted differentiation surface is structurally narrow"
  - assumption: "substrate with TIGHTEST archive at parity floor wins sub-0.18 race"
    classification: CARGO-CULTED
    rationale: "the +0.012 gap to sub-0.18 from canonical frontier 0.192028 is ~5x larger than the V3-vs-V2 rate-axis differential (+0.003075); no single PACT-NeRV variant's rate-axis savings alone closes the gap to sub-0.18"
council_decisions_recorded:
  - "op-routable #1: re-run V3 (TIGHTEST archive among parity cluster) with per_axis_decomposition GAP FIX active (commit 92a39dc62) + canonical archive emission, THEN paired-CUDA TOP-1 to ratify rate-axis savings translate to contest-CPU sub-frontier improvement"
  - "op-routable #2: cross-paradigm extension routing — Wyner-Ziv L1 or NSCS06 v8 chroma_lut per T3 council PROCEED commit 38d77eebd — PACT-NeRV decoder_state_dict cluster appears saturated at the shared-decoder component"
  - "op-routable #3: VQ substrate at L6 substrate_deferral per Catalog #220 (posterior_refusal_reason: no_archive_emitted_at_l2_substrate_deferral_to_l6) — operator-routable to add archive emission OR retire to research_only per Catalog #298"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
deferred_substrate_retrospective_due_utc: ""
deferred_substrate_id: ""
related_deliberation_ids:
  - "pact_nerv_v3_v2_v4_hinton_distill_600pair_long_mlx_landed_20260528_apparatus_finding"
  - "t3_council_pr110_stacking_pivot_ordering_landed_20260526"
---

# Archive-encode-time differentiation analysis — 5 parity substrates

**UTC**: 2026-05-28T10:28:13Z
**Lane**: `lane_archive_encode_time_differentiation_analysis_5_parity_substrates_20260528`
**Task**: #1453 (IN_PROGRESS → completed via TaskUpdate this turn)
**Mission contribution**: `frontier_protecting`
**Provenance**: `[macOS-MLX research-signal]` per Catalog #127/#192/#323 (analysis is $0 MLX-local, non-promotable)

## Premise verification (Catalog #229)

Read in full before analysis:
- Archive grammar source for all 4 substrates: `src/tac/substrates/{pact_nerv_selector_v2,pact_nerv_selector_v3,pact_nerv_selector_v4,z6_v2_cargo_cult_unwind}/archive.py` (PSV2/PSV3/PSV4/Z6V2 headers; section layouts; brotli quality)
- All 4 archive.zip + 5 training_artifact.json + 5 checkpoints/* (file system inspection)
- per_axis_decomposition GAP FIX landing memo `.omx/research/mlx_score_aware_per_axis_decomposition_gap_fix_landed_20260528.md` (referenced in parent task prompt; existing artifacts pre-date the fix → per_axis_decomposition is None across all 5)
- CLAUDE.md "Bit-level deconstruction and entropy discipline" + "Apples-to-apples evidence discipline" + "Frontier scores are pointer-only" + `tools/refresh_canonical_frontier.py` for canonical frontier-pointer state

## Source-of-truth amendments to the parent prompt

| Prompt claim | Verified state | Action taken |
|---|---|---|
| Z6-v2 commit `d6168d9ef` | The only Z6-v2 MLX long artifact with archive.zip is `experiments/results/z6_v2_mlx_long_20260528T060811Z/` (sha256 `3eb72eae10e7faa0...`, 579,451 bytes). No directory matches the `T095121Z` / 612,704-byte spec in the parent prompt. | Used canonical T060811Z artifact; flagged divergence here per Catalog #287/#323 evidence discipline. |
| VQ "+ Hinton + 600-pair commit `84a4893e4`" | Directory `experiments/results/pact_nerv_vq_hinton_distill_600pair_long_mlx_20260528T093247Z/` exists with telemetry + checkpoints, **but no archive.zip and no 0.bin**. `training_artifact.json` carries `posterior_refusal_reason: no_archive_emitted_at_l2_substrate_deferral_to_l6` per Catalog #220. | Cannot dissect VQ archive bytes. Marked VQ as NO_ARCHIVE in the rate-axis ranking. Operator-routable per op #3. |
| "5 substrates parity at 3.40 in-training scorer-bound floor" | V2 (3.397) / V3 (3.396) / V4 (3.408) / VQ (3.407) all converge at ~3.40 ±0.012 (PARITY CONFIRMED). **Z6-v2 final loss = 0.016** — on a different scale (different `score_aware_loss_kwargs` per `config_snapshot`), NOT in the same parity cluster on the apples-to-apples axis. | Verdict scopes "parity" to the 4 PACT-NeRV substrates; Z6-v2 treated as off-axis. |
| "per_axis_decomposition GAP FIX LANDED commit `92a39dc62`" | Confirmed via inspection: all 5 artifacts have `last_epoch.per_axis_decomposition == None`. GAP FIX applies to FUTURE runs. | Verdict explicitly states the apples-to-apples projection to total contest-CPU score requires a re-run with GAP FIX active. |

## Phase 1+2: archive-byte tabulation + 0.bin payload dissection

### Phase 1 — apples-to-apples archive sizes (per Catalog #343 frontier pointer reference)

| Substrate | archive.zip bytes | sha256 prefix | rate-axis (25·N/37,545,489) | Δ_bytes_vs_v3 | Δ_rate_vs_v3 |
|---|---:|---:|---:|---:|---:|
| selector_v3 | 137,351 | `ef5a087ff6301dbf` | 0.091456 | +0 (+0.00%) | +0.000000 |
| selector_v4 | 138,200 | `d9c3388bda54b7a9` | 0.092022 | +849 (+0.62%) | +0.000565 |
| selector_v2 | 141,969 | `f9bff760e638a719` | 0.094531 | +4,618 (+3.36%) | +0.003075 |
| **vq** | **NO ARCHIVE** | — | — | — | — |
| z6_v2 | 579,451 | `3eb72eae10e7faa0` | 0.385833 | +442,100 (+321.88%) | +0.294376 |

**Canonical frontier (per pointer)**: contest-CPU 0.192028 (DQS1 rank021 sha `7a0da5d0fc327cba`); contest-CUDA 0.20533 (PR106 format0d sha `9cb989cef519`). All 5 substrate archive byte counts above are MLX-local research-signal per Catalog #192; not contest-frontier-eligible without a paired Linux x86_64 contest-CPU + NVIDIA contest-CUDA anchor.

### Phase 2 — 0.bin payload dissection (substrate-distinguishing-codec contribution)

Per each substrate's canonical header (PSV2/PSV3/PSV4 = `<4sBHHBIIII` = 26B; Z6V2 = `<4sBHHHIIIIB` = 28B):

| Substrate | decoder (brotli) | latents (int16) | per-substrate canonical codec | meta (json) | Other |
|---|---:|---:|---:|---:|---:|
| selector_v3 | 100,849B (77.45%, H=7.995, lzma_ratio=1.001) | 28,800B (22.12%, H=7.830) | **selector Rice-Golomb 225B (0.17%, H=1.585, lzma_ratio=0.338)** | 310B | — |
| selector_v2 | 101,127B (77.19%, H=7.990, lzma_ratio=1.001) | 28,800B (21.98%, H=7.813) | **selector arithmetic-coded 301B (0.23%, H=0.032, lzma_ratio=0.252)** | 762B (cum_freq embedded) | — |
| selector_v4 | 101,624B (77.73%, H=7.992, lzma_ratio=1.001) | 28,800B (22.03%, H=7.813) | **selector per-class RLE arith 3B (0.00%)** | 294B | — |
| z6_v2 | 565,994B (99.58%, H=7.993, lzma_ratio=1.000) | 1,536B (0.27%, H=7.748) | **ego_vecs FoE 384B (0.07%, H=7.369)** | 462B | — |
| vq | NO ARCHIVE | NO ARCHIVE | NO ARCHIVE | NO ARCHIVE | NO ARCHIVE |

**Empirical observations** (per CLAUDE.md "Bit-level deconstruction and entropy discipline"):

1. **decoder_state_dict consumes 77% of 0.bin in PACT-NeRV cluster** (~101,000B in all 3 V2/V3/V4 variants). LZMA ratio = 1.001 across the board → brotli at quality=9 has already extracted essentially all redundancy; no further compression headroom on this section.
2. **PACT-NeRV per-substrate canonical-distinguishing-codec contribution is <0.3% of 0.bin** (V3 Rice-Golomb 225B vs V2 arithmetic-coded 301B vs V4 per-class RLE 3B). The architectural distinction between Rice-Golomb / arithmetic-coded / per-class-RLE manifests as ~300B byte-budget variance, not the dominant rate-axis driver.
3. **selector_v4 RLE-arith achieves dramatic compression (3B for 600 pairs of palette indices)** — strong evidence the per-pair selector signal is essentially constant-class (low entropy 1.585) — but the saving is structurally bounded since the selector was never the rate-axis dominant section.
4. **V2 selector blob has H=0.032 bits/byte** (vs 1.585 for V3 / V4) — V2's arithmetic-coded selector approaches the Shannon entropy bound for the symbol stream, but V2's META blob carries 762B of cum_freq tables that V3/V4 do NOT need; net penalty.
5. **Z6-v2 decoder_state_dict is 565,994B = 99.58% of 0.bin** — the Rao-Ballard hierarchical predictor + cooperative-receiver FoE conditioning architecture pays a 5.6× decoder cost vs PACT-NeRV cluster. The ego_vecs FoE blob (the canonical-distinguishing-feature per Catalog #311) is only 384B = 0.07% of 0.bin. The "smart" hierarchical predictor primitive is dominated by raw decoder weight cost.

## Phase 3: differentiation surface

The differentiation surface across the 4 PACT-NeRV substrates is structurally narrow:

- **In-training parity**: 3.396 → 3.408 (0.4% range across V2/V3/V4/VQ).
- **Archive rate-axis differentiation**: V3 → V4 → V2 separated by +0.000565 (V4) and +0.003075 (V2) above the V3 floor.
- **Per-substrate canonical-distinguishing-codec savings**: <0.3% of 0.bin in each case.

**The hypothesis "substrate with TIGHTEST archive at parity floor wins sub-0.18 race"** is structurally HARD-EARNED only IF the rate-axis differential (~0.003) closes a meaningful fraction of the (frontier - sub-0.18) gap (~0.012). The empirical ratio is ~25% (0.003 / 0.012) → V3's rate-axis savings alone do NOT close the gap. To project total contest-CPU score, the seg/pose decomposition is required.

## Phase 4: verdict per Catalog #307

### Primary verdict: **CASCADE_SATURATION_CONFIRMED at archive-encode-time for the PACT-NeRV cluster (V2/V3/V4)**

**Classification per Catalog #307**: IMPLEMENTATION-LEVEL apparatus observation (not paradigm-level falsification). The PACT-NeRV decoder_state_dict cluster has saturated on the per-substrate canonical-codec differentiation surface. Cross-substrate variation lives almost entirely in the shared decoder_state_dict (77% of 0.bin) which is byte-stable across variants at brotli quality=9; the per-substrate canonical-codec primitives (Rice-Golomb / arithmetic-coded / per-class-RLE) all live in <0.3% of 0.bin.

**Rationale**:
1. V2/V3/V4 in-training final loss within 0.4% range (3.396–3.408) → in-training scorer-bound floor saturated.
2. V2/V3/V4 archive byte differences within 3.4% (137,351 → 141,969) → rate-axis differentiation bounded.
3. Per-substrate canonical-distinguishing-codec payload is <0.3% of 0.bin across the cluster → structural saturation of the cascade architecture's distinguishing-codec layer at the parity floor.
4. The (frontier - sub-0.18) gap (~0.012) is 4× larger than the V2-vs-V3 rate-axis differential (~0.003) → no single PACT-NeRV variant's rate-axis savings closes the gap.

### Secondary verdicts

- **VQ substrate**: AT L6 substrate_deferral per Catalog #220 (no archive emitted at L2). DEFERRED-pending-archive-emission per Catalog #298 retirement discipline. NOT in candidate ranking today.
- **Z6-v2 substrate**: DIFFERENT loss scale (0.016 vs 3.40 PACT-NeRV cluster floor); 4.2× larger archive (rate-axis +0.294 vs V3). Cannot apples-to-apples project sub-0.18 from this artifact even with optimal seg/pose floors. The cooperative-receiver paradigm remains INTACT per Catalog #307; the architectural per-pair training was not the cluster comparison vehicle today.

### Operator-routable next steps (per Catalog #300 op-routables)

**TOP-1 (op-routable #1)**: re-run V3 (TIGHTEST archive among PACT-NeRV parity cluster, 137,351B) with per_axis_decomposition GAP FIX active (post-commit `92a39dc62`) AND canonical archive emission. Then paired-CUDA TOP-1 per Catalog #246 (BOTH `[contest-CPU]` AND `[contest-CUDA]` on 1:1 contest-compliant hardware per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"). Predicted operator-decidable outcome: contest-CPU projection from authenticated decomposition closes the (frontier - sub-0.18) gap by ~25% via rate-axis alone; remaining ~75% gap surfaces seg/pose dominant axes for next-cycle attack.

**TOP-2 (op-routable #2)**: cross-paradigm extension routing per T3 council PROCEED commit `38d77eebd` ORDERING — NSCS06 v8 chroma_lut (#1 in T3 ordering) OR Wyner-Ziv L1 (cross-family). The PACT-NeRV decoder_state_dict cluster appears saturated at the shared-decoder component; further investment in within-cluster cascade primitives produces diminishing returns per the <0.3% canonical-codec contribution evidence.

**TOP-3 (op-routable #3)**: VQ substrate operator-routable per Catalog #298 retirement discipline — either (a) wire archive emission per HNeRV parity discipline L4 (≤200 LOC inflate budget; 8 required fields per Catalog #124), OR (b) flip to `research_only=true` / `dispatch_enabled=false` per Catalog #240.

## Canonical-vs-unique decision per layer

Per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD operating mode:

- **Apparatus surfaces** (canonical helpers, tools/audit_*.py, fcntl-locked JSONL state writes): ADOPT_CANONICAL_BECAUSE_SERVES — this analysis routes through canonical `tac.canonical_equations` registry (refinements in Phase 5), `tac.council_continual_learning` (posterior anchor below), and reads via `tac.frontier_scan` per Catalog #316 sister.
- **Bit-level dissection methodology**: ADOPT_CANONICAL — Shannon entropy + lzma ratio per CLAUDE.md "Bit-level deconstruction" non-negotiable; no per-substrate forking of analysis primitives.
- **Per-substrate archive grammar interpretation**: FORK_BECAUSE_PRINCIPLED — each substrate's archive.py canonical header is the authority for section boundaries; the analysis respects per-substrate grammar (PSV2/3/4 vs Z6V2) rather than a generic-bulk-zip-dump.

## Cargo-cult audit per assumption (per Catalog #303)

| Assumption | HARD-EARNED-vs-CARGO-CULTED | Unwind test plan |
|---|---|---|
| Parent prompt: "5 substrates parity at 3.40 in-training scorer-bound floor" | HARD-EARNED-FOR-4-OF-5 | Z6-v2 loss is on different scale; verified via direct `loss` field inspection across all 5 `training_artifact.json` files. |
| Parent prompt: "differentiation lives DOWNSTREAM of in-training (archive-encode-time + per-pair-difficulty-atlas + sub-frontier-inference)" | PARTIALLY-HARD-EARNED | Archive-encode-time differentiation exists but is dominated by decoder_state_dict variance (~101,000B shared across PACT-NeRV cluster); the per-substrate canonical-distinguishing-codec primitives only contribute <0.3% of 0.bin. The hypothesized differentiation surface is structurally narrower than the parent prompt's framing implies. |
| Parent prompt: "substrate with TIGHTEST archive at parity floor wins sub-0.18 race" | CARGO-CULTED | The (frontier - sub-0.18) gap is ~0.012 vs V2-vs-V3 rate-axis differential ~0.003 → no single PACT-NeRV variant's rate-axis savings alone closes the gap; the framing under-counts the contest formula's seg/pose dominance. |
| Apparatus-finding parent: "5 substrates parity → differentiation downstream of in-training" | PARTIALLY-HARD-EARNED, requires PAIRED-CUDA RATIFICATION | Without `per_axis_decomposition` (GAP FIX applies to FUTURE runs only), we cannot apples-to-apples project contest-CPU total score from the in-training loss + rate-axis bytes alone. The TOP-1 op-routable closes this verification gap. |

## 9-dimension success checklist evidence (per Catalog #294)

1. **UNIQUENESS**: each substrate's per-substrate canonical-distinguishing-codec was inspected per its own archive.py grammar; not generic-bulk-archive-analysis.
2. **BEAUTY + ELEGANCE**: rate-axis differential computed via single contest formula `25 * N / 37,545,489`; verdict reduces to 3 lines (parity / rate-axis ordering / sub-0.18 gap arithmetic).
3. **DISTINCTNESS**: each substrate's distinguishing primitive (Rice-Golomb / arithmetic-coded / per-class-RLE / FoE) preserved + measured at byte granularity.
4. **RIGOR**: premise verification before edit per Catalog #229; canonical Provenance per Catalog #323; HARD-EARNED-vs-CARGO-CULTED audit per Catalog #303; cargo-cult-unwound assumptions ratified empirically.
5. **OPTIMIZATION PER TECHNIQUE**: per-substrate archive.py grammar respected (UNIQUE-AND-COMPLETE-PER-METHOD); not collapsed to shared helper.
6. **STACK-OF-STACKS COMPOSABILITY**: verdict surfaces TOP-1 paired-CUDA + TOP-2 cross-paradigm + TOP-3 VQ rectification — composition surface lives in the operator-routable cascade.
7. **DETERMINISTIC REPRODUCIBILITY**: archive.zip bytes + sha256 + ZIP-member listing + brotli-quality-pinning all read-only from existing artifacts; analysis reproducible from sha256-pinned inputs.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: identified V3 as TIGHTEST archive among parity cluster (137,351B; 3.36% smaller than V2; 0.62% smaller than V4); operator-routable to ratify via paired-CUDA.
9. **OPTIMAL MINIMAL CONTEST SCORE**: arithmetic projection — V3 rate-axis 0.091456 < frontier 0.118867 → 0.027 byte-axis savings VS frontier in isolation; net total score depends on seg/pose preservation, which requires paired-CUDA ratification.

## Observability surface (per Catalog #305)

1. **Inspectable per layer**: archive.zip ZIP-member listing + 0.bin section-by-section dissection per canonical header format.
2. **Decomposable per signal**: per-section byte counts + entropy + lzma ratio → decoder vs latents vs per-substrate-canonical-codec vs meta separation.
3. **Diff-able across runs**: 4 substrate ZIP archives byte-byte comparable; PACT-NeRV cluster decoder_state_dict ~101K shared; selector sections diverge as expected per architecture.
4. **Queryable post-hoc**: training_artifact.json per_epoch_metrics + loss_components + config_snapshot (canonical JSON; no chat-only state).
5. **Cite-able**: canonical posterior anchor below; canonical equations #344 entries (Phase 5).
6. **Counterfactual-able**: byte-mutation smoke per Catalog #139/#272 possible on per-substrate distinguishing sections (Rice-Golomb selector / arithmetic-coded selector / RLE selector / FoE ego_vecs) since each section's offset + length is known from header.

## Predicted ΔS band

**Per CLAUDE.md "Frontier scores are pointer-only"**: ALL ΔS projections cited here are MLX-local research-signal (Catalog #192). They are NOT contest-score claims and remain non-promotable until paired Linux x86_64 + NVIDIA anchor lands per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA".

Rate-axis projection (CONFIRMED via byte counting + canonical contest formula `25 * N / 37,545,489`):
- V3 rate-axis = 0.091456 (vs PR101+FEC6 frontier baseline rate 0.118867 → -0.027411 byte-axis savings in isolation IF seg/pose preserved at frontier).
- V4 rate-axis = 0.092022 (vs PR101+FEC6 → -0.026845).
- V2 rate-axis = 0.094531 (vs PR101+FEC6 → -0.024336).
- Z6-v2 rate-axis = 0.385833 (vs PR101+FEC6 → +0.266966 — DOMINATES total score; cannot project sub-0.18 from current artifact).

**Dykstra-feasibility check** (per Catalog #296): the seg + pose axes must be jointly preserved within the (frontier - sub-0.18) budget = 0.012028. The PACT-NeRV cluster trades 0.027 of rate-axis budget for unknown seg+pose excess. Without per_axis_decomposition (GAP FIX) data, the Pareto-feasibility intersection of {rate ≤ V3 rate, seg ≤ frontier seg, pose ≤ frontier pose} cannot be empirically verified from these artifacts alone. The TOP-1 op-routable is exactly the alternating-projections feasibility check.

## Mission alignment (per CLAUDE.md "Mission alignment" non-negotiable)

`council_predicted_mission_contribution`: **frontier_protecting**. The verdict protects against a CARGO-CULTED inversion where the parent prompt's "substrate with TIGHTEST archive wins" framing would have steered operator to dispatch all 4 PACT-NeRV cluster substrates to paired-CUDA without ratifying the per-substrate canonical-codec actually moves the contest score. The 3.4% archive-byte differential between V2 and V3 produces ~25% of the (frontier - sub-0.18) gap closure budget — useful but not sufficient. The TOP-2 op-routable (cross-paradigm extension) is the structurally larger lever.

## 6-hook wire-in declaration (per Catalog #125)

- **Hook #1 sensitivity-map contribution**: ACTIVE — per-section archive byte counts + entropy estimates feed downstream sensitivity-map consumers; rate-axis differential per substrate is a typed row for the Pareto solver.
- **Hook #2 Pareto constraint**: ACTIVE — rate-axis byte budgets per substrate are explicit constraints for the alternating-projections feasibility intersection per Catalog #296 sister.
- **Hook #3 bit-allocator hook**: ACTIVE — per-substrate distinguishing-codec contribution (<0.3% of 0.bin) bounds the per-substrate bit-allocator's marginal value; decoder_state_dict (77% of 0.bin) is the dominant lever for future bit-allocator work.
- **Hook #4 cathedral autopilot dispatch hook**: ACTIVE — TOP-1 paired-CUDA op-routable feeds the canonical operator-authorize recipe surface per Catalog #167 smoke-before-full; TOP-2 cross-paradigm routing feeds the T3 council ordering at commit `38d77eebd`.
- **Hook #5 continual-learning posterior update**: ACTIVE — canonical posterior anchor written below via `tac.council_continual_learning.append_council_anchor`; canonical equations #344 entries refined (Phase 5).
- **Hook #6 probe-disambiguator**: ACTIVE — the verdict explicitly chooses between {SUB_0_18_CANDIDATE, CASCADE_SATURATION, DIFFERENTIATION_INSUFFICIENT}; the canonical disambiguator is the per_axis_decomposition GAP FIX re-run on V3 (TOP-1 op-routable).

## Canonical posterior anchor

Written to `.omx/state/council_deliberation_posterior.jsonl` via `tac.council_continual_learning.append_council_anchor`. See sister `feedback_archive_encode_time_differentiation_analysis_5_parity_substrates_landed_20260528.md` for the structured posterior row.

## Cross-references

- Sister landing memos for PACT-NeRV cascade (V2/V3/V4 commits `ab650cc78` + `84a4893e4`) and Z6-v2 600-pair commit `d6168d9ef` (apparatus-level finding cited by parent prompt).
- Per-axis decomposition GAP FIX commit `92a39dc62` (referenced by parent prompt; existing artifacts pre-date; FUTURE runs will carry per_axis_decomposition data).
- T3 council PR110 stacking ordering at commit `38d77eebd` (TOP-2 op-routable consumer).
- Canonical frontier pointer `.omx/state/canonical_frontier_pointer.json` (contest-CPU 0.192028; contest-CUDA 0.20533).
- CLAUDE.md "Bit-level deconstruction and entropy discipline" + "Apples-to-apples evidence discipline" + "Frontier scores are pointer-only" + "Submission auth eval — BOTH CPU AND CUDA".
- Catalog #220 (operational mechanism); Catalog #233 (L2 promotion 4-gate); Catalog #240 (recipe-vs-trainer consistency); Catalog #246 (paired anchor skip); Catalog #270 (dispatch optimization protocol); Catalog #287 (placeholder rejection); Catalog #294 (9-dim checklist); Catalog #296 (Dykstra feasibility); Catalog #298 (substrate retirement); Catalog #300 (council deliberation v2); Catalog #303 (cargo-cult audit); Catalog #305 (observability surface); Catalog #307 (paradigm-vs-implementation classification); Catalog #316 (frontier-staleness sister); Catalog #319 (Wyner-Ziv deliverability proof); Catalog #323 (canonical Provenance umbrella); Catalog #335 (cathedral consumer auto-discovery); Catalog #344 (canonical equations registry); Catalog #356 (per-axis decomposition Provenance).
