# Codex routing directive: A-2 N-7 CPU-axis Tier-1 standard compressor sweep — MANDATORY EMPIRICAL ANCHOR
# Date: 2026-05-18
# Authority: ADVERSARIAL rate-attack paradigm challenger memo `.omx/research/adversarial_rate_attack_paradigm_challenger_20260518.md` commit 4c6e46bfa
# Per ADVERSARIAL quartet (Tao+Carmack+Hotz+Boyd) recommendation: A-2 N-7 is the MANDATORY $0 empirical anchor that tests the SHARED-CARGO-CULTED ASSUMPTION underlying ALL 43+ rate-attack vectors
# Per CLAUDE.md "Mission alignment" Consequence 4 (frontier-breaking dominates rigor budget) + Catalog #229 premise-verification-before-edit
# Per operator standing directive "must pursue and confirm all closure operations complete and correct" applied at the WAVE level

## CANONICAL POINTERS (read FIRST)

1. `/Users/adpena/Projects/pact/CLAUDE.md` (FULL; especially "Apples-to-apples evidence discipline" + "Bit-level deconstruction and entropy discipline" + Catalog #287 evidence tags + Catalog #313 probe outcomes ledger + Catalog #316 frontier scan)
2. `/Users/adpena/Projects/pact/AGENTS.md`
3. `.omx/research/adversarial_rate_attack_paradigm_challenger_20260518.md` (THE AUTHORITY; section A-2 N-7 specifies the empirical anchor precisely)
4. `.omx/research/rate_attack_research_context_supplement_per_axis_hardware_plus_dual_device_master_gradient_20260518.md` (Part 3 inventory of available signals)
5. `reports/latest.md` (current frontier per Catalog #316)
6. Current frontier archives (per Catalog #316 frontier scan):
   - `0.19205 [contest-CPU]`: `pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean` / archive `6bae0201...`
   - `0.20533 [contest-CUDA]`: `pr106_format0d_latent_score_table` / archive `9cb989cef519...`
   - Plus: A1 archive `87ec7ca5...` + PR101_lc_v2 anchor `f174192aeadf...` + DP1 archives + fec6 stacking outputs

## STRATEGIC MOTIVATION (why this anchor is MANDATORY before further rate-attack research)

The 43+ rate-attack vectors enumerated in `.omx/research/rate_attack_research_context_supplement_per_axis_hardware_plus_dual_device_master_gradient_20260518.md` ALL share ONE foundational assumption:

> **"Current archive packets have remaining rate-extractable structure that standard compressors miss"**

The PRIMARY rate-attack subagent (`a703d2b74784d4f00`) is currently researching these 43 vectors under the assumption that this is HARD-EARNED. The ADVERSARIAL quartet (Tao+Carmack+Hotz+Boyd) classified it as **CARGO-CULTED-PENDING-EMPIRICAL** with 60% predicted SATURATION_HARD_EARNED / 40% SATURATION_REFUTED.

The A-2 N-7 sweep EMPIRICALLY tests this assumption at $0 cost / 4-6 hours wall-clock. If saturation is confirmed:
- Most of 43 vectors are CARGO-CULTED (Boyd's Pareto-feasibility critique is dispositive)
- 5-10 days of design work on them would be misallocated rigor budget
- Per CLAUDE.md "Forbidden premature KILL" + "Mission alignment" Consequence 4: pivot to ALTERNATIVE paradigm (e.g., substrate-class-shift, score-axis-only optimization, hardware-acceleration)

If saturation is refuted:
- Rate-attack vectors are HARD-EARNED
- Codex executes the TOP-3 routing directives from PRIMARY's TOP-5 (when PRIMARY lands)
- Frontier displacement predicted [0.165, 0.185] per cross-stack synthesis

Either outcome is HIGHEST-EV information per Catalog #229 premise-verification-before-edit. **Run A-2 N-7 BEFORE committing to the 43 vectors.**

## WHAT CODEX BUILDS

### Phase 1: Sweep tool (`tools/probe_a2_n7_standard_compressor_sweep.py` ~400-600 LOC)

Apply ALL standard compressors to each current frontier archive's payload bytes:

```python
# Compressors to sweep (canonical lossless on contest-CPU axis):
COMPRESSORS = [
    ("zstd",      [1, 3, 9, 15, 22]),           # levels low to ultra
    ("brotli",    [4, 6, 9, 11]),                # levels mid to max
    ("lzma",      [1, 6, 9]),                    # levels low to max
    ("xz",        [1, 6, 9]),                    # levels low to max
    ("zlib",      [1, 6, 9]),                    # baseline reference
    ("zstd_long", [1, 9, 22]),                   # --long mode for cross-block reference
    ("bzip2",     [1, 6, 9]),                    # block-sort canonical
    ("lzfse",     [None]),                        # Apple's canonical for fast decode
]

# Per-archive per-compressor measurement:
def sweep_archive(archive_path: Path) -> SweepReport:
    """Apply each compressor at each level to archive payload bytes.
    Measure: compressed_size_bytes / original_size_bytes / ratio / wall_clock.
    Report: SATURATION_HARD_EARNED if best ratio >= 0.95 (≤5% rate-extractable)
            SATURATION_REFUTED  if best ratio <= 0.85 (≥15% rate-extractable)
            SATURATION_PENDING  if 0.85 < best ratio < 0.95 (5-15% gray zone)
    """
```

Apply to current frontier archives:
- PR101 GOLD anchor `f174192aeadf...`
- A1 `87ec7ca5...`
- PR101 fec6 fixed_huffman_k16_clean `6bae0201...`
- PR106 format0d `9cb989cef519...`
- DP1 codebook + latent blobs

### Phase 2: Per-block / per-section sweep (extends to byte-region granularity)

Per archive, split payload bytes into per-tensor-region sections (per Catalog #319 byte-level deconstruction). For each section, sweep compressors. Identify regions with HIGHEST rate-extractable structure vs regions already at entropy boundary.

Output: per-section table showing which sections (if any) have remaining rate structure. This is INFORMATIVE even if global SATURATION is confirmed — surfaces local rate-attack opportunities.

### Phase 3: Comma2k19-derived dictionary sweep (per quartet member Carmack's Vector C-2)

Build canonical Brotli compression dictionary from Comma2k19 dashcam data (via `tac.substrates.pretrained_driving_prior.local_chunk_cache.Comma2k19LocalCache` per Catalog #213). Re-sweep with Brotli + dictionary. Test whether OOD dictionary improves compression on contest archives.

### Phase 4: Report + verdict

Write `experiments/results/a2_n7_compressor_sweep_<utc>/report.json` with:
- Per-archive per-compressor compressed_size + ratio + wall-clock
- Per-archive verdict: SATURATION_HARD_EARNED / SATURATION_REFUTED / SATURATION_PENDING
- Per-section verdict (Phase 2)
- Comma2k19 dictionary improvement signal (Phase 3)
- Aggregate verdict across all current frontier archives

Append probe outcome to `.omx/state/probe_outcomes.jsonl` per Catalog #313:
```python
register_probe_outcome(
    probe_id="a2_n7_cpu_axis_tier1_standard_compressor_sweep_20260518",
    verdict=<SATURATION_HARD_EARNED | SATURATION_REFUTED | SATURATION_PENDING>,
    status="adjudicated",
    blocks_recipes=[<rate-attack research directives if SATURATION_HARD_EARNED>],
    rationale=<aggregate report path>,
    expires_at_utc=<+30d>,
    agent="codex",
)
```

### Phase 5: Routing-decision tree (consume verdict)

If SATURATION_HARD_EARNED:
- Append canonical_task_status row marking PRIMARY rate-attack research as DEFERRED-pending-paradigm-pivot
- Route operator decision: pivot to substrate-class-shift (per cargo-cult resurrection top-3 + Z6/Z7/Z8 cascade) OR continue rate-attack at lower priority + smaller scope

If SATURATION_REFUTED:
- Re-route PRIMARY rate-attack TOP-5 directives at full priority
- Operator can authorize Modal/Lightning dispatch for empirical TOP-3 validation per Catalog #270 protocol

If SATURATION_PENDING:
- Schedule extended sweep (Phase 2 per-section deeper) + dictionary sweep (Phase 3 with multiple OOD sources)

## DISCIPLINE

- Catalog #229 premise verification BEFORE compressor sweep: verify each frontier archive's actual filesystem location matches the reports/latest.md claim
- Catalog #287 evidence tags on every numeric output (compression_ratio + wall_clock + verdict)
- Catalog #206 checkpoint discipline every ~10 tool uses
- Catalog #117/#157/#174 commit serializer with POST-EDIT sha for the probe tool + report
- Catalog #186 catalog # claim transactional (if new STRICT gate needed; likely NOT for this probe — it's diagnostic not protective)
- Catalog #313 register probe outcome via canonical helper `tac.probe_outcomes_ledger.register_probe_outcome`
- Catalog #245 4-layer pattern N/A (this is a single-shot probe; not a recurring channel)
- Catalog #314 absorption avoidance: scope is `tools/probe_a2_n7_standard_compressor_sweep.py` + `experiments/results/a2_n7_compressor_sweep_*` + `.omx/state/probe_outcomes.jsonl` registration
- Catalog #270 dispatch protocol N/A (this is CPU-only probe; no GPU dispatch)
- Catalog #287 evidence-tag discipline: aggregate verdict tagged `[empirical:experiments/results/a2_n7_compressor_sweep_<utc>/report.json]`

## EXIT CRITERIA

- [ ] `tools/probe_a2_n7_standard_compressor_sweep.py` exists; runnable via CLI
- [ ] Sweep applied to ALL 5 current frontier archives (PR101 GOLD + A1 + PR101 fec6 + PR106 format0d + DP1)
- [ ] Per-archive per-compressor compressed_size + ratio recorded
- [ ] Per-archive verdict: SATURATION_HARD_EARNED / SATURATION_REFUTED / SATURATION_PENDING
- [ ] Per-section verdict (Phase 2) for non-globally-saturated archives
- [ ] Comma2k19 dictionary sweep (Phase 3) executed + reported
- [ ] Aggregate verdict appended to `.omx/state/probe_outcomes.jsonl` per Catalog #313
- [ ] `experiments/results/a2_n7_compressor_sweep_<utc>/report.json` lands
- [ ] codex_persistent_session_state row appended with `directive_executed=a2_n7_cpu_axis_tier1_standard_compressor_sweep_empirical_anchor`
- [ ] canonical_task_status row updated 'completed' with commit_shas + verdict
- [ ] Memory entry `feedback_a2_n7_compressor_sweep_landed_20260518.md` documenting verdict + operator-routable next action

## OPERATOR-FACING NOTE

This empirical anchor SHOULD COMPLETE WITHIN 4-6 HOURS WALL-CLOCK at $0 cost. Result determines whether the in-flight rate-attack research subagent (PRIMARY a703d2b74784d4f00) findings are HARD-EARNED or CARGO-CULTED. Per ADVERSARIAL quartet 60% saturation prediction, this is HIGH-EXPECTED-INFORMATION-GAIN.

After Codex executes + verdict lands, operator decides:
- SATURATION_HARD_EARNED → pivot rate-attack research to lower-priority + smaller-scope; recommit rigor budget to substrate-class-shift candidates (Z6/Z7/Z8 cascade per pose-axis council)
- SATURATION_REFUTED → green-light PRIMARY's TOP-3 routing directives for Modal/Lightning dispatch
- SATURATION_PENDING → request operator decision on extended sweep budget

## SISTER COORDINATION

In-flight at directive-write time:
- PRIMARY rate-attack research subagent (`a703d2b74784d4f00`) — may still be doing 43-vector research; the A-2 N-7 verdict will inform its TOP-5 ranking when it lands
- Codex session `019de465` — continues autonomously; will pick up THIS directive per /goal LOOP

Sister memos (do NOT duplicate; cite cross-references):
- `.omx/research/adversarial_rate_attack_paradigm_challenger_20260518.md` (THE AUTHORITY for A-2 N-7 specification)
- `.omx/research/rate_attack_research_context_supplement_per_axis_hardware_plus_dual_device_master_gradient_20260518.md` (Part 3 inventory)

— Main-Claude 2026-05-18 (ADVERSARIAL-authorized routing per quartet mandatory empirical anchor + operator standing "deeply and broadly and passionately researching" directive applied at premise-verification level per Catalog #229)
