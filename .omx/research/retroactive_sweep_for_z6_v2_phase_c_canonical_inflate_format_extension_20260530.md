# Retroactive sweep for z6_v2 Phase C canonical inflate format extension 2026-05-30

Per Catalog #348 `check_new_gate_landing_includes_retroactive_sweep_evidence`: this landing introduces NO NEW STRICT preflight gate. The required-by-Catalog-#348 4-field contract therefore applies in the conservative orientation: bug-class symptom signature + pre-fix window + historical KILL/DEFER/FALSIFY search + per-finding RE-EVAL-priority.

## 1. Bug-class symptom signature

**Symptom class**: substrate inflate.py emits PNG output rather than canonical contest `.raw` (uint8 concatenated H×W×C bytes), with `output_dir/{frame_idx}.png` per-frame files instead of single `output_dir/{base}.raw` file per video. Sister symptoms include archive size exceeding rate-axis budget by significant margin (rate term 0.386 vs frontier 0.192 = 2x) due to fp16+pickle+brotli q=9 vs canonical L21+L29+L32 INT8+fp16scales+brotli q=11 binding, and frame count truncated to MLX-LOCAL smoke pair count rather than canonical 1200 contest frames.

## 2. Pre-fix window

Pre-fix window: any substrate inflate.py landed BEFORE 2026-05-30T23:00:00Z that satisfies the symptom signature. This Phase C extension only modified z6_v2; sister substrates retain their respective inflate paths.

## 3. Historical KILL/DEFER/FALSIFY search

Search for substrate inflate.py instances that may have produced KILL/DEFER/FALSIFY verdicts due to inflate format gap (NOT substrate-paradigm gap):

| Substrate | Status | Verdict | Reactivation criterion satisfied? |
|---|---|---|---|
| z6_v2_cargo_cult_unwind | DEFERRED 2026-05-30 (Phase C BLOCKED) | DEFER pending canonical inflate extension | **SATISFIED THIS LANDING** — Phase C reactivation criterion #1 closes |
| z5_predictive_coding_world_model | LANDED with canonical .raw inflate | PROCEED | N/A (already canonical) |
| z8_hierarchical_predictive_coding | LANDED with canonical .raw inflate | PROCEED | N/A (already canonical) |
| Other PR101/PR106/HNeRV-family substrates | LANDED with canonical .raw inflate | PROCEED | N/A (already canonical) |

**Per CLAUDE.md "Forbidden premature KILL without research exhaustion"**: no historical KILL verdict on z6_v2 paradigm; the DEFER verdict on Phase C was an IMPLEMENTATION-LEVEL gap per Catalog #307. This landing closes that gap structurally.

## 4. Per-finding RE-EVAL-priority assignment

| Finding | RE-EVAL priority | Reason |
|---|---|---|
| z6_v2 Phase C BLOCKED → LANDED (THIS landing) | HIGH (this landing) | Phase D operator-attended paired-CUDA Modal T4 verification anchor per Catalog #246 within $0.30-1.00 envelope — once trainer wires v2 archive format + substrate shrinkage closes the 34KB gap to 290KB frontier |
| Sister substrates that emit canonical .raw (z5, z8, PR101-family) | N/A | Already canonical; no re-eval needed |
| Other Z6-v2 PROCEED_WITH_REVISIONS council verdicts | LOW | Per Catalog #315: any PROCEED_WITH_REVISIONS council anchor on Z6-v2 sub-surfaces remains dormant pending iteration; Phase C extension closes ONE specific implementation gap, does NOT reset wider council deliberation surface |

## Cross-references

* Landing memo: `feedback_z6_v2_phase_c_canonical_inflate_format_extension_landed_20260530.md`
* Predecessor: `feedback_z6_v2_canonical_29650ep_mlx_local_full_run_landed_20260530.md` Phase C BLOCKED
* Probe outcome: `z6_v2_phase_c_canonical_inflate_format_extension_20260530T230000Z` PROCEED 14-day advisory
* Lane: `lane_z6_v2_phase_c_canonical_inflate_format_extension_20260530` L1 (impl_complete + memory_entry)
* Sister canonical reference: `src/tac/substrates/z5_predictive_coding_world_model/inflate.py:133-180`
* Canonical helper: `tac.substrates._shared.inflate_runtime.write_rgb_pair_to_raw`
* CLAUDE.md Catalog #146 (contest-compliant inflate runtime template), Catalog #205 (canonical select_inflate_device), Catalog #367 (raw-byte fail-closed), Catalog #110/#113 (APPEND-ONLY HISTORICAL_PROVENANCE for v1 schema)
