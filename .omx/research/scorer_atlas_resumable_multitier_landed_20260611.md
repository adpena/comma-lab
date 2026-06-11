# Scorer spectral-sensitivity atlas: resumable, multi-tier, exhaustive daemon RELAUNCHED (2026-06-11)

**Authority:** `[macOS-CPU advisory]` / `exact_pair_scorer` -> `mechanism_update_eligible` ONLY (inherited
from the v2 physics). NOT a score row; does NOT move the canonical frontier pointer (UNMOVED at
0.19109982). Lane: `lane_atlas_resumable_multitier_20260611` (L1: impl_complete + three_clean_review +
memory_entry). Commit: `ad07cb838`. Implements operator standing directive 2026-06-11
(`feedback_long_resumable_saturation_sweeps_standing_directive_20260611.md`).

## The defect this fixes (the killed 31h runaway)

The v2 atlas (`tools/measure_scorer_spectral_sensitivity.py v2`) held every cell in memory and wrote the
final JSON only at the END. A kill/crash/reboot lost ALL progress — exactly why the prior 31h orphan
(cell 800/3200, ~5-day ETA) had to be killed with the per-cell H values stranded in a text log
(`scorer_spectral_atlas_v2_partial_20260609/`). Per the directive, the runtime length + orphan status
were NOT the defect — the ONLY real defect was no per-cell durable checkpoint + no resume. This landing
makes the sweep resumable-by-construction so a days-long detached orphan is safe.

## What changed (extend, not rewrite — SEARCH-AND-FAMILIARIZE)

Reused the entire existing v2 physics (band synthesis, channel-basis rotation, coordinate conversion,
energy audit, three response levels, the exact `DistortionNet` path). Only ADDED resume + tiers + analysis.

1. **`src/tac/analysis/scorer_spectral_sensitivity_v2.py`** — refactored into a streaming producer +
   aggregator so the all-at-once and the resumable paths share ONE cell-production codepath:
   - `cell_seed_for(...)` — the deterministic per-cell RNG seed, now keyed off the cell's INTRINSIC
     identity (band index + the FIXED canonical label-space index of each categorical axis + the amplitude
     value) instead of the axis's POSITION within the swept grid subset. This is the contract that makes
     resume bit-exact: a cell drawn in any run (any ordering, any axis subset) draws the same random-phase
     field (CLAUDE.md "Seeds pinned").
   - `cell_key(...)` / `cell_key_str(...)` — the canonical 6-axis identity dict + its compact dedup token
     (amplitude formatted at fixed precision so float noise can't split one logical cell into two keys).
   - `iter_atlas_cells(...)` — the RESUMABLE generator: yields `(cell_dict, key_str)` per cell, skipping
     any `skip_cell_keys` already present (resume), measured with the deterministic seed.
   - `aggregate_atlas_from_cells(...)` — rebuilds the final atlas (schema + headline peaks + authority
     flags) from a (possibly partial) list of cells; re-aggregatable any time from the JSONL.
   - `measure_atlas(...)` is now a thin wrapper over those two — so the existing end-to-end test still
     passes and the two paths are provably the same.
   - **Vectorized `_area_downsample`** (energy-audit resize proxy) via `np.add.reduceat`: the prior
     nested-Python double-loop over the 384x512 scorer-input grid was ~0.8s/cell of HIDDEN cost (a 24-cell
     stub run dropped 20s -> <1s). Bit-identical to the loop (max |diff| 4e-16 across 5 shape cases incl
     the upsample fallback). This also speeds the REAL pipeline — the killed run paid this per cell.

2. **`src/tac/analysis/scorer_spectral_atlas_runner.py`** (NEW) — the reusable resumable orchestration:
   - **Durable JSONL custody** — `append_cell_jsonl` appends one record `{"key","cell_index","ts_utc",
     "cell"}` per cell under an exclusive `fcntl.flock` + `flush` + `fsync`. It REPAIRS a missing trailing
     newline before appending (a crash mid-write would otherwise MERGE the next line into the corrupt
     partial and lose a cell — caught by the resume test).
   - `read_cells_jsonl` / `load_completed_cells` — skip blank/corrupt trailing lines (the in-flight cell
     is simply re-measured); last-writer-wins per key (no double-count on a re-run).
   - **Tier presets** `TIER_PRESETS` + `grid_for_tier`: `quick` (~32 cells, minutes), `medium` (~192
     cells, ~1-2h — the right-sized actionable-bands recipe), `exhaustive` (6400 cells, ~days — full CI).
   - **`run_resumable_atlas`** — reads the JSONL skip-set on startup, streams each new cell to the JSONL,
     refreshes the progress sidecar, re-aggregates the final atlas from the FULL JSONL, and emits the
     lowering analysis. Accepts a pre-built scorer + baseline + threshold so resume sessions don't reload.
   - **Marker-on-exit + progress**: `write_done_marker` (EXIT + completed/total) and
     `write_progress_sidecar` (`atlas_progress.json`, atomic replace) so an external check sees
     %-complete + ETA without parsing the JSONL.
   - **`analyze_lowering_opportunities`** — the lowering-oriented analysis (below).

3. **`tools/measure_scorer_spectral_sensitivity.py`** — two new thin subcommands delegating to the runner:
   - `v2-resume --tier {quick,medium,exhaustive} [grid overrides] --work-dir ...` — the resumable sweep.
     Decodes the source once, prints cell count + rough ETA, runs/RESUMES, writes a DONE.marker in a
     `finally` (so a crash still records the exit code).
   - `v2-aggregate --cells-jsonl ... --out ...` — re-aggregate the atlas + lowering analysis from a
     partial/complete JSONL WITHOUT loading the scorer (lets you inspect a still-running or killed sweep).

## Tier cell-counts + ETAs (measured / estimated)

| tier | cells | n_pairs x phases | rough ETA (CPU) | purpose |
|---|---|---|---|---|
| quick | 32 (preset) | 4 x 1 | ~minutes | coarse peak / smoke |
| medium | 192 (preset) | 12 x 1 | ~1-2h | actionable bands (the kill-memo right-sized recipe) |
| exhaustive | 6400 | 12 x 2 | ~144h (~6 days) measured from the live daemon's first-cell rate | full cross-product / CI |

(The 5 oriented x 5 amplitude x 8-channel x 4-incidence x 8-band exhaustive grid = 6400; the daemon's
own first-cell timing reported ETA~144h.)

## Resume-idempotency: tests GREEN (the NO-FAKE contract)

`src/tac/tests/test_scorer_spectral_atlas_runner.py` — 18 tests; full module pair = **48 tests pass**
(46 fast in 1.4s + 2 torch-gated slow in 84s). The headline guards:

- **`test_resume_is_bit_identical_to_uninterrupted`** — runs a 24-cell sweep, KILLS it (truncates the
  JSONL to ~60% + a half-written final line, the realistic crash signature), RESUMES with a fresh scorer
  instance, and asserts (a) the skip-set was exactly the completed cells, (b) the resume recomputed ONLY
  the remaining cells, and (c) EVERY cell value + both headline peaks are BIT-IDENTICAL to an
  uninterrupted run. If resume re-seeded differently this FAILS.
- **`test_resume_end_to_end_real_scorer_bit_identical`** (torch-gated) — the SAME bit-identity claim
  through the EXACT `DistortionNet` (the authority surface): kill after 1 cell, resume, assert every cell
  matches the uninterrupted real-scorer run.
- **`test_truncated_final_jsonl_line_is_skipped_and_remeasured`** — a crash mid-write leaves a half-line;
  it is ignored on read and the cell is re-measured.
- `test_cell_seed_is_order_invariant_and_key_derived`, the tier-size monotonicity, and the lowering
  analysis spend/shed tests.

The `_area_downsample` vectorization was verified bit-identical to the prior loop (max |diff| 4.4e-16).

## Exhaustive daemon RELAUNCHED (detached, resumable, checkpointing)

- **PID:** `26486` (the `v2-resume` python; wrapped by a `nohup ... ; OUTER_DONE.marker` bash launcher).
- **Tier:** `exhaustive` (6400 cells), device `cpu`, `--progress-every 1`.
- **Work dir (SSD):** `/Volumes/VertigoDataTier/pact/scorer_spectral_atlas_exhaustive_20260611T080857Z/`
- **JSONL (the resume store):** `.../atlas_cells.jsonl`
- **Sidecars:** `.../atlas_progress.json` (heartbeat %-complete + ETA), `.../DONE.marker` (inner,
  exit-code), `.../OUTER_DONE.marker` (launcher), `.../daemon.log`, final `.../atlas.json`.
- **Verified checkpointing as it runs:** cell 1/6400 landed in the JSONL
  (`b0|o:isotropic|a:0.5|cb:rgb|ch:all|fi:frame0_only`), progress sidecar = `in_progress 1/6400`, and a
  read-only `v2-aggregate` of the LIVE partial JSONL succeeded WITHOUT touching the daemon (daemon stayed
  alive). It started fresh (the killed run's per-25-cell text log is NOT a resumable JSONL, so there was
  no JSONL to resume from — noted).
- **Coexistence:** the decisive capstone daemon (pid `72123`) is UNTOUCHED and still running. On the
  16-core M5 Max the two advisory/decisive sweeps each have cores.
- **Orphaning is APPROVED:** it's fine for this to run ~6 days orphaned now that it checkpoints. A
  kill/crash/reboot loses at most the in-flight cell; re-launching with the same `--work-dir` RESUMES
  from the JSONL. To resume after a reboot, re-run the exact `v2-resume --tier exhaustive --work-dir <the
  same dir>` command.

To check progress without parsing the JSONL:
```bash
cat /Volumes/VertigoDataTier/pact/scorer_spectral_atlas_exhaustive_20260611T080857Z/atlas_progress.json
```

## Lowering-oriented analysis (the GOAL_v3 invisible-subspace rate lever)

`analyze_lowering_opportunities` emits BOTH halves of the atlas:

(a) **spend_here_freq_budget** — per-axis (band/orientation/channel/incidence) ranking by contest-weighted
    sensitivity `100*|H_seg| + sqrt(10*max(H_pose,0))` (the score units the scorer actually moves), plus a
    per-band table with each band's `siren_w_equivalent` (the carrier's omega). This is WHERE TO SPEND
    BYTES (the carrier's frequency budget — the arbitrariness cure).

(b) **shed_here_low_sensitivity_cells** — the SCORE-LOWERING lever: cells where the scorer is effectively
    blind (combined sensitivity below a near-zero blind floor `1e-4` in score units, OR bottom-quartile
    AND below both the seg+pose response medians). The carrier can spend FEWER bytes there invisibly.
    Consumer = the **bit-allocator / waterfiller**: spend bits where H is high, shed where H ~ 0.

This runs at the end of every `v2-resume` and standalone via `v2-aggregate` on a partial JSONL — so the
shed-bytes opportunity surfaces continuously as the exhaustive sweep accumulates cells.

## #1 score-lowering opportunity the partial/early atlas suggests

From the preserved partial signal (the killed run's completed isotropic all-band sweep, cells 1-625) +
the early exhaustive cell: **SegNet is broadly BLIND to band-limited perturbations (max H_seg ~0.009;
most cells H_seg < 0.003), while PoseNet's response is concentrated at LOW spatial frequency +
HORIZONTAL orientation.** The direct rate-shedding lever: the carrier is paying bytes for **high-frequency
content and SegNet-targeted spectral detail that the scorer does not react to**. The waterfiller should
SHED bytes in the high-frequency bands and in the broadly-flat SegNet response surface (spend the freed
budget on the low-freq/horizontal pose-sensitive bands the carrier actually needs). Concretely this
validates a LOW carrier frequency budget (small `grid_pe_num_freqs` / small SIREN omega) AND says the HF
tail of the carrier's spectrum is sheddable rate. The exhaustive atlas will quantify which exact
(band, orientation, channel, incidence) cells are the cheapest to shed (the shed_here list); the medium
tier already produces an actionable version in ~1-2h if a faster turnaround is wanted.

## 6-hook wire-in (Catalog #125)

- #1 sensitivity-map: ACTIVE — the atlas IS a per-(band,orientation,channel,incidence) scorer sensitivity
  map (the spend_here ranking).
- #2 Pareto: N/A (advisory measurement; not a candidate archive).
- #3 bit-allocator: ACTIVE (design) — `analyze_lowering_opportunities` declares the consumer =
  bit_allocator_waterfiller, with both the spend (freq budget) and shed (low-sensitivity cells) feeds.
- #4 cathedral autopilot: N/A (advisory, non-promotable).
- #5 continual-learning posterior: N/A (advisory measurement; does not write the score posterior).
- #6 probe-disambiguator: ACTIVE — the atlas disambiguates "where the carrier should place frequency
  content" (the arbitrary-w question) and "where the carrier can shed bytes."

## Authority / NO-FAKE

Every artifact carries `authority_tier=exact_cpu_advisory`, `promotable=false`,
`mechanism_update_eligible=true`. The resume produces bit-identical results to an uninterrupted run (the
deterministic per-cell seed) — proven by the kill/resume tests through both the stub and the EXACT
`DistortionNet`. NO MPS. The exhaustive sweep runs on CPU only.
