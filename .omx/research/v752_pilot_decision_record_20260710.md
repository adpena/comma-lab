# v7.5.2 FROM-SCRATCH PILOT — pre-registered decision record (operator-driven hold-gate 2, 2026-07-10)

**STORES CONSULTED:** `DUAL_CHAIN_BRIEF_385_20260710.md` (ADDENDUM v2 + operator GO) · owed-16 verdict
(`owed16_verdict_20260710.json`, EmpiricalAnchor `owed16_realized_transfer_measured_zero_20260710`) ·
DAG FEED-owed16-verdict (reformulation queue) · `SPEC_v75_optimal_single_trunk_20260708.md` §4/§5/§8 ·
mod32cap run dir `experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z` (run.log verdict
rows, MEASURED) · crucible_v752 compile (`tac.witness_autoconfig.compile_crucible_v752_launch_config`,
commit `3b028a374`). Pointer contest-CPU **0.19110 UNMOVED** — everything here is MEANS.

## 1. WHY (the hold's second gate — operator verbatim "We don't want to launch v7.5.2 if we know its confounded or not optimal")

The self-orient-OFF decision (owed-16 P9 RESOLVED-REFUTING) rests on **warm-start evidence only**
(verdict_scope: FORMULATION — bounded ep650→700 fine-tune from a self-orient-TRAINED parent). The
uncovered arm is FROM-SCRATCH: the parent trunk may carry internalized directional structure that
persists in the OFF ablation, so OFF-from-scratch could lag at PARTITION FORMATION (ep0–300) even though
OFF-from-warm-start is ≈0-loss. This pilot closes that formulation-scope gap BEFORE the 6–16 h launch
commits. (DAG FEED-owed16-verdict reformulation-queue item 1, now elevated to a launch gate.)

## 2. THE PILOT (config = the launch's own first 300 epochs; VERIFIED byte-close)

`compile_crucible_v752_launch_config(gt_n600, num_pairs=600, epochs=300)` — the GO'd self-orient-OFF
launch config with ONLY the epoch cap changed. **MEASURED argv diff vs the epochs-3000 launch config:
exactly 2 tokens** — `--epochs 300` (vs 3000) and `--polyak-finisher-start-epoch 301` (vs 2546; the
degenerate clamp = epochs+1 ⇒ count=0). Both Polyak starts are > 300 ⇒ **the pilot's 300 epochs are
training-dynamics-IDENTICAL to the launch's first 300 epochs** (all schedule pins are ABSOLUTE:
anneal-den 3000, tau@300, muon/pose-finish caps 726 — none rescale with --epochs). Already in-config:
`--seed 0 --eval-every 25 --verdict-pairs 0 --verdict-batch 32 --ckpt-every 25 --stage-checkpoints`
(resumability P0). DSL-validated 0 violations; parses the real trainer argparse clean.

**The RESUME-FROM-PILOT option (why the pilot costs ~nothing if it passes):** because the pilot IS the
launch's first 300 epochs, a PASS verdict lets the real launch fire as
`--config crucible_v752` (sealed epochs 3000) + `--extra-trainer-flags "--resume-from <pilot_out_dir>"`
— restoring the pilot's ep300 checkpoint and continuing to 3000, trajectory-faithful to an
uninterrupted launch (the trainer's resume contract). The pilot's wall-clock is then simply the
launch's own first ~10%, not an added cost.

## 3. PINNED GOVERNED PILOT COMMAND (fires ONLY after gate-1's machine frees; see §5)

```bash
cd /Users/adpena/Projects/pact
.venv/bin/python tools/launch_witness_run.py \
  --config crucible_v752 \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --num-pairs 600 \
  --epochs 300 \
  --out-dir experiments/results/levelset_v752_pilot_<TS>
```

(launcher stamps the explicit-epochs override note; pilot wall-clock budget DERIVES to 0.831 d; full
gate chain — memory-preflight ~24.5 GiB projected, DSL-config, schedule-provenance, safe-compile,
system-admission, throughput — runs unmodified; durable governed spawn.) Wall-clock estimate: see the
dry-start report (`experiments/results/__v752_drystart__/dry_start_report.json`) for the measured
sec/ep; mod32cap's from-scratch cadence anchor is ~116 s/ep (ep25→300 in 8.88 h, self-orient-ON) —
the OFF pilot should run at or below that; ~300 ep ⇒ roughly a 6–10 h read-out.

## 4. PRE-REGISTERED COMPARISON PROTOCOL + PASS BAND

**Reference = mod32cap's banked from-scratch n600 trajectory** (run dir above; MEASURED verdict cells,
CE stage, self-orient ON, mod-dim 32):

| ep | 0 | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 | 225 | 250 | 275 | 300 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d_seg | 0.7439 | 0.009288 | 0.007248 | 0.006354 | 0.005856 | 0.005519 | 0.005288 | 0.005121 | 0.004963 | 0.004869 | 0.004751 | 0.004682 | 0.004571 |

**HONESTY CAVEAT (binding on the verdict's scope):** mod32cap is NOT a controlled self-orient A/B
against the pilot — it differs in the WHOLE v7.5 lever stack (no counter-force / ladder-birth /
dash-comb / temporal-screw / taper / unify-τ; different curriculum pins; islands-unborn DELIBERATE).
It is the REFERENCE ANCHOR for "healthy from-scratch formation at this capacity on this GT", not an
isolation of the basis. A controlled from-scratch OFF-vs-ON matched pair remains reformulation-queue
item 1 if this pilot is ambiguous.

**Facets read (holistic, per the SPEC §5 watch-list — never the composite alone):**
1. **Composite d_seg descent** at matched epochs (the primary band, below).
2. **Descent RATE** — the log-slope over ep100→300 (mod32cap: ln(0.005856/0.004571)/200 ≈ 1.24e-3/ep).
3. **Island birth** — part_frac[lane]>0 by ep25 (the paint-then-SDF seed admission gate) and
   lane/movable part_frac approaching the Chan-Vese equilibrium band ≈1.25×GT (v752-only machinery;
   mod32cap has NO birth stack — v752 should be STRICTLY better here; a v752 lane/movable
   islands-unborn read-out is a REGRESSION vs its own design, independent of mod32cap).
4. **Per-class d_seg** — Road PRIMARY (SPEC §5: the Chan-Vese success target; run-1's 13.8× over-paint
   is the failure signature to watch), Lane/Movable vs the birth-stack expectation.
5. **d_pose** — WATCH-ONLY at ep≤300 (pose-finish gates at the muon event ≥726; pose-blind by design
   in the pilot window; mod32cap's ~104–134 band is the analog).

**PASS BAND (pre-registered):**
* **PASS → launch OFF** (optionally resume-from-pilot-ep300): pilot composite d_seg ≤ **1.15×**
  mod32cap's matched-epoch cell at ep100–300 (≤ **1.25×** at ep25–75, the noisier formation cells),
  AND log-slope(ep100→300) ≥ **0.7×** mod32cap's, AND island-birth facet healthy (lane part_frac>0 by
  ep25; no run-1-style >4× GT over-paint persisting past ep150 against the Chan-Vese equilibrium).
* **LAG → basis matters at formation:** pilot composite d_seg > the band at ≥2 consecutive matched
  cells in ep100–300, OR log-slope < 0.7× — the launch config switches to
  **self-orient-REBALANCED-early-annealed-OFF** (directional channels ON at formation with the
  freq-along-heavy allocation per the owed16v2 rebalance verdict, annealed off after the partition
  forms), compiled through the DSL as its own amendment before any launch.
* **AMBIGUOUS** (band-straddling / facet disagreement): NO launch; run the controlled from-scratch
  OFF-vs-ON matched pair (reformulation-queue item 1) before committing. One crisp verdict, then act.
* Band rationale: 1.15× ≈ the largest adjacent-cell step in the reference's ep100–300 window (so a
  within-band pilot is indistinguishable from one verdict-cadence of ordinary progress); the P2
  single-seed noise floor is unmeasured at these cells (owed) — the band is therefore deliberately
  WIDE and the LAG trigger requires 2 consecutive cells, never one.

## 5. SEQUENCING (the two pre-registered hold gates, then the launch fires with no further gates)

1. **Gate 1 — owed16v2 rebalance verdict** (in flight, `owed16v2_rebalanced_ON_20260710T114759Z`,
   pid 64206): reads out the freq-along-heavy warm-start arm vs the banked OFF trajectory. Feeds the
   LAG branch's allocation (and could independently inform the launch config).
2. **Gate 2 — this pilot:** fires AFTER the rebalance arm completes and frees the machine (~22 GiB
   pilot admits easily solo). Read-out per §4.
3. **Launch:** whichever config §4 selects, via the governed launcher; if PASS, optionally
   `--resume-from` the pilot ep300 checkpoint.

**Pointer 0.19110 UNMOVED — this record is MEANS.** Only a byte-closed `upstream/evaluate.py` n600 row
< 0.19110 moves it.
