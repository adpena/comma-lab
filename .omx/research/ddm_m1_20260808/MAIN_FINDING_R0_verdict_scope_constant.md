# M1 SEAL — MAIN FINDING (round 0, self-reported): the verdict instrument hardcodes its own scope

status: OPEN · severity: MEDIUM · class: silent-instrument / label-is-a-constant (m50 vacuity genus)
found_by: MAIN, 2026-08-08, while harvesting the finished arms (NOT by the two live review passes)
axis: `[macOS-CPU advisory torch upstream SegNet]` · score_claim: false

## The defect

`experiments/ddm_mx1_pr130_semantic_renderer.py:2007` returns, in the torch-verdict payload:

```python
"verdict_scope": "n32 arm-selection instrument",
```

It is a **string literal**, independent of what the verdict actually measured. Sister sites with the
same class: `:1382` and `:1860` (`"n32 arm-instrument checkpoint-series facets"`).

`pair_ids` is in scope at the same return site (`"pair_count": len(pair_ids)` two lines below), so
the scope was derivable and simply was not derived.

## What it did to OUR OWN seal artifacts (measured, at source)

Both M1 sigma d_seg verdicts ran **n120** and are labelled **n32**:

| receipt | pair_count | verdict_scope (emitted) |
|---|---|---|
| `sigma/dseg_fp16/verdict_result.json` | 120 | `n32 arm-selection instrument` |
| `sigma/dseg_fp32/verdict_result.json` | 120 | `n32 arm-selection instrument` |

The **numbers are correct** (`per_pair_d_seg` carries 120 entries; `pair_count` is honest). Only the
scope LABEL lies. A reader or automated consumer routing on `verdict_scope` mis-scopes the row by
3.75× in population.

## Blast radius

- **The burn is affected.** The same emission fires at `eval_every=50` throughout the ~12h n120 burn,
  so every in-run verdict row would carry a wrong scope label.
- **The sigma verdict is NOT invalidated.** The σ=0 argument reasons from checkpoint byte-identity
  and `pair_count`, never from the scope string. The falsifier arithmetic stands.
- **No score claim is affected** (`score_claim: false` throughout; advisory axis).

## Two things this finding also SETTLES (measured while checking it)

1. **fp16 vs fp32 verdicts derive from DIFFERENT checkpoints** — sha `56047d05…` (831,396 B) vs
   `9f5ec7ef3…` (831,391 B). The exact d_seg equality (0.0010835435655381944 both) is therefore a
   real measurement, not a cache collision or a path bug. (This is m1r2's Q3, answered affirmatively.)
2. **GT-cache identity IS recorded** — `input_cache` / `target_cache` both name
   `/Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/gt_seg_cache.pt` (GT/GT). This is the
   rr16-class provenance gap, and our receipts do carry it — **as a PATH, not a content hash.**
   rr16 (2026-08-07) measured that the cache binding is load-bearing: parser-default replay gave
   d_seg 0.004567 vs GT/GT 0.001073, a 4.3× swing. A path is a weaker identity than a hash.
   Secondary finding, LOW: add a cache content-hash to the verdict receipt.

## The cure (not yet applied — see sequencing)

Derive the scope instead of asserting it, at all three sites; e.g. at `:2007`

```python
"verdict_scope": f"n{len(pair_ids)} arm-selection instrument",
```

plus the sister facets sites, plus a warn-only guard that refuses a hardcoded `n<digits>` scope
literal in a payload whose own dict carries a pair count.

## Counter-reset (the honest bookkeeping)

Per the gc21 discipline and the concurrency caveat written into both live charters: **ANY finding
resets the 3-clean-pass counter to 0.** This is a MAIN-found finding, so the counter resets the same
way it would for an arm-found one. The two live passes (`ddm_m1r2` mechanics, `ddm_m1r3` science)
are allowed to FINISH — their verdicts remain informative and may surface more — but neither can
count toward the 3, because the cure changes the artifact they reviewed.

Sequencing (avoids reviewing a moving target):
1. This record lands now. Ticket NOT mutated while the passes are live.
2. Both passes land; their findings are collected.
3. ONE amendment applies every cure + resets `review_passes` to `[]`.
4. Three fresh passes run against the cured artifact.

## Why this was worth a reset

A clean pass I did not earn is worse than a reset (my own words in both charters). Sitting on a known
label defect and letting two arms return CLEAN would have laundered it through the gate — and the
defective emission would then have run 65+ times during the burn, producing a full run of rows whose
scope field nobody could trust.
