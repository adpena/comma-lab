# PR95 HNeRV-Muon curriculum — schedule + a realized training trajectory (transparency / open-science)

**Why this exists.** The winning comma.ai compression PR (referred to here as "PR95", the HNeRV-Muon
substrate the 0.193-class leaderboard cluster was built on) publishes only its **static submission**
(`archive.zip` = final trained decoder weights + `inflate.sh` + source code). Like essentially all contest
PRs, **it does not publish the training trajectory** — no per-epoch loss, no per-stage d_seg/d_pose curve.
Those metrics only ever existed in the author's local training logs. We confirmed this against our captured
intake (`pr95_src/` contains only the two frozen scorer weights; the intake profile records "no static score
terms were provided in the intake bundle").

So there is no external reference curve to compare against — and, to our knowledge, **no one has published a
per-stage realized trajectory for this curriculum.** We are publishing ours for transparency.

## What's here

| file | what it is | provenance |
|---|---|---|
| `pr95_8stage_schedule.csv` | the 8-stage curriculum **schedule** (epochs / loss form / LR / Muon / QAT / C1a-λ / σ) | **publicly derivable** — reverse-engineered from the public PR95 source; 29,650 epochs total; matches the published schedule digest |
| `our_realized_trajectory_exact_points.csv` | **our** realized per-stage **exact-eval** reference curve (global_epoch, stage, d_seg, d_pose, rate, archive_bytes, score S) | **our own run** — a faithful PR95-curriculum reproduction (see "Faithfulness" below) |

The realized curve is the 149 periodic exact-eval points of a single continuous run (the full per-epoch log
is ~11k rows of loss/LR/grad-norm; the exact-eval points are the score-bearing reference). Snapshot is
**live / mid-flight** — see the status note below; it will be finalized at run completion.

## Faithfulness — what our run shares with PR95, and where it deliberately differs

Our run executes the **exact PR95 8-stage schedule** (`pr95_8stage_schedule.csv`, read off the vendored
stage builders) with three deliberate, documented deltas (so this is "PR95-curriculum-faithful", **not**
bit-identical PR95):

1. **Architecture**: a smaller tapered HNeRV decoder (`base_channels=20`, channel taper
   `[16,16,17,19,19,14,10]`) — a rate-reducing variant of the vendored decoder.
2. **Seg loss**: the detector-informed `margin_hinge` surrogate substituted for the vendored seg surrogate.
3. **Optimizer fix**: the stage-8 Muon LR-floor fix (`muon_lr_floor_fix`), so the final Muon polish anneals to
   the intended fine-LR rather than flooring at 50% of peak.

It also runs on Apple MPS for the training gradient (CPU for the authority eval), not the author's CUDA.

## The honest comparison — endpoints, not paths (NO score claim)

- **PR95 published score**: ~0.193 `[contest-CPU]` (the leaderboard axis). That is the author's *final*
  byte-closed contest result.
- **Our run**: every number in `our_realized_trajectory_exact_points.csv` is **`[contest-CPU advisory]`
  (macOS, non-promotable)** — it is NOT an authoritative contest score. An authoritative score requires
  `upstream/evaluate.py` on contest-CPU (Linux x86_64) / contest-CUDA over the 600 samples on the exact
  byte-closed `archive.zip`. Our canonical exact frontier pointer is unchanged at **0.19110** (a separate,
  byte-closed result); this run has not yet produced a byte-closed exact row.
- **Where the run is** (snapshot): the advisory S has descended ~2.32 → ~0.309 across stages 1→5. The binding
  term is **d_seg** (≈65% of S); it descends slowly in the AdamW stages (1–7) and the bulk of its closing is
  reserved for stage 5 (C1a-L7) and the spectral **stage 8 (Muon)** finisher, which is where the decisive
  read will land.

## Reuse / reproduction

The schedule CSV is everything needed to reproduce the curriculum; the realized CSV is a reference for what a
faithful reproduction's per-stage d_seg/d_pose/S curve actually looks like. Anyone reproducing PR95 can use
the realized curve to sanity-check their own run stage-by-stage instead of flying blind to the final score.

*All values advisory; no contest-score claim is made by this document. See the comma.ai compression challenge
for the authoritative scorer.*
