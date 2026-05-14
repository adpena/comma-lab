# Comprehensive handoff — 2026-04-07

This document is the authoritative handoff for the current state of the comma-lab repository after the post-filter promotion cycle.

It is intended for a trusted partner agent or engineer who needs enough context to continue work without relying on chat history.

---

## 1. Current authoritative state

### Track A — `exact_current`

- Purpose: transparency / current-workflow exploit lane only
- Status: intentionally non-rule-faithful
- Use: keep runnable, but do not confuse with honest submission quality

### Track B — `robust_current`

Current best authoritative floor:

- `current_workflow`: **2.05**
- `current_workflow` bytes: **861,986**
- `rule_faithful`: **2.0778631822069484**
- `rule_faithful` bytes: **896,432**
- config:
  - `522x392`
  - `libsvtav1`
  - `preset0`
  - `crf34`
  - `film-grain22`
  - `lanczos`
  - `sharpness=1`
  - tiny learned int8 post-filter in inflate path

Authoritative scorer result:
- PoseNet distortion: `0.07996829`
- SegNet distortion: `0.00586716`

Why it matters:
- this is the first decode-side learned lane in the repo that beat the prior `2.08` honest floor in a real scorer-backed run

---

## 2. Authoritative evidence locations

### Promoted 2.05 post-filter floor

- scorer report:
  - `reports/raw/2026-04-07-postfilter/robust_current-exp-postfilter-current_workflow-cpu-report.txt`
- scorer summary:
  - `reports/raw/2026-04-07-postfilter/robust_current-exp-postfilter-current_workflow-cpu-summary.json`
- smoke:
  - `reports/raw/2026-04-07-postfilter/robust_current-exp-postfilter-smoke.json`
- canonical live summary/report pair:
  - `reports/raw/robust_current-current_workflow-cpu-summary.json`
  - `reports/raw/robust_current-current_workflow-cpu-report.txt`

### Key comparison evidence

- first post-filter on the wrong train distribution:
  - `reports/raw/2026-04-06-av1-roi-experiments/exp_postfilter_v1_report.txt`
  - `reports/raw/2026-04-06-av1-roi-experiments/exp_postfilter_v1_scorer.log`
- grain-mask final verified reject:
  - `reports/raw/2026-04-06-av1-roi-experiments/exp_grain_mask_final_report.txt`
  - `reports/raw/2026-04-06-av1-roi-experiments/exp_grain_mask_final_scorer.log`
- consensus stack reject:
  - `reports/raw/2026-04-06-pre-submit-round/robust_current-exp-h-current_workflow-cpu-summary.json`
  - `reports/raw/2026-04-06-pre-submit-round/robust_current-exp-h-current_workflow-cpu-report.txt`
- preprocess reject:
  - `reports/raw/2026-04-06-pre-submit-round/robust_current-exp-j-current_workflow-cpu-summary.json`
  - `reports/raw/2026-04-06-pre-submit-round/robust_current-exp-j-current_workflow-cpu-report.txt`
- sharpness=1 promoted baseline:
  - `reports/raw/2026-04-06-pre-submit-round/robust_current-sharpness1-current_workflow-cpu-summary.json`
  - `reports/raw/2026-04-06-pre-submit-round/robust_current-sharpness1-current_workflow-cpu-report.txt`

---

## 3. Current canonical runtime payload

The honest installed/runtime payload for the promoted lane now includes:

- `archive.zip`
- `inflate.sh`
- `inflate.py`
- `inflate_postfilter.py`
- `inflate_grain_mask.py`
- `postfilter_int8.pt`
- `config.env`
- `analyze_roi.py`

Important:
- `rule_faithful` must charge the payload under test, not an ad hoc larger local tree
- the canonical install payload definition lives in `src/comma_lab/install.py`
- if any summary surface lists only a subset of the shipped post-filter files under `rule_faithful_bundle_paths`, treat the path list as stale and refresh it before citing it
- if any writeup or ledger shows a smaller rule-faithful byte total than the archive itself, treat it as suspect until reconciled

---

## 4. What we learned from the experiments

### 4.1 The big lessons

#### A. PoseNet is the real constraint
Most losing experiments did not fail because of bytes alone.
They failed because PoseNet distortion rose too much.

#### B. Broad preprocessing is mostly a dead end here
The following families were repeatedly bad:
- blur preprocessing
- gentle preprocess
- ROI corridor preprocess
- chroma-only degradation
- broad denoise-first variants

These can save bytes, but they usually hurt PoseNet too much.

#### C. Encoder-side safe wins exist, but they are small
Confirmed useful knobs:
- `sharpness=1`
- repaired/canonical inflate path
- better upscale kernel choices

These matter, but mostly in the hundredths, not tenths.

#### D. Grain is not cosmetic noise
The `film-grain=0` hypothesis failed badly.
Film-grain synthesis appears to preserve task-relevant structure for PoseNet.

#### E. Task-aware decode correction is the strongest remaining lane
The learned post-filter is the first lane that moved the honest floor meaningfully beyond the prior AV1 tuning plateau.
That is the highest-signal lesson in the repo right now.

---

## 5. Important scored frontier points

### Strongest honest results

- **2.05** — learned post-filter (promoted)
- **2.08** — sharpness=1 floor
- **2.09** — ROI map / saliency ROI / sharpness=2 / Rubin stack cluster
- **2.12** — earlier colorspace-hardening floor
- **2.18** — earlier Lanczos-upscale floor

### Important verified rejects

- **2.35** — first post-filter trained on the wrong distribution
- **2.13** — consensus stack (`crf33 + scd0 + hqdn3d + sharpness=1`)
- **2.30** — saliency-masked grain recovery lane
- **2.52** — ROI preprocessing stack
- **2.94** — `film-grain=0` catastrophe family

### Interpretation

The path from here is not:
- more broad preprocessing
- more random codec knob sweeps
- more geometry poking for its own sake

The path from here is:
- task-aware decode correction
- very disciplined model-size / architecture / objective sweeps
- possibly ROI-local residual correction if it stays tiny and measurable

---

## 6. Learned post-filter lane — current status

### Verified facts

- post-filter code exists:
  - `submissions/robust_current/inflate_postfilter.py`
- shipped weights exist:
  - `submissions/robust_current/postfilter_int8.pt`
- payload manifest includes them:
  - `src/comma_lab/install.py`
- inflate routing includes `PYTHON_INFLATE=postfilter` branch:
  - `submissions/robust_current/inflate.sh`
- local isolated smoke passed
- BAT00 smoke passed
- local authoritative scorer passed at **2.05**
- the first post-filter variant was rejected at **2.35** because it was trained on the wrong archive distribution

### Why this lane is special

It is not just a proxy curiosity anymore.
It survived:
1. packaging
2. inflate
3. smoke
4. full scorer

That makes it a real promoted mechanism.

### Follow-on branches worth testing

These are still speculative until measured:

1. **Slightly larger model**
   - hidden channels `16 -> 24`
   - hidden channels `16 -> 32`
   - same 3-layer topology first

2. **Smaller model**
   - hidden channels `16 -> 8`
   - 2-layer residual variant
   - luma-only post-filter

3. **Cheaper architecture**
   - depthwise-separable residual block
   - per-channel kernels + tiny pointwise mix
   - BSConv-style variant

4. **Better objective / data**
   - train on the real fg22/sharpness operating point, not just a recovery path
   - ROI-weighted loss
   - temporal asymmetry / odd-frame emphasis if justified

5. **Deeper speculative hybrids**
   - LUT-like correction stage
   - residual patch + post-filter hybrid
   - post-filter + ROI map stack (only after one-axis evidence)

### Byte-cost warning

Model bytes are cheap enough that a slightly larger model is worth exploring.
Roughly:
- +10 KB payload ≈ +0.0067 score from the rate term
- +20 KB ≈ +0.0133
- +50 KB ≈ +0.0333

That means "bigger" is a legitimate direction, not just "smaller" and "cheaper."

---

## 7. Grain-mask lane — current status

### Verified result

- score: **2.30**
- bytes: **716,797**
- PoseNet: `0.15428504`
- SegNet: `0.00577725`

### Lesson

This lane recovered a large portion of the `film-grain=0` disaster, but it still did not get near the honest floor.
It is useful as a research result, not a promotion candidate.

### Warning

Do not let the strong byte savings mislead you.
This lane is not competitive unless PoseNet can be improved dramatically.

---

## 8. Public leaderboard / PR context that matters

As of the latest checked public state:

- PR #31: `1.95`
  - ROI-aware preprocessing + unsharp
  - still the key no-GPU public leader to study
- PR #32: `1.77`
  - gradient-optimized decode correction
  - GPU-required
  - extremely relevant as proof that task-aware decode optimization can dominate classic codec tuning
- PR #37: `2.16`
  - spline downscale
  - not competitive with our current floor

### Public takeaway

The public field, as checked on 2026-04-07, reinforces the same conclusion:
- classic AV1 tuning matters, but it is not enough to win now
- task-aware decode correction is where the remaining upside lives

---

## 9. BAT00 status and warnings

### BAT00 role
BAT00 is now useful as a serious side runner.
Use it for:
- smoke/ranking
- architecture ablations
- long-running non-authoritative scorer cross-checks
- tooling bring-up / parity checks

### BAT00 is still non-authoritative
Do not promote from BAT00 alone.
It is a side lane until the winning config is rerun locally on the authoritative scorer path.

### Important BAT00 lesson
The first BAT00 smoke batch was invalid.
Cause:
- remote worker skipped `--package`
- reused shared mutable state

Fixes that are now in place:
- per-job workspace copy
- per-job config snapshot
- `smoke-submission --package`
- per-job isolated upstream root
- per-job `manifest.json` / `status.json`
- remote ledger under `~/bat00-runs`

### BAT00 warning
The overnight BAT00 post-filter cross-check lives under a UTC-next-day timestamp:
- `~/bat00-runs/exp_postfilter/20260408T003020Z/*`

That is still the same promotion cycle; it just crossed midnight UTC.

---

## 10. Process / orchestration failures we hit

These are important. They will bite a new agent if ignored.

### A. Tool-host exhaustion (`Too many open files`)
Observed repeatedly:
- new shell/process/file-edit attempts failed with `os error 24`
- once that happened, only some MCP paths still worked

Likely causes:
- too many long-lived `exec_command` sessions
- repeated polling sessions
- SSH wrappers to BAT00
- long scorer runs through `tee`
- overlapping ffmpeg / rebuild / scoring processes

### B. Stale lock collisions
Observed:
- stale PID metadata blocked reruns
- manual cleanup was required in some cases

### C. Shared-tree drift
There is another trusted partner working in the repo.
Do not assume shared `submissions/robust_current/archive.zip` or shared reports always reflect your own intended run.

### D. Background-session sprawl
Old monitoring jobs and shell wrappers can silently accumulate and increase failure risk.

---

## 11. Operational warnings for the next agent

### Do not do these

- do not overwrite shared canonical files casually while another partner is active
- do not promote from BAT00 alone
- do not trust any summary that contradicts archive size or payload composition
- do not assume preprocessing is worth revisiting without a very specific new hypothesis
- do not run a large number of long-lived shell sessions in parallel unless you control cleanup tightly

### Do these instead

- run authoritative work in `<scratch>/pact-authoritative` or another isolated workspace (use repo-relative `experiments/results/<lane_id>_<timestamp>/` or `.omx/tmp/` per CLAUDE.md "Forbidden /tmp paths in any persisted artifact")
- only copy into shared canonical paths at promotion time
- use BAT00 for smoke / ranking / non-authoritative scoring
- keep one owner per long-running job
- prefer file-based polling over many live shell sessions
- clear stale locks only deliberately and record why

---

## 12. Current writeup/site state

The site and side-by-side media were rebuilt from the verified 2.05 post-filter archive.

What still needs follow-up:
- a full recursive senior-engineer + senior-editor cleanup loop
- final consistency pass across generated site outputs after the 2.05 update
- possible improvement to the media builder so it can accept an explicit promoted artifact source instead of assuming shared `submissions/robust_current/archive.zip`

### Warning

Do not assume every generated text surface is already fully clean just because the rebuild succeeded.
A final review loop is still required.

---

## 13. Recommended next steps

### Immediate

1. run a full senior-engineer review on the new 2.05 writeup/site state
2. run a full senior-editor review on the same surfaces
3. fix issues
4. rebuild site/media
5. repeat until the reviews stop finding real issues

### Then

6. run the next capped three-experiment post-filter cycle:
   - slightly larger model
   - smaller or luma-only model
   - cheaper architecture

### Only after that

7. consider whether the 2.05 floor is ready for publishing/submission

---

## 14. If you must recover from a broken session

If shell tooling starts failing again with `Too many open files`:

1. restart the Codex/session tool host
2. kill stale local scorer/eval processes
3. kill stale BAT00 wrapper sessions
4. clear stale `.omx/locks` deliberately
5. verify only the intended scorer jobs remain
6. resume from the evidence in `reports/raw/**` and this handoff document

---

## 15. Bottom line

The project is no longer primarily a codec-tuning project.
It is now a task-aware decode-correction project with a real honest floor at **2.05**.

The biggest lessons are:
- broad preprocessing mostly loses
- grain is structurally important
- tiny learned post-filters are real
- orchestration hygiene matters almost as much as model quality

If you pick up from here, protect the 2.05 floor, keep the evidence honest, and push the post-filter family first.
