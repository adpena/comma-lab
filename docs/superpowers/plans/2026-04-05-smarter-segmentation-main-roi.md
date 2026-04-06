# Smarter Segmentation Main-ROI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a compression-side dynamic main-ROI experiment that preserves the central driving corridor, optionally adds one auxiliary ROI, and produces one authoritative local score against the current 3.33 floor.

**Architecture:** Add a small `uv run` Python analysis helper that computes temporally smoothed ROI metadata from motion/staticness heuristics, feed that metadata into the existing multi-stream ROI path in `submissions/robust_current`, then run a single measured local evaluation cycle with explicit current_workflow vs rule_faithful reporting.

**Tech Stack:** bash, ffmpeg/x265, Python via uv, existing comma-lab evaluation helpers, JSON metadata.

---

## File structure

- Modify: `submissions/robust_current/compress.sh`
  - Load ROI metadata when enabled and translate it into per-video/per-window crop coordinates.
- Modify: `submissions/robust_current/inflate.sh`
  - Keep reconstruction compatible with metadata-driven main ROI + optional aux ROI overlays.
- Create: `submissions/robust_current/analyze_roi.py`
  - Compression-side ROI analysis helper, run with `uv run`.
- Create: `experiments/runs/2026-04-05-dynamic-main-roi/README.md`
  - Experiment intent, commands, and result summary.
- Create: `experiments/runs/2026-04-05-dynamic-main-roi/notes.md`
  - Per-run decisions and metadata interpretation notes.
- Modify: `docs/speculative_lanes.md`
  - Record the refined dynamic-main-ROI hypothesis and guardrails.
- Modify: `.omx/state/current_focus.md`
- Modify: `.omx/state/next_experiments.md`
- Modify: `.omx/research/findings.md`
- Modify: `.ralph/run_log.md`
- Modify: `reports/latest.md`
- Modify: `reports/results.jsonl`
- Modify: `reports/timeline.jsonl`
- Modify: `reports/writeup_working.md`
- Create under evaluation output as needed: `reports/raw/2026-04-05-dynamic-main-roi/*`

### Task 1: Define the ROI metadata contract

**Files:**
- Modify: `submissions/robust_current/compress.sh`
- Create: `experiments/runs/2026-04-05-dynamic-main-roi/README.md`

- [ ] **Step 1: Document the metadata shape in the experiment README**
  - Include fields for `video`, `window_start`, `window_end`, `main_roi`, `aux_roi`, and summary stats.

- [ ] **Step 2: Add config/env flags for metadata-driven ROI mode**
  - Add flags such as `ROI_METADATA_ENABLE`, `ROI_METADATA_PATH`, and analysis window controls.

- [ ] **Step 3: Run a shell syntax check**
  - Run: `bash -n submissions/robust_current/compress.sh`
  - Expected: no output, exit 0.

### Task 2: Implement compression-side ROI analysis helper

**Files:**
- Create: `submissions/robust_current/analyze_roi.py`
- Test: manual invocation against one source video via `uv run`

- [ ] **Step 1: Write the helper to emit JSON metadata**
  - Inputs: source video path, target scale, tile size, smoothing factors.
  - Features: frame-diff magnitude, edge energy, center prior, optional road-band prior.
  - Outputs: temporally smoothed main ROI plus optional aux ROI.

- [ ] **Step 2: Run the helper on one sample video**
  - Run: `uv run python submissions/robust_current/analyze_roi.py --video workspace/upstream/comma_video_compression_challenge/videos/0.mkv --scale-w 432 --scale-h 324 --out /tmp/roi-meta.json`
  - Expected: JSON file written with bounded ROI coordinates.

- [ ] **Step 3: Inspect output sanity**
  - Verify main ROI is always present.
  - Verify aux ROI is absent unless justified.
  - Verify coordinates stay in-bounds and even-sized.

### Task 3: Plumb metadata into the existing ROI encode path

**Files:**
- Modify: `submissions/robust_current/compress.sh`

- [ ] **Step 1: Load per-video metadata if enabled**
  - Fall back to static ROI settings when metadata is disabled or missing.

- [ ] **Step 2: Encode base + main ROI + optional aux ROI using metadata-driven boxes**
  - Preserve the main ROI as mandatory.
  - Emit aux ROI only when metadata calls for it.

- [ ] **Step 3: Re-run shell syntax check**
  - Run: `bash -n submissions/robust_current/compress.sh`
  - Expected: no output, exit 0.

### Task 4: Keep inflate compatible and simple

**Files:**
- Modify: `submissions/robust_current/inflate.sh`

- [ ] **Step 1: Confirm inflate expects the same archive layout produced by metadata-driven encode**
  - Keep base/main/aux composition explicit.

- [ ] **Step 2: Run shell syntax check**
  - Run: `bash -n submissions/robust_current/inflate.sh`
  - Expected: no output, exit 0.

### Task 5: Smoke test packaging and inflation

**Files:**
- Modify as needed: `experiments/runs/2026-04-05-dynamic-main-roi/notes.md`
- Create under raw outputs: `reports/raw/2026-04-05-dynamic-main-roi/*`

- [ ] **Step 1: Run one packaging pass**
  - Run: `bash submissions/robust_current/compress.sh`
  - Expected: `submissions/robust_current/archive.zip` created.

- [ ] **Step 2: Run one local inflation/eval smoke test**
  - Run: `uv run comma-lab eval-submission robust_current --device cpu --no-sync`
  - Expected: inflation succeeds and shape/frame checks pass.

- [ ] **Step 3: Record bytes and any runtime anomalies**
  - Save notes and raw logs under the experiment directory.

### Task 6: Run one authoritative local evaluation cycle

**Files:**
- Create/update: `reports/raw/2026-04-05-dynamic-main-roi/*`
- Modify: `reports/results.jsonl`
- Modify: `reports/timeline.jsonl`
- Modify: `reports/latest.md`
- Modify: `reports/writeup_working.md`

- [ ] **Step 1: Run the authoritative local CPU evaluation**
  - Run: `uv run comma-lab eval-submission robust_current --device cpu`
  - Expected: summary JSON + report saved.

- [ ] **Step 2: Compare against the 3.33 floor**
  - Promote only if the full measured result is better.
  - Otherwise record explicit rejection and why.

- [ ] **Step 3: Update results ledgers**
  - Add explicit `current_workflow` result.
  - Add separate `rule_faithful` note/estimate if applicable.

### Task 7: Update durable state and research notes

**Files:**
- Modify: `.omx/state/current_focus.md`
- Modify: `.omx/state/next_experiments.md`
- Modify: `.omx/research/findings.md`
- Modify: `.ralph/run_log.md`
- Modify: `docs/speculative_lanes.md`

- [ ] **Step 1: Summarize what the dynamic main-ROI experiment proved**
- [ ] **Step 2: Explicitly state whether the main ROI prior helped**
- [ ] **Step 3: Queue the next best experiment based on measured evidence**

### Task 8: Verification pass before completion

**Files:**
- No new files required beyond outputs above

- [ ] **Step 1: Run compile/static checks needed for changed scripts**
  - Run: `python3 -m py_compile submissions/robust_current/analyze_roi.py`
  - Expected: exit 0.

- [ ] **Step 2: Re-run shell syntax checks**
  - Run: `bash -n submissions/robust_current/compress.sh && bash -n submissions/robust_current/inflate.sh`
  - Expected: exit 0.

- [ ] **Step 3: Confirm no claim exceeds evidence**
  - Check that any promotion is backed by a saved measured summary.
  - Check that BAT00 is labeled non-authoritative if referenced.
