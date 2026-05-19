# Historical: Task-Aware Compression (`tac`) Lossless Implementation Plan

> Historical implementation plan. Several tasks below have since landed or
> changed shape, and some named entry points are no longer live. Treat this as
> provenance for the lossless subsystem, not as the current queue. Prefer
> `README.md`, `docs/README.md`, and `src/tac/README.md` for current commands.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a canonical `tac/lossless` subsystem and `tac` CLI that support exact commavq-style lossless experiments without trampling the lossy path.

**Architecture:** Add a new `src/tac/lossless/` subsystem with typed contracts, exact evaluation, packaging, profiles, and CLI routing. Standardize the algorithm-facing command surface under `src/tac/cli.py` while keeping repo-ops in `src/comma_lab/cli.py`. Lossless state remains separate from lossy state.

**Tech Stack:** Python stdlib, `argparse`, existing `tac` patterns, commavq evaluation contract, unittest.

---

### Task 1: Create lossless typed contracts and profile surfaces

**Files:**
- Create: `src/tac/lossless/__init__.py`
- Create: `src/tac/lossless/contracts.py`
- Create: `src/tac/lossless/profiles.py`
- Test: `experiments/test_tac_lossless_contracts.py`

- [ ] **Step 1: Write failing tests for lossless result/profile models**

Cover:
- compression result typing
- exact verification result typing
- named profile lookup

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest experiments.test_tac_lossless_contracts -v`
Expected: FAIL because the subsystem does not exist yet.

- [ ] **Step 3: Implement the minimal typed contracts**

Add explicit models for:
- compression rate result
- exact decompression verification
- failure signature
- promoted lossless result

- [ ] **Step 4: Implement minimal lossless profiles**

Add profiles:
- `lzma_baseline`
- `zpaq_baseline`
- `gpt_arithmetic_small`
- `gpt_arithmetic_large`
- `neural_codec_smoke`

- [ ] **Step 5: Run tests to verify green**

Run: `python3 -m unittest experiments.test_tac_lossless_contracts -v`
Expected: PASS.

### Task 2: Add exact commavq data and evaluation helpers

**Files:**
- Create: `src/tac/lossless/data.py`
- Create: `src/tac/lossless/evaluate.py`
- Test: `experiments/test_tac_lossless_evaluate.py`

- [ ] **Step 1: Write failing tests for exact data/eval helpers**

Cover:
- token ordering
- exact equality verification
- compression-rate calculation
- extracted bundle / decompressed directory handling

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest experiments.test_tac_lossless_evaluate -v`
Expected: FAIL.

- [ ] **Step 3: Implement minimal data loader and evaluator**

Implement:
- exact token compare
- archive-size based rate
- contract-compatible evaluation helper

- [ ] **Step 4: Run test to verify green**

Run: `python3 -m unittest experiments.test_tac_lossless_evaluate -v`
Expected: PASS.

### Task 3: Add lossless packaging helpers

**Files:**
- Create: `src/tac/lossless/submission.py`
- Test: `experiments/test_tac_lossless_submission.py`

- [ ] **Step 1: Write failing tests for submission packaging**

Cover:
- deterministic zip assembly
- required `decompress.py`
- challenge-compliant layout

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest experiments.test_tac_lossless_submission -v`
Expected: FAIL.

- [ ] **Step 3: Implement minimal packaging path**

Build exact submission zip helpers and preflight validation.

- [ ] **Step 4: Run test to verify green**

Run: `python3 -m unittest experiments.test_tac_lossless_submission -v`
Expected: PASS.

### Task 4: Add canonical `tac` CLI with lossy/lossless routing

**Files:**
- Create: `src/tac/cli.py`
- Test: `experiments/test_tac_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Cover:
- `tac lossy ...` routing shell
- `tac lossless evaluate ...`
- `tac lossless profiles`

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest experiments.test_tac_cli -v`
Expected: FAIL.

- [ ] **Step 3: Implement minimal `tac` CLI**

Use `argparse`.
Do not add a second legacy CLI.

- [ ] **Step 4: Run test to verify green**

Run: `python3 -m unittest experiments.test_tac_cli -v`
Expected: PASS.

### Task 5: Standardize current lossy entrypoint story

**Files:**
- Modify: `experiments/train_tac.py`
- Test: `experiments/test_train_tac_entrypoint.py`

- [ ] **Step 1: Write failing test for canonical train-tac routing**

Test that `train_tac.py` is only a thin shell over canonical `tac` CLI or shared `tac` entry logic.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest experiments.test_train_tac_entrypoint -v`
Expected: FAIL.

- [ ] **Step 3: Implement thin-shell standardization**

Make `train_tac.py` clearly secondary to canonical `tac` command routing.

- [ ] **Step 4: Run test to verify green**

Run: `python3 -m unittest experiments.test_train_tac_entrypoint -v`
Expected: PASS.

### Task 6: Add separate lossless repo state surfaces

**Files:**
- Create: `.omx/state/lossless_focus.md`
- Create: `.omx/state/lossless_next_experiments.md`
- Create: `.omx/research/lossless_findings.md`
- Create: `reports/lossless_results.jsonl`
- Create: `reports/lossless_timeline.jsonl`
- Create: `reports/lossless_latest.md`

- [ ] **Step 1: Create empty but structured lossless state surfaces**

Keep them clearly separate from lossy state.

- [ ] **Step 2: Add minimal documentation note**

Record that lossless and lossy promotion/state flows are intentionally separate.

### Task 7: Full verification

**Files:**
- No new files; verify all touched surfaces

- [ ] **Step 1: Run lossless/tac tests**

Run:
`python3 -m unittest experiments.test_tac_lossless_contracts experiments.test_tac_lossless_evaluate experiments.test_tac_lossless_submission experiments.test_tac_cli experiments.test_train_tac_entrypoint -v`

- [ ] **Step 2: Re-run Kaggle/tac cleanup tests**

Run:
`python3 -m unittest experiments.test_kaggle_bootstrap_template experiments.test_kaggle_kernel_builder experiments.test_build_kaggle_kernels experiments.test_build_kaggle_assets_dataset experiments.test_kaggle_output_ingest -v`

- [ ] **Step 3: Re-run trainer/tac tests**

Run:
`uv run --with torch --with numpy --with av --with safetensors --with einops --with timm --with segmentation-models-pytorch python -m unittest experiments.test_tac_entrypoints experiments.test_train_postfilter_dilated_h64 experiments.test_cloud_segnet_attack_h32_trainer -v`

- [ ] **Step 4: Run diff sanity check**

Run:
`git diff --check -- src/tac src/comma_lab experiments docs .omx reports`

- [ ] **Step 5: Summarize final structure**

Report:
- canonical `tac` CLI status
- `tac/lossless` module coverage
- remaining non-canonical entrypoints, if any
