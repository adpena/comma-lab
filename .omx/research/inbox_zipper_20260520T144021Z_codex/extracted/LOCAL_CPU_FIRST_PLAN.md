# Local CPU‑First Plan

This document outlines how to maximise progress using only local macOS CPU (and
optionally MPS) before renting any cloud GPU resources.  Running these tasks
first ensures that the heavy GPU experiments will be well‑informed and
efficient.

## 1. Environment setup and reproducibility

1. **Clone repositories:**
   ```bash
   git clone https://github.com/commaai/comma_video_compression_challenge.git
   git clone https://github.com/adpena/comma-lab.git
   git clone https://github.com/adpena/tac.git
   ```
2. **Create a Python virtual environment** and install dependencies:
   ```bash
   python3 -m venv venv && source venv/bin/activate
   pip install torch numpy brotli
   pip install -e tac[mlx,viz,notebooks]
   ```
3. **Run unit tests** in `tac` to ensure the local installation is healthy:
   ```bash
   cd tac && pytest
   ```
4. **Check reproducibility of PR #110** by running the commands in
   `EXPERIMENT_QUEUE.json` (`exp_s1_verify_pr110`).  Verify that the SHA and
   scores match the claims【294212394795766†L235-L244】【294212394795766†L354-L357】.

## 2. Static audits

1. **Read and classify the full‑stack source map:** Use the table in
   `comma-lab/docs/full_stack_source_map.md` to build a spreadsheet of
   candidate families, their status (EMPIRICAL/SCAFFOLDED/...), and local
   surfaces【58280996536521†L18-L32】.  This will guide which modules to
   explore first.
2. **Review the `tac` README and examples:** Understand the meta search
   engine, predictor and preflight checks【635462165059268†L118-L142】.  Run the
   example in `tac/examples/quickstart.py` on CPU to see how the loop
   orchestrates candidate ranking.
3. **Inspect `frame_selector.py` and `model.py`:** Read the FEC6 selector and
   HNeRV decoder code to understand how transforms are selected and encoded.
   Confirm that the implementation matches the description in PR #110 and
   note any code quality issues.
4. **Check compliance scripts:** Identify any pre‑submission or manifest
   scripts in `comma-lab` that check archive completeness.  Run them on the
   PR #110 archive to ensure there are no hidden sidecars.

## 3. Byte profiling

1. **Write a byte profiling tool:** Create a script that perturbs bits in the
   `archive.zip` member `x` and measures the effect on the score by running
   `upstream/evaluate.py` on CPU.  Use finite differences: flip one bit,
   recompute score, record the delta.  Repeat for a subset of bits.
2. **Visualise sensitivity:** Aggregate per‑byte sensitivity into a heatmap
   (e.g. using matplotlib).  Identify which frames and selector bits dominate
   the distortion metrics.  This will inform whether adding more selector
   modes (FEC7/FEC8) is worth the extra bytes.

## 4. Deterministic export tooling

1. **Packet compiler:** Write a simple Python script that accepts a source
   payload file, a selector sidecar and metadata (e.g. compression method,
   member names) and produces a zip file with deterministic ordering and
   `ZIP_STORED` compression【58280996536521†L200-L206】.
2. **Manifest generator:** Implement a manifest generator that records the
   archive SHA‑256, member names, sizes, and the versions of `torch`,
   `numpy` and `brotli` used.  Store this manifest alongside each archive.
3. **Preflight checks:** Use or extend `tac.preflight` to perform
   sanity checks on candidate archives (e.g. no network calls, no extraneous
   files, correct compression method).

## 5. Predictor calibration

1. **Collect anchors:** As you reproduce PR #95–#110 and any local smokes,
   record their archive sizes, segmentation distortions, pose distortions and
   total scores.  These form the anchor set for the predictor.
2. **Fit the predictor:** Use `tac.predictor.score_band` to fit the
   calibration parameters on the anchor set.  Evaluate the predictor on a
   held‑out subset to measure prediction error and refusal rates【635462165059268†L118-L142】.
3. **Tune refusal thresholds:** Adjust thresholds for `insufficient_anchors`
   and `extrapolation` to achieve a conservative predictor that rarely
   underestimates scores.  This will help prioritise which candidates to send
   to GPU.

## 6. Implementation exercises (no GPU)

1. **Prototype FEC7:** Modify `frame_selector.py` to support a larger
   transform palette (e.g. 63 modes with 32 active).  Write an offline search
   loop that selects per‑frame transforms based on a local distortion proxy.
   Use the packet compiler to build a new archive and evaluate on CPU.  Use
   this to decide whether to pursue a GPU training for FEC7.
2. **Small E‑NeRV train:** Use a small CPU‑friendly HNeRV variant (e.g. tiny
   hidden dimension) to ensure the training loop works and exports bytes.
   Although this will not produce contest‑quality results, it uncovers
   implementation issues before larger GPU runs.
3. **Readiness of foveation masks:** Implement basic foveation fields
   (e.g. radial falloff around the vanishing point) and integrate them into
   the decoder’s forward pass.  Confirm that the code runs on CPU and that
   you can export variable‑resolution frames.

Completing these local tasks will produce baseline metrics, diagnostic tools
and deterministic export mechanisms that underpin all subsequent GPU‑heavy
experiments.  Only after this foundation is solid should you proceed to the
parallel execution plan and the cloud GPU spend outlined there.
