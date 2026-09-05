# Runbook — ANE screening backend (`ddm_ane1`, 2026-09-05)

The Apple Neural Engine is fp16-only. Under the law CLAUDE.md applies to MPS it is a **screening**
device and never a score: it may rank, and every pair it adopts is re-measured on `cpu_torch` fp32
before any number leaves an instrument. Contract + tests: `src/tac/ane_screening.py`,
`src/tac/tests/test_ane_screening.py`. Measured verdicts:
`.omx/research/ddm_ane1_ane_screening_lane_20260905.md`.

**Read the verdict before you reach for the ANE.** On the pose axis the fp16 screen ranks at chance
(argmin agreement 4/39 = 10.26%) and its picks are worse than shipped on 34 of 39 pairs. The
acceleration that IS admissible is `coreml_cpu_fp32`: bit-exact SegNet argmax over 600 pairs at
3.28×, and 3.1e-07 of d_pose at 5.12×.

Everything below runs under `.venv_executorch_spike` (coremltools 9.0, torch 2.12.0, Brotli 1.2.0).
The shared `.venv` carries no coremltools and must not be changed to.

```bash
OUT=/Volumes/VertigoDataTier/pact/ddm_ane1_ane_screening

# 1. convert the frozen scorers (upstream stays READ-ONLY; copies are traced in memory)
PYTHONPATH=src:upstream .venv_executorch_spike/bin/python \
  experiments/ddm_ane1_ane_screening.py convert \
  --out-dir "$OUT/mlpackages" --model both --precision both --batch 1 \
  --out "$OUT/convert_manifest.json"

# 2. PROVE placement per-op (MLComputePlan) and time the triad
PYTHONPATH=src:upstream .venv_executorch_spike/bin/python \
  experiments/ddm_ane1_ane_screening.py placement \
  --manifest "$OUT/convert_manifest.json" --reps 30 --out "$OUT/placement_v2.json"

# 3. n600 drift on REAL frames (the shipped body's own decode)
PYTHONPATH=src:upstream .venv_executorch_spike/bin/python \
  experiments/ddm_ane1_ane_screening.py fidelity \
  --manifest "$OUT/convert_manifest.json" \
  --raw /Volumes/APDataStore/pact/ddm_to1/advisory/attempt_0002/work/inflated/0.raw \
  --pairs 600 --model both --threads 1 --out "$OUT/fidelity/fidelity_n600.json"

# 4. all-pixel SegNet top-2 margin census, then price the exact hybrid (never built)
PYTHONPATH=src:upstream .venv_executorch_spike/bin/python \
  experiments/ddm_ane1_ane_screening.py margins \
  --raw /Volumes/APDataStore/pact/ddm_to1/advisory/attempt_0002/work/inflated/0.raw \
  --pairs 600 --threads 1 --out "$OUT/fidelity/margin_census_n600.json"
PYTHONPATH=src .venv_executorch_spike/bin/python \
  experiments/ddm_ane1_ane_screening.py price \
  --fidelity "$OUT/fidelity/fidelity_n600.json" \
  --placement "$OUT/placement_v2.json" \
  --margin-census "$OUT/fidelity/margin_census_n600.json" \
  --out "$OUT/hybrid_price.json"

# 5. replay pr1's 39-point selector sweep on a backend
bash "$OUT/replay/run_selector_replay.sh" validate   # all 8 modes confirmed -> rank agreement
bash "$OUT/replay/run_selector_replay.sh" screen     # 2 confirms/pair -> the production scheme
bash "$OUT/replay/run_selector_replay.sh" cpu        # cpu_torch control
```

Launch anything above three minutes through `tools/launch_detached_process.py` with a MEASURED
`--measured-peak-rss-gib` (this lane measured 1.32 GiB for `convert`, 1.28 GiB for `fidelity`).

## The two flags, and what they mean

* `ddm_pr1 ... selector --scorer-backend {cpu_torch,coreml_cpu_fp32,ane_fp16_screen}
  [--pose-mlpackage PATH] [--confirm-all-modes]` — the backend RANKS the 8 selector modes. The two
  values that leave the sweep (d_pose at the shipped mode and at the chosen mode) are always
  re-measured on `cpu_torch` fp32. `--confirm-all-modes` re-measures all 8 instead: it buys no
  wall-clock and is the experiment that measures whether the screen's ranking agrees with the
  authority. Without it the report says so, and reports
  `screened_picks_that_survive_confirmation` rather than a rank agreement it cannot compute.
* `ddm_fs1 ... measure --scorer-backend ...` — this path emits the PRICED d_pose, so it **refuses**
  every non-authority backend before it measures anything. The flag exists so the receipt states the
  backend instead of implying it.

## Budgeting a speedup — read this before you plan one

The ANE trunk is 63.8× (SegNet) and 74.1× (PoseNet) faster than 1-thread CPU-torch. **That is not
the instrument's speedup.** In pr1's selector sweep, 76.82 ms of every 160.28 ms forward is
`render_frame0` + `preprocess_input`, which stay in torch. Measured end-to-end: 2.06× per forward,
and 1.36× for the production 8-screen/2-confirm scheme. Amdahl, measured inside one run — budget
from the instrument, never from the trunk.
