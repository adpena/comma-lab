# Cloud GPU Spend Plan

This plan outlines how to allocate cloud GPU resources (T4, 4090, A100) for
mid‑term and long‑term experiments while controlling costs and ensuring that
each experiment yields actionable evidence.  The goal is to minimise wasted
GPU hours by staging cheap smoke tests before full‑scale training.

## Hardware tiers and recommendations

| Tier | Examples | Typical cost/hour | Suitable workloads | Comments |
|---|---|---|---|---|
| **Tier 1: T4** | NVIDIA Tesla T4 in cloud providers (Modal, AWS g4, Vast.ai)| $0.35–$0.45 | Small HNeRV/HiNeRV/E‑NeRV smokes; FEC7 training; foveation prototypes; predictor calibration runs | Good balance of cost and memory (~16 GB).  Suitable for small models and short runs (1–6 h). |
| **Tier 2: 4090/4080** | Consumer GPUs on Vast.ai; A10G | $0.60–$0.80 | SIREN and VQ‑VAE smokes; RAFT pose priors; moderate‑size MAE pretraining | More memory (24 GB) and faster compute.  Ideal for experiments that need to train moderate CNNs but not huge models. |
| **Tier 3: A100/A40** | Data‑centre GPUs on Modal/Azure | $1.50–$3.00 | Full runs of promising candidates; large neural codecs; long‑epoch MAE or V‑JEPA training | Use only after a candidate passes smoke tests and predictor suggests high potential. |

## Spend phases

1. **Validation phase (week 1):** Use Tier 1 (T4) instances for all smoke
   tests (M1–M8).  Cap each run at 4 hours.  Budget ≈ $200.

2. **Exploration phase (weeks 2–3):** For candidates that pass smoke tests,
   allocate up to 20 hours on Tier 2 hardware to train on several videos and
   measure scaling behavior.  Budget ≈ $800.  Prioritise candidates with
   predicted score improvements >0.0005.

3. **Full‑scale phase (week 4+):** Reserve Tier 3 hardware for one or two
   candidates that show clear promise (e.g. SIREN or VQ‑VAE with strong
   smokes).  Limit runs to 48–72 hours.  Collect thorough logs, intermediate
   checkpoints and evaluation results.  Budget ≈ $2 000.

## Cost control rules

1. **Set per‑run cost caps:** Use the `--max-hours` flag or provider limits
   to enforce a maximum runtime.  Abort jobs that exceed budget.
2. **Use spot/preemptible instances:** For non‑critical smokes, rent
   spot/interruptible GPUs at lower prices.  Save checkpoints frequently to
   resume if interrupted.
3. **Batch experiments:** Run multiple smokes sequentially on a single GPU
   instance to amortise setup time.  Use job arrays or simple queue scripts.
4. **Monitor GPU utilisation:** Use `nvidia-smi` and logging to ensure that
   GPUs are not idle.  Adjust batch size or workload to maximise usage.
5. **Record cost vs. benefit:** For each experiment, record the dollar cost,
   score improvement, and bytes saved.  Prioritise future work based on
   improvements per dollar.

## Evidence harvesting

* **Store evaluation outputs:** Collect `contest_auth_eval.json`, `report.txt`
  and logs for every run.  Record the commit hashes of both the candidate
  code and the upstream evaluator.
* **Capture intermediate checkpoints:** For long trainings, save checkpoints at
  regular intervals (e.g. every 4 hours).  These can be used to diagnose
  training collapse or to fine‑tune hyperparameters without starting from
  scratch.
* **Document hyperparameters:** Store all hyperparameters (learning rate,
  hidden size, number of modes, etc.) in a structured YAML or JSON file that
  is hashed and committed.  This ensures that each run can be exactly
  reproduced.

## Escalation criteria

* **From Tier 1 to Tier 2:** Candidate passes smoke tests with ≥0.0003 score
  improvement on CPU/CUDA and sidecar byte cost ≤500 bytes.

* **From Tier 2 to Tier 3:** Candidate demonstrates ≥0.0008 score improvement
  on multiple videos and evidence suggests scalability.  Predictor must
  estimate that full‑dataset training has a >50% chance of surpassing the
  current baseline by ≥0.001.  A written justification should be prepared.

* **Terminate early:** If during a Tier 3 run the score stagnates or degrades
  after the first 24 hours, abort and revert to exploring other candidates.
