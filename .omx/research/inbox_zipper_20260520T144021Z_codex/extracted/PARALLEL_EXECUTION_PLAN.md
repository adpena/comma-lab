# Parallel Execution Plan

Efficient use of local and cloud resources requires orchestrating many
experiments in parallel while respecting dependencies and avoiding duplicate
work.  This plan identifies which tasks can proceed concurrently and which
should wait for prerequisite artifacts.

## Core principles

1. **Separate CPU‑bound setup from GPU‑bound training.**  Use idle CPU cores to
   build tooling (packet compiler, predictor calibration) while GPU tasks run.

2. **Batch small GPU smokes.**  For early‑stage candidates, queue several
   short runs on a single GPU instance; release the instance once all runs
   complete.

3. **Reuse artifacts across experiments.**  The deterministic packet compiler,
   baseline scores and predictor anchors should be produced once and reused.

4. **Gate full runs.**  Only launch long GPU trainings after a candidate has
   passed its smoke tests and the expected improvement meets the promotion gate.

## Parallelizable tasks

The following tasks can run concurrently once the environment is prepared:

* **Baseline verification (S1) and packet compiler build (S2):** These are
  CPU‑only.  They can be executed in parallel with any GPU training.

* **HNeRV variant sweep (S3) and byte profiling (S4):** Each archive can be
  evaluated independently on separate CPU or GPU cores.  Use a thread pool or
  job queue to dispatch them concurrently.  Ensure that only one run writes
  to `contest_auth_eval.json` at a time.

* **Selector extensions (M2), E‑NeRV (M1) and SIREN/VQ‑VAE (M3/M4):** After
  the packet compiler is ready, these candidates can train and export
  concurrently on different GPU instances.  For example, run FEC7 on a T4
  while SIREN trains on a 4090.  Use distinct output directories to avoid
  collisions.

* **Foveation (M6) and RAFT pose prior (M8):** These involve different
  components and can run in parallel on separate GPUs.  Coordinate the
  queue so that only one experiment uses the same GPU at a time.

* **Predictor tuning (M7):** This is CPU‑bound and can run during any GPU
  training.  As soon as new evaluation results arrive, append them to the
  anchor set and re‑fit the predictor.

## Blocking tasks

Some tasks depend on previous results and should wait for them:

1. **FEC7 smoke (M2) depends on the packet compiler (S2).**  Without a
   deterministic compiler, comparing sidecar sizes is meaningless.

2. **HNeRV variant sweep (S3) should complete before RAFT and foveation
   experiments (M8, M6).**  The sweep establishes the baseline and may
   reveal that some videos are more sensitive to pose or foveation changes.

3. **Predictor tuning (M7) requires anchors from S1–S3.**  Tune after a
   representative set of candidate scores is available.

4. **Composition experiments (L6) require at least two successful candidate
   lanes.**  Do not start them until mid‑term experiments yield positive
   results.

## Artifact production sequence

1. **Environment setup and repository checkout (S6).**  Must occur first.

2. **Packet compiler and baseline scores (S1–S2).**  Generate once; publish SHA
   and manifest.

3. **HNeRV sweep and predictor anchors (S3, M7).**  Use to calibrate the
   predictor and to measure improvements.

4. **Byte profiling (S4).**  Use to inform the design of selectors and
   foveation masks.

5. **Run mid‑term candidate smokes (M1–M4, M6, M8).**  Distribute across GPUs.

6. **Collect results and update predictor.**  After each batch of smokes,
   append new anchors and re‑run the predictor calibration.

## Avoiding duplicate work

* **Version control:** Commit the packet compiler, training scripts and
  configuration files to a repository with clear version tags.  Never
  manually copy scripts between experiments.

* **Central results registry:** Keep a central `results.csv` or JSONL file
  recording each experiment’s configuration, random seed, and measured
  scores.  Use this registry to detect duplicate runs.

* **Configuration hashes:** For each experiment, compute a hash of the
  configuration (model type, hyperparameters, random seed).  Skip running an
  experiment if a matching hash is found in the registry.
