# Research Package Overview

This directory contains the deliverables requested for a comprehensive
review of PR #110 and a research roadmap for the `comma_video_compression_challenge`.
All files are plain text (Markdown, YAML or JSON) except the
implementation scaffolds, which are Python pseudocode.  Use these
documents and scripts to guide further development and to provide
evidence to contest maintainers and collaborators.

## Structure

* **PR110_EXECUTIVE_REVIEW.md** – High‑level verdict and recommended edits for the PR body.
* **PR110_FACT_AUDIT_TABLE.md** – Table mapping each PR claim to verification method and recommendations.
* **PR110_PATCH_SUGGESTIONS.md** – Patch‑ready language to tighten the PR description.
* **PR110_MAINTAINER_READER_MODEL.md** – Model of how maintainers view the PR and how to reduce friction.
* **FRONTIER_STAIRCASE.md** – A stepwise research roadmap divided into short‑term, mid‑term and long‑term tasks.
* **FRONTIER_GRAPH.yaml** – Machine‑readable graph of candidate lanes with dependencies and gating criteria.
* **EXPERIMENT_QUEUE.json** – Queue of experiments with commands, resources, expected outputs and success/failure criteria.
* **PARALLEL_EXECUTION_PLAN.md** – Guidance on running experiments in parallel without conflicts.
* **LOCAL_CPU_FIRST_PLAN.md** – Tasks to run on local CPU/MPS before any GPU spend.
* **CLOUD_GPU_SPEND_PLAN.md** – Recommendations for allocating T4/4090/A100 GPU hours and controlling costs.
* **THEORETICAL_FLOOR_MEMO.md** – Discussion of the contest’s theoretical score floor and where uncertainty lies.
* **IMPLEMENTATION_SCAFFOLD/** – Directory containing pseudocode for several promising candidate codecs and tools:
  * `fec7_selector.py` – 63‑mode FEC7 selector with offline search and fixed Huffman code.
  * `siren_inr_codec.py` – SIREN‑based implicit neural representation (INR) codec skeleton.
  * `vqvae_codec.py` – Minimal VQ‑VAE codec with training and token export functions.
  * `foveation_field.py` – Generation of foveation masks and training of foveated decoders.
  * `raft_pose_prior.py` – Integration of RAFT/LA‑Pose pose priors into decoder training.

## Recommended execution order

1. **Read the executive review and fact audit** to understand the state of PR #110 and apply any suggested edits.
2. **Set up your environment** as described in `LOCAL_CPU_FIRST_PLAN.md`.  Clone the necessary repositories and install dependencies.
3. **Reproduce PR #110’s scores** using the commands in the first experiment of `EXPERIMENT_QUEUE.json` (`exp_s1_verify_pr110`).  Verify that your local results match the claimed score.
4. **Build the deterministic packet compiler** (S2) and confirm that it reproduces the official archive bytes.  This tool will be reused for all subsequent experiments.
5. **Audit the HNeRV variants** by recreating the table of baseline scores (S3).  This informs whether new selectors or architectures make a real difference.
6. **Perform byte profiling** (S4) to understand which bits matter most.  The provided pseudocode does not implement this, but the concept is described in `LOCAL_CPU_FIRST_PLAN.md`.
7. **Review the frontier staircase** and choose mid‑term experiments.  The YAML graph and experiment queue define dependencies and resource requirements.
8. **Implement and run small smokes** for promising candidates using the pseudocode in `IMPLEMENTATION_SCAFFOLD/`.  For example, test FEC7 with `fec7_selector.py` and a deterministic compiler.  Evaluate results against promotion gates.
9. **Follow the cloud GPU spend plan** when scaling up experiments.  Start with T4 instances for smokes and only move to 4090 or A100 once a candidate shows meaningful improvements.
10. **Iteratively update the experiment queue and graph** as evidence accumulates.  Document all runs, commit hashes, and evaluation outputs for reproducibility.

## Next steps for coding agents

When handing these materials to other coding agents (e.g. Codex or Claude), provide them with:

1. The **experiment queue** for immediate tasks and corresponding scripts.
2. The **packet compiler scaffold** and instructions to integrate it with candidate models.
3. The **implementation scaffolds** for FEC7, SIREN, VQ‑VAE, foveation, and pose priors.  Ask the agents to flesh out these pseudocode files into working modules that interface with `comma-lab` and `tac`.
4. The **frontier graph** to prioritise which experiments to run in parallel and which to postpone.

By following this roadmap and using the included scaffolds, developers can systematically explore the space of task‑aware video compression techniques and approach the contest’s theoretical floor.
