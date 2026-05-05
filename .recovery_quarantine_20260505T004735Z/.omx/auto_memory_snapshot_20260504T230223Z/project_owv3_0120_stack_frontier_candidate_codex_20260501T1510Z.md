# Codex Memory - OWV3 0120 Stack Frontier Candidate

Date: 2026-05-01T15:10Z

Repository: `/Users/adpena/Projects/pact`

Context:

- The project is pursuing the contest-compliant Shannon-floor frontier under
  AGENTS.md evidence rules.
- CUDA auth eval through `archive.zip -> inflate.sh -> upstream/evaluate.py`
  is the only score truth for GPU-dependent claims. CPU/MPS/proxy scores are
  non-promotable.
- Exact T4/equivalent confirmation is required before a frontier candidate can
  be treated as promotion-grade/A++.

Candidate:

- Lane/archive: `owv3_0120_stack`.
- Source eval: Vast RTX 4090 instance `35959478`, label
  `owv3_wave3_chain_v11_self_bootstrap`.
- Local harvest dir:
  `/Users/adpena/Projects/pact/experiments/results/vast_harvest/owv3_0120_stack_rtx4090_20260501T1501Z`.
- Archive:
  `/Users/adpena/Projects/pact/experiments/results/vast_harvest/owv3_0120_stack_rtx4090_20260501T1501Z/owv3_0120_stack_archive.zip`.
- Archive bytes: `609963`.
- Archive SHA-256:
  `1e9195cb6e0e08fc98ee393590770e2b22905a2ee2718edb8b737cada125f279`.
- RTX 4090 CUDA auth eval recomputed score: `0.997430122363832`.
- Components: PoseNet `0.00356167`, SegNet `0.00402557`, `600` samples.
- Current evidence grade: A score-grade only, pending T4/equivalent exact eval.

T4 promotion job:

- Wrapper:
  `scripts/lightning_exact_eval_repro.py`.
- Job name:
  `exact_eval_owv3_0120_stack_t4_20260501T150652Z`.
- Lightning SDK job name:
  `exact-eval-owv3-0120-stack-t4-20260501t150652z`.
- Local planned artifact dir:
  `/Users/adpena/Projects/pact/experiments/results/lightning_batch/exact_eval_owv3_0120_stack_t4_20260501T150652Z`.
- State record:
  `/Users/adpena/Projects/pact/.omx/state/exact_eval_owv3_0120_stack_t4_20260501T150652Z_lightning_batch_record.json`.
- Source manifest:
  `/Users/adpena/Projects/pact/.omx/state/exact_eval_owv3_0120_stack_t4_20260501T150652Z_manifest.json`.
- Local supply-chain scan:
  `/Users/adpena/Projects/pact/.omx/state/exact_eval_owv3_0120_stack_t4_20260501T150652Z_local_lightning_supply_chain_scan.json`.
- Baseline T4 reference:
  `/Users/adpena/Projects/pact/experiments/results/lightning_batch/exact_eval_owv3_0120_wave3_t4_20260501T130313Z/contest_auth_eval.json`,
  score `1.0021175309471926`, bytes `617410`, PoseNet `0.00356094`,
  SegNet `0.00402305`.
- First refresh status: `Pending` on Lightning `T4_SMALL`.
- Duplicate wall-clock hedge job:
  `exact_eval_owv3_0120_stack_t4aws_20260501T151050Z`.
- Duplicate SDK job name:
  `exact-eval-owv3-0120-stack-t4aws-20260501t151050z`.
- Duplicate machine class: Lightning `T4` via `g4dn.2xlarge`.
- Duplicate uses the same archive bytes/SHA and same adjudication gates, with
  `duplicate_of=exact_eval_owv3_0120_stack_t4_20260501T150652Z` in queue
  metadata. It is a wall-clock hedge only, not a separate score claim.

Compute hygiene:

- Vast H100 instance `35961748` / `arith_coding_h100` was destroyed after live
  API and SSH inspection showed idle GPU, no `/workspace/pact`, no tmux, no
  active payload, and effectively empty disk.
- Audit record:
  `/Users/adpena/Projects/pact/.omx/state/vast_destroy_35961748_20260501T1507Z.json`.

Next:

1. Refresh Lightning status for the T4 job until completed.
2. Harvest with `scripts/launch_lightning_batch_job.py harvest-ssh` using the
   state-derived path; do not hand-compose artifact paths.
3. Validate archive SHA/bytes, `contest_auth_eval.json`,
   `contest_auth_eval.adjudicated.json`, runner preflight, and supply-chain
   scans.
4. If T4 confirms the sub-1.0 recomputed score, update the claim matrix and
   progress doc as current exact frontier. If it diverges, preserve all
   artifacts and run mismatch forensics before any lane conclusion.
