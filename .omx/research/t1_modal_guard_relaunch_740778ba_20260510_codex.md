# T1 Modal guard relaunch after canonical A1 payload mount fix — Codex ledger

Date: 2026-05-10

## Dispatch

- Lane: `t1_balle_128k_endtoend`
- Label / instance job id: `t1_balle_modal_guard_740778ba_20260510T1000Z`
- Modal app: `comma-t1-balle-endtoend`
- Modal run URL: `https://modal.com/apps/adpena/main/ap-DFruX1atbAs8UnQYECRrOx`
- Function call id: `fc-01KR8HFZPJRQHWTXAX7TT72D04`
- Local committed code: `740778ba harden score evidence and T1 dispatch gates`
- Estimated cost cap: `$80.00`; plan estimate: `$14.16` for the 24h T4 function budget.
- Claim state: `active_dispatching` recorded by `tools/claim_lane_dispatch.py`.

## Why this relaunch exists

The previous T1 Modal guard reached the remote worker but failed before training:
`FrozenA1EncoderError` reported that `/workspace/pact/experiments/results/A1_canonical`
was missing. Commit `740778ba` fixed the Modal actuator so the canonical A1
directory and `.omx/state/canonical_a1_designation.md` are mounted into the
remote runtime and validated in the local plan before GPU dispatch.

## Canonical A1 payload mounted

- Archive: `experiments/results/A1_canonical/harvested_artifacts/finetuned_archive/archive.zip`
  - bytes: `178262`
  - SHA-256: `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`
- Checkpoint: `experiments/results/A1_canonical/harvested_artifacts/train/checkpoint_best_proxy.pt`
  - bytes: `925250`
  - SHA-256: `5846043656d8261855d58de6d9c3568c7b2c4ecabdc3b4b1729aef913f7cb272`
- Extracted latents: `experiments/results/A1_canonical/harvested_artifacts/extracted_frozen_latents.pt`
  - bytes: `69088`
  - SHA-256: `5ba13604837e27b834867d2ace06d7c21228e8b97d241da6744020bee9f79090`
- Designation memo: `.omx/state/canonical_a1_designation.md`
  - bytes: `3617`
  - SHA-256: `064842006b08e5d7d0527bfa3dc06c112ed3de97d423a1fd43f25a6169c2ea45`

## Recovery

Initial recovery attempt at dispatch time returned:

```text
NOT READY: call_id=fc-01KR8HFZPJRQHWTXAX7TT72D04 still queued or running. Re-run later.
```

Run:

```bash
.venv/bin/python experiments/modal_t1_balle_endtoend.py recover --label t1_balle_modal_guard_740778ba_20260510T1000Z
```

Score status: no score claim. Promotion status: no promotion eligibility. This
dispatch is only score evidence after recovery returns exact contest-CUDA
auth-eval JSON with `auth_eval_schema` blockers equal to zero and closes the
active claim with a terminal row.
