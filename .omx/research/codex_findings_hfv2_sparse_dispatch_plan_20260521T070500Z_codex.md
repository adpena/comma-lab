# Codex Findings: HFV2 Sparse Sidecar Dispatch Plan

- timestamp_utc: 2026-05-21T07:05:00Z
- lane: hfv2_pair_sparse_pr101_hfv1_sidecar_exact_eval
- status: LANDED_CANONICAL_PAIRED_DISPATCH_PLAN
- score_claim: false
- promotion_eligible: false
- exact_eval_executed: false

## What changed

`tools/build_hfv1_sparse_sidecar_candidate.py` now embeds the canonical paired
Modal auth-eval command templates in every generated HFV2 sparse sidecar
manifest. The command path is `tools/dispatch_modal_paired_auth_eval.py`, not a
single-axis wrapper.

The artifact remains research-only until paired CPU/CUDA exact eval lands.

## Current artifact

- Output directory: `experiments/results/hfv2_sparse_sidecar_candidate_20260521T070412Z`
- Archive: `experiments/results/hfv2_sparse_sidecar_candidate_20260521T070412Z/archive.zip`
- Submission runtime: `experiments/results/hfv2_sparse_sidecar_candidate_20260521T070412Z/submission_dir_hfv2`
- Manifest: `experiments/results/hfv2_sparse_sidecar_candidate_20260521T070412Z/hfv2_sparse_manifest.json`
- Paired dispatch plan: `experiments/results/hfv2_sparse_sidecar_candidate_20260521T070412Z/paired_dispatch_plan.json`

Hashes:

```text
488f2e53d81d6442d189b4f882508af0d4184010ca67558e83bfadf822138ee2  archive.zip
eee1c0dd1677cfc21c45f9bf21468cc9d7d0ff65dc90ff4037f2007a7a012c4b  hfv2_sparse_manifest.json
3a5169abb2bac946055da637e4080e94c604e50fa9ed9943d09903760563377d  submission_dir_hfv2/archive_manifest.json
b301c1766db88a812708be8be40d05e0bcf943158f06f4660d46dce995753eb5  paired_dispatch_plan.json
```

## Plan verification

Plan-only dispatch command was run with `--json-out` and did not execute Modal:

```bash
.venv/bin/python tools/dispatch_modal_paired_auth_eval.py \
  --archive experiments/results/hfv2_sparse_sidecar_candidate_20260521T070412Z/submission_dir_hfv2/archive.zip \
  --submission-dir experiments/results/hfv2_sparse_sidecar_candidate_20260521T070412Z/submission_dir_hfv2 \
  --inflate-sh inflate.sh \
  --label hfv2_pair_sparse_pr101_hfv1_sidecar_exact_eval \
  --expected-archive-sha256 488f2e53d81d6442d189b4f882508af0d4184010ca67558e83bfadf822138ee2 \
  --run-id hfv2_pair_sparse_pr101_488f2e53d81d \
  --pair-group-id pair_hfv2_pair_sparse_pr101_hfv1_sidecar_exact_eval_488f2e53d81d \
  --lane-id-base hfv2_pair_sparse_pr101_hfv1_sidecar_exact_eval \
  --output-root experiments/results \
  --modal-bin .venv/bin/modal \
  --gpu T4 \
  --claim-agent codex:hfv2_sparse_sidecar \
  --claim-notes 'HFV2 pair-sparse sidecar candidate; score_claim=false until paired contest CPU/CUDA exact eval harvest.' \
  --expected-runtime-tree-sha256 auto \
  --skip-axis-if-promotable-anchor-exists \
  --json-out experiments/results/hfv2_sparse_sidecar_candidate_20260521T070412Z/paired_dispatch_plan.json
```

Plan result:

- archive SHA matched expected
- no existing promotable CPU anchor was found
- no existing promotable CUDA anchor was found
- CPU lane: `hfv2_pair_sparse_pr101_hfv1_sidecar_exact_eval_contest_cpu`
- CUDA lane: `hfv2_pair_sparse_pr101_hfv1_sidecar_exact_eval_contest_cuda`
- CPU runtime tree: `24693357fe6e62509b304417613955e1e2d0584cbfdf1c644af4400d56a86a2f`
- CUDA runtime tree: `7295c296ff0655c5f58d34d2ac13e5b882ed9a898466f1470ffc9a441b6c9b4c`
- runtime content tree both axes: `568d5e68459822bf2686688ce384bf512e1189b08b7027e492147f9f17077e33`

## Why not executed

I did not pass `--execute`. The dispatch surface still has active
Claude-owned DP1 claims:

```text
lane_dp1_original_baseline_first_paired_anchor_20260520
lane_dp1_procedural_codebook_replacement_first_paired_smoke_20260520
```

The HFV2 packet is launch-ready once that surface clears; executing now would
mix ownership and risk stepping on the partner Modal queue.

## Verification commands

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  tools/build_hfv1_sparse_sidecar_candidate.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py scan

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py mark-file \
  tools/build_hfv1_sparse_sidecar_candidate.py --status reviewed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py policy-check \
  tools/build_hfv1_sparse_sidecar_candidate.py
```

Review tracker result:

- `tools/build_hfv1_sparse_sidecar_candidate.py`: 21 entities reviewed
- policy: NORMAL, 21 entities compliant, 0 violations

## Next action

When the DP1 claims terminalize, run the generated execute command from
`hfv2_sparse_manifest.md`. The artifact only needs `0.000338256348186` component
gain over FEC6/PR110 to clear the CPU-axis rate hurdle.
