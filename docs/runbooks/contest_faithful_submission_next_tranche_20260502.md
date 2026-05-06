# Contest-Faithful Submission Next Tranche - 2026-05-02

Scope: local docs, packet automation, and deterministic artifact review only.
Do not dispatch remote or GPU jobs from this tranche.

## Active Exact Frontier

- Evidence: `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/contest_auth_eval.adjudicated.json`
- Grade: `A++` field-supported exact Tesla T4 CUDA, 600 samples.
- Score field: `0.31561703078448233`.
- Archive bytes: `276214`.
- Archive SHA-256: `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`.
- Packet policy: submission packets record custody metadata only; `score_claim=false`, `ranking_claim=false`, and `promotion_claim=false`.

## Non-Score Supporting Inputs

These artifacts may guide implementation order and writeup structure. They are
not score authorities and must not appear in frontier/ranking rows.

- Learned optimizer feedback: use as proposal routing for charged archive atoms only.
- `MM-GUMBEL-KNAPSACK`: planning/proxy until a deterministic archive consumes selected atoms and passes local smoke plus exact CUDA auth eval.
- `PMG-HOTSPOT`: planning/proxy until the decoder/runtime, archive wrapper, validator, and custody packet exist.
- `QBF1-v2`: local byte win and geometry/runtime closure first; no exact eval request from byte-negative or no-op QBF1-v1 evidence.
- Yousfi-Fridrich observability visualizations: report-audit aids only.
- C-067 byte-accounting markdown/PNG profiles:
  `experiments/results/c067_archive_byte_accounting_20260502/archive_byte_accounting.md`
  and
  `experiments/results/c067_self_compression_profile_20260502/archive_byte_accounting.png`.
  They show a `23454` byte unchanged-distortion gap to sub-`0.300` and the
  stream split `219472/55965/677`, but they are `empirical` observability
  artifacts only.
- PMG atomtop4068 exact L40S result:
  `experiments/results/lightning_batch/exact_eval_pmg_hotspot_atomtop4068_l40sdiag_20260502T1445Z/contest_auth_eval.json`.
  Treat as `A-negative scoped forensic`: score `28.41411894150047`, PoseNet
  `62.34251404`, archive `195762` bytes. It blocks PMG row-run-only T4
  promotion, not learned mask grammar or atom planning.
- SJ-KL tensor-prep manifests:
  `experiments/results/sjkl_tensor_prep_c067_smoke_20260502/sjkl_pair_tensor_prep_manifest.json`
  and
  `experiments/results/sjkl_tensor_prep_c067_full_20260502/sjkl_pair_tensor_prep_manifest.json`.
  Both are `build_tensor_prep_only`, `score_claim=false`, and
  `promotion_eligible=false`; the full manifest is the next local build input.

## Local Automation Order

1. Build or refresh candidate metadata locally only.
2. Validate deterministic archive custody, hidden-file exclusion, zip-slip closure, and no sidecar dependencies.
3. Generate a metadata-only submission packet with `scripts/build_contest_submission_packet.py`, using the adjudicated exact eval JSON as the score authority.
4. Record planner ledgers, visualizations, and this next-action tranche as non-score supporting artifacts in the packet manifest.
5. Keep exact-score wording pinned to the active exact CUDA JSON until a new exact CUDA auth eval JSON exists.
6. For SJ-KL, build `sjkl.bin` from the full `600`-pair tensor-prep manifest,
   package it as a charged archive member, then run local decode parity and
   validator checks before any dispatch claim.

## Promotion Blockers

- No new sub-`0.30` score exists in this tranche.
- Planner ledgers and visualizations are not score evidence.
- H100/L40S diagnostics are not T4/equivalent promotion evidence unless explicitly adjudicated under the project rules.
- PMG row-run-only rescue is blocked by the atomtop4068 PoseNet collapse; do
  not dispatch another row-run-only PMG T4 job.
- SJ-KL has runtime/tensor-prep readiness only. It cannot claim a score until
  the residual payload is inside a deterministic archive with SHA/bytes,
  runtime tree hash, target-slot custody, and exact CUDA auth eval.
- Remote/GPU dispatch is outside this ownership slice.
