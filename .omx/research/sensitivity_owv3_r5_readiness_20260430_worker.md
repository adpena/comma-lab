# Sensitivity / OWV3 R5 Readiness - 2026-04-30

Owner: Worker Sensitivity/OWV3.

This is a readiness/progress note, not a score ledger. No CUDA eval was run by
this patch and no OWV3 result is promoted here.

## Source Context

- OWV3 r4 exact CUDA/T4 packet scored
  `score_recomputed_from_components=1.0378905176070103`, but failed the
  predeclared SegNet relative gate: `0.00402120 / 0.00400656 = 1.003654`
  against cap `1.002`.
- Grand Council verdict keeps r4 non-promotable and requires either paired
  same-run PFP16 calibration on the r4 runner or SegNet-conservative R5 exact
  eval candidates. No retroactive gate relaxation is assumed.

## Patch Landed

Changed code paths:

- `experiments/profile_component_sensitivity.py`
- `experiments/sweep_owv3_byte_plan.py`
- `src/tac/tests/test_profile_component_sensitivity.py`
- `src/tac/tests/test_sweep_owv3_byte_plan.py`

Sensitivity producer improvements:

- Writes `perturbation_basis_v1.json` with deterministic atom IDs, selected
  channel order, epsilon ladder, split hash, absolute calibration/holdout pair
  IDs, sign convention, normalization, and input custody fields.
- Response curves now include raw Fisher-map predicted deltas,
  least-squares unit calibration to observed holdout component deltas, fitted
  absolute/relative error, Pearson/Spearman correlation, sign accuracy, and
  gate diagnostics.
- The profiler remains non-promotable because maps are still Fisher proxies,
  not official finite-difference component maps. The blocker is explicit and
  preserved.

OWV3 R5 selection improvement:

- `experiments/sweep_owv3_byte_plan.py` can now rank
  `r5_segnet_conservative_candidates` when passed an exact-evaluated failed
  reference candidate id.
- The selector keeps byte feasibility, requires fewer OWV2-low-bit channels
  than the failed reference, and ranks the smallest conservative move first.
- This is a candidate-selection aid only. It makes no score claim and requires
  paired same-run PFP16 calibration plus exact CUDA component gates.

## Fastest Next R5 Candidate Logic

For the known r4 candidate
`owv3_0018_bbr0p69_protect0p0014_aggr1em05`, the R5 policy should search
byte-feasible candidates with fewer OWV2-low-bit channels than r4 and avoid
larger codec perturbations unless exact diagnostics justify them.

The existing sweep packet already contains plausible neighbors around the r4
threshold. The local selector over the existing packet ranks the first
SegNet-conservative byte-feasible neighbor as:

```text
owv3_0047_bbr0p67_protect0p00135_aggr1em05
archive_bytes=686468
frontier_delta_bytes=-167
owv2_low_bit_channels=62
reduction_vs_r4=3
```

This is byte/candidate-selection evidence only, not a score claim. The next
dispatch should be generated from the sweep summary with:

```bash
.venv/bin/python experiments/sweep_owv3_byte_plan.py \
  --sensitivity-map experiments/results/lane_g_v3_owv3_fisher_lightning_20260430_codex_r2/owv3_sensitivity_map.pt \
  --output-dir experiments/results/lane_g_v3_owv3_r5_candidate_sweep_20260430 \
  --preset frontier \
  --archive-policy selected \
  --decode-verify selected \
  --r5-reference-candidate-id owv3_0018_bbr0p69_protect0p0014_aggr1em05
```

Then exact eval must run on the selected archive only after paired PFP16
calibration or an equivalent reviewed same-run reference is queued.

## Remaining Blockers

- No CUDA-authored `component_sensitivity_v1` promotion artifact exists.
- Current component profiler still emits Fisher-proxy maps and cannot assemble
  a promotable manifest.
- R5 still needs exact CUDA/T4 archive evaluation on exact bytes with the
  predeclared component gates and same-run PFP16 calibration if using paired
  calibration rationale.
- The local machine is not a CUDA scorer host, so no authoritative score or
  component-response run was possible in this patch.

## Worker 2 R5 Queue Packet

Local-only update; no scorer or remote eval was run.

- Added an explicit R5 eval-selection mode to
  `experiments/sweep_owv3_byte_plan.py` so the selected archive is the first
  SegNet-conservative neighbor, not the failed r4 byte-best reference.
- Hardened R5 selection to exclude non-promotable OWV3 plans such as
  diagnostic FP16 fallback.
- Generated the R5 packet at
  `experiments/results/lane_g_v3_owv3_r5_candidate_sweep_20260430_worker2/`.
- Added queue/runbook guidance at `docs/runbooks/owv3_r5_exact_eval_queue.md`.

Ranked R5 candidate set:

| rank | candidate_id | bytes | sha256 | frontier delta | OWV2-low channels |
|---:|---|---:|---|---:|---:|
| 1 | `owv3_0047_bbr0p67_protect0p00135_aggr1em05` | 686468 | `16ab95220c8add11b0bc40fb632bc8421f8bb8ad1cfba145f0b6058075237518` | -167 | 62 |
| 2 | `owv3_0062_bbr0p66_protect0p00135_aggr1em05` | 686190 | `6393607a2c7b517fe9415a254481a54f189c0bca0e25bec6bc3032dfc6eded39` | -445 | 62 |
| 3 | `owv3_0077_bbr0p65_protect0p00135_aggr1em05` | 686014 | `5355143f1d7bfdda8bf5a8160723ed305c4c8e1af7aa074d16e5365ac0e5d7f0` | -621 | 62 |
| 4 | `owv3_0092_bbr0p64_protect0p00135_aggr1em05` | 685487 | `862a429629b1ea754970507a78b20a839428dfe00185e574ce0cfd41510241d0` | -1148 | 62 |
| 5 | `owv3_0076_bbr0p65_protect0p0013_aggr1em05` | 686531 | `9f7528bade11bf9cdf3df68f8073d11f196a6d5f48475a8680c21fb58c878c91` | -104 | 58 |

R5 promotion remains blocked until:

1. PFP16 A++ is exact-evaled on the same CUDA/T4 runner.
2. R5 rank 1 is exact-evaled on the exact archive bytes.
3. `scripts/adjudicate_contest_auth_eval.py` passes with
   `--max-segnet-relative 1.002`, `--max-posenet-relative 1.002`,
   `--required-device cuda`, and `--required-samples 600`, using the paired
   PFP16 JSON component values as reference.
