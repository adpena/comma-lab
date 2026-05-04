# PR95 HNeRV Residual Atom Planning - 2026-05-04

## Scope

Local-only worker slice. No remote GPU dispatch. Owned additive files only:

- `experiments/build_pr95_hnerv_residual_atom_plan.py`
- `src/tac/tests/test_build_pr95_hnerv_residual_atom_plan.py`
- `experiments/results/pr95_hnerv_residual_atom_plan_20260504_codex/`

The purpose is to convert the confirmed PR95 96-byte repack frontier into a
hard-pair / latent-residual planning surface without risking PR95's rate
advantage through arbitrary runtime edits.

## Exact Baseline

Evidence input:

- `experiments/results/lightning_batch/exact_eval_pr95_hnerv_muon_repacked_t4_fix2_20260504T0848Z/contest_auth_eval.adjudicated.json`

Confirmed baseline used by the tool:

- score: `0.23091954465634829`
- archive bytes: `178321`
- archive SHA-256: `2b9b471358f5bba97ea809dcca544ecca26b504fde770a9002046830f469368b`
- avg PoseNet distance: `0.00017185`
- avg SegNet distance: `0.00070728`
- runtime tree SHA-256: `a3f8ab2cfbbdfab53a6d437b0c39f525e0adde2d8bd971765de96aeda4da3dc7`
- evidence grade: A++ T4 exact eval

## Artifact Outputs

Generated with:

```bash
.venv/bin/python experiments/build_pr95_hnerv_residual_atom_plan.py \
  --archive experiments/results/pr95_hnerv_muon_packing_profile_20260504_codex/archive.pr95_repacked.zip \
  --exact-json experiments/results/lightning_batch/exact_eval_pr95_hnerv_muon_repacked_t4_fix2_20260504T0848Z/contest_auth_eval.adjudicated.json \
  --output-dir experiments/results/pr95_hnerv_residual_atom_plan_20260504_codex \
  --top-k 80 \
  --stdout
```

Outputs:

- `experiments/results/pr95_hnerv_residual_atom_plan_20260504_codex/pr95_hnerv_residual_atom_manifest.json`
- `experiments/results/pr95_hnerv_residual_atom_plan_20260504_codex/pr95_hnerv_pair_opportunity_ledger.json`
- `experiments/results/pr95_hnerv_residual_atom_plan_20260504_codex/pr95_hnerv_residual_atom_plan.proxy.json`

The archive structure parsed from the PR95 repack:

- member: `0.bin`
- member bytes: `178213`
- member SHA-256: `d9fa160d366d0aed105d14458e289f88ec5b71ec4d5de318ddb2ec0d44b50bf5`
- meta brotli bytes: `68`
- decoder brotli bytes: `162265`
- latent brotli bytes: `15868`
- latent raw bytes: `33720`
- latent pairs: `600`
- latent dim: `28`

## Current Signal

No PR95 per-pair component trace is present in the harvested T4 artifact, so the
ledger is intentionally marked `latent_proxy_only`. It is useful for choosing
where to run component trace and optimizer sweeps, but it is not promotable
score evidence.

The top proxy-ranked pairs after removing the pair-0 initialization artifact:

| rank | pair | top latent dims | proxy signal | estimated min atom bytes | score break-even |
|---:|---:|---|---:|---:|---:|
| 1 | 517 | 13,24,26,19 | `410.7646300852321` | 16 | `0.000010653743249954742` |
| 2 | 222 | 12,15,14,0 | `398.4715940215767` | 16 | `0.000010653743249954742` |
| 3 | 133 | 26,22,21,4 | `390.4683082873793` | 16 | `0.000010653743249954742` |
| 4 | 232 | 22,17,15,4 | `387.6649661984598` | 16 | `0.000010653743249954742` |
| 5 | 523 | 17,11,0,12 | `387.3367272235565` | 16 | `0.000010653743249954742` |
| 6 | 515 | 11,3,24,9 | `385.4897245938309` | 16 | `0.000010653743249954742` |
| 7 | 216 | 18,2,7,25 | `381.979685805574` | 16 | `0.000010653743249954742` |
| 8 | 337 | 26,17,16,24 | `381.135454277643` | 16 | `0.000010653743249954742` |
| 9 | 514 | 14,20,24,11 | `376.1248754015726` | 16 | `0.000010653743249954742` |
| 10 | 219 | 15,20,4,16 | `374.8919984004885` | 16 | `0.000010653743249954742` |

Break-even math:

- rate score per added byte: `6.658589531221714e-07`
- 16 charged atom bytes cost score `0.000010653743249954742`
- equivalent SegNet-distance reduction: `1.0653743249954742e-07`
- equivalent PoseNet-distance reduction at PR95 pose level: `8.832974774966603e-08`

These numbers show why micro-residual atoms are still plausible on PR95: even a
tiny correction can beat its byte cost, but only if the sign and target are
scorer-aligned.

## Guardrails Landed

The builder scaffolding fails closed on:

- non-single-member PR95 archives;
- zip-slip unsafe members;
- source archive SHA mismatch;
- source member SHA mismatch;
- atom plans that allow sidecars;
- empty atom lists;
- no-op latent atoms;
- out-of-range pair/dim/value references;
- atom output that leaves the latent raw stream, member blob, or archive SHA unchanged.

The default proxy atom file is deliberately non-dispatchable:

- `exact_eval_ready=false`
- every generated atom has `dispatchable=false`
- every generated atom says it requires component-trace or optimizer sign evidence

## Next Action

Highest-EV next step is a PR95 component trace on the confirmed repack bytes,
then re-run this planner with `--component-trace-json`. That turns the ranking
from latent-proxy into component-weighted hard-pair atoms and can produce a
small explicit atom plan for exact eval.

Before any remote trace or exact eval:

1. claim the lane with `tools/claim_lane_dispatch.py claim ...`;
2. use the existing PR95 runtime inflate path;
3. keep the exact archive SHA fixed to
   `2b9b471358f5bba97ea809dcca544ecca26b504fde770a9002046830f469368b`;
4. require adjudication and preserve component trace JSON.

Candidate builder command once a signed atom plan exists:

```bash
.venv/bin/python experiments/build_pr95_hnerv_residual_atom_plan.py \
  --archive experiments/results/pr95_hnerv_muon_packing_profile_20260504_codex/archive.pr95_repacked.zip \
  --exact-json experiments/results/lightning_batch/exact_eval_pr95_hnerv_muon_repacked_t4_fix2_20260504T0848Z/contest_auth_eval.adjudicated.json \
  --component-trace-json ${PR95_COMPONENT_TRACE_JSON} \
  --build-plan-json ${PR95_SIGNED_ATOM_PLAN_JSON} \
  --output-dir experiments/results/pr95_hnerv_residual_atom_candidate_${UTC_STAMP} \
  --top-k 80 \
  --stdout
```

No score claim is made by this ledger or by any candidate until exact CUDA
auth eval of the exact output archive bytes succeeds.

## Component-Trace Readiness Hardening Addendum

Patch scope:

- `experiments/build_pr95_hnerv_residual_atom_plan.py`
- `src/tac/tests/test_build_pr95_hnerv_residual_atom_plan.py`

Changes:

- Component traces are now parsed as the canonical
  `diagnostic_component_trace` schema from `experiments/contest_component_trace.py`.
- Trace intake fails closed on wrong `schema_version`, non-diagnostic
  `score_claim`, wrong evidence grade, duplicate/missing pair indices,
  non-finite component distances, archive SHA/byte mismatches, failed
  `contest_auth_eval_cross_check`, and component averages/scores that do not
  match the exact PR95 baseline within `1e-5`.
- When a valid PR95 component trace is present, the planner emits
  `pr95_hnerv_residual_atom_plan.signed.json`.
- The signed policy excludes pair `0`, ranks by component-trace first-order
  score, and assigns deterministic signs by shrinking the selected latent
  dimensions one uint8 step toward the previous pair.
- Candidate building rejects no-op atoms, duplicate pair/dim rewrites, and
  source-preserving plans before archive emission.
- `--build-generated-signed-policy` is local-only and builds a candidate archive
  from the generated signed policy; it does not launch exact eval or any remote
  job.

Verification:

```bash
.venv/bin/python -m py_compile \
  experiments/build_pr95_hnerv_residual_atom_plan.py \
  src/tac/tests/test_build_pr95_hnerv_residual_atom_plan.py

.venv/bin/python -m pytest \
  src/tac/tests/test_build_pr95_hnerv_residual_atom_plan.py -q
# 5 passed

.venv/bin/python experiments/build_pr95_hnerv_residual_atom_plan.py --help

.venv/bin/python experiments/build_pr95_hnerv_residual_atom_plan.py \
  --archive experiments/results/pr95_hnerv_muon_packing_profile_20260504_codex/archive.pr95_repacked.zip \
  --exact-json experiments/results/lightning_batch/exact_eval_pr95_hnerv_muon_repacked_t4_fix2_20260504T0848Z/contest_auth_eval.adjudicated.json \
  --output-dir /tmp/pr95_residual_atom_plan_smoke \
  --top-k 10 \
  --stdout
# latent-proxy smoke preserved top pairs: 517,222,133,232,523,515,216,337,514,219
```

Command to run after the PR95 component trace is harvested:

```bash
UTC_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
PR95_COMPONENT_TRACE_JSON=experiments/results/lightning_batch/exact_eval_pr95_hnerv_muon_repack_component_trace_t4_20260504T0914Z/component_trace.json

.venv/bin/python experiments/build_pr95_hnerv_residual_atom_plan.py \
  --archive experiments/results/pr95_hnerv_muon_packing_profile_20260504_codex/archive.pr95_repacked.zip \
  --exact-json experiments/results/lightning_batch/exact_eval_pr95_hnerv_muon_repacked_t4_fix2_20260504T0848Z/contest_auth_eval.adjudicated.json \
  --component-trace-json "${PR95_COMPONENT_TRACE_JSON}" \
  --output-dir "experiments/results/pr95_hnerv_residual_atom_candidate_${UTC_STAMP}" \
  --top-k 80 \
  --signed-policy-pairs 10 \
  --signed-policy-dims-per-pair 2 \
  --build-generated-signed-policy \
  --stdout
```

The resulting candidate archive is still non-promotable until a lane claim is
made and exact CUDA auth eval succeeds on the exact output bytes.
