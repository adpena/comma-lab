# G70 specification — exact receiver/scorer finite-effect materializer

Date: 2026-07-27  
Lane: `lane_g70_canonical_receiver_scorer_effect_materializer_20260727`  
Implementation-start HEAD: `0dd5bbf346`  
Status: exact finite endpoint emitter implemented; transition-anchored JVP/VJP
closure blocked; no score, candidate, promotion, dispatch, or pointer claim

## Scope

G70 is the smallest real replacement seam for the missing canonical
receiver/`R`/scorer effect materializer named by G67. It consumes two exact
`G17CandidateForwardObservationV1` objects that retain:

- the same exact target object and ordered pair IDs `0..599`;
- different outer `archive.zip` bytes;
- receiver receipts and decoded output bytes;
- realized-through-`R` tensors;
- frozen SegNet labels and PoseNet-6 outputs; and
- frozen scorer and runtime custody bytes.

The materializer does not accept caller-provided effect, JVP, or VJP arrays.
It reopens both ZIP member streams, proves that each observation's exact
member bytes occur in its outer ZIP, and derives the full-n600 endpoint effects
from the typed observations.

## Exact finite effects

For pair `i`, let `s_i` and `p_i` be the exact G17 per-pair Seg mismatch
fraction and Pose6 MSE. The emitted `(600,2)` float64 array has axes
`[seg_score_term_delta, pose_score_term_delta]`.

The Seg attribution is

`e_seg_i = 100 * (s_candidate_i - s_baseline_i) / 600`.

The Pose attribution is the Aumann-Shapley straight-line integral from the
baseline population MSE to the candidate population MSE. Its 600 terms sum to

`sqrt(10 * d_pose_candidate) - sqrt(10 * d_pose_baseline)`.

The exact outer-ZIP byte delta is measured from the reopened byte strings and
passed through `build_serialized_archive_delta_contract`. Its rate effect is

`25 * (candidate_zip_bytes - baseline_zip_bytes) / 37_545_489`.

The receipt separately records the aggregate endpoint Seg, Pose, rate, and
total deltas. These are endpoint effects, not a contest score or candidate
claim.

## Why JVP/VJP remains blocked

The SSD contains a real terminal frozen-scorer VJP campaign:

`/Volumes/VertigoDataTier/pact/evidence/vjp_custody_20260719/extension_n600_20260720/campaign_receipt.json`

Its current receipt says `COMPLETE_N600`, 600 completed pair IDs, no final
refusals, and no missing pairs. That is real scorer-autograd evidence. It is
not the requested transition evidence:

- its producer reads `gt_f0`/`gt_f1` from
  `experiments/results/mlx_fleet_gt_cache/gt_n600.npz`;
- its active arrangements are source-cache/native-winner arrangements;
- its manifest schema carries no G17 baseline observation receipt,
  candidate observation receipt, archive ZIP identity, transition ID, or
  baseline input-frame identity; and
- it contains VJPs at the source arrangement, not realized JVP/VJP
  contractions for a G17 archive transition.

G70 validates the campaign's terminal n600 coverage but returns
`G70_GT_SOURCE_VJP_CANNOT_BE_RELABELED_AS_G17_TRANSITION_JVP_VJP` and
`G70_TRANSITION_ANCHORED_SCORER_DIFFERENTIATION_OWED`. Booleans, arbitrary
NPZ arrays, or the source VJP campaign cannot clear those blockers.

## Strict consumer boundary

`require_g70_actionable_costate_input` is the production integration guard.
It always refuses. The deterministic emitted receipt has:

- `actionable_costate_input=false`;
- `actionable_consumers=[]`;
- `score_claim=false`;
- `candidate_claim=false`;
- `promotion_eligible=false`; and
- `ready_for_exact_eval_dispatch=false`.

G67 remains unchanged and blocked. A future positive admission requires a
separate adversarial review after a real producer differentiates the frozen
scorers at the exact retained G17 baseline, contracts the exact
baseline-to-candidate receiver/`R` direction, and seals both derivative
outputs to the same transition ID and archive/object hashes.

## Next executable prerequisite

Run a governed, resumable full-n600 producer whose per-pair immutable sidecars
are created from the baseline G17 observation's exact retained camera/R
tensors, and whose manifest seals:

1. baseline and candidate G17 forward receipt SHA-256;
2. baseline and candidate outer-ZIP SHA-256 and byte counts;
3. baseline and candidate receiver-receipt and decoded-output SHA-256;
4. exact per-pair input-frame hashes for the differentiation anchor and
   transition direction;
5. frozen scorer/runtime hashes;
6. actual autograd VJP tensors and computed JVP/VJP contractions; and
7. ordered terminal coverage `0..599` with no refused or missing pairs.

That heavy run was not launched in G70.

## Files and verification

- `src/tac/witness_control/taskspace_g70_receiver_scorer_effect_materializer_v1.py`
- `src/tac/witness_control/tests/test_taskspace_g70_receiver_scorer_effect_materializer_v1.py`
- this specification
- `g70_receiver_scorer_effect_materializer_blocker_receipt_20260727.json`

Focused verification:

```bash
uv run pytest -q \
  src/tac/witness_control/tests/test_taskspace_g70_receiver_scorer_effect_materializer_v1.py

uv run ruff check \
  src/tac/witness_control/taskspace_g70_receiver_scorer_effect_materializer_v1.py \
  src/tac/witness_control/tests/test_taskspace_g70_receiver_scorer_effect_materializer_v1.py

uv run ruff format --check \
  src/tac/witness_control/taskspace_g70_receiver_scorer_effect_materializer_v1.py \
  src/tac/witness_control/tests/test_taskspace_g70_receiver_scorer_effect_materializer_v1.py

python3 -m py_compile \
  src/tac/witness_control/taskspace_g70_receiver_scorer_effect_materializer_v1.py \
  src/tac/witness_control/tests/test_taskspace_g70_receiver_scorer_effect_materializer_v1.py
```
