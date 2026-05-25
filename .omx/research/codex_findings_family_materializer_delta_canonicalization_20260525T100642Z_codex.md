# Codex Findings - Family Materializer Delta Canonicalization

Timestamp: 2026-05-25T10:06:42Z
Agent: Codex
Lane: `codex_materializer_generic_feedback_hardening_20260525`

## Finding

The prior receiver-safe feedback bridge preserved `section_recode` archive
deltas, but packet/tensor/sibling family materializers still had signal-loss
edges:

- `packet_member_recompress_v1` emitted byte savings only under
  `selected_compression`.
- `tensor_factorize_v1` emitted byte savings only under `factorization`.
- sibling families used `selected_merge`, `selected_payload`, and
  `selected_elision`.
- runner auto-discovery only recognized explicit `serialized_archive_delta`,
  so receiver-satisfied family manifests without that object could remain leaf
  artifacts instead of planner feedback.

## Landing

- Added `tac.optimization.materializer_feedback` as the canonical extractor for
  family-local materializer byte deltas.
- All family-agnostic materializers now emit a neutral
  `serialized_archive_delta_contract.v1` alongside family-local detail.
- Queue observation records normalize family-local deltas into both namespaced
  fields and `serialized_archive_delta_*` fields.
- Runner feedback and auto-discovery preserve legacy manifests with
  family-local deltas while avoiding over-discovery of generic
  source/candidate JSON files.
- Inverse-steganalysis queue observation intake reads packet/tensor/sibling
  normalized fields, so positive receiver-satisfied rows stay rate-positive and
  receiver-negative rows remain blocking feedback.
- Family materializer queue postconditions now require the neutral
  `serialized_archive_delta.schema` plus source/candidate archive byte fields.
- Recursive adversarial pass closed the paired action-functional leak: ranked
  acquisition rows and discrete action cells now both zero priority when
  materializer archive-delta feedback blocks receiver/rate-safe water-bucket
  selection.

## Authority Contract

The new delta contracts and empirical rows are planning signal only:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

They may steer local water-bucket replanning, demotion, or receiver repair.
They do not promote, rank/kill, submit, or skip contest CPU/CUDA auth eval.

## Verification

```bash
.venv/bin/python -m ruff check \
  src/tac/optimization/materializer_feedback.py \
  src/tac/optimization/family_agnostic_materializers.py \
  src/comma_lab/scheduler/experiment_queue_observer.py \
  src/tac/optimization/inverse_steganalysis_acquisition.py \
  src/comma_lab/scheduler/byte_shaving_campaign_queue.py \
  tools/run_byte_shaving_materializer_campaign.py \
  tools/run_family_agnostic_materializer_sweep.py \
  src/tac/tests/test_family_agnostic_materializers.py \
  src/tac/tests/test_experiment_queue_observer.py \
  src/tac/tests/test_byte_shaving_materializer_campaign_runner.py \
  src/tac/tests/test_inverse_steganalysis_acquisition.py \
  src/tac/tests/test_byte_shaving_campaign_queue.py \
  src/tac/tests/test_family_agnostic_materializer_sweep.py \
  --no-cache
```

Result: passed.

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_family_agnostic_materializers.py \
  src/tac/tests/test_experiment_queue_observer.py \
  src/tac/tests/test_byte_shaving_materializer_campaign_runner.py \
  src/tac/tests/test_inverse_steganalysis_acquisition.py \
  src/tac/tests/test_byte_shaving_campaign_queue.py \
  src/tac/tests/test_family_agnostic_materializer_sweep.py \
  -q
```

Result: 238 passed.

Additional focused regression after recursive adversarial fix:

```bash
.venv/bin/python -m ruff check \
  src/tac/optimization/inverse_steganalysis_acquisition.py \
  src/tac/tests/test_inverse_steganalysis_acquisition.py \
  --no-cache
PYTHONPATH=. .venv/bin/pytest \
  src/tac/tests/test_inverse_steganalysis_acquisition.py \
  -q
```

Result: passed; 49 passed.

## Next Integration

The remaining score-lowering work is not more leaf accounting. The next step is
to feed these family-neutral empirical deltas into higher-level bucket assembly:
compare packet, tensor, section, header, and inverse-cell candidates in one
action surface, then materialize only the combinations whose receiver/rate
contract survives local proof and exact-axis dispatch gates.
