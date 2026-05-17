# Pre-Entropy Catalog #321 Absolute-Path Custody Fix - 2026-05-17

## Status

`tools/pre_entropy_substrate_pivot_prober.py` had a custody false negative:
the live FEC6 canonical contest archive was accepted when passed as a relative
path but rejected when passed as the equivalent absolute path under the repo.

Regression anchor:

- relative path: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip`
- absolute path: `/Users/adpena/Projects/pact/experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip`
- prior verdict split: relative `VALIDATED_CONTEST_MEMBER`, absolute rejected as outside canonical root

## Fix

Path validation now resolves relative candidates against the repository root for
file IO, canonical-root checks, ZIP validity checks, whole-archive probes, and
member-level archive probes. The original path string is still preserved in the
emitted diagnostic result so existing manifests keep the operator-supplied
target surface.

## Tests

Focused:

```bash
.venv/bin/python -m pytest src/tac/tests/test_pre_entropy_substrate_pivot_prober_phantom_score_fix.py -q
```

Result: `22 passed in 0.36s`.

Settled partner-wave cluster:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_consumer_15_wyner_ziv_hoist_real_jacobian.py \
  src/tac/tests/test_check_321_no_phantom_wyner_ziv_savings_from_research_sidecar.py \
  src/tac/tests/test_low_gap_closure_widened_namespace_wire_ins.py \
  src/tac/tests/test_low_gap_closure_widened_bucket_b_extensions.py \
  src/tac/tests/test_low_gap_closure_widened_bucket_c_autopilot_sister_817_consumption.py \
  src/tac/tests/test_pre_entropy_substrate_pivot_prober.py \
  src/tac/tests/test_pre_entropy_substrate_pivot_prober_phantom_score_fix.py \
  src/tac/tests/test_q6_preprobe_extended.py \
  src/tac/tests/test_q6_preprobe_pairwise_composition_alpha.py \
  src/tac/tests/test_bit_allocator_end_to_end.py -q
```

Result: `196 passed in 199.11s`.

## Evidence Discipline

This is a diagnostic custody fix, not a score claim. It does not promote a
candidate archive, does not assert CUDA or CPU score movement, and does not make
any archive submission-ready. It prevents a valid canonical archive from being
misclassified solely because a CLI or consumer passed an absolute repo-local
path.
