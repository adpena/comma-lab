# L5 v2 / TT5L proof custody provenance hardening - 2026-05-16

## Scope

This fix wave hardens the L5 v2 / TT5L staircase against a false promotion path in
the temporal side-info proof. The previous contest-full-frame side-info proof
checked that two output directories differed, but it trusted caller-supplied
directories. That was parser/output evidence, not causal inflate provenance.

No score claim, promotion claim, or exact-eval readiness claim is made here.

## Fixes landed

- TT5L contest side-info proof now requires baseline and mutated inflate
  provenance manifests.
- Provenance must bind archive SHA-256, runtime tree SHA-256, file-list SHA-256,
  output aggregate SHA-256, output paths, command, exit code, and non-promotion
  authority flags.
- Non-target identity is split into payload sections and TT5L header. A changed
  TT5L header is allowed only when the delta is restricted to `side_len`; world
  model, AC state, and metadata payloads must remain byte-identical.
- L5 v2 dispatch readiness now rejects malformed gate-evidence mappings instead
  of crashing or coercing arbitrary values into authority.
- TT5L Dykstra feasibility artifacts must now carry literal JSON `false` for
  `score_claim`, `promotion_eligible`, and `ready_for_exact_eval_dispatch`.
  Missing fields, string `"false"`, numeric `0`, or any non-literal value block
  readiness.
- Dykstra artifacts now declare `feasibility_scope=score_axis_sanity_only` and
  `move_level_constraint_proof=false`; L5 v2 readiness rejects any artifact that
  is missing these fields or claims move-level feasibility authority.
- The C1/Z5/TT5L probe CLI exits nonzero when observations are missing and the
  verdict is blocked.
- If a probe template already exists, the next action advances to populating and
  evaluating observations rather than re-emitting the same template.

## Current staircase status

The older committed proof artifact is intentionally no longer sufficient for the
TT5L side-info gate because it was not bound to inflate provenance. The next
valid TT5L action is to rematerialize the contest-full-frame side-info proof with
the hardened command:

```bash
.venv/bin/python tools/build_tt5l_contest_sideinfo_consumption_proof.py \
  --baseline-archive <tt5l_baseline_archive_0.bin> \
  --mutated-archive <tt5l_sideinfo_mutated_archive_0.bin> \
  --baseline-output-dir <baseline_inflated_raw_dir> \
  --mutated-output-dir <mutated_inflated_raw_dir> \
  --file-list <contest_file_list.txt> \
  --baseline-inflate-provenance <baseline_inflate_provenance.json> \
  --mutated-inflate-provenance <mutated_inflate_provenance.json> \
  --artifact-out experiments/results/time_traveler_l5_v2/tt5l_contest_sideinfo_consumption_proof.json \
  --manifest-out experiments/results/time_traveler_l5_v2/tt5l_contest_sideinfo_outputs_manifest.json
```

Readiness after the patch:

```text
dykstra_valid= True
sideinfo_valid= False
timing_allowed= False
next_action= materialize_tt5l_contest_full_frame_sideinfo_consumption_proof
```

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/substrates/time_traveler_l5_autonomy/tests/test_consumption_proof.py \
  src/tac/tests/test_l5_staircase_v2.py \
  src/tac/tests/test_l5_v2_probe_disambiguator.py -q
```

Result: `109 passed in 2.01s`.

```bash
.venv/bin/python -m ruff check \
  src/tac/substrates/time_traveler_l5_autonomy/consumption_proof.py \
  src/tac/substrates/time_traveler_l5_autonomy/tests/test_consumption_proof.py \
  src/tac/optimization/l5_staircase_v2.py \
  src/tac/tests/test_l5_staircase_v2.py \
  tools/build_tt5l_contest_sideinfo_consumption_proof.py \
  tools/probe_l5_v2_staircase_disambiguator.py \
  src/tac/tests/test_l5_v2_probe_disambiguator.py
```

Result: `All checks passed!`.

Follow-up Dykstra flag regression:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_check_substrate_dykstra_feasibility.py \
  src/tac/tests/test_l5_staircase_v2.py -q
```

Result: `88 passed in 0.71s`.
