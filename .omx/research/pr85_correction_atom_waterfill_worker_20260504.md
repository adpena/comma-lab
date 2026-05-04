# PR85 Correction Atom Waterfill Worker - 2026-05-04

## Scope

Planning-only worker artifact for fine-grained PR85 correction side-channel search. Owned code surfaces:

- `experiments/plan_pr85_correction_atom_waterfill.py`
- `src/tac/tests/test_plan_pr85_correction_atom_waterfill.py`
- `.omx/research/pr85_correction_atom_waterfill_worker_20260504.md`

No remote or GPU dispatch was performed.

## Contract

The planner uses `tac.pr85_bundle.parse_pr85_bundle()` as the byte slicing source of truth, then decomposes only PR85 correction sidechannels:

- `post`: `post_stage1` through `post_stage4`
- `shift`: `motion_shift`
- `frac`: `motion_frac`
- `frac2`: `motion_frac2`
- `frac3`: `motion_frac3`
- `bias`
- `region`
- `randmulti`: 72 sparse replay groups from `PR85_HEADERLESS_RANDMULTI_SPECS`

Every atom records source segment bytes, estimated group bytes, decoded group bytes, semantic SHA-256, non-default count, no-op risk, and two strict gates:

- decoded-output-changing neutralization requires exact component-response before eval
- lossless recode requires decoded-output parity before eval

## Exact Negative Constraint

The prior `preserve_post_all_shift_frac2_frac3` candidate neutralized `motion_frac` only and saved 97 archive bytes, but exact T4 auth eval made the score worse:

- baseline PR85: score `0.25806611029397786`, bytes `236328`, pose contribution `0.043520110293977884`
- preserve_post_all_shift_frac2_frac3: score `0.27100583104425036`, bytes `236231`, pose contribution `0.056524331044250316`
- score delta: `+0.0129397207502725`
- pose contribution delta: `+0.013004220750272432`
- rate contribution delta: `-0.0000645`

Constraint: `motion_frac` must not be blindly deleted or treated as byte-only slack. Any finer-grained `motion_frac` atom requires exact component-response before eval.

## Candidate Policies

The planner emits these planning-only policies:

1. `decoded_parity_recode_all_correction_streams`
   - Lossless byte search across all correction streams.
   - Gate: decoded-output parity for every changed stream before eval.

2. `component_response_motion_frac_microatoms`
   - Fine-grained search inside the known-sensitive `motion_frac` correction.
   - Gate: exact component-response on `motion_frac` atoms before eval.
   - Blocked by the `preserve_post_all_shift_frac2_frac3` exact negative.

3. `component_response_bias_region_sidechannels`
   - Bias/region correction atom screen with fixed-v5 byte constraints.
   - Gate: exact component-response or runtime-decoded parity evidence before eval.

4. `randmulti_dense_group_response_top008`
   - Group-level `randmulti` probe, not whole-stream deletion.
   - Gate: exact component-response for selected groups before eval.

5. `small_byte_correction_response_top012`
   - Low-byte atom screen across post/motion/bias/region/randmulti.
   - Gate: exact component-response for each selected atom before eval.

## Dispatch Blockers

- Planning-only ledger; no archive candidate is emitted by this tool.
- This worker was explicitly forbidden to dispatch remote/GPU work.
- Any decoded-output-changing atom needs exact component-response before eval.
- Any lossless recode needs decoded-output parity before eval.
- The `motion_frac` byte saving is an exact negative until finer response evidence proves a safe sub-atom.

## Live Planner Run

Command:

```bash
.venv/bin/python experiments/plan_pr85_correction_atom_waterfill.py --json-out /tmp/pr85_correction_atom_waterfill_plan.json
```

The JSON artifact was written outside the repo to avoid creating files beyond this worker's ownership set. The durable summary is:

- schema: `pr85_correction_atom_waterfill_plan_v1`
- score claim: `false`
- dispatch performed: `false`
- source archive: `experiments/results/public_pr85_intake_20260503_codex/archive.zip`
- atom count: `82`
- exact-negative status: `exact_negative`
- blocked atom: `motion_frac`

Per-stream source bytes:

- `post`: `1400` bytes, `4` groups
- `shift`: `226` bytes, `1` group
- `frac`: `106` bytes, `1` group
- `frac2`: `149` bytes, `1` group
- `frac3`: `154` bytes, `1` group
- `bias`: `223` bytes, `1` group
- `region`: `273` bytes, `1` group
- `randmulti`: `16101` bytes, `72` groups, `13496` nonzero entries

Top ranked atoms by non-default decoded signal, with byte attribution caveat for shared streams:

1. `pr85_randmulti_g000`: estimated `1430` bytes, `1194` non-default entries
2. `pr85_post_stage3`: estimated `350` bytes, `597` non-default choices
3. `pr85_post_stage2`: estimated `350` bytes, `593` non-default choices
4. `pr85_post_stage1`: estimated `350` bytes, `546` non-default choices
5. `pr85_randmulti_g019`: estimated `634` bytes, `532` non-default entries
6. `pr85_randmulti_g063`: estimated `453` bytes, `380` non-default entries
7. `pr85_randmulti_g021`: estimated `438` bytes, `367` non-default entries
8. `pr85_randmulti_g020`: estimated `416` bytes, `349` non-default entries

Verification:

```bash
.venv/bin/python -m py_compile experiments/plan_pr85_correction_atom_waterfill.py src/tac/tests/test_plan_pr85_correction_atom_waterfill.py
.venv/bin/python -m pytest src/tac/tests/test_plan_pr85_correction_atom_waterfill.py -q
```

Result: `3 passed`.
