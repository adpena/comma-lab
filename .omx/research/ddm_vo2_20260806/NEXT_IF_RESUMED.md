# NEXT_IF_RESUMED - ddm_vo2

Resume from R2-partial. Do not reseal from this generation.

Generation-2 state: R1 was re-run and matched the prior manifest; R2 batch 1 graded 23 selected
rows in `.omx/research/ddm_vo2_20260806/R2_ELEMENT_DECOMPOSITION.jsonl` with all ten charter
elements present. Already covered: `iteration_cap_stop_defaults`, the six CA1 Class-B cap sites,
eight SW1 project-after/metric/uint8 seam rows, three DK1 realizer rows, and five highest-fanout
`vo2-new` source candidates. `ROUND_SUMMARY.json` says `round_reached=R2-partial`; do not treat
this as dry or sealed.

Covered aggregate form-grade reference: `form_grade_ref:iteration_cap_stop_defaults`.

1. Re-run:

   ```bash
   .venv/bin/python tools/build_ddm_vo2_instrument_registry.py --out-dir .omx/research/ddm_vo2_20260806
   ```

   Compare row counts and the R1 hashes before trusting prior counts:

   - `INSTRUMENT_REGISTRY.jsonl`: `947b7faaa3ba61dfad567b434075c8151028e8fd5e6dbe3c38cbcb4ccc43b936`
   - pre-R2 builder `ROUND_SUMMARY.json`: `0b450d49d33d1ba8e756b1d16d031f144fd05a490ace2c4292b577d4bb2b4393`

   The builder overwrites `ROUND_SUMMARY.json` to the R1 shape. After the rebuild comparison, restore
   or refresh the R2 summary fields from `R2_ELEMENT_DECOMPOSITION.jsonl` rather than silently dropping
   R2.

2. Continue R2 by verdict-fanout ranking, never alphabetical:

   - Remaining `vo2-new` source candidates after the five already graded:
     `experiments/train_levelset_witness_realized_through_R_mlx.py`,
     `tools/build_ddm_vo2_instrument_registry.py`, `src/tac/ddm_costate_organ.py`,
     `src/tac/preflight.py`, and `tools/levelset_byte_close_and_eval.py`.
   - Then recurse into the remaining CA1/SW1/DK1/VO1 low-fanout rows by stakes and cure availability.
   - Keep UNKNOWN as a valid grade; do not invent receipts.

3. For each row, grade all ten elements:

   `initialization`, `proposal_step_rule`, `stopping_rule`, `metric_inner_product`,
   `subset_sampling`, `realization`, `projection_constraint_handling`, `tie_breaks`,
   `seed_determinism`, and `caches_staleness`.

4. R3 must recurse calibration lineage. Every validator named in a row becomes an instrument row
   unless it terminates at exact authority or is marked `UNANCHORED`.

   Known newly surfaced R3 lineage from R2 batch 1:

   - `src/tac/canonical_equations/trajectory_derived_stopping_20260805.py`

5. R4 must grade VO2's own instruments:

   - `tools/build_ddm_vo2_instrument_registry.py`
   - source-token denominator heuristic
   - `tools/check_instrument_registry_form_grade_refs.py`
   - the scoped receipt scan that currently cites zero registry IDs

6. Repeat R1-R4 over new instruments surfaced during R2/R3/R4. Seal only on a round that adds zero new
   rows. Record the dry trajectory in `ROUND_SUMMARY.json` and this receipt family.

Boundaries still bind: scorer-free unless a later charter explicitly owns the slot, no `/tmp` persisted
evidence, no protected-file edits, and serializer commits with post-edit SHA-256s.
