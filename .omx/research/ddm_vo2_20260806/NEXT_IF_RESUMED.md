# NEXT_IF_RESUMED - ddm_vo2

Resume from R2. Do not reseal from this generation.

1. Re-run:

   ```bash
   .venv/bin/python tools/build_ddm_vo2_instrument_registry.py --out-dir .omx/research/ddm_vo2_20260806
   ```

   Compare `ROUND_SUMMARY.json` row counts and `MANIFEST.sha256.json` before trusting prior counts.

2. Start R2 with the confirmed top families, in this order:

   - `iteration_cap_stop_defaults` and CA1 Class B cap sites.
   - SW1 `project_after_seam` / `euclidean_metric_site` rows.
   - DK1 `float-first` / `uint8` realization rows.
   - highest-fanout source candidates from the `vo2-new` family.

3. For each row, grade all ten elements:

   `initialization`, `proposal_step_rule`, `stopping_rule`, `metric_inner_product`,
   `subset_sampling`, `realization`, `projection_constraint_handling`, `tie_breaks`,
   `seed_determinism`, and `caches_staleness`.

4. R3 must recurse calibration lineage. Every validator named in a row becomes an instrument row
   unless it terminates at exact authority or is marked `UNANCHORED`.

5. R4 must grade VO2's own instruments:

   - `tools/build_ddm_vo2_instrument_registry.py`
   - source-token denominator heuristic
   - `tools/check_instrument_registry_form_grade_refs.py`
   - the scoped receipt scan that currently cites zero registry IDs

6. Repeat R1-R4 over new instruments surfaced during R2/R3/R4. Seal only on a round that adds zero new
   rows. Record the dry trajectory in `ROUND_SUMMARY.json` and this receipt family.

Boundaries still bind: scorer-free unless a later charter explicitly owns the slot, no `/tmp` persisted
evidence, no protected-file edits, and serializer commits with post-edit SHA-256s.
