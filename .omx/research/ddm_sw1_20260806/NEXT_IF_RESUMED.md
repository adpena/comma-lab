# NEXT_IF_RESUMED - ddm_sw1

Resume from the committed runner and receipts, not from chat.

1. Verify custody first:
   - parent archive sha256 must remain `b35e756829306a85ec2ad51634bde74523d89df9046c682253176b393bd59c06`;
   - bulk rows sha256 must remain `5994edf22d5af37a5cbfe17712d4ae9ad610aacb86453120c65a9ddd00d8026a`;
   - committed summary sha256 must remain `231c11e958731450c8821d70d494ff26e7d4959276dd085ac80bd4a8330f5ac1`.

2. If the scorer lane is still owned by et2, do not launch n600. SW1 is eligible only for n32 expansion:
   `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/ddm_sw1_null_basis_phase_solve.py --limit 8 --steps 15 --resume`
   remains inside the charter bound; for n32, first get explicit lane clearance and update the receipt denominator.

3. Do not reroute through project-after. The measured n=4 aggregate is:
   - Euclidean project-after eta `0.0940882598`;
   - diagonal metric project-after eta `0.0982514571`;
   - solve-within eta `0.3114071607`;
   - bar `0.1710048742`.

4. dk1 interface work should consume the `dk1_realizer_interface` in each row:
   `NullRealizer.apply_coefficients(pair, coeffs, basis_id, rounding_policy)`.
   The first useful dk1 measurement is round-vs-lattice on pairs `[0, 20, 32, 48]` with the same basis certificate and same target field.

5. Full MS4D substitution remains queued, not consumed. Do not claim MS4D was used by SW1 unless a follow-on row binds receiver object builder, realized uint8 quantum, candidate delta, dimension rate home, and coder payload owner.

6. No exact-row language is allowed until a receiver-closed archive is built, counted, parsed back, and scored on the authority axis. Current SW1 artifacts are small-n advisory and `score_claim=false`.
