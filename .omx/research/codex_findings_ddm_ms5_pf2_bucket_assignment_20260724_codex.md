# Codex findings — DDM MS5 PF2 bucket assignment

Evidence is `[macOS-CPU frozen-scorer advisory]`; `score_claim=false`; pointer
`0.1910828242 [contest-CPU]` is unchanged. MAIN landing review is required.

1. **MEASURED — PF2 pair membership is recoverable exactly.** The SHA-bound
   re-walk reproduced all `1,200/1,200` typed keys and conserved the exact
   `4,011,236` raw G4/V12 flip events. For every row, filtering the reconstructed
   raw event set by the emitted `pair_ids` reproduces the same boolean event set.
   No source pair is orphaned.

2. **CRITICAL — actuator and signed-direction identity is not recoverable from
   the PF2 construction lineage.** PF2 partitions predicted-to-target class
   flips by class pair, stratum, visibility, temporal class, and five-type role.
   It predates and stores no foreign key to a J2 receiver DOF or G2F/G2G paired
   secant. Consequently all 1,200 rows carry
   `ASSIGNMENT_UNRECOVERABLE_PF2_CONSTRUCTION_HAS_NO_ACTUATOR_DIRECTION_FOREIGN_KEY`;
   none invents a join from spatial overlap or class labels. Scope:
   `INSTANCE(current PF2 construction lineage) < FORMULATION < FAMILY < PARADIGM`.

3. **CUSTODIED — the relevant vocabularies exist but do not imply a join.**
   The source-bound table records 374 unique receiver actuator stable IDs
   (368 current J2 lifted receiver DOFs plus six measured G2G G2CS1 addresses)
   and the exact G2F `NEGATIVE_ONE_QUANTUM` / `POSITIVE_ONE_QUANTUM`
   convention. Exact join rows remain `0/1,200`.

4. **MEASURED — coverage is sparse in typed-key space, not pair space.**
   `37` buckets contain PF2 events and `1,163` contain none; zero-event rows
   remain explicit measurement debt rather than asserted zero scorer geometry.
   Every source pair appears in 10–31 occupied buckets. Multi-actuator bucket
   count is zero because no actuator join is admissible.

5. **HELD — the MS4 producers were not rerun.** The existing harness now
   accepts a separately SHA-bound MS5 receipt/table and validates it strictly.
   Its audit reports `assigned_count=0`, `unassigned_count=1,200`, so coverage
   does not reach even the G3 top24 block. Re-running Pose or fabricating Seg,
   composite-R, and dual rows would add cost without authority.

Machine artifacts:

- `.omx/research/ddm_ms5_pf2_bucket_assignment_20260724T044736Z/pf2_bucket_assignment_table.json`
  (`file_sha256=20fa2b2ce2bd96b91c64d4e1342109dd7dab399d4769cd372dbf67fbcdf97d8d`);
- `.omx/research/ddm_ms5_pf2_bucket_assignment_20260724T044736Z/ddm_ms5_pf2_bucket_assignment_receipt.json`
  (`file_sha256=3d0b9fcc738a1092bad495b0dbce2b022451e1442814a7cc274da41e43d455d6`).

## Round-1 adversarial review

The first pass correctly recovered membership and refused the missing join.
Adversarial review then found two custody weaknesses and fixed both before
landing: the membership claim initially relied only on reconstructed mass, so
the final implementation also performs exact assignment-filter boolean-set
equality for every bucket; and the receipt initially bound artifacts but not
the implementation sources used to re-walk PF2/J2, so five exact source hashes
are now input-bound and revalidated.

The remaining blocker is scientific, not mechanical. A successor must measure
a causal receiver-support relation: for each stable J2/G2G actuator and each
G2F-compatible sign, apply the receiver quantum, round-trip through the actual
receiver/R surface, and record which PF2 raw events it perturbs. Only that new
measurement can populate the lost foreign key.
