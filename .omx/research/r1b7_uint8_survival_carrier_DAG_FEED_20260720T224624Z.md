# FEED-R1B7-UINT8-SURVIVAL-20260720

`research_only=true`
Authority: `[macOS-CPU advisory]`, `score_claim=false`
Pointer: `0.1910828242 [contest-CPU] UNMOVED`

`verdict_scope`: sealed R1b4 n16, 498 exact-feasible Fisher-ordered fixed-arm
sites, bounded top-8 integer-aware prefix, seed 1234/batch 16 CPU Torch. Excludes
n600, contest CPU/CUDA, other bases, other receivers, and carrier-family claims.

## Executed readiness DAG

1. `PASS source_custody`
   - R1b6 receipt, baseline/candidate archives, Fisher ordering, target raw,
     decoder, and frozen scorer weights pinned by SHA-256.
2. `PASS storage_waterfall`
   - SSD root selected with 690,998,198,272 bytes free versus 2,831,133,440
     required.
3. `PASS sealed_double_decode_baseline_fixed`
   - baseline raw SHA `8c0bd711…9383`; fixed raw SHA `b0dbc05a…e4adb`.
4. `PASS fixed_constructor_equivalence`
   - reconstructed replay equals sealed R1b6 byte for byte, SHA
     `063986d4…765c`; the requested fixed arm is not distinct.
5. `PASS exact_498_site_stage_autopsy`
   - histogram `[uint8=0, resize=0, stem=0, head-same-rival=204,
     head-wrong-rival=0, survived-collateral=5, survived-clean=289]`;
   - exact total 498; head rank 4; margin reconstruction error 4.41447e-6.
6. `BLOCK fixed_hard_and_rate_admission`
   - combined recovery -0.000660674492;
   - 22,891 archive bytes; 45.96586345 bytes/site; break-even 0.
7. `PASS bounded_integer_proposal_search`
   - top-8 EV sites, 63 exact proposals, 0 wrong-to-target hard crossings;
   - post-run review invalidated 4 already-correct positive-margin candidates;
   - no infeasibility claim.
8. `PASS diagnostic_integer_receiver_closed_archive`
   - 38 writes, 624 replay bytes, archive SHA `2f18fa52…d9b3f`;
   - deterministic double-decode raw SHA `1fca2060…f0d9c`.
9. `BLOCK diagnostic_integer_composed_hard_and_rate_admission`
   - Seg flips unchanged at 10,002;
   - combined recovery -4.29028077e-7;
   - 184 archive bytes; break-even 0.
10. `BLOCK marginal_prefix_waterfill_unmeasured`
    - full fixed set and diagnostic integer composition rejected;
    - individual receiver-composed prefixes remain open and unmeasured.
11. `REFUSE n600_dispatch`
    - prerequisite “n16 positive with margin and byte-paying” is false.
12. `PASS disk_hygiene_success_path`
    - six success raw files certified then deleted; sealed archives retained.
13. `BLOCK_KEEP_BYTES attempt1_cleanup`
    - fail-closed parity attempt has no success cleanup authority; four raw
      files retained and machine-recorded, no deletion.

## Solver-stack feed

- Sensitivity map: preserve Fisher/necessity order, but mark fixed same-bin
  magnitude as receiver-composed negative on this n16 formulation.
- Pareto constraint: hard combined recovery must be positive before rate; the
  full fixed arm and diagnostic four-site composition violate it.
- Bit allocator: zero conditional byte budget via equation ID
  `realization_breakeven_bytes_v1` for those two measured composed sets only;
  no claim on unevaluated marginal prefixes.
- Cathedral/autopilot: refuse redispatch of the exact fixed or top-8 integer
  recipe; reopen only with composed collateral/pose admission and a measured
  byte-paying prefix.
- Continual learning: append a scoped probe outcome; do not overwrite the old
  positive R2b n600 anchor.
- Probe disambiguator: fixed-vs-integer modes were both executed; integer hard
  crossings exist individually but do not survive composed score/rate admission.

Receipt:
`.omx/research/r1b7_uint8_survival_carrier_20260720T224624Z.json` SHA-256
`61f3d03930ac765b3ad5a287cbff29a3073c800eb5a5f2b98b8a701bc086d03c`.

MAIN landing review is required before merge.
