# DAG FEED — REGMAX-FAMILY-PROBES-20260720

**Pointer `0.1910828242 [contest-CPU Linux x86_64]` UNMOVED.** `research_only=true`; no live config,
training, paid dispatch, evaluator score, promotion, or family-negative authority.

```text
09cc0026cb preregistration
  + real n600 fp16 logits sha 41d3ef535f5b...
  + real n600 hard labels sha 36c6be718916...
  -> unit-scale target-map measurement (117,964,800 pixels)
       -> sparsemax exact one-hot = 0.973330917358
       -> Lane exact one-hot = 0.257648351744
       -> target-only result; never HARD_ACCEPT
  -> receiver gate: regmax logits -> decoded uint8 candidate [ABSENT]
       -> sparsemax probe = N-A; falsifier NOT FIRED
       -> Hopfield prototype gate [ABSENT]
            -> entropy-prox probe = N-A; falsifier NOT FIRED
  -> principal rank-4 A,b fixture [ABSENT]
       + Aurenhammer min-generator same-coder comparator [ABSENT]
       -> tropical representative probe = N-A; falsifier NOT FIRED
```

## Exact falsifiers retained

1. `probe_sparsemax_margin_band_preimage_ab_v1`: falsified only if, on identical real cells and budget, the
   matched sparsemax receiver arm does not improve hard accepts, exact-oracle calls, or bytes versus
   entropy/Cole–Hopf.
2. `probe_tropical_residuation_principal_cell_representative_v1`: falsified only if the constructed,
   gauge-fixed principal representative changes the hard cell, is longer under the same coder, or requires
   uncounted state.
3. `probe_entropy_hopfield_preprox_uint8_v1`: falsified only if one step using a frozen rank-4 valid-cell
   prototype bank does not improve hard accepts or exact-call cost versus no-prox on identical cells/budget.

## Reopen edges and consumers (not authorization)

- Land a typed, test-population-independent logits/rank-4 pre-step to decoded uint8 preimage adapter that
  reuses the unchanged bounded solver and frozen CPU-Torch hard oracle. A positive sparsemax receipt may then
  feed the **band-slack/annulus loss config**.
- Land a frozen prototype artifact with source split/hash and the same adapter. A positive prox receipt may
  then feed the **R1 `d_B` pre-step**.
- Land the principal-cell inequality fixture, Aurenhammer comparator, gauge convention, and one same-coder
  serializer. A positive tropical receipt may then feed the **PDW/quotient representative coder**.

No edge may be activated from target debt alone. MAIN review is required before merge or rerun.
