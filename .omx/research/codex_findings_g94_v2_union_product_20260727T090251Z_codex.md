# Codex findings — G94 V2 typed union product

UTC: `2026-07-27T09:02:51Z`  
Lane reviewed: `lane_g101_g94_v2_typed_union_product_20260727`  
Verdict scope: typed composition and source-backed fixture only

## Verdict

The V2 algebra closes the ownership gap:

```text
(base PVSA * cumulative final-Y1)
    * (G88 conditional-Y0 | G95 P-once basis * indexed chunks)
```

The right factor is a closed tagged sum. It is impossible for the typed encoder
or strict parser to admit G88 and G95 simultaneously, and exact EOF prevents a
second dead owner from hiding after the selected branch. Base, cumulative Y1,
basis, and chunks each have one counted byte home. The G95 parent key binds the
owner-independent preconditional member, avoiding a circular outer-product
hash.

The receiver reconstructs G98 final Y1 before either Y0 branch, requires
byte-identical Y1 afterward, and keeps scorer/target/evaluator machinery out of
decode. The G95 branch additionally requires exact ordered 0..599 chunk
coverage and a whole-state hash over the actual final-G98 population.

## Adversarial finding and fix

The deliberately stale G95 fixture failed closed correctly, but the outer
receiver replaced its precise chunk-state mismatch with the generic message
`G95 exclusive Y0 transition failed`. That suppressed the exact blocker signal
agents and controllers need. The wrapper now preserves the nested failure text
and labels it as chunk/state custody. The source-backed regression timeout was
raised from 180 to 300 seconds because the real fixture measured 214.13 seconds
under the concurrent governed G95 run.

Root validation:

- focused source-backed tests: 4 passed in 214.13 seconds;
- Ruff check: clean;
- Ruff format: clean;
- Python compilation: clean.

## Authority boundary and next action

This is not a candidate or score row. The current fixture intentionally proves
that mere n600 selector coverage is insufficient. The G95 branch must be
freshly refit and every chunk re-emitted against the chosen final semantic-Y1
whole state. Either branch then needs public `inflate.sh` recursive runtime
closure, same-archive double decode, exact n600 upstream evaluation, and actual
outer archive pricing.

Pointer delta: none.

STORES CONSULTED: committed G88, G95, G98, outer-archive, compact-PVSA and
carrier-compose implementations; G94 V2 module, fixture, and specification;
source-backed G85 archive.

HISTORICAL_PROVENANCE: first Codex adversarial review of the G94 V2
sparse-Y1/exclusive-Y0 typed product; preserves the closed algebra and prevents
chunk-state blocker suppression.
