# DAG FEED — R5 witness-anchor reverse-waterfill (Task #578-R5)

`research_only=true` · `[macOS-CPU advisory]` · pointer `0.1910828242 [contest-CPU]` unchanged · MAIN review required.

## Executed dependency graph

```text
hash-verified ep725 archive (149fefd0..., 83,838 B)
  -> unchanged LVLS1 receiver -> raw hash 8565df10...
  -> D1 n600 hard scorer, batch32/seed1234, 19 checkpoints
       -> receiver miss-site cache + per-class split
       -> D2a #453 exact mutated archive -> full n600 -> REJECT
       -> D2d #336 exact mutated archive -> full n600 -> ADMIT
       -> D2f R3 set intersection -> >5% screen PASS
             -> impossible all-overlap score bound -> REJECT; no splice
  -> structural gates
       -> #140 no dxi -> N/A
       -> #400 self_orient + no joint accept adapter -> REFUSE
       -> #557 no LVLS1 alternate-coder consumer -> BLOCKED
       -> #553 no RGB pullback -> BLOCKED
  -> D3 one admitted receiver-closed stream
       -> pairwise interaction set is empty -> singleton union-once closure
       -> v5 = #336 archive d2ad27cf..., 83,827 B
  -> D4 axis-separated pointer comparison -> NO PAID ROW; pointer unchanged
```

## Measured feed

- Anchor: `(0.0035127175506204367, 127.35957336425781, 83,838)`, advisory `S=36.0945691012573`.
- #453: `delta S=+0.316621139582665`; do not reuse the previous prefix as a zero-distortion transfer prior.
- #336 bounded point: `delta S=-0.04473334744707813`; exact singleton candidate `(0.003522830316796899, 127.03333282470703, 83,827)`.
- R3: 33,787 overlapping sites / 414,377 witness misses (`8.153686136054848%`), but impossible best-case `delta S=+0.09134352513598683`; screen pass is not admission.
- #557: alternate coder bytes are measurement-only until a strict LVLS1 consumer exists.
- D4: Pose contributes `99.34696718054592%` of the non-comparable numeric v5-to-pointer gap.

## Unified-solver hooks

1. **Sensitivity map:** receiver-miss chunks and the D1 per-class split are the reusable exact support; no proxy map is promoted.
2. **Pareto constraint:** admission is exact measured `delta_nonrate_S + RATE_PRICE_S_PER_BYTE * delta_bytes < 0`.
3. **Bit allocator:** #336 receives one admitted exact anchor-local point; the full tensor/bit frontier remains open.
4. **Cathedral/autopilot:** consume `NO_PAID_ROW_POSE_BINDING`; do not dispatch or move the pointer from this advisory lane.
5. **Continual learning:** canonical input is SSD `receipt_v3.json` SHA-256 `0c471598bd1ad9204488d4ed69705900d986c97be76e805a166695e8fa69ee8a` plus the durable memo/reuse manifest.
6. **Probe disambiguator:** receiver-bound full n600 score, not isolated compressed bytes, arbitrates JRD/requant/coder choices.

## Reactivation / settling measurements

- #400: build a receiver-bound joint Seg/Pose proposal/accept adapter, then full n600 hard score.
- #557 block-FP/context: add a strict content-bound LVLS1 parser/consumer, parse back logical arrays, full decode, hard-score n600.
- #553/R3: add a counted scorer-free spatial/RGB inverse-R consumer before any splice claim.
- #140: reactivate only on a manifest with an actual counted dxi/pose stream.
- #336: extend cheapest-first one tensor/bit at a time, stopping at measured rate break-even; do not infer monotonicity from this singleton.

No heavy launch, paid dispatch, live-run mutation, or pointer mutation is authorized by this FEED.
