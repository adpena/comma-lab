# M1 facet observer — build and live attachment receipt

Date: 2026-07-21 UTC  
Authority: `codex_delegate:m1_facet_observer:20260721T005534Z`  
Lane: `lane_m1_facet_observer_20260721`  
Axis: `[macOS-CPU advisory]`  
Promotion authority: **none**; score claim: **false**; pointer: **unchanged**

## SCOPE_AMENDMENT_ACK

The observer consumes all five binding operator amendments received through
the per-arm inbox at `2026-07-21T01:20:52Z`, `01:24:41Z`, `01:25:35Z`, and
`01:26:01Z`, plus the estimator retraction at `01:54:58Z`. The fixed random-n24,
hardest-six, BIC, Neyman, and Good-Turing panel designs are superseded.
The binding design is now:

- first-checkpoint n600 per-pair receiver/scorer sweep when the explicit
  footprint preflight accepts it, else a labelled PCG64 n128 fallback;
- recurring top-32 d_seg plus 16 seeded-background cohort;
- measured 50%/90% d_seg-mass prefixes from the exhaustive census, direct
  class-flip-composition strata with one exemplar per nonempty stratum, and
  `realization_breakeven_bytes_v1` consumed by ID as fix-EV telemetry only;
- lossless native-resolution PNG diagnostics, exact signed-delta `.npy`
  sidecars, bit-exact persisted-map verification, and eight-per-row
  concentration contact sheet;
- out-of-band class-pair excursion, per-pair d_seg/d_pose tails, and explicitly
  pair-internal temporal argmax instability.

## Live source and custody

The live materializer completed before observer attachment and success-cleaned
its rebuildable `base_camera_frames.raw`. It retained a stable read-only
`base_scorer_planes.npy`, shape `[600,2,384,512,3]`, dtype `uint8`, bytes
`707788928`, declared SHA-256
`d5523f2d972d57ac8152978b53a943c332a56db5633b279c14469de6da9806f0`.
The observer accepts this as a fallback only after hashing and validating its
materialization receipt, then factor-2 realizes both camera frames with exact
receiver proofs into observer-owned storage. It never writes into the run.

Other sealed custody:

- band manifest SHA-256:
  `2fd10841dc0cb344454e4af55bd8d27e5e1d819a97df3fc03307604dfffcc367`;
- frozen n600 GT cache SHA-256:
  `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`;
- carrier binding is hash-validated at observer startup;
- every output remains `[macOS-CPU advisory]`, `score_claim=false`,
  `subsample_advisory=true`, and `promotion_eligible=false`.

## Build verification

Status at this checkpoint: implementation review and final amended tests are in
progress. Runtime PID, launch receipt, first rank/facet rows, measured census
derivation, and panel receipts will be appended before the delegated arm is
closed.

## Operator tail commands

```bash
tail -f /Volumes/VertigoDataTier/pact/evidence/m1_facet_observer_20260721/facets.jsonl
tail -f /Volumes/VertigoDataTier/pact/evidence/m1_facet_observer_20260721/facets_perpair_rank.jsonl
```

## STORES CONSULTED

- delegated authority file and SHA/byte contract;
- `CLAUDE.md` and `AGENTS.md`;
- per-arm and broadcast inbox JSONL;
- `.omx/state/lane_registry.json` and subagent progress ledger;
- live run checkpoint directory and its base-plane materialization receipt,
  read-only;
- band manifest, carrier binding, frozen GT cache, frozen upstream scorers;
- canonical equation registry entry `realization_breakeven_bytes_v1`;
- latest canonical frontier surfaces only to preserve pointer separation.

## Verdict scope

This artifact can establish only that a read-only observer produced decomposed,
receiver-closed trajectory telemetry for the retained M1 checkpoints on
macOS CPU. It cannot establish an n600 contest score, CPU/CUDA parity,
promotion readiness, or a frontier-pointer change. MAIN must perform a
mandatory full `base..branch` landing review before merge.
