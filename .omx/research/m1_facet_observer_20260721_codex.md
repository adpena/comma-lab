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

The first live scoring attempt failed closed because replaying sealed GT labels
through candidate batch-16 SegNet did not reproduce the batch-32 cache exactly.
An exact self-replay smoke over pairs 0..15 measured maximum per-pair d_seg
offset `5.086263020833333e-06`; batch-16 Pose candidate versus sealed cached
poses measured maximum d_pose `2.1832103254867025e-11`. The corrected observer
therefore uses the sealed cache as GT authority, runs both candidate scorers at
the binding batch 16, and emits `segnet_batch_geometry_parity=false` on every
row. This is trajectory telemetry only, never contest or promotion evidence.

## Build verification

Implementation and live attachment are complete:

- focused verification: Ruff, py_compile, and `17 passed`; both changed Python
  files received two review-tracker passes with 100% entity coverage;
- commits: `014b87715a`, `2e21f249e4`, and `14ec523233`, all without co-author
  trailers;
- detached observer PID `77097`, CWD verified as this isolated worktree,
  launch receipt
  `/Volumes/VertigoDataTier/pact/evidence/m1_facet_observer_20260721/launch_batch_geometry_fix/launch_manifest.json`,
  SHA-256 `bd963f9260f16483a04d5e286541e5aa25743b3f0fb332c3f69fc42b6e0b2c08`;
- exhaustive rank: exactly 600 unique pair rows from one checkpoint, SHA-256
  `355c09b59c455a9edb16046f29314ab771c6ce6e71c1a3e25ed63f15f8f00243`;
- measured concentration: 259 pairs for 50% mass, 522 for 90%, Gini
  `0.09654806223979684`, and 19 nonempty direct class-flip strata;
- visual plan: all 262 full panels admitted; 90%-mass contact cohort explicitly
  capped from 522 to 396 pairs. The preserved 396-pair snapshot is
  `2,417,190,464` bytes, SHA-256
  `49028ff18dc33fd42abdd83919da318c012317ef2e2b619a12d159913121d199`;
- certified cleanup removed the observer-owned full-n600 scratch after rank,
  recurring, and visual snapshots became durable. Cleanup receipt SHA-256:
  `6f02f475409b4f2732a66d4a1a262572719be74fb3eb7c2a63b1e32955061a80`;
- first facet line SHA-256
  `cf4e2c6dcb27fbff2ec2fba1baead79c23dfffa9882100ca44309fb35346ccdb`:
  n48 d_seg `0.004533873664008247`, d_pose `124.3037419718479`,
  pair-internal flicker `0.5098681979709202`, excursion `0.0`, exact parsed
  receiver/emitter equality, and exact factor-2 proof. It is not a score.

The observer remains live and is processing the retained checkpoints in order.
Stage-complete rows will add the lossless native-resolution panels and exact
sidecars. MAIN must review the full `base..branch` diff before landing.

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
