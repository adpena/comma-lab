Implemented, committed, launched, and completed the first probe.

Key result: epoch 2’s real RC64 token payload is **118,292 B**, versus the 118,277 B surrogate: **ratio 1.0001268209**, or **+15 B / +0.01268%**. The native decoder reproduced the expected token SHA-256 `9ba2e52b…ea52`.

Continuous-phase IHS1 packing was honestly rejected: `IHS1 pack changed the deployed state-dict schema`. Therefore:

- No model+token total or archive size is claimed.
- No `real_bytes ≈ a·surrogate+b` fit or terminal projection is yet possible.
- The first receipt remains `ADVISORY_MIDRUN_PROBE_NOT_CANONICAL`, `score_claim=false`.

Artifacts:

- [Probe implementation](/Users/adpena/Projects/pact/experiments/ddm_rx2_midrun_serialization_probe.py)
- [Probe result](/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/probes/ep0002/PROBE_RESULT.json)
- [Calibration result](/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/probes/ep0002/CALIBRATION_RESULT.json)
- [Detached launch manifest](/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/probes/detached_cadence/launch_manifest.json)
- [Append-only receipt stream](/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/probes/serialization_probe.jsonl)

The probe retained 2,526 files totaling 1,422,718,135 B. The cadence worker is polling for the next due immutable checkpoint; its done receipt remains pending because it exits only on the three chartered alert conditions. Torch/BLAS were capped at two threads. Both `nice -n 10` and macOS background-policy enforcement were attempted but rejected by the sandbox with `Operation not permitted`.

Commits: `82d82a88f8`, `b1b613e1a0`, `e637aa6e26`, `9b0e5013d5`. Static checks, payload-retention guard, focused tests, two review passes per edit, and serializer SHA gates passed.

Own-vehicle frontier: unchanged at `S=0.1619344578804448 @ 186,269 B [contest-CUDA T4, n600]`; this advisory probe made no score claim.

## NEXT_IF_RESUMED

- `FIRED` — owner: detached `ddm_sp2_rx2_midrun_serialization_probe`; consumer store: `/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/probes/serialization_probe.jsonl`; fire trigger: the next cadence-eligible immutable checkpoint appears, with every-two-epoch probing around the QAT knee.
- `QUEUED-WITH-A-FIRE-ORDER` — owner: MAIN RX2 harvester; consumer store: `/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac/probes/serialization_alert.json`; fire trigger: a real archive falls below 186,269 B, the calibrated terminal projection exceeds the bar, or lossless round-trip fails.

## LIVE-HYPOTHESES

- The first discrete-QAT checkpoint may become IHS1-serializable because the terminal representation is designed for deployed discrete state; this is plausible but untested.
- The token surrogate may remain nearly exact across training: epoch 2 differed by only 15 B, but one checkpoint cannot establish calibration stability.
- A serializable checkpoint may beat the archive bar because its 144,075 B surrogate joint total has substantial nominal headroom; real model/container bytes remain the unresolved term.

## DEAD-ENDS

- A continuous epoch-2 whole-archive price is closed: IHS1 round-trip changed the deployed state-dict schema, so model and archive bytes are unavailable rather than estimated.
- Mid-run scorer evaluation is closed for RX2: fixed losslessly decoded MC36 tokens leave bytes as the only uncertain axis.
- Host scheduling demotion through `nice` or `taskpolicy` is closed for this launch because both were denied by the sandbox; the two-thread cap is the available containment.
- AppleDouble `._epoch_*.pt` sidecars are closed as checkpoint candidates by an explicit discovery guard.