The real MICRO35 union is terminally refused; no fire order was issued.

| Gate | Required | Measured |
|---|---:|---:|
| Seg gain | ≥35 flips | **35, pass** |
| Archive delta | ≤+29 B | **+44 B, fail by 15 B** |
| Pose delta | ≤5.974×10⁻¹⁰ | **+5.263×10⁻⁹, fail by 8.81×** |
| Receiver parse-back | exact | **pass** |

The deterministic archive and repeat are both 186,296 B with SHA-256 `ca0e2e785ff65260d63673bb8a734cfbe835b345395ad2bea19523f4c94ec4f1`. All intermediate payloads, scorer fields, checkpoints, and exact-object compensation receipts are retained under `/Volumes/VertigoDataTier/pact/ddm_mc35_20260814/`.

No Modal, Metal, or exact contest evaluation was launched. The frontier remains `S=0.16195513827824176`, 186,252 B `[contest-CUDA]`.

Artifacts:

- [Evidence memo](/Users/adpena/Projects/pact/.omx/research/ddm_mc35_micro35_union_build_20260814.md)
- [Resumable builder](/Users/adpena/Projects/pact/experiments/ddm_mc35_micro35_union_build.py)
- [Retained archive](/Volumes/VertigoDataTier/pact/ddm_mc35_20260814/micro35_candidate/archive.zip)
- Commit: `538d28f55169b70452d1f4ffeeb2f7f4936aee9e`

Verification passed lint, bytecode compilation, payload-retention enforcement, archive integrity, exact parse-back, deterministic repeat, two review-tracker passes, and serializer post-commit hash verification.

## NEXT_IF_RESUMED

- **Disposition:** QUEUED-WITH-A-FIRE-ORDER. **Owner:** MAIN under a fresh charter. **Consumer store:** `/Volumes/VertigoDataTier/pact/ddm_mc35_successor_pair105/`. **Fire trigger:** first produce an archive ≤186,281 B, then fresh-solve pair 105; dispatch T4 only if every original MICRO35 gate passes.
- **Disposition:** QUEUED-WITH-A-FIRE-ORDER. **Owner:** MAIN under that successor charter. **Consumer store:** `/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532/`. **Fire trigger:** build the seven-object union without pair 532 and proceed only if the receiver-closed object retains ≥35 flips and recovers the 15-byte deficit.

## LIVE-HYPOTHESES

- Pair 105 is the pose bottleneck: it contributes `+5.6822×10⁻⁹`, while the other seven pairs collectively improve pose slightly.
- Removing pair 532 may raise Seg gain from 35 to 37 and reduce rate, but this requires a newly built receiver-closed object.
- A joint overlay/container representation may recover the remaining 15 bytes; this has not been tested on the exact union.

## DEAD-ENDS

- Exact evaluation of the current archive is closed because two mandatory local gates failed.
- Additive QS2 + RE1 + HP4 projections are closed as evidence; the built object measured +44 B and non-additive pose behavior.
- Reusing stale QS2 compensation is closed; all eight final objects required fresh fingerprint-bound solves.
- Calling the 35-flip result frontier progress is closed; the pointer did not move.