# R1b5 row closer — exact partial verdict

`verdict=PARTIAL_EXACT_R1B5_ROW_BLOCKED_AT_PRODUCTION_SECANT_AND_REPLAY`

`verdict_scope=VJP campaign closure, encoder-side Fisher ordering, exact resize-support coupling audit, n8 conditioned xi0 calibration, and exact carrier byte accounting only; no compiled candidate, no n600 candidate decode, no contest CPU/CUDA score, and no family negative`

`[macOS-CPU advisory]` · `pointer=0.19108 [contest-CPU] UNMOVED` · `MAIN landing review REQUIRED`

## Outcome

R1b5 closed the VJP campaign to `COMPLETE_N600`: 600/600 pairs, no missing IDs, and no effective refusals. All seven preregistered native retries `[11,245,277,482,514,532,574]` completed. The new generic recovery path was exercised by an unexpected refusal at pair 484: it preserved the completed prefix, isolated 484 into a fresh-native singleton, and resumed the tail instead of aborting the chunk.

The row itself is not admissible. The terminal compiler emitted no candidate and names exactly three blockers:

1. `R1B2_RANK4_FIRST_ORDER_REALIZED_SECANT_CUSTODY_ABSENT`
2. `R1B2_FULL_KERNEL_MDL_SELECTION_AND_COMPACT_REPLAY_ABSENT`
3. `VJP_FULL_SIDECAR_REHASH_DEFERRED_UNTIL_PRODUCER_INPUTS_PRESENT`

The third is deliberate gate ordering, not weakened custody: the strict all-sidecar rehash remains mandatory before a candidate can compile, but it no longer rereads roughly 90 GB while cheaper predecessor artifacts are absent. There is no valid weaker score test because boundary bytes do not exist without the rank-4/secant solve, and the receiver cannot produce the final camera bytes without exact full-kernel MDL replay.

## L3 ordering and interaction geometry

The encoder-side artifact orders all 38,077 exact PDW1/n24 realization mismatches in the Fisher/margin chart. It uses `0.5*sech(margin/2)^2`, winner-rival flip distance, exact resize support, edge necessity, and pair-native VJP head arrangements. Counts are 14,538 Road/Lane-edge, 11,643 other-edge, and 11,896 non-edge; 33,886 rows match the native VJP arrangement and 4,191 do not. It is an ordering, not an admission decision: exact prefix-byte marginals and realized backbone secants are still absent.

The exact local coupling audit falsified the predicted path-component structure for the measured 16,319 moderate-margin R2b cells. Every align-corners-false bilinear camera support has four taps, yet all supports are disjoint at this downsampling geometry: the component histogram is `{1: 16319}`. Therefore local resize inversion is 16,319 independent singleton solves; surrogate group-vs-singleton covariance is not applicable. Wider EfficientNet receptive-field coupling remains outside that partition and must be measured by typed realized endpoint secants. A secant is not a derivative.

The downstream solve is now explicitly scorer-native: rank-4 winner-rival hyperplanes, public channel/YUV6 structure, pair-dependent gradients, exact resize range/null coordinates, and curvelet/shearlet residual bases. It must target the max-min-margin Chebyshev center of `(argmax cell ∩ pose tube ∩ uint8 box)`, not mere feasibility. Intrinsic structures are mandatory: compact 38,077-row ordering, union-find over 16,319 cells, rank-4 GEMV, and tiny per-cell atoms—never a 117.96M-pixel ambient mask.

## xi0 coordinate-to-warp calibration

The fit was conditioned correctly: decoded control frame0 was paired with exact source frame1, so the Seg state was fixed before pose descent. Pairs 0–3 were training; pairs 4–7 were held out; every scorer call used the same B8 geometry. The existing positive mapping `(xi0-31)` worsened conditioned held-out `d_pose` from 14.3421116 to 16.5640812. The fitted scalar `shift=round(-0.36*xi0)` realized `-12 px` on all eight prefix rows and improved held-out `d_pose` to 9.9205946. The fitted affine reached 11.1373968.

On the actual decoded receiver prefix, the scalar reduced `d_pose` 150.9269426 → 126.4600577. Every policy retained byte-identical frame1, identical Seg logits, identical per-pair `d_seg`, and aggregate `d_seg=0.0031541188558`. Pose coordinate 0 accounts for more than 99.99% of held-out conditioned squared error. This consumes the exact upstream bilinear RGB→YUV6 chain and is a 33-point realized integer-warp secant measurement, not an RGB intuition or derivative claim.

Production adoption remains blocked on a real Seg-correct n600 boundary state, n600 validation with literal `37×B16+B8` scorer geometry, and contest CPU/CUDA parity.

## Carrier break-even

The settled typed fixture is exactly 2,114 B, 262 B above the 1,852 B conditional cap. Even making xi0 free leaves 1,865 B, so an xi0-only change cannot pass: at least 13 non-xi0 bytes must also disappear.

With the real banked xi0 and zero boundary/replay bodies, the exact reference carrier is 3,223 B: manifest 1,070 compressed, xi0 1,135, boundary 395, replay 179, and ZIP structure 444. The current xi0 body stores 600 float16 coordinates although the selected actuator consumes only five integer shifts (`-13:1, -12:145, -11:440, -10:13, -8:1`). Direct int8 shifts are exactly lossless relative to that selected mapping and the 600-byte body raw-DEFLATEs to 100 B. The empty replay also spends 179 compressed bytes on a derivable JSON envelope.

A concrete compact-binary common descriptor plus direct-shift payload projects to 1,273 B, 579 B under cap. That projection is intentionally `parse_back=false`, `receiver_bound=false`, and `admissible=false`: it proves the layout has sufficient structural headroom, not that a candidate exists. The implementation gate is legacy-compatible v2 codec/parser/receiver binding plus real boundary/replay bytes and n600 calibration. Boundary int8 coefficients + fp16 scales retain the settled mixed-precision shape, but their production error-to-admission tolerance is not yet custodied.

## Evidence

- VJP campaign: `/Volumes/VertigoDataTier/pact/evidence/vjp_custody_20260719/extension_n600_20260720/campaign_receipt.json`, SHA-256 `11d441acec2640f607e87514e35c9244c9d8dfec617a810b2c9763b2e116d284`.
- Fisher/coupling receipt: `/Volumes/VertigoDataTier/pact/evidence/r1b5_row_closer_20260720/fisher_ev/receipt.json`, SHA-256 `0d2b1a103f4830bc70ec9dbe10a3c18a710e980f968a2c0d424d4f7bb2cbce97`.
- xi0 receipt: `/Volumes/VertigoDataTier/pact/evidence/r1b5_row_closer_20260720/xi0_calibration/receipt.json`, SHA-256 `8fb2058d9f71ab6b27909757755cd44d352bcd0b0117cf40dd9563b264fe4fbf`.
- Carrier receipt: `/Volumes/VertigoDataTier/pact/evidence/r1b5_row_closer_20260720/carrier_audit/receipt.json`, SHA-256 `1efa17c7422d710709c7386364675df424abfa44abd183960bed7b87b5c9a7da`.
- Terminal compiler receipt: `.omx/research/r1b5_r1b2_compile_gate_20260720.json`, SHA-256 `d1fd50e8f1dc3feec43d23d39dd1e3773a2e3e7f643b9ca308f0417086d36acd`.
- Machine synthesis: `.omx/research/r1b5_row_closer_receipt_20260720.json`.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`; v7.5/v8 canonical SPECs; R1b4 receiver artifacts and handoff; latest Codex findings/session summaries; latest T3/design memos; `reports/latest.md`; lane registry; subagent progress; master gradient anchors; probe outcomes; active dispatch claims; R1b5 private and broadcast inboxes; exact source and all evidence receipts named above.

## Triality and pointer delta

- DSL: unchanged; no new runtime flag or launch surface.
- DAG: companion R1b5 feed records the closed and blocking edges.
- Equations: no new canonical law; measured results instantiate existing Fisher-margin, exact resize, max-min-margin, and score-byte laws.
- Pointer delta: none. No score, candidate archive, promotion, or GO is claimed.

## MAIN landing requirement

MAIN must review the isolated commit range, artifact hashes, compiler blocker scope, and the claim-ledger closure before cherry-picking. In particular, MAIN must not reinterpret the 1,273 B projection as a compiled archive or the n8 xi0 row as n600/contest authority.
