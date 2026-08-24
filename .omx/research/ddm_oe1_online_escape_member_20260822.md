# ddm_oe1 — the zero-stored causal escape member loses 10,818–12,305 real bytes

## Outcome

**MEASURED — `[macOS-CPU advisory / scorer-free exact RC64 byte measurement]`, full n600.**
The canonical zero-stored causal uniform escape member does not recover the DX2 anti-predicted
mass. All four noncontrol responsiveness rungs grow the real RC64 stream. The least-bad rung,
`escape_w64`, is **124,595 B**, or **+10,818 B** against the shipped **113,777 B** stream.
It recovers **2,303.197892 B** on the exact 93,580 anti-predicted positions but spends
**13,121.849451 B** on the other 117,871,220 positions, for selectivity **0.175524**. The
prior-law prediction required a positive real saving above 7,993 B and selectivity above 1.5;
the measured sign and ratio both falsify it.

The adaptation-zero positive control is byte-identical to the retained shipped stream:
**113,777 B**, sha256
`e2af55e641c4f2d3c1f81d75af2ce0453dd44263ac3cbd84f129eadf7b8a4ac5`.
Every rung independently decodes all **117,964,800 / 117,964,800** tokens to TO2 sha256
`cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`.

This is a completed negative, not a shipping candidate. The exact frontier did not move.

## Real n600 curve

All row sizes are retained physical RC64 streams. “Recovered” and “spent” are the signed sums
of exact RC64 integer-frequency code-length changes on the two fixed AE1 sets. The final net
adjudication is always the physical stream stat, not the model sum.

| rung | history window | real stream | real saving vs 113,777 B | recovered on 93,580 | spent on 117,871,220 | selectivity | rate-only delta S |
|---|---:|---:|---:|---:|---:|---:|---:|
| `control_w0` | 0 | **113,777 B** | **0 B** | 0 B | 0 B | n/a | 0 |
| `escape_w1` | 1 frame | 126,082 B | **−12,305 B** | 864.413035 B | 13,170.064928 B | 0.065635 | **+0.0081933944** |
| `escape_w4` | 4 frames | 125,509 B | **−11,732 B** | 1,486.242636 B | 13,218.995278 B | 0.112432 | **+0.0078118572** |
| `escape_w16` | 16 frames | 124,898 B | **−11,121 B** | 2,026.623439 B | 13,147.542772 B | 0.154145 | **+0.0074050174** |
| `escape_w64` | 64 frames | **124,595 B** | **−10,818 B** | **2,303.197892 B** | **13,121.849451 B** | **0.175524** | **+0.0072032622** |

The rate-only delta uses TX1’s fixed exchange rate
`25 / 37,545,489 = 6.658590e-07 S/B`. Negative “saving” means growth: w64 consumes another
**25.5250%** of the 42,382 B shedding demand instead of satisfying any of it. Its recovered
2,303.197892 B is only **8.6440%** of AE1’s 26,645.297908 B gross ceiling.

Finite-coder reconciliation is small and sign-safe:

| rung | integer-frequency model saving | real saving | real minus model |
|---|---:|---:|---:|
| w1 | −12,305.651893 B | −12,305 B | +0.651893 B |
| w4 | −11,732.752642 B | −11,732 B | +0.752642 B |
| w16 | −11,120.919334 B | −11,121 B | −0.080666 B |
| w64 | −10,818.651559 B | −10,818 B | +0.651559 B |

The loss is therefore the probability law, not terminal RC64 rounding.

## AE1 reproduction before the sweep

OE1 reused AE1’s complete field read-only, copied it to the charter-authorized local retention
tier, and reproduced the charter pins before any RC64 rung ran:

- **93,580 / 117,964,800** positions have positive excess over `log2(5)`;
- **117,871,220 / 117,964,800** positions were already below uniform;
- gross excess is **213,162.38326091738 bits = 26,645.297907614673 B**;
- retained excess field: **943,718,400 B**, sha256
  `45f94cdaeeda86a7f4e467af1f182c73a2c5de76d08ed7c0a22c3b0f8af879ed`;
- retained packed mask: **14,745,600 B**, sha256
  `a1fadb5a966343f79649dcd4af892e373868bb93cf6ab2347fd1f3ef4a274d18`.

The result is `PASS_BEFORE_SWEEP`; the 93,580-set joined the shipped RC64 integer frequencies
exactly at every encoded group. No BL1, AE1, MS9, MST1, or LD1-owned tree was modified.

## The member that actually ran

The shipped 19-member HPAC/FreeCorrector law, its 190 groups, RC64 coder, group map,
serialization order, table correction, addressing, and all existing online state remain fixed.
OE1 adds one uniform expert. For a position in the already-existing
`group × (boundary bucket, predicted class)` cell, its weight is

`alpha = min(0.5, anti_predicted_count / seen_count)`

over the completed prior 1, 4, 16, or 64 occurrences of that cell’s group. A cold cell has
`alpha = 0`, so it nests the shipped law exactly. The current group’s decoded symbols update
the rolling counts only after coding. Encoder and decoder both begin with zero state and derive
the same state solely from already-decoded token history.

This is the reference zero-stored causal formulation:

- stored video-derived parameter bytes: **0**;
- new non-token descriptor bytes: **0**;
- other shipped mixture members changed: **0**;
- generic estimator replacement: **no**;
- lookahead, target access at decode, side channel, or transmitted history: **none**.

The slow-window trend is monotone but not near a sign change: w64 still spends **5.697×** what
it recovers. Extending the history merely approaches the already-dead static member; it does
not supply evidence for another fire.

## Zero-storage and receiver proof

The exact DX2 stored member is **180,268 B**. Its token stream is the trailing 113,777 B, and
the retained non-token prefix is **66,491 B**, sha256
`0e2dd639e50795a00a3013f1ba66efa06495ed7b0a2ea6bbd920aa50b4ad1877`.
Every OE1 member is this byte-identical prefix followed only by that rung’s real RC64 stream.
Thus member growth equals stream growth and non-token descriptor growth is exactly zero.

The w64 retained member is **191,086 B**, sha256
`cec655f5715bd6b1174c8d01a8d799cf1d934f9e1ae1951791c24d1525ada4af`;
its retained stream is **124,595 B**, sha256
`1f45a4bdd59dfec6e75c5ed052e2e26600514e3ed461abd73924e92f6ae2ef3b`.
It is evidence, not a candidate archive.

The encoder and decoder each preserved **30 / 30** distinct 20-frame checkpoints. Each encode
checkpoint contains the complete structurally captured shipped corrector, the five causal
history states, the prior decoded frame, and all five RC64 interval snapshots. Each decode
checkpoint retains the same receiver state, all five RC64 decoder states, and all five decoded
stage payloads. The full decoded field is retained separately for every rung; every sha is the
TO2 pin above.

## Prediction, falsifier, and verdict scope

The registered prediction was:

- real recovery above **7,993 B**, more than 30% of AE1 gross and about 18.9% of demand;
- selectivity above **1.5** at the best response rate.

The registered falsifier was every rung net nonpositive or selectivity below 1 everywhere.
**Both falsifiers landed:** all four physical streams grow by 10,818–12,305 B and all four
ratios are 0.0656–0.1755.

**Verdict scope — FAMILY:** fixed-DX2 anti-predicted uniform-member routes, combining AE1’s
measured stored/static forms with OE1’s zero-stored causal recent-escape responsiveness sweep
over `{1, 4, 16, 64}`. The already-cheap 99.920671% of positions dominates the recoverable
0.079329% even when signalling cost is exactly zero. This closes the anti-predicted uniform
escape member family on this fixed DX2 probability object. It does not claim that every legal
content-aware learned probability program is impossible.

No follow-on is fired or queued. A longer-window rerun is folded into the static limit already
measured by AE1; a stored flag/weight rerun is folded into AE1’s negative; a generic replacement
is folded into EF1/CX3; an order/addressing/coder rerun is folded into TO2/AD2/RB1.

## Authority and score boundaries

- Axis: `[macOS-CPU advisory / scorer-free exact RC64 byte measurement]`.
- The physical stream byte counts and token identity are measured; no archive was submitted.
- No scorer ran. No DALI-GT field was read. OE1 makes **no d_seg or d_pose claim**.
- Losslessness is against the retained TO2 decoded-token field, not a distortion measurement.
- The rate-only deltas are arithmetic applications of `25/37,545,489`, not exact score rows.
- No shipping candidate exists and the pointer is unchanged.

The old common-contract `qo1` frontier paragraph was stale for this run. Live authority was
re-read from `.omx/state/main_hot_state.md`: DX2 remains
**S = 0.14821987563243377 @ 180,368 B [contest-CUDA T4, n600]**, archive sha256
`976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`.

## Retention and reproducibility

The charter’s explicit local-disk opt-in was used because both SSD tiers remained at 100%.
OE1 wrote no new bytes under `/Volumes/*`.

- root: `/Users/adpena/Projects/pact/.omx/tmp/arm_receipts_local/ddm_oe1_online_escape_member/`;
- `RESULT.json`: **20,454 B**, sha256
  `15b3894b6a9355ce9356f091f8b0df345b44ef945bd051b388733519b71d849b`;
- `MANIFEST.json`: **173,571 B**, sha256
  `f195d951a79dc1e1ecbb26a29192dbb2b0246ac11b24471b2f28488ac4af164a`;
- manifest: **633 artifacts / 2,222,452,394 B**;
- post-run manifest rehash: **PASS for all 633 / 633 artifacts**;
- post-run local free space: about **476 GiB**;
- launch config records argv, cwd, interpreter, no RNG, 4 torch threads, and local-only writes.

The pre-commit secret scanner classified source line 66’s pinned public TO2 decoded-token
SHA-256 as a `generic-api-key` false positive. The hook’s deliberate `TAC_SECRETS_WAIVE=1`
exception was used rather than changing the measured source after its implementation hash was
bound into every receipt. This was not `REVIEW_GATE_OVERRIDE`; both required Python review
passes remained enforced.

No listed payload may be removed or moved without replacement custody evidence.

## RECALL EVIDENCE

Searches were corpus-wide rather than charter-only:

- `.omx/research/`, `experiments/`, `src/`, arm receipts, and runtime sources with content queries
  `anti-predicted|anti predicted|escape member|uniform member|online mixture|causal mixture|KT|PPM`,
  plus the exact 93,580 / 213,162 / 113,777 pins;
- `.venv/bin/python tools/list_canonical_equations.py --json`, filtered for token rate,
  direction dependence, entropy, mixture, and context laws;
- `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks, design docs, SPECs, task ledgers,
  lane registries, active claims, and checkpoints with `OE1|AE1|escape|uniform|HPAC|RC64`;
- read-only receiver and encoder surfaces in DX2, BL1, JG2, MA1, AE1, TO2, EF1, CX3, AD2,
  and TX1.

Findings beyond the charter’s seeds changed the execution in three ways:

1. Canonical equation `token_rate_model_direction_dependence_v1` says modelled token prices can
   be directionally wrong. OE1 therefore adjudicated every rung by a retained real RC64 stream
   and reported the finite-coder residual rather than treating selectivity sums as bytes.
2. MA1 proved the live DX2 object already carries zero-stored causal within-miss state. OE1
   extended that exact 19-member object instead of replacing it and kept its other state fixed.
3. JG2’s prior resume defect showed that hand-picked corrector keys silently lose live mixer
   state. OE1 used JG2’s structural capture/detector and preserved 30 encode plus 30 decode
   stage checkpoints.

No prior implementation or retained real-byte sweep of this same zero-stored causal uniform
escape member was found in those scopes. AE1 had correctly left it `UNKNOWN_NOT_BUILT_OR_REPLAYED`.

## Dead ends consumed

- Stored flags: AE1 measured a **−103,582.702092 B** net ceiling; not retried.
- Static uniform overlays: AE1 measured **−14 B** global and **−34.468770 B** group190; not retried.
- Generic context estimators: EF1’s best is **3.21086×** the incumbent density; not substituted.
- Named summaries: CX3 supplies **0 B** against DX2; not reopened.
- Token order: TO2 alternatives are **196.07%–686.94% worse**; order stayed fixed.
- Addressing: AD2 shows DX2 addressing is already implicit/free; no descriptor dodge was used.
- Probability-only distortion: LX2’s lesson is honored; no d_seg claim is attached.
- Causal uniform escape: OE1’s four real streams all grow and all selectivity ratios stay below
  0.176; closed at the verdict scope above.

**Own-vehicle frontier unchanged: S = 0.14821987563243377 @ 180,368 B [contest-CUDA T4, n600].**
