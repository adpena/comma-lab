# ddm_pz4a — sensitivity-allocated coefficient precision pre-proof

`verdict_scope: INSTANCE/FORMULATION`

**Verdict: `REFUTED` at INSTANCE/FORMULATION scope.** On the exact pass-03 LC2/CPR1 coefficient
state, the best of five sensitivity-allocated variable-precision rungs saved **500 B gross** in the
held-basis whole-carrier Brotli-q9 cell, then paid **2,732 B** for the allocation wire: **−2,232 B
net**, which is 4,232 B below the required +2,000 B gate. Even a hypothetical zero-byte allocation
map would leave the measured gross ceiling at 500 B, 1,500 B below the gate. The candidate is dropped
at T7; no receiver, compensation build, scorer run, or evaluator row is owed by this result.

Axis throughout: `[scorer-free exact coder measurement + pass-02-parent Jacobian planning
derivation]`. Every sensitivity and distortion number below is **PLANNING-BAND only**: the Jacobian
producer parent is `b8c3b1187cff48...`, while the selected pass-03 consumer archive is
`93f8d7b4b668...`; the retained freshness receipt says `freshness_ok=false`.

## Gate curve

The baseline is the selected archive's actual coded object, not the 14,528-B NPY file. The exact inner
CPR1 coefficient component is 9,945 B. Because the selected CX2 archive jointly applies Brotli q9 to
the held-basis full carrier, the governing comparison is its exact 23,054-B carrier cell. Positive
deltas are savings; negative deltas are growth.

| Status / axis | Pose-contribution tolerance | Predicted induced contribution | Mean depth (bits) | Raw int16 projection | Inner CPR1 gross | Joint q9 gross | Depth wire | Governing net |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PLANNING-BAND; scorer-free bytes | 0.008300 | 0.008277153 | 7.142500 | 7,971.750 B | −11 B | +422 B | 2,748 B | **−2,326 B** |
| PLANNING-BAND; scorer-free bytes | 0.008675 | 0.008653055 | 7.085139 | 8,023.375 B | −12 B | +452 B | 2,743 B | **−2,291 B** |
| PLANNING-BAND; scorer-free bytes | 0.009050 | 0.009024827 | 7.034861 | 8,068.625 B | −12 B | +468 B | 2,738 B | **−2,270 B** |
| PLANNING-BAND; scorer-free bytes | 0.009425 | 0.009401948 | 6.959167 | 8,136.750 B | −13 B | +495 B | 2,734 B | **−2,239 B** |
| PLANNING-BAND; scorer-free bytes | 0.009800 | 0.009770097 | 6.928750 | **8,164.125 B** | −12 B | **+500 B** | **2,732 B** | **−2,232 B** |

The raw-domain prior was directionally right and operationally misleading: a physical int16-width
calculation does show 7.97–8.16 KB, but the logical signed-int12 projection is only 4.37–4.56 KB, and
neither is a coded result. Exact recoding reverses the inner-component sign.

## Coefficient error handed to any hypothetical compensator

Maximum absolute error is in signed int12 quanta, by coefficient dimension 0–11. This is not rendered
or scorer-measured distortion.

| Status / axis | Tolerance | d0 | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 | d10 | d11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PLANNING-BAND; Jacobian-derived | 0.008300 | 429 | 218 | 413 | 263 | 486 | 385 | 252 | 356 | 427 | 252 | 286 | 137 |
| PLANNING-BAND; Jacobian-derived | 0.008675 | 429 | 218 | 413 | 263 | 486 | 385 | 252 | 356 | 427 | 252 | 286 | 137 |
| PLANNING-BAND; Jacobian-derived | 0.009050 | 429 | 474 | 413 | 279 | 486 | 385 | 252 | 475 | 427 | 318 | 286 | 137 |
| PLANNING-BAND; Jacobian-derived | 0.009425 | 429 | 474 | 413 | 279 | 486 | 385 | 252 | 475 | 427 | 318 | 286 | 178 |
| PLANNING-BAND; Jacobian-derived | 0.009800 | 429 | 474 | 413 | 279 | 486 | 385 | 252 | 475 | 427 | 318 | 430 | 178 |

## Sensitivity and shipped-coder census

`|J|` is the Euclidean norm across the six PoseNet-output coordinates for each pair/dimension. Active
count is out of 600 pairs; each pair contributes exactly three active dimensions.

| Status / axis | Dim | active/600 | median `|J|` | p90 `|J|` | max `|J|` | shipped Rice k | shipped Rice bits |
|---|---:|---:|---:|---:|---:|---:|---:|
| PLANNING-BAND; stale pass-02 parent | 0 | 104 | 0.00182212 | 0.00453959 | 0.00790604 | 9 | 6,689 |
| PLANNING-BAND; stale pass-02 parent | 1 | 109 | 0.00156868 | 0.00389399 | 0.00705599 | 9 | 6,846 |
| PLANNING-BAND; stale pass-02 parent | 2 | 152 | 0.00206318 | 0.00533816 | 0.01095078 | 9 | 6,658 |
| PLANNING-BAND; stale pass-02 parent | 3 | 217 | 0.00258096 | 0.00582716 | 0.01121347 | 8 | 6,089 |
| PLANNING-BAND; stale pass-02 parent | 4 | 164 | 0.00199609 | 0.00523647 | 0.01085177 | 8 | 6,240 |
| PLANNING-BAND; stale pass-02 parent | 5 | 140 | 0.00170583 | 0.00446056 | 0.01123248 | 9 | 6,828 |
| PLANNING-BAND; stale pass-02 parent | 6 | 172 | 0.00236161 | 0.00586021 | 0.01225983 | 9 | 6,506 |
| PLANNING-BAND; stale pass-02 parent | 7 | 124 | 0.00202642 | 0.00495802 | 0.01123910 | 9 | 6,836 |
| PLANNING-BAND; stale pass-02 parent | 8 | 192 | 0.00222466 | 0.00586092 | 0.01406261 | 9 | 6,586 |
| PLANNING-BAND; stale pass-02 parent | 9 | 118 | 0.00162637 | 0.00439100 | 0.01077761 | 9 | 6,646 |
| PLANNING-BAND; stale pass-02 parent | 10 | 112 | 0.00193729 | 0.00468332 | 0.00982586 | 9 | 6,440 |
| PLANNING-BAND; stale pass-02 parent | 11 | 196 | 0.00200764 | 0.00536790 | 0.01205829 | 9 | 6,683 |

## Mechanism verdict

The shipped CPR1 component already delta-codes each dimension and selects per-dimension Rice parameters
(`k=8–9`). Sensitivity-aware rounding of absolute coefficient codes does not make those temporal deltas
cheaper: the exact Rice component grows by 11–13 B at every rung. The outer q9 transform recovers only
422–500 B, while the real PZ4D depth-map race selects Brotli q11 at 2,715–2,731 payload bytes plus its
17-B parse header. Thus the failure is not merely metadata overhead: zero metadata would still miss the
gate. This directly tests and supports the charter's likely refutation mechanism on this exact instance.

The negative does not kill all lossy pose representations. It closes this formulation: independently
rounding the selected `(600,12)` absolute CPR1 coefficient lattice according to the stale-parent
sensitivity waterfill, then recoding through the exact CPR1/Brotli cell.

## Payload custody and reproducibility

Primary result: `/Volumes/APDataStore/pact/ddm_pz4a/retained/preproof_v2/FINAL_RESULT.json`,  SHA-256
`31d77d401d67c699a3559d4cdd4636c13641e412d335c76f586011e6217f0c0e`.

The run retained every baseline and rung payload plus a byte-identical repeat: coefficient NPY, depth
NPY, exact CPR1 coefficient component, complete CPR1 carrier, Brotli-q9 carrier cell, raw nibble depth
map, Brotli-q11 depth stream, LZMA1 depth stream, and selected PZ4D wire. It also retained the complete
allocation checkpoint and per-rung receipts under
`/Volumes/APDataStore/pact/ddm_pz4a/retained/preproof_v2/`.

- `RETENTION_AUDIT.json`: SHA-256 `bb5b6ef03417c16f527eacd12175ed036df9477610ec2e8d7e1449cea8bbb0dc`.
- `stages/01_allocations.npz`: SHA-256 `55c25d2c443b112586de05b6fa3b5893d63f7b8cfac13fa6353a4d3e97d24888`.
- Independent read-only audit: 108 artifact records re-hashed; all five coefficient/depth/CPR1/q9
  parse-backs and gate arithmetic passed.
- Completed resume rerun returned the same `REFUTED`, −2,232-B gate row.
- Command:
  `.venv/bin/python experiments/ddm_pz4a_precision_preproof.py --output /Volumes/APDataStore/pact/ddm_pz4a/retained/preproof_v2 --resume-from /Volumes/APDataStore/pact/ddm_pz4a/retained/preproof_v2/state.json`.

The initial `preproof_v1` payloads remain retained. That run exposed a coarse-to-fine allocation
direction error by overshooting the full tolerance band and producing duplicate near-exact rows; its
`SUPERSEDED.md` points to v2. Nothing was deleted or moved.

Pinned inputs re-verified before both runs:

| Status / axis | Object | Bytes | SHA-256 |
|---|---|---:|---|
| PLANNING-BAND input custody | pass-03 coefficients NPY | 14,528 | `2daec0ae99e86f2a6583a96561335186992a2a1235791af461083ada44d3503d` |
| PLANNING-BAND input custody | pass-03 CPR1 carrier | 23,050 | `a532057d6c786c5e367d83c0a686d7b0c313a7d5b2a2fa6bd2ed7fc47e837684` |
| PLANNING-BAND input custody | pass-03 selected archive | 187,222 | `93f8d7b4b668919d2357a02cde2a96fc0488ec7e2ac00a250f509d27dbef4c6e` |
| PLANNING-BAND stale-parent diagnostic | sensitivity NPZ | 285,478 | `1ac48d8323526729c4ed1d4d507a85e9d22a53a7c0ffeaaedc4e735daed020de` |

## Verification

- `ruff check`: pass.
- `ruff format --check`: pass.
- `py_compile`: pass.
- Focused tests: **6 passed**.
- P0 payload-retention detector: **0 findings**.
- Exact shipped CPR1 decode/re-encode equality: pass.
- Exact selected carrier Brotli-q9 reproduction: pass.
- Two manual review-tracker passes: pass; both files report zero policy violations.

## RECALL EVIDENCE

Recall preceded the decision and covered research, equations, memory, DAG, council, task, and docs
stores with this exact query:

`variable precision pose coefficient sensitivity waterfill CPR1 Rice pz4`

The corpus query searched 8,400 research records, 886 equation records, 915 DAG records, 531 task
records, plus the remaining named stores. I also searched the canonical research indexes and sub-0.15
DAG by content, listed the canonical equation registry, and read the current hot-state/queue rows.

Beyond the charter seeds, four findings changed or constrained the implementation:

1. JS1 Amendment 10 confirms that depth changes must be representation moves priced through the real
   coder, with whole-candidate remeasurement rather than raw-bit arithmetic.
2. LP135 closes the exact same-state F26/ANS and CAP1 lossless races and explicitly leaves only a
   representation- or learned-state change open. PZ4A therefore changed coefficient values and did not
   re-litigate those settled coder rows.
3. OP1R records the exact 23,054-B CPR1 carrier and its existing delta/Rice `k=8–9`; TW1 shows coder
   context makes byte price state-dependent. Together they required complete joint carrier re-encoding
   at every rung rather than summing per-dimension estimates.
4. PH3 says adaptive precision must be raced per stream because earlier coarse-to-fine token rungs were
   dominated; PZ4P/PZ4R show why an envelope is not a receiver or realized score. These prevented an
   adoption claim and kept this result strictly scorer-free/pre-proof.

The canonical registry supplied `score_marginal_lagrange_multipliers_v1` and broader rate-distortion /
waterfill laws, but no exact PZ4A allocation equation was found in the bounded search. The implementation
therefore uses the full retained `(600,6,12)` Jacobian directly and reports its stale-parent boundary.

## Measured, not measured, and boundaries

**Measured:** exact baseline and candidate CPR1 bytes; exact held-basis Brotli-q9 bytes; three real
metadata coder outputs; exact parse-back; all retained SHA-256/byte records; sensitivity summaries;
full-J linearized coefficient-error contribution; raw-width projections explicitly labeled projections.

**Not measured:** PoseNet, SegNet, rendered frames, compensation recovery, terminal-parent sensitivity,
a receiver-framed archive carrying PZ4D, runtime, complete archive size, `d_pose`, `d_seg`, or any exact
contest score. No MPS, GPU, Modal, scorer lane, evaluator, public-PR mutation, upstream edit, or live
solve-tree write occurred. This arm did not move the frontier and did not reach sub-0.15.

Git custody: pending the serializer attempt. The source and focused
test post-edit SHA-256 values are `6a152a0828cc04cf4603b0c313f5bfc67549be9e3eb9ae23fdbe5156cfdcb3a4`
and `05d76ea13b7defbd8032e0bd4b88541c54c3d1604745d7da7d740aad62e825f9`.

## LIVE-HYPOTHESES

- None remain inside the chartered PZ4A formulation. The metadata-free exact gross upper bound is only
  500 B, so neither a better depth-map codec nor compensation can make this representation clear the
  2,000-B byte pre-proof gate.

## DEAD-ENDS

- Absolute-code sensitivity coarsening on this exact pass-03 CPR1 state is closed at
  INSTANCE/FORMULATION scope: best exact gross joint saving is 500 B and best counted net is −2,232 B.
- Raw int16-width accounting is closed as a gate proxy: it predicts 8.16 KB where the exact inner coder
  grows by 12 B.
- Retrying the same-state F26/ANS or CAP1 lossless races is closed by LP135; PZ4A did not produce a new
  state that reopens them.
- The coarse-to-fine v1 allocator is closed as an implementation direction because it overshot the
  tolerance band and collapsed all five rungs; its payloads remain retained only as custody history.

Standing frontier unchanged: own-vehicle LC2 **S = 0.16959899569230852 @ 187,226 B
`[contest-CUDA T4, n600]`**; effective floor CP135 composed **S = 0.16195513827824176 @ 186,252 B
`[contest-CUDA T4, n600]` (ours)**.
