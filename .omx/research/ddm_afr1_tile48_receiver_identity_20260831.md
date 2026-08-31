# AFR1: `tile48_groupbin8` native receiver identity

Date: 2026-08-31

Owner: `ddm_afr1`

Verdict: **PASS / QUEUED-WITH-A-FIRE-ORDER**

Authority: `[macOS-CPU full receiver identity / no score claim]`

## Conclusion

The address-free `tile48_groupbin8` interaction is portable to the staged generation-23 native C receiver. The Python and C context functions agree at every image coordinate, and the native receiver decoded the exact 180,002-byte AFC1 candidate into the same 3,662,409,600 raw bytes as the retained Python-reference path over all 600 pairs. The prior-law prediction survived its falsifier.

This is receiver closure, not a score. No scorer ran, no Modal call was made, no authority axis was inferred, and the canonical frontier did not move. MAIN owns the remaining byte-close revalidation, two content-derived seals, and the sequential contest-CUDA/contest-CPU pair recorded in the retained fire order.

## Control reproduction came first

The first executed stage revalidated AFC1 from its retained physical artifacts before the native port was trusted.

| `[macOS-CPU scorer-free retained physical byte control]` | LB1 control | `tile48_groupbin8` candidate | deterministic repeat | delta |
|---|---:|---:|---:|---:|
| token stream | 113,492 B, `8838e44f…` | 113,411 B, `5601d6fd…` | 113,411 B, `5601d6fd…` | **-81 B** |
| `archive.zip` | 180,083 B, `5b856e66…` | 180,002 B, `cbb8d928…` | 180,002 B, `cbb8d928…` | **-81 B** |
| decoded symbols changed | 0 | 0 | 0 | **0** |
| pure-rate score delta | — | — | — | **-0.00005393457520289588** |

The source adjudication and manifest matched their charter pins exactly: `9bda316e278e6bf37e762c6c1308cc014db2f76703ce327eef0bad064b6ed841` and `1e8a111e8f5d010d67ac34e212a81370341743c4e9ab148c14b2ceb22425a425`. Control free space was 8,964,206,592 B before capture and 8,931,266,560 B afterward, above the 8,142,450,560 B identity floor. Streams, archives, per-frame ledgers, encoder states, and 25-frame checkpoints were copied to the Vertigo store before the port proceeded.

Authority receipt: `/Volumes/VertigoDataTier/pact/ddm_afr1_tile48_receiver_identity/measurement_v1/CONTROL_REPRODUCTION.json`.

## RECALL EVIDENCE

The bounded recall queried `tile48_groupbin8`, `tile48`, `groupbin8`, `native C receiver`, `receiver identity`, `RC64`, and `address-free` across `.omx/research/`, `experiments/`, `runtime-rs/native/`, the research index and DAG, the hot state, and the task ledger. `tools/list_canonical_equations.py --json` was also consulted for governing score/identity relations.

The recall found two load-bearing predecessors beyond the charter text:

- AFC1 is the Python authority and retained full-n600 joint re-encode: `.omx/research/ddm_afc1_address_free_census_20260831.md` and `experiments/ddm_afc1_address_free_census.py`.
- LB1 is the exact generation-22 native receiver precedent: `.omx/research/ddm_lb1_banked_lossless_joint_collect_20260829.md`, `experiments/ddm_lb1_banked_lossless_joint_collect.py`, and `runtime-rs/native/f26-corrector/gb1_20family/README.md`.

The bounded scopes did not contain another current `tile48_groupbin8` native port. AFR1 therefore derived generation 23 from LB1's exact C source (`ad8cc276…`, 49,539 B), copied AFC1's exact Python authority (`6462ba51…`, 31,470 B), and added only the named rule. The staged C source is `3e2705f5…`, 50,299 B. Runtime pin consistency and the Python/native configuration gate both passed.

## Python/C parity

The context calculation was tested exhaustively, not sampled:

```text
tile48    = ((y // 64) * 8 + (x // 64))
groupbin8 = (((x % 64) + 2 * (y % 64)) * 8) // 190
context   = tile48 * 8 + groupbin8
```

| `[macOS-CPU exhaustive context parity / no score claim]` | result |
|---|---:|
| coordinate denominator | **196,608 / 196,608** |
| mismatches | **0** |
| context cells observed | **384 / 384**, spanning 0 through 383 |
| Python context payload | 393,216 B, `630950a3…` |
| native context payload | 393,216 B, `630950a3…` |

Both deterministic, loadable native builds are 34,576 B with SHA-256 `99454f9c71ba3d1c47cff383c30a7fecbce4e313b1f2194c316b6fa4f4e8b66e`. The parity receipt is `/Volumes/VertigoDataTier/pact/ddm_afr1_tile48_receiver_identity/measurement_v1/CONFIGURATION_PARITY.json`.

## Full receiver identity

The sole full run invoked the staged `inflate.sh` with `NativeFreeCorrector` and the Python HPAC token decoder. It retained the candidate stream, archive, native sources/builds, token checkpoint, logs, and raw output.

| `[macOS-CPU full receiver identity / no score claim]` | result |
|---|---:|
| pairs | **600 / 600** |
| candidate archive | **180,002 B**, `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25` |
| decoded tokens | **117,964,800 / 117,964,800**, `cc10a7b0…` |
| raw output bytes | **3,662,409,600 / 3,662,409,600**, `7246a4ff…` |
| bytes differing from retained Python-reference raw | **0 / 3,662,409,600** |
| candidate stream binding | **PASS**, `5601d6fd…` |
| archive binding | **PASS**, `cbb8d928…` |
| corrected CDF-input field | `370a5e2a…` |
| corrected quantized-logit field | `8269fe1a…` |
| token decode | 431.9605 s |
| decode plus render | 694.6181 s |
| wrapper wall time | 698.3239 s |

The explicit byte-close revalidation also passed all six gates: control, port, exhaustive parity, full identity, deterministic build, and receiver pin consistency. Receipts:

- `/Volumes/VertigoDataTier/pact/ddm_afr1_tile48_receiver_identity/measurement_v1/NATIVE_IDENTITY.json`
- `/Volumes/VertigoDataTier/pact/ddm_afr1_tile48_receiver_identity/measurement_v1/BYTE_CLOSE_REVALIDATION.json`

## Retained custody

The retained root is `/Volumes/VertigoDataTier/pact/ddm_afr1_tile48_receiver_identity/`. Its manifest records 124 files and 3,815,662,560 B, with 4,776,378,368 B free after capture. The raw payload, full token checkpoint, candidate and repeat streams/archives, successful repeat builds, failed build diagnostics, sources, logs, and receipts remain present. Nothing was deleted or moved after measurement.

Manifest: `/Volumes/VertigoDataTier/pact/ddm_afr1_tile48_receiver_identity/measurement_v1/MANIFEST.json`, SHA-256 `c792611b304201d39860aa83e8a2f5f339ff8039816ecf6a34f6fa23563dbe97`.

## Typed MAIN fire order

Disposition: **QUEUED-WITH-A-FIRE-ORDER**.

Owner: **MAIN sole scorer-lane router**.

Consumer store: `/Volumes/VertigoDataTier/pact/ddm_afr1_tile48_receiver_identity/fire_main/`.

Fire trigger: the AFR1 serializer landing exists; the effective pointer is still LB1 archive `5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9`; no scorer or duplicate lane is active; MAIN records the two named axis claims; the byte-close receipt and both seals revalidate from disk.

The exact commands and lane IDs are machine-readable in `/Volumes/VertigoDataTier/pact/ddm_afr1_tile48_receiver_identity/measurement_v1/MAIN_FIRE_ORDER.json`. Order is:

1. Re-run the scorer-free byte-close gate.
2. Create the contest-CUDA seal with `tools/make_candidate_seal.py`.
3. Create the contest-CPU seal over the same archive/runtime identity.
4. Fire CUDA first, then CPU, through `tools/fire_modal_auth_eval.py` with distinct claims and pair group `ddm_afr1_tile48_groupbin8_dual_axis`.

No SHA is accepted from a hand-typed seal argument. The candidate's lossless rate-only projection is `S = 0.14797617125559107`, but that is **UNSCORED and non-authoritative**; only the retained dual-axis exact rows can admit or refute it.

## Denominator and authority line

Control: **600/600 frames and 117,964,800/117,964,800 tokens**; candidate/repeat archive and stream identity passed; **0 changed symbols**. Parity: **196,608/196,608 coordinates**, **0 mismatches**. Receiver identity: **600/600 pairs**, **117,964,800/117,964,800 tokens**, and **3,662,409,600/3,662,409,600 raw bytes**, with **0 differing raw bytes**. No scorer denominator exists because no scorer ran.

## LIVE-HYPOTHESES

- **INSTANCE:** the exact contest-CUDA row will preserve LB1's measured distortion and realize the full 81-byte rate credit, because the candidate archive is 81 B smaller and its native full-receiver raw bytes are identical to the Python/LB1 reference. This remains untested until MAIN fires the sealed row.
- **INSTANCE:** the Linux contest-CPU receiver will remain byte-identical and finish within budget, because the portable integer context function is exhaustive-parity clean and the macOS CPU full receiver completed in 698.3 s. Linux build/runtime behavior remains untested.

## DEAD-ENDS

- **INSTANCE closed:** a repeat build whose output name leaks into the dylib identity is not a deterministic-build proof; the two hashes differed. The fixed install name cured it.
- **INSTANCE closed:** `-Wl,-no_uuid` makes deterministic Mach-O bytes but produces an unloadable dylib on this host (`missing LC_UUID load command`). Those failed artifacts were retained and are excluded from the receiver proof.
- **FAMILY closed on inherited evidence:** do not sum another banked lossless delta without a joint re-encode; #1269 showed non-additivity.
- **FAMILY closed on inherited evidence:** do not retry the reorder, coder, alternate decode-derived conditioning, or packet/tree-shake axes. RR9, JT23/OC2, and DCF1 already closed them in their measured scopes; AFR1 admitted only this native-port/identity exception.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: `MAIN sole scorer-lane router`; consumer store: `/Volumes/VertigoDataTier/pact/ddm_afr1_tile48_receiver_identity/fire_main/`; fire trigger: serializer landing present, LB1 still the effective pointer, no active scorer/duplicate lane, named CUDA and CPU claims recorded, and byte-close plus both disk-derived seals valid; action: execute the retained CUDA-first then CPU fire order exactly once per axis.
- **ADMIT-OR-REFUTE** — owner: `MAIN`; consumer store: the two `fire_main/` axis result stores plus the canonical frontier pointer; fire trigger: both exact rows are terminal and harvested; action: recompute each score from `d_seg`, `d_pose`, and archive bytes, admit only a qualifying lower exact row, otherwise record the typed falsifier without promoting the projection.

[contest-CUDA T4 n600] own-vehicle frontier: LB1 — S=0.14803010583079396, archive=180,083 B, d_seg=0.00020139, d_pose=6.37e-6, SHA-256=5b856e667961dd9ab68ddd7166384662bfb5912fabc8c9270098ea63a8ad28c9.
