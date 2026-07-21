# Task #578 S4 archive composer — Codex findings

UTC: 2026-07-21T12:50:20Z  
Lane: `lane_s4_archive_composer_578_20260721`  
Authority: research-only; MAIN landing review required; pointer `0.1910828242 [contest-CPU]` unchanged.

## Outcome first

A self-contained, one-member archive vehicle now exists. The exact `archive.zip`
is 451,191 bytes (`d84f2fe0...96ed`); its sole `0.bin` is 1,285,943 bytes
(`595e69d4...955b`). The receiver has no `tac` imports, scorer weights, GT tables,
source video, or video-derived sidecars, and uses only NumPy and Brotli beyond the
standard library.

Independent repository-native replay and the standalone runtime are byte-identical
at n16, n64, and n600. Two fresh standalone n600 decodes both emitted stream SHA-256
`01f45813...83f8` for 3,662,409,600 bytes.

## A1–A4 verdicts

| Gate | Verdict | Evidence |
|---|---|---|
| A1 monolithic builder | PASS | deterministic ZIP; one `0.bin`; strict length/hash parse-back; real ZIP stat charged |
| A2 standalone tree | PASS | no repo/scorer/GT imports; atomic output; PPCS/base/event/component/xi/lane/factor-2 replay; exact #557 static range twin |
| A3 parity | PASS | exact n16/n64/n600 repo-native parity and standalone double-decode determinism |
| A4 local advisory | MEASURED, NON-PROMOTABLE | `[macOS-CPU advisory]`: d_seg 0.60198647, d_pose 163.11865234, 451,191 bytes, score 100.89; evaluate 351.01 s |

The A4 negative is narrowly scoped to the current merged sections and the current
same-frame realization. It does not reject the representation family or the hot-swap
composer. It does establish that this exact realization is not a candidate: the
frame-0 pose carrier is absent and semantic fidelity is far from frontier quality.

## Fresh-eyes correction that became a guard

The first independent parity replay rejected the standalone lane raster. The
standalone draft used non-authoritative camera constants and omitted the native
half-width floor. The final receiver reads the camera constants from counted
video-derived manifest metadata and matches the native lane raster exactly. This is
now covered by the full n16/n64/n600 output parity gate.

## Cleanliness and custody

The constant audit classifies every shipped constant on the required three-way
spine. Generic interpreter geometry and the LZMA filter are uncounted; lane camera
intrinsics and the LBND2 render header are video-derived and counted; the public
SegNet-derived R2 palette is weight-derived and counted. Inflate never loads a
scorer.

The 3.66 GB evaluation raw was created only on the SSD tier, hash-certified as
`01f45813...83f8`, evaluated, and deleted after success. The rebuild command,
archive hash, byte count, and reason are preserved in the cleanup receipt.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, v7.5 §8 operating contract, and v8 spec.
- `.omx/state/lane_registry.json`, `.omx/state/subagent_progress.jsonl`, both live inboxes.
- The merged PPCS, predictor-R3 base/event/component receipts and exact SSD bytes.
- `upstream/evaluate.py` and the frozen local evaluator fixture.

## Pointer delta and triality

- Pointer delta: none.
- DSL leg: no new scoring lever; selected bytes are versioned registry inputs, and
  future policy values must arrive in counted typed sections.
- DAG leg: `s4_archive_composer_DAG_FEED_20260721.md`.
- Equation leg: exact factor-2 support fill plus the contest action decomposition
  recorded in the measurement receipt; no proxy was promoted.
