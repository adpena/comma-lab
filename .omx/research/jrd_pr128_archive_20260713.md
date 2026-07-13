# JRD PR128 intake archive coefficient-prefix completion

## Verdict

`grammar_compat=PASS`, `parse_back=PASS`, `result=NULL`, `recommendation=NO-GO`,
`score_claim=false`, `pointer_moved=false`, `research_only=true`.

**VERDICT-SCOPE:** INSTANCE — the archive bytes with `[MEASURED]` SHA-256
`196acd18e4ca10a3ab0d826436aa46014a44cba8a55eb4abf9931876cc7e98b5`, the
PR110 FP11/CTXR/FECa/DQS1 runtime named below, and the uniform plus Laplace-dead-zone
prefix families over the signed-int8 decoder tensors. This NULL does not kill JRD, learned
rate control, non-prefix coefficient changes, latent sections, selectors, sidecars, or a
different archive instance.

This is a borrowed-PR128-click bank/reference result. It is not a submittable pointer move and
has no promotion authority.

## Grammar compatibility and custody

| Surface | Result |
|---|---|
| Source archive | `[MEASURED]` `experiments/results/pr128_click_import_forensics_20260712/import_candidate_archive.zip` |
| Source archive bytes | `[MEASURED]` `176564` |
| Source archive SHA-256 | `[MEASURED]` `196acd18e4ca10a3ab0d826436aa46014a44cba8a55eb4abf9931876cc7e98b5` |
| Runtime grammar | `[MEASURED]` PR110 FP11/CTXR/FECa/DQS1 parse PASS |
| Signed-int8 decoder tensors parsed | `[MEASURED]` `28` |
| Signed-int8 decoder coefficients parsed | `[MEASURED]` `228958` |
| No-op decoder re-encode | `[MEASURED]` byte-identical PASS |
| Eligible surface | `[MEASURED]` signed-int8 decoder q-tensors only |
| Preserved outside the search | `[MEASURED]` affine uint8 latents, fp16 latent header, latent sidecar, FECa selector, DQS1 tail, and fp16 decoder scales |

**Grammar-compat verdict:** PASS. The adapter parsed the expected decoder grammar, derived all
runtime tensor shapes/order/maps, inverted each byte map exactly, and reproduced the complete
archive bytes with a no-op decoder re-encode. No non-int8 section was reinterpreted.

## Sealed exact-through-R baseline

Authority is `[macOS-CPU advisory]` using the archive's own deterministic CPU receiver, NumPy
reference arrays, frozen local CPU-torch SegNet/PoseNet, and the exact resize/uint8 path. No MPS
surface was used, and the command did not call `upstream/evaluate.py`.

| Quantity | Baseline | Positive repeat | Negative control |
|---|---:|---:|---:|
| Evaluated pairs | `[MEASURED] 600` | `[MEASURED] 600` | `[MEASURED] 600` |
| Archive bytes | `[MEASURED] 176564` | `[MEASURED] 176564` | `[MEASURED] 15572` |
| d_seg | `[MEASURED] 0.0005336422402857958` | `[MEASURED] 0.0005336422402857958` | `[MEASURED] 0.5048244815568129` |
| d_pose | `[MEASURED] 0.000029371462599859417` | `[MEASURED] 0.000029371462599859417` | `[MEASURED] 160.75346749623617` |
| Rendered raw bytes | `[MEASURED] 3662409600` | `[MEASURED] 3662409600` | `[MEASURED] 3662409600` |
| Rendered raw SHA-256 | `[MEASURED] 465781ba6f334ae4837ad99de8b8787e1d5f312553cff6d75364f888c1007a2f` | `[MEASURED] 465781ba6f334ae4837ad99de8b8787e1d5f312553cff6d75364f888c1007a2f` | `[MEASURED] 300680725d8a1eb471e031363d74c4b25080ae5dcb64aa59881e8ae2306103f2` |

The positive repeat is `[MEASURED]` exactly equal on archive bytes/SHA, raw bytes/SHA, d_seg,
and d_pose. The all-decoder-zero negative control is `[MEASURED]` distinct on both d_seg and
d_pose, so the component meters are responsive.

The advisory score decomposition is:

- Seg term: `[DERIVED] 100 * d_seg = 0.05336422402857958`.
- Pose term: `[DERIVED] sqrt(10 * d_pose) = 0.01713810450425`.
- Rate term: `[DERIVED] 25 * 176564 / 37545489 = 0.11756672019906306`.
- Advisory S: `[DERIVED] 0.18806904873189262`.

`score_claim=false`: this is not a fresh contest-CPU or contest-CUDA evaluation.

## Exact control law

For a proposed nested prefix packet `p`, admission required all of:

- receiver parse-back PASS;
- `[MEASURED] bytes(p) < [MEASURED] bytes(baseline)`;
- `[MEASURED] d_seg(p) <= [MEASURED] d_seg(baseline)` and
   `[MEASURED] d_pose(p) <= [MEASURED] d_pose(baseline)`.

After each possible admission, the machinery was required to rebuild and remeasure the complete
accepted replacement set. No proposal reached that stage because none passed the strict
single-tensor byte-decrease plus component-safe screen.

## Admissions table

The search measured `[MEASURED] 448` prefix candidates: `[DERIVED] 28` tensors times
`[DERIVED] 2` registered prefix families times `[DERIVED] 8` nested planes. The receipt contains
`[MEASURED] 56` tensor-family summaries.

| Decoder tensor | Families measured | Byte-smaller component-safe choices | Equal-byte component-safe last planes | Combined admissions |
|---|---:|---:|---:|---:|
| `skips.2.weight` | `[MEASURED] 2` | `[MEASURED] 0` | `[MEASURED] 0` | `[MEASURED] 0` |
| `refine.1.weight` | `[MEASURED] 2` | `[MEASURED] 0` | `[MEASURED] 0` | `[MEASURED] 0` |
| `blocks.2.bias` | `[MEASURED] 2` | `[MEASURED] 0` | `[MEASURED] 0` | `[MEASURED] 0` |
| `blocks.2.weight` | `[MEASURED] 2` | `[MEASURED] 0` | `[MEASURED] 0` | `[MEASURED] 0` |
| `skips.4.bias` | `[MEASURED] 2` | `[MEASURED] 0` | `[MEASURED] 0` | `[MEASURED] 0` |
| `blocks.4.weight` | `[MEASURED] 2` | `[MEASURED] 0` | `[MEASURED] 0` | `[MEASURED] 0` |
| `rgb_0.bias` | `[MEASURED] 2` | `[MEASURED] 0` | `[MEASURED] 1` | `[MEASURED] 0` |
| `blocks.1.weight` | `[MEASURED] 2` | `[MEASURED] 0` | `[MEASURED] 0` | `[MEASURED] 0` |
| `refine.0.weight` | `[MEASURED] 2` | `[MEASURED] 0` | `[MEASURED] 0` | `[MEASURED] 0` |
| `blocks.3.bias` | `[MEASURED] 2` | `[MEASURED] 0` | `[MEASURED] 0` | `[MEASURED] 0` |
| `blocks.5.weight` | `[MEASURED] 2` | `[MEASURED] 0` | `[MEASURED] 0` | `[MEASURED] 0` |
| `skips.2.bias` | `[MEASURED] 2` | `[MEASURED] 0` | `[MEASURED] 0` | `[MEASURED] 0` |
| `blocks.1.bias` | `[MEASURED] 2` | `[MEASURED] 0` | `[MEASURED] 0` | `[MEASURED] 0` |
| `blocks.4.bias` | `[MEASURED] 2` | `[MEASURED] 0` | `[MEASURED] 0` | `[MEASURED] 0` |
| `skips.4.weight` | `[MEASURED] 2` | `[MEASURED] 0` | `[MEASURED] 0` | `[MEASURED] 0` |
| `stem.bias` | `[MEASURED] 2` | `[MEASURED] 0` | `[MEASURED] 0` | `[MEASURED] 0` |
| `refine.0.bias` | `[MEASURED] 2` | `[MEASURED] 0` | `[MEASURED] 0` | `[MEASURED] 0` |
| `blocks.0.bias` | `[MEASURED] 2` | `[MEASURED] 0` | `[MEASURED] 0` | `[MEASURED] 0` |
| `rgb_1.bias` | `[MEASURED] 2` | `[MEASURED] 0` | `[MEASURED] 0` | `[MEASURED] 0` |
| `blocks.5.bias` | `[MEASURED] 2` | `[MEASURED] 0` | `[MEASURED] 0` | `[MEASURED] 0` |
| `blocks.0.weight` | `[MEASURED] 2` | `[MEASURED] 0` | `[MEASURED] 0` | `[MEASURED] 0` |
| `rgb_1.weight` | `[MEASURED] 2` | `[MEASURED] 0` | `[MEASURED] 0` | `[MEASURED] 0` |
| `rgb_0.weight` | `[MEASURED] 2` | `[MEASURED] 0` | `[MEASURED] 0` | `[MEASURED] 0` |
| `skips.3.bias` | `[MEASURED] 2` | `[MEASURED] 0` | `[MEASURED] 0` | `[MEASURED] 0` |
| `skips.3.weight` | `[MEASURED] 2` | `[MEASURED] 0` | `[MEASURED] 0` | `[MEASURED] 0` |
| `refine.1.bias` | `[MEASURED] 2` | `[MEASURED] 0` | `[MEASURED] 1` | `[MEASURED] 0` |
| `blocks.3.weight` | `[MEASURED] 2` | `[MEASURED] 0` | `[MEASURED] 0` | `[MEASURED] 0` |
| `stem.weight` | `[MEASURED] 2` | `[MEASURED] 0` | `[MEASURED] 0` | `[MEASURED] 0` |

The `[MEASURED] 2` equal-byte last planes failed the strict byte-decrease gate and were not
admitted. Final accepted sections: `[MEASURED] 0`. Combined proposal rows: `[MEASURED] 0`.

## Final archive and receiver parse-back

| Quantity | Result |
|---|---|
| Final archive | `[MEASURED]` `experiments/results/jrd_pr128_completion_20260713T022712Z/archive.zip` |
| Final bytes | `[MEASURED] 176564` |
| Final SHA-256 | `[MEASURED] 196acd18e4ca10a3ab0d826436aa46014a44cba8a55eb4abf9931876cc7e98b5` |
| Bytes saved | `[MEASURED] 0` |
| Final delta d_seg | `[MEASURED] 0.0` |
| Final delta d_pose | `[MEASURED] 0.0` |
| Rate-term delta | `[DERIVED] -25 * 0 / 37545489 = 0.0` |
| Advisory delta S | `[DERIVED] 0.0` |

`archive.zip` is `[MEASURED]` byte-identical to the measured candidate and source archive. It is
delivered because the mission requires a byte-closed artifact even for an honest NULL result.

The frozen runtime was executed through a streaming FIFO twice:

| Parse-back pass | Return code | Raw bytes | Raw SHA-256 | Reader cleanly joined |
|---|---:|---:|---|---|
| `[MEASURED] 1` | `[MEASURED] 0` | `[MEASURED] 3662409600` | `[MEASURED] 465781ba6f334ae4837ad99de8b8787e1d5f312553cff6d75364f888c1007a2f` | `[MEASURED] true` |
| `[MEASURED] 2` | `[MEASURED] 0` | `[MEASURED] 3662409600` | `[MEASURED] 465781ba6f334ae4837ad99de8b8787e1d5f312553cff6d75364f888c1007a2f` | `[MEASURED] true` |

Parse-back is `[MEASURED] PASS`; repeated output is bit-identical, matches the in-process
exact-R raw SHA, materializes no bulk raw file, and reports scratch cleanup on success.

## Triality and reproducibility

- **DSL/argv leg:** exact argv is SHA-bound in
  `experiments/results/jrd_pr128_completion_20260713T022712Z/run_fingerprint.json`; no flag was
  invented.
- **DAG leg:** grammar parse -> no-op identity -> screen baseline/repeat/negative -> all nested
  tensor prefixes -> strict byte/component filter -> combined recheck for any survivor -> n600
  baseline/repeat/negative -> repeated runtime FIFO parse-back -> byte-closed handoff.
- **Equation leg:** the admission inequalities and contest score decomposition above are the
  canonical decision law.
- **Resume leg:** row, tensor, final-meter, and receiver checkpoints are preserved under
  `experiments/results/jrd_pr128_completion_20260713T022712Z/`.
- **Pointer delta:** `pointer_moved=false`; the delivered bytes equal the borrowed bank archive.

## Durable receipts

- `experiments/results/jrd_pr128_completion_20260713T022712Z/archive.zip`
- `experiments/results/jrd_pr128_completion_20260713T022712Z/measurement_receipt.json`
- `experiments/results/jrd_pr128_completion_20260713T022712Z/section_precision_response_curves.json`
- `experiments/results/jrd_pr128_completion_20260713T022712Z/runtime_inflate_proof.json`
- `experiments/results/jrd_pr128_completion_20260713T022712Z/run_fingerprint.json`
- `experiments/results/jrd_pr128_completion_20260713T022712Z/resume/state.json`

## STORES CONSULTED

- `CLAUDE.md`
- `AGENTS.md`
- `docs/operating_manual_craft_handoff.md`
- `reports/latest.md`
- `.omx/state/lane_registry.json`
- `.omx/state/active_lane_dispatch_claims.md`
- `.omx/state/subagent_progress.jsonl`
- latest Codex findings/session-summary, council, and design memo pointers from preflight
- source archive, PR110 runtime tree, adapter/oracle bytes, local scorer weights, and n600 GT cache
  SHA-bound in `run_fingerprint.json`
