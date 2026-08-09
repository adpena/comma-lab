# FX4 — GT-cache lineage, measured confound, and graph-memory cure

**Date:** 2026-08-09  
**Mode:** n600 full-population scorer-free cache comparison plus source/receipt audit  
**Axis:** `[macOS-CPU advisory]`, `score_claim=false`, no candidate score  
**PR130 source pin:** `e34f31bc4969042c0051ac81aa3c56884419a231`  
**RR1 pin:** `ecfd4ec595`

## 1. Measured inter-cache delta

The retained semantic cache and retained official-Ada cache are materially different targets.
Across all 600 pairs, with no prefix and no sampling, they differ at **20,750 / 117,964,800
segmentation sites = 0.00017589992947048612 d_seg**. Every pair differs: **600 / 600**.
The stored six-coordinate PoseNet targets differ at **3,600 / 3,600 elements**, with
**d_pose = MSE = 0.00014004340079290474**.

In the contest score's units, the target-to-target separation is:

| component | measured target separation | score-form term |
|---|---:|---:|
| Seg | `20,750 / 117,964,800 = 0.00017589992947048612` | `100*d_seg = 0.017589992947048612` |
| Pose | `MSE = 0.00014004340079290474` over `3,600 / 3,600` elements | `sqrt(10*d_pose) = 0.03742237309323191` |
| Rate | same cache comparison, no archive mutation | `0` |
| **Total separation scale** | | **`0.05501236604028052`** |

This `0.05501236604028052` is **not a candidate score and not a generally additive penalty**.
It is the score-form distance between the two target tensors. A real candidate can have residuals
that partly cancel or reinforce the target shift, and the pose term is nonlinear. The directly
measured fixed-renderer effect is narrower: the selected quantized semantic renderer scores
`0.0002764044867621528` against the retained AV-like target and
`0.0002857038709852431` against the retained DALI target, so changing only the target adds
`9.299384223090295e-06 d_seg = 0.0009299384223090295 S_seg` on that renderer
`[macOS-Metal advisory]`, with pose unmeasured by that semantic-leg receipt.

### Per-pair and per-class denominators

The per-pair flip count is min `15`, q25 `27.75`, median `33`, mean `34.583333333333336`,
q75 `40`, max `103`, over **600 / 600 pairs**.

| class | AV-like sites | flips leaving AV-like class | fraction | DALI sites | flips entering/leaving DALI class | fraction |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 27,407,045 | 9,144 | 0.0003336368441034048 | 27,407,369 | 9,468 | 0.00034545453815723793 |
| 1 | 690,639 | 2,788 | 0.004036841244123196 | 690,874 | 3,023 | 0.004375616972125163 |
| 2 | 58,413,281 | 3,820 | 0.0000653960868933214 | 58,413,418 | 3,957 | 0.00006774128505885411 |
| 3 | 1,460,325 | 1,735 | 0.001188091691917895 | 1,460,279 | 1,689 | 0.0011566282881558935 |
| 4 | 29,993,510 | 3,263 | 0.00010879020161361574 | 29,992,860 | 2,613 | 0.0000871207347348669 |

Class 1 has the largest conditional flip rate on both denominators, but class 0 has the largest
absolute flip count. The complete AV-like-to-DALI 5x5 transition matrix and per-dimension pose
MSE/max-absolute deltas are in `FX4_GT_LINEAGE_RECEIPT.json`.

## 2. Cache lineages and receipts

### `gt_cache_600.pt` — retained semantic target

- Restored bytes: `117,981,133`; SHA-256
  `8248a60da56119eb4b3ad76bfa32f5498dee849eaf4b83b304275064141fd828`.
- Compressed intake source: `artifacts/caches/gt_cache_600.pt.xz`, `527,296 B`, SHA-256
  `67fa351ac728332b0e9b186706a6596f3f9b85e15303b55f007473fdec6d248f`.
- `scripts/train.sh prepare` only restores it with `xz -dc` and verifies the hash; that is not
  its producer command. The exact historical producer command and log were **not found** in the
  five-commit PR130 Git history, the intake working tree, or the Pact research scopes listed in
  `RECALL EVIDENCE` below.
- Content pins it as **AV-like**, not as proven historical PyAV provenance: compared with the
  completed same-host T4 AV cache, it differs at **1 / 117,964,800 seg sites** and has pose MSE
  `1.0483765967618982e-12`.
- The compatible decoder path is `AVVideoDataset`: PyAV decodes planar YUV420, then the official
  `frame_utils.yuv420_to_rgb` performs bilinear chroma upsampling with `align_corners=False` and
  BT.601 limited-range conversion. PyAV `rgb24` was not used by this retained builder family.
- **Lineage boundary:** content nearly identifies the AV target object, but it does not recover
  the original executable, argv, package versions, device, or color-conversion receipt.

### `gt_cache_600_official_ada.pt` — retained official DALI target

- Restored bytes: `117,981,301`; SHA-256
  `382d7dfe38b37c0cc5017e5645032faa045af6924db66e0b67549cc96c840195`.
- Compressed intake source: `artifacts/caches/gt_cache_600_official_ada.pt.xz`, `526,820 B`,
  SHA-256 `233884c672eff22258376cf9532bb69a52017980000a2615bbd917ba7a8ec3dc`.
- The retained producer implementation is `code/build_gt_cache_official.py`. Its DALI mode uses
  `DaliVideoDataset`/NVDEC RGB output and does **not** call `frame_utils.yuv420_to_rgb`; its AV mode
  uses the canonical conversion above. The strict graph's stage `01_targets` explicitly supplies
  `--dataset dali` with batch size 16, two threads, prefetch depth 4, and seed 1234 inside the
  dataset constructor.
- The exact historical shell command/log that created this particular `382d...` cache was **not
  found**. Therefore the strict graph command is evidence of the retained DALI formulation, not
  proof that this exact invocation created the retained bytes.
- DALI identity is additionally supported by the cache's independent-official-Ada role and its
  deployed token identity. It is not interchangeable with every DALI build: the completed T4
  DALI cache differs at **1,644 / 117,964,800 seg sites**, pose MSE
  `1.0652253518058493e-06`, score-form separation scale `0.004657415091793526`.

The source cache objects contain only `pose` and `seg`. Neither embeds decoder, color conversion,
challenge commit, scorer-weight hashes, video hashes, command, hardware, or package versions.

## 3. Canonical target for our iteration

**Recommendation: use the DALI target family, and use the retained official-Ada cache
`382d7dfe...0195` for apples-to-apples iteration on the PR130 base. Do not train new PR130 legs
against `gt_cache_600.pt`.** Final promotion still requires exact DALI replay on the intended
contest-CUDA runtime; a cache is a training/evaluation target asset, not score authority.

The recommendation is derived rather than preferential:

1. PR130 CPR1's base row is `S = 0.172141297491896447` at `191,052 B`
   `[contest-CUDA, DALI GT, n600]`. The object to beat is therefore DALI-scored.
2. The retained AV-like and retained DALI targets are separated by `0.017589992947048612 S_seg`
   and a `0.03742237309323191` root-pose separation scale. Optimizing AV while scoring DALI is
   not an innocuous label choice.
3. On the actual selected semantic renderer, DALI evaluation is already
   `0.0009299384223090295 S_seg` worse than AV evaluation. The same weights reproduce each
   lineage's own recorded number; that is direct evidence that the optimization target matters.
4. The retained `382d...` cache is the exact DALI target tied to PR130's carrier, HPAC, deployed
   token golden, and published lineage. The fresh T4 `a91d...` cache is a second DALI-family
   runtime object, useful for promotion checks but not a silent replacement for the PR130 base.

For future reports, `DALI` alone is not a sufficient cache ID. Every local result must carry the
target cache SHA-256; authority results must additionally carry the actual evaluator/runtime axis.

## 4. Graph-provenance cure

The immutable PR130 intake was not edited. The retained replay now has machine-readable graph
memory in `FX4_GT_PROVENANCE_MANIFEST.json`:

- `semantic -> retained_semantic_av_like_8248a60d`;
- `carrier`, `hpac`, and `encode_tokens -> retained_official_dali_382d7dfe`;
- strict raw-video E2E -> one dynamic `strict_fresh_dali` target produced at stage 01 and consumed
  by **41 / 49 stages**.

The manifest therefore preserves the distinction the 49-stage code erased. It also labels the
strict graph correctly as **DALI-for-all formulation**, not an exact replay of the historical
mixed-cache selected treatment.

`fx4_gt_provenance_guard.sh LEFT_LEG RIGHT_LEG` is the L3 verdict-clearance control. It reads the
manifest and returns typed `REFUSE` with exit code 42 when target-axis IDs differ. It passes
same-target comparisons and fails closed on unknown legs. Demonstrated controls:

| comparison | expected | observed |
|---|---|---|
| `semantic` vs `carrier` | refuse cross-axis comparison | `REFUSE`, rc 42 |
| `carrier` vs `hpac` | allow same-axis comparison | `PASS`, rc 0 |
| unknown leg vs `hpac` | fail closed | rc 2 |

This cures graph memory for the retained replay without mutating the intake. It does not make an
unrelated ad hoc script consult the manifest automatically; any maintained comparison consumer must
invoke the guard before interpreting cross-leg metrics.

## 5. Consequence for #906

The DALI-vs-AV Modal job is **complete, not pending**. Before consuming it, FX4 asserted terminal
`status=OK`, coverage `600 / 600` AV and `600 / 600` DALI, on one Tesla T4 with torch
`2.5.1+cu124`. The same-host result, re-reduced exactly from the terminal cache tensors, is:

- Seg disagreement `0.00017523023817274304`, or `0.017523023817274306 S_seg`;
- pose MSE `0.00014061324889363773`, root term `0.03749843315308491`;
- total score-form target separation scale `0.05502145697035922`.

The remote receipt's float32 reductions are `0.00017523023416288197` for Seg disagreement and
`0.00014061325055081397` for pose MSE; FX4's integer/float64 re-reductions are
`20,671 / 117,964,800 = 0.00017523023817274304` and `0.00014061324889363773`. The tiny numerical
difference is reduction precision, not a different cache pair.

The retained-cache result is nearly the same at the aggregate decoder-separation level:
`0.05501236604028052`, a difference of about `9.09e-06 S` in the score-form scale. This makes the
decoder split the dominant explanation of the historical AV-like-versus-DALI confound, while the
within-DALI `0.004657415091793526` separation proves runtime/hardware version drift is not zero.

**Dispatch consequence:** another DALI-vs-AV Modal job is unnecessary for #906. The completed job
already bought both durable caches and isolated the decoder on one host. Future CUDA spend is required
only for a byte-closed candidate's exact contest replay or for a deliberately pinned attempt to
reproduce the historical `382d...` DALI cache, not to re-answer the AV/DALI question.

## 6. Ranked residuals and falsifiers

1. **Historical producer argv is absent — HIGH, `INSTANCE`.** Neither retained cache is
   self-authenticating, and the exact producer command/log was not found in the bounded scopes.
   **Falsifier:** a hash-linked historical runner marker or log naming the exact output hash,
   decoder mode, challenge/scorer/video hashes, runtime, and argv.
2. **DALI-family cache drift remains — MEDIUM-HIGH, `FORMULATION`.** Retained Ada DALI and fresh T4
   DALI differ by 1,644 seg sites and pose MSE `1.0652253518058493e-06`.
   **Falsifier:** a pinned historical environment that reproduces both raw `seg` and `pose` tensor
   hashes exactly, or a source proof assigning the drift to a different named input.
3. **Candidate-level pose effect is not measured — MEDIUM, `INSTANCE`.** Target-to-target root-pose
   separation is not the change in any candidate's nonlinear pose term.
   **Falsifier:** score the same byte-closed candidate outputs against both hash-pinned target caches
   with the same frozen scorer outputs and recompute both full-precision component formulas.
4. **The immutable intake does not call the sidecar guard — MEDIUM, `FORMULATION`.** FX4 preserved
   graph memory and supplied a fail-closed consumer, but an unrelated ad hoc comparator can ignore it.
   **Falsifier:** a maintained comparison harness that makes the manifest/guard mandatory before every
   cross-leg verdict, with the semantic-vs-carrier positive refusal as a regression test.

## RECALL EVIDENCE

Searches performed before adjudication:

- `.omx/research/**` and the PR130 intake for `gt_cache_600.pt`,
  `gt_cache_600_official_ada.pt`, both uncompressed hashes, `build_gt_cache_official`,
  `decoder/target confound`, `DALI-vs-AV`, and `#906`;
- the complete five-commit PR130 Git history with content-history searches for both cache names and
  the producer builder;
- `.omx/state/main_hot_state.md`, `active_lane_dispatch_claims.md`, the operator P0 ledger, and
  task-related research rows for `#906`, `#995`, PR130, DALI, and PyAV;
- `CANONICAL_RESEARCH_INDEX*` and `sub015_DAG_*` content searches for PR130/cache/decoder terms;
- `.venv/bin/python tools/list_canonical_equations.py --json`: 429 entries searched for PR130,
  DALI, PyAV, `gt_cache`, decoder, and chroma-siting terms.

Beyond the charter seeds, the search found the terminal same-host #906 result and OP1R's complete
n600 preserved-cache receipt. That changed the plan in three ways: no new Modal dispatch; direct
n600 re-derivation instead of a sample; and separate IDs for retained DALI versus fresh T4 DALI.
No equation specific to this GT-cache lineage/refusal surface was found in the 429-entry registry
search. That is a scoped registry negative, not a claim that no related equation exists.

## Could not check and why

- The exact historical producer commands for both retained caches: absent from the searched intake
  Git history, retained receipts, and named Pact research scopes.
- Historical hardware, package versions, scorer/video hashes, and color-conversion metadata for the
  retained caches: not embedded in the cache objects or linked by a surviving runner receipt.
- The exact argv of the completed #906 remote run: its terminal result preserves leg labels, runtime,
  output hashes, coverage, and builder stdout, but not a top-level command field.
- A candidate-level full-score delta between AV and DALI: no scorer slot was owned or needed for this
  scorer-free cache comparison, and target separation must not be relabeled as candidate impact.
- Automatic enforcement inside the PR130 intake: intake is binding read-only. The cure is a maintained
  sidecar manifest plus fail-closed guard in our repository.

## Follow-on dispositions

| disposition | owner | consumer store | fire trigger | action |
|---|---|---|---|---|
| `FIRED` | FX4 | `FX4_GT_PROVENANCE_MANIFEST.json` + guard | this landing | Preserve per-leg target IDs and refuse semantic-to-DALI-leg comparisons. |
| `FOLDED` | MAIN / #906 | `.omx/state/main_hot_state.md` consumers | terminal #906 receipt verified | Do not dispatch another generic DALI-vs-AV cache job. |
| `QUEUED-WITH-A-FIRE-ORDER` | MAIN / PR130 semantic owner | next semantic training ticket and result receipt | before the next PR130 semantic training step | Point training/evaluation at retained DALI cache SHA `382d...0195`; stamp the cache hash in every result. |
| `QUEUED-WITH-A-FIRE-ORDER` | MAIN / comparison-harness owner | maintained PR130 comparison harness | before the next cross-leg metric verdict | Invoke the FX4 guard and retain its PASS/REFUSE row with the comparison receipt. |
| `QUEUED-WITH-A-FIRE-ORDER` | MAIN / exact-eval owner | contest-CUDA candidate receipt store | once a DALI-trained byte-closed candidate passes local gates | Run exact n600 contest-CUDA/DALI evaluation; do not promote cache separation as a score. |

## Landing status

The required serializer was invoked with post-edit SHA-256 pins for all four FX4 files and commit
message tags `[no-triality] [p0-ledger-ok]`. It failed before staging with Git rc 128:
`unable to create temporary file: Operation not permitted` and `failed to insert into database`.
The managed sandbox exposes `.git` read-only, so no compliant commit can be created in this turn.
The staged index remained unchanged; the four validated FX4 artifacts are present as untracked
working-tree files. No bare `git commit`, alternate index, or intake mutation was attempted.

## Frontier honesty

FX4 moved no candidate score and no exact pointer. The PR130 base remains
**S = 0.172141297491896447 at 191,052 bytes `[contest-CUDA, DALI GT, n600]`**.
