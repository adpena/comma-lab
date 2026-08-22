# ddm_rc1 rate crush — terminal temporal-program representation

Date: 2026-08-22  
Verdict: **BUILT AND REAL-CODER MEASURED; QUEUED-WITH-FIRE-ORDER, BLOCKED**  
Measurement axis: `[macOS-CPU scorer-free real-coder token-representation n600]`  
Authority claim: `score_claim=false`, `frontier_moved=false`

## Answer first

RC1 found a new representation with enough measured byte mass to cover the
42,382-byte sub-0.12 demand. The selected scorer-free point replaces DX2's
113,777-byte RC64 token stream and 13,515-byte HPAC probability model with a
counted 59,884-byte terminal temporal-program payload. In a real, retained ZIP
shadow container with the exact DX2 semantic, carrier, and residual streams,
the complete size is **113,006 B**, a **67,362 B** cut from DX2 and **24,980 B**
below the strict 137,986-byte ceiling.

That is a byte result, not a candidate score. The research receiver reconstructs
the complete `600x384x512` categorical token tensor, but it does not run the
shipping renderer to full RGB. No SegNet, PoseNet, exact evaluator, Modal job,
or score authority ran. The selected reconstruction agrees with the source token
tensor on **98.7959704929%** of labels, but that diagnostic is not evaluator
evidence and its rare-class retention is weak. RC1 therefore has one sealed,
blocked fire-order for MAIN rather than a promotion claim.

## Dimension choice

The measured DX2 anatomy fixes the choice:

| physical region | DX2 bytes | can contain the 42,382 B cut alone? |
|---|---:|---|
| ZIP framing + RX1 header | 114 | no |
| HPAC probability model | 13,515 | no |
| semantic renderer | 30,856 | no |
| carrier | 22,010 | no |
| residual | 96 | no |
| semantic tokens | **113,777** | **yes** |

The token stream is 63.0805% of the archive and is the only physical region
larger than the demand. If every other DX2 byte stays fixed, the replacement
token payload must be at most

```text
113,777 - 42,382 = 71,395 B.
```

RC1 therefore operates at the **terminal categorical-token lattice**: one
length-600 temporal program per spatial site, expanded by generic receiver code.
This is not another coder race on the fixed RC64 representation. The video-
derived program dictionary and site assignments are counted; only the generic
parser, coder implementations, and expansion loop are free receiver code.

Inflated RGB, literal evaluator planes, explicit sparse grids, and smaller
archive streams were not selected. They either lack enough measured mass or are
already closed at the scopes recorded below. NR1 remains a sibling evaluator-
cell quotient route; RC1 does not edit, depend on, or claim credit for it.

## Mechanism at its measured optimal form

The retained source contains 196,608 spatial sites and 30,428 distinct
length-600 categorical trajectories. RC1 learns a deterministic temporal
dictionary plus a `384x512` spatial assignment lattice:

1. For `K<=256`, population-weighted categorical k-modes fits whole temporal
   trajectories with lexical and smallest-index tie breaks.
2. For larger `K`, the fitted K=256 basis is extended by exact source programs
   in descending `population x residual Hamming debt` order. This is optimal
   within the declared nested exact-program policy; it does not claim a global
   large-K optimum and does not silently reassign unrelated programs.
3. Assignments race five reversible layouts (`row`, `serpentine`, both delta
   forms, and row RLE) through Brotli q11, raw LZMA1, and zlib9: 15 retained
   streams per point.
4. Codebooks race four reversible layouts (`row`, `time-major`, temporal
   delta, and transition events) through the same three coders: 12 retained
   streams per point.
5. Every learned model, every coder variant, selected payload and repeat,
   decoded token tensor, shadow archive and repeat, and mutation negative is
   retained. No materialized payload is reduced to scalar-only evidence.

The code is in `src/tac/optimization/rc1_terminal_program_vq.py`; the resumable
materializer is `experiments/ddm_rc1_rate_crush.py`. The canonical run retains
the exact producer-source bytes and refuses resume if the workspace source no
longer matches them.

## Real-coder price curve

All rows below are measured at
`[macOS-CPU scorer-free real-coder token-representation n600]`. `agreement` and
per-class IoU compare reconstructed tokens with the retained DX2 token tensor;
they are diagnostics only. `shadow B` is a real deterministic ZIP, but not an
evaluator-runnable submission archive.

| K | RC1 payload B | token cut B | shadow B | headroom to 137,986 B | token agreement | class-1 IoU | class-3 IoU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 868 | 112,909 | 53,990 | 83,996 | 97.340836% | 0.0020 | 0.0000 |
| 8 | 1,641 | 112,136 | 54,763 | 83,223 | 97.343365% | 0.0063 | 0.0000 |
| 16 | 2,979 | 110,798 | 56,101 | 81,885 | 97.348633% | 0.0153 | 0.0000 |
| 32 | 4,679 | 109,098 | 57,801 | 80,185 | 97.383228% | 0.0236 | 0.0052 |
| 64 | 5,520 | 108,257 | 58,642 | 79,344 | 97.792871% | 0.0245 | 0.3759 |
| 128 | 7,137 | 106,640 | 60,259 | 77,727 | 98.335036% | 0.0309 | 0.5597 |
| 256 | 9,338 | 104,439 | 62,460 | 75,526 | 98.481602% | 0.0539 | 0.6376 |
| 512 | 18,888 | 94,889 | 72,010 | 65,976 | 98.544393% | 0.0587 | 0.6672 |
| 1,024 | 33,816 | 79,961 | 86,938 | 51,048 | 98.642058% | 0.0859 | 0.6997 |
| **2,048** | **59,884** | **53,893** | **113,006** | **24,980** | **98.795970%** | **0.1460** | **0.7321** |
| 4,096 | 105,811 | 7,966 | 158,933 | -20,947 | 99.034394% | 0.2736 | 0.7704 |

K=2,048 is the largest measured point that passes both the conservative
71,395-byte token-only bar and the complete shadow-container bar. Its payload is
64 B of header, a 10,900 B row+Brotli-q11 assignment stream, and a 48,920 B
row+raw-LZMA1 codebook. It has 11,511 B of headroom to the token-only bar. The
K=4,096 point improves overall agreement but is byte-dead at this formulation:
it exceeds the token-only bar by 34,416 B and the complete shadow bar by 20,947 B.

For K=2,048, the measured shadow rate term is
`25*113006/37545489 = 0.0752460568565241`. If distortion were unchanged, pure
arithmetic would map DX2 from `0.14821987563243377` to
`0.10336628483221807`, leaving `0.01663371516778192 S` of total distortion-tax
room under 0.12. This is a counterfactual admission bound, **not a projected or
measured candidate score**. The 1,420,331 changed token labels and especially
class-1 IoU 0.145961 make exact full-RGB scoring load-bearing.

## Receiver contract

The counted RC1 payload contains an explicit version, assignment/codebook coder
IDs, assignment-width flag, tensor geometry, exact section lengths, per-section
CRCs, and the SHA-256 of the reconstructed token tensor. The receiver:

- rejects unknown versions, methods, flags, invalid classes, missing codewords,
  non-canonical ULEB fields, wrong lengths, trailing bytes, decompressor residue,
  CRC differences, and non-canonical re-encodings;
- reconstructs the complete token tensor frame by frame and verifies its digest;
- consumes the copied semantic, carrier, and residual sections in the shadow
  container and exhausts its exact declared length;
- produces byte-identical RC1 payload repeats and byte-identical ZIP repeats;
- refuses one-bit mutations independently in assignment, codebook, semantic,
  carrier, and residual sections, plus a trailing-byte negative.

The contract currently stops at tokens. `shipping_integration=false`,
`full_rgb_render=false`, and `evaluator_runnable=false` are recorded in every
candidate result. A representation with no decode path would not be a
candidate; RC1 has a strict token decode path, but not yet the final renderer
leg required for an exact-row claim.

## Prior-law prediction

The charter predicted that the same terminal mechanism would yield materially
more when placed on the 113,777-byte token bulk than on a smaller stream, with
roughly size-proportional yield as the falsifier. **This prediction remains
UNTESTED.** The measured token placement has sufficient mass, but the other DX2
streams are compressed parameter packets rather than homologous categorical
time lattices. Applying a different transform to those bytes would not be the
same mechanism, and comparing RC1 with an old fixed-stream coder would fake the
falsifier. The result is consistent with choosing the high-mass placement but
does not confirm the 26x placement law.

## Fire disposition

Disposition: **QUEUED-WITH-FIRE-ORDER, BLOCKED**. MAIN owns the scorer lane and
no lane was claimed or fired by RC1. The single seal is:

`/Volumes/APDataStore/pact/ddm_rc1_rate_crush/measurement_v4/SEALED_FIRE_ORDER.json`

SHA-256: `0d683cd3ee46dce4ed4d5b5b14d49ef608365537fcea29754619e833907eae56`.

It selects K=2,048 by the declared scorer-free rule “highest source-token
agreement under the conservative byte bar.” Dispatch argv is deliberately null.
The fire trigger requires MAIN to own an idle unique n600 scorer lane and a
fresh MAIN-owned runtime to integrate this exact RC1 payload into the real DX2
full-RGB receiver, with parse-back, repeat, every-paid-section mutation refusal,
and retained exact archive custody passing. MAIN then evaluates that exact
archive and folds immediately if its recomputed score does not improve the
canonical pointer or it exceeds 137,986 B.

## Retained custody

Canonical root:
`/Volumes/APDataStore/pact/ddm_rc1_rate_crush/measurement_v4/`.

| artifact | bytes | SHA-256 |
|---|---:|---|
| `RESULT.json` | 393,133 | `d51e92a37bddca462a381eec66f4dbc37ff4a38f941fe2e033fc51c3c31e119c` |
| `RUN_MANIFEST.json` | 2,866 | `80a7269ab00a3b6dcc67b5da896ff29f5903861cab24210564948103be8229d3` |
| `retained/source_index/MANIFEST.json` | 1,099 | `6cac81f93ad742d45b27228d011400e51dfc67f7ec16e751a449456d0348fc90` |
| selected `tokens.rc1v` | 59,884 | `eab66bad9d113ed79475a810f4002ec821deb335c3e87fc1b1e90ef2b8e61164` |
| selected decoded `tokens.u8` | 117,964,800 | `2c85d29698782b2b12f75a897665f80c59a40a9549f0697e18db16feaca93168` |
| selected shadow `archive.zip` | 113,006 | `6756ae8f39116907828ee27b8f9686b9935eaae94c61f68c3eb02de16d45e87a` |
| `SEALED_FIRE_ORDER.json` | 1,928 | `0d683cd3ee46dce4ed4d5b5b14d49ef608365537fcea29754619e833907eae56` |

Input custody passed for DX2 archive SHA-256
`976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`
at 180,368 B and source token SHA-256
`cc10a7b09353c0af1ebe4e52a1640df1fadac4d245a27f41aff8cf0992636efb`
at 117,964,800 B. The canonical root occupies about 1.4 GB and passed the
APDataStore storage waterfall with an 8 GiB reserve. Earlier `measurement_v1`
through `measurement_v3` development outputs remain retained and superseded;
no evidence bytes were deleted.

## RECALL EVIDENCE

Sources searched included the governing files, full `.omx/research/` memo and
receipt corpus, `CANONICAL_RESEARCH_INDEX*`, `sub015_DAG_*` FEED blocks, design
SPECs, canonical task/bridge stores, `main_hot_state.md`, the canonical equation
registry from `tools/list_canonical_equations.py --json`, the shipped DX2
receiver, and `src/tac` implementation surfaces.

Content queries included:

- `terminal neural|learned prior|sleeper|decode-time expansion|task-cell quotient`;
- `semantic token stream|RC64|token predictor|token factor|hyperprior|codebook|implicit`;
- equations filtered by `rate|byte|entropy|token|quotient|prior|receiver|codebook|latent|context|grammar`;
- `RX1_MODEL_HEADER|RC64|semantic token|token_stream` across code and receipts.

Findings beyond the charter seeds that changed the plan:

1. TM1 had already tested terminal token reversion on the PR130 fixed model and
   won only 296 B, with a 56 B confidence bound and no receiver. RC1 therefore
   did not repeat a fixed-context temporal tweak; it replaced the represented
   object with a counted temporal-program dictionary.
2. The exact DX2 decode checkpoint retained the whole categorical tensor under
   an archive-bound SHA. Its 30,428 distinct site trajectories made a complete
   population temporal dictionary measurable without a scorer or live-run read.
3. NR1's prebuild is an evaluator-cell quotient sibling with zero measured
   candidate bytes, not a surface RC1 may edit or call its own. RC1 selected the
   token lattice so the campaign does not have a single representation route.
4. GC18's current representation addendum makes payload or decoded-output
   compression legal when all learned/video-derived state is counted. That
   moved the design from another archive-byte coder to terminal decode-time
   expansion.
5. The live pointer is DX2 at 180,368 B, not the stale rc2/common-contract row.
   All demand, rate, and fire arithmetic was therefore recomputed from DX2.

## Verification and boundaries

- Provenance pins passed for RB1
  `fa26a44444a57428910565956011e0bb26c6680174a71bfbb914002f9f564f09`
  and TL1
  `d307c971f7cdb41806f39135acbc5ff68549283700699ae7a8b1bd77d60ecf15`.
- A final-source canonical rerun reproduced all 11 prices. An immediate resume
  rehashed and reused every completed candidate in 0.65 s.
- Independent custody verification rehashed **451 retained files** across all
  11 candidates, reparsed every selected payload and shadow ZIP, reconstructed
  every token digest, and passed every recorded mutation refusal.
- Synthetic receiver coverage passed every 15 assignment by 12 codebook coder
  combination for both 8-bit (`K=3`) and 16-bit (`K=300`) assignments.
- `ruff check`, `py_compile`, and `git diff --check` passed for both RC1 Python
  files. Two genuine review-tracker passes completed after the final fixes.
- Developer preflight was **18/25 green, 7 red**. A bounded non-strict rerun of
  every red gate found zero references to either RC1 Python file or this memo:
  the existing failures are one strict-state-writer violation, one custody-tag
  bypass, one shared-state writer, 25 legacy launch patterns, one dispatch-
  helper violation, 124 older landing memos, and five substrate scorer-contract
  violations. RC1 added no waiver or unrelated repair.
- Scorer forwards: **0**. Modal dispatches: **0**. Full-RGB renders: **0**.
  Exact evaluator runs: **0**. Shipped receiver edits: **0**. Reads/writes under
  the sacred JO2 run directory: **0**.

## NEXT_IF_RESUMED

- **QUEUED-WITH-FIRE-ORDER, BLOCKED** — owner: MAIN exact-row and unique n600 scorer-lane owner; consumer store: `/Volumes/APDataStore/pact/ddm_rc1_rate_crush/measurement_v4/main_fire/`; fire trigger: after the live joint solve is harvested and MAIN owns an idle lane, integrate selected payload `eab66bad…e61164` into a fresh owned full-RGB DX2 receiver, retain an exact repeat and all-paid-section mutation controls, then evaluate only that exact archive and fold if recomputed S does not improve the pointer or bytes exceed 137,986.

## LIVE-HYPOTHESES

- A class-balanced or boundary-debt temporal dictionary could preserve rare
  class-1/3 programs far better at the same K, because the current population-
  weighted objective spends most distortion on majority classes and the
  assignment stream is only 10,900 B at K=2,048.
- Jointly coding spatial assignments with a multiscale boundary grammar could
  buy more codebook capacity under 71,395 B, because the selected codebook costs
  48,920 B while the spatial map costs only 10,900 B and the K=4,096 point shows
  additional agreement exists just beyond the byte bar.
- The evaluator may tolerate materially more token Hamming than exact token
  identity requires if changed programs stay inside the same scorer cells; this
  is plausible because RC1 compresses whole temporal programs, but only the
  full-RGB receiver and exact scorer can test it.
- A hybrid RC1/NR1 program may use RC1's cheap temporal-site routing as counted
  conditioning for NR1's evaluator-cell renderer, because the two routes spend
  bytes on different objects and neither has yet measured the composition.

## DEAD-ENDS

- Another fixed-RC64 coder/context race is closed on DX2: the composable 88 B
  ceiling is already shipped and cannot supply 42,382 B.
- PR130 memoryless entropy numbers are not DX2 floors; transferring them across
  objects is invalid.
- MZ2 exact semantic recoding is closed on its tested forms at +340 B; post-hoc
  carrier refit misses break-even by 35.5x; WD4 and FS2/FS3 achieve smaller
  archives with the wrong complete-score sign.
- DC1S sparse-grid Family A is closed at full n600 FX5 scope: 388,326 B versus
  the 113,777 B member, a 274,549 B loss with all 190 groups negative.
- Literal C1 plane storage and PP1/SP1/WS1 shipping streams exceed the corridor
  before realization and should not be relabeled as RC1 variants.
- K=4,096 under the measured nested temporal-program formulation is byte-dead:
  105,811 B payload and 158,933 B shadow. This closes that point, not the family.
- Overall token agreement is not an admissible candidate score. K=2,048's
  98.796% headline hides class-1 IoU 0.146 and cannot be promoted without the
  exact full-RGB receiver and scorer.

Own-vehicle frontier: **DX2 S 0.14821987563243377 @ 180,368 B `[contest-CUDA T4 n600]`**, archive SHA-256 `976f706d5af6070f9785e495d35f2bd1bf10159a154fa19b45aefbf8f6de6674`; **UNMOVED**.
